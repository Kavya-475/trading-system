"""
tools/monte_carlo_trades.py — Monte Carlo trade removal analysis
================================================================
Randomly removes X% of completed round-trip trades and recalculates
CAGR / Sharpe / max-drawdown across N simulations.

Answers: how sensitive is the strategy to any random subset of trades?
If we had missed X% of entries, what is the distribution of outcomes?

Method
------
1. Match BUY → SELL rows to get completed round-trips with net P&L.
2. For each simulation: randomly select removal_rate% of trades to drop.
3. Dropped trades: their capital earns cash_yield instead (daily_rf per day held).
4. Adjusted daily P&L = base daily P&L − removed_trade_daily_pnl + cash_gain.
   (trade P&L is spread linearly across its holding days — good approximation
   for Sharpe/maxDD; CAGR is exact to within cash-yield rounding.)
5. Report percentile distribution across simulations.

Usage
-----
    python tools/monte_carlo_trades.py                          # defaults
    python tools/monte_carlo_trades.py --remove 0.1 0.2 0.3 0.4 0.5
    python tools/monte_carlo_trades.py --n 2000 --remove 0.2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

TRADE_LOG  = os.path.join(HERE, "trade_log_daily.csv")
EQUITY_CSV = os.path.join(HERE, "equity_curve_daily.csv")
RISK_FREE  = 0.065   # India 10yr G-Sec, matches config.py


# ── data loading ──────────────────────────────────────────────────────────────

def load_round_trips(path: str) -> pd.DataFrame:
    """Match BUY rows to SELL rows (FIFO per ticker) → completed round trips."""
    df   = pd.read_csv(path)
    buys = {}
    rows = []
    for _, r in df.iterrows():
        act    = str(r["action"]).strip()
        ticker = str(r["ticker"]).strip()
        date   = pd.Timestamp(r["date"])
        val    = float(r["value"])
        cost   = float(r["cost"]) if pd.notna(r.get("cost")) else 0.0
        tax    = float(r["tax"])  if pd.notna(r.get("tax"))  else 0.0

        if act == "BUY":
            buys.setdefault(ticker, []).append(
                {"entry_date": date, "entry_value": val, "entry_cost": cost}
            )
        elif act.startswith("SELL"):
            queue = buys.get(ticker, [])
            if not queue:
                continue
            buy      = queue.pop(0)
            net_pnl  = val - buy["entry_value"] - tax - buy["entry_cost"]
            holding  = max(1, (date - buy["entry_date"]).days)
            rows.append({
                "ticker":       ticker,
                "entry_date":   buy["entry_date"],
                "exit_date":    date,
                "entry_value":  buy["entry_value"],
                "exit_value":   val,
                "net_pnl":      net_pnl,
                "holding_days": holding,
                "return_pct":   net_pnl / buy["entry_value"] if buy["entry_value"] > 0 else 0.0,
                "sell_reason":  act,
            })
    return pd.DataFrame(rows)


def load_equity(path: str):
    eq = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    pnl = eq["value"].diff().fillna(0)
    return eq["value"], pnl


# ── simulation core ───────────────────────────────────────────────────────────

def _adjust(base_pnl: pd.Series, removed: pd.DataFrame, cash_yield: float) -> pd.Series:
    """
    Subtract removed trades' P&L and add cash yield for the same period.
    Trade P&L is spread evenly across holding days (linear approximation).
    """
    adj       = base_pnl.copy().astype(float)
    daily_rf  = cash_yield / 252
    for _, t in removed.iterrows():
        mask   = (adj.index >= t["entry_date"]) & (adj.index <= t["exit_date"])
        n_days = int(mask.sum())
        if n_days == 0:
            continue
        daily_trade  = t["net_pnl"] / n_days
        daily_cash   = t["entry_value"] * daily_rf
        adj[mask]   += -daily_trade + daily_cash
    return adj


def _metrics(adj_pnl: pd.Series, v0: float) -> dict:
    equity  = (v0 + adj_pnl.cumsum()).clip(lower=1)
    years   = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr    = (equity.iloc[-1] / v0) ** (1 / max(years, 1e-6)) - 1
    rets    = equity.pct_change().dropna()
    ann_vol = rets.std() * np.sqrt(252)
    sharpe  = (cagr - RISK_FREE) / ann_vol if ann_vol > 1e-9 else 0.0
    peak    = equity.cummax()
    maxdd   = ((equity - peak) / peak).min()
    return {"cagr": cagr, "sharpe": sharpe, "maxdd": maxdd}


def simulate(trips: pd.DataFrame, base_pnl: pd.Series, portfolio: pd.Series,
             removal_rate: float, n_sims: int, cash_yield: float,
             seed: int = 42) -> pd.DataFrame:
    rng  = np.random.default_rng(seed)
    v0   = float(portfolio.iloc[0])
    n    = len(trips)
    k    = max(1, int(round(n * removal_rate)))
    rows = []
    for _ in range(n_sims):
        idx     = rng.choice(n, size=k, replace=False)
        removed = trips.iloc[idx]
        adj     = _adjust(base_pnl, removed, cash_yield)
        rows.append(_metrics(adj, v0))
    return pd.DataFrame(rows)


# ── reporting ─────────────────────────────────────────────────────────────────

PCTS = [5, 10, 25, 50, 75, 90, 95]

def _pct_row(values: pd.Series, fmt: str) -> str:
    return "  ".join(fmt % (v * 100 if "%" in fmt else v)
                     for v in [values.quantile(p / 100) for p in PCTS])


def report(trips: pd.DataFrame, base_pnl: pd.Series, portfolio: pd.Series,
           removal_rate: float, n_sims: int, cash_yield: float, seed: int):
    df   = simulate(trips, base_pnl, portfolio, removal_rate, n_sims, cash_yield, seed)
    v0, vf = float(portfolio.iloc[0]), float(portfolio.iloc[-1])
    years  = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
    base_cagr = (vf / v0) ** (1 / years) - 1

    k = max(1, int(round(len(trips) * removal_rate)))
    print("\n  ── Remove %.0f%% (%d of %d trades) · %d sims ──" %
          (removal_rate * 100, k, len(trips), n_sims))
    print("  Base CAGR (0%% removed): %.1f%%" % (base_cagr * 100))
    hdr = "  %-8s " % "metric" + "  ".join("%6s" % ("p%d" % p) for p in PCTS)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for col, label, fmt in [
        ("cagr",   "CAGR   ", "%5.1f%%"),
        ("sharpe", "Sharpe ", "%6.2f "),
        ("maxdd",  "MaxDD  ", "%5.1f%%"),
    ]:
        vals = [df[col].quantile(p / 100) for p in PCTS]
        if col in ("cagr", "maxdd"):
            vals = [v * 100 for v in vals]
        print("  %-8s " % label + "  ".join(fmt % v for v in vals))

    pos  = 100 * (df["cagr"] > 0).mean()
    beat = 100 * (df["cagr"] > base_cagr * 0.75).mean()
    print("  positive: %.0f%%   within 75%% of base: %.0f%%" % (pos, beat))


def sell_breakdown(trips: pd.DataFrame):
    print("\n  Exit reason breakdown:")
    grp = trips.groupby("sell_reason").agg(
        count=("net_pnl", "count"),
        median_ret=("return_pct", "median"),
        win_rate=("return_pct", lambda x: (x > 0).mean()),
        total_pnl=("net_pnl", "sum"),
    ).sort_values("count", ascending=False)
    for reason, row in grp.iterrows():
        print("    %-20s  n=%3d  median=%+5.1f%%  win=%.0f%%  total_pnl=₹%+.0f" % (
            reason, row["count"], row["median_ret"] * 100,
            row["win_rate"] * 100, row["total_pnl"]))


def best_worst(trips: pd.DataFrame, n: int = 10):
    s = trips.sort_values("net_pnl")
    print("\n  Worst %d trades (by net P&L):" % n)
    for _, t in s.head(n).iterrows():
        print("    %s  %-16s  %+5.1f%%  ₹%+.0f  (%d days)" % (
            t["entry_date"].date(), t["ticker"],
            t["return_pct"] * 100, t["net_pnl"], t["holding_days"]))
    print("\n  Best %d trades (by net P&L):" % n)
    for _, t in s.tail(n).iterrows():
        print("    %s  %-16s  %+5.1f%%  ₹%+.0f  (%d days)" % (
            t["entry_date"].date(), t["ticker"],
            t["return_pct"] * 100, t["net_pnl"], t["holding_days"]))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", type=float, nargs="+",
                    default=[0.1, 0.2, 0.3, 0.4, 0.5],
                    help="Removal fractions, e.g. 0.1 0.2 0.3")
    ap.add_argument("--n",           type=int,   default=2000)
    ap.add_argument("--cash-yield",  type=float, default=0.05)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--top",         type=int,   default=10,
                    help="Best/worst trades to list")
    args = ap.parse_args()

    trips    = load_round_trips(TRADE_LOG)
    port, pnl = load_equity(EQUITY_CSV)

    if trips.empty:
        print("No completed round trips found — run a backtest first.")
        sys.exit(1)

    v0, vf = float(port.iloc[0]), float(port.iloc[-1])
    years  = (port.index[-1] - port.index[0]).days / 365.25
    base_cagr = (vf / v0) ** (1 / years) - 1
    rets      = port.pct_change().dropna()
    ann_vol   = rets.std() * np.sqrt(252)
    base_sharpe = (base_cagr - RISK_FREE) / ann_vol if ann_vol > 0 else 0
    peak  = port.cummax()
    base_maxdd = ((port - peak) / peak).min()

    print("=" * 72)
    print("  MONTE CARLO TRADE REMOVAL")
    print("  Trade log : %s" % TRADE_LOG)
    print("  Period    : %s → %s  (%.1f yrs)" % (
        port.index[0].date(), port.index[-1].date(), years))
    print("  Trades    : %d completed round-trips  (of %d log rows)" % (
        len(trips), pd.read_csv(TRADE_LOG).shape[0]))
    print("  Base      : CAGR %.1f%%  |  Sharpe %.2f  |  MaxDD %.1f%%" % (
        base_cagr * 100, base_sharpe, base_maxdd * 100))
    print("  Sims/rate : %d   |  percentiles shown: %s" % (
        args.n, ", ".join("p%d" % p for p in PCTS)))
    print("=" * 72)

    sell_breakdown(trips)
    best_worst(trips, args.top)

    print("\n" + "=" * 72)
    print("  REMOVAL SENSITIVITY")
    print("=" * 72)

    for rate in sorted(args.remove):
        report(trips, pnl, port, rate, args.n, args.cash_yield, args.seed)

    print()


if __name__ == "__main__":
    main()

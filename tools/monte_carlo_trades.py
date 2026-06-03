"""
tools/monte_carlo_trades.py — Monte Carlo trade removal (EXACT re-simulation)
=============================================================================
Robustness test: "if I'd randomly MISSED p% of my entries, what happens?"

Method (ground truth — no rupee arithmetic, no compounding approximation):
  For each simulation, the backtester is re-run with backtester.MC_ENTRY_SKIP_PROB
  set: every attempt to OPEN A NEW POSITION is skipped with probability p, and that
  slot's capital stays in CASH (earns CASH_YIELD, is NOT swept into other holdings).
  Each sim uses a fresh RNG seed, so the portfolio path genuinely diverges — exit
  timing, replacements, taxes and sizing are all re-derived correctly.

Why re-simulation (not subtracting trade P&L from the equity curve): a momentum
strategy RECYCLES capital across hundreds of sequential trades, so the sum of
realised trade P&L is many times the net portfolio growth. Subtracting absolute
trade P&L from a compounding curve is mathematically invalid (it drives equity
negative and pins CAGR to the wipe-out floor). Re-running is the only honest way.

Usage
-----
    python tools/monte_carlo_trades.py                       # 0.1 0.2 0.3, 500 sims
    python tools/monte_carlo_trades.py --remove 0.1 0.2 0.3 0.4 0.5 --n 1000
    python tools/monte_carlo_trades.py --workers 8

⚠  Uses the CURRENT config.py window/params. Set START_DATE/END_DATE first
   (e.g. the clean bias-free 2007-01-01 → 2020-07-31 block) for an honest read.
"""
import argparse
import contextlib
import os
import sys
import time
import multiprocessing as mp

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

TRADE_LOG  = os.path.join(HERE, "trade_log_daily.csv")
OUT        = os.path.join(HERE, "mc_trades_results.csv")
PCTS       = [5, 10, 25, 50, 75, 90, 95]


# ── worker: each sim re-runs the backtest with a per-entry skip probability ────

def _init():
    """Cache the (deterministic) data load so 2000 sims don't re-read CSVs."""
    import backtester as bt
    orig, cache = bt.load_data, {}
    def cached():
        if "d" not in cache:
            cache["d"] = orig()
        return cache["d"]
    bt.load_data = cached


def evaluate(args):
    rate, seed = args
    import backtester as bt
    bt.MC_ENTRY_SKIP_PROB = rate
    bt.MC_ENTRY_SKIP_SEED = seed
    try:
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
            m = bt.run_backtest(save=False)
        att = bt._mc_stats["attempted"]
        return {"rate": rate, "seed": seed,
                "cagr": float(m["cagr"]), "sharpe": float(m["sharpe"]),
                "maxdd": float(m["maxdd"]), "edge_cagr": float(m["edge_cagr"]),
                "skipped": bt._mc_stats["skipped"], "attempted": att,
                "skip_frac": (bt._mc_stats["skipped"] / att) if att else 0.0}
    except Exception as ex:                                  # noqa: BLE001
        return {"rate": rate, "seed": seed, "cagr": None, "err": str(ex)[:140]}


# ── descriptive stats from the baseline trade log (still valid) ────────────────

def load_round_trips(path):
    df, buys, rows = pd.read_csv(path), {}, []
    for _, r in df.iterrows():
        act, tk = str(r["action"]).strip(), str(r["ticker"]).strip()
        if act.startswith("BUY"):
            buys.setdefault(tk, []).append(
                {"d": pd.Timestamp(r["date"]), "v": float(r["value"]),
                 "c": float(r["cost"]) if pd.notna(r.get("cost")) else 0.0})
        elif act.startswith("SELL"):
            q = buys.get(tk, [])
            if not q:
                continue
            b   = q.pop(0)
            tax = float(r["tax"]) if pd.notna(r.get("tax")) else 0.0
            pnl = float(r["value"]) - b["v"] - tax - b["c"]
            rows.append({"ticker": tk, "entry_date": b["d"],
                         "net_pnl": pnl, "ret": pnl / b["v"] if b["v"] > 0 else 0.0,
                         "days": max(1, (pd.Timestamp(r["date"]) - b["d"]).days),
                         "reason": act})
    return pd.DataFrame(rows)


def descriptive(trips, top):
    print("\n  Exit-reason breakdown:")
    g = trips.groupby("reason").agg(n=("net_pnl", "count"),
                                    med=("ret", "median"),
                                    win=("ret", lambda x: (x > 0).mean()),
                                    pnl=("net_pnl", "sum")).sort_values("n", ascending=False)
    for r, row in g.iterrows():
        print("    %-18s n=%3d  median=%+5.1f%%  win=%.0f%%  total=₹%+.0f"
              % (r, row["n"], row["med"] * 100, row["win"] * 100, row["pnl"]))
    s = trips.sort_values("net_pnl")
    print("\n  Worst %d / best %d trades (net P&L):" % (top, top))
    for _, t in s.head(top).iterrows():
        print("    %s  %-14s %+6.1f%%  ₹%+.0f" % (t["entry_date"].date(), t["ticker"], t["ret"] * 100, t["net_pnl"]))
    print("    ...")
    for _, t in s.tail(top).iterrows():
        print("    %s  %-14s %+6.1f%%  ₹%+.0f" % (t["entry_date"].date(), t["ticker"], t["ret"] * 100, t["net_pnl"]))


# ── reporting ──────────────────────────────────────────────────────────────────

def report_rate(df, rate, base_cagr):
    sub = df[(df["rate"] == rate) & df["cagr"].notna()]
    if sub.empty:
        print("\n  ── Remove %.0f%% — NO valid sims ──" % (rate * 100));  return
    print("\n  ── Skip %.0f%% of entries · %d sims · mean skipped %.0f%% of attempts ──"
          % (rate * 100, len(sub), sub["skip_frac"].mean() * 100))
    hdr = "  %-8s " % "metric" + "  ".join("%7s" % ("p%d" % p) for p in PCTS)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for col, label, scale in [("cagr", "CAGR  ", 100), ("sharpe", "Sharpe", 1), ("maxdd", "MaxDD ", 100)]:
        vals = [sub[col].quantile(p / 100) * scale for p in PCTS]
        fmt  = "%6.1f%%" if scale == 100 else "%7.2f"
        print("  %-8s " % label + "  ".join(fmt % v for v in vals))
    pos  = 100 * (sub["cagr"] > 0).mean()
    keep = 100 * (sub["cagr"] >= base_cagr * 0.75).mean()
    print("  positive: %.0f%%   ≥75%% of base CAGR: %.0f%%   median CAGR %.1f%% (base %.1f%%)"
          % (pos, keep, sub["cagr"].median() * 100, base_cagr * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove",  type=float, nargs="+", default=[0.1, 0.2, 0.3])
    ap.add_argument("--n",       type=int, default=500, help="sims per removal rate")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--top",     type=int, default=8)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    import backtester as bt, config as cfg

    if args.report_only:
        df = pd.read_csv(OUT)
        base = df[df["rate"] == 0.0]["cagr"]
        base_cagr = float(base.iloc[0]) if len(base) else float("nan")
        for rate in sorted(r for r in df["rate"].unique() if r > 0):
            report_rate(df, rate, base_cagr)
        return

    # Baseline (p=0): one run, save outputs so the trade-log descriptive stats match config.
    print("Baseline run (p=0, saving trade log)…", flush=True)
    bt.MC_ENTRY_SKIP_PROB = 0.0
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
        m0 = bt.run_backtest(save=True)
    base_cagr = float(m0["cagr"])

    print("=" * 74)
    print("  MONTE CARLO TRADE REMOVAL  (exact re-simulation)")
    print("  Window : %s → %s   config: weights %.2f/%.2f/%.2f/%.2f/%.2f · TOP_N %d · cut %d"
          % (cfg.START_DATE, cfg.END_DATE, cfg.W_MOM_12M, cfg.W_MOM_6M, cfg.W_MOM_3M,
             cfg.W_VOL, cfg.W_DIST_DMA, cfg.TOP_N, cfg.EXIT_RANK_CUTOFF))
    print("  Base   : CAGR %.1f%%  |  Sharpe %.2f  |  MaxDD %.1f%%"
          % (base_cagr * 100, m0["sharpe"], m0["maxdd"] * 100))
    print("  Sims   : %d per rate × %d rates · %d workers" % (args.n, len(args.remove), args.workers))
    print("=" * 74)

    if os.path.exists(TRADE_LOG):
        descriptive(load_round_trips(TRADE_LOG), args.top)

    # Build the (rate, seed) work list — distinct seeds per rate for independent paths.
    jobs = [(rate, 1000 * j + i)
            for j, rate in enumerate(sorted(args.remove))
            for i in range(args.n)]
    print("\nRunning %d simulations…" % len(jobs), flush=True)

    t0, done, rows = time.time(), 0, [{"rate": 0.0, "seed": -1, "cagr": base_cagr,
                                       "sharpe": m0["sharpe"], "maxdd": m0["maxdd"],
                                       "edge_cagr": m0["edge_cagr"], "skip_frac": 0.0}]
    with mp.Pool(args.workers, initializer=_init) as pool:
        for r in pool.imap_unordered(evaluate, jobs, chunksize=4):
            rows.append(r); done += 1
            if done % 100 == 0 or done == len(jobs):
                el = time.time() - t0
                print("  %d/%d  elapsed %.1fm  eta %.1fm"
                      % (done, len(jobs), el / 60, el / done * (len(jobs) - done) / 60), flush=True)

    pd.DataFrame(rows).to_csv(OUT, index=False)

    print("\n" + "=" * 74); print("  REMOVAL SENSITIVITY"); print("=" * 74)
    df = pd.DataFrame(rows)
    for rate in sorted(args.remove):
        report_rate(df, rate, base_cagr)
    print("\n  Full per-sim metrics: %s\n" % OUT)


if __name__ == "__main__":
    main()

"""
tools/forensics.py  —  post-trade forensic analysis of momentum trades
======================================================================
Reconstructs every completed trade from trade_log_daily.csv, attaches the
point-in-time TECHNICAL characteristics that existed at the entry date (computed
only from data up to entry — no look-ahead), buckets winners/losers, and runs a
separation analysis to see which entry characteristics distinguish outcomes.

IMPORTANT — fundamentals gap (read this):
  Steps 3/5/7 of the brief ask for fundamental quality metrics at entry (ROE,
  D/E, OCF, promoter pledge, P/E, market cap, …). The repo has NO point-in-time
  fundamentals database, and yfinance only exposes CURRENT fundamentals — using
  those would be look-ahead bias, which the brief forbids. So this script does
  the full analysis on factors it CAN compute point-in-time from the price/volume
  cache (momentum, volatility, liquidity/ADV, holding period, exit reason,
  sector), writes trades_master.csv with empty fundamental columns ready to be
  joined, and prints exactly what external data is needed to finish.

Outputs:  trades_master.csv  (one row per completed trade)
"""
import os
import numpy as np
import pandas as pd

HERE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRADES = os.path.join(HERE, "trade_log_daily.csv")
PRICE  = os.path.join(HERE, "nse_data", "price_cache.csv")
VOL    = os.path.join(HERE, "nse_data", "volume_cache.csv")
UNIV   = os.path.join(HERE, "universe_history.csv")
OUT    = os.path.join(HERE, "trades_master.csv")

LB12, LB6, LB3, SKIP = 280, 140, 80, 25

# Fundamental fields the brief wants — left empty (need a point-in-time source).
FUND_COLS = ["roe", "roce", "ocf", "ocf_margin", "interest_cover", "debt_equity",
             "current_ratio", "market_cap", "promoter_holding", "promoter_pledge",
             "pe", "pb", "sales_growth", "earnings_growth"]


def reconstruct_trades(t):
    """Position lifecycle per ticker: open on first BUY-from-flat, close on the
    SELL that takes the position to zero (sells are full-position). Entry price =
    share-weighted avg of the buys that built the position."""
    out = []
    for tk, g in t[t.ticker != "-"].sort_values("date").groupby("ticker"):
        sh, cost, entry = 0, 0.0, None
        for _, r in g.iterrows():
            if r.action.startswith("BUY"):
                if sh == 0:
                    entry, cost = r.date, 0.0
                sh += r.shares; cost += r.value
            elif r.action.startswith("SELL") and sh > 0:
                avg = cost / sh
                out.append(dict(ticker=tk, entry_date=entry, exit_date=r.date,
                                hold_days=(r.date - entry).days, entry_px=avg,
                                exit_px=r.price, ret=(r.price / avg - 1.0),
                                exit_reason=r.action))
                sh, cost, entry = 0, 0.0, None
    return pd.DataFrame(out)


def entry_features(close, vol, tk, entry, exit):
    """Point-in-time technicals AT entry (only data <= entry) + max drawdown
    during the holding window."""
    f = dict.fromkeys(["mom_12m", "mom_6m", "mom_3m", "vol_6m", "adv_cr", "maxdd_hold"], np.nan)
    if tk not in close.columns:
        return f
    c = close[tk].loc[:entry].dropna()
    if len(c) >= LB12 + SKIP:
        p = c.iloc[-(SKIP + 1)]
        f["mom_12m"] = p / c.iloc[-(LB12 + SKIP)] - 1
        f["mom_6m"]  = p / c.iloc[-(LB6 + SKIP)] - 1
        f["mom_3m"]  = p / c.iloc[-(LB3 + SKIP)] - 1
    if len(c) >= LB6:
        f["vol_6m"] = c.iloc[-LB6:].pct_change().std() * np.sqrt(252)
    if tk in vol.columns:
        v = vol[tk].loc[:entry]
        val = (close[tk].loc[:entry] * v).dropna()
        if len(val) >= 20:
            f["adv_cr"] = val.iloc[-60:].mean() / 1e7      # ₹ crore, split-adjusted
    # max drawdown during the holding window
    h = close[tk].loc[entry:exit].dropna()
    if len(h) > 1:
        f["maxdd_hold"] = (h / h.cummax() - 1).min()
    return f


def main():
    t = pd.read_csv(TRADES, parse_dates=["date"])
    close = pd.read_csv(PRICE, index_col=0, parse_dates=True)
    vol   = pd.read_csv(VOL,   index_col=0, parse_dates=True)
    sect  = (pd.read_csv(UNIV).drop_duplicates("ticker").set_index("ticker")["sector"].to_dict()
             if os.path.exists(UNIV) else {})

    td = reconstruct_trades(t)
    feats = td.apply(lambda r: entry_features(close, vol, r.ticker, r.entry_date, r.exit_date),
                     axis=1, result_type="expand")
    td = pd.concat([td, feats], axis=1)
    td["sector"] = td.ticker.map(sect).fillna("Unknown")
    for c in FUND_COLS:
        td[c] = np.nan                                   # placeholder — needs PIT fundamentals

    # ── Buckets (data-driven) ──
    q20, q80 = td.ret.quantile(0.20), td.ret.quantile(0.80)
    td["bucket"] = "mid"
    td.loc[td.ret >= q80, "bucket"] = "A_winner"
    td.loc[td.ret <= q20, "bucket"] = "B_loser"
    td.loc[td.ret <= -0.50, "bucket"] = "C_catastrophic"
    td.loc[td.ret >= 1.00,  "bucket"] = "D_exceptional"
    td.to_csv(OUT, index=False)

    FACT = ["mom_12m", "mom_6m", "mom_3m", "vol_6m", "adv_cr", "hold_days", "maxdd_hold"]

    def stats(df):
        return pd.DataFrame({m: {"median": df[m].median(), "mean": df[m].mean(),
                                 "std": df[m].std(), "p25": df[m].quantile(.25),
                                 "p75": df[m].quantile(.75)} for m in FACT}).T

    print("=" * 70)
    print(f"  POST-TRADE FORENSICS — {len(td)} completed trades "
          f"({td.entry_date.min().date()} → {td.exit_date.max().date()})")
    print("=" * 70)
    print(f"  return: median {td.ret.median()*100:+.1f}%  mean {td.ret.mean()*100:+.1f}%  "
          f"min {td.ret.min()*100:.0f}%  max {td.ret.max()*100:.0f}%")
    print(f"  win rate: {(td.ret>0).mean()*100:.0f}%   |  buckets: "
          + ", ".join(f"{k}={v}" for k, v in td.bucket.value_counts().items()))
    print(f"  thresholds: winner ≥ {q80*100:+.0f}%   loser ≤ {q20*100:+.0f}%")

    win, los = td[td.bucket.isin(["A_winner","D_exceptional"])], td[td.bucket=="B_loser"]
    print("\n--- WINNERS (top quintile + exceptional) ---"); print(stats(win).round(3).to_string())
    print("\n--- LOSERS (bottom quintile) ---");              print(stats(los).round(3).to_string())

    # ── Separation: median diff + effect size + Spearman vs return ──
    print("\n" + "=" * 70)
    print("  SEPARATION ANALYSIS (look-ahead-free factors, ranked by |Spearman|)")
    print("=" * 70)
    rows = []
    for m in FACT:
        a, b = win[m].dropna(), los[m].dropna()
        pooled = np.sqrt((a.var() + b.var()) / 2) if len(a) > 1 and len(b) > 1 else np.nan
        d = (a.median() - b.median())
        cohen = (a.mean() - b.mean()) / pooled if pooled else np.nan
        sub = td[[m, "ret"]].dropna()
        # Spearman = Pearson on ranks (avoids a scipy dependency)
        rho = sub[m].rank().corr(sub.ret.rank()) if len(sub) > 10 else np.nan
        rows.append({"factor": m, "win_median": round(a.median(),3), "los_median": round(b.median(),3),
                     "median_diff": round(d,3), "cohen_d": round(cohen,2), "spearman_ret": round(rho,3)})
    sep = pd.DataFrame(rows).reindex(pd.DataFrame(rows).spearman_ret.abs().sort_values(ascending=False).index)
    print(sep.to_string(index=False))

    # ── Catastrophic / worst-trade + named-blowup check ──
    print("\n" + "=" * 70)
    print("  STEP 6 — CATASTROPHIC / WORST-TRADE INVESTIGATION")
    print("=" * 70)
    cat = td[td.ret <= -0.50]
    print(f"  trades with return ≤ -50% (catastrophic): {len(cat)}")
    if cat.empty:
        print("  → NONE. The 200-DMA + rank exits cap losses; worst-decile shown instead.")
    worst = td.nsmallest(10, "ret")[["ticker","entry_date","exit_date","hold_days","ret","exit_reason","mom_12m","vol_6m","adv_cr"]]
    print(worst.round(3).to_string(index=False))
    print("\n  Were the classic blow-ups ever held by the strategy?")
    for nm in ["VAKRANGEE","PCJEWELLER","GITANJALI","JETAIRWAYS","YESBANK","DHFL","RCOM"]:
        sub = td[td.ticker == nm]
        if len(sub):
            reasons = sorted(set(sub.exit_reason))
            print(f"    {nm:12} held {len(sub)}x, worst {sub.ret.min()*100:.0f}%, reasons {reasons}")
        else:
            print(f"    {nm:12} NEVER bought (never a top-momentum name → no exposure)")
    print(f"\nWrote {OUT}  ({len(td)} trades × {td.shape[1]} cols; fundamental cols empty pending PIT data)")


if __name__ == "__main__":
    main()

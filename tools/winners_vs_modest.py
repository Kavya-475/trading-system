"""
tools/winners_vs_modest.py
==========================
What separates EXCEPTIONAL winners (ret >= +100%) from MODEST winners (~+10%)
using ONLY entry-date variables (point-in-time, no look-ahead):
  momentum rank, 12M/6M/3M return, volatility, distance above 250-DMA, sector, liquidity.

Builds on trades_master.csv (from forensics.py); adds momentum rank + DMA distance,
which need the per-date cross-sectional universe.
"""
import os
import sys
import numpy as np
import pandas as pd
HERE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)                      # repo root (config.py, backtester.py)
import config as cfg
from backtester import load_universe_history, get_universe_for_date
TM    = os.path.join(HERE, "trades_master.csv")
PRICE = os.path.join(HERE, "nse_data", "price_cache.csv")
LB12, LB6, LB3, SKIP, DMA = cfg.LOOKBACK_12M, cfg.LOOKBACK_6M, cfg.LOOKBACK_3M, cfg.SKIP_RECENT, cfg.DMA_EXIT

close = pd.read_csv(PRICE, index_col=0, parse_dates=True)
hist  = load_universe_history()
_cache = {}

def ranking_on(date):
    """Composite-momentum rank (1=best) within the eligible universe on `date`,
    exactly as the strategy ranks (z-scored 12/6/3M momentum, cfg weights)."""
    key = pd.Timestamp(date)
    if key in _cache:
        return _cache[key]
    uni = get_universe_for_date(date, hist) if hist else {}
    sl  = close.loc[:date]
    recs = {}
    for tk in uni:
        if tk not in sl.columns:
            continue
        c = sl[tk].dropna()
        if len(c) < LB12 + SKIP:
            continue
        p = c.iloc[-(SKIP + 1)]
        recs[tk] = (p / c.iloc[-(LB12+SKIP)] - 1, p / c.iloc[-(LB6+SKIP)] - 1, p / c.iloc[-(LB3+SKIP)] - 1)
    if not recs:
        _cache[key] = {}; return {}
    df = pd.DataFrame(recs, index=["m12","m6","m3"]).T
    z  = lambda s: (s - s.mean())/s.std() if s.std() > 0 else s*0
    df["score"] = cfg.W_MOM_12M*z(df.m12) + cfg.W_MOM_6M*z(df.m6) + cfg.W_MOM_3M*z(df.m3)
    rank = {tk: i+1 for i, tk in enumerate(df.sort_values("score", ascending=False).index)}
    _cache[key] = rank
    return rank

def dist_above_dma(tk, date):
    if tk not in close.columns:
        return np.nan
    c = close[tk].loc[:date].dropna()
    if len(c) < DMA:
        return np.nan
    return c.iloc[-1] / c.iloc[-DMA:].mean() - 1

def main():
    td = pd.read_csv(TM, parse_dates=["entry_date","exit_date"])
    td["mom_rank"] = td.apply(lambda r: ranking_on(r.entry_date).get(r.ticker, np.nan), axis=1)
    td["dist_dma"] = td.apply(lambda r: dist_above_dma(r.ticker, r.entry_date), axis=1)

    exc = td[td.ret >= 1.00]                       # exceptional: +100%+
    mod = td[(td.ret >= 0.05) & (td.ret <= 0.20)]  # modest: ~+10% band
    FACT = ["mom_rank","mom_12m","mom_6m","mom_3m","vol_6m","dist_dma","adv_cr"]

    print("="*72)
    print(f"  +100% WINNERS (n={len(exc)})  vs  +10% MODEST WINNERS (n={len(mod)})")
    print(f"  entry-date variables only (point-in-time)")
    print("="*72)
    hdr = f"  {'factor':10} {'exc_med':>9} {'mod_med':>9} {'diff':>9} {'exc_mean':>9} {'mod_mean':>9}"
    print(hdr); print("  " + "-"*68)
    for m in FACT:
        e, o = exc[m].dropna(), mod[m].dropna()
        print(f"  {m:10} {e.median():9.3f} {o.median():9.3f} {e.median()-o.median():9.3f} "
              f"{e.mean():9.3f} {o.mean():9.3f}")

    # Spearman of each entry var vs return, across ALL winners (ret>0) — direction/strength
    win = td[td.ret > 0]
    print("\n  Rank-correlation with return (across all positive trades, n=%d):" % len(win))
    rows = []
    for m in FACT:
        s = win[[m,"ret"]].dropna()
        rho = s[m].rank().corr(s.ret.rank()) if len(s) > 10 else np.nan
        rows.append((m, rho))
    for m, rho in sorted(rows, key=lambda x: -abs(x[1]) if pd.notna(x[1]) else 0):
        print(f"     {m:10} spearman = {rho:+.3f}")

    print("\n  Sector mix:")
    for lbl, grp in [("+100%", exc), ("+10%", mod)]:
        vc = grp.sector.value_counts(normalize=True).head(5)
        print(f"     {lbl:5}: " + ", ".join(f"{k} {v*100:.0f}%" for k, v in vc.items()))

    print("\n  Holding period (context — an OUTCOME, not an entry feature):")
    print(f"     +100% median hold {exc.hold_days.median():.0f}d   |   +10% median hold {mod.hold_days.median():.0f}d")

    td.to_csv(TM, index=False)   # persist the two new columns
    print(f"\n  (added mom_rank + dist_dma to {os.path.basename(TM)})")

if __name__ == "__main__":
    main()

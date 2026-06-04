"""
tools/daily_ranks.py — the full daily momentum ranking the algo generates
=========================================================================
Prints every stock the live engine ranks today, best→worst, using the EXACT
same path as execution.py: apply_liquidity_filter → compute_scores (signals.py),
with the deployed opt#7 weights from config.py. Reads only the local cache (no
network). Marks the TOP_N target-holdings line and the EXIT_RANK_CUTOFF line, and
lists any universe names excluded (illiquid / insufficient history) so all 250 are
accounted for.

    python tools/daily_ranks.py                 # full table to stdout
    python tools/daily_ranks.py --csv           # also write daily_ranks.csv
    python tools/daily_ranks.py --top 50        # only show the top 50

Run on the VM after data_manager updates the cache to see "today's" ranking.
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import config as cfg
from signals import UNIVERSE, compute_scores, apply_liquidity_filter
from data_manager import load_for_signals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=0, help="show only the top N (0 = all)")
    ap.add_argument("--csv", action="store_true", help="also write daily_ranks.csv")
    args = ap.parse_args()

    close, volume = load_for_signals()
    asof = close.index[-1].date()
    universe = list(UNIVERSE.keys())

    liquid = apply_liquidity_filter(close, volume, universe)
    scored = compute_scores(close, liquid)            # ranked best→worst, opt#7 weights
    if scored.empty:
        sys.exit("No stocks scored — check the cache (run data_manager.py).")

    ranked = list(scored.index)
    excluded = [t for t in universe if t not in ranked]   # illiquid or < LOOKBACK_12M+SKIP history

    print("=" * 100)
    print(f"  DAILY MOMENTUM RANKING — as of {asof} close   (signals use the close {cfg.SKIP_RECENT}d before)")
    print(f"  weights {cfg.W_MOM_12M}/{cfg.W_MOM_6M}/{cfg.W_MOM_3M}/{cfg.W_VOL}/{cfg.W_DIST_DMA}"
          f" · TOP_N {cfg.TOP_N} · exit-cutoff {cfg.EXIT_RANK_CUTOFF}"
          f" · ranked {len(ranked)}/{len(universe)} (excluded {len(excluded)})")
    print("=" * 100)
    print(f"  {'#':>3}  {'ticker':<12}{'score':>7}  {'sector':<14}{'price':>9}"
          f"{'12m':>7}{'6m':>7}{'3m':>7}{'vol':>6}{'distDMA':>8}")
    print("  " + "-" * 96)

    n_show = args.top if args.top > 0 else len(ranked)
    for i, (t, r) in enumerate(scored.head(n_show).iterrows()):
        rank = i + 1
        print(f"  {rank:>3}  {t:<12}{r['score']:>+7.2f}  {r['sector']:<14}{r['price']:>9,.1f}"
              f"{r['mom_12m']*100:>6.0f}%{r['mom_6m']*100:>6.0f}%{r['mom_3m']*100:>6.0f}%"
              f"{r['vol_6m']*100:>5.0f}%{r['dist_dma']*100:>+7.0f}%")
        if rank == cfg.TOP_N and n_show > cfg.TOP_N:
            print(f"  {'':>3}  {'── TOP ' + str(cfg.TOP_N) + ' (target holdings) ' + '─'*70}")
        if rank == cfg.EXIT_RANK_CUTOFF and n_show > cfg.EXIT_RANK_CUTOFF:
            print(f"  {'':>3}  {'── EXIT CUTOFF ' + str(cfg.EXIT_RANK_CUTOFF) + ' (held names must stay above) ' + '─'*52}")

    if excluded:
        print("\n  Excluded (illiquid or < %d trading days history) — %d names:"
              % (cfg.LOOKBACK_12M + cfg.SKIP_RECENT, len(excluded)))
        print("    " + ", ".join(sorted(excluded)))

    if args.csv:
        out = os.path.join(HERE, "daily_ranks.csv")
        cols = ["sector", "score", "price", "mom_12m", "mom_6m", "mom_3m", "vol_6m", "dist_dma"]
        df = scored[cols].copy()
        df.insert(0, "rank", range(1, len(df) + 1))
        df.to_csv(out)
        print(f"\n  Wrote {out}  ({len(df)} ranked stocks)")


if __name__ == "__main__":
    main()

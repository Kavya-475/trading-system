"""
tools/bootstrap.py  —  single-path risk via block bootstrap
===========================================================
A backtest is ONE realization of history. This resamples the strategy's own
daily returns in *blocks* (to preserve momentum/autocorrelation) into thousands
of alternate histories of the same length, then reports the DISTRIBUTION of CAGR
and max drawdown — so you see the realistic range, not just the lucky/unlucky
single number.

Reads the equity curve produced by backtester.py.

Usage:
    python tools/bootstrap.py
    python tools/bootstrap.py --equity equity_curve_daily.csv --n 10000 --block 21
"""
import argparse
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity", default=os.path.join(HERE, "equity_curve_daily.csv"))
    ap.add_argument("--n",     type=int, default=10000, help="number of resampled paths")
    ap.add_argument("--block", type=int, default=21,    help="block length in trading days (~1 month)")
    ap.add_argument("--seed",  type=int, default=42)
    args = ap.parse_args()

    eq = pd.read_csv(args.equity, index_col=0, parse_dates=True).iloc[:, 0].dropna()
    r  = eq.pct_change().dropna().values
    T  = len(r)
    years = T / 252.0
    rng = np.random.default_rng(args.seed)

    def cagr(growth_last):  return growth_last ** (252.0 / T) - 1.0

    # realized path
    g_real   = np.cumprod(1 + r)
    real_cagr = cagr(g_real[-1])
    real_dd   = (g_real / np.maximum.accumulate(g_real) - 1).min()

    # ── vectorized block bootstrap ──
    L  = args.block
    nb = int(np.ceil(T / L))
    starts = rng.integers(0, T - L, size=(args.n, nb))           # (N, nb)
    cols   = (starts[:, :, None] + np.arange(L)).reshape(args.n, -1)[:, :T]
    paths  = r[cols]                                             # (N, T) daily returns
    growth = np.cumprod(1 + paths, axis=1)
    cagrs  = growth[:, -1] ** (252.0 / T) - 1.0
    peak   = np.maximum.accumulate(growth, axis=1)
    dds    = (growth / peak - 1.0).min(axis=1)

    def pct(a, p): return np.percentile(a, p)
    def rank(a, v): return (a < v).mean() * 100

    print("=" * 60)
    print(f"  BLOCK-BOOTSTRAP  ({args.n:,} paths, {L}-day blocks, {years:.1f}y each)")
    print(f"  source: {os.path.basename(args.equity)}")
    print("=" * 60)
    print(f"  REALIZED (the actual backtest):  CAGR {real_cagr*100:5.1f}%   maxDD {real_dd*100:6.1f}%")
    print( "  ── it sits at the {:.0f}th percentile of CAGR, {:.0f}th of drawdown-severity ──"
           .format(rank(cagrs, real_cagr), rank(-dds, -real_dd)))
    print("-" * 60)
    print("  CAGR distribution across alternate histories:")
    for p in (5, 25, 50, 75, 95):
        print(f"     {p:3d}th percentile : {pct(cagrs, p)*100:6.1f}%")
    print(f"     P(CAGR < 0)       : {(cagrs < 0).mean()*100:4.1f}%")
    print("-" * 60)
    print("  MAX DRAWDOWN distribution (how bad it can get):")
    for p, lbl in ((50, "median"), (75, "bad"), (95, "very bad"), (99, "worst-case")):
        print(f"     {lbl:10s} ({p:2d}th): {pct(dds, 100-p)*100:6.1f}%")
    print("=" * 60)
    print("  Read it as: plan around the MEDIAN CAGR and the 95th-percentile")
    print("  drawdown — NOT the single realized number.")
    print("=" * 60)


if __name__ == "__main__":
    main()

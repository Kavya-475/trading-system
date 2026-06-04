"""
tools/optimize_structural.py — multi-period walk-forward over STRUCTURAL params
================================================================================
Phase-1 portfolio-construction sweep. Varies ONLY:

    TOP_N   ·   MAX_PER_SECTOR   ·   EXIT_RANK_CUTOFF

holding the SCORING WEIGHTS and lookbacks at their config.py values (opt#7), and
drives the EXISTING backtester (backtester.run_backtest) — no logic reimplemented.

Same walk-forward harness as optimize_weights.py: each combo is scored on several
disjoint windows tiling 2010→2020 (five 2-yr periods), so every window is an
independent out-of-sample test. Records per-window CAGR / Sharpe / maxDD and the
mean / worst across windows.

IMPORTANT — ranking: TOP_N and MAX_PER_SECTOR trade return for risk almost
mechanically (fewer names / higher sector cap → higher CAGR AND deeper drawdown),
so ranking by raw CAGR just picks maximum concentration. Default objective is
**mean_sharpe** (risk-adjusted); min_cagr (worst fold) and mean_maxdd are shown
alongside. Read TOP_N as a risk/capacity choice, not a pure backtest optimum.

Run (small grid → minutes, exhaustive):
    python tools/optimize_structural.py
    python tools/optimize_structural.py --objective min_cagr     # robustness
    python tools/optimize_structural.py --objective mean_cagr    # raw return (concentration!)
    python tools/optimize_structural.py --report-only
"""
import argparse
import contextlib
import itertools
import os
import statistics
import sys
import time
import multiprocessing as mp

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, HERE)                              # config, data caches, outputs (root)
sys.path.insert(0, os.path.join(HERE, "research"))    # backtester.py (research/)
OUT  = os.path.join(HERE, "wfo_struct_results.csv")
TOP  = os.path.join(HERE, "wfo_struct_top.csv")

WINDOWS = []   # filled by main(), passed to workers (spawn-safe)

# ── Phase-1 structural grid (weights/lookbacks stay at config.py) ───────────
SPACE = {
    "TOP_N":            [10, 12, 15, 18, 20, 25],
    "MAX_PER_SECTOR":   [3, 4, 5, 6, 8],
    "EXIT_RANK_CUTOFF": [40, 50, 65],
}
PARAMS = list(SPACE.keys())


def build_grid():
    """Full product — structural combos are all distinct (no scale-degeneracy).
    Drops combos where the exit band is tighter than the portfolio (cutoff<TOP_N)."""
    out = []
    for vals in itertools.product(*(SPACE[p] for p in PARAMS)):
        c = dict(zip(PARAMS, vals))
        if c["EXIT_RANK_CUTOFF"] < c["TOP_N"]:
            continue                       # cutoff must exceed holdings
        out.append(c)
    return out


def _worker_init(windows):
    global WINDOWS
    WINDOWS = windows
    import backtester as bt
    orig, cache = bt.load_data, {}
    def cached():
        if "d" not in cache:
            cache["d"] = orig()
        return cache["d"]
    bt.load_data = cached


def evaluate(combo):
    """Run the CURRENT backtester (weights = config/opt#7) on every window for one
    structural combo."""
    import config as cfg, backtester as bt
    for k, v in combo.items():
        setattr(cfg, k, int(v))
    row = dict(combo)
    cagrs, sharpes, mdds = [], [], []
    try:
        for label, s, e in WINDOWS:
            cfg.START_DATE, cfg.END_DATE = s, e
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), \
                    contextlib.redirect_stderr(dn):
                m = bt.run_backtest(save=False)
            row[f"cagr_{label}"] = round(float(m["cagr"]), 6)
            cagrs.append(float(m["cagr"])); sharpes.append(float(m["sharpe"])); mdds.append(float(m["maxdd"]))
        row["mean_cagr"]   = round(sum(cagrs) / len(cagrs), 6)
        row["min_cagr"]    = round(min(cagrs), 6)
        row["mean_sharpe"] = round(sum(sharpes) / len(sharpes), 6)
        row["mean_maxdd"]  = round(sum(mdds) / len(mdds), 6)
        row["worst_maxdd"] = round(min(mdds), 6)
        row["ok"] = 1
    except Exception as ex:                                          # noqa: BLE001
        row["ok"], row["err"] = 0, str(ex)[:160]
    return row


def calibrate():
    _worker_init(WINDOWS)
    base = {p: SPACE[p][0] for p in PARAMS}
    base["EXIT_RANK_CUTOFF"] = 65          # ensure cutoff>TOP_N for the warm-up
    evaluate(base)
    t0 = time.time(); evaluate(base)
    return time.time() - t0


def report(objective):
    if not os.path.exists(OUT):
        print("No results yet."); return
    df = pd.read_csv(OUT)
    df = df[df.get("ok", 1) == 1].dropna(subset=[objective])
    if df.empty:
        print("No successful runs."); return
    asc = (objective in ("mean_maxdd", "worst_maxdd"))      # less-negative is better
    df = df.sort_values(objective, ascending=asc)
    df.head(20).to_csv(TOP, index=False)

    win = [c for c in df.columns if c.startswith("cagr_")]
    show = PARAMS + win + ["mean_cagr", "min_cagr", "mean_sharpe", "mean_maxdd"]
    show = [c for c in show if c in df.columns]
    print("\n" + "=" * 104)
    print(f"  STRUCTURAL SWEEP — top by {objective}   ({len(df)} combos × {len(win)} periods, weights=opt#7)")
    print("  reminder: rank by mean_sharpe; mean_cagr alone just favours concentration (higher CAGR + deeper DD)")
    print("=" * 104)
    with pd.option_context("display.width", 240, "display.max_columns", 60,
                           "display.float_format", lambda x: f"{x:.4f}"):
        print(df[show].head(20).to_string(index=False))
    print("\n  Marginal mean", objective, "by value:")
    for p in PARAMS:
        g = df.groupby(p)[objective].mean().sort_values(ascending=asc)
        print(f"    {p:18} " + "  ".join(f"{int(k)}:{v:.3f}" for k, v in g.items()))
    print(f"\n  Full: {OUT}\n  Top : {TOP}")


def _shift(d, n):
    y, m, day = (int(x) for x in d.split("-")); return f"{y+n:04d}-{m:02d}-{day:02d}"


def make_windows(start_year, end_year, win):
    out, y = [], start_year
    while y + win <= end_year:
        out.append((f"{y}-{(y+win-1)%100:02d}", f"{y}-01-01", f"{y+win-1}-12-31"))
        y += win
    return out


def main():
    global WINDOWS
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="mean_sharpe",
                    help="mean_sharpe | min_cagr | mean_cagr | mean_maxdd | worst_maxdd")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--start-year", type=int, default=2010)
    ap.add_argument("--end-year",   type=int, default=2020)
    ap.add_argument("--window-years", type=int, default=2)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    WINDOWS = make_windows(args.start_year, args.end_year, args.window_years)
    if args.report_only:
        report(args.objective); return

    grid = build_grid()
    import config as cfg
    print(f"Structural combos: {len(grid)}  (weights held at "
          f"{cfg.W_MOM_12M}/{cfg.W_MOM_6M}/{cfg.W_MOM_3M}/{cfg.W_VOL}/{cfg.W_DIST_DMA})")
    print(f"Periods ({len(WINDOWS)}): " + "  ".join(f"{s[:4]}-{e[2:4]}" for _, s, e in WINDOWS))
    sec = calibrate()
    print(f"  ~{sec:.1f}s/combo · {args.workers} workers · ~{len(grid)*sec/args.workers/60:.0f} min  | rank by {args.objective}\n")

    if os.path.exists(OUT):
        os.replace(OUT, OUT + ".prev")
    done, t0, header = 0, time.time(), False
    try:
        with mp.Pool(args.workers, initializer=_worker_init, initargs=(WINDOWS,)) as pool, \
                open(OUT, "w") as fh:
            for r in pool.imap_unordered(evaluate, grid, chunksize=1):
                pd.DataFrame([r]).to_csv(fh, header=not header, index=False); fh.flush(); header = True
                done += 1
                if done % 10 == 0 or done == len(grid):
                    print(f"  {done}/{len(grid)}  {(time.time()-t0)/60:.1f}m", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted — partial results.")
    finally:
        report(args.objective)


if __name__ == "__main__":
    main()

"""
tools/optimize_lookback.py — multi-period walk-forward over the LOOKBACK horizons
=================================================================================
Phase-2 sweep. Varies ONLY the momentum lookback windows:

    LOOKBACK_12M   ·   LOOKBACK_6M   ·   LOOKBACK_3M   (trading days)

holding weights (opt#7), TOP_N/sector/cutoff and SKIP_RECENT at their config.py
values, and driving the EXISTING backtester. Same harness as the weight/structural
sweeps: each combo scored on five disjoint 2-yr periods tiling 2010→2020.

Lookbacks define the momentum SIGNAL itself, so they're the easiest thing to
overfit — keep the grid coarse and trust only settings that win across folds AND
survive the full-window validation. Defaults 280/140/80 (~12/6/3 months) are
robust priors; deviate only on strong, consistent evidence.

    python tools/optimize_lookback.py
    python tools/optimize_lookback.py --objective min_cagr
    python tools/optimize_lookback.py --report-only
"""
import argparse
import contextlib
import itertools
import os
import sys
import time
import multiprocessing as mp

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, HERE)                              # config, data caches, outputs (root)
sys.path.insert(0, os.path.join(HERE, "research"))    # backtester.py (research/)
OUT = os.path.join(HERE, "wfo_lb_results.csv")
TOP = os.path.join(HERE, "wfo_lb_top.csv")

WINDOWS = []

SPACE = {
    "LOOKBACK_12M": [250, 280, 300],
    "LOOKBACK_6M":  [120, 140, 160],
    "LOOKBACK_3M":  [60, 80, 100],
    "SKIP_RECENT":  [0, 15, 25, 40],     # days of recent data skipped (reversal guard)
}
PARAMS = list(SPACE.keys())


def build_grid():
    out = []
    for vals in itertools.product(*(SPACE[p] for p in PARAMS)):
        c = dict(zip(PARAMS, vals))
        if not (c["LOOKBACK_12M"] > c["LOOKBACK_6M"] > c["LOOKBACK_3M"]):
            continue                       # horizons must be ordered
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
        row["ok"] = 1
    except Exception as ex:                                          # noqa: BLE001
        row["ok"], row["err"] = 0, str(ex)[:160]
    return row


def calibrate():
    _worker_init(WINDOWS)
    base = {p: SPACE[p][1] for p in PARAMS}     # 280/140/80
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
    asc = (objective == "mean_maxdd")
    df = df.sort_values(objective, ascending=asc)
    df.to_csv(TOP, index=False)
    win = [c for c in df.columns if c.startswith("cagr_")]
    show = PARAMS + win + ["mean_cagr", "min_cagr", "mean_sharpe", "mean_maxdd"]
    show = [c for c in show if c in df.columns]
    print("\n" + "=" * 104)
    print(f"  LOOKBACK SWEEP — by {objective}   ({len(df)} combos × {len(win)} periods, weights=opt#7, struct=current)")
    print("  current default = 280/140/80; lookbacks overfit easily — trust only fold-consistent, full-window-validated moves")
    print("=" * 104)
    with pd.option_context("display.width", 240, "display.max_columns", 60,
                           "display.float_format", lambda x: f"{x:.4f}"):
        print(df[show].to_string(index=False))
    print("\n  Marginal mean", objective, "by value:")
    for p in PARAMS:
        g = df.groupby(p)[objective].mean().sort_values(ascending=asc)
        print(f"    {p:14} " + "  ".join(f"{int(k)}:{v:.3f}" for k, v in g.items()))
    print(f"\n  Full: {OUT}\n  Top : {TOP}")


def make_windows(sy, ey, w):
    out, y = [], sy
    while y + w <= ey:
        out.append((f"{y}-{(y+w-1)%100:02d}", f"{y}-01-01", f"{y+w-1}-12-31")); y += w
    return out


def main():
    global WINDOWS
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="mean_sharpe",
                    help="mean_sharpe | min_cagr | mean_cagr | mean_maxdd")
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
    print(f"Lookback combos: {len(grid)}  (weights {cfg.W_MOM_12M}/{cfg.W_MOM_6M}/{cfg.W_MOM_3M}/{cfg.W_VOL}/{cfg.W_DIST_DMA}, "
          f"struct {cfg.TOP_N}/{cfg.MAX_PER_SECTOR}/{cfg.EXIT_RANK_CUTOFF})")
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
                if done % 5 == 0 or done == len(grid):
                    print(f"  {done}/{len(grid)}  {(time.time()-t0)/60:.1f}m", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted — partial results.")
    finally:
        report(args.objective)


if __name__ == "__main__":
    main()

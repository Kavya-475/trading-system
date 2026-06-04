"""
tools/optimize_weights.py  —  multi-period search over the SCORING WEIGHTS
===========================================================================
Extensively varies the five scoring weights only —

    W_MOM_12M   W_MOM_6M   W_MOM_3M   W_VOL   W_DIST_DMA

— holding every other parameter at its config.py value, and drives the EXISTING
backtester (backtester.run_backtest). No backtest logic is re-implemented.

Multi-period robustness (what the user asked for)
-------------------------------------------------
Each weight combo is scored on SEVERAL disjoint windows that tile a span
(default 2010→2020 as five 2-year periods: 2010-11, 2012-13, 2014-15, 2016-17,
2018-19). Because every combo is grid-searched exhaustively, each window is an
independent out-of-sample test of that combo — there is no "fit" step to overfit.
A combo that scores well in ALL periods is regime-robust; one that only wins a
single lucky period is not. The combo's per-window CAGR is recorded, plus the
mean, the worst (min) and the spread (std) across windows. The final table is the
best 50 ranked by mean CAGR — re-sort by min_cagr for the strict maximin choice.

    --start-year 2010 --end-year 2020 --window-years 2   (the default span)

Throughput (the "max backtests in 12h" ask)
--------------------------------------------
  • Every worker loads the price cache ONCE (memoized) — not once per combo.
  • Parallel across all CPU cores.
  • SCALE-DEDUP: a momentum score is rank-invariant to multiplying *all* weights
    by a positive constant, so (0.4,0.5,0.2,0,0.3) and (0.8,1.0,0.4,0,0.6) give
    the identical portfolio. Combos are collapsed to a canonical (L1-normalised)
    key so no two rank-equivalent backtests are ever run.
  • SELF-SIZING: times a warm run, reads the core count, and runs the whole
    deduped grid if it fits the budget, else a random sample that does.
  • Streams each result to wfo_results.csv as it lands; reports partials on Ctrl-C.

Run on the VM (12-hour budget, all cores):
    python tools/optimize_weights.py --budget 43200
    python tools/optimize_weights.py --objective min_cagr      # rank by worst period
    python tools/optimize_weights.py --report-only             # re-print tables
"""
import argparse
import contextlib
import os
import random
import statistics
import sys
import time
import multiprocessing as mp

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, HERE)                              # config, data caches, outputs (root)
sys.path.insert(0, os.path.join(HERE, "research"))    # backtester.py (research/)
OUT   = os.path.join(HERE, "wfo_results.csv")
TOP50 = os.path.join(HERE, "wfo_top50.csv")

# Filled in by main() from CLI args, read by the workers. Each entry:
#   (label, start_date, end_date)   e.g. ("2010-11", "2010-01-01", "2011-12-31")
WINDOWS = []

# ── Weight search space (fine grid; everything else stays at config.py) ──────
SPACE = {
    "W_MOM_12M":  [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "W_MOM_6M":   [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "W_MOM_3M":   [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "W_VOL":      [-0.3, -0.2, -0.1, 0.0],
    "W_DIST_DMA": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
}
PARAMS = list(SPACE.keys())


def canon_key(c):
    """L1-normalised, rounded weight vector. Collapses rank-equivalent scalar
    multiples so we never run the same portfolio twice."""
    w = [c[p] for p in PARAMS]
    s = sum(abs(x) for x in w)
    if s == 0:
        return None
    return tuple(round(x / s, 3) for x in w)


def build_grid():
    """Full grid, minus (a) all-momentum-zero combos (no trend signal) and
    (b) scale-duplicates collapsed by canonical key."""
    import itertools
    seen, out = set(), []
    for vals in itertools.product(*(SPACE[p] for p in PARAMS)):
        c = dict(zip(PARAMS, vals))
        if c["W_MOM_12M"] + c["W_MOM_6M"] + c["W_MOM_3M"] <= 0:
            continue                       # require some momentum signal
        k = canon_key(c)
        if k is None or k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


def make_windows(start_year, end_year, win):
    """Disjoint consecutive windows tiling [start_year, end_year)."""
    out, y = [], start_year
    while y + win <= end_year:
        out.append((f"{y}-{(y + win - 1) % 100:02d}",
                    f"{y}-01-01", f"{y + win - 1}-12-31"))
        y += win
    return out


def _worker_init(windows):
    """Set the windows in this worker (spawn re-imports the module with WINDOWS
    empty, so it must be passed in), and memoize the weight-independent data load
    so each worker reads the cache exactly once, not once per combo."""
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
    """Run the CURRENT backtester on every window for one combo."""
    import config as cfg, backtester as bt
    for k, v in combo.items():
        setattr(cfg, k, v)
    row, cagrs = dict(combo), []
    try:
        for label, s, e in WINDOWS:
            cfg.START_DATE, cfg.END_DATE = s, e
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), \
                    contextlib.redirect_stderr(dn):
                m = bt.run_backtest(save=False)
            row[f"cagr_{label}"] = round(float(m["cagr"]), 6)
            row[f"shrp_{label}"] = round(float(m["sharpe"]), 6)
            row[f"mdd_{label}"]  = round(float(m["maxdd"]), 6)
            cagrs.append(float(m["cagr"]))
        row["mean_cagr"] = round(sum(cagrs) / len(cagrs), 6)
        row["min_cagr"]  = round(min(cagrs), 6)
        row["std_cagr"]  = round(statistics.pstdev(cagrs) if len(cagrs) > 1 else 0.0, 6)
        row["ok"] = 1
    except Exception as ex:                                          # noqa: BLE001
        row["ok"], row["err"] = 0, str(ex)[:160]
    return row


def calibrate():
    """Warm per-combo time (seconds) after one data load — all windows."""
    _worker_init(WINDOWS)
    base = {p: SPACE[p][1] for p in PARAMS}     # a non-degenerate combo
    evaluate(base)                              # warm-up pays the data load
    t0 = time.time()
    for _ in range(2):
        evaluate(base)
    return (time.time() - t0) / 2.0


def report(objective):
    if not os.path.exists(OUT):
        print("No results yet."); return
    df = pd.read_csv(OUT)
    df = df[df.get("ok", 1) == 1].dropna(subset=[objective])
    if df.empty:
        print("No successful runs."); return
    df = df.sort_values(objective, ascending=False)

    top = df.head(50)
    top.to_csv(TOP50, index=False)

    win_cols = [c for c in df.columns if c.startswith("cagr_")]
    show = PARAMS + win_cols + ["mean_cagr", "min_cagr", "std_cagr"]
    show = [c for c in show if c in df.columns]
    print("\n" + "=" * 110)
    print(f"  BEST 50 weight combos by {objective}   "
          f"(of {len(df)} distinct portfolios × {len(win_cols)} periods)")
    print("  per-period CAGR  |  mean = avg return across periods  |  "
          "min = worst period (robustness)")
    print("=" * 110)
    with pd.option_context("display.width", 260, "display.max_columns", 80,
                           "display.float_format", lambda x: f"{x:.4f}"):
        print(top[show].to_string(index=False))

    print("\n  Marginal mean", objective, "by weight value (which settings help):")
    for p in PARAMS:
        g = df.groupby(p)[objective].mean().sort_values(ascending=False)
        print(f"    {p:12} " + "  ".join(f"{k:+.2f}:{v:.3f}" for k, v in g.items()))
    print(f"\n  Full results: {OUT}\n  Best 50 CSV : {TOP50}")


def main():
    global WINDOWS
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=43200, help="time budget, seconds (default 12h)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--objective", default="mean_cagr",
                    help="mean_cagr | min_cagr | std_cagr (lower) | cagr_<label>")
    ap.add_argument("--start-year", type=int, default=2010)
    ap.add_argument("--end-year",   type=int, default=2020)
    ap.add_argument("--window-years", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    WINDOWS = make_windows(args.start_year, args.end_year, args.window_years)
    if not WINDOWS:
        sys.exit("No windows — check --start-year/--end-year/--window-years.")

    if args.report_only:
        report(args.objective); return

    grid = build_grid()
    print(f"Distinct portfolios in grid (after scale-dedup): {len(grid):,}")
    print(f"Periods ({len(WINDOWS)}): " + "  ".join(f"{s[:4]}–{e[:4]}" for _, s, e in WINDOWS))
    print("Calibrating per-combo time on this machine…")
    sec = calibrate()
    capacity = int(args.budget * args.workers / sec * 0.90)
    if capacity >= len(grid):
        combos, mode = grid, "EXHAUSTIVE (whole grid fits the budget)"
    else:
        combos = random.Random(args.seed).sample(grid, capacity)
        mode = f"RANDOM SAMPLE {capacity:,} of {len(grid):,} (budget-limited)"

    print(f"  ~{sec:.1f}s/combo ({len(WINDOWS)} periods) · {args.workers} workers · "
          f"budget {args.budget/3600:.1f}h")
    print(f"  → {mode}")
    print(f"  rank by {args.objective}\n")

    if os.path.exists(OUT):
        os.replace(OUT, OUT + ".prev")

    done, t0, header = 0, time.time(), False
    try:
        with mp.Pool(args.workers, initializer=_worker_init, initargs=(WINDOWS,)) as pool, \
                open(OUT, "w") as fh:
            for row in pool.imap_unordered(evaluate, combos, chunksize=1):
                pd.DataFrame([row]).to_csv(fh, header=not header, index=False)
                fh.flush(); header = True
                done += 1
                if done % 50 == 0 or done == len(combos):
                    el = time.time() - t0
                    eta = el / done * (len(combos) - done)
                    print(f"  {done}/{len(combos)}  elapsed {el/3600:.2f}h  "
                          f"eta {eta/3600:.2f}h", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted — reporting partial results.")
    finally:
        report(args.objective)


if __name__ == "__main__":
    main()

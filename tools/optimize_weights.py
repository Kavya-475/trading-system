"""
tools/optimize_weights.py  —  walk-forward search over the SCORING WEIGHTS
===========================================================================
Extensively varies the five scoring weights only —

    W_MOM_12M   W_MOM_6M   W_MOM_3M   W_VOL   W_DIST_DMA

— holding every other parameter at its config.py value, and drives the EXISTING
backtester (backtester.run_backtest). No backtest logic is re-implemented.

Walk-forward (what the user asked for)
--------------------------------------
Each weight combo is run on TWO consecutive windows:
  • TRAIN  : a 2-year window  (the "optimize for two years")
  • FORWARD: the next 2 years (the "then run for 2" out-of-sample test)
The combo's TRAIN numbers and its FORWARD numbers are both recorded, so you can
see which combos that looked good in-sample actually held up forward. The final
table is the **best 50 combos ranked by FORWARD return (CAGR)**, with the
train→forward gap shown alongside (a big positive gap = overfit to the train).

Default windows are fully pre-2020 (bias-free bhavcopy era, before the stale
post-2020 universe): TRAIN 2016-01-01→2017-12-31, FORWARD 2018-01-01→2019-12-31.
Override with --train-start / --train-years / --test-years.

Throughput (the "max backtests in 12h" ask)
--------------------------------------------
  • Every worker loads the price cache ONCE (memoized) — not once per combo.
  • Parallel across all CPU cores.
  • SCALE-DEDUP: a momentum score is rank-invariant to multiplying *all* weights
    by a positive constant, so (0.4,0.5,0.2,0,0.3) and (0.8,1.0,0.4,0,0.6) give
    the identical portfolio. Combos are collapsed to a canonical (L1-normalised)
    key so no two redundant, rank-equivalent backtests are ever run — every
    backtest in the budget is informationally distinct.
  • SELF-SIZING: times a warm run, reads the core count, and runs the whole
    deduped grid if it fits the budget, else a random sample that does.
  • Streams each result to wfo_results.csv as it lands; reports partials on Ctrl-C.

Run on the VM (12-hour budget, all cores):
    python tools/optimize_weights.py --budget 43200

    python tools/optimize_weights.py --train-start 2014-01-01            # different fold
    python tools/optimize_weights.py --objective fwd_sharpe              # rank by risk-adj
    python tools/optimize_weights.py --report-only                       # re-print tables
"""
import argparse
import contextlib
import os
import random
import sys
import time
import multiprocessing as mp

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
OUT   = os.path.join(HERE, "wfo_results.csv")
TOP50 = os.path.join(HERE, "wfo_top50.csv")

# Filled in by main() from CLI args, read by the workers.
TRAIN = ("2016-01-01", "2017-12-31")
FWD   = ("2018-01-01", "2019-12-31")

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
    (b) scale-duplicates collapsed by canonical key. Returns one representative
    combo per distinct portfolio."""
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


def _worker_init():
    """Memoize the (weight-independent) data load: each worker reads the cache
    exactly once, not once per combo."""
    import backtester as bt
    orig, cache = bt.load_data, {}
    def cached():
        if "d" not in cache:
            cache["d"] = orig()
        return cache["d"]
    bt.load_data = cached


def evaluate(combo):
    """Run the CURRENT backtester on the train + forward windows for one combo."""
    import config as cfg, backtester as bt
    for k, v in combo.items():
        setattr(cfg, k, v)
    row = dict(combo)
    try:
        for tag, (s, e) in (("tr", TRAIN), ("fwd", FWD)):
            cfg.START_DATE, cfg.END_DATE = s, e
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), \
                    contextlib.redirect_stderr(dn):
                m = bt.run_backtest(save=False)
            for kk, vv in m.items():
                row[f"{tag}_{kk}"] = round(float(vv), 6)
        row["gap_cagr"] = round(row["tr_cagr"] - row["fwd_cagr"], 6)  # overfit tell
        row["ok"] = 1
    except Exception as ex:                                          # noqa: BLE001
        row["ok"], row["err"] = 0, str(ex)[:160]
    return row


def calibrate():
    """Warm per-combo time (seconds) after one data load — both windows."""
    _worker_init()
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

    show = PARAMS + ["tr_cagr", "fwd_cagr", "gap_cagr", "fwd_sharpe",
                     "fwd_maxdd", "fwd_edge_cagr", "fwd_calmar"]
    show = [c for c in show if c in df.columns]
    print("\n" + "=" * 96)
    print(f"  BEST 50 weight combos by {objective}   "
          f"(of {len(df)} distinct portfolios evaluated)")
    print(f"  TRAIN {TRAIN[0]}→{TRAIN[1]}   FORWARD {FWD[0]}→{FWD[1]}   "
          f"gap_cagr = train−forward (large + = overfit)")
    print("=" * 96)
    with pd.option_context("display.width", 240, "display.max_columns", 60,
                           "display.float_format", lambda x: f"{x:.4f}"):
        print(top[show].to_string(index=False))

    print("\n  Marginal mean", objective, "by weight value (which settings help):")
    for p in PARAMS:
        g = df.groupby(p)[objective].mean().sort_values(ascending=False)
        print(f"    {p:12} " + "  ".join(f"{k:+.2f}:{v:.3f}" for k, v in g.items()))
    print(f"\n  Full results: {OUT}\n  Best 50 CSV : {TOP50}")


def _shift_years(d, n):
    y, m, day = (int(x) for x in d.split("-"))
    return f"{y + n:04d}-{m:02d}-{day:02d}"


def main():
    global TRAIN, FWD
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=43200, help="time budget, seconds (default 12h)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--objective", default="fwd_cagr",
                    help="fwd_cagr | fwd_sharpe | fwd_calmar | fwd_edge_cagr | fwd_sortino")
    ap.add_argument("--train-start", default="2016-01-01")
    ap.add_argument("--train-years", type=int, default=2)
    ap.add_argument("--test-years",  type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    ts = args.train_start
    te = _shift_years(ts, args.train_years)            # exclusive boundary →
    fe = _shift_years(te, args.test_years)
    # windows are inclusive end-dates one day before the next window starts
    def _minus1d(d):
        from datetime import datetime, timedelta
        return (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    TRAIN = (ts, _minus1d(te))
    FWD   = (te, _minus1d(fe))

    if args.report_only:
        report(args.objective); return

    grid = build_grid()
    print(f"Distinct portfolios in grid (after scale-dedup): {len(grid):,}")
    print("Calibrating per-combo time on this machine…")
    sec = calibrate()
    capacity = int(args.budget * args.workers / sec * 0.90)
    if capacity >= len(grid):
        combos = grid                                  # exhaustive — run them all
        mode = "EXHAUSTIVE (whole grid fits the budget)"
    else:
        rng = random.Random(args.seed)
        combos = rng.sample(grid, capacity)            # random subset that fits
        mode = f"RANDOM SAMPLE {capacity:,} of {len(grid):,} (budget-limited)"

    print(f"  ~{sec:.1f}s/combo (train+forward) · {args.workers} workers · "
          f"budget {args.budget/3600:.1f}h")
    print(f"  → {mode}")
    print(f"  TRAIN {TRAIN[0]}→{TRAIN[1]}  |  FORWARD {FWD[0]}→{FWD[1]}  |  "
          f"rank by {args.objective}\n")

    if os.path.exists(OUT):
        os.replace(OUT, OUT + ".prev")

    done, t0, header = 0, time.time(), False
    try:
        with mp.Pool(args.workers, initializer=_worker_init) as pool, open(OUT, "w") as fh:
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

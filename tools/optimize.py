"""
tools/optimize.py  —  parameter optimizer (random search, train/validate)
=========================================================================
Searches the critical strategy parameters for a better configuration, driving
the EXISTING backtester (backtester.run_backtest) — no re-implementation of the
backtest logic. Every candidate is scored on an OUT-OF-SAMPLE validation window,
because optimizing on a single in-sample window is how you overfit.

Design
------
- TRAIN window 2007–2014 (incl. the 2008 bear), VALIDATE window 2015–2019.
  Both pre-2020, to avoid the stale post-2020 universe. Ranking is by the
  VALIDATION objective; the train number is reported so you can see the
  train→validate gap (overfitting tell).
- Random search over the joint parameter space (covers interactions; scales to a
  time budget far better than a full grid).
- Parallel across CPU cores; each worker loads the price cache ONCE (memoized).
- SELF-SIZING: times a warm calibration run, reads core count, and picks how many
  combos to sample so the whole run finishes within --budget seconds (default 1h).
- Resilient: writes each result to optimize_results.csv as it completes, and
  prints the ranking even if interrupted.

Run on the VM:
    python tools/optimize.py                      # ~1 hour, all cores
    python tools/optimize.py --budget 3600 --workers 6 --objective va_sharpe
    python tools/optimize.py --objective va_calmar   # rank by CAGR/|maxDD| instead
"""
import argparse
import contextlib
import itertools
import os
import random
import sys
import time
import multiprocessing as mp

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "optimize_results.csv")

TRAIN = ("2007-01-01", "2014-12-31")   # includes the 2008 GFC
VAL   = ("2015-01-01", "2019-12-31")   # out-of-sample, still pre-2020

# ── Parameter search space (the critical knobs) ─────────────────────────────
SPACE = {
    "TOP_N":              [8, 10, 12, 15, 20],
    "MAX_PER_SECTOR":     [3, 4, 5],
    "EXIT_RANK_CUTOFF":   [20, 25, 30, 40],
    "W_MOM_12M":          [0.4, 0.5, 0.6, 0.7],
    "W_MOM_6M":           [0.2, 0.3, 0.4, 0.5],
    "W_MOM_3M":           [0.1, 0.2, 0.3, 0.4],
    "W_VOL":              [0.0, -0.1, -0.2],
    "W_DIST_DMA":         [0.0, 0.1, 0.2, 0.3],
    "SKIP_RECENT":        [15, 20, 25],
    "DMA_EXIT":           [150, 200, 250],
    "LOOKBACK_12M":       [250, 280, 300],
    "POSITION_WEIGHTING": ["equal", "tiered"],
    "RANK_TIER_WEIGHTS":  [(1.5, 1.1, 0.7), (2.0, 1.0, 0.5), (1.3, 1.0, 0.8)],
    "MIN_AVG_VALUE_CR":   [0.1, 1.0, 5.0],
    "BACKTEST_FORCE_RISK_ON": [False, True],   # False = regime filter ON (live)
}
PARAMS = list(SPACE.keys())


def _worker_init():
    """Memoize the (param-independent) data load so each worker reads the cache
    only once, not once per combo."""
    import backtester as bt
    orig, cache = bt.load_data, {}
    def cached():
        if "d" not in cache:
            cache["d"] = orig()
        return cache["d"]
    bt.load_data = cached


def evaluate(combo):
    """Run the CURRENT backtester on train + validate windows for one combo."""
    import config as cfg, backtester as bt
    for k, v in combo.items():
        setattr(cfg, k, v)
    row = dict(combo)
    try:
        for tag, (s, e) in (("tr", TRAIN), ("va", VAL)):
            cfg.START_DATE, cfg.END_DATE = s, e
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
                m = bt.run_backtest(save=False)
            for kk, vv in m.items():
                row[f"{tag}_{kk}"] = round(float(vv), 6)
        row["ok"] = 1
    except Exception as ex:
        row["ok"], row["err"] = 0, str(ex)[:120]
    return row


def sample_combos(n, seed):
    rng = random.Random(seed)
    seen, out = set(), []
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        c = {p: rng.choice(SPACE[p]) for p in PARAMS}
        key = tuple(c[p] for p in PARAMS)
        if key not in seen:
            seen.add(key); out.append(c)
    return out


def calibrate():
    """Warm per-combo time (seconds), measured in-process after one data load."""
    _worker_init()
    import config as cfg, backtester as bt  # noqa
    base = {p: SPACE[p][0] for p in PARAMS}
    evaluate(base)                  # warm-up (pays the one-time data load)
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
    cols = PARAMS + ["tr_sharpe", "va_sharpe", "va_cagr", "va_maxdd", "va_edge_cagr", "va_calmar"]
    cols = [c for c in cols if c in df.columns]
    print("\n" + "=" * 70)
    print(f"  TOP 15 by {objective}  (of {len(df)} evaluated)   train→val gap = overfit tell")
    print("=" * 70)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df[cols].head(15).to_string(index=False))
    # Per-parameter marginal effect on the validation objective
    print("\n  Marginal mean", objective, "by parameter value (which settings help):")
    for p in PARAMS:
        g = df.groupby(p)[objective].mean().sort_values(ascending=False)
        best = ", ".join(f"{k}={v:.2f}" for k, v in g.items())
        print(f"    {p:22} {best}")
    print(f"\n  Full results: {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=3600, help="time budget in seconds (~1h)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--objective", default="va_sharpe",
                    help="va_sharpe | va_calmar | va_edge_cagr | va_cagr | va_sortino")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        report(args.objective); return

    print(f"Calibrating per-combo time on this machine…")
    sec = calibrate()
    n = max(args.workers * 4, int(args.budget * args.workers / sec * 0.85))
    total_space = 1
    for p in PARAMS:
        total_space *= len(SPACE[p])
    n = min(n, total_space)
    print(f"  ~{sec:.1f}s/combo (train+val) · {args.workers} workers · budget {args.budget/60:.0f} min")
    print(f"  → sampling {n:,} random combos from a {total_space:,}-point space")
    print(f"  TRAIN {TRAIN[0]}→{TRAIN[1]}  |  VALIDATE {VAL[0]}→{VAL[1]}  |  rank by {args.objective}\n")

    combos = sample_combos(n, args.seed)
    if os.path.exists(OUT):
        os.replace(OUT, OUT + ".prev")

    done, t0, header = 0, time.time(), False
    try:
        with mp.Pool(args.workers, initializer=_worker_init) as pool:
            with open(OUT, "w") as fh:
                for row in pool.imap_unordered(evaluate, combos, chunksize=1):
                    df1 = pd.DataFrame([row])
                    df1.to_csv(fh, header=not header, index=False)
                    fh.flush(); header = True
                    done += 1
                    if done % 25 == 0 or done == len(combos):
                        el = time.time() - t0
                        eta = el / done * (len(combos) - done)
                        print(f"  {done}/{len(combos)}  elapsed {el/60:.1f}m  eta {eta/60:.1f}m", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted — reporting partial results.")
    finally:
        report(args.objective)


if __name__ == "__main__":
    main()

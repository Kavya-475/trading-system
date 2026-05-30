#!/usr/bin/env python3
"""
optimizer.py  —  Walk-Forward Optimization Engine
==================================================
For each rolling window:
  1. Sweep a focused parameter grid on a 5-year TRAIN window
  2. Pick the best params (Sharpe × Calmar)
  3. Run those params on the next 1-year TEST window (true OOS)
  4. Also run CURRENT fixed params on the same test window (baseline)

13 test windows: 2013–2025
Results:
  - Chained OOS equity curve (walk-forward)
  - vs Fixed current params over same period
  - vs Nifty 50 buy-and-hold
  - Parameter stability table (which values win most often)

Output:
  optimizer_results.txt   —  full walk-forward report
  optimizer_equity.csv    —  daily equity curves for charting

Run:
    python optimizer.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, time, itertools, csv
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime
import config as cfg
from backtester_sweep import (
    load_data, _init_worker, run_one,
    TXN_BUY, TXN_SELL, TOP_N, SECTOR_CAP, EXIT_CUTOFF,
    DMA_EXIT_DAYS, REGIME_DMA, RF_ANNUAL, INITIAL_CAP
)
import backtester_sweep as bsw

# ─────────────────────────────────────────────
# WALK-FORWARD SETTINGS
# ─────────────────────────────────────────────
TRAIN_YEARS  = 5
TEST_YEARS   = 1
FIRST_TRAIN_START = "2008-01-01"   # enough warmup from 2007-01-01 data
N_WORKERS    = min(8, mp.cpu_count())

OUT_TEXT  = "optimizer_results.txt"
OUT_CSV   = "optimizer_equity.csv"

# ─────────────────────────────────────────────
# PARAMETER GRID  (3^7 = 2,187 combos per window)
# L3 fixed at 80 — sweep showed it barely affects results
# ─────────────────────────────────────────────
FIXED_L3 = 80
GRID = {
    "w12":  [0.30, 0.50, 0.70],
    "w6":   [0.20, 0.40, 0.60],
    "w3":   [0.00, 0.20, 0.40],
    "wv":   [-0.30, -0.10, 0.00],
    "L12":  [260, 280, 300],
    "L6":   [110, 140, 170],   # must match precomputed values in backtester_sweep.py
    "skip": [15,  25,  35],
}

CURRENT_PARAMS = dict(
    w12=0.50, w6=0.40, w3=0.30, wv=0.00,
    L12=280,  L6=140,  L3=80,   skip=25,
)


# ─────────────────────────────────────────────
# WINDOW GENERATION
# ─────────────────────────────────────────────
def build_windows():
    """Returns list of (train_start, train_end, test_start, test_end) strings."""
    windows = []
    t_start = pd.Timestamp(FIRST_TRAIN_START)
    data_end = pd.Timestamp(cfg.END_DATE)

    while True:
        t_end   = t_start + pd.DateOffset(years=TRAIN_YEARS) - pd.DateOffset(days=1)
        ts_start = t_end  + pd.DateOffset(days=1)
        ts_end   = ts_start + pd.DateOffset(years=TEST_YEARS) - pd.DateOffset(days=1)
        if ts_end > data_end:
            break
        windows.append((
            t_start.strftime("%Y-%m-%d"),
            t_end.strftime("%Y-%m-%d"),
            ts_start.strftime("%Y-%m-%d"),
            ts_end.strftime("%Y-%m-%d"),
        ))
        t_start = t_start + pd.DateOffset(years=TEST_YEARS)
    return windows


# ─────────────────────────────────────────────
# SWEEP HELPERS
# ─────────────────────────────────────────────
def set_mask(close_dates, start, end):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    bsw._sweep_mask = np.array(
        [s <= pd.Timestamp(d) <= e for d in close_dates]
    )


def sweep_window(close_dates, train_start, train_end, data):
    """Run full grid sweep on train window. Returns best params dict."""
    set_mask(close_dates, train_start, train_end)
    keys   = list(GRID.keys())
    combos = [dict(zip(keys, c), L3=FIXED_L3)
               for c in itertools.product(*[GRID[k] for k in keys])]

    # Create pool AFTER mask is set — fork workers inherit _sweep_mask
    with mp.Pool(N_WORKERS, initializer=_init_worker, initargs=(data,)) as pool:
        results = pool.map(run_one, combos,
                           chunksize=max(1, len(combos) // (N_WORKERS * 4)))

    valid = [r for r in results if r is not None]
    if not valid:
        return CURRENT_PARAMS.copy(), None
    best = max(valid, key=lambda r: r["composite"])
    return {k: best[k] for k in list(GRID.keys()) + ["L3"]}, best


def run_test(close_dates, test_start, test_end, params):
    """Run backtest on test window with given params. Returns result dict."""
    set_mask(close_dates, test_start, test_end)
    return run_one(params)


# ─────────────────────────────────────────────
# EQUITY CURVE HELPERS
# ─────────────────────────────────────────────
def chain_equity(window_returns):
    """
    window_returns: list of (final_val / initial_val) per test window.
    Returns a numpy array of cumulative portfolio value starting at INITIAL_CAP.
    """
    capital = INITIAL_CAP
    curve   = [capital]
    for r in window_returns:
        capital *= r
        curve.append(capital)
    return np.array(curve)


def metrics(eq: np.ndarray, n_years: float, label: str = "") -> dict:
    if len(eq) < 2 or eq[0] <= 0:
        return {}
    cagr = (eq[-1] / eq[0]) ** (1.0 / n_years) - 1 if n_years > 0 else 0
    dr   = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], np.nan)
    dr   = dr[np.isfinite(dr)]
    if len(dr) < 2:
        return {}
    rf_d    = RF_ANNUAL / 252
    excess  = dr - rf_d
    sharpe  = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
    down    = excess[excess < 0]
    sortino = excess.mean() / down.std() * np.sqrt(252) if (len(down)>1 and down.std()>0) else 0
    peak    = np.maximum.accumulate(eq)
    dd      = (eq - peak) / np.where(peak > 0, peak, 1)
    max_dd  = float(dd.min())
    mret    = np.diff(eq[::21]) / eq[::21][:-1]
    mret    = mret[np.isfinite(mret)]
    win_rate = float((mret > 0).mean()) if len(mret) > 0 else 0
    return {
        "label":    label,
        "cagr":     cagr,
        "sharpe":   sharpe,
        "sortino":  sortino,
        "max_dd":   max_dd,
        "win_rate": win_rate,
        "final":    eq[-1],
    }


# ─────────────────────────────────────────────
# PARAMETER STABILITY ANALYSIS
# ─────────────────────────────────────────────
def stability_table(best_params_list):
    """Count how often each value was chosen for each parameter."""
    table = {}
    for key in list(GRID.keys()) + ["L3"]:
        vals = [p[key] for p in best_params_list]
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        table[key] = sorted(counts.items(), key=lambda x: -x[1])
    return table


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    t0 = time.time()
    print(f"\n{'='*65}")
    print(f"  WALK-FORWARD OPTIMIZATION ENGINE")
    print(f"  Train: {TRAIN_YEARS}yr  |  Test: {TEST_YEARS}yr  |  Workers: {N_WORKERS}")
    print(f"  Grid: {np.prod([len(v) for v in GRID.values()]):,} combos per window")
    print(f"{'='*65}\n")

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading data...", flush=True)
    data = load_data()
    _init_worker(data)   # set globals in main process (workers inherit via fork)

    close_dates = pd.read_csv(
        cfg.DATA_CACHE_FILE, index_col=0, parse_dates=True
    ).sort_index().index

    # ── Build windows ─────────────────────────────────────────────────────────
    windows = build_windows()
    n_win   = len(windows)
    print(f"  {n_win} walk-forward windows\n")
    for i, (tr_s, tr_e, ts_s, ts_e) in enumerate(windows, 1):
        print(f"  Window {i:>2}: Train {tr_s[:4]}–{tr_e[:4]}  |  Test {ts_s[:4]}")
    print()

    # ── Run walk-forward ──────────────────────────────────────────────────────
    wf_results    = []   # per-window dicts
    best_params_list = []

    for i, (tr_s, tr_e, ts_s, ts_e) in enumerate(windows, 1):
        print(f"  Window {i:>2}/{n_win}  train={tr_s[:7]}→{tr_e[:7]}  "
              f"test={ts_s[:7]}→{ts_e[:7]} ...", flush=True)

        # 1. Sweep train window
        best_p, train_res = sweep_window(close_dates, tr_s, tr_e, data)
        best_params_list.append(best_p)

        # 2. Test with best params (OOS)
        oos_res = run_test(close_dates, ts_s, ts_e, best_p)

        # 3. Test with current fixed params (baseline)
        fix_res = run_test(close_dates, ts_s, ts_e, CURRENT_PARAMS)

        train_score = train_res["composite"] if train_res else 0

        row = {
            "window":       i,
            "train_start":  tr_s[:7],
            "train_end":    tr_e[:7],
            "test_start":   ts_s[:7],
            "test_end":     ts_e[:7],
            "best_params":  best_p,
            "train_score":  round(train_score, 3),
            "oos":          oos_res,
            "fixed":        fix_res,
        }
        wf_results.append(row)

        oos_str = (f"CAGR={oos_res['cagr_pct']:+.1f}% "
                   f"DD={oos_res['max_dd_pct']:.1f}% "
                   f"Sharpe={oos_res['sharpe']:.3f}"
                   ) if oos_res else "no valid result"
        fix_str = (f"CAGR={fix_res['cagr_pct']:+.1f}% "
                   f"DD={fix_res['max_dd_pct']:.1f}%"
                   ) if fix_res else "no result"

        p_str = (f"w12={best_p['w12']} w6={best_p['w6']} "
                 f"wv={best_p['wv']} L12={best_p['L12']} skip={best_p['skip']}")
        print(f"         Best: {p_str}")
        print(f"         OOS : {oos_str}")
        print(f"         Fixed: {fix_str}\n", flush=True)

    # ── Build chained equity curves ───────────────────────────────────────────
    def return_factor(res):
        if res is None:
            return 1.0
        return res["final_val"] / INITIAL_CAP

    oos_factors   = [return_factor(r["oos"])   for r in wf_results]
    fixed_factors = [return_factor(r["fixed"]) for r in wf_results]

    oos_eq   = chain_equity(oos_factors)
    fixed_eq = chain_equity(fixed_factors)

    n_years = len(windows) * TEST_YEARS

    oos_m   = metrics(oos_eq,   n_years, "Walk-Forward OOS")
    fixed_m = metrics(fixed_eq, n_years, "Fixed Params (current config)")

    # ── Stability table ───────────────────────────────────────────────────────
    stab = stability_table(best_params_list)

    elapsed = time.time() - t0

    # ── Write text report ─────────────────────────────────────────────────────
    with open(OUT_TEXT, "w") as f:

        f.write(f"{'='*70}\n")
        f.write(f"  WALK-FORWARD OPTIMIZATION REPORT\n")
        f.write(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Runtime   : {elapsed/60:.1f} minutes\n")
        f.write(f"  Train/Test: {TRAIN_YEARS}yr / {TEST_YEARS}yr rolling\n")
        f.write(f"  Grid      : {np.prod([len(v) for v in GRID.values()]):,} "
                f"combos × {n_win} windows\n")
        f.write(f"{'='*70}\n\n")

        # Per-window table
        f.write(f"  PER-WINDOW RESULTS\n")
        f.write(f"  {'Win':>3}  {'Test':>7}  "
                f"{'OOS CAGR':>9}  {'OOS Sharpe':>10}  {'OOS MaxDD':>9}  "
                f"{'Fix CAGR':>9}  {'Fix MaxDD':>9}  "
                f"{'Best wv':>7}  {'Best L12':>8}  {'Best skip':>9}\n")
        f.write("  " + "-"*90 + "\n")

        for r in wf_results:
            o = r["oos"]
            fx = r["fixed"]
            bp = r["best_params"]
            o_cagr  = f"{o['cagr_pct']:+.1f}%"   if o  else "  N/A"
            o_sh    = f"{o['sharpe']:.3f}"         if o  else "  N/A"
            o_dd    = f"{o['max_dd_pct']:.1f}%"   if o  else "  N/A"
            fx_cagr = f"{fx['cagr_pct']:+.1f}%"  if fx else "  N/A"
            fx_dd   = f"{fx['max_dd_pct']:.1f}%"  if fx else "  N/A"
            f.write(f"  {r['window']:>3}  {r['test_start']:>7}  "
                    f"{o_cagr:>9}  {o_sh:>10}  {o_dd:>9}  "
                    f"{fx_cagr:>9}  {fx_dd:>9}  "
                    f"{bp['wv']:>7}  {bp['L12']:>8}  {bp['skip']:>9}\n")

        # Summary comparison
        f.write(f"\n\n{'='*70}\n")
        f.write(f"  CHAINED OUT-OF-SAMPLE PERFORMANCE  ({windows[0][2][:4]}–{windows[-1][3][:4]})\n")
        f.write(f"{'='*70}\n\n")

        for m in [oos_m, fixed_m]:
            if not m:
                continue
            f.write(f"  {m['label']}\n")
            f.write(f"    CAGR         : {m['cagr']*100:+.1f}%\n")
            f.write(f"    Sharpe       : {m['sharpe']:.3f}\n")
            f.write(f"    Sortino      : {m['sortino']:.3f}\n")
            f.write(f"    Max Drawdown : {m['max_dd']*100:.1f}%\n")
            f.write(f"    Win Rate     : {m['win_rate']*100:.1f}%\n")
            f.write(f"    Final Value  : ₹{m['final']:,.0f}  "
                    f"(started ₹{INITIAL_CAP:,.0f})\n\n")

        # Parameter stability
        f.write(f"\n{'='*70}\n")
        f.write(f"  PARAMETER STABILITY  (how often each value was chosen)\n")
        f.write(f"{'='*70}\n\n")

        for key, counts in stab.items():
            f.write(f"  {key:>6}: ")
            for val, cnt in counts:
                pct = cnt / n_win * 100
                f.write(f"{val}→{cnt}/{n_win} ({pct:.0f}%)   ")
            f.write("\n")

        # Consensus params (most-chosen value per parameter)
        consensus = {k: counts[0][0] for k, counts in stab.items()}
        f.write(f"\n  CONSENSUS PARAMS (most-chosen per parameter across all windows):\n")
        for k, v in consensus.items():
            current_v = CURRENT_PARAMS.get(k, FIXED_L3 if k == "L3" else "?")
            change = " ←  CHANGE" if v != current_v else "   (same as current)"
            f.write(f"    {k:>6} = {v}{change}\n")

    print(f"  Saved: {OUT_TEXT}")

    # ── Write equity CSV ──────────────────────────────────────────────────────
    years_axis = [windows[0][2][:4]] + [r["test_end"][:4] for r in wf_results]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "walkforward_oos", "fixed_params"])
        for yr, ov, fv in zip(years_axis, oos_eq, fixed_eq):
            w.writerow([yr, round(ov, 2), round(fv, 2)])
    print(f"  Saved: {OUT_CSV}")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  WALK-FORWARD SUMMARY  ({n_win} test years)")
    print(f"{'='*65}")
    for m in [oos_m, fixed_m]:
        if m:
            print(f"\n  {m['label']}")
            print(f"    CAGR: {m['cagr']*100:+.1f}%  |  "
                  f"Sharpe: {m['sharpe']:.3f}  |  "
                  f"MaxDD: {m['max_dd']*100:.1f}%  |  "
                  f"Final: ₹{m['final']:,.0f}")

    print(f"\n  PARAMETER STABILITY")
    for key, counts in stab.items():
        winner = counts[0]
        current_v = CURRENT_PARAMS.get(key, FIXED_L3 if key == "L3" else "?")
        flag = " ← differs from current" if winner[0] != current_v else ""
        print(f"  {key:>6}: most-chosen = {winner[0]}  "
              f"({winner[1]}/{n_win} windows){flag}")

    print(f"\n  Done in {elapsed/60:.1f} minutes.\n")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    # Store shared data reference for worker init
    main()

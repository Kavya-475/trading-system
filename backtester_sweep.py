#!/usr/bin/env python3
"""
backtester_sweep.py
===================
VM-optimised parameter sweep  (8 vCPU / 32 GB  —  t2d-standard-8).

Daily-scored momentum backtest over a 2008-01-01 → 2015-12-31 training
window.  Results are ranked by Sharpe × Calmar.

Parameters swept  (5 values each  →  5^8 = 390,625 combinations):
  Weights  : W_MOM_12M, W_MOM_6M, W_MOM_3M, W_VOL
  Lookbacks: LOOKBACK_12M, LOOKBACK_6M, LOOKBACK_3M, SKIP_RECENT

Fixed (not swept):
  TOP_N=10, MAX_PER_SECTOR=3, EXIT_RANK_CUTOFF=25, DMA_EXIT=250

Run in background on the VM:
    nohup python backtester_sweep.py > sweep_vm.log 2>&1 &
    tail -f sweep_vm.log          # monitor progress

Output:
    sweep_vm_results.csv     — all valid combos, sorted by Sharpe × Calmar
    sweep_vm_top100.txt      — human-readable top-100 + sensitivity table
"""

import os, sys, time, csv, itertools, warnings
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# SWEEP SETTINGS
# ─────────────────────────────────────────────
SWEEP_START  = "2008-01-01"
SWEEP_END    = "2015-12-31"
N_WORKERS    = 8
INITIAL_CAP  = 100_000.0

# Fixed strategy constants
TOP_N         = 10
SECTOR_CAP    = 3
EXIT_CUTOFF   = 25
DMA_EXIT_DAYS = 250
REGIME_DMA    = 200
RF_ANNUAL     = 0.065   # India 10yr G-Sec

# Simplified transaction costs (no tax — fine for relative ranking)
TXN_BUY  = 0.00165   # stamp + exchange + SEBI on buy
TXN_SELL = 0.00170   # STT + exchange + SEBI on sell

OUT_CSV  = "sweep_vm_results.csv"
OUT_TEXT = "sweep_vm_top100.txt"

# ─────────────────────────────────────────────
# PARAMETER GRID   (5^8 = 390,625 combinations)
# Estimated runtime on VM: ~40 minutes (8 vCPU, vectorised scoring)
# ─────────────────────────────────────────────
GRID = {
    "w12":  [0.10, 0.25, 0.40, 0.55, 0.70],   # 12-month momentum weight  (current: 0.50)
    "w6":   [0.10, 0.25, 0.40, 0.55, 0.70],   # 6-month  momentum weight  (current: 0.40)
    "w3":   [0.00, 0.10, 0.20, 0.30, 0.45],   # 3-month  momentum weight  (current: 0.30; 0 = off)
    "wv":   [-0.30, -0.20, -0.10, 0.00, 0.10],# volatility penalty        (current: 0.00)
    "L12":  [220, 250, 280, 310, 340],         # 12m lookback in days      (current: 280)
    "L6":   [110, 125, 140, 155, 170],         # 6m  lookback in days      (current: 140)
    "L3":   [55,  67,  80,  93, 105],          # 3m  lookback in days      (current: 80)
    "skip": [0,   10,  20,  30,  40],          # skip-recent days          (current: 25)
}

# ─────────────────────────────────────────────
# WORKER GLOBALS  (populated by fork)
# ─────────────────────────────────────────────
_close_arr  = None   # (N_days, N_stocks) float32  — adj-close
_above_dma  = None   # (N_days, N_stocks) bool     — close > 250-DMA
_regime_on  = None   # (N_days,) bool              — Nifty500 > 200-DMA
_month_arr  = None   # (N_days,) int               — YYYYMM for rebalance detect
_sweep_mask = None   # (N_days,) bool              — inside sweep window
_vol_arrs   = None   # {L6_val: (N_days, N_stocks) float32} — pre-computed rolling vol
_sec_idx    = None   # (N_stocks,) int             — sector index per stock
_n_stocks   = None


def _init_worker(shared: dict):
    global _close_arr, _above_dma, _regime_on, _month_arr
    global _sweep_mask, _vol_arrs, _sec_idx, _n_stocks
    _close_arr  = shared["close_arr"]
    _above_dma  = shared["above_dma"]
    _regime_on  = shared["regime_on"]
    _month_arr  = shared["month_arr"]
    _sweep_mask = shared["sweep_mask"]
    _vol_arrs   = shared["vol_arrs"]
    _sec_idx    = shared["sec_idx"]
    _n_stocks   = shared["n_stocks"]


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_data() -> dict:
    import config as cfg
    from signals import UNIVERSE

    print("Loading cached data...", flush=True)
    close_df  = pd.read_csv(cfg.DATA_CACHE_FILE, index_col=0, parse_dates=True).sort_index()
    regime_df = pd.read_csv(cfg.REGIME_CACHE,    index_col=0, parse_dates=True).sort_index()

    # Only tickers present in the cache
    tickers = [t for t in UNIVERSE if t in close_df.columns]
    sectors = [UNIVERSE[t] for t in tickers]
    n_stocks = len(tickers)

    close_df = close_df[tickers].astype(float).ffill()

    # 250-DMA exit signal (pre-computed, fixed)
    dma250    = close_df.rolling(DMA_EXIT_DAYS).mean()
    above_dma = (close_df.values > dma250.values)
    above_dma = np.nan_to_num(above_dma.astype(np.float32), nan=0).astype(bool)

    # Regime: nifty500 vs 200-DMA
    n500_col = "nifty500" if "nifty500" in regime_df.columns else regime_df.columns[0]
    n500 = regime_df[n500_col].reindex(close_df.index).ffill().values.astype(np.float32)
    n500_dma = pd.Series(n500).rolling(REGIME_DMA).mean().values.astype(np.float32)
    regime_on = (n500 > n500_dma)

    # Month tag for rebalance detection (YYYYMM)
    month_arr = np.array([
        pd.Timestamp(d).year * 100 + pd.Timestamp(d).month
        for d in close_df.index
    ], dtype=np.int32)

    # Sweep window mask
    s_start   = pd.Timestamp(SWEEP_START)
    s_end     = pd.Timestamp(SWEEP_END)
    sweep_mask = np.array([s_start <= pd.Timestamp(d) <= s_end for d in close_df.index])

    # Pre-compute rolling volatility for each unique L6 value
    daily_ret = close_df.pct_change()
    vol_arrs  = {}
    for L6 in sorted(set(GRID["L6"])):
        vol_arrs[L6] = (daily_ret.rolling(L6).std() * np.sqrt(252)).values.astype(np.float32)
    print(f"  Precomputed vol for L6 values: {sorted(vol_arrs)}", flush=True)

    # Sector encoding
    sec_names = sorted(set(sectors))
    sec_map   = {s: i for i, s in enumerate(sec_names)}
    sec_idx   = np.array([sec_map[s] for s in sectors], dtype=np.int32)
    n_sec     = len(sec_names)

    close_arr = close_df.values.astype(np.float32)

    n_sweep = sweep_mask.sum()
    print(f"  {n_stocks} stocks | {n_sweep} sweep days | "
          f"{n_sec} sectors | {len(close_arr)} total days", flush=True)

    return {
        "close_arr":  close_arr,
        "above_dma":  above_dma,
        "regime_on":  regime_on,
        "month_arr":  month_arr,
        "sweep_mask": sweep_mask,
        "vol_arrs":   vol_arrs,
        "sec_idx":    sec_idx,
        "n_stocks":   n_stocks,
        "n_sec":      n_sec,
        "tickers":    tickers,
        "sectors":    sectors,
        "sec_names":  sec_names,
    }


# ─────────────────────────────────────────────
# CORE: one backtest per parameter combo
# ─────────────────────────────────────────────
def run_one(params: dict):
    w12  = params["w12"]
    w6   = params["w6"]
    w3   = params["w3"]
    wv   = params["wv"]
    L12  = params["L12"]
    L6   = params["L6"]
    L3   = params["L3"]
    skip = params["skip"]

    S = _n_stocks
    close = _close_arr          # (N, S)
    N = close.shape[0]

    min_hist = L12 + skip       # warmup needed before first valid score

    # Day indices within sweep window that have enough history
    sweep_idx = np.where(_sweep_mask)[0]
    sweep_idx = sweep_idx[sweep_idx >= min_hist]
    T = len(sweep_idx)
    if T < 20:
        return None

    # ── Vectorised score computation (all sweep days at once) ─────────────────
    d = sweep_idx                          # (T,)

    p_now = close[d - skip].astype(np.float64)          # (T, S)
    p_12m = close[d - L12 - skip].astype(np.float64)    # (T, S)
    p_6m  = close[d - L6  - skip].astype(np.float64)    # (T, S)
    p_3m  = close[d - L3  - skip].astype(np.float64)    # (T, S)

    with np.errstate(invalid="ignore", divide="ignore"):
        m12 = (p_now - p_12m) / p_12m      # (T, S)
        m6  = (p_now - p_6m)  / p_6m
        m3  = (p_now - p_3m)  / p_3m

    # Valid mask: prices must be positive and finite
    valid = (
        np.isfinite(m12) & np.isfinite(m6) & np.isfinite(m3) &
        (p_now > 0) & (p_12m > 0) & (p_6m > 0) & (p_3m > 0)
    )

    # Cross-sectional z-score (per day)
    def czs(arr: np.ndarray) -> np.ndarray:
        masked = np.where(valid, arr, np.nan)
        mu = np.nanmean(masked, axis=1, keepdims=True)
        sd = np.nanstd(masked,  axis=1, keepdims=True)
        sd = np.where(sd > 1e-9, sd, 1.0)
        return np.nan_to_num((masked - mu) / sd)

    z12 = czs(m12)
    z6  = czs(m6)
    z3  = czs(m3)

    if abs(wv) > 1e-9:
        vol_full = _vol_arrs[L6]            # (N, S) float32
        vol_at_d = vol_full[d].astype(np.float64)   # (T, S)
        zv = czs(vol_at_d)
        score_mat = w12*z12 + w6*z6 + w3*z3 + wv*zv
    else:
        score_mat = w12*z12 + w6*z6 + w3*z3

    # Mask invalid stocks with a large negative sentinel
    score_mat = np.where(valid, score_mat, -9999.0)

    # ── Portfolio simulation ───────────────────────────────────────────────────
    cash        = INITIAL_CAP
    shares      = np.zeros(S, dtype=np.float64)
    eq_curve    = np.empty(T, dtype=np.float64)
    trades      = 0
    last_month  = -1

    for ti in range(T):
        di     = sweep_idx[ti]
        prices = close[di].astype(np.float64)
        scores = score_mat[ti]

        # Portfolio value (any position with price=0/NaN excluded)
        px_safe  = np.where(prices > 0, prices, 0.0)
        port_val = cash + float(np.dot(shares, px_safe))

        # ── RISK-OFF: liquidate all ──────────────────────────────────────────
        if not _regime_on[di]:
            held = shares > 0
            if held.any():
                px_held  = np.where(prices > 0, prices, 0.0)
                proceeds = float(np.dot(shares, px_held * held))
                cash    += proceeds * (1.0 - TXN_SELL)
                shares  *= 0.0
                trades  += int(held.sum())
            eq_curve[ti] = cash + float(np.dot(shares, px_safe))
            last_month = -1
            continue

        # ── Rank stocks by today's score ─────────────────────────────────────
        rank_order = np.argsort(-scores)   # best→worst

        # Top EXIT_CUTOFF set (for rank-exit check)
        in_top_cutoff = set(rank_order[:EXIT_CUTOFF].tolist())

        # Sector-constrained portfolio (top TOP_N)
        portfolio = []
        sec_count = {}
        for i in rank_order:
            if scores[i] <= -9000 or len(portfolio) >= TOP_N:
                break
            sc = int(_sec_idx[i])
            if sec_count.get(sc, 0) < SECTOR_CAP:
                portfolio.append(i)
                sec_count[sc] = sec_count.get(sc, 0) + 1
        portfolio_set = set(portfolio)

        # ── Daily exits: rank dropout + below DMA ────────────────────────────
        for i in range(S):
            if shares[i] <= 0 or prices[i] <= 0:
                continue
            should_exit = (i not in in_top_cutoff) or (not _above_dma[di, i])
            if should_exit:
                cash += shares[i] * prices[i] * (1.0 - TXN_SELL)
                shares[i] = 0.0
                trades += 1

        # ── Monthly rotation: sell holdings not in current top-N ─────────────
        cur_month = int(_month_arr[di])
        if cur_month != last_month:
            for i in range(S):
                if shares[i] > 0 and i not in portfolio_set and prices[i] > 0:
                    cash += shares[i] * prices[i] * (1.0 - TXN_SELL)
                    shares[i] = 0.0
                    trades += 1
            last_month = cur_month

        # ── Buy to fill target portfolio ──────────────────────────────────────
        port_val     = cash + float(np.dot(shares, px_safe))
        target_each  = port_val / TOP_N if TOP_N > 0 else 0.0

        for i in portfolio:
            px = prices[i]
            if px <= 0:
                continue
            cur_val = shares[i] * px
            if cur_val < target_each * 0.95:
                n_buy = int((target_each - cur_val) / px)
                if n_buy > 0:
                    cost = n_buy * px * (1.0 + TXN_BUY)
                    if cash >= cost:
                        cash     -= cost
                        shares[i] += n_buy
                        trades   += 1

        port_val      = cash + float(np.dot(shares, np.where(prices > 0, prices, 0.0)))
        eq_curve[ti]  = port_val

    # ── Performance metrics ───────────────────────────────────────────────────
    eq = eq_curve
    if eq[0] <= 0 or not np.all(np.isfinite(eq)):
        return None

    years = T / 252.0
    if years <= 0:
        return None

    cagr = (eq[-1] / eq[0]) ** (1.0 / years) - 1.0

    dr = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], np.nan)
    dr = dr[np.isfinite(dr)]
    if len(dr) < 10:
        return None

    rf_d    = RF_ANNUAL / 252.0
    excess  = dr - rf_d
    std_exc = excess.std()
    sharpe  = (excess.mean() / std_exc * np.sqrt(252)) if std_exc > 0 else 0.0

    down     = excess[excess < 0]
    sortino  = (excess.mean() / down.std() * np.sqrt(252)) if (len(down) > 1 and down.std() > 0) else 0.0

    peak    = np.maximum.accumulate(eq)
    dd      = (eq - peak) / np.where(peak > 0, peak, 1.0)
    max_dd  = float(dd.min())

    calmar    = cagr / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0
    composite = sharpe * calmar

    # Approximate monthly win rate (sample every ~21 trading days)
    monthly     = eq[::21]
    mret        = np.diff(monthly) / np.where(monthly[:-1] > 0, monthly[:-1], np.nan)
    mret        = mret[np.isfinite(mret)]
    win_rate    = float((mret > 0).mean()) if len(mret) > 0 else 0.0

    return {
        **params,
        "cagr_pct":   round(cagr   * 100, 2),
        "sharpe":     round(float(sharpe),    3),
        "sortino":    round(float(sortino),   3),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar":     round(float(calmar),    3),
        "win_rate":   round(win_rate * 100, 1),
        "trades":     trades,
        "composite":  round(float(composite), 4),
        "final_val":  round(float(eq[-1]),    0),
    }


# ─────────────────────────────────────────────
# OUTPUT HELPERS
# ─────────────────────────────────────────────
def write_csv(results: list):
    if not results:
        return
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n  Saved: {OUT_CSV}  ({len(results):,} rows)")


def write_top100(results: list, elapsed: float, n_total: int):
    top = results[:100]

    with open(OUT_TEXT, "w") as f:
        f.write(f"{'='*88}\n")
        f.write(f"  VM PARAMETER SWEEP — TOP 100 RESULTS\n")
        f.write(f"  Period    : {SWEEP_START} → {SWEEP_END}\n")
        f.write(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Runtime   : {elapsed/3600:.2f} hours  |  "
                f"Combinations: {n_total:,}  |  Valid: {len(results):,}\n")
        f.write(f"  Ranking   : Sharpe × Calmar  |  Daily scoring  |  "
                f"TOP_N={TOP_N}, SectorCap={SECTOR_CAP}, ExitCutoff={EXIT_CUTOFF}\n")
        f.write(f"{'='*88}\n\n")

        hdr = (f"{'Rank':>4}  "
               f"{'w12':>5} {'w6':>5} {'w3':>5} {'wv':>6}  "
               f"{'L12':>4} {'L6':>4} {'L3':>3} {'skip':>4}  |  "
               f"{'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} "
               f"{'MaxDD%':>7} {'Calmar':>7} {'Win%':>5} {'Trades':>6} {'Score':>8}\n")
        f.write(hdr)
        f.write("-" * (len(hdr) - 1) + "\n")

        for rank, r in enumerate(top, 1):
            f.write(
                f"{rank:>4}  "
                f"{r['w12']:>5.2f} {r['w6']:>5.2f} {r['w3']:>5.2f} {r['wv']:>6.2f}  "
                f"{r['L12']:>4} {r['L6']:>4} {r['L3']:>3} {r['skip']:>4}  |  "
                f"{r['cagr_pct']:>6.1f}% {r['sharpe']:>7.3f} {r['sortino']:>8.3f} "
                f"{r['max_dd_pct']:>6.1f}% {r['calmar']:>7.3f} {r['win_rate']:>5.1f}% "
                f"{r['trades']:>6} {r['composite']:>8.4f}\n"
            )

        # ── Sensitivity table (top 20% of valid results) ───────────────────────
        n_top = max(1, len(results) // 5)
        top20 = results[:n_top]

        f.write(f"\n\n{'='*88}\n")
        f.write(f"  PARAMETER SENSITIVITY  (averaged over top {n_top:,} results = top 20%)\n")
        f.write(f"{'='*88}\n")

        for key in GRID:
            f.write(f"\n  ── {key} ──\n")
            f.write(f"  {'Value':>8}  {'CAGR%':>7}  {'Sharpe':>7}  "
                    f"{'MaxDD%':>7}  {'Calmar':>7}  {'Score':>8}  {'Count':>6}\n")
            for val in sorted(GRID[key]):
                sub = [r for r in top20 if r[key] == val]
                if not sub:
                    continue
                f.write(
                    f"  {val:>8}  "
                    f"{np.mean([r['cagr_pct']   for r in sub]):>6.1f}%  "
                    f"{np.mean([r['sharpe']      for r in sub]):>7.3f}  "
                    f"{np.mean([r['max_dd_pct']  for r in sub]):>6.1f}%  "
                    f"{np.mean([r['calmar']      for r in sub]):>7.3f}  "
                    f"{np.mean([r['composite']   for r in sub]):>8.4f}  "
                    f"{len(sub):>6}\n"
                )

    print(f"  Saved: {OUT_TEXT}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  VM PARAMETER SWEEP")
    print(f"  Window  : {SWEEP_START} → {SWEEP_END}")
    print(f"  Workers : {N_WORKERS}")
    print(f"  Scoring : DAILY  |  Ranking: Sharpe × Calmar")
    print(f"{'='*60}\n")

    data    = load_data()
    tickers = data["tickers"]

    # Build full parameter list
    keys       = list(GRID.keys())
    all_params = [dict(zip(keys, c))
                  for c in itertools.product(*[GRID[k] for k in keys])]
    n_total = len(all_params)

    print(f"\n  Total combinations : {n_total:,}")
    print(f"  Started            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"  Progress logged every 100 completions.\n", flush=True)

    results = []
    done    = 0

    chunksize = max(1, n_total // (N_WORKERS * 20))

    with mp.Pool(N_WORKERS, initializer=_init_worker, initargs=(data,)) as pool:
        for res in pool.imap_unordered(run_one, all_params, chunksize=chunksize):
            done += 1
            if res is not None:
                results.append(res)

            if done % 100 == 0 or done == n_total:
                elapsed = time.time() - t0
                rate    = done / elapsed if elapsed > 0 else 1
                eta     = (n_total - done) / rate
                print(
                    f"  {done:>5}/{n_total}  |  valid: {len(results):>5}  |  "
                    f"elapsed: {elapsed/60:>5.1f}m  |  ETA: {eta/60:>5.1f}m",
                    flush=True
                )

    if not results:
        print("\nERROR: No valid results. Check cache files and parameters.")
        sys.exit(1)

    elapsed = time.time() - t0

    # Sort: Sharpe × Calmar descending
    results.sort(key=lambda r: r["composite"], reverse=True)

    print(f"\n  Sweep complete in {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours).")
    print(f"  Valid : {len(results):,} / {n_total:,}")

    write_csv(results)
    write_top100(results, elapsed, n_total)

    print(f"\n  ── TOP 5 ──")
    print(f"  {'Rank':>4}  {'w12':>5} {'w6':>5} {'w3':>5} {'wv':>6}  "
          f"{'L12':>4} {'L6':>4} {'L3':>3} {'skip':>4}  "
          f"{'CAGR%':>7} {'Sharpe':>7} {'MaxDD%':>7} {'Score':>8}")
    for i, r in enumerate(results[:5], 1):
        print(f"  {i:>4}  {r['w12']:>5.2f} {r['w6']:>5.2f} {r['w3']:>5.2f} {r['wv']:>6.2f}  "
              f"{r['L12']:>4} {r['L6']:>4} {r['L3']:>3} {r['skip']:>4}  "
              f"{r['cagr_pct']:>6.1f}% {r['sharpe']:>7.3f} {r['max_dd_pct']:>6.1f}% "
              f"{r['composite']:>8.4f}")

    print(f"\n  Done. Results in: {OUT_CSV}  |  {OUT_TEXT}")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)   # Linux COW — workers share read-only data
    main()

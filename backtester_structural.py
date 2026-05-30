#!/usr/bin/env python3
"""
backtester_structural.py
========================
Sweeps structural/construction parameters with signal weights fixed
at current config values. Trains on 2008-2015, validates top 50
combos on 2016-2026 (OOS).

Parameters swept (4,320 combinations):
  TOP_N             : [7, 8, 10, 12, 15]
  MAX_PER_SECTOR    : [2, 3, 4]
  EXIT_RANK_BUFFER  : [8, 13, 18, 23]  → cutoff = TOP_N + buffer
  DMA_EXIT          : [100, 150, 200, 250]
  REGIME_DMA        : [150, 200]
  MIN_PRICE         : [0, 50, 100]
  MIN_AVG_VALUE_CR  : [0, 5, 10]

Signal weights fixed at current config:
  W_MOM_12M=0.50  W_MOM_6M=0.40  W_MOM_3M=0.30  W_VOL=0.00
  LOOKBACK_12M=280  LOOKBACK_6M=140  LOOKBACK_3M=80  SKIP_RECENT=25

Run in background on VM:
    nohup python backtester_structural.py > structural_sweep.log 2>&1 &
    tail -f structural_sweep.log

Output:
    structural_results.csv      — all 4,320 train results ranked by Sharpe × Calmar
    structural_oos.csv          — top 50 combos validated on 2016-2026
    structural_top100.txt       — human-readable report
"""

import os, sys, time, csv, itertools, warnings
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime
import config as cfg

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
TRAIN_START = "2008-01-01"
TRAIN_END   = "2015-12-31"
OOS_START   = "2016-01-01"
OOS_END     = "2026-05-29"
N_WORKERS   = 8
INITIAL_CAP = 100_000.0
RF_ANNUAL   = 0.065
OOS_TOP_N   = 50          # validate top N train combos on OOS

# Fixed signal weights (current config)
W12, W6, W3, WV       = 0.50, 0.40, 0.30, 0.00
L12, L6, L3, SKIP     = 280,  140,  80,   25

TXN_BUY  = 0.00165
TXN_SELL = 0.00170

OUT_CSV     = "structural_results.csv"
OUT_OOS_CSV = "structural_oos.csv"
OUT_TEXT    = "structural_top100.txt"

# ─────────────────────────────────────────────
# PARAMETER GRID  (4,320 combinations)
# ─────────────────────────────────────────────
GRID = {
    "top_n":            [7, 8, 10, 12, 15],
    "max_per_sector":   [2, 3, 4],
    "exit_rank_buffer": [8, 13, 18, 23],    # EXIT_RANK_CUTOFF = top_n + buffer
    "dma_exit":         [100, 150, 200, 250],
    "regime_dma":       [150, 200],
    "min_price":        [0, 50, 100],
    "min_avg_value_cr": [0, 5, 10],
}

# ─────────────────────────────────────────────
# WORKER GLOBALS
# ─────────────────────────────────────────────
_close_arr      = None   # (N_days, N_stocks) float32
_score_mat      = None   # (N_days, N_stocks) float64 — fixed weights, all stocks
_above_dma_dict = None   # {dma_val: (N_days, N_stocks) bool}
_regime_dict    = None   # {regime_dma: (N_days,) bool}
_avg_val_arr    = None   # (N_days, N_stocks) float32 — 60d avg traded value (crores)
_month_arr      = None   # (N_days,) int32 YYYYMM
_sec_idx        = None   # (N_stocks,) int32
_n_stocks       = None
_train_mask     = None   # (N_days,) bool
_oos_mask       = None   # (N_days,) bool


def _init_worker(shared: dict):
    global _close_arr, _score_mat, _above_dma_dict, _regime_dict
    global _avg_val_arr, _month_arr, _sec_idx, _n_stocks
    global _train_mask, _oos_mask
    _close_arr      = shared["close_arr"]
    _score_mat      = shared["score_mat"]
    _above_dma_dict = shared["above_dma_dict"]
    _regime_dict    = shared["regime_dict"]
    _avg_val_arr    = shared["avg_val_arr"]
    _month_arr      = shared["month_arr"]
    _sec_idx        = shared["sec_idx"]
    _n_stocks       = shared["n_stocks"]
    _train_mask     = shared["train_mask"]
    _oos_mask       = shared["oos_mask"]


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_data() -> dict:
    from signals import UNIVERSE
    print("Loading data...", flush=True)

    close_df  = pd.read_csv(cfg.DATA_CACHE_FILE,  index_col=0, parse_dates=True).sort_index()
    vol_df    = pd.read_csv(cfg.VOLUME_CACHE,      index_col=0, parse_dates=True).sort_index()
    regime_df = pd.read_csv(cfg.REGIME_CACHE,      index_col=0, parse_dates=True).sort_index()

    tickers  = [t for t in UNIVERSE if t in close_df.columns]
    sectors  = [UNIVERSE[t] for t in tickers]
    n_stocks = len(tickers)
    close_df = close_df[tickers].astype(float).ffill()

    dates = close_df.index
    N     = len(dates)

    # ── Score matrix (fixed weights, computed once) ───────────────
    print("  Computing score matrix (fixed weights)...", flush=True)
    close_arr = close_df.values.astype(np.float32)
    d_all     = np.arange(N)
    score_mat = np.full((N, n_stocks), np.nan, dtype=np.float64)

    min_hist = L12 + SKIP
    valid_d  = d_all[d_all >= min_hist]

    p_now = close_arr[valid_d - SKIP]
    p_12m = close_arr[valid_d - L12 - SKIP]
    p_6m  = close_arr[valid_d - L6  - SKIP]
    p_3m  = close_arr[valid_d - L3  - SKIP]

    with np.errstate(invalid="ignore", divide="ignore"):
        m12 = (p_now - p_12m) / p_12m
        m6  = (p_now - p_6m)  / p_6m
        m3  = (p_now - p_3m)  / p_3m

    valid_mask = (
        np.isfinite(m12) & np.isfinite(m6) & np.isfinite(m3) &
        (p_now > 0) & (p_12m > 0) & (p_6m > 0) & (p_3m > 0)
    )

    def czs(arr):
        out = np.where(valid_mask, arr, np.nan)
        mu  = np.nanmean(out, axis=1, keepdims=True)
        sd  = np.nanstd(out,  axis=1, keepdims=True)
        return np.nan_to_num(np.where(sd > 0, (out - mu) / sd, 0))

    score_mat[valid_d] = W12*czs(m12) + W6*czs(m6) + W3*czs(m3)
    score_mat[~np.isfinite(score_mat)] = -9999.0

    # ── Pre-compute above_dma for each DMA_EXIT value ─────────────
    print("  Precomputing DMA arrays...", flush=True)
    above_dma_dict = {}
    for dma in sorted(set(GRID["dma_exit"])):
        dma_arr = close_df.rolling(dma).mean().values
        above_dma_dict[dma] = (close_arr > dma_arr.astype(np.float32)).astype(bool)

    # ── Pre-compute regime_on for each REGIME_DMA value ──────────
    print("  Precomputing regime arrays...", flush=True)
    regime_col = "nifty500" if "nifty500" in regime_df.columns else regime_df.columns[0]
    n500 = regime_df[regime_col].reindex(dates).ffill().values.astype(np.float32)
    regime_dict = {}
    for rdma in sorted(set(GRID["regime_dma"])):
        dma_s = pd.Series(n500).rolling(rdma).mean().values
        regime_dict[rdma] = (n500 > dma_s).astype(bool)

    # ── 60-day avg traded value (crore) ──────────────────────────
    print("  Precomputing liquidity matrix...", flush=True)
    vol_df = vol_df[tickers].astype(float).ffill()
    traded = (close_df * vol_df).rolling(60).mean() / 1e7   # crores
    avg_val_arr = traded.values.astype(np.float32)

    # ── Sector encoding ───────────────────────────────────────────
    sec_names = sorted(set(sectors))
    sec_map   = {s: i for i, s in enumerate(sec_names)}
    sec_idx   = np.array([sec_map[s] for s in sectors], dtype=np.int32)

    # ── Date helpers ──────────────────────────────────────────────
    month_arr = np.array(
        [pd.Timestamp(d).year * 100 + pd.Timestamp(d).month for d in dates],
        dtype=np.int32
    )
    ts, te  = pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)
    os_, oe = pd.Timestamp(OOS_START),   pd.Timestamp(OOS_END)
    train_mask = np.array([ts <= pd.Timestamp(d) <= te for d in dates])
    oos_mask   = np.array([os_ <= pd.Timestamp(d) <= oe for d in dates])

    print(f"  {n_stocks} stocks | {train_mask.sum()} train days | "
          f"{oos_mask.sum()} OOS days", flush=True)

    return {
        "close_arr":      close_arr,
        "score_mat":      score_mat,
        "above_dma_dict": above_dma_dict,
        "regime_dict":    regime_dict,
        "avg_val_arr":    avg_val_arr,
        "month_arr":      month_arr,
        "sec_idx":        sec_idx,
        "n_stocks":       n_stocks,
        "train_mask":     train_mask,
        "oos_mask":       oos_mask,
        "tickers":        tickers,
        "sectors":        sectors,
        "sec_names":      sec_names,
    }


# ─────────────────────────────────────────────
# BACKTEST CORE
# ─────────────────────────────────────────────
def _run(params: dict, use_oos: bool = False):
    top_n        = params["top_n"]
    max_sec      = params["max_per_sector"]
    exit_cutoff  = top_n + params["exit_rank_buffer"]
    dma_exit     = params["dma_exit"]
    regime_dma   = params["regime_dma"]
    min_price    = params["min_price"]
    min_val_cr   = params["min_avg_value_cr"]

    above_dma  = _above_dma_dict[dma_exit]
    regime_on  = _regime_dict[regime_dma]
    mask       = _oos_mask if use_oos else _train_mask

    day_idx = np.where(mask)[0]
    day_idx = day_idx[day_idx >= (L12 + SKIP)]
    T = len(day_idx)
    if T < 20:
        return None

    S          = _n_stocks
    cash       = INITIAL_CAP
    shares     = np.zeros(S, dtype=np.float64)
    eq_curve   = np.empty(T, dtype=np.float64)
    trades     = 0
    last_month = -1

    for ti in range(T):
        di     = day_idx[ti]
        prices = _close_arr[di].astype(np.float64)
        scores = _score_mat[di].copy()

        px_safe  = np.where(prices > 0, prices, 0.0)
        port_val = cash + float(np.dot(shares, px_safe))

        # ── RISK-OFF ─────────────────────────────────────────────
        if not regime_on[di]:
            held = shares > 0
            if held.any():
                cash += float(np.dot(shares * held, px_safe)) * (1 - TXN_SELL)
                shares[:] = 0
                trades += int(held.sum())
            eq_curve[ti] = cash
            last_month = -1
            continue

        # ── Liquidity / price filter (mask ineligible stocks) ────
        if min_price > 0:
            scores[prices < min_price] = -9999.0
        if min_val_cr > 0:
            scores[_avg_val_arr[di] < min_val_cr] = -9999.0

        # ── Rank and select portfolio ─────────────────────────────
        rank_order = np.argsort(-scores)
        in_cutoff  = set(rank_order[:exit_cutoff].tolist())

        portfolio = []
        sec_count = {}
        for i in rank_order:
            if scores[i] <= -9000 or len(portfolio) >= top_n:
                break
            sc = int(_sec_idx[i])
            if sec_count.get(sc, 0) < max_sec:
                portfolio.append(i)
                sec_count[sc] = sec_count.get(sc, 0) + 1
        portfolio_set = set(portfolio)

        # ── Daily exits ───────────────────────────────────────────
        for i in range(S):
            if shares[i] <= 0 or prices[i] <= 0:
                continue
            if (i not in in_cutoff) or (not above_dma[di, i]):
                cash += shares[i] * prices[i] * (1 - TXN_SELL)
                shares[i] = 0.0
                trades += 1

        # ── Monthly rotation ──────────────────────────────────────
        cur_month = int(_month_arr[di])
        if cur_month != last_month:
            for i in range(S):
                if shares[i] > 0 and i not in portfolio_set and prices[i] > 0:
                    cash += shares[i] * prices[i] * (1 - TXN_SELL)
                    shares[i] = 0.0
                    trades += 1
            last_month = cur_month

        # ── Buy to fill portfolio ─────────────────────────────────
        port_val    = cash + float(np.dot(shares, px_safe))
        target_each = port_val / top_n if top_n > 0 else 0

        for i in portfolio:
            px = prices[i]
            if px <= 0:
                continue
            cur_val = shares[i] * px
            if cur_val < target_each * 0.95:
                n_buy = int((target_each - cur_val) / px)
                if n_buy > 0:
                    cost = n_buy * px * (1 + TXN_BUY)
                    if cash >= cost:
                        cash    -= cost
                        shares[i] += n_buy
                        trades  += 1

        eq_curve[ti] = cash + float(np.dot(shares, px_safe))

    # ── Metrics ───────────────────────────────────────────────────
    eq = eq_curve
    if eq[0] <= 0 or not np.all(np.isfinite(eq)):
        return None

    years  = T / 252.0
    cagr   = (eq[-1] / eq[0]) ** (1 / years) - 1 if years > 0 else 0
    dr     = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], np.nan)
    dr     = dr[np.isfinite(dr)]
    if len(dr) < 10:
        return None

    rf_d    = RF_ANNUAL / 252
    excess  = dr - rf_d
    sharpe  = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
    down    = excess[excess < 0]
    sortino = excess.mean() / down.std() * np.sqrt(252) if (len(down) > 1 and down.std() > 0) else 0
    peak    = np.maximum.accumulate(eq)
    dd      = (eq - peak) / np.where(peak > 0, peak, 1)
    max_dd  = float(dd.min())
    calmar  = cagr / abs(max_dd) if abs(max_dd) > 1e-9 else 0
    mret    = np.diff(eq[::21]) / eq[::21][:-1]
    win_rate = float((mret[np.isfinite(mret)] > 0).mean()) if len(mret) > 0 else 0
    composite = float(sharpe) * float(calmar)

    return {
        **params,
        "exit_cutoff": exit_cutoff,
        "cagr_pct":   round(cagr * 100, 2),
        "sharpe":     round(float(sharpe), 3),
        "sortino":    round(float(sortino), 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar":     round(float(calmar), 3),
        "win_rate":   round(win_rate * 100, 1),
        "trades":     trades,
        "composite":  round(composite, 4),
    }


def run_one_train(params: dict):
    return _run(params, use_oos=False)

def run_one_oos(params: dict):
    return _run(params, use_oos=True)


# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────
def write_report(train_results, oos_results, elapsed):
    top100 = train_results[:100]

    with open(OUT_TEXT, "w") as f:
        f.write(f"{'='*90}\n")
        f.write(f"  STRUCTURAL PARAMETER SWEEP — TOP 100 RESULTS\n")
        f.write(f"  Train  : {TRAIN_START} → {TRAIN_END}\n")
        f.write(f"  OOS    : {OOS_START} → {OOS_END}  (top {OOS_TOP_N} combos validated)\n")
        f.write(f"  Fixed  : W12={W12} W6={W6} W3={W3} WV={WV}  "
                f"L12={L12} L6={L6} L3={L3} skip={SKIP}\n")
        f.write(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Runtime   : {elapsed/60:.1f} min  |  "
                f"Combinations: {len(train_results):,}\n")
        f.write(f"  Ranking   : Sharpe × Calmar\n")
        f.write(f"{'='*90}\n\n")

        hdr = (f"{'Rk':>3}  {'N':>3} {'Sec':>3} {'Buf':>3} {'Cut':>4}  "
               f"{'DMAx':>5} {'Rgm':>4}  {'MinP':>5} {'MinV':>5}  |  "
               f"{'CAGR%':>7} {'Sharpe':>7} {'Sort':>6} "
               f"{'MaxDD%':>7} {'Calmar':>7} {'Win%':>5} {'Trd':>5} {'Score':>8}\n")
        f.write(hdr)
        f.write("  " + "-" * (len(hdr) - 3) + "\n")

        for rk, r in enumerate(top100, 1):
            f.write(
                f"{rk:>3}  {r['top_n']:>3} {r['max_per_sector']:>3} "
                f"{r['exit_rank_buffer']:>3} {r['exit_cutoff']:>4}  "
                f"{r['dma_exit']:>5} {r['regime_dma']:>4}  "
                f"{r['min_price']:>5} {r['min_avg_value_cr']:>5}  |  "
                f"{r['cagr_pct']:>6.1f}% {r['sharpe']:>7.3f} {r['sortino']:>6.3f} "
                f"{r['max_dd_pct']:>6.1f}% {r['calmar']:>7.3f} {r['win_rate']:>5.1f}% "
                f"{r['trades']:>5} {r['composite']:>8.4f}\n"
            )

        # Sensitivity analysis
        f.write(f"\n\n{'='*90}\n")
        f.write(f"  PARAMETER SENSITIVITY  (top 20% of train results)\n")
        f.write(f"{'='*90}\n")

        top20 = train_results[:max(1, len(train_results)//5)]
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

        # OOS validation
        if oos_results:
            f.write(f"\n\n{'='*90}\n")
            f.write(f"  OOS VALIDATION  (top {OOS_TOP_N} train combos tested on {OOS_START[:4]}–{OOS_END[:4]})\n")
            f.write(f"{'='*90}\n\n")
            f.write(f"  {'Rk':>3}  {'N':>3} {'Sec':>3} {'Buf':>3} {'Cut':>4}  "
                    f"{'DMAx':>5} {'Rgm':>4}  {'MinP':>5} {'MinV':>5}  |  "
                    f"  Train{'':>2}→  OOS CAGR%   OOS Sharpe   OOS MaxDD%  OOS Score\n")
            f.write("  " + "-"*85 + "\n")
            for i, (tr, oos) in enumerate(zip(train_results[:OOS_TOP_N], oos_results), 1):
                if oos is None:
                    continue
                f.write(
                    f"  {i:>3}  {tr['top_n']:>3} {tr['max_per_sector']:>3} "
                    f"{tr['exit_rank_buffer']:>3} {tr['exit_cutoff']:>4}  "
                    f"{tr['dma_exit']:>5} {tr['regime_dma']:>4}  "
                    f"{tr['min_price']:>5} {tr['min_avg_value_cr']:>5}  |  "
                    f"{tr['composite']:>8.4f}  →  "
                    f"{oos['cagr_pct']:>+8.1f}%   {oos['sharpe']:>10.3f}   "
                    f"{oos['max_dd_pct']:>9.1f}%  {oos['composite']:>9.4f}\n"
                )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    t0 = time.time()
    total = int(np.prod([len(v) for v in GRID.values()]))

    print(f"\n{'='*60}")
    print(f"  STRUCTURAL PARAMETER SWEEP")
    print(f"  Combinations : {total:,}")
    print(f"  Workers      : {N_WORKERS}")
    print(f"  Train        : {TRAIN_START} → {TRAIN_END}")
    print(f"  OOS top {OOS_TOP_N}   : {OOS_START} → {OOS_END}")
    print(f"  Started      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    data = load_data()
    _init_worker(data)

    keys       = list(GRID.keys())
    all_params = [dict(zip(keys, c))
                  for c in itertools.product(*[GRID[k] for k in keys])]

    # ── Train sweep ───────────────────────────────────────────────
    print(f"  Running train sweep ({total:,} combos)...\n", flush=True)
    train_results = []
    done = 0
    chunksize = max(1, total // (N_WORKERS * 20))

    with mp.Pool(N_WORKERS, initializer=_init_worker, initargs=(data,)) as pool:
        for res in pool.imap_unordered(run_one_train, all_params, chunksize=chunksize):
            done += 1
            if res is not None:
                train_results.append(res)
            if done % 500 == 0 or done == total:
                el  = time.time() - t0
                eta = (total - done) / (done / el) if done > 0 else 0
                print(f"  {done:>5}/{total}  valid: {len(train_results):>5}  "
                      f"elapsed: {el/60:>5.1f}m  ETA: {eta/60:>5.1f}m", flush=True)

    train_results.sort(key=lambda r: r["composite"], reverse=True)

    el_train = time.time() - t0
    print(f"\n  Train sweep done in {el_train/60:.1f} min")
    print(f"  Valid results: {len(train_results):,} / {total:,}\n")

    # ── OOS validation (top N combos) ─────────────────────────────
    print(f"  Running OOS validation (top {OOS_TOP_N} combos)...", flush=True)
    top_params = [
        {k: r[k] for k in GRID} for r in train_results[:OOS_TOP_N]
    ]
    with mp.Pool(N_WORKERS, initializer=_init_worker, initargs=(data,)) as pool:
        oos_results = pool.map(run_one_oos, top_params)

    # ── Write outputs ─────────────────────────────────────────────
    elapsed = time.time() - t0

    fields_train = list(all_params[0].keys()) + [
        "exit_cutoff","cagr_pct","sharpe","sortino","max_dd_pct",
        "calmar","win_rate","trades","composite"
    ]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields_train)
        w.writeheader()
        w.writerows(train_results)
    print(f"  Saved: {OUT_CSV}")

    oos_valid = [r for r in oos_results if r is not None]
    if oos_valid:
        fields_oos = list(oos_valid[0].keys())
        with open(OUT_OOS_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields_oos)
            w.writeheader()
            w.writerows(oos_valid)
        print(f"  Saved: {OUT_OOS_CSV}")

    write_report(train_results, oos_results, elapsed)
    print(f"  Saved: {OUT_TEXT}")

    # ── Console summary ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DONE in {elapsed/60:.1f} min")
    print(f"\n  TOP 5 (train):")
    print(f"  {'N':>3} {'Sec':>3} {'Buf':>3} {'Cut':>4}  "
          f"{'DMAx':>5} {'Rgm':>4}  {'MinP':>5} {'MinV':>5}  "
          f"{'CAGR%':>7} {'Sharpe':>7} {'MaxDD%':>7} {'Score':>8}")
    for r in train_results[:5]:
        print(f"  {r['top_n']:>3} {r['max_per_sector']:>3} "
              f"{r['exit_rank_buffer']:>3} {r['exit_cutoff']:>4}  "
              f"{r['dma_exit']:>5} {r['regime_dma']:>4}  "
              f"{r['min_price']:>5} {r['min_avg_value_cr']:>5}  "
              f"{r['cagr_pct']:>6.1f}% {r['sharpe']:>7.3f} "
              f"{r['max_dd_pct']:>6.1f}% {r['composite']:>8.4f}")

    if oos_results:
        print(f"\n  TOP 5 OOS VALIDATION:")
        print(f"  {'Train→OOS':>12}  {'CAGR%':>8}  {'Sharpe':>8}  {'MaxDD%':>8}  {'Score':>8}")
        for tr, oos in zip(train_results[:5], oos_results[:5]):
            if oos:
                print(f"  {tr['composite']:>8.4f} →  "
                      f"{oos['cagr_pct']:>+7.1f}%  {oos['sharpe']:>8.3f}  "
                      f"{oos['max_dd_pct']:>7.1f}%  {oos['composite']:>8.4f}")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()

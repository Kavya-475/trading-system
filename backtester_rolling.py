#!/usr/bin/env python3
"""
backtester_rolling.py
=====================
Rolling window confidence analysis.
Runs every (start_year, duration) combination:
  Start years : 2007 → 2025
  Durations   : 1 → 18 years
  ~171 valid windows

Uses real regime filter (FORCE_RISK_ON=False — no override).
Includes capital gains tax and full transaction costs.
Score matrix precomputed once; portfolio sim vectorised.

Run on VM:
    nohup python backtester_rolling.py > rolling.log 2>&1 &
    tail -f rolling.log

Output:
    rolling_results.csv   — per-window metrics
    rolling_summary.txt   — CAGR / MaxDD / Sharpe matrices + stats
"""

import os, sys, time, csv, warnings, itertools
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime
import config as cfg

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STRATEGY PARAMETERS  (current config)
# ─────────────────────────────────────────────
TOP_N        = cfg.TOP_N            # 10
SECTOR_CAP   = cfg.MAX_PER_SECTOR   # 3
EXIT_CUTOFF  = cfg.EXIT_RANK_CUTOFF # 25
DMA_EXIT_D   = cfg.DMA_EXIT         # 250
REGIME_DMA   = cfg.REGIME_DMA       # 200
W12          = cfg.W_MOM_12M        # 0.50
W6           = cfg.W_MOM_6M         # 0.40
W3           = cfg.W_MOM_3M         # 0.30
WV           = cfg.W_VOL            # 0.00
L12          = cfg.LOOKBACK_12M     # 280
L6           = cfg.LOOKBACK_6M      # 140
L3           = cfg.LOOKBACK_3M      # 80
SKIP         = cfg.SKIP_RECENT      # 25
INITIAL_CAP  = float(cfg.INITIAL_CAPITAL)
RF_ANNUAL    = cfg.RISK_FREE_RATE

# Transaction costs (one-way)
TXN_BUY  = cfg.STAMP_DUTY + cfg.EXCHANGE_CHARGE + cfg.SEBI_CHARGE
TXN_SELL = cfg.STT_SELL   + cfg.EXCHANGE_CHARGE + cfg.SEBI_CHARGE

# Limit-order buffers (tiered by rank in full universe)
BUY_BUF_TOP5 = cfg.BUY_BUFFER_TOP5   # 5% — ranks 0-4
BUY_BUF_MID  = cfg.BUY_BUFFER_MID    # 3% — ranks 5-11
BUY_BUF_REST = cfg.BUY_BUFFER_REST   # 2% — rank 12+
SELL_BUF     = cfg.SELL_BUFFER        # 1% discount on sell limit

# Capital gains tax cutoff date (Budget 2024)
TAX_CUTOFF_ORD = pd.Timestamp("2024-07-23").toordinal()
STCG_PRE, STCG_POST = cfg.STCG_RATE_PRE, cfg.STCG_RATE_POST
LTCG_PRE, LTCG_POST = cfg.LTCG_RATE_PRE, cfg.LTCG_RATE_POST

N_WORKERS = 8
OUT_CSV   = "rolling_results.csv"
OUT_TEXT  = "rolling_summary.txt"

# ─────────────────────────────────────────────
# WORKER GLOBALS
# ─────────────────────────────────────────────
_close_arr  = None
_open_arr   = None   # next-day open prices for realistic fills
_score_mat  = None
_above_dma  = None
_regime_on  = None
_month_arr  = None
_date_ord   = None
_sec_idx    = None
_n_stocks   = None
_has_open   = False  # whether open price cache is available


def _init(shared):
    global _close_arr, _open_arr, _score_mat, _above_dma, _regime_on
    global _month_arr, _date_ord, _sec_idx, _n_stocks, _has_open
    _close_arr = shared["close_arr"]
    _open_arr  = shared["open_arr"]
    _has_open  = shared["has_open"]
    _score_mat = shared["score_mat"]
    _above_dma = shared["above_dma"]
    _regime_on = shared["regime_on"]
    _month_arr = shared["month_arr"]
    _date_ord  = shared["date_ord"]
    _sec_idx   = shared["sec_idx"]
    _n_stocks  = shared["n_stocks"]


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_data():
    from signals import UNIVERSE
    print("Loading data...", flush=True)

    close_df  = pd.read_csv(cfg.DATA_CACHE_FILE, index_col=0,
                             parse_dates=True).sort_index()
    regime_df = pd.read_csv(cfg.REGIME_CACHE,    index_col=0,
                             parse_dates=True).sort_index()

    tickers  = [t for t in UNIVERSE if t in close_df.columns]
    sectors  = [UNIVERSE[t] for t in tickers]
    S        = len(tickers)
    close_df = close_df[tickers].astype(float).ffill()
    dates    = close_df.index
    N        = len(dates)

    close_arr = close_df.values.astype(np.float32)

    # ── Open prices (for realistic next-day fill simulation) ──────
    has_open = False
    open_arr = np.zeros_like(close_arr)   # fallback: zeros → use limit price
    if os.path.exists(cfg.OPEN_CACHE):
        try:
            open_df  = pd.read_csv(cfg.OPEN_CACHE, index_col=0,
                                   parse_dates=True).sort_index()
            open_df  = open_df.reindex(columns=tickers).reindex(dates).ffill()
            open_arr = open_df.values.astype(np.float32)
            has_open = True
            print("  Open price cache loaded — using next-day open fills.", flush=True)
        except Exception as e:
            print(f"  Open cache error ({e}) — using limit-price approximation.", flush=True)
    else:
        print("  Open cache missing — run data_manager.py first for exact fills.", flush=True)
        print("  Falling back to limit-price approximation (close × buffer).", flush=True)

    # ── Score matrix (precomputed once) ──────────────────────────
    print("  Precomputing score matrix...", flush=True)
    min_hist  = L12 + SKIP
    valid_d   = np.arange(min_hist, N)
    score_mat = np.full((N, S), -9999.0, dtype=np.float64)

    p_now = close_arr[valid_d - SKIP].astype(np.float64)
    p_12m = close_arr[valid_d - L12 - SKIP].astype(np.float64)
    p_6m  = close_arr[valid_d - L6  - SKIP].astype(np.float64)
    p_3m  = close_arr[valid_d - L3  - SKIP].astype(np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        m12 = (p_now - p_12m) / p_12m
        m6  = (p_now - p_6m)  / p_6m
        m3  = (p_now - p_3m)  / p_3m

    ok = (np.isfinite(m12) & np.isfinite(m6) & np.isfinite(m3) &
          (p_now > 0) & (p_12m > 0) & (p_6m > 0) & (p_3m > 0))

    def czs(arr):
        a = np.where(ok, arr, np.nan)
        mu = np.nanmean(a, axis=1, keepdims=True)
        sd = np.nanstd(a,  axis=1, keepdims=True)
        return np.nan_to_num(np.where(sd > 0, (a - mu) / sd, 0))

    scores = W12*czs(m12) + W6*czs(m6) + W3*czs(m3)
    scores = np.where(ok, scores, -9999.0)
    score_mat[valid_d] = scores

    # ── 250-DMA above/below ───────────────────────────────────────
    dma = close_df.rolling(DMA_EXIT_D).mean().values.astype(np.float32)
    above_dma = (close_arr > dma).astype(bool)

    # ── Regime filter ─────────────────────────────────────────────
    rcol  = "nifty500" if "nifty500" in regime_df.columns else regime_df.columns[0]
    n500  = regime_df[rcol].reindex(dates).ffill().values.astype(np.float32)
    n500_dma = pd.Series(n500).rolling(REGIME_DMA).mean().values
    regime_on = (n500 > n500_dma).astype(bool)

    # ── Helpers ───────────────────────────────────────────────────
    month_arr = np.array(
        [d.year * 100 + d.month for d in dates], dtype=np.int32
    )
    date_ord = np.array([d.toordinal() for d in dates], dtype=np.int32)

    sec_names = sorted(set(sectors))
    sec_map   = {s: i for i, s in enumerate(sec_names)}
    sec_idx   = np.array([sec_map[s] for s in sectors], dtype=np.int32)

    print(f"  {S} stocks | {N} total days | "
          f"{int(regime_on.sum())} RISK-ON days ({regime_on.mean()*100:.0f}%)",
          flush=True)

    return {
        "close_arr": close_arr,
        "open_arr":  open_arr,
        "has_open":  has_open,
        "score_mat": score_mat,
        "above_dma": above_dma,
        "regime_on": regime_on,
        "month_arr": month_arr,
        "date_ord":  date_ord,
        "sec_idx":   sec_idx,
        "n_stocks":  S,
        "dates":     dates,
        "tickers":   tickers,
    }


# ─────────────────────────────────────────────
# CAPITAL GAINS TAX
# ─────────────────────────────────────────────
def _cgt(gain: float, entry_ord: int, exit_ord: int) -> float:
    if gain <= 0:
        return 0.0
    held_days = exit_ord - entry_ord
    post = exit_ord >= TAX_CUTOFF_ORD
    if held_days < 365:
        rate = STCG_POST if post else STCG_PRE
    else:
        rate = LTCG_POST if post else LTCG_PRE
    return gain * rate


# ─────────────────────────────────────────────
# SINGLE WINDOW BACKTEST
# ─────────────────────────────────────────────
def run_window(params: dict):
    start_str = params["start"]
    end_str   = params["end"]
    s_ts      = pd.Timestamp(start_str)
    e_ts      = pd.Timestamp(end_str)

    N, S = _close_arr.shape[0], _n_stocks

    # Day indices for this window
    # We need ALL data for warmup, only simulate within window
    all_d  = np.arange(N)
    win_mask = np.zeros(N, dtype=bool)
    # Load dates from shared close index — reconstruct from month_arr / date_ord
    for i in range(N):
        d_ord = int(_date_ord[i])
        d_ts  = pd.Timestamp.fromordinal(d_ord)
        if s_ts <= d_ts <= e_ts:
            win_mask[i] = True

    day_idx = all_d[win_mask]
    min_hist = L12 + SKIP
    day_idx  = day_idx[day_idx >= min_hist]
    T = len(day_idx)

    if T < 20:
        return None

    N_all      = _close_arr.shape[0]
    cash       = INITIAL_CAP
    shares     = np.zeros(S, dtype=np.float64)
    avg_px     = np.zeros(S, dtype=np.float64)
    entry_ord  = np.zeros(S, dtype=np.int32)
    eq_curve   = np.empty(T, dtype=np.float64)
    trades     = 0
    last_month = -1
    in_roff    = False

    for ti in range(T):
        di      = day_idx[ti]
        prices  = _close_arr[di].astype(np.float64)   # today's close
        scores  = _score_mat[di].copy()
        d_ord   = int(_date_ord[di])
        next_di = di + 1 if di + 1 < N_all else di    # next trading day

        # Next-day open prices for fills
        if _has_open:
            opens = _open_arr[next_di].astype(np.float64)
        else:
            opens = None   # will use limit-price fallback

        px_safe  = np.where(prices > 0, prices, 0.0)
        port_val = cash + float(np.dot(shares, px_safe))

        # ── Helper: sell fill price ───────────────────────────────
        def sell_fill(close_px, open_px):
            """
            Limit sell order placed at close × (1 - SELL_BUF).
            If open data available: actual fill is at next-day open
            (which is typically >= limit so sell at open gives a better price,
            but we conservatively use open × (1 - SELL_BUF)).
            Fallback: close × (1 - SELL_BUF).
            """
            if open_px is not None and open_px > 0:
                return open_px * (1.0 - SELL_BUF)
            return close_px * (1.0 - SELL_BUF)

        # ── Helper: buy fill price ────────────────────────────────
        def buy_fill(close_px, open_px, universe_rank):
            """
            Limit buy placed at close × (1 + buf).
            Fill price = next-day open if open <= limit (order filled at open).
            If open > limit: gap-up — order missed, return None.
            Fallback (no open data): use close price as proxy for next-day open.
            This is slightly optimistic but avoids inflating fills with the buffer.
            """
            if universe_rank < 5:
                buf = BUY_BUF_TOP5
            elif universe_rank < 12:
                buf = BUY_BUF_MID
            else:
                buf = BUY_BUF_REST
            limit = close_px * (1.0 + buf)
            if open_px is not None and open_px > 0:
                if open_px <= limit:
                    return open_px   # filled at open (better than limit)
                return None          # gap-up exceeded limit — missed fill
            return close_px          # fallback: close ≈ next-day open

        # ── Rank lookup array (universe rank per stock) ───────────
        rank_order  = np.argsort(-scores)
        uni_rank    = np.empty(S, dtype=np.int32)
        uni_rank[rank_order] = np.arange(len(rank_order), dtype=np.int32)

        in_cutoff   = set(rank_order[:EXIT_CUTOFF].tolist())

        # ── RISK-OFF ──────────────────────────────────────────────
        if not _regime_on[di]:
            if not in_roff:
                for i in range(S):
                    if shares[i] > 0 and prices[i] > 0:
                        op       = float(opens[i]) if opens is not None else None
                        fp       = sell_fill(prices[i], op)
                        proceeds = shares[i] * fp
                        cost     = proceeds * TXN_SELL
                        gain     = (fp - avg_px[i]) * shares[i]
                        tax      = _cgt(gain, entry_ord[i], d_ord)
                        cash    += proceeds - cost - tax
                        shares[i] = 0.0
                        trades   += 1
                in_roff = True
            eq_curve[ti] = cash
            last_month = -1
            continue

        in_roff = False

        # ── Portfolio selection ───────────────────────────────────
        portfolio, sec_count = [], {}
        for i in rank_order:
            if scores[i] <= -9000 or len(portfolio) >= TOP_N:
                break
            sc = int(_sec_idx[i])
            if sec_count.get(sc, 0) < SECTOR_CAP:
                portfolio.append(i)
                sec_count[sc] = sec_count.get(sc, 0) + 1
        portfolio_set = set(portfolio)

        # ── Daily exits (rank dropout + below DMA) ────────────────
        for i in range(S):
            if shares[i] <= 0 or prices[i] <= 0:
                continue
            if (i not in in_cutoff) or (not _above_dma[di, i]):
                op       = float(opens[i]) if opens is not None else None
                fp       = sell_fill(prices[i], op)
                proceeds = shares[i] * fp
                cost     = proceeds * TXN_SELL
                gain     = (fp - avg_px[i]) * shares[i]
                tax      = _cgt(gain, entry_ord[i], d_ord)
                cash    += proceeds - cost - tax
                shares[i] = 0.0
                trades   += 1

        # ── Monthly rotation ──────────────────────────────────────
        cur_month = int(_month_arr[di])
        if cur_month != last_month:
            for i in range(S):
                if shares[i] > 0 and i not in portfolio_set and prices[i] > 0:
                    op       = float(opens[i]) if opens is not None else None
                    fp       = sell_fill(prices[i], op)
                    proceeds = shares[i] * fp
                    cost     = proceeds * TXN_SELL
                    gain     = (fp - avg_px[i]) * shares[i]
                    tax      = _cgt(gain, entry_ord[i], d_ord)
                    cash    += proceeds - cost - tax
                    shares[i] = 0.0
                    trades   += 1
            last_month = cur_month

        # ── Buy to fill portfolio ─────────────────────────────────
        port_val    = cash + float(np.dot(shares, px_safe))
        target_each = port_val / TOP_N if TOP_N > 0 else 0

        for i in portfolio:
            px = prices[i]
            if px <= 0:
                continue
            cur_val = shares[i] * px
            if cur_val < target_each * 0.95:
                op   = float(opens[i]) if opens is not None else None
                fp   = buy_fill(px, op, int(uni_rank[i]))
                if fp is None:
                    continue   # gap-up missed the limit order
                n_buy = int((target_each - cur_val) / fp)
                if n_buy > 0:
                    cost = n_buy * fp * (1.0 + TXN_BUY)
                    if cash >= cost:
                        old_sh    = shares[i]
                        new_sh    = old_sh + n_buy
                        old_avg   = avg_px[i] if old_sh > 0 else fp
                        avg_px[i] = (old_avg * old_sh + fp * n_buy) / new_sh
                        if old_sh == 0:
                            entry_ord[i] = d_ord
                        shares[i] = new_sh
                        cash -= cost
                        trades += 1

        eq_curve[ti] = cash + float(np.dot(shares, px_safe))

    # ── Metrics ───────────────────────────────────────────────────
    eq = eq_curve
    if eq[0] <= 0 or not np.all(np.isfinite(eq)):
        return None

    years = T / 252.0
    if years < 0.08:   # < 1 month of data — skip
        return None

    cagr    = (eq[-1] / eq[0]) ** (1.0 / years) - 1
    dr      = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], np.nan)
    dr      = dr[np.isfinite(dr)]
    if len(dr) < 5:
        return None

    rf_d    = RF_ANNUAL / 252
    excess  = dr - rf_d
    sharpe  = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
    down    = excess[excess < 0]
    sortino = excess.mean() / down.std() * np.sqrt(252) if (len(down)>1 and down.std()>0) else 0
    peak    = np.maximum.accumulate(eq)
    dd      = (eq - peak) / np.where(peak > 0, peak, 1)
    max_dd  = float(dd.min())
    calmar  = cagr / abs(max_dd) if abs(max_dd) > 1e-9 else 0
    mret    = np.diff(eq[::21]) / eq[::21][:-1]
    win_rate = float((mret[np.isfinite(mret)] > 0).mean()) if len(mret) > 0 else 0

    return {
        "start":      start_str,
        "end":        end_str,
        "start_year": params["start_year"],
        "duration":   params["duration"],
        "trade_days": T,
        "trades":     trades,
        "cagr_pct":   round(cagr   * 100, 1),
        "sharpe":     round(float(sharpe),  3),
        "sortino":    round(float(sortino), 3),
        "max_dd_pct": round(max_dd * 100, 1),
        "calmar":     round(float(calmar),  3),
        "win_rate":   round(win_rate * 100, 1),
        "final_val":  round(float(eq[-1]),  0),
    }


# ─────────────────────────────────────────────
# MATRIX FORMATTING
# ─────────────────────────────────────────────
def fmt_matrix(results: list, metric: str, fmt: str, start_years, durations):
    """Format a metric as a start_year × duration matrix."""
    lookup = {(r["start_year"], r["duration"]): r[metric] for r in results}

    col_w  = 6
    lines  = []
    header = f"{'':>6}" + "".join(f"{d:>{col_w}}yr" for d in durations)
    lines.append(header)
    lines.append("─" * len(header))

    for sy in start_years:
        row = f"{sy:>6}"
        for dur in durations:
            val = lookup.get((sy, dur))
            if val is None:
                row += f"{'--':>{col_w+2}}"
            else:
                row += f"{fmt.format(val):>{col_w+2}}"
        lines.append(row)

    return "\n".join(lines)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  ROLLING WINDOW CONFIDENCE ANALYSIS")
    print(f"  Regime : REAL (FORCE_RISK_ON=False)")
    print(f"  Workers: {N_WORKERS}")
    print(f"  Config : N={TOP_N} Sec={SECTOR_CAP} Cut={EXIT_CUTOFF} "
          f"DMA={DMA_EXIT_D} Rgm={REGIME_DMA}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    data = load_data()
    _init(data)
    dates = data["dates"]

    # ── Build window list ─────────────────────────────────────────
    data_end = pd.Timestamp(cfg.END_DATE)
    start_years = list(range(2007, 2026))   # 2007 to 2025
    durations   = list(range(1, 19))        # 1 to 18

    all_params = []
    for sy in start_years:
        for dur in durations:
            end_year = sy + dur - 1
            if end_year >= 2026:
                end_date = cfg.END_DATE
            else:
                end_date = f"{end_year}-12-31"
            if pd.Timestamp(end_date) > data_end:
                continue
            if pd.Timestamp(f"{sy}-01-01") > data_end:
                continue
            all_params.append({
                "start":      f"{sy}-01-01",
                "end":        end_date,
                "start_year": sy,
                "duration":   dur,
            })

    print(f"  Total windows: {len(all_params)}", flush=True)

    # ── Run all windows ───────────────────────────────────────────
    results = []
    done = 0
    chunksize = max(1, len(all_params) // (N_WORKERS * 4))

    with mp.Pool(N_WORKERS, initializer=_init, initargs=(data,)) as pool:
        for res in pool.imap_unordered(run_window, all_params,
                                       chunksize=chunksize):
            done += 1
            if res is not None:
                results.append(res)
            if done % 20 == 0 or done == len(all_params):
                el = time.time() - t0
                print(f"  {done:>4}/{len(all_params)}  valid: {len(results):>4}  "
                      f"elapsed: {el:.1f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s  |  {len(results)} valid windows\n",
          flush=True)

    if not results:
        print("ERROR: No valid results.")
        return

    results.sort(key=lambda r: (r["start_year"], r["duration"]))

    # ── CSV ───────────────────────────────────────────────────────
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"  Saved: {OUT_CSV}")

    # ── Summary text ──────────────────────────────────────────────
    with open(OUT_TEXT, "w") as f:

        f.write(f"{'='*75}\n")
        f.write(f"  ROLLING WINDOW CONFIDENCE ANALYSIS\n")
        f.write(f"  Strategy : Nifty LargeMidCap 250 Momentum\n")
        f.write(f"  Regime   : REAL regime filter (FORCE_RISK_ON=False)\n")
        f.write(f"  Config   : TOP_N={TOP_N}  SEC={SECTOR_CAP}  EXIT={EXIT_CUTOFF}  "
                f"DMA={DMA_EXIT_D}  RgmDMA={REGIME_DMA}  skip={SKIP}\n")
        f.write(f"  Includes : Transaction costs + Capital gains tax\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                f"  |  Runtime: {elapsed:.1f}s\n")
        f.write(f"{'='*75}\n\n")

        # CAGR matrix
        f.write("  CAGR (% per year)\n")
        f.write("  Rows = start year  |  Cols = duration\n\n")
        f.write("  " + fmt_matrix(results, "cagr_pct", "{:.0f}%",
                                   start_years, durations) + "\n\n")

        # MaxDD matrix
        f.write("  MAX DRAWDOWN (%)\n\n")
        f.write("  " + fmt_matrix(results, "max_dd_pct", "{:.0f}%",
                                   start_years, durations) + "\n\n")

        # Sharpe matrix
        f.write("  SHARPE RATIO\n\n")
        f.write("  " + fmt_matrix(results, "sharpe", "{:.2f}",
                                   start_years, durations) + "\n\n")

        # ── Summary stats ─────────────────────────────────────────
        f.write(f"{'='*75}\n")
        f.write(f"  SUMMARY STATISTICS\n")
        f.write(f"{'='*75}\n\n")

        cagrs   = [r["cagr_pct"]   for r in results]
        dds     = [r["max_dd_pct"] for r in results]
        sharpes = [r["sharpe"]     for r in results]

        pos = sum(1 for c in cagrs if c > 0)
        f.write(f"  Total windows     : {len(results)}\n")
        f.write(f"  Positive CAGR     : {pos}/{len(results)} "
                f"({pos/len(results)*100:.0f}%)\n")
        f.write(f"  Avg CAGR          : {np.mean(cagrs):.1f}%\n")
        f.write(f"  Median CAGR       : {np.median(cagrs):.1f}%\n")
        f.write(f"  Avg MaxDD         : {np.mean(dds):.1f}%\n")
        f.write(f"  Avg Sharpe        : {np.mean(sharpes):.2f}\n")
        f.write(f"  Best window       : {max(results, key=lambda r: r['cagr_pct'])['start'][:4]}"
                f"–{max(results, key=lambda r: r['cagr_pct'])['end'][:4]}"
                f"  CAGR {max(cagrs):.0f}%\n")
        f.write(f"  Worst window      : {min(results, key=lambda r: r['cagr_pct'])['start'][:4]}"
                f"–{min(results, key=lambda r: r['cagr_pct'])['end'][:4]}"
                f"  CAGR {min(cagrs):.0f}%\n\n")

        # By duration
        f.write("  AVG METRICS BY DURATION:\n")
        f.write(f"  {'Dur':>4}  {'Count':>5}  {'AvgCAGR':>8}  "
                f"{'AvgDD':>7}  {'AvgSharpe':>9}  {'%Positive':>9}\n")
        f.write("  " + "-"*52 + "\n")
        for dur in durations:
            sub = [r for r in results if r["duration"] == dur]
            if not sub:
                continue
            pct_pos = sum(1 for r in sub if r["cagr_pct"] > 0) / len(sub) * 100
            f.write(f"  {dur:>4}  {len(sub):>5}  "
                    f"{np.mean([r['cagr_pct'] for r in sub]):>7.1f}%  "
                    f"{np.mean([r['max_dd_pct'] for r in sub]):>6.1f}%  "
                    f"{np.mean([r['sharpe'] for r in sub]):>9.2f}  "
                    f"{pct_pos:>8.0f}%\n")

        # By start year (individual year returns)
        f.write(f"\n  INDIVIDUAL YEAR RETURNS (1-year windows only):\n")
        f.write(f"  {'Year':>6}  {'CAGR':>7}  {'MaxDD':>7}  "
                f"{'Sharpe':>7}  {'Trades':>7}  {'Regime'}\n")
        f.write("  " + "-"*55 + "\n")
        one_yr = sorted([r for r in results if r["duration"] == 1],
                        key=lambda r: r["start_year"])
        for r in one_yr:
            f.write(f"  {r['start_year']:>6}  {r['cagr_pct']:>6.1f}%  "
                    f"{r['max_dd_pct']:>6.1f}%  {r['sharpe']:>7.2f}  "
                    f"{r['trades']:>7}  ")
            if r["trades"] == 0:
                f.write("(no trades — insufficient warmup)\n")
            else:
                f.write("\n")

    print(f"  Saved: {OUT_TEXT}")

    # Console preview
    print(f"\n  INDIVIDUAL YEAR RETURNS:")
    print(f"  {'Year':>6}  {'CAGR':>7}  {'MaxDD':>7}  {'Sharpe':>7}")
    for r in [r for r in results if r["duration"] == 1]:
        print(f"  {r['start_year']:>6}  {r['cagr_pct']:>6.1f}%  "
              f"{r['max_dd_pct']:>6.1f}%  {r['sharpe']:>7.2f}")

    print(f"\n  {'='*40}")
    print(f"  {pos}/{len(results)} windows positive  |  "
          f"Avg CAGR: {np.mean(cagrs):.1f}%  |  "
          f"Avg Sharpe: {np.mean(sharpes):.2f}")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()

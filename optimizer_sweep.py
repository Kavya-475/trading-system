"""
optimizer_sweep.py — Comprehensive Parameter Sweep Backtester
==============================================================
Designed for: GCP c4-standard-32 (32 vCPUs, 128 GB RAM)
Time budget : ≤ 12 hours
Period      : 2008-01-01 to 2015-12-31

Architecture:
  1. Loads price/volume/regime data ONCE into shared memory
  2. Runs backtest as a pure function (no subprocess, no config mutation)
  3. Uses multiprocessing.Pool with 30 workers (leave 2 cores for OS)
  4. Latin Hypercube Sampling for efficient parameter space coverage

Parameters swept (8 dimensions, symmetric ranges):
  ┌──────────────────┬──────────┬──────────────────────────────┐
  │ Parameter        │ Current  │ Sweep Range                  │
  ├──────────────────┼──────────┼──────────────────────────────┤
  │ TOP_N            │ 10       │ [5, 7, 10, 12, 15, 18, 20]  │
  │ MAX_PER_SECTOR   │ 3        │ [1, 2, 3, 4, 5, 6]          │
  │ W_MOM_12M        │ 0.40     │ [0.0, 0.10, 0.20, 0.30, ... │
  │ W_MOM_6M         │ 0.35     │      0.40, 0.50, 0.60, 0.70]│
  │ W_MOM_3M         │ 0.15     │ [0.0, 0.05, 0.10, 0.15, ... │
  │                  │          │      0.20, 0.25, 0.30]       │
  │ W_VOL            │ -0.10    │ [-0.30, -0.20, -0.15, -0.10,│
  │                  │          │  -0.05, 0.0, 0.05, 0.10]     │
  │ SKIP_RECENT      │ 20       │ [0, 5, 10, 15, 20, 30, 42]  │
  │ MIN_PRICE        │ 0        │ [0, 50, 100, 200, 500]       │
  │ MIN_AVG_VALUE_CR │ 0        │ [0, 1, 5, 10, 25, 50]        │
  └──────────────────┴──────────┴──────────────────────────────┘

Run:
    python optimizer_sweep.py                 # full sweep
    python optimizer_sweep.py --max-combos 500  # quick test
    python optimizer_sweep.py --workers 8       # fewer workers

Output:
    sweep_results.csv          — all results sorted by Sharpe
    sweep_results_summary.txt  — top 50 + parameter sensitivity analysis
"""

import os
import sys
import time
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
from multiprocessing import Pool, Manager, cpu_count
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import traceback

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

BACKTEST_START = "2008-01-01"
BACKTEST_END   = "2015-12-31"
DATA_FETCH_START = "2007-01-01"  # 1yr warmup
INITIAL_CAPITAL  = 100_000
RISK_FREE_RATE   = 0.065

# Fixed parameters (not being swept)
EXIT_RANK_CUTOFF = 40
LOOKBACK_12M     = 252
LOOKBACK_6M      = 126
LOOKBACK_3M      = 63
REGIME_DMA       = 200
DMA_EXIT         = 250
FORCE_RISK_ON    = True

# Transaction costs (Zerodha CNC)
STT_BUY        = 0.0
STT_SELL       = 0.001
EXCHANGE_CHARGE = 0.0000325
SEBI_CHARGE     = 0.000001
STAMP_DUTY      = 0.00015

# Tax rates (pre-2024 for this period)
STCG_RATE  = 0.15
LTCG_RATE  = 0.10

# ═══════════════════════════════════════════════════════════════════════════
# PARAMETER GRID — symmetric ranges, no directional bias
# ═══════════════════════════════════════════════════════════════════════════

PARAM_GRID = {
    # Portfolio construction
    "TOP_N":            [5, 7, 10, 12, 15, 18, 20],
    "MAX_PER_SECTOR":   [1, 2, 3, 4, 5, 6],

    # Signal weights — full range from 0 to dominant
    "W_MOM_12M":        [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
    "W_MOM_6M":         [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
    "W_MOM_3M":         [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],

    # Volatility penalty — from strong penalty to mild reward
    "W_VOL":            [-0.30, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10],

    # Momentum skip period (trading days)
    "SKIP_RECENT":      [0, 5, 10, 15, 20, 30, 42],

    # Universe filters
    "MIN_PRICE":        [0, 50, 100, 200, 500],
    "MIN_AVG_VALUE_CR": [0, 1, 5, 10, 25, 50],
}

# ═══════════════════════════════════════════════════════════════════════════
# UNIVERSE (same as backtester.py)
# ═══════════════════════════════════════════════════════════════════════════

UNIVERSE = {
    # Auto
    "APOLLOTYRE":"Auto","BAJAJ-AUTO":"Auto","BALKRISIND":"Auto","BHARATFORG":"Auto",
    "BOSCHLTD":"Auto","EICHERMOT":"Auto","ENDURANCE":"Auto","EXIDEIND":"Auto",
    "HEROMOTOCO":"Auto","HYUNDAI":"Auto","M&M":"Auto","MARUTI":"Auto",
    "MOTHERSON":"Auto","MRF":"Auto","SCHAEFFLER":"Auto","TIINDIA":"Auto",
    "TMPV":"Auto","TVSMOTOR":"Auto","UNOMINDA":"Auto",
    # Capital Goods
    "ABB":"Capital Goods","AIAENG":"Capital Goods","APARINDS":"Capital Goods",
    "APLAPOLLO":"Capital Goods","ASHOKLEY":"Capital Goods","ASTRAL":"Capital Goods",
    "BDL":"Capital Goods","BEL":"Capital Goods","BHEL":"Capital Goods",
    "CGPOWER":"Capital Goods","COCHINSHIP":"Capital Goods","CUMMINSIND":"Capital Goods",
    "ENRIN":"Capital Goods","ESCORTS":"Capital Goods","GVT&D":"Capital Goods",
    "HAL":"Capital Goods","HONAUT":"Capital Goods","KEI":"Capital Goods",
    "MAZDOCK":"Capital Goods","POLYCAB":"Capital Goods","POWERINDIA":"Capital Goods",
    "PREMIERENE":"Capital Goods","SIEMENS":"Capital Goods","SUPREMEIND":"Capital Goods",
    "SUZLON":"Capital Goods","THERMAX":"Capital Goods","TMCV":"Capital Goods",
    "WAAREEENER":"Capital Goods",
    # Cement
    "ACC":"Cement","AMBUJACEM":"Cement","DALBHARAT":"Cement","GRASIM":"Cement",
    "JKCEMENT":"Cement","SHREECEM":"Cement","ULTRACEMCO":"Cement",
    # Chemicals
    "COROMANDEL":"Chemicals","FLUOROCHEM":"Chemicals","LINDEINDIA":"Chemicals",
    "PIDILITIND":"Chemicals","PIIND":"Chemicals","SOLARINDS":"Chemicals",
    "SRF":"Chemicals","UPL":"Chemicals",
    # Consumer
    "ASIANPAINT":"Consumer","BERGEPAINT":"Consumer","BLUESTARCO":"Consumer",
    "DIXON":"Consumer","DMART":"Consumer","ETERNAL":"Consumer","HAVELLS":"Consumer",
    "INDHOTEL":"Consumer","IRCTC":"Consumer","ITCHOTELS":"Consumer",
    "JUBLFOOD":"Consumer","KALYANKJIL":"Consumer","KPRMILL":"Consumer",
    "LENSKART":"Consumer","LGEINDIA":"Consumer","NAUKRI":"Consumer",
    "NYKAA":"Consumer","PAGEIND":"Consumer","SWIGGY":"Consumer","TITAN":"Consumer",
    "TRENT":"Consumer","VMM":"Consumer","VOLTAS":"Consumer",
    # Energy
    "ADANIENSOL":"Energy","ADANIGREEN":"Energy","ADANIPOWER":"Energy","ATGL":"Energy",
    "BPCL":"Energy","COALINDIA":"Energy","GAIL":"Energy","HINDPETRO":"Energy",
    "IOC":"Energy","JSWENERGY":"Energy","NHPC":"Energy","NLCINDIA":"Energy",
    "NTPC":"Energy","NTPCGREEN":"Energy","OIL":"Energy","ONGC":"Energy",
    "PETRONET":"Energy","POWERGRID":"Energy","RELIANCE":"Energy","SJVN":"Energy",
    "TATAPOWER":"Energy","TORNTPOWER":"Energy",
    # FMCG
    "AWL":"FMCG","BRITANNIA":"FMCG","COLPAL":"FMCG","DABUR":"FMCG",
    "GODFRYPHLP":"FMCG","GODREJCP":"FMCG","HINDUNILVR":"FMCG","ITC":"FMCG",
    "MARICO":"FMCG","NESTLEIND":"FMCG","PATANJALI":"FMCG","RADICO":"FMCG",
    "TATACONSUM":"FMCG","UBL":"FMCG","UNITDSPR":"FMCG","VBL":"FMCG",
    # Financials
    "360ONE":"Financials","ABCAPITAL":"Financials","AIIL":"Financials",
    "AUBANK":"Financials","AXISBANK":"Financials","BAJAJFINSV":"Financials",
    "BAJAJHFL":"Financials","BAJAJHLDNG":"Financials","BAJFINANCE":"Financials",
    "BANKBARODA":"Financials","BANKINDIA":"Financials","BSE":"Financials",
    "CANBK":"Financials","CHOLAFIN":"Financials","CRISIL":"Financials",
    "FEDERALBNK":"Financials","GICRE":"Financials","GROWW":"Financials",
    "HDBFS":"Financials","HDFCAMC":"Financials","HDFCBANK":"Financials",
    "HDFCLIFE":"Financials","HUDCO":"Financials","ICICIAMC":"Financials",
    "ICICIBANK":"Financials","ICICIGI":"Financials","ICICIPRULI":"Financials",
    "IDFCFIRSTB":"Financials","INDIANB":"Financials","INDUSINDBK":"Financials",
    "IREDA":"Financials","IRFC":"Financials","JIOFIN":"Financials",
    "KOTAKBANK":"Financials","LICHSGFIN":"Financials","LICI":"Financials",
    "LTF":"Financials","M&MFIN":"Financials","MAHABANK":"Financials",
    "MCX":"Financials","MFSL":"Financials","MOTILALOFS":"Financials",
    "MUTHOOTFIN":"Financials","NAM-INDIA":"Financials","NIACL":"Financials",
    "PAYTM":"Financials","PFC":"Financials","PNB":"Financials",
    "POLICYBZR":"Financials","RECLTD":"Financials","SBICARD":"Financials",
    "SBILIFE":"Financials","SBIN":"Financials","SHRIRAMFIN":"Financials",
    "SUNDARMFIN":"Financials","TATACAP":"Financials","TATAINVEST":"Financials",
    "UNIONBANK":"Financials","YESBANK":"Financials",
    # Healthcare
    "ABBOTINDIA":"Healthcare","AJANTPHARM":"Healthcare","ALKEM":"Healthcare",
    "ANTHEM":"Healthcare","APOLLOHOSP":"Healthcare","AUROPHARMA":"Healthcare",
    "BIOCON":"Healthcare","CIPLA":"Healthcare","DIVISLAB":"Healthcare",
    "DRREDDY":"Healthcare","FORTIS":"Healthcare","GLAXO":"Healthcare",
    "GLENMARK":"Healthcare","IPCALAB":"Healthcare","LAURUSLABS":"Healthcare",
    "LUPIN":"Healthcare","MANKIND":"Healthcare","MAXHEALTH":"Healthcare",
    "MEDANTA":"Healthcare","SUNPHARMA":"Healthcare","TORNTPHARM":"Healthcare",
    "ZYDUSLIFE":"Healthcare",
    # IT
    "COFORGE":"IT","HCLTECH":"IT","HEXT":"IT","INFY":"IT","KPITTECH":"IT",
    "LTM":"IT","LTTS":"IT","MPHASIS":"IT","OFSS":"IT","PERSISTENT":"IT",
    "TATAELXSI":"IT","TCS":"IT","TECHM":"IT","WIPRO":"IT",
    # Industrials
    "3MINDIA":"Industrials","ADANIPORTS":"Industrials","CONCOR":"Industrials",
    "GMRAIRPORT":"Industrials","GODREJIND":"Industrials","INDIGO":"Industrials",
    "JSWINFRA":"Industrials","LT":"Industrials","RVNL":"Industrials",
    # Metals
    "ADANIENT":"Metals","HINDALCO":"Metals","HINDZINC":"Metals",
    "JINDALSTEL":"Metals","JSL":"Metals","JSWSTEEL":"Metals","LLOYDSME":"Metals",
    "NATIONALUM":"Metals","NMDC":"Metals","SAIL":"Metals","TATASTEEL":"Metals",
    "VEDL":"Metals",
    # Realty
    "DLF":"Realty","GODREJPROP":"Realty","LODHA":"Realty","OBEROIRLTY":"Realty",
    "PHOENIXLTD":"Realty","PRESTIGE":"Realty",
    # Telecom
    "BHARTIARTL":"Telecom","BHARTIHEXA":"Telecom","IDEA":"Telecom",
    "INDUSTOWER":"Telecom","TATACOMM":"Telecom",
}


# ═══════════════════════════════════════════════════════════════════════════
# TRANSACTION COST & TAX (inline — no config dependency)
# ═══════════════════════════════════════════════════════════════════════════

def txn_cost(value, side):
    cost = value * (EXCHANGE_CHARGE + SEBI_CHARGE)
    cost += value * STAMP_DUTY  if side == "buy"  else 0
    cost += value * STT_SELL    if side == "sell"  else 0
    cost += 15.93               if side == "sell"  else 0  # DP charge
    return cost

def capital_gains_tax(gain, entry_date, sell_date):
    if gain <= 0:
        return 0.0
    days_held = (pd.Timestamp(sell_date) - pd.Timestamp(entry_date)).days
    return gain * (LTCG_RATE if days_held >= 365 else STCG_RATE)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING (one-time)
# ═══════════════════════════════════════════════════════════════════════════

def load_data_once():
    """Load all data from cache files. Called once before spawning workers."""
    print("Loading cached data...")
    close     = pd.read_csv("price_data_cache.csv",  index_col=0, parse_dates=True)
    volume    = pd.read_csv("volume_data_cache.csv",  index_col=0, parse_dates=True)
    regime_df = pd.read_csv("regime_data_cache.csv",  index_col=0, parse_dates=True)

    open_path = "open_data_cache.csv"
    if os.path.exists(open_path):
        open_prices = pd.read_csv(open_path, index_col=0, parse_dates=True)
    else:
        open_prices = close.copy()

    nifty500 = regime_df["nifty500"]
    nifty50  = regime_df["nifty50"] if "nifty50" in regime_df.columns else nifty500

    print(f"Loaded: {close.shape[1]} stocks × {close.shape[0]} days")
    print(f"Date range: {close.index[0].strftime('%Y-%m-%d')} → {close.index[-1].strftime('%Y-%m-%d')}")
    return close, volume, nifty500, nifty50, open_prices


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE (pure function — no global state)
# ═══════════════════════════════════════════════════════════════════════════

def get_regime(nifty500, date):
    d = nifty500.loc[:date].dropna()
    if len(d) < REGIME_DMA:
        return "RISK-ON"
    dma = d.rolling(REGIME_DMA).mean()
    return "RISK-ON" if d.iloc[-1] > dma.iloc[-1] else "RISK-OFF"


def is_above_dma_exit(close, ticker, date):
    if ticker not in close.columns:
        return True
    p = close[ticker].loc[:date].dropna()
    if len(p) < DMA_EXIT:
        return True
    return p.iloc[-1] >= p.rolling(DMA_EXIT).mean().iloc[-1]


def compute_scores(close, volume, date, tickers, params):
    """Score stocks using parameter overrides."""
    records = []
    c_slice = close.loc[:date]

    for t in tickers:
        if t not in c_slice.columns:
            continue
        c = c_slice[t].dropna()
        v = volume[t].loc[:date].dropna() if t in volume.columns else pd.Series(dtype=float)

        # Liquidity filters
        if len(c) < 60 or c.iloc[-1] < params["MIN_PRICE"]:
            continue
        if len(v) >= 60:
            if (c * v).rolling(60).mean().iloc[-1] / 1e7 < params["MIN_AVG_VALUE_CR"]:
                continue

        s = params["SKIP_RECENT"]
        if len(c) < LOOKBACK_12M + s + 1:
            continue

        p_now  = c.iloc[-(s+1)]
        mom12  = (p_now - c.iloc[-(LOOKBACK_12M+s)]) / c.iloc[-(LOOKBACK_12M+s)]
        mom6   = (p_now - c.iloc[-(LOOKBACK_6M+s)])  / c.iloc[-(LOOKBACK_6M+s)]
        mom3   = (p_now - c.iloc[-(LOOKBACK_3M+s)])  / c.iloc[-(LOOKBACK_3M+s)]
        vol6   = c.iloc[-LOOKBACK_6M:].pct_change().dropna().std() * np.sqrt(252)

        records.append({
            "ticker": t, "sector": UNIVERSE.get(t, "Unknown"),
            "mom12": mom12, "mom6": mom6, "mom3": mom3, "vol6": vol6
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("ticker")

    def z(s):
        return (s - s.mean()) / s.std() if s.std() > 0 else s * 0

    df["score"] = (params["W_MOM_12M"] * z(df["mom12"]) +
                   params["W_MOM_6M"]  * z(df["mom6"])  +
                   params["W_MOM_3M"]  * z(df["mom3"])  +
                   params["W_VOL"]     * z(df["vol6"]))
    return df.sort_values("score", ascending=False)


def pick_portfolio(scored, params):
    sel, sc = [], {}
    for t, row in scored.iterrows():
        s = row["sector"]
        if sc.get(s, 0) < params["MAX_PER_SECTOR"]:
            sel.append(t)
            sc[s] = sc.get(s, 0) + 1
        if len(sel) == params["TOP_N"]:
            break
    return sel


def find_replacement(scored, current_holdings, exits, params):
    if scored.empty:
        return []
    remaining  = [t for t in current_holdings if t not in exits]
    top_cutoff = scored.head(EXIT_RANK_CUTOFF).index.tolist()
    candidates = [t for t in top_cutoff if t not in remaining]

    sector_count = {}
    for t in remaining:
        s = UNIVERSE.get(t, "Unknown")
        sector_count[s] = sector_count.get(s, 0) + 1

    replacements = []
    for t in candidates:
        s = UNIVERSE.get(t, "Unknown")
        if sector_count.get(s, 0) < params["MAX_PER_SECTOR"]:
            replacements.append(t)
            sector_count[s] = sector_count.get(s, 0) + 1
        if len(replacements) == len(exits):
            break
    return replacements


def get_price(close, ticker, date):
    try:
        return float(close[ticker].loc[:date].dropna().iloc[-1])
    except Exception:
        return 0.0

def get_open_price(open_prices, ticker, date):
    try:
        if ticker not in open_prices.columns:
            return 0.0
        if date in open_prices.index:
            val = open_prices[ticker].loc[date]
            return float(val) if pd.notna(val) else 0.0
        return 0.0
    except Exception:
        return 0.0


def get_fill_price(close_price, open_price, side, rank=0):
    if side == "buy":
        if rank < 5:
            buffer = 0.050
        elif rank < 12:
            buffer = 0.030
        else:
            buffer = 0.020
        limit = close_price * (1 + buffer)
        if open_price > 0 and open_price <= limit:
            return open_price
        elif open_price == 0:
            return close_price
        else:
            return None
    else:
        limit = close_price * 0.990
        if open_price > 0 and open_price >= limit:
            return open_price
        elif open_price == 0:
            return close_price
        else:
            return None


def run_single_backtest(close, volume, nifty500, nifty50, open_prices, params):
    """
    Run one complete backtest with the given parameters.
    Returns dict with all performance metrics.
    """
    all_days = close.loc[BACKTEST_START:BACKTEST_END].index
    if len(all_days) < 60:
        return None

    cash       = float(INITIAL_CAPITAL)
    holdings   = {}
    entry_info = {}
    eq_curve   = []
    tickers    = list(UNIVERSE.keys())
    trade_count = 0

    cached_scored   = pd.DataFrame()
    cached_top_25   = []
    cached_top_n    = []
    in_risk_off     = False
    last_rebalance_month = None

    all_days_list = list(all_days)

    for day_idx, day in enumerate(all_days_list):
        next_day  = all_days_list[day_idx + 1] if day_idx + 1 < len(all_days_list) else day
        cur_month = pd.Timestamp(day).to_period("M")
        is_rebalance = (cur_month != last_rebalance_month)

        # Regime check
        if FORCE_RISK_ON:
            regime = "RISK-ON"
        else:
            regime = get_regime(nifty500, day)

        # RISK-OFF: liquidate
        if regime == "RISK-OFF":
            if not in_risk_off:
                for t, sh in list(holdings.items()):
                    if sh > 0 and t in close.columns:
                        px = get_price(close, t, day)
                        if px > 0:
                            open_px = get_open_price(open_prices, t, next_day)
                            fill_px = get_fill_price(px, open_px, "sell")
                            fill_px = fill_px if fill_px else open_px if open_px > 0 else px
                            proceeds = sh * fill_px
                            cost     = txn_cost(proceeds, "sell")
                            info     = entry_info.get(t, {})
                            gain     = (fill_px - info.get("avg_price", fill_px)) * sh
                            tax      = capital_gains_tax(gain, info.get("entry_date", day), day)
                            cash    += proceeds - cost - tax
                            entry_info.pop(t, None)
                            trade_count += 1
                holdings    = {}
                in_risk_off = True
                cached_scored = pd.DataFrame()
                cached_top_25 = []
                cached_top_n  = []
            eq_curve.append({"date": day, "value": cash})
            if is_rebalance:
                last_rebalance_month = cur_month
            continue

        in_risk_off = False

        # Rebalance day: recompute scores
        if is_rebalance:
            cached_scored = compute_scores(close, volume, day, tickers, params)
            if not cached_scored.empty:
                cached_top_25 = cached_scored.head(EXIT_RANK_CUTOFF).index.tolist()
                cached_top_n  = pick_portfolio(cached_scored, params)

        # Daily exit check
        current_held = [t for t, s in holdings.items() if s > 0]
        exits = []
        for t in current_held:
            exit_reason = None
            if cached_top_25 and t not in cached_top_25:
                exit_reason = "rank"
            if t in close.columns and not is_above_dma_exit(close, t, day):
                exit_reason = "100DMA"
            if exit_reason:
                exits.append((t, exit_reason))

        exit_tickers = [t for t, _ in exits]

        # Process exits
        for t, reason in exits:
            sh = holdings.get(t, 0)
            if sh > 0:
                px = get_price(close, t, day)
                if px > 0:
                    open_px  = get_open_price(open_prices, t, next_day)
                    fill_px  = get_fill_price(px, open_px, "sell")
                    fill_px  = fill_px if fill_px else open_px if open_px > 0 else px
                    proceeds = sh * fill_px
                    cost     = txn_cost(proceeds, "sell")
                    info     = entry_info.get(t, {})
                    gain     = (fill_px - info.get("avg_price", fill_px)) * sh
                    tax      = capital_gains_tax(gain, info.get("entry_date", day), day)
                    cash    += proceeds - cost - tax
                    entry_info.pop(t, None)
                    holdings[t] = 0
                    trade_count += 1

        # Buy replacements for exits
        if exit_tickers and not cached_scored.empty:
            replacements = find_replacement(cached_scored, current_held, exit_tickers, params)
            port_val = cash
            for t, sh in holdings.items():
                if sh > 0:
                    port_val += get_price(close, t, day) * sh

            stocks_to_hold = max(1, params["TOP_N"])
            target = port_val / stocks_to_hold

            for t in replacements:
                px = get_price(close, t, day)
                if px <= 0:
                    continue
                rank    = cached_scored.index.get_loc(t) if t in cached_scored.index else 99
                open_px = get_open_price(open_prices, t, next_day)
                fill_px = get_fill_price(px, open_px, "buy", rank)
                if fill_px is None:
                    continue
                cur_val = holdings.get(t, 0) * fill_px
                if cur_val < target * 0.95:
                    n    = int((target - cur_val) / fill_px)
                    cost = n * fill_px
                    tc   = txn_cost(cost, "buy")
                    if n > 0 and cash >= cost + tc:
                        cash -= (cost + tc)
                        old_sh  = holdings.get(t, 0)
                        new_sh  = old_sh + n
                        old_avg = entry_info.get(t, {}).get("avg_price", fill_px)
                        new_avg = (old_sh * old_avg + n * fill_px) / new_sh if new_sh > 0 else fill_px
                        entry_info[t] = {
                            "avg_price":  new_avg,
                            "entry_date": entry_info.get(t, {}).get("entry_date", day),
                        }
                        holdings[t] = new_sh
                        trade_count += 1

        # Monthly full rotation
        if is_rebalance and not cached_scored.empty:
            current_held_after = [t for t, s in holdings.items() if s > 0]
            rotate_out = [t for t in current_held_after if t not in cached_top_n]

            for t in rotate_out:
                sh = holdings.get(t, 0)
                if sh > 0:
                    px = get_price(close, t, day)
                    if px > 0:
                        proceeds = sh * px
                        cost     = txn_cost(proceeds, "sell")
                        info     = entry_info.get(t, {})
                        gain     = (px - info.get("avg_price", px)) * sh
                        tax      = capital_gains_tax(gain, info.get("entry_date", day), day)
                        cash    += proceeds - cost - tax
                        entry_info.pop(t, None)
                        holdings[t] = 0
                        trade_count += 1

            # Recompute portfolio value
            port_val = cash
            for t, sh in holdings.items():
                if sh > 0:
                    port_val += get_price(close, t, day) * sh

            stocks_to_hold = max(1, params["TOP_N"])
            target = port_val / stocks_to_hold

            bought = 0
            for t in cached_top_n:
                if bought >= stocks_to_hold:
                    break
                px = get_price(close, t, day)
                if px <= 0 or px > target:
                    continue
                rank    = cached_scored.index.get_loc(t) if t in cached_scored.index else 99
                open_px = get_open_price(open_prices, t, next_day)
                fill_px = get_fill_price(px, open_px, "buy", rank)
                if fill_px is None:
                    continue
                cur_val = holdings.get(t, 0) * fill_px
                if cur_val < target * 0.95:
                    n    = int((target - cur_val) / fill_px)
                    cost = n * fill_px
                    tc   = txn_cost(cost, "buy")
                    if n > 0 and cash >= cost + tc:
                        cash -= (cost + tc)
                        old_sh  = holdings.get(t, 0)
                        new_sh  = old_sh + n
                        old_avg = entry_info.get(t, {}).get("avg_price", fill_px)
                        new_avg = (old_sh * old_avg + n * fill_px) / new_sh if new_sh > 0 else fill_px
                        entry_info[t] = {
                            "avg_price":  new_avg,
                            "entry_date": entry_info.get(t, {}).get("entry_date", day),
                        }
                        holdings[t] = new_sh
                        bought += 1
                        trade_count += 1

            last_rebalance_month = cur_month

        # Record daily value
        day_val = cash
        for t, sh in holdings.items():
            if sh > 0:
                day_val += sh * get_price(close, t, day)
        eq_curve.append({"date": day, "value": day_val})

    # ── Compute metrics ───────────────────────────────────────────────────
    if len(eq_curve) < 60:
        return None

    eq_df  = pd.DataFrame(eq_curve).set_index("date")
    eq_df.index = pd.to_datetime(eq_df.index)
    equity = eq_df["value"]

    monthly = equity.resample("MS").last().dropna()
    if len(monthly) < 12:
        return None

    ret    = monthly.pct_change().dropna()
    total  = (equity.iloc[-1] / equity.iloc[0]) - 1
    n_yr   = len(ret) / 12
    cagr   = (1 + total) ** (1 / n_yr) - 1 if n_yr > 0 else 0
    mrf    = (1 + RISK_FREE_RATE) ** (1/12) - 1
    exc    = ret - mrf
    sharpe = exc.mean() / exc.std() * np.sqrt(12) if exc.std() > 0 else 0
    down   = exc[exc < 0]
    sortino = exc.mean() / down.std() * np.sqrt(12) if len(down) > 0 and down.std() > 0 else 0
    rollmax = equity.cummax()
    maxdd   = float(((equity - rollmax) / rollmax).min())
    winrate = float((ret > 0).sum() / len(ret)) if len(ret) > 0 else 0

    # Calmar ratio
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0

    # Annual returns
    yearly = equity.resample("YS").last().dropna().pct_change().dropna()
    worst_year  = float(yearly.min()) if len(yearly) > 0 else 0
    best_year   = float(yearly.max()) if len(yearly) > 0 else 0

    # Benchmark comparison
    bench_raw   = nifty50.loc[BACKTEST_START:BACKTEST_END].dropna()
    if len(bench_raw) > 12:
        bench_curve = (bench_raw / bench_raw.iloc[0]) * INITIAL_CAPITAL
        bench_monthly = bench_curve.resample("MS").last().dropna()
        bench_ret = bench_monthly.pct_change().dropna()
        bench_total = (bench_curve.iloc[-1] / bench_curve.iloc[0]) - 1
        bench_n_yr  = len(bench_ret) / 12
        bench_cagr  = (1 + bench_total) ** (1 / bench_n_yr) - 1 if bench_n_yr > 0 else 0
    else:
        bench_cagr = 0

    return {
        "cagr":         round(cagr * 100, 2),
        "sharpe":       round(sharpe, 4),
        "sortino":      round(sortino, 4),
        "maxdd":        round(maxdd * 100, 2),
        "calmar":       round(calmar, 4),
        "winrate":      round(winrate * 100, 1),
        "total_return": round(total * 100, 2),
        "worst_year":   round(worst_year * 100, 2),
        "best_year":    round(best_year * 100, 2),
        "trades":       trade_count,
        "final_value":  round(float(equity.iloc[-1]), 0),
        "alpha_cagr":   round((cagr - bench_cagr) * 100, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# SAMPLING STRATEGY
# ═══════════════════════════════════════════════════════════════════════════

def generate_grid_combos():
    """Full cartesian product — use only if small enough."""
    keys = list(PARAM_GRID.keys())
    vals = [PARAM_GRID[k] for k in keys]
    total = 1
    for v in vals:
        total *= len(v)
    return total, keys, vals


def latin_hypercube_sample(n_samples, seed=42):
    """
    Latin Hypercube Sampling across the parameter grid.
    Maps continuous LHS samples to discrete grid values.
    """
    rng  = np.random.RandomState(seed)
    keys = list(PARAM_GRID.keys())
    dims = len(keys)

    # Generate LHS samples
    samples = np.zeros((n_samples, dims))
    for i in range(dims):
        perm = rng.permutation(n_samples)
        samples[:, i] = (perm + rng.uniform(size=n_samples)) / n_samples

    combos = []
    for row in samples:
        combo = {}
        for j, key in enumerate(keys):
            grid_vals = PARAM_GRID[key]
            idx = int(row[j] * len(grid_vals))
            idx = min(idx, len(grid_vals) - 1)
            combo[key] = grid_vals[idx]
        combos.append(combo)

    return combos


def generate_combos(max_combos=None):
    """
    Generate parameter combinations.
    Uses full grid if small enough, otherwise LHS sampling.
    """
    full_grid_size, keys, vals = generate_grid_combos()
    print(f"\nFull grid size: {full_grid_size:,} combinations")

    if max_combos is None:
        # Estimate: each backtest takes ~2-3 seconds in-process
        # With 30 workers on c4-standard-32: ~30 combos/sec throughput
        # 12 hours = 43,200 sec → ~1.3M combos theoretical max
        # Safety margin (50%): target 600,000 combos
        # But scoring is the bottleneck. Realistic: ~5-8 sec per backtest
        # → 30 workers × 43200/7 ≈ 185,000 combos
        # Conservative target: 100,000 combos
        max_combos = min(full_grid_size, 100_000)

    if full_grid_size <= max_combos:
        print(f"Using full grid: {full_grid_size:,} combinations")
        combos = []
        for combo_vals in product(*vals):
            combos.append(dict(zip(keys, combo_vals)))
        return combos
    else:
        print(f"Grid too large. Using Latin Hypercube Sampling: {max_combos:,} samples")
        # Also always include the current default + anchor points
        combos = latin_hypercube_sample(max_combos - 1, seed=42)

        # Ensure current default is always tested
        default = {
            "TOP_N": 10, "MAX_PER_SECTOR": 3,
            "W_MOM_12M": 0.40, "W_MOM_6M": 0.35, "W_MOM_3M": 0.15,
            "W_VOL": -0.10, "SKIP_RECENT": 20,
            "MIN_PRICE": 0, "MIN_AVG_VALUE_CR": 0,
        }
        combos.insert(0, default)

        # Deduplicate
        seen = set()
        unique = []
        for c in combos:
            key = tuple(sorted(c.items()))
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique


# ═══════════════════════════════════════════════════════════════════════════
# WORKER (multiprocessing entry point)
# ═══════════════════════════════════════════════════════════════════════════

# Global data references (set in worker initializer)
_CLOSE = None
_VOLUME = None
_NIFTY500 = None
_NIFTY50 = None
_OPEN = None

def init_worker(close_path, volume_path, regime_path, open_path):
    """Each worker loads data from CSV (shared memory via OS page cache)."""
    global _CLOSE, _VOLUME, _NIFTY500, _NIFTY50, _OPEN
    _CLOSE   = pd.read_csv(close_path,  index_col=0, parse_dates=True)
    _VOLUME  = pd.read_csv(volume_path, index_col=0, parse_dates=True)
    regime   = pd.read_csv(regime_path, index_col=0, parse_dates=True)
    _NIFTY500 = regime["nifty500"]
    _NIFTY50  = regime["nifty50"] if "nifty50" in regime.columns else regime["nifty500"]
    if os.path.exists(open_path):
        _OPEN = pd.read_csv(open_path, index_col=0, parse_dates=True)
    else:
        _OPEN = _CLOSE.copy()


def worker_run(args):
    """Worker entry: runs one backtest, returns (combo_id, params, metrics)."""
    combo_id, params = args
    try:
        metrics = run_single_backtest(_CLOSE, _VOLUME, _NIFTY500, _NIFTY50, _OPEN, params)
        return (combo_id, params, metrics)
    except Exception as e:
        return (combo_id, params, {"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS & REPORTING
# ═══════════════════════════════════════════════════════════════════════════

def parameter_sensitivity(df, param_name):
    """Compute average metrics for each value of a parameter."""
    if param_name not in df.columns:
        return pd.DataFrame()
    grouped = df.groupby(param_name).agg({
        "cagr":    ["mean", "std", "count"],
        "sharpe":  ["mean", "std"],
        "maxdd":   ["mean", "std"],
        "sortino": ["mean"],
        "calmar":  ["mean"],
    }).round(3)
    return grouped


def generate_report(results_df, elapsed_sec):
    """Generate comprehensive summary report."""
    lines = []
    lines.append("=" * 80)
    lines.append("  PARAMETER SWEEP — COMPREHENSIVE RESULTS")
    lines.append(f"  Period: {BACKTEST_START} → {BACKTEST_END}")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Runtime: {elapsed_sec/3600:.1f} hours ({elapsed_sec:.0f} seconds)")
    lines.append(f"  Combinations tested: {len(results_df):,}")
    lines.append("=" * 80)

    # ── TOP 50 BY SHARPE ─────────────────────────────────────────────────
    lines.append("\n" + "─" * 80)
    lines.append("  TOP 50 COMBINATIONS — RANKED BY SHARPE RATIO")
    lines.append("─" * 80)

    top50 = results_df.nlargest(50, "sharpe")
    header = (f"{'Rank':>4} {'N':>3} {'S':>2} {'w12':>5} {'w6':>5} {'w3':>5} "
              f"{'wV':>6} {'Skip':>4} {'MinP':>5} {'MinV':>5} | "
              f"{'CAGR':>7} {'Sharpe':>7} {'Sort':>7} {'MaxDD':>7} {'Calm':>6} {'Trades':>6}")
    lines.append(header)
    lines.append("-" * len(header))

    for i, (_, r) in enumerate(top50.iterrows(), 1):
        lines.append(
            f"{i:>4} {r['TOP_N']:>3.0f} {r['MAX_PER_SECTOR']:>2.0f} "
            f"{r['W_MOM_12M']:>5.2f} {r['W_MOM_6M']:>5.2f} {r['W_MOM_3M']:>5.2f} "
            f"{r['W_VOL']:>6.2f} {r['SKIP_RECENT']:>4.0f} {r['MIN_PRICE']:>5.0f} "
            f"{r['MIN_AVG_VALUE_CR']:>5.0f} | "
            f"{r['cagr']:>6.1f}% {r['sharpe']:>7.3f} {r['sortino']:>7.3f} "
            f"{r['maxdd']:>6.1f}% {r['calmar']:>6.3f} {r['trades']:>6.0f}"
        )

    # ── TOP 30 BY CALMAR ─────────────────────────────────────────────────
    lines.append("\n" + "─" * 80)
    lines.append("  TOP 30 COMBINATIONS — RANKED BY CALMAR RATIO (CAGR / MaxDD)")
    lines.append("─" * 80)

    top30_calmar = results_df.nlargest(30, "calmar")
    lines.append(header)
    lines.append("-" * len(header))

    for i, (_, r) in enumerate(top30_calmar.iterrows(), 1):
        lines.append(
            f"{i:>4} {r['TOP_N']:>3.0f} {r['MAX_PER_SECTOR']:>2.0f} "
            f"{r['W_MOM_12M']:>5.2f} {r['W_MOM_6M']:>5.2f} {r['W_MOM_3M']:>5.2f} "
            f"{r['W_VOL']:>6.2f} {r['SKIP_RECENT']:>4.0f} {r['MIN_PRICE']:>5.0f} "
            f"{r['MIN_AVG_VALUE_CR']:>5.0f} | "
            f"{r['cagr']:>6.1f}% {r['sharpe']:>7.3f} {r['sortino']:>7.3f} "
            f"{r['maxdd']:>6.1f}% {r['calmar']:>6.3f} {r['trades']:>6.0f}"
        )

    # ── PARAMETER SENSITIVITY ────────────────────────────────────────────
    lines.append("\n" + "=" * 80)
    lines.append("  PARAMETER SENSITIVITY ANALYSIS")
    lines.append("  (Average metrics across all combos with each parameter value)")
    lines.append("=" * 80)

    for param in PARAM_GRID.keys():
        lines.append(f"\n  ── {param} ──")
        if param not in results_df.columns:
            lines.append("    (not found in results)")
            continue

        grp = results_df.groupby(param).agg(
            cagr_mean=("cagr", "mean"),
            cagr_std=("cagr", "std"),
            sharpe_mean=("sharpe", "mean"),
            sharpe_std=("sharpe", "std"),
            maxdd_mean=("maxdd", "mean"),
            sortino_mean=("sortino", "mean"),
            calmar_mean=("calmar", "mean"),
            count=("cagr", "count"),
        ).round(3)

        lines.append(f"  {'Value':>8} {'CAGR(μ)':>8} {'CAGR(σ)':>8} {'Sharpe(μ)':>10} "
                     f"{'MaxDD(μ)':>9} {'Calmar(μ)':>10} {'Count':>6}")
        lines.append("  " + "-" * 65)

        for val, row in grp.iterrows():
            lines.append(
                f"  {val:>8} {row['cagr_mean']:>7.1f}% {row['cagr_std']:>7.1f}% "
                f"{row['sharpe_mean']:>10.3f} {row['maxdd_mean']:>8.1f}% "
                f"{row['calmar_mean']:>10.3f} {row['count']:>6.0f}"
            )

    # ── OVERALL STATISTICS ───────────────────────────────────────────────
    lines.append("\n" + "=" * 80)
    lines.append("  OVERALL DISTRIBUTION")
    lines.append("=" * 80)
    for metric in ["cagr", "sharpe", "sortino", "maxdd", "calmar"]:
        if metric in results_df.columns:
            s = results_df[metric]
            lines.append(
                f"  {metric:>8}: mean={s.mean():>8.2f}  std={s.std():>7.2f}  "
                f"min={s.min():>8.2f}  p25={s.quantile(0.25):>8.2f}  "
                f"median={s.median():>8.2f}  p75={s.quantile(0.75):>8.2f}  "
                f"max={s.max():>8.2f}"
            )

    # ── STABLE REGIONS ───────────────────────────────────────────────────
    lines.append("\n" + "=" * 80)
    lines.append("  STABLE PARAMETER REGIONS")
    lines.append("  (Top 10% by Sharpe — parameter ranges that consistently work)")
    lines.append("=" * 80)

    top10pct = results_df.nlargest(max(10, len(results_df) // 10), "sharpe")
    for param in PARAM_GRID.keys():
        if param in top10pct.columns:
            vals = top10pct[param]
            mode = vals.mode()
            mode_str = ", ".join(str(v) for v in mode.values[:3])
            lines.append(
                f"  {param:>18}: range [{vals.min()}, {vals.max()}]  "
                f"median={vals.median():.2f}  mode={mode_str}"
            )

    lines.append("\n" + "=" * 80)
    report = "\n".join(lines)
    return report


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Parameter Sweep Backtester")
    parser.add_argument("--max-combos", type=int, default=None,
                        help="Max combinations to test (default: auto)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: CPU count - 2)")
    parser.add_argument("--output", type=str, default="sweep_results.csv",
                        help="Output CSV file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for LHS sampling")
    args = parser.parse_args()

    n_workers = args.workers or max(1, cpu_count() - 2)

    print("=" * 70)
    print("  PARAMETER SWEEP BACKTESTER")
    print(f"  Period     : {BACKTEST_START} → {BACKTEST_END}")
    print(f"  Workers    : {n_workers}")
    print(f"  CPU count  : {cpu_count()}")
    print(f"  Target VM  : c4-standard-32 (32 vCPUs)")
    print("=" * 70)

    # ── Validate data exists ────────────────────────────────────────────
    for f in ["price_data_cache.csv", "volume_data_cache.csv", "regime_data_cache.csv"]:
        if not os.path.exists(f):
            print(f"ERROR: Missing {f}. Run backtester.py first to create cache.")
            sys.exit(1)

    # ── Quick data validation ───────────────────────────────────────────
    close, volume, nifty500, nifty50, open_prices = load_data_once()
    bt_days = close.loc[BACKTEST_START:BACKTEST_END].index
    print(f"Trading days in period: {len(bt_days)}")

    # ── Generate combinations ───────────────────────────────────────────
    combos = generate_combos(args.max_combos)
    total  = len(combos)
    print(f"\nTotal combinations to test: {total:,}")

    # Time estimate
    est_per_combo = 5.0  # seconds (conservative)
    est_total     = (total * est_per_combo) / n_workers / 3600
    print(f"Estimated runtime: {est_total:.1f} hours (at ~{est_per_combo}s/combo, {n_workers} workers)")
    if est_total > 12:
        print(f"WARNING: Estimated {est_total:.1f}h exceeds 12h budget!")
        # Auto-reduce
        max_feasible = int(12 * 3600 * n_workers / est_per_combo * 0.9)
        print(f"Auto-reducing to {max_feasible:,} combinations")
        combos = combos[:max_feasible]
        total  = len(combos)
        print(f"Adjusted total: {total:,}")

    # ── Run benchmark first ─────────────────────────────────────────────
    print("\n── Running current default configuration... ──")
    default_params = {
        "TOP_N": 10, "MAX_PER_SECTOR": 3,
        "W_MOM_12M": 0.40, "W_MOM_6M": 0.35, "W_MOM_3M": 0.15,
        "W_VOL": -0.10, "SKIP_RECENT": 20,
        "MIN_PRICE": 0, "MIN_AVG_VALUE_CR": 0,
    }
    t0_bench = time.time()
    bench_result = run_single_backtest(close, volume, nifty500, nifty50, open_prices, default_params)
    bench_time = time.time() - t0_bench
    print(f"Default result (took {bench_time:.1f}s):")
    if bench_result and "error" not in bench_result:
        print(f"  CAGR={bench_result['cagr']:.1f}%  Sharpe={bench_result['sharpe']:.3f}  "
              f"MaxDD={bench_result['maxdd']:.1f}%  Sortino={bench_result['sortino']:.3f}")

    # Update time estimate with actual benchmark
    est_per_combo = bench_time * 1.1  # 10% safety margin
    est_total = (total * est_per_combo) / n_workers / 3600
    print(f"\nRevised estimate: {est_total:.1f} hours (actual {bench_time:.1f}s/combo)")

    if est_total > 12:
        max_feasible = int(12 * 3600 * n_workers / est_per_combo * 0.9)
        print(f"Auto-reducing to {max_feasible:,} combinations to fit 12h budget")
        combos = combos[:max_feasible]
        total  = len(combos)

    # ── Prepare work items ──────────────────────────────────────────────
    work_items = [(i, combo) for i, combo in enumerate(combos)]

    # ── Parallel execution ──────────────────────────────────────────────
    open_path = "open_data_cache.csv"
    print(f"\n{'='*70}")
    print(f"  LAUNCHING {total:,} BACKTESTS across {n_workers} workers")
    print(f"{'='*70}\n")

    results  = []
    errors   = 0
    t_start  = time.time()
    last_print = 0

    with Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=("price_data_cache.csv", "volume_data_cache.csv",
                   "regime_data_cache.csv", open_path),
    ) as pool:
        for result in pool.imap_unordered(worker_run, work_items, chunksize=10):
            combo_id, params, metrics = result

            if metrics is None or (isinstance(metrics, dict) and "error" in metrics):
                errors += 1
                continue

            row = {**params, **metrics}
            results.append(row)

            done = len(results) + errors
            elapsed = time.time() - t_start

            # Progress every 100 combos or 30 seconds
            if done - last_print >= 100 or elapsed - (last_print * elapsed / max(done, 1)) > 30:
                pct  = done / total * 100
                rate = done / elapsed if elapsed > 0 else 0
                eta  = (total - done) / rate / 3600 if rate > 0 else 0
                best_sharpe = max((r.get("sharpe", 0) for r in results), default=0)
                print(f"[{done:>6,}/{total:,}] {pct:>5.1f}% | "
                      f"{rate:.1f}/s | ETA {eta:.1f}h | "
                      f"Best Sharpe: {best_sharpe:.3f} | Errors: {errors}")
                last_print = done

    elapsed_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  COMPLETE: {len(results):,} successful / {errors} errors "
          f"in {elapsed_total/3600:.1f} hours")
    print(f"{'='*70}")

    # ── Save results ────────────────────────────────────────────────────
    if not results:
        print("ERROR: No successful backtests. Check data.")
        sys.exit(1)

    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output} ({len(df):,} rows)")

    # ── Generate report ─────────────────────────────────────────────────
    report = generate_report(df, elapsed_total)
    report_file = args.output.replace(".csv", "_summary.txt")
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Saved: {report_file}")
    print(report)

    # ── Save parameter grid for reproducibility ─────────────────────────
    meta = {
        "backtest_start": BACKTEST_START,
        "backtest_end":   BACKTEST_END,
        "param_grid":     {k: [str(v) for v in vals] for k, vals in PARAM_GRID.items()},
        "total_combos":   total,
        "successful":     len(results),
        "errors":         errors,
        "runtime_hours":  round(elapsed_total / 3600, 2),
        "workers":        n_workers,
        "generated_at":   datetime.now().isoformat(),
    }
    meta_file = args.output.replace(".csv", "_meta.json")
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {meta_file}")


if __name__ == "__main__":
    main()

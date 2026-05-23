"""
data_manager.py
===============
Manages the price data cache for the trading system.

Run this FIRST every day before execution.py.
It fetches only the latest few days and appends to the existing cache.
Much faster than re-downloading everything — takes ~30 seconds.

Daily workflow:
    3:40 PM → python data_manager.py    (update cache with today's close)
    3:45 PM → python execution.py       (run signals + place orders)

Run manually anytime:
    python data_manager.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings("ignore")

import config as cfg

# ── Universe (must match signals.py) ───────────────────────────────────────
UNIVERSE_TICKERS = [
    "TCS","INFY","HCLTECH","WIPRO","TECHM","LTIM","MPHASIS","PERSISTENT",
    "COFORGE","KPITTECH","TATAELXSI","OFSS",
    "HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","INDUSINDBK",
    "BANDHANBNK","FEDERALBNK","IDFCFIRSTB","PNB","BANKBARODA","CANBK",
    "UNIONBANK","AUBANK","RBLBANK",
    "BAJFINANCE","BAJAJFINSV","CHOLAFIN","MUTHOOTFIN","MANAPPURAM",
    "LICHSGFIN","SUNDARMFIN","SHRIRAMFIN",
    "SBILIFE","HDFCLIFE","ICICIGI","LICI","STARHEALTH",
    "SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","LUPIN","TORNTPHARM",
    "AUROPHARMA","ALKEM","ZYDUSLIFE","IPCALAB","ABBOTINDIA",
    "HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR","MARICO",
    "COLPAL","GODREJCP","TATACONSUM","VBL","UBL",
    "MARUTI","TATAMOTORS","EICHERMOT","BAJAJ-AUTO","HEROMOTOCO","ASHOKLEY",
    "MRF","BALKRISIND","MOTHERSON","TVSMOTOR","M&M","BHARATFORG","TIINDIA",
    "LT","SIEMENS","ABB","HAVELLS","BEL","HAL","BHEL","CGPOWER","THERMAX",
    "CUMMINSIND","VOLTAS","SUZLON","KEC",
    "RELIANCE","ONGC","BPCL","GAIL","TATAPOWER","ADANIGREEN","NTPC",
    "POWERGRID","NHPC","TORNTPOWER","SJVN",
    "PIDILITIND","DEEPAKNTR","SRF","AARTIIND","NAVINFLUOR","ATUL","ALKYLAMINE",
    "ASIANPAINT","BERGEPAINT","KANSAINER",
    "TITAN","TRENT","DMART","PAGEIND","JUBLFOOD","IRCTC","NAUKRI",
    "ZOMATO","INDIAMART",
    "ULTRACEMCO","SHREECEM","AMBUJACEM","GRASIM","JKCEMENT",
    "JSWSTEEL","TATASTEEL","HINDALCO","VEDL","SAIL","COALINDIA","APLAPOLLO",
    "BHARTIARTL","TATACOMM",
    "DLF","GODREJPROP","OBEROIRLTY","PRESTIGE","LODHA","PHOENIXLTD","BRIGADE",
    "APOLLOHOSP","MAXHEALTH","FORTIS",
    "ADANIPORTS","ADANIENT","IRB",
    "CDSL","BSE","MCX","ANGELONE","CAMS","360ONE","HDFCAMC",
]

INDEX_TICKERS = {
    "nifty500": cfg.REGIME_TICKER,
    "nifty50" : cfg.BENCHMARK_TICKER,
}


# ─────────────────────────────────────────────
# SAFE COLUMN READER
# Handles yfinance 2.x None column issue
# ─────────────────────────────────────────────
def safe_get_column(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Safely extracts a column from a yfinance MultiIndex DataFrame.
    Returns empty Series if column is missing or None.

    This fixes the TypeError: 'NoneType' object is not subscriptable
    which happens with yfinance 2.x when a ticker download fails.
    """
    try:
        series = df[col]
        if series is None:
            return pd.Series(dtype=float)
        return series.dropna()
    except (KeyError, TypeError):
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────
# FETCH FRESH DATA FOR A DATE RANGE
# ─────────────────────────────────────────────
def fetch_fresh(tickers: list, start: str, end: str) -> tuple:
    """
    Downloads OHLCV data for given tickers and date range.
    Returns (close_df, volume_df) with clean ticker column names.
    Handles yfinance 2.x column format safely.
    """
    yf_tickers = [t + ".NS" for t in tickers]
    print(f"  Downloading {len(yf_tickers)} stocks ({start} to {end})...")

    try:
        raw = yf.download(
            yf_tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker",   # group by ticker for cleaner structure
        )
    except Exception as e:
        print(f"  Download error: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # ── Extract close and volume safely ─────────────────────────────────
    close_data  = {}
    volume_data = {}

    for t in tickers:
        yf_t = t + ".NS"
        try:
            # yfinance 2.x with group_by="ticker" uses (ticker, field) MultiIndex
            if isinstance(raw.columns, pd.MultiIndex):
                if yf_t in raw.columns.get_level_values(0):
                    c = raw[yf_t]["Close"]
                    v = raw[yf_t]["Volume"]
                elif yf_t in raw.columns.get_level_values(1):
                    c = raw["Close"][yf_t]
                    v = raw["Volume"][yf_t]
                else:
                    continue
            else:
                # Single ticker fallback
                c = raw["Close"]
                v = raw["Volume"]

            if c is not None and len(c.dropna()) > 0:
                close_data[t]  = c
                volume_data[t] = v

        except Exception:
            continue

    close_df  = pd.DataFrame(close_data)
    volume_df = pd.DataFrame(volume_data)

    print(f"  Got data for {len(close_df.columns)}/{len(tickers)} stocks")
    return close_df, volume_df


# ─────────────────────────────────────────────
# UPDATE CACHE
# ─────────────────────────────────────────────
def update_cache():
    """
    Checks if cache exists and is up to date.
    If stale, fetches only the missing days and appends.
    If cache does not exist, does a full download.

    This is the function to run daily.
    """
    today     = datetime.today().date()
    fetch_end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"\n{'='*50}")
    print(f"  DATA MANAGER — {today}")
    print(f"{'='*50}")

    # ── Case 1: No cache exists — full download ──────────────────────────
    if not os.path.exists(cfg.DATA_CACHE_FILE):
        print("No cache found. Running full download (this takes ~4 minutes)...")
        fetch_start = cfg.DATA_FETCH_START
        close, volume = fetch_fresh(UNIVERSE_TICKERS, fetch_start, fetch_end)

        if not close.empty:
            close.to_csv(cfg.DATA_CACHE_FILE)
            volume.to_csv(cfg.VOLUME_CACHE)
            print(f"Cache created: {close.shape[0]} days × {close.shape[1]} stocks")

        _update_index_cache(fetch_start, fetch_end)
        return

    # ── Case 2: Cache exists — check if up to date ───────────────────────
    existing = pd.read_csv(cfg.DATA_CACHE_FILE, index_col=0, parse_dates=True)
    last_date = existing.index[-1].date()
    print(f"Cache last updated : {last_date}")
    print(f"Today              : {today}")

    # If market was open today (weekday) and cache doesn't have today
    # fetch from last cached date onwards
    if last_date >= today:
        print("Cache is up to date. No update needed.")
        return

    # ── Case 3: Cache is stale — fetch only missing days ─────────────────
    fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Fetching new data from {fetch_start}...")

    new_close, new_volume = fetch_fresh(UNIVERSE_TICKERS, fetch_start, fetch_end)

    if new_close.empty:
        print("No new data returned. Market may have been closed.")
        return

    # Load existing volume cache
    existing_vol = pd.read_csv(cfg.VOLUME_CACHE, index_col=0, parse_dates=True)

    # Align columns — only update columns that exist in both
    common_cols  = existing.columns.intersection(new_close.columns)
    new_cols     = new_close.columns.difference(existing.columns)

    # Append new rows
    updated_close  = pd.concat([existing, new_close[common_cols]])
    updated_vol    = pd.concat([existing_vol, new_volume[common_cols]])

    # Remove duplicate dates (keep latest)
    updated_close  = updated_close[~updated_close.index.duplicated(keep="last")]
    updated_vol    = updated_vol[~updated_vol.index.duplicated(keep="last")]

    # Add any new tickers as new columns
    for col in new_cols:
        updated_close[col] = new_close[col]
        updated_vol[col]   = new_volume.get(col, pd.Series())

    updated_close.to_csv(cfg.DATA_CACHE_FILE)
    updated_vol.to_csv(cfg.VOLUME_CACHE)

    print(f"Cache updated: added {len(new_close)} new days")
    print(f"Cache now has: {updated_close.shape[0]} days × {updated_close.shape[1]} stocks")
    print(f"Date range: {updated_close.index[0].date()} → {updated_close.index[-1].date()}")

    # Update index data too
    _update_index_cache(fetch_start, fetch_end)


def _update_index_cache(start: str, end: str):
    """Updates Nifty 500 and Nifty 50 index cache."""
    print("\nUpdating index data (Nifty 500 + Nifty 50)...")
    try:
        r500 = yf.download(cfg.REGIME_TICKER,    start=start, end=end,
                           auto_adjust=True, progress=False)
        r50  = yf.download(cfg.BENCHMARK_TICKER, start=start, end=end,
                           auto_adjust=True, progress=False)

        if os.path.exists(cfg.REGIME_CACHE):
            existing_idx = pd.read_csv(cfg.REGIME_CACHE, index_col=0, parse_dates=True)
            new_idx = pd.DataFrame({
                "nifty500": r500["Close"].squeeze(),
                "nifty50" : r50["Close"].squeeze(),
            })
            updated_idx = pd.concat([existing_idx, new_idx])
            updated_idx = updated_idx[~updated_idx.index.duplicated(keep="last")]
        else:
            updated_idx = pd.DataFrame({
                "nifty500": r500["Close"].squeeze(),
                "nifty50" : r50["Close"].squeeze(),
            })

        updated_idx.to_csv(cfg.REGIME_CACHE)
        print(f"Index cache updated → last date: {updated_idx.index[-1].date()}")

    except Exception as e:
        print(f"Index update failed: {e}")


# ─────────────────────────────────────────────
# LOAD FOR SIGNALS (used by signals.py)
# ─────────────────────────────────────────────
def load_for_signals() -> tuple:
    """
    Loads close prices and volume from cache.
    Returns (close_df, volume_df) ready for signal computation.
    Called by signals.py instead of fetching fresh data.
    """
    if not os.path.exists(cfg.DATA_CACHE_FILE):
        raise FileNotFoundError(
            "Price cache not found. Run: python data_manager.py"
        )

    close  = pd.read_csv(cfg.DATA_CACHE_FILE, index_col=0, parse_dates=True)
    volume = pd.read_csv(cfg.VOLUME_CACHE,    index_col=0, parse_dates=True)

    print(f"Loaded cache: {close.shape[1]} stocks | "
          f"last date: {close.index[-1].date()}")
    return close, volume


def load_index_data() -> tuple:
    """Loads Nifty 500 and Nifty 50 from cache."""
    if not os.path.exists(cfg.REGIME_CACHE):
        raise FileNotFoundError(
            "Index cache not found. Run: python data_manager.py"
        )
    df = pd.read_csv(cfg.REGIME_CACHE, index_col=0, parse_dates=True)
    return df["nifty500"], df["nifty50"]


# ─────────────────────────────────────────────
# CACHE STATUS REPORT
# ─────────────────────────────────────────────
def print_cache_status():
    """Prints a summary of what data is currently cached."""
    print(f"\n{'─'*40}")
    print("  CACHE STATUS")
    print(f"{'─'*40}")

    for label, filepath in [
        ("Price cache",  cfg.DATA_CACHE_FILE),
        ("Volume cache", cfg.VOLUME_CACHE),
        ("Index cache",  cfg.REGIME_CACHE),
    ]:
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1_000_000
            df      = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"  {label:<15}: {df.shape[0]} rows × {df.shape[1]} cols | "
                  f"{size_mb:.1f} MB | last: {df.index[-1].date()}")
        else:
            print(f"  {label:<15}: NOT FOUND")
    print(f"{'─'*40}\n")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    update_cache()
    print_cache_status()
    print("Done. You can now run: python execution.py")
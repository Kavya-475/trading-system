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

# ── Universe ticker list ─────────────────────────────────────────────────────
# Must stay in sync with the UNIVERSE dict in signals.py.
# yfinance requires the ".NS" suffix for NSE stocks (added in fetch_fresh below).
UNIVERSE_TICKERS = [
    "360ONE",
    "3MINDIA",
    "ABB",
    "ABBOTINDIA",
    "ABCAPITAL",
    "ACC",
    "ADANIENSOL",
    "ADANIENT",
    "ADANIGREEN",
    "ADANIPORTS",
    "ADANIPOWER",
    "AIAENG",
    "AIIL",
    "AJANTPHARM",
    "ALKEM",
    "AMBUJACEM",
    "ANTHEM",
    "APARINDS",
    "APLAPOLLO",
    "APOLLOHOSP",
    "APOLLOTYRE",
    "ASHOKLEY",
    "ASIANPAINT",
    "ASTRAL",
    "ATGL",
    "AUBANK",
    "AUROPHARMA",
    "AWL",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJAJFINSV",
    "BAJAJHFL",
    "BAJAJHLDNG",
    "BAJFINANCE",
    "BALKRISIND",
    "BANKBARODA",
    "BANKINDIA",
    "BDL",
    "BEL",
    "BERGEPAINT",
    "BHARATFORG",
    "BHARTIARTL",
    "BHARTIHEXA",
    "BHEL",
    "BIOCON",
    "BLUESTARCO",
    "BOSCHLTD",
    "BPCL",
    "BRITANNIA",
    "BSE",
    "CANBK",
    "CGPOWER",
    "CHOLAFIN",
    "CIPLA",
    "COALINDIA",
    "COCHINSHIP",
    "COFORGE",
    "COLPAL",
    "CONCOR",
    "COROMANDEL",
    "CRISIL",
    "CUMMINSIND",
    "DABUR",
    "DALBHARAT",
    "DIVISLAB",
    "DIXON",
    "DLF",
    "DMART",
    "DRREDDY",
    "EICHERMOT",
    "ENDURANCE",
    "ENRIN",
    "ESCORTS",
    "ETERNAL",
    "EXIDEIND",
    "FEDERALBNK",
    "FLUOROCHEM",
    "FORTIS",
    "GAIL",
    "GICRE",
    "GLAXO",
    "GLENMARK",
    "GMRAIRPORT",
    "GODFRYPHLP",
    "GODREJCP",
    "GODREJIND",
    "GODREJPROP",
    "GRASIM",
    "GROWW",
    "GVT&D",
    "HAL",
    "HAVELLS",
    "HCLTECH",
    "HDBFS",
    "HDFCAMC",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HEXT",
    "HINDALCO",
    "HINDPETRO",
    "HINDUNILVR",
    "HINDZINC",
    "HONAUT",
    "HUDCO",
    "HYUNDAI",
    "ICICIAMC",
    "ICICIBANK",
    "ICICIGI",
    "ICICIPRULI",
    "IDEA",
    "IDFCFIRSTB",
    "INDHOTEL",
    "INDIANB",
    "INDIGO",
    "INDUSINDBK",
    "INDUSTOWER",
    "INFY",
    "IOC",
    "IPCALAB",
    "IRCTC",
    "IREDA",
    "IRFC",
    "ITC",
    "ITCHOTELS",
    "JINDALSTEL",
    "JIOFIN",
    "JKCEMENT",
    "JSL",
    "JSWENERGY",
    "JSWINFRA",
    "JSWSTEEL",
    "JUBLFOOD",
    "KALYANKJIL",
    "KEI",
    "KOTAKBANK",
    "KPITTECH",
    "KPRMILL",
    "LAURUSLABS",
    "LENSKART",
    "LGEINDIA",
    "LICHSGFIN",
    "LICI",
    "LINDEINDIA",
    "LLOYDSME",
    "LODHA",
    "LT",
    "LTF",
    "LTM",
    "LTTS",
    "LUPIN",
    "M&M",
    "M&MFIN",
    "MAHABANK",
    "MANKIND",
    "MARICO",
    "MARUTI",
    "MAXHEALTH",
    "MAZDOCK",
    "MCX",
    "MEDANTA",
    "MFSL",
    "MOTHERSON",
    "MOTILALOFS",
    "MPHASIS",
    "MRF",
    "MUTHOOTFIN",
    "NAM-INDIA",
    "NATIONALUM",
    "NAUKRI",
    "NESTLEIND",
    "NHPC",
    "NIACL",
    "NLCINDIA",
    "NMDC",
    "NTPC",
    "NTPCGREEN",
    "NYKAA",
    "OBEROIRLTY",
    "OFSS",
    "OIL",
    "ONGC",
    "PAGEIND",
    "PATANJALI",
    "PAYTM",
    "PERSISTENT",
    "PETRONET",
    "PFC",
    "PHOENIXLTD",
    "PIDILITIND",
    "PIIND",
    "PNB",
    "POLICYBZR",
    "POLYCAB",
    "POWERGRID",
    "POWERINDIA",
    "PREMIERENE",
    "PRESTIGE",
    "RADICO",
    "RECLTD",
    "RELIANCE",
    "RVNL",
    "SAIL",
    "SBICARD",
    "SBILIFE",
    "SBIN",
    "SCHAEFFLER",
    "SHREECEM",
    "SHRIRAMFIN",
    "SIEMENS",
    "SJVN",
    "SOLARINDS",
    "SRF",
    "SUNDARMFIN",
    "SUNPHARMA",
    "SUPREMEIND",
    "SUZLON",
    "SWIGGY",
    "TATACAP",
    "TATACOMM",
    "TATACONSUM",
    "TATAELXSI",
    "TATAINVEST",
    "TATAPOWER",
    "TATASTEEL",
    "TCS",
    "TECHM",
    "THERMAX",
    "TIINDIA",
    "TITAN",
    "TMCV",
    "TMPV",
    "TORNTPHARM",
    "TORNTPOWER",
    "TRENT",
    "TVSMOTOR",
    "UBL",
    "ULTRACEMCO",
    "UNIONBANK",
    "UNITDSPR",
    "UNOMINDA",
    "UPL",
    "VBL",
    "VEDL",
    "VMM",
    "VOLTAS",
    "WAAREEENER",
    "WIPRO",
    "YESBANK",
    "ZYDUSLIFE",
]

# ─────────────────────────────────────────────
# SAFE COLUMN READER
# yfinance 2.x changed its DataFrame structure and sometimes returns None
# for a ticker column instead of raising a KeyError. This wrapper handles both.
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
# Downloads OHLCV data from Yahoo Finance for the given tickers and date range.
# yfinance 2.x can return MultiIndex DataFrames in different layouts depending
# on whether one or many tickers are downloaded — this handles both layouts.
# ─────────────────────────────────────────────
def fetch_fresh(tickers: list, start: str, end: str) -> tuple:
    """
    Downloads adjusted OHLCV data from Yahoo Finance.
    Returns (close_df, volume_df, open_df) — each a DataFrame with
    clean NSE ticker names as columns (the .NS suffix is stripped).
    """
    yf_tickers = [t + ".NS" for t in tickers]   # yfinance requires .NS suffix for NSE stocks
    print(f"  Downloading {len(yf_tickers)} stocks ({start} to {end})...")

    try:
        raw = yf.download(
            yf_tickers,
            start=start,
            end=end,
            auto_adjust=True,     # adjusts for splits and dividends automatically
            progress=False,
            group_by="ticker",    # groups columns by ticker for easier access
        )
    except Exception as e:
        print(f"  Download error: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ── Extract close, volume, open per ticker — handle yfinance column layouts ─
    close_data  = {}
    volume_data = {}
    open_data   = {}

    for t in tickers:
        yf_t = t + ".NS"
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                # yfinance 2.x multi-ticker layout — two possible orderings
                if yf_t in raw.columns.get_level_values(0):
                    # Layout: (ticker, field) — ticker is the first level
                    c = raw[yf_t]["Close"]
                    v = raw[yf_t]["Volume"]
                    o = raw[yf_t]["Open"]
                elif yf_t in raw.columns.get_level_values(1):
                    # Layout: (field, ticker) — field is the first level
                    c = raw["Close"][yf_t]
                    v = raw["Volume"][yf_t]
                    o = raw["Open"][yf_t]
                else:
                    continue    # ticker not in download result — skip
            else:
                # Single ticker download — flat column names
                c = raw["Close"]
                v = raw["Volume"]
                o = raw["Open"]

            if c is not None and len(c.dropna()) > 0:
                close_data[t]  = c     # store with clean ticker name (no .NS)
                volume_data[t] = v
                open_data[t]   = o

        except Exception:
            continue    # silently skip any ticker that failed to parse

    close_df  = pd.DataFrame(close_data)
    volume_df = pd.DataFrame(volume_data)
    open_df   = pd.DataFrame(open_data)

    print(f"  Got data for {len(close_df.columns)}/{len(tickers)} stocks")
    return close_df, volume_df, open_df


# ─────────────────────────────────────────────
# UPDATE CACHE
# Three-case logic:
#   Case 1 — No cache files exist → full download from DATA_FETCH_START (~4 min)
#   Case 2 — Cache is current (last date ≥ today) → nothing to do
#   Case 3 — Cache is stale → incremental update (only missing days, ~30 sec)
# ─────────────────────────────────────────────
def update_cache():
    """
    Main function to call daily (via scheduler.py or manually at 3:40 PM IST).
    Keeps the CSV caches up to date without re-downloading historical data.
    """
    today     = datetime.today().date()
    fetch_end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")   # +1 so today is included

    print(f"\n{'='*50}")
    print(f"  DATA MANAGER — {today}")
    print(f"{'='*50}")

    # ── Case 1: No cache — full download ─────────────────────────────────
    if not os.path.exists(cfg.DATA_CACHE_FILE):
        print("No cache found. Running full download (this takes ~4 minutes)...")
        fetch_start = cfg.DATA_FETCH_START
        close, volume, open_prices = fetch_fresh(UNIVERSE_TICKERS, fetch_start, fetch_end)

        if not close.empty:
            close.to_csv(cfg.DATA_CACHE_FILE)
            volume.to_csv(cfg.VOLUME_CACHE)
            open_prices.to_csv(cfg.OPEN_CACHE)
            print(f"Cache created: {close.shape[0]} days × {close.shape[1]} stocks")

        _update_index_cache(fetch_start, fetch_end)
        return

    # ── Case 2: Cache exists — check last date ────────────────────────────
    existing  = pd.read_csv(cfg.DATA_CACHE_FILE, index_col=0, parse_dates=True)
    last_date = existing.index[-1].date()
    print(f"Cache last updated : {last_date}")
    print(f"Today              : {today}")

    if last_date >= today:
        print("Cache is up to date. No update needed.")
        return

    # ── Case 3: Stale cache — fetch only missing days ─────────────────────
    # Start from the day after the last cached date
    fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Fetching new data from {fetch_start}...")

    new_close, new_volume, new_open = fetch_fresh(UNIVERSE_TICKERS, fetch_start, fetch_end)

    if new_close.empty:
        # Happens on market holidays or if run before market close
        print("No new data returned. Market may have been closed.")
        return

    existing_vol  = pd.read_csv(cfg.VOLUME_CACHE, index_col=0, parse_dates=True)
    existing_open = pd.read_csv(cfg.OPEN_CACHE,   index_col=0, parse_dates=True) if os.path.exists(cfg.OPEN_CACHE) else pd.DataFrame()

    # Only update columns that already exist in the cache (common between old and new)
    common_cols = existing.columns.intersection(new_close.columns)
    new_cols    = new_close.columns.difference(existing.columns)   # newly listed stocks

    # Concatenate old + new rows for each cache
    updated_close = pd.concat([existing,      new_close[common_cols]])
    updated_vol   = pd.concat([existing_vol,  new_volume[common_cols]])

    # Remove any duplicate dates that might arise from running twice on the same day
    updated_close = updated_close[~updated_close.index.duplicated(keep="last")]
    updated_vol   = updated_vol[~updated_vol.index.duplicated(keep="last")]

    # Append data for any newly listed stocks not in the existing cache
    for col in new_cols:
        updated_close[col] = new_close[col]
        updated_vol[col]   = new_volume.get(col, pd.Series())

    updated_close.to_csv(cfg.DATA_CACHE_FILE)
    updated_vol.to_csv(cfg.VOLUME_CACHE)

    if not existing_open.empty:
        open_common  = existing_open.columns.intersection(new_open.columns)
        updated_open = pd.concat([existing_open, new_open[open_common]])
        updated_open = updated_open[~updated_open.index.duplicated(keep="last")]
        for col in new_cols:
            updated_open[col] = new_open.get(col, pd.Series())
        updated_open.to_csv(cfg.OPEN_CACHE)

    print(f"Cache updated: added {len(new_close)} new days")
    print(f"Cache now has: {updated_close.shape[0]} days × {updated_close.shape[1]} stocks")
    print(f"Date range: {updated_close.index[0].date()} → {updated_close.index[-1].date()}")

    # Update the index cache (Nifty 500 + Nifty 50) for the same date range
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
            }, index=r500.index)
            updated_idx = pd.concat([existing_idx, new_idx])
            updated_idx = updated_idx[~updated_idx.index.duplicated(keep="last")]
        else:
            updated_idx = pd.DataFrame({
                "nifty500": r500["Close"].squeeze(),
                "nifty50" : r50["Close"].squeeze(),
            }, index=r500.index)

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
    """Loads all regime indices from cache."""
    if not os.path.exists(cfg.REGIME_CACHE):
        raise FileNotFoundError(
            "Index cache not found. Run: python data_manager.py"
        )
    df = pd.read_csv(cfg.REGIME_CACHE, index_col=0, parse_dates=True)
    nifty500  = df["nifty500"]
    nifty50   = df["nifty50"]
    nifty100  = df["nifty100"]    if "nifty100"    in df.columns else nifty500
    nifty_mid = df["nifty_midcap"]if "nifty_midcap"in df.columns else nifty500
    return nifty500, nifty50, nifty100, nifty_mid


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
        ("Open cache",   cfg.OPEN_CACHE),
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
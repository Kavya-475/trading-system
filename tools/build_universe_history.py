"""
build_universe_history.py
=========================
Converts NSE LargeMidCap 250 constituent snapshot files into
universe_history.csv — the point-in-time universe used by backtester.py
to eliminate survivorship bias.

HOW TO GET THE RAW DATA (one-time manual step)
-----------------------------------------------
1. Go to: nseindia.com → Market Data → Live Market → Indices
2. Search "NIFTY LARGEMIDCAP 250" → click the index
3. Click "Download" (top right) to get the CURRENT constituent list as CSV
4. For HISTORICAL snapshots:
   a. Go to: nseindia.com → Products → Indices → Equity Indices
   b. Click "NIFTY LARGEMIDCAP 250" → "Historical Data"
   c. Alternatively: niftyindices.com → Reports → Index Constituents
      → Select index → download one CSV per rebalance date (semi-annual: March & September)
5. Save each downloaded file into a folder called  nse_snapshots/
   Rename each file to its effective date:  YYYY-MM-DD.csv
   e.g.  nse_snapshots/2024-09-27.csv
         nse_snapshots/2024-03-28.csv
         nse_snapshots/2023-09-29.csv  ... and so on back to 2011

NSE rebalancing is semi-annual. You need roughly 26 files (2011–2024).
The more files you download, the more accurate the point-in-time universe.

EXPECTED FILE FORMAT (NSE's standard CSV download)
----------------------------------------------------
Column names may vary slightly — the script handles common variants.
Required columns (detected automatically):
  Symbol / Ticker / SYMBOL
  Industry / Sector / INDUSTRY / GICS Sector

RUN
---
    python build_universe_history.py

OUTPUT
------
  universe_history.csv   — used by backtester.py automatically
  new_tickers.txt        — tickers not in current price cache; run
                           data_manager.py after adding these to UNIVERSE_TICKERS
"""

import os
import sys
import pandas as pd
from pathlib import Path

SNAPSHOTS_DIR   = "nse_snapshots"
OUTPUT_FILE     = "universe_history.csv"
CURRENT_CACHE   = "price_data_cache.csv"

# ── Column name aliases ────────────────────────────────────────────────────────
SYMBOL_ALIASES  = ["Symbol", "SYMBOL", "Ticker", "TICKER", "NSE Symbol", "NSE CODE"]
SECTOR_ALIASES  = ["Industry", "INDUSTRY", "Sector", "SECTOR", "Sub-Sector",
                   "GICS Sector", "Industry/ Sector", "Industry/Sector"]


def detect_column(df, aliases):
    for alias in aliases:
        if alias in df.columns:
            return alias
    # fuzzy match
    for col in df.columns:
        for alias in aliases:
            if alias.lower() in col.lower():
                return col
    return None


def parse_snapshot(filepath):
    path = Path(filepath)
    try:
        if path.suffix in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        else:
            # try common encodings
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    df = pd.read_csv(filepath, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
    except Exception as e:
        print(f"  ERROR reading {path.name}: {e}")
        return None

    sym_col = detect_column(df, SYMBOL_ALIASES)
    sec_col = detect_column(df, SECTOR_ALIASES)

    if sym_col is None:
        print(f"  SKIP {path.name} — cannot find symbol column. Columns: {list(df.columns)}")
        return None

    result = pd.DataFrame()
    result["ticker"] = df[sym_col].astype(str).str.strip().str.upper()
    result["sector"] = df[sec_col].astype(str).str.strip() if sec_col else "Unknown"

    # Remove empty / header rows
    result = result[result["ticker"].str.len() > 0]
    result = result[~result["ticker"].str.contains("SYMBOL|TICKER|N/A", case=False, na=False)]
    result = result.drop_duplicates(subset="ticker")

    return result


def build():
    snapshots_path = Path(SNAPSHOTS_DIR)

    if not snapshots_path.exists():
        print(f"ERROR: Folder '{SNAPSHOTS_DIR}/' not found.")
        print("Create it and add your NSE constituent snapshot files.")
        print("See the docstring at the top of this file for instructions.")
        sys.exit(1)

    files = sorted([
        f for f in snapshots_path.iterdir()
        if f.suffix in (".csv", ".xlsx", ".xls")
    ])

    if not files:
        print(f"No CSV/Excel files found in '{SNAPSHOTS_DIR}/'.")
        sys.exit(1)

    print(f"Found {len(files)} snapshot files in '{SNAPSHOTS_DIR}/':\n")

    rows = []
    all_tickers = set()

    for f in files:
        # Try to parse effective date from filename
        stem = f.stem  # e.g. "2024-09-27"
        try:
            effective_date = pd.to_datetime(stem).date()
        except Exception:
            print(f"  SKIP {f.name} — filename must be YYYY-MM-DD (got '{stem}')")
            continue

        df = parse_snapshot(f)
        if df is None or df.empty:
            continue

        print(f"  {effective_date}  →  {len(df)} stocks")

        for _, row in df.iterrows():
            rows.append({
                "effective_date": effective_date,
                "ticker":         row["ticker"],
                "sector":         row["sector"],
            })
            all_tickers.add(row["ticker"])

    if not rows:
        print("\nNo data parsed. Check your file format.")
        sys.exit(1)

    history = pd.DataFrame(rows)
    history["effective_date"] = pd.to_datetime(history["effective_date"])
    history = history.sort_values(["effective_date", "ticker"])
    history.to_csv(OUTPUT_FILE, index=False)
    print(f"\nWrote {len(history)} rows → {OUTPUT_FILE}")
    print(f"Unique tickers across all snapshots: {len(all_tickers)}")

    # ── Find tickers not in current price cache ────────────────────────────────
    if os.path.exists(CURRENT_CACHE):
        cached_cols = pd.read_csv(CURRENT_CACHE, index_col=0, nrows=0).columns.tolist()
        new_tickers = sorted(all_tickers - set(cached_cols))
        if new_tickers:
            with open("new_tickers.txt", "w") as fh:
                fh.write("\n".join(new_tickers))
            print(f"\n{len(new_tickers)} tickers not in price cache → new_tickers.txt")
            print("Add these to UNIVERSE_TICKERS in data_manager.py and re-run:")
            print("    python data_manager.py")
        else:
            print("\nAll tickers already in price cache. No data download needed.")
    else:
        print("\nNo price cache found — run data_manager.py to download price data.")

    print(f"\nDone. backtester.py will use {OUTPUT_FILE} automatically.")


if __name__ == "__main__":
    build()

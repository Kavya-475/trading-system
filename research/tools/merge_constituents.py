"""
tools/merge_constituents.py
===========================
Merge dated NIFTY LargeMidcap 250 constituent files into universe_history.csv to
close the point-in-time gap (e.g. the 2020→2026 stretch that NSE only serves via
browser-gated, anti-bot downloads and isn't auto-fetchable).

WHY MANUAL DOWNLOAD: niftyindices.com / archives.nseindia.com return their HTML
SPA (not the file) to non-browser requests, and the Wayback Machine archived only
one constituent snapshot (2023-08) in that window. So the files must be saved by
hand from a browser — but that's the only manual step; this script does the rest.

HOW TO GET THE FILES (≈10 min):
  1. Browser → niftyindices.com → search "NIFTY LargeMidcap 250" → the index page.
  2. The "Download Constituents" button gives ind_niftylargemidcap250list.csv
     (columns: Company Name, Industry, Symbol, Series, ISIN Code) — the CURRENT
     list. For HISTORICAL semi-annual lists (NSE rebalances end-Mar & end-Sep):
       • niftyindices.com → Resources → Index Reconstitution / press releases, OR
       • download the current list each rebalance going forward.
     Realistically, grab whatever semi-annual lists you can for 2020-09 → 2025-09.
  3. Save each as  nse_snapshots/YYYY-MM-DD.csv  using its effective (rebalance)
     date, e.g.  nse_snapshots/2022-09-30.csv

THEN:
    python tools/merge_constituents.py        # merges into universe_history.csv
    python backtester.py                       # re-run (no cache rebuild needed —
                                               #  prices for these names already exist)

Only the universe MEMBERSHIP changes; the bias-free price cache already contains
all recent symbols, so you do NOT need to rebuild nse_data caches.
"""
import os
import sys
import glob
import pandas as pd

HERE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
SNAPS  = os.path.join(HERE, "nse_snapshots")
HIST   = os.path.join(HERE, "universe_history.csv")
PRICE  = os.path.join(HERE, "nse_data", "price_cache.csv")

SYM_ALIASES = ["Symbol", "SYMBOL", "Ticker", "TICKER"]
SEC_ALIASES = ["Industry", "INDUSTRY", "Sector", "SECTOR", "Macro-Economic Sector"]


def _col(df, aliases):
    for a in aliases:
        if a in df.columns:
            return a
    for c in df.columns:
        if any(a.lower() in c.lower() for a in aliases):
            return c
    return None


def parse_file(path):
    date = pd.to_datetime(os.path.basename(path)[:10], errors="coerce")
    if pd.isna(date):
        print(f"  SKIP {os.path.basename(path)} — name must start YYYY-MM-DD")
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    sc, kc = _col(df, SYM_ALIASES), _col(df, SEC_ALIASES)
    if sc is None:
        print(f"  SKIP {os.path.basename(path)} — no Symbol column ({list(df.columns)})")
        return None
    out = pd.DataFrame({
        "effective_date": date,
        "ticker": df[sc].astype(str).str.strip().str.upper(),
        "sector": (df[kc].astype(str).str.strip() if kc else "Unknown"),
    })
    out = out[out.ticker.str.len() > 0]
    out = out[~out.ticker.str.contains("SYMBOL|DUMMY", case=False, na=False)]
    return out.drop_duplicates("ticker")


def main():
    files = sorted(glob.glob(os.path.join(SNAPS, "*.csv")))
    if not files:
        print(f"No constituent files in {SNAPS}/ — see this script's docstring.")
        sys.exit(1)
    if not os.path.exists(HIST):
        print(f"Missing {HIST}"); sys.exit(1)

    hist = pd.read_csv(HIST, parse_dates=["effective_date"])
    pd.read_csv(HIST).to_csv(HIST + ".bak", index=False)   # backup
    before_dates = set(hist.effective_date.dt.date)

    added = []
    for f in files:
        d = parse_file(f)
        if d is not None and len(d):
            added.append(d)
            print(f"  + {d.effective_date.iloc[0].date()}  ({len(d)} constituents)  {os.path.basename(f)}")

    if not added:
        print("Nothing parsed."); sys.exit(1)

    merged = (pd.concat([hist] + added, ignore_index=True)
                .drop_duplicates(["effective_date", "ticker"])
                .sort_values(["effective_date", "ticker"]))
    merged.to_csv(HIST, index=False)

    snaps = sorted(merged.effective_date.dt.date.unique())
    new   = [s for s in snaps if s not in before_dates]
    print(f"\nuniverse_history.csv: {len(hist)} → {len(merged)} rows, "
          f"{len(snaps)} snapshots ({len(new)} new: {new})")

    # Coverage check against the price cache
    if os.path.exists(PRICE):
        priced = set(pd.read_csv(PRICE, index_col=0, nrows=0).columns)
        miss = sorted(set(merged.ticker) - priced)
        if miss:
            print(f"⚠ {len(miss)} constituent tickers have NO price in nse_data cache "
                  f"(add an alias in build_caches.py or accept the gap): {miss[:15]}")

    # Remaining internal gaps > ~9 months
    gaps = [(a, b, (pd.Timestamp(b) - pd.Timestamp(a)).days)
            for a, b in zip(snaps, snaps[1:])
            if (pd.Timestamp(b) - pd.Timestamp(a)).days > 270]
    if gaps:
        print("Remaining gaps > 270d (add files for these to fully close):")
        for a, b, dd in gaps:
            print(f"   {a} → {b}  ({dd}d)")
    else:
        print("No remaining gaps > 270d — point-in-time universe is contiguous. ✅")


if __name__ == "__main__":
    main()

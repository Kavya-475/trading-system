"""
nse_data/fetch_splits.py
========================
Fetch AUTHORITATIVE split/bonus ratios for the universe from yfinance, so the
bhavcopy price cache can be corporate-action-adjusted correctly.

Why not derive splits from bhavcopy?  NSE's PREVCLOSE is the raw previous close
(never adjusted), and a 1:1 bonus is indistinguishable from a −50% crash using
close prices alone. So heuristic detection mislabels ordinary −20% circuit days
as "4:5 bonuses". The split CALENDAR (which dates, what ratio) has to come from a
clean source. yfinance has it, and there is no survivorship issue here: we only
need splits for names the strategy can hold — i.e. index members, ~all survivors.

yfinance convention: Ticker(...).splits gives shares-multiplier on the ex-date
(e.g. 10.0 for a 1:10 split, 2.0 for a 1:1 bonus). The PRICE multiplier we apply
to back-adjust is 1 / that.

Output: nse_data/splits.csv  [symbol, date, price_mult]

Usage:
    python nse_data/fetch_splits.py            # universe_history symbols
    python nse_data/fetch_splits.py --all      # every symbol in price_cache.csv
"""
import argparse
import os
import time

import pandas as pd
import yfinance as yf

HERE      = os.path.dirname(__file__)
OUT       = os.path.join(HERE, "splits.csv")
PRICE     = os.path.join(HERE, "price_cache.csv")
UNIVERSE  = os.path.join(os.path.dirname(HERE), "universe_history.csv")

# yfinance NSE tickers keep the special characters (M&M.NS, BAJAJ-AUTO.NS).
def yf_ticker(sym):
    return sym + ".NS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="fetch for every symbol in price_cache.csv (slow)")
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    if args.all and os.path.exists(PRICE):
        symbols = sorted(pd.read_csv(PRICE, index_col=0, nrows=0).columns)
    elif os.path.exists(UNIVERSE):
        symbols = sorted(set(pd.read_csv(UNIVERSE)["ticker"].astype(str)))
    else:
        raise SystemExit("Need universe_history.csv or price_cache.csv.")

    print(f"Fetching split/bonus history for {len(symbols)} symbols from yfinance…")
    rows, have, fail = [], 0, []
    for i, sym in enumerate(symbols, 1):
        try:
            s = yf.Ticker(yf_ticker(sym)).splits
        except Exception:
            s = None
        if s is not None and len(s):
            have += 1
            for dt, ratio in s.items():
                if ratio and ratio > 0:
                    rows.append({"symbol": sym,
                                 "date": pd.Timestamp(dt).tz_localize(None).date(),
                                 "price_mult": round(1.0 / float(ratio), 8)})
        if i % 50 == 0 or i == len(symbols):
            print(f"  [{i}/{len(symbols)}] symbols with splits: {have}, events: {len(rows)}")
        time.sleep(args.sleep)

    df = pd.DataFrame(rows).sort_values(["date", "symbol"])
    df.to_csv(OUT, index=False)
    print(f"\nWrote {len(df)} split/bonus events for {have} symbols → {OUT}")
    if len(df):
        print("Sample:")
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

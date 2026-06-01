"""
nse_data/download.py
====================
Resilient downloader for NSE end-of-day archives — the raw material for a
SURVIVORSHIP-BIAS-FREE price database.

Why bhavcopy?  NSE's daily bhavcopy is the official EOD snapshot of every
security that traded that day. A stock that later delisted/merged/went bankrupt
still appears in every bhavcopy up to its last trading day — so downloading the
full archive gives you the dead names too, which yfinance (today's survivors
only) never will.

What it fetches (per trading day in the range):
  - EQUITY bhavcopy  → per-symbol OHLC, volume, PREVCLOSE, ISIN
      old format (< 2024-07-08): .../EQUITIES/<YYYY>/<MON>/cm<DD><MON><YYYY>bhav.csv.zip
      UDiFF      (>= 2024-07-08): .../cm/BhavCopy_NSE_CM_0_0_0_<YYYYMMDD>_F_0000.csv.zip
  - INDEX  bhavcopy  → Nifty 50/100/500/Midcap/LargeMidcap250 close
      .../indices/ind_close_all_<DDMMYYYY>.csv

Design:
  - Resumable: skips files already on disk, and writes a .holiday marker on a
    404 so weekends/holidays aren't re-requested.
  - Polite: throttled, retries with backoff, browser headers.
  - Runs for a long time (~5000 files for 20 years). Safe to Ctrl-C and re-run;
    it picks up where it stopped. Consider running year-by-year in the background.

Usage:
    python nse_data/download.py --start 2007-01-01 --end 2026-05-31
    python nse_data/download.py --start 2020-01-01 --end 2020-12-31 --no-index
    python nse_data/download.py --start 2024-01-01 --end 2024-12-31 --equity-only
"""
import argparse
import os
import time
from datetime import date, timedelta

import requests

RAW_DIR   = os.path.join(os.path.dirname(__file__), "raw")
EQ_DIR    = os.path.join(RAW_DIR, "equity")
IDX_DIR   = os.path.join(RAW_DIR, "index")

UDIFF_CUTOVER = date(2024, 7, 8)   # NSE switched equity bhavcopy to UDiFF here
MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
       "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.nseindia.com/",
    "Accept-Language": "en-US,en;q=0.9",
}


def equity_url(d):
    if d >= UDIFF_CUTOVER:
        return (f"https://archives.nseindia.com/content/cm/"
                f"BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip")
    return (f"https://archives.nseindia.com/content/historical/EQUITIES/"
            f"{d.year}/{MON[d.month-1]}/cm{d.day:02d}{MON[d.month-1]}{d.year}bhav.csv.zip")


def index_url(d):
    return (f"https://archives.nseindia.com/content/indices/"
            f"ind_close_all_{d:%d%m%Y}.csv")


def equity_path(d):
    ext = "zip"
    return os.path.join(EQ_DIR, str(d.year), f"eq_{d:%Y%m%d}.{ext}")


def index_path(d):
    return os.path.join(IDX_DIR, str(d.year), f"idx_{d:%Y%m%d}.csv")


def _fetch(session, url, dest, retries=3):
    """Download url → dest. Returns 'ok' | 'holiday' | 'fail'."""
    marker = dest + ".holiday"
    if os.path.exists(dest):
        return "ok"          # already have it
    if os.path.exists(marker):
        return "holiday"     # known non-trading day
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    backoff = 1.5
    for attempt in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=40)
            if r.status_code == 200 and len(r.content) > 200:
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    f.write(r.content)
                os.replace(tmp, dest)
                return "ok"
            if r.status_code == 404:
                open(marker, "w").close()   # weekend/holiday — don't retry later
                return "holiday"
        except requests.RequestException:
            pass
        time.sleep(backoff * (attempt + 1))
    return "fail"


def daterange(start, end):
    d = start
    while d <= end:
        if d.weekday() < 5:          # skip Sat/Sun without a request
            yield d
        d += timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end",   required=True, help="YYYY-MM-DD")
    ap.add_argument("--sleep", type=float, default=0.4, help="seconds between requests")
    ap.add_argument("--no-index",    action="store_true", help="skip index bhavcopy")
    ap.add_argument("--equity-only", action="store_true", help="alias for --no-index")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)
    do_index = not (args.no_index or args.equity_only)

    days = list(daterange(start, end))
    print(f"Downloading {len(days)} weekdays  {start} → {end}")
    print(f"  equity → {EQ_DIR}")
    if do_index:
        print(f"  index  → {IDX_DIR}")

    session = requests.Session()
    stats = {"ok": 0, "holiday": 0, "fail": 0}
    fails = []

    for i, d in enumerate(days, 1):
        st = _fetch(session, equity_url(d), equity_path(d))
        stats[st] += 1
        if st == "fail":
            fails.append(("eq", d))
        if st != "ok":   # if equity is a holiday, index is too — skip the index call
            do_idx_today = (st != "holiday") and do_index
        else:
            do_idx_today = do_index

        if do_idx_today:
            sti = _fetch(session, index_url(d), index_path(d))
            if sti == "fail":
                fails.append(("idx", d))

        if i % 50 == 0 or i == len(days):
            print(f"  [{i}/{len(days)}] {d}  ok={stats['ok']} "
                  f"holiday={stats['holiday']} fail={stats['fail']}")
        time.sleep(args.sleep)

    print(f"\nDone. ok={stats['ok']} holiday={stats['holiday']} fail={stats['fail']}")
    if fails:
        print(f"{len(fails)} failures (re-run to retry): "
              f"{', '.join(f'{k}:{v}' for k, v in fails[:10])}"
              f"{' …' if len(fails) > 10 else ''}")


if __name__ == "__main__":
    main()

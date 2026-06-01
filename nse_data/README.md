# Bias-free NSE price cache (from bhavcopy archives)

Builds a **survivorship-bias-free, corporate-action-adjusted** price cache from
NSE's official daily bhavcopy archives, so the local `backtester.py` becomes
genuinely bias-free — it includes delisted/merged/bankrupt names (AMTEKAUTO,
BHUSHAN, ESSAR, …) that yfinance can never provide.

## Why this is the real fix

- **Survivorship**: a stock that delisted in 2015 appears in every bhavcopy
  through 2015, then disappears. Downloading the full archive captures the dead
  names exactly as they traded — no look-back filtering on today's survivors.
- **Renames**: the cache is keyed by **symbol**; ticker renames are repaired via a
  curated, ISIN-verified alias list in `build_caches.py` (ALIASES) — *not* automatic
  ISIN stitching. Renames outside that list won't be merged; add them to ALIASES.
- **Corporate actions**: back-adjusted using NSE's own **PREVCLOSE** (it prints
  an adjusted previous close on split/bonus/rights ex-dates). No separate file.
- **True benchmark**: the index bhavcopy carries **NIFTY LargeMidcap 250** itself.

## Runbook

```bash
# 1. Download the archive (LONG — ~4800 files for 2007→now). Resumable: safe to
#    Ctrl-C and re-run, or run year-by-year. Consider running in the background.
for y in $(seq 2007 2026); do python nse_data/download.py --start $y-01-01 --end $y-12-31; done

# 2. Fetch the authoritative split/bonus CALENDAR from yfinance (one-time, ~5 min).
#    NSE's PREVCLOSE is raw, so splits can't be derived from bhavcopy prices alone
#    (a 1:1 bonus looks identical to a −50% crash). yfinance gives the ex-dates;
#    build_caches reads the exact ratio from NSE's own price jump at each one.
python nse_data/fetch_splits.py        # → nse_data/splits.csv

# 3. Build the bias-free, split-adjusted cache.
python nse_data/build_caches.py
#    → price_cache.csv, open_cache.csv, volume_cache.csv,
#      regime_cache.csv, adjustments_report.csv

# 4. Point the backtester at it and run.
#    Set USE_NSE_BHAVCOPY = True in config.py, then:
python backtester.py
```

Notes:
- **Equity prices** come from bhavcopy (survivorship-free). **Index/regime series**
  (Nifty 50/500/…) are read from the existing yfinance `regime_data_cache.csv`,
  because NSE's `ind_close_all` archive only exists from ~2015 and indices have no
  survivorship bias anyway — this keeps the regime filter active back to 2008.
- Re-running step 2 is only needed when new corporate actions occur.

## Verify the build (do this — don't trust blindly)

- **Coverage line** at the end of `build_caches.py` should show far fewer missing
  universe tickers than yfinance's 47/420 gap. Remaining misses are genuine
  never-on-NSE-cash or symbol-mismatch cases.
- **`adjustments_report.csv`** lists every detected corporate action
  (isin, symbol, date, ratio). Eyeball a few known ones (e.g. a 1:1 bonus → 0.5,
  a 1:5 split → 0.2). If you see many tiny ratios, raise `--action-threshold`;
  if real splits are missed, lower it.

## Important caveats

- **Adjustment needs contiguous daily data.** The PREVCLOSE ratio compares
  consecutive *trading* days, so download a continuous range. A sparse/partial
  download fabricates false ratios (and a tiny sample under-reports coverage).
- **Dividends are not adjusted** — only splits/bonus/rights (capital actions).
  This matches a price-index convention and is appropriate for price-momentum +
  DMA. (yfinance `auto_adjust=True` *did* fold in dividends, so absolute returns
  will differ slightly from the old cache — this version is the cleaner basis.)
- **EQ series only** (delivery). Trade-to-trade (BE/BZ) names are excluded.
- The live pipeline is untouched: `data_manager.py`/`execution.py` always use the
  yfinance caches. `USE_NSE_BHAVCOPY` only affects `backtester.py`.

## Files

| File | Role |
|---|---|
| `download.py` | fetch raw equity + index bhavcopy zips (resumable) |
| `build_caches.py` | parse → ISIN-stitch → adjust → emit caches |
| `raw/` | downloaded archive (git-ignore this; it's large) |
| `price_cache.csv` etc. | the bias-free cache the backtester reads |
| `adjustments_report.csv` | every corporate action applied — audit this |

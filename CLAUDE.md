# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this system does

Quantitative momentum trading system for NSE (India) equities. Holds top-momentum stocks from the Nifty LargeMidCap 250 universe, with daily exit monitoring and monthly full rebalance. Live execution via Zerodha Kite Connect; paper mode by default.

## Daily workflow

```bash
# Step 1 — update price cache (~3:40 PM IST after market close)
python data_manager.py

# Step 2 — run signals and place orders (~3:45 PM IST)
python execution.py

# Or both in one shot (used by cron — also auto-handles Kite login)
python scheduler.py
```

## Running backtests

```bash
python backtester.py       # India — daily exits, monthly rotation
```

Two data sources, selected by `USE_NSE_BHAVCOPY` in `config.py`:
- **`False`** → yfinance cache (`price_data_cache.csv`, `volume_data_cache.csv`, `regime_data_cache.csv`). Fast, but **survivorship-biased** (yfinance only carries today's survivors) and the universe membership is whatever `universe_history.csv` holds.
- **`True`** (default) → the **bias-free** cache under `nse_data/` built from NSE bhavcopy archives: includes delisted names, corporate-action-adjusted, point-in-time universe. This is the honest backtest. See `nse_data/README.md` to (re)build it.

The backtest exercises the regime filter (`BACKTEST_FORCE_RISK_ON = False`), real STCG/LTCG tax settled annually, brokerage/STT/slippage, next-day-open fills with missed gap-ups, and idle-cash interest. It benchmarks against a same-universe equal-weight index and Nifty 50.

## Running signals standalone

```bash
python signals.py          # reads cache, prints current top holdings and any exits
```

## Architecture

```
config.py             ← single source of truth for ALL parameters
data_manager.py       ← incremental cache management (yfinance → CSV)
signals.py            ← signal engine (cache-only, zero network calls)
execution.py          ← order placement via Kite Connect + Telegram alerts
kite_login.py         ← Playwright-based headless Kite login (TOTP auto-fill)
scheduler.py          ← thin cron wrapper: kite_login → update_cache → run_execution
backtester.py         ← walk-forward India backtest (daily exits + monthly rotation)
universe_history.csv  ← point-in-time Nifty LargeMidCap 250 membership (anti-survivorship)

tools/                ← infrequently-run universe-maintenance utilities
  build_universe_history.py ← rebuilds universe_history.csv from nse_snapshots/
  merge_constituents.py     ← merges dated constituent files into universe_history.csv

nse_data/             ← survivorship-bias-free price cache from NSE bhavcopy
  download.py         ← fetch raw daily bhavcopy (equity + index), resumable
  fetch_splits.py     ← authoritative split/bonus calendar from yfinance
  build_caches.py     ← parse → split-adjust → emit bias-free price/open/volume cache
  README.md           ← runbook + caveats
```

**Data flow:** `data_manager` fetches from Yahoo Finance and writes CSVs → `signals` and `execution` read only from those CSVs — no live network calls during trading hours.

**Rebalancing logic (two-tier):**
- *Daily:* Check every holding against exit rules (rank dropout from top 25, price < 250 DMA, regime flip). On exit, immediately buy the best available replacement from the top 25.
- *Monthly (day 1–3):* Full portfolio rotation — sell anything not in the new top N, buy the full top N.
- *Immediate:* On RISK-OFF (Nifty 500 < 200 DMA), liquidate everything same day regardless of rebalance schedule.

**Regime filter:** Nifty 500 (`^CRSLDX`) vs its 200-day moving average. RISK-OFF → hold cash. RISK-ON → run the momentum strategy.

**Scoring formula** (weights live in `config.py` — these are current defaults):
```
score = 0.50 * z(mom_12m) + 0.40 * z(mom_6m) + 0.30 * z(mom_3m) + 0.00 * z(vol_6m)
```
All terms are z-scored cross-sectionally before weighting. `SKIP_RECENT = 25` days skips the most recent month to avoid short-term reversal. `W_VOL` is 0 by default (set negative to penalise high-vol stocks).

**Limit order buffers (tiered):** Buy orders use a 5% buffer for the top 5 ranked stocks, 3% for ranks 6–12, and 2% for the rest. Sell orders use a 1% discount.

## Key configuration (config.py)

All strategy parameters live here — never hardcode them elsewhere. Important values:
- `TOP_N = 10` — simultaneous holdings
- `EXIT_RANK_CUTOFF = 25` — sell if rank drops below this
- `MAX_PER_SECTOR = 3` — sector concentration cap
- `DMA_EXIT = 250` — exit stock if price drops below this DMA
- `RISK_FREE_RATE = 0.065` — used for Sharpe/Sortino (India 10yr G-Sec)
- `START_DATE` / `END_DATE` — backtest window (change here, not in backtester files); default spans 2007→2026 to capture 2008/2020/2022 drawdowns
- `REGIME_WEIGHTED = False` — if True, uses a weighted composite of Nifty 500/100/Midcap instead of binary regime

**Live vs backtest regime flags (decoupled — do not conflate):**
- `FORCE_RISK_ON = True` — affects **live/paper** (`execution.py`) only; overrides RISK-OFF so paper testing stays invested. **Set False before going live.**
- `BACKTEST_FORCE_RISK_ON = False` — affects **`backtester.py`** only; keep False so the regime filter (the main drawdown defense) is actually exercised.

**Backtest realism knobs (config.py):**
- `USE_NSE_BHAVCOPY = True` — use the bias-free `nse_data/` cache instead of yfinance
- `ANNUAL_TAX = True` — settle STCG/LTCG once per financial year with the LTCG exemption (vs per-trade)
- `SLIPPAGE_BPS = 5` — per-side slippage on fills
- `CASH_YIELD = 0.05` — interest earned on idle/RISK-OFF cash

## Environment variables (.env)

```
KITE_API_KEY=
KITE_API_SECRET=
KITE_ACCESS_TOKEN=      # refreshed daily after Kite login
KITE_USER_ID=           # required by kite_login.py for headless login
KITE_PASSWORD=          # required by kite_login.py for headless login
KITE_TOTP_SECRET=       # TOTP seed for kite_login.py (pyotp)
TELEGRAM_BOT_TOKEN=     # optional — for order notifications
TELEGRAM_CHAT_ID=
TRADING_CAPITAL=100000  # starting capital in ₹
```

## Live trading

Two flags must both be flipped before going live:
1. `PAPER_MODE = True` in `execution.py` — flip to `False`
2. `FORCE_RISK_ON = True` in `config.py` — flip to `False` (this is the **live** flag; the separate `BACKTEST_FORCE_RISK_ON` does not affect live)

In paper mode, orders are logged to `orders.log` but nothing touches Zerodha. Live mode requires `kiteconnect` installed (`pip install kiteconnect`). Headless Kite login via `kite_login.py` requires `playwright` and `pyotp`.

## State files

- `current_holdings.json` — persists live share counts + avg price + entry date across runs; edit manually if needed to sync with actual Zerodha positions
- `orders.log` — append-only order log
- `scheduler.log` — append-only cron run log
- `price_data_cache.csv`, `volume_data_cache.csv`, `open_data_cache.csv`, `regime_data_cache.csv` — yfinance data cache (safe to delete to force full re-download); also used live
- `nse_data/price_cache.csv`, `open_cache.csv`, `volume_cache.csv`, `regime_cache.csv`, `splits.csv`, `adjustments_report.csv` — bias-free backtest cache (regenerate via `nse_data/build_caches.py`)
- `nse_data/raw/` — bhavcopy download dir (git-ignored; created on demand by `download.py`, deleted after the cache is built)
- `equity_curve_daily.csv`, `trade_log_daily.csv` — backtest output files
- `backups/` — auto-created daily backups of `current_holdings.json`, last 7 days kept

## Universe

- **Live** (`signals.py` `UNIVERSE` dict, `data_manager.py` `UNIVERSE_TICKERS` list) must stay in sync; update **both** if you add/remove tickers.
- **Backtest** uses `universe_history.csv` (point-in-time membership) when present, which is what removes survivorship bias from universe selection; it falls back to the hardcoded `UNIVERSE` dict in `backtester.py` only if that file is missing. `universe_history.csv` is stale after 2020-07-31 — extend it via `tools/merge_constituents.py` (see `nse_data/README.md`).

## Dependencies

Python 3.9. Install with:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

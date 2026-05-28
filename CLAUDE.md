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
python backtester.py       # India — daily exits, monthly rotation (primary)
python backtester_us.py    # US S&P 500 — comparison only
python backtester_uk.py    # UK FTSE — comparison only
```

Backtests prefer cached data (`price_data_cache.csv`, `volume_data_cache.csv`, `regime_data_cache.csv`). If cache is missing they fall back to a full yfinance download (~4 min).

## Running signals standalone

```bash
python signals.py          # reads cache, prints current top holdings and any exits
```

## Architecture

```
config.py          ← single source of truth for ALL parameters
data_manager.py    ← incremental cache management (yfinance → CSV)
signals.py         ← signal engine (cache-only, zero network calls)
execution.py       ← order placement via Kite Connect + Telegram alerts
kite_login.py      ← Playwright-based headless Kite login (TOTP auto-fill)
scheduler.py       ← thin cron wrapper: kite_login → update_cache → run_execution
backtester.py      ← walk-forward India backtest (daily exits + monthly rotation)
backtester_us.py   ← US comparison backtest
backtester_uk.py   ← UK comparison backtest
```

**Data flow:** `data_manager` fetches from Yahoo Finance and writes CSVs → `signals` and `execution` read only from those CSVs — no live network calls during trading hours.

**Rebalancing logic (two-tier):**
- *Daily:* Check every holding against exit rules (rank dropout from top 25, price < 250 DMA, regime flip). On exit, immediately buy the best available replacement from the top 25.
- *Monthly (day 1–3):* Full portfolio rotation — sell anything not in the new top N, buy the full top N.
- *Immediate:* On RISK-OFF (Nifty 500 < 200 DMA), liquidate everything same day regardless of rebalance schedule.

**Regime filter:** Nifty 500 (`^CRSLDX`) vs its 200-day moving average. RISK-OFF → hold cash. RISK-ON → run the momentum strategy.

**Scoring formula:**
```
score = 0.40 * z(mom_12m) + 0.35 * z(mom_6m) + 0.15 * z(mom_3m) - 0.10 * z(vol_6m)
```
All terms are z-scored cross-sectionally before weighting. `SKIP_RECENT = 20` days skips the most recent month to avoid short-term reversal.

**Limit order buffers (tiered):** Buy orders use a 5% buffer for the top 5 ranked stocks, 3% for ranks 6–12, and 2% for the rest. Sell orders use a 1% discount.

## Key configuration (config.py)

All strategy parameters live here — never hardcode them elsewhere. Important values:
- `TOP_N = 12` — simultaneous holdings
- `EXIT_RANK_CUTOFF = 25` — sell if rank drops below this
- `MAX_PER_SECTOR = 3` — sector concentration cap
- `DMA_EXIT = 250` — exit stock if price drops below this DMA
- `RISK_FREE_RATE = 0.065` — used for Sharpe/Sortino (India 10yr G-Sec)
- `START_DATE` / `END_DATE` — backtest window (change here, not in backtester files)
- `FORCE_RISK_ON = True` — overrides RISK-OFF regime for paper testing; **set to False before going live**
- `REGIME_WEIGHTED = False` — if True, uses a weighted composite of Nifty 500/100/Midcap instead of binary regime

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
1. `PAPER_MODE = True` in `execution.py` (line 43) — flip to `False`
2. `FORCE_RISK_ON = True` in `config.py` (line 82) — flip to `False`

In paper mode, orders are logged to `orders.log` but nothing touches Zerodha. Live mode requires `kiteconnect` installed (`pip install kiteconnect`). Headless Kite login via `kite_login.py` requires `playwright` and `pyotp`.

## State files

- `current_holdings.json` — persists live share counts + avg price + entry date across runs; edit manually if needed to sync with actual Zerodha positions
- `orders.log` — append-only order log
- `scheduler.log` — append-only cron run log
- `price_data_cache.csv`, `volume_data_cache.csv`, `open_data_cache.csv`, `regime_data_cache.csv` — data cache (safe to delete to force full re-download)
- `equity_curve_daily.csv`, `trade_log_daily.csv` — backtest output files
- `backups/` — auto-created daily backups of `current_holdings.json`, last 7 days kept

## Universe sync requirement

The `UNIVERSE` dict in `signals.py` and the `UNIVERSE_TICKERS` list in `data_manager.py` must stay in sync. If you add or remove tickers, update **both** files. The backtester files each contain their own copy of `UNIVERSE` for self-containment.

## Dependencies

Python 3.9. Install with:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

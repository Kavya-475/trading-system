"""
config.py
=========
Single source of truth for all strategy parameters.
Edit numbers here — signals.py and backtester.py import from here.
"""

# ─────────────────────────────────────────────
# PORTFOLIO CONSTRUCTION
# How many stocks to hold and how concentrated sectors can be.
# ─────────────────────────────────────────────

# Number of stocks held simultaneously in the portfolio
TOP_N            = 10

# Maximum stocks from any single sector — prevents sector concentration
MAX_PER_SECTOR   = 3

# Sell a held stock if its momentum rank drops below this threshold in the universe
EXIT_RANK_CUTOFF = 25


# ─────────────────────────────────────────────
# SIGNAL WEIGHTS
# These determine how much each momentum factor contributes to the score.
# Only ratios matter — all terms are z-scored cross-sectionally before weighting,
# so absolute scale has no effect on stock rankings.
# ─────────────────────────────────────────────

W_MOM_12M   = 0.50    # 12-month momentum weight — strongest predictor
W_MOM_6M    = 0.40    # 6-month momentum weight — medium-term trend
W_MOM_3M    = 0.30    # 3-month momentum weight — short-term trend
W_VOL       = 0.00    # volatility penalty — set negative to penalise high-vol stocks


# ─────────────────────────────────────────────
# LOOKBACK PERIODS (trading days)
# How far back each momentum calculation reaches.
# ─────────────────────────────────────────────

LOOKBACK_12M    = 280   # ~12 months of trading days
LOOKBACK_6M     = 140   # ~6 months of trading days
LOOKBACK_3M     = 80    # ~3 months of trading days

# How many of the most recent days to skip before measuring momentum.
# Skipping ~1 month avoids short-term reversal (stocks that just ran up tend to pull back).
SKIP_RECENT     = 25

# Number of days for the Nifty 500 moving average used to determine market regime
REGIME_DMA      = 200

# Exit a stock if its price falls below this many-day moving average (trend breakdown)
DMA_EXIT        = 250


# ─────────────────────────────────────────────
# UNIVERSE FILTERS
# Minimum quality thresholds for a stock to be eligible for scoring.
# Currently both set to 0 (disabled) — no filtering at ₹1L capital.
# ─────────────────────────────────────────────

MIN_PRICE           = 0    # ₹ minimum stock price (0 = off)
MIN_AVG_VALUE_CR    = 0    # crore — minimum 60-day avg traded value (0 = off)


# ─────────────────────────────────────────────
# CAPITAL GAINS TAX (India equity — CNC delivery)
# India Budget 2024 changed rates effective 2024-07-23.
# STCG applies when holding period < 1 year, LTCG when >= 1 year.
# ─────────────────────────────────────────────

STCG_RATE_PRE  = 0.15    # Short-term rate before 2024-07-23
STCG_RATE_POST = 0.20    # Short-term rate from 2024-07-23
LTCG_RATE_PRE  = 0.10    # Long-term rate before 2024-07-23 (ignoring ₹1L exemption)
LTCG_RATE_POST = 0.125   # Long-term rate from 2024-07-23 (ignoring ₹1.25L exemption)


# ─────────────────────────────────────────────
# TRANSACTION COSTS (CNC Delivery — Zerodha)
# These are applied per trade to simulate real execution costs.
# Breakdown:
#   Brokerage    : ₹0 (Zerodha CNC is free)
#   STT          : 0.1% on sell side only
#   Exchange NSE : 0.00325% both sides
#   SEBI         : 0.0001% both sides
#   Stamp duty   : 0.015% on buy side only
#   Effective round-trip cost ≈ 0.17%
# ─────────────────────────────────────────────

STT_BUY         = 0.0        # Securities Transaction Tax — not charged on buy for CNC
STT_SELL        = 0.001      # 0.1% of sell value
EXCHANGE_CHARGE = 0.0000325  # NSE exchange fee — charged on both buy and sell
SEBI_CHARGE     = 0.000001   # SEBI regulatory fee — charged on both sides
STAMP_DUTY      = 0.00015    # Stamp duty — charged on buy side only


# ─────────────────────────────────────────────
# ORDER EXECUTION BUFFERS
# Limit orders are placed above/below the last close price so they fill
# even if the stock gaps up/down slightly at next-day open.
# Higher-ranked stocks get a larger buffer because we really want them.
# ─────────────────────────────────────────────

BUY_BUFFER_TOP5 = 0.050   # 5% above close for the top 5 ranked stocks
BUY_BUFFER_MID  = 0.030   # 3% above close for ranks 6–12
BUY_BUFFER_REST = 0.020   # 2% above close for remaining stocks
SELL_BUFFER     = 0.010   # 1% below close on sell orders (ensures fill)


# ─────────────────────────────────────────────
# BACKTEST SETTINGS
# Date range and starting capital for backtester.py simulations.
# ─────────────────────────────────────────────

START_DATE      = "2008-01-01"   # First date the backtest places orders
END_DATE        = "2026-05-29"   # Last date included in the backtest
INITIAL_CAPITAL = 100000         # ₹1 lakh starting capital
RISK_FREE_RATE  = 0.065          # 6.5% — India 10-year G-Sec, used for Sharpe/Sortino


# ─────────────────────────────────────────────
# DATA
# File paths for the price/volume/regime caches and index tickers.
# ─────────────────────────────────────────────

DATA_FETCH_START = "2007-01-01"          # Download history from this date (1yr before START_DATE for warmup)
DATA_FETCH_END   = "2026-05-29"          # Download up to this date
DATA_CACHE_FILE  = "price_data_cache.csv"   # Daily adjusted close prices for all 250 stocks
VOLUME_CACHE     = "volume_data_cache.csv"  # Daily trading volume for all 250 stocks
OPEN_CACHE       = "open_data_cache.csv"    # Daily open prices (used for realistic fill simulation)
REGIME_CACHE     = "regime_data_cache.csv"  # Nifty 500 and Nifty 50 index prices
BENCHMARK_TICKER = "^NSEI"               # Nifty 50 — used as benchmark in backtests
REGIME_TICKER    = "^CRSLDX"            # Nifty 500 — used for market regime filter


# ─────────────────────────────────────────────
# PAPER TESTING FLAGS
# Both must be flipped before going live.
# ─────────────────────────────────────────────

# When True, overrides the regime filter so the strategy always stays invested.
# Used during paper testing so we can observe signals even in RISK-OFF markets.
# Set to False before going live.
FORCE_RISK_ON   = True

# Emergency kill switch — set True to immediately stop all trading and signal processing
TRADING_HALTED  = False


# ─────────────────────────────────────────────
# WEIGHTED REGIME SETTINGS
# Alternative to the simple binary RISK-ON/RISK-OFF regime filter.
# Uses a composite of three indices to determine what fraction of capital to deploy.
# Only active when REGIME_WEIGHTED = True.
# ─────────────────────────────────────────────

# Set True to use the weighted composite regime instead of binary on/off
REGIME_WEIGHTED = False

# Weight of each index in the composite regime strength score (must sum to 1.0)
REGIME_WEIGHT_NIFTY500  = 0.40
REGIME_WEIGHT_NIFTY100  = 0.35
REGIME_WEIGHT_MIDCAP    = 0.25

# If composite score is below DEPLOY_MIN → hold 0% (full cash)
# If composite score is above DEPLOY_MAX → deploy 100% (fully invested)
# Between the two thresholds → deploy linearly (e.g. 50% deployed)
REGIME_DEPLOY_MIN = -0.02   # -2% below DMA → move to cash
REGIME_DEPLOY_MAX =  0.02   # +2% above DMA → fully invest


# ─────────────────────────────────────────────
# REGIME CONFIRMATION FILTER
# Requires the market to stay above its DMA for N consecutive days
# before re-entering after a RISK-OFF period.
# RISK-OFF exit is always immediate (asymmetric — exit fast, enter carefully).
# ─────────────────────────────────────────────

# How many consecutive days above DMA before flipping to RISK-ON
# 0 means flip immediately on the first day above DMA (original behaviour)
REGIME_CONFIRM_DAYS = 0

"""
config.py
=========
Single source of truth for all strategy parameters.
Edit numbers here — signals.py and backtester.py import from here.

AGGRESSIVE configuration selected on 2026-05-23.
"""

# ─────────────────────────────────────────────
# PORTFOLIO CONSTRUCTION
# ─────────────────────────────────────────────
TOP_N             = 7      # stocks to hold simultaneously (was 10)
MAX_PER_SECTOR    = 3      # max stocks from one sector (was 2)
EXIT_RANK_CUTOFF  = 25     # sell if rank drops below this (was 20)

# ─────────────────────────────────────────────
# SIGNAL WEIGHTS  (must sum to 1.0 for long legs)
# ─────────────────────────────────────────────
W_MOM_12M   = 0.40    # 12-month momentum weight
W_MOM_6M    = 0.35    # 6-month momentum weight
W_MOM_3M    = 0.15    # 3-month momentum weight  ← NEW
W_VOL       = -0.10   # volatility penalty (negative = penalise)

# ─────────────────────────────────────────────
# LOOKBACK PERIODS (trading days)
# ─────────────────────────────────────────────
LOOKBACK_12M    = 252   # ~12 months
LOOKBACK_6M     = 126   # ~6 months
LOOKBACK_3M     = 63    # ~3 months  ← NEW
SKIP_RECENT     = 20    # skip last 1 month in momentum (avoids reversal)
REGIME_DMA      = 200   # moving average for market regime filter
DMA_EXIT        = 100   # exit stock if price drops below this DMA

# ─────────────────────────────────────────────
# UNIVERSE FILTERS
# ─────────────────────────────────────────────
MIN_PRICE           = 100    # ₹ minimum stock price
MIN_AVG_VALUE_CR    = 20     # crore — minimum 60-day avg traded value

# ─────────────────────────────────────────────
# TRANSACTION COSTS (CNC Delivery — Zerodha)
# ─────────────────────────────────────────────
# Brokerage    : ₹0 (Zerodha CNC)
# STT          : 0.1% on sell side only
# Exchange NSE : 0.00325% both sides
# SEBI         : 0.0001% both sides
# Stamp duty   : 0.015% on buy side
# Effective round-trip cost ≈ 0.17%
# CORRECT
STT_BUY  = 0.001      # 0.1% on buy side
STT_SELL = 0.001      # 0.1% on sell side
EXCHANGE_CHARGE = 0.0000325
SEBI_CHARGE     = 0.000001
STAMP_DUTY      = 0.00015

# ─────────────────────────────────────────────
# BACKTEST SETTINGS
# ─────────────────────────────────────────────
# Change these two lines
# config.py — change back to original
START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"
INITIAL_CAPITAL = 100000      # ₹1 lakh (results scale %-wise regardless)
RISK_FREE_RATE  = 0.065       # 6.5% — India 10yr G-Sec

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
DATA_FETCH_START = "2018-01-01"   # extra buffer for DMA calculations
DATA_FETCH_END   = "2025-01-01"
DATA_CACHE_FILE  = "price_data_cache.csv"
VOLUME_CACHE     = "volume_data_cache.csv"
REGIME_CACHE     = "regime_data_cache.csv"
BENCHMARK_TICKER = "^NSEI"        # Nifty 50
REGIME_TICKER    = "^CRSLDX"      # Nifty 500
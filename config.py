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
# results are slightly changing 
TOP_N            = 10
MAX_PER_SECTOR   = 3
# barely changing 
EXIT_RANK_CUTOFF = 40     # sell if rank drops below this (was 20)

# ─────────────────────────────────────────────
# SIGNAL WEIGHTS  (must sum to 1.0 for long legs)
# ─────────────────────────────────────────────

# results are heavily changing
W_MOM_12M   = 0.40    # 12-month momentum weight
W_MOM_6M    = 0.35    # 6-month momentum weight
W_MOM_3M    = 0.15    # 3-month momentum weight  ← NEW
# quite interesting removing it kept cagr same but reduced dd by 10% -> good
# needs optimzation 
W_VOL       = -0.10   # volatility penalty (negative = penalise)

# ─────────────────────────────────────────────
# LOOKBACK PERIODS (trading days)
# ─────────────────────────────────────────────
# changing drastically 
LOOKBACK_12M    = 252   # ~12 months
LOOKBACK_6M     = 126   # ~6 months
LOOKBACK_3M     = 63    # ~3 months  ← NEW
# INTERESTING , changing it single handedly reduces dd by 12%(when kept 0 --> dd=57) needs optimization
SKIP_RECENT     = 20    # skip last 1 month in momentum (avoids reversal)

# mostly the results are remaining same for these 
# 
REGIME_DMA      = 200   # moving average for market regime filter
DMA_EXIT        = 250   # exit stock if price drops below this DMA

# ─────────────────────────────────────────────
# UNIVERSE FILTERS
# ─────────────────────────────────────────────
# results are changing too much 
MIN_PRICE           = 0    # ₹ minimum stock price
MIN_AVG_VALUE_CR    = 0    # crore — minimum 60-day avg traded value

# ─────────────────────────────────────────────
# CAPITAL GAINS TAX (India equity — CNC delivery)
# ─────────────────────────────────────────────
# Budget 2024 (effective 2024-07-23): STCG 15%→20%, LTCG 10%→12.5%
STCG_RATE_PRE  = 0.15    # < 1 year hold, before 2024-07-23
STCG_RATE_POST = 0.20    # < 1 year hold, from 2024-07-23
LTCG_RATE_PRE  = 0.10    # >= 1 year hold, before 2024-07-23 (ignoring ₹1L exemption)
LTCG_RATE_POST = 0.125   # >= 1 year hold, from 2024-07-23 (ignoring ₹1.25L exemption)

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
STT_BUY  = 0.0        # STT not applicable on buy side for CNC delivery
STT_SELL = 0.001      # 0.1% on sell side
EXCHANGE_CHARGE = 0.0000325
SEBI_CHARGE     = 0.000001
STAMP_DUTY      = 0.00015

# ─────────────────────────────────────────────
# ORDER EXECUTION BUFFERS
# ─────────────────────────────────────────────
BUY_BUFFER_TOP5 = 0.050   # limit buffer for top 5 ranked stocks
BUY_BUFFER_MID  = 0.030   # limit buffer for ranks 6–12
BUY_BUFFER_REST = 0.020   # limit buffer for remaining stocks
SELL_BUFFER     = 0.010   # limit discount on sell orders

# ─────────────────────────────────────────────
# BACKTEST SETTINGS
# ─────────────────────────────────────────────
START_DATE = "2008-01-01"
END_DATE   = "2026-05-22"
INITIAL_CAPITAL = 100000      # ₹1 lakh (results scale %-wise regardless)
RISK_FREE_RATE  = 0.065       # 6.5% — India 10yr G-Sec

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
DATA_FETCH_START = "2007-01-01"   # 1yr warmup buffer before START_DATE
DATA_FETCH_END   = "2026-05-28"   # must match or exceed END_DATE
DATA_CACHE_FILE  = "price_data_cache.csv"
VOLUME_CACHE     = "volume_data_cache.csv"
OPEN_CACHE       = "open_data_cache.csv"
REGIME_CACHE     = "regime_data_cache.csv"
BENCHMARK_TICKER = "^NSEI"        # Nifty 50
REGIME_TICKER    = "^CRSLDX"      # Nifty 500
# ─────────────────────────────────────────────
# PAPER TESTING FLAGS
# ─────────────────────────────────────────────
FORCE_RISK_ON   = True   # Set False before going live
TRADING_HALTED  = False  # Safety kill switch — set True to suspend all trading

# ─────────────────────────────────────────────
# WEIGHTED REGIME SETTINGS
# ─────────────────────────────────────────────
REGIME_WEIGHTED = False   # True = weighted strength, False = original binary

# Index weights for regime strength (must sum to 1.0)
REGIME_WEIGHT_NIFTY500  = 0.40
REGIME_WEIGHT_NIFTY100  = 0.35
REGIME_WEIGHT_MIDCAP    = 0.25

# Deployment thresholds
# Below DEPLOY_MIN → 0% invested (full cash)
# Above DEPLOY_MAX → 100% invested (full 7 stocks)
# Linear between the two
REGIME_DEPLOY_MIN = -0.02   # -2% below DMA → cash
REGIME_DEPLOY_MAX =  0.02   # +2% above DMA → fully invested

# ─────────────────────────────────────────────
# REGIME CONFIRMATION FILTER
# ─────────────────────────────────────────────
# Number of consecutive days above 200 DMA before triggering RISK-ON
# RISK-OFF is still immediate (asymmetric — protect capital fast)
REGIME_CONFIRM_DAYS = 0   # 0 = original behaviour (no confirmation needed)

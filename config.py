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
TOP_N            = 15

# Maximum stocks from any single sector — prevents sector concentration
MAX_PER_SECTOR   = 5

# Sell a held stock if its momentum rank drops below this threshold in the universe
EXIT_RANK_CUTOFF = 50

# Rebalance cadence for the full rotation (sell non-top-N + refill to top-N):
#   "monthly" | "weekly" | "2x-week" | "none" (build once, then only exit-driven turnover)
REBALANCE_FREQ   = "monthly"

# Rank-velocity exit: sell if a holding falls more than RANK_DROP_EXIT positions
# vs its rank RANK_DROP_LOOKBACK trading days ago. 0 = off.
RANK_DROP_EXIT     = 0
RANK_DROP_LOOKBACK = 3

# Position sizing at the monthly rebalance:
#   "equal"  → 1/N each
#   "tiered" → higher momentum rank gets more capital (forensics: rank was the #1
#              separator of exceptional winners). Also stops dropping costly high-rank
#              names: they're bought with their (larger) budget and leftover is
#              redistributed by the same weights.
POSITION_WEIGHTING = "tiered"
# Tier multipliers for ranks [top 5] / [6–12] / [rest], normalised across the held set.
RANK_TIER_WEIGHTS  = (1.3, 1.0, 0.8)


# ─────────────────────────────────────────────
# SIGNAL WEIGHTS
# These determine how much each momentum factor contributes to the score.
# Only ratios matter — all terms are z-scored cross-sectionally before weighting,
# so absolute scale has no effect on stock rankings.
# ─────────────────────────────────────────────

W_MOM_12M   = 0.40    # 12-month momentum weight (optimizer: tilt down toward 6M)
W_MOM_6M    = 0.50    # 6-month momentum weight — medium-term trend (optimizer favored)
W_MOM_3M    = 0.20    # 3-month momentum weight — short-term trend
W_VOL       = 0    # volatility penalty — set negative to penalise high-vol stocks
W_DIST_DMA  = 0.30    # weight on z-scored distance above the 250-DMA (trend extension).
                      # Forensics: dist-above-DMA was the #2 separator of big winners. 0 = off.


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
# Buffer below the DMA before the EXIT fires: sell only when price is this fraction
# below the DMA (0.02 = 2%). The entry gate stays strict (buy at/above DMA), so this
# creates a hysteresis band that cuts DMA ping-pong churn. 0 = exit exactly at the line.
DMA_EXIT_BUFFER = 0.0
# Master switch for the 250-DMA trend filter (both the entry gate AND the exit).
# False = the DMA plays no role (the dist_dma SCORING factor is separate & unaffected).
USE_DMA         = True

# Hard stop-loss: sell if a position is down this fraction from its avg cost.
# 0 = off (the 250-DMA is then the only, loose, trend stop).
STOP_LOSS_PCT   = 0.0


# ─────────────────────────────────────────────
# UNIVERSE FILTERS
# Minimum quality thresholds for a stock to be eligible for scoring.
# Currently both set to 0 (disabled) — no filtering at ₹1L capital.
# ─────────────────────────────────────────────

MIN_PRICE           = 0    # ₹ minimum stock price (0 = off)
MIN_AVG_VALUE_CR    = 1.0    # crore — min 60-day avg traded value (1.0 for live tradability;
                             # optimizer wanted 0.1 but sub-1cr names aren't realistically tradable)

# "Flat" / stale-price filter. The bias-free cache forward-fills non-trading days,
# so a suspended or delisted name can appear as a flat line (and, if it ran up
# before going flat, score high on stale momentum). Exclude any stock whose last
# 60 days are >this fraction zero-change. 0 = off.
MAX_FLAT_FRAC       = 0.5  # >50% zero-change days in last 60 → treat as untradeable


# ─────────────────────────────────────────────
# CAPITAL GAINS TAX (India equity — CNC delivery)
# India Budget 2024 changed rates effective 2024-07-23.
# STCG applies when holding period < 1 year, LTCG when >= 1 year.
# ─────────────────────────────────────────────

STCG_RATE_PRE  = 0.15    # Short-term rate before 2024-07-23
STCG_RATE_POST = 0.20    # Short-term rate from 2024-07-23
LTCG_RATE_PRE  = 0.10    # Long-term rate before 2024-07-23
LTCG_RATE_POST = 0.125   # Long-term rate from 2024-07-23

# Annual LTCG exemption (applied once per financial year, not per trade).
LTCG_EXEMPTION_PRE  = 100000   # ₹1.00L exemption before 2024-07-23
LTCG_EXEMPTION_POST = 125000   # ₹1.25L exemption from 2024-07-23

# When True, capital-gains tax is netted and settled once per financial year
# (Apr–Mar), honouring the annual LTCG exemption and short-term loss offset —
# this is how tax actually works and lets gains compound within the year.
# When False, tax is deducted per trade (older, more conservative behaviour).
ANNUAL_TAX = True


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

STT_BUY         = 0.001      # 0.1% — delivery STT is charged on BOTH buy and sell
STT_SELL        = 0.001      # 0.1% of sell value
EXCHANGE_CHARGE = 0.0000325  # NSE exchange fee — charged on both buy and sell
SEBI_CHARGE     = 0.000001   # SEBI regulatory fee — charged on both sides
STAMP_DUTY      = 0.00015    # Stamp duty — charged on buy side only
GST_RATE        = 0.18       # 18% GST on (brokerage + exchange + SEBI) charges

# Per-side slippage + half-spread applied to the simulated fill price, on top of
# the next-day-open limit model. Models the gap between the printed open and the
# price you actually get.
SLIPPAGE_BPS    = 5          # adverse slippage, bps per side
SPREAD_BPS      = 3          # half bid-ask spread paid per side, bps

# Max fraction of a stock's 60-day average daily volume a single order may take
# (participation limit). At ₹1L this rarely binds; matters at larger capital.
PARTICIPATION_LIMIT = 0.10   # 10% of ADV; set 0 to disable

# Annualised yield on idle cash. Zerodha trading cash earns NOTHING unless you
# actively sweep into a liquid/overnight fund (and that return is taxable). Set
# to 0 to avoid the optimistic free-interest assumption; raise it only if you
# model an explicit liquid-fund sweep.
CASH_YIELD      = 0.0        # % p.a. on idle cash (0 = broker cash earns nothing)


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

START_DATE      = "2007-01-01"   # First date the backtest places orders.
                                 # Bias-free cache now extends back to 2005-01, so
                                 # the 305-day momentum warmup is satisfied by 2006
                                 # data and trading begins cleanly in Jan 2007 —
                                 # capturing the FULL 2008 GFC peak-to-trough
                                 # (the earlier 2007-10 start started mid-warmup,
                                 #  leaving 2008 only half-tested).
END_DATE        = "2020-8-30"   # Last date included in the backtest
INITIAL_CAPITAL = 100000         # ₹1 lakh starting capital
RISK_FREE_RATE  = 0.065          # 6.5% — India 10-year G-Sec, used for Sharpe/Sortino


# ─────────────────────────────────────────────
# DATA
# File paths for the price/volume/regime caches and index tickers.
# ─────────────────────────────────────────────

DATA_FETCH_START = "2005-01-01"          # Download history from this date — needs 305 trading days before START_DATE for warmup
DATA_FETCH_END   = "2026-05-29"          # Download up to this date
DATA_CACHE_FILE  = "price_data_cache.csv"   # Daily adjusted close prices for all 250 stocks
VOLUME_CACHE     = "volume_data_cache.csv"  # Daily trading volume for all 250 stocks
OPEN_CACHE       = "open_data_cache.csv"    # Daily open prices (used for realistic fill simulation)
REGIME_CACHE     = "regime_data_cache.csv"  # Nifty 500 and Nifty 50 index prices
BENCHMARK_TICKER = "^NSEI"               # Nifty 50 — used as benchmark in backtests
REGIME_TICKER    = "^CRSLDX"            # Nifty 500 — used for market regime filter

# ── BIAS-FREE CACHE (NSE bhavcopy, built by nse_data/build_caches.py) ────────
# When True, the backtester loads the survivorship-bias-free, corporate-action-
# adjusted cache built from NSE daily bhavcopy archives (includes delisted names)
# instead of the yfinance cache. The live pipeline (data_manager/execution) is
# unaffected — it always uses the yfinance caches above.
USE_NSE_BHAVCOPY = True
NSE_PRICE_CACHE  = "nse_data/price_cache.csv"
NSE_OPEN_CACHE   = "nse_data/open_cache.csv"
NSE_VOLUME_CACHE = "nse_data/volume_cache.csv"
NSE_REGIME_CACHE = "nse_data/regime_cache.csv"


# ─────────────────────────────────────────────
# PAPER TESTING FLAGS
# Both must be flipped before going live.
# ─────────────────────────────────────────────

# When True, overrides the regime filter so the strategy always stays invested.
# Used during paper testing so we can observe signals even in RISK-OFF markets.
# Set to False before going live.
FORCE_RISK_ON   = True

# Backtest-only override, decoupled from the live FORCE_RISK_ON above so that
# changing backtest realism never alters live/paper execution behaviour.
# MUST be False for a realistic backtest — the regime filter is the strategy's
# main drawdown defense and has to be exercised over 2008/2020/2022. This is the
# INTENDED strategy; only flip to True to study an always-invested variant.
BACKTEST_FORCE_RISK_ON = True   # regime filter OFF — always invested (the 16.7%/-60% max-return profile)

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

"""
signals.py
==========
Signal engine — reads price data from the CSV cache and produces
momentum scores, portfolio selection, and exit signals.

Reads entirely from the local cache (no network calls).
Run data_manager.py first each day to ensure the cache is up to date.

Daily workflow:
    python data_manager.py   ← updates cache (run once at 3:40 PM)
    python execution.py      ← reads cache, runs signals, places orders

Run standalone to see today's top 10 and any exit signals:
    python signals.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import config as cfg
from data_manager import load_for_signals, load_index_data

# ─────────────────────────────────────────────
# NIFTY LARGEMIDCAP 250 UNIVERSE
# Maps each NSE ticker to its sector for the sector concentration cap.
# Must stay in sync with UNIVERSE_TICKERS in data_manager.py.
# ─────────────────────────────────────────────
UNIVERSE = {

    # ── Auto ──
    "APOLLOTYRE": "Auto",
    "BAJAJ-AUTO": "Auto",
    "BALKRISIND": "Auto",
    "BHARATFORG": "Auto",
    "BOSCHLTD": "Auto",
    "EICHERMOT": "Auto",
    "ENDURANCE": "Auto",
    "EXIDEIND": "Auto",
    "HEROMOTOCO": "Auto",
    "HYUNDAI": "Auto",
    "M&M": "Auto",
    "MARUTI": "Auto",
    "MOTHERSON": "Auto",
    "MRF": "Auto",
    "SCHAEFFLER": "Auto",
    "TIINDIA": "Auto",
    "TMPV": "Auto",
    "TVSMOTOR": "Auto",
    "UNOMINDA": "Auto",

    # ── Capital Goods ──
    "ABB": "Capital Goods",
    "AIAENG": "Capital Goods",
    "APARINDS": "Capital Goods",
    "APLAPOLLO": "Capital Goods",
    "ASHOKLEY": "Capital Goods",
    "ASTRAL": "Capital Goods",
    "BDL": "Capital Goods",
    "BEL": "Capital Goods",
    "BHEL": "Capital Goods",
    "CGPOWER": "Capital Goods",
    "COCHINSHIP": "Capital Goods",
    "CUMMINSIND": "Capital Goods",
    "ENRIN": "Capital Goods",
    "ESCORTS": "Capital Goods",
    "GVT&D": "Capital Goods",
    "HAL": "Capital Goods",
    "HONAUT": "Capital Goods",
    "KEI": "Capital Goods",
    "MAZDOCK": "Capital Goods",
    "POLYCAB": "Capital Goods",
    "POWERINDIA": "Capital Goods",
    "PREMIERENE": "Capital Goods",
    "SIEMENS": "Capital Goods",
    "SUPREMEIND": "Capital Goods",
    "SUZLON": "Capital Goods",
    "THERMAX": "Capital Goods",
    "TMCV": "Capital Goods",
    "WAAREEENER": "Capital Goods",

    # ── Cement ──
    "ACC": "Cement",
    "AMBUJACEM": "Cement",
    "DALBHARAT": "Cement",
    "GRASIM": "Cement",
    "JKCEMENT": "Cement",
    "SHREECEM": "Cement",
    "ULTRACEMCO": "Cement",

    # ── Chemicals ──
    "COROMANDEL": "Chemicals",
    "FLUOROCHEM": "Chemicals",
    "LINDEINDIA": "Chemicals",
    "PIDILITIND": "Chemicals",
    "PIIND": "Chemicals",
    "SOLARINDS": "Chemicals",
    "SRF": "Chemicals",
    "UPL": "Chemicals",

    # ── Consumer ──
    "ASIANPAINT": "Consumer",
    "BERGEPAINT": "Consumer",
    "BLUESTARCO": "Consumer",
    "DIXON": "Consumer",
    "DMART": "Consumer",
    "ETERNAL": "Consumer",
    "HAVELLS": "Consumer",
    "INDHOTEL": "Consumer",
    "IRCTC": "Consumer",
    "ITCHOTELS": "Consumer",
    "JUBLFOOD": "Consumer",
    "KALYANKJIL": "Consumer",
    "KPRMILL": "Consumer",
    "LENSKART": "Consumer",
    "LGEINDIA": "Consumer",
    "NAUKRI": "Consumer",
    "NYKAA": "Consumer",
    "PAGEIND": "Consumer",
    "SWIGGY": "Consumer",
    "TITAN": "Consumer",
    "TRENT": "Consumer",
    "VMM": "Consumer",
    "VOLTAS": "Consumer",

    # ── Energy ──
    "ADANIENSOL": "Energy",
    "ADANIGREEN": "Energy",
    "ADANIPOWER": "Energy",
    "ATGL": "Energy",
    "BPCL": "Energy",
    "COALINDIA": "Energy",
    "GAIL": "Energy",
    "HINDPETRO": "Energy",
    "IOC": "Energy",
    "JSWENERGY": "Energy",
    "NHPC": "Energy",
    "NLCINDIA": "Energy",
    "NTPC": "Energy",
    "NTPCGREEN": "Energy",
    "OIL": "Energy",
    "ONGC": "Energy",
    "PETRONET": "Energy",
    "POWERGRID": "Energy",
    "RELIANCE": "Energy",
    "SJVN": "Energy",
    "TATAPOWER": "Energy",
    "TORNTPOWER": "Energy",

    # ── FMCG ──
    "AWL": "FMCG",
    "BRITANNIA": "FMCG",
    "COLPAL": "FMCG",
    "DABUR": "FMCG",
    "GODFRYPHLP": "FMCG",
    "GODREJCP": "FMCG",
    "HINDUNILVR": "FMCG",
    "ITC": "FMCG",
    "MARICO": "FMCG",
    "NESTLEIND": "FMCG",
    "PATANJALI": "FMCG",
    "RADICO": "FMCG",
    "TATACONSUM": "FMCG",
    "UBL": "FMCG",
    "UNITDSPR": "FMCG",
    "VBL": "FMCG",

    # ── Financials ──
    "360ONE": "Financials",
    "ABCAPITAL": "Financials",
    "AIIL": "Financials",
    "AUBANK": "Financials",
    "AXISBANK": "Financials",
    "BAJAJFINSV": "Financials",
    "BAJAJHFL": "Financials",
    "BAJAJHLDNG": "Financials",
    "BAJFINANCE": "Financials",
    "BANKBARODA": "Financials",
    "BANKINDIA": "Financials",
    "BSE": "Financials",
    "CANBK": "Financials",
    "CHOLAFIN": "Financials",
    "CRISIL": "Financials",
    "FEDERALBNK": "Financials",
    "GICRE": "Financials",
    "GROWW": "Financials",
    "HDBFS": "Financials",
    "HDFCAMC": "Financials",
    "HDFCBANK": "Financials",
    "HDFCLIFE": "Financials",
    "HUDCO": "Financials",
    "ICICIAMC": "Financials",
    "ICICIBANK": "Financials",
    "ICICIGI": "Financials",
    "ICICIPRULI": "Financials",
    "IDFCFIRSTB": "Financials",
    "INDIANB": "Financials",
    "INDUSINDBK": "Financials",
    "IREDA": "Financials",
    "IRFC": "Financials",
    "JIOFIN": "Financials",
    "KOTAKBANK": "Financials",
    "LICHSGFIN": "Financials",
    "LICI": "Financials",
    "LTF": "Financials",
    "M&MFIN": "Financials",
    "MAHABANK": "Financials",
    "MCX": "Financials",
    "MFSL": "Financials",
    "MOTILALOFS": "Financials",
    "MUTHOOTFIN": "Financials",
    "NAM-INDIA": "Financials",
    "NIACL": "Financials",
    "PAYTM": "Financials",
    "PFC": "Financials",
    "PNB": "Financials",
    "POLICYBZR": "Financials",
    "RECLTD": "Financials",
    "SBICARD": "Financials",
    "SBILIFE": "Financials",
    "SBIN": "Financials",
    "SHRIRAMFIN": "Financials",
    "SUNDARMFIN": "Financials",
    "TATACAP": "Financials",
    "TATAINVEST": "Financials",
    "UNIONBANK": "Financials",
    "YESBANK": "Financials",

    # ── Healthcare ──
    "ABBOTINDIA": "Healthcare",
    "AJANTPHARM": "Healthcare",
    "ALKEM": "Healthcare",
    "ANTHEM": "Healthcare",
    "APOLLOHOSP": "Healthcare",
    "AUROPHARMA": "Healthcare",
    "BIOCON": "Healthcare",
    "CIPLA": "Healthcare",
    "DIVISLAB": "Healthcare",
    "DRREDDY": "Healthcare",
    "FORTIS": "Healthcare",
    "GLAXO": "Healthcare",
    "GLENMARK": "Healthcare",
    "IPCALAB": "Healthcare",
    "LAURUSLABS": "Healthcare",
    "LUPIN": "Healthcare",
    "MANKIND": "Healthcare",
    "MAXHEALTH": "Healthcare",
    "MEDANTA": "Healthcare",
    "SUNPHARMA": "Healthcare",
    "TORNTPHARM": "Healthcare",
    "ZYDUSLIFE": "Healthcare",

    # ── IT ──
    "COFORGE": "IT",
    "HCLTECH": "IT",
    "HEXT": "IT",
    "INFY": "IT",
    "KPITTECH": "IT",
    "LTM": "IT",
    "LTTS": "IT",
    "MPHASIS": "IT",
    "OFSS": "IT",
    "PERSISTENT": "IT",
    "TATAELXSI": "IT",
    "TCS": "IT",
    "TECHM": "IT",
    "WIPRO": "IT",

    # ── Industrials ──
    "3MINDIA": "Industrials",
    "ADANIPORTS": "Industrials",
    "CONCOR": "Industrials",
    "GMRAIRPORT": "Industrials",
    "GODREJIND": "Industrials",
    "INDIGO": "Industrials",
    "JSWINFRA": "Industrials",
    "LT": "Industrials",
    "RVNL": "Industrials",

    # ── Metals ──
    "ADANIENT": "Metals",
    "HINDALCO": "Metals",
    "HINDZINC": "Metals",
    "JINDALSTEL": "Metals",
    "JSL": "Metals",
    "JSWSTEEL": "Metals",
    "LLOYDSME": "Metals",
    "NATIONALUM": "Metals",
    "NMDC": "Metals",
    "SAIL": "Metals",
    "TATASTEEL": "Metals",
    "VEDL": "Metals",

    # ── Realty ──
    "DLF": "Realty",
    "GODREJPROP": "Realty",
    "LODHA": "Realty",
    "OBEROIRLTY": "Realty",
    "PHOENIXLTD": "Realty",
    "PRESTIGE": "Realty",

    # ── Telecom ──
    "BHARTIARTL": "Telecom",
    "BHARTIHEXA": "Telecom",
    "IDEA": "Telecom",
    "INDUSTOWER": "Telecom",
    "TATACOMM": "Telecom",
}


# ─────────────────────────────────────────────
# REGIME FILTER
# Determines whether the market is RISK-ON (invest) or RISK-OFF (hold cash).
# Uses Nifty 500 vs its REGIME_DMA-day moving average.
# An optional confirmation period requires the index to stay above the DMA
# for N consecutive days before re-entering after a RISK-OFF period.
# ─────────────────────────────────────────────
def get_regime(nifty500: pd.Series) -> str:
    clean  = nifty500.dropna()                            # remove any NaN dates

    # ── FAIL-SAFE: a DATA problem must never look like a sell signal ──────────
    # If we can't compute the DMA (too little history) or the latest value/DMA is
    # NaN, return "UNKNOWN" so the caller ABORTS the run and keeps holdings —
    # rather than silently reading missing data as "below DMA → RISK-OFF" and
    # liquidating the whole book on a download glitch.
    if len(clean) < cfg.REGIME_DMA:
        print(f"\nREGIME CHECK\n  ⚠ Insufficient Nifty 500 history "
              f"({len(clean)} < {cfg.REGIME_DMA} days) — regime UNKNOWN")
        return "UNKNOWN"

    dma    = clean.rolling(cfg.REGIME_DMA).mean()         # rolling 200-day average
    latest = clean.iloc[-1]                               # today's Nifty 500 value
    l_dma  = dma.iloc[-1]                                 # today's 200-DMA value

    if pd.isna(latest) or pd.isna(l_dma):
        print("\nREGIME CHECK\n  ⚠ NaN in Nifty 500 / 200-DMA — regime UNKNOWN")
        return "UNKNOWN"

    # If REGIME_CONFIRM_DAYS > 0, require that many consecutive days above DMA
    # before declaring RISK-ON. This prevents false re-entries during choppy markets.
    # RISK-OFF is still immediate — asymmetric to protect capital faster.
    confirm_days = getattr(cfg, "REGIME_CONFIRM_DAYS", 0)
    if confirm_days > 0 and len(clean) >= confirm_days:
        recent     = clean.iloc[-confirm_days:]           # last N days of index
        recent_dma = dma.iloc[-confirm_days:]             # corresponding DMA values
        all_above  = all(p > d for p, d in zip(recent, recent_dma))
        regime     = "RISK-ON" if all_above else "RISK-OFF"
    else:
        # Default: single day above/below DMA determines regime
        regime = "RISK-ON" if latest > l_dma else "RISK-OFF"

    print(f"\nREGIME CHECK")
    print(f"  Nifty 500 : {latest:,.2f}")
    print(f"  200 DMA   : {l_dma:,.2f}")
    if confirm_days > 0:
        print(f"  Confirm   : {confirm_days} days required for RISK-ON")
    print(f"  Status    : {regime}")
    return regime


# ─────────────────────────────────────────────
# LIQUIDITY FILTER
# Removes stocks that are too illiquid or too cheap to trade reliably.
# Both filters are currently disabled (MIN_PRICE=0, MIN_AVG_VALUE_CR=0)
# but can be enabled in config as capital scales up.
# ─────────────────────────────────────────────
def apply_liquidity_filter(close: pd.DataFrame, volume: pd.DataFrame,
                            tickers: list) -> list:
    passed = []
    for t in tickers:
        if t not in close.columns:
            continue     # skip tickers not in cache (recently listed, data gap)
        try:
            c = close[t].dropna()
            v = volume[t].dropna() if t in volume.columns else pd.Series()

            # Require at least 60 days of price history and price above minimum
            if len(c) < 60 or c.iloc[-1] < cfg.MIN_PRICE:
                continue

            # Check 60-day average traded value in crores (price × volume / 1e7)
            if len(v) >= 60:
                traded_val_cr = (c * v).rolling(60).mean().iloc[-1] / 1e7
                if traded_val_cr < cfg.MIN_AVG_VALUE_CR:
                    continue

            passed.append(t)
        except Exception:
            continue    # silently skip any stock with malformed data
    print(f"\nLIQUIDITY FILTER: {len(passed)}/{len(tickers)} passed")
    return passed


# ─────────────────────────────────────────────
# COMPUTE SCORES
# Core of the momentum strategy. Scores every stock using a weighted
# combination of z-scored momentum and volatility factors.
# Formula: score = W12*z(mom_12m) + W6*z(mom_6m) + W3*z(mom_3m) + WV*z(vol_6m)
# ─────────────────────────────────────────────
def compute_scores(close: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Scores each stock on momentum (12M, 6M, 3M) and optionally volatility.
    All factors are z-scored cross-sectionally before weighting so that
    only relative rankings matter, not absolute return levels.
    Returns a DataFrame sorted best→worst, with score and raw factors.
    """
    records = []
    for t in tickers:
        if t not in close.columns:
            continue
        try:
            prices = close[t].dropna()

            # Need enough history to compute the longest lookback after skipping recent days
            if len(prices) < cfg.LOOKBACK_12M + cfg.SKIP_RECENT:
                continue

            s     = cfg.SKIP_RECENT    # number of recent days to skip (avoids short-term reversal)

            # Price at "now" = SKIP_RECENT days ago (not today) to avoid reversal effect
            p_now = prices.iloc[-(s+1)]

            # Prices at each lookback horizon (also shifted back by SKIP_RECENT)
            p_12m = prices.iloc[-(cfg.LOOKBACK_12M+s)]
            p_6m  = prices.iloc[-(cfg.LOOKBACK_6M+s)]
            p_3m  = prices.iloc[-(cfg.LOOKBACK_3M+s)]

            # Momentum = simple return from lookback price to "now" price
            mom_12m = (p_now - p_12m) / p_12m
            mom_6m  = (p_now - p_6m)  / p_6m
            mom_3m  = (p_now - p_3m)  / p_3m

            # Annualised 6-month realised volatility (std of daily returns × √252)
            vol_6m = prices.iloc[-cfg.LOOKBACK_6M:].pct_change().dropna().std() * np.sqrt(252)

            # Distance above the 250-DMA (trend extension) — mirrors the backtester
            dma_d    = prices.rolling(cfg.DMA_EXIT).mean().iloc[-1] if len(prices) >= cfg.DMA_EXIT else np.nan
            dist_dma = (prices.iloc[-1] / dma_d - 1) if (dma_d and not np.isnan(dma_d)) else 0.0

            records.append({
                "ticker" : t,
                "sector" : UNIVERSE.get(t, "Unknown"),
                "price"  : prices.iloc[-1],     # today's actual close (for display)
                "mom_12m": mom_12m,
                "mom_6m" : mom_6m,
                "mom_3m" : mom_3m,
                "vol_6m" : vol_6m,
                "dist_dma": dist_dma,
            })
        except Exception:
            continue    # skip any stock with insufficient or malformed data

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("ticker")

    # Cross-sectional z-score: subtract universe mean, divide by universe std
    # This makes all factors comparable regardless of their raw scale
    def z(s): return (s - s.mean()) / s.std() if s.std() > 0 else s * 0

    df["z_12m"]  = z(df["mom_12m"])
    df["z_6m"]   = z(df["mom_6m"])
    df["z_3m"]   = z(df["mom_3m"])
    df["z_vol"]  = z(df["vol_6m"])
    df["z_dist"] = z(df["dist_dma"])

    # Composite score: weighted sum of z-scored factors
    df["score"] = (cfg.W_MOM_12M * df["z_12m"] +
                   cfg.W_MOM_6M  * df["z_6m"]  +
                   cfg.W_MOM_3M  * df["z_3m"]  +
                   cfg.W_VOL     * df["z_vol"] +
                   getattr(cfg, "W_DIST_DMA", 0.0) * df["z_dist"])

    return df.sort_values("score", ascending=False)    # rank best to worst


# ─────────────────────────────────────────────
# PORTFOLIO SELECTION
# Picks the top TOP_N stocks from the ranked score list,
# subject to the sector concentration cap (MAX_PER_SECTOR).
# Equal-weighted — each selected stock gets 1/N of capital.
# ─────────────────────────────────────────────
def _entry_above_dma(close, ticker) -> bool:
    """True if the stock is at/above its DMA_EXIT moving average. This is the
    ENTRY gate — it matches the backtester's pick_portfolio so the live selector
    and the backtest simulate the same strategy (don't initiate downtrending names)."""
    if close is None or ticker not in close.columns:
        return True
    p = close[ticker].dropna()
    if len(p) < cfg.DMA_EXIT:
        return True
    return p.iloc[-1] >= p.rolling(cfg.DMA_EXIT).mean().iloc[-1]


def select_portfolio(scored: pd.DataFrame, close: pd.DataFrame = None) -> pd.DataFrame:
    selected, sc = [], {}     # selected = chosen tickers, sc = sector counts
    for ticker, row in scored.iterrows():
        # Entry gate: skip names below their DMA_EXIT (matches the backtester)
        if not _entry_above_dma(close, ticker):
            continue
        s = row["sector"]
        # Only include this stock if its sector isn't already at the cap
        if sc.get(s, 0) < cfg.MAX_PER_SECTOR:
            selected.append(ticker)
            sc[s] = sc.get(s, 0) + 1
        if len(selected) == cfg.TOP_N:
            break     # reached the target portfolio size
    if not selected:
        return pd.DataFrame()
    port = scored.loc[selected].copy()
    port["weight"] = 1.0 / len(port)    # equal weight for each selected stock
    return port


# ─────────────────────────────────────────────
# EXIT SIGNALS
# Checks each currently held stock against two exit rules:
#   1. Rank dropout: stock falls outside the top EXIT_RANK_CUTOFF (default 25)
#   2. DMA breach: stock price falls below its DMA_EXIT-day moving average
# Either rule alone triggers an exit — both are checked every day.
# ─────────────────────────────────────────────
def check_exit_signals(close: pd.DataFrame, scored: pd.DataFrame,
                        current_holdings: list) -> list:
    exits = []
    # The top EXIT_RANK_CUTOFF stocks by score — held stocks must stay within this group
    top_n = scored.head(cfg.EXIT_RANK_CUTOFF).index.tolist()

    for t in current_holdings:
        reason = None

        # Rule 1: rank dropout — stock fell out of the top EXIT_RANK_CUTOFF by momentum score
        if t not in top_n:
            reason = f"dropped out of top {cfg.EXIT_RANK_CUTOFF}"

        # Rule 2: DMA breach — stock's price has dropped below its long-term trend line
        if t in close.columns:
            try:
                p        = close[t].dropna()
                dma_exit = p.rolling(cfg.DMA_EXIT).mean().iloc[-1]   # e.g. 250-day moving average
                # Exit only when >DMA_EXIT_BUFFER below the line (matches the backtester).
                if p.iloc[-1] < dma_exit * (1 - getattr(cfg, "DMA_EXIT_BUFFER", 0.0)):
                    reason = f"below {cfg.DMA_EXIT}-DMA ({dma_exit:.0f})"
            except Exception:
                pass    # if DMA can't be computed (insufficient history), don't exit

        if reason:
            exits.append(t)
            print(f"  EXIT → {t}: {reason}")

    return exits


# ─────────────────────────────────────────────
# MAIN — reads from cache, zero network calls
# ─────────────────────────────────────────────

def regime_strength(nifty500, nifty100, nifty_midcap) -> float:
    """
    Alternative regime mode (only used when cfg.REGIME_WEIGHTED = True).
    Instead of a binary RISK-ON/OFF switch, computes a continuous 0.0–1.0 strength score
    from three indices (Nifty 500, Nifty 100, Nifty Midcap) weighted by config values.
    The strength score is then used to scale position count and capital deployed.
    Returns 0.0 (all cash) to 1.0 (fully invested).
    """
    def margin_above_dma(series, dma=200):
        """Computes how far above/below the DMA the index is, as a percentage."""
        d = series.dropna()
        if len(d) < dma:
            return 0.0    # insufficient history — treat as neutral
        latest = float(d.iloc[-1])
        avg    = float(d.rolling(dma).mean().iloc[-1])
        return (latest - avg) / avg    # positive = above DMA, negative = below

    m500 = margin_above_dma(nifty500)
    m100 = margin_above_dma(nifty100)
    mmid = margin_above_dma(nifty_midcap)

    # Weighted composite of all three indices
    composite = (cfg.REGIME_WEIGHT_NIFTY500 * m500 +
                 cfg.REGIME_WEIGHT_NIFTY100  * m100 +
                 cfg.REGIME_WEIGHT_MIDCAP    * mmid)

    # Map composite to 0–1 using the configured thresholds (linear interpolation)
    fraction = (composite - cfg.REGIME_DEPLOY_MIN) / (cfg.REGIME_DEPLOY_MAX - cfg.REGIME_DEPLOY_MIN)
    fraction = max(0.0, min(1.0, fraction))    # clamp to [0, 1]

    print(f"  Nifty 500  margin: {m500*100:>+.2f}%")
    print(f"  Nifty 100  margin: {m100*100:>+.2f}%")
    print(f"  Midcap     margin: {mmid*100:>+.2f}%")
    print(f"  Composite  margin: {composite*100:>+.2f}%")
    print(f"  Deploy fraction  : {fraction*100:.0f}%")

    return fraction




def run_signals(current_holdings: list = []) -> dict:
    """
    Main signal pipeline — called by execution.py each day.
    Reads entirely from the local CSV cache (no network calls).
    Returns a dict with:
      regime    : 'RISK-ON' | 'RISK-OFF' | 'HALTED'
      portfolio : DataFrame of top N scored stocks with weights
      exits     : list of currently held stocks that triggered exit rules
      strength  : float 0.0–1.0 (regime strength, 1.0 in binary mode)
    """
    # Safety check — abort immediately if kill switch is set in config
    if getattr(cfg, 'TRADING_HALTED', False):
        print("TRADING_HALTED — signals suppressed")
        return {'regime': 'HALTED', 'portfolio': pd.DataFrame(), 'exits': [], 'strength': 0.0}

    # Load price/volume cache and index data from disk
    close, volume = load_for_signals()
    nifty500, nifty50, nifty100, nifty_mid = load_index_data()

    # ── FAIL-SAFE data guard (both regime modes) ─────────────────────────────
    # Not enough regime history ⇒ we cannot judge the market. Abort the run and
    # hold positions unchanged. This is a DATA problem, NOT a market signal, so it
    # must bypass the FORCE_RISK_ON override below (which would wrongly force ON).
    if len(nifty500.dropna()) < cfg.REGIME_DMA:
        print(f"REGIME data insufficient "
              f"({len(nifty500.dropna())} < {cfg.REGIME_DMA} days) — aborting run.")
        return {'regime': 'UNKNOWN', 'portfolio': pd.DataFrame(), 'exits': [], 'strength': 0.0}

    strength = 1.0   # default to fully deployed (overridden in REGIME_WEIGHTED mode)

    # Determine market regime — binary (RISK-ON/OFF) or weighted (0.0–1.0 strength)
    if cfg.REGIME_WEIGHTED:
        strength = regime_strength(nifty500, nifty100, nifty_mid)
        regime   = 'RISK-ON' if strength > 0 else 'RISK-OFF'
    else:
        regime = get_regime(nifty500)     # binary: Nifty 500 vs 200 DMA

    # Data problem (NaN in the index series) → abort, hold positions. Must come
    # before the FORCE_RISK_ON override so a glitch can't be forced to RISK-ON.
    if regime == 'UNKNOWN':
        print('REGIME UNKNOWN → aborting run (data issue); holdings unchanged.')
        return {'regime': 'UNKNOWN', 'portfolio': pd.DataFrame(), 'exits': [], 'strength': 0.0}

    # Handle RISK-OFF — either exit all or override for paper testing
    if regime == 'RISK-OFF':
        print('\nRISK-OFF → Hold cash. Sell all holdings.')
        if cfg.FORCE_RISK_ON:
            # Paper testing override: pretend market is always RISK-ON
            regime   = 'RISK-ON'
            strength = 1.0
        else:
            # Real RISK-OFF: return all current holdings as exits (sell everything)
            return {'regime': regime, 'portfolio': pd.DataFrame(),
                    'exits': current_holdings, 'strength': 0.0}

    # Apply liquidity filter to narrow the universe before scoring
    tickers = list(UNIVERSE.keys())
    liquid  = apply_liquidity_filter(close, volume, tickers)

    print(f"\nComputing scores for {len(liquid)} stocks...")
    scored  = compute_scores(close, liquid)

    if scored.empty:
        print("No stocks passed scoring. Check cache data.")
        return {"regime": regime, "portfolio": pd.DataFrame(), "exits": [], "strength": 1.0}

    # Select top N portfolio and check which held stocks should exit
    portfolio = select_portfolio(scored, close)
    exits     = check_exit_signals(close, scored, current_holdings)

    return {"regime": regime, "portfolio": portfolio, "exits": exits, "strength": strength}


# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("  AGGRESSIVE MOMENTUM SIGNAL ENGINE")
    print(f"  Run date : {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Stocks   : {cfg.TOP_N} | Sector cap : {cfg.MAX_PER_SECTOR} | Exit rank : {cfg.EXIT_RANK_CUTOFF}")
    print("="*55)

    result = run_signals([])

    if not result["portfolio"].empty:
        print(f"\n{'='*55}")
        print(f"  TOP {cfg.TOP_N} PORTFOLIO")
        print("="*55)
        cols = ["sector","price","mom_12m","mom_6m","mom_3m","score","weight"]
        d    = result["portfolio"][cols].copy()
        for col in ["mom_12m","mom_6m","mom_3m"]:
            d[col] = (d[col]*100).round(1).astype(str)+"%"
        d["price"]  = d["price"].round(1)
        d["score"]  = d["score"].round(3)
        d["weight"] = (d["weight"]*100).round(1).astype(str)+"%"
        print(d.to_string())

    if result["exits"]:
        print(f"\n  SELL: {', '.join(result['exits'])}")
    print("\nRun every evening after 3:30 PM IST.")

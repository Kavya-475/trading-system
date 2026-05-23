"""
signals.py
==========
Core signal engine for the Statistical Quality + Momentum strategy.
Universe : Nifty 200 (approximated with major constituents)
Regime   : Nifty 500 close vs 200-DMA  (Risk-On / Risk-Off)
Score    : 0.45 * z(12M mom) + 0.35 * z(6M mom) - 0.20 * z(6M vol)
           Momentum excludes the most recent month (t-20 days)
Output   : Top 10 ranked stocks with max 2 per sector

Run:
    pip install yfinance pandas numpy
    python signals.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STRATEGY PARAMETERS  (edit these freely)
# ─────────────────────────────────────────────
TOP_N               = 10       # number of stocks to hold
MAX_PER_SECTOR      = 2        # sector concentration cap
MIN_PRICE           = 100      # ₹ minimum closing price filter
MIN_AVG_VALUE_CR    = 20       # crore — 60-day avg traded value filter
LOOKBACK_12M_DAYS   = 252      # trading days in 12 months
LOOKBACK_6M_DAYS    = 126      # trading days in 6 months
SKIP_RECENT_DAYS    = 20       # exclude last ~1 month for momentum (avoids reversal)
REGIME_DMA          = 200      # DMA period for regime filter
DATA_YEARS          = 2        # years of historical data to fetch

# ─────────────────────────────────────────────
# NIFTY 200 UNIVERSE
# NSE tickers in yfinance format (append .NS)
# Source: NSE India — update this list semi-annually
# ─────────────────────────────────────────────
UNIVERSE = {
    # ticker        : sector
    "RELIANCE"      : "Energy",
    "ONGC"          : "Energy",
    "BPCL"          : "Energy",
    "GAIL"          : "Energy",
    "TATAPOWER"     : "Energy",
    "ADANIGREEN"    : "Energy",
    "NTPC"          : "Energy",
    "POWERGRID"     : "Energy",
    "NHPC"          : "Energy",

    "TCS"           : "IT",
    "INFOSYS"       : "IT",
    "HCLTECH"       : "IT",
    "WIPRO"         : "IT",
    "TECHM"         : "IT",
    "LTIM"          : "IT",
    "MPHASIS"       : "IT",
    "PERSISTENT"    : "IT",
    "COFORGE"       : "IT",
    "KPIT"          : "IT",

    "HDFCBANK"      : "Banking",
    "ICICIBANK"     : "Banking",
    "SBIN"          : "Banking",
    "KOTAKBANK"     : "Banking",
    "AXISBANK"      : "Banking",
    "INDUSINDBK"    : "Banking",
    "BANDHANBNK"    : "Banking",
    "FEDERALBNK"    : "Banking",
    "IDFCFIRSTB"    : "Banking",
    "PNB"           : "Banking",
    "BANKBARODA"    : "Banking",

    "BAJFINANCE"    : "NBFC",
    "BAJAJFINSV"    : "NBFC",
    "CHOLAFIN"      : "NBFC",
    "MUTHOOTFIN"    : "NBFC",
    "MANAPPURAM"    : "NBFC",
    "LICHSGFIN"     : "NBFC",

    "SBILIFE"       : "Insurance",
    "HDFCLIFE"      : "Insurance",
    "ICICIlombard"  : "Insurance",
    "LICI"          : "Insurance",
    "STARHEALTH"    : "Insurance",

    "SUNPHARMA"     : "Pharma",
    "DRREDDY"       : "Pharma",
    "CIPLA"         : "Pharma",
    "DIVISLAB"      : "Pharma",
    "LUPIN"         : "Pharma",
    "TORNTPHARM"    : "Pharma",
    "AUROPHARMA"    : "Pharma",
    "ALKEM"         : "Pharma",
    "ZYDUSLIFE"     : "Pharma",
    "BIOCON"        : "Pharma",

    "HINDUNILVR"    : "FMCG",
    "ITC"           : "FMCG",
    "NESTLEIND"     : "FMCG",
    "BRITANNIA"     : "FMCG",
    "DABUR"         : "FMCG",
    "MARICO"        : "FMCG",
    "COLPAL"        : "FMCG",
    "GODREJCP"      : "FMCG",
    "TATACONSUM"    : "FMCG",

    "MARUTI"        : "Auto",
    "TATAMOTORS"    : "Auto",
    "EICHERMOT"     : "Auto",
    "BAJAJ-AUTO"    : "Auto",
    "HEROMOTOCO"    : "Auto",
    "ASHOKLEY"      : "Auto",
    "MRF"           : "Auto",
    "BALKRISIND"    : "Auto",
    "MOTHERSON"     : "Auto",

    "LT"            : "Capital Goods",
    "SIEMENS"       : "Capital Goods",
    "ABB"           : "Capital Goods",
    "HAVELLS"       : "Capital Goods",
    "BEL"           : "Capital Goods",
    "HAL"           : "Capital Goods",
    "BHEL"          : "Capital Goods",
    "BHARATFORG"    : "Capital Goods",
    "TIINDIA"       : "Capital Goods",

    "ASIANPAINT"    : "Paints",
    "BERGEPAINT"    : "Paints",
    "PIDILITIND"    : "Chemicals",
    "DEEPAKNTR"     : "Chemicals",
    "SRF"           : "Chemicals",
    "AARTI"         : "Chemicals",
    "NAVINFLUOR"    : "Chemicals",

    "TITAN"         : "Consumer",
    "TRENT"         : "Consumer",
    "DMART"         : "Consumer",
    "PAGEIND"       : "Consumer",
    "JUBLFOOD"      : "Consumer",
    "IRCTC"         : "Consumer",

    "ULTRACEMCO"    : "Cement",
    "SHREECEM"      : "Cement",
    "AMBUJACEM"     : "Cement",
    "GRASIM"        : "Cement",

    "JSWSTEEL"      : "Metals",
    "TATASTEEL"     : "Metals",
    "HINDALCO"      : "Metals",
    "VEDL"          : "Metals",
    "SAIL"          : "Metals",
    "COALINDIA"     : "Metals",

    "BHARTIARTL"    : "Telecom",
    "TATACOMM"      : "Telecom",

    "DLF"           : "Realty",
    "GODREJPROP"    : "Realty",
    "OBEROIRLTY"    : "Realty",
    "PRESTIGE"      : "Realty",
    "LODHA"         : "Realty",

    "APOLLOHOSP"    : "Healthcare",
    "MAXHEALTH"     : "Healthcare",
    "FORTIS"        : "Healthcare",

    "ADANIPORTS"    : "Infrastructure",
    "CONCOR"        : "Infrastructure",
    "ADANIENT"      : "Infrastructure",

    "ZOMATO"        : "New Age",
    "NAUKRI"        : "New Age",
    "INDIAMART"     : "New Age",

    "CDSL"          : "Financials",
    "BSE"           : "Financials",
    "MCX"           : "Financials",
    "ANGELONE"      : "Financials",
    "CAMS"          : "Financials",
    "360ONE"        : "Financials",
}

REGIME_TICKER = "^CRSLDX"   # Nifty 500 index on yfinance


# ─────────────────────────────────────────────
# STEP 1: FETCH DATA
# ─────────────────────────────────────────────
def fetch_data(tickers: list, years: int = DATA_YEARS) -> pd.DataFrame:
    """
    Downloads OHLCV data for all tickers using yfinance.
    Returns a DataFrame with MultiIndex columns (OHLCV, ticker).
    """
    end   = datetime.today()
    start = end - timedelta(days=years * 365 + 60)   # extra buffer for DMA

    yf_tickers = [t + ".NS" for t in tickers]
    print(f"\nFetching data for {len(yf_tickers)} stocks...")

    raw = yf.download(
        yf_tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    print("Data fetched successfully.")
    return raw


def fetch_regime_data(years: int = DATA_YEARS) -> pd.Series:
    """Fetches Nifty 500 closing prices for regime filter."""
    end   = datetime.today()
    start = end - timedelta(days=years * 365 + 60)
    data  = yf.download(
        REGIME_TICKER,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    return data["Close"].squeeze()


# ─────────────────────────────────────────────
# STEP 2: REGIME FILTER
# ─────────────────────────────────────────────
def get_regime(nifty500_close: pd.Series) -> str:
    """
    Risk-On  : Nifty 500 > 200-DMA  → proceed with signals
    Risk-Off : Nifty 500 < 200-DMA  → liquidate, hold cash
    """
    dma_200    = nifty500_close.rolling(REGIME_DMA).mean()
    latest     = nifty500_close.iloc[-1]
    latest_dma = dma_200.iloc[-1]

    regime = "RISK-ON" if latest > latest_dma else "RISK-OFF"
    print(f"\nREGIME CHECK")
    print(f"  Nifty 500  : {latest:,.2f}")
    print(f"  200 DMA    : {latest_dma:,.2f}")
    print(f"  Status     : {regime}")
    return regime


# ─────────────────────────────────────────────
# STEP 3: LIQUIDITY FILTER
# ─────────────────────────────────────────────
def apply_liquidity_filter(raw: pd.DataFrame, tickers: list) -> list:
    """
    Keeps only stocks where:
      - Latest close > ₹MIN_PRICE
      - 60-day average traded value > ₹MIN_AVG_VALUE_CR crore
    """
    passed = []

    for ticker in tickers:
        yf_ticker = ticker + ".NS"
        try:
            close  = raw["Close"][yf_ticker].dropna()
            volume = raw["Volume"][yf_ticker].dropna()

            if len(close) < 60:
                continue

            latest_price = close.iloc[-1]
            if latest_price < MIN_PRICE:
                continue

            # Traded value = Close * Volume (proxy for turnover)
            traded_value  = (close * volume).rolling(60).mean().iloc[-1]
            traded_value_cr = traded_value / 1e7   # convert to crore

            if traded_value_cr >= MIN_AVG_VALUE_CR:
                passed.append(ticker)

        except Exception:
            continue

    print(f"\nLIQUIDITY FILTER: {len(passed)}/{len(tickers)} stocks passed")
    return passed


# ─────────────────────────────────────────────
# STEP 4: COMPUTE MOMENTUM & VOLATILITY SCORES
# ─────────────────────────────────────────────
def compute_scores(raw: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    For each stock:
      mom_12m = return from -252 days to -20 days
      mom_6m  = return from -126 days to -20 days
      vol_6m  = std of daily returns over last 126 days (annualised)

    Then z-score each factor across the universe.
    Final score = 0.45*z_12m + 0.35*z_6m - 0.20*z_vol
    """
    records = []

    for ticker in tickers:
        yf_ticker = ticker + ".NS"
        try:
            close = raw["Close"][yf_ticker].dropna()

            if len(close) < LOOKBACK_12M_DAYS + SKIP_RECENT_DAYS:
                continue

            # Momentum: return excluding the most recent month
            price_now       = close.iloc[-(SKIP_RECENT_DAYS + 1)]
            price_12m_ago   = close.iloc[-(LOOKBACK_12M_DAYS + SKIP_RECENT_DAYS)]
            price_6m_ago    = close.iloc[-(LOOKBACK_6M_DAYS  + SKIP_RECENT_DAYS)]

            mom_12m = (price_now - price_12m_ago) / price_12m_ago
            mom_6m  = (price_now - price_6m_ago)  / price_6m_ago

            # Volatility: annualised std of daily returns over last 6 months
            recent_close = close.iloc[-LOOKBACK_6M_DAYS:]
            daily_ret    = recent_close.pct_change().dropna()
            vol_6m       = daily_ret.std() * np.sqrt(252)

            records.append({
                "ticker"  : ticker,
                "sector"  : UNIVERSE.get(ticker, "Unknown"),
                "price"   : close.iloc[-1],
                "mom_12m" : mom_12m,
                "mom_6m"  : mom_6m,
                "vol_6m"  : vol_6m,
            })

        except Exception:
            continue

    df = pd.DataFrame(records).set_index("ticker")

    if df.empty:
        print("ERROR: No valid stocks found. Check data.")
        return df

    # Z-score each factor (mean=0, std=1 across universe)
    def zscore(series):
        return (series - series.mean()) / series.std()

    df["z_12m"] = zscore(df["mom_12m"])
    df["z_6m"]  = zscore(df["mom_6m"])
    df["z_vol"] = zscore(df["vol_6m"])

    # Blended score
    df["score"] = (
        0.45 * df["z_12m"] +
        0.35 * df["z_6m"]  -
        0.20 * df["z_vol"]
    )

    return df.sort_values("score", ascending=False)


# ─────────────────────────────────────────────
# STEP 5: SECTOR CAP & FINAL PORTFOLIO
# ─────────────────────────────────────────────
def select_portfolio(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Picks top N stocks with max MAX_PER_SECTOR per sector.
    Equal weight across all selected stocks.
    """
    selected   = []
    sector_count = {}

    for ticker, row in scored.iterrows():
        sector = row["sector"]
        count  = sector_count.get(sector, 0)

        if count < MAX_PER_SECTOR:
            selected.append(ticker)
            sector_count[sector] = count + 1

        if len(selected) == TOP_N:
            break

    portfolio = scored.loc[selected].copy()
    portfolio["weight"] = 1.0 / len(portfolio)
    return portfolio


# ─────────────────────────────────────────────
# STEP 6: CHECK 100-DMA EXIT SIGNAL
# ─────────────────────────────────────────────
def check_exit_signals(raw: pd.DataFrame, current_holdings: list) -> list:
    """
    Returns list of tickers to EXIT because price < 100-DMA.
    Pass in your current holdings list.
    If empty list passed, skips this check.
    """
    exit_list = []

    for ticker in current_holdings:
        yf_ticker = ticker + ".NS"
        try:
            close    = raw["Close"][yf_ticker].dropna()
            dma_100  = close.rolling(100).mean().iloc[-1]
            latest   = close.iloc[-1]

            if latest < dma_100:
                exit_list.append(ticker)
                print(f"  EXIT SIGNAL → {ticker}: price {latest:.1f} < 100-DMA {dma_100:.1f}")

        except Exception:
            continue

    return exit_list


# ─────────────────────────────────────────────
# MAIN: RUN SIGNAL GENERATION
# ─────────────────────────────────────────────
def run_signals(current_holdings: list = []) -> dict:
    """
    Full signal pipeline. Returns a dict with:
      - regime     : "RISK-ON" or "RISK-OFF"
      - portfolio  : DataFrame of top 10 stocks to hold
      - exits      : list of current holdings to sell
    """
    tickers = list(UNIVERSE.keys())

    # Fetch all data
    raw        = fetch_data(tickers)
    nifty500   = fetch_regime_data()

    # Regime check first — if Risk-Off, skip everything
    regime = get_regime(nifty500)
    if regime == "RISK-OFF":
        print("\nMARKET IS RISK-OFF. Hold cash. No new positions.")
        exits = current_holdings   # sell everything
        return {"regime": regime, "portfolio": pd.DataFrame(), "exits": exits}

    # Liquidity filter
    liquid_tickers = apply_liquidity_filter(raw, tickers)

    # Score remaining stocks
    print("\nComputing momentum and volatility scores...")
    scored = compute_scores(raw, liquid_tickers)

    # Select top 10 with sector cap
    portfolio = select_portfolio(scored)

    # Check exit signals for current holdings
    exits = check_exit_signals(raw, current_holdings)

    # Also exit if a holding dropped out of top 20
    top_20 = scored.head(20).index.tolist()
    for h in current_holdings:
        if h not in top_20 and h not in exits:
            exits.append(h)
            print(f"  EXIT SIGNAL → {h}: dropped out of top 20")

    return {"regime": regime, "portfolio": portfolio, "exits": exits}


# ─────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  STATISTICAL QUALITY + MOMENTUM SIGNAL ENGINE")
    print(f"  Run date: {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 55)

    # Pass your current holdings here on subsequent runs
    # Example: current_holdings = ["RELIANCE", "TCS", "HDFCBANK"]
    current_holdings = []

    result = run_signals(current_holdings)

    if not result["portfolio"].empty:
        print("\n" + "=" * 55)
        print("  TOP 10 PORTFOLIO — BUY / HOLD TOMORROW")
        print("=" * 55)
        display_cols = ["sector", "price", "mom_12m", "mom_6m", "vol_6m", "score", "weight"]
        display      = result["portfolio"][display_cols].copy()
        display["mom_12m"] = (display["mom_12m"] * 100).round(1).astype(str) + "%"
        display["mom_6m"]  = (display["mom_6m"]  * 100).round(1).astype(str) + "%"
        display["vol_6m"]  = (display["vol_6m"]  * 100).round(1).astype(str) + "%"
        display["price"]   = display["price"].round(1)
        display["score"]   = display["score"].round(3)
        display["weight"]  = (display["weight"] * 100).round(1).astype(str) + "%"
        print(display.to_string())

    if result["exits"]:
        print(f"\n  SELL TOMORROW: {', '.join(result['exits'])}")

    print("\nDone. Run this every evening after 3:30 PM IST.")
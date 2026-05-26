"""
signals.py  [AGGRESSIVE CONFIG — cache-based]
==============================================
Reads price data from cache instead of fetching from Yahoo Finance.
Run data_manager.py first to ensure cache is up to date.

Daily workflow:
    python data_manager.py   ← updates cache (run once at 3:40 PM)
    python execution.py      ← reads cache, runs signals, places orders

Run standalone:
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
# ─────────────────────────────────────────────
def get_regime(nifty500: pd.Series) -> str:
    clean  = nifty500.dropna()
    dma    = clean.rolling(cfg.REGIME_DMA).mean()
    latest = clean.iloc[-1]
    l_dma  = dma.iloc[-1]

    # Confirmation filter — require N consecutive days above DMA for RISK-ON
    # RISK-OFF is immediate (asymmetric — protect capital faster)
    confirm_days = getattr(cfg, "REGIME_CONFIRM_DAYS", 0)
    if confirm_days > 0 and len(clean) >= confirm_days:
        recent        = clean.iloc[-confirm_days:]
        recent_dma    = dma.iloc[-confirm_days:]
        all_above     = all(p > d for p, d in zip(recent, recent_dma))
        regime        = "RISK-ON" if all_above else "RISK-OFF"
    else:
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
# Uses cache data — no network call
# ─────────────────────────────────────────────
def apply_liquidity_filter(close: pd.DataFrame, volume: pd.DataFrame,
                            tickers: list) -> list:
    passed = []
    for t in tickers:
        if t not in close.columns:
            continue
        try:
            c = close[t].dropna()
            v = volume[t].dropna() if t in volume.columns else pd.Series()
            if len(c) < 60 or c.iloc[-1] < cfg.MIN_PRICE:
                continue
            if len(v) >= 60:
                traded_val_cr = (c * v).rolling(60).mean().iloc[-1] / 1e7
                if traded_val_cr < cfg.MIN_AVG_VALUE_CR:
                    continue
            passed.append(t)
        except Exception:
            continue
    print(f"\nLIQUIDITY FILTER: {len(passed)}/{len(tickers)} passed")
    return passed


# ─────────────────────────────────────────────
# COMPUTE SCORES
# ─────────────────────────────────────────────
def compute_scores(close: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Scores each stock on momentum (12M, 6M, 3M) and volatility.
    Reads entirely from cache — zero network calls.
    """
    records = []
    for t in tickers:
        if t not in close.columns:
            continue
        try:
            prices = close[t].dropna()
            if len(prices) < cfg.LOOKBACK_12M + cfg.SKIP_RECENT:
                continue

            s        = cfg.SKIP_RECENT
            p_now    = prices.iloc[-(s+1)]
            p_12m    = prices.iloc[-(cfg.LOOKBACK_12M+s)]
            p_6m     = prices.iloc[-(cfg.LOOKBACK_6M+s)]
            p_3m     = prices.iloc[-(cfg.LOOKBACK_3M+s)]

            mom_12m  = (p_now-p_12m)/p_12m
            mom_6m   = (p_now-p_6m)/p_6m
            mom_3m   = (p_now-p_3m)/p_3m
            vol_6m   = prices.iloc[-cfg.LOOKBACK_6M:].pct_change().dropna().std()*np.sqrt(252)

            records.append({
                "ticker" :t, "sector":UNIVERSE.get(t,"Unknown"),
                "price"  :prices.iloc[-1],
                "mom_12m":mom_12m,"mom_6m":mom_6m,"mom_3m":mom_3m,"vol_6m":vol_6m,
            })
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("ticker")

    def z(s): return (s-s.mean())/s.std() if s.std()>0 else s*0
    df["z_12m"] = z(df["mom_12m"])
    df["z_6m"]  = z(df["mom_6m"])
    df["z_3m"]  = z(df["mom_3m"])
    df["z_vol"] = z(df["vol_6m"])
    df["score"] = (cfg.W_MOM_12M*df["z_12m"] + cfg.W_MOM_6M*df["z_6m"] +
                   cfg.W_MOM_3M*df["z_3m"]  + cfg.W_VOL*df["z_vol"])
    return df.sort_values("score", ascending=False)


# ─────────────────────────────────────────────
# PORTFOLIO SELECTION
# ─────────────────────────────────────────────
def select_portfolio(scored: pd.DataFrame) -> pd.DataFrame:
    selected, sc = [], {}
    for ticker, row in scored.iterrows():
        s = row["sector"]
        if sc.get(s,0) < cfg.MAX_PER_SECTOR:
            selected.append(ticker)
            sc[s] = sc.get(s,0)+1
        if len(selected) == cfg.TOP_N:
            break
    port = scored.loc[selected].copy()
    port["weight"] = 1.0/len(port)
    return port


# ─────────────────────────────────────────────
# EXIT SIGNALS
# ─────────────────────────────────────────────
def check_exit_signals(close: pd.DataFrame, scored: pd.DataFrame,
                        current_holdings: list) -> list:
    exits  = []
    top_n  = scored.head(cfg.EXIT_RANK_CUTOFF).index.tolist()
    for t in current_holdings:
        reason = None
        if t not in top_n:
            reason = f"dropped out of top {cfg.EXIT_RANK_CUTOFF}"
        if t in close.columns:
            try:
                p      = close[t].dropna()
                dma100 = p.rolling(cfg.DMA_EXIT).mean().iloc[-1]
                if p.iloc[-1] < dma100:
                    reason = f"below 100-DMA ({dma100:.0f})"
            except Exception:
                pass
        if reason:
            exits.append(t)
            print(f"  EXIT → {t}: {reason}")
    return exits


# ─────────────────────────────────────────────
# MAIN — reads from cache, zero network calls
# ─────────────────────────────────────────────

def regime_strength(nifty500, nifty100, nifty_midcap) -> float:
    """
    Weighted regime strength score.
    Returns 0.0 (full cash) to 1.0 (fully invested).
    Only used when cfg.REGIME_WEIGHTED = True.
    """
    def margin_above_dma(series, dma=200):
        d = series.dropna()
        if len(d) < dma:
            return 0.0
        latest = float(d.iloc[-1])
        avg    = float(d.rolling(dma).mean().iloc[-1])
        return (latest - avg) / avg

    m500  = margin_above_dma(nifty500)
    m100  = margin_above_dma(nifty100)
    mmid  = margin_above_dma(nifty_midcap)

    composite = (cfg.REGIME_WEIGHT_NIFTY500 * m500 +
                 cfg.REGIME_WEIGHT_NIFTY100  * m100 +
                 cfg.REGIME_WEIGHT_MIDCAP    * mmid)

    # Normalise to 0-1
    fraction = (composite - cfg.REGIME_DEPLOY_MIN) / (cfg.REGIME_DEPLOY_MAX - cfg.REGIME_DEPLOY_MIN)
    fraction = max(0.0, min(1.0, fraction))

    print(f"  Nifty 500  margin: {m500*100:>+.2f}%")
    print(f"  Nifty 100  margin: {m100*100:>+.2f}%")
    print(f"  Midcap     margin: {mmid*100:>+.2f}%")
    print(f"  Composite  margin: {composite*100:>+.2f}%")
    print(f"  Deploy fraction  : {fraction*100:.0f}%")

    return fraction




def run_signals(current_holdings: list = []) -> dict:
    """
    Full signal pipeline using cached data.
    No Yahoo Finance fetch — reads from disk only.
    """
    # Load from cache
    close, volume = load_for_signals()
    nifty500, nifty50, nifty100, nifty_mid = load_index_data()

    strength = 1.0  # default — fully deployed

    if cfg.REGIME_WEIGHTED:
        strength = regime_strength(nifty500, nifty100, nifty_mid)
        regime   = 'RISK-ON' if strength > 0 else 'RISK-OFF'
        print(f'  Deploy fraction  : {strength*100:.0f}%')
    else:
        regime = get_regime(nifty500)

    if regime == 'RISK-OFF':
        print('\nRISK-OFF → Hold cash. Sell all holdings.')
        if cfg.FORCE_RISK_ON:
            regime   = 'RISK-ON'
            strength = 1.0
        else:
            return {'regime':regime,'portfolio':pd.DataFrame(),'exits':current_holdings,'strength':0.0}
    tickers  = list(UNIVERSE.keys())
    liquid   = apply_liquidity_filter(close, volume, tickers)
    print(f"\nComputing scores for {len(liquid)} stocks...")
    scored   = compute_scores(close, liquid)

    if scored.empty:
        print("No stocks passed scoring. Check cache data.")
        return {"regime":regime,"portfolio":pd.DataFrame(),"exits":[]}

    portfolio = select_portfolio(scored)
    exits     = check_exit_signals(close, scored, current_holdings)
    return {"regime":regime,"portfolio":portfolio,"exits":exits,"strength":strength}


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

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
    "TCS":"IT","INFY":"IT","HCLTECH":"IT","WIPRO":"IT","TECHM":"IT",
    "LTIM":"IT","MPHASIS":"IT","PERSISTENT":"IT","COFORGE":"IT",
    "KPITTECH":"IT","TATAELXSI":"IT","OFSS":"IT",

    "HDFCBANK":"Banking","ICICIBANK":"Banking","SBIN":"Banking",
    "KOTAKBANK":"Banking","AXISBANK":"Banking","INDUSINDBK":"Banking",
    "BANDHANBNK":"Banking","FEDERALBNK":"Banking","IDFCFIRSTB":"Banking",
    "PNB":"Banking","BANKBARODA":"Banking","CANBK":"Banking",
    "UNIONBANK":"Banking","AUBANK":"Banking","RBLBANK":"Banking",

    "BAJFINANCE":"NBFC","BAJAJFINSV":"NBFC","CHOLAFIN":"NBFC",
    "MUTHOOTFIN":"NBFC","MANAPPURAM":"NBFC","LICHSGFIN":"NBFC",
    "SUNDARMFIN":"NBFC","SHRIRAMFIN":"NBFC",

    "SBILIFE":"Insurance","HDFCLIFE":"Insurance","ICICIGI":"Insurance",
    "LICI":"Insurance","STARHEALTH":"Insurance",

    "SUNPHARMA":"Pharma","DRREDDY":"Pharma","CIPLA":"Pharma",
    "DIVISLAB":"Pharma","LUPIN":"Pharma","TORNTPHARM":"Pharma",
    "AUROPHARMA":"Pharma","ALKEM":"Pharma","ZYDUSLIFE":"Pharma",
    "IPCALAB":"Pharma","ABBOTINDIA":"Pharma",

    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG",
    "BRITANNIA":"FMCG","DABUR":"FMCG","MARICO":"FMCG",
    "COLPAL":"FMCG","GODREJCP":"FMCG","TATACONSUM":"FMCG","VBL":"FMCG","UBL":"FMCG",

    "MARUTI":"Auto","TATAMOTORS":"Auto","EICHERMOT":"Auto",
    "BAJAJ-AUTO":"Auto","HEROMOTOCO":"Auto","ASHOKLEY":"Auto",
    "MRF":"Auto","BALKRISIND":"Auto","MOTHERSON":"Auto",
    "TVSMOTOR":"Auto","M&M":"Auto","BHARATFORG":"Auto","TIINDIA":"Auto",

    "LT":"Capital Goods","SIEMENS":"Capital Goods","ABB":"Capital Goods",
    "HAVELLS":"Capital Goods","BEL":"Capital Goods","HAL":"Capital Goods",
    "BHEL":"Capital Goods","CGPOWER":"Capital Goods","THERMAX":"Capital Goods",
    "CUMMINSIND":"Capital Goods","VOLTAS":"Capital Goods","SUZLON":"Capital Goods",
    "KEC":"Capital Goods",

    "RELIANCE":"Energy","ONGC":"Energy","BPCL":"Energy","GAIL":"Energy",
    "TATAPOWER":"Energy","ADANIGREEN":"Energy","NTPC":"Energy",
    "POWERGRID":"Energy","NHPC":"Energy","TORNTPOWER":"Energy","SJVN":"Energy",

    "PIDILITIND":"Chemicals","DEEPAKNTR":"Chemicals","SRF":"Chemicals",
    "AARTIIND":"Chemicals","NAVINFLUOR":"Chemicals","ATUL":"Chemicals",
    "ALKYLAMINE":"Chemicals",

    "ASIANPAINT":"Paints","BERGEPAINT":"Paints","KANSAINER":"Paints",

    "TITAN":"Consumer","TRENT":"Consumer","DMART":"Consumer",
    "PAGEIND":"Consumer","JUBLFOOD":"Consumer","IRCTC":"Consumer",
    "NAUKRI":"Consumer","ZOMATO":"Consumer","INDIAMART":"Consumer",

    "ULTRACEMCO":"Cement","SHREECEM":"Cement","AMBUJACEM":"Cement",
    "GRASIM":"Cement","JKCEMENT":"Cement",

    "JSWSTEEL":"Metals","TATASTEEL":"Metals","HINDALCO":"Metals",
    "VEDL":"Metals","SAIL":"Metals","COALINDIA":"Metals","APLAPOLLO":"Metals",

    "BHARTIARTL":"Telecom","TATACOMM":"Telecom",

    "DLF":"Realty","GODREJPROP":"Realty","OBEROIRLTY":"Realty",
    "PRESTIGE":"Realty","LODHA":"Realty","PHOENIXLTD":"Realty","BRIGADE":"Realty",

    "APOLLOHOSP":"Healthcare","MAXHEALTH":"Healthcare","FORTIS":"Healthcare",

    "ADANIPORTS":"Infrastructure","ADANIENT":"Infrastructure","IRB":"Infrastructure",

    "CDSL":"Financials","BSE":"Financials","MCX":"Financials",
    "ANGELONE":"Financials","CAMS":"Financials","360ONE":"Financials","HDFCAMC":"Financials",
}


# ─────────────────────────────────────────────
# REGIME FILTER
# ─────────────────────────────────────────────
def get_regime(nifty500: pd.Series) -> str:
    clean  = nifty500.dropna()          # remove NaN gaps before rolling
    dma    = clean.rolling(cfg.REGIME_DMA).mean()
    latest = clean.iloc[-1]
    l_dma  = dma.iloc[-1]
    regime = "RISK-ON" if latest > l_dma else "RISK-OFF"
    print(f"\nREGIME CHECK")
    print(f"  Nifty 500 : {latest:,.2f}")
    print(f"  200 DMA   : {l_dma:,.2f}")
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
def run_signals(current_holdings: list = []) -> dict:
    """
    Full signal pipeline using cached data.
    No Yahoo Finance fetch — reads from disk only.
    """
    # Load from cache
    close, volume = load_for_signals()
    nifty500, _   = load_index_data()

    regime = get_regime(nifty500)

    if regime == "RISK-OFF":
        print("\nRISK-OFF → Hold cash. Sell all holdings.")
        if cfg.FORCE_RISK_ON:
            regime = "RISK-ON"  # Paper test override
        else:
            return {"regime":regime,"portfolio":pd.DataFrame(),"exits":current_holdings}
    tickers  = list(UNIVERSE.keys())
    liquid   = apply_liquidity_filter(close, volume, tickers)
    print(f"\nComputing scores for {len(liquid)} stocks...")
    scored   = compute_scores(close, liquid)

    if scored.empty:
        print("No stocks passed scoring. Check cache data.")
        return {"regime":regime,"portfolio":pd.DataFrame(),"exits":[]}

    portfolio = select_portfolio(scored)
    exits     = check_exit_signals(close, scored, current_holdings)
    return {"regime":regime,"portfolio":portfolio,"exits":exits}


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

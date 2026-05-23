"""
signals.py  [AGGRESSIVE CONFIG]
================================
Core signal engine — imports all parameters from config.py.

Changes from conservative version:
  - TOP_N          : 10 → 7
  - MAX_PER_SECTOR : 2  → 3
  - EXIT_RANK      : 20 → 25
  - Formula adds 3M momentum, halves volatility penalty
  - All parameters imported from config.py

Run:
    python signals.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

import config as cfg

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
    "COLPAL":"FMCG","GODREJCP":"FMCG","TATACONSUM":"FMCG",
    "VBL":"FMCG","UBL":"FMCG",

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


def fetch_data(tickers):
    end   = datetime.today()
    start = end - timedelta(days=2*365+60)
    print(f"\nFetching data for {len(tickers)} stocks...")
    raw = yf.download(
        [t+".NS" for t in tickers],
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True, progress=False
    )
    print("Done.")
    return raw


def fetch_regime_data():
    end   = datetime.today()
    start = end - timedelta(days=2*365+60)
    data  = yf.download(cfg.REGIME_TICKER, start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    return data["Close"].squeeze()


def get_regime(nifty500):
    dma    = nifty500.rolling(cfg.REGIME_DMA).mean()
    latest = nifty500.iloc[-1]
    l_dma  = dma.iloc[-1]
    regime = "RISK-ON" if latest > l_dma else "RISK-OFF"
    print(f"\nREGIME CHECK\n  Nifty 500 : {latest:,.2f}\n  200 DMA   : {l_dma:,.2f}\n  Status    : {regime}")
    return regime


def apply_liquidity_filter(raw, tickers):
    passed = []
    for t in tickers:
        try:
            c = raw["Close"][t+".NS"].dropna()
            v = raw["Volume"][t+".NS"].dropna()
            if len(c) < 60 or c.iloc[-1] < cfg.MIN_PRICE:
                continue
            if (c*v).rolling(60).mean().iloc[-1] / 1e7 >= cfg.MIN_AVG_VALUE_CR:
                passed.append(t)
        except Exception:
            continue
    print(f"\nLIQUIDITY FILTER: {len(passed)}/{len(tickers)} passed")
    return passed


def compute_scores(raw, tickers):
    records = []
    for t in tickers:
        try:
            close = raw["Close"][t+".NS"].dropna()
            if len(close) < cfg.LOOKBACK_12M + cfg.SKIP_RECENT:
                continue
            s = cfg.SKIP_RECENT
            p_now    = close.iloc[-(s+1)]
            p_12m    = close.iloc[-(cfg.LOOKBACK_12M+s)]
            p_6m     = close.iloc[-(cfg.LOOKBACK_6M+s)]
            p_3m     = close.iloc[-(cfg.LOOKBACK_3M+s)]
            mom_12m  = (p_now-p_12m)/p_12m
            mom_6m   = (p_now-p_6m)/p_6m
            mom_3m   = (p_now-p_3m)/p_3m
            vol_6m   = close.iloc[-cfg.LOOKBACK_6M:].pct_change().dropna().std()*np.sqrt(252)
            records.append({"ticker":t,"sector":UNIVERSE.get(t,"Unknown"),
                            "price":close.iloc[-1],"mom_12m":mom_12m,
                            "mom_6m":mom_6m,"mom_3m":mom_3m,"vol_6m":vol_6m})
        except Exception:
            continue

    df = pd.DataFrame(records).set_index("ticker")
    if df.empty:
        return df

    def z(s): return (s-s.mean())/s.std() if s.std()>0 else s*0
    df["z_12m"] = z(df["mom_12m"])
    df["z_6m"]  = z(df["mom_6m"])
    df["z_3m"]  = z(df["mom_3m"])
    df["z_vol"] = z(df["vol_6m"])
    df["score"] = (cfg.W_MOM_12M*df["z_12m"] + cfg.W_MOM_6M*df["z_6m"] +
                   cfg.W_MOM_3M*df["z_3m"]  + cfg.W_VOL*df["z_vol"])
    return df.sort_values("score", ascending=False)


def select_portfolio(scored):
    selected, sector_count = [], {}
    for ticker, row in scored.iterrows():
        s = row["sector"]
        if sector_count.get(s,0) < cfg.MAX_PER_SECTOR:
            selected.append(ticker)
            sector_count[s] = sector_count.get(s,0)+1
        if len(selected) == cfg.TOP_N:
            break
    port = scored.loc[selected].copy()
    port["weight"] = 1.0/len(port)
    return port


def check_exit_signals(raw, scored, current_holdings):
    exits  = []
    top_n  = scored.head(cfg.EXIT_RANK_CUTOFF).index.tolist()
    for t in current_holdings:
        reason = None
        if t not in top_n:
            reason = f"dropped out of top {cfg.EXIT_RANK_CUTOFF}"
        try:
            c      = raw["Close"][t+".NS"].dropna()
            dma100 = c.rolling(cfg.DMA_EXIT).mean().iloc[-1]
            if c.iloc[-1] < dma100:
                reason = f"below 100-DMA ({dma100:.0f})"
        except Exception:
            pass
        if reason:
            exits.append(t)
            print(f"  EXIT → {t}: {reason}")
    return exits


def run_signals(current_holdings=[]):
    tickers  = list(UNIVERSE.keys())
    raw      = fetch_data(tickers)
    nifty500 = fetch_regime_data()
    regime   = get_regime(nifty500)

    if regime == "RISK-OFF":
        print("\nRISK-OFF → Hold cash. Sell all holdings.")
        return {"regime":regime,"portfolio":pd.DataFrame(),"exits":current_holdings}

    liquid    = apply_liquidity_filter(raw, tickers)
    scored    = compute_scores(raw, liquid)
    portfolio = select_portfolio(scored)
    exits     = check_exit_signals(raw, scored, current_holdings)
    return {"regime":regime,"portfolio":portfolio,"exits":exits}


if __name__ == "__main__":
    print("="*55)
    print("  AGGRESSIVE MOMENTUM SIGNAL ENGINE")
    print(f"  Run date : {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Stocks   : {cfg.TOP_N} | Sector cap : {cfg.MAX_PER_SECTOR} | Exit rank : {cfg.EXIT_RANK_CUTOFF}")
    print(f"  Formula  : {cfg.W_MOM_12M}*z12M + {cfg.W_MOM_6M}*z6M + {cfg.W_MOM_3M}*z3M + {cfg.W_VOL}*zVol")
    print("="*55)

    result = run_signals([])

    if not result["portfolio"].empty:
        print(f"\n{'='*55}")
        print(f"  TOP {cfg.TOP_N} PORTFOLIO — BUY / HOLD TOMORROW")
        print("="*55)
        cols = ["sector","price","mom_12m","mom_6m","mom_3m","vol_6m","score","weight"]
        d    = result["portfolio"][cols].copy()
        for col in ["mom_12m","mom_6m","mom_3m","vol_6m"]:
            d[col] = (d[col]*100).round(1).astype(str)+"%"
        d["price"]  = d["price"].round(1)
        d["score"]  = d["score"].round(3)
        d["weight"] = (d["weight"]*100).round(1).astype(str)+"%"
        print(d.to_string())

    if result["exits"]:
        print(f"\n  SELL TOMORROW : {', '.join(result['exits'])}")
    print("\nRun every evening after 3:30 PM IST.")
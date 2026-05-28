"""
backtester.py  [DAILY EXIT VERSION]
=====================================
Walk-forward backtest with daily exit monitoring.

Key improvement over monthly-only version:
  - Regime check runs EVERY trading day
  - 100 DMA exit checked EVERY trading day
  - Rank exit checked EVERY trading day (using last monthly scores)
  - Exit triggers immediate replacement buy — no cash sitting idle
  - Full portfolio rotation on first trading day of each month only
  - Score computation monthly only (performance + realism)

This more accurately reflects live execution.py behaviour.
Expected result vs monthly backtester:
  - Slightly lower CAGR (more transaction costs from faster exits)
  - Lower max drawdown (exits happen days faster)
  - More realistic overall

Run:
    python backtester.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")

import config as cfg

# ── Universe ───────────────────────────────────────────────────────────────
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


# ── Transaction costs ──────────────────────────────────────────────────────
def txn_cost(value, side):
    cost = value * (cfg.EXCHANGE_CHARGE + cfg.SEBI_CHARGE)
    cost += value * cfg.STAMP_DUTY  if side == "buy"  else 0
    cost += value * cfg.STT_SELL    if side == "sell" else 0
    cost += 15.93                   if side == "sell" else 0   # DP charge flat per sell
    return cost


# ── Data loading ───────────────────────────────────────────────────────────
def load_data():
    if os.path.exists(cfg.DATA_CACHE_FILE):
        print("Loading cached data...")
        close     = pd.read_csv(cfg.DATA_CACHE_FILE,  index_col=0, parse_dates=True)
        volume    = pd.read_csv(cfg.VOLUME_CACHE,     index_col=0, parse_dates=True)
        regime_df = pd.read_csv(cfg.REGIME_CACHE,     index_col=0, parse_dates=True)
        print(f"Loaded {close.shape[1]} stocks, {close.shape[0]} days.")
        nifty100  = regime_df["nifty100"]    if "nifty100"    in regime_df.columns else regime_df["nifty500"]
        nifty_mid = regime_df["nifty_midcap"]if "nifty_midcap"in regime_df.columns else regime_df["nifty500"]
        open_prices = pd.read_csv(cfg.OPEN_CACHE, index_col=0, parse_dates=True) if os.path.exists(cfg.OPEN_CACHE) else close.copy()
        return close, volume, regime_df["nifty500"], regime_df["nifty50"], nifty100, nifty_mid, open_prices

    print(f"Downloading {len(UNIVERSE)} stocks...")
    tickers = [t+".NS" for t in UNIVERSE]
    raw     = yf.download(tickers, start=cfg.DATA_FETCH_START,
                          end=cfg.DATA_FETCH_END, auto_adjust=True, progress=True)
    close   = raw["Close"].copy()
    volume  = raw["Volume"].copy()
    close.columns  = [c.replace(".NS","") for c in close.columns]
    volume.columns = [c.replace(".NS","") for c in volume.columns]
    close.to_csv(cfg.DATA_CACHE_FILE)
    volume.to_csv(cfg.VOLUME_CACHE)

    r500 = yf.download(cfg.REGIME_TICKER,    start=cfg.DATA_FETCH_START,
                       end=cfg.DATA_FETCH_END, auto_adjust=True, progress=False)
    r50  = yf.download(cfg.BENCHMARK_TICKER, start=cfg.DATA_FETCH_START,
                       end=cfg.DATA_FETCH_END, auto_adjust=True, progress=False)
    rd   = pd.DataFrame({"nifty500": r500["Close"].squeeze(),
                         "nifty50":  r50["Close"].squeeze()})
    rd.to_csv(cfg.REGIME_CACHE)
    return close, volume, rd["nifty500"], rd["nifty50"]


# ── Signal helpers ─────────────────────────────────────────────────────────
def get_regime(nifty500, date):
    d   = nifty500.loc[:date].dropna()
    if len(d) < cfg.REGIME_DMA:
        return "RISK-ON"
    dma = d.rolling(cfg.REGIME_DMA).mean()

    # Confirmation filter
    confirm_days = getattr(cfg, "REGIME_CONFIRM_DAYS", 0)
    if confirm_days > 0 and len(d) >= confirm_days:
        recent     = d.iloc[-confirm_days:]
        recent_dma = dma.iloc[-confirm_days:]
        all_above  = all(p > m for p, m in zip(recent, recent_dma))
        return "RISK-ON" if all_above else "RISK-OFF"

    return "RISK-ON" if d.iloc[-1] > dma.iloc[-1] else "RISK-OFF"



def get_regime_strength(nifty500, nifty100, nifty_mid, date) -> float:
    """Weighted regime strength 0.0-1.0 for backtesting."""
    def margin(series):
        d = series.loc[:date].dropna()
        if len(d) < cfg.REGIME_DMA:
            return 0.0
        latest = float(d.iloc[-1])
        avg    = float(d.rolling(cfg.REGIME_DMA).mean().iloc[-1])
        return (latest - avg) / avg

    m500  = margin(nifty500)
    m100  = margin(nifty100)
    mmid  = margin(nifty_mid)
    composite = (cfg.REGIME_WEIGHT_NIFTY500 * m500 +
                 cfg.REGIME_WEIGHT_NIFTY100  * m100 +
                 cfg.REGIME_WEIGHT_MIDCAP    * mmid)
    fraction = (composite - cfg.REGIME_DEPLOY_MIN) / (cfg.REGIME_DEPLOY_MAX - cfg.REGIME_DEPLOY_MIN)
    return max(0.0, min(1.0, fraction))

def is_above_100dma(close, ticker, date):
    if ticker not in close.columns:
        return True
    p = close[ticker].loc[:date].dropna()
    if len(p) < cfg.DMA_EXIT:
        return True
    return p.iloc[-1] >= p.rolling(cfg.DMA_EXIT).mean().iloc[-1]


def compute_scores_on(close, volume, date, tickers):
    """Monthly score computation. Used for ranking and rotation."""
    records = []
    c_slice = close.loc[:date]

    for t in tickers:
        if t not in c_slice.columns:
            continue
        c = c_slice[t].dropna()
        v = volume[t].loc[:date].dropna() if t in volume.columns else pd.Series()

        # Liquidity filter
        if len(c) < 60 or c.iloc[-1] < cfg.MIN_PRICE:
            continue
        if len(v) >= 60:
            if (c * v).rolling(60).mean().iloc[-1] / 1e7 < cfg.MIN_AVG_VALUE_CR:
                continue

        if len(c) < cfg.LOOKBACK_12M + cfg.SKIP_RECENT:
            continue

        s      = cfg.SKIP_RECENT
        p_now  = c.iloc[-(s+1)]
        mom12  = (p_now - c.iloc[-(cfg.LOOKBACK_12M+s)]) / c.iloc[-(cfg.LOOKBACK_12M+s)]
        mom6   = (p_now - c.iloc[-(cfg.LOOKBACK_6M+s)])  / c.iloc[-(cfg.LOOKBACK_6M+s)]
        mom3   = (p_now - c.iloc[-(cfg.LOOKBACK_3M+s)])  / c.iloc[-(cfg.LOOKBACK_3M+s)]
        vol6   = c.iloc[-cfg.LOOKBACK_6M:].pct_change().dropna().std() * np.sqrt(252)

        records.append({"ticker":t, "sector":UNIVERSE.get(t,"Unknown"),
                        "mom12":mom12, "mom6":mom6, "mom3":mom3, "vol6":vol6})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("ticker")
    def z(s): return (s-s.mean())/s.std() if s.std()>0 else s*0
    df["score"] = (cfg.W_MOM_12M*z(df["mom12"]) + cfg.W_MOM_6M*z(df["mom6"]) +
                   cfg.W_MOM_3M*z(df["mom3"])   + cfg.W_VOL*z(df["vol6"]))
    return df.sort_values("score", ascending=False)


def pick_portfolio(scored):
    sel, sc = [], {}
    for t, row in scored.iterrows():
        s = row["sector"]
        if sc.get(s,0) < cfg.MAX_PER_SECTOR:
            sel.append(t)
            sc[s] = sc.get(s,0)+1
        if len(sel) == cfg.TOP_N:
            break
    return sel


def find_replacement(scored, current_holdings, exits):
    """
    Finds immediate replacements for exited stocks.
    Same logic as execution.py — picks best available from top 25
    respecting sector cap and existing holdings.
    """
    if scored.empty:
        return []

    remaining    = [t for t in current_holdings if t not in exits]
    top_25       = scored.head(cfg.EXIT_RANK_CUTOFF).index.tolist()
    candidates   = [t for t in top_25 if t not in remaining]

    sector_count = {}
    for t in remaining:
        s = UNIVERSE.get(t, "Unknown")
        sector_count[s] = sector_count.get(s, 0) + 1

    replacements = []
    for t in candidates:
        s = UNIVERSE.get(t, "Unknown")
        if sector_count.get(s, 0) < cfg.MAX_PER_SECTOR:
            replacements.append(t)
            sector_count[s] = sector_count.get(s, 0) + 1
        if len(replacements) == len(exits):
            break

    return replacements


def get_price(close, ticker, date):
    """Safe price lookup — returns last close on or before date."""
    try:
        return float(close[ticker].loc[:date].dropna().iloc[-1])
    except Exception:
        return 0.0

def get_open_price(open_prices, ticker, date):
    """Returns opening price ON the given date (not before)."""
    try:
        if ticker not in open_prices.columns:
            return 0.0
        if date in open_prices.index:
            val = open_prices[ticker].loc[date]
            return float(val) if pd.notna(val) else 0.0
        return 0.0
    except Exception:
        return 0.0

def get_fill_price(close_price, open_price, side, rank=0):
    """
    Realistic fill model using next day open price.
    Buy:  fills at open price if open <= close x buffer, else missed
    Sell: fills at open price if open >= close x 0.99, else missed
    """
    if side == "buy":
        if rank < 5:
            buffer = 0.050
        elif rank < 12:
            buffer = 0.030
        else:
            buffer = 0.020
        limit = close_price * (1 + buffer)
        if open_price > 0 and open_price <= limit:
            return open_price
        elif open_price == 0:
            return close_price
        else:
            return None
    else:
        limit = close_price * 0.990
        if open_price > 0 and open_price >= limit:
            return open_price
        elif open_price == 0:
            return close_price
        else:
            return None
def print_metrics(equity, label):
    # Resample to monthly for consistent metric calculation
    monthly = equity.resample("MS").last().dropna()
    ret     = monthly.pct_change().dropna()
    total   = (equity.iloc[-1] / equity.iloc[0]) - 1
    n_yr    = len(ret) / 12
    cagr    = (1 + total) ** (1 / n_yr) - 1 if n_yr > 0 else 0
    mrf     = (1 + cfg.RISK_FREE_RATE) ** (1/12) - 1
    exc     = ret - mrf
    sharpe  = exc.mean()/exc.std()*np.sqrt(12) if exc.std() > 0 else 0
    down    = exc[exc < 0]
    sortino = exc.mean()/down.std()*np.sqrt(12) if len(down) > 0 and down.std() > 0 else 0
    rollmax = equity.cummax()
    maxdd   = ((equity - rollmax) / rollmax).min()
    winrate = (ret > 0).sum() / len(ret) if len(ret) > 0 else 0

    print(f"\n{'='*48}")
    print(f"  {label}")
    print(f"{'='*48}")
    print(f"  Period       : {equity.index[0].strftime('%b %Y')} → {equity.index[-1].strftime('%b %Y')}")
    print(f"  Total Return : {total*100:+.1f}%")
    print(f"  CAGR         : {cagr*100:.1f}%")
    print(f"  Sharpe       : {sharpe:.2f}")
    print(f"  Sortino      : {sortino:.2f}")
    print(f"  Max Drawdown : {maxdd*100:.1f}%")
    print(f"  Win Rate     : {winrate*100:.0f}% of months")
    print(f"  Best Month   : {ret.max()*100:+.1f}%")
    print(f"  Worst Month  : {ret.min()*100:+.1f}%")
    return {"cagr":cagr,"sharpe":sharpe,"sortino":sortino,"maxdd":maxdd,"winrate":winrate}


# ── Main backtest ──────────────────────────────────────────────────────────
def run_backtest():
    close, volume, nifty500, nifty50, nifty100, nifty_mid, open_prices = load_data()

    # All trading days in backtest window
    all_days = close.loc[cfg.START_DATE:cfg.END_DATE].index

    cash        = float(cfg.INITIAL_CAPITAL)
    holdings    = {}
    eq_curve    = []
    trade_log   = []
    tickers     = list(UNIVERSE.keys())

    cached_scored        = pd.DataFrame()
    cached_top_25        = []
    cached_top_7         = []
    in_risk_off          = False
    last_rebalance_month = None   # month comparison — avoids type mismatch bug

    print(f"\n{'='*55}")
    print(f"  BACKTEST [DAILY EXITS]  |  {cfg.START_DATE} → {cfg.END_DATE}")
    print(f"  Capital : ₹{cfg.INITIAL_CAPITAL:,.0f}  |  Stocks: {cfg.TOP_N}  |  Sector cap: {cfg.MAX_PER_SECTOR}")
    print(f"  Formula : {cfg.W_MOM_12M}*z12M + {cfg.W_MOM_6M}*z6M + {cfg.W_MOM_3M}*z3M + {cfg.W_VOL}*zVol")
    print(f"  Exit monitoring: DAILY  |  Full rebalance: MONTHLY")
    print(f"{'='*55}\n")

    all_days_list = list(all_days)
    for day_idx, day in enumerate(all_days_list):
        date_str    = pd.Timestamp(day).strftime("%Y-%m-%d")
        # Next trading day — used for realistic fill simulation
        next_day    = all_days_list[day_idx + 1] if day_idx + 1 < len(all_days_list) else day
        cur_month    = pd.Timestamp(day).to_period("M")
        # First trading day of each month = month changed since last rebalance
        is_rebalance = (cur_month != last_rebalance_month)

        # ── REGIME CHECK (daily) ────────────────────────────────────────────
        nifty500_regime = get_regime(nifty500, day)
        if nifty500_regime == "RISK-ON":
            strength = 1.0
            regime   = "RISK-ON"
        elif cfg.REGIME_WEIGHTED:
            strength = get_regime_strength(nifty500, nifty100, nifty_mid, day)
            regime   = "RISK-ON" if strength > 0 else "RISK-OFF"
        else:
            strength = 0.0
            regime   = "RISK-OFF"

        # ── RISK-OFF: liquidate everything ──────────────────────────────────
        if regime == "RISK-OFF":
            if not in_risk_off:
                # First day of risk-off — sell everything
                for t, sh in list(holdings.items()):
                    if sh > 0 and t in close.columns:
                        px = get_price(close, t, day)
                        if px > 0:
                            open_px  = get_open_price(open_prices, t, next_day)
                            fill_px  = get_fill_price(px, open_px, "sell")
                            fill_px  = fill_px if fill_px else open_px if open_px > 0 else px
                            proceeds = sh * fill_px
                            cost     = txn_cost(proceeds, "sell")
                            cash    += proceeds - cost
                            trade_log.append({
                                "date":date_str,"ticker":t,"action":"SELL(RISK-OFF)",
                                "shares":sh,"price":fill_px,"value":proceeds,"cost":cost
                            })
                holdings     = {}
                in_risk_off  = True
                cached_scored = pd.DataFrame()
                cached_top_25 = []
                cached_top_7  = []

            # Record daily value (all cash)
            eq_curve.append({"date":day,"value":cash,"regime":"OFF"})

            if is_rebalance:
                print(f"{date_str} | RISK-OFF  | Cash : ₹{cash:>12,.0f}")
            continue

        # ── RISK-ON ─────────────────────────────────────────────────────────
        in_risk_off = False

        # ── REBALANCE DAY: recompute full scores ─────────────────────────────
        if is_rebalance:
            cached_scored = compute_scores_on(close, volume, day, tickers)
            if not cached_scored.empty:
                cached_top_25 = cached_scored.head(cfg.EXIT_RANK_CUTOFF).index.tolist()
                cached_top_7  = pick_portfolio(cached_scored)

        # ── DAILY EXIT CHECK ─────────────────────────────────────────────────
        # Check every current holding against exit rules using today's prices
        current_held = [t for t, s in holdings.items() if s > 0]
        exits        = []

        for t in current_held:
            exit_reason = None

            # Rule 1: dropped out of top 25 (using cached monthly scores)
            if cached_top_25 and t not in cached_top_25:
                exit_reason = "rank"

            # Rule 2: price below 100 DMA (checked with today's price)
            if t in close.columns and not is_above_100dma(close, t, day):
                exit_reason = "100DMA"

            if exit_reason:
                exits.append((t, exit_reason))

        # ── PROCESS EXITS + IMMEDIATE REPLACEMENT ───────────────────────────
        exit_tickers = [t for t, _ in exits]

        for t, reason in exits:
            sh = holdings.get(t, 0)
            if sh > 0:
                px = get_price(close, t, day)
                if px > 0:
                    open_px  = get_open_price(open_prices, t, next_day)
                    fill_px  = get_fill_price(px, open_px, "sell")
                    fill_px  = fill_px if fill_px else open_px if open_px > 0 else px
                    proceeds = sh * fill_px
                    cost     = txn_cost(proceeds, "sell")
                    cash    += proceeds - cost
                    trade_log.append({
                        "date":date_str,"ticker":t,"action":f"SELL({reason})",
                        "shares":sh,"price":fill_px,"value":proceeds,"cost":cost
                    })
                    holdings[t] = 0

        # Immediately find and buy replacements for exits
        if exit_tickers and not cached_scored.empty:
            replacements = find_replacement(cached_scored, current_held, exit_tickers)

            # Compute target allocation
            port_val = cash
            for t, sh in holdings.items():
                if sh > 0:
                    px = get_price(close, t, day)
                    port_val += sh * px

            stocks_to_hold = max(1, round(cfg.TOP_N * strength))
            target         = (port_val * strength) / stocks_to_hold

            for t in replacements:
                px = get_price(close, t, day)
                if px <= 0:
                    continue
                rank     = cached_scored.index.get_loc(t) if t in cached_scored.index else 99
                open_px  = get_open_price(open_prices, t, next_day)
                fill_px  = get_fill_price(px, open_px, "buy", rank)
                if fill_px is None:
                    continue  # missed — gap-up too large
                cur_val = holdings.get(t, 0) * fill_px
                if cur_val < target * 0.95:
                    n    = int((target - cur_val) / fill_px)
                    cost = n * fill_px
                    tc   = txn_cost(cost, "buy")
                    if n > 0 and cash >= cost + tc:
                        cash -= (cost + tc)
                        holdings[t] = holdings.get(t, 0) + n
                        bought += 1
                        trade_log.append({
                            "date":date_str,"ticker":t,"action":"BUY(replacement)",
                            "shares":n,"price":fill_px,"value":cost,"cost":tc
                        })

        # ── MONTHLY FULL ROTATION ────────────────────────────────────────────
        # On rebalance day: rotate underperformers + top up to full allocation
        if is_rebalance and not cached_scored.empty:
            current_held_after = [t for t, s in holdings.items() if s > 0]

            # Find holdings not in current top 7 — rotate them out
            rotate_out = [t for t in current_held_after if t not in cached_top_7]
            for t in rotate_out:
                sh = holdings.get(t, 0)
                if sh > 0:
                    px = get_price(close, t, day)
                    if px > 0:
                        proceeds = sh * px
                        cost     = txn_cost(proceeds, "sell")
                        cash    += proceeds - cost
                        trade_log.append({
                            "date":date_str,"ticker":t,"action":"SELL(rotation)",
                            "shares":sh,"price":px,"value":proceeds,"cost":cost
                        })
                        holdings[t] = 0

            # Recompute portfolio value
            port_val = cash
            for t, sh in holdings.items():
                if sh > 0:
                    port_val += get_price(close, t, day) * sh

            stocks_to_hold = max(1, round(cfg.TOP_N * strength))
            target         = (port_val * strength) / stocks_to_hold

            # Buy top N (new positions + top-ups) with realistic fills
            bought = 0
            for t in cached_top_7:
                if bought >= stocks_to_hold:
                    break
                px = get_price(close, t, day)
                if px <= 0:
                    continue
                # Skip if price > target allocation (expensive stock)
                if px > target:
                    continue
                rank    = cached_scored.index.get_loc(t) if t in cached_scored.index else 99
                open_px = get_open_price(open_prices, t, next_day)
                fill_px = get_fill_price(px, open_px, "buy", rank)
                if fill_px is None:
                    continue  # missed — gap-up too large
                cur_val = holdings.get(t, 0) * fill_px
                if cur_val < target * 0.95:
                    n    = int((target - cur_val) / fill_px)
                    cost = n * fill_px
                    tc   = txn_cost(cost, "buy")
                    if n > 0 and cash >= cost + tc:
                        cash -= (cost + tc)
                        holdings[t] = holdings.get(t, 0) + n
                        bought += 1
                        trade_log.append({
                            "date":date_str,"ticker":t,"action":"BUY",
                            "shares":n,"price":fill_px,"value":cost,"cost":tc
                        })

            # Print monthly summary
            held = [t for t, s in holdings.items() if s > 0]
            port_val = cash + sum(
                holdings.get(t,0) * get_price(close,t,day)
                for t in held
            )
            print(f"{date_str} | RISK-ON   | ₹{port_val:>12,.0f} | {', '.join(held[:7])}")
            last_rebalance_month = cur_month

        # ── RECORD DAILY PORTFOLIO VALUE ─────────────────────────────────────
        day_val = cash
        for t, sh in holdings.items():
            if sh > 0:
                day_val += sh * get_price(close, t, day)
        eq_curve.append({"date":day,"value":day_val,"regime":"ON"})

    # ── Build equity curve ─────────────────────────────────────────────────
    eq_df  = pd.DataFrame(eq_curve).set_index("date")
    eq_df.index = pd.to_datetime(eq_df.index)
    equity = eq_df["value"]

    # Benchmark
    bench_raw   = nifty50.loc[cfg.START_DATE:cfg.END_DATE].dropna()
    bench_curve = (bench_raw / bench_raw.iloc[0]) * cfg.INITIAL_CAPITAL

    # ── Print results ──────────────────────────────────────────────────────
    m_s = print_metrics(equity,      "STRATEGY [DAILY EXITS] — Aggressive Momentum")
    m_b = print_metrics(bench_curve, "BENCHMARK              — Nifty 50 Buy & Hold")

    print(f"\n{'='*48}")
    print(f"  Outperformance vs Nifty 50")
    print(f"  CAGR delta   : {(m_s['cagr']-m_b['cagr'])*100:+.1f}% per year")
    print(f"  Sharpe delta : {m_s['sharpe']-m_b['sharpe']:+.2f}")
    print(f"  DD improvement: {(m_b['maxdd']-m_s['maxdd'])*100:+.1f}%")
    print(f"{'='*48}")

    # Save outputs
    equity.to_csv("equity_curve_daily.csv")
    pd.DataFrame(trade_log).to_csv("trade_log_daily.csv", index=False)
    print(f"\nSaved: equity_curve_daily.csv  |  trade_log_daily.csv")
    print(f"Total trades executed: {len(trade_log)}")


if __name__ == "__main__":
    run_backtest()
"""
backtester.py  [AGGRESSIVE CONFIG]
====================================
Walk-forward backtest — imports all parameters from config.py.

Aggressive changes vs conservative:
  - TOP_N 10 → 7  (more concentrated positions)
  - MAX_PER_SECTOR 2 → 3  (ride hot sectors harder)
  - EXIT_RANK_CUTOFF 20 → 25  (let winners run longer)
  - Formula: adds 3M momentum, halves volatility penalty

Run:
    pip install yfinance pandas numpy python-dateutil
    python backtester.py

First run downloads data (~4 mins). Saves to disk. Subsequent runs use cache.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")

import config as cfg

# ── Universe (same as signals.py) ──────────────────────────────────────────
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
    "CUMMINSIND":"Capital Goods","VOLTAS":"Capital Goods","SUZLON":"Capital Goods","KEC":"Capital Goods",
    "RELIANCE":"Energy","ONGC":"Energy","BPCL":"Energy","GAIL":"Energy",
    "TATAPOWER":"Energy","ADANIGREEN":"Energy","NTPC":"Energy",
    "POWERGRID":"Energy","NHPC":"Energy","TORNTPOWER":"Energy","SJVN":"Energy",
    "PIDILITIND":"Chemicals","DEEPAKNTR":"Chemicals","SRF":"Chemicals",
    "AARTIIND":"Chemicals","NAVINFLUOR":"Chemicals","ATUL":"Chemicals","ALKYLAMINE":"Chemicals",
    "ASIANPAINT":"Paints","BERGEPAINT":"Paints","KANSAINER":"Paints",
    "TITAN":"Consumer","TRENT":"Consumer","DMART":"Consumer",
    "PAGEIND":"Consumer","JUBLFOOD":"Consumer","IRCTC":"Consumer",
    "NAUKRI":"Consumer","ZOMATO":"Consumer","INDIAMART":"Consumer",
    "ULTRACEMCO":"Cement","SHREECEM":"Cement","AMBUJACEM":"Cement","GRASIM":"Cement","JKCEMENT":"Cement",
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


# ── Transaction cost calculator ────────────────────────────────────────────
def txn_cost(value, side):
    cost = value * (cfg.EXCHANGE_CHARGE + cfg.SEBI_CHARGE)
    cost += value * cfg.STAMP_DUTY  if side == "buy"  else 0
    cost += value * cfg.STT_BUY     if side == "buy"  else 0   # ← add this
    cost += value * cfg.STT_SELL    if side == "sell" else 0
    return cost


# ── Data fetching with cache ───────────────────────────────────────────────
def load_data():
    if os.path.exists(cfg.DATA_CACHE_FILE):
        print("Loading cached data...")
        close  = pd.read_csv(cfg.DATA_CACHE_FILE,  index_col=0, parse_dates=True)
        volume = pd.read_csv(cfg.VOLUME_CACHE,     index_col=0, parse_dates=True)
        regime_df = pd.read_csv(cfg.REGIME_CACHE,  index_col=0, parse_dates=True)
        print(f"Loaded {close.shape[1]} stocks, {close.shape[0]} days.")
        return close, volume, regime_df["nifty500"], regime_df["nifty50"]

    print(f"First run — downloading {len(UNIVERSE)} stocks ({cfg.DATA_FETCH_START} to {cfg.DATA_FETCH_END})...")
    print("This takes ~4 minutes. Saved to disk after.\n")

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
    print(f"\nData cached. {close.shape[1]} stocks loaded.")
    return close, volume, rd["nifty500"], rd["nifty50"]


# ── Signal functions (no lookahead bias) ───────────────────────────────────
def regime_on(nifty500, date):
    d = nifty500.loc[:date].dropna()
    if len(d) < cfg.REGIME_DMA:
        return "RISK-ON"
    return "RISK-ON" if d.iloc[-1] > d.rolling(cfg.REGIME_DMA).mean().iloc[-1] else "RISK-OFF"


def liquid_on(close, volume, date, tickers):
    out = []
    for t in tickers:
        if t not in close.columns:
            continue
        c = close[t].loc[:date].dropna()
        v = volume[t].loc[:date].dropna()
        if len(c) < 60 or c.iloc[-1] < cfg.MIN_PRICE:
            continue
        if (c*v).rolling(60).mean().iloc[-1]/1e7 >= cfg.MIN_AVG_VALUE_CR:
            out.append(t)
    return out


def scores_on(close, date, tickers):
    rows, c_slice = [], close.loc[:date]
    for t in tickers:
        if t not in c_slice.columns:
            continue
        p = c_slice[t].dropna()
        if len(p) < cfg.LOOKBACK_12M + cfg.SKIP_RECENT:
            continue
        s = cfg.SKIP_RECENT
        p_now = p.iloc[-(s+1)]
        mom12 = (p_now - p.iloc[-(cfg.LOOKBACK_12M+s)]) / p.iloc[-(cfg.LOOKBACK_12M+s)]
        mom6  = (p_now - p.iloc[-(cfg.LOOKBACK_6M+s)])  / p.iloc[-(cfg.LOOKBACK_6M+s)]
        mom3  = (p_now - p.iloc[-(cfg.LOOKBACK_3M+s)])  / p.iloc[-(cfg.LOOKBACK_3M+s)]
        vol6  = p.iloc[-cfg.LOOKBACK_6M:].pct_change().dropna().std() * np.sqrt(252)
        rows.append({"ticker":t,"sector":UNIVERSE.get(t,"Unknown"),
                     "mom12":mom12,"mom6":mom6,"mom3":mom3,"vol6":vol6})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("ticker")
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


def above_dma(close, t, date):
    if t not in close.columns:
        return True
    p = close[t].loc[:date].dropna()
    if len(p) < cfg.DMA_EXIT:
        return True
    return p.iloc[-1] >= p.rolling(cfg.DMA_EXIT).mean().iloc[-1]


# ── Performance metrics ────────────────────────────────────────────────────
def print_metrics(equity, label):
    ret     = equity.pct_change().dropna()
    total   = (equity.iloc[-1]/equity.iloc[0]) - 1
    n_yr    = len(ret)/12
    cagr    = (1+total)**(1/n_yr) - 1
    mrf     = (1+cfg.RISK_FREE_RATE)**(1/12)-1
    exc     = ret - mrf
    sharpe  = exc.mean()/exc.std()*np.sqrt(12) if exc.std()>0 else 0
    down    = exc[exc<0]
    sortino = exc.mean()/down.std()*np.sqrt(12) if len(down)>0 and down.std()>0 else 0
    rollmax = equity.cummax()
    maxdd   = ((equity-rollmax)/rollmax).min()
    winrate = (ret>0).sum()/len(ret)

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


# ── Main backtest loop ─────────────────────────────────────────────────────
def run_backtest():
    close, volume, nifty500, nifty50 = load_data()

    all_dates     = close.loc[cfg.START_DATE:cfg.END_DATE].index
    month_starts  = (pd.Series(all_dates)
                     .groupby(pd.Series(all_dates).dt.to_period("M"))
                     .first().values)

    cash      = float(cfg.INITIAL_CAPITAL)
    holdings  = {}   # {ticker: shares}
    eq_curve  = []
    trade_log = []
    tickers   = list(UNIVERSE.keys())

    print(f"\n{'='*55}")
    print(f"  BACKTEST  |  {cfg.START_DATE} → {cfg.END_DATE}")
    print(f"  Capital   : ₹{cash:,.0f}  |  Stocks: {cfg.TOP_N}  |  Sector cap: {cfg.MAX_PER_SECTOR}")
    print(f"  Formula   : {cfg.W_MOM_12M}*z12M + {cfg.W_MOM_6M}*z6M + {cfg.W_MOM_3M}*z3M + {cfg.W_VOL}*zVol")
    print(f"{'='*55}\n")

    for rb_date in month_starts:
        date_str = pd.Timestamp(rb_date).strftime("%Y-%m-%d")

        # Portfolio value at start of month
        port_val = cash
        for t, sh in holdings.items():
            if sh > 0 and t in close.columns:
                try: port_val += sh * close[t].loc[:rb_date].dropna().iloc[-1]
                except: pass

        regime = regime_on(nifty500, rb_date)

        # ── RISK-OFF: sell everything ────────────────────────────────────
        if regime == "RISK-OFF":
            for t, sh in list(holdings.items()):
                if sh > 0 and t in close.columns:
                    try:
                        px      = close[t].loc[:rb_date].dropna().iloc[-1]
                        proceed = sh * px
                        cost    = txn_cost(proceed, "sell")
                        cash   += proceed - cost
                        trade_log.append({"date":date_str,"ticker":t,"action":"SELL(OFF)",
                                          "shares":sh,"price":px,"value":proceed,"cost":cost})
                    except: pass
            holdings = {}
            eq_curve.append({"date":rb_date,"value":cash,"regime":"OFF"})
            print(f"{date_str} | RISK-OFF  | Cash : ₹{cash:>12,.0f}")
            continue

        # ── RISK-ON: generate signals ────────────────────────────────────
        liquid  = liquid_on(close, volume, rb_date, tickers)
        scored  = scores_on(close, rb_date, liquid)
        if scored.empty:
            eq_curve.append({"date":rb_date,"value":port_val,"regime":"ON"})
            continue

        top25   = scored.head(cfg.EXIT_RANK_CUTOFF).index.tolist()
        new_port = pick_portfolio(scored)

        # ── Exit rules ───────────────────────────────────────────────────
        for t, sh in list(holdings.items()):
            if sh == 0: continue
            sell = (t not in top25) or (not above_dma(close, t, rb_date))
            if sell:
                try:
                    px      = close[t].loc[:rb_date].dropna().iloc[-1]
                    proceed = sh * px
                    cost    = txn_cost(proceed, "sell")
                    cash   += proceed - cost
                    reason  = "100DMA" if not above_dma(close,t,rb_date) else "RANK"
                    trade_log.append({"date":date_str,"ticker":t,"action":f"SELL({reason})",
                                      "shares":sh,"price":px,"value":proceed,"cost":cost})
                    holdings[t] = 0
                except: pass

        # ── Recompute portfolio value before buying ──────────────────────
        port_val = cash
        for t, sh in holdings.items():
            if sh > 0 and t in close.columns:
                try: port_val += sh * close[t].loc[:rb_date].dropna().iloc[-1]
                except: pass

        target = port_val / cfg.TOP_N

        # ── Buy new / top-up positions ───────────────────────────────────
        for t in new_port:
            try:
                px      = close[t].loc[:rb_date].dropna().iloc[-1]
                cur_val = holdings.get(t, 0) * px
                if cur_val < target * 0.95:
                    buy_val = target - cur_val
                    n_shares = int(buy_val / px)
                    cost_buy = n_shares * px
                    tc       = txn_cost(cost_buy, "buy")
                    if n_shares > 0 and cash >= cost_buy + tc:
                        cash -= (cost_buy + tc)
                        holdings[t] = holdings.get(t,0) + n_shares
                        trade_log.append({"date":date_str,"ticker":t,"action":"BUY",
                                          "shares":n_shares,"price":px,"value":cost_buy,"cost":tc})
            except: pass

        # ── Record month value ───────────────────────────────────────────
        month_val = cash
        for t, sh in holdings.items():
            if sh > 0 and t in close.columns:
                try: month_val += sh * close[t].loc[:rb_date].dropna().iloc[-1]
                except: pass

        eq_curve.append({"date":rb_date,"value":month_val,"regime":"ON"})
        held = [t for t,s in holdings.items() if s>0]
        print(f"{date_str} | RISK-ON   | ₹{month_val:>12,.0f} | {', '.join(held[:7])}")

    # ── Results ───────────────────────────────────────────────────────────
    eq_df = pd.DataFrame(eq_curve).set_index("date")
    eq_df.index = pd.to_datetime(eq_df.index)
    equity = eq_df["value"]

    bench_raw   = nifty50.loc[cfg.START_DATE:cfg.END_DATE].resample("MS").first().dropna()
    bench_curve = (bench_raw / bench_raw.iloc[0]) * cfg.INITIAL_CAPITAL

    m_strat = print_metrics(equity,      "STRATEGY  — Aggressive Momentum")
    m_bench = print_metrics(bench_curve, "BENCHMARK — Nifty 50 Buy & Hold")

    print(f"\n{'='*48}")
    print(f"  Outperformance vs Nifty 50")
    print(f"  CAGR delta   : {(m_strat['cagr']-m_bench['cagr'])*100:+.1f}% per year")
    print(f"  Sharpe delta : {m_strat['sharpe']-m_bench['sharpe']:+.2f}")
    print(f"{'='*48}")

    equity.to_csv("equity_curve.csv")
    pd.DataFrame(trade_log).to_csv("trade_log.csv", index=False)
    print("\nSaved: equity_curve.csv  |  trade_log.csv")


if __name__ == "__main__":
    run_backtest()

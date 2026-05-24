"""
backtester_us.py
================
Same Statistical Quality + Momentum strategy applied to US markets.
Universe  : S&P 500 large caps
Regime    : S&P 500 (^GSPC) vs 200-DMA
Benchmark : S&P 500 Buy & Hold
Period    : 2020-01-01 to 2024-12-31 (same as India backtest)

Purpose   : Comparison only — does momentum work in US markets?
            US market is more efficient than India.
            Lower alpha expected but strategy logic should still hold.

Run:
    python backtester_us.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STRATEGY PARAMETERS (identical to India)
# ─────────────────────────────────────────────
TOP_N            = 7
MAX_PER_SECTOR   = 3
EXIT_RANK_CUTOFF = 25
W_MOM_12M        = 0.40
W_MOM_6M         = 0.35
W_MOM_3M         = 0.15
W_VOL            = -0.10
LOOKBACK_12M     = 252
LOOKBACK_6M      = 126
LOOKBACK_3M      = 63
SKIP_RECENT      = 20
REGIME_DMA       = 200
DMA_EXIT         = 100
MIN_PRICE        = 10       # USD — filters penny stocks
RISK_FREE_RATE   = 0.04     # 4% — US Fed Funds rate approx

# ─────────────────────────────────────────────
# US TRANSACTION COSTS
# Much lower than India — no STT, no stamp duty
# SEC fee: $0.0000278 per $1 sold
# FINRA TAF: negligible
# Brokerage: $0 (most US brokers)
# Effective round trip: ~0.003%
# ─────────────────────────────────────────────
SEC_FEE = 0.0000278    # sell side only
BROKERAGE = 0.0        # zero commission brokers

def txn_cost(value, side):
    cost = BROKERAGE
    if side == "sell":
        cost += value * SEC_FEE
    return cost

# ─────────────────────────────────────────────
# BACKTEST SETTINGS
# ─────────────────────────────────────────────
START_DATE       = "2018-01-01"
END_DATE         = "2026-05-22"
INITIAL_CAPITAL  = 100000
DATA_FETCH_START = "2018-01-01"
DATA_FETCH_END   = "2026-05-22"

# Separate cache files — does not touch Indian cache
US_PRICE_CACHE   = "us_price_cache.csv"
US_VOLUME_CACHE  = "us_volume_cache.csv"
US_INDEX_CACHE   = "us_index_cache.csv"

REGIME_TICKER    = "^GSPC"    # S&P 500
BENCHMARK_TICKER = "^GSPC"    # S&P 500 buy and hold

# ─────────────────────────────────────────────
# S&P 500 UNIVERSE — Large caps by sector
# Using standard NYSE/NASDAQ tickers (no suffix needed)
# ─────────────────────────────────────────────
UNIVERSE = {
    # ── Technology ──────────────────────────
    "AAPL"  : "Technology",
    "MSFT"  : "Technology",
    "NVDA"  : "Technology",
    "AVGO"  : "Technology",
    "ORCL"  : "Technology",
    "CSCO"  : "Technology",
    "ACN"   : "Technology",
    "IBM"   : "Technology",
    "NOW"   : "Technology",
    "ADBE"  : "Technology",
    "CRM"   : "Technology",
    "INTC"  : "Technology",
    "AMD"   : "Technology",
    "QCOM"  : "Technology",
    "TXN"   : "Technology",
    "AMAT"  : "Technology",
    "MU"    : "Technology",
    "KLAC"  : "Technology",
    "LRCX"  : "Technology",
    "MCHP"  : "Technology",

    # ── Communication Services ───────────────
    "GOOGL" : "Communication",
    "META"  : "Communication",
    "NFLX"  : "Communication",
    "DIS"   : "Communication",
    "CMCSA" : "Communication",
    "T"     : "Communication",
    "VZ"    : "Communication",
    "TMUS"  : "Communication",
    "EA"    : "Communication",
    "TTWO"  : "Communication",

    # ── Consumer Discretionary ───────────────
    "AMZN"  : "Consumer Disc",
    "TSLA"  : "Consumer Disc",
    "HD"    : "Consumer Disc",
    "MCD"   : "Consumer Disc",
    "NKE"   : "Consumer Disc",
    "SBUX"  : "Consumer Disc",
    "TJX"   : "Consumer Disc",
    "BKNG"  : "Consumer Disc",
    "LOW"   : "Consumer Disc",
    "GM"    : "Consumer Disc",
    "F"     : "Consumer Disc",
    "ABNB"  : "Consumer Disc",
    "RCL"   : "Consumer Disc",
    "CCL"   : "Consumer Disc",

    # ── Consumer Staples ─────────────────────
    "WMT"   : "Consumer Staples",
    "PG"    : "Consumer Staples",
    "KO"    : "Consumer Staples",
    "PEP"   : "Consumer Staples",
    "COST"  : "Consumer Staples",
    "PM"    : "Consumer Staples",
    "MO"    : "Consumer Staples",
    "MDLZ"  : "Consumer Staples",
    "CL"    : "Consumer Staples",
    "GIS"   : "Consumer Staples",

    # ── Healthcare ───────────────────────────
    "LLY"   : "Healthcare",
    "UNH"   : "Healthcare",
    "JNJ"   : "Healthcare",
    "ABBV"  : "Healthcare",
    "MRK"   : "Healthcare",
    "TMO"   : "Healthcare",
    "ABT"   : "Healthcare",
    "DHR"   : "Healthcare",
    "PFE"   : "Healthcare",
    "AMGN"  : "Healthcare",
    "ISRG"  : "Healthcare",
    "SYK"   : "Healthcare",
    "BSX"   : "Healthcare",
    "GILD"  : "Healthcare",
    "VRTX"  : "Healthcare",
    "REGN"  : "Healthcare",
    "ZTS"   : "Healthcare",
    "ELV"   : "Healthcare",
    "CI"    : "Healthcare",
    "CVS"   : "Healthcare",

    # ── Financials ───────────────────────────
    "BRK-B" : "Financials",
    "JPM"   : "Financials",
    "V"     : "Financials",
    "MA"    : "Financials",
    "BAC"   : "Financials",
    "WFC"   : "Financials",
    "GS"    : "Financials",
    "MS"    : "Financials",
    "BLK"   : "Financials",
    "SCHW"  : "Financials",
    "AXP"   : "Financials",
    "SPGI"  : "Financials",
    "ICE"   : "Financials",
    "CME"   : "Financials",
    "C"     : "Financials",

    # ── Industrials ──────────────────────────
    "CAT"   : "Industrials",
    "RTX"   : "Industrials",
    "HON"   : "Industrials",
    "UPS"   : "Industrials",
    "BA"    : "Industrials",
    "DE"    : "Industrials",
    "GE"    : "Industrials",
    "MMM"   : "Industrials",
    "LMT"   : "Industrials",
    "NOC"   : "Industrials",
    "FDX"   : "Industrials",
    "EMR"   : "Industrials",
    "ETN"   : "Industrials",
    "PH"    : "Industrials",
    "ROK"   : "Industrials",

    # ── Energy ───────────────────────────────
    "XOM"   : "Energy",
    "CVX"   : "Energy",
    "COP"   : "Energy",
    "EOG"   : "Energy",
    "SLB"   : "Energy",
    "MPC"   : "Energy",
    "PSX"   : "Energy",
    "VLO"   : "Energy",
    "OXY"   : "Energy",
    "HAL"   : "Energy",

    # ── Materials ────────────────────────────
    "LIN"   : "Materials",
    "APD"   : "Materials",
    "SHW"   : "Materials",
    "ECL"   : "Materials",
    "NEM"   : "Materials",
    "FCX"   : "Materials",
    "NUE"   : "Materials",
    "VMC"   : "Materials",
    "MLM"   : "Materials",

    # ── Real Estate ──────────────────────────
    "PLD"   : "Real Estate",
    "AMT"   : "Real Estate",
    "EQIX"  : "Real Estate",
    "CCI"   : "Real Estate",
    "SPG"   : "Real Estate",
    "O"     : "Real Estate",
    "WELL"  : "Real Estate",
    "DLR"   : "Real Estate",

    # ── Utilities ────────────────────────────
    "NEE"   : "Utilities",
    "DUK"   : "Utilities",
    "SO"    : "Utilities",
    "D"     : "Utilities",
    "AEP"   : "Utilities",
    "EXC"   : "Utilities",
    "XEL"   : "Utilities",
}


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_data():
    if os.path.exists(US_PRICE_CACHE):
        print("Loading cached US data...")
        close  = pd.read_csv(US_PRICE_CACHE, index_col=0, parse_dates=True)
        volume = pd.read_csv(US_VOLUME_CACHE, index_col=0, parse_dates=True)
        idx    = pd.read_csv(US_INDEX_CACHE,  index_col=0, parse_dates=True)
        print(f"Loaded {close.shape[1]} stocks, {close.shape[0]} days.")
        return close, volume, idx["sp500"], idx["sp500"]

    print(f"Downloading {len(UNIVERSE)} US stocks...")
    print("First run — takes ~3 minutes.\n")

    tickers = list(UNIVERSE.keys())
    raw     = yf.download(
        tickers,
        start=DATA_FETCH_START,
        end=DATA_FETCH_END,
        auto_adjust=True,
        progress=True,
    )

    # Handle MultiIndex columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        close  = raw["Close"].copy()
        volume = raw["Volume"].copy()
    else:
        close  = raw[["Close"]].copy()
        volume = raw[["Volume"]].copy()

    close.to_csv(US_PRICE_CACHE)
    volume.to_csv(US_VOLUME_CACHE)

    # Fetch S&P 500 index
    sp500 = yf.download(REGIME_TICKER, start=DATA_FETCH_START,
                        end=DATA_FETCH_END, auto_adjust=True, progress=False)
    idx_df = pd.DataFrame({"sp500": sp500["Close"].squeeze()})
    idx_df.to_csv(US_INDEX_CACHE)

    print(f"\nData cached. {close.shape[1]} stocks loaded.")
    return close, volume, idx_df["sp500"], idx_df["sp500"]


# ─────────────────────────────────────────────
# SIGNAL FUNCTIONS (identical logic to India)
# ─────────────────────────────────────────────
def regime_on(sp500, date):
    d = sp500.loc[:date].dropna()
    if len(d) < REGIME_DMA:
        return "RISK-ON"
    return "RISK-ON" if d.iloc[-1] > d.rolling(REGIME_DMA).mean().iloc[-1] else "RISK-OFF"


def liquid_on(close, date, tickers):
    out = []
    for t in tickers:
        if t not in close.columns:
            continue
        c = close[t].loc[:date].dropna()
        if len(c) < 60 or c.iloc[-1] < MIN_PRICE:
            continue
        out.append(t)
    return out


def scores_on(close, date, tickers):
    rows    = []
    c_slice = close.loc[:date]

    for t in tickers:
        if t not in c_slice.columns:
            continue
        p = c_slice[t].dropna()
        if len(p) < LOOKBACK_12M + SKIP_RECENT:
            continue
        s      = SKIP_RECENT
        p_now  = p.iloc[-(s+1)]
        mom12  = (p_now - p.iloc[-(LOOKBACK_12M+s)]) / p.iloc[-(LOOKBACK_12M+s)]
        mom6   = (p_now - p.iloc[-(LOOKBACK_6M+s)])  / p.iloc[-(LOOKBACK_6M+s)]
        mom3   = (p_now - p.iloc[-(LOOKBACK_3M+s)])  / p.iloc[-(LOOKBACK_3M+s)]
        vol6   = p.iloc[-LOOKBACK_6M:].pct_change().dropna().std() * np.sqrt(252)
        rows.append({"ticker":t, "sector":UNIVERSE.get(t,"Unknown"),
                     "mom12":mom12, "mom6":mom6, "mom3":mom3, "vol6":vol6})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("ticker")
    def z(s): return (s-s.mean())/s.std() if s.std()>0 else s*0
    df["score"] = (W_MOM_12M*z(df["mom12"]) + W_MOM_6M*z(df["mom6"]) +
                   W_MOM_3M*z(df["mom3"])   + W_VOL*z(df["vol6"]))
    return df.sort_values("score", ascending=False)


def pick_portfolio(scored):
    sel, sc = [], {}
    for t, row in scored.iterrows():
        s = row["sector"]
        if sc.get(s,0) < MAX_PER_SECTOR:
            sel.append(t)
            sc[s] = sc.get(s,0)+1
        if len(sel) == TOP_N:
            break
    return sel


def above_dma(close, t, date):
    if t not in close.columns:
        return True
    p = close[t].loc[:date].dropna()
    if len(p) < DMA_EXIT:
        return True
    return p.iloc[-1] >= p.rolling(DMA_EXIT).mean().iloc[-1]


# ─────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────
def print_metrics(equity, label):
    ret    = equity.pct_change().dropna()
    total  = (equity.iloc[-1]/equity.iloc[0]) - 1
    n_yr   = len(ret)/12
    cagr   = (1+total)**(1/n_yr) - 1
    mrf    = (1+RISK_FREE_RATE)**(1/12) - 1
    exc    = ret - mrf
    sharpe = exc.mean()/exc.std()*np.sqrt(12) if exc.std()>0 else 0
    down   = exc[exc<0]
    sortino = exc.mean()/down.std()*np.sqrt(12) if len(down)>0 and down.std()>0 else 0
    maxdd  = ((equity - equity.cummax())/equity.cummax()).min()
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
    return {"cagr":cagr, "sharpe":sharpe, "maxdd":maxdd}


# ─────────────────────────────────────────────
# MAIN BACKTEST
# ─────────────────────────────────────────────
def run_backtest():
    close, volume, sp500, benchmark = load_data()

    all_dates    = close.loc[START_DATE:END_DATE].index
    month_starts = (pd.Series(all_dates)
                    .groupby(pd.Series(all_dates).dt.to_period("M"))
                    .first().values)

    cash     = float(INITIAL_CAPITAL)
    holdings = {}
    eq_curve = []
    tickers  = list(UNIVERSE.keys())

    print(f"\n{'='*55}")
    print(f"  US BACKTEST  |  {START_DATE} → {END_DATE}")
    print(f"  Capital      : ${cash:,.0f}  |  Stocks: {TOP_N}  |  Sector cap: {MAX_PER_SECTOR}")
    print(f"  Formula      : {W_MOM_12M}*z12M + {W_MOM_6M}*z6M + {W_MOM_3M}*z3M + {W_VOL}*zVol")
    print(f"{'='*55}\n")

    for rb_date in month_starts:
        date_str = pd.Timestamp(rb_date).strftime("%Y-%m-%d")
        regime   = regime_on(sp500, rb_date)

        # ── RISK-OFF: liquidate ──────────────────────────────────────────
        if regime == "RISK-OFF":
            for t, sh in list(holdings.items()):
                if sh > 0 and t in close.columns:
                    try:
                        px   = close[t].loc[:rb_date].dropna().iloc[-1]
                        cash += sh * px - txn_cost(sh * px, "sell")
                        holdings[t] = 0
                    except: pass
            eq_curve.append({"date":rb_date, "value":cash, "regime":"OFF"})
            print(f"{date_str} | RISK-OFF  | Cash : ${cash:>12,.0f}")
            continue

        # ── RISK-ON: signals ─────────────────────────────────────────────
        liquid   = liquid_on(close, rb_date, tickers)
        scored   = scores_on(close, rb_date, liquid)
        if scored.empty:
            eq_curve.append({"date":rb_date, "value":cash, "regime":"ON"})
            continue

        top25    = scored.head(EXIT_RANK_CUTOFF).index.tolist()
        new_port = pick_portfolio(scored)

        # Exits
        for t, sh in list(holdings.items()):
            if sh == 0: continue
            if (t not in top25) or (not above_dma(close, t, rb_date)):
                try:
                    px   = close[t].loc[:rb_date].dropna().iloc[-1]
                    cash += sh * px - txn_cost(sh * px, "sell")
                    holdings[t] = 0
                except: pass

        # Portfolio value
        port_val = cash
        for t, sh in holdings.items():
            if sh > 0 and t in close.columns:
                try: port_val += sh * close[t].loc[:rb_date].dropna().iloc[-1]
                except: pass

        target = port_val / TOP_N

        # Buys
        for t in new_port:
            try:
                px      = close[t].loc[:rb_date].dropna().iloc[-1]
                cur_val = holdings.get(t, 0) * px
                if cur_val < target * 0.95:
                    n       = int((target - cur_val) / px)
                    cost    = n * px
                    tc      = txn_cost(cost, "buy")
                    if n > 0 and cash >= cost + tc:
                        cash -= (cost + tc)
                        holdings[t] = holdings.get(t, 0) + n
            except: pass

        # Month value
        month_val = cash
        for t, sh in holdings.items():
            if sh > 0 and t in close.columns:
                try: month_val += sh * close[t].loc[:rb_date].dropna().iloc[-1]
                except: pass

        eq_curve.append({"date":rb_date, "value":month_val, "regime":"ON"})
        held = [t for t,s in holdings.items() if s>0]
        print(f"{date_str} | RISK-ON   | ${month_val:>12,.0f} | {', '.join(held[:7])}")

    # ── Results ───────────────────────────────────────────────────────────
    eq_df  = pd.DataFrame(eq_curve).set_index("date")
    eq_df.index = pd.to_datetime(eq_df.index)
    equity = eq_df["value"]

    bench_raw   = benchmark.loc[START_DATE:END_DATE].resample("MS").first().dropna()
    bench_curve = (bench_raw / bench_raw.iloc[0]) * INITIAL_CAPITAL

    m_s = print_metrics(equity,      "US STRATEGY  — Momentum (S&P 500 universe)")
    m_b = print_metrics(bench_curve, "US BENCHMARK — S&P 500 Buy & Hold")

    print(f"\n{'='*48}")
    print(f"  US vs India Comparison")
    print(f"{'='*48}")
    print(f"  US Strategy CAGR    : {m_s['cagr']*100:.1f}%")
    print(f"  US Benchmark CAGR   : {m_b['cagr']*100:.1f}%")
    print(f"  US Outperformance   : {(m_s['cagr']-m_b['cagr'])*100:+.1f}% per year")
    print(f"\n  India Strategy CAGR : ~54% (2020-2024 backtest)")
    print(f"  India Benchmark CAGR: ~15% (Nifty 50)")
    print(f"  India Outperformance: ~+39% per year")
    print(f"{'='*48}")

    equity.to_csv("us_equity_curve.csv")
    print("\nSaved: us_equity_curve.csv")


if __name__ == "__main__":
    run_backtest()
"""
backtester_uk.py
================
Same Statistical Quality + Momentum strategy applied to UK markets.
Universe  : FTSE 100 + select FTSE 250 stocks (yfinance .L suffix)
Regime    : FTSE 100 (^FTSE) vs 200-DMA
Benchmark : FTSE 100 Buy & Hold
Period    : 2018-01-01 to 2026-05-22

Purpose   : Comparison — does momentum work in UK markets?
            UK sits between India (inefficient) and US (very efficient).
            FTSE 100 heavy in energy/mining/banks — mean-reverting sectors.
            FTSE 250 has more growth names where momentum persists.

UK Transaction costs (significantly higher than US):
  Stamp Duty Reserve Tax : 0.5% on BUYS only (10x India's stamp duty)
  Broker commission      : £0 (zero commission brokers) to £3/trade
  No STT equivalent      : unlike India

Run:
    python backtester_uk.py
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
MIN_PRICE        = 50        # pence — filters very cheap stocks
RISK_FREE_RATE   = 0.045     # 4.5% — UK base rate approx

# ─────────────────────────────────────────────
# UK TRANSACTION COSTS
# Stamp Duty Reserve Tax is the dominant cost
# Much higher than US, comparable to India overall
# ─────────────────────────────────────────────
SDRT_BUY    = 0.005     # 0.5% stamp duty on buys — UK specific
BROKER_FEE  = 0.001     # 0.1% broker commission both sides (conservative)

def txn_cost(value, side):
    cost = value * BROKER_FEE
    if side == "buy":
        cost += value * SDRT_BUY    # 0.5% stamp duty on buys only
    return cost

# ─────────────────────────────────────────────
# BACKTEST SETTINGS
# ─────────────────────────────────────────────
START_DATE       = "2016-01-01"
END_DATE         = "2026-05-22"
INITIAL_CAPITAL  = 100000       # £ pounds
DATA_FETCH_START = "2016-01-01" # extra buffer for DMA calculations
DATA_FETCH_END   = "2026-05-24"

UK_PRICE_CACHE   = "uk_price_cache.csv"
UK_VOLUME_CACHE  = "uk_volume_cache.csv"
UK_INDEX_CACHE   = "uk_index_cache.csv"

REGIME_TICKER    = "^FTSE"     # FTSE 100
BENCHMARK_TICKER = "^FTSE"     # FTSE 100 buy and hold

# ─────────────────────────────────────────────
# FTSE 100 + FTSE 250 UNIVERSE
# All UK tickers use .L suffix on yfinance
# ─────────────────────────────────────────────
UNIVERSE = {
    # ── Energy ──────────────────────────────
    "SHEL.L"  : "Energy",       # Shell
    "BP.L"    : "Energy",       # BP
    "TLW.L"   : "Energy",       # Tullow Oil

    # ── Mining & Materials ───────────────────
    "RIO.L"   : "Mining",       # Rio Tinto
    "GLEN.L"  : "Mining",       # Glencore
    "AAL.L"   : "Mining",       # Anglo American
    "ANTO.L"  : "Mining",       # Antofagasta
    "FRES.L"  : "Mining",       # Fresnillo
    "NEM.L"   : "Mining",       # Newmont (listed London)

    # ── Financials — Banks ───────────────────
    "HSBA.L"  : "Banks",        # HSBC
    "BARC.L"  : "Banks",        # Barclays
    "LLOY.L"  : "Banks",        # Lloyds
    "NWG.L"   : "Banks",        # NatWest
    "STAN.L"  : "Banks",        # Standard Chartered
    "VMUK.L"  : "Banks",        # Virgin Money

    # ── Financials — Insurance ───────────────
    "AV.L"    : "Insurance",    # Aviva
    "LGEN.L"  : "Insurance",    # Legal & General
    "PRU.L"   : "Insurance",    # Prudential
    "SBRE.L"  : "Insurance",    # Sabre Insurance
    "DLG.L"   : "Insurance",    # Direct Line

    # ── Financials — Asset Management ────────
    "HLMA.L"  : "Financials",   # Halma
    "EXPN.L"  : "Financials",   # Experian
    "ICG.L"   : "Financials",   # Intermediate Capital
    "MNG.L"   : "Financials",   # M&G

    # ── Healthcare & Pharma ──────────────────
    "AZN.L"   : "Healthcare",   # AstraZeneca — UK's biggest momentum winner
    "GSK.L"   : "Healthcare",   # GSK
    "HIK.L"   : "Healthcare",   # Hikma
    "COB.L"   : "Healthcare",   # Cobham (defence/health)
    "SGRO.L"  : "Healthcare",   # Segro (REIT — healthcare real estate)

    # ── Consumer Staples ─────────────────────
    "ULVR.L"  : "Consumer Staples",  # Unilever
    "DGE.L"   : "Consumer Staples",  # Diageo
    "IMB.L"   : "Consumer Staples",  # Imperial Brands
    "BATS.L"  : "Consumer Staples",  # British American Tobacco
    "TSCO.L"  : "Consumer Staples",  # Tesco
    "SBRY.L"  : "Consumer Staples",  # Sainsbury's
    "MKS.L"   : "Consumer Staples",  # M&S

    # ── Consumer Discretionary ───────────────
    "NXT.L"   : "Consumer Disc",     # Next
    "JD.L"    : "Consumer Disc",     # JD Sports
    "WPP.L"   : "Consumer Disc",     # WPP
    "IPG.L"   : "Consumer Disc",     # Inchcape
    "AUTO.L"  : "Consumer Disc",     # Auto Trader
    "OCDO.L"  : "Consumer Disc",     # Ocado

    # ── Industrials ──────────────────────────
    "RR.L"    : "Industrials",   # Rolls-Royce — massive momentum winner 2023-24
    "BAE.L"   : "Industrials",   # BAE Systems
    "IMI.L"   : "Industrials",   # IMI
    "WEIR.L"  : "Industrials",   # Weir Group
    "SN.L"    : "Industrials",   # Smith & Nephew
    "SPX.L"   : "Industrials",   # Spirax-Sarco
    "SDRC.L"  : "Industrials",   # Serco
    "GKN.L"   : "Industrials",   # Melrose (GKN)

    # ── Technology ───────────────────────────
    "SAGE.L"  : "Technology",   # Sage Group
    "CTEC.L"  : "Technology",   # ConvaTec
    "MONY.L"  : "Technology",   # Moneysupermarket
    "FDM.L"   : "Technology",   # FDM Group
    "DPLM.L"  : "Technology",   # Diploma

    # ── Telecom ──────────────────────────────
    "VOD.L"   : "Telecom",      # Vodafone
    "BT-A.L"  : "Telecom",      # BT Group

    # ── Utilities ────────────────────────────
    "NG.L"    : "Utilities",    # National Grid
    "SSE.L"   : "Utilities",    # SSE
    "UU.L"    : "Utilities",    # United Utilities
    "SVT.L"   : "Utilities",    # Severn Trent
    "CNA.L"   : "Utilities",    # Centrica

    # ── Real Estate ──────────────────────────
    "LAND.L"  : "Real Estate",  # Land Securities
    "BLND.L"  : "Real Estate",  # British Land
    "LXI.L"   : "Real Estate",  # LXI REIT

    # ── Aerospace & Defence ──────────────────
    "QQ.L"    : "Defence",      # QinetiQ
    "COG.L"   : "Defence",      # Chemring
}


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_data():
    if os.path.exists(UK_PRICE_CACHE):
        print("Loading cached UK data...")
        close  = pd.read_csv(UK_PRICE_CACHE, index_col=0, parse_dates=True)
        volume = pd.read_csv(UK_VOLUME_CACHE, index_col=0, parse_dates=True)
        idx    = pd.read_csv(UK_INDEX_CACHE,  index_col=0, parse_dates=True)
        print(f"Loaded {close.shape[1]} stocks, {close.shape[0]} days.")
        return close, volume, idx["ftse100"], idx["ftse100"]

    tickers = list(UNIVERSE.keys())
    print(f"Downloading {len(tickers)} UK stocks (.L suffix)...")
    print("First run — takes ~3 minutes.\n")

    raw = yf.download(
        tickers,
        start=DATA_FETCH_START,
        end=DATA_FETCH_END,
        auto_adjust=True,
        progress=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        close  = raw["Close"].copy()
        volume = raw["Volume"].copy()
    else:
        close  = raw[["Close"]].copy()
        volume = raw[["Volume"]].copy()

    # Remove .L suffix from column names for cleaner display
    close.columns  = [c.replace(".L", "") if isinstance(c, str) else c for c in close.columns]
    volume.columns = [c.replace(".L", "") if isinstance(c, str) else c for c in volume.columns]

    close.to_csv(UK_PRICE_CACHE)
    volume.to_csv(UK_VOLUME_CACHE)

    ftse = yf.download(REGIME_TICKER, start=DATA_FETCH_START,
                       end=DATA_FETCH_END, auto_adjust=True, progress=False)
    idx_df = pd.DataFrame({"ftse100": ftse["Close"].squeeze()})
    idx_df.to_csv(UK_INDEX_CACHE)

    print(f"\nData cached. {close.shape[1]} stocks loaded.")
    return close, volume, idx_df["ftse100"], idx_df["ftse100"]


# ─────────────────────────────────────────────
# SIGNAL FUNCTIONS
# ─────────────────────────────────────────────
def regime_on(ftse, date):
    d = ftse.loc[:date].dropna()
    if len(d) < REGIME_DMA:
        return "RISK-ON"
    return "RISK-ON" if d.iloc[-1] > d.rolling(REGIME_DMA).mean().iloc[-1] else "RISK-OFF"


def liquid_on(close, date, tickers):
    out = []
    for t in tickers:
        # Try with and without .L suffix in column names
        col = t.replace(".L", "") if t.endswith(".L") else t
        if col not in close.columns:
            continue
        c = close[col].loc[:date].dropna()
        if len(c) < 60 or c.iloc[-1] < MIN_PRICE:
            continue
        out.append(col)
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

        # Map back to sector using original ticker
        orig_ticker = t + ".L" if not t.endswith(".L") else t
        sector = UNIVERSE.get(orig_ticker, UNIVERSE.get(t, "Unknown"))

        rows.append({"ticker":t, "sector":sector,
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
    ret     = equity.pct_change().dropna()
    total   = (equity.iloc[-1]/equity.iloc[0]) - 1
    n_yr    = len(ret)/12
    cagr    = (1+total)**(1/n_yr) - 1
    mrf     = (1+RISK_FREE_RATE)**(1/12) - 1
    exc     = ret - mrf
    sharpe  = exc.mean()/exc.std()*np.sqrt(12) if exc.std()>0 else 0
    down    = exc[exc<0]
    sortino = exc.mean()/down.std()*np.sqrt(12) if len(down)>0 and down.std()>0 else 0
    maxdd   = ((equity - equity.cummax())/equity.cummax()).min()
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
    close, volume, ftse, benchmark = load_data()

    all_dates    = close.loc[START_DATE:END_DATE].index
    month_starts = (pd.Series(all_dates)
                    .groupby(pd.Series(all_dates).dt.to_period("M"))
                    .first().values)

    cash     = float(INITIAL_CAPITAL)
    holdings = {}
    eq_curve = []
    tickers  = list(UNIVERSE.keys())

    print(f"\n{'='*55}")
    print(f"  UK BACKTEST  |  {START_DATE} → {END_DATE}")
    print(f"  Capital      : £{cash:,.0f}  |  Stocks: {TOP_N}  |  Sector cap: {MAX_PER_SECTOR}")
    print(f"  Costs        : 0.5% SDRT on buys + 0.1% broker both sides")
    print(f"  Formula      : {W_MOM_12M}*z12M + {W_MOM_6M}*z6M + {W_MOM_3M}*z3M + {W_VOL}*zVol")
    print(f"{'='*55}\n")

    for rb_date in month_starts:
        date_str = pd.Timestamp(rb_date).strftime("%Y-%m-%d")
        regime   = regime_on(ftse, rb_date)

        # ── RISK-OFF ─────────────────────────────────────────────────────
        if regime == "RISK-OFF":
            for t, sh in list(holdings.items()):
                if sh > 0 and t in close.columns:
                    try:
                        px   = close[t].loc[:rb_date].dropna().iloc[-1]
                        cash += sh * px - txn_cost(sh * px, "sell")
                        holdings[t] = 0
                    except: pass
            eq_curve.append({"date":rb_date, "value":cash, "regime":"OFF"})
            print(f"{date_str} | RISK-OFF  | Cash : £{cash:>12,.0f}")
            continue

        # ── RISK-ON ──────────────────────────────────────────────────────
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
        print(f"{date_str} | RISK-ON   | £{month_val:>12,.0f} | {', '.join(held[:7])}")

    # ── Results ───────────────────────────────────────────────────────────
    eq_df  = pd.DataFrame(eq_curve).set_index("date")
    eq_df.index = pd.to_datetime(eq_df.index)
    equity = eq_df["value"]

    bench_raw   = benchmark.loc[START_DATE:END_DATE].resample("MS").first().dropna()
    bench_curve = (bench_raw / bench_raw.iloc[0]) * INITIAL_CAPITAL

    m_s = print_metrics(equity,      "UK STRATEGY  — Momentum (FTSE 100/250 universe)")
    m_b = print_metrics(bench_curve, "UK BENCHMARK — FTSE 100 Buy & Hold")

    print(f"\n{'='*55}")
    print(f"  Three-Market Comparison (2018–2026)")
    print(f"{'='*55}")
    print(f"  India Strategy CAGR    : ~32%  | vs Nifty 50 ~11%  | Alpha: +21%")
    print(f"  UK    Strategy CAGR    : {m_s['cagr']*100:.1f}%  | vs FTSE  {m_b['cagr']*100:.1f}%  | Alpha: {(m_s['cagr']-m_b['cagr'])*100:+.1f}%")
    print(f"  US    Strategy CAGR    : ~11%  | vs S&P   ~13%  | Alpha: -2%")
    print(f"{'='*55}")

    equity.to_csv("uk_equity_curve.csv")
    print("\nSaved: uk_equity_curve.csv")


if __name__ == "__main__":
    run_backtest()
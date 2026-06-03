"""
backtester.py  [DAILY EXIT VERSION]
=====================================
Walk-forward backtest with daily exit monitoring.

Key improvement over monthly-only version:
  - Regime check runs EVERY trading day
  - DMA exit (cfg.DMA_EXIT days) checked EVERY trading day
  - Rank exit checked EVERY trading day (using daily-recomputed scores)
  - Exit triggers immediate replacement buy — no cash sitting idle
  - Full portfolio rotation on first trading day of each month only
  - Score computation DAILY — matches live execution.py behaviour
  - DMA exit uses cfg.DMA_EXIT (250) — not 100

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
    brokerage = 0.0                                   # Zerodha CNC delivery = ₹0
    exch      = value * cfg.EXCHANGE_CHARGE
    sebi      = value * cfg.SEBI_CHARGE
    stt       = value * (cfg.STT_BUY if side == "buy" else cfg.STT_SELL)   # both sides for delivery
    stamp     = value * cfg.STAMP_DUTY if side == "buy"  else 0.0
    dp        = 15.93                  if side == "sell" else 0.0          # flat DP charge per sell
    gst       = (brokerage + exch + sebi) * getattr(cfg, "GST_RATE", 0.0) # 18% GST on charges
    return brokerage + exch + sebi + stt + stamp + dp + gst

_TAX_CHANGE        = pd.Timestamp("2024-07-23")   # STCG 15→20%, LTCG 10→12.5%
_LTCG_TAXABLE_FROM = pd.Timestamp("2018-04-01")   # before this, listed-equity LTCG was EXEMPT (Sec 10(38))

def _stcg_rate(sell_date):
    return cfg.STCG_RATE_POST if pd.Timestamp(sell_date) >= _TAX_CHANGE else cfg.STCG_RATE_PRE

def _ltcg_rate(sell_date):
    d = pd.Timestamp(sell_date)
    if d < _LTCG_TAXABLE_FROM:
        return 0.0                                # listed-equity LTCG exempt pre-2018
    return cfg.LTCG_RATE_POST if d >= _TAX_CHANGE else cfg.LTCG_RATE_PRE


def capital_gains_tax(gain, entry_date, sell_date):
    """Per-trade STCG/LTCG tax on a realised gain. Returns 0 for losses.
    Used only when cfg.ANNUAL_TAX is False (older, conservative behaviour)."""
    if gain <= 0:
        return 0.0
    days_held = (pd.Timestamp(sell_date) - pd.Timestamp(entry_date)).days
    rate = _ltcg_rate(sell_date) if days_held >= 365 else _stcg_rate(sell_date)
    return gain * rate


def _financial_year(date):
    """Indian financial year (Apr–Mar). Returns the starting calendar year."""
    d = pd.Timestamp(date)
    return d.year if d.month >= 4 else d.year - 1


class TaxLedger:
    """
    Accrues realised capital gains/losses and settles tax once per Indian
    financial year (Apr–Mar), the way it actually works:
      - short-term (held < 365d) and long-term (>= 365d) gains are netted
        within their buckets;
      - a net short-term loss is set off against long-term gains;
      - the annual LTCG exemption is applied once to net long-term gains;
      - rates follow the 2024-07-23 change (post rates used for FY2024-25+).
    This lets gains compound within the year instead of being skimmed per trade.
    """

    def __init__(self):
        self._buckets  = {}         # fy_start_year -> list[(gain, is_long, sell_date)]
        self.settled   = []         # list of dicts, for reporting
        self.carry_st  = 0.0        # carried-forward short-term capital loss (<=0)
        self.carry_lt  = 0.0        # carried-forward long-term capital loss (<=0)

    def record(self, gain, entry_date, sell_date):
        days_held = (pd.Timestamp(sell_date) - pd.Timestamp(entry_date)).days
        fy = _financial_year(sell_date)
        self._buckets.setdefault(fy, []).append((gain, days_held >= 365, pd.Timestamp(sell_date)))

    @staticmethod
    def _weighted_rate(lots, rate_fn):
        """Gain-weighted effective rate over the profitable lots, using each lot's
        own sell-date rate (handles the 2018 LTCG-exemption and 2024 rate changes
        within a single financial year)."""
        gpos = [(g, d) for g, d in lots if g > 0]
        tot  = sum(g for g, _ in gpos)
        if tot <= 0:
            return 0.0
        return sum(g * rate_fn(d) for g, d in gpos) / tot

    def _compute(self, fy, lots):
        short = [(g, d) for g, is_long, d in lots if not is_long]
        long  = [(g, d) for g, is_long, d in lots if is_long]
        short_net = sum(g for g, _ in short) + self.carry_st   # apply carried STCL
        long_net  = sum(g for g, _ in long)  + self.carry_lt   # apply carried LTCL
        self.carry_st = self.carry_lt = 0.0

        # Net short-term loss sets off against long-term gain; residual carries fwd.
        if short_net < 0:
            absorbed   = min(-short_net, max(0.0, long_net))
            long_net  -= absorbed
            short_net += absorbed
            self.carry_st = short_net            # remaining STCL (<=0) carries forward
            short_net = 0.0
        # Net long-term loss can only offset LTCG → carries forward.
        if long_net < 0:
            self.carry_lt = long_net
            long_net = 0.0

        exempt = cfg.LTCG_EXEMPTION_POST if fy >= 2024 else cfg.LTCG_EXEMPTION_PRE
        stcg_r = self._weighted_rate(short, _stcg_rate)
        ltcg_r = self._weighted_rate(long,  _ltcg_rate)   # 0% pre-2018 (exempt)
        long_taxable  = max(0.0, long_net - exempt)
        short_taxable = max(0.0, short_net)
        return short_taxable * stcg_r + long_taxable * ltcg_r

    def settle_completed(self, current_date):
        """Settle every financial year strictly before current_date's FY.
        Returns total tax to deduct now."""
        cur_fy = _financial_year(current_date)
        total  = 0.0
        for fy in sorted(self._buckets):
            if fy < cur_fy:
                tax = self._compute(fy, self._buckets.pop(fy))
                if tax > 0:
                    self.settled.append({"fy": f"{fy}-{fy+1}", "tax": tax})
                    total += tax
        return total

    def settle_all(self):
        """Settle any remaining (current/open) financial year at backtest end."""
        total = 0.0
        for fy in sorted(self._buckets):
            tax = self._compute(fy, self._buckets.pop(fy))
            if tax > 0:
                self.settled.append({"fy": f"{fy}-{fy+1}", "tax": tax})
                total += tax
        return total


def realize_tax(ledger, gain, entry_date, sell_date):
    """Single entry point for a realised sale's tax.
    Annual mode: defer to the ledger (settled per FY), charge 0 at trade time.
    Per-trade mode: charge STCG/LTCG immediately (legacy conservative path)."""
    if getattr(cfg, "ANNUAL_TAX", True):
        ledger.record(gain, entry_date, sell_date)
        return 0.0
    return capital_gains_tax(gain, entry_date, sell_date)


def realize_lots(ledger, lots_t, fill_px, sell_date):
    """Close a fully-sold position FIFO lot-by-lot: each acquisition lot is taxed on
    its OWN cost basis and holding period (real demat FIFO), instead of one merged
    average price + earliest date. Returns immediate tax (0 in annual mode)."""
    tax = 0.0
    for sh, price, date in (lots_t or []):
        tax += realize_tax(ledger, (fill_px - price) * sh, date, sell_date)
    return tax


# ── Data loading ───────────────────────────────────────────────────────────
def load_nse_bhavcopy_cache():
    """Load the survivorship-bias-free cache built from NSE bhavcopy archives
    (nse_data/build_caches.py). Same return shape as load_data()."""
    print("Loading BIAS-FREE NSE bhavcopy cache…")
    # sort_index() everywhere: a rebuilt cache can come back with an out-of-order
    # (non-monotonic) date index, which breaks label-based date slicing (.loc[s:e]).
    close  = pd.read_csv(cfg.NSE_PRICE_CACHE,  index_col=0, parse_dates=True).sort_index()
    open_p = (pd.read_csv(cfg.NSE_OPEN_CACHE,  index_col=0, parse_dates=True).sort_index()
              if os.path.exists(cfg.NSE_OPEN_CACHE) else close.copy())
    volume = (pd.read_csv(cfg.NSE_VOLUME_CACHE, index_col=0, parse_dates=True).sort_index()
              if os.path.exists(cfg.NSE_VOLUME_CACHE) else close.copy())
    # Index/regime series carry NO survivorship bias, so prefer the longer
    # yfinance regime cache (2006→) over the NSE index cache (only 2015→, since
    # NSE's ind_close_all archive doesn't exist pre-2015). This keeps the regime
    # filter active through 2008/2011 while equity prices stay bias-free.
    regime_path = cfg.REGIME_CACHE if os.path.exists(cfg.REGIME_CACHE) else cfg.NSE_REGIME_CACHE
    print(f"  regime/benchmark series from {regime_path}")
    regime = pd.read_csv(regime_path, index_col=0, parse_dates=True)
    regime = regime[~regime.index.duplicated(keep="last")].sort_index()
    n500 = regime["nifty500"]
    n50  = regime["nifty50"]      if "nifty50"      in regime else n500
    n100 = regime["nifty100"]     if "nifty100"     in regime else n500
    nmid = regime["nifty_midcap"] if "nifty_midcap" in regime else n500
    print(f"  {close.shape[1]} symbols × {close.shape[0]} days, "
          f"{close.index.min().date()} → {close.index.max().date()}")
    return close, volume, n500, n50, n100, nmid, open_p


def load_data():
    if getattr(cfg, "USE_NSE_BHAVCOPY", False) and os.path.exists(cfg.NSE_PRICE_CACHE):
        return load_nse_bhavcopy_cache()

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
    open_prices = raw["Open"].copy()
    close.columns       = [c.replace(".NS","") for c in close.columns]
    volume.columns      = [c.replace(".NS","") for c in volume.columns]
    open_prices.columns = [c.replace(".NS","") for c in open_prices.columns]
    close.to_csv(cfg.DATA_CACHE_FILE)
    volume.to_csv(cfg.VOLUME_CACHE)
    open_prices.to_csv(cfg.OPEN_CACHE)

    r500 = yf.download(cfg.REGIME_TICKER,    start=cfg.DATA_FETCH_START,
                       end=cfg.DATA_FETCH_END, auto_adjust=True, progress=False)
    r50  = yf.download(cfg.BENCHMARK_TICKER, start=cfg.DATA_FETCH_START,
                       end=cfg.DATA_FETCH_END, auto_adjust=True, progress=False)
    r100 = yf.download("^CNX100",            start=cfg.DATA_FETCH_START,
                       end=cfg.DATA_FETCH_END, auto_adjust=True, progress=False)
    rmid = yf.download("^CNXMID",            start=cfg.DATA_FETCH_START,
                       end=cfg.DATA_FETCH_END, auto_adjust=True, progress=False)
    rd   = pd.DataFrame({
        "nifty500":    r500["Close"].squeeze(),
        "nifty50":     r50["Close"].squeeze(),
        "nifty100":    r100["Close"].squeeze() if not r100.empty else r500["Close"].squeeze(),
        "nifty_midcap":rmid["Close"].squeeze() if not rmid.empty else r500["Close"].squeeze(),
    })
    rd.to_csv(cfg.REGIME_CACHE)
    nifty100  = rd["nifty100"]
    nifty_mid = rd["nifty_midcap"]
    return close, volume, rd["nifty500"], rd["nifty50"], nifty100, nifty_mid, open_prices


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

def is_above_exit_dma(close, ticker, date):
    if not getattr(cfg, "USE_DMA", True):     # DMA filter off → no entry gate
        return True
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


def pick_portfolio(scored, close, day):
    sel, sc = [], {}
    for t, row in scored.iterrows():
        if not is_above_exit_dma(close, t, day):
            continue
        s = row["sector"]
        if sc.get(s,0) < cfg.MAX_PER_SECTOR:
            sel.append(t)
            sc[s] = sc.get(s,0)+1
        if len(sel) == cfg.TOP_N:
            break
    return sel


def find_replacement(scored, current_holdings, exits, close, day, universe=None):
    """
    Finds immediate replacements for exited stocks.
    Same logic as execution.py — picks best available from top 25
    respecting sector cap and existing holdings.
    """
    if scored.empty:
        return []

    if universe is None:
        universe = UNIVERSE

    remaining    = [t for t in current_holdings if t not in exits]
    top_25       = scored.head(cfg.EXIT_RANK_CUTOFF).index.tolist()
    candidates   = [t for t in top_25 if t not in remaining and is_above_exit_dma(close, t, day)]

    sector_count = {}
    for t in remaining:
        s = universe.get(t, "Unknown")
        sector_count[s] = sector_count.get(s, 0) + 1

    replacements = []
    for t in candidates:
        s = universe.get(t, "Unknown")
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

def _apply_slippage(price, side):
    """Adverse slippage + half bid-ask spread on the fill: buys fill higher,
    sells fill lower."""
    bps = (getattr(cfg, "SLIPPAGE_BPS", 0) + getattr(cfg, "SPREAD_BPS", 0)) / 10000.0
    if bps <= 0:
        return price
    return price * (1 + bps) if side == "buy" else price * (1 - bps)


def get_fill_price(close_price, open_price, side, rank=0):
    """
    Realistic fill model using next day open price, then adverse slippage.
    Buy:  fills at open if open <= close x buffer, else missed (gap-up)
    Sell: fills at open if open >= close x (1-SELL_BUFFER), else missed (gap-down)
    No next-session open (open_price == 0 → suspended/halt/no data) is a MISS, not
    a forced fill at a stale close. Returns None when the order would not fill.
    """
    if open_price <= 0:
        return None                      # no tradeable open → order does not fill
    if side == "buy":
        if rank < 5:
            buffer = cfg.BUY_BUFFER_TOP5
        elif rank < 12:
            buffer = cfg.BUY_BUFFER_MID
        else:
            buffer = cfg.BUY_BUFFER_REST
        limit = close_price * (1 + buffer)
        if open_price > limit:
            return None                  # gap-up past limit → missed
        # Slippage can never push the fill ABOVE the limit on a buy.
        return min(_apply_slippage(open_price, "buy"), limit)
    else:
        limit = close_price * (1 - cfg.SELL_BUFFER)
        if open_price < limit:
            return None                  # gap-down past limit → missed
        # Slippage can never push the fill BELOW the limit on a sell.
        return max(_apply_slippage(open_price, "sell"), limit)
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


# ── Point-in-time universe loader ─────────────────────────────────────────
def load_universe_history():
    """
    Loads universe_history.csv if present.
    Returns a list of (effective_date, {ticker: sector}) sorted by date,
    or None if the file doesn't exist (falls back to hardcoded UNIVERSE).
    """
    path = "universe_history.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["effective_date"])
    history = []
    for date, grp in df.groupby("effective_date"):
        snapshot = dict(zip(grp["ticker"], grp["sector"]))
        history.append((pd.Timestamp(date), snapshot))
    history.sort(key=lambda x: x[0])
    print(f"Loaded universe history: {len(history)} snapshots, "
          f"{len(set(t for _, u in history for t in u))} unique tickers")

    # Staleness check — flag BOTH a stale tail (newest snapshot long before the
    # backtest end) AND large INTERNAL gaps between snapshots inside the window
    # (where the universe is frozen and membership/survivorship bias creeps in).
    GAP_DAYS = 270   # > ~2 missed semi-annual rebalances
    start, end = pd.Timestamp(cfg.START_DATE), pd.Timestamp(cfg.END_DATE)
    dates = [d for d, _ in history]
    latest = dates[-1]
    if (end - latest).days > GAP_DAYS:
        print(f"⚠ Universe TAIL stale: newest snapshot {latest.date()}, backtest ends "
              f"{end.date()} ({(end-latest).days}d frozen).")
    # internal gaps overlapping the [start, end] window
    gaps = []
    for a, b in zip(dates, dates[1:]):
        if b > start and a < end and (b - a).days > GAP_DAYS:
            gaps.append((a.date(), b.date(), (b - a).days))
    if gaps:
        print(f"⚠ Universe has {len(gaps)} large INTERNAL gap(s) inside the test window "
              f"(frozen membership → residual survivorship bias):")
        for a, b, d in gaps[:5]:
            print(f"     {a} → {b}  ({d}d, ~{d//182} missed rebalances)")
        print( "   Fill with semi-annual constituent files via tools/merge_constituents.py "
               "(see nse_data/README.md).")
    return history


def get_universe_for_date(date, history):
    """Returns the most recent universe snapshot on or before date."""
    ts = pd.Timestamp(date)
    current = history[0][1]
    for snap_date, snapshot in history:
        if snap_date <= ts:
            current = snapshot
        else:
            break
    return current


def build_universe_benchmark(close, all_days, valid_tickers, history, dynamic):
    """
    Equal-weight, daily-rebalanced total-return index of the *same* universe the
    strategy picks from, built from the *same* adjusted prices. This is the fair
    benchmark: identical data basis (so no price-vs-total-return mismatch) and
    point-in-time membership (so no survivorship edge). Answers "did stock
    selection beat just owning the whole universe equally?".
    """
    px   = close[valid_tickers].reindex(all_days).ffill()
    rets = px.pct_change()
    level, out = float(cfg.INITIAL_CAPITAL), []
    for i, day in enumerate(all_days):
        if i == 0:
            out.append(level)
            continue
        members = set(get_universe_for_date(day, history)) if dynamic else set(valid_tickers)
        cols    = [t for t in valid_tickers if t in members]
        r       = rets.loc[day, cols].dropna() if cols else pd.Series(dtype=float)
        level  *= (1 + (r.mean() if len(r) else 0.0))
        out.append(level)
    return pd.Series(out, index=all_days)


# ── Monte-Carlo trade removal ────────────────────────────────────────────────
# When MC_ENTRY_SKIP_PROB > 0, each attempt to ESTABLISH A NEW POSITION is skipped
# with that probability (capital stays in cash — NOT swept into other holdings).
# This re-simulates the true divergent path of "what if I'd randomly missed p% of
# my entries", the exact ground-truth version of the trade-removal robustness test.
# Set MC_ENTRY_SKIP_PROB + MC_ENTRY_SKIP_SEED, then call run_backtest(save=False).
MC_ENTRY_SKIP_PROB = 0.0
MC_ENTRY_SKIP_SEED = None
_mc_rng   = None
_mc_stats = {"attempted": 0, "skipped": 0}


def _mc_skip_new_entry():
    """For Monte-Carlo trade removal: True ⇒ skip establishing this new position.
    Only meaningful when the caller is about to open a brand-new holding (shares==0)."""
    global _mc_stats
    if MC_ENTRY_SKIP_PROB <= 0 or _mc_rng is None:
        return False
    _mc_stats["attempted"] += 1
    if _mc_rng.random() < MC_ENTRY_SKIP_PROB:
        _mc_stats["skipped"] += 1
        return True
    return False


# ── Main backtest ──────────────────────────────────────────────────────────
def run_backtest(save=True):
    global _mc_rng, _mc_stats
    _mc_rng   = np.random.default_rng(MC_ENTRY_SKIP_SEED) if MC_ENTRY_SKIP_PROB > 0 else None
    _mc_stats = {"attempted": 0, "skipped": 0}

    close, volume, nifty500, nifty50, nifty100, nifty_mid, open_prices = load_data()

    # Load point-in-time universe history (fixes survivorship bias).
    # Falls back to hardcoded UNIVERSE if universe_history.csv doesn't exist.
    universe_history = load_universe_history()
    using_dynamic_universe = universe_history is not None
    if using_dynamic_universe:
        all_historical_tickers = list({t for _, u in universe_history for t in u})
        print(f"Dynamic universe: {len(all_historical_tickers)} total historical tickers")
    else:
        all_historical_tickers = list(UNIVERSE.keys())
        print("Static universe (survivorship bias present — run tools/build_universe_history.py to fix)")

    # All trading days in backtest window
    all_days = close.loc[cfg.START_DATE:cfg.END_DATE].index
    tickers  = all_historical_tickers

    # ── Survivorship-gap report ──────────────────────────────────────────────
    # Universe members with NO price data are silently un-tradeable. These are
    # mostly delisted/acquired names (yfinance drops them), so their absence is
    # residual survivorship bias. We log it so the bias is measured, not hidden.
    valid_tickers = [t for t in tickers if t in close.columns]
    dropped       = sorted(set(tickers) - set(valid_tickers))
    if dropped:
        print(f"\n⚠ Survivorship gap: {len(dropped)}/{len(tickers)} "
              f"({100*len(dropped)/len(tickers):.0f}%) universe tickers have NO price "
              f"data and are excluded from the backtest.")
        print(f"   e.g. {', '.join(dropped[:12])}{' …' if len(dropped) > 12 else ''}")
        print( "   → these are mostly delisted names; results are optimistic by their absence.\n")

    # ── Pre-compute score matrix + above-DMA flags (one-time, O(N×S)) ────────
    # Replaces per-day compute_scores_on() which was O(N²×S) overall.
    S             = len(valid_tickers)
    t_to_i        = {t: i for i, t in enumerate(valid_tickers)}

    full_close = close[valid_tickers].ffill().values.astype(np.float64)  # (N_all, S)
    full_dates = close.index
    N_all      = len(full_dates)
    date_to_di = {d: i for i, d in enumerate(full_dates)}

    min_hist  = cfg.LOOKBACK_12M + cfg.SKIP_RECENT
    valid_d   = np.arange(min_hist, N_all)
    sk        = cfg.SKIP_RECENT

    p_now = full_close[valid_d - sk]
    p_12m = full_close[valid_d - cfg.LOOKBACK_12M - sk]
    p_6m  = full_close[valid_d - cfg.LOOKBACK_6M  - sk]
    p_3m  = full_close[valid_d - cfg.LOOKBACK_3M  - sk]

    with np.errstate(invalid="ignore", divide="ignore"):
        m12 = (p_now - p_12m) / p_12m
        m6  = (p_now - p_6m)  / p_6m
        m3  = (p_now - p_3m)  / p_3m

    ok = (np.isfinite(m12) & np.isfinite(m6) & np.isfinite(m3)
          & (p_now > 0) & (p_12m > 0) & (p_6m > 0) & (p_3m > 0))

    # Raw factor matrices over ALL days (NaN where not computable). Z-scoring is
    # done PER DAY over the point-in-time eligible universe in day_scores() below,
    # NOT once over the whole cached population — matching the live signal engine.
    m12_mat = np.full((N_all, S), np.nan); m12_mat[valid_d] = m12
    m6_mat  = np.full((N_all, S), np.nan); m6_mat[valid_d]  = m6
    m3_mat  = np.full((N_all, S), np.nan); m3_mat[valid_d]  = m3
    ok_mat  = np.zeros((N_all, S), dtype=bool); ok_mat[valid_d] = ok

    vol_mat  = (pd.DataFrame(full_close, columns=valid_tickers)
                .pct_change().rolling(cfg.LOOKBACK_6M).std().values * np.sqrt(252))

    # ── Eligibility mask: liquidity (₹cr traded value), min price, flat/stale ────
    # Applied to the scores so ineligible names can never be ranked/bought. (The
    # old per-day path applied MIN_AVG_VALUE_CR; the vectorized path didn't — this
    # restores it, plus a flat-price filter for the forward-filled bias-free cache.)
    full_close_df = pd.DataFrame(full_close, index=full_dates, columns=valid_tickers)
    vol_df   = volume.reindex(index=full_dates, columns=valid_tickers).fillna(0.0)
    val_cr   = (full_close_df * vol_df).rolling(60, min_periods=20).mean().values / 1e7

    eligible = np.isfinite(val_cr)                         # need 60d to assess
    n_liq = n_flat = n_price = 0
    if cfg.MIN_AVG_VALUE_CR > 0:
        m = val_cr >= cfg.MIN_AVG_VALUE_CR
        n_liq = int((~m & eligible).sum()); eligible &= m
    if cfg.MIN_PRICE > 0:
        m = full_close >= cfg.MIN_PRICE
        n_price = int((~m & eligible).sum()); eligible &= m
    max_flat = getattr(cfg, "MAX_FLAT_FRAC", 0.0)
    if max_flat > 0:
        zero_chg  = (full_close_df.diff().abs() < 1e-9)
        flat_frac = zero_chg.rolling(60, min_periods=20).mean().values
        m = flat_frac <= max_flat
        n_flat = int((~m & eligible).sum()); eligible &= m
    print(f"Eligibility filters → blocked stock-days: liquidity<{cfg.MIN_AVG_VALUE_CR}cr={n_liq:,}, "
          f"flat>{max_flat:.0%}={n_flat:,}, price<{cfg.MIN_PRICE}={n_price:,}")

    # 60-day average daily volume (shares) — for the participation cap on fills.
    adv_shares = vol_df.rolling(60, min_periods=20).mean().values

    dma_vals  = pd.DataFrame(full_close, index=full_dates,
                             columns=valid_tickers).rolling(cfg.DMA_EXIT).mean().values
    # Exit fires only when price is more than DMA_EXIT_BUFFER below the DMA (hysteresis
    # vs the strict entry gate) — cuts DMA ping-pong churn.
    above_dma = full_close > dma_vals * (1 - getattr(cfg, "DMA_EXIT_BUFFER", 0.0))   # (N_all, S) bool — daily exit checks
    with np.errstate(invalid="ignore", divide="ignore"):
        dist_dma_mat = full_close / dma_vals - 1.0   # distance above the 250-DMA (trend extension)

    valid_tickers_arr = np.array(valid_tickers, dtype=object)

    def day_scores(di, uni_set):
        """Composite momentum score, z-scored WITHIN the day's investable
        point-in-time universe (cur_universe ∩ eligible ∩ computable factors) —
        matching signals.py, which standardises over the eligible set rather than
        the whole cache. Returns a Series sorted best→worst (may be empty)."""
        base = ok_mat[di] & eligible[di]
        idx  = [j for j in np.nonzero(base)[0] if valid_tickers[j] in uni_set]
        if not idx:
            return pd.Series(dtype=float)
        idx = np.array(idx)
        def z(a):
            mu, sd = np.nanmean(a), np.nanstd(a)
            return (a - mu) / sd if sd > 0 else np.zeros_like(a)
        sc = (cfg.W_MOM_12M * z(m12_mat[di, idx]) + cfg.W_MOM_6M * z(m6_mat[di, idx])
              + cfg.W_MOM_3M * z(m3_mat[di, idx]) + cfg.W_VOL * z(vol_mat[di, idx]))
        wdd = getattr(cfg, "W_DIST_DMA", 0.0)
        if wdd:
            sc = sc + wdd * z(dist_dma_mat[di, idx])     # tilt toward trend-extension
        return pd.Series(sc, index=valid_tickers_arr[idx]).sort_values(ascending=False)

    def cap_shares(n, di, t):
        """Cap an order at PARTICIPATION_LIMIT × 60-day ADV (shares). Rarely binds
        at ₹1L; prevents unrealistic fills in thin names at larger capital."""
        lim = getattr(cfg, "PARTICIPATION_LIMIT", 0.0)
        ti  = t_to_i.get(t)
        if lim > 0 and ti is not None and di is not None:
            adv = adv_shares[di, ti]
            if np.isfinite(adv) and adv > 0:
                return max(0, min(n, int(lim * adv)))
        return n

    def rank_weights(n):
        """Per-position capital weights (sum to 1) in rank order. 'equal' = 1/n;
        'tiered' = ranks [top5]/[6–12]/[rest] get RANK_TIER_WEIGHTS multipliers
        (higher rank → more capital)."""
        if n <= 0:
            return []
        if getattr(cfg, "POSITION_WEIGHTING", "equal") != "tiered":
            return [1.0 / n] * n
        tw  = getattr(cfg, "RANK_TIER_WEIGHTS", (1.5, 1.1, 0.7))
        raw = [tw[0] if i < 5 else (tw[1] if i < 12 else tw[2]) for i in range(n)]
        s   = sum(raw)
        return [r / s for r in raw]

    print(f"Per-day scoring ready: {S} stocks × {N_all} days")

    cash        = float(cfg.INITIAL_CAPITAL)
    holdings    = {}
    lots        = {}   # ticker → list of [shares, price, date]  (FIFO acquisition lots)
    eq_curve    = []
    trade_log   = []
    tax_ledger  = TaxLedger()
    cash_interest_total = 0.0
    daily_cash_rate     = getattr(cfg, "CASH_YIELD", 0.0) / 252.0

    cached_scored        = pd.DataFrame()
    cached_top_25        = []
    cached_portfolio     = []
    rank_hist            = []     # per-day {ticker: rank} for the rank-velocity exit
    in_risk_off          = False
    last_rebalance_period = None  # cadence comparison (see REBALANCE_FREQ)

    print(f"\n{'='*55}")
    print(f"  BACKTEST [DAILY EXITS]  |  {cfg.START_DATE} → {cfg.END_DATE}")
    print(f"  Capital : ₹{cfg.INITIAL_CAPITAL:,.0f}  |  Stocks: {cfg.TOP_N}  |  Sector cap: {cfg.MAX_PER_SECTOR}")
    print(f"  Formula : {cfg.W_MOM_12M}*z12M + {cfg.W_MOM_6M}*z6M + {cfg.W_MOM_3M}*z3M + {cfg.W_VOL}*zVol")
    print(f"  Exit monitoring: DAILY  |  Full rebalance: MONTHLY")
    print(f"{'='*55}\n")

    all_days_list = list(all_days)
    for day_idx, day in enumerate(all_days_list):
        date_str    = pd.Timestamp(day).strftime("%Y-%m-%d")
        di          = date_to_di.get(day)          # integer index into full_close / score_mat
        next_day    = all_days_list[day_idx + 1] if day_idx + 1 < len(all_days_list) else day
        # Rebalance cadence (cfg.REBALANCE_FREQ): "monthly" | "weekly" | "2x-week" | "none"
        _freq = getattr(cfg, "REBALANCE_FREQ", "monthly")
        _d    = pd.Timestamp(day); _iso = _d.isocalendar()
        if _freq == "weekly":
            rebal_period = (_iso[0], _iso[1])                      # first trading day of each ISO week
        elif _freq in ("2x-week", "biweekly", "twice"):
            rebal_period = (_iso[0], _iso[1], _d.weekday() >= 3)   # Mon–Wed vs Thu–Fri ≈ twice/week
        elif _freq == "none":
            rebal_period = "once"                                 # build once, then never rotate
        else:
            rebal_period = _d.to_period("M")                      # monthly (default)
        is_rebalance = (rebal_period != last_rebalance_period)

        # ── IDLE-CASH INTEREST (daily accrual on the cash balance) ──────────
        if daily_cash_rate and cash > 0:
            interest             = cash * daily_cash_rate
            cash                += interest
            cash_interest_total += interest

        # ── CAPITAL-GAINS TAX SETTLEMENT (once per completed financial year) ─
        tax_due = tax_ledger.settle_completed(day)
        if tax_due > 0:
            cash -= tax_due
            trade_log.append({
                "date":date_str,"ticker":"-","action":"TAX(FY settle)",
                "shares":0,"price":0,"value":0,"cost":0,"tax":tax_due
            })

        # ── REGIME CHECK (daily) ────────────────────────────────────────────
        if getattr(cfg, "BACKTEST_FORCE_RISK_ON", cfg.FORCE_RISK_ON):
            strength = 1.0
            regime   = "RISK-ON"
        else:
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

        # ── RECORD NAV at today's close, BEFORE today's trades ──────────────
        # Trades decided today fill at NEXT day's open and show up in tomorrow's
        # NAV; today's holdings (filled this morning from yesterday's decisions)
        # are what's marked at today's close. No forward-dating of future fills.
        day_val = cash + sum(sh * get_price(close, t, day)
                             for t, sh in holdings.items() if sh > 0)
        eq_curve.append({"date": day, "value": day_val,
                         "regime": "OFF" if regime == "RISK-OFF" else "ON"})

        # ── RISK-OFF: liquidate, retrying missed sells on subsequent days ────
        if regime == "RISK-OFF":
            for t, sh in list(holdings.items()):
                if sh <= 0 or t not in close.columns:
                    continue
                px = get_price(close, t, day)
                if px <= 0:
                    continue
                open_px = get_open_price(open_prices, t, next_day)
                fill_px = get_fill_price(px, open_px, "sell")
                if fill_px is None:
                    continue                 # order didn't fill — keep, retry tomorrow
                proceeds = sh * fill_px
                cost     = txn_cost(proceeds, "sell")
                tax      = realize_lots(tax_ledger, lots.get(t, []), fill_px, day)
                cash    += proceeds - cost - tax
                lots.pop(t, None)
                holdings[t] = 0
                trade_log.append({
                    "date":date_str,"ticker":t,"action":"SELL(RISK-OFF)",
                    "shares":sh,"price":fill_px,"value":proceeds,"cost":cost,"tax":tax
                })
            in_risk_off  = True
            cached_scored = pd.DataFrame(); cached_top_25 = []; cached_portfolio = []
            if is_rebalance:
                unsold = sum(1 for s in holdings.values() if s > 0)
                tag    = f"  (unsold: {unsold})" if unsold else ""
                print(f"{date_str} | RISK-OFF  | Cash : ₹{cash:>12,.0f}{tag}")
            continue

        # ── RISK-ON ─────────────────────────────────────────────────────────
        in_risk_off = False

        # ── DAILY SCORES — z-scored WITHIN the point-in-time eligible universe ─
        cur_universe = (get_universe_for_date(day, universe_history)
                        if using_dynamic_universe else UNIVERSE)
        if di is not None and di >= min_hist:
            ranked = list(day_scores(di, set(cur_universe)).index)
            cached_scored    = pd.DataFrame(
                {"sector": [cur_universe.get(t, "Unknown") for t in ranked]}, index=ranked
            )
            cached_top_25    = ranked[:cfg.EXIT_RANK_CUTOFF]
            cached_portfolio = pick_portfolio(cached_scored, close, day)
        else:
            ranked           = []
            cached_scored    = pd.DataFrame()
            cached_top_25    = []
            cached_portfolio = []
        rank_hist.append({tk: i for i, tk in enumerate(ranked)})   # today's ranks (rank-velocity exit)

        # ── DAILY EXIT CHECK ─────────────────────────────────────────────────
        # Check every current holding against exit rules using today's prices
        current_held = [t for t, s in holdings.items() if s > 0]
        exits        = []

        for t in current_held:
            exit_reason = None

            # Rule 1: dropped out of top 25 (using today's scores)
            if cached_top_25 and t not in cached_top_25:
                exit_reason = "rank"

            # Rule 2: price below DMA_EXIT (checked with today's price) — if enabled
            ti = t_to_i.get(t)
            if getattr(cfg, "USE_DMA", True) and ti is not None and di is not None and not above_dma[di, ti]:
                exit_reason = f"{cfg.DMA_EXIT}DMA"

            # Rule 3: hard stop-loss — position down >= STOP_LOSS_PCT from avg cost
            slp = getattr(cfg, "STOP_LOSS_PCT", 0.0)
            if slp > 0 and lots.get(t):
                tot = sum(s for s, _, _ in lots[t])
                avg = sum(s * p for s, p, _ in lots[t]) / tot if tot else 0.0
                px  = get_price(close, t, day)
                if avg > 0 and px > 0 and (px / avg - 1.0) <= -slp:
                    exit_reason = "stop"

            # Rule 4: rank-velocity — fell > RANK_DROP_EXIT ranks vs RANK_DROP_LOOKBACK days ago
            dexit = getattr(cfg, "RANK_DROP_EXIT", 0)
            lb    = getattr(cfg, "RANK_DROP_LOOKBACK", 3)
            if dexit and len(rank_hist) > lb:
                rnow, rprev = rank_hist[-1].get(t), rank_hist[-1 - lb].get(t)
                if rnow is not None and rprev is not None and (rnow - rprev) > dexit:
                    exit_reason = "rankvel"

            if exit_reason:
                exits.append((t, exit_reason))

        # ── PROCESS EXITS + IMMEDIATE REPLACEMENT ───────────────────────────
        # Missed sells (gap-down past limit / no open) are NOT force-filled — the
        # position is kept and re-checked next day. Only actually-sold names free a
        # slot, so replacements buy exactly that many.
        sold = []
        for t, reason in exits:
            sh = holdings.get(t, 0)
            if sh > 0:
                px = get_price(close, t, day)
                if px > 0:
                    open_px  = get_open_price(open_prices, t, next_day)
                    fill_px  = get_fill_price(px, open_px, "sell")
                    if fill_px is None:
                        continue            # order didn't fill — keep, retry tomorrow
                    proceeds = sh * fill_px
                    cost     = txn_cost(proceeds, "sell")
                    tax      = realize_lots(tax_ledger, lots.get(t, []), fill_px, day)
                    cash    += proceeds - cost - tax
                    lots.pop(t, None)
                    holdings[t] = 0
                    sold.append(t)
                    trade_log.append({
                        "date":date_str,"ticker":t,"action":f"SELL({reason})",
                        "shares":sh,"price":fill_px,"value":proceeds,"cost":cost,"tax":tax
                    })

        # Immediately find and buy replacements for what actually sold.
        # NOTE: same-day re-entry of a just-sold name is now ALLOWED (the
        # exited-today guard was removed by request) — find_replacement will pick
        # it again if it's still top-ranked and above its DMA.
        if sold and not cached_scored.empty:
            replacements = find_replacement(cached_scored, current_held, sold, close, day, cur_universe)

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
                if holdings.get(t, 0) == 0 and _mc_skip_new_entry():
                    continue                          # MC: drop this entry (cash stays idle)
                rank     = cached_scored.index.get_loc(t) if t in cached_scored.index else 99
                open_px  = get_open_price(open_prices, t, next_day)
                fill_px  = get_fill_price(px, open_px, "buy", rank)
                if fill_px is None:
                    continue  # missed — gap-up too large
                cur_val = holdings.get(t, 0) * fill_px
                if cur_val < target * 0.95:
                    n    = cap_shares(int((target - cur_val) / fill_px), di, t)
                    cost = n * fill_px
                    tc   = txn_cost(cost, "buy")
                    if n > 0 and cash >= cost + tc:
                        cash -= (cost + tc)
                        holdings[t] = holdings.get(t, 0) + n
                        lots.setdefault(t, []).append([n, fill_px, day])
                        trade_log.append({
                            "date":date_str,"ticker":t,"action":"BUY(replacement)",
                            "shares":n,"price":fill_px,"value":cost,"cost":tc
                        })

        # ── MONTHLY FULL ROTATION ────────────────────────────────────────────
        # On rebalance day: rotate underperformers + top up to full allocation
        if is_rebalance and not cached_scored.empty:
            current_held_after = [t for t, s in holdings.items() if s > 0]

            # Find holdings not in current top N — rotate them out
            rotate_out = [t for t in current_held_after if t not in cached_portfolio]
            for t in rotate_out:
                sh = holdings.get(t, 0)
                if sh > 0:
                    px = get_price(close, t, day)
                    if px > 0:
                        open_px  = get_open_price(open_prices, t, next_day)
                        fill_px  = get_fill_price(px, open_px, "sell")
                        if fill_px is None:
                            continue        # order didn't fill — keep, retry next day
                        proceeds = sh * fill_px
                        cost     = txn_cost(proceeds, "sell")
                        tax      = realize_lots(tax_ledger, lots.get(t, []), fill_px, day)
                        cash    += proceeds - cost - tax
                        lots.pop(t, None)
                        holdings[t] = 0
                        trade_log.append({
                            "date":date_str,"ticker":t,"action":"SELL(rotation)",
                            "shares":sh,"price":fill_px,"value":proceeds,"cost":cost,"tax":tax
                        })

            # Recompute portfolio value
            port_val = cash
            for t, sh in holdings.items():
                if sh > 0:
                    port_val += get_price(close, t, day) * sh

            stocks_to_hold = max(1, round(cfg.TOP_N * strength))
            selected       = cached_portfolio[:stocks_to_hold]
            deploy_capital = port_val * strength
            weights        = rank_weights(len(selected))   # rank-tiered (or equal)

            # Buy the selected names toward their rank-weighted target. Costly
            # high-rank names are NOT skipped: if their slot buys 0 shares but one
            # share is within 2× the slot and affordable, take a single share;
            # leftover is redistributed (weighted) in the second pass.
            bought = 0
            mc_reserved = 0.0          # MC: capital of skipped new entries — kept idle, not swept
            for i, t in enumerate(selected):
                px = get_price(close, t, day)
                if px <= 0:
                    continue
                tgt_i   = deploy_capital * weights[i]
                if holdings.get(t, 0) == 0 and _mc_skip_new_entry():
                    mc_reserved += tgt_i              # MC: drop this entry (reserve its slot as cash)
                    continue
                rank    = cached_scored.index.get_loc(t) if t in cached_scored.index else 99
                open_px = get_open_price(open_prices, t, next_day)
                fill_px = get_fill_price(px, open_px, "buy", rank)
                if fill_px is None:
                    continue  # missed — gap-up too large
                cur_val = holdings.get(t, 0) * fill_px
                if cur_val >= tgt_i * 0.95:
                    bought += 1
                    continue
                n = int((tgt_i - cur_val) / fill_px)
                if n == 0 and holdings.get(t, 0) == 0 and fill_px <= 2 * tgt_i:
                    n = 1                                  # include the costly high-rank name
                n    = cap_shares(n, di, t)
                cost = n * fill_px
                tc   = txn_cost(cost, "buy")
                if n > 0 and cash >= cost + tc:
                    cash -= (cost + tc)
                    holdings[t] = holdings.get(t, 0) + n
                    lots.setdefault(t, []).append([n, fill_px, day])
                    bought += 1
                    trade_log.append({
                        "date":date_str,"ticker":t,"action":"BUY",
                        "shares":n,"price":fill_px,"value":cost,"cost":tc
                    })

            # ── Second pass: redistribute leftover cash from integer rounding ─
            if bought > 0:
                in_scope  = [t for t in cached_portfolio if holdings.get(t, 0) > 0]
                scope_val = sum(holdings.get(t, 0) * float(full_close[di, t_to_i[t]])
                                for t in in_scope if t_to_i.get(t) is not None)
                remaining = (port_val * strength) - scope_val - mc_reserved
                if remaining > 500 and in_scope:
                    w2     = rank_weights(len(in_scope))     # redistribute by the same weights
                    total2 = scope_val + remaining
                    for i, t in enumerate(in_scope):
                        ti = t_to_i.get(t)
                        px = float(full_close[di, ti]) if ti is not None else 0.0
                        if px <= 0:
                            continue
                        new_target = total2 * w2[i]
                        rank    = cached_scored.index.get_loc(t) if t in cached_scored.index else 99
                        open_px = get_open_price(open_prices, t, next_day)
                        fill_px = get_fill_price(px, open_px, "buy", rank)
                        if fill_px is None:
                            continue
                        cur_val = holdings.get(t, 0) * fill_px
                        if cur_val < new_target * 0.95:
                            n    = cap_shares(int((new_target - cur_val) / fill_px), di, t)
                            cost = n * fill_px
                            tc   = txn_cost(cost, "buy")
                            if n > 0 and cash >= cost + tc:
                                cash -= (cost + tc)
                                holdings[t] = holdings.get(t, 0) + n
                                lots.setdefault(t, []).append([n, fill_px, day])
                                trade_log.append({
                                    "date":date_str,"ticker":t,"action":"BUY(topup)",
                                    "shares":n,"price":fill_px,"value":cost,"cost":tc
                                })

            # Print monthly summary
            held = [t for t, s in holdings.items() if s > 0]
            port_val = cash + sum(
                holdings.get(t,0) * get_price(close,t,day)
                for t in held
            )
            print(f"{date_str} | RISK-ON   | ₹{port_val:>12,.0f} | {', '.join(held[:7])}")
            last_rebalance_period = rebal_period

        # NOTE: daily NAV is recorded at the TOP of the loop (pre-trade) so that
        # trades filling at next-day open are reflected in tomorrow's equity, not
        # forward-dated into today. (No append here on purpose.)

    # ── Settle any open (final) financial year's capital-gains tax ───────────
    final_tax = tax_ledger.settle_all()

    # ── Build equity curve ─────────────────────────────────────────────────
    eq_df  = pd.DataFrame(eq_curve).set_index("date")
    eq_df.index = pd.to_datetime(eq_df.index)
    equity = eq_df["value"]
    # Reflect the still-unpaid final-FY tax in the closing equity value.
    if final_tax > 0 and len(equity):
        equity.iloc[-1] -= final_tax

    total_tax = sum(s["tax"] for s in tax_ledger.settled)

    # Benchmarks
    bench_raw   = nifty50.loc[cfg.START_DATE:cfg.END_DATE].dropna()
    bench_curve = (bench_raw / bench_raw.iloc[0]) * cfg.INITIAL_CAPITAL
    # Fair, same-data benchmark: equal-weight TR index of the picked universe.
    eqw_curve   = build_universe_benchmark(
        close, all_days, valid_tickers, universe_history, using_dynamic_universe
    )

    # ── Print results ──────────────────────────────────────────────────────
    m_s = print_metrics(equity,      "STRATEGY [DAILY EXITS] — Aggressive Momentum")
    m_e = print_metrics(eqw_curve,   "BENCHMARK — Universe Equal-Weight (gross: daily-rebal, no costs/div)")
    m_b = print_metrics(bench_curve, "BENCHMARK — Nifty 50 (price index, reference)")

    print(f"\n{'='*48}")
    print(f"  Realism accounting")
    print(f"  Capital-gains tax paid : ₹{total_tax:>12,.0f}")
    print(f"  Idle-cash interest     : ₹{cash_interest_total:>12,.0f}")
    print(f"  Slippage+spread assumed: {getattr(cfg,'SLIPPAGE_BPS',0)}+{getattr(cfg,'SPREAD_BPS',0)}"
          f" = {getattr(cfg,'SLIPPAGE_BPS',0)+getattr(cfg,'SPREAD_BPS',0)} bps/side")
    print(f"{'='*48}")
    print(f"  Outperformance vs Universe Equal-Weight (fair)")
    print(f"  CAGR delta   : {(m_s['cagr']-m_e['cagr'])*100:+.1f}% per year")
    print(f"  Sharpe delta : {m_s['sharpe']-m_e['sharpe']:+.2f}")
    print(f"  DD improvement: {(m_s['maxdd']-m_e['maxdd'])*100:+.1f}%  (positive = shallower)")
    print(f"{'='*48}")

    # Save outputs (skipped during parameter sweeps to avoid file churn/collisions)
    if save:
        equity.to_csv("equity_curve_daily.csv")
        pd.DataFrame(trade_log).to_csv("trade_log_daily.csv", index=False)
        print(f"\nSaved: equity_curve_daily.csv  |  trade_log_daily.csv")
        print(f"Total trades executed: {len(trade_log)}")

    # Return key metrics so external drivers can score a run without parsing stdout.
    return {
        "cagr":     m_s["cagr"],   "sharpe":  m_s["sharpe"], "sortino": m_s["sortino"],
        "maxdd":    m_s["maxdd"],  "winrate": m_s["winrate"],
        "edge_cagr":   m_s["cagr"]   - m_e["cagr"],     # vs same-universe equal-weight
        "edge_sharpe": m_s["sharpe"] - m_e["sharpe"],
        "calmar":   (m_s["cagr"] / abs(m_s["maxdd"])) if m_s["maxdd"] else 0.0,
        "tax":      total_tax, "trades": len(trade_log),
    }


if __name__ == "__main__":
    run_backtest()
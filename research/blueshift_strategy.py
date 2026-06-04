"""
blueshift_strategy.py — NSE LargeMidcap-250 momentum, ported to Blueshift
=========================================================================
Mirrors the local strategy AS-IS (config = opt#7 weights):

  score = 0.20·z(mom12) + 0.30·z(mom6) + 0.20·z(mom3) − 0.20·z(vol6) + 0.50·z(dist_dma)
  mom_k = return from (LOOKBACK_k + SKIP_RECENT) days ago → SKIP_RECENT days ago
  LOOKBACK 280/140/80, SKIP 25 ; vol6 = ann. 6M realised vol ; dist_dma = px/250DMA − 1
  All factors z-scored cross-sectionally. Hold top TOP_N=15 by score,
  MAX_PER_SECTOR=5, 250-DMA entry gate, tiered weights (1.3/1.0/0.8 for ranks
  [top5]/[6-12]/[rest]). Daily exits (rank>50 OR px<250DMA), monthly full rotation.
  Regime OFF (FORCE_RISK_ON=True) → always invested.

⚠️  NOT equivalent to the bias-free local backtester:
  - Blueshift data = current/surviving names only (survivorship bias; optimistic).
  - Universe below is TODAY's membership (no point-in-time index reconstruction).
  - Fills/commission/slippage use Blueshift models, not our next-open + STT/GST/tax.
  - Needs >~310 daily bars of history before the start date.
Use as a forward/paper cross-check; the local bias-free backtester is the truth.
"""
from blueshift.api import (
    symbol, order_target_percent, schedule_function,
    date_rules, time_rules, get_datetime, set_commission, set_slippage,
)
import numpy as np
import pandas as pd

# Cost-model classes live in different places across Blueshift versions — import
# defensively so a missing class can't crash the algo at load.
try:
    from blueshift.finance import commission, slippage
    HAVE_COSTS = True
except Exception:
    HAVE_COSTS = False

# ── PARAMETERS (mirror config.py) ───────────────────────────────────────────
TOP_N            = 15
MAX_PER_SECTOR   = 5
EXIT_RANK_CUTOFF = 50
RANK_TIER_WEIGHTS = (1.3, 1.0, 0.8)        # ranks [top5] / [6-12] / [rest]

W_MOM_12M, W_MOM_6M, W_MOM_3M, W_VOL, W_DIST_DMA = 0.20, 0.30, 0.20, -0.20, 0.50
LOOKBACK_12M, LOOKBACK_6M, LOOKBACK_3M = 280, 140, 80
SKIP_RECENT      = 25
DMA_EXIT         = 250
USE_DMA          = True
MIN_AVG_VALUE_CR = 1.0                      # 60-day avg traded value, ₹ crore
MAX_FLAT_FRAC    = 0.5
HISTORY_BARS     = LOOKBACK_12M + SKIP_RECENT + 5    # 310

# ── UNIVERSE (NSE LargeMidcap-250) → sector, mirrors signals.UNIVERSE ────────
UNIVERSE = {
    "APOLLOTYRE":"Auto","BAJAJ-AUTO":"Auto","BALKRISIND":"Auto","BHARATFORG":"Auto","BOSCHLTD":"Auto",
    "EICHERMOT":"Auto","ENDURANCE":"Auto","EXIDEIND":"Auto","HEROMOTOCO":"Auto","HYUNDAI":"Auto",
    "M&M":"Auto","MARUTI":"Auto","MOTHERSON":"Auto","MRF":"Auto","SCHAEFFLER":"Auto","TIINDIA":"Auto",
    "TMPV":"Auto","TVSMOTOR":"Auto","UNOMINDA":"Auto",
    "ABB":"Capital Goods","AIAENG":"Capital Goods","APARINDS":"Capital Goods","APLAPOLLO":"Capital Goods",
    "ASHOKLEY":"Capital Goods","ASTRAL":"Capital Goods","BDL":"Capital Goods","BEL":"Capital Goods",
    "BHEL":"Capital Goods","CGPOWER":"Capital Goods","COCHINSHIP":"Capital Goods","CUMMINSIND":"Capital Goods",
    "ENRIN":"Capital Goods","ESCORTS":"Capital Goods","GVT&D":"Capital Goods","HAL":"Capital Goods",
    "HONAUT":"Capital Goods","KEI":"Capital Goods","MAZDOCK":"Capital Goods","POLYCAB":"Capital Goods",
    "POWERINDIA":"Capital Goods","PREMIERENE":"Capital Goods","SIEMENS":"Capital Goods","SUPREMEIND":"Capital Goods",
    "SUZLON":"Capital Goods","THERMAX":"Capital Goods","TMCV":"Capital Goods","WAAREEENER":"Capital Goods",
    "ACC":"Cement","AMBUJACEM":"Cement","DALBHARAT":"Cement","GRASIM":"Cement","JKCEMENT":"Cement",
    "SHREECEM":"Cement","ULTRACEMCO":"Cement",
    "COROMANDEL":"Chemicals","FLUOROCHEM":"Chemicals","LINDEINDIA":"Chemicals","PIDILITIND":"Chemicals",
    "PIIND":"Chemicals","SOLARINDS":"Chemicals","SRF":"Chemicals","UPL":"Chemicals",
    "ASIANPAINT":"Consumer","BERGEPAINT":"Consumer","BLUESTARCO":"Consumer","DIXON":"Consumer","DMART":"Consumer",
    "ETERNAL":"Consumer","HAVELLS":"Consumer","INDHOTEL":"Consumer","IRCTC":"Consumer","ITCHOTELS":"Consumer",
    "JUBLFOOD":"Consumer","KALYANKJIL":"Consumer","KPRMILL":"Consumer","LENSKART":"Consumer","LGEINDIA":"Consumer",
    "NAUKRI":"Consumer","NYKAA":"Consumer","PAGEIND":"Consumer","SWIGGY":"Consumer","TITAN":"Consumer",
    "TRENT":"Consumer","VMM":"Consumer","VOLTAS":"Consumer",
    "ADANIENSOL":"Energy","ADANIGREEN":"Energy","ADANIPOWER":"Energy","ATGL":"Energy","BPCL":"Energy",
    "COALINDIA":"Energy","GAIL":"Energy","HINDPETRO":"Energy","IOC":"Energy","JSWENERGY":"Energy",
    "NHPC":"Energy","NLCINDIA":"Energy","NTPC":"Energy","NTPCGREEN":"Energy","OIL":"Energy","ONGC":"Energy",
    "PETRONET":"Energy","POWERGRID":"Energy","RELIANCE":"Energy","SJVN":"Energy","TATAPOWER":"Energy","TORNTPOWER":"Energy",
    "AWL":"FMCG","BRITANNIA":"FMCG","COLPAL":"FMCG","DABUR":"FMCG","GODFRYPHLP":"FMCG","GODREJCP":"FMCG",
    "HINDUNILVR":"FMCG","ITC":"FMCG","MARICO":"FMCG","NESTLEIND":"FMCG","PATANJALI":"FMCG","RADICO":"FMCG",
    "TATACONSUM":"FMCG","UBL":"FMCG","UNITDSPR":"FMCG","VBL":"FMCG",
    "360ONE":"Financials","ABCAPITAL":"Financials","AIIL":"Financials","AUBANK":"Financials","AXISBANK":"Financials",
    "BAJAJFINSV":"Financials","BAJAJHFL":"Financials","BAJAJHLDNG":"Financials","BAJFINANCE":"Financials",
    "BANKBARODA":"Financials","BANKINDIA":"Financials","BSE":"Financials","CANBK":"Financials","CHOLAFIN":"Financials",
    "CRISIL":"Financials","FEDERALBNK":"Financials","GICRE":"Financials","GROWW":"Financials","HDBFS":"Financials",
    "HDFCAMC":"Financials","HDFCBANK":"Financials","HDFCLIFE":"Financials","HUDCO":"Financials","ICICIAMC":"Financials",
    "ICICIBANK":"Financials","ICICIGI":"Financials","ICICIPRULI":"Financials","IDFCFIRSTB":"Financials","INDIANB":"Financials",
    "INDUSINDBK":"Financials","IREDA":"Financials","IRFC":"Financials","JIOFIN":"Financials","KOTAKBANK":"Financials",
    "LICHSGFIN":"Financials","LICI":"Financials","LTF":"Financials","M&MFIN":"Financials","MAHABANK":"Financials",
    "MCX":"Financials","MFSL":"Financials","MOTILALOFS":"Financials","MUTHOOTFIN":"Financials","NAM-INDIA":"Financials",
    "NIACL":"Financials","PAYTM":"Financials","PFC":"Financials","PNB":"Financials","POLICYBZR":"Financials",
    "RECLTD":"Financials","SBICARD":"Financials","SBILIFE":"Financials","SBIN":"Financials","SHRIRAMFIN":"Financials",
    "SUNDARMFIN":"Financials","TATACAP":"Financials","TATAINVEST":"Financials","UNIONBANK":"Financials","YESBANK":"Financials",
    "ABBOTINDIA":"Healthcare","AJANTPHARM":"Healthcare","ALKEM":"Healthcare","ANTHEM":"Healthcare","APOLLOHOSP":"Healthcare",
    "AUROPHARMA":"Healthcare","BIOCON":"Healthcare","CIPLA":"Healthcare","DIVISLAB":"Healthcare","DRREDDY":"Healthcare",
    "FORTIS":"Healthcare","GLAXO":"Healthcare","GLENMARK":"Healthcare","IPCALAB":"Healthcare","LAURUSLABS":"Healthcare",
    "LUPIN":"Healthcare","MANKIND":"Healthcare","MAXHEALTH":"Healthcare","MEDANTA":"Healthcare","SUNPHARMA":"Healthcare",
    "TORNTPHARM":"Healthcare","ZYDUSLIFE":"Healthcare",
    "COFORGE":"IT","HCLTECH":"IT","HEXT":"IT","INFY":"IT","KPITTECH":"IT","LTM":"IT","LTTS":"IT","MPHASIS":"IT",
    "OFSS":"IT","PERSISTENT":"IT","TATAELXSI":"IT","TCS":"IT","TECHM":"IT","WIPRO":"IT",
    "3MINDIA":"Industrials","ADANIPORTS":"Industrials","CONCOR":"Industrials","GMRAIRPORT":"Industrials",
    "GODREJIND":"Industrials","INDIGO":"Industrials","JSWINFRA":"Industrials","LT":"Industrials","RVNL":"Industrials",
    "ADANIENT":"Metals","HINDALCO":"Metals","HINDZINC":"Metals","JINDALSTEL":"Metals","JSL":"Metals",
    "JSWSTEEL":"Metals","LLOYDSME":"Metals","NATIONALUM":"Metals","NMDC":"Metals","SAIL":"Metals",
    "TATASTEEL":"Metals","VEDL":"Metals",
    "DLF":"Realty","GODREJPROP":"Realty","LODHA":"Realty","OBEROIRLTY":"Realty","PHOENIXLTD":"Realty","PRESTIGE":"Realty",
    "BHARTIARTL":"Telecom","BHARTIHEXA":"Telecom","IDEA":"Telecom","INDUSTOWER":"Telecom","TATACOMM":"Telecom",
}


def initialize(context):
    # Resolve tradable symbols (skip any not on Blueshift / delisted).
    context.assets, context.sector = [], {}
    for tk, sec in UNIVERSE.items():
        try:
            a = symbol(tk)
        except Exception:
            continue
        if a is not None:
            context.assets.append(a)
            context.sector[a] = sec
    print("Universe resolved: %d / %d symbols" % (len(context.assets), len(UNIVERSE)))

    # Cost model matched to the local backtester as closely as Blueshift allows:
    #   commission ≈ 0.11%/side  (STT 0.10% + stamp/exchange/SEBI + 18% GST on charges)
    #   slippage   = 8 bps/side  (our SLIPPAGE_BPS 5 + SPREAD_BPS 3)
    # NOT representable here: the flat ₹15.93 DP charge per sell (Blueshift's PerDollar
    # model can't mix a percentage with a flat-per-trade fee) — a small *understatement*
    # of cost on small positions.
    if HAVE_COSTS:
        try:
            set_commission(commission.PerDollar(cost=0.0011))          # ≈ 0.11% / side
        except Exception as e:
            print("commission not set (Blueshift default): %r" % e)
        # FixedBasisPointsSlippage's arg NAME varies by engine version (5.x rejects
        # 'basis_points'), so pass positionally: (basis_points, volume_limit).
        sl = None
        for args in [(8.0, 0.1), (8.0,)]:
            try:
                sl = slippage.FixedBasisPointsSlippage(*args)
                break
            except Exception:
                sl = None
        try:
            if sl is not None:
                set_slippage(sl)
            else:
                print("slippage signature not matched — using Blueshift default")
        except Exception as e:
            print("slippage not set (Blueshift default): %r" % e)
    else:
        print("blueshift.finance unavailable — set commission/slippage in the backtest UI")

    # Run shortly AFTER the open: signals use history through the PRIOR close (no
    # look-ahead) and orders fill at the open — matching the backtester's
    # rank-on-close / fill-next-open model.
    schedule_function(daily_strategy, date_rules.every_day(), time_rules.market_open(minutes=30))


def compute_factors(closes, vols):
    """Per-asset factor table from a (bars × assets) close/volume DataFrame.
    Mirrors signals.compute_scores exactly."""
    s = SKIP_RECENT
    rows = {}
    for a in closes.columns:
        c = closes[a].dropna()
        v = vols[a].dropna() if a in vols else pd.Series(dtype=float)
        if len(c) < LOOKBACK_12M + s + 1:
            continue
        p_now = c.iloc[-(s + 1)]
        p12, p6, p3 = c.iloc[-(LOOKBACK_12M + s)], c.iloc[-(LOOKBACK_6M + s)], c.iloc[-(LOOKBACK_3M + s)]
        if min(p12, p6, p3) <= 0:
            continue
        # Liquidity: 60-day avg traded value ≥ MIN_AVG_VALUE_CR (₹ crore)
        if len(v) >= 60:
            tv_cr = (c.tail(60) * v.tail(60)).mean() / 1e7
            if tv_cr < MIN_AVG_VALUE_CR:
                continue
        # Flat / stale filter
        if (c.tail(60).pct_change().fillna(0) == 0).mean() > MAX_FLAT_FRAC:
            continue
        dma = c.tail(DMA_EXIT).mean() if len(c) >= DMA_EXIT else np.nan
        if not (dma and dma > 0):
            continue
        vol6 = c.tail(LOOKBACK_6M).pct_change().dropna().std() * np.sqrt(252)
        rows[a] = {
            "mom12": (p_now - p12) / p12, "mom6": (p_now - p6) / p6, "mom3": (p_now - p3) / p3,
            "vol6": vol6, "dist": c.iloc[-1] / dma - 1.0, "px": c.iloc[-1], "dma": dma,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def compute_scores(context, data):
    closes = data.history(context.assets, "close", HISTORY_BARS, "1d")
    vols   = data.history(context.assets, "volume", 60, "1d")
    df = compute_factors(closes, vols)
    if df.empty:
        return df
    def z(x):
        sd = x.std()
        return (x - x.mean()) / sd if sd > 0 else x * 0.0
    df["score"] = (W_MOM_12M * z(df["mom12"]) + W_MOM_6M * z(df["mom6"]) +
                   W_MOM_3M * z(df["mom3"]) + W_VOL * z(df["vol6"]) +
                   W_DIST_DMA * z(df["dist"]))
    return df.sort_values("score", ascending=False)


def tier_weights(n):
    tw = [RANK_TIER_WEIGHTS[0] if i < 5 else (RANK_TIER_WEIGHTS[1] if i < 12 else RANK_TIER_WEIGHTS[2])
          for i in range(n)]
    s = sum(tw)
    return [w / s for w in tw]


def select_top(scored, sector):
    """Top-N by score with 250-DMA entry gate + sector cap, in rank order."""
    chosen, sec_count = [], {}
    for a, row in scored.iterrows():
        if USE_DMA and row["px"] < row["dma"]:        # DMA entry gate
            continue
        sec = sector.get(a, "X")
        if sec_count.get(sec, 0) >= MAX_PER_SECTOR:
            continue
        chosen.append(a); sec_count[sec] = sec_count.get(sec, 0) + 1
        if len(chosen) == TOP_N:
            break
    return chosen


def daily_strategy(context, data):
    # Catch + log any per-day error so the backtest completes (and shows the cause)
    # instead of halting with "Stopped" and no diagnostics.
    try:
        run_strategy(context, data)
    except Exception as e:
        print("%s  ERROR in run_strategy: %r" % (get_datetime().date(), e))


def run_strategy(context, data):
    scored = compute_scores(context, data)
    if scored.empty:
        print("%s  no stocks scored — insufficient history or no symbols resolved" % get_datetime().date())
        return

    ranked = list(scored.index)
    top_cutoff = set(ranked[:EXIT_RANK_CUTOFF])          # held names must stay within this
    held = [a for a in context.portfolio.positions if context.portfolio.positions[a].quantity > 0]

    is_rebalance = (get_datetime().day <= 3)             # first trading days of the month

    if is_rebalance or not held:
        target = select_top(scored, context.sector)     # full top-N rebuild
    else:
        # Daily exits: drop holdings that fell out of top-EXIT_RANK_CUTOFF or below DMA.
        survivors = []
        for a in held:
            if a not in top_cutoff:
                continue
            if USE_DMA and a in scored.index and scored.loc[a, "px"] < scored.loc[a, "dma"]:
                continue
            survivors.append(a)
        target = list(survivors)
        # Refill to TOP_N with best non-held names (DMA gate + sector cap).
        sec_count = {}
        for a in target:
            sc = context.sector.get(a, "X"); sec_count[sc] = sec_count.get(sc, 0) + 1
        for a in ranked:
            if len(target) >= TOP_N:
                break
            if a in target:
                continue
            if USE_DMA and scored.loc[a, "px"] < scored.loc[a, "dma"]:
                continue
            sc = context.sector.get(a, "X")
            if sec_count.get(sc, 0) >= MAX_PER_SECTOR:
                continue
            target.append(a); sec_count[sc] = sec_count.get(sc, 0) + 1

    # Order to tiered target weights; close anything no longer held.
    target = target[:TOP_N]
    weights = tier_weights(len(target)) if target else []
    tgt_w = dict(zip(target, weights))
    for a in held:
        if a not in tgt_w:
            order_target_percent(a, 0.0)
    for a, w in tgt_w.items():
        order_target_percent(a, w)

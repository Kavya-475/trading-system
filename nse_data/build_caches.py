"""
nse_data/build_caches.py
========================
Turns the raw NSE bhavcopy archive (downloaded by download.py) into a
SURVIVORSHIP-BIAS-FREE, corporate-action-adjusted price cache that the local
backtester.py can consume directly.

Pipeline
--------
1. Parse every equity bhavcopy (old + UDiFF formats), keep SERIES/SctySrs == EQ.
2. Repair renames via a curated, ISIN-verified ALIASES list (the cache is keyed
   by SYMBOL, not auto-stitched by ISIN): a company that changed
   ticker keeps one continuous series; delisted names keep their history up to
   their last trading day (that is the bias-free part).
3. Corporate-action adjust using NSE's own PREVCLOSE: on a split/bonus/rights
   ex-date NSE prints an adjusted previous close, so
       implied_factor = PREVCLOSE_today / CLOSE_prev_traded
   deviates from 1 only on real capital actions. Back-adjust close & open from
   those factors. (Ordinary dividends are NOT adjusted — same convention as a
   price index; this is appropriate for price-momentum + DMA.)
4. Reconcile symbols with universe_history.csv and report coverage vs the old
   yfinance cache (the whole point: far fewer missing/delisted names).
5. Emit caches under nse_data/ in the exact shape backtester.py expects
   (Date index, one column per symbol). Set cfg.USE_NSE_BHAVCOPY = True to use.

Index bhavcopy → regime cache (Nifty 50/100/500/Midcap150 + LargeMidcap250).

Usage:
    python nse_data/build_caches.py
    python nse_data/build_caches.py --action-threshold 0.20
"""
import argparse
import glob
import io
import os
import zipfile

import numpy as np
import pandas as pd

HERE      = os.path.dirname(__file__)
EQ_GLOB   = os.path.join(HERE, "raw", "equity", "*", "eq_*.zip")
IDX_GLOB  = os.path.join(HERE, "raw", "index",  "*", "idx_*.csv")

OUT_PRICE  = os.path.join(HERE, "price_cache.csv")
OUT_OPEN   = os.path.join(HERE, "open_cache.csv")
OUT_VOLUME = os.path.join(HERE, "volume_cache.csv")
OUT_REGIME = os.path.join(HERE, "regime_cache.csv")
OUT_REPORT = os.path.join(HERE, "adjustments_report.csv")

UNIVERSE_HISTORY = os.path.join(os.path.dirname(HERE), "universe_history.csv")

# Universe-history names that trade in bhavcopy under a DIFFERENT NSE symbol than
# the index used. Each verified by matching the recovered series' delisting date
# to the company's real history (e.g. ESSAROIL ends 2016, RPL ends 2009, IL&FSENGG
# ends Oct-2018). Recovers delisted names that would otherwise be survivorship gaps.
ALIASES = {
    "ESSAR": "ESSAROIL", "IVRCL": "IVRCLINFRA", "JAICORP": "JAICORPLTD",
    "MCLEODRUSSL": "MCLEODRUSS", "INGERSRAND": "INGERRAND", "SUNDRAMFAST": "SUNDRMFAST",
    "HINDUSTANUNILVR": "HINDUNILVR", "BOMBAYDYEING": "BOMDYEING", "WOCKHARDT": "WOCKPHARMA",
    "PENINSULA": "PENINLAND", "PRISMJOINTS": "PRISMCEM", "CORE": "COREEDUTEC",
    "GOKALDAS": "GOKEX", "HCLINSYS": "HCL-INSYS", "NARAYANA": "NH",
    "BF-UTIL": "BFUTILITIE", "RPETRO": "RPL", "JAYPEEINFRA": "JPINFRATEC",
    "PUNJTRACP": "PUNJABTRAC", "ILFSENGINE": "IL&FSENGG",
}

# Index name (as printed by NSE) → regime-cache column name.
INDEX_MAP = {
    "Nifty 50":               "nifty50",
    "Nifty 100":              "nifty100",
    "Nifty 500":              "nifty500",
    "NIFTY Midcap 150":       "nifty_midcap",
    "Nifty Midcap 150":       "nifty_midcap",
    "NIFTY LargeMidcap 250":  "largemidcap250",
}


# ── Equity parsing ──────────────────────────────────────────────────────────
def _date_from_name(path):
    base = os.path.basename(path)            # eq_YYYYMMDD.zip
    return pd.Timestamp(base[3:11])


def parse_equity_zip(path):
    """Return a normalized DataFrame: [symbol, isin, open, close, prevclose, volume]
    for the EQ series only, or None if unreadable/empty."""
    try:
        with zipfile.ZipFile(path) as z:
            inner = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not inner:
                return None
            raw = z.read(inner[0])
    except (zipfile.BadZipFile, OSError):
        return None

    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]

    if "TckrSymb" in df.columns:             # UDiFF (>= 2024-07-08)
        df = df[(df.get("FinInstrmTp") == "STK") & (df["SctySrs"] == "EQ")]
        out = pd.DataFrame({
            "symbol":    df["TckrSymb"].astype(str).str.strip(),
            "isin":      df["ISIN"].astype(str).str.strip(),
            "open":      pd.to_numeric(df["OpnPric"], errors="coerce"),
            "close":     pd.to_numeric(df["ClsPric"], errors="coerce"),
            "prevclose": pd.to_numeric(df["PrvsClsgPric"], errors="coerce"),
            "volume":    pd.to_numeric(df["TtlTradgVol"], errors="coerce"),
        })
    elif "SYMBOL" in df.columns:             # old format (< 2024-07-08)
        df = df[df["SERIES"].astype(str).str.strip() == "EQ"]
        out = pd.DataFrame({
            "symbol":    df["SYMBOL"].astype(str).str.strip(),
            "isin":      df["ISIN"].astype(str).str.strip() if "ISIN" in df.columns else "",
            "open":      pd.to_numeric(df["OPEN"], errors="coerce"),
            "close":     pd.to_numeric(df["CLOSE"], errors="coerce"),
            "prevclose": pd.to_numeric(df["PREVCLOSE"], errors="coerce"),
            "volume":    pd.to_numeric(df["TOTTRDQTY"], errors="coerce"),
        })
    else:
        return None

    # Key on SYMBOL (present in every era). ISIN only appeared in bhavcopy in
    # mid-2011, so requiring it would silently discard 2007–2011 (incl. the 2008
    # crash). ISIN is kept where present, used only as a secondary rename hint.
    out = out[(out["symbol"].str.len() > 0) & (out["close"] > 0)]
    return out if len(out) else None


def load_all_equity():
    files = sorted(glob.glob(EQ_GLOB))
    if not files:
        raise SystemExit(f"No equity bhavcopy found under {EQ_GLOB}. Run download.py first.")
    print(f"Parsing {len(files)} equity bhavcopy files…")
    frames = []
    for i, f in enumerate(files, 1):
        d = parse_equity_zip(f)
        if d is not None:
            d["date"] = _date_from_name(f)
            frames.append(d)
        if i % 500 == 0 or i == len(files):
            print(f"  parsed {i}/{len(files)}")
    long = pd.concat(frames, ignore_index=True)
    print(f"  {len(long):,} EQ rows, {long['symbol'].nunique():,} unique symbols, "
          f"{long['date'].nunique():,} trading days "
          f"({long['date'].min().date()} → {long['date'].max().date()})")
    return long


# ── Corporate-action back-adjustment ─────────────────────────────────────────
# AUTHORITATIVE source: nse_data/splits.csv (from fetch_splits.py / yfinance) —
# the exact ex-date + price multiplier for every split/bonus. PREVCLOSE in
# bhavcopy is raw, and a 1:1 bonus is indistinguishable from a −50% crash using
# close prices, so we never guess from prices when we have the real calendar.
#
# FALLBACK (only for symbols absent from splits.csv, e.g. delisted names yfinance
# lacks): a deliberately conservative price-jump detector that fires only on very
# large jumps snapping to a big clean ratio — well clear of ±20% circuit moves —
# so it cannot mislabel ordinary crashes. It will miss small bonuses (acceptable
# for obscure delisted names the strategy rarely holds).
SPLITS_CSV   = os.path.join(HERE, "splits.csv")
FB_RATIOS    = np.array([1/2, 1/3, 1/4, 1/5, 1/8, 1/10, 1/20, 1/50, 1/100,
                         2.0, 3.0, 4.0, 5.0, 10.0])
FB_MIN_MOVE  = 0.45    # only jumps > ±45% (circuit max is ±20%) — no crash collision
FB_TOL       = 0.02


def _clean_ratio_set():
    """Dense set of plausible split/bonus price multipliers p/q and inverses,
    so an observed jump can be snapped to the exact corporate-action ratio
    (handles combined split+bonus and ratios yfinance reports incompletely)."""
    rs = set()
    for q in range(1, 13):
        for p in range(1, 13):
            if p != q:
                rs.add(round(p / q, 6))
    for x in (1/12, 1/15, 1/20, 1/25, 1/50, 1/100, 12, 15, 20, 25, 50, 100):
        rs.add(round(x, 6))
    return np.array(sorted(rs))


SNAP = _clean_ratio_set()
SNAP_WINDOW   = 3      # search ± this many trading days around the yfinance ex-date
SNAP_MIN_MOVE = 0.10   # require a real discontinuity (>10%) to apply anything


def authoritative_eff(close_w, splits):
    """eff matrix from splits.csv. yfinance gives the trustworthy ex-date *calendar*;
    the exact multiplier is read from NSE's own price discontinuity near that date
    (most extreme 1-day move in a small window) and snapped to a clean ratio. This
    fixes yfinance ratios that miss a companion bonus (e.g. BAJFINANCE 0.5 vs the
    real 0.2) and small ex-date offsets, without ever guessing at non-event dates."""
    eff = pd.DataFrame(1.0, index=close_w.index, columns=close_w.columns)
    rows = []
    if splits is None or splits.empty:
        return eff, pd.DataFrame(columns=["symbol", "date", "price_mult", "source"])
    cols = set(close_w.columns)
    for sym, grp in splits.groupby("symbol"):
        if sym not in cols:
            continue
        col = close_w[sym].dropna()
        if len(col) < 2:
            continue
        ratios = col / col.shift(1)
        applied_days = set()
        for _, r in grp.iterrows():
            d   = pd.Timestamp(r["date"])
            pos = col.index.searchsorted(d)
            lo, hi = max(1, pos - SNAP_WINDOW), min(len(col), pos + SNAP_WINDOW + 1)
            if lo >= hi:
                continue
            window = ratios.iloc[lo:hi]
            ex = window.sub(1.0).abs().idxmax()        # the real ex-date = biggest jump
            if ex in applied_days:
                continue
            obs = float(ratios.loc[ex])
            if not np.isfinite(obs) or abs(obs - 1.0) < SNAP_MIN_MOVE:
                continue
            snapped = float(SNAP[np.abs(SNAP - obs).argmin()])
            eff.at[ex, sym] *= snapped
            applied_days.add(ex)
            rows.append({"symbol": sym, "date": ex.date(),
                         "price_mult": round(snapped, 6), "source": "yfinance"})
    return eff, pd.DataFrame(rows)


def fallback_eff(close_w, covered_symbols):
    """Conservative price-jump detector for symbols with NO authoritative splits."""
    todo = [c for c in close_w.columns if c not in covered_symbols]
    eff  = pd.DataFrame(1.0, index=close_w.index, columns=close_w.columns)
    if not todo:
        return eff, pd.DataFrame(columns=["symbol", "date", "price_mult", "source"])
    sub = close_w[todo]
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (sub / sub.shift(1)).values
    best = np.full(R.shape, np.inf); bestk = np.ones(R.shape)
    for k in FB_RATIOS:
        d = np.abs(R / k - 1.0)
        upd = d < best
        best = np.where(upd, d, best); bestk = np.where(upd, k, bestk)
    action = np.isfinite(R) & (best < FB_TOL) & (np.abs(R - 1.0) > FB_MIN_MOVE)
    e = np.where(action, bestk, 1.0)
    eff[todo] = e
    rpos = np.where(action)
    rep = pd.DataFrame({
        "symbol": sub.columns[rpos[1]],
        "date":   [d.date() for d in sub.index[rpos[0]]],
        "price_mult": np.round(e[rpos], 6),
        "source": "heuristic",
    })
    return eff, rep


def build_price_caches(long):
    # One row per (date, symbol) so .pivot is unambiguous and fast.
    long = long.sort_values("date").drop_duplicates(["date", "symbol"], keep="last")

    print("Pivoting to wide matrices (close / open / volume)…")
    close_w = long.pivot(index="date", columns="symbol", values="close").sort_index()
    open_w  = long.pivot(index="date", columns="symbol", values="open").reindex_like(close_w)
    vol_w   = long.pivot(index="date", columns="symbol", values="volume").reindex_like(close_w)

    splits = pd.read_csv(SPLITS_CSV) if os.path.exists(SPLITS_CSV) else None
    if splits is None:
        print("⚠ nse_data/splits.csv not found — run fetch_splits.py for correct "
              "adjustment. Falling back to the conservative price-jump heuristic only.")
    print("Applying corporate-action adjustment "
          f"({0 if splits is None else len(splits)} authoritative split events)…")

    eff_a, rep_a = authoritative_eff(close_w, splits)
    covered = set(rep_a["symbol"]) if len(rep_a) else set()
    # symbols that HAVE authoritative data at all (even if no events) shouldn't
    # also get heuristic treatment → treat every symbol present in splits.csv as covered.
    if splits is not None:
        covered |= set(splits["symbol"])
    eff_b, rep_b = fallback_eff(close_w, covered)

    eff = eff_a * eff_b
    suffix     = eff.iloc[::-1].cumprod().iloc[::-1]
    adj_factor = suffix.shift(-1).fillna(1.0)
    adj_close  = close_w * adj_factor
    adj_open   = open_w  * adj_factor
    # Volume must move OPPOSITE to price so turnover (price×volume) stays split-
    # invariant and shares are expressed in the same adjusted units as price.
    # (Previously raw volume was paired with adjusted price → liquidity/ADV and
    #  whole-share affordability were wrong around splits.)
    adj_vol    = vol_w / adj_factor.replace(0, np.nan)

    report = pd.concat([rep_a, rep_b], ignore_index=True).sort_values(["date", "symbol"])

    # Expose recovered delisted/renamed names under the universe-history symbol
    # (duplicate the column — never drop the source, which other names may use).
    def _alias(df):
        for univ, bhav in ALIASES.items():
            if bhav in df.columns and univ not in df.columns:
                df[univ] = df[bhav]
        return df

    return _alias(adj_close).sort_index(), _alias(adj_open).sort_index(), _alias(adj_vol).sort_index(), report


# ── Index bhavcopy → regime cache ───────────────────────────────────────────
def build_regime_cache():
    files = sorted(glob.glob(IDX_GLOB))
    if not files:
        print("No index bhavcopy found — skipping regime cache "
              "(keep the yfinance regime_data_cache.csv).")
        return None
    print(f"Parsing {len(files)} index bhavcopy files…")
    rows = []
    for f in files:
        d = pd.Timestamp(os.path.basename(f)[4:12])
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        df.columns = [c.strip() for c in df.columns]
        name_col  = "Index Name"
        close_col = next((c for c in df.columns if "Closing" in c), None)
        if name_col not in df.columns or close_col is None:
            continue
        for _, r in df.iterrows():
            col = INDEX_MAP.get(str(r[name_col]).strip())
            if col:
                rows.append({"date": d, "col": col,
                             "val": pd.to_numeric(r[close_col], errors="coerce")})
    if not rows:
        return None
    long = pd.DataFrame(rows)
    wide = long.pivot_table(index="date", columns="col", values="val", aggfunc="last").sort_index()
    return wide


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    argparse.ArgumentParser().parse_args()   # no options; keeps --help working

    universe_symbols = set()
    if os.path.exists(UNIVERSE_HISTORY):
        universe_symbols = set(pd.read_csv(UNIVERSE_HISTORY)["ticker"].astype(str))

    long = load_all_equity()
    adj_close, adj_open, vol, report = build_price_caches(long)

    adj_close.to_csv(OUT_PRICE)
    adj_open.to_csv(OUT_OPEN)
    vol.to_csv(OUT_VOLUME)
    report.to_csv(OUT_REPORT, index=False)

    regime = build_regime_cache()
    if regime is not None:
        regime.to_csv(OUT_REGIME)

    # ── Coverage report — the payoff vs yfinance ──
    print("\n" + "=" * 60)
    print("  BIAS-FREE CACHE BUILT")
    print("=" * 60)
    print(f"  Price cache : {adj_close.shape[1]} symbols × {adj_close.shape[0]} days "
          f"→ {OUT_PRICE}")
    print(f"  Span        : {adj_close.index.min().date()} → {adj_close.index.max().date()}")
    print(f"  Splits/bonus: {len(report)} adjustments → {OUT_REPORT}")
    if universe_symbols:
        covered = universe_symbols & set(adj_close.columns)
        missing = sorted(universe_symbols - set(adj_close.columns))
        print(f"  Universe coverage: {len(covered)}/{len(universe_symbols)} "
              f"({100*len(covered)/len(universe_symbols):.0f}%) of historical tickers priced")
        print(f"    (yfinance covered 373/420 = 89%; gap was delisted names)")
        if missing:
            print(f"  Still missing {len(missing)}: {', '.join(missing[:15])}"
                  f"{' …' if len(missing) > 15 else ''}")
    if regime is not None:
        print(f"  Regime cache: {list(regime.columns)} → {OUT_REGIME}")
    print("\n  Next: set cfg.USE_NSE_BHAVCOPY = True, then run  python backtester.py")


if __name__ == "__main__":
    main()

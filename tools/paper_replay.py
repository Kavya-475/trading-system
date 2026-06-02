"""
tools/paper_replay.py — walk-forward PAPER replay with realistic next-open fills
================================================================================
Replays the LIVE strategy day-by-day from --start, exactly as execution.py would
have run each evening — but with honest fills:

  • On day T (using only data up to T's close — NO look-ahead) it ranks the
    universe, decides exits/rotations/buys, and places limit orders.
  • On day T+1 those orders FILL AT THE OPEN — a buy fills only if the open is
    at/below its limit (else it's missed, a gap-up), a sell fills only if the
    open is at/above its limit. Cost basis = the actual open fill price.
  • Cash, transaction costs and FIFO average price are tracked through every fill.

It reuses the real strategy logic from signals.py (so it matches execution.py:
compute_scores, select_portfolio, check_exit_signals, get_regime) and txn_cost
from backtester.py. It honours the current config — regime via FORCE_RISK_ON,
EXIT_RANK_CUTOFF, TOP_N, sector cap, DMA, the buy/sell buffers.

At the end it prints a day-by-day log + final P&L (marked to the last close) and,
unless --dry-run, writes current_holdings.json with the open-price cost basis so
the live paper state is accurate. Orders placed on the LAST replay day are shown
as "pending" — those are what the next live (3:40pm) run actually places.

Needs the live yfinance cache to span the window, so run it on the VM:
    python tools/paper_replay.py --start 2026-05-15
    python tools/paper_replay.py --start 2026-05-15 --dry-run     # don't touch holdings
"""
import argparse
import contextlib
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import config as cfg
from signals import (UNIVERSE, compute_scores, apply_liquidity_filter,
                     select_portfolio, check_exit_signals, get_regime)
from backtester import txn_cost
from data_manager import load_for_signals, load_index_data

HOLDINGS_FILE = os.path.join(HERE, "current_holdings.json")


def _buy_limit(px, rank):
    b = cfg.BUY_BUFFER_TOP5 if rank < 5 else cfg.BUY_BUFFER_MID if rank < 12 else cfg.BUY_BUFFER_REST
    return round(px * (1 + b), 1)


def _last(df, t):
    s = df[t].dropna() if t in df.columns else pd.Series(dtype=float)
    return float(s.iloc[-1]) if len(s) else 0.0


def _quiet(fn, *a, **k):
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
        return fn(*a, **k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="first replay day, e.g. 2026-05-15")
    ap.add_argument("--end", default=None, help="last replay day (default: last cached day)")
    ap.add_argument("--capital", type=float, default=float(os.getenv("TRADING_CAPITAL", "100000")))
    ap.add_argument("--dry-run", action="store_true", help="don't overwrite current_holdings.json")
    args = ap.parse_args()

    close, volume = load_for_signals()
    nifty500, _, _, _ = load_index_data()
    open_df = pd.read_csv(cfg.OPEN_CACHE, index_col=0, parse_dates=True)
    # Sort + dedupe so date slicing is robust to unsorted / duplicated index rows
    # (a freshly rebuilt cache can come back out of order).
    close = close.sort_index(); close = close[~close.index.duplicated(keep="last")]
    volume = volume.sort_index(); volume = volume[~volume.index.duplicated(keep="last")]
    nifty500 = nifty500.sort_index(); nifty500 = nifty500[~nifty500.index.duplicated(keep="last")]
    open_df = open_df.sort_index(); open_df = open_df[~open_df.index.duplicated(keep="last")]

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else close.index.max()
    days = close.index[(close.index >= start) & (close.index <= end)]
    if len(days) < 2:
        sys.exit(f"Need ≥2 trading days in range; cache covers to {close.index.max().date()}.")

    universe = list(UNIVERSE.keys())
    holdings = {}            # ticker -> {shares, avg_price, entry_date}
    cash = args.capital
    pending = []             # orders placed the prior day, fill at this day's open
    force_on = getattr(cfg, "FORCE_RISK_ON", False)

    print(f"PAPER REPLAY  {days[0].date()} → {days[-1].date()}  ({len(days)} trading days)")
    print(f"  capital ₹{args.capital:,.0f} · regime {'OFF (always invested)' if force_on else 'ON'} · "
          f"TOP_N {cfg.TOP_N} · exit-rank {cfg.EXIT_RANK_CUTOFF}\n")

    final_pending = []
    for di, day in enumerate(days):
        fills = []
        # ── 1. fill yesterday's orders at TODAY's open ──────────────────────
        for o in pending:
            t = o["ticker"]
            opx = float(open_df.loc[day, t]) if (day in open_df.index and t in open_df.columns
                                                 and not pd.isna(open_df.loc[day, t])) else None
            if opx is None:
                fills.append(f"no-open {t} (no fill)"); continue
            if o["side"] == "buy":
                if opx <= o["limit"]:
                    val = o["shares"] * opx
                    tot = val + txn_cost(val, "buy")
                    if cash >= tot:
                        ex = holdings.get(t, {}).get("shares", 0)
                        avg = holdings.get(t, {}).get("avg_price", 0.0)
                        ns = ex + o["shares"]
                        navg = (avg * ex + opx * o["shares"]) / ns if ex > 0 else opx
                        holdings[t] = {"shares": ns, "avg_price": round(navg, 2),
                                       "entry_date": holdings.get(t, {}).get("entry_date", str(day.date()))}
                        cash -= tot
                        fills.append(f"BUY  {t} ×{o['shares']} @ {opx:.1f}")
                    else:
                        fills.append(f"BUY  {t} skipped (cash ₹{cash:,.0f} < ₹{tot:,.0f})")
                else:
                    fills.append(f"BUY  {t} MISSED (open {opx:.1f} > limit {o['limit']:.1f})")
            else:  # sell
                if opx >= o["limit"]:
                    sh = min(o["shares"], holdings.get(t, {}).get("shares", 0))
                    if sh > 0:
                        val = sh * opx
                        cash += val - txn_cost(val, "sell")
                        rem = holdings[t]["shares"] - sh
                        if rem > 0:
                            holdings[t]["shares"] = rem
                        else:
                            holdings.pop(t, None)
                        fills.append(f"SELL {t} ×{sh} @ {opx:.1f}")
                else:
                    fills.append(f"SELL {t} MISSED (open {opx:.1f} < limit {o['limit']:.1f})")
        pending = []

        # ── 2. rank as of TODAY's close (no look-ahead) ─────────────────────
        close_t = close[close.index <= day]
        vol_t = volume[volume.index <= day]
        n5_t = nifty500[nifty500.index <= day]
        regime = "RISK-ON" if force_on else _quiet(get_regime, n5_t)

        mtm = sum(h["shares"] * _last(close_t, t) for t, h in holdings.items())
        pv = cash + mtm
        orders = []

        if regime == "RISK-OFF":
            for t, h in holdings.items():
                orders.append({"ticker": t, "side": "sell", "shares": h["shares"],
                               "limit": round(_last(close_t, t) * (1 - cfg.SELL_BUFFER), 1)})
            decision = "RISK-OFF → liquidate"
        else:
            liquid = _quiet(apply_liquidity_filter, close_t, vol_t, universe)
            scored = _quiet(compute_scores, close_t, liquid)
            port = _quiet(select_portfolio, scored, close_t) if not scored.empty else pd.DataFrame()
            topn = port.index.tolist() if not port.empty else []
            exits = _quiet(check_exit_signals, close_t, scored, list(holdings)) if not scored.empty else []
            # Force a full build on a rebalance day OR whenever the book is empty
            # (mirrors execution.py: rebalance = is_rebalance_day() or len==0).
            is_reb = (day.day <= 3 and day.weekday() < 5) or (not holdings)

            to_sell = {t for t in holdings if t in exits}
            if is_reb:
                to_sell |= {t for t in holdings if t not in topn}
            for t in to_sell:
                orders.append({"ticker": t, "side": "sell", "shares": holdings[t]["shares"],
                               "limit": round(_last(close_t, t) * (1 - cfg.SELL_BUFFER), 1)})

            buy_list = topn if is_reb else [t for t in topn if t not in holdings][:max(0, len(exits))]

            def _heldval(t):  # current value of a name we're NOT selling (0 if being sold)
                return (holdings.get(t, {}).get("shares", 0) * _last(close_t, t)) if t not in to_sell else 0.0

            order_sh = {}                                   # pass 1: equal target = pv / TOP_N
            target = pv / max(1, cfg.TOP_N)
            for t in buy_list:
                px = _last(close_t, t)
                if px <= 0 or px > target or _heldval(t) >= target * 0.95:
                    continue
                s = int((target - _heldval(t)) / px)
                if s > 0:
                    order_sh[t] = s
            bought = [t for t in buy_list if t in order_sh]
            if bought:                                      # pass 2: redistribute leftover cash
                spent = sum(order_sh[t] * _last(close_t, t) for t in bought)
                held_keep = sum(_heldval(t) for t in bought)
                leftover = pv - held_keep - spent
                if leftover > 500:
                    t2 = (held_keep + spent + leftover) / len(bought)
                    for t in bought:
                        px = _last(close_t, t)
                        cur = _heldval(t) + order_sh[t] * px
                        if px <= t2 and cur < t2 * 0.95:
                            add = int((t2 - cur) / px)
                            if add > 0:
                                order_sh[t] += add
            for rank, t in enumerate(buy_list):
                if t in order_sh:
                    orders.append({"ticker": t, "side": "buy", "shares": order_sh[t],
                                   "limit": _buy_limit(_last(close_t, t), rank)})
            decision = f"{regime} | {'REBAL' if is_reb else 'monitor'} | top{len(topn)} | exits {exits or '-'}"

        # orders placed on the last replay day fill BEYOND the window → report as pending
        if di < len(days) - 1:
            pending = orders
        else:
            final_pending = orders

        held = sorted(holdings, key=lambda t: -holdings[t]["shares"] * _last(close_t, t))
        print(f"{day.date()}  pv ₹{pv:,.0f}  cash ₹{cash:,.0f}  [{len(held)} held]  {decision}")
        for f in fills:
            print(f"      ↳ {f}")

    # ── final P&L (marked to last close) ────────────────────────────────────
    last_day = days[-1]
    cl = close.loc[:last_day]
    total_cost = sum(h["shares"] * h["avg_price"] for h in holdings.values())
    total_val = sum(h["shares"] * _last(cl, t) for t, h in holdings.items())
    print("\n" + "=" * 60)
    print(f"  FINAL PORTFOLIO as of {last_day.date()}")
    print("=" * 60)
    for t in sorted(holdings, key=lambda t: -holdings[t]["shares"] * _last(cl, t)):
        h = holdings[t]; cur = _last(cl, t); val = h["shares"] * cur
        pnl = val - h["shares"] * h["avg_price"]; pct = pnl / (h["shares"] * h["avg_price"]) * 100 if h["avg_price"] else 0
        print(f"  {t:<12} ×{h['shares']:<5} avg {h['avg_price']:>8.1f}  now {cur:>8.1f}  "
              f"PnL {pnl:>+8.0f} ({pct:>+5.1f}%)")
    inv_pnl = total_val - total_cost
    equity = cash + total_val
    print("-" * 60)
    print(f"  invested cost ₹{total_cost:,.0f}  value ₹{total_val:,.0f}  → position PnL ₹{inv_pnl:+,.0f}")
    print(f"  cash ₹{cash:,.0f}   TOTAL EQUITY ₹{equity:,.0f}   (net ₹{equity-args.capital:+,.0f}, "
          f"{(equity/args.capital-1)*100:+.2f}% from ₹{args.capital:,.0f})")
    if final_pending:
        print(f"\n  Orders placed {last_day.date()} (fill next session — i.e. the live 3:40pm run):")
        for o in final_pending:
            print(f"    {o['side'].upper()} {o['ticker']} ×{o['shares']} limit {o['limit']}")

    if args.dry_run:
        print("\n  --dry-run: current_holdings.json NOT modified.")
    else:
        if os.path.exists(HOLDINGS_FILE):
            import shutil
            shutil.copy(HOLDINGS_FILE, HOLDINGS_FILE + ".prereplay.bak")
        with open(HOLDINGS_FILE, "w") as f:
            json.dump(holdings, f, indent=2)
        print(f"\n  Wrote {HOLDINGS_FILE} (backup: current_holdings.json.prereplay.bak)")


if __name__ == "__main__":
    main()

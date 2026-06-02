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
from data_manager import load_for_signals, load_index_data
import paper_engine as pe

HOLDINGS_FILE = os.path.join(HERE, "current_holdings.json")




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
    realized = 0.0           # cumulative realized P&L (after costs)
    strength = 1.0           # binary regime → fully deployed when RISK-ON
    pending = []             # orders placed the prior day, fill at this day's open
    force_on = getattr(cfg, "FORCE_RISK_ON", False)

    print(f"PAPER REPLAY  {days[0].date()} → {days[-1].date()}  ({len(days)} trading days)")
    print(f"  capital ₹{args.capital:,.0f} · regime {'OFF (always invested)' if force_on else 'ON'} · "
          f"TOP_N {cfg.TOP_N} · exit-rank {cfg.EXIT_RANK_CUTOFF}\n")

    final_pending = []
    for di, day in enumerate(days):
        # ── 1. confirm yesterday's orders at TODAY's open (shared engine) ───
        opens_today = {t: float(open_df.loc[day, t]) for t in open_df.columns
                       if day in open_df.index and not pd.isna(open_df.loc[day, t])}
        fills, cash, realized, pending = pe.fill_orders(
            pending, opens_today, holdings, cash, realized, str(day.date()))

        # ── 2. rank as of TODAY's close (no look-ahead) ─────────────────────
        close_t = close[close.index <= day]
        vol_t = volume[volume.index <= day]
        n5_t = nifty500[nifty500.index <= day]
        regime = "RISK-ON" if force_on else _quiet(get_regime, n5_t)

        mtm = sum(h["shares"] * _last(close_t, t) for t, h in holdings.items())
        pv = cash + mtm
        scored = pd.DataFrame()
        if regime == "RISK-OFF":
            top_n, exits = [], []
            sell_list = [t for t in holdings if pe.shares_of(holdings, t) > 0]
            decision = "RISK-OFF → liquidate"
        else:
            liquid = _quiet(apply_liquidity_filter, close_t, vol_t, universe)
            scored = _quiet(compute_scores, close_t, liquid)
            port = _quiet(select_portfolio, scored, close_t) if not scored.empty else pd.DataFrame()
            top_n = port.index.tolist() if not port.empty else []
            exits = _quiet(check_exit_signals, close_t, scored, list(holdings)) if not scored.empty else []
            # Rebalance day (or empty book) → sell anything not in top_n (rotation);
            # monitor day → sell only exits. Buys auto-deploy available cash.
            is_reb = (day.day <= 3 and day.weekday() < 5) or (not holdings)
            sell_list = ([t for t in holdings if t not in top_n] if is_reb else list(exits))
            decision = f"{regime} | {'REBAL' if is_reb else 'monitor'} | top{len(top_n)} | exits {exits or '-'}"

        ranks = {t: i for i, t in enumerate(scored.index)} if not scored.empty else {}
        close_px = {t: _last(close_t, t) for t in set(top_n) | set(holdings)}
        orders = pe.generate_orders(close_px, top_n, ranks, holdings, cash,
                                    sell_list=sell_list, strength=strength)

        # orders placed on the last replay day fill BEYOND the window → report as pending
        if di < len(days) - 1:
            pending = pending + orders          # still-working (from fill_orders) + new
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
        print("\n  --dry-run: current_holdings.json / paper_state.json NOT modified.")
    else:
        import shutil
        for o in final_pending:                       # tag with the placement day
            o["placed"] = str(last_day.date())
        if os.path.exists(HOLDINGS_FILE):
            shutil.copy(HOLDINGS_FILE, HOLDINGS_FILE + ".prereplay.bak")
        with open(HOLDINGS_FILE, "w") as f:
            json.dump(holdings, f, indent=2)
        # Seed the live paper ledger so execution.py continues from the replay:
        # the final day's orders become the working (pending) orders to confirm.
        state_file = os.path.join(HERE, "paper_state.json")
        with open(state_file, "w") as f:
            json.dump({"capital": args.capital, "cash": round(cash, 2),
                       "realized": round(realized, 2), "pending": final_pending,
                       "last_date": str(last_day.date())}, f, indent=2)
        print(f"\n  Wrote {HOLDINGS_FILE} (backup .prereplay.bak)\n  Wrote {state_file} "
              f"(cash ₹{cash:,.0f}, {len(final_pending)} working orders)")


if __name__ == "__main__":
    main()

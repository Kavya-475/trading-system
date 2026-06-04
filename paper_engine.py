"""
paper_engine.py — the order-sizing + fill simulator (shared by live-paper and replay)
=====================================================================================
"Paper trading" = trading with fake money but a REALISTIC simulation, to prove the
system works before risking real capital. This module is the engine that (a) decides
exactly how many shares to buy/sell, and (b) simulates whether those orders fill. It's
deliberately the SAME code used by:
  • execution.py            — one real day at a time, state saved in paper_state.json
  • tools/paper_replay.py   — the same step looped over history to backfill/validate
so paper results and the live path can never silently diverge.

It mirrors the backtester's model exactly, which is what makes paper results comparable:

  HOW ORDERS ARE SIZED (generate_orders):
    Capital is split by TIERED weights (top-ranked names get more) — see tiered_weights.
    It's a two-pass fill: pass 1 buys each name toward its target with the cash on hand;
    pass 2 sweeps any leftover rupees into the highest-rank affordable name. A name too
    expensive for even one share is skipped and its budget flows elsewhere.

  HOW FILLS ARE SIMULATED (fill_orders) — the realistic, no-look-ahead part:
    An order decided using day T's CLOSE is a LIMIT order that fills at day T+1's OPEN —
    a BUY fills only if the open is at/below its limit, a SELL only if at/above. If the
    market gaps through the limit, the order is MISSED (just like real life). The cost
    basis recorded is the ACTUAL open fill, not the close we decided at.

  SELL→CONFIRM→BUY SEQUENCING: buys only spend cash that has actually settled, so a sell
    placed today frees cash only when it confirms tomorrow — money can't be double-spent.

Holdings format: {ticker: {shares, avg_price, entry_date, lots:[{shares,price,date}...]}}
  where `lots` is the real per-purchase record (see the PER-LOT TRACKING section) and
  shares/avg_price/entry_date are derived from it for convenient display.
"""
import config as cfg
from costs import txn_cost


def tiered_weights(n):
    """Per-position capital weights (sum to 1), rank order. Mirrors
    backtester.rank_weights: 'equal' = 1/n, 'tiered' = RANK_TIER_WEIGHTS."""
    if n <= 0:
        return []
    if getattr(cfg, "POSITION_WEIGHTING", "equal") != "tiered":
        return [1.0 / n] * n
    tw = getattr(cfg, "RANK_TIER_WEIGHTS", (1.5, 1.1, 0.7))
    raw = [tw[0] if i < 5 else (tw[1] if i < 12 else tw[2]) for i in range(n)]
    s = sum(raw)
    return [r / s for r in raw]


def _buy_buffer(rank):
    """How far ABOVE the close we set a buy limit, so the order actually fills at the
    next open instead of being left behind by a small gap-up. Higher-conviction (better
    rank) names get a wider buffer — we're more willing to chase them a little."""
    return (cfg.BUY_BUFFER_TOP5 if rank < 5 else
            cfg.BUY_BUFFER_MID if rank < 12 else cfg.BUY_BUFFER_REST)


def shares_of(holdings, t):
    """Shares currently held of ticker t (0 if none). Tolerates both the rich dict
    format and a bare integer, so it works on any holdings snapshot."""
    v = holdings.get(t, {})
    return v.get("shares", 0) if isinstance(v, dict) else int(v or 0)


# ── PER-LOT TRACKING (shared by live + paper) ────────────────────────────────
# Each position keeps a `lots` list — the real demat acquisition record:
#   {ticker: {shares, avg_price, entry_date, lots:[{shares, price, date}, ...]}}
# `lots` is the source of truth; shares/avg_price/entry_date are DERIVED (kept for
# backward-compatible display). Sells consume lots FIFO so per-lot cost basis and
# holding period (for STCG/LTCG) are preserved exactly as the backtester does.

def _recompute(entry):
    """Refresh derived shares/avg_price/entry_date from the lot list."""
    lots = entry.get("lots", [])
    sh   = sum(l["shares"] for l in lots)
    entry["shares"] = sh
    if sh > 0:
        entry["avg_price"]  = round(sum(l["shares"] * l["price"] for l in lots) / sh, 2)
        entry["entry_date"] = min(l["date"] for l in lots)   # oldest remaining lot
    else:
        entry["avg_price"] = 0.0
    return entry


def migrate_lots(holdings):
    """Ensure every position has a `lots` list. Legacy {shares,avg_price,entry_date}
    (or a bare int) → one synthesized lot. Mutates and returns holdings."""
    for t, v in list(holdings.items()):
        if not isinstance(v, dict):
            v = {"shares": int(v or 0), "avg_price": 0.0, "entry_date": "unknown"}
            holdings[t] = v
        if "lots" not in v:
            sh = v.get("shares", 0)
            v["lots"] = ([{"shares": sh, "price": v.get("avg_price", 0.0),
                           "date": v.get("entry_date") or "unknown"}] if sh > 0 else [])
            _recompute(v)
    return holdings


def add_lot(holdings, ticker, shares, price, date):
    """Record a BUY as a new acquisition lot; refresh derived fields."""
    if shares <= 0:
        return
    entry = holdings.setdefault(ticker, {"lots": []})
    entry.setdefault("lots", []).append(
        {"shares": int(shares), "price": round(float(price), 2), "date": str(date)})
    _recompute(entry)


def realize_fifo(holdings, ticker, shares_to_sell, fill_px, sell_date):
    """Consume lots FIFO for a (partial or full) sell. Returns
    (gross_pnl, consumed) where consumed = [(shares, cost_price, buy_date), ...] for
    per-lot tax. Refreshes derived fields; pops the position when fully sold. Cost
    (brokerage/STT) is NOT subtracted here — the caller applies it."""
    entry = holdings.get(ticker)
    if not isinstance(entry, dict):
        return 0.0, []
    # Self-heal: a position with shares but no lots (un-migrated) → synthesize one.
    if not entry.get("lots") and entry.get("shares", 0) > 0:
        entry["lots"] = [{"shares": entry["shares"], "price": entry.get("avg_price", 0.0),
                          "date": entry.get("entry_date") or "unknown"}]
    lots, remaining, gross, consumed = entry.get("lots", []), int(shares_to_sell), 0.0, []
    while remaining > 0 and lots:
        lot  = lots[0]
        take = min(remaining, lot["shares"])
        gross += (fill_px - lot["price"]) * take
        consumed.append((take, lot["price"], lot["date"]))
        lot["shares"] -= take
        remaining     -= take
        if lot["shares"] <= 0:
            lots.pop(0)
    if lots:
        _recompute(entry)
    else:
        holdings.pop(ticker, None)
    return gross, consumed


def generate_orders(close_px, top_n, ranks, holdings, cash, *,
                    sell_list, strength=1.0, concentration_cap=1.5):
    """Build the LIMIT orders to place at today's close.

    close_px : {ticker: close}          top_n     : ranked list of target names
    ranks    : {ticker: rank_index}     holdings  : {ticker:{shares,...}}
    sell_list: tickers to sell in full (exits on a monitor day, or names dropped
               from top_n on a rebalance day) — caller decides.

    SELLS are placed for sell_list. BUYS deploy only the AVAILABLE CASH (not
    unsettled sell proceeds) into top_n names below their tiered target — so a
    sell placed today frees cash only when it confirms tomorrow, and its buy is
    placed the run after (the requested sell→confirm→buy sequencing). Tiered
    targets are sized off total equity; the spend each run is capped by cash on
    hand. Costly high-rank names get a single share; leftover cash is swept into
    the highest-rank affordable name (capped at concentration_cap × its slot).
    """
    orders = []
    for t in sell_list:
        sh = shares_of(holdings, t)
        px = close_px.get(t, 0)
        if sh > 0 and px > 0:
            orders.append({"ticker": t, "side": "sell", "shares": sh,
                           "limit": round(px * (1 - cfg.SELL_BUFFER), 1)})

    equity = cash + sum(shares_of(holdings, t) * close_px.get(t, 0) for t in holdings)
    deploy = equity * strength
    w = tiered_weights(len(top_n))
    budget = cash * strength                            # spend only settled cash on hand
    plan = {}                                           # ticker -> shares to BUY

    def held_val(t):                                    # value kept (0 if selling)
        return 0.0 if t in sell_list else shares_of(holdings, t) * close_px.get(t, 0)

    if budget > 500:
        # Pass 1 — move each under-target name toward its tiered slot, in rank
        # order, capped by remaining budget.
        for i, t in enumerate(top_n):
            if budget <= 0:
                break
            px = close_px.get(t, 0)
            if px <= 0:
                continue
            tgt = deploy * w[i]
            short = tgt - held_val(t)
            if short < px:                              # ~at target already
                if held_val(t) == 0 and px <= budget and px <= 2 * tgt:
                    plan[t] = 1; budget -= px           # costly high-rank single share
                continue
            n = int(min(short, budget) / px)
            if n > 0:
                plan[t] = plan.get(t, 0) + n
                budget -= n * px
        # Pass 2 — sweep leftover cash into the highest-rank affordable name that
        # is still below concentration_cap × its slot (skips unaffordable names).
        changed = True
        while budget > 0 and changed:
            changed = False
            for i, t in enumerate(top_n):
                px = close_px.get(t, 0)
                if px <= 0 or px > budget:
                    continue
                cur = held_val(t) + plan.get(t, 0) * px
                if cur < deploy * w[i] * concentration_cap:
                    plan[t] = plan.get(t, 0) + 1
                    budget -= px
                    changed = True
                    break

    for t in top_n:
        if plan.get(t, 0) > 0:
            orders.append({"ticker": t, "side": "buy", "shares": plan[t],
                           "limit": round(close_px[t] * (1 + _buy_buffer(ranks.get(t, 99))), 1)})
    return orders


def fill_orders(pending, opens, holdings, cash, realized, today):
    """Confirm yesterday's LIMIT orders against today's OPEN. Sells are processed
    first so their proceeds fund the buys (mirrors the backtester's same-bar
    rebalance). Mutates `holdings`; returns (confirmations, cash, realized, still_open).

    Buy fills if open ≤ limit, sell fills if open ≥ limit, else MISSED. No open
    price yet → kept working (returned in still_open)."""
    confirms, still_open = [], []
    for o in sorted(pending, key=lambda x: 0 if x["side"] == "sell" else 1):
        t, side, sh, lim = o["ticker"], o["side"], o["shares"], o["limit"]
        op = opens.get(t)
        if op is None or op <= 0:
            still_open.append(o)
            confirms.append(f"⏳ {side.upper()} {t} ×{sh} — no open yet, still working")
            continue
        if side == "buy":
            if op <= lim:
                val = sh * op
                tc = txn_cost(val, "buy")
                if cash >= val + tc:
                    add_lot(holdings, t, sh, op, today)     # records a new acquisition lot
                    cash -= (val + tc)
                    confirms.append(f"✅ BUY  {t} ×{sh} filled @ ₹{op:.1f}")
                else:
                    confirms.append(f"❌ BUY  {t} ×{sh} unfilled — short ₹{val+tc-cash:,.0f} cash")
            else:
                confirms.append(f"⚠️ BUY  {t} ×{sh} MISSED — open ₹{op:.1f} > limit ₹{lim:.1f} (gap-up)")
        else:  # sell
            cur = shares_of(holdings, t)
            n = min(sh, cur)
            if op >= lim and n > 0:
                val = n * op
                tc = txn_cost(val, "sell")
                gross, _ = realize_fifo(holdings, t, n, op, today)   # FIFO lot cost basis
                realized += gross - tc
                cash += val - tc
                confirms.append(f"✅ SELL {t} ×{n} filled @ ₹{op:.1f}")
            elif n > 0:
                confirms.append(f"⚠️ SELL {t} ×{n} MISSED — open ₹{op:.1f} < limit ₹{lim:.1f} (gap-down)")
    return confirms, cash, realized, still_open

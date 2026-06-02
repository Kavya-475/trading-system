"""
paper_engine.py — shared paper-trading engine that mirrors backtester.py
========================================================================
One source of truth for paper fills + sizing, used by both:
  • execution.py  (one step per day, state persisted in paper_state.json)
  • tools/paper_replay.py  (the same step looped over history)

It reproduces the backtester's model exactly:
  • TIERED position sizing (rank_weights → RANK_TIER_WEIGHTS), two-pass with
    leftover redistribution; the costly-high-rank single-share rule; pass 2
    only tops up names already in scope (so names too expensive for even one
    share are skipped, and their budget flows to affordable names).
  • Orders are placed at day T's CLOSE (the price known then) as LIMIT orders,
    and FILL AT THE NEXT OPEN (buy if open ≤ close×buffer, sell if open ≥
    close×(1-SELL_BUFFER); otherwise missed — a gap-through). Cost basis = the
    actual open fill. This matches backtester.get_fill_price.

Holdings format (execution.py's): {ticker: {shares, avg_price, entry_date}}.
"""
import config as cfg
from backtester import txn_cost


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
    return (cfg.BUY_BUFFER_TOP5 if rank < 5 else
            cfg.BUY_BUFFER_MID if rank < 12 else cfg.BUY_BUFFER_REST)


def shares_of(holdings, t):
    v = holdings.get(t, {})
    return v.get("shares", 0) if isinstance(v, dict) else int(v or 0)


def generate_orders(close_px, top_n, ranks, holdings, cash, *,
                    sell_list, buy_names, full_rebuild, strength=1.0):
    """Build the list of LIMIT orders to place at today's close.

    close_px : {ticker: close}          top_n     : ranked list of target names
    ranks    : {ticker: rank_index}     holdings  : {ticker:{shares,...}}
    sell_list: tickers to sell in full (exits and/or rotations)
    buy_names: tickers to place buy orders for — the whole top_n on a rebalance
               day, or just the replacement names on a monitor day. Each is sized
               to its tiered slot (its rank position within top_n).
    full_rebuild: True on a rebalance day → also run the pass-2 leftover
                  redistribution across in-scope names.

    Returns a list of {ticker, side, shares, limit} (shares sized off the close;
    fills happen next open via fill_orders).
    """
    orders = []
    for t in sell_list:
        sh = shares_of(holdings, t)
        px = close_px.get(t, 0)
        if sh > 0 and px > 0:
            orders.append({"ticker": t, "side": "sell", "shares": sh,
                           "limit": round(px * (1 - cfg.SELL_BUFFER), 1)})

    # Portfolio value = cash + all held positions (names being sold count as the
    # cash they'll free), exactly like the backtester's port_val for the rebalance.
    port_val = cash + sum(shares_of(holdings, t) * close_px.get(t, 0) for t in holdings)
    deploy = port_val * strength
    w = tiered_weights(len(top_n))                      # tiered slot per rank

    def held_kept(t):                                   # value we keep (0 if selling it)
        return 0.0 if t in sell_list else shares_of(holdings, t) * close_px.get(t, 0)

    plan = {}                                           # ticker -> shares to BUY

    # ── Pass 1: tiered target per name ──────────────────────────────────────
    for t in buy_names:
        if t not in top_n:
            continue
        i = top_n.index(t)
        px = close_px.get(t, 0)
        if px <= 0:
            continue
        tgt = deploy * w[i]
        cur = held_kept(t)
        if cur >= tgt * 0.95:
            continue
        n = int((tgt - cur) / px)
        if n == 0 and held_kept(t) == 0 and px <= 2 * tgt:
            n = 1                                       # costly high-rank: take one share
        if n > 0:
            plan[t] = n

    # ── Pass 2: redistribute leftover across names IN SCOPE (held or bought) ─
    # Names too expensive for even one share never enter scope, so their budget
    # flows to affordable names instead of sitting idle. Only on a full rebuild.
    if full_rebuild and plan:
        in_scope = [t for t in top_n if held_kept(t) > 0 or t in plan]
        scope_val = sum(held_kept(t) + plan.get(t, 0) * close_px.get(t, 0) for t in in_scope)
        remaining = deploy - scope_val
        if remaining > 500 and in_scope:
            w2 = tiered_weights(len(in_scope))
            total2 = scope_val + remaining
            for j, t in enumerate(in_scope):
                px = close_px.get(t, 0)
                if px <= 0:
                    continue
                new_tgt = total2 * w2[j]
                cur = held_kept(t) + plan.get(t, 0) * px
                if cur < new_tgt * 0.95:
                    add = int((new_tgt - cur) / px)
                    if add > 0:
                        plan[t] = plan.get(t, 0) + add

    for t in buy_names:
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
                    ex = shares_of(holdings, t)
                    avg = holdings.get(t, {}).get("avg_price", 0.0) if isinstance(holdings.get(t), dict) else 0.0
                    ns = ex + sh
                    navg = (avg * ex + op * sh) / ns if ex > 0 and avg > 0 else op
                    holdings[t] = {"shares": ns, "avg_price": round(navg, 2),
                                   "entry_date": (holdings.get(t, {}) or {}).get("entry_date") or today}
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
                avg = holdings.get(t, {}).get("avg_price", 0.0) if isinstance(holdings.get(t), dict) else 0.0
                realized += (op - avg) * n - tc
                cash += val - tc
                if cur - n > 0:
                    holdings[t]["shares"] = cur - n
                else:
                    holdings.pop(t, None)
                confirms.append(f"✅ SELL {t} ×{n} filled @ ₹{op:.1f}")
            elif n > 0:
                confirms.append(f"⚠️ SELL {t} ×{n} MISSED — open ₹{op:.1f} < limit ₹{lim:.1f} (gap-down)")
    return confirms, cash, realized, still_open

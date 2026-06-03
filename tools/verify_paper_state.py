"""
tools/verify_paper_state.py — integrity check for the paper-trading state files
================================================================================
Verifies current_holdings.json + paper_state.json are stored correctly:

  • Lot integrity — every position has a non-empty `lots` list; each lot has
    shares>0, price>0, a date; and the DERIVED fields are consistent:
        shares     == Σ lot shares
        avg_price  == Σ(shares·price)/Σshares   (rounded, source = lots)
        entry_date == oldest lot date
  • Ledger — capital/cash/realized present and sane; pending orders well-formed;
    pending sells reference actually-held shares.
  • Accounting identity — capital + realized − (cash + cost_basis) must equal the
    total BUY-side costs paid (≥ 0, small). A negative or large gap means the
    ledger drifted.
  • Optional mark-to-market vs the live price cache (informational).

Run on the VM after a replay / live run:
    python tools/verify_paper_state.py
Exit code 0 = all checks pass, 1 = a check failed.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

HOLDINGS = os.path.join(HERE, "current_holdings.json")
STATE    = os.path.join(HERE, "paper_state.json")
EPS      = 0.05            # ₹ tolerance for rounding in derived fields

_fails = []
def check(cond, label, detail=""):
    print(f"  {'✓' if cond else '✗'} {label}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        _fails.append(label)
    return cond


def main():
    if not os.path.exists(HOLDINGS) or not os.path.exists(STATE):
        sys.exit(f"Missing state file(s): "
                 f"{'' if os.path.exists(HOLDINGS) else HOLDINGS} "
                 f"{'' if os.path.exists(STATE) else STATE}")

    holdings = json.load(open(HOLDINGS))
    state    = json.load(open(STATE))

    print("=" * 64)
    print("  PAPER STATE INTEGRITY CHECK")
    print("=" * 64)

    # ── 1. Lot integrity + derived fields ────────────────────────────────────
    print("\n[1] Holdings — per-lot integrity")
    cost_basis = 0.0
    for t, v in holdings.items():
        if not isinstance(v, dict):
            check(False, f"{t}: dict shape", f"got {type(v).__name__}"); continue
        lots = v.get("lots")
        if not check(isinstance(lots, list) and len(lots) > 0, f"{t}: has lots"):
            continue
        ok_lots = all(isinstance(l, dict) and l.get("shares", 0) > 0
                      and l.get("price", 0) > 0 and l.get("date") for l in lots)
        check(ok_lots, f"{t}: every lot has shares>0, price>0, date")

        lot_sh   = sum(l["shares"] for l in lots)
        lot_avg  = round(sum(l["shares"] * l["price"] for l in lots) / lot_sh, 2) if lot_sh else 0
        lot_date = min(l["date"] for l in lots)
        check(v.get("shares") == lot_sh, f"{t}: shares == Σlots", f"{v.get('shares')} vs {lot_sh}")
        check(abs(v.get("avg_price", 0) - lot_avg) <= EPS,
              f"{t}: avg_price == weighted lot avg", f"{v.get('avg_price')} vs {lot_avg}")
        check(v.get("entry_date") == lot_date,
              f"{t}: entry_date == oldest lot", f"{v.get('entry_date')} vs {lot_date}")
        cost_basis += sum(l["shares"] * l["price"] for l in lots)

    held = {t: v for t, v in holdings.items() if v.get("shares", 0) > 0}
    print(f"\n  positions: {len(held)}   cost basis ₹{cost_basis:,.2f}")

    # ── 2. Ledger fields + pending orders ────────────────────────────────────
    print("\n[2] Ledger (paper_state.json)")
    for k in ("capital", "cash", "realized", "pending", "last_date"):
        check(k in state, f"has key '{k}'")
    cap, cash, realized = state.get("capital", 0), state.get("cash", 0), state.get("realized", 0)
    check(cap > 0, "capital > 0", str(cap))
    check(cash >= -EPS, "cash >= 0", f"₹{cash:,.2f}")

    pend = state.get("pending", [])
    print(f"\n  pending orders: {len(pend)}")
    for o in pend:
        ok = (o.get("ticker") and o.get("side") in ("buy", "sell")
              and o.get("shares", 0) > 0 and o.get("limit", 0) > 0 and o.get("placed"))
        check(ok, f"order well-formed: {o.get('side')} {o.get('ticker')} ×{o.get('shares')}")
        if o.get("side") == "sell":
            h = held.get(o["ticker"], {}).get("shares", 0)
            check(h >= o["shares"], f"sell {o['ticker']} backed by holding", f"have {h}, selling {o['shares']}")

    # ── 3. Accounting identity ───────────────────────────────────────────────
    # capital + realized - (cash + cost_basis) == total BUY-side costs paid (>=0).
    print("\n[3] Accounting identity")
    gap = cap + realized - (cash + cost_basis)
    print(f"  capital ₹{cap:,.2f} + realized ₹{realized:,.2f} "
          f"- (cash ₹{cash:,.2f} + cost ₹{cost_basis:,.2f}) = ₹{gap:,.2f}")
    check(gap >= -EPS, "identity gap >= 0 (= buy costs paid)", f"₹{gap:,.2f}")
    check(gap <= 0.03 * cap, "identity gap is small (< 3% of capital)", f"₹{gap:,.2f}")
    print("    (this gap = cumulative buy-side brokerage/STT — small & positive is correct)")

    # ── 4. Mark-to-market (informational, needs the price cache) ─────────────
    try:
        from execution import load_latest_prices
        px = load_latest_prices()
        if px:
            mtm = sum(v["shares"] * px.get(t, 0) for t, v in held.items())
            equity = cash + mtm
            print("\n[4] Mark-to-market (latest cache close)")
            print(f"  holdings ₹{mtm:,.0f} + cash ₹{cash:,.0f} = equity ₹{equity:,.0f}  "
                  f"(net ₹{equity-cap:+,.0f}, {(equity/cap-1)*100:+.2f}%)")
    except Exception as e:
        print(f"\n[4] MTM skipped: {e}")

    print("\n" + "=" * 64)
    if _fails:
        print(f"  RESULT: ✗ {len(_fails)} check(s) FAILED — {', '.join(_fails[:6])}"
              + (" …" if len(_fails) > 6 else ""))
        sys.exit(1)
    print("  RESULT: ✓ ALL CHECKS PASSED — paper state is stored correctly")
    sys.exit(0)


if __name__ == "__main__":
    main()

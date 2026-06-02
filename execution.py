"""
execution.py
============
Execution engine — reads signals and places orders via Kite Connect.

Three-tier execution structure:
  DAILY        : Check regime + exit signals. If exit triggered, immediately
                 buy replacement from current top 25. Sell and replace same day.
  MONTHLY      : Full portfolio review. Rotate any position overtaken by
                 significantly better ranked stock even if no exit triggered.
  IMMEDIATELY  : Regime flip RISK-OFF → sell everything, no waiting.

PAPER_MODE = True  → logs orders, touches nothing real (default)
PAPER_MODE = False → places real orders (only after API approved + tested)

Run:
    python execution.py

Schedule every weekday at 3:45 PM IST.
"""

import os
import json
import logging
from datetime import datetime, date
from dotenv import load_dotenv
import pandas as pd

import config as cfg
from signals import run_signals, UNIVERSE, compute_scores, apply_liquidity_filter
from data_manager import load_for_signals
import paper_engine as pe

# ── Load secrets from .env file ─────────────────────────────────────────────
# Must be called before any os.getenv() calls so the file values are available
load_dotenv()
KITE_API_KEY      = os.getenv("KITE_API_KEY",       "your_api_key_here")
KITE_API_SECRET   = os.getenv("KITE_API_SECRET",    "your_api_secret_here")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN",  "")   # refreshed daily by kite_login.py
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID",   "")

# ── Master switch ────────────────────────────────────────────────────────────
# PAPER_MODE = True  → logs orders but never calls Kite API (safe for testing)
# PAPER_MODE = False → places real orders via Kite Connect (only after full testing)
PAPER_MODE = True

# ── Module-level settings ────────────────────────────────────────────────────
CAPITAL        = float(os.getenv("TRADING_CAPITAL", "100000"))  # total capital to deploy
HOLDINGS_FILE  = "current_holdings.json"   # persists share counts + avg price across runs
LOG_FILE       = "orders.log"              # append-only log of every order placed

# ── Logging setup ────────────────────────────────────────────────────────────
# Writes to orders.log AND prints to console simultaneously
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console = logging.StreamHandler()           # add a second handler for console output
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# REBALANCE DAY CHECK
# ─────────────────────────────────────────────
def is_rebalance_day() -> bool:
    """
    Returns True on the first trading day of each month (calendar day 1-3 that is a weekday).
    On rebalance days: full portfolio rotation — compare all holdings against current top N
    and rotate out underperformers even if no hard exit rule triggered.
    On non-rebalance days: only hard exits (rank dropout / DMA breach) are acted on,
    and a replacement is bought immediately for each exit.
    """
    today = date.today()
    # Day 1-3 of the month AND a weekday (0=Mon, 4=Fri, 5=Sat, 6=Sun)
    return today.day <= 3 and today.weekday() < 5


# ─────────────────────────────────────────────
# LOAD LATEST PRICES FROM CACHE
# ─────────────────────────────────────────────
def load_latest_prices() -> dict:
    """
    Reads the most recent closing price for every stock from the CSV cache.
    Used to calculate limit order prices and P&L. No network calls.
    Returns a dict: {ticker: last_close_price}.
    """
    prices = {}
    if os.path.exists(cfg.DATA_CACHE_FILE):
        try:
            close = pd.read_csv(cfg.DATA_CACHE_FILE, index_col=0, parse_dates=True)
            for ticker in close.columns:
                series = close[ticker].dropna()      # remove NaN entries (weekends/holidays)
                if len(series) > 0:
                    prices[ticker] = float(series.iloc[-1])   # take the most recent price
            log.info(f"Loaded latest prices for {len(prices)} stocks from cache")
        except Exception as e:
            log.error(f"Could not read price cache: {e}")
    return prices


# ─────────────────────────────────────────────
# FIND BEST REPLACEMENT for exited position
# ─────────────────────────────────────────────
def find_replacement(
    scored: pd.DataFrame,
    current_holdings: list,
    exits: list
) -> list:
    """
    When a stock exits mid-month (rank dropout or DMA breach), immediately
    buy the best available replacement — don't wait for month-end rebalance.
    Looks in the top EXIT_RANK_CUTOFF ranked stocks for candidates not already held.
    Respects the sector cap so we don't end up over-concentrated.
    Returns a list of replacement tickers (one per exit).
    """
    # Determine which stocks we still hold after the exits
    remaining = [t for t in current_holdings if t not in exits]

    # Best candidates are top-ranked stocks we don't already hold
    top_25     = scored.head(cfg.EXIT_RANK_CUTOFF).index.tolist()
    candidates = [t for t in top_25 if t not in remaining]

    # Count how many stocks from each sector are already in remaining holdings
    sector_count = {}
    for t in remaining:
        sector = UNIVERSE.get(t, "Unknown")
        sector_count[sector] = sector_count.get(sector, 0) + 1

    # Walk candidates in rank order, pick those that fit the sector cap
    replacements = []
    for t in candidates:
        sector = UNIVERSE.get(t, "Unknown")
        if sector_count.get(sector, 0) < cfg.MAX_PER_SECTOR:
            replacements.append(t)
            sector_count[sector] = sector_count.get(sector, 0) + 1
        if len(replacements) == len(exits):    # one replacement per exit
            break

    if replacements:
        log.info(f"Replacements found: {replacements} for exits: {exits}")
    else:
        log.warning("No suitable replacements found — holding partial cash")

    return replacements


# ─────────────────────────────────────────────
# KITE CONNECT CLIENT
# ─────────────────────────────────────────────
def get_kite_client():
    """
    Returns an authenticated KiteConnect client for live order placement.
    In PAPER_MODE returns None (no real API calls are made).
    In live mode, uses the access token written to .env by kite_login.py.
    If no token is present, falls back to an interactive manual login flow.
    """
    if PAPER_MODE:
        log.info("PAPER MODE — Kite Connect not initialised")
        return None
    try:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=KITE_API_KEY)
        if KITE_ACCESS_TOKEN:
            # Normal path — token was written to .env by kite_login.py this morning
            kite.set_access_token(KITE_ACCESS_TOKEN)
            log.info("Kite Connect authenticated")
        else:
            # Fallback path — no token in .env, ask user to login manually
            login_url = kite.login_url()
            print(f"\nOpen this URL to login:\n{login_url}\n")
            request_token = input("Paste request_token from redirect URL: ").strip()
            data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
            kite.set_access_token(data["access_token"])
            with open(".env", "a") as f:
                f.write(f'\nKITE_ACCESS_TOKEN={data["access_token"]}')
            log.info("New access token saved")
        return kite
    except ImportError:
        log.error("kiteconnect not installed. Run: pip install kiteconnect")
        return None
    except Exception as e:
        log.error(f"Kite login failed: {e}")
        return None


# ─────────────────────────────────────────────
# PORTFOLIO STATE  (read/write current_holdings.json)
# ─────────────────────────────────────────────

def load_holdings() -> dict:
    """
    Reads current_holdings.json from disk.
    Each entry: {ticker: {shares, avg_price, entry_date}}.
    Handles the old format (bare integer share counts) by migrating them
    to the new dict format with avg_price=0 and a 'migrated' flag.
    Returns empty dict if the file doesn't exist yet.
    """
    if os.path.exists(HOLDINGS_FILE):
        with open(HOLDINGS_FILE, "r") as f:
            raw = json.load(f)
        migrated = {}
        for ticker, val in raw.items():
            if isinstance(val, (int, float)):
                # Old format — just an integer share count; migrate it
                migrated[ticker] = {"shares": int(val), "avg_price": 0.0, "entry_date": "unknown", "migrated": True}
            else:
                migrated[ticker] = val
        return migrated
    return {}


def save_holdings(holdings: dict):
    """
    Writes the holdings dict back to current_holdings.json.
    Also creates a dated backup in backups/ and removes backups older than 7 days.
    This runs after every execution so the state is always current.
    """
    with open(HOLDINGS_FILE, "w") as f:
        json.dump(holdings, f, indent=2)
    log.info(f"Holdings saved: { {t: get_shares(holdings,t) for t in holdings if get_shares(holdings,t) > 0} }")

    # Daily backup in backups/ — named by date so we can recover any day's state
    try:
        import shutil
        from datetime import date as _date
        _backup_dir = "backups"
        os.makedirs(_backup_dir, exist_ok=True)
        _backup_file = os.path.join(_backup_dir, f"holdings_{_date.today()}.json")
        shutil.copy(HOLDINGS_FILE, _backup_file)
        # Purge backups older than 7 days to prevent indefinite growth
        import time
        _cutoff = time.time() - (7 * 86400)
        for _f in os.listdir(_backup_dir):
            _fp = os.path.join(_backup_dir, _f)
            if os.path.getmtime(_fp) < _cutoff:
                os.remove(_fp)
    except Exception as _e:
        log.warning(f"Backup failed: {_e}")


def get_shares(holdings: dict, ticker: str) -> int:
    """Returns the number of shares held for a ticker (0 if not held)."""
    val = holdings.get(ticker, {})
    return val.get("shares", 0) if isinstance(val, dict) else int(val or 0)


def get_avg_price(holdings: dict, ticker: str) -> float:
    """Returns the weighted average buy price for a ticker (0.0 if unknown)."""
    val = holdings.get(ticker, {})
    return val.get("avg_price", 0.0) if isinstance(val, dict) else 0.0


def set_holding(holdings: dict, ticker: str, shares: int,
                avg_price: float = 0.0, entry_date: str = ""):
    """Writes or updates a position in the holdings dict (does not save to disk)."""
    holdings[ticker] = {
        "shares"    : shares,
        "avg_price" : avg_price,
        "entry_date": entry_date or str(date.today())
    }

def build_pnl_summary(holdings: dict, prices: dict) -> str:
    """
    Builds a formatted P&L string for all held positions.
    Shows each stock's current price, unrealised P&L in rupees and percent.
    Shows 'Entry: N/A' for migrated positions where avg_price is unknown.
    Appends a total P&L and total value line at the bottom.
    Used in the Telegram end-of-day notification.
    """
    lines       = []
    total_cost  = 0.0
    total_value = 0.0
    for ticker in holdings:
        shares    = get_shares(holdings, ticker)
        avg_price = get_avg_price(holdings, ticker)
        if shares <= 0:
            continue
        cur   = prices.get(ticker, 0.0)
        value = shares * cur
        total_value += value
        is_migrated = isinstance(holdings.get(ticker), dict) and holdings[ticker].get("migrated", False)
        if avg_price == 0.0 or is_migrated:
            lines.append(
                "  " + ticker.ljust(12) + " x" + str(shares).ljust(4) +
                " @ " + f"{cur:>8.1f}" +
                "  Entry: N/A"
            )
        else:
            cost      = shares * avg_price
            pnl       = value - cost
            pnl_pct   = pnl / cost * 100
            total_cost += cost
            sign = "+" if pnl >= 0 else "-"
            lines.append(
                sign + " " + ticker.ljust(12) + " x" + str(shares).ljust(4) +
                " @ " + f"{cur:>8.1f}" +
                "  PnL: " + f"{pnl:>+8.0f}" +
                " (" + f"{pnl_pct:>+.1f}" + "%)"
            )
    total_pnl     = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0
    sep = "-" * 40
    result = chr(10).join(lines)
    result += chr(10) + sep
    result += chr(10) + "Total PnL: " + (f"{total_pnl:>+,.0f} ({total_pnl_pct:>+.1f}%)" if total_cost > 0 else "N/A (migrated positions)")
    result += chr(10) + "Value:     " + f"{total_value:>10,.0f}"
    return result

def get_portfolio_value(kite) -> float:
    """
    Returns the total current portfolio value.
    In PAPER_MODE returns the fixed CAPITAL constant (no real account data).
    In live mode, queries Kite for the actual margin balance and open positions.
    Falls back to CAPITAL if the Kite API call fails.
    """
    if PAPER_MODE:
        return CAPITAL
    try:
        margins     = kite.margins(segment="equity")
        cash        = margins["net"]
        positions   = kite.positions()["net"]
        stock_value = sum(
            p["quantity"] * p["last_price"]
            for p in positions if p["quantity"] > 0
        )
        return cash + stock_value
    except Exception as e:
        log.error(f"Could not fetch portfolio value: {e}")
        return CAPITAL


# ─────────────────────────────────────────────
# ORDER PLACEMENT
# Thin wrappers around the Kite Connect API.
# In PAPER_MODE, all orders are logged but no real API call is made.
# In live mode, places a LIMIT DAY order on NSE as a CNC (delivery) product.
# Returns a dict with 'status': 'paper' | 'placed' | 'failed'.
# ─────────────────────────────────────────────
def place_buy_order(kite, ticker: str, shares: int, limit_price: float) -> dict:
    if PAPER_MODE:
        log.info(
            f"[PAPER] BUY  | {ticker:<15} | "
            f"{shares:>5} shares @ ₹{limit_price:>8.1f} | "
            f"Total: ₹{shares*limit_price:>10,.0f}"
        )
        return {"status": "paper"}
    try:
        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NSE,
            tradingsymbol=ticker,
            transaction_type=kite.TRANSACTION_TYPE_BUY,
            quantity=shares,
            product=kite.PRODUCT_CNC,
            order_type=kite.ORDER_TYPE_LIMIT,
            price=round(limit_price, 1),
            validity=kite.VALIDITY_DAY,
        )
        log.info(f"[LIVE]  BUY  | {ticker:<15} | {shares:>5} shares @ ₹{limit_price:.1f} | ID: {order_id}")
        return {"status": "placed", "order_id": order_id}
    except Exception as e:
        log.error(f"BUY failed for {ticker}: {e}")
        return {"status": "failed", "error": str(e)}


def place_sell_order(kite, ticker: str, shares: int, limit_price: float) -> dict:
    if PAPER_MODE:
        log.info(
            f"[PAPER] SELL | {ticker:<15} | "
            f"{shares:>5} shares @ ₹{limit_price:>8.1f} | "
            f"Total: ₹{shares*limit_price:>10,.0f}"
        )
        return {"status": "paper"}
    try:
        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NSE,
            tradingsymbol=ticker,
            transaction_type=kite.TRANSACTION_TYPE_SELL,
            quantity=shares,
            product=kite.PRODUCT_CNC,
            order_type=kite.ORDER_TYPE_LIMIT,
            price=round(limit_price, 1),
            validity=kite.VALIDITY_DAY,
        )
        log.info(f"[LIVE]  SELL | {ticker:<15} | {shares:>5} shares @ ₹{limit_price:.1f} | ID: {order_id}")
        return {"status": "placed", "order_id": order_id}
    except Exception as e:
        log.error(f"SELL failed for {ticker}: {e}")
        return {"status": "failed", "error": str(e)}


# ─────────────────────────────────────────────
# TELEGRAM NOTIFICATIONS
# Sends the end-of-day summary (sells, buys, P&L) to the configured chat.
# Silently skips if bot token or chat ID are not set in .env.
# ─────────────────────────────────────────────
def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.info("Telegram not configured — skipping")
        return
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"},
            timeout=10
        )
        log.info("Telegram notification sent")
    except Exception as e:
        log.error(f"Telegram failed: {e}")


# ─────────────────────────────────────────────
# PLACE BUYS FOR A LIST OF TICKERS
# Shared logic used by both the immediate-replacement path and the monthly rebalance.
# Two-pass approach:
#   Pass 1 — buy each ticker up to the equal-weight target allocation
#   Pass 2 — redistribute any leftover cash across all in-scope positions
# ─────────────────────────────────────────────
def execute_buys(kite, tickers_to_buy: list, holdings: dict,
                 prices: dict, port_value: float, strength: float = 1.0, stocks_to_hold: int = None) -> list:
    """
    Places buy orders for the given list of tickers.
    strength: regime strength 0.0-1.0 — scales down capital deployed when < 1.0.
    stocks_to_hold: total target position count (used to set target per stock).
    Returns a list of human-readable buy descriptions for the Telegram message.
    Avg price in holdings is recorded at the market close price (not limit price)
    because that is the realistic fill price on next-day open.
    """
    if stocks_to_hold is None:
        stocks_to_hold = max(1, round(cfg.TOP_N * strength))
    buy_lines        = []
    capital_deployed = port_value * strength
    target           = capital_deployed / stocks_to_hold
    log.info(f"Regime strength: {strength*100:.0f}% | Deploying {stocks_to_hold} positions @ {target:,.0f} each")

    # Assign limit order buffer per ticker based on its rank in tickers_to_buy.
    # Higher-ranked (earlier in list) stocks get a larger buffer so we really fill them.
    buf_map = {}
    for i, t in enumerate(tickers_to_buy):
        if i < 5:
            buf_map[t] = cfg.BUY_BUFFER_TOP5    # top 5 → 5% above close
        elif i < 12:
            buf_map[t] = cfg.BUY_BUFFER_MID     # ranks 6–12 → 3% above close
        else:
            buf_map[t] = cfg.BUY_BUFFER_REST    # rest → 2% above close

    def _limit(ticker, close_price):
        """Computes the limit order price = close × (1 + buffer)."""
        return round(close_price * (1 + buf_map.get(ticker, cfg.BUY_BUFFER_REST)), 1)

    # ── First pass: buy each ticker up to equal-weight target ────────────────
    bought_count = 0
    for ticker in tickers_to_buy:
        if bought_count >= stocks_to_hold:
            break                              # already at full position count

        price = prices.get(ticker, 0)
        if price <= 0:
            log.warning(f"No price for {ticker} — skipping")
            continue

        if price > target:
            # Single share costs more than the whole target allocation — skip
            log.info(f"Skipping {ticker} @ ₹{price:,.0f} — too expensive for ₹{target:,.0f} allocation")
            continue

        current_shares = get_shares(holdings, ticker)
        current_value  = current_shares * price

        if current_value < target * 0.95:    # only buy if meaningfully underweight (< 95%)
            buy_value     = target - current_value
            shares_to_buy = int(buy_value / price)      # floor to whole shares
            limit_price   = _limit(ticker, price)

            if shares_to_buy > 0:
                res = place_buy_order(kite, ticker, shares_to_buy, limit_price)
                # Log with market price (not limit) — that's the realistic fill price
                buy_lines.append(f"BUY {ticker} ×{shares_to_buy} @ ₹{price}")
                if res["status"] in ("paper", "placed"):
                    # Update holdings with new weighted average price
                    existing_shares = get_shares(holdings, ticker)
                    new_shares = existing_shares + shares_to_buy
                    old_avg = get_avg_price(holdings, ticker)
                    new_avg = (old_avg * existing_shares + price * shares_to_buy) / new_shares if existing_shares > 0 and old_avg > 0 else price
                    set_holding(holdings, ticker, new_shares, new_avg)
                    bought_count += 1
            else:
                log.info(f"Skipping {ticker} @ ₹{price:,.0f} — insufficient allocation for 1 share")

    # ── Second pass: redistribute leftover cash across bought positions ───────
    # After the first pass, some cash may remain because integer share rounding
    # means we can't buy exactly the target value. Redistribute this across
    # the in-scope positions, buying more shares where possible.
    if bought_count > 0:
        in_scope       = [t for t in tickers_to_buy if get_shares(holdings, t) > 0]
        scope_val      = sum(get_shares(holdings, t) * prices.get(t, 0) for t in in_scope)
        remaining_cash = capital_deployed - scope_val     # cash not yet deployed
        if remaining_cash > 500 and in_scope:
            new_target = (scope_val + remaining_cash) / len(in_scope)    # new equal target
            log.info(f"Redistributing ₹{remaining_cash:,.0f} unused cash across {len(in_scope)} in-scope positions")
            for ticker in in_scope:
                price = prices.get(ticker, 0)
                if price <= 0 or price > new_target:
                    continue
                current_value = get_shares(holdings, ticker) * price
                if current_value < new_target * 0.95:
                    buy_value     = new_target - current_value
                    shares_to_buy = int(buy_value / price)
                    limit_price   = _limit(ticker, price)
                    if shares_to_buy > 0:
                        res = place_buy_order(kite, ticker, shares_to_buy, limit_price)
                        buy_lines.append(f"BUY {ticker} ×{shares_to_buy} @ ₹{price} [top-up]")
                        if res["status"] in ("paper", "placed"):
                            existing_shares = get_shares(holdings, ticker)
                            new_shares = existing_shares + shares_to_buy
                            old_avg = get_avg_price(holdings, ticker)
                            new_avg = (old_avg * existing_shares + price * shares_to_buy) / new_shares if existing_shares > 0 and old_avg > 0 else price
                            set_holding(holdings, ticker, new_shares, new_avg)

    return buy_lines


# ─────────────────────────────────────────────
# MAIN EXECUTION FLOW
# Called daily at 3:45 PM IST by scheduler.py (or directly).
# Full flow:
#   1. Load current holdings from disk
#   2. Run the signal engine (regime check + momentum scores)
#   3. If RISK-OFF: sell everything
#   4. If RISK-ON: process exits, buy replacements, run monthly rebalance
#   5. Save updated holdings + send Telegram summary
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# PAPER LEDGER (cash + open limit orders, confirmed at the next open)
# Mirrors the backtester via paper_engine: orders placed at today's close fill
# at the NEXT open, and a sell's freed cash funds buys only after it confirms.
# State persists in paper_state.json: {capital, cash, realized, pending, last_date}.
# ─────────────────────────────────────────────
PAPER_STATE_FILE = "paper_state.json"


def load_latest_opens() -> dict:
    """Most recent open price per ticker from the open cache (fill reference)."""
    if not os.path.exists(cfg.OPEN_CACHE):
        return {}
    o = pd.read_csv(cfg.OPEN_CACHE, index_col=0, parse_dates=True)
    return {t: float(o[t].dropna().iloc[-1]) for t in o.columns if len(o[t].dropna())}


def load_paper_state() -> dict:
    s = {"capital": CAPITAL, "cash": CAPITAL, "realized": 0.0, "pending": [], "last_date": None}
    if os.path.exists(PAPER_STATE_FILE):
        with open(PAPER_STATE_FILE) as f:
            s.update(json.load(f))
    return s


def save_paper_state(state: dict):
    with open(PAPER_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _paper_telegram(holdings, close_px, state, fills, new_orders, regime):
    cash = state["cash"]; cap = state.get("capital", CAPITAL); realized = state.get("realized", 0.0)
    held = [t for t in holdings if get_shares(holdings, t) > 0]
    posval = sum(get_shares(holdings, t) * close_px.get(t, 0.0) for t in held)
    equity = cash + posval
    net    = equity - cap                          # true net P&L (incl. all costs)
    unreal = sum((close_px.get(t, 0.0) - get_avg_price(holdings, t)) * get_shares(holdings, t)
                 for t in held if get_avg_price(holdings, t) > 0)
    sells = [o for o in new_orders if o["side"] == "sell"]
    buys  = [o for o in new_orders if o["side"] == "buy"]

    L = [f"{'📋 PAPER' if PAPER_MODE else '✅ LIVE'} — Daily Run · {date.today()}",
         f"Regime: {regime}"]
    if fills:
        L += ["", "*Confirmed today (filled at open):*", *fills]

    # Tomorrow's orders (placed today, fill at tomorrow's open)
    L += ["", "*Tomorrow's SELL orders:*" + ("" if sells else " none")]
    L += [f"  SELL {o['ticker']} ×{o['shares']} @ ≥₹{o['limit']:,.0f}" for o in sells]
    L += ["*Tomorrow's BUY orders:*" + ("" if buys else " none")]
    L += [f"  BUY  {o['ticker']} ×{o['shares']} @ ≤₹{o['limit']:,.0f}" for o in buys]

    if held:
        L += ["", f"*Holdings ({len(held)}):*"]
        for t in sorted(held, key=lambda t: -get_shares(holdings, t) * close_px.get(t, 0)):
            sh = get_shares(holdings, t); avg = get_avg_price(holdings, t); cur = close_px.get(t, 0)
            pnl = (cur - avg) * sh; pct = (pnl / (avg * sh) * 100) if avg > 0 else 0.0
            L.append(f"  {t:<11} ×{sh:<4} @₹{cur:,.0f}  {pnl:+,.0f} ({pct:+.1f}%)")

    L += ["", "─" * 24,
          f"Cash:        ₹{cash:,.0f}",
          f"Holdings:    ₹{posval:,.0f}",
          f"*Total Value: ₹{equity:,.0f}*",
          f"*Net P&L: ₹{net:+,.0f} ({net / cap * 100:+.2f}%)*",
          f"  (realized ₹{realized:+,.0f} · unrealized ₹{unreal:+,.0f})"]
    return "\n".join(L)


def run_execution_paper(holdings, close_px, signals):
    """PAPER daily step. Confirm yesterday's open orders at today's open, then
    place today's new limit orders via paper_engine (sells for rotations/exits;
    buys deploy available cash), persist the ledger, and Telegram the result."""
    state = load_paper_state()
    opens = load_latest_opens()
    today = str(date.today())

    # 1. confirm yesterday's working orders at today's OPEN
    fills, state["cash"], state["realized"], still = pe.fill_orders(
        state.get("pending", []), opens, holdings, state["cash"], state["realized"], today)

    # 2. decide target + size today's new orders
    regime   = signals.get("regime", "RISK-ON")
    strength = signals.get("strength", 1.0)
    if cfg.FORCE_RISK_ON:
        regime, strength = "RISK-ON", 1.0
    exits  = signals.get("exits", [])
    held   = [t for t in holdings if get_shares(holdings, t) > 0]
    scored = pd.DataFrame()
    if regime == "RISK-OFF":
        top_n, sell_list = [], held
    else:
        close, volume = load_for_signals()
        liquid = apply_liquidity_filter(close, volume, list(UNIVERSE.keys()))
        scored = compute_scores(close, liquid)
        port   = signals.get("portfolio")
        top_n  = (port.index.tolist() if (port is not None and not port.empty)
                  else (scored.head(cfg.TOP_N).index.tolist() if not scored.empty else []))
        is_reb = is_rebalance_day() or len(held) == 0
        sell_list = [t for t in held if t not in top_n] if is_reb else list(exits)

    ranks = {t: i for i, t in enumerate(scored.index)} if not scored.empty else {}
    cpx   = {t: close_px.get(t, 0.0) for t in set(top_n) | set(holdings)}
    new_orders = pe.generate_orders(cpx, top_n, ranks, holdings, state["cash"],
                                    sell_list=sell_list, strength=strength)
    for o in new_orders:
        o["placed"] = today
    state["pending"], state["last_date"] = still + new_orders, today

    # 3. persist + notify
    save_holdings(holdings)
    save_paper_state(state)
    send_telegram(_paper_telegram(holdings, close_px, state, fills, new_orders, regime))
    eq = state["cash"] + sum(get_shares(holdings, t) * close_px.get(t, 0.0) for t in held)
    log.info(f"[PAPER] equity ₹{eq:,.0f} | cash ₹{state['cash']:,.0f} | "
             f"working orders {len(state['pending'])} | held {len(held)}")


def run_execution():
    if cfg.TRADING_HALTED:
        log.info("TRADING_HALTED flag set — aborting run")
        return

    log.info("=" * 55)
    log.info(f"EXECUTION RUN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"MODE          : {'PAPER' if PAPER_MODE else '⚠️  LIVE'}")
    log.info("=" * 55)

    # ── Load current portfolio state ───────────────────────────────────────
    holdings        = load_holdings()
    current_tickers = [t for t in holdings if get_shares(holdings, t) > 0]
    # Force full rebalance if portfolio is empty (first run or after RISK-OFF liquidation)
    rebalance = is_rebalance_day() or len(current_tickers) == 0
    log.info(f"Current holdings: {current_tickers or 'None (empty)'}")

    # ── Load today's closing prices from cache ─────────────────────────────
    latest_prices = load_latest_prices()
    if not latest_prices:
        log.error("No price data. Run data_manager.py first.")
        return

    # ── Connect to broker (returns None in PAPER_MODE) ─────────────────────
    kite = get_kite_client()

    # ── Run signal engine — regime + scores + exit signals ─────────────────
    log.info("Running signal engine...")
    signals        = run_signals(current_tickers)

    # ── PAPER MODE → ledger engine (open-fill confirmation, cash tracking) ──
    if PAPER_MODE:
        run_execution_paper(holdings, latest_prices, signals)
        log.info("=" * 55)
        return

    strength       = signals.get("strength", 1.0)             # 0.0–1.0 from regime strength
    stocks_to_hold = max(1, round(cfg.TOP_N * strength))      # scale position count by strength
    regime         = signals["regime"]
    if cfg.FORCE_RISK_ON:
        regime = "RISK-ON"   # override regime for paper testing — remove before going live
    portfolio = signals["portfolio"]    # DataFrame of top N scored stocks
    exits     = signals["exits"]        # list of currently held stocks that triggered exit rules
    log.info(f"Regime: {regime}")

    # ── RISK-OFF: sell everything immediately ───────────────────────────────
    if regime == "RISK-OFF":
        log.info("RISK-OFF — liquidating all positions immediately")
        sell_lines = []
        for ticker in current_tickers:
            shares = get_shares(holdings, ticker)
            if shares > 0:
                price      = latest_prices.get(ticker, 0)
                sell_price = round(price * (1 - cfg.SELL_BUFFER), 1)
                res        = place_sell_order(kite, ticker, shares, sell_price)
                sell_lines.append(f"SELL {ticker} ×{shares} @ ₹{sell_price}")
                if res["status"] in ("paper", "placed"):
                    set_holding(holdings, ticker, 0)
        save_holdings(holdings)
        send_telegram(
            f"🔴 *RISK-OFF — Market Alert*\n"
            f"Date: {date.today()}\n"
            f"All positions liquidated\n\n"
            + ("\n".join(sell_lines) if sell_lines else "No open positions")
        )
        log.info("Liquidation complete.")
        return

    # ── RISK-ON: compute scores for replacement logic ───────────────────────
    # Load raw data for scoring
    close, volume = load_for_signals()
    tickers       = list(UNIVERSE.keys())
    liquid        = apply_liquidity_filter(close, volume, tickers)
    scored        = compute_scores(close, liquid)

    buy_lines  = []
    sell_lines = []

    # ── Process exits ───────────────────────────────────────────────────────
    if exits:
        log.info(f"Exit signals: {exits}")
        for ticker in exits:
            shares = get_shares(holdings, ticker)
            if shares > 0:
                price      = latest_prices.get(ticker, 0)
                sell_price = round(price * (1 - cfg.SELL_BUFFER), 1)
                res        = place_sell_order(kite, ticker, shares, sell_price)
                sell_lines.append(f"SELL {ticker} ×{shares} @ ₹{sell_price}")
                if res["status"] in ("paper", "placed"):
                    set_holding(holdings, ticker, 0)

        # ── Immediately find and buy replacements ───────────────────────────
        # This is the key change — don't wait for month-end rebalance
        if not scored.empty:
            replacements = find_replacement(scored, current_tickers, exits)
            if replacements:
                port_value = get_portfolio_value(kite)
                new_lines  = execute_buys(
                    kite, replacements, holdings, latest_prices, port_value, strength
                )
                buy_lines.extend(new_lines)
                log.info(f"Immediate replacements bought: {replacements}")

    # ── REBALANCE DAY: full portfolio review ────────────────────────────────
    # Even if no exit triggered, check if any holding has been significantly
    # overtaken by a better ranked stock. Rotate if gap is large enough.
    if rebalance and not scored.empty:
        log.info("Full monthly rebalance — reviewing all positions...")

        top_n        = portfolio.index.tolist() if not portfolio.empty else []
        held_tickers = [t for t in holdings if get_shares(holdings, t) > 0]

        # Find holdings not in current top N that should be rotated out
        rotate_out = []
        for t in held_tickers:
            if t not in top_n and t not in exits:
                rank = scored.index.tolist().index(t) if t in scored.index else 999
                log.info(f"  Rotation candidate: {t} (rank {rank+1})")
                rotate_out.append(t)

        # Sell rotation exits
        for ticker in rotate_out:
            shares = get_shares(holdings, ticker)
            if shares > 0:
                price      = latest_prices.get(ticker, 0)
                sell_price = round(price * (1 - cfg.SELL_BUFFER), 1)
                res        = place_sell_order(kite, ticker, shares, sell_price)
                sell_lines.append(f"SELL {ticker} ×{shares} @ ₹{sell_price} [rotation]")
                if res["status"] in ("paper", "placed"):
                    set_holding(holdings, ticker, 0)

        # Buy full top N (including rotations)
        port_value    = get_portfolio_value(kite)
        new_lines     = execute_buys(
            kite, top_n, holdings, latest_prices, port_value, strength, stocks_to_hold
        )
        buy_lines.extend(new_lines)

    # ── Save and notify ─────────────────────────────────────────────────────
    save_holdings(holdings)

    held_now  = [t for t in holdings if get_shares(holdings, t) > 0]
    mode_icon = "📋" if PAPER_MODE else "✅"
    run_type  = "Full Rebalance" if rebalance else "Monitoring"

    msg = (
        f"{mode_icon} *Trading System — {run_type}*\n"
        f"Date   : {date.today()}\n"
        f"Mode   : {'Paper' if PAPER_MODE else '⚠️ LIVE'}\n"
        f"Regime : {regime}\n\n"
    )
    if sell_lines:
        msg += "*Sells:*\n" + "\n".join(sell_lines) + "\n\n"
    if buy_lines:
        msg += "*Buys:*\n" + "\n".join(buy_lines) + "\n\n"
    if not sell_lines and not buy_lines:
        msg += "_No changes today_\n\n"
    if held_now:
        msg += "*Portfolio & P&L:*\n"
        msg += build_pnl_summary(holdings, latest_prices)

    send_telegram(msg)

    log.info(f"Final portfolio: {[t for t in holdings if get_shares(holdings, t) > 0]}")
    log.info("=" * 55)


if __name__ == "__main__":
    run_execution()

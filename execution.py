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
import numpy as np

import config as cfg
from signals import run_signals, UNIVERSE, compute_scores, apply_liquidity_filter
from data_manager import load_for_signals, load_index_data

# ── Load secrets ────────────────────────────────────────────────────────────
load_dotenv()
KITE_API_KEY      = os.getenv("KITE_API_KEY",       "your_api_key_here")
KITE_API_SECRET   = os.getenv("KITE_API_SECRET",    "your_api_secret_here")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN",  "")
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID",   "")

# ── Master switch ───────────────────────────────────────────────────────────
PAPER_MODE = True   # flip to False only after full testing

# ── Settings ────────────────────────────────────────────────────────────────
LIMIT_BUFFER   = 0.005
CAPITAL        = float(os.getenv("TRADING_CAPITAL", "100000"))
HOLDINGS_FILE  = "current_holdings.json"
LOG_FILE       = "orders.log"

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# REBALANCE DAY CHECK
# ─────────────────────────────────────────────
def is_rebalance_day() -> bool:
    """
    True on the first trading day of each month (day 1-3, weekday).
    On rebalance days: full portfolio review, rotate underperformers.
    On monitoring days: exit monitoring + immediate replacement only.
    """
    today = date.today()
    return today.day <= 3 and today.weekday() < 5


# ─────────────────────────────────────────────
# LOAD LATEST PRICES FROM CACHE
# ─────────────────────────────────────────────
def load_latest_prices() -> dict:
    """Reads latest closing price for every stock from cache. No re-fetch."""
    prices = {}
    if os.path.exists(cfg.DATA_CACHE_FILE):
        try:
            close = pd.read_csv(cfg.DATA_CACHE_FILE, index_col=0, parse_dates=True)
            for ticker in close.columns:
                series = close[ticker].dropna()
                if len(series) > 0:
                    prices[ticker] = float(series.iloc[-1])
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
    When a stock exits mid-month, immediately find the best replacement
    from the current top 25 that isn't already held.

    Returns list of tickers to buy as replacements.
    """
    # Remaining holdings after exits
    remaining = [t for t in current_holdings if t not in exits]

    # Candidates: top 25 ranked stocks not already in portfolio
    top_25      = scored.head(cfg.EXIT_RANK_CUTOFF).index.tolist()
    candidates  = [t for t in top_25 if t not in remaining]

    # Apply sector cap — count sectors in remaining holdings
    sector_count = {}
    for t in remaining:
        sector = UNIVERSE.get(t, "Unknown")
        sector_count[sector] = sector_count.get(sector, 0) + 1

    # Pick replacements respecting sector cap
    replacements = []
    for t in candidates:
        sector = UNIVERSE.get(t, "Unknown")
        if sector_count.get(sector, 0) < cfg.MAX_PER_SECTOR:
            replacements.append(t)
            sector_count[sector] = sector_count.get(sector, 0) + 1
        if len(replacements) == len(exits):
            break

    if replacements:
        log.info(f"Replacements found: {replacements} for exits: {exits}")
    else:
        log.warning("No suitable replacements found — holding partial cash")

    return replacements


# ─────────────────────────────────────────────
# KITE CONNECT LOGIN
# ─────────────────────────────────────────────
def get_kite_client():
    if PAPER_MODE:
        log.info("PAPER MODE — Kite Connect not initialised")
        return None
    try:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=KITE_API_KEY)
        if KITE_ACCESS_TOKEN:
            kite.set_access_token(KITE_ACCESS_TOKEN)
            log.info("Kite Connect authenticated")
        else:
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
# PORTFOLIO STATE
# ─────────────────────────────────────────────
def load_holdings() -> dict:
    if os.path.exists(HOLDINGS_FILE):
        with open(HOLDINGS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_holdings(holdings: dict):
    with open(HOLDINGS_FILE, "w") as f:
        json.dump(holdings, f, indent=2)
    log.info(f"Holdings saved: { {t:s for t,s in holdings.items() if s>0} }")


def get_portfolio_value(kite, holdings: dict, prices: dict) -> float:
    if PAPER_MODE:
        stock_value = sum(
            shares * prices.get(ticker, 0)
            for ticker, shares in holdings.items() if shares > 0
        )
        return CAPITAL + stock_value
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
        from kiteconnect import KiteConnect
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
        from kiteconnect import KiteConnect
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
# TELEGRAM
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
# Shared logic used by both monitoring and rebalance paths
# ─────────────────────────────────────────────
def execute_buys(kite, tickers_to_buy: list, holdings: dict,
                 prices: dict, port_value: float) -> list:
    """
    Places buy orders for given tickers.
    Returns list of buy order descriptions for Telegram.
    """
    buy_lines  = []
    target     = port_value / cfg.TOP_N

    for ticker in tickers_to_buy:
        price = prices.get(ticker, 0)
        if price <= 0:
            log.warning(f"No price for {ticker} — skipping")
            continue

        current_shares = holdings.get(ticker, 0)
        current_value  = current_shares * price

        if current_value < target * 0.95:
            buy_value     = target - current_value
            shares_to_buy = int(buy_value / price)
            limit_price   = round(price * (1 + LIMIT_BUFFER), 1)

            if shares_to_buy > 0:
                res = place_buy_order(kite, ticker, shares_to_buy, limit_price)
                buy_lines.append(f"BUY {ticker} ×{shares_to_buy} @ ₹{limit_price}")
                if res["status"] in ("paper", "placed"):
                    holdings[ticker] = current_shares + shares_to_buy

    return buy_lines


# ─────────────────────────────────────────────
# MAIN EXECUTION FLOW
# ─────────────────────────────────────────────
def run_execution():
    log.info("=" * 55)
    log.info(f"EXECUTION RUN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"MODE          : {'PAPER' if PAPER_MODE else '⚠️  LIVE'}")

    rebalance = is_rebalance_day()
    log.info(f"TYPE          : {'FULL REBALANCE' if rebalance else 'MONITORING + EXIT CHECK'}")
    log.info("=" * 55)

    # ── Load state ─────────────────────────────────────────────────────────
    holdings        = load_holdings()
    current_tickers = [t for t, s in holdings.items() if s > 0]
    log.info(f"Current holdings: {current_tickers or 'None (empty)'}")

    # ── Load prices and data ────────────────────────────────────────────────
    latest_prices = load_latest_prices()
    if not latest_prices:
        log.error("No price data. Run data_manager.py first.")
        return

    # ── Connect to broker ───────────────────────────────────────────────────
    kite = get_kite_client()

    # ── Run signal engine ───────────────────────────────────────────────────
    log.info("Running signal engine...")
    signals   = run_signals(current_tickers)
    regime    = signals["regime"]
    portfolio = signals["portfolio"]
    exits     = signals["exits"]
    log.info(f"Regime: {regime}")

    # ── RISK-OFF: sell everything immediately ───────────────────────────────
    if regime == "RISK-OFF":
        log.info("RISK-OFF — liquidating all positions immediately")
        sell_lines = []
        for ticker in current_tickers:
            shares = holdings.get(ticker, 0)
            if shares > 0:
                price      = latest_prices.get(ticker, 0)
                sell_price = round(price * 0.999, 1)
                res        = place_sell_order(kite, ticker, shares, sell_price)
                sell_lines.append(f"SELL {ticker} ×{shares} @ ₹{sell_price}")
                if res["status"] in ("paper", "placed"):
                    holdings[ticker] = 0
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
            shares = holdings.get(ticker, 0)
            if shares > 0:
                price      = latest_prices.get(ticker, 0)
                sell_price = round(price * 0.999, 1)
                res        = place_sell_order(kite, ticker, shares, sell_price)
                sell_lines.append(f"SELL {ticker} ×{shares} @ ₹{sell_price}")
                if res["status"] in ("paper", "placed"):
                    holdings[ticker] = 0

        # ── Immediately find and buy replacements ───────────────────────────
        # This is the key change — don't wait for month-end rebalance
        if not scored.empty:
            replacements = find_replacement(scored, current_tickers, exits)
            if replacements:
                port_value = get_portfolio_value(kite, holdings, latest_prices)
                new_lines  = execute_buys(
                    kite, replacements, holdings, latest_prices, port_value
                )
                buy_lines.extend(new_lines)
                log.info(f"Immediate replacements bought: {replacements}")

    # ── REBALANCE DAY: full portfolio review ────────────────────────────────
    # Even if no exit triggered, check if any holding has been significantly
    # overtaken by a better ranked stock. Rotate if gap is large enough.
    if rebalance and not scored.empty:
        log.info("Full monthly rebalance — reviewing all positions...")

        top_7        = portfolio.index.tolist() if not portfolio.empty else []
        held_tickers = [t for t, s in holdings.items() if s > 0]

        # Find holdings not in current top 7 that should be rotated out
        rotate_out = []
        for t in held_tickers:
            if t not in top_7 and t not in exits:
                rank = scored.index.tolist().index(t) if t in scored.index else 999
                log.info(f"  Rotation candidate: {t} (rank {rank+1})")
                rotate_out.append(t)

        # Sell rotation exits
        for ticker in rotate_out:
            shares = holdings.get(ticker, 0)
            if shares > 0:
                price      = latest_prices.get(ticker, 0)
                sell_price = round(price * 0.999, 1)
                res        = place_sell_order(kite, ticker, shares, sell_price)
                sell_lines.append(f"SELL {ticker} ×{shares} @ ₹{sell_price} [rotation]")
                if res["status"] in ("paper", "placed"):
                    holdings[ticker] = 0

        # Buy full top 7 (including rotations)
        port_value    = get_portfolio_value(kite, holdings, latest_prices)
        new_lines     = execute_buys(
            kite, top_7, holdings, latest_prices, port_value
        )
        buy_lines.extend(new_lines)

    # ── Save and notify ─────────────────────────────────────────────────────
    save_holdings(holdings)

    held_now  = [f"{t} ×{s}" for t, s in holdings.items() if s > 0]
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
        msg += "*Portfolio:*\n" + "\n".join(held_now)

    send_telegram(msg)

    log.info(f"Final portfolio: {[t for t,s in holdings.items() if s>0]}")
    log.info("=" * 55)


if __name__ == "__main__":
    run_execution()

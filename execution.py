"""
execution.py
============
Execution engine — reads signals and places orders via Kite Connect.

PAPER_MODE = True  → logs orders, touches nothing real (default)
PAPER_MODE = False → places real orders (only after API approved + tested)

Flow:
  1. Load current holdings from disk
  2. Load latest prices from cache (no second fetch — avoids rate limiting)
  3. Run signal engine to get buy/sell list
  4. Place limit orders at last close + 0.5% buffer
  5. Send Telegram confirmation
  6. Log everything to orders.log

Run:
    python execution.py

Schedule this every weekday at 3:45 PM IST (after market close).
"""

import os
import json
import logging
from datetime import datetime, date
from dotenv import load_dotenv
import pandas as pd
import numpy as np

import config as cfg
from signals import run_signals, UNIVERSE

# ── Load secrets from .env ──────────────────────────────────────────────────
load_dotenv()
KITE_API_KEY      = os.getenv("KITE_API_KEY",       "your_api_key_here")
KITE_API_SECRET   = os.getenv("KITE_API_SECRET",    "your_api_secret_here")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN",  "")
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID",   "")

# ── Master switch ───────────────────────────────────────────────────────────
# True  = paper mode (safe, logs only, zero real orders)
# False = live mode  (real orders — only flip after full testing)
PAPER_MODE = True

# ── Settings ────────────────────────────────────────────────────────────────
LIMIT_BUFFER   = 0.005     # 0.5% above last close for buy limit orders
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
# LOAD LATEST PRICES FROM CACHE
# No second Yahoo Finance fetch — avoids rate limiting
# ─────────────────────────────────────────────
def load_latest_prices() -> dict:
    """
    Reads the most recent closing price for every stock
    directly from the cache file already on disk.

    This avoids fetching data twice in one run —
    which causes Yahoo Finance rate limiting errors.
    """
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
    else:
        log.warning(f"Cache file not found: {cfg.DATA_CACHE_FILE}")
        log.warning("Run backtester.py first to generate the cache")

    return prices


# ─────────────────────────────────────────────
# KITE CONNECT LOGIN
# ─────────────────────────────────────────────
def get_kite_client():
    """
    Returns authenticated Kite Connect client.
    Kite requires a fresh access token every day — expires at midnight.
    In paper mode returns None (no connection needed).
    """
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
            log.warning(f"No access token. Login here: {login_url}")
            print(f"\nOpen this URL to login:\n{login_url}\n")
            request_token = input("Paste the request_token from redirect URL: ").strip()
            data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
            kite.set_access_token(data["access_token"])
            # Save token for today
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
    """Loads current holdings from disk. Format: { TICKER: shares }"""
    if os.path.exists(HOLDINGS_FILE):
        with open(HOLDINGS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_holdings(holdings: dict):
    """Saves current holdings to disk after every run."""
    with open(HOLDINGS_FILE, "w") as f:
        json.dump(holdings, f, indent=2)
    log.info(f"Holdings saved: { {t:s for t,s in holdings.items() if s>0} }")


def get_portfolio_value(kite, holdings: dict, prices: dict) -> float:
    """
    Returns total portfolio value = cash + market value of stock holdings.
    In paper mode uses cached prices.
    In live mode reads from Kite margins API.
    """
    if PAPER_MODE:
        stock_value = sum(
            shares * prices.get(ticker, 0)
            for ticker, shares in holdings.items()
            if shares > 0
        )
        # In paper mode, CAPITAL is our starting cash
        # In a real run, cash decreases as we buy
        # We use CAPITAL as proxy here since we track holdings separately
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
        log.error(f"Could not fetch portfolio value from Kite: {e}")
        return CAPITAL


# ─────────────────────────────────────────────
# ORDER PLACEMENT
# ─────────────────────────────────────────────
def place_buy_order(kite, ticker: str, shares: int, limit_price: float) -> dict:
    """
    Places a CNC limit buy order.

    PAPER MODE  → logs without placing
    LIVE MODE   → sends to Kite Connect API

    limit_price = last close * 1.005
    The 0.5% buffer ensures the order fills at market open
    without chasing the price.
    """
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
            variety          = kite.VARIETY_REGULAR,
            exchange         = kite.EXCHANGE_NSE,
            tradingsymbol    = ticker,
            transaction_type = kite.TRANSACTION_TYPE_BUY,
            quantity         = shares,
            product          = kite.PRODUCT_CNC,
            order_type       = kite.ORDER_TYPE_LIMIT,
            price            = round(limit_price, 1),
            validity         = kite.VALIDITY_DAY,
        )
        log.info(f"[LIVE]  BUY  | {ticker:<15} | {shares:>5} shares @ ₹{limit_price:.1f} | ID: {order_id}")
        return {"status": "placed", "order_id": order_id}
    except Exception as e:
        log.error(f"BUY failed for {ticker}: {e}")
        return {"status": "failed", "error": str(e)}


def place_sell_order(kite, ticker: str, shares: int, limit_price: float) -> dict:
    """
    Places a CNC limit sell order.
    limit_price = last close * 0.999 — slight discount to ensure fill.
    """
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
            variety          = kite.VARIETY_REGULAR,
            exchange         = kite.EXCHANGE_NSE,
            tradingsymbol    = ticker,
            transaction_type = kite.TRANSACTION_TYPE_SELL,
            quantity         = shares,
            product          = kite.PRODUCT_CNC,
            order_type       = kite.ORDER_TYPE_LIMIT,
            price            = round(limit_price, 1),
            validity         = kite.VALIDITY_DAY,
        )
        log.info(f"[LIVE]  SELL | {ticker:<15} | {shares:>5} shares @ ₹{limit_price:.1f} | ID: {order_id}")
        return {"status": "placed", "order_id": order_id}
    except Exception as e:
        log.error(f"SELL failed for {ticker}: {e}")
        return {"status": "failed", "error": str(e)}


# ─────────────────────────────────────────────
# TELEGRAM NOTIFICATIONS
# ─────────────────────────────────────────────
def send_telegram(message: str):
    """Sends alert to your Telegram. Configure bot token in .env"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.info("Telegram not configured — skipping")
        return
    try:
        import requests
        url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"},
            timeout=10
        )
        if resp.status_code == 200:
            log.info("Telegram notification sent")
        else:
            log.warning(f"Telegram returned {resp.status_code}")
    except Exception as e:
        log.error(f"Telegram failed: {e}")


# ─────────────────────────────────────────────
# MAIN EXECUTION FLOW
# ─────────────────────────────────────────────
def run_execution():
    log.info("=" * 55)
    log.info(f"EXECUTION RUN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"MODE: {'PAPER TRADING' if PAPER_MODE else '⚠️  LIVE TRADING'}")
    log.info("=" * 55)

    # ── 1. Load state from disk ────────────────────────────────────────────
    holdings        = load_holdings()
    current_tickers = [t for t, s in holdings.items() if s > 0]
    log.info(f"Current holdings: {current_tickers or 'None (empty)'}")

    # ── 2. Load latest prices from cache (single source, no re-fetch) ──────
    latest_prices = load_latest_prices()
    if not latest_prices:
        log.error("No price data available. Run backtester.py first to build cache.")
        return

    # ── 3. Connect to broker ───────────────────────────────────────────────
    kite = get_kite_client()

    # ── 4. Run signal engine (single data fetch happens inside here) ────────
    log.info("Running signal engine...")
    signals = run_signals(current_tickers)

    regime    = signals["regime"]
    portfolio = signals["portfolio"]
    exits     = signals["exits"]
    log.info(f"Regime: {regime}")

    # ── 5. RISK-OFF → sell everything, go to cash ──────────────────────────
    if regime == "RISK-OFF":
        log.info("RISK-OFF — liquidating all positions")
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

        msg = (
            f"🔴 *RISK-OFF — Market Alert*\n"
            f"Date: {date.today()}\n"
            f"All positions liquidated — holding cash\n\n"
            + ("\n".join(sell_lines) if sell_lines else "No open positions to sell")
        )
        send_telegram(msg)
        log.info("Liquidation complete. Holding 100% cash.")
        return

    # ── 6. RISK-ON: process individual exits first ─────────────────────────
    if exits:
        log.info(f"Exit signals for: {exits}")
        for ticker in exits:
            shares = holdings.get(ticker, 0)
            if shares > 0:
                price      = latest_prices.get(ticker, 0)
                sell_price = round(price * 0.999, 1)
                res        = place_sell_order(kite, ticker, shares, sell_price)
                if res["status"] in ("paper", "placed"):
                    holdings[ticker] = 0

    # ── 7. Calculate portfolio value and target allocation ─────────────────
    port_value       = get_portfolio_value(kite, holdings, latest_prices)
    target_per_stock = port_value / cfg.TOP_N
    log.info(f"Portfolio value  : ₹{port_value:,.0f}")
    log.info(f"Target per stock : ₹{target_per_stock:,.0f}")

    # ── 8. Buy new positions or top up underweight ones ────────────────────
    buy_lines = []
    if not portfolio.empty:
        for ticker in portfolio.index.tolist():
            price = latest_prices.get(ticker, 0)
            if price <= 0:
                log.warning(f"No price for {ticker} — skipping")
                continue

            current_shares = holdings.get(ticker, 0)
            current_value  = current_shares * price

            if current_value < target_per_stock * 0.95:   # 5% tolerance band
                buy_value     = target_per_stock - current_value
                shares_to_buy = int(buy_value / price)
                limit_price   = round(price * (1 + LIMIT_BUFFER), 1)

                if shares_to_buy > 0:
                    res = place_buy_order(kite, ticker, shares_to_buy, limit_price)
                    buy_lines.append(f"BUY {ticker} ×{shares_to_buy} @ ₹{limit_price}")
                    if res["status"] in ("paper", "placed"):
                        holdings[ticker] = current_shares + shares_to_buy

    # ── 9. Save updated holdings ───────────────────────────────────────────
    save_holdings(holdings)

    # ── 10. Send Telegram summary ──────────────────────────────────────────
    held_now  = [f"{t} ×{s}" for t, s in holdings.items() if s > 0]
    mode_icon = "📋" if PAPER_MODE else "✅"

    msg = (
        f"{mode_icon} *Trading System — Daily Run*\n"
        f"Date   : {date.today()}\n"
        f"Mode   : {'Paper' if PAPER_MODE else '⚠️ LIVE'}\n"
        f"Regime : {regime}\n"
        f"Value  : ₹{port_value:,.0f}\n"
    )
    if buy_lines:
        msg += "\n*Orders placed:*\n" + "\n".join(buy_lines)
    if exits:
        msg += f"\n\n*Exits:* {', '.join(exits)}"
    if held_now:
        msg += "\n\n*Portfolio:*\n" + "\n".join(held_now)

    send_telegram(msg)

    log.info("Execution run complete.")
    log.info(f"Final portfolio: {[t for t,s in holdings.items() if s>0]}")
    log.info("=" * 55)


# ──────────────────────────────────────────────
if __name__ == "__main__":
    run_execution()
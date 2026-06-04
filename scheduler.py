"""
scheduler.py — the daily cron entry point that runs the whole pipeline
======================================================================
This is the single command the server's cron job runs each weekday afternoon
(~3:40 PM IST, after the market closes). It is a thin ORCHESTRATOR: it contains no
trading logic itself — it just calls the other modules in the right order and makes
sure one failing step doesn't silently kill the rest.

The order is deliberate:
  0. git pull      — self-update: pick up any code/config pushed to `main` since
                     yesterday, so the server always runs the latest version.
  1. kite_login    — refresh the daily-expiring Kite access token (live only).
  2. data_manager  — append today's prices to the local cache. MUST run before
                     step 3, because signals read ONLY from that cache.
  3. execution     — run signals + place (or paper-simulate) orders + Telegram summary.

Each step is wrapped in try/except so a non-fatal failure (e.g. Kite login while in
paper mode) is logged and the pipeline keeps going. If execution itself fails, we try
to send a Telegram alert so the problem gets noticed.

Run manually with:  python scheduler.py
"""
import os, logging, subprocess
from datetime import datetime
from dotenv import load_dotenv

# Load API keys and secrets from .env file into environment variables
load_dotenv()

import config as cfg

# Set up a file-based logger that appends to scheduler.log with timestamps
logging.basicConfig(filename="scheduler.log", level=logging.INFO,
    format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


def run_pipeline():
    """
    Main daily pipeline — runs every weekday at 3:40 PM IST via cron.
    Steps in order:
      1. Pull latest code from GitHub
      2. Log in to Kite Connect (gets a fresh access token)
      3. Update price/volume cache from Yahoo Finance
      4. Run signals and place orders via execution.py
    Halts gracefully on weekends or if TRADING_HALTED is set.
    """

    log.info(f"SCHEDULER START — {datetime.now()}")

    # Skip weekends — weekday() returns 0=Mon … 6=Sun, so >= 5 means Sat/Sun
    if datetime.today().weekday() >= 5:
        log.info("Weekend — skipping")
        return

    # Respect the emergency kill switch in config.py
    if cfg.TRADING_HALTED:
        log.info("TRADING_HALTED flag set — skipping")
        return

    # ── Step 0: Pull latest code from GitHub ─────────────────────────────
    # Ensures any config or strategy changes pushed to main are applied before running
    try:
        result = subprocess.run(
            ["git", "pull", "origin", "main"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))   # run in the project directory
        )
        log.info(f"git pull: {(result.stdout or result.stderr).strip()}")
    except Exception as e:
        log.warning(f"git pull failed: {e}")

    # ── Step 1: Kite Connect login ────────────────────────────────────────
    # Kite access tokens expire daily — get_access_token() logs in headlessly
    # using Playwright + TOTP and writes the new token to .env
    try:
        import asyncio
        from kite_login import get_access_token
        asyncio.run(get_access_token())
        log.info("Kite login successful")
    except Exception as e:
        log.error(f"Kite login failed: {e}")

    # ── Step 2: Update price/volume cache ────────────────────────────────
    # Fetches today's closing prices from Yahoo Finance and appends to the CSV cache.
    # This must complete before execution.py runs since signals read from the cache.
    try:
        from data_manager import update_cache
        update_cache()
        log.info("Cache updated")
    except Exception as e:
        log.error(f"Cache failed: {e}")

    # ── Step 3: Run signals and place orders ──────────────────────────────
    # Reads the updated cache, scores stocks, determines exits/buys, places limit orders.
    # If execution fails, sends a Telegram alert so the user knows to intervene manually.
    try:
        from execution import run_execution
        run_execution()
        log.info("Execution complete")
    except Exception as e:
        log.error(f"Execution failed: {e}")
        # Attempt to notify via Telegram so the failure doesn't go unnoticed
        try:
            import requests
            token = os.getenv("TELEGRAM_BOT_TOKEN", "")
            chat  = os.getenv("TELEGRAM_CHAT_ID", "")
            if token and chat:
                requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    data={"chat_id": chat, "text": f"Error: {e}"},
                    timeout=10
                )
        except:
            pass   # if Telegram also fails, nothing we can do — error is in the log

    log.info("SCHEDULER DONE")


if __name__ == "__main__":
    run_pipeline()

import os, logging, subprocess
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
import config as cfg

logging.basicConfig(filename="scheduler.log", level=logging.INFO,
    format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

def run_pipeline():
    log.info(f"SCHEDULER START — {datetime.now()}")
    if datetime.today().weekday() >= 5:
        log.info("Weekend — skipping")
        return

    if cfg.TRADING_HALTED:
        log.info("TRADING_HALTED flag set — skipping")
        return

    # Pull latest code from GitHub before running
    try:
        result = subprocess.run(
            ["git", "pull", "origin", "main"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        log.info(f"git pull: {(result.stdout or result.stderr).strip()}")
    except Exception as e:
        log.warning(f"git pull failed: {e}")

    # Step 1: Kite login
    try:
        import asyncio
        from kite_login import get_access_token
        asyncio.run(get_access_token())
        log.info("Kite login successful")
    except Exception as e:
        log.error(f"Kite login failed: {e}")

    # Step 2: Update cache
    try:
        from data_manager import update_cache
        update_cache()
        log.info("Cache updated")
    except Exception as e:
        log.error(f"Cache failed: {e}")

    # Step 3: Execute
    try:
        from execution import run_execution
        run_execution()
        log.info("Execution complete")
    except Exception as e:
        log.error(f"Execution failed: {e}")
        try:
            import requests
            token = os.getenv("TELEGRAM_BOT_TOKEN","")
            chat  = os.getenv("TELEGRAM_CHAT_ID","")
            if token and chat:
                requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                    data={"chat_id":chat,"text":f"Error: {e}"},timeout=10)
        except: pass

    log.info("SCHEDULER DONE")

if __name__ == "__main__":
    run_pipeline()

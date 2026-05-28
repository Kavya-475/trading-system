import os, logging
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(filename="scheduler.log", level=logging.INFO,
    format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

def run_pipeline():
    log.info(f"SCHEDULER START — {datetime.now()}")
    if datetime.today().weekday() >= 5:
        log.info("Weekend — skipping")
        return

    # Pull latest code from GitHub before running
    import subprocess
    pull = subprocess.run(["git", "pull", "origin", "main"],
                         capture_output=True, text=True)
    log.info(f"Git pull: {pull.stdout.strip() or pull.stderr.strip()}")

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

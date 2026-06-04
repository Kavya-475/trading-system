"""
kite_login.py — headless daily Zerodha Kite Connect login
=========================================================
WHY THIS EXISTS: Kite Connect access tokens expire every single day. Live trading
therefore needs a fresh token each morning before any orders can be placed. Doing
that by hand (open browser → type password → type 2FA code → click Authorise) every
day is impractical for an automated system, so this script performs the entire login
flow automatically in an invisible (headless) browser.

HOW IT WORKS (the Kite OAuth-style handshake):
  1. Ask KiteConnect for the login URL for our app (api_key).
  2. Drive a headless Chromium browser through the login form (user id + password).
  3. Generate the current 6-digit TOTP 2-factor code from the shared secret and enter it.
  4. Click "Authorise" — Kite then redirects to our app with ?request_token=... in the URL.
  5. Intercept that request_token, then exchange it (+ api_secret) for an access_token.
  6. Write the access_token into .env, where execution.py reads it to authenticate.

WHERE IT RUNS: scheduler.py calls get_access_token() once at the start of the daily
cron run, before data_manager and execution. In PAPER_MODE this is not needed (no real
broker calls), so a failure here is logged but doesn't stop paper trading.

SECRETS: every credential below comes from .env (never hard-coded). KITE_TOTP_SECRET is
the seed that lets pyotp reproduce the same time-based codes as your authenticator app.
"""
import os, asyncio, pyotp
from playwright.async_api import async_playwright
from kiteconnect import KiteConnect
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

# Load Kite credentials from .env into environment variables
load_dotenv()

# Read all required credentials from environment
API_KEY     = os.getenv("KITE_API_KEY")
API_SECRET  = os.getenv("KITE_API_SECRET")
USER_ID     = os.getenv("KITE_USER_ID")
PASSWORD    = os.getenv("KITE_PASSWORD")
TOTP_SECRET = os.getenv("KITE_TOTP_SECRET")   # seed used to generate 6-digit TOTP codes


async def get_access_token():
    """
    Performs a fully headless Kite Connect login using Playwright.
    Flow:
      1. Open the Kite Connect login URL in a hidden Chromium browser
      2. Fill in user ID and password
      3. Generate and fill the TOTP code (2-factor auth)
      4. Click Authorise to grant the app permission
      5. Intercept the request_token from the redirect URL
      6. Exchange request_token + API secret for an access_token
      7. Write the new access_token into .env for execution.py to use

    Kite access tokens expire every day, so this runs before each execution.
    """

    # Create a KiteConnect client and get the login URL to open in the browser
    kite      = KiteConnect(api_key=API_KEY)
    login_url = kite.login_url()
    print(f"Login URL: {login_url}")

    async with async_playwright() as p:
        # Launch a headless (invisible) Chromium browser
        browser = await p.chromium.launch(headless=True)
        page    = await browser.new_page()

        request_token = None   # will be captured from the redirect URL

        # Log every page navigation so we can debug if the flow breaks
        page.on("framenavigated", lambda f: print(f"NAV: {f.url[:100]}"))

        # ── Intercept request_token from redirect URL ─────────────────────
        # After the user authorises the app, Kite redirects to a URL containing
        # ?request_token=<token>. We capture this before the page loads.
        def handle_request(request):
            nonlocal request_token
            url = request.url
            if "request_token" in url:
                params = parse_qs(urlparse(url).query)
                token  = params.get("request_token", [None])[0]
                if token:
                    request_token = token
                    print(f"Got request_token: {token[:10]}...")

        page.on("request", handle_request)

        # ── Navigate to Kite login page ───────────────────────────────────
        print("Navigating to Kite Connect login URL...")
        try:
            await page.goto(login_url, timeout=15000)
        except Exception as e:
            # Playwright raises on redirect — that's fine, we catch it and continue
            print(f"Navigation note: {e}")

        # Wait for the page to finish rendering after navigation
        await page.wait_for_timeout(2000)
        print(f"Current URL: {page.url}")

        # ── Fill user ID and password, then submit ────────────────────────
        try:
            await page.fill('input[type="text"]', USER_ID)
            await page.fill('input[type="password"]', PASSWORD)
            await page.click('button[type="submit"]')
            print("Submitted login form")
            await page.wait_for_timeout(3000)   # wait for TOTP screen to appear
            print(f"After login URL: {page.url}")
        except Exception as e:
            print(f"Login form error: {e}")

        # ── Generate and fill TOTP code ───────────────────────────────────
        # pyotp generates the current 6-digit TOTP from the secret seed.
        # Codes are time-based and change every 30 seconds — must be entered quickly.
        try:
            totp_code = pyotp.TOTP(TOTP_SECRET).now()
            print(f"TOTP code: {totp_code}")
            await page.fill('input[type="number"]', totp_code)
            await page.wait_for_timeout(3000)   # wait for Authorise screen
            print(f"After TOTP URL: {page.url}")
        except Exception as e:
            print(f"TOTP error: {e}")

        # ── Click the Authorise / Allow button ────────────────────────────
        # This grants our app permission and triggers the redirect with request_token
        try:
            btn = await page.wait_for_selector(
                'button:has-text("Authorise"), button:has-text("Allow"), button:has-text("Authorize")',
                timeout=8000
            )
            print(f"Found button: {await btn.inner_text()}")
            await btn.click()
            await page.wait_for_timeout(5000)   # wait for the redirect to complete
            print(f"After authorize URL: {page.url}")
        except Exception as e:
            print(f"Authorize button error: {e}")
            # Print page HTML snippet to help debug what went wrong
            content = await page.content()
            print(f"Page content (500 chars): {content[:500]}")

        await browser.close()

    # ── Validate that we captured the request_token ───────────────────────
    if not request_token:
        raise Exception("Could not get request_token — check debug output above")

    # ── Exchange request_token for access_token ───────────────────────────
    # This is a one-time exchange — the request_token is single-use.
    # The resulting access_token is valid for the rest of the trading day.
    session_data = kite.generate_session(request_token, api_secret=API_SECRET)
    access_token = session_data["access_token"]

    # ── Write the new access_token to .env ───────────────────────────────
    # execution.py reads KITE_ACCESS_TOKEN from .env to authenticate API calls.
    # We update the existing line in place to avoid duplicates.
    lines = []
    if os.path.exists(".env"):
        with open(".env") as f:
            lines = f.readlines()
    updated = False
    for i, line in enumerate(lines):
        if line.startswith("KITE_ACCESS_TOKEN="):
            lines[i] = f"KITE_ACCESS_TOKEN={access_token}\n"
            updated = True
            break
    if not updated:
        lines.append(f"KITE_ACCESS_TOKEN={access_token}\n")
    with open(".env", "w") as f:
        f.writelines(lines)

    print(f"Login successful. Token: {access_token[:10]}...")
    return access_token


if __name__ == "__main__":
    # Run login manually to test or refresh the token outside of the daily cron
    asyncio.run(get_access_token())
    print("Token saved to .env")

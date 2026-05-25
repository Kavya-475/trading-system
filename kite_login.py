import os, asyncio, pyotp
from playwright.async_api import async_playwright
from kiteconnect import KiteConnect
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
load_dotenv()

API_KEY     = os.getenv("KITE_API_KEY")
API_SECRET  = os.getenv("KITE_API_SECRET")
USER_ID     = os.getenv("KITE_USER_ID")
PASSWORD    = os.getenv("KITE_PASSWORD")
TOTP_SECRET = os.getenv("KITE_TOTP_SECRET")

async def get_access_token():
    kite      = KiteConnect(api_key=API_KEY)
    login_url = kite.login_url()
    print(f"Login URL: {login_url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page    = await browser.new_page()

        request_token = None

        # Log all navigations
        page.on("framenavigated", lambda f: print(f"NAV: {f.url[:100]}"))

        # Capture request_token from any URL
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

        # Go directly to Kite Connect login URL
        print("Navigating to Kite Connect login URL...")
        try:
            await page.goto(login_url, timeout=15000)
        except Exception as e:
            print(f"Navigation note: {e}")

        await page.wait_for_timeout(2000)
        print(f"Current URL: {page.url}")

        # Fill login form
        try:
            await page.fill('input[type="text"]', USER_ID)
            await page.fill('input[type="password"]', PASSWORD)
            await page.click('button[type="submit"]')
            print("Submitted login form")
            await page.wait_for_timeout(3000)
            print(f"After login URL: {page.url}")
        except Exception as e:
            print(f"Login form error: {e}")

        # Fill TOTP
        try:
            totp_code = pyotp.TOTP(TOTP_SECRET).now()
            print(f"TOTP code: {totp_code}")
            await page.fill('input[type="number"]', totp_code)
            await page.wait_for_timeout(3000)
            print(f"After TOTP URL: {page.url}")
        except Exception as e:
            print(f"TOTP error: {e}")

        # Click Authorise button
        try:
            btn = await page.wait_for_selector(
                'button:has-text("Authorise"), button:has-text("Allow"), button:has-text("Authorize")',
                timeout=8000
            )
            print(f"Found button: {await btn.inner_text()}")
            await btn.click()
            await page.wait_for_timeout(5000)
            print(f"After authorize URL: {page.url}")
        except Exception as e:
            print(f"Authorize button error: {e}")
            # Print page content to debug
            content = await page.content()
            print(f"Page content (500 chars): {content[:500]}")

        await browser.close()

    if not request_token:
        raise Exception("Could not get request_token — check debug output above")

    session_data = kite.generate_session(request_token, api_secret=API_SECRET)
    access_token = session_data["access_token"]

    lines = open(".env").readlines() if os.path.exists(".env") else []
    updated = False
    for i, line in enumerate(lines):
        if line.startswith("KITE_ACCESS_TOKEN="):
            lines[i] = f"KITE_ACCESS_TOKEN={access_token}\n"
            updated = True
            break
    if not updated:
        lines.append(f"KITE_ACCESS_TOKEN={access_token}\n")
    open(".env","w").writelines(lines)

    print(f"Login successful. Token: {access_token[:10]}...")
    return access_token

if __name__ == "__main__":
    asyncio.run(get_access_token())
    print("Token saved to .env")

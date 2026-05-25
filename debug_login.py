import pyotp, requests
from dotenv import load_dotenv
import os
load_dotenv()

session = requests.Session()

# Login
resp = session.post("https://kite.zerodha.com/api/login",
    data={"user_id":os.getenv("KITE_USER_ID"),"password":os.getenv("KITE_PASSWORD")})
data = resp.json()
request_id = data["data"]["request_id"]

# TOTP
totp_code = pyotp.TOTP(os.getenv("KITE_TOTP_SECRET")).now()
session.post("https://kite.zerodha.com/api/twofa",
    data={"user_id":os.getenv("KITE_USER_ID"),"request_id":request_id,
          "twofa_value":totp_code,"twofa_type":"totp"})

# Get authorize page
from kiteconnect import KiteConnect
kite = KiteConnect(api_key=os.getenv("KITE_API_KEY"))
login_url = kite.login_url()

resp2 = session.get(login_url, allow_redirects=True)
print("Status:", resp2.status_code)
print("Final URL:", resp2.url)
print("Content (first 2000 chars):")
print(resp2.text[:2000])

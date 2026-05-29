import requests
from http.cookiejar import MozillaCookieJar
from pathlib import Path

COOKIES = Path.home() / "Desktop" / "cookies.txt"

session = requests.Session()
cj = MozillaCookieJar(str(COOKIES))
cj.load(ignore_discard=True, ignore_expires=True)
session.cookies = cj
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
})

# Test one folder
url = "https://media.talkbank.org/dementia/English/Pitt/Control/cookie/"

print(f"Fetching: {url}\n")
r = session.get(url, timeout=60)

print(f"Status Code: {r.status_code}")
print(f"Content-Type: {r.headers.get('Content-Type')}")
print(f"\n{'='*50}")
print("PAGE CONTENT (first 3000 chars):")
print("="*50)
print(r.text[:3000])
print("\n" + "="*50)
print("ALL LINKS FOUND:")
print("="*50)

from bs4 import BeautifulSoup
soup = BeautifulSoup(r.text, "lxml")
for a in soup.find_all("a"):
    href = a.get("href", "")
    print(f"  {href}")
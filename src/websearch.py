"""
Optional web fallback: called only when a user explicitly turns it on AND
the document has already been asked and found not to contain the answer.
See README.md, "Web fallback", for the full picture - this file is
deliberately the ONLY place in the project that makes a network call to
anything other than the local Ollama server.

TWO backends, chosen via config.WEB_SEARCH_BACKEND, not one hardcoded
choice - this file has already been rewritten once to swap backends
(BUILD_LOG.md, "Day 10": Google -> DuckDuckGo, over an invalid API key) and
once back (Google, over DuckDuckGo's live rate-limiting turning out to be a
real everyday problem, not just a theoretical one). Rather than rewrite a
third time if the trade-off flips again, both implementations live here
side by side and config.py picks one:

  "google"     - Google Custom Search API. Real, versioned results; needs a
                 Google Cloud API key + Search Engine ID (see README.md,
                 "Web fallback", for setup) and is capped at 100 free
                 searches/day. The credential is easy to misconfigure -
                 wrong project, API not enabled, key restricted to the
                 wrong APIs - and Google's error message for most of those
                 is the same unhelpful "API key not valid."
  "duckduckgo" - No signup, no key, works immediately. Scrapes DuckDuckGo's
                 HTML results page rather than calling a sanctioned API, and
                 - confirmed live, not assumed - rate-limits fast: a second
                 search within about a minute of the first can come back
                 blocked.

Run it directly to sanity-check whichever backend is active, without going
through either UI:
    python src\\websearch.py "what is the capital of Japan"
"""

import sys
import time
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

from config import (GOOGLE_SEARCH_API_KEY, GOOGLE_SEARCH_ENGINE_ID,
                    WEB_FALLBACK_MAX_RESULTS, WEB_SEARCH_BACKEND)

GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"
DUCKDUCKGO_SEARCH_URL = "https://html.duckduckgo.com/html/"
# A generic request without a real User-Agent is disproportionately likely
# to be blocked by DuckDuckGo's HTML endpoint - this is a known, common
# workaround for scraping it, not a way of hiding what the request is.
DUCKDUCKGO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                 "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}


def is_configured():
    """Whether search_web() can be called right now with the ACTIVE
    backend - lets the UI grey out the toggle with an explanation instead
    of offering it and then failing when clicked.
    """
    if WEB_SEARCH_BACKEND == "google":
        return bool(GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID)
    return True  # duckduckgo needs no credentials


def search_web(query, max_results=WEB_FALLBACK_MAX_RESULTS):
    """Return up to max_results results as a list of
    {"title", "snippet", "link"} dicts, most relevant first, from whichever
    backend config.WEB_SEARCH_BACKEND selects.
    """
    if WEB_SEARCH_BACKEND == "google":
        return _search_google(query, max_results)
    if WEB_SEARCH_BACKEND == "duckduckgo":
        return _search_duckduckgo(query, max_results)
    raise RuntimeError(
        f"Unknown WEB_SEARCH_BACKEND {WEB_SEARCH_BACKEND!r} in config.py - "
        "expected 'google' or 'duckduckgo'."
    )


# --------------------------------------------------------------------------
# Google Custom Search API
# --------------------------------------------------------------------------

def _search_google(query, max_results):
    if not is_configured():
        raise RuntimeError(
            "Web fallback (Google) is not configured. Set "
            "GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID as "
            "environment variables, then restart the app. See README.md, "
            "'Web fallback', for where to get both - and for exactly what "
            "'API key not valid' means if you already have and it still "
            "fails."
        )

    try:
        response = requests.get(
            GOOGLE_SEARCH_URL,
            params={
                "key": GOOGLE_SEARCH_API_KEY,
                "cx": GOOGLE_SEARCH_ENGINE_ID,
                "q": query,
                "num": max(1, min(max_results, 10)),  # API caps at 10
            },
            timeout=15,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not reach Google's Custom Search API - check your "
            "internet connection."
        ) from None

    if response.status_code != 200:
        # Google's error body names the actual problem (bad key, search
        # engine not configured to search the web, quota exceeded, ...) -
        # surfacing it beats a bare status code on a feature this fiddly to
        # set up correctly on the first try.
        raise RuntimeError(
            f"Google Custom Search returned {response.status_code}\n"
            f"Server said: {response.text.strip()}"
        )

    data = response.json()
    items = data.get("items", [])
    return [
        {
            "title": item.get("title", ""),
            "snippet": item.get("snippet", "").replace("\n", " "),
            "link": item.get("link", ""),
        }
        for item in items[:max_results]
    ]


# --------------------------------------------------------------------------
# DuckDuckGo (no credentials, scraped HTML - see this file's docstring)
# --------------------------------------------------------------------------

def _unwrap_ddg_redirect(href):
    """DuckDuckGo's HTML results wrap every outbound link in its own
    redirector: //duckduckgo.com/l/?uddg=<url-encoded target>&rut=... .
    Unwrapped here so the UI can link straight to the real page instead of
    DuckDuckGo's tracking redirect.
    """
    if href.startswith("//"):
        href = "https:" + href
    query = parse_qs(urlparse(href).query)
    if "uddg" in query:
        return unquote(query["uddg"][0])
    return href


_ddg_blocked_since = None
# Originally guessed at "a minute or two" (Day 10, from one observed 202).
# Measured live instead of re-guessed (Day 11): still blocked over 50
# minutes after the first 202 in the same session, across several spaced-out
# attempts, not just rapid-fire ones. The real duration is unknown - this is
# a lower bound, not a documented policy Google or DuckDuckGo publishes - so
# the cooldown here is deliberately long and the message says "at least",
# not a specific promise.
DDG_COOLDOWN_SECONDS = 600


def _search_duckduckgo(query, max_results):
    global _ddg_blocked_since

    if _ddg_blocked_since is not None:
        elapsed = time.time() - _ddg_blocked_since
        if elapsed < DDG_COOLDOWN_SECONDS:
            # Fail immediately, no network call at all. Two reasons: (1) no
            # point making the user wait out a request that will almost
            # certainly come back blocked again, and (2) retrying while
            # already blocked is plausibly what EXTENDS a block in the first
            # place - every attempt during the cooldown, from any source,
            # keeps resetting this timer.
            remaining = int(DDG_COOLDOWN_SECONDS - elapsed)
            raise RuntimeError(
                f"Still cooling down from DuckDuckGo's rate limit (blocked "
                f"{int(elapsed)}s ago, waiting at least {remaining}s more) - "
                "not retrying yet, since another blocked attempt likely "
                "extends it further rather than clearing it. This block has "
                "measurably lasted over 50 minutes in testing, not the "
                "\"a minute or two\" first assumed - if you need the web "
                "fallback working NOW, switch WEB_SEARCH_BACKEND to "
                "\"google\" in config.py instead (needs a working API key)."
            )

    try:
        response = requests.post(
            DUCKDUCKGO_SEARCH_URL, data={"q": query},
            headers=DUCKDUCKGO_HEADERS, timeout=15,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not reach DuckDuckGo - check your internet connection."
        ) from None

    if response.status_code == 202:
        # Observed live (BUILD_LOG.md, Days 10-11): DuckDuckGo's anomaly/
        # rate-limit detection, not a genuine error - and measurably NOT a
        # brief one. First seen after a second request within about a
        # minute; still returning 202 over 50 minutes later across several
        # spaced-out real attempts in the same session. Recorded here so
        # the next attempt fails fast via the cooldown check above instead
        # of wasting a request that will almost certainly also be blocked.
        _ddg_blocked_since = time.time()
        raise RuntimeError(
            "DuckDuckGo is rate-limiting this machine (HTTP 202 - its bot/"
            "anomaly detection, not a real error). This has measurably "
            "lasted over 50 minutes in testing - not the \"a minute or two\" "
            "first assumed - so don't expect it to clear quickly. If you "
            "need the web fallback working now, switch WEB_SEARCH_BACKEND "
            "to \"google\" in config.py instead (needs a working API key)."
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"DuckDuckGo returned {response.status_code}\n"
            f"Body (truncated): {response.text[:300].strip()}"
        )

    soup = BeautifulSoup(response.text, "html.parser")
    rows = soup.select(".result")

    if not rows:
        # Empty is a valid outcome for a genuinely obscure query, but it is
        # ALSO what a blocked or restructured page looks like - distinguish
        # them for whoever reads this later by checking for the page's own
        # "no results" marker rather than assuming.
        if soup.select_one(".no-results"):
            return []
        raise RuntimeError(
            "DuckDuckGo returned a page this scraper doesn't recognise - "
            "their result markup may have changed. Try "
            "'python src\\websearch.py \"test\"' directly to confirm."
        )

    results = []
    for row in rows[:max_results]:
        title_el = row.select_one(".result__a")
        if not title_el:
            continue
        snippet_el = row.select_one(".result__snippet")
        results.append({
            "title": title_el.get_text(strip=True),
            "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
            "link": _unwrap_ddg_redirect(title_el.get("href", "")),
        })
    return results


if __name__ == "__main__":
    # Found live (see BUILD_LOG.md, Day 10): Windows' default console
    # codepage (cp1252) can't encode every character a real web result may
    # contain - crashing this demo script's own print() calls with
    # UnicodeEncodeError even when search_web() itself succeeded. Not a
    # problem for either UI - the browser and Flask's response encoding are
    # UTF-8 throughout - only for this direct-CLI demo path.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print('Usage: python src\\websearch.py "your query"')
        sys.exit(1)

    print(f"Backend: {WEB_SEARCH_BACKEND}")
    for i, result in enumerate(search_web(" ".join(sys.argv[1:])), start=1):
        print(f"\n[{i}] {result['title']}")
        print(f"    {result['link']}")
        print(f"    {result['snippet']}")

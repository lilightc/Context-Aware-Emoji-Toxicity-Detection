"""Urban Dictionary scraper for emoji slang definitions."""

from __future__ import annotations

import time
import urllib.parse

import requests
from bs4 import BeautifulSoup

_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120.0"}


def scrape_urban_definition(term: str) -> str | None:
    """Scrape the top definition from Urban Dictionary for a given term."""
    safe_term = urllib.parse.quote(term)
    url = f"https://www.urbandictionary.com/define.php?term={safe_term}"

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=5)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        def_block = soup.find("div", class_="definition")
        if def_block:
            return def_block.get_text().replace("[", "").replace("]", "").strip()
    except Exception:
        return None
    return None


def load_urban_dict_entries(symbols: list[str], delay: float = 1.5) -> list[dict]:
    """Scrape Urban Dictionary for a list of emoji symbols.

    Args:
        symbols: List of emoji characters to look up.
        delay: Seconds between requests to avoid rate limiting.

    Returns list of dicts with keys: symbol, slang_definition, source.
    """
    entries = []
    for i, sym in enumerate(symbols):
        definition = scrape_urban_definition(sym)
        if definition:
            entries.append({
                "symbol": sym,
                "slang_definition": definition,
                "source": "UrbanDictionary",
            })
            print(f"  [{i + 1}/{len(symbols)}] {sym} — found definition")
        else:
            print(f"  [{i + 1}/{len(symbols)}] {sym} — no definition")
        if i < len(symbols) - 1:
            time.sleep(delay)
    return entries

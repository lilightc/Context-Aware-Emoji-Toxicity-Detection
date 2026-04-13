"""Unicode CLDR emoji descriptions — canonical short names for ~3,700 emoji."""

from __future__ import annotations

import emoji

from emoji_toxicity.utils import cldr_name


def load_cldr_entries() -> list[dict]:
    """Get official Unicode CLDR short names for all emoji.

    Returns list of dicts with keys: symbol, literal_meaning, source.
    """
    entries = []
    for sym in emoji.EMOJI_DATA:
        name = cldr_name(sym)
        if name:
            entries.append({"symbol": sym, "literal_meaning": name, "source": "CLDR"})
    return entries

"""Load Silent Signals dataset — emoji dog-whistle meanings from social media."""

from __future__ import annotations

from datasets import load_dataset

from emoji_toxicity.config import settings
from emoji_toxicity.utils import extract_emojis


def load_silent_signals_entries() -> list[dict]:
    """Load emoji dog-whistle data from the Silent Signals dataset.

    Returns list of dicts with keys: symbol, slang_meaning, context, source.
    """
    try:
        ds = load_dataset("MMHS/SilentSignals", token=settings.hf_token)
    except Exception:
        # Fallback: try alternative dataset name patterns
        try:
            ds = load_dataset("cambridge-alt/silent-signals", token=settings.hf_token)
        except Exception:
            print("  [WARN] Silent Signals dataset not available, skipping.")
            return []

    entries = []
    seen = set()

    for split in ds:
        for row in ds[split]:
            text = row.get("text", "")
            label = row.get("label", "")

            for em in extract_emojis(text):
                if em not in seen:
                    seen.add(em)
                    entries.append({
                        "symbol": em,
                        "slang_meaning": label if isinstance(label, str) else "",
                        "context": text[:200],
                        "source": "SilentSignals",
                    })

    return entries

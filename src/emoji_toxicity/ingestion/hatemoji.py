"""Load HatemojiBuild dataset (Kirk et al., 2022) — 5,912 examples of emoji-based hate speech."""

from __future__ import annotations

from datasets import load_dataset

from emoji_toxicity.config import settings
from emoji_toxicity.utils import extract_emojis


def load_hatemoji_entries() -> list[dict]:
    """Extract emoji entries from HatemojiBuild with usage examples and frequency data.

    Returns list of dicts with keys: symbol, examples, toxic_count, total_count, source.
    """
    ds = load_dataset("HannahRoseKirk/HatemojiBuild", token=settings.hf_token)
    train = ds["train"]

    emoji_data: dict[str, dict] = {}

    for row in train:
        text = row["text"]
        label = row["label_gold"]  # 1 = hate, 0 = not hate

        for em in extract_emojis(text):
            if em not in emoji_data:
                emoji_data[em] = {
                    "symbol": em,
                    "examples": [],
                    "toxic_count": 0,
                    "total_count": 0,
                    "source": "HatemojiBuild",
                }
            entry = emoji_data[em]
            entry["total_count"] += 1
            if label == 1:
                entry["toxic_count"] += 1
                if len(entry["examples"]) < 5:
                    entry["examples"].append(text)

    return list(emoji_data.values())

"""Incremental index updates — upsert new KB entries without full rebuild.

Appends validated entries to the KB JSONL and upserts their vectors into
Pinecone. Existing entries are never deleted (append-only). Each entry is
version-tagged with a timestamp for freshness tracking.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from emoji_toxicity.config import settings, KB_PATH, PROCESSED_DIR
from emoji_toxicity.utils import make_vec_id
from emoji_toxicity.vectorstore.index import _format_embedding_text, _make_metadata


def upsert_entries(
    entries: list[dict],
    tag: str = "",
) -> int:
    """Append entries to the KB and upsert vectors into Pinecone.

    Args:
        entries: List of validated KB entries (from validation.validate_candidates).
        tag: Optional batch tag for provenance (e.g. "reddit_2026w16").

    Returns:
        Number of vectors upserted.
    """
    if not entries:
        return 0

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    # Normalize entries to KB schema and add version metadata
    normalized = []
    for entry in entries:
        norm = {
            "symbol": entry["symbol"],
            "literal_meaning": entry.get("literal_meaning", ""),
            "slang_meaning": entry.get("slang_meaning", ""),
            "slang_definition": entry.get("slang_definition", ""),
            "risk_category": entry.get("risk_category", ""),
            "toxic_signals": entry.get("toxic_signals", []),
            "benign_signals": entry.get("benign_signals", []),
            "sources": [f"dynamic:{tag or 'update'}"],
            "added_at": timestamp,
            "batch_tag": tag,
        }
        normalized.append(norm)

    # Append to KB JSONL (append-only)
    with open(KB_PATH, "a") as f:
        for entry in normalized:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Appended {len(normalized)} entries to {KB_PATH}")

    # Embed and upsert into Pinecone
    print("Embedding new entries...")
    model = SentenceTransformer(settings.embedding_model)
    texts = [_format_embedding_text(e) for e in normalized]
    vectors = model.encode(texts, batch_size=32)

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    batch = []
    for entry, vector in zip(normalized, vectors):
        batch.append({
            "id": make_vec_id(entry["symbol"]),
            "values": vector.tolist(),
            "metadata": _make_metadata(entry),
        })

    if batch:
        index.upsert(vectors=batch)

    print(f"Upserted {len(batch)} vectors into '{settings.pinecone_index_name}'")
    return len(batch)

"""Build/rebuild the Pinecone index from the knowledge base."""

from __future__ import annotations

import json
import time

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from emoji_toxicity.config import settings, KB_PATH
from emoji_toxicity.utils import make_vec_id


def _format_embedding_text(entry: dict) -> str:
    """Format a KB entry into text for embedding."""
    parts = [f"Symbol: {entry['symbol']}"]

    if entry.get("literal_meaning"):
        parts.append(f"Literal meaning: {entry['literal_meaning']}")
    if entry.get("slang_meaning"):
        parts.append(f"Slang meaning: {entry['slang_meaning']}")
    if entry.get("slang_definition"):
        parts.append(f"Slang definition: {entry['slang_definition']}")
    if entry.get("toxic_signals"):
        parts.append(f"Toxic context triggers: {', '.join(entry['toxic_signals'])}")
    if entry.get("benign_signals"):
        parts.append(f"Benign signals: {', '.join(entry['benign_signals'])}")

    return "\n".join(parts)


def _make_metadata(entry: dict) -> dict:
    """Build Pinecone metadata from a KB entry (string values only for Pinecone)."""
    return {
        "symbol": entry["symbol"],
        "literal_meaning": entry.get("literal_meaning", ""),
        "slang_meaning": entry.get("slang_meaning", ""),
        "risk_category": entry.get("risk_category", "Unknown"),
        "toxic_signals": ", ".join(entry.get("toxic_signals", [])),
        "benign_signals": ", ".join(entry.get("benign_signals", [])),
        "text": _format_embedding_text(entry),
    }


def build_index(kb_path: str | None = None, batch_size: int = 50) -> int:
    """Build/rebuild the Pinecone index from the knowledge base.

    Args:
        kb_path: Path to knowledge_base.jsonl. Uses default if None.
        batch_size: Upload batch size.

    Returns:
        Number of vectors uploaded.
    """
    kb_file = kb_path or str(KB_PATH)

    # Load KB
    entries = []
    with open(kb_file) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"Loaded {len(entries)} KB entries")

    # Filter: only entries with meaningful content (not just CLDR stubs)
    entries = [
        e for e in entries
        if e.get("risk_category") and e["risk_category"] != "Safe"
        or e.get("slang_meaning")
        or e.get("slang_definition")
    ]
    print(f"Indexing {len(entries)} non-trivial entries")

    # Connect to Pinecone
    pc = Pinecone(api_key=settings.pinecone_api_key)

    # Create index if it doesn't exist
    existing = pc.list_indexes().names()
    if settings.pinecone_index_name not in existing:
        print(f"Creating index '{settings.pinecone_index_name}'...")
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        time.sleep(10)  # Wait for index to be ready

    index = pc.Index(settings.pinecone_index_name)

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(settings.embedding_model)

    # Batch-encode all texts at once (much faster than per-doc encode).
    print("Embedding entries...")
    texts = [_format_embedding_text(e) for e in entries]
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True)

    # Upload in batches
    vectors_uploaded = 0
    pending = []
    for entry, vector in zip(entries, vectors):
        pending.append({
            "id": make_vec_id(entry["symbol"]),
            "values": vector.tolist(),
            "metadata": _make_metadata(entry),
        })
        if len(pending) >= batch_size:
            index.upsert(vectors=pending)
            vectors_uploaded += len(pending)
            pending = []
            time.sleep(0.3)

    if pending:
        index.upsert(vectors=pending)
        vectors_uploaded += len(pending)

    print(f"Uploaded {vectors_uploaded} vectors to '{settings.pinecone_index_name}'")
    return vectors_uploaded

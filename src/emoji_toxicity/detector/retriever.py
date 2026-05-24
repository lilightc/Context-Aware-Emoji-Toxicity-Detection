"""Hybrid retrieval: exact-symbol lookup (primary) + dense semantic search (fallback).

Diagnostic showed dense top-1 recall on bare-emoji queries was only ~77%. The
text encoder confuses near-identical emoji (e.g. retrieving 🐵 when the query
is 🐒). Exact-symbol lookup via deterministic vector IDs is 100% accurate when
the symbol is indexed, so we use it as the primary key and fall back to dense
for free-form context queries and unindexed emoji.

Each result carries an origin tag ("exact" or "dense") and a score:
  - "exact"  → score 1.0 (deterministic key match)
  - "dense"  → cosine similarity from Pinecone
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document

from emoji_toxicity.utils import cldr_name, extract_emojis


@dataclass
class RetrievalResult:
    documents: list[Document]
    scores: list[float]
    origins: list[str]  # "exact" or "dense", aligned with documents
    query: str
    expanded_query: str
    emoji_found: list[str]


def _expand_query(text: str) -> str:
    """Expand emoji in text to their CLDR names for better dense retrieval.

    Example: "She is a 🌽 star" → "She is a 🌽 (ear of corn) star"
    """
    out = []
    for ch in text:
        name = cldr_name(ch)
        out.append(f"{ch} ({name})" if name else ch)
    return "".join(out)


@lru_cache(maxsize=1)
def _get_vectorstore():
    from emoji_toxicity.vectorstore.store import get_vectorstore
    return get_vectorstore()


def _dense_search(query: str, k: int) -> tuple[list[Document], list[float]]:
    """Cosine-similarity search via PineconeVectorStore."""
    results = _get_vectorstore().similarity_search_with_score(query, k=k)
    return [d for d, _ in results], [float(s) for _, s in results]


def retrieve(text: str, k: int = 3) -> RetrievalResult:
    """Hybrid retrieval. Exact-symbol matches fill the result first; dense
    semantic search backfills remaining slots with non-duplicate symbols.

    No emoji in text → falls back to dense-only on the original message.
    """
    from emoji_toxicity.vectorstore.store import fetch_by_symbols

    emojis = extract_emojis(text)
    expanded = _expand_query(text)

    docs: list[Document] = []
    scores: list[float] = []
    origins: list[str] = []
    seen_symbols: set[str] = set()

    if emojis:
        unique_emojis = list(dict.fromkeys(emojis))  # preserve order, dedupe
        for doc in fetch_by_symbols(unique_emojis):
            sym = doc.metadata.get("symbol")
            if sym in seen_symbols:
                continue
            docs.append(doc)
            scores.append(1.0)
            origins.append("exact")
            seen_symbols.add(sym)
            if len(docs) >= k:
                break

    if len(docs) < k:
        dense_docs, dense_scores = _dense_search(expanded, k=k)
        for doc, score in zip(dense_docs, dense_scores):
            sym = doc.metadata.get("symbol")
            if sym in seen_symbols:
                continue
            docs.append(doc)
            scores.append(score)
            origins.append("dense")
            seen_symbols.add(sym)
            if len(docs) >= k:
                break

    return RetrievalResult(
        documents=docs,
        scores=scores,
        origins=origins,
        query=text,
        expanded_query=expanded,
        emoji_found=emojis,
    )

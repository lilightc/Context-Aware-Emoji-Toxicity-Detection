"""Query expansion + multi-result retrieval with similarity scores."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.documents import Document

from emoji_toxicity.utils import cldr_name, extract_emojis


@dataclass
class RetrievalResult:
    """Result from retrieval with similarity scores."""
    documents: list[Document]
    scores: list[float]  # cosine similarity per document (1.0 = perfect match)
    query: str
    expanded_query: str
    emoji_found: list[str]


def _expand_query(text: str) -> str:
    """Expand emoji in text to their CLDR names for better retrieval.

    Example: "She is a 🌽 star" → "She is a 🌽 (ear of corn) star"
    """
    out = []
    for ch in text:
        name = cldr_name(ch)
        if name:
            out.append(f"{ch} ({name})")
        else:
            out.append(ch)
    return "".join(out)


@lru_cache(maxsize=1)
def _get_vectorstore():
    from emoji_toxicity.vectorstore.store import get_vectorstore
    return get_vectorstore()


def retrieve(text: str, k: int = 3) -> RetrievalResult:
    """Retrieve relevant KB entries with similarity scores.

    Uses similarity_search_with_score to return cosine similarity alongside
    each document, so the LLM can discount low-relevance retrievals.
    """
    emojis = extract_emojis(text)
    expanded = _expand_query(text)

    vs = _get_vectorstore()
    results = vs.similarity_search_with_score(expanded, k=k)

    docs = [doc for doc, _score in results]
    # Pinecone returns cosine similarity (higher = better, max 1.0)
    scores = [float(score) for _doc, score in results]

    return RetrievalResult(
        documents=docs,
        scores=scores,
        query=text,
        expanded_query=expanded,
        emoji_found=emojis,
    )

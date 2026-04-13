"""Query expansion + multi-result retrieval with confidence scoring."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document

from emoji_toxicity.utils import cldr_name, extract_emojis


@dataclass
class RetrievalResult:
    """Result from retrieval with confidence metadata."""
    documents: list[Document]
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
def _get_retriever(k: int):
    # Lazy import keeps Pinecone out of import-time path (tests don't need it).
    from emoji_toxicity.vectorstore.store import get_vectorstore

    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def retrieve(text: str, k: int = 3) -> RetrievalResult:
    """Retrieve relevant KB entries for a text message.

    Performs query expansion (emoji → CLDR name) then similarity search.
    """
    emojis = extract_emojis(text)
    expanded = _expand_query(text)
    docs = _get_retriever(k).invoke(expanded)

    return RetrievalResult(
        documents=docs,
        query=text,
        expanded_query=expanded,
        emoji_found=emojis,
    )

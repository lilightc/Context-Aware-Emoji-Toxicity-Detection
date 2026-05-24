"""Pinecone vector store via LangChain, plus low-level fetch-by-ID helper."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from emoji_toxicity.config import settings
from emoji_toxicity.utils import make_vec_id
from emoji_toxicity.vectorstore.embedder import get_embeddings


def get_vectorstore() -> PineconeVectorStore:
    """Return a PineconeVectorStore connected to the emoji-toxicity index."""
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=get_embeddings(),
        pinecone_api_key=settings.pinecone_api_key,
    )


@lru_cache(maxsize=1)
def _raw_index():
    """Pinecone index handle for low-level operations (fetch by ID)."""
    return Pinecone(api_key=settings.pinecone_api_key).Index(settings.pinecone_index_name)


def _vector_to_document(vec) -> Document:
    """Convert a Pinecone Vector record to a LangChain Document.

    Matches PineconeVectorStore's similarity_search convention: the metadata
    "text" field becomes page_content; remaining keys stay in metadata.
    """
    meta = dict(getattr(vec, "metadata", None) or {})
    page_content = meta.pop("text", "")
    return Document(page_content=page_content, metadata=meta)


def fetch_by_symbols(symbols: list[str]) -> list[Document]:
    """Exact-key lookup: fetch KB entries by emoji symbol via deterministic vector IDs.

    Returns Documents in the same order as ``symbols``. Symbols not in the index
    are silently skipped (the caller can detect this via len mismatch).
    """
    if not symbols:
        return []
    id_to_symbol = {make_vec_id(s): s for s in symbols}
    res = _raw_index().fetch(ids=list(id_to_symbol.keys()))
    vectors = getattr(res, "vectors", None) or {}

    docs_by_symbol: dict[str, Document] = {}
    for vec_id, vec in vectors.items():
        symbol = id_to_symbol.get(vec_id)
        if symbol is not None:
            docs_by_symbol[symbol] = _vector_to_document(vec)

    return [docs_by_symbol[s] for s in symbols if s in docs_by_symbol]

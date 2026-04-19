"""Sentence-transformer embedding wrapper using LangChain's HuggingFaceEmbeddings."""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from emoji_toxicity.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance (all-MiniLM-L6-v2, 384-dim)."""
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)

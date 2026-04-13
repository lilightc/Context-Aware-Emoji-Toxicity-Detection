"""Sentence-transformer embedding wrapper using LangChain's HuggingFaceEmbeddings."""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from emoji_toxicity.config import settings


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a configured HuggingFaceEmbeddings instance (all-MiniLM-L6-v2, 384-dim)."""
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)

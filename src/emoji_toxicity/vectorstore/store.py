"""Pinecone vector store via LangChain."""

from __future__ import annotations

from langchain_pinecone import PineconeVectorStore

from emoji_toxicity.config import settings
from emoji_toxicity.vectorstore.embedder import get_embeddings


def get_vectorstore() -> PineconeVectorStore:
    """Return a PineconeVectorStore connected to the emoji-toxicity index."""
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=get_embeddings(),
        pinecone_api_key=settings.pinecone_api_key,
    )

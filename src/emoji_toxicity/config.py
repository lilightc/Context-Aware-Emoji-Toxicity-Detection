"""Centralized configuration via pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

KB_PATH = PROCESSED_DIR / "knowledge_base.jsonl"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API keys
    openai_api_key: str = ""
    hf_token: str = ""
    pinecone_api_key: str = ""

    # Pinecone
    pinecone_index_name: str = "emoji-toxicity"

    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # LLM
    llm_model: str = "gpt-5"
    llm_temperature: float = 0.0

    # Retrieval
    retrieval_k: int = 3

    # Confidence thresholds (two-threshold gating)
    toxic_threshold: float = 0.7
    safe_threshold: float = 0.3

    # Detector mode: "agent" (tool-calling, default) or "workflow" (fixed retrieve→classify)
    detector_mode: str = "workflow"
    agent_max_iterations: int = 4

    # Dynamic KB
    kb_update_min_sources: int = 3  # minimum independent sources to confirm new slang
    kb_update_confidence: float = 0.7  # LLM extraction confidence threshold
    reddit_client_id: str = ""
    reddit_client_secret: str = ""


settings = Settings()

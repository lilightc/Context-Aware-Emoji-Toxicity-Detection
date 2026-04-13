"""CLI: Embed and index the knowledge base into Pinecone.

Usage:
    python -m scripts.build_index [--kb-path PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from emoji_toxicity.vectorstore.index import build_index


def main():
    parser = argparse.ArgumentParser(description="Build Pinecone index from knowledge base")
    parser.add_argument("--kb-path", type=str, default=None, help="Path to knowledge_base.jsonl")
    args = parser.parse_args()

    build_index(kb_path=args.kb_path)


if __name__ == "__main__":
    main()

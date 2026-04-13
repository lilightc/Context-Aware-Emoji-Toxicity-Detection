# Context-Aware Emoji Toxicity Detector

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/framework-LangChain-green.svg)](https://langchain.com)
[![Gradio](https://img.shields.io/badge/demo-Gradio-orange.svg)](https://gradio.app)
[![Pinecone](https://img.shields.io/badge/vectordb-Pinecone-purple.svg)](https://pinecone.io)

A RAG-based system that detects toxic emoji usage by understanding **context**. The same emoji can be harmful or harmless depending on how it's used — this system retrieves emoji slang knowledge and applies an LLM judge to classify usage accurately.

## Key Insight

| Message | Context | Verdict |
|---------|---------|---------|
| She is a 🌽 star | "Check out my exclusive content!" | **TOXIC** (sexual) |
| I love 🌽 on the cob! | "What's your favorite summer food?" | **SAFE** |
| 🐵 go back to the jungle | — | **TOXIC** (hate speech) |
| 🐵 see no evil 🙈 | — | **SAFE** |

## Architecture

```
Message + Context
       │
       ▼
┌──────────────┐    ┌───────────────────┐
│ Query        │───▶│ Pinecone Vector   │
│ Expansion    │    │ Store (384-dim)   │
│ (emoji→CLDR) │    │ 500+ emoji entries│
└──────────────┘    └────────┬──────────┘
                             │ top-k results
                             ▼
                    ┌───────────────────┐
                    │ LLM Judge         │
                    │ (GPT-5)           │
                    │ Structured output │
                    └────────┬──────────┘
                             │
                             ▼
                    ┌───────────────────┐
                    │ Confidence Gating │
                    │ TOXIC/SAFE/       │
                    │ UNCERTAIN         │
                    └───────────────────┘
```

**Pipeline:**
1. **Query Expansion** — emoji characters expanded to CLDR names for better retrieval
2. **RAG Retrieval** — Pinecone similarity search against knowledge base of emoji slang meanings
3. **LLM Judge** — GPT-5 with structured output: verdict, confidence, per-emoji analysis
4. **Confidence Gating** — two-threshold system outputs TOXIC / SAFE / UNCERTAIN

## Knowledge Base

Multi-source knowledge base with provenance tracking:

| Source | Entries | Description |
|--------|---------|-------------|
| Seed KB (GPT-enriched) | 509 | Urban Dictionary scraping + GPT structured enrichment |
| Unicode CLDR | ~3,700 | Official emoji short names and descriptions |
| HatemojiBuild | ~150 unique emoji | Academic dataset of emoji-based hate speech (Kirk et al., 2022) |
| Silent Signals | varies | Emoji dog-whistle meanings (optional) |

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys:
# - OPENAI_API_KEY
# - PINECONE_API_KEY
# - HF_TOKEN (for gated datasets)
```

### 3. Build Knowledge Base

```bash
python -m scripts.build_knowledge_base --skip-gpt  # fast: seed + CLDR only
python -m scripts.build_knowledge_base              # full: includes GPT enrichment
```

### 4. Build Vector Index

```bash
python -m scripts.build_index
```

### 5. Launch Demo

```bash
python app.py
# Opens at http://localhost:7860
```

### 6. Run Evaluation

```bash
python -m scripts.evaluate --skip-hatemoji           # adversarial set only
python -m scripts.evaluate --max-hatemoji 200         # with HatemojiCheck subset
```

## Project Structure

```
├── src/emoji_toxicity/
│   ├── config.py                # pydantic-settings configuration
│   ├── ingestion/               # Data collection from 4+ sources
│   │   ├── hatemoji.py          # HatemojiBuild dataset loader
│   │   ├── silent_signals.py    # Silent Signals dog whistles
│   │   ├── cldr.py              # Unicode CLDR emoji descriptions
│   │   ├── urban_dict.py        # Urban Dictionary scraper
│   │   └── build_kb.py          # Orchestrator: merge → unified KB
│   ├── vectorstore/
│   │   ├── embedder.py          # HuggingFaceEmbeddings wrapper
│   │   ├── store.py             # Pinecone vector store
│   │   └── index.py             # Build/rebuild Pinecone index
│   ├── detector/
│   │   ├── retriever.py         # Query expansion + retrieval
│   │   ├── classifier.py        # LLM judge with structured output
│   │   └── pipeline.py          # End-to-end ToxicityDetector
│   └── evaluation/
│       ├── datasets.py          # HatemojiCheck + adversarial test set
│       ├── baselines.py         # Keyword + raw LLM baselines
│       ├── metrics.py           # F1, AUROC, precision, recall
│       └── run_eval.py          # Benchmark runner
├── scripts/                     # CLI entry points
├── app.py                       # Gradio demo (HF Spaces compatible)
├── data/
│   ├── processed/knowledge_base.jsonl
│   └── results/eval_results.json
└── notebooks/exploration.ipynb
```

## Evaluation

Three-way comparison:

| Method | Description |
|--------|-------------|
| **Keyword Baseline** | Blacklist of ~20 known toxic emoji — no context awareness |
| **Raw LLM** | GPT-5 without RAG — no emoji slang knowledge |
| **RAG Pipeline** (ours) | Full system: retrieval + LLM judge + confidence gating |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | LangChain |
| Vector DB | Pinecone (free tier) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| LLM | OpenAI GPT-5 |
| Demo | Gradio on HuggingFace Spaces |

## References

- Kirk, H. R., et al. (2022). *HatemojiBuild: A hate speech dataset annotated with emoji-based hate speech.* ACL.
- Unicode CLDR — Common Locale Data Repository for emoji descriptions.

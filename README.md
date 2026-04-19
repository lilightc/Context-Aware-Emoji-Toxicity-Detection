# Context-Aware Emoji Toxicity Detector

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/framework-LangChain-green.svg)](https://langchain.com)
[![Gradio](https://img.shields.io/badge/demo-Gradio-orange.svg)](https://gradio.app)
[![Pinecone](https://img.shields.io/badge/vectordb-Pinecone-purple.svg)](https://pinecone.io)

A RAG-based system that detects toxic emoji usage by understanding **context**. The same emoji can be harmful or harmless depending on how it's used — this system retrieves emoji slang knowledge from a curated knowledge base and applies an LLM judge to classify coded vs. literal usage. The KB is designed to be **dynamic** — updated continuously from social media, community reports, and emerging slang.

## The Problem

Standard content moderation fails on emoji-encoded toxicity because emoji are **context-dependent signals**: 🌽 means corn in a recipe thread but "porn" in a sexual context; ❄️ means snow in a weather post but cocaine in a party invitation. Keyword blacklists produce false positives on literal usage, and even GPT-5 without retrieval misses 47% of coded emoji in our benchmark (0% detection on drug-coded emoji, 35% on sexual-coded emoji).

## Key Results

### Context-Flip Benchmark (n=155)

| Method | Accuracy | Macro F1 | AUROC | FP | FN | Latency |
|--------|----------|----------|-------|----|----|---------|
| Keyword Baseline | 0.516 | 0.485 | 0.515 | 56 | 19 | <1s |
| Raw LLM (GPT-5, no RAG) | 0.761 | 0.748 | 0.912 | 0 | 37 | 14.3s |
| **RAG Workflow (ours)** | **0.948** [0.92, 0.98] | **0.948** | **0.997** | **0** | **8** | **11.0s** |

### Real-World Benchmark (n=55, threads + sarcasm + plausible deniability)

| Method | Accuracy | Macro F1 | FP | FN |
|--------|----------|----------|----|----|
| Raw LLM (GPT-5) | 0.709 | 0.709 | 1 | 15 |
| **RAG Workflow** | **0.909** [0.84, 0.98] | **0.906** | **0** | **5** |

### Cost-Efficiency: Weaker Model + RAG

| Method | Accuracy | Latency | ~Cost |
|--------|----------|---------|-------|
| Raw GPT-5 (no RAG) | 0.761 | 14.3s | $$$ |
| **gpt-4o-mini + RAG** | **0.858** | **2.4s** | **$** |

A ~20× cheaper model with RAG **outperforms** the flagship model without it, at 6× lower latency.

### Where RAG Helps Most

| Category | Raw LLM | RAG Workflow | Lift |
|----------|---------|--------------|------|
| Drug-coded emoji (❄️🍄🌿💊) | **0%** | **79%** | +79pp |
| Sexual-coded emoji (🌽🍑🍆🍒🥜) | **35%** | **91%** | +56pp |
| Novel slang NOT in KB (🧊🫠🛣️) | 62% | 62% | 0pp |

The last row is the honest limitation: **RAG can't help on emoji not in the KB.** This is why the knowledge base must be dynamic.

## Architecture

### Detection Pipeline (Workflow Mode)

```
Message + Context
       │
       ▼
┌──────────────────┐    ┌─────────────────────┐
│ Query Expansion  │───▶│ Pinecone Vector     │
│ (emoji → CLDR    │    │ Store (384-dim)     │
│  short name)     │    │ 534 entries         │
└──────────────────┘    └──────────┬──────────┘
                                   │ top-k=3 + similarity scores
                                   ▼
                        ┌─────────────────────┐
                        │ LLM Judge (GPT-5)   │
                        │ → toxicity_score    │
                        │ → reasoning         │
                        │ → emoji_analysis[]  │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │ Score Gate           │
                        │ ≥0.7 → TOXIC        │
                        │ ≤0.3 → SAFE         │
                        │ else → UNCERTAIN    │
                        └─────────────────────┘
```

### Dynamic KB Update Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ Collect           → Extract         → Validate → Index  │
│ Reddit, Emojipedia   LLM identifies   ≥3 sources  Upsert│
│ User submissions     coded emoji      conf ≥ 0.7  into  │
│                      from raw text    no conflicts Pinecone│
│                                                         │
│ Monitor: re-run benchmarks → alert on regression        │
└─────────────────────────────────────────────────────────┘
```

The KB is **append-only** (old slang doesn't stop being slang) and **version-tagged** (freshness tracking). User-flagged misclassifications feed back into the update cycle via the Gradio demo's "Flag Misclassification" tab.

### Three Operating Modes

| Mode | Description | Best for |
|------|-------------|----------|
| `workflow` | Always retrieve → classify → gate | Production (best accuracy + speed) |
| `adaptive` | Heuristic gate decides whether to retrieve | Cost optimization on mostly-safe traffic |
| `agent` | GPT-5 tool-calling decides what to retrieve | Research / demo (shows tool-calling capability) |

## Quick Start

### 1. Install & Configure

```bash
pip install -e .
cp .env.example .env
# Edit .env: OPENAI_API_KEY, PINECONE_API_KEY, HF_TOKEN (optional)
```

### 2. Build Knowledge Base & Index

```bash
python -m scripts.build_knowledge_base --skip-gpt   # seed + CLDR + HatemojiBuild
python -m scripts.build_index                        # 534 entries → Pinecone
```

### 3. Launch Demo

```bash
python app.py    # http://localhost:7860 — two tabs: Detect + Flag Misclassification
```

### 4. Run Evaluation

```bash
python -m scripts.evaluate --bench                   # context-flip benchmark
python -m scripts.evaluate --bench --compare-modes   # all modes side-by-side
LLM_MODEL=gpt-4o-mini python -m scripts.evaluate --bench  # cost-efficiency test
```

### 5. Update Knowledge Base (Dynamic)

```bash
python -m scripts.update_kb                          # full: collect → extract → validate → index
python -m scripts.update_kb --dry-run                # validate only, don't index
python -m scripts.update_kb --health                 # print KB health report
python -m scripts.update_kb --save-baseline          # save eval results as regression baseline
```

### 6. Calibrate Thresholds

```bash
python -m scripts.calibrate_thresholds               # sweep on val split, report on test
```

## Project Structure

```
├── src/emoji_toxicity/
│   ├── config.py                     # pydantic-settings: env vars, thresholds, mode
│   ├── utils.py                      # shared: extract_emojis, cldr_name, verdict_from_score, format_retrieved_docs
│   │
│   ├── ingestion/                    # Data collection + dynamic KB
│   │   ├── build_kb.py               # Static KB: merge seed + CLDR + HatemojiBuild + GPT enrichment
│   │   ├── collectors.py             # Dynamic: Reddit, Emojipedia, user-submission scrapers
│   │   ├── slang_extractor.py        # LLM-assisted slang extraction from raw posts
│   │   ├── validation.py             # Multi-source confirmation + conflict checking
│   │   ├── monitor.py                # KB health report + accuracy regression detection
│   │   ├── cldr.py                   # Unicode CLDR emoji descriptions
│   │   ├── hatemoji.py               # HatemojiBuild dataset loader
│   │   ├── silent_signals.py         # Silent Signals dog-whistle data
│   │   └── urban_dict.py             # Urban Dictionary scraper
│   │
│   ├── vectorstore/
│   │   ├── embedder.py               # HuggingFaceEmbeddings wrapper
│   │   ├── store.py                  # Pinecone vector store connection
│   │   ├── index.py                  # Full rebuild: batch-encode KB → Pinecone (with safe anchors)
│   │   └── incremental.py            # Dynamic: append-only upsert of new entries
│   │
│   ├── detector/
│   │   ├── retriever.py              # Query expansion + similarity scores
│   │   ├── classifier.py             # LLM judge: toxicity_score with relevance-aware prompt
│   │   ├── retrieval_gate.py         # Heuristic: does this message need KB lookup?
│   │   ├── tools.py                  # Agent tools: lookup, search, cldr, submit_verdict
│   │   ├── agent.py                  # Tool-calling agent loop
│   │   └── pipeline.py               # ToxicityDetector: workflow | adaptive | agent
│   │
│   └── evaluation/
│       ├── context_flip_bench.py     # 155-sample context-sensitivity benchmark
│       ├── realworld_bench.py        # 55-sample real-world scenarios
│       ├── datasets.py               # HatemojiBuild loader + stratified sampling
│       ├── baselines.py              # Keyword blacklist + raw LLM baselines
│       ├── metrics.py                # Bootstrap 95% CIs on all metrics
│       └── run_eval.py               # Multi-method runner + per-sample trace logging
│
├── scripts/
│   ├── build_knowledge_base.py       # CLI: build static KB
│   ├── build_index.py                # CLI: full Pinecone index rebuild
│   ├── evaluate.py                   # CLI: run eval benchmarks
│   ├── calibrate_thresholds.py       # CLI: threshold sweep on val split
│   └── update_kb.py                  # CLI: dynamic KB update pipeline
│
├── app.py                            # Gradio: Detect tab + Flag Misclassification tab
├── docs/PROJECT_REPORT.md            # Full analysis, results, and architecture discussion
├── notebooks/exploration.ipynb       # KB stats + eval visualization
├── data/
│   ├── knowledge_base_enriched.json  # seed KB (509 entries)
│   ├── processed/knowledge_base.jsonl
│   ├── user_submissions.jsonl        # flagged misclassifications from Gradio
│   └── results/
│       ├── eval_results.json
│       ├── eval_traces.jsonl
│       ├── eval_baseline.json        # accuracy baseline for regression detection
│       └── kb_validation*.jsonl      # validation audit logs
└── tests/test_core.py                # 19 unit tests, no API keys needed
```

## Tech Stack

| Component | Technology | Reason |
|-----------|------------|--------|
| Framework | LangChain 0.3+ | Retriever abstractions, structured output, tool-calling |
| Vector DB | Pinecone Starter (free 2GB) | Cloud-hosted, incremental upsert, HF Spaces compatible |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384-dim) | Free, local, fast |
| LLM | OpenAI GPT-5 (configurable) | Structured output; also tested with gpt-4o-mini |
| Demo | Gradio | Detect + Flag tabs, HF Spaces deployment |
| Config | pydantic-settings | Single `.env` source of truth |
| Build | hatchling | Modern Python packaging |

## References

- Kirk, H. R., et al. (2022). *HatemojiBuild: A hate speech dataset annotated with emoji-based hate speech.* ACL.
- Unicode CLDR — Common Locale Data Repository for emoji descriptions.

See [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) for the full analysis, experimental results, cost-efficiency findings, evaluation bias discussion, and dynamic KB design.

# Context-Aware Emoji Toxicity Detector — Project Report

## 1. Goal & Motivation

Modern content-moderation systems struggle with **emoji-encoded toxicity**. The same emoji can be entirely benign or deeply harmful depending on context:

| Message | Context | Intent |
|---|---|---|
| "I love 🌽 on the cob!" | Food discussion | benign |
| "She is a 🌽 star" | "Check out my exclusive content!" | sexual (corn = "porn") |
| "🐵 see no evil 🙈" | — | idiom |
| "🐵 go back to the jungle" | — | racial hate speech |

Keyword blacklists fail (false positives on food posts), and raw LLMs miss the slang because emoji dog-whistles aren't well-represented in pre-training. The project's thesis: **retrieve emoji-slang knowledge first, then let an LLM judge the message in context.**

## 2. Method — RAG Pipeline

```
Message + Context
       │
       ▼
┌──────────────────┐    ┌────────────────────┐
│ Query Expansion  │───▶│ Pinecone Vector    │
│ (emoji → CLDR    │    │ Store (384-dim)    │
│  name)           │    │ ~500 emoji entries │
└──────────────────┘    └─────────┬──────────┘
                                  │ top-k results
                                  ▼
                         ┌────────────────────┐
                         │ LLM Judge          │
                         │ (GPT-5)            │
                         │ Structured output  │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │ Confidence Gating  │
                         │ TOXIC / SAFE /     │
                         │ UNCERTAIN          │
                         └────────────────────┘
```

**Stages:**
1. **Query expansion** — each emoji is annotated with its CLDR name (`🌽 (ear of corn)`) so the embedding model has lexical anchors, not just opaque codepoints.
2. **Retrieval** — sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) embeds the expanded query against a Pinecone index of emoji slang knowledge; top-k=3 nearest entries are returned with metadata.
3. **LLM judge** — GPT-5 receives the message, surrounding context, and retrieved knowledge, and emits a structured Pydantic object: `verdict`, `confidence`, `reasoning`, `risk_category`, plus a per-emoji breakdown. GPT-5 was chosen over the older GPT-4o family because the contextual judgment task — distinguishing literal vs. coded use of the *same* emoji — rewards stronger reasoning, and the LLM is the only non-cacheable cost in the pipeline.
4. **Confidence gating** — instead of forcing a binary, a two-threshold gate produces:
   - `confidence ≥ 0.7` → trust the LLM verdict (typically TOXIC)
   - `confidence ≤ 0.3` → SAFE
   - in-between → UNCERTAIN (the model abstains rather than guess)

## 3. Knowledge Base — Multi-Source

The KB is built by `src/emoji_toxicity/ingestion/build_kb.py`, which merges entries from up to five sources with **provenance tracking** (each entry records the `sources` it came from):

| Source | Approx. size | Role |
|---|---|---|
| **Seed KB** (`data/knowledge_base_enriched.json`) | 509 entries | Pre-existing GPT-enriched prototype data — `literal_meaning`, `slang_meaning`, `risk_category`, `toxic_signals`, `benign_signals` |
| **Unicode CLDR** (via the `emoji` package) | ~3,700 emoji | Canonical short names — every emoji gets at least a literal description |
| **HatemojiBuild** (Kirk et al., 2022, HF gated dataset) | ~5,900 examples | Per-emoji `toxic_count` / `total_count` frequency stats and example messages |
| **Silent Signals** (optional) | varies | Dog-whistle meanings from social-media data |
| **Urban Dictionary** (optional, scraped) | up to 100 | Informal slang definitions for emoji missing structured signals |

After merging, entries that still lack a `risk_category` or `toxic_signals` are passed through GPT-5 for structured enrichment. Output: `data/processed/knowledge_base.jsonl`.

## 4. Project Structure

```
emoji-toxicity-detector/
├── README.md
├── pyproject.toml                    # hatchling, pinned deps
├── .env.example                      # API keys + index name
│
├── src/emoji_toxicity/
│   ├── config.py                     # pydantic-settings: env, paths, thresholds
│   ├── utils.py                      # shared: extract_emojis, cldr_name, make_vec_id
│   │
│   ├── ingestion/                    # 5 data sources → unified KB
│   │   ├── cldr.py
│   │   ├── hatemoji.py
│   │   ├── silent_signals.py
│   │   ├── urban_dict.py
│   │   └── build_kb.py               # orchestrator + GPT enrichment
│   │
│   ├── vectorstore/
│   │   ├── embedder.py               # HuggingFaceEmbeddings wrapper
│   │   ├── store.py                  # Pinecone vector store
│   │   └── index.py                  # Build/rebuild index (batched encoding)
│   │
│   ├── detector/
│   │   ├── retriever.py              # query expansion + cached retriever
│   │   ├── classifier.py             # cached LLM chain, structured output
│   │   └── pipeline.py               # ToxicityDetector → DetectionResult
│   │
│   └── evaluation/
│       ├── datasets.py               # HatemojiCheck loader + 16 hand-crafted adversarial pairs
│       ├── baselines.py              # keyword blacklist + raw LLM (no RAG)
│       ├── metrics.py                # accuracy, precision, recall, macro F1, AUROC
│       └── run_eval.py               # unified runner, no duplicate inference
│
├── scripts/
│   ├── build_knowledge_base.py       # CLI: ingest sources → KB
│   ├── build_index.py                # CLI: KB → Pinecone
│   └── evaluate.py                   # CLI: run eval suite
│
├── app.py                            # Gradio demo (HF Spaces compatible)
├── notebooks/exploration.ipynb       # KB stats + eval plots
├── data/
│   ├── knowledge_base_enriched.json  # seed KB (509 entries)
│   ├── processed/knowledge_base.jsonl
│   └── results/eval_results.json
└── tests/test_core.py                # 6 unit tests, no API needed
```

## 5. Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Framework | **LangChain** (current `langchain` packages, not legacy `langchain_classic`) | Built-in retriever abstractions, structured output with Pydantic |
| Vector DB | **Pinecone** Starter (free 2 GB) | Cloud-hosted, no infra; HF Spaces friendly |
| Embeddings | **sentence-transformers / all-MiniLM-L6-v2** (384-dim) | Free, runs locally, fast |
| LLM | **GPT-5** | Strongest current OpenAI model with structured output — context-flip judgment is the hard part of the task and benefits from stronger reasoning |
| Demo | **Gradio** | One-command UI, deployable to HF Spaces (Secrets keep keys hidden) |
| Config | **pydantic-settings** | Single source of truth from `.env` |
| Build | **hatchling** with `packages = ["src/emoji_toxicity"]` | Modern, no `setup.py` |

## 6. Evaluation Setup

Three-way comparison on the same samples:

| Method | Description |
|---|---|
| **Keyword baseline** | Hard-coded blacklist of ~20 commonly toxic emoji — no context |
| **Raw LLM** | GPT-5 with no RAG — tests whether retrieval actually adds value over a strong base model |
| **RAG Pipeline** (ours) | Full system: query expansion → retrieval → LLM judge → confidence gating |

Datasets:
- **HatemojiCheck** (filtered to emoji-containing rows, capped at 200 by default for cost control)
- **16 hand-crafted adversarial examples** designed as **context-flip pairs** — same emoji, one toxic and one safe, e.g. corn/porn, peach/cobbler, eggplant/recipe, monkey/idiom-vs-slur, clown/insult-vs-party, skull/threat-vs-laughter, snake/betrayal-vs-zoo, OK-hand/approval-vs-extremism

Metrics: accuracy, macro precision/recall/F1, AUROC, confusion matrix. The runner also produces a **per-perturbation-type breakdown** for the adversarial set (context_flip, hate_speech, threat, etc.) so failure modes are visible — and crucially, it **reuses RAG predictions from the main eval pass** instead of re-running inference.

## 7. Demo

`app.py` is a Gradio interface with:
- Two text inputs (message + optional context)
- An HTML report: colored verdict badge, confidence %, risk category, reasoning, per-emoji table with risk colors, retrieved KB citations
- 10 pre-loaded examples that demonstrate context-flip pairs side by side (the thesis-of-the-project demo)
- Compatible with **HuggingFace Spaces Secrets** so API keys stay private when deployed

## 8. Engineering Quality Notes

The project went through a code-review pass that produced these improvements over the rough prototype:

- **Hot-path caching with `lru_cache`**: vectorstore, embedding model, LLM chains, OpenAI client are all built **once** rather than per-request — without this, every Gradio call would reload `all-MiniLM-L6-v2` from disk
- **Batched embedding** in `index.py` — `model.encode(texts, batch_size=32)` is 5–10× faster than per-doc encoding
- **No duplicate inference** in `run_eval.py` — the adversarial breakdown reuses cached predictions
- **Unified eval runner** — three near-identical `evaluate_*` functions collapsed into one `_evaluate(samples, classify_fn, name)` helper
- **Shared utilities** in `utils.py` (`extract_emojis`, `cldr_name`, `make_vec_id`) — eliminates the `[ch for ch in text if ch in emoji.EMOJI_DATA]` pattern that was duplicated across 5 files
- **Lazy Pinecone import** in `retriever.py` so `tests/test_core.py` can run without API keys (6 tests passing offline)
- **Pinecone version pinned** `>=6.0,<8.0` to match `langchain-pinecone`'s constraint (caught during install validation)

## 9. How to Run

```bash
pip install -e .
cp .env.example .env   # add OPENAI_API_KEY, PINECONE_API_KEY, HF_TOKEN

python -m scripts.build_knowledge_base --skip-gpt   # fast: seed + CLDR
python -m scripts.build_knowledge_base               # full: with GPT enrichment
python -m scripts.build_index                        # embed → Pinecone
python app.py                                        # launch demo at :7860
python -m scripts.evaluate --skip-hatemoji           # adversarial only
python -m scripts.evaluate --max-hatemoji 200        # full benchmark
```

## 10. What's Implemented vs Future Work

**Implemented end-to-end:**
- All 5 ingestion sources, KB orchestrator with provenance, GPT enrichment
- Pinecone index builder with batched embedding
- Full RAG pipeline with structured output and confidence gating
- 3-way evaluation framework with metrics + per-perturbation breakdown
- Gradio demo with examples
- 6 unit tests (no API keys required)
- pydantic-settings config, modern packaging, README

**Stretch / not yet built:**
- Chrome extension that highlights toxic emoji on Twitter/X via FastAPI backend (mentioned as stretch goal)
- Perspective API as a third baseline (scaffolded but not wired)
- Async/parallel GPT enrichment (currently sequential)
- Live HF Space deployment (code is HF-Spaces-ready but the upload itself is manual)

## 11. The Core Contribution

The project's value isn't novel ML — it's **wiring together known components into a system that actually demonstrates context-sensitivity for emoji toxicity**, with:
- A reproducible multi-source KB (not just one dataset)
- Honest evaluation against two baselines on adversarial pairs designed to expose context-blindness
- A confidence gate that lets the model say "I don't know" instead of forcing a binary
- A polished demo that makes the thesis visible in 10 seconds

That's the gap between a 4-script prototype and a project worth showing.

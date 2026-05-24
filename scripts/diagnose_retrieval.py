"""Diagnostic: does dense retrieval recover the *correct* emoji's KB entry?

For each sampled emoji that's in the KB:
  1. Query the vector store with just the emoji (and with a context-bearing
     message containing it).
  2. Check whether the emoji's own KB entry appears in top-1 / top-3.
  3. Compare against an exact-symbol fetch (Pinecone .fetch by deterministic ID).

If dense top-1 recovery is < 90%, exact-symbol lookup is the right primary
strategy. If it's already > 95%, hybrid won't move the needle and effort is
better spent on a reranker for context-flip cases.

Usage:
    python -m scripts.diagnose_retrieval                   # 30 samples
    python -m scripts.diagnose_retrieval --n 50 --seed 1   # custom
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pinecone import Pinecone

from emoji_toxicity.config import KB_PATH, settings
from emoji_toxicity.detector.retriever import retrieve
from emoji_toxicity.utils import make_vec_id


def _load_kb_emojis_with_slang() -> list[dict]:
    """Return KB entries that carry a non-trivial slang meaning (the interesting set)."""
    entries = []
    with open(KB_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("slang_meaning") and e.get("risk_category") not in ("", "Safe"):
                entries.append(e)
    return entries


def _query_messages_for(symbol: str) -> list[tuple[str, str]]:
    """Two query variants per emoji: bare symbol, and an in-context sentence."""
    return [
        (symbol, "bare"),
        (f"What does this mean? {symbol}", "in_sentence"),
    ]


def _exact_symbol_fetch_works(pc_index, symbol: str) -> bool:
    """Return True if the deterministic vector ID exists in the Pinecone index."""
    vec_id = make_vec_id(symbol)
    res = pc_index.fetch(ids=[vec_id])
    vectors = getattr(res, "vectors", None) or res.get("vectors", {})
    return vec_id in vectors


def diagnose(n: int = 30, seed: int = 0, k: int = 3) -> None:
    kb_entries = _load_kb_emojis_with_slang()
    if not kb_entries:
        print("No KB entries with slang_meaning found — build the KB first.")
        return

    rng = random.Random(seed)
    sample = rng.sample(kb_entries, min(n, len(kb_entries)))
    symbols = [e["symbol"] for e in sample]
    print(f"Sampled {len(symbols)} emoji with slang meanings (seed={seed}).\n")

    pc = Pinecone(api_key=settings.pinecone_api_key)
    pc_index = pc.Index(settings.pinecone_index_name)

    results: dict[str, dict[str, int]] = {
        "bare": {"top1": 0, "topk": 0, "indexed": 0},
        "in_sentence": {"top1": 0, "topk": 0, "indexed": 0},
    }
    misses: list[dict] = []

    for symbol in symbols:
        indexed = _exact_symbol_fetch_works(pc_index, symbol)

        for query, variant in _query_messages_for(symbol):
            results[variant]["indexed"] += int(indexed)
            r = retrieve(query, k=k)
            retrieved_symbols = [d.metadata.get("symbol") for d in r.documents]

            if not retrieved_symbols:
                continue
            if retrieved_symbols[0] == symbol:
                results[variant]["top1"] += 1
            if symbol in retrieved_symbols:
                results[variant]["topk"] += 1
            else:
                misses.append({
                    "symbol": symbol,
                    "variant": variant,
                    "query": query,
                    "indexed": indexed,
                    "top_k_retrieved": retrieved_symbols,
                    "scores": [round(s, 3) for s in r.scores],
                })

    n_total = len(symbols)
    print(f"{'Variant':<14}{'Indexed':>10}{'Top-1':>10}{'Top-' + str(k):>10}")
    print("-" * 44)
    for variant, counts in results.items():
        top1_pct = counts["top1"] / n_total
        topk_pct = counts["topk"] / n_total
        idx_pct = counts["indexed"] / n_total
        print(
            f"{variant:<14}"
            f"{idx_pct:>9.0%}"
            f"{top1_pct:>9.0%}"
            f"{topk_pct:>9.0%}"
        )

    print(f"\n{len(misses)} miss(es) where the correct emoji was NOT in top-{k}.")
    if misses:
        print("\nFirst 10 misses (correct emoji absent from top-k):")
        for m in misses[:10]:
            indexed_marker = "" if m["indexed"] else " [NOT INDEXED]"
            print(
                f"  {m['symbol']!r}{indexed_marker} ({m['variant']}): "
                f"top-{k} = {m['top_k_retrieved']} scores={m['scores']}"
            )

    print("\nInterpretation:")
    bare_top1 = results["bare"]["top1"] / n_total
    if bare_top1 >= 0.95:
        print("  Dense retrieval is already strong. Hybrid won't help much.")
        print("  Spend effort on a cross-encoder reranker for context-flip cases.")
    elif bare_top1 >= 0.80:
        print("  Dense retrieval works most of the time.")
        print("  Adding exact-symbol lookup as a primary key would close the gap cheaply.")
    else:
        print("  Dense retrieval is weak on bare-emoji queries.")
        print("  Use exact-symbol lookup as the primary retrieval strategy.")
        print("  Keep dense as a fallback for context queries / multi-emoji combos.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, default=30, help="Number of emoji to sample.")
    p.add_argument("--seed", type=int, default=0, help="Sample seed.")
    p.add_argument("--k", type=int, default=3, help="Top-k for retrieval.")
    args = p.parse_args()
    diagnose(n=args.n, seed=args.seed, k=args.k)


if __name__ == "__main__":
    main()

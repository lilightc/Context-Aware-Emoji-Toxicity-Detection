"""CLI: Dynamic KB update pipeline.

Runs the full collect → extract → validate → index → monitor cycle.

Usage:
    python -m scripts.update_kb                        # full pipeline
    python -m scripts.update_kb --skip-collect --posts-file data/raw/posts.jsonl
    python -m scripts.update_kb --dry-run              # validate only, don't index
    python -m scripts.update_kb --health               # print KB health report only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    p = argparse.ArgumentParser(description="Dynamic KB update pipeline")
    p.add_argument("--skip-collect", action="store_true",
                   help="Skip collection step (use --posts-file instead)")
    p.add_argument("--posts-file", type=str, default=None,
                   help="Path to pre-collected posts JSONL (one {text, source} per line)")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate candidates but don't index them")
    p.add_argument("--health", action="store_true",
                   help="Print KB health report and exit")
    p.add_argument("--save-baseline", action="store_true",
                   help="Save current eval results as accuracy baseline")
    p.add_argument("--tag", type=str, default="",
                   help="Batch tag for provenance tracking (e.g. 'reddit_2026w16')")
    p.add_argument("--min-sources", type=int, default=None,
                   help="Override minimum sources for validation")
    p.add_argument("--min-confidence", type=float, default=None,
                   help="Override minimum confidence for validation")
    args = p.parse_args()

    from emoji_toxicity.ingestion.monitor import kb_health_report, check_accuracy_regression, save_baseline

    # Health report only
    if args.health:
        report = kb_health_report()
        print(json.dumps(report, indent=2))
        return

    # Save baseline only
    if args.save_baseline:
        save_baseline()
        return

    # ---- Step 1: Collect ----
    if args.skip_collect and args.posts_file:
        print(f"Loading posts from {args.posts_file}...")
        with open(args.posts_file) as f:
            posts = [json.loads(line) for line in f if line.strip()]
        print(f"  -> {len(posts)} posts loaded")
    elif args.skip_collect:
        print("ERROR: --skip-collect requires --posts-file")
        return
    else:
        from emoji_toxicity.ingestion.collectors import collect_all
        print("=" * 60)
        print("Step 1: Collecting posts from sources...")
        print("=" * 60)
        posts = collect_all()

    if not posts:
        print("No posts collected. Nothing to do.")
        return

    # ---- Step 2: Extract ----
    from emoji_toxicity.ingestion.slang_extractor import extract_slang_candidates
    print()
    print("=" * 60)
    print("Step 2: Extracting slang candidates with LLM...")
    print("=" * 60)
    candidates = extract_slang_candidates(posts)
    print(f"  -> {len(candidates)} candidates extracted")

    if not candidates:
        print("No slang candidates found. Nothing to index.")
        return

    for c in candidates:
        conf = c.get("confidence", 0)
        print(f"  {c.get('symbol', '?')} → {c.get('slang_meaning', '?')[:60]} (conf={conf:.2f})")

    # ---- Step 3: Validate ----
    from emoji_toxicity.ingestion.validation import validate_candidates, save_validation_log
    print()
    print("=" * 60)
    print("Step 3: Validating candidates...")
    print("=" * 60)
    result = validate_candidates(
        candidates,
        min_sources=args.min_sources,
        min_confidence=args.min_confidence,
    )
    print(f"  Accepted:  {len(result.accepted)}")
    print(f"  Rejected:  {len(result.rejected)}")
    print(f"  Conflicts: {len(result.conflicts)}")

    for entry in result.accepted:
        print(f"    ✓ {entry.get('symbol', '?')} → {entry.get('slang_meaning', '?')[:50]}")
    for entry in result.rejected:
        print(f"    ✗ {entry.get('symbol', '?')} — {entry.get('rejection_reason', '?')}")
    for entry in result.conflicts:
        print(f"    ⚠ {entry.get('symbol', '?')} — {entry.get('conflict_type', '?')}: "
              f"{entry.get('existing_risk', '?')} vs {entry.get('new_risk', '?')}")

    log_path = save_validation_log(result, tag=args.tag)
    print(f"  Validation log: {log_path}")

    if args.dry_run:
        print("\n[DRY RUN] Skipping indexing.")
        return

    if not result.accepted:
        print("No candidates passed validation. Nothing to index.")
        return

    # ---- Step 4: Index ----
    from emoji_toxicity.vectorstore.incremental import upsert_entries
    print()
    print("=" * 60)
    print("Step 4: Indexing accepted entries...")
    print("=" * 60)
    n_upserted = upsert_entries(result.accepted, tag=args.tag)

    # ---- Step 5: Monitor ----
    print()
    print("=" * 60)
    print("Step 5: KB health check...")
    print("=" * 60)
    health = kb_health_report()
    print(f"  Total entries: {health['total_entries']}")
    print(f"  Dynamic entries: {health['dynamic_entries']}")
    print(f"  Static entries: {health['static_entries']}")
    print(f"  Newest: {health.get('newest_entry', 'N/A')}")

    regression = check_accuracy_regression()
    if regression.get("regression"):
        print("\n  ⚠ ACCURACY REGRESSION DETECTED:")
        for r in regression["regressions"]:
            print(f"    {r['method']}: {r['baseline_accuracy']:.3f} → {r['current_accuracy']:.3f} ({r['delta']:+.3f})")
        print("  Consider rolling back the last batch of entries.")
    elif regression.get("reason") == "no baseline — saving current as baseline":
        print("  No baseline found. Run --save-baseline after your next eval.")
    else:
        print("  ✓ No accuracy regression detected.")

    print(f"\nDone. {n_upserted} entries added to KB and Pinecone index.")


if __name__ == "__main__":
    main()

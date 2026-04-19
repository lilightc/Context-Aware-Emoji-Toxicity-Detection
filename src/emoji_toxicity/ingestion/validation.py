"""Validation gate for candidate KB entries before indexing.

Candidates must pass:
1. Confidence threshold (LLM extraction confidence ≥ kb_update_confidence)
2. Multi-source confirmation (N ≥ kb_update_min_sources independent sources)
3. No destructive conflict with existing high-confidence KB entries
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

from emoji_toxicity.config import settings, KB_PATH, RESULTS_DIR


@dataclass
class ValidationResult:
    """Outcome of validating a set of candidates."""
    accepted: list[dict]
    rejected: list[dict]
    conflicts: list[dict]  # candidates that conflict with existing KB


def load_existing_kb() -> dict[str, dict]:
    """Load the current KB as a symbol → entry map."""
    if not KB_PATH.exists():
        return {}
    kb = {}
    with open(KB_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                kb[entry["symbol"]] = entry
    return kb


def validate_candidates(
    candidates: list[dict],
    min_sources: int | None = None,
    min_confidence: float | None = None,
) -> ValidationResult:
    """Validate candidate KB entries against quality gates.

    Args:
        candidates: List of extraction results from slang_extractor.
        min_sources: Minimum independent sources. Defaults to config.
        min_confidence: Minimum LLM extraction confidence. Defaults to config.

    Returns:
        ValidationResult with accepted, rejected, and conflict lists.
    """
    min_sources = min_sources or settings.kb_update_min_sources
    min_confidence = min_confidence or settings.kb_update_confidence

    existing_kb = load_existing_kb()

    # Group candidates by symbol to count independent sources
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        sym = c.get("symbol", "")
        if sym:
            by_symbol[sym].append(c)

    accepted, rejected, conflicts = [], [], []

    for sym, entries in by_symbol.items():
        # Pick highest-confidence extraction for this symbol
        best = max(entries, key=lambda e: e.get("confidence", 0))
        confidence = best.get("confidence", 0)

        # Count unique sources across all extractions for this symbol
        sources = set()
        for e in entries:
            for post in e.get("source_posts", []):
                sources.add(post.get("source", "unknown"))
            if not e.get("source_posts"):
                sources.add(e.get("extraction_model", "unknown"))
        n_sources = len(sources)

        # Gate 1: confidence threshold
        if confidence < min_confidence:
            rejected.append({
                **best,
                "rejection_reason": f"confidence {confidence:.2f} < {min_confidence}",
                "n_sources": n_sources,
            })
            continue

        # Gate 2: multi-source confirmation
        if n_sources < min_sources:
            rejected.append({
                **best,
                "rejection_reason": f"n_sources {n_sources} < {min_sources}",
                "n_sources": n_sources,
            })
            continue

        # Gate 3: conflict check against existing KB
        if sym in existing_kb:
            existing = existing_kb[sym]
            existing_risk = existing.get("risk_category", "")
            new_risk = best.get("risk_category", "")

            if existing_risk and new_risk and existing_risk != new_risk:
                conflicts.append({
                    **best,
                    "conflict_type": "risk_category_mismatch",
                    "existing_risk": existing_risk,
                    "new_risk": new_risk,
                    "n_sources": n_sources,
                })
                # Still accept if new has more sources — but flag for review
                if n_sources >= min_sources + 1:
                    accepted.append({**best, "n_sources": n_sources, "overrides_existing": True})
                continue

        # Passed all gates
        accepted.append({**best, "n_sources": n_sources})

    return ValidationResult(accepted=accepted, rejected=rejected, conflicts=conflicts)


def save_validation_log(result: ValidationResult, tag: str = "") -> Path:
    """Write validation results to a JSONL log for audit."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"kb_validation{'_' + tag if tag else ''}.jsonl"
    with open(log_path, "a") as f:
        for entry in result.accepted:
            f.write(json.dumps({"status": "accepted", **entry}, ensure_ascii=False) + "\n")
        for entry in result.rejected:
            f.write(json.dumps({"status": "rejected", **entry}, ensure_ascii=False) + "\n")
        for entry in result.conflicts:
            f.write(json.dumps({"status": "conflict", **entry}, ensure_ascii=False) + "\n")
    return log_path

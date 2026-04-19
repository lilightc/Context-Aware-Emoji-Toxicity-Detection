"""KB health monitoring — track coverage, freshness, and accuracy over time.

Run after each KB update to ensure new entries don't degrade performance.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from emoji_toxicity.config import KB_PATH, RESULTS_DIR


def kb_health_report() -> dict:
    """Compute KB health metrics: size, freshness, source distribution."""
    if not KB_PATH.exists():
        return {"error": "KB not found"}

    entries = []
    with open(KB_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    n_total = len(entries)
    n_with_slang = sum(1 for e in entries if e.get("slang_meaning"))
    n_dynamic = sum(1 for e in entries if any(
        s.startswith("dynamic:") for s in e.get("sources", [])
    ))

    # Source distribution
    all_sources = []
    for e in entries:
        all_sources.extend(e.get("sources", []))
    source_counts = dict(Counter(all_sources).most_common())

    # Freshness — entries with timestamps
    timestamps = [e["added_at"] for e in entries if "added_at" in e]
    n_versioned = len(timestamps)
    newest = max(timestamps) if timestamps else None
    oldest = min(timestamps) if timestamps else None

    # Risk category distribution
    risk_dist = dict(Counter(e.get("risk_category", "Unknown") for e in entries).most_common())

    return {
        "total_entries": n_total,
        "entries_with_slang": n_with_slang,
        "dynamic_entries": n_dynamic,
        "static_entries": n_total - n_dynamic,
        "versioned_entries": n_versioned,
        "newest_entry": newest,
        "oldest_entry": oldest,
        "source_distribution": source_counts,
        "risk_distribution": risk_dist,
    }


def check_accuracy_regression(
    current_results_path: Path | None = None,
    threshold: float = 0.05,
) -> dict:
    """Compare current eval results against the last saved baseline.

    Returns {"regression": True/False, "delta": ..., "details": ...}.
    """
    results_path = current_results_path or RESULTS_DIR / "eval_results.json"
    baseline_path = RESULTS_DIR / "eval_baseline.json"

    if not results_path.exists():
        return {"regression": False, "reason": "no current results"}
    if not baseline_path.exists():
        return {"regression": False, "reason": "no baseline — saving current as baseline"}

    with open(results_path) as f:
        current = json.load(f)
    with open(baseline_path) as f:
        baseline = json.load(f)

    regressions = []
    for curr_method in current.get("results", []):
        name = curr_method["name"]
        base_method = next(
            (r for r in baseline.get("results", []) if r["name"] == name), None
        )
        if not base_method:
            continue

        curr_acc = curr_method["seed_0"]["accuracy"]["value"]
        base_acc = base_method["seed_0"]["accuracy"]["value"]
        delta = curr_acc - base_acc

        if delta < -threshold:
            regressions.append({
                "method": name,
                "baseline_accuracy": base_acc,
                "current_accuracy": curr_acc,
                "delta": delta,
            })

    return {
        "regression": len(regressions) > 0,
        "regressions": regressions,
        "threshold": threshold,
    }


def save_baseline():
    """Copy current eval_results.json as the accuracy baseline."""
    results_path = RESULTS_DIR / "eval_results.json"
    baseline_path = RESULTS_DIR / "eval_baseline.json"
    if results_path.exists():
        import shutil
        shutil.copy2(results_path, baseline_path)
        print(f"Saved baseline: {baseline_path}")
    else:
        print("No eval_results.json to save as baseline.")

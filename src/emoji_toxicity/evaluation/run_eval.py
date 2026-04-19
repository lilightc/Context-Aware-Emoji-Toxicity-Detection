"""Run all evaluation benchmarks with bootstrap CIs, multi-seed aggregation,
and per-sample trace logging for post-hoc inspection."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Callable

import numpy as np
from tqdm import tqdm

from emoji_toxicity.config import RESULTS_DIR
from emoji_toxicity.detector.pipeline import DetectionResult, ToxicityDetector
from emoji_toxicity.evaluation.baselines import keyword_baseline, raw_llm_baseline
from emoji_toxicity.evaluation.context_flip_bench import load_context_flip_bench
from emoji_toxicity.evaluation.datasets import (
    EvalSample,
    load_adversarial_test_set,
    load_hatemoji_build_test,
    load_hatemoji_check,
    stratified_sample,
)
from emoji_toxicity.evaluation.metrics import EvalMetrics, compute_metrics
from emoji_toxicity.utils import verdict_from_score


def _verdict_to_label(verdict: str) -> int:
    """UNCERTAIN maps to TOXIC (conservative)."""
    return 0 if verdict == "SAFE" else 1


# Each classify fn returns (score, extra) where extra is a method-specific dict
# captured in the per-sample trace (e.g. agent tool calls for introspection).
ClassifyReturn = tuple[float, dict]
ClassifyFn = Callable[[EvalSample], ClassifyReturn]


def _run_classifier_once(
    samples: list[EvalSample],
    classify_fn: ClassifyFn,
    desc: str,
    n_bootstrap: int,
) -> tuple[EvalMetrics, list[dict]]:
    """Run a classifier over samples once. Returns (metrics, per_sample_records)."""
    y_true, y_pred, y_scores = [], [], []
    records: list[dict] = []

    for sample in tqdm(samples, desc=desc):
        try:
            score, extra = classify_fn(sample)
        except Exception as e:
            print(f"  [ERROR] {sample.text[:50]}... -> {e}")
            continue
        verdict = verdict_from_score(score)
        y_true.append(sample.label)
        y_pred.append(_verdict_to_label(verdict))
        y_scores.append(score)
        records.append({
            "text": sample.text,
            "context": sample.context,
            "label": sample.label,
            "pred_verdict": verdict,
            "pred_label": _verdict_to_label(verdict),
            "toxicity_score": score,
            "source": sample.source,
            "perturbation_type": sample.perturbation_type,
            **extra,
        })

    metrics = compute_metrics(y_true, y_pred, y_scores, n_bootstrap=n_bootstrap)
    return metrics, records


def _aggregate_across_seeds(per_seed_metrics: list[EvalMetrics]) -> dict:
    """Return mean ± std for each scalar metric across seeds."""
    if not per_seed_metrics:
        return {}
    fields = ["accuracy", "precision", "recall", "f1_macro"]
    if per_seed_metrics[0].auroc is not None:
        fields.append("auroc")

    out: dict = {"n_seeds": len(per_seed_metrics)}
    for f in fields:
        vals = np.array([getattr(m, f).value for m in per_seed_metrics])
        out[f] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
    return out


def _agent_trace_summary(result: DetectionResult) -> dict:
    """Extract a compact trace record from an agent-mode result."""
    if result.agent_trace is None:
        return {}
    t = result.agent_trace
    tool_names = [c["name"] for c in t.tool_calls if c["name"] != "submit_verdict"]
    return {
        "agent_iterations": t.iterations,
        "agent_info_tool_calls": len(tool_names),
        "agent_tools_used": tool_names,
        "agent_terminated_normally": t.terminated_normally,
    }


def run_full_evaluation(
    include_hatemoji: bool = True,
    sample_size: int | None = None,
    n_seeds: int = 1,
    n_bootstrap: int = 1000,
    sample_seed: int = 0,
    compare_modes: bool = False,
    use_context_flip_bench: bool = False,
) -> dict:
    """Run all benchmarks and save results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if use_context_flip_bench:
        adversarial = load_context_flip_bench()
        print(f"Loaded {len(adversarial)} context-flip benchmark samples")
    else:
        adversarial = load_adversarial_test_set()
        print(f"Loaded {len(adversarial)} adversarial test samples")

    all_samples = adversarial[:]
    if include_hatemoji and not use_context_flip_bench:
        hatemoji = load_hatemoji_check()
        hatemoji_source = "HatemojiCheck"
        if not hatemoji:
            print("  Falling back to HatemojiBuild test split (non-gated, same tokens).")
            hatemoji = load_hatemoji_build_test()
            hatemoji_source = "HatemojiBuild-test"
        print(f"  {hatemoji_source} size (emoji-filtered): {len(hatemoji)}")
        if sample_size is not None and sample_size < len(hatemoji):
            hatemoji = stratified_sample(hatemoji, sample_size, seed=sample_seed)
            labels = [s.label for s in hatemoji]
            print(
                f"  Stratified sample: n={len(hatemoji)}, "
                f"toxic={sum(labels)}, safe={len(labels) - sum(labels)}"
            )
        all_samples.extend(hatemoji)

    print(f"Total evaluation samples per seed: {len(all_samples)}")
    print(f"n_seeds={n_seeds}, n_bootstrap={n_bootstrap}\n")

    agent_detector = ToxicityDetector(mode="agent")
    workflow_detector = ToxicityDetector(mode="workflow") if compare_modes else None
    adaptive_detector = ToxicityDetector(mode="adaptive") if compare_modes else None

    def make_fns(seed: int | None) -> dict[str, ClassifyFn]:
        def kw_fn(s: EvalSample) -> ClassifyReturn:
            return keyword_baseline(s.text, s.context).toxicity_score, {}

        def llm_fn(s: EvalSample) -> ClassifyReturn:
            return raw_llm_baseline(s.text, s.context, seed=seed).toxicity_score, {}

        def agent_fn(s: EvalSample) -> ClassifyReturn:
            r = agent_detector.detect(s.text, s.context, seed=seed)
            return r.toxicity_score, _agent_trace_summary(r)

        fns: dict[str, ClassifyFn] = {
            "keyword_baseline": kw_fn,
            "raw_llm": llm_fn,
            "rag_agent": agent_fn,
        }

        if workflow_detector is not None:
            def workflow_fn(s: EvalSample) -> ClassifyReturn:
                r = workflow_detector.detect(s.text, s.context, seed=seed)
                return r.toxicity_score, {}
            fns["rag_workflow"] = workflow_fn

        if adaptive_detector is not None:
            def adaptive_fn(s: EvalSample) -> ClassifyReturn:
                r = adaptive_detector.detect(s.text, s.context, seed=seed)
                return r.toxicity_score, {"adaptive_retrieved": "(retrieval skipped" not in str(r.citations)}
            fns["rag_adaptive"] = adaptive_fn

        return fns

    method_names = ["keyword_baseline", "raw_llm", "rag_agent"]
    if compare_modes:
        method_names.extend(["rag_workflow", "rag_adaptive"])

    per_seed_results: dict[str, list[EvalMetrics]] = defaultdict(list)
    per_sample_records: dict[str, list[dict]] = defaultdict(list)

    for seed_idx in range(n_seeds):
        effective_seed = seed_idx if n_seeds > 1 else None
        fns = make_fns(effective_seed)
        print("=" * 60)
        print(f"Seed {seed_idx + 1}/{n_seeds} (OpenAI seed={effective_seed})")
        print("=" * 60)

        for name in method_names:
            if name == "keyword_baseline" and seed_idx > 0:
                continue
            metrics, records = _run_classifier_once(
                all_samples, fns[name], desc=f"{name} seed={seed_idx}", n_bootstrap=n_bootstrap
            )
            per_seed_results[name].append(metrics)
            if seed_idx == 0:
                per_sample_records[name] = records
            print(f"\n[{name}]")
            print(metrics.summary())
            print()

    # Summary output
    output: dict = {
        "config": {
            "n_seeds": n_seeds,
            "n_bootstrap": n_bootstrap,
            "n_samples": len(all_samples),
            "sample_size_hatemoji": sample_size,
            "sample_seed": sample_seed,
            "compare_modes": compare_modes,
        },
        "results": [],
    }
    for name in method_names:
        runs = per_seed_results[name]
        if not runs:
            continue
        entry: dict = {"name": name, "seed_0": runs[0].to_dict()}
        if len(runs) > 1:
            entry["across_seeds"] = _aggregate_across_seeds(runs)
        output["results"].append(entry)

    results_path = RESULTS_DIR / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Per-sample trace dump for post-hoc analysis (JSONL, one record per row)
    traces_path = RESULTS_DIR / "eval_traces.jsonl"
    with open(traces_path, "w") as f:
        for name, records in per_sample_records.items():
            for rec in records:
                f.write(json.dumps({"method": name, **rec}, ensure_ascii=False) + "\n")
    print(f"Per-sample traces saved to {traces_path}")

    # Cross-seed summary table
    if n_seeds > 1:
        print("\n" + "=" * 60)
        print(f"Cross-seed summary (mean ± std over n_seeds={n_seeds}):")
        print("=" * 60)
        print(f"{'method':<20} {'accuracy':<18} {'macro F1':<18} {'AUROC':<18}")
        for name in method_names:
            runs = per_seed_results[name]
            if not runs:
                continue
            if len(runs) == 1:
                print(f"{name:<20} (deterministic — single run)")
                continue
            agg = _aggregate_across_seeds(runs)

            def fmt(k):
                if k not in agg:
                    return "n/a".ljust(18)
                return f"{agg[k]['mean']:.3f} ± {agg[k]['std']:.3f}".ljust(18)

            print(f"{name:<20} {fmt('accuracy')} {fmt('f1_macro')} {fmt('auroc')}")

    # Agent tool-use summary (how often did it skip retrieval?)
    agent_records = per_sample_records.get("rag_agent", [])
    if agent_records:
        n_total = len(agent_records)
        n_no_tools = sum(1 for r in agent_records if r.get("agent_info_tool_calls", 0) == 0)
        n_failed = sum(1 for r in agent_records if r.get("agent_terminated_normally") is False)
        avg_tools = np.mean([r.get("agent_info_tool_calls", 0) for r in agent_records])
        tool_histogram: dict[str, int] = defaultdict(int)
        for r in agent_records:
            for t in r.get("agent_tools_used", []):
                tool_histogram[t] += 1
        print("\n" + "=" * 60)
        print("Agent tool-use summary:")
        print(f"  skipped retrieval entirely: {n_no_tools}/{n_total} ({n_no_tools / n_total:.0%})")
        print(f"  avg info-tool calls/sample: {avg_tools:.2f}")
        print(f"  failed to terminate normally: {n_failed}/{n_total}")
        for name, count in sorted(tool_histogram.items(), key=lambda kv: -kv[1]):
            print(f"  {name}: {count}")

    # Breakdown by difficulty (context-flip bench) or perturbation type (adversarial)
    if adversarial and agent_records:
        print("\n" + "=" * 60)
        print("Adversarial set breakdown by perturbation type (rag_agent, seed 0):")
        adv_texts = {s.text for s in adversarial}
        by_type: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
        for rec in agent_records:
            if rec["text"] not in adv_texts:
                continue
            pt = rec["perturbation_type"] or "unknown"
            by_type[pt]["total"] += 1
            if rec["pred_label"] == rec["label"]:
                by_type[pt]["correct"] += 1
        for pt, counts in sorted(by_type.items()):
            total = counts["total"]
            acc = counts["correct"] / total if total > 0 else 0
            marker = "  !! n<5" if total < 5 else ""
            print(f"  {pt}: {counts['correct']}/{total} ({acc:.0%}){marker}")

    return output

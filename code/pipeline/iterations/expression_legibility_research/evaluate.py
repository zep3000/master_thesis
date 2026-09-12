#!/usr/bin/env python3
"""Evaluation-side metrics, cost accounting, and promotion ranking."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from strategies import ALL_STRATEGIES, LEGIBILITY


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "expression_legibility_research"
DATA = HERE / "data"
OUTPUT = HERE / "output"
EVAL = HERE / "evaluation"
LEG_INDEX = {value: index for index, value in enumerate(LEGIBILITY)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_results(path: Path) -> dict[str, dict[str, Any]]:
    latest = {}
    for row in read_jsonl(path):
        latest[str(row["logical_key"])] = row
    return latest


def usage_cost(usage: dict[str, Any] | None) -> float:
    if not usage:
        return 0.0
    if isinstance(usage.get("cost"), (int, float)):
        return float(usage["cost"])
    return float((usage.get("cost_details") or {}).get("upstream_inference_cost") or 0.0)


def usage_tokens(usage: dict[str, Any] | None) -> tuple[int, int]:
    if not usage:
        return 0, 0
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def weighted_kappa(gold: list[int], pred: list[int]) -> float:
    if not gold:
        return 0.0
    n = len(LEGIBILITY)
    observed = [[0.0] * n for _ in range(n)]
    gh = [0.0] * n; ph = [0.0] * n
    for g, p in zip(gold, pred):
        observed[g][p] += 1; gh[g] += 1; ph[p] += 1
    observed_disagreement = sum(((i - j) ** 2 / (n - 1) ** 2) * observed[i][j] for i in range(n) for j in range(n)) / len(gold)
    expected_disagreement = sum(((i - j) ** 2 / (n - 1) ** 2) * gh[i] * ph[j] for i in range(n) for j in range(n)) / (len(gold) ** 2)
    return 1.0 - observed_disagreement / expected_disagreement if expected_disagreement else 1.0


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("pred_legibility") in LEG_INDEX]
    n = len(rows); nv = len(valid)
    gold = [LEG_INDEX[row["gold_legibility"]] for row in valid]
    pred = [LEG_INDEX[row["pred_legibility"]] for row in valid]
    per_level = {}
    recalls = []
    for level in LEGIBILITY:
        level_rows = [row for row in valid if row["gold_legibility"] == level]
        correct = sum(row["pred_legibility"] == level for row in level_rows)
        recall = correct / len(level_rows) if level_rows else None
        per_level[level] = {"n": len(level_rows), "correct": correct, "recall": recall}
        if recall is not None:
            recalls.append(recall)
    exact = sum(g == p for g, p in zip(gold, pred)) / nv if nv else 0.0
    within1 = sum(abs(g - p) <= 1 for g, p in zip(gold, pred)) / nv if nv else 0.0
    mae = mean(abs(g - p) for g, p in zip(gold, pred)) if nv else math.inf
    bias = mean(p - g for g, p in zip(gold, pred)) if nv else 0.0
    under = sum(p < g for g, p in zip(gold, pred)) / nv if nv else 0.0
    over = sum(p > g for g, p in zip(gold, pred)) / nv if nv else 0.0
    nonzero = [(g, p) for g, p in zip(gold, pred) if g > 0]
    zeros = [(g, p) for g, p in zip(gold, pred) if g == 0]
    gold_gaze = [row for row in valid if row.get("gold_gaze") not in {None, "", "not_assessable"}]
    gold_smile = [row for row in valid if row.get("gold_smile") in {"yes", "no"}]
    return {
        "n_expected": n, "n_valid": nv, "coverage": nv / n if n else 0.0,
        "balanced_accuracy": mean(recalls) if recalls else 0.0,
        "exact_accuracy": exact, "within_one_accuracy": within1,
        "mean_absolute_error": mae, "mean_signed_error": bias,
        "under_rate": under, "over_rate": over,
        "quadratic_weighted_kappa": weighted_kappa(gold, pred),
        "nonzero_recall": sum(p > 0 for _, p in nonzero) / len(nonzero) if nonzero else 0.0,
        "zero_specificity": sum(p == 0 for _, p in zeros) / len(zeros) if zeros else 0.0,
        "moderate_plus_recall": sum(p >= 2 for g, p in zip(gold, pred) if g >= 2) / sum(g >= 2 for g in gold) if any(g >= 2 for g in gold) else 0.0,
        "gaze_substantive_n": len(gold_gaze),
        "gaze_output_coverage": sum(row.get("pred_gaze") not in {None, "", "not_assessable"} for row in gold_gaze) / len(gold_gaze) if gold_gaze else 0.0,
        "gaze_exact": sum(row.get("pred_gaze") == row.get("gold_gaze") for row in gold_gaze) / len(gold_gaze) if gold_gaze else 0.0,
        "smile_binary_n": len(gold_smile),
        "smile_output_coverage": sum(row.get("pred_smile") in {"yes", "no"} for row in gold_smile) / len(gold_smile) if gold_smile else 0.0,
        "smile_exact": sum(row.get("pred_smile") == row.get("gold_smile") for row in gold_smile) / len(gold_smile) if gold_smile else 0.0,
        "per_level": per_level,
        "confusion": dict(Counter(f"{row['gold_legibility']} -> {row['pred_legibility']}" for row in valid)),
    }


def paired_bootstrap(rows_by_strategy: dict[str, list[dict[str, Any]]], baseline: str, strategy: str, draws: int = 5000) -> dict[str, Any]:
    left = {row["case_id"]: row for row in rows_by_strategy[baseline]}
    right = {row["case_id"]: row for row in rows_by_strategy[strategy]}
    ids = sorted(set(left) & set(right))
    if not ids:
        return {"n": 0}
    rng = random.Random(20260812)
    differences = []
    for _ in range(draws):
        sampled = [ids[rng.randrange(len(ids))] for _ in ids]
        b = mean(left[c]["pred_legibility"] == left[c]["gold_legibility"] for c in sampled)
        s = mean(right[c]["pred_legibility"] == right[c]["gold_legibility"] for c in sampled)
        differences.append(s - b)
    differences.sort()
    return {
        "n": len(ids), "exact_accuracy_difference": mean(differences),
        "ci95_low": differences[int(0.025 * (draws - 1))],
        "ci95_high": differences[int(0.975 * (draws - 1))],
        "probability_improvement": sum(value > 0 for value in differences) / draws,
    }


def md_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    args = parse_args()
    run_dir = OUTPUT / args.run_name
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    model = config["model"]
    manifest = json.loads((DATA / "cases_manifest.json").read_text(encoding="utf-8"))
    labels = json.loads((DATA / "gold_labels.json").read_text(encoding="utf-8"))["cases"]
    case_ids = manifest["pilot_case_ids"] if config.get("pilot") else [case["case_id"] for case in manifest["cases"]]
    latest = latest_results(run_dir / "results.jsonl")
    ledger = read_jsonl(run_dir / "request_ledger.jsonl")

    available = ["frozen_p2_baseline"]
    for strategy in ALL_STRATEGIES:
        if strategy == "p2_zero_verify":
            if any(f"{model}::{strategy}::decision::{case_id}" in latest for case_id in case_ids):
                available.append(strategy)
        elif any(f"{model}::{strategy}::primary::{case_id}" in latest for case_id in case_ids):
            available.append(strategy)

    rows_by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    detailed = []
    costs: dict[str, dict[str, Any]] = {}
    for strategy in available:
        prompt_total = completion_total = 0; cost_total = 0.0; attributed_calls = 0
        for case_id in case_ids:
            item = labels[case_id]
            if strategy == "frozen_p2_baseline":
                pred = item["frozen_p2_baseline"]
                usages = [pred.get("usage")]
                triggered = None
            elif strategy == "p2_zero_verify":
                decision = latest.get(f"{model}::{strategy}::decision::{case_id}")
                pred = decision.get("model_annotation") if decision and decision.get("ok") else {}
                primary_usage = item["frozen_p2_baseline"].get("usage")
                verifier = latest.get(f"{model}::{strategy}::verifier::{case_id}")
                usages = [primary_usage]
                if decision and decision.get("verifier_triggered"):
                    usages.append(verifier.get("usage") if verifier else None)
                triggered = bool(decision and decision.get("verifier_triggered"))
            else:
                result = latest.get(f"{model}::{strategy}::primary::{case_id}")
                pred = result.get("model_annotation") if result and result.get("ok") else {}
                usages = [result.get("usage") if result else None]
                triggered = None
            for usage in usages:
                if usage:
                    pt, ct = usage_tokens(usage)
                    prompt_total += pt; completion_total += ct; cost_total += usage_cost(usage); attributed_calls += 1
            row = {
                "strategy": strategy, "case_id": case_id,
                "source_image_id": item["source_image_id"], "face_size_bin": item["face_size_bin"],
                "human_depiction_type": item["human_depiction_type"], "human_orientation": item["human_orientation"],
                "selection_stratum": item["selection_stratum"],
                "gold_legibility": item["gold"]["face_expression_legibility"],
                "pred_legibility": pred.get("face_expression_legibility"),
                "gold_gaze": item["gold"].get("gaze_target"), "pred_gaze": pred.get("gaze_target"),
                "gold_smile": item["gold"].get("smile_present"), "pred_smile": pred.get("smile_present"),
                "gold_intensity": item["gold"].get("smile_intensity"), "pred_intensity": pred.get("smile_intensity"),
                "confidence": pred.get("confidence"), "verifier_triggered": triggered,
            }
            rows_by_strategy[strategy].append(row); detailed.append(row)
        costs[strategy] = {
            "attributed_calls": attributed_calls, "prompt_tokens": prompt_total,
            "completion_tokens": completion_total, "reported_cost_usd": cost_total,
            "calls_per_case": attributed_calls / len(case_ids),
            "prompt_tokens_per_case": prompt_total / len(case_ids),
            "completion_tokens_per_case": completion_total / len(case_ids),
            "reported_cost_per_case_usd": cost_total / len(case_ids),
        }

    metrics = {strategy: metric_summary(rows) for strategy, rows in rows_by_strategy.items()}
    for strategy in metrics:
        metrics[strategy]["cost"] = costs[strategy]
    candidates = [strategy for strategy in available if strategy != "frozen_p2_baseline" and metrics[strategy]["coverage"] == 1.0]
    ranking = sorted(candidates, key=lambda strategy: (
        -metrics[strategy]["balanced_accuracy"], -metrics[strategy]["exact_accuracy"],
        metrics[strategy]["mean_absolute_error"], -metrics[strategy]["nonzero_recall"],
    ))
    bootstraps = {strategy: paired_bootstrap(rows_by_strategy, "frozen_p2_baseline", strategy) for strategy in candidates}
    physical_cost = sum(usage_cost(row.get("usage")) for row in ledger)
    report = {
        "run_name": args.run_name, "model": model, "pilot": config.get("pilot"),
        "case_count": len(case_ids), "available_strategies": available,
        "promotion_rule": "balanced_accuracy desc, exact_accuracy desc, MAE asc, nonzero_recall desc",
        "ranking": ranking, "metrics": metrics, "paired_bootstrap_vs_frozen_p2": bootstraps,
        "physical_requests": len(ledger), "physical_reported_cost_usd": physical_cost,
        "cost_accounting_note": "p2_zero_verify attribution includes the already-incurred frozen P2 primary plus any new verifier; physical run totals include only requests made in this experiment.",
    }
    out = EVAL / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (out / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed[0]))
        writer.writeheader(); writer.writerows(detailed)

    lines = [
        f"# Expression-legibility results: {args.run_name}", "",
        f"Model: `{model}`. Cases: {len(case_ids)}. Physical requests: {len(ledger)}. Physical recorded cost: ${physical_cost:.6f}.", "",
        "This diagnostic set is stratified using frozen first-100 errors and is not prevalence-representative.", "",
        "| Rank | Strategy | Balanced acc. | Exact | Within 1 | MAE | Signed bias | Under | Nonzero recall | Zero specificity | Moderate+ recall | Smile exact | Gaze exact | Calls/case | $/case |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    display = ["frozen_p2_baseline", *ranking]
    for rank, strategy in enumerate(display):
        m = metrics[strategy]; c = m["cost"]
        lines.append(
            f"| {'baseline' if rank == 0 else rank} | {strategy} | {md_pct(m['balanced_accuracy'])} | {md_pct(m['exact_accuracy'])} | "
            f"{md_pct(m['within_one_accuracy'])} | {m['mean_absolute_error']:.3f} | {m['mean_signed_error']:+.3f} | "
            f"{md_pct(m['under_rate'])} | {md_pct(m['nonzero_recall'])} | {md_pct(m['zero_specificity'])} | "
            f"{md_pct(m['moderate_plus_recall'])} | {md_pct(m['smile_exact'])} | {md_pct(m['gaze_exact'])} | "
            f"{c['calls_per_case']:.2f} | ${c['reported_cost_per_case_usd']:.6f} |"
        )
    lines += ["", "## Recall by gold legibility", "", "| Strategy | Zero | Low | Moderate | High |", "|---|---:|---:|---:|---:|"]
    for strategy in display:
        per = metrics[strategy]["per_level"]
        lines.append(f"| {strategy} | " + " | ".join(md_pct(per[level]["recall"]) for level in LEGIBILITY) + " |")
    lines += ["", "## Paired exact-accuracy difference from frozen P2", "", "| Strategy | Difference | Bootstrap 95% interval | P(improvement) |", "|---|---:|---:|---:|"]
    for strategy in ranking:
        b = bootstraps[strategy]
        lines.append(f"| {strategy} | {b.get('exact_accuracy_difference', 0):+.3f} | [{b.get('ci95_low', 0):+.3f}, {b.get('ci95_high', 0):+.3f}] | {md_pct(b.get('probability_improvement', 0))} |")
    lines += ["", "`p2_zero_verify` strategy cost includes its frozen P2 primary and conditional verifier. Physical run cost includes only new experiment requests.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(out / "report.md")
    print("Ranking:", ranking)


if __name__ == "__main__":
    main()

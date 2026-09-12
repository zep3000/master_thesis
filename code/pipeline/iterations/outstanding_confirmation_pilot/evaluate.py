#!/usr/bin/env python3
"""Gold-reading evaluation for the frozen outstanding-confirmation pilot."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "outstanding_confirmation_pilot"
CANONICAL = ROOT / "qwen_iteration" / "canonical_candidate_280"
sys.path.insert(0, str(CANONICAL))


def module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path); assert spec and spec.loader
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value); return value


ce = module(CANONICAL / "evaluate.py", "pilot_canonical_evaluate")
common = module(CANONICAL / "common.py", "pilot_canonical_common")


def predictions(cohort: str, variant: str) -> dict[str, dict[str, Any]]:
    if variant == "baseline":
        path = CANONICAL / "output" / "assembled" / cohort / "canonical_candidate_v1.jsonl"
    else:
        path = HERE / "output" / "assembled" / f"{cohort}.jsonl"
    return {row["image_id"]: row["annotation"] for row in common.rows(path) if row.get("ok")}


def compact(value: dict[str, Any]) -> dict[str, Any]:
    role = value["fields"]["people"]["annotation_role"]
    outstanding = role["class_metrics"].get("outstanding_individual", {})
    return {
        "person_f1": value["detection"]["people"]["f1"],
        "person_precision": value["detection"]["people"]["precision"],
        "person_recall": value["detection"]["people"]["recall"],
        "role_exact": role["exact_accuracy"],
        "outstanding_support": outstanding.get("support"),
        "outstanding_precision": outstanding.get("precision"),
        "outstanding_recall": outstanding.get("recall"),
        "has_outstanding_exact": value["fields"]["ads"]["has_outstanding_individuals"]["exact_accuracy"],
        "has_outstanding_yes_recall": value["fields"]["ads"]["has_outstanding_individuals"]["class_metrics"].get("yes", {}).get("recall"),
        "person_field_exact": {field: metric.get("exact_accuracy") for field, metric in value["fields"]["people"].items()},
    }


def main() -> int:
    ev = ce.frozen(); report: dict[str, Any] = {"cohorts": {}, "pooled": {}}
    pooled_h = {}; pooled_ids = []; pooled_predictions = {"baseline": {}, "confirmation": {}}
    for cohort in ["difficult140", "stratified140"]:
        human = ce.gold(cohort); ids = ce.selected_ids(cohort); report["cohorts"][cohort] = {}
        for variant in ["baseline", "confirmation"]:
            pred = predictions(cohort, variant)
            metric = ce.route_mode(ev, human, pred, ids, "legacy_strict", variant, 1000)
            report["cohorts"][cohort][variant] = compact(metric)
        for image_id in ids:
            key = f"{cohort}::{image_id}"; pooled_ids.append(key)
            if image_id in human: pooled_h[key] = human[image_id]
            for variant in pooled_predictions:
                pred = predictions(cohort, variant)
                if image_id in pred: pooled_predictions[variant][key] = pred[image_id]
    metrics = {}
    for variant, pred in pooled_predictions.items():
        metrics[variant] = ce.route_mode(ev, pooled_h, pred, pooled_ids, "legacy_strict", variant, 2000)
        report["pooled"][variant] = compact(metrics[variant])
    base = report["pooled"]["baseline"]; test = report["pooled"]["confirmation"]
    base_tp = round((base["outstanding_support"] or 0) * (base["outstanding_recall"] or 0))
    test_tp = round((test["outstanding_support"] or 0) * (test["outstanding_recall"] or 0))
    precision_gain = (test["outstanding_precision"] or 0) - (base["outstanding_precision"] or 0)
    report["gate"] = {
        "precision_gain": precision_gain,
        "baseline_true_positive_outstanding": base_tp,
        "confirmation_true_positive_outstanding": test_tp,
        "true_positive_loss": base_tp - test_tp,
        "person_f1_delta": test["person_f1"] - base["person_f1"],
        "criteria": {"precision_gain_at_least_0.15": precision_gain >= 0.15, "true_positive_loss_at_most_1": base_tp - test_tp <= 1, "person_f1_non_decreasing": test["person_f1"] >= base["person_f1"]},
    }
    report["gate"]["promote"] = all(report["gate"]["criteria"].values())
    path = HERE / "evaluation" / "pilot_results.json"; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["pooled"] | {"gate": report["gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

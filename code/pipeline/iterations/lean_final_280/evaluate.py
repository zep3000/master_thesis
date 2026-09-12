#!/usr/bin/env python3
"""Evaluation-only strict/lenient comparison for lean_final_280.

This module is the only file in the package that reads human annotations.  It
also reports the pre-existing evaluator's standalone human-no-group slice.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "lean_final_280"
FROZEN_DIR = ROOT / "qwen_iteration" / "first100_pipeline_matrix"
GOLD = {
    "difficult140": ROOT / "annotation_results" / "test_collection_200_difficult_joined_v1_2026-08-02.json",
    "stratified140": ROOT / "annotation_results" / "economist_decade_face_count_stratified_200_seed20260812_min2_2026-08-13_full.json",
}


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def frozen_evaluator():
    spec = importlib.util.spec_from_file_location("lean_frozen_eval", FROZEN_DIR / "evaluate.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_gold(cohort: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(GOLD[cohort].read_text(encoding="utf-8"))
    result = {}
    for row in payload.get("annotations") or []:
        if cohort == "difficult140" and row.get("assignment_code") != "79201188":
            continue
        if isinstance(row.get("payload"), dict):
            result[str(row["image_id"])] = row["payload"]
    return result


def ids_for(cohort: str, scope: str) -> list[str]:
    manifest = json.loads((HERE / "data" / f"manifest_{cohort}.json").read_text(encoding="utf-8"))
    all_ids = [item["image_id"] for item in manifest["images"]]
    if scope == "all":
        return all_ids
    pilots = json.loads((HERE / "data" / "pilot_ids.json").read_text(encoding="utf-8"))["ids"][cohort]
    return pilots


def load_predictions(cohort: str, route: str) -> dict[str, dict[str, Any]]:
    if route.startswith("structure_"):
        path = HERE / "output" / cohort / f"{route}.jsonl"
        return {row["task_key"]: row["model_annotation"] for row in rows(path) if row.get("ok")}
    path = HERE / "output" / cohort / "assembled" / f"{route}.jsonl"
    return {row["image_id"]: row["annotation"] for row in rows(path) if row.get("ok") and isinstance(row.get("annotation"), dict)}


def compact(summary: dict[str, Any]) -> dict[str, Any]:
    output = {"evaluated_pages": summary["evaluated_pages"], "missing_prediction": summary["missing_prediction"]}
    for mode in ["strict", "lenient"]:
        value = summary["match_modes"][mode]
        output[mode] = {
            "detection_f1": {name: value["detection"][name]["f1"] for name in ["ads", "people", "groups"]},
            "no_group": {
                "pages": value["human_no_group"]["pages"],
                "group_free_specificity": value["human_no_group"]["page_group_free_specificity"],
                "person_f1": value["human_no_group"]["person_detection"]["f1"],
            },
            "group_positive": {
                "pages": value["human_group_positive"]["pages"],
                "group_presence_recall": value["human_group_positive"]["page_group_presence_recall"],
            },
        }
        if "people" in value["fields"]:
            output[mode]["person_exact"] = {field: value["fields"]["people"][field]["exact_accuracy"] for field in [
                "perceived_age", "perceived_gender_presentation", "face_expression_legibility", "face_orientation", "gaze_target", "smile_present", "smile_intensity"
            ]}
            output[mode]["person_lenient"] = {field: value["fields"]["people"][field]["lenient_accuracy"] for field in [
                "perceived_age", "face_expression_legibility", "smile_intensity"
            ]}
            output[mode]["group_exact"] = {field: value["fields"]["groups"][field]["exact_accuracy"] for field in [
                "age_composition", "gender_presentation_composition", "expression_legibility_distribution", "dominant_gaze", "smile_prevalence", "dominant_smile_intensity"
            ]}
    return output


def category_brand_audit(predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    categories = Counter()
    brands = Counter()
    confidence_by_category: defaultdict[str, list[float]] = defaultdict(list)
    all_ads = []
    for image_id, annotation in predictions.items():
        for ad in annotation.get("advertisements") or []:
            category = str(ad.get("ad_category") or "missing")
            categories[category] += 1
            confidence_by_category[category].append(float(ad.get("ad_category_confidence") or 0))
            brand = ad.get("brand_or_advertiser")
            if brand:
                brands[str(brand)] += 1
            all_ads.append((image_id, ad))
    low_confidence = sorted(
        [{"image_id": image_id, "advertisement_id": ad.get("advertisement_id"), "category": ad.get("ad_category"), "category_confidence": ad.get("ad_category_confidence"), "brand": ad.get("brand_or_advertiser"), "brand_confidence": ad.get("brand_confidence")} for image_id, ad in all_ads],
        key=lambda item: float(item.get("category_confidence") or 0),
    )[:30]
    return {
        "ads": len(all_ads),
        "category_counts": dict(categories.most_common()),
        "category_mean_confidence": {key: sum(values) / len(values) for key, values in confidence_by_category.items()},
        "brand_present_rate": (sum(brands.values()) / len(all_ads)) if all_ads else None,
        "brand_counts_top30": dict(brands.most_common(30)),
        "lowest_category_confidence": low_confidence,
    }


def cost_summary() -> dict[str, Any]:
    ledger = rows(HERE / "output" / "request_ledger.jsonl")
    result = {"requests": len(ledger), "successful": sum(bool(row.get("ok")) for row in ledger), "by_configuration": {}}
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        key = "/".join(str(row.get(field, "-")) for field in ["stage", "variant", "style", "gaze"])
        grouped[key].append(row)
    for key, items in grouped.items():
        usage = Counter()
        cost = 0.0
        for item in items:
            raw = item.get("usage") or {}
            for field in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                usage[field] += int(raw.get(field) or 0)
            cost += float(raw.get("cost") or 0)
        result["by_configuration"][key] = {"requests": len(items), **dict(usage), "cost_usd": cost, "mean_cost_per_request_usd": cost / len(items)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["pilot", "all"], required=True)
    parser.add_argument("--cohorts", nargs="+", choices=["difficult140", "stratified140"], default=["difficult140", "stratified140"])
    parser.add_argument("--routes", nargs="+", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    evaluator = frozen_evaluator()
    report: dict[str, Any] = {"tag": args.tag, "scope": args.scope, "cohorts": {}, "cost": cost_summary()}
    pooled_gold: dict[str, dict[str, Any]] = {}
    pooled_pred: dict[str, dict[str, dict[str, Any]]] = {route: {} for route in args.routes}
    pooled_ids = []
    for cohort in args.cohorts:
        ids = ids_for(cohort, args.scope)
        if args.limit:
            ids = ids[: args.limit]
        gold = load_gold(cohort)
        report["cohorts"][cohort] = {}
        pooled_gold.update(gold)
        pooled_ids.extend(ids)
        for route in args.routes:
            predictions = load_predictions(cohort, route)
            pooled_pred[route].update(predictions)
            evaluation = evaluator.evaluate_scope(gold, predictions, ids)
            report["cohorts"][cohort][route] = {"evaluation": evaluation, "summary": compact(evaluation), "category_brand_audit": category_brand_audit(predictions)}
            print(json.dumps({"cohort": cohort, "route": route, **compact(evaluation)}, ensure_ascii=False))
    report["pooled"] = {}
    for route in args.routes:
        evaluation = evaluator.evaluate_scope(pooled_gold, pooled_pred[route], pooled_ids)
        report["pooled"][route] = {"evaluation": evaluation, "summary": compact(evaluation), "category_brand_audit": category_brand_audit(pooled_pred[route])}
        print(json.dumps({"cohort": "pooled", "route": route, **compact(evaluation)}, ensure_ascii=False))
    out = HERE / "evaluation" / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

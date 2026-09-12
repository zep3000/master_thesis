#!/usr/bin/env python3
"""Evaluation-only module. This is the sole package file that reads gold."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from common import BASE, HERE, LEDGER, manifest, rows


ROOT = Path(r".")
FROZEN = ROOT / "qwen_iteration" / "first100_pipeline_matrix" / "evaluate.py"
GOLD = {
    "difficult140": ROOT / "annotation_results" / "test_collection_200_difficult_joined_v1_2026-08-02.json",
    "stratified140": ROOT / "annotation_results" / "economist_decade_face_count_stratified_200_seed20260812_min2_2026-08-13_full.json",
}
ORDER = ["0_not_legible", "1_low_legibility", "2_moderate_legibility", "3_high_legibility"]


def evaluator():
    spec = importlib.util.spec_from_file_location("final_refinement_frozen", FROZEN)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def gold(cohort: str) -> dict[str, dict]:
    payload = json.loads(GOLD[cohort].read_text(encoding="utf-8")); result = {}
    for row in payload.get("annotations") or []:
        if cohort == "difficult140" and row.get("assignment_code") != "79201188":
            continue
        if isinstance(row.get("payload"), dict):
            result[str(row["image_id"])] = row["payload"]
    return result


def predictions(cohort: str, route: str) -> dict[str, dict]:
    if route.startswith("structure_"):
        variant = route.removeprefix("structure_")
        return {x["task_key"]: x["model_annotation"] for x in rows(HERE / "output" / "structure" / cohort / f"{variant}.jsonl") if x.get("ok")}
    return {x["image_id"]: x["annotation"] for x in rows(HERE / "output" / "assembled" / cohort / f"{route}.jsonl") if x.get("ok")}


def ids(cohort: str, scope: str) -> list[str]:
    values = [x["image_id"] for x in manifest(cohort)["images"]]
    if scope == "pilot":
        wanted = set(json.loads((HERE / "data" / "pilot_ids.json").read_text(encoding="utf-8"))["ids"][cohort])
        values = [x for x in values if x in wanted]
    return values


def compact(summary: dict) -> dict:
    out = {"pages": summary["evaluated_pages"], "missing": len(summary["missing_prediction"])}
    for mode in ["strict", "lenient"]:
        value = summary["match_modes"][mode]
        out[mode] = {
            "f1": {x: value["detection"][x]["f1"] for x in ["ads", "people", "groups"]},
            "no_group_specificity": value["human_no_group"]["page_group_free_specificity"],
            "no_group_person_f1": value["human_no_group"]["person_detection"]["f1"],
            "group_presence_recall": value["human_group_positive"]["page_group_presence_recall"],
            "person": {x: value["fields"]["people"][x]["exact_accuracy"] for x in ["perceived_age", "perceived_gender_presentation", "face_expression_legibility", "face_orientation", "gaze_target", "smile_present", "smile_intensity"]},
            "person_lenient": {x: value["fields"]["people"][x]["lenient_accuracy"] for x in ["perceived_age", "face_expression_legibility", "smile_intensity"]},
            "group": {x: value["fields"]["groups"][x]["exact_accuracy"] for x in ["group_type", "age_composition", "gender_presentation_composition", "expression_legibility_distribution", "dominant_gaze", "smile_prevalence", "dominant_smile_intensity"]},
        }
    return out


def matched_people(ev: Any, human: dict, pred: dict, mode: str, wanted: set[str] | None = None) -> list[Any]:
    pairs = []
    for image_id in sorted(set(human) & set(pred)):
        ad_pairs = ev.pair_items(ev.all_items(image_id, human[image_id], "ads"), ev.all_items(image_id, pred[image_id], "ads"), "ads", mode)
        for ad_pair in ad_pairs:
            found = ev.pair_items(ev.person_items(image_id, ad_pair.human.data), ev.person_items(image_id, ad_pair.predicted.data), "people", mode)
            pairs.extend(p for p in found if wanted is None or f"{image_id}::{p.predicted.item_id}" in wanted)
    return pairs


def qwk(matrix: list[list[int]]) -> float | None:
    n = sum(map(sum, matrix))
    if not n:
        return None
    hr = [sum(r) for r in matrix]; pc = [sum(matrix[i][j] for i in range(4)) for j in range(4)]
    observed = sum(((i-j)**2 / 9) * matrix[i][j] for i in range(4) for j in range(4)) / n
    expected = sum(((i-j)**2 / 9) * hr[i] * pc[j] / n for i in range(4) for j in range(4)) / n
    return 1 - observed / expected if expected else None


def ordinal(ev: Any, pairs: list[Any]) -> dict:
    matrix = [[0] * 4 for _ in range(4)]
    for pair in pairs:
        h = ev.normalize_value(pair.human.data.get("face_expression_legibility"), "face_expression_legibility")
        p = ev.normalize_value(pair.predicted.data.get("face_expression_legibility"), "face_expression_legibility")
        if h in ORDER and p in ORDER:
            matrix[ORDER.index(h)][ORDER.index(p)] += 1
    n = sum(map(sum, matrix)); exact = sum(matrix[i][i] for i in range(4)); within = sum(matrix[i][j] for i in range(4) for j in range(4) if abs(i-j) <= 1)
    recalls = [matrix[i][i] / sum(matrix[i]) if sum(matrix[i]) else None for i in range(4)]
    mae = sum(abs(i-j) * matrix[i][j] for i in range(4) for j in range(4)) / n if n else None
    bias = sum((j-i) * matrix[i][j] for i in range(4) for j in range(4)) / n if n else None
    static_exact = sum(matrix[2]) / n if n else None
    return {"n": n, "exact": exact/n if n else None, "within_one": within/n if n else None, "mae": mae, "bias": bias,
            "macro_recall": statistics.fmean(x for x in recalls if x is not None) if n else None, "per_class_recall": dict(zip(ORDER, recalls)),
            "qwk": qwk(matrix), "static_moderate_exact": static_exact, "confusion_human_rows": matrix,
            "predicted_distribution": {ORDER[j]: sum(matrix[i][j] for i in range(4))/n if n else None for j in range(4)}}


def costs() -> dict:
    result = {"attempts": len(rows(LEDGER)), "by_stage": {}}
    grouped: dict[str, list] = {}
    for row in rows(LEDGER): grouped.setdefault(f"{row.get('stage')}/{row.get('variant')}", []).append(row)
    for key, values in grouped.items():
        result["by_stage"][key] = {"requests": len(values), "cost_usd": sum(float((x.get("usage") or {}).get("cost") or 0) for x in values),
                                   "tokens": sum(int((x.get("usage") or {}).get("total_tokens") or 0) for x in values)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["pilot", "all"], required=True)
    parser.add_argument("--routes", nargs="+", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args(); ev = evaluator()
    fer_wanted = {x["task_id"] for x in json.loads((HERE / "data" / "fer_pilot.json").read_text(encoding="utf-8"))["tasks"]} if args.scope == "pilot" else None
    report: dict[str, Any] = {"tag": args.tag, "scope": args.scope, "costs": costs(), "cohorts": {}, "pooled": {}}
    pooled_h, pooled_ids = {}, []
    pooled_p = {route: {} for route in args.routes}
    for cohort in ["difficult140", "stratified140"]:
        human, selected = gold(cohort), ids(cohort, args.scope); pooled_h.update(human); pooled_ids += selected; report["cohorts"][cohort] = {}
        cohort_wanted = {x for x in fer_wanted or set() if x.split("::", 1)[0] in set(selected)} if fer_wanted is not None else None
        for route in args.routes:
            pred = predictions(cohort, route); pooled_p[route].update(pred)
            full = ev.evaluate_scope(human, pred, selected)
            fer = {mode: ordinal(ev, matched_people(ev, human, pred, mode, cohort_wanted)) for mode in ["strict", "lenient"]}
            report["cohorts"][cohort][route] = {"summary": compact(full), "fer": fer, "full": full}
            print(json.dumps({"cohort": cohort, "route": route, "summary": compact(full), "fer": fer}, ensure_ascii=False))
    for route in args.routes:
        full = ev.evaluate_scope(pooled_h, pooled_p[route], pooled_ids)
        fer = {mode: ordinal(ev, matched_people(ev, pooled_h, pooled_p[route], mode, fer_wanted)) for mode in ["strict", "lenient"]}
        report["pooled"][route] = {"summary": compact(full), "fer": fer, "full": full}
    out = HERE / "evaluation" / f"{args.tag}.json"; out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

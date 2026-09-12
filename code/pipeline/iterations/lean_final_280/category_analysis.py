#!/usr/bin/env python3
"""Evaluation-only per-gold-category agreement for final filtered routes."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import evaluate as package_eval


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "lean_final_280"
FROZEN = ROOT / "qwen_iteration" / "first100_pipeline_matrix" / "evaluate.py"


def frozen():
    spec = importlib.util.spec_from_file_location("lean_category_frozen", FROZEN)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pairs_for(ev: Any, gold: dict[str, Any], predicted: dict[str, Any], ids: list[str], mode: str) -> dict[str, list[Any]]:
    pairs = {"people": [], "groups": []}
    for image_id in ids:
        h_ads = ev.all_items(image_id, gold[image_id], "ads")
        p_ads = ev.all_items(image_id, predicted[image_id], "ads")
        for ad_pair in ev.pair_items(h_ads, p_ads, "ads", mode):
            pairs["people"].extend(ev.pair_items(ev.person_items(image_id, ad_pair.human.data), ev.person_items(image_id, ad_pair.predicted.data), "people", mode))
            pairs["groups"].extend(ev.pair_items(ev.group_items(image_id, ad_pair.human.data), ev.group_items(image_id, ad_pair.predicted.data), "groups", mode))
    return pairs


def stats(ev: Any, pairs: list[Any], field: str) -> dict[str, Any]:
    values: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"matched": 0, "exact": 0, "lenient": 0})
    for pair in pairs:
        human = ev.normalize_value(pair.human.data.get(field), field)
        predicted = ev.normalize_value(pair.predicted.data.get(field), field)
        if human is None:
            continue
        row = values[str(human)]
        row["matched"] += 1
        row["exact"] += int(ev.exact_label(field, human, predicted))
        row["lenient"] += int(ev.lenient_label(field, human, predicted))
    return {key: {**value, "exact_accuracy": value["exact"] / value["matched"], "lenient_accuracy": value["lenient"] / value["matched"]} for key, value in sorted(values.items())}


def main() -> int:
    ev = frozen()
    ids = []
    gold = {}
    for cohort in ["difficult140", "stratified140"]:
        cohort_ids = package_eval.ids_for(cohort, "all")
        ids.extend(cohort_ids)
        gold.update(package_eval.load_gold(cohort))
    routes = ["lean_s2c_direct_nogaze", "lean_s2c_direct_gaze", "lean_s2c_direct_hybrid"]
    fields = {
        "people": ["perceived_gender_presentation", "face_expression_legibility", "face_orientation", "gaze_target", "smile_present", "smile_intensity"],
        "groups": ["expression_legibility_distribution", "dominant_gaze", "smile_prevalence", "dominant_smile_intensity"],
    }
    report = {}
    for route in routes:
        predicted = {}
        for cohort in ["difficult140", "stratified140"]:
            predicted.update(package_eval.load_predictions(cohort, route))
        report[route] = {}
        for mode in ["strict", "lenient"]:
            pairs = pairs_for(ev, gold, predicted, ids, mode)
            report[route][mode] = {entity: {field: stats(ev, pairs[entity], field) for field in entity_fields} for entity, entity_fields in fields.items()}
    out = HERE / "evaluation" / "final_category_analysis.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

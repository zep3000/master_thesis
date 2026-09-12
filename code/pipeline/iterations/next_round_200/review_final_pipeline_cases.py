#!/usr/bin/env python3
"""Evaluation-only audit for the final-pipeline narrowing discussion.

The script compares R0/R1/R4 to the two development gold sets, extracts the
user-identified pages, and reports recurrent field confusions. It is never
imported by inference and writes only into the evaluation directory.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "next_round_200"
OLD = ROOT / "qwen_iteration" / "first100_pipeline_matrix"
RUN = "scale_v1"
OUTPUT = HERE / "evaluation" / RUN / "user_case_audit.json"
OLD_GOLD = ROOT / "annotation_results" / "test_collection_200_difficult_joined_v1_2026-08-02.json"
NEW_GOLD = ROOT / "annotation_results" / "economist_decade_face_count_stratified_200_seed20260812_min2_52123740_2026-08-12.json"

CITED = {
    "difficult100": [
        "1995-1104-0074_0075", "1955-0409-0028", "2002-0525-0019",
        "1957-1005-0008", "2003-0607-0020",
    ],
    "stratified100": [
        "1965-1113-0053", "1945-1110-0020", "1998-0523-0038_0039",
        "1953-0808-0025", "1968-1012-0087", "2001-0707-0109",
        "1994-0917-0034_0035", "1997-0809-0070_0071",
    ],
}

FIELDS = [
    "face_orientation", "gaze_target", "perceived_gender_presentation",
    "face_expression_legibility", "smile_present",
]


def load_evaluator() -> Any:
    spec = importlib.util.spec_from_file_location("final_case_eval", OLD / "evaluate.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EV = load_evaluator()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_gold(cohort: str) -> dict[str, dict[str, Any]]:
    if cohort == "difficult100":
        payload = json.loads(OLD_GOLD.read_text(encoding="utf-8"))
        return {
            row["image_id"]: row["payload"]
            for row in payload["annotations"]
            if row.get("assignment_code") == "79201188" and isinstance(row.get("payload"), dict)
        }
    payload = json.loads(NEW_GOLD.read_text(encoding="utf-8"))
    rows = sorted(payload["annotations"], key=lambda row: int(row["sort_order"]))[:100]
    return {row["image_id"]: row["payload"] for row in rows if isinstance(row.get("payload"), dict)}


def load_route(cohort: str, route: str) -> dict[str, dict[str, Any]]:
    path = HERE / "output" / RUN / cohort / "assembled" / f"{route}.jsonl"
    return {
        row["image_id"]: row["annotation"]
        for row in read_jsonl(path)
        if row.get("ok") and isinstance(row.get("annotation"), dict)
    }


def paired_people(gold: dict[str, Any], pred: dict[str, Any], image_ids: list[str]) -> list[Any]:
    pairs = []
    for image_id in image_ids:
        if image_id not in gold or image_id not in pred:
            continue
        human_ads = EV.all_items(image_id, gold[image_id], "ads")
        pred_ads = EV.all_items(image_id, pred[image_id], "ads")
        for ad_pair in EV.pair_items(human_ads, pred_ads, "ads", "strict"):
            pairs.extend(
                EV.pair_items(
                    EV.person_items(image_id, ad_pair.human.data),
                    EV.person_items(image_id, ad_pair.predicted.data),
                    "people",
                    "strict",
                )
            )
    return pairs


def field_audit(pairs: list[Any], field: str) -> dict[str, Any]:
    confusion: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    assessed = exact = 0
    for pair in pairs:
        human = EV.normalize_value(pair.human.data.get(field), field)
        pred = EV.normalize_value(pair.predicted.data.get(field), field)
        if human is None and pred is None:
            continue
        assessed += 1
        exact += int(human == pred)
        key = (str(human), str(pred))
        confusion[key] += 1
        if human != pred and pair.human.image_id not in examples[key]:
            examples[key].append(pair.human.image_id)
    return {
        "matched_assessed": assessed,
        "exact": exact,
        "exact_accuracy": exact / assessed if assessed else None,
        "top_confusions": [
            {
                "human": human,
                "predicted": pred,
                "n": count,
                "example_image_ids": examples[(human, pred)][:8],
            }
            for (human, pred), count in confusion.most_common()
            if human != pred
        ][:8],
    }


def ambiguity_audit(pairs: list[Any]) -> dict[str, Any]:
    predicted_ambiguous = [
        pair for pair in pairs
        if pair.predicted.data.get("perceived_gender_presentation") == "ambiguous_or_androgynous"
    ]
    correct = sum(
        pair.human.data.get("perceived_gender_presentation") == "ambiguous_or_androgynous"
        for pair in predicted_ambiguous
    )
    human_ambiguous = sum(
        pair.human.data.get("perceived_gender_presentation") == "ambiguous_or_androgynous"
        for pair in pairs
    )
    return {
        "matched_people": len(pairs),
        "predicted_ambiguous": len(predicted_ambiguous),
        "predicted_ambiguous_rate": len(predicted_ambiguous) / len(pairs) if pairs else None,
        "predicted_ambiguous_correct": correct,
        "predicted_ambiguous_precision": correct / len(predicted_ambiguous) if predicted_ambiguous else None,
        "human_ambiguous": human_ambiguous,
        "false_ambiguous_examples": list(dict.fromkeys(
            pair.human.image_id for pair in predicted_ambiguous
            if pair.human.data.get("perceived_gender_presentation") != "ambiguous_or_androgynous"
        ))[:15],
    }


def summarize_annotation(annotation: dict[str, Any] | None) -> dict[str, Any] | None:
    if not annotation:
        return None
    ads = []
    for ad in annotation.get("advertisements") or []:
        ads.append({
            "advertisement_id": ad.get("advertisement_id") or ad.get("ad_id"),
            "bbox": ad.get("bbox_1000") or ad.get("bbox"),
            "face_count_band": ad.get("face_depiction_count_band"),
            "has_outstanding_individuals": ad.get("has_outstanding_individuals"),
            "people": [
                {
                    "person_id": person.get("person_id"),
                    "role": person.get("annotation_role"),
                    "bbox": person.get("face_bbox_1000") or person.get("face_bbox"),
                    "gender": person.get("perceived_gender_presentation"),
                    "legibility": person.get("face_expression_legibility"),
                    "orientation": person.get("face_orientation"),
                    "gaze": person.get("gaze_target"),
                    "smile": person.get("smile_present"),
                }
                for person in ad.get("people") or []
            ],
            "groups": [
                {
                    "group_id": group.get("group_id"),
                    "bbox": group.get("bbox_1000") or group.get("bbox"),
                    "legibility": group.get("expression_legibility_distribution"),
                    "gaze": group.get("dominant_gaze"),
                    "smile_prevalence": group.get("smile_prevalence"),
                }
                for group in ad.get("groups") or []
            ],
        })
    return {"page": annotation.get("page"), "advertisements": ads}


def legacy_orientation_values(route: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for image_id, annotation in route.items():
        for ad in annotation.get("advertisements") or []:
            for person in ad.get("people") or []:
                value = person.get("face_orientation")
                if value in {"tilted_down", "tilted_up"}:
                    values[value] += 1
                    if image_id not in examples[value]:
                        examples[value].append(image_id)
    return {value: {"n": count, "example_image_ids": examples[value][:15]} for value, count in values.items()}


def main() -> int:
    output: dict[str, Any] = {
        "purpose": "evaluation-only user case audit; never used by inference",
        "run": RUN,
        "cohorts": {},
        "pooled_r1": {},
    }
    pooled_pairs: list[Any] = []
    for cohort in ["difficult100", "stratified100"]:
        manifest = json.loads((HERE / "data" / f"manifest_{cohort}.json").read_text(encoding="utf-8"))
        ids = [row["image_id"] for row in manifest["images"][:100]]
        gold = load_gold(cohort)
        routes = {route: load_route(cohort, route) for route in ["r0", "r1", "r4"]}
        route_pairs = {route: paired_people(gold, pred, ids) for route, pred in routes.items()}
        pooled_pairs.extend(route_pairs["r1"])
        output["cohorts"][cohort] = {
            "field_metrics": {
                route: {field: field_audit(pairs, field) for field in FIELDS}
                for route, pairs in route_pairs.items()
            },
            "r1_gender_ambiguity": ambiguity_audit(route_pairs["r1"]),
            "r1_legacy_orientations": legacy_orientation_values(routes["r1"]),
            "cited_cases": {
                image_id: {
                    "gold": summarize_annotation(gold.get(image_id)),
                    "r0": summarize_annotation(routes["r0"].get(image_id)),
                    "r1": summarize_annotation(routes["r1"].get(image_id)),
                    "r4": summarize_annotation(routes["r4"].get(image_id)),
                }
                for image_id in CITED[cohort]
            },
        }
    output["pooled_r1"] = {
        "field_metrics": {field: field_audit(pooled_pairs, field) for field in FIELDS},
        "gender_ambiguity": ambiguity_audit(pooled_pairs),
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

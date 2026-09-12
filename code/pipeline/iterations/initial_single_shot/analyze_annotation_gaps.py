"""Reproducible diagnostics for Qwen annotation runs versus assignment 79201188.

This analysis reuses the matching rules from the existing Venice evaluator and
adds route-aware error decomposition, current-playbook compatibility metrics,
and validation/post-processing summaries. It never feeds gold data into a
model run; it only reads completed exports after inference.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(r".")
QWEN_DIR = ROOT / "qwen_iteration"
HUMAN_PATH = ROOT / "annotation_results" / "test_collection_200_difficult_joined_v1_2026-08-02.json"
ASSIGNMENT_CODE = "79201188"
OUT_PATH = QWEN_DIR / "analysis" / "qwen_vs_human_gap_metrics.json"

EVALUATOR_PATH = (
    QWEN_DIR
    / "venice_first30_single_shot"
    / "evaluate_venice_vs_human_79201188.py"
)

RUNS = {
    "deepinfra_single_shot": {
        "export": ROOT / "openrouter_qwen" / "output" / "qwen_iteration_first20_single_shot_comparison.jsonl",
        "completed": QWEN_DIR / "output" / "first20_single_shot.jsonl",
        "shared20_eval": QWEN_DIR / "evaluation_shared_first20" / "deepinfra_single_shot_first20_vs_human_79201188_field_evaluation.json",
    },
    "venice_single_shot": {
        "export": ROOT / "openrouter_qwen" / "output" / "qwen35_9b_venice_first30_single_shot_full_annotation.jsonl",
        "completed": QWEN_DIR / "venice_first30_single_shot" / "output" / "venice_first30_single_shot.completed.jsonl",
        "shared20_eval": QWEN_DIR / "evaluation_shared_first20" / "venice_single_shot_shared_first20_vs_human_79201188_field_evaluation.json",
        "common30_eval": QWEN_DIR / "venice_first30_single_shot" / "evaluation" / "venice_vs_human_79201188_field_evaluation.json",
    },
    "venice_two_step": {
        "export": ROOT / "openrouter_qwen" / "output" / "qwen35_9b_venice_first40_two_step_full_annotation.jsonl",
        "completed": QWEN_DIR / "venice_first40_two_step" / "output" / "venice_first40_two_step.completed.jsonl",
        "shared20_eval": QWEN_DIR / "evaluation_shared_first20" / "venice_two_step_shared_first20_vs_human_79201188_field_evaluation.json",
        "common30_eval": QWEN_DIR / "venice_first40_two_step" / "evaluation" / "venice_first40_two_step_common_first30_vs_human_79201188_field_evaluation.json",
    },
    "venice_four_pass": {
        "export": ROOT / "openrouter_qwen" / "output" / "qwen35_9b_venice_first30_four_pass_full_annotation.jsonl",
        "completed": QWEN_DIR / "venice_first30_four_pass" / "output" / "venice_first30_four_pass.completed.jsonl",
        "shared20_eval": QWEN_DIR / "evaluation_shared_first20" / "venice_four_pass_shared_first20_vs_human_79201188_field_evaluation.json",
        "common30_eval": QWEN_DIR / "venice_first30_four_pass" / "evaluation" / "venice_first30_four_pass_vs_human_79201188_field_evaluation.json",
    },
}

ILLUSTRATION_VALUES = {"naturalistic_illustration", "stylized_illustration", "illustration"}
LEGACY_VERTICAL_ORIENTATION = {
    "tilted_down",
    "tilted_up",
    "frontal_head_angled_down",
    "frontal_head_angled_up",
}


def load_evaluator() -> Any:
    spec = importlib.util.spec_from_file_location("qwen_existing_evaluator", EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load evaluator at {EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVAL = load_evaluator()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_human() -> dict[str, dict[str, Any]]:
    export = read_json(HUMAN_PATH)
    return {
        row["image_id"]: row["payload"]
        for row in export["annotations"]
        if str(row.get("assignment_code")) == ASSIGNMENT_CODE
    }


def load_llm(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["image_id"]: row["annotation"]
        for row in read_jsonl(path)
        if isinstance(row.get("annotation"), dict)
    }


def pct(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def eval_snapshot(path: Path) -> dict[str, Any]:
    data = read_json(path)
    detection = data["detection"]["matched_detection"]
    boxes = {item["label"]: item for item in data["box_metrics"]}
    fields: dict[str, dict[str, Any]] = {}
    for section in (
        "page_field_summary",
        "ad_field_summary",
        "person_iou_field_summary",
        "group_field_summary",
    ):
        fields[section] = {
            name: {
                "n": item["n_comparable_or_one_missing"],
                "strict_matches": item["strict_matches"],
                "strict_accuracy": item["strict_accuracy"],
                "loose_matches": item["loose_ordinal_matches"],
                "loose_accuracy": item["loose_ordinal_accuracy"],
                "one_missing": item["one_missing_included"],
                "top_confusions": item["top_confusions"],
            }
            for name, item in data[section].items()
        }
    return {
        "pages": data["scope"]["overlap_pages"],
        "detection": detection,
        "person_box_mean_iou": boxes["people IoU >= .20"]["mean_iou"],
        "person_box_median_iou": boxes["people IoU >= .20"]["median_iou"],
        "group_box_mean_iou": boxes["groups IoU >= .18"]["mean_iou"],
        "ad_box_mean_iou": boxes["ads IoU >= .20"]["mean_iou"],
        "fields": fields,
    }


def item_key(image_id: str, ad_id: str | None, entity_id: str | None) -> tuple[str, str | None, str | None]:
    return image_id, ad_id, entity_id


def center_inside(box: list[float] | None, container: list[float] | None) -> bool:
    if not box or not container:
        return False
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return container[0] <= cx <= container[2] and container[1] <= cy <= container[3]


def band_route(value: Any) -> str:
    return "crowd_10_plus" if str(value) in {"10_20", "20_plus"} else "exact_under_10"


def normalize_v117_depiction(value: Any) -> Any:
    value = EVAL.normalize_value(value, "depiction_type")
    return "illustration" if value in ILLUSTRATION_VALUES else value


def accuracy(values: Iterable[tuple[Any, Any]]) -> dict[str, Any]:
    pairs = list(values)
    matches = sum(h == l for h, l in pairs)
    return {"n": len(pairs), "matches": matches, "accuracy": pct(matches, len(pairs))}


def confidence_summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": sum(values) / len(values) if values else None,
        "at_least_0_8": sum(value >= 0.8 for value in values),
    }


def field_pair_stats(person_pairs: list[Any]) -> dict[str, Any]:
    fields = [name for name, _ in EVAL.PERSON_FIELDS]
    result: dict[str, Any] = {}
    for field in fields:
        pairs: list[tuple[Any, Any]] = []
        confusion: Counter[tuple[str, str]] = Counter()
        for pair in person_pairs:
            human = EVAL.normalize_value(pair.human.get(field), field)
            llm = EVAL.normalize_value(pair.llm.get(field), field)
            if human is None and llm is None:
                continue
            pairs.append((human, llm))
            confusion[(str(human), str(llm))] += 1
        matches = sum(h == l for h, l in pairs)
        result[field] = {
            "n": len(pairs),
            "matches": matches,
            "accuracy": pct(matches, len(pairs)),
            "confusions": [
                {"human": h, "llm": l, "n": n}
                for (h, l), n in confusion.most_common()
            ],
        }
    both_legible = [
        pair
        for pair in person_pairs
        if EVAL.normalize_value(pair.human.get("face_expression_legibility"), "face_expression_legibility") != "0_not_legible"
        and EVAL.normalize_value(pair.llm.get("face_expression_legibility"), "face_expression_legibility") != "0_not_legible"
    ]
    for field in ("smile_present", "smile_intensity", "mouth_covered"):
        comparable = []
        for pair in both_legible:
            human = EVAL.normalize_value(pair.human.get(field), field)
            llm = EVAL.normalize_value(pair.llm.get(field), field)
            if human is None and llm is None:
                continue
            comparable.append((human, llm))
        result[f"{field}_both_legible"] = accuracy(comparable)
    cross_field = Counter()
    for pair in person_pairs:
        legibility = EVAL.normalize_value(pair.llm.get("face_expression_legibility"), "face_expression_legibility")
        smile = EVAL.normalize_value(pair.llm.get("smile_present"), "smile_present")
        mouth = EVAL.normalize_value(pair.llm.get("mouth_covered"), "mouth_covered")
        gaze = EVAL.normalize_value(pair.llm.get("gaze_target"), "gaze_target")
        if legibility in {"1_low_legibility", "2_moderate_legibility", "3_high_legibility"}:
            cross_field["legible_people"] += 1
            if smile in {None, "not_assessable"}:
                cross_field["legible_but_smile_abstained"] += 1
            if mouth == "not_assessable":
                cross_field["legible_but_mouth_not_assessable"] += 1
        if legibility in {"2_moderate_legibility", "3_high_legibility"}:
            cross_field["moderate_or_high_legibility"] += 1
            if smile in {None, "not_assessable"}:
                cross_field["moderate_or_high_but_smile_abstained"] += 1
        if legibility == "0_not_legible":
            cross_field["zero_legibility_people"] += 1
            if smile is not None or gaze is not None:
                cross_field["zero_legibility_route_violation_after_normalization"] += 1
    result["cross_field_consistency"] = dict(cross_field)
    return result


def current_playbook_compatibility(ad_pairs: list[Any], person_pairs: list[Any], group_pairs: list[Any]) -> dict[str, Any]:
    ad_depiction = accuracy(
        (normalize_v117_depiction(pair.human.get("depiction_type")), normalize_v117_depiction(pair.llm.get("depiction_type")))
        for pair in ad_pairs
    )
    person_depiction = accuracy(
        (normalize_v117_depiction(pair.human.get("depiction_type")), normalize_v117_depiction(pair.llm.get("depiction_type")))
        for pair in person_pairs
    )
    orientation_pairs = []
    removed_legacy = 0
    for pair in person_pairs:
        human = EVAL.normalize_value(pair.human.get("face_orientation"), "face_orientation")
        llm = EVAL.normalize_value(pair.llm.get("face_orientation"), "face_orientation")
        if human in LEGACY_VERTICAL_ORIENTATION or llm in LEGACY_VERTICAL_ORIENTATION:
            removed_legacy += 1
            continue
        if human is None and llm is None:
            continue
        orientation_pairs.append((human, llm))
    ad_extent_both_present = []
    ad_duplicate_both_present = []
    ad_unique_count_both_present = []
    for pair in ad_pairs:
        for field, target in (
            ("extent", ad_extent_both_present),
            ("duplicate_faces_present", ad_duplicate_both_present),
            ("unique_face_count", ad_unique_count_both_present),
        ):
            human = EVAL.normalize_value(pair.human.get(field), field)
            llm = EVAL.normalize_value(pair.llm.get(field), field)
            if human is not None and llm is not None:
                target.append((human, llm))
    group_intensity_route_equivalent = []
    for pair in group_pairs:
        human = EVAL.normalize_value(pair.human.get("dominant_smile_intensity"), "dominant_smile_intensity")
        llm = EVAL.normalize_value(pair.llm.get("dominant_smile_intensity"), "dominant_smile_intensity")
        if human in {None, "not_applicable"}:
            human = "not_applicable_or_null"
        if llm in {None, "not_applicable"}:
            llm = "not_applicable_or_null"
        group_intensity_route_equivalent.append((human, llm))
    return {
        "ad_depiction_type_v117_collapse": ad_depiction,
        "person_depiction_type_v117_collapse": person_depiction,
        "orientation_without_removed_vertical_tilt": {
            **accuracy(orientation_pairs),
            "legacy_pairs_removed": removed_legacy,
        },
        "ad_extent_when_both_present": accuracy(ad_extent_both_present),
        "ad_duplicate_when_both_present": accuracy(ad_duplicate_both_present),
        "ad_unique_count_when_both_present": accuracy(ad_unique_count_both_present),
        "group_dominant_smile_intensity_null_equals_not_applicable": accuracy(group_intensity_route_equivalent),
    }


def route_and_detection_diagnostics(
    human_by_image: dict[str, dict[str, Any]],
    llm_by_image: dict[str, dict[str, Any]],
    image_ids: list[str],
) -> dict[str, Any]:
    ad_pairs: list[Any] = []
    person_pairs: list[Any] = []
    group_pairs: list[Any] = []
    for image_id in image_ids:
        human = human_by_image[image_id]
        llm = llm_by_image[image_id]
        ad_pairs.extend(EVAL.greedy_iou_pairs(image_id, EVAL.flatten_ads(human), EVAL.flatten_ads(llm), 0.20, "iou>=.20"))
        person_pairs.extend(EVAL.greedy_iou_pairs(image_id, EVAL.flatten_people(human), EVAL.flatten_people(llm), 0.20, "iou>=.20"))
        group_pairs.extend(EVAL.greedy_iou_pairs(image_id, EVAL.flatten_groups(human), EVAL.flatten_groups(llm), 0.18, "iou>=.18"))

    route_confusion: Counter[tuple[str, str]] = Counter()
    crowd = Counter()
    exact = Counter()
    for pair in ad_pairs:
        human_route = band_route(pair.human.get("face_depiction_count_band"))
        llm_route = band_route(pair.llm.get("face_depiction_count_band"))
        route_confusion[(human_route, llm_route)] += 1
        bucket = crowd if human_route == "crowd_10_plus" else exact
        bucket["matched_ads"] += 1
        bucket["human_people"] += len(pair.human.get("people") or [])
        bucket["llm_people"] += len(pair.llm.get("people") or [])
        bucket["human_groups"] += len(pair.human.get("groups") or [])
        bucket["llm_groups"] += len(pair.llm.get("groups") or [])

    matched_llm = {
        item_key(pair.image_id, pair.llm_ad_id, pair.llm_id)
        for pair in person_pairs
    }
    matched_human = {
        item_key(pair.image_id, pair.human_ad_id, pair.human_id)
        for pair in person_pairs
    }
    unmatched_llm = Counter()
    unmatched_human = Counter()
    unmatched_llm_pages: Counter[str] = Counter()
    missed_human_pages: Counter[str] = Counter()
    matched_person_confidence: list[float] = []
    unmatched_person_confidence: list[float] = []
    page_correct_confidence: list[float] = []
    page_incorrect_confidence: list[float] = []
    page_audit = Counter()
    for image_id in image_ids:
        human = human_by_image[image_id]
        llm = llm_by_image[image_id]
        human_groups = EVAL.flatten_groups(human)
        has_gold_group = bool(human_groups)
        human_count = EVAL.normalize_value(EVAL.value_at(human, "page.qualifying_ad_count"), "qualifying_ad_count")
        llm_count = EVAL.normalize_value(EVAL.value_at(llm, "page.qualifying_ad_count"), "qualifying_ad_count")
        page_ok = human_count == llm_count
        page_audit["correct" if page_ok else "incorrect"] += 1
        page_flags = llm.get("review_flags") or []
        if page_flags:
            page_audit["correct_flagged" if page_ok else "incorrect_flagged"] += 1
        root_confidence = llm.get("confidence")
        if isinstance(root_confidence, (int, float)):
            (page_correct_confidence if page_ok else page_incorrect_confidence).append(float(root_confidence))
        for ad_id, person_id, person, box in EVAL.flatten_people(llm):
            confidence = person.get("confidence")
            is_matched = item_key(image_id, ad_id, person_id) in matched_llm
            if isinstance(confidence, (int, float)):
                (matched_person_confidence if is_matched else unmatched_person_confidence).append(float(confidence))
            if is_matched:
                continue
            unmatched_llm["total"] += 1
            unmatched_llm_pages[image_id] += 1
            if has_gold_group:
                unmatched_llm["on_page_with_gold_group"] += 1
            else:
                unmatched_llm["on_page_without_gold_group"] += 1
            if any(center_inside(box, group_box) for *_rest, group_box in human_groups):
                unmatched_llm["center_inside_gold_group_box"] += 1
        for ad_id, person_id, _person, _box in EVAL.flatten_people(human):
            if item_key(image_id, ad_id, person_id) in matched_human:
                continue
            unmatched_human["total"] += 1
            missed_human_pages[image_id] += 1
            if has_gold_group:
                unmatched_human["on_page_with_gold_group"] += 1
            else:
                unmatched_human["on_page_without_gold_group"] += 1

    exact_route_agree = route_confusion[("exact_under_10", "exact_under_10")]
    crowd_route_agree = route_confusion[("crowd_10_plus", "crowd_10_plus")]
    return {
        "matched_ads": len(ad_pairs),
        "matched_people": len(person_pairs),
        "matched_groups": len(group_pairs),
        "ad_route_confusion": [
            {"human": h, "llm": l, "n": n}
            for (h, l), n in sorted(route_confusion.items())
        ],
        "ad_route_accuracy": pct(exact_route_agree + crowd_route_agree, len(ad_pairs)),
        "matched_gold_crowd_ads": dict(crowd),
        "matched_gold_exact_ads": dict(exact),
        "unmatched_llm_people": {
            **dict(unmatched_llm),
            "top_pages": [{"image_id": image_id, "n": n} for image_id, n in unmatched_llm_pages.most_common(10)],
        },
        "missed_human_people": {
            **dict(unmatched_human),
            "top_pages": [{"image_id": image_id, "n": n} for image_id, n in missed_human_pages.most_common(10)],
        },
        "confidence_audit": {
            "page_count_decisions": dict(page_audit),
            "page_confidence_correct": confidence_summary(page_correct_confidence),
            "page_confidence_incorrect": confidence_summary(page_incorrect_confidence),
            "matched_person_confidence": confidence_summary(matched_person_confidence),
            "unmatched_person_confidence": confidence_summary(unmatched_person_confidence),
        },
        "current_playbook_compatibility": current_playbook_compatibility(ad_pairs, person_pairs, group_pairs),
        "person_fields": field_pair_stats(person_pairs),
    }


def canonical_validation_error(error: str) -> str:
    if "another_person" in error or "gaze_target_person" in error:
        return "gaze_reference_consistency"
    if "required property" in error or "advertisements length" in error:
        return "missing_or_placeholder_structure"
    if "duplicate role" in error or "duplicate_of_person_id" in error or "identity" in error:
        return "duplicate_identity_consistency"
    if "face route" in error or "crowd route" in error or "has_outstanding_individuals" in error or "face_depiction_count_band" in error:
        return "count_route_consistency"
    if "bbox" in error or "box" in error:
        return "bbox_schema_or_geometry"
    return "other"


def validation_snapshot(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    error_counts: Counter[str] = Counter()
    records_by_error: Counter[str] = Counter()
    normalization_counts: Counter[str] = Counter()
    records_with_normalization = 0
    for row in rows:
        seen: set[str] = set()
        for error in row.get("validation_errors") or []:
            category = canonical_validation_error(str(error))
            error_counts[category] += 1
            seen.add(category)
        for category in seen:
            records_by_error[category] += 1
        actions = row.get("normalization_actions") or []
        if actions:
            records_with_normalization += 1
        for action in actions:
            normalized = re.sub(r"advertisements\[\d+\]", "advertisements[*]", str(action))
            normalized = re.sub(r"\.people\[\d+\]", ".people[*]", normalized)
            normalized = re.sub(r"\.groups\[\d+\]", ".groups[*]", normalized)
            normalized = re.sub(r"\s*->.*$", " -> ...", normalized)
            normalization_counts[normalized] += 1
    return {
        "records": len(rows),
        "strictly_valid": sum(row.get("ok") is True for row in rows),
        "invalid": sum(row.get("ok") is not True for row in rows),
        "validation_error_instances": sum(error_counts.values()),
        "validation_errors_by_category": dict(error_counts.most_common()),
        "records_by_error_category": dict(records_by_error.most_common()),
        "normalization_actions": sum(normalization_counts.values()),
        "records_with_normalization": records_with_normalization,
        "top_normalization_actions": [
            {"action": action, "n": n}
            for action, n in normalization_counts.most_common(12)
        ],
    }


def main() -> None:
    human = load_human()
    shared20_ids = read_json(RUNS["deepinfra_single_shot"]["shared20_eval"])["scope"]["overlap_image_ids"]
    common30_ids = read_json(RUNS["venice_single_shot"]["common30_eval"])["scope"]["overlap_image_ids"]

    output: dict[str, Any] = {
        "analysis_scope": {
            "human_path": str(HUMAN_PATH),
            "human_assignment_records": len(human),
            "assignment_code": ASSIGNMENT_CODE,
            "shared20_image_ids": shared20_ids,
            "common30_image_ids": common30_ids,
            "matching": {
                "ads_iou": 0.20,
                "people_iou": 0.20,
                "groups_iou": 0.18,
            },
        },
        "runs": {},
    }
    for run_name, spec in RUNS.items():
        llm = load_llm(spec["export"])
        run_output: dict[str, Any] = {
            "shared20": eval_snapshot(spec["shared20_eval"]),
            "validation": validation_snapshot(spec["completed"]),
        }
        if spec.get("common30_eval"):
            run_output["common30"] = eval_snapshot(spec["common30_eval"])
            run_output["common30_route_diagnostics"] = route_and_detection_diagnostics(human, llm, common30_ids)
        output["runs"][run_name] = run_output

    four_crop_path = QWEN_DIR / "venice_first30_four_pass" / "output" / "crop_features.jsonl"
    crop_rows = read_jsonl(four_crop_path)
    output["four_pass_crop_self_check"] = {
        "crop_records": len(crop_rows),
        "crop_face_fit": dict(Counter((row.get("model_annotation") or {}).get("crop_face_fit") for row in crop_rows)),
        "suggest_rebox": dict(Counter(str((row.get("model_annotation") or {}).get("suggest_rebox")) for row in crop_rows)),
        "actual_correction_tasks": len(read_jsonl(QWEN_DIR / "venice_first30_four_pass" / "output" / "crop_corrections.jsonl")),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()

"""Evaluate direct and individual-first group annotations against human gold.

This file contains the *predeclared* deterministic reducer used for the primary
analysis.  It is intentionally separate from run.py: inference cannot import or
read the gold labels, while this evaluator joins predictions to gold only after
all model calls have finished.

Primary reducer thresholds (frozen before inspecting model outputs):
  - at least 50% of relevant faces must be assessable;
  - >=90% agreement produces all/only;
  - >=60% agreement produces mostly/dominant;
  - smile prevalence uses 0, (0,1/3], (1/3,2/3), [2/3,1), 1.

The report also evaluates a small, explicitly labelled threshold-sensitivity
grid.  Those variants are robustness checks, not fitted alternatives.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "group_aggregation_experiment"
GOLD_PATH = HERE / "evaluation" / "gold_groups_first50.json"
OUTPUT_DIR = HERE / "output"
EVALUATION_DIR = HERE / "evaluation"

FIELDS = [
    "group_type",
    "age_composition",
    "gender_presentation_composition",
    "expression_legibility_distribution",
    "dominant_gaze",
    "smile_prevalence",
    "dominant_smile_intensity",
]
EXPRESSION_FIELDS = [
    "expression_legibility_distribution",
    "dominant_gaze",
    "smile_prevalence",
    "dominant_smile_intensity",
]
PRIMARY_THRESHOLDS = {"assessable": 0.50, "mostly": 0.60, "all": 0.90}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--evaluation-dir", type=Path, default=EVALUATION_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_ok(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        latest[str(row["task_key"])] = row
    return {key: row for key, row in latest.items() if row.get("ok") is True}


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def normalize_group(annotation: dict[str, Any] | None) -> dict[str, Any] | None:
    if annotation is None:
        return None
    result = {field: annotation.get(field) for field in FIELDS}
    if result["dominant_smile_intensity"] == "not_applicable":
        result["dominant_smile_intensity"] = None
    return result


def top_category(values: list[str]) -> tuple[str | None, float]:
    if not values:
        return None, 0.0
    counts = Counter(values)
    value, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return value, count / len(values)


def deterministic_reduce(
    group_type: str | None,
    individuals: list[dict[str, Any]],
    *,
    assessable_threshold: float = PRIMARY_THRESHOLDS["assessable"],
    mostly_threshold: float = PRIMARY_THRESHOLDS["mostly"],
    all_threshold: float = PRIMARY_THRESHOLDS["all"],
) -> dict[str, Any]:
    """Map independent face labels to the annotation-playbook group schema."""
    people = [face for face in individuals if isinstance(face, dict)]
    count = len(people)
    result: dict[str, Any] = {
        "group_type": group_type,
        "age_composition": None,
        "gender_presentation_composition": None,
        "expression_legibility_distribution": None,
        "dominant_gaze": None,
        "smile_prevalence": None,
        "dominant_smile_intensity": None,
    }
    if count == 0:
        return result

    # Age: reduce seven face-level values to the three playbook age families.
    age_map = {
        "infant": "young",
        "child": "young",
        "adolescent": "young",
        "young_adult": "young",
        "middle_adult": "middle",
        "older_adult": "older",
    }
    ages = [age_map[face.get("perceived_age")] for face in people if face.get("perceived_age") in age_map]
    if len(ages) / count < assessable_threshold:
        result["age_composition"] = "not_assessable"
    else:
        age, share = top_category(ages)
        if share >= all_threshold:
            result["age_composition"] = f"{age}_only"
        elif share >= mostly_threshold:
            result["age_composition"] = f"mostly_{age}"
        else:
            result["age_composition"] = "mixed"

    genders = [
        face.get("perceived_gender_presentation")
        for face in people
        if face.get("perceived_gender_presentation") in {"feminine", "masculine", "ambiguous_or_androgynous"}
    ]
    if len(genders) / count < assessable_threshold:
        result["gender_presentation_composition"] = "not_assessable"
    elif "ambiguous_or_androgynous" in genders:
        # The target category is explicitly a presence flag, not a majority.
        result["gender_presentation_composition"] = "ambiguous_or_androgynous_present"
    else:
        gender, share = top_category(genders)
        if share >= all_threshold:
            result["gender_presentation_composition"] = f"{gender}_only"
        elif share >= mostly_threshold:
            result["gender_presentation_composition"] = f"mostly_{gender}"
        else:
            result["gender_presentation_composition"] = "mixed"

    legibility_values = [
        face.get("face_expression_legibility")
        for face in people
        if face.get("face_expression_legibility")
        in {"0_not_legible", "1_low_legibility", "2_moderate_legibility", "3_high_legibility"}
    ]
    legibility, legibility_share = top_category(legibility_values)
    if legibility is not None:
        if legibility_share >= all_threshold:
            result["expression_legibility_distribution"] = f"all_{legibility}"
        elif legibility_share >= mostly_threshold:
            result["expression_legibility_distribution"] = f"mostly_{legibility}"
        else:
            result["expression_legibility_distribution"] = "mixed_legibility"

    if result["expression_legibility_distribution"] == "all_0_not_legible":
        return result

    gaze_map = {
        "viewer_camera": "toward_viewer_camera",
        "another_person": "toward_each_other",
        "advertised_product": "toward_object",
        "other_object": "toward_object",
        "off_frame_or_scene_direction": "off_frame_or_scene_direction",
    }
    gazes = [gaze_map[face.get("gaze_target")] for face in people if face.get("gaze_target") in gaze_map]
    if len(gazes) / count < assessable_threshold:
        result["dominant_gaze"] = "not_assessable"
    else:
        gaze, share = top_category(gazes)
        result["dominant_gaze"] = gaze if share >= mostly_threshold else "mixed"

    expression_visible = [
        face
        for face in people
        if face.get("face_expression_legibility") in {"1_low_legibility", "2_moderate_legibility", "3_high_legibility"}
    ]
    substantive_smiles = [face for face in expression_visible if face.get("smile_present") in {"yes", "no"}]
    if not expression_visible or len(substantive_smiles) / len(expression_visible) < assessable_threshold:
        result["smile_prevalence"] = "not_assessable"
        return result

    yes = [face for face in substantive_smiles if face.get("smile_present") == "yes"]
    ratio = len(yes) / len(substantive_smiles)
    if ratio == 0:
        result["smile_prevalence"] = "none"
    elif ratio <= 1 / 3:
        result["smile_prevalence"] = "minority"
    elif ratio < 2 / 3:
        result["smile_prevalence"] = "about_half"
    elif ratio < 1:
        result["smile_prevalence"] = "majority"
    else:
        result["smile_prevalence"] = "all"

    if not yes:
        return result
    intensity_map = {
        "1_slight": "slight",
        "2_clear": "clear",
        "3_broad": "broad_or_laughter_like",
        "4_laughter_like": "broad_or_laughter_like",
    }
    intensities = [intensity_map[face.get("smile_intensity")] for face in yes if face.get("smile_intensity") in intensity_map]
    intensity, share = top_category(intensities)
    if intensity is not None:
        result["dominant_smile_intensity"] = intensity if share >= mostly_threshold else "mixed"
    return result


def deterministic_reduce_tally(
    annotation: dict[str, Any],
    *,
    assessable_threshold: float = PRIMARY_THRESHOLDS["assessable"],
    mostly_threshold: float = PRIMARY_THRESHOLDS["mostly"],
    all_threshold: float = PRIMARY_THRESHOLDS["all"],
) -> dict[str, Any]:
    """Apply the same frozen thresholds directly to crop-level percentages."""
    result: dict[str, Any] = {
        "group_type": annotation.get("group_type"),
        "age_composition": None,
        "gender_presentation_composition": None,
        "expression_legibility_distribution": None,
        "dominant_gaze": None,
        "smile_prevalence": None,
        "dominant_smile_intensity": None,
    }

    age = annotation.get("age_percent") or {}
    age_assessable = sum(float(age.get(key, 0)) for key in ("young", "middle", "older"))
    if age_assessable < 100 * assessable_threshold:
        result["age_composition"] = "not_assessable"
    elif age_assessable:
        category = max(("young", "middle", "older"), key=lambda key: (float(age.get(key, 0)), key))
        share = float(age.get(category, 0)) / age_assessable
        result["age_composition"] = f"{category}_only" if share >= all_threshold else f"mostly_{category}" if share >= mostly_threshold else "mixed"

    gender = annotation.get("gender_percent") or {}
    gender_assessable = sum(float(gender.get(key, 0)) for key in ("feminine", "masculine", "ambiguous_or_androgynous"))
    if gender_assessable < 100 * assessable_threshold:
        result["gender_presentation_composition"] = "not_assessable"
    elif float(gender.get("ambiguous_or_androgynous", 0)) > 0:
        result["gender_presentation_composition"] = "ambiguous_or_androgynous_present"
    elif gender_assessable:
        category = max(("feminine", "masculine"), key=lambda key: (float(gender.get(key, 0)), key))
        share = float(gender.get(category, 0)) / gender_assessable
        result["gender_presentation_composition"] = f"{category}_only" if share >= all_threshold else f"mostly_{category}" if share >= mostly_threshold else "mixed"

    legibility = annotation.get("legibility_percent") or {}
    legibility_keys = ("0_not_legible", "1_low_legibility", "2_moderate_legibility", "3_high_legibility")
    total_legibility = sum(float(legibility.get(key, 0)) for key in legibility_keys)
    if total_legibility:
        category = max(legibility_keys, key=lambda key: (float(legibility.get(key, 0)), key))
        share = float(legibility.get(category, 0)) / total_legibility
        result["expression_legibility_distribution"] = f"all_{category}" if share >= all_threshold else f"mostly_{category}" if share >= mostly_threshold else "mixed_legibility"
    if result["expression_legibility_distribution"] == "all_0_not_legible":
        return result

    gaze = annotation.get("gaze_percent") or {}
    gaze_keys = ("toward_viewer_camera", "toward_each_other", "toward_object", "off_frame_or_scene_direction")
    gaze_assessable = sum(float(gaze.get(key, 0)) for key in gaze_keys)
    if gaze_assessable < 100 * assessable_threshold:
        result["dominant_gaze"] = "not_assessable"
    elif gaze_assessable:
        category = max(gaze_keys, key=lambda key: (float(gaze.get(key, 0)), key))
        result["dominant_gaze"] = category if float(gaze.get(category, 0)) / gaze_assessable >= mostly_threshold else "mixed"

    smile = annotation.get("smile_percent") or {}
    yes, no = float(smile.get("yes", 0)), float(smile.get("no", 0))
    smile_assessable = yes + no
    if smile_assessable < 100 * assessable_threshold:
        result["smile_prevalence"] = "not_assessable"
        return result
    ratio = yes / smile_assessable if smile_assessable else 0
    if ratio == 0:
        result["smile_prevalence"] = "none"
    elif ratio <= 1 / 3:
        result["smile_prevalence"] = "minority"
    elif ratio < 2 / 3:
        result["smile_prevalence"] = "about_half"
    elif ratio < 1:
        result["smile_prevalence"] = "majority"
    else:
        result["smile_prevalence"] = "all"
    if yes == 0:
        return result

    intensity = annotation.get("smiling_face_intensity_percent") or {}
    intensity_keys = ("slight", "clear", "broad_or_laughter_like")
    intensity_assessable = sum(float(intensity.get(key, 0)) for key in intensity_keys)
    if intensity_assessable:
        category = max(intensity_keys, key=lambda key: (float(intensity.get(key, 0)), key))
        share = float(intensity.get(category, 0)) / intensity_assessable
        result["dominant_smile_intensity"] = category if share >= mostly_threshold else "mixed"
    return result


def exact_metrics(gold: dict[str, dict[str, Any]], predictions: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    per_field: dict[str, Any] = {}
    confusions: dict[str, dict[str, int]] = {}
    available = sum(predictions.get(key) is not None for key in gold)
    for field in FIELDS:
        pairs = [(truth[field], (predictions.get(key) or {}).get(field)) for key, truth in gold.items()]
        correct = sum(expected == observed for expected, observed in pairs)
        non_null = [(expected, observed) for expected, observed in pairs if expected is not None]
        non_null_correct = sum(expected == observed for expected, observed in non_null)
        predicted_non_null = sum(observed is not None for _, observed in pairs)
        per_field[field] = {
            "correct": correct,
            "n": len(pairs),
            "accuracy": correct / len(pairs) if pairs else None,
            "gold_non_null_n": len(non_null),
            "gold_non_null_correct": non_null_correct,
            "gold_non_null_accuracy": non_null_correct / len(non_null) if non_null else None,
            "predicted_non_null_n": predicted_non_null,
        }
        confusion = Counter(f"{expected} -> {observed}" for expected, observed in pairs)
        confusions[field] = dict(sorted(confusion.items(), key=lambda item: (-item[1], item[0])))
    all_cells = [value for field in per_field.values() for value in [field["accuracy"]] if value is not None]
    expression_cells = [per_field[field]["accuracy"] for field in EXPRESSION_FIELDS]
    return {
        "available_groups": available,
        "gold_groups": len(gold),
        "coverage": available / len(gold) if gold else None,
        "field_macro_accuracy": sum(all_cells) / len(all_cells) if all_cells else None,
        "expression_field_macro_accuracy": sum(expression_cells) / len(expression_cells) if expression_cells else None,
        "per_field": per_field,
        "confusions": confusions,
    }


def ordinal_value(field: str, value: Any) -> int | None:
    if value is None:
        return None
    if field == "expression_legibility_distribution":
        for level in range(4):
            if f"_{level}_" in str(value):
                return level
    if field == "smile_prevalence":
        return {"none": 0, "minority": 1, "about_half": 2, "majority": 3, "all": 4}.get(value)
    if field == "dominant_smile_intensity":
        return {"slight": 1, "clear": 2, "broad_or_laughter_like": 3}.get(value)
    return None


def relaxed_metrics(gold: dict[str, dict[str, Any]], predictions: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in ("expression_legibility_distribution", "smile_prevalence", "dominant_smile_intensity"):
        comparable = 0
        within_one = 0
        base_exact = 0
        for key, truth in gold.items():
            expected = ordinal_value(field, truth[field])
            observed = ordinal_value(field, (predictions.get(key) or {}).get(field))
            if expected is None or observed is None:
                continue
            comparable += 1
            base_exact += expected == observed
            within_one += abs(expected - observed) <= 1
        output[field] = {
            "ordinal_comparable_n": comparable,
            "base_level_accuracy": base_exact / comparable if comparable else None,
            "within_one_accuracy": within_one / comparable if comparable else None,
        }
    return output


def proposal_diagnostics(
    gold_rows: list[dict[str, Any]],
    direct: dict[str, dict[str, Any]],
    inventory: dict[str, dict[str, Any]],
    face_summary: dict[str, dict[str, Any]],
    face_rows: dict[str, dict[str, Any]],
    face_manifest: dict[str, Any],
) -> dict[str, Any]:
    accepted_by_group = Counter()
    for task in face_manifest.get("tasks", []):
        record = face_rows.get(task["face_task_id"])
        if record and record["model_annotation"].get("eligible_face_visible") is True:
            accepted_by_group[task["group_key"]] += 1
    methods: dict[str, dict[str, int | None]] = {
        "direct_estimate": {},
        "one_shot_inventory": {},
        "merged_full_plus_tiles_pre_cap": {},
        "spatially_selected_for_attributes": {},
        "accepted_face_attribute_targets": {},
    }
    for row in gold_rows:
        key = row["group_key"]
        direct_ann = (direct.get(key) or {}).get("model_annotation", {})
        inv_ann = (inventory.get(key) or {}).get("model_annotation", {})
        methods["direct_estimate"][key] = direct_ann.get("estimated_eligible_face_count")
        methods["one_shot_inventory"][key] = len(inv_ann.get("faces", [])) if key in inventory else None
        methods["merged_full_plus_tiles_pre_cap"][key] = (face_summary.get(key) or {}).get("merged_candidates")
        methods["spatially_selected_for_attributes"][key] = (face_summary.get(key) or {}).get("selected_face_tasks")
        methods["accepted_face_attribute_targets"][key] = accepted_by_group.get(key, 0)
    result: dict[str, Any] = {}
    for method, values in methods.items():
        observed = sorted(value for value in values.values() if isinstance(value, int))
        result[method] = {
            "groups_with_value": len(observed),
            "total": sum(observed),
            "minimum": min(observed) if observed else None,
            "median": observed[len(observed) // 2] if observed else None,
            "maximum": max(observed) if observed else None,
            "counts": values,
        }
    result["notes"] = {
        "human_group_face_count_gold_available": False,
        "advertisement_count_band_not_used_for_group_scoring": True,
        "inventory_faces_truncated_groups": sum(
            (row.get("model_annotation") or {}).get("faces_truncated") is True for row in inventory.values()
        ),
        "groups_truncated_by_12_face_cap": sum(
            (face_summary.get(row["group_key"]) or {}).get("truncated_by_face_cap") is True for row in gold_rows
        ),
    }
    return result


def ledger_metrics(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    stages: dict[str, Counter[str]] = defaultdict(Counter)
    prompt_tokens = completion_tokens = total_tokens = 0
    cost = 0.0
    cost_seen = False
    for row in rows:
        stage = str(row.get("stage"))
        stages[stage]["calls"] += 1
        stages[stage]["ok"] += row.get("ok") is True
        usage = row.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)
        if usage.get("cost") is not None:
            cost += float(usage["cost"])
            cost_seen = True
    return {
        "calls": len(rows),
        "successful_calls": sum(row.get("ok") is True for row in rows),
        "by_stage": {stage: dict(counter) for stage, counter in sorted(stages.items())},
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "reported_cost": cost if cost_seen else None,
        },
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Qwen group aggregation experiment: first 50 pages",
        "",
        f"Generated: {results['generated_at']}",
        "",
        "This is an oracle-geometry experiment: human people-area crop boundaries are supplied, but human group labels are isolated from all Qwen calls. It measures aggregate-label inference and face discovery inside a known group, not end-to-end group detection.",
        "",
        "## Exact agreement",
        "",
        "| Condition | Available groups | All-field macro | Expression-field macro |",
        "|---|---:|---:|---:|",
    ]
    for condition, metrics in results["conditions"].items():
        lines.append(
            f"| {condition} | {metrics['exact']['available_groups']}/{metrics['exact']['gold_groups']} | "
            f"{fmt(metrics['exact']['field_macro_accuracy'])} | {fmt(metrics['exact']['expression_field_macro_accuracy'])} |"
        )
    lines.extend(["", "### Per-field exact accuracy", "", "| Condition | " + " | ".join(FIELDS) + " |", "|---|" + "---:|" * len(FIELDS)])
    for condition, metrics in results["conditions"].items():
        values = [fmt(metrics["exact"]["per_field"][field]["accuracy"]) for field in FIELDS]
        lines.append(f"| {condition} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "### Exact accuracy where human gold is non-null",
            "",
            "This removes the easy route-equivalent null cases from gaze, smile prevalence, and especially smile intensity.",
            "",
            "| Condition | Dominant gaze | Smile prevalence | Smile intensity |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition, metrics in results["conditions"].items():
        per_field = metrics["exact"]["per_field"]
        lines.append(
            f"| {condition} | {fmt(per_field['dominant_gaze']['gold_non_null_accuracy'])} "
            f"| {fmt(per_field['smile_prevalence']['gold_non_null_accuracy'])} "
            f"| {fmt(per_field['dominant_smile_intensity']['gold_non_null_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## Face-proposal diagnostics",
            "",
            "The human count band is advertisement-level, not group-level, so it is not used as group face-count gold. These values are descriptive diagnostics only.",
            "",
            "| Method | Groups | Total | Min | Median | Max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method, metrics in results["face_proposal_diagnostics"].items():
        if method == "notes":
            continue
        lines.append(f"| {method} | {metrics['groups_with_value']} | {metrics['total']} | {fmt(metrics['minimum'])} | {fmt(metrics['median'])} | {fmt(metrics['maximum'])} |")
    lines.extend(
        [
            "",
            "## Reliability and call budget",
            "",
            f"Calls: {results['ledger']['calls']}; online schema-valid calls: {results['ledger']['successful_calls']}; final valid task outputs after retries/offline parser repair: {results['final_valid_task_outputs']['total']}; token total: {results['ledger']['usage']['total_tokens']}; reported cost: {fmt(results['ledger']['usage']['reported_cost'])}.",
            "",
            "## Interpretation cautions",
            "",
            "- Only 22 human group regions occur in the first 50 manifest pages. Estimates are descriptive, not a stable benchmark.",
            "- The first 25/second 25 split is reported as a stability check; no threshold was fitted on either subset.",
            "- Per-face assessment is capped at 12 spatially distributed faces per group. Proposal counts before and after that cap are reported descriptively.",
            "- Exact group labels are sensitive to category boundaries such as `all` versus `mostly`; relaxed ordinal metrics are included in the JSON output.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.evaluation_dir.mkdir(parents=True, exist_ok=True)
    gold_rows = read_json(args.gold)["groups"]
    gold = {row["group_key"]: normalize_group(row["gold"]) for row in gold_rows}

    direct_rows = latest_ok(args.output_dir / "direct_groups.jsonl")
    inventory_rows = latest_ok(args.output_dir / "individual_inventory.jsonl")
    face_rows = latest_ok(args.output_dir / "face_attributes.jsonl")
    reducer_rows = latest_ok(args.output_dir / "llm_reducers.jsonl")
    tally_rows = latest_ok(args.output_dir / "distribution_tallies.jsonl")
    face_manifest_path = args.output_dir / "face_tasks_manifest.json"
    face_manifest = read_json(face_manifest_path) if face_manifest_path.exists() else {"tasks": [], "group_summary": []}
    face_summary = {row["group_key"]: row for row in face_manifest.get("group_summary", [])}

    direct_predictions = {key: normalize_group(row["model_annotation"]) for key, row in direct_rows.items()}
    inventory_predictions: dict[str, dict[str, Any]] = {}
    tiled_predictions: dict[str, dict[str, Any]] = {}
    faces_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in face_manifest.get("tasks", []):
        record = face_rows.get(task["face_task_id"])
        if record and record["model_annotation"].get("eligible_face_visible") is True:
            faces_by_group[task["group_key"]].append(record["model_annotation"])
    for row in gold_rows:
        key = row["group_key"]
        inventory_record = inventory_rows.get(key)
        if inventory_record:
            annotation = inventory_record["model_annotation"]
            inventory_predictions[key] = deterministic_reduce(annotation.get("group_type"), annotation.get("faces") or [])
            tiled_predictions[key] = deterministic_reduce(annotation.get("group_type"), faces_by_group.get(key, []))
    reducer_predictions = {key: normalize_group(row["model_annotation"]) for key, row in reducer_rows.items()}
    tally_predictions = {key: deterministic_reduce_tally(row["model_annotation"]) for key, row in tally_rows.items()}

    prediction_sets = {
        "direct_group": direct_predictions,
        "one_shot_individuals_deterministic": inventory_predictions,
        "tiled_individuals_deterministic": tiled_predictions,
        "tiled_individuals_llm_reducer": reducer_predictions,
        "group_distribution_tally_deterministic_followup": tally_predictions,
    }
    conditions: dict[str, Any] = {}
    for name, predictions in prediction_sets.items():
        exact = exact_metrics(gold, predictions)
        conditions[name] = {"exact": exact, "relaxed": relaxed_metrics(gold, predictions)}
        for split_name, predicate in {
            "manifest_pages_1_25": lambda row: row["manifest_index"] < 25,
            "manifest_pages_26_50": lambda row: row["manifest_index"] >= 25,
        }.items():
            split_gold = {row["group_key"]: gold[row["group_key"]] for row in gold_rows if predicate(row)}
            conditions[name].setdefault("stability_splits", {})[split_name] = exact_metrics(split_gold, predictions)

    sensitivity = []
    tiled_sensitivity = []
    for mostly in (0.50, 0.60, 0.70):
        for all_threshold in (0.90, 1.00):
            predictions = {}
            tiled_variant = {}
            for row in gold_rows:
                key = row["group_key"]
                inventory_record = inventory_rows.get(key)
                if inventory_record:
                    ann = inventory_record["model_annotation"]
                    predictions[key] = deterministic_reduce(
                        ann.get("group_type"), ann.get("faces") or [], mostly_threshold=mostly, all_threshold=all_threshold
                    )
                    tiled_variant[key] = deterministic_reduce(
                        ann.get("group_type"), faces_by_group.get(key, []), mostly_threshold=mostly, all_threshold=all_threshold
                    )
            metrics = exact_metrics(gold, predictions)
            tiled_metrics = exact_metrics(gold, tiled_variant)
            sensitivity.append(
                {
                    "mostly_threshold": mostly,
                    "all_threshold": all_threshold,
                    "all_field_macro_accuracy": metrics["field_macro_accuracy"],
                    "expression_field_macro_accuracy": metrics["expression_field_macro_accuracy"],
                }
            )
            tiled_sensitivity.append(
                {
                    "mostly_threshold": mostly,
                    "all_threshold": all_threshold,
                    "all_field_macro_accuracy": tiled_metrics["field_macro_accuracy"],
                    "expression_field_macro_accuracy": tiled_metrics["expression_field_macro_accuracy"],
                }
            )

    results = {
        "schema_version": "qwen_group_aggregation_evaluation_v1",
        "generated_at": iso_now(),
        "scope": {"manifest_pages": 50, "gold_groups": len(gold), "oracle_group_geometry": True},
        "primary_deterministic_thresholds": PRIMARY_THRESHOLDS,
        "conditions": conditions,
        "threshold_sensitivity_one_shot_inventory": sensitivity,
        "threshold_sensitivity_tiled_individuals": tiled_sensitivity,
        "face_proposal_diagnostics": proposal_diagnostics(gold_rows, direct_rows, inventory_rows, face_summary, face_rows, face_manifest),
        "ledger": ledger_metrics(args.output_dir / "call_ledger.jsonl"),
        "final_valid_task_outputs": {
            "direct": len(direct_rows),
            "inventory": len(inventory_rows),
            "tiles": len(latest_ok(args.output_dir / "tile_localization.jsonl")),
            "faces": len(face_rows),
            "reducers": len(reducer_rows),
            "tally_followup": len(tally_rows),
            "total": len(direct_rows) + len(inventory_rows) + len(latest_ok(args.output_dir / "tile_localization.jsonl")) + len(face_rows) + len(reducer_rows) + len(tally_rows),
        },
    }
    (args.evaluation_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.evaluation_dir / "results.md").write_text(render_markdown(results), encoding="utf-8")

    csv_path = args.evaluation_dir / "per_group_predictions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        columns = ["group_key", "manifest_index", "advertisement_face_depiction_count_band", "condition", *FIELDS]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        row_lookup = {row["group_key"]: row for row in gold_rows}
        for key, truth in gold.items():
            base = {"group_key": key, "manifest_index": row_lookup[key]["manifest_index"], "advertisement_face_depiction_count_band": row_lookup[key]["advertisement_face_depiction_count_band"]}
            writer.writerow({**base, "condition": "human_gold", **truth})
            for condition, predictions in prediction_sets.items():
                writer.writerow({**base, "condition": condition, **(predictions.get(key) or {})})
    print(json.dumps({"results": str(args.evaluation_dir / "results.json"), "markdown": str(args.evaluation_dir / "results.md"), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()

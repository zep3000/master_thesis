#!/usr/bin/env python3
"""Gold-free assemblies of current geometry with group/FER/zero-gate variants."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from common import BASE, HERE, normalize_group, normalize_person, rows


ENTITY_FIELDS = {
    "person": ["depiction_type", "perceived_age", "perceived_gender_presentation", "face_expression_legibility", "face_orientation", "gaze_target", "gaze_target_person_unboxed", "mouth_covered", "mouth_covering", "smile_present", "smile_intensity", "confidence", "review_flags"],
    "group": ["group_type", "age_composition", "gender_presentation_composition", "expression_legibility_distribution", "dominant_gaze", "smile_prevalence", "dominant_smile_intensity", "confidence", "review_flags"],
}


def lookup(path: Path, kind: str, gated: bool | None = None) -> dict[str, dict]:
    result = {}
    for row in rows(path):
        if not row.get("ok"):
            continue
        if gated is None:
            ann = row.get("model_annotation") or {}
        elif kind == "person":
            ann = normalize_person(row.get("model_annotation_raw") or {}, row["task_key"], gated)
        else:
            ann = normalize_group(row.get("model_annotation_raw") or {}, row["task_key"], gated)
        result[row["task_key"]] = ann
    return result


def replace_entities(annotation: dict, image_id: str, people: dict[str, dict], groups: dict[str, dict]) -> dict:
    result = copy.deepcopy(annotation)
    for ad in result.get("advertisements") or []:
        for person in ad.get("people") or []:
            values = people.get(f"{image_id}::{person['person_id']}")
            if values:
                for field in ENTITY_FIELDS["person"]:
                    if field in values:
                        person[field] = values[field]
        for group in ad.get("groups") or []:
            values = groups.get(f"{image_id}::{group['group_id']}")
            if values:
                for field in ENTITY_FIELDS["group"]:
                    if field in values:
                        group[field] = values[field]
    return result


def apply_fer(annotation: dict, image_id: str, thresholds: dict[str, dict], moderate_only: bool) -> dict:
    result = copy.deepcopy(annotation)
    for ad in result.get("advertisements") or []:
        for person in ad.get("people") or []:
            threshold = thresholds.get(f"{image_id}::{person['person_id']}")
            if threshold and (not moderate_only or person.get("face_expression_legibility") == "2_moderate_legibility"):
                person["face_expression_legibility"] = threshold["face_expression_legibility"]
    return result


def write_jsonl(path: Path, values: list[dict]) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in values), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["difficult140", "stratified140"], required=True)
    args = parser.parse_args()
    cohort = args.cohort
    baseline_rows = [x for x in rows(BASE / "output" / cohort / "assembled" / "lean_s2c_direct_gaze.jsonl") if x.get("ok")]
    raw_path = BASE / "output" / cohort / "entities_s2_direct_gaze.jsonl"
    people_gated, people_raw = lookup(raw_path, "person", True), lookup(raw_path, "person", False)
    groups_gated, groups_raw = lookup(raw_path, "group", True), lookup(raw_path, "group", False)
    thresholds = lookup(HERE / "output" / "fer" / cohort / "thresholds.jsonl", "person", None)
    moderate_audit = lookup(HERE / "output" / "fer" / cohort / "moderate_audit.jsonl", "person", None)

    variants: dict[str, tuple[dict[str, dict], dict[str, dict], str | None]] = {
        "current_direct_gated": (people_gated, groups_gated, None),
        "current_direct_ungated": (people_raw, groups_raw, None),
    }
    for group_variant in ["multiscale", "multiscale_context"]:
        path = HERE / "output" / "groups" / cohort / f"{group_variant}.jsonl"
        if path.exists():
            variants[f"current_group_{group_variant}_gated"] = (people_gated, lookup(path, "group", True), None)
            variants[f"current_group_{group_variant}_ungated"] = (people_raw, lookup(path, "group", False), None)
    if thresholds:
        variants["current_threshold_universal_ungated"] = (people_raw, groups_raw, "universal")
        variants["current_threshold_moderate_only_ungated"] = (people_raw, groups_raw, "moderate")
    if moderate_audit:
        variants["current_moderate_audit_ungated"] = (people_raw, groups_raw, "audit")

    for route, (people, groups, fer_mode) in variants.items():
        output = []
        for base in baseline_rows:
            image_id = base["image_id"]
            annotation = replace_entities(base["annotation"], image_id, people, groups)
            if fer_mode:
                selected = moderate_audit if fer_mode == "audit" else thresholds
                annotation = apply_fer(annotation, image_id, selected, fer_mode in {"moderate", "audit"})
            output.append({"image_id": image_id, "filename": base.get("filename"), "cohort": cohort, "route": route, "ok": True, "annotation": annotation})
        write_jsonl(HERE / "output" / "assembled" / cohort / f"{route}.jsonl", output)
        print(route, len(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

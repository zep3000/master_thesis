#!/usr/bin/env python3
"""Create canonical-null and raw-preserving views from frozen selected outputs."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common import HERE, OUTPUT, baseline_rows, rows, stable_hash, write_jsonl


PERSON_RAW_FIELDS = [
    "depiction_type", "perceived_age", "perceived_gender_presentation", "face_expression_legibility",
    "face_orientation", "gaze_target", "gaze_target_person_unboxed", "mouth_covered", "mouth_covering",
    "smile_present", "smile_intensity", "confidence", "review_flags",
]
GROUP_RAW_FIELDS = [
    "group_type", "age_composition", "gender_presentation_composition", "expression_legibility_distribution",
    "dominant_gaze", "smile_prevalence", "dominant_smile_intensity", "confidence", "review_flags",
]


def business_map(cohort: str) -> dict[str, dict[str, Any]]:
    path = OUTPUT / "business" / f"{cohort}.jsonl"
    return {row["task_key"]: row["model_annotation"] for row in rows(path) if row.get("ok")}


def raw_record(image_id: str, ad: dict[str, Any], kind: str, entity: dict[str, Any]) -> dict[str, Any]:
    entity_id = entity["person_id" if kind == "person" else "group_id"]
    fields = PERSON_RAW_FIELDS if kind == "person" else GROUP_RAW_FIELDS
    values = {field: copy.deepcopy(entity.get(field)) for field in fields}
    return {
        "task_key": f"{image_id}::{entity_id}",
        "image_id": image_id,
        "advertisement_id": ad["advertisement_id"],
        "entity_type": kind,
        "entity_id": entity_id,
        "raw_normalized_values": values,
        "source": "final_refinement_280/current_direct_ungated",
    }


def log_change(log: list[dict[str, Any]], counts: Counter[str], key: str, field: str, old: Any, new: Any, rule: str) -> None:
    if old == new:
        return
    counts[rule] += 1
    log.append({"task_key": key, "field": field, "from": old, "to": new, "rule": rule})


def set_logged(entity: dict[str, Any], field: str, value: Any, key: str, rule: str, log: list[dict[str, Any]], counts: Counter[str]) -> None:
    old = copy.deepcopy(entity.get(field))
    entity[field] = value
    log_change(log, counts, key, field, old, value, rule)


def canonicalize(cohort: str, require_business: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    business = business_map(cohort)
    output: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    missing_business: list[str] = []
    for source in baseline_rows(cohort):
        image_id = source["image_id"]
        annotation = copy.deepcopy(source["annotation"])
        annotation["schema_version"] = "qwen_canonical_candidate_v1"
        for ad in annotation.get("advertisements") or []:
            ad_key = f"{image_id}::{ad['advertisement_id']}"
            classified = business.get(ad_key)
            if classified:
                for field in ["ad_category", "brand_or_advertiser"]:
                    set_logged(ad, field, classified.get(field), ad_key, f"warc_business_sidecar:{field}", log, counts)
            else:
                missing_business.append(ad_key)
            # Identity candidates are deliberately not confirmed identity labels.
            set_logged(ad, "duplicate_faces_present", None, ad_key, "unresolved_identity_is_null", log, counts)
            set_logged(ad, "unique_face_count", None, ad_key, "unresolved_identity_is_null", log, counts)
            ad.pop("ad_category_confidence", None)
            ad.pop("brand_confidence", None)
            for person in ad.get("people") or []:
                key = f"{image_id}::{person['person_id']}"
                raw.append(raw_record(image_id, ad, "person", person))
                set_logged(person, "duplicate_of_person_id", None, key, "all_depictions_remain_independent", log, counts)
                set_logged(person, "duplicate_person_ids", [], key, "all_depictions_remain_independent", log, counts)
                if person.get("face_expression_legibility") == "0_not_legible":
                    for field in ["gaze_target", "gaze_target_person_unboxed", "smile_present", "smile_intensity"]:
                        set_logged(person, field, None, key, "v1.17_zero_legibility_null", log, counts)
                elif person.get("smile_present") != "yes":
                    set_logged(person, "smile_intensity", None, key, "v1.17_no_smile_intensity_null", log, counts)
            for group in ad.get("groups") or []:
                key = f"{image_id}::{group['group_id']}"
                raw.append(raw_record(image_id, ad, "group", group))
                if group.get("expression_legibility_distribution") == "all_0_not_legible":
                    for field in ["dominant_gaze", "smile_prevalence", "dominant_smile_intensity"]:
                        set_logged(group, field, None, key, "v1.17_group_zero_legibility_null", log, counts)
                elif group.get("smile_prevalence") in {"none", "not_assessable", None}:
                    set_logged(group, "dominant_smile_intensity", None, key, "v1.17_group_no_smile_intensity_null", log, counts)
        output.append({
            "image_id": image_id,
            "filename": source.get("filename"),
            "cohort": cohort,
            "route": "canonical_candidate_v1",
            "ok": not (require_business and any(key.startswith(f"{image_id}::") for key in missing_business)),
            "source_annotation_hash": stable_hash(source["annotation"]),
            "annotation": annotation,
        })
    summary = {
        "cohort": cohort,
        "pages": len(output),
        "complete_pages": sum(bool(row["ok"]) for row in output),
        "raw_entity_records": len(raw),
        "normalization_events": len(log),
        "normalization_counts": dict(sorted(counts.items())),
        "missing_business_tasks": missing_business,
    }
    return output, raw, log, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["difficult140", "stratified140"], required=True)
    parser.add_argument("--allow-missing-business", action="store_true")
    args = parser.parse_args()
    out, raw, log, summary = canonicalize(args.cohort, not args.allow_missing_business)
    write_jsonl(OUTPUT / "assembled" / args.cohort / "canonical_candidate_v1.jsonl", out)
    write_jsonl(OUTPUT / "raw_values" / f"{args.cohort}.jsonl", raw)
    write_jsonl(OUTPUT / "normalization" / f"{args.cohort}.jsonl", log)
    path = OUTPUT / "normalization" / f"{args.cohort}_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

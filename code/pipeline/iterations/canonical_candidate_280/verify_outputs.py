#!/usr/bin/env python3
"""Gold-free integrity checks for promoted development artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import prompts
from common import HERE, OUTPUT, rows


def main() -> int:
    summary = {"cohorts": {}, "checks": []}
    assembler_source = (HERE / "assemble.py").read_text(encoding="utf-8")
    assert "duplicate_candidates" not in assembler_source
    summary["checks"].append("main assembler has no duplicate-candidate input path")
    production_sources = [HERE / name for name in ["assemble.py", "common.py", "prompts.py", "run_business.py", "run_duplicates.py"]]
    forbidden_geometry_inputs = ["TheEconomistHistoricalArchives-Faces", "detector_half_center", "detector_full_center"]
    assert not any(token in source.read_text(encoding="utf-8") for source in production_sources for token in forbidden_geometry_inputs)
    summary["checks"].append("production modules have no legacy Faces-Dataset geometry input")
    for cohort in ["difficult140", "stratified140"]:
        canonical_rows = [row for row in rows(OUTPUT / "assembled" / cohort / "canonical_candidate_v1.jsonl") if row.get("ok")]
        duplicate_rows = [row for row in rows(OUTPUT / "duplicate_candidates_promoted" / f"{cohort}.jsonl") if row.get("ok")]
        business_rows = [row for row in rows(OUTPUT / "business" / f"{cohort}.jsonl") if row.get("ok")]
        assert len(canonical_rows) == 140
        assert len({row["task_key"] for row in duplicate_rows}) == len(duplicate_rows)
        assert all(row["model_annotation"]["ad_category"] in prompts.WARC_CATEGORIES for row in business_rows)
        assert all(set(row["model_annotation"]) == {"task_key", "ad_category", "brand_or_advertiser"} for row in business_rows)
        for canonical in canonical_rows:
            for ad in canonical["annotation"].get("advertisements") or []:
                assert ad.get("duplicate_faces_present") is None and ad.get("unique_face_count") is None
                assert ad.get("ad_category") in prompts.WARC_CATEGORIES
                assert "ad_category_confidence" not in ad and "brand_confidence" not in ad
                for person in ad.get("people") or []:
                    assert person.get("duplicate_of_person_id") is None and person.get("duplicate_person_ids") == []
                    if person.get("face_expression_legibility") == "0_not_legible":
                        assert all(person.get(field) is None for field in ["gaze_target", "gaze_target_person_unboxed", "smile_present", "smile_intensity"])
                for group in ad.get("groups") or []:
                    if group.get("expression_legibility_distribution") == "all_0_not_legible":
                        assert all(group.get(field) is None for field in ["dominant_gaze", "smile_prevalence", "dominant_smile_intensity"])
        summary["cohorts"][cohort] = {"canonical_pages": 140, "business_ads": len(business_rows), "duplicate_sidecar_ads": len(duplicate_rows)}
    path = HERE / "evaluation" / "output_integrity.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

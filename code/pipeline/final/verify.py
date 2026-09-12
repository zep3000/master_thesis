"""Fail-fast integrity checks for the standalone final package and its outputs."""

from __future__ import annotations

import json
from pathlib import Path

import prompts


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
EXPECTED = {"difficult": 198, "stratified": 200}


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    inference_text = "\n".join((HERE / name).read_text(encoding="utf-8") for name in ["run.py", "prompts.py"])
    forbidden = ["annotation_results", "Faces Dataset", "face_boxes", "archived detector", "detector-center", "detector_center"]
    hits = [token for token in forbidden if token.lower() in inference_text.lower()]
    if hits:
        raise AssertionError(f"inference code contains forbidden gold/detector references: {hits}")
    summary = {"inference_gold_or_detector_references": 0, "cohorts": {}}
    total_pages = 0
    for cohort, expected in EXPECTED.items():
        manifest = json.loads((HERE / "data" / f"manifest_{cohort}.json").read_text(encoding="utf-8"))
        if len(manifest["images"]) != expected:
            raise AssertionError(f"{cohort}: manifest count")
        assembled = rows(OUTPUT / cohort / "assembled" / "final.jsonl")
        if assembled and len(assembled) != expected:
            raise AssertionError(f"{cohort}: expected {expected} assembled rows, found {len(assembled)}")
        complete = 0; ads = people = groups = invalid_categories = nonnull_identity = 0
        for row in assembled:
            complete += int(bool(row.get("ok")))
            annotation = row["annotation"]
            for ad in annotation.get("advertisements") or []:
                ads += 1
                category = ad.get("ad_category")
                unresolved = "ad_category_unresolved" in (ad.get("review_flags") or [])
                invalid_categories += int(category not in prompts.AD_CATEGORIES and not (category is None and unresolved))
                nonnull_identity += int(ad.get("duplicate_faces_present") is not None or ad.get("unique_face_count") is not None)
                for person in ad.get("people") or []:
                    people += 1
                    nonnull_identity += int(person.get("duplicate_of_person_id") is not None or bool(person.get("duplicate_person_ids")))
                groups += len(ad.get("groups") or [])
        if invalid_categories or nonnull_identity:
            raise AssertionError(f"{cohort}: invalid_categories={invalid_categories}, identity_leak={nonnull_identity}")
        raw = rows(OUTPUT / "raw_values" / f"{cohort}.jsonl")
        normalization = rows(OUTPUT / "normalization" / f"{cohort}.jsonl")
        duplicates = rows(OUTPUT / "duplicate_candidates" / f"{cohort}.jsonl")
        summary["cohorts"][cohort] = {
            "manifest_pages": expected, "assembled_pages": len(assembled), "complete_pages": complete,
            "advertisements": ads, "people": people, "groups": groups,
            "raw_entity_records": len(raw), "normalization_events": len(normalization),
            "isolated_duplicate_tasks": len(duplicates),
        }
        total_pages += len(assembled)
    ledger = rows(OUTPUT / "request_ledger.jsonl")
    summary["total_assembled_pages"] = total_pages
    summary["physical_attempts"] = len(ledger)
    summary["failed_attempts"] = sum(not row.get("ok") for row in ledger)
    unrecovered = 0
    for path in OUTPUT.glob("**/*.jsonl"):
        if path.name == "request_ledger.jsonl":
            continue
        task_status: dict[str, bool] = {}
        for row in rows(path):
            if "ok" not in row:
                continue
            key = str(row.get("task_key") or row.get("image_id") or len(task_status))
            task_status[key] = task_status.get(key, False) or bool(row.get("ok"))
        unrecovered += sum(not status for status in task_status.values())
    summary["unrecovered_tasks_in_result_files"] = unrecovered
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

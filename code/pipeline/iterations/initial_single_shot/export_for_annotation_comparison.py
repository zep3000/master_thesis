#!/usr/bin/env python3
"""Convert qwen_iteration output to annotation_comparison_app-compatible JSONL."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "qwen_iteration" / "output" / "first20_single_shot.jsonl"
DEFAULT_OUTPUT = ROOT / "openrouter_qwen" / "output" / "qwen_iteration_first20_single_shot_comparison.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="Export only locally valid records. Default exports all parsed comparable records for review.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = read_jsonl(args.input)
    converted = []
    for record in records:
        if args.valid_only and record.get("ok") is not True:
            continue
        annotation = record.get("model_annotation")
        if not is_comparable(annotation):
            continue
        converted.append(convert_record(record, annotation))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in converted),
        encoding="utf-8",
    )
    print(f"wrote {len(converted)} records -> {args.output}")
    return 0


def convert_record(record: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    converted_annotation = {
        "schema_version": "ad_face_annotation_v2_comparison_from_qwen_iteration",
        "status": "complete" if record.get("ok") is True else "review",
        "image": {
            "image_id": record.get("image_id"),
            "filename": record.get("filename"),
            "path": record.get("image_path"),
            "metadata": record.get("manifest_metadata") or {},
        },
        "page": annotation.get("page") or {},
        "advertisements_truncated": annotation.get("advertisements_truncated", False),
        "advertisements": [convert_ad(ad) for ad in annotation.get("advertisements", []) if isinstance(ad, dict)],
        "urgent_comments": annotation.get("urgent_comments") or [],
        "machine_annotation": {
            "model": record.get("request", {}).get("model", ""),
            "provider_tag": record.get("request", {}).get("provider_tag", ""),
            "prompt_version": record.get("prompt_version", ""),
            "processed_at": record.get("processed_at", ""),
            "source": "qwen_iteration",
        },
    }
    return {
        "image_id": record.get("image_id"),
        "filename": record.get("filename"),
        "image_file": record.get("filename"),
        "image_path": record.get("image_path"),
        "ok": record.get("ok") is True,
        "model": record.get("request", {}).get("model", ""),
        "processed_at": record.get("processed_at") or iso_now(),
        "task": record.get("prompt_version", ""),
        "validation_errors": record.get("validation_errors") or [],
        "normalization_actions": record.get("normalization_actions") or [],
        "source_record_schema": "qwen_iteration_comparison_jsonl_v1",
        "annotation": converted_annotation,
    }


def convert_ad(ad: dict[str, Any]) -> dict[str, Any]:
    ad_id = ad.get("ad_id") or ad.get("advertisement_id")
    return {
        **ad,
        "ad_id": ad_id,
        "advertisement_id": ad.get("advertisement_id") or ad_id,
        "bbox": normalize_box(ad.get("bbox") or ad.get("bbox_1000")),
        "people": [convert_person(person) for person in ad.get("people", []) if isinstance(person, dict)],
        "groups": [convert_group(group) for group in ad.get("groups", []) if isinstance(group, dict)],
    }


def convert_person(person: dict[str, Any]) -> dict[str, Any]:
    cause = person.get("mouth_covering")
    if cause is None:
        cause = person.get("mouth_covering_cause")
    other_text = person.get("mouth_covering_other_text")
    if other_text is None:
        other_text = person.get("mouth_covering_cause_other_text")
    return {
        **person,
        "face_bbox": normalize_box(person.get("face_bbox") or person.get("face_bbox_1000")),
        "mouth_covering": cause,
        "mouth_covering_other_text": other_text,
    }


def convert_group(group: dict[str, Any]) -> dict[str, Any]:
    return {
        **group,
        "bbox": normalize_box(group.get("bbox") or group.get("bbox_1000")),
    }


def normalize_box(box: Any) -> list[float] | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    values = [float(value) for value in box]
    if any(value > 1 for value in values):
        values = [value / 1000 for value in values]
    return [round(max(0.0, min(1.0, value)), 4) for value in values]


def is_comparable(annotation: Any) -> bool:
    return isinstance(annotation, dict) and isinstance(annotation.get("page"), dict) and isinstance(annotation.get("advertisements"), list)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())

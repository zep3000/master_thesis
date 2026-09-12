#!/usr/bin/env python3
"""Export grounding-only matrix results for the annotation comparison app."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "qwen_iteration" / "output" / "first10_grounding_matrix.json"
DEFAULT_MANIFEST = ROOT / "annotation_app_v2" / "data" / "test_collection_200_difficult_joined_hosted_manifest.json"
DEFAULT_OUTPUT = ROOT / "openrouter_qwen" / "output" / "qwen35_9b_venice_first10_fullpage_grounding.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provider", default="venice/fp8")
    parser.add_argument("--configuration", default="full_page")
    parser.add_argument("--variant", default="predictions")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = json.loads(args.input.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_by_id = {row["image_id"]: row for row in manifest["images"]}
    records = []
    for result in matrix.get("results", []):
        if result.get("ok") is not True:
            continue
        if result.get("provider_tag") != args.provider or result.get("configuration") != args.configuration:
            continue
        variant = (result.get("variants") or {}).get(args.variant) or {}
        boxes = [normalize_box(box) for box in variant.get("boxes", [])]
        boxes = [box for box in boxes if box]
        image_id = result["image_id"]
        image_row = manifest_by_id.get(image_id, {})
        records.append(convert_result(matrix, result, image_row, boxes, args.variant))

    requested_order = {image_id: index for index, image_id in enumerate(matrix.get("image_ids", []))}
    records.sort(key=lambda row: requested_order.get(row["image_id"], 10**9))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    expected = sum(
        result.get("provider_tag") == args.provider and result.get("configuration") == args.configuration
        for result in matrix.get("results", [])
    )
    print(f"wrote {len(records)} usable records from {expected} attempted pages -> {args.output}")
    return 0


def convert_result(
    matrix: dict[str, Any],
    result: dict[str, Any],
    image_row: dict[str, Any],
    boxes: list[list[float]],
    variant: str,
) -> dict[str, Any]:
    image_id = result["image_id"]
    filename = image_row.get("filename") or f"{image_id}.jpg"
    people = [
        {
            "person_id": f"venice_face_{index:03d}",
            "face_bbox": box,
            "annotation_role": "individual",
        }
        for index, box in enumerate(boxes, start=1)
    ]
    processed_at = matrix.get("updated_at") or iso_now()
    annotation = {
        "schema_version": "ad_face_annotation_v2_grounding_only_comparison",
        "status": "complete",
        "image": {
            "image_id": image_id,
            "filename": filename,
            "metadata": image_row.get("metadata") or {},
        },
        "page": {
            "grounding_only": True,
            "qualifying_ad_count": None,
        },
        "advertisements_truncated": False,
        "advertisements": [
            {
                "ad_id": "venice_grounding_faces",
                "advertisement_id": "venice_grounding_faces",
                "bbox": None,
                "bbox_source": "not_annotated_grounding_only",
                "people": people,
                "groups": [],
            }
        ],
        "urgent_comments": [],
        "machine_annotation": {
            "model": matrix.get("model", "qwen/qwen3.5-9b"),
            "provider_tag": result.get("provider_tag", ""),
            "prompt_version": "native_face_bbox_grounding_v1",
            "processed_at": processed_at,
            "source": "qwen_iteration_first10_grounding_matrix",
            "configuration": result.get("configuration", ""),
            "variant": variant,
            "grounding_only": True,
        },
    }
    return {
        "image_id": image_id,
        "filename": filename,
        "image_file": filename,
        "ok": True,
        "model": matrix.get("model", "qwen/qwen3.5-9b"),
        "processed_at": processed_at,
        "task": "native_face_bbox_grounding_v1",
        "validation_errors": [],
        "normalization_actions": ["wrapped_grounding_boxes_in_synthetic_ad_without_ad_bbox"],
        "source_record_schema": "qwen_grounding_comparison_jsonl_v1",
        "annotation": annotation,
    }


def normalize_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    box = [float(item) for item in value]
    if any(item > 1 for item in box):
        box = [item / 1000 for item in box]
    x1, y1, x2, y2 = [round(max(0.0, min(1.0, item)), 6) for item in box]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())

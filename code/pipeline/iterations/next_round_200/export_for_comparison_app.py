#!/usr/bin/env python3
"""Export next-round routes for visual review in annotation_comparison_app.

This is a format-only export. It never reads human annotations or gold labels.
By default it publishes:

* R1: recommended production route (P2 structure + focused FER specialist)
* R4: maximum-spatial-score comparator (detector-assisted additions)

It also writes one standalone manifest containing the difficult100 followed by
the stratified100 cohort, so all 200 pages can be browsed in a single app view.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_NAME = "scale_v1"
DEFAULT_OUTPUT_DIR = ROOT / "openrouter_qwen" / "output"
DEFAULT_MANIFEST_OUTPUT = (
    ROOT / "annotation_app_v2" / "data" / "qwen_next_round_200_visual_comparison_manifest.json"
)
DEFAULT_STRATIFIED_MANIFEST_OUTPUT = (
    ROOT / "annotation_app_v2" / "data" / "qwen_next_round_stratified100_visual_comparison_manifest.json"
)

COHORTS = {
    "difficult100": {
        "manifest": EXPERIMENT_DIR / "data" / "manifest_difficult100.json",
        "image_dir": ROOT / "code" / "test_collection_200_difficult_joined_pages",
    },
    "stratified100": {
        "manifest": EXPERIMENT_DIR / "data" / "manifest_stratified100.json",
        "image_dir": ROOT / "master_thesis" / "data" / "images" / "full_pages_1940_2007_joined",
    },
}

ROUTES = {
    "r1": {
        "output_name": "qwen_next_round_200_r1_recommended.jsonl",
        "label": "R1 recommended: P2 structure + focused FER",
        "description": "Recommended robust production route.",
    },
    "r4": {
        "output_name": "qwen_next_round_200_r4_max_spatial.jsonl",
        "label": "R4 comparator: detector-assisted maximum spatial score",
        "description": "Higher pooled spatial score, retained as a visual comparator.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--routes", nargs="+", choices=sorted(ROUTES), default=["r1", "r4"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    parser.add_argument(
        "--stratified-manifest-output",
        type=Path,
        default=DEFAULT_STRATIFIED_MANIFEST_OUTPUT,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_items = load_manifest_items()
    exported_at = iso_now()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for route in args.routes:
        records = export_route(route, args.run_name, manifest_items, exported_at)
        output_path = args.output_dir / ROUTES[route]["output_name"]
        write_jsonl(output_path, records)
        print(f"wrote {len(records)} records -> {output_path}")

    combined_manifest = {
        "schema_version": "qwen_next_round_visual_comparison_manifest_v1",
        "task_id": "qwen_next_round_200_visual_comparison",
        "name": "Qwen next round: difficult100 + stratified100",
        "description": (
            "Neutral 200-page visual-review manifest. Contains no annotations or gold labels; "
            "difficult100 pages precede stratified100 pages."
        ),
        "created_at": exported_at,
        "images": [item["manifest_item"] for item in manifest_items],
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(combined_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(manifest_items)} images -> {args.manifest_output}")

    stratified_items = [
        item["manifest_item"] for item in manifest_items if item["cohort"] == "stratified100"
    ]
    stratified_manifest = {
        "schema_version": "qwen_next_round_visual_comparison_manifest_v1",
        "task_id": "qwen_next_round_stratified100_visual_comparison",
        "name": "Qwen next round: stratified100",
        "description": (
            "Neutral visual-review manifest for the first 100 pages of the freshly "
            "human-annotated stratified sample. Contains no annotations or gold labels."
        ),
        "created_at": exported_at,
        "images": stratified_items,
    }
    args.stratified_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.stratified_manifest_output.write_text(
        json.dumps(stratified_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(stratified_items)} images -> {args.stratified_manifest_output}")
    return 0


def load_manifest_items() -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cohort, config in COHORTS.items():
        payload = json.loads(config["manifest"].read_text(encoding="utf-8"))
        for cohort_index, raw_item in enumerate(payload.get("images", [])):
            image_id = str(raw_item.get("image_id") or Path(raw_item["filename"]).stem)
            filename = str(raw_item.get("filename") or f"{image_id}.jpg")
            image_path = config["image_dir"] / filename
            if image_id in seen:
                raise ValueError(f"duplicate image_id across cohorts: {image_id}")
            if not image_path.is_file():
                raise FileNotFoundError(f"missing image: {image_path}")
            seen.add(image_id)
            metadata = copy.deepcopy(raw_item.get("metadata") or {})
            metadata.update({"cohort": cohort, "cohort_index": cohort_index})
            combined.append(
                {
                    "cohort": cohort,
                    "cohort_index": cohort_index,
                    "image_id": image_id,
                    "filename": filename,
                    "image_path": str(image_path.resolve()),
                    "metadata": metadata,
                    "manifest_item": {
                        "image_id": image_id,
                        "filename": filename,
                        "path": str(image_path.resolve()),
                        "metadata": metadata,
                    },
                }
            )
    if len(combined) != 200:
        raise ValueError(f"expected 200 manifest images, found {len(combined)}")
    return combined


def export_route(
    route: str,
    run_name: str,
    manifest_items: list[dict[str, Any]],
    exported_at: str,
) -> list[dict[str, Any]]:
    source_by_image: dict[str, dict[str, Any]] = {}
    for cohort in COHORTS:
        source_path = EXPERIMENT_DIR / "output" / run_name / cohort / "assembled" / f"{route}.jsonl"
        for record in read_jsonl(source_path):
            image_id = str(record.get("image_id") or Path(record.get("filename", "")).stem)
            if image_id in source_by_image:
                raise ValueError(f"duplicate {route} record: {image_id}")
            if not is_comparable(record.get("annotation")):
                raise ValueError(f"non-comparable {route} annotation: {image_id}")
            source_by_image[image_id] = {**record, "source_path": str(source_path.resolve())}

    records: list[dict[str, Any]] = []
    for item in manifest_items:
        image_id = item["image_id"]
        if image_id not in source_by_image:
            raise ValueError(f"missing {route} record for {image_id}")
        records.append(convert_record(source_by_image[image_id], item, route, run_name, exported_at))
    extras = set(source_by_image) - {item["image_id"] for item in manifest_items}
    if extras:
        raise ValueError(f"unexpected {route} records: {sorted(extras)[:5]}")
    return records


def convert_record(
    source: dict[str, Any],
    item: dict[str, Any],
    route: str,
    run_name: str,
    exported_at: str,
) -> dict[str, Any]:
    original = source["annotation"]
    annotation = {
        **copy.deepcopy(original),
        "schema_version": "ad_face_annotation_v2_comparison_qwen_next_round_200",
        "status": "complete" if source.get("ok") is True else "review",
        "image": {
            "image_id": item["image_id"],
            "filename": item["filename"],
            "path": item["image_path"],
            "metadata": item["metadata"],
        },
        "advertisements_truncated": original.get("advertisements_truncated", False),
        "advertisements": [
            convert_ad(ad) for ad in original.get("advertisements", []) if isinstance(ad, dict)
        ],
        "machine_annotation": {
            "model": "qwen/qwen3.5-9b",
            "provider_tag": "venice/fp8",
            "route": route,
            "route_label": ROUTES[route]["label"],
            "experiment_run": run_name,
            "processed_at": exported_at,
            "source": "qwen_iteration/next_round_200",
            "source_file": source["source_path"],
            "format_only_export": True,
        },
    }
    return {
        "image_id": item["image_id"],
        "filename": item["filename"],
        "image_file": item["filename"],
        "image_path": item["image_path"],
        "cohort": item["cohort"],
        "cohort_index": item["cohort_index"],
        "ok": source.get("ok") is True,
        "model": "qwen/qwen3.5-9b",
        "processed_at": exported_at,
        "task": ROUTES[route]["label"],
        "route": route,
        "experiment_run": run_name,
        "validation_errors": source.get("validation_errors") or [],
        "normalization_actions": [
            "Added comparison-app bbox/face_bbox aliases normalized to [0,1].",
            "Added image, route, and provenance metadata; annotation values were not changed.",
        ],
        "source_record_schema": "qwen_next_round_200_comparison_jsonl_v1",
        "annotation": annotation,
    }


def convert_ad(ad: dict[str, Any]) -> dict[str, Any]:
    ad_id = ad.get("ad_id") or ad.get("advertisement_id")
    return {
        **copy.deepcopy(ad),
        "ad_id": ad_id,
        "advertisement_id": ad.get("advertisement_id") or ad_id,
        "bbox": normalize_box(ad.get("bbox") or ad.get("bbox_1000")),
        "people": [convert_person(person) for person in ad.get("people", []) if isinstance(person, dict)],
        "groups": [convert_group(group) for group in ad.get("groups", []) if isinstance(group, dict)],
    }


def convert_person(person: dict[str, Any]) -> dict[str, Any]:
    return {
        **copy.deepcopy(person),
        "face_bbox": normalize_box(person.get("face_bbox") or person.get("face_bbox_1000")),
    }


def convert_group(group: dict[str, Any]) -> dict[str, Any]:
    return {
        **copy.deepcopy(group),
        "bbox": normalize_box(group.get("bbox") or group.get("bbox_1000")),
    }


def normalize_box(box: Any) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    values = [float(value) for value in box]
    if any(value > 1 for value in values):
        values = [value / 1000 for value in values]
    return [round(max(0.0, min(1.0, value)), 4) for value in values]


def is_comparable(annotation: Any) -> bool:
    return (
        isinstance(annotation, dict)
        and isinstance(annotation.get("page"), dict)
        and isinstance(annotation.get("advertisements"), list)
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())

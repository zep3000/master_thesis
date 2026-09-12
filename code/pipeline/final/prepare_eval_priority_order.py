#!/usr/bin/env python3
"""Prepend the unprocessed Main Evaluation Set to an existing processing order.

The already-run prefix is retained byte-for-byte as an image-id sequence. Main
Evaluation Set members outside that prefix are then promoted in their original
evaluation-manifest order. All other images retain their relative order from
the supplied continuation. This changes scheduling only and is deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "output" / "production" / "full_pages_joined_v1" / "manifest.json"
DEFAULT_SOURCE_ORDER = HERE / "data" / "full_pages_decade_balanced_after_2000_seed20260819.json"
DEFAULT_EVALUATION_EXPORT = (
    HERE.parent.parent
    / "annotation_results"
    / "karin_economist_decade_face_count_stratified_600_v3_60109424_2026-08-10.json"
)
DEFAULT_OUTPUT = HERE / "data" / "full_pages_main_evaluation_600_priority_after_11000.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_manifest_hash(manifest: dict) -> str:
    identity = [
        (row["image_id"], row["filename"], row.get("year"), row.get("source_size_bytes"))
        for row in manifest["images"]
    ]
    encoded = json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def distribution(ids: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(image_id[:3] + "0" for image_id in ids).items()))


def build(manifest: dict, source_order: dict, evaluation_export: dict, preserve_prefix: int) -> dict:
    manifest_ids = [row["image_id"] for row in manifest["images"]]
    manifest_set = set(manifest_ids)
    source_ids = list(source_order["image_ids"])
    evaluation_manifest = evaluation_export["annotation_set"]["manifest"]
    evaluation_ids = [row["image_id"] for row in evaluation_manifest["images"]]

    if preserve_prefix < 0 or preserve_prefix > len(source_ids):
        raise ValueError("preserved prefix is outside the source processing order")
    if len(source_ids) != len(manifest_ids) or set(source_ids) != manifest_set:
        raise ValueError("source processing order is not a complete permutation of the source manifest")
    if len(evaluation_ids) != 600 or len(set(evaluation_ids)) != 600:
        raise ValueError("Main Evaluation Set must contain exactly 600 unique image IDs")
    missing = [image_id for image_id in evaluation_ids if image_id not in manifest_set]
    if missing:
        raise ValueError(f"evaluation IDs are absent from the source manifest: {missing[:10]}")

    prefix = source_ids[:preserve_prefix]
    prefix_set = set(prefix)
    promoted = [image_id for image_id in evaluation_ids if image_id not in prefix_set]
    promoted_set = set(promoted)
    continuation = [image_id for image_id in source_ids[preserve_prefix:] if image_id not in promoted_set]
    ordered = prefix + promoted + continuation

    if len(ordered) != len(manifest_ids) or len(set(ordered)) != len(manifest_ids):
        raise AssertionError("generated processing order is not a complete, duplicate-free permutation")
    if ordered[:preserve_prefix] != prefix:
        raise AssertionError("generated processing order changed the preserved prefix")
    if ordered[preserve_prefix : preserve_prefix + len(promoted)] != promoted:
        raise AssertionError("generated processing order did not preserve evaluation-manifest order")

    source_bytes = json.dumps(source_order, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    evaluation_identity = {
        "task_id": evaluation_manifest.get("task_id"),
        "image_ids": evaluation_ids,
    }
    evaluation_sha = sha256_bytes(
        json.dumps(evaluation_identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    priority_end = preserve_prefix + len(promoted)
    return {
        "schema_version": "qwen_processing_order_v1",
        "order_id": f"main_evaluation_600_priority_after_{preserve_prefix}",
        "strategy": "preserve_run_prefix_then_main_evaluation_manifest_order_then_existing_decade_balanced_order",
        "preserved_prefix_count": preserve_prefix,
        "source_manifest_sha256": stable_manifest_hash(manifest),
        "source_processing_order": {
            "order_id": source_order.get("order_id"),
            "sha256": sha256_bytes(source_bytes),
        },
        "priority_set": {
            "name": evaluation_export["annotation_set"].get("name"),
            "task_id": evaluation_manifest.get("task_id"),
            "manifest_identity_sha256": evaluation_sha,
            "total_images": len(evaluation_ids),
            "already_in_preserved_prefix": len(evaluation_ids) - len(promoted),
            "promoted_after_prefix": len(promoted),
            "priority_start": preserve_prefix,
            "priority_end_exclusive": priority_end,
        },
        "image_count": len(ordered),
        "decade_counts_all": distribution(ordered),
        "prefix_diagnostics": {
            str(preserve_prefix): distribution(ordered[:preserve_prefix]),
            str(priority_end): distribution(ordered[:priority_end]),
            str(min(len(ordered), preserve_prefix + 10_000)): distribution(
                ordered[: min(len(ordered), preserve_prefix + 10_000)]
            ),
            str(len(ordered)): distribution(ordered),
        },
        "image_ids": ordered,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-order", type=Path, default=DEFAULT_SOURCE_ORDER)
    parser.add_argument("--evaluation-export", type=Path, default=DEFAULT_EVALUATION_EXPORT)
    parser.add_argument("--preserve-prefix", type=int, default=11_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_order = json.loads(args.source_order.read_text(encoding="utf-8"))
    evaluation_export = json.loads(args.evaluation_export.read_text(encoding="utf-8"))
    output = build(manifest, source_order, evaluation_export, args.preserve_prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ["order_id", "priority_set", "image_count", "prefix_diagnostics"]}, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

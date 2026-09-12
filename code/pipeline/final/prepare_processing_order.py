#!/usr/bin/env python3
"""Build a deterministic decade-balanced processing order from a neutral manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "output" / "production" / "full_pages_joined_v1" / "manifest.json"
DEFAULT_OUTPUT = HERE / "data" / "full_pages_decade_balanced_after_2000_seed20260819.json"


def decade(row: dict) -> int:
    year = int(row["year"])
    return year // 10 * 10


def stable_manifest_hash(manifest: dict) -> str:
    identity = [(row["image_id"], row["filename"], row.get("year"), row.get("source_size_bytes")) for row in manifest["images"]]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def build(manifest: dict, preserve_prefix: int, seed: int) -> dict:
    images = manifest["images"]
    if preserve_prefix < 0 or preserve_prefix > len(images):
        raise ValueError("preserved prefix is outside the manifest")
    prefix = list(images[:preserve_prefix])
    remaining: dict[int, list[dict]] = {}
    for row in images[preserve_prefix:]:
        remaining.setdefault(decade(row), []).append(row)
    for value, rows in remaining.items():
        random.Random(seed + value).shuffle(rows)

    counts = Counter(decade(row) for row in prefix)
    continuation = []
    while remaining:
        available = [value for value, rows in remaining.items() if rows]
        if not available:
            break
        selected = min(available, key=lambda value: (counts[value], value))
        continuation.append(remaining[selected].pop())
        counts[selected] += 1
        if not remaining[selected]:
            del remaining[selected]
    ordered = prefix + continuation
    if len(ordered) != len(images) or len({row["image_id"] for row in ordered}) != len(images):
        raise AssertionError("processing order is not a complete permutation")

    def distribution(rows):
        return {str(key): value for key, value in sorted(Counter(decade(row) for row in rows).items())}

    checkpoints = sorted({preserve_prefix, min(len(ordered), preserve_prefix + 1000), min(len(ordered), preserve_prefix + 5000), len(ordered)})
    return {
        "schema_version": "qwen_processing_order_v1",
        "order_id": f"decade_deficit_balanced_after_{preserve_prefix}_seed{seed}",
        "strategy": "preserve_completed_prefix_then_randomized_within_decade_deficit_balancing",
        "seed": seed,
        "preserved_prefix_count": preserve_prefix,
        "source_manifest_sha256": stable_manifest_hash(manifest),
        "image_count": len(ordered),
        "decade_counts_all": distribution(ordered),
        "prefix_diagnostics": {
            str(stop): distribution(ordered[:stop])
            for stop in checkpoints
        },
        "image_ids": [row["image_id"] for row in ordered],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preserve-prefix", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = build(manifest, args.preserve_prefix, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ["order_id", "image_count", "decade_counts_all", "prefix_diagnostics"]}, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

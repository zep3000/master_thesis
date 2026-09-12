#!/usr/bin/env python3
"""Create the frozen first-30 manifest without annotation or prior-model fields."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
SOURCE = ROOT / "annotation_app_v2" / "data" / "test_collection_200_difficult_joined_hosted_manifest.json"
OUTPUT = SCRIPT_DIR / "input_manifest_first30.json"


def main() -> int:
    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes)
    images = []
    for index, item in enumerate(source["images"][:30]):
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        images.append(
            {
                "image_id": item["image_id"],
                "filename": item["filename"],
                "page_type": item.get("page_type") or "unknown",
                "metadata": {
                    key: metadata[key]
                    for key in ("year", "decade", "issue")
                    if key in metadata
                },
                "source_manifest_index": index,
            }
        )
    payload = {
        "schema_version": "qwen_venice_first30_four_pass_input_manifest_v1",
        "task_id": "venice_first30_four_pass",
        "selection": "first_30_in_source_order",
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "excluded_metadata": "All annotation, disagreement, and prior-model fields are excluded.",
        "images": images,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(images)} sanitized rows -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

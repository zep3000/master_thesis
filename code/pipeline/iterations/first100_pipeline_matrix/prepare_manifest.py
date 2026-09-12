"""Create a gold-free manifest for the first 100 difficult pages."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "first100_pipeline_matrix"
SOURCE = ROOT / "annotation_app_v2" / "data" / "test_collection_200_difficult_joined_hosted_manifest.json"
OUTPUT = HERE / "data" / "input_manifest_first100.json"


def main() -> None:
    raw = SOURCE.read_bytes()
    source = json.loads(raw)
    images = []
    for index, item in enumerate(source["images"][:100]):
        source_metadata = item.get("metadata") or {}
        images.append(
            {
                "manifest_index": index,
                "image_id": item.get("image_id") or Path(item["filename"]).stem,
                "filename": item["filename"],
                "page_type": item.get("page_type", "unknown"),
                # Selection diagnostics include earlier model judgments and must
                # not be available to inference, even if current prompts happen
                # not to interpolate them.
                "metadata": {key: source_metadata.get(key) for key in ["year", "decade", "issue"] if source_metadata.get(key) is not None},
            }
        )
    payload = {
        "schema_version": "qwen_first100_pipeline_matrix_input_v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "selection": "first_100_in_hosted_manifest_order",
        "source_manifest": str(SOURCE),
        "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "gold_or_prior_model_annotation_fields_present": False,
        "images": images,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "images": len(images)}, indent=2))


if __name__ == "__main__":
    main()

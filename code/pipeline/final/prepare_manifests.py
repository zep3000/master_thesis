"""Create label-free inference manifests from the two archived test manifests.

This script reads image identifiers and ordering only. Annotation payloads are never
copied into the inference package, which keeps the gold/evaluation boundary explicit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
HUMAN_GOLD = (
    REPOSITORY_ROOT / "artifacts" / "human-validation" / "raw" / "pipeline-development"
)

SOURCES = {
    "difficult": {
        "gold": HUMAN_GOLD / "difficult-198.json",
        "image_dir": Path(
            os.environ.get(
                "PIPELINE_DIFFICULT_IMAGE_DIR",
                REPOSITORY_ROOT / "data" / "images" / "pipeline-development-difficult",
            )
        ),
        "expected": 198,
    },
    "stratified": {
        "gold": HUMAN_GOLD / "stratified-200.json",
        "image_dir": Path(
            os.environ.get(
                "PIPELINE_IMAGE_DIR",
                REPOSITORY_ROOT / "data" / "images" / "full_pages_1940_2007_joined",
            )
        ),
        "expected": 200,
    },
}


def main() -> int:
    data_dir = HERE / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cohort, spec in SOURCES.items():
        source = json.loads(spec["gold"].read_text(encoding="utf-8"))
        source_images = source["annotation_set"]["manifest"]["images"]
        images = []
        missing = []
        for index, item in enumerate(source_images):
            filename = item["filename"]
            image_path = spec["image_dir"] / filename
            if not image_path.exists():
                missing.append(str(image_path))
            metadata = item.get("metadata") or {}
            images.append(
                {
                    "index": index,
                    "image_id": item["image_id"],
                    "filename": filename,
                    "cohort": cohort,
                    "year": metadata.get("year"),
                }
            )
        if len(images) != spec["expected"]:
            raise RuntimeError(f"{cohort}: expected {spec['expected']} images, found {len(images)}")
        if missing:
            raise FileNotFoundError(f"{cohort}: {len(missing)} missing image files; first={missing[0]}")
        manifest = {
            "schema_version": "qwen_final_label_free_manifest_v1",
            "cohort": cohort,
            "image_dir": str(spec["image_dir"].resolve()),
            "images": images,
            "splits": {
                "development140": [0, 140],
                "reserve": [140, len(images)],
                "first60": [0, 60],
                "historical_next40": [60, 100],
                "reserve_first40": [140, min(180, len(images))],
            },
        }
        target = data_dir / f"manifest_{cohort}.json"
        target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        summary[cohort] = {"path": str(target), "images": len(images), "missing": 0}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

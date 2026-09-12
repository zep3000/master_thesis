"""Prepare oracle-boundary group crops for the first 50 manifest pages.

The inference manifest contains source identity, crop geometry, and image paths,
but no human group labels.  Gold labels are written to a separate evaluation
file that the inference runner does not open.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "group_aggregation_experiment"
SOURCE_MANIFEST = ROOT / "annotation_app_v2" / "data" / "test_collection_200_difficult_joined_hosted_manifest.json"
HUMAN_EXPORT = ROOT / "annotation_results" / "test_collection_200_difficult_joined_v1_2026-08-02.json"
IMAGE_DIR = ROOT / "code" / "test_collection_200_difficult_joined_pages"
ASSIGNMENT_CODE = "79201188"
INPUT_MANIFEST = HERE / "data" / "input_manifest_first50_groups.json"
GOLD_PATH = HERE / "evaluation" / "gold_groups_first50.json"
GROUP_CROP_DIR = HERE / "crops" / "groups"
TILE_DIR = HERE / "crops" / "tiles"

GROUP_FIELDS = [
    "group_type",
    "age_composition",
    "gender_presentation_composition",
    "expression_legibility_distribution",
    "dominant_gaze",
    "smile_prevalence",
    "dominant_smile_intensity",
]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_id_from_item(item: dict[str, Any]) -> str:
    return str(item.get("image_id") or Path(str(item["filename"])).stem)


def normalized_box(entity: dict[str, Any]) -> list[float] | None:
    box = entity.get("bbox")
    if isinstance(box, list) and len(box) == 4:
        return [float(value) for value in box]
    box = entity.get("bbox_1000")
    if isinstance(box, list) and len(box) == 4:
        return [float(value) / 1000.0 for value in box]
    return None


def pixel_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(round(box[0] * width))))
    y1 = max(0, min(height - 1, int(round(box[1] * height))))
    x2 = max(x1 + 1, min(width, int(round(box[2] * width))))
    y2 = max(y1 + 1, min(height, int(round(box[3] * height))))
    return x1, y1, x2, y2


def tile_boxes(width: int, height: int) -> list[tuple[str, tuple[int, int, int, int]]]:
    # Two-by-two tiles with 20% overlap around the center seams.
    x_ranges = [(0, int(round(width * 0.60))), (int(round(width * 0.40)), width)]
    y_ranges = [(0, int(round(height * 0.60))), (int(round(height * 0.40)), height)]
    output = []
    for row, (top, bottom) in enumerate(y_ranges, start=1):
        for col, (left, right) in enumerate(x_ranges, start=1):
            output.append((f"r{row}c{col}", (left, top, right, bottom)))
    return output


def ensure_minimum_side(image: Image.Image, minimum: int = 64) -> Image.Image:
    """Upscale exceptionally tiny tiles to the provider's accepted image size."""
    shortest = min(image.width, image.height)
    if shortest >= minimum:
        return image
    scale = minimum / shortest
    size = (max(minimum, round(image.width * scale)), max(minimum, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def main() -> None:
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    first50 = list(source_manifest["images"][:50])
    human = json.loads(HUMAN_EXPORT.read_text(encoding="utf-8"))
    human_by_image = {
        row["image_id"]: row["payload"]
        for row in human["annotations"]
        if str(row.get("assignment_code")) == ASSIGNMENT_CODE
    }

    GROUP_CROP_DIR.mkdir(parents=True, exist_ok=True)
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)

    input_groups: list[dict[str, Any]] = []
    gold_groups: list[dict[str, Any]] = []
    for manifest_index, item in enumerate(first50):
        image_id = image_id_from_item(item)
        payload = human_by_image.get(image_id)
        if not isinstance(payload, dict):
            continue
        image_path = IMAGE_DIR / str(item["filename"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            for ad_index, ad in enumerate(payload.get("advertisements") or [], start=1):
                for group_index, group in enumerate(ad.get("groups") or [], start=1):
                    box = normalized_box(group)
                    if box is None:
                        continue
                    group_id = str(group.get("group_id") or f"ad{ad_index}_g{group_index}")
                    group_key = f"{image_id}::{group_id}"
                    page_crop_box = pixel_box(box, rgb.width, rgb.height)
                    group_crop = rgb.crop(page_crop_box)
                    crop_path = GROUP_CROP_DIR / f"{safe(group_key)}.jpg"
                    group_crop.save(crop_path, format="JPEG", quality=95, optimize=True)
                    tiles = []
                    for tile_id, tile_box in tile_boxes(group_crop.width, group_crop.height):
                        tile_path = TILE_DIR / f"{safe(group_key)}__{tile_id}.jpg"
                        tile_image = ensure_minimum_side(group_crop.crop(tile_box))
                        tile_image.save(tile_path, format="JPEG", quality=95, optimize=True)
                        tiles.append(
                            {
                                "tile_id": tile_id,
                                "tile_bbox_pixels_in_group_crop": list(tile_box),
                                "tile_encoded_width": tile_image.width,
                                "tile_encoded_height": tile_image.height,
                                "tile_upscaled_for_provider": tile_image.size != (tile_box[2] - tile_box[0], tile_box[3] - tile_box[1]),
                                "tile_path": str(tile_path),
                                "tile_sha256": sha256(tile_path),
                            }
                        )
                    input_groups.append(
                        {
                            "group_key": group_key,
                            "image_id": image_id,
                            "filename": str(item["filename"]),
                            "manifest_index": manifest_index,
                            "source_image_path": str(image_path),
                            "source_image_sha256": sha256(image_path),
                            "ad_ordinal": ad_index,
                            "group_ordinal": group_index,
                            "oracle_group_bbox_normalized": box,
                            "group_crop_bbox_pixels_on_page": list(page_crop_box),
                            "group_crop_path": str(crop_path),
                            "group_crop_sha256": sha256(crop_path),
                            "group_crop_width": group_crop.width,
                            "group_crop_height": group_crop.height,
                            "tiles": tiles,
                        }
                    )
                    gold_groups.append(
                        {
                            "group_key": group_key,
                            "image_id": image_id,
                            "manifest_index": manifest_index,
                            "advertisement_face_depiction_count_band": ad.get("face_depiction_count_band"),
                            "gold": {field: group.get(field) for field in GROUP_FIELDS},
                        }
                    )

    input_payload = {
        "schema_version": "qwen_group_aggregation_input_manifest_v1",
        "generated_at": iso_now(),
        "scope": {
            "manifest": str(SOURCE_MANIFEST),
            "manifest_indices": [0, 49],
            "pages": 50,
            "groups": len(input_groups),
            "assignment_code_used_only_by_preparation": ASSIGNMENT_CODE,
        },
        "data_isolation": {
            "oracle_human_group_geometry_used": True,
            "human_group_labels_in_this_file": False,
            "gold_label_file_read_by_inference_runner": False,
            "purpose": "isolate aggregate-label inference from group detection and matching",
        },
        "groups": input_groups,
    }
    gold_payload = {
        "schema_version": "qwen_group_aggregation_gold_v1",
        "generated_at": iso_now(),
        "human_export": str(HUMAN_EXPORT),
        "assignment_code": ASSIGNMENT_CODE,
        "groups": gold_groups,
    }
    INPUT_MANIFEST.write_text(json.dumps(input_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    GOLD_PATH.write_text(json.dumps(gold_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"input_manifest": str(INPUT_MANIFEST), "gold": str(GOLD_PATH), "groups": len(input_groups)}, indent=2))


if __name__ == "__main__":
    main()

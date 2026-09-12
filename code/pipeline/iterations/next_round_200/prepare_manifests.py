"""Evaluation-side creation of gold-free manifests for the two 100-page cohorts.

This is the only preparation program that reads the fresh human export. It writes
only image identity, filename, neutral bibliographic metadata, and provenance.
Annotation status, labels, timing, and detector face-count metadata are excluded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "next_round_200"
OLD_SOURCE = ROOT / "annotation_app_v2" / "data" / "test_collection_200_difficult_joined_hosted_manifest.json"
FRESH_EXPORT = ROOT / "annotation_results" / "economist_decade_face_count_stratified_200_seed20260812_min2_52123740_2026-08-12.json"
OLD_NEUTRAL = ROOT / "qwen_iteration" / "first100_pipeline_matrix" / "data" / "input_manifest_first100.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, images: list[dict[str, Any]], source: Path, selection: str) -> None:
    assert len(images) == 100
    assert len({row["image_id"] for row in images}) == 100
    out = HERE / "data" / f"manifest_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "qwen_next_round_neutral_manifest_v1",
        "cohort": name,
        "selection": selection,
        "source_sha256": sha(source),
        "gold_fields_included": False,
        "images": images,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "images": len(images), "sha256": sha(out)}))


def main() -> None:
    old = json.loads(OLD_NEUTRAL.read_text(encoding="utf-8"))["images"][:100]
    old_images = [{
        "manifest_index": i,
        "image_id": row["image_id"],
        "filename": row["filename"],
        "metadata": {"year": int(row["metadata"]["year"])},
    } for i, row in enumerate(old)]
    write("difficult100", old_images, OLD_SOURCE, "indices 0..99 of difficult-200 manifest")

    export = json.loads(FRESH_EXPORT.read_text(encoding="utf-8"))
    rows = sorted(export["annotations"], key=lambda row: int(row["sort_order"]))[:100]
    assert [int(row["sort_order"]) for row in rows] == list(range(100))
    fresh_images = []
    for i, row in enumerate(rows):
        image = row["payload"]["image"]
        fresh_images.append({
            "manifest_index": i,
            "image_id": str(row["image_id"]),
            "filename": str(row["filename"]),
            "metadata": {"year": int(image["metadata"]["year"])},
        })
    assert not ({row["image_id"] for row in old_images} & {row["image_id"] for row in fresh_images})
    write("stratified100", fresh_images, FRESH_EXPORT, "sort_order 0..99 of stratified-200 export")


if __name__ == "__main__":
    main()

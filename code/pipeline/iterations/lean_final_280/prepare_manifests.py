#!/usr/bin/env python3
"""Create gold-free first-140 manifests and a documented pilot selection.

This preparation program may read hosted exports to preserve their image order,
but writes no annotations, labels, assignment metadata, or gold-derived fields.
Inference reads only the resulting neutral manifests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "lean_final_280"
DIFFICULT_EXPORT = ROOT / "annotation_results" / "test_collection_200_difficult_joined_v1_2026-08-02.json"
STRATIFIED_EXPORT = ROOT / "annotation_results" / "economist_decade_face_count_stratified_200_seed20260812_min2_2026-08-13_full.json"
IMAGE_DIRS = {
    "difficult140": ROOT / "code" / "test_collection_200_difficult_joined_pages",
    "stratified140": ROOT / "master_thesis" / "data" / "images" / "full_pages_1940_2007_joined",
}

USER_CASES = {
    "difficult140": [
        "1995-1104-0074_0075", "1955-0409-0028", "2002-0525-0019",
        "1957-1005-0008", "2003-0607-0020",
    ],
    "stratified140": [
        "1965-1113-0053", "1945-1110-0020", "1998-0523-0038_0039",
        "1953-0808-0025", "1968-1012-0087", "2001-0707-0109",
        "1994-0917-0034_0035", "1997-0809-0070_0071",
    ],
}


def ordered_images(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("annotations") or []
    by_sort = sorted(rows, key=lambda row: int(row.get("sort_order", 10**9)))
    images = []
    seen = set()
    for row in by_sort:
        image_id = str(row.get("image_id") or "")
        filename = str(row.get("image_file") or row.get("filename") or f"{image_id}.jpg")
        if not image_id or image_id in seen:
            continue
        seen.add(image_id)
        images.append({
            "image_id": image_id,
            "filename": filename,
            "metadata": {"year": int(image_id[:4]) if image_id[:4].isdigit() else None},
        })
    return images


def main() -> int:
    HERE.joinpath("data").mkdir(parents=True, exist_ok=True)
    sources = {
        "difficult140": ordered_images(DIFFICULT_EXPORT),
        "stratified140": ordered_images(STRATIFIED_EXPORT),
    }
    assert len(sources["difficult140"]) >= 140
    assert len(sources["stratified140"]) == 200

    pilot: dict[str, list[str]] = {}
    for cohort, source in sources.items():
        selected = source[:140]
        image_dir = IMAGE_DIRS[cohort]
        for index, item in enumerate(selected):
            item["manifest_index"] = index
            item["metadata"]["cohort"] = cohort
            item["metadata"]["cohort_index"] = index
            assert (image_dir / item["filename"]).is_file(), image_dir / item["filename"]
        ids = {item["image_id"] for item in selected}
        assert all(image_id in ids for image_id in USER_CASES[cohort])
        pilot_ids = list(USER_CASES[cohort])
        pilot_ids.extend(item["image_id"] for item in selected if item["image_id"] not in pilot_ids)
        pilot[cohort] = pilot_ids[:30]
        manifest = {
            "schema_version": "lean_final_neutral_manifest_v1",
            "cohort": cohort,
            "selection": "first 140 in hosted annotation order; indices 140+ reserved",
            "gold_fields_included": False,
            "image_dir": str(image_dir),
            "images": selected,
        }
        path = HERE / "data" / f"manifest_{cohort}.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{cohort}: source={len(source)} selected={len(selected)} reserved={len(source)-140} -> {path}")

    pilot_path = HERE / "data" / "pilot_ids.json"
    pilot_path.write_text(
        json.dumps({"schema_version": "lean_final_pilot_ids_v1", "ids": pilot}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"pilot: 30+30 -> {pilot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Format-only export of final candidates for annotation_comparison_app.

No gold annotations are read.  The two outputs share one neutral 280-image
manifest (difficult140 followed by stratified140).
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "lean_final_280"
APP_OUTPUT = ROOT / "openrouter_qwen" / "output"
APP_MANIFEST = ROOT / "annotation_app_v2" / "data" / "qwen_lean_final_280_visual_comparison_manifest.json"
ROUTES = {
    "lean_s2c_direct_nogaze": "Qwen lean final: no individual/group gaze",
    "lean_s2c_direct_hybrid": "Qwen lean final recommended: individual gaze dropped, group gaze retained",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    numbers = [float(item) for item in value]
    if any(item > 1 for item in numbers):
        numbers = [item / 1000 for item in numbers]
    return [round(max(0, min(1, item)), 4) for item in numbers]


def convert(annotation: dict[str, Any], image: dict[str, Any], route: str, exported: str) -> dict[str, Any]:
    output = copy.deepcopy(annotation)
    output["schema_version"] = "ad_face_annotation_v2_comparison_qwen_lean_final_280"
    output["status"] = "complete"
    output["image"] = copy.deepcopy(image)
    for ad in output.get("advertisements") or []:
        ad["ad_id"] = ad.get("advertisement_id")
        ad["bbox"] = normalize_box(ad.get("bbox_1000"))
        for person in ad.get("people") or []:
            person["face_bbox"] = normalize_box(person.get("face_bbox_1000"))
        for group in ad.get("groups") or []:
            group["bbox"] = normalize_box(group.get("bbox_1000"))
    output["machine_annotation"] = {
        "model": "qwen/qwen3.5-9b",
        "provider_tag": "venice/fp8",
        "route": route,
        "route_label": ROUTES[route],
        "experiment": "qwen_iteration/lean_final_280",
        "processed_at": exported,
        "format_only_export": True,
    }
    return output


def main() -> int:
    exported = now()
    items = []
    source = {}
    for cohort in ["difficult140", "stratified140"]:
        manifest = json.loads((HERE / "data" / f"manifest_{cohort}.json").read_text(encoding="utf-8"))
        for cohort_index, raw in enumerate(manifest["images"]):
            path = Path(manifest["image_dir"]) / raw["filename"]
            metadata = copy.deepcopy(raw.get("metadata") or {})
            metadata.update({"cohort": cohort, "cohort_index": cohort_index})
            item = {"image_id": raw["image_id"], "filename": raw["filename"], "path": str(path), "metadata": metadata}
            items.append(item)
        for route in ROUTES:
            path = HERE / "output" / cohort / "assembled" / f"{route}.jsonl"
            source[(cohort, route)] = {row["image_id"]: row for row in rows(path)}
    APP_OUTPUT.mkdir(parents=True, exist_ok=True)
    for route, label in ROUTES.items():
        records = []
        for item in items:
            cohort = item["metadata"]["cohort"]
            row = source[(cohort, route)][item["image_id"]]
            annotation = convert(row["annotation"], item, route, exported)
            records.append({
                "image_id": item["image_id"], "filename": item["filename"], "image_file": item["filename"],
                "image_path": item["path"], "cohort": cohort, "cohort_index": item["metadata"]["cohort_index"],
                "ok": True, "model": "qwen/qwen3.5-9b", "processed_at": exported, "task": label,
                "route": route, "annotation": annotation,
            })
        out = APP_OUTPUT / f"qwen_lean_final_280_{'nogaze_minimal' if route.endswith('nogaze') else 'hybrid_recommended'}.jsonl"
        out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        print(f"wrote {len(records)} -> {out}")
    manifest = {
        "schema_version": "qwen_lean_final_280_visual_manifest_v1",
        "task_id": "qwen_lean_final_280_visual_comparison",
        "name": "Qwen lean final: difficult140 + stratified140",
        "description": "Neutral 280-page manifest; records after index 139 remain reserved (58 joined difficult and 60 stratified).",
        "created_at": exported,
        "images": items,
    }
    APP_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    APP_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(items)} -> {APP_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

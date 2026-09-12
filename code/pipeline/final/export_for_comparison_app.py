#!/usr/bin/env python3
"""Format-only exports for annotation_comparison_app; reads no human labels."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(r".")
HERE = Path(__file__).resolve().parent
APP_OUTPUT = ROOT / "openrouter_qwen" / "output"
APP_DATA = ROOT / "annotation_app_v2" / "data"


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    numbers = [float(item) for item in value]
    if any(item > 1 for item in numbers):
        numbers = [item / 1000 for item in numbers]
    return [round(max(0, min(1, item)), 4) for item in numbers]


def convert(annotation: dict[str, Any], image: dict[str, Any], exported: str) -> dict[str, Any]:
    output = copy.deepcopy(annotation)
    output["schema_version"] = "ad_face_annotation_v2_comparison_qwen_final_standalone"
    output["status"] = "complete"
    output["image"] = copy.deepcopy(image)
    for ad in output.get("advertisements") or []:
        ad["ad_id"] = ad.get("advertisement_id")
        ad["bbox"] = box(ad.get("bbox_1000"))
        for person in ad.get("people") or []:
            person["face_bbox"] = box(person.get("face_bbox_1000"))
        for group in ad.get("groups") or []:
            group["bbox"] = box(group.get("bbox_1000"))
    output["machine_annotation"] = {
        "model": "qwen/qwen3.5-9b", "provider_tag": "venice/fp8",
        "route": "standalone_final_v1", "route_label": "Qwen standalone final: tiled page + direct entities + gaze + WARC",
        "experiment": "qwen_iteration/final_pipeline_standalone", "processed_at": exported,
        "format_only_export": True, "duplicate_sidecar_promoted": False,
    }
    return output


def main() -> int:
    exported = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    APP_OUTPUT.mkdir(parents=True, exist_ok=True); APP_DATA.mkdir(parents=True, exist_ok=True)
    result = {}
    for cohort in ["difficult", "stratified"]:
        manifest = json.loads((HERE / "data" / f"manifest_{cohort}.json").read_text(encoding="utf-8"))
        predictions = {row["image_id"]: row for row in rows(HERE / "output" / cohort / "assembled" / "final.jsonl")}
        items = []; records = []
        for index, raw in enumerate(manifest["images"]):
            image_path = Path(manifest["image_dir"]) / raw["filename"]
            split = "development140" if index < 140 else "reserve"
            item = {
                "image_id": raw["image_id"], "filename": raw["filename"], "path": str(image_path),
                "metadata": {"cohort": cohort, "cohort_index": index, "split": split, "year": raw.get("year")},
            }
            items.append(item)
            annotation = convert(predictions[raw["image_id"]]["annotation"], item, exported)
            records.append({
                "image_id": raw["image_id"], "filename": raw["filename"], "image_file": raw["filename"], "image_path": str(image_path),
                "cohort": cohort, "cohort_index": index, "ok": True, "model": "qwen/qwen3.5-9b", "processed_at": exported,
                "task": "Qwen standalone final", "route": "standalone_final_v1", "annotation": annotation,
            })
        out = APP_OUTPUT / f"qwen_final_standalone_{cohort}.jsonl"
        out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        visual_manifest = {
            "schema_version": "qwen_final_standalone_visual_manifest_v1", "task_id": f"qwen_final_standalone_{cohort}",
            "name": f"Qwen standalone final — {cohort}", "description": f"{len(items)} pages; indices 0-139 development, 140+ reserve.",
            "created_at": exported, "images": items,
        }
        manifest_path = APP_DATA / f"qwen_final_standalone_{cohort}_visual_comparison_manifest.json"
        manifest_path.write_text(json.dumps(visual_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        result[cohort] = {"records": len(records), "jsonl": str(out), "manifest": str(manifest_path)}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

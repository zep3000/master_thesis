#!/usr/bin/env python3
"""Build the neutral diagnostic manifest and separate evaluation labels.

This is evaluation-side code: it reads frozen strict matches and human labels.
The inference runner never imports this module and never reads gold_labels.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "expression_legibility_research"
SOURCE = ROOT / "qwen_iteration" / "first100_pipeline_matrix"
MATCHED = SOURCE / "evaluation" / "first100_frozen_v1" / "visual_audit" / "person_expression_matched.csv"
P2_PERSONS = SOURCE / "output" / "first100_frozen_v1" / "p2" / "persons.jsonl"
DATA = HERE / "data"
VARIANTS = DATA / "images"

TARGETS = {
    "0_not_legible": {"all": 14},
    "1_low_legibility": {"under": 9, "equal": 4},
    "2_moderate_legibility": {"under": 9, "non_under": 4},
    "3_high_legibility": {"all": 7},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_tie(row: dict[str, str]) -> str:
    value = f"{row['image_id']}::{row['predicted_person_id']}"
    return hashlib.sha256(value.encode()).hexdigest()


def choose(rows: list[dict[str, str]], n: int, selected: list[dict[str, str]]) -> list[dict[str, str]]:
    chosen: list[dict[str, str]] = []
    remaining = list(rows)
    page_counts = Counter(row["image_id"] for row in selected)
    depiction_counts = Counter(row["human_depiction_type"] for row in selected)
    size_counts = Counter(row["face_size_bin"] for row in selected)
    while remaining and len(chosen) < n:
        remaining.sort(key=lambda row: (
            page_counts[row["image_id"]],
            depiction_counts[row["human_depiction_type"]],
            size_counts[row["face_size_bin"]],
            stable_tie(row),
        ))
        row = remaining.pop(0)
        chosen.append(row)
        page_counts[row["image_id"]] += 1
        depiction_counts[row["human_depiction_type"]] += 1
        size_counts[row["face_size_bin"]] += 1
    if len(chosen) != n:
        raise RuntimeError(f"Could select only {len(chosen)}/{n} cases")
    return chosen


def latest_person_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(P2_PERSONS):
        if row.get("ok") is True:
            records[str(row["task_key"])] = row
    return records


def make_variants(source: Path, case_id: str) -> tuple[Path, Path]:
    with Image.open(source) as opened:
        composite = opened.convert("RGB")
    if composite.size != (1400, 740):
        raise ValueError(f"Unexpected composite size {composite.size}: {source}")
    face = composite.crop((0, 40, 700, 740))
    face_path = VARIANTS / f"{case_id}_face.jpg"
    face.save(face_path, "JPEG", quality=95, optimize=True)

    enhanced = ImageOps.autocontrast(face, cutoff=1)
    enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1.1, percent=115, threshold=3))
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.06)
    canvas = Image.new("RGB", (1400, 740), "white")
    canvas.paste(face, (0, 40))
    canvas.paste(enhanced, (700, 40))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 10), "ORIGINAL TARGET FACE", fill="black", font=font)
    draw.text((710, 10), "DETERMINISTIC CONTRAST/SHARPNESS VIEW", fill="black", font=font)
    dual_path = VARIANTS / f"{case_id}_dual.jpg"
    canvas.save(dual_path, "JPEG", quality=95, optimize=True)
    return face_path, dual_path


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    VARIANTS.mkdir(parents=True, exist_ok=True)
    with MATCHED.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["pipeline"] == "p2" and row["human_legibility"]]

    selected: list[dict[str, str]] = []
    for level, quotas in TARGETS.items():
        pool = [row for row in rows if row["human_legibility"] == level]
        if "all" in quotas:
            if len(pool) != quotas["all"]:
                raise RuntimeError(f"Expected {quotas['all']} {level} rows, got {len(pool)}")
            selected.extend(choose(pool, quotas["all"], selected))
            continue
        if level == "1_low_legibility":
            selected.extend(choose([row for row in pool if row["legibility_relation"] == "under"], quotas["under"], selected))
            selected.extend(choose([row for row in pool if row["legibility_relation"] == "equal"], quotas["equal"], selected))
        elif level == "2_moderate_legibility":
            selected.extend(choose([row for row in pool if row["legibility_relation"] == "under"], quotas["under"], selected))
            selected.extend(choose([row for row in pool if row["legibility_relation"] != "under"], quotas["non_under"], selected))

    # Stable anonymous case IDs do not encode gold level or error direction.
    selected.sort(key=stable_tie)
    records = latest_person_records()
    manifest_cases = []
    gold_cases: dict[str, Any] = {}
    prior_outputs: dict[str, Any] = {}
    selection_rows = []
    for index, row in enumerate(selected, start=1):
        case_id = f"expr_{index:03d}"
        task_key = f"{row['image_id']}::{row['predicted_person_id']}"
        record = records.get(task_key)
        if not record:
            raise KeyError(f"Missing P2 person call: {task_key}")
        source_path = Path(record["image_path"]).resolve()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        face_path, dual_path = make_variants(source_path, case_id)
        manifest_cases.append({
            "case_id": case_id,
            "source_composite_path": str(source_path),
            "face_only_path": str(face_path.resolve()),
            "dual_view_path": str(dual_path.resolve()),
        })
        usage = record.get("usage") or {}
        gold_cases[case_id] = {
            "source_image_id": row["image_id"],
            "human_person_id": row["human_person_id"],
            "matched_p2_person_id": row["predicted_person_id"],
            "iou": float(row["iou"]),
            "face_size_bin": row["face_size_bin"],
            "human_depiction_type": row["human_depiction_type"],
            "human_orientation": row["human_orientation"],
            "gold": {
                "face_expression_legibility": row["human_legibility"],
                "gaze_target": row["human_gaze"] or None,
                "smile_present": row["human_smile"] or None,
                "smile_intensity": row["human_intensity"] or None,
            },
            "frozen_p2_baseline": {
                "face_expression_legibility": row["predicted_legibility"],
                "gaze_target": row["predicted_gaze"] or None,
                "smile_present": row["predicted_smile"] or None,
                "smile_intensity": row["predicted_intensity"] or None,
                "usage": usage,
            },
            "selection_stratum": f"{row['human_legibility']}::{row['legibility_relation']}",
        }
        prior_outputs[case_id] = {
            "case_id": case_id,
            "source_task_key": task_key,
            "model": "qwen/qwen3.5-9b",
            "model_annotation": record["model_annotation"],
            "usage": usage,
        }
        selection_rows.append({"case_id": case_id, **row})

    # Two cases per gold level, selected evaluation-side. Only anonymous IDs are
    # copied into the neutral manifest.
    pilot_ids = []
    for level in TARGETS:
        candidates = [case_id for case_id, item in gold_cases.items() if item["gold"]["face_expression_legibility"] == level]
        pilot_ids.extend(candidates[:2])

    manifest = {
        "schema_version": "expression_legibility_neutral_manifest_v1",
        "description": "Gold-free face-crop inputs selected evaluation-side from frozen P2 strict matches.",
        "cases": manifest_cases,
        "pilot_case_ids": pilot_ids,
    }
    labels = {
        "schema_version": "expression_legibility_evaluation_labels_v1",
        "warning": "Evaluation side only. Inference must never read this file.",
        "cases": gold_cases,
    }
    (DATA / "cases_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (DATA / "frozen_p2_outputs.json").write_text(json.dumps({
        "schema_version": "expression_legibility_prior_model_outputs_v1",
        "description": "Inference-safe prior Qwen outputs; contains no human labels.",
        "cases": prior_outputs,
    }, indent=2) + "\n", encoding="utf-8")
    (DATA / "gold_labels.json").write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    with (DATA / "selection_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0]))
        writer.writeheader()
        writer.writerows(selection_rows)
    print(f"Wrote {len(manifest_cases)} neutral cases; pilot={len(pilot_ids)}")
    print(Counter(item["gold"]["face_expression_legibility"] for item in gold_cases.values()))
    print(f"Pages={len(set(item['source_image_id'] for item in gold_cases.values()))}")


if __name__ == "__main__":
    main()

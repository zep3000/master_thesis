#!/usr/bin/env python3
"""Post-pipeline legacy-box matching experiment on frozen model face boxes.

This script is not part of the Qwen annotation pipeline. It reads the completed
canonical output and the historical Faces Dataset CSV, then writes a separate
derivative under this experiment directory. No canonical file is modified.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANONICAL = ROOT / "qwen_iteration" / "canonical_candidate_280"
sys.path.insert(0, str(CANONICAL))

from common import rows, write_jsonl  # noqa: E402


CSV_PATH = ROOT / "master_thesis" / "data" / "processed" / "TheEconomistHistoricalArchives-Faces-deduplicated_cleaned.csv"
CANONICAL_OUTPUT = CANONICAL / "output"
OUTPUT = HERE / "output"


def source_id(image_id: str) -> str:
    match = re.fullmatch(r"(\d{4}-\d{4}-\d{4})_(\d{4})", image_id)
    return f"{match.group(1)},{match.group(2)}" if match else image_id


def iou(a: list[int], b: list[int]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def center_in(box: list[int], outer: list[int]) -> bool:
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return outer[0] <= x <= outer[2] and outer[1] <= y <= outer[3]


def nms(values: list[dict[str, Any]], threshold: float = 0.35) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for value in sorted(values, key=lambda row: row["confidence"], reverse=True):
        if not any(iou(value["bbox_1000"], old["bbox_1000"]) >= threshold for old in kept):
            kept.append(value)
    return kept


def detections(image_ids: list[str], confidence: float) -> dict[str, list[dict[str, Any]]]:
    wanted = {source_id(image_id): image_id for image_id in image_ids}
    result = {image_id: [] for image_id in image_ids}
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = wanted.get(row["Filename"].split("_", 1)[0])
            if not image_id or float(row["Segmentation confidence score"]) < confidence:
                continue
            box = [round(float(row[f"Bounding Box relative {axis}"]) * 1000) for axis in ["X1", "Y1", "X2", "Y2"]]
            if box[0] < box[2] and box[1] < box[3]:
                result[image_id].append({"bbox_1000": box, "confidence": float(row["Segmentation confidence score"])})
    return {key: nms(value) for key, value in result.items()}


def recenter(box: list[int], detector: list[int], fraction: float) -> list[int]:
    width, height = box[2] - box[0], box[3] - box[1]
    bx, by = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    dx, dy = (detector[0] + detector[2]) / 2, (detector[1] + detector[3]) / 2
    cx, cy = bx + fraction * (dx - bx), by + fraction * (dy - by)
    x1, y1 = max(0, min(1000 - width, round(cx - width / 2))), max(0, min(1000 - height, round(cy - height / 2)))
    return [x1, y1, x1 + width, y1 + height]


def refine(annotation: dict[str, Any], page_detections: list[dict[str, Any]], fraction: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output = copy.deepcopy(annotation)
    changes: list[dict[str, Any]] = []
    for ad in output.get("advertisements") or []:
        candidates = [row for row in page_detections if center_in(row["bbox_1000"], ad["bbox_1000"])]
        pairs = []
        for pi, person in enumerate(ad.get("people") or []):
            for di, detector in enumerate(candidates):
                score = iou(person["face_bbox_1000"], detector["bbox_1000"])
                if score >= 0.10:
                    pairs.append((score, pi, di))
        used_people: set[int] = set()
        used_detectors: set[int] = set()
        for score, pi, di in sorted(pairs, reverse=True):
            if pi in used_people or di in used_detectors:
                continue
            used_people.add(pi); used_detectors.add(di)
            person = ad["people"][pi]
            old = person["face_bbox_1000"]
            new = recenter(old, candidates[di]["bbox_1000"], fraction)
            person["face_bbox_1000"] = new
            changes.append({
                "advertisement_id": ad["advertisement_id"], "person_id": person["person_id"],
                "old_bbox_1000": old, "new_bbox_1000": new, "detector_bbox_1000": candidates[di]["bbox_1000"],
                "model_detector_iou": score, "fraction": fraction,
            })
    return output, changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["difficult140", "stratified140"], default="difficult140")
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--apply-selection-biased", action="store_true", help="Required to apply (not validate) on the detector-selected stratified cohort.")
    args = parser.parse_args()
    if args.cohort == "stratified140" and not args.apply_selection_biased:
        raise SystemExit("Refusing stratified detector application without --apply-selection-biased; never use it for detector validation.")
    base_path = OUTPUT / "assembled" / args.cohort / "canonical_candidate_v1.jsonl"
    base = [row for row in rows(base_path) if row.get("ok")]
    found = detections([row["image_id"] for row in base], args.confidence)
    summary: dict[str, Any] = {"cohort": args.cohort, "confidence": args.confidence, "selection_bias_warning": args.cohort == "stratified140", "validation_permitted": args.cohort == "difficult140", "variants": {}}
    for route, fraction in [("detector_half_center", 0.5), ("detector_full_center", 1.0)]:
        values, log = [], []
        for row in base:
            annotation, changes = refine(row["annotation"], found.get(row["image_id"], []), fraction)
            values.append({**row, "route": route, "annotation": annotation})
            log.extend({"image_id": row["image_id"], **change} for change in changes)
        write_jsonl(OUTPUT / "assembled" / args.cohort / f"{route}.jsonl", values)
        write_jsonl(OUTPUT / "detector_center" / f"{args.cohort}_{route}_changes.jsonl", log)
        summary["variants"][route] = {"pages": len(values), "moved_boxes": len(log)}
    path = OUTPUT / "detector_center" / f"{args.cohort}_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

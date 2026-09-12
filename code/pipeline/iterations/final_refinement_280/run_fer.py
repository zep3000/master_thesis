#!/usr/bin/env python3
"""Run the separate three-threshold expression-legibility specialist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import prompts
from common import BASE, HERE, execute, image_data_url, manifest, normalize_moderate_audit, normalize_threshold, person_composite, rows


def current_tasks(cohort: str) -> list[dict]:
    result = []
    for row in rows(BASE / "output" / cohort / "assembled" / "lean_s2c_direct_gaze.jsonl"):
        for ad in (row.get("annotation") or {}).get("advertisements") or []:
            for person in ad.get("people") or []:
                result.append({
                    "cohort": cohort, "image_id": row["image_id"], "task_id": f"{row['image_id']}::{person['person_id']}",
                    "person_id": person["person_id"], "ad_box": ad["bbox_1000"], "face_box": person["face_bbox_1000"],
                    "direct_legibility": person.get("face_expression_legibility"),
                })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["difficult140", "stratified140"], required=True)
    parser.add_argument("--scope", choices=["pilot", "all"], required=True)
    parser.add_argument("--variant", choices=["nested_visibility", "moderate_audit"], default="nested_visibility")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-requests", type=int, default=3000)
    args = parser.parse_args()
    tasks = current_tasks(args.cohort)
    if args.variant == "moderate_audit":
        tasks = [x for x in tasks if x["direct_legibility"] == "2_moderate_legibility"]
    if args.scope == "pilot":
        wanted = {x["task_id"] for x in json.loads((HERE / "data" / "fer_pilot.json").read_text(encoding="utf-8"))["tasks"] if x["cohort"] == args.cohort}
        tasks = [x for x in tasks if x["task_id"] in wanted]
    data = manifest(args.cohort)
    by_id = {x["image_id"]: x for x in data["images"]}
    image_dir = Path(data["image_dir"])
    jobs = []
    for task in tasks:
        page = image_dir / by_id[task["image_id"]]["filename"]
        out = HERE / "output" / "crops" / "fer" / f"{task['image_id']}__{task['person_id']}.jpg"
        composite = person_composite(page, task["ad_box"], task["face_box"], out)
        task_id = task["task_id"]
        if args.variant == "moderate_audit":
            prompt = prompts.fer_moderate_audit_prompt(task_id)
            normalizer = lambda raw, task_id=task_id: normalize_moderate_audit(raw, task_id)
        else:
            prompt = prompts.fer_threshold_prompt(task_id)
            normalizer = lambda raw, task_id=task_id: normalize_threshold(raw, task_id)
        jobs.append({
            "task_key": task_id,
            "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_data_url(composite)}}],
            "max_tokens": 650, "normalize": normalizer,
            "meta": {"stage": "fer_threshold", "cohort": args.cohort, "variant": args.variant, "scope": args.scope},
        })
    name = "moderate_audit.jsonl" if args.variant == "moderate_audit" else "thresholds.jsonl"
    execute(jobs, HERE / "output" / "fer" / args.cohort / name, args.max_requests, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

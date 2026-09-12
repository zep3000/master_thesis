#!/usr/bin/env python3
"""Run one-call multiscale group aggregate variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import prompts
from common import HERE, execute, group_composite, image_data_url, manifest, normalize_group


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["difficult140", "stratified140"], required=True)
    parser.add_argument("--variant", choices=["multiscale", "multiscale_context"], required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-requests", type=int, default=3000)
    args = parser.parse_args()
    tasks = json.loads((HERE / "data" / "current_groups.json").read_text(encoding="utf-8"))["tasks"]
    tasks = [x for x in tasks if x["cohort"] == args.cohort]
    data = manifest(args.cohort)
    by_id = {x["image_id"]: x for x in data["images"]}
    image_dir = Path(data["image_dir"])
    include_context = args.variant.endswith("context")
    jobs = []
    for task in tasks:
        page = image_dir / by_id[task["image_id"]]["filename"]
        out = HERE / "output" / "crops" / "groups" / args.variant / f"{task['image_id']}__{task['group_id']}.jpg"
        composite = group_composite(page, task["ad_box"], task["group_box"], out, include_context)
        task_id = task["task_id"]
        jobs.append({
            "task_key": task_id,
            "content": [{"type": "text", "text": prompts.group_prompt(task_id, include_context)}, {"type": "image_url", "image_url": {"url": image_data_url(composite)}}],
            "max_tokens": 1400, "normalize": lambda raw, task_id=task_id: normalize_group(raw, task_id, False),
            "meta": {"stage": "group", "cohort": args.cohort, "variant": args.variant},
        })
    execute(jobs, HERE / "output" / "groups" / args.cohort / f"{args.variant}.jsonl", args.max_requests, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

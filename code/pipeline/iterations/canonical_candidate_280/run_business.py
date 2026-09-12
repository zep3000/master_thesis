#!/usr/bin/env python3
"""Gold-blind WARC/brand backfill over frozen advertisement boxes.

The production candidate keeps these fields in the page call.  This isolated
backfill is only needed to reconcile already-frozen legacy development outputs
without rerunning or changing their geometry.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from PIL import Image

import prompts
from common import MODEL, OUTPUT, baseline_rows, execute, image_path, jpeg_data_url, to_pixels


def normalize(raw: dict[str, Any], task_key: str) -> dict[str, Any]:
    category = raw.get("ad_category")
    if isinstance(category, list) and len(category) == 1:
        category = category[0]
    if category not in prompts.WARC_CATEGORIES:
        raise ValueError(f"invalid WARC category: {category!r}")
    brand = raw.get("brand_or_advertiser")
    if isinstance(brand, list) and len(brand) == 1:
        brand = brand[0]
    if not isinstance(brand, str) or not brand.strip():
        brand = None
    return {"task_key": task_key, "ad_category": category, "brand_or_advertiser": brand.strip() if brand else None}


def jobs(cohort: str, scope: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in baseline_rows(cohort):
        page = Image.open(image_path(cohort, row["filename"])).convert("RGB")
        for ad in row["annotation"].get("advertisements") or []:
            task_key = f"{row['image_id']}::{ad['advertisement_id']}"
            crop = page.crop(to_pixels(ad["bbox_1000"], page.size))
            crop.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            result.append({
                "task_key": task_key,
                "content": [
                    {"type": "text", "text": prompts.business_prompt(task_key)},
                    {"type": "image_url", "image_url": {"url": jpeg_data_url(crop)}},
                ],
                "max_tokens": 500,
                "normalize": lambda raw, task_key=task_key: normalize(raw, task_key),
                "meta": {"stage": "business", "cohort": cohort, "variant": "warc_backfill_v1", "model": MODEL},
                "result_meta": {"image_id": row["image_id"], "advertisement_id": ad["advertisement_id"]},
            })
    if scope == "pilot":
        result = sorted(result, key=lambda x: hashlib.sha256(x["task_key"].encode()).hexdigest())[:20]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["difficult140", "stratified140"], required=True)
    parser.add_argument("--scope", choices=["pilot", "all"], default="pilot")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--maximum", type=int, default=3000)
    args = parser.parse_args()
    execute(jobs(args.cohort, args.scope), OUTPUT / "business" / f"{args.cohort}.jsonl", args.maximum, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

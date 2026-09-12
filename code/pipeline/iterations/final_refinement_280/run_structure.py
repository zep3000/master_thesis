#!/usr/bin/env python3
"""Run page-structure tiling and group-partition variants."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

import prompts
from common import HERE, clean_structure, execute, image_data_url, jpeg_data_url, normalize_structure, selected_images


def tiles(path: Path, variant: str) -> list[str]:
    page = Image.open(path).convert("RGB")
    if variant.startswith("fixed_") or .8 < page.width / page.height < 1.25:
        regions = [(0, 0, .6, .6), (.4, 0, 1, .6), (0, .4, .6, 1), (.4, .4, 1, 1)]
    elif page.width / page.height >= 1.25:
        regions = [(x1, y1, x2, y2) for y1, y2 in [(0, .55), (.45, 1)] for x1, x2 in [(0, .4), (.3, .7), (.6, 1)]]
    else:
        regions = [(x1, y1, x2, y2) for y1, y2 in [(0, .4), (.3, .7), (.6, 1)] for x1, x2 in [(0, .55), (.45, 1)]]
    result = []
    for box in regions:
        x1, y1, x2, y2 = box
        crop = page.crop((round(x1 * page.width), round(y1 * page.height), round(x2 * page.width), round(y2 * page.height)))
        crop.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
        result.append(jpeg_data_url(crop))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["difficult140", "stratified140"], required=True)
    parser.add_argument("--variant", choices=["fixed_original", "fixed_partition", "aspect_original", "aspect_partition"], required=True)
    parser.add_argument("--scope", choices=["pilot", "all"], required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-requests", type=int, default=3000)
    args = parser.parse_args()
    manifest, images = selected_images(args.cohort, args.scope)
    image_dir = Path(manifest["image_dir"])
    partition = "revised" if args.variant.endswith("partition") else "original"
    jobs = []
    for item in images:
        path = image_dir / item["filename"]
        content = [{"type": "text", "text": prompts.structure_prompt(item["image_id"], item.get("metadata", {}).get("year"), partition)},
                   {"type": "image_url", "image_url": {"url": image_data_url(path)}}]
        for index, url in enumerate(tiles(path, args.variant), 1):
            content.extend([{"type": "text", "text": f"DETAIL CROP {index}"}, {"type": "image_url", "image_url": {"url": url}}])
        image_id = item["image_id"]
        jobs.append({
            "task_key": image_id, "content": content, "max_tokens": 8000,
            "normalize": lambda raw, image_id=image_id: clean_structure(normalize_structure(raw, image_id))[0],
            "meta": {"stage": "structure", "cohort": args.cohort, "variant": args.variant, "scope": args.scope},
        })
    execute(jobs, HERE / "output" / "structure" / args.cohort / f"{args.variant}.jsonl", args.max_requests, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Propose exact-duplicate candidates without modifying the main pipeline.

This runner reads only frozen model annotations and pixels.  It writes a
separate sidecar that no assembly module imports.  Every face remains an
independently annotated depiction whether or not this runner proposes a link.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

import prompts
from common import MODEL, OUTPUT, baseline_rows, execute, expand_box, image_data_url, image_path, stable_hash, to_pixels


COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#17becf", "#bcbd22"]


def normalize(raw: dict[str, Any], task_key: str, allowed_ids: list[str]) -> dict[str, Any]:
    allowed = set(allowed_ids)
    source = raw.get("candidate_clusters") if isinstance(raw.get("candidate_clusters"), list) else []
    clusters: list[dict[str, Any]] = []
    for item in source:
        if not isinstance(item, dict):
            continue
        ids = item.get("person_ids") if isinstance(item.get("person_ids"), list) else []
        ids = sorted({str(value) for value in ids if str(value) in allowed})
        strength = item.get("candidate_strength")
        if len(ids) >= 2 and strength in {"high", "medium"}:
            clusters.append({"person_ids": ids, "candidate_strength": strength})
    # Overlapping model clusters describe one connected candidate identity.
    merged: list[dict[str, Any]] = []
    for cluster in clusters:
        overlapping = [x for x in merged if set(x["person_ids"]) & set(cluster["person_ids"])]
        if not overlapping:
            merged.append(cluster)
            continue
        ids = set(cluster["person_ids"])
        strength = cluster["candidate_strength"]
        for old in overlapping:
            ids.update(old["person_ids"])
            strength = "medium" if "medium" in {strength, old["candidate_strength"]} else "high"
            merged.remove(old)
        merged.append({"person_ids": sorted(ids), "candidate_strength": strength})
    return {"task_key": task_key, "candidate_clusters": sorted(merged, key=lambda x: x["person_ids"])}


def fit_panel(image: Image.Image, width: int, height: int, label: str) -> Image.Image:
    panel = Image.new("RGB", (width, height), "white")
    work = image.copy()
    work.thumbnail((width - 16, height - 46), Image.Resampling.LANCZOS)
    panel.paste(work, ((width - work.width) // 2, 40 + (height - 40 - work.height) // 2))
    ImageDraw.Draw(panel).text((10, 12), label, fill="black", font=ImageFont.load_default())
    return panel


def montage(page_path: Path, ad: dict[str, Any], out_path: Path, crop_scale: float) -> Path:
    if out_path.exists():
        return out_path
    page = Image.open(page_path).convert("RGB")
    ad_px = to_pixels(ad["bbox_1000"], page.size)
    ad_image = page.crop(ad_px)
    draw = ImageDraw.Draw(ad_image)
    ax1, ay1, _, _ = ad_px
    people = ad.get("people") or []
    for index, person in enumerate(people):
        x1, y1, x2, y2 = to_pixels(person["face_bbox_1000"], page.size)
        color = COLORS[index % len(COLORS)]
        draw.rectangle((x1 - ax1, y1 - ay1, x2 - ax1, y2 - ay1), outline=color, width=max(3, round(min(page.size) / 350)))
        draw.rectangle((x1 - ax1, max(0, y1 - ay1 - 18), x1 - ax1 + 32, y1 - ay1), fill=color)
        draw.text((x1 - ax1 + 3, max(0, y1 - ay1 - 16)), f"P{index + 1}", fill="white", font=ImageFont.load_default())
    panels = [fit_panel(ad_image, 760, 620, "FROZEN ADVERTISEMENT CONTEXT")]
    for index, person in enumerate(people):
        face_px = to_pixels(person["face_bbox_1000"], page.size)
        face = page.crop(expand_box(face_px, page.size, crop_scale))
        panels.append(fit_panel(face, 300, 300, f"P{index + 1}: {person['person_id']}"))
    right_columns = 3
    right_rows = math.ceil(len(people) / right_columns)
    canvas = Image.new("RGB", (760 + right_columns * 300, max(620, right_rows * 300)), "white")
    canvas.paste(panels[0], (0, 0))
    for index, panel in enumerate(panels[1:]):
        canvas.paste(panel, (760 + (index % right_columns) * 300, (index // right_columns) * 300))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "JPEG", quality=92, optimize=True)
    return out_path


def jobs(cohort: str, scope: str, variant: str = "v1") -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in baseline_rows(cohort):
        for ad in row["annotation"].get("advertisements") or []:
            people = ad.get("people") or []
            if len(people) < 2:
                continue
            task_key = f"{row['image_id']}::{ad['advertisement_id']}"
            person_ids = [str(person["person_id"]) for person in people]
            source = {"advertisement_id": ad["advertisement_id"], "bbox_1000": ad["bbox_1000"], "people": [{"person_id": p["person_id"], "face_bbox_1000": p["face_bbox_1000"]} for p in people]}
            conservative = variant == "conservative_v2"
            path = montage(
                image_path(cohort, row["filename"]),
                ad,
                OUTPUT / "duplicate_montages" / variant / cohort / f"{task_key.replace('::', '__')}.jpg",
                1.4 if conservative else 2.2,
            )
            result.append({
                "task_key": task_key,
                "content": [
                    {"type": "text", "text": prompts.duplicate_prompt(task_key, person_ids, conservative)},
                    {"type": "image_url", "image_url": {"url": image_data_url(path)}},
                ],
                "max_tokens": 700,
                "normalize": lambda raw, task_key=task_key, person_ids=person_ids: normalize(raw, task_key, person_ids),
                "meta": {"stage": "duplicate_candidates", "cohort": cohort, "variant": f"isolated_frozen_montage_{variant}", "model": MODEL},
                "result_meta": {
                    "image_id": row["image_id"],
                    "advertisement_id": ad["advertisement_id"],
                    "person_ids": person_ids,
                    "source_annotation_hash": stable_hash(source),
                },
            })
    if scope == "pilot":
        result = sorted(result, key=lambda x: hashlib.sha256(x["task_key"].encode()).hexdigest())[:20]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["difficult140", "stratified140"], required=True)
    parser.add_argument("--scope", choices=["pilot", "all"], default="pilot")
    parser.add_argument("--variant", choices=["v1", "conservative_v2"], default="conservative_v2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--maximum", type=int, default=3000)
    args = parser.parse_args()
    suffix = "" if args.variant == "v1" else f"_{args.variant}"
    execute(jobs(args.cohort, args.scope, args.variant), OUTPUT / "duplicate_candidates" / f"{args.cohort}{suffix}.jsonl", args.maximum, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a deterministic, category-stratified visual audit of predicted ads.

The sampler never reads human gold annotations.  It selects two advertisements
per currently predicted category (preferring one from each cohort), then fills
to 40 with the lowest-confidence remaining advertisements.  Selection within
strata is deterministic under AUDIT_SEED.  The output manifest records every
selection decision and the rendered contact sheets contain only model output
and source pixels, so this can be reproduced for a methods appendix.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "category_taxonomy_reuse"
INPUT = ROOT / "openrouter_qwen" / "output" / "qwen_lean_final_280_hybrid_recommended.jsonl"
OUT = HERE / "audit_50"
AUDIT_SEED = "warc-fit-audit-v1"
TARGET_N = 50
TARGETED_STRESS_CASES = [
    ("1944-0408-0014", "ad_1"),  # alcoholic drink inside the old combined food/beverage/tobacco class
    ("1955-0611-0051", "ad_1"),  # soft drink inside the old combined class
    ("1962-0728-0007", "ad_1"),  # chewing gum / packaged food boundary
    ("1982-0522-0103", "ad_1"),  # watch: accessories versus consumer goods
    ("1986-0510-0105", "ad_1"),  # clothing manufacturer/retailer boundary
    ("2000-0304-0018", "ad_1"),  # postal/logistics service previously mislabeled technology
    ("2001-0505-0049", "ad_1"),  # industrial-gas company previously mislabeled technology
    ("1989-1104-0103", "ad_1"),  # industrial company versus technology boundary
    ("1953-0207-0044", "ad_3"),  # telecom service boundary
    ("1997-1011-0046", "ad_1"),  # sponsored prize / advertiser-versus-topic boundary
]


def stable_key(row: dict[str, Any]) -> str:
    token = f"{AUDIT_SEED}|{row['image_id']}|{row['ad_id']}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def load_ads() -> list[dict[str, Any]]:
    ads: list[dict[str, Any]] = []
    for line in INPUT.read_text(encoding="utf-8").splitlines():
        page = json.loads(line)
        for ad in page["annotation"].get("advertisements") or []:
            ads.append(
                {
                    "image_id": page["image_id"],
                    "cohort": page["cohort"],
                    "image_path": page["image_path"],
                    "ad_id": ad["advertisement_id"],
                    "bbox_1000": ad["bbox_1000"],
                    "predicted_category": ad.get("ad_category"),
                    "predicted_category_confidence": ad.get("ad_category_confidence"),
                    "predicted_brand": ad.get("brand_or_advertiser"),
                    "predicted_brand_confidence": ad.get("brand_confidence"),
                }
            )
    return ads


def select(ads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ads:
        by_category[str(row["predicted_category"])].append(row)

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str]] = set()
    for category in sorted(by_category):
        rows = by_category[category]
        chosen: list[dict[str, Any]] = []
        for cohort in ("difficult140", "stratified140"):
            candidates = sorted((r for r in rows if r["cohort"] == cohort), key=stable_key)
            if candidates:
                chosen.append(candidates[0])
        if len(chosen) < min(2, len(rows)):
            for row in sorted(rows, key=stable_key):
                if (row["image_id"], row["ad_id"]) not in {(r["image_id"], r["ad_id"]) for r in chosen}:
                    chosen.append(row)
                    if len(chosen) == min(2, len(rows)):
                        break
        for row in chosen:
            key = (row["image_id"], row["ad_id"])
            if key not in selected_keys:
                row = {**row, "selection_reason": "two_per_predicted_category"}
                selected.append(row)
                selected_keys.add(key)

    # Add predeclared boundary cases before the confidence fill.  The list is
    # fixed in source so it cannot drift in response to later model outputs.
    by_key = {(r["image_id"], r["ad_id"]): r for r in ads}
    for key in TARGETED_STRESS_CASES:
        if key in selected_keys or key not in by_key:
            continue
        selected.append({**by_key[key], "selection_reason": "predeclared_taxonomy_stress_case"})
        selected_keys.add(key)

    remaining = [r for r in ads if (r["image_id"], r["ad_id"]) not in selected_keys]
    remaining.sort(
        key=lambda r: (
            float(r.get("predicted_category_confidence") or 0),
            float(r.get("predicted_brand_confidence") or 0),
            stable_key(r),
        )
    )
    for row in remaining:
        if len(selected) >= TARGET_N:
            break
        selected.append({**row, "selection_reason": "lowest_remaining_confidence"})
    return selected[:TARGET_N]


def to_pixels(box: list[int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    return tuple(round(value * (width if i % 2 == 0 else height) / 1000) for i, value in enumerate(box))  # type: ignore[return-value]


def panel(row: dict[str, Any], index: int) -> Image.Image:
    image = Image.open(row["image_path"]).convert("RGB")
    crop = image.crop(to_pixels(row["bbox_1000"], image.size))
    crop.thumbnail((860, 500), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (900, 650), "white")
    canvas.paste(crop, ((900 - crop.width) // 2, 135 + (500 - crop.height) // 2))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    draw.text((16, 12), f"{index:02d}  {row['image_id']}  {row['ad_id']}  [{row['cohort']}]", fill="black", font=font)
    category = f"current: {row['predicted_category']} ({float(row['predicted_category_confidence']):.2f})"
    brand = f"brand: {row['predicted_brand']} ({float(row['predicted_brand_confidence']):.2f})"
    reason = f"sample: {row['selection_reason']}"
    for offset, text in enumerate((category, brand, reason)):
        wrapped = textwrap.wrap(text, width=92)[:2]
        draw.multiline_text((16, 38 + offset * 29), "\n".join(wrapped), fill=(30, 30, 30), font=font, spacing=2)
    return canvas


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = select(load_ads())
    manifest = {
        "audit_id": AUDIT_SEED,
        "sampling_frame": str(INPUT),
        "target_n": TARGET_N,
        "actual_n": len(rows),
        "gold_used_for_selection": False,
        "method": "two per predicted category, preferring one per cohort; fill by ascending model confidence",
        "ads": rows,
    }
    (OUT / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    panels = [panel(row, i) for i, row in enumerate(rows, 1)]
    sheet_count = (len(panels) + 9) // 10
    for sheet_index in range(sheet_count):
        block = panels[sheet_index * 10 : (sheet_index + 1) * 10]
        sheet = Image.new("RGB", (1800, 3250), (225, 225, 225))
        for local_index, item in enumerate(block):
            sheet.paste(item, ((local_index % 2) * 900, (local_index // 2) * 650))
        sheet.save(OUT / f"contact_{sheet_index + 1}.jpg", quality=92, optimize=True)
    print(json.dumps({"selected": len(rows), "output": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

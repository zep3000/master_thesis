#!/usr/bin/env python3
"""Render human-vs-final overlays for the user-named critical cases.

This is evaluation-only: it reads gold after inference is complete.  Blue is
human, red is the recommended hybrid candidate.  Ad boxes are thick; person
boxes thin; group boxes dashed where possible.  Category/brand are prediction
only and shown as text because no gold exists yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "lean_final_280"
CASES = {
    "difficult140": ["1995-1104-0074_0075", "1955-0409-0028", "2002-0525-0019", "1957-1005-0008", "2003-0607-0020"],
    "stratified140": ["1965-1113-0053", "1945-1110-0020", "1998-0523-0038_0039", "1953-0808-0025", "1968-1012-0087", "2001-0707-0109", "1994-0917-0034_0035", "1997-0809-0070_0071"],
}
GOLD = {
    "difficult140": ROOT / "annotation_results" / "test_collection_200_difficult_joined_v1_2026-08-02.json",
    "stratified140": ROOT / "annotation_results" / "economist_decade_face_count_stratified_200_seed20260812_min2_2026-08-13_full.json",
}


def load_gold(cohort: str) -> dict[str, dict[str, Any]]:
    raw = json.loads(GOLD[cohort].read_text(encoding="utf-8"))
    return {str(row["image_id"]): row["payload"] for row in raw["annotations"] if isinstance(row.get("payload"), dict) and (cohort != "difficult140" or row.get("assignment_code") == "79201188")}


def load_pred(cohort: str) -> dict[str, dict[str, Any]]:
    path = HERE / "output" / cohort / "assembled" / "lean_s2c_direct_hybrid.jsonl"
    return {row["image_id"]: row["annotation"] for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())}


def box_px(entity: dict[str, Any], key: str, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    raw = entity.get(key) or entity.get(f"{key}_1000")
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    scale = 1000 if any(float(value) > 1 for value in raw) else 1
    width, height = size
    return tuple(round(float(value) / scale * (width if index % 2 == 0 else height)) for index, value in enumerate(raw))  # type: ignore[return-value]


def counts(annotation: dict[str, Any]) -> tuple[int, int, int]:
    ads = annotation.get("advertisements") or []
    return len(ads), sum(len(ad.get("people") or []) for ad in ads), sum(len(ad.get("groups") or []) for ad in ads)


def draw_annotation(image: Image.Image, annotation: dict[str, Any], color: tuple[int, int, int]) -> None:
    draw = ImageDraw.Draw(image)
    width = max(2, round(min(image.size) / 300))
    for ad in annotation.get("advertisements") or []:
        box = box_px(ad, "bbox", image.size)
        if box:
            draw.rectangle(box, outline=color, width=width * 2)
        for person in ad.get("people") or []:
            box = box_px(person, "face_bbox", image.size)
            if box:
                draw.rectangle(box, outline=color, width=width)
        for group in ad.get("groups") or []:
            box = box_px(group, "bbox", image.size)
            if box:
                draw.rectangle(box, outline=color, width=width)


def panel(page: Image.Image, image_id: str, human: dict[str, Any], predicted: dict[str, Any]) -> Image.Image:
    target_h = 520
    page.thumbnail((500, target_h), Image.Resampling.LANCZOS)
    left, right = page.copy(), page.copy()
    draw_annotation(left, human, (0, 100, 255))
    draw_annotation(right, predicted, (230, 20, 20))
    canvas = Image.new("RGB", (1020, 650), "white")
    canvas.paste(left, ((500 - left.width) // 2, 55))
    canvas.paste(right, (510 + (500 - right.width) // 2, 55))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 10), image_id, fill="black", font=font)
    draw.text((10, 30), f"HUMAN blue  ads/people/groups={counts(human)}", fill=(0, 80, 200), font=font)
    draw.text((520, 30), f"QWEN red  ads/people/groups={counts(predicted)}", fill=(200, 0, 0), font=font)
    labels = []
    for ad in predicted.get("advertisements") or []:
        labels.append(f"{ad.get('advertisement_id')}: {ad.get('ad_category')} | {ad.get('brand_or_advertiser')}")
    text = " ; ".join(labels)
    if len(text) > 155:
        text = text[:152] + "..."
    draw.text((10, 625), text, fill=(40, 40, 40), font=font)
    return canvas


def main() -> int:
    panels = []
    audit = []
    for cohort, ids in CASES.items():
        manifest = json.loads((HERE / "data" / f"manifest_{cohort}.json").read_text(encoding="utf-8"))
        items = {item["image_id"]: item for item in manifest["images"]}
        human = load_gold(cohort)
        predicted = load_pred(cohort)
        for image_id in ids:
            page = Image.open(Path(manifest["image_dir"]) / items[image_id]["filename"]).convert("RGB")
            panels.append(panel(page, image_id, human[image_id], predicted[image_id]))
            audit.append({"cohort": cohort, "image_id": image_id, "human_counts": counts(human[image_id]), "predicted_counts": counts(predicted[image_id]), "predicted_business": [{"ad": ad.get("advertisement_id"), "category": ad.get("ad_category"), "brand": ad.get("brand_or_advertiser")} for ad in predicted[image_id].get("advertisements") or []]})
    sheet = Image.new("RGB", (2040, 650 * ((len(panels) + 1) // 2)), (230, 230, 230))
    for index, item in enumerate(panels):
        sheet.paste(item, ((index % 2) * 1020, (index // 2) * 650))
    outdir = HERE / "evaluation" / "visual_audit"
    outdir.mkdir(parents=True, exist_ok=True)
    sheet.save(outdir / "user_cases_final_human_vs_hybrid.jpg", quality=92)
    (outdir / "user_cases_final.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(outdir / "user_cases_final_human_vs_hybrid.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

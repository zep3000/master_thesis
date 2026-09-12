"""Smile disagreement audit for Venice runs vs human assignment 79201188."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from compare_venice_first30_strict_loose_legibility import (
    ASSIGNMENT_CODE,
    HUMAN_PATH,
    RUNS,
    Pair,
    box_iou,
    containment,
    flatten_people,
    is_legible,
    is_not_legible,
    load_jsonl,
    normalize_value,
    pair_items,
    value_at,
)


ROOT = Path(r".")
IMAGE_DIR = ROOT / "code" / "test_collection_200_difficult_joined_pages"
OUT_DIR = ROOT / "qwen_iteration" / "venice_first40_two_step" / "evaluation" / "smile_disagreements"
OUT_JSON = OUT_DIR / "smile_disagreement_breakdown.json"
OUT_CSV = OUT_DIR / "smile_disagreement_pairs.csv"
OUT_MD = OUT_DIR / "smile_disagreement_breakdown.md"


def pct(numerator: int, denominator: int) -> str:
    return "n/a" if denominator == 0 else f"{(numerator / denominator) * 100:.1f}%"


def norm_field(item: dict[str, Any], field: str) -> Any:
    return normalize_value(value_at(item, field), field.split(".")[-1])


def first30_human() -> tuple[list[str], dict[str, dict[str, Any]]]:
    export = json.loads(HUMAN_PATH.read_text(encoding="utf-8"))
    rows = sorted(
        [row for row in export["annotations"] if str(row.get("assignment_code")) == ASSIGNMENT_CODE],
        key=lambda row: row["sort_order"],
    )[:30]
    return [row["image_id"] for row in rows], {row["image_id"]: row["payload"] for row in rows}


def run_annotations(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["image_id"]: row["annotation"]
        for row in load_jsonl(path)
        if isinstance(row.get("annotation"), dict)
    }


def matched_people(
    human_by_image: dict[str, dict[str, Any]],
    llm_by_image: dict[str, dict[str, Any]],
    first30_ids: list[str],
    mode: str,
) -> list[Pair]:
    pairs: list[Pair] = []
    for image_id in first30_ids:
        if image_id not in human_by_image or image_id not in llm_by_image:
            continue
        pairs.extend(
            pair_items(
                image_id,
                flatten_people(human_by_image[image_id]),
                flatten_people(llm_by_image[image_id]),
                "people",
                mode,
            )
        )
    return pairs


def comparable_pairs(pairs: list[Pair], field: str) -> list[Pair]:
    out = []
    for pair in pairs:
        human = norm_field(pair.human.data, field)
        llm = norm_field(pair.llm.data, field)
        if human is None and llm is None:
            continue
        out.append(pair)
    return out


def confusion(pairs: list[Pair], field: str) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for pair in comparable_pairs(pairs, field):
        counts[(str(norm_field(pair.human.data, field)), str(norm_field(pair.llm.data, field)))] += 1
    return counts


def field_agreement(pairs: list[Pair], field: str) -> dict[str, Any]:
    comparable = comparable_pairs(pairs, field)
    matches = sum(norm_field(pair.human.data, field) == norm_field(pair.llm.data, field) for pair in comparable)
    return {
        "n": len(comparable),
        "matches": matches,
        "mismatches": len(comparable) - matches,
        "accuracy": matches / len(comparable) if comparable else None,
        "confusion": [
            {"human": human, "llm": llm, "n": n}
            for (human, llm), n in confusion(pairs, field).most_common()
        ],
    }


def pair_row(run: str, mode: str, pair: Pair, disagreement_field: str) -> dict[str, Any]:
    h = pair.human.data
    l = pair.llm.data
    return {
        "run": run,
        "match_mode": mode,
        "image_id": pair.image_id,
        "human_ad_id": pair.human.ad_id,
        "human_person_id": pair.human.item_id,
        "llm_ad_id": pair.llm.ad_id,
        "llm_person_id": pair.llm.item_id,
        "match_reason": pair.reason,
        "iou": pair.iou,
        "containment": pair.containment,
        "disagreement_field": disagreement_field,
        "human_smile_present": norm_field(h, "smile_present"),
        "llm_smile_present": norm_field(l, "smile_present"),
        "human_smile_intensity": norm_field(h, "smile_intensity"),
        "llm_smile_intensity": norm_field(l, "smile_intensity"),
        "human_expression_legibility": norm_field(h, "face_expression_legibility"),
        "llm_expression_legibility": norm_field(l, "face_expression_legibility"),
        "human_mouth_covered": norm_field(h, "mouth_covered"),
        "llm_mouth_covered": norm_field(l, "mouth_covered"),
        "human_orientation": norm_field(h, "face_orientation"),
        "llm_orientation": norm_field(l, "face_orientation"),
        "human_face_bbox": json.dumps(pair.human.box),
        "llm_face_bbox": json.dumps(pair.llm.box),
    }


def smile_disagreement_pairs(pairs: list[Pair]) -> list[tuple[Pair, str]]:
    out = []
    for pair in comparable_pairs(pairs, "smile_present"):
        if norm_field(pair.human.data, "smile_present") != norm_field(pair.llm.data, "smile_present"):
            out.append((pair, "smile_present"))
    for pair in comparable_pairs(pairs, "smile_intensity"):
        if norm_field(pair.human.data, "smile_intensity") != norm_field(pair.llm.data, "smile_intensity"):
            out.append((pair, "smile_intensity"))
    seen = set()
    deduped = []
    for pair, field in out:
        key = (pair.image_id, pair.human.item_id, pair.llm.item_id, field)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((pair, field))
    return deduped


def breakdown(pairs: list[Pair]) -> dict[str, Any]:
    human_zero = [p for p in pairs if is_not_legible(value_at(p.human.data, "face_expression_legibility"))]
    human_zero_llm_legible = [
        p for p in human_zero if is_legible(value_at(p.llm.data, "face_expression_legibility"))
    ]
    drop_h0 = [p for p in pairs if not is_not_legible(value_at(p.human.data, "face_expression_legibility"))]
    drop_h0_l0 = [
        p
        for p in drop_h0
        if not is_not_legible(value_at(p.llm.data, "face_expression_legibility"))
    ]
    by_human_legibility = defaultdict(list)
    by_llm_legibility = defaultdict(list)
    by_mouth_combo = defaultdict(list)
    for pair in comparable_pairs(pairs, "smile_present"):
        if norm_field(pair.human.data, "smile_present") == norm_field(pair.llm.data, "smile_present"):
            continue
        by_human_legibility[str(norm_field(pair.human.data, "face_expression_legibility"))].append(pair)
        by_llm_legibility[str(norm_field(pair.llm.data, "face_expression_legibility"))].append(pair)
        mouth_key = (
            str(norm_field(pair.human.data, "mouth_covered")),
            str(norm_field(pair.llm.data, "mouth_covered")),
        )
        by_mouth_combo[mouth_key].append(pair)
    return {
        "matched_people": len(pairs),
        "human_0_not_legible_matched": len(human_zero),
        "human_0_not_legible_marked_legible_by_llm": len(human_zero_llm_legible),
        "smile_present_all": field_agreement(pairs, "smile_present"),
        "smile_intensity_all": field_agreement(pairs, "smile_intensity"),
        "smile_present_drop_human_0": field_agreement(drop_h0, "smile_present"),
        "smile_intensity_drop_human_0": field_agreement(drop_h0, "smile_intensity"),
        "smile_present_drop_human_and_llm_0": field_agreement(drop_h0_l0, "smile_present"),
        "smile_intensity_drop_human_and_llm_0": field_agreement(drop_h0_l0, "smile_intensity"),
        "smile_present_disagreements_by_human_legibility": {
            key: len(value) for key, value in sorted(by_human_legibility.items())
        },
        "smile_present_disagreements_by_llm_legibility": {
            key: len(value) for key, value in sorted(by_llm_legibility.items())
        },
        "smile_present_disagreements_by_mouth_covered_pair": {
            f"{key[0]} -> {key[1]}": len(value)
            for key, value in sorted(by_mouth_combo.items(), key=lambda item: (-len(item[1]), item[0]))
        },
    }


def crop_for_pair(image: Image.Image, pair: Pair, pad_factor: float = 4.0) -> Image.Image:
    width, height = image.size
    boxes = [box for box in [pair.human.box, pair.llm.box] if box]
    if not boxes:
        return image.copy()
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    bw = max(x2 - x1, 0.03)
    bh = max(y2 - y1, 0.03)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    crop_w = max(bw * pad_factor, 0.12)
    crop_h = max(bh * pad_factor, 0.12)
    x1 = max(0.0, cx - crop_w / 2)
    y1 = max(0.0, cy - crop_h / 2)
    x2 = min(1.0, cx + crop_w / 2)
    y2 = min(1.0, cy + crop_h / 2)
    px = [round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)]
    return image.crop(px), (x1, y1, x2, y2)


def draw_box(draw: ImageDraw.ImageDraw, box: list[float] | None, crop_norm: tuple[float, float, float, float], size: tuple[int, int], color: tuple[int, int, int]) -> None:
    if not box:
        return
    x1, y1, x2, y2 = crop_norm
    width, height = size
    xy = [
        (box[0] - x1) / (x2 - x1) * width,
        (box[1] - y1) / (y2 - y1) * height,
        (box[2] - x1) / (x2 - x1) * width,
        (box[3] - y1) / (y2 - y1) * height,
    ]
    for offset in range(3):
        draw.rectangle([xy[0] - offset, xy[1] - offset, xy[2] + offset, xy[3] + offset], outline=color)


def make_sheet(run: str, mode: str, disagreements: list[tuple[Pair, str]]) -> Path | None:
    if not disagreements:
        return None
    try:
        font = ImageFont.truetype("arial.ttf", 15)
        small = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    tiles = []
    for pair, field in disagreements:
        path = IMAGE_DIR / f"{pair.image_id}.jpg"
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        crop, crop_norm = crop_for_pair(image, pair)
        tile_w, tile_h, label_h = 280, 250, 90
        crop.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tile_w, tile_h + label_h), "white")
        canvas.paste(crop, ((tile_w - crop.width) // 2, label_h + (tile_h - crop.height) // 2))
        draw = ImageDraw.Draw(canvas)
        offset_x = (tile_w - crop.width) // 2
        offset_y = label_h + (tile_h - crop.height) // 2
        overlay = Image.new("RGBA", (crop.width, crop.height), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        draw_box(odraw, pair.human.box, crop_norm, crop.size, (255, 30, 30))
        draw_box(odraw, pair.llm.box, crop_norm, crop.size, (30, 100, 255))
        canvas.paste(Image.alpha_composite(crop.convert("RGBA"), overlay).convert("RGB"), (offset_x, offset_y))
        hs = norm_field(pair.human.data, "smile_present")
        ls = norm_field(pair.llm.data, "smile_present")
        hi = norm_field(pair.human.data, "smile_intensity")
        li = norm_field(pair.llm.data, "smile_intensity")
        hleg = norm_field(pair.human.data, "face_expression_legibility")
        lleg = norm_field(pair.llm.data, "face_expression_legibility")
        draw.text((6, 4), f"{pair.image_id} {field}", fill="black", font=font)
        draw.text((6, 24), f"H smile={hs} int={hi}", fill=(150, 0, 0), font=small)
        draw.text((6, 42), f"L smile={ls} int={li}", fill=(0, 55, 170), font=small)
        draw.text((6, 60), f"leg H={hleg} L={lleg}", fill=(40, 40, 40), font=small)
        draw.text((6, 76), f"IoU={pair.iou:.2f} cont={pair.containment:.2f}", fill=(40, 40, 40), font=small)
        tiles.append(canvas)
    if not tiles:
        return None
    cols = 4
    gap = 12
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * 280 + (cols + 1) * gap, rows * 340 + (rows + 1) * gap), (235, 235, 235))
    for i, tile in enumerate(tiles):
        x = gap + (i % cols) * (280 + gap)
        y = gap + (i // cols) * (340 + gap)
        sheet.paste(tile, (x, y))
    out = OUT_DIR / f"{run}_{mode}_smile_disagreements.jpg"
    sheet.save(out, quality=92)
    return out


def table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    first30_ids, human_by_image = first30_human()
    result = {
        "assignment_code": ASSIGNMENT_CODE,
        "first30_image_ids": first30_ids,
        "runs": {},
    }
    csv_rows = []
    sheet_rows = []
    summary_rows = []
    confusion_rows = []
    filter_rows = []
    for run, path in RUNS.items():
        llm_all = run_annotations(path)
        llm_by_image = {image_id: llm_all[image_id] for image_id in first30_ids if image_id in llm_all}
        result["runs"][run] = {}
        for mode in ["strict", "loose"]:
            pairs = matched_people(human_by_image, llm_by_image, first30_ids, mode)
            b = breakdown(pairs)
            result["runs"][run][mode] = b
            disagreements = smile_disagreement_pairs(pairs)
            for pair, field in disagreements:
                csv_rows.append(pair_row(run, mode, pair, field))
            sheet = make_sheet(run, mode, disagreements)
            if sheet:
                sheet_rows.append([run, mode, str(sheet)])
            sp = b["smile_present_all"]
            si = b["smile_intensity_all"]
            summary_rows.append(
                [
                    run,
                    mode,
                    b["matched_people"],
                    f"{sp['matches']}/{sp['n']} ({pct(sp['matches'], sp['n'])})",
                    sp["mismatches"],
                    f"{si['matches']}/{si['n']} ({pct(si['matches'], si['n'])})",
                    si["mismatches"],
                ]
            )
            filter_rows.append(
                [
                    run,
                    mode,
                    b["human_0_not_legible_matched"],
                    b["human_0_not_legible_marked_legible_by_llm"],
                    f"{b['smile_present_drop_human_0']['matches']}/{b['smile_present_drop_human_0']['n']} ({pct(b['smile_present_drop_human_0']['matches'], b['smile_present_drop_human_0']['n'])})",
                    f"{b['smile_present_drop_human_and_llm_0']['matches']}/{b['smile_present_drop_human_and_llm_0']['n']} ({pct(b['smile_present_drop_human_and_llm_0']['matches'], b['smile_present_drop_human_and_llm_0']['n'])})",
                ]
            )
            for item in b["smile_present_all"]["confusion"]:
                if item["human"] != item["llm"]:
                    confusion_rows.append([run, mode, "smile_present", item["human"], item["llm"], item["n"]])
            for item in b["smile_intensity_all"]["confusion"]:
                if item["human"] != item["llm"]:
                    confusion_rows.append([run, mode, "smile_intensity", item["human"], item["llm"], item["n"]])

    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "# Smile disagreement breakdown",
        "",
        "Scope: first 30 human pages for assignment 79201188. Red boxes are human faces; blue boxes are LLM faces.",
        "",
        "## Summary",
        "",
        table(
            ["run", "match", "matched people", "smile_present", "present mismatches", "smile_intensity", "intensity mismatches"],
            summary_rows,
        ),
        "",
        "## Legibility Filter Effect",
        "",
        table(
            [
                "run",
                "match",
                "human 0_not_legible",
                "LLM marked those legible",
                "smile_present after drop human 0",
                "smile_present after drop human+LLM 0",
            ],
            filter_rows,
        ),
        "",
        "## Disagreement Confusions",
        "",
        table(["run", "match", "field", "human", "llm", "n"], confusion_rows),
        "",
        "## Visual Contact Sheets",
        "",
        table(["run", "match", "image"], sheet_rows),
        "",
        f"- Pair CSV: `{OUT_CSV}`",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD)
    for _run, _mode, sheet in sheet_rows:
        print(sheet)


if __name__ == "__main__":
    main()

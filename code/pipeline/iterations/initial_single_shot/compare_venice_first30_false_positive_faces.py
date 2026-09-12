"""Visualize LLM face false positives for Venice runs vs human assignment 79201188."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from compare_venice_first30_strict_loose_legibility import (
    ASSIGNMENT_CODE,
    HUMAN_PATH,
    RUNS,
    Item,
    Pair,
    flatten_people,
    load_jsonl,
    pair_items,
    value_at,
)


ROOT = Path(r".")
IMAGE_DIR = ROOT / "code" / "test_collection_200_difficult_joined_pages"
OUT_DIR = ROOT / "qwen_iteration" / "venice_first40_two_step" / "evaluation" / "false_positive_faces"
OUT_MD = OUT_DIR / "false_positive_face_breakdown.md"
OUT_CSV = OUT_DIR / "false_positive_faces.csv"
OUT_JSON = OUT_DIR / "false_positive_face_breakdown.json"


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


def item_key(item: Item) -> tuple[str | None, str | None, tuple[float, ...] | None]:
    return (item.ad_id, item.item_id, tuple(round(v, 6) for v in item.box) if item.box else None)


def unmatched_llm_people(
    image_id: str,
    human_annotation: dict[str, Any],
    llm_annotation: dict[str, Any],
    mode: str,
) -> tuple[list[Item], list[Item], list[Pair]]:
    human_people = flatten_people(human_annotation)
    llm_people = flatten_people(llm_annotation)
    pairs = pair_items(image_id, human_people, llm_people, "people", mode)
    matched_llm = {item_key(pair.llm) for pair in pairs}
    return human_people, [item for item in llm_people if item_key(item) not in matched_llm], pairs


def center(box: list[float] | None) -> tuple[float, float]:
    if not box:
        return (0.5, 0.5)
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def crop_norm_for_item(item: Item, human_people: list[Item]) -> tuple[float, float, float, float]:
    boxes = [item.box] if item.box else []
    cx, cy = center(item.box)
    nearby = []
    for human in human_people:
        if not human.box:
            continue
        hx, hy = center(human.box)
        if abs(hx - cx) <= 0.18 and abs(hy - cy) <= 0.18:
            nearby.append(human.box)
    boxes.extend(nearby[:6])
    if not boxes:
        return (0.0, 0.0, 1.0, 1.0)
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    bw = max(x2 - x1, 0.04)
    bh = max(y2 - y1, 0.04)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    crop_w = max(bw * 4.5, 0.16)
    crop_h = max(bh * 4.5, 0.16)
    return (
        max(0.0, cx - crop_w / 2),
        max(0.0, cy - crop_h / 2),
        min(1.0, cx + crop_w / 2),
        min(1.0, cy + crop_h / 2),
    )


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: list[float] | None,
    crop_norm: tuple[float, float, float, float],
    size: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
) -> None:
    if not box:
        return
    x1, y1, x2, y2 = crop_norm
    w, h = size
    if x2 <= x1 or y2 <= y1:
        return
    xy = [
        (box[0] - x1) / (x2 - x1) * w,
        (box[1] - y1) / (y2 - y1) * h,
        (box[2] - x1) / (x2 - x1) * w,
        (box[3] - y1) / (y2 - y1) * h,
    ]
    for offset in range(width):
        draw.rectangle([xy[0] - offset, xy[1] - offset, xy[2] + offset, xy[3] + offset], outline=color)


def make_sheet(run: str, mode: str, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    try:
        font = ImageFont.truetype("arial.ttf", 15)
        small = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    tiles = []
    for row in rows:
        image_id = row["image_id"]
        image_path = IMAGE_DIR / f"{image_id}.jpg"
        if not image_path.exists():
            continue
        page = Image.open(image_path).convert("RGB")
        w, h = page.size
        crop_norm = row["crop_norm"]
        crop_px = [
            round(crop_norm[0] * w),
            round(crop_norm[1] * h),
            round(crop_norm[2] * w),
            round(crop_norm[3] * h),
        ]
        crop = page.crop(crop_px)
        tile_w, tile_h, label_h = 280, 245, 95
        crop.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        for human_box in row["nearby_human_boxes"]:
            draw_box(odraw, human_box, crop_norm, crop.size, (255, 30, 30), 2)
        draw_box(odraw, row["llm_box"], crop_norm, crop.size, (30, 100, 255), 3)
        composed = Image.alpha_composite(crop.convert("RGBA"), overlay).convert("RGB")
        canvas = Image.new("RGB", (tile_w, tile_h + label_h), "white")
        x = (tile_w - composed.width) // 2
        y = label_h + (tile_h - composed.height) // 2
        canvas.paste(composed, (x, y))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 4), f"{image_id} {row['llm_person_id']}", fill="black", font=font)
        draw.text((6, 24), f"ad={row['llm_ad_id']} human nearby={len(row['nearby_human_boxes'])}", fill=(40, 40, 40), font=small)
        draw.text((6, 42), f"LLM leg={row['llm_legibility']} smile={row['llm_smile']}", fill=(0, 55, 170), font=small)
        draw.text((6, 60), f"age={row['llm_age']} gender={row['llm_gender']}", fill=(40, 40, 40), font=small)
        draw.text((6, 78), "red=human nearby blue=unmatched LLM", fill=(40, 40, 40), font=small)
        tiles.append(canvas)
    if not tiles:
        return None
    cols = 4
    gap = 12
    rows_n = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * 280 + (cols + 1) * gap, rows_n * 340 + (rows_n + 1) * gap), (235, 235, 235))
    for i, tile in enumerate(tiles):
        x = gap + (i % cols) * (280 + gap)
        y = gap + (i // cols) * (340 + gap)
        sheet.paste(tile, (x, y))
    out = OUT_DIR / f"{run}_{mode}_false_positive_faces.jpg"
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
    all_csv_rows = []
    result: dict[str, Any] = {"assignment_code": ASSIGNMENT_CODE, "first30_image_ids": first30_ids, "runs": {}}
    summary_rows = []
    sheet_rows = []
    by_page_rows = []
    by_label_rows = []
    for run, llm_path in RUNS.items():
        llm_all = run_annotations(llm_path)
        result["runs"][run] = {}
        for mode in ["strict", "loose"]:
            fp_rows = []
            human_total = llm_total = matched_total = 0
            matched_pairs_total = 0
            for image_id in first30_ids:
                if image_id not in llm_all:
                    continue
                human_people, false_positives, pairs = unmatched_llm_people(
                    image_id, human_by_image[image_id], llm_all[image_id], mode
                )
                human_total += len(human_people)
                llm_people = flatten_people(llm_all[image_id])
                llm_total += len(llm_people)
                matched_total += len(pairs)
                matched_pairs_total += len(pairs)
                for item in false_positives:
                    cx, cy = center(item.box)
                    nearby = []
                    for human in human_people:
                        if not human.box:
                            continue
                        hx, hy = center(human.box)
                        if abs(hx - cx) <= 0.18 and abs(hy - cy) <= 0.18:
                            nearby.append(human.box)
                    row = {
                        "run": run,
                        "match_mode": mode,
                        "image_id": image_id,
                        "llm_ad_id": item.ad_id,
                        "llm_person_id": item.item_id,
                        "llm_box": item.box,
                        "nearby_human_boxes": nearby[:6],
                        "crop_norm": crop_norm_for_item(item, human_people),
                        "llm_legibility": value_at(item.data, "face_expression_legibility"),
                        "llm_smile": value_at(item.data, "smile_present"),
                        "llm_age": value_at(item.data, "perceived_age"),
                        "llm_gender": value_at(item.data, "perceived_gender_presentation"),
                        "llm_role": value_at(item.data, "annotation_role"),
                    }
                    fp_rows.append(row)
                    all_csv_rows.append(
                        {
                            key: json.dumps(value) if isinstance(value, list) else value
                            for key, value in row.items()
                            if key not in {"crop_norm", "nearby_human_boxes"}
                        }
                    )
            precision = matched_total / llm_total if llm_total else None
            result["runs"][run][mode] = {
                "human_people": human_total,
                "llm_people": llm_total,
                "matched_people": matched_total,
                "false_positive_llm_people": len(fp_rows),
                "precision": precision,
                "false_positive_by_page": Counter(row["image_id"] for row in fp_rows).most_common(),
                "false_positive_legibility": Counter(str(row["llm_legibility"]) for row in fp_rows).most_common(),
                "false_positive_smile": Counter(str(row["llm_smile"]) for row in fp_rows).most_common(),
                "false_positive_role": Counter(str(row["llm_role"]) for row in fp_rows).most_common(),
            }
            summary_rows.append(
                [
                    run,
                    mode,
                    human_total,
                    llm_total,
                    matched_pairs_total,
                    len(fp_rows),
                    f"{precision * 100:.1f}%" if precision is not None else "n/a",
                ]
            )
            for page, n in result["runs"][run][mode]["false_positive_by_page"][:12]:
                by_page_rows.append([run, mode, page, n])
            for kind, counter_key in [
                ("legibility", "false_positive_legibility"),
                ("smile", "false_positive_smile"),
                ("role", "false_positive_role"),
            ]:
                for label, n in result["runs"][run][mode][counter_key]:
                    by_label_rows.append([run, mode, kind, label, n])
            sheet = make_sheet(run, mode, fp_rows)
            if sheet:
                sheet_rows.append([run, mode, str(sheet)])

    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if all_csv_rows:
        with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_csv_rows)

    lines = [
        "# False Positive Face Visualization",
        "",
        "Scope: first 30 human pages for assignment 79201188. These are LLM face boxes not matched to any human face under the selected spatial matching mode.",
        "",
        "Red boxes are nearby human face annotations. Blue boxes are unmatched LLM faces.",
        "",
        "## Precision Breakdown",
        "",
        table(["run", "match", "human faces", "LLM faces", "matched", "unmatched LLM", "precision"], summary_rows),
        "",
        "## Top Pages By Unmatched LLM Faces",
        "",
        table(["run", "match", "image_id", "unmatched LLM faces"], by_page_rows),
        "",
        "## Unmatched LLM Label Breakdown",
        "",
        table(["run", "match", "kind", "label", "n"], by_label_rows),
        "",
        "## Visual Contact Sheets",
        "",
        table(["run", "match", "image"], sheet_rows),
        "",
        f"- CSV: `{OUT_CSV}`",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD)
    for _run, _mode, sheet in sheet_rows:
        print(sheet)


if __name__ == "__main__":
    main()

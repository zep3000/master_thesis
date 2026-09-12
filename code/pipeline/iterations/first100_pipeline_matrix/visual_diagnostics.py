"""Create visual audit boards for group routing and expression disagreements.

This is an evaluation-only utility.  It reads human gold and completed pipeline
outputs, draws comparison overlays, and writes case manifests.  It is not
imported by the inference runner and never makes model calls or changes an
annotation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

import evaluate as ev


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IMAGE_DIR = ROOT / "code" / "test_collection_200_difficult_joined_pages"
OUT_ROOT = HERE / "evaluation" / "first100_frozen_v1" / "visual_audit"
PIPELINES = ["p1", "p2", "p3", "p4"]

COLORS = {
    "ad": "#42a5f5",
    "person": "#42d77d",
    "group": "#ff9f43",
    "target": "#ff3b6b",
}


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("arialbd.ttf" if bold else "arial.ttf"),
        Path("segoeuib.ttf" if bold else "segoeui.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONTS = {"title": font(30, True), "label": font(23, True), "body": font(18), "small": font(15)}


def annotations(run_name: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    gold = ev.load_gold()
    predictions = {pipeline: ev.load_predictions(run_name, pipeline) for pipeline in PIPELINES}
    return gold, predictions


def scale_box(box: list[float] | None, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not box:
        return None
    return tuple(int(round(value * (width if index % 2 == 0 else height))) for index, value in enumerate(box))


def draw_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int] | None, color: str, label: str, width: int) -> None:
    if not box:
        return
    draw.rectangle(box, outline=color, width=width)
    x1, y1, _x2, _y2 = box
    text_bbox = draw.textbbox((x1, y1), label, font=FONTS["small"])
    draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill="#111111")
    draw.text((x1, y1), label, fill=color, font=FONTS["small"])


def entity_summary(annotation: dict[str, Any]) -> str:
    ads = annotation.get("advertisements") or []
    people = sum(len(ad.get("people") or []) for ad in ads)
    groups = sum(len(ad.get("groups") or []) for ad in ads)
    bands = ",".join(str(ad.get("face_depiction_count_band") or "?") for ad in ads) or "-"
    return f"ads={len(ads)}  people={people}  groups={groups}  count-band={bands}"


def overlay_panel(image: Image.Image, annotation: dict[str, Any], heading: str, panel_w: int = 720, panel_h: int = 900) -> Image.Image:
    header_h = 84
    available_h = panel_h - header_h
    canvas = Image.new("RGB", (panel_w, panel_h), "#f6f7f9")
    fitted = ImageOps.contain(image.convert("RGB"), (panel_w - 20, available_h - 10), Image.Resampling.LANCZOS)
    xoff = (panel_w - fitted.width) // 2
    yoff = header_h + (available_h - fitted.height) // 2
    canvas.paste(fitted, (xoff, yoff))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), heading, fill="#111111", font=FONTS["label"])
    draw.text((12, 42), entity_summary(annotation), fill="#333333", font=FONTS["body"])

    for ad_i, ad in enumerate(annotation.get("advertisements") or [], 1):
        ad_box = ev.normalized_box(ad)
        if ad_box:
            mapped = [xoff + ad_box[0] * fitted.width, yoff + ad_box[1] * fitted.height,
                      xoff + ad_box[2] * fitted.width, yoff + ad_box[3] * fitted.height]
            draw_box(draw, tuple(map(int, mapped)), COLORS["ad"], f"A{ad_i}", 3)
        for person_i, person in enumerate(ad.get("people") or [], 1):
            box = ev.normalized_box(person, face=True)
            if box:
                mapped = [xoff + box[0] * fitted.width, yoff + box[1] * fitted.height,
                          xoff + box[2] * fitted.width, yoff + box[3] * fitted.height]
                draw_box(draw, tuple(map(int, mapped)), COLORS["person"], f"P{person_i}", 4)
        for group_i, group in enumerate(ad.get("groups") or [], 1):
            box = ev.normalized_box(group)
            if box:
                mapped = [xoff + box[0] * fitted.width, yoff + box[1] * fitted.height,
                          xoff + box[2] * fitted.width, yoff + box[3] * fitted.height]
                draw_box(draw, tuple(map(int, mapped)), COLORS["group"], f"G{group_i}", 5)
    return canvas


def group_case_kinds(gold: dict[str, dict[str, Any]], predictions: dict[str, dict[str, dict[str, Any]]], image_id: str) -> list[str]:
    gold_groups = len(ev.all_items(image_id, gold[image_id], "groups"))
    counts = {pipeline: len(ev.all_items(image_id, predictions[pipeline][image_id], "groups")) for pipeline in PIPELINES}
    kinds = []
    if gold_groups == 0:
        for pipeline, count in counts.items():
            if count:
                kinds.append(f"{pipeline}_false_group")
        if counts["p2"] and not counts["p4"]:
            kinds.append("p4_corrects_p2_false_group")
        if counts["p3"] and not counts["p4"]:
            kinds.append("p4_corrects_p3_false_group")
    else:
        for pipeline, count in counts.items():
            if not count:
                kinds.append(f"{pipeline}_missed_group")
        if counts["p4"] and not counts["p2"]:
            kinds.append("p4_recovers_p2_group")
        if counts["p2"] and not counts["p4"]:
            kinds.append("p2_recovers_p4_group")
    return kinds


def build_group_cases(gold: dict[str, dict[str, Any]], predictions: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    ids = sorted(set(gold).intersection(*(set(predictions[p]) for p in PIPELINES)))
    cases = []
    for image_id in ids:
        kinds = group_case_kinds(gold, predictions, image_id)
        if not kinds:
            continue
        row: dict[str, Any] = {
            "image_id": image_id,
            "kinds": kinds,
            "gold_people": len(ev.all_items(image_id, gold[image_id], "people")),
            "gold_groups": len(ev.all_items(image_id, gold[image_id], "groups")),
        }
        for pipeline in PIPELINES:
            row[f"{pipeline}_people"] = len(ev.all_items(image_id, predictions[pipeline][image_id], "people"))
            row[f"{pipeline}_groups"] = len(ev.all_items(image_id, predictions[pipeline][image_id], "groups"))
        cases.append(row)
    return cases


def group_board(case: dict[str, Any], gold: dict[str, dict[str, Any]], predictions: dict[str, dict[str, dict[str, Any]]], out_dir: Path) -> Path:
    image_id = case["image_id"]
    source = Image.open(IMAGE_DIR / f"{image_id}.jpg")
    title_h = 100
    panel_w, panel_h = 720, 900
    board = Image.new("RGB", (panel_w * 3, title_h + panel_h * 2), "#e9edf2")
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), image_id, fill="#111111", font=FONTS["title"])
    wrapped = textwrap.fill(", ".join(case["kinds"]), width=105)
    draw.text((20, 50), wrapped, fill="#444444", font=FONTS["body"])
    panels = [("GOLD", gold[image_id])] + [(pipeline.upper(), predictions[pipeline][image_id]) for pipeline in PIPELINES]
    panels.append(("ORIGINAL", {"advertisements": []}))
    for index, (heading, annotation) in enumerate(panels):
        panel = overlay_panel(source, annotation, heading, panel_w, panel_h)
        board.paste(panel, ((index % 3) * panel_w, title_h + (index // 3) * panel_h))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{image_id}.jpg"
    board.save(path, quality=92)
    return path


def matched_entity_pairs(image_id: str, human: dict[str, Any], predicted: dict[str, Any], entity: str, mode: str = "strict") -> list[ev.Pair]:
    h_ads = ev.all_items(image_id, human, "ads")
    p_ads = ev.all_items(image_id, predicted, "ads")
    ad_pairs = ev.pair_items(h_ads, p_ads, "ads", mode)
    pairs: list[ev.Pair] = []
    getter = ev.person_items if entity == "people" else ev.group_items
    for ad_pair in ad_pairs:
        pairs.extend(ev.pair_items(getter(image_id, ad_pair.human.data), getter(image_id, ad_pair.predicted.data), entity, mode))
    return pairs


def expression_cases(gold: dict[str, dict[str, Any]], predictions: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for pipeline in ["p2", "p4"]:
        for image_id in sorted(set(gold) & set(predictions[pipeline])):
            for pair in matched_entity_pairs(image_id, gold[image_id], predictions[pipeline][image_id], "people", "strict"):
                h, p = pair.human.data, pair.predicted.data
                hleg = ev.normalize_value(h.get("face_expression_legibility"), "face_expression_legibility")
                pleg = ev.normalize_value(p.get("face_expression_legibility"), "face_expression_legibility")
                box = pair.human.box
                area = ev.area(box)
                downward = hleg in ev.ORDERS["face_expression_legibility"] and pleg in ev.ORDERS["face_expression_legibility"] and ev.ORDERS["face_expression_legibility"].index(pleg) < ev.ORDERS["face_expression_legibility"].index(hleg)
                smile_lost = h.get("smile_present") in {"yes", "no"} and p.get("smile_present") is None
                gaze_lost = h.get("gaze_target") is not None and p.get("gaze_target") is None
                if not (downward or smile_lost or gaze_lost):
                    continue
                rows.append({
                    "pipeline": pipeline,
                    "image_id": image_id,
                    "human_person_id": pair.human.item_id,
                    "predicted_person_id": pair.predicted.item_id,
                    "human_box": box,
                    "predicted_box": pair.predicted.box,
                    "box_area": area,
                    "iou": pair.iou,
                    "downward_legibility": downward,
                    "smile_lost_to_null": smile_lost,
                    "gaze_lost_to_null": gaze_lost,
                    "human_legibility": hleg,
                    "predicted_legibility": pleg,
                    "human_gaze": h.get("gaze_target"),
                    "predicted_gaze": p.get("gaze_target"),
                    "human_smile": h.get("smile_present"),
                    "predicted_smile": p.get("smile_present"),
                    "human_intensity": h.get("smile_intensity"),
                    "predicted_intensity": p.get("smile_intensity"),
                })
    return rows


def crop_from_box(image: Image.Image, box: list[float], padding: float = .5) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0.0, x1 - bw * padding); x2 = min(1.0, x2 + bw * padding)
    y1 = max(0.0, y1 - bh * padding); y2 = min(1.0, y2 + bh * padding)
    return image.crop((int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)))


def expression_card(case: dict[str, Any], width: int = 560, height: int = 500) -> Image.Image:
    source = Image.open(IMAGE_DIR / f"{case['image_id']}.jpg").convert("RGB")
    crop = crop_from_box(source, case["human_box"], .85)
    visual_h = 310
    canvas = Image.new("RGB", (width, height), "#fafafa")
    fitted = ImageOps.contain(crop, (width - 20, visual_h - 20), Image.Resampling.LANCZOS)
    canvas.paste(fitted, ((width - fitted.width) // 2, 10 + (visual_h - 20 - fitted.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width - 1, height - 1), outline="#888888", width=2)
    draw.text((12, visual_h), f"{case['image_id']}  {case['pipeline'].upper()}  IoU={case['iou']:.2f}", fill="#111111", font=FONTS["label"])
    human_text = f"Gold: leg={case['human_legibility']}  gaze={case['human_gaze']}  smile={case['human_smile']}/{case['human_intensity']}"
    pred_text = f"Model: leg={case['predicted_legibility']}  gaze={case['predicted_gaze']}  smile={case['predicted_smile']}/{case['predicted_intensity']}"
    y = visual_h + 38
    for line, color in [(human_text, "#145a32"), (pred_text, "#8e2b2b")]:
        for wrapped in textwrap.wrap(line, width=61):
            draw.text((12, y), wrapped, fill=color, font=FONTS["body"])
            y += 23
        y += 3
    return canvas


def expression_boards(cases: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Prioritize large, well-matched faces while keeping several severity levels.
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    buckets = [
        lambda row: row["human_legibility"] == "3_high_legibility" and row["predicted_legibility"] in {"0_not_legible", "1_low_legibility"},
        lambda row: row["human_legibility"] == "2_moderate_legibility" and row["predicted_legibility"] == "0_not_legible",
        lambda row: row["human_legibility"] == "2_moderate_legibility" and row["predicted_legibility"] == "1_low_legibility",
        lambda row: row["human_legibility"] == "1_low_legibility" and row["predicted_legibility"] == "0_not_legible",
    ]
    for pipeline in ["p2", "p4"]:
        pipeline_rows = [row for row in cases if row["pipeline"] == pipeline and row["iou"] >= .35]
        for predicate in buckets:
            choices = sorted((row for row in pipeline_rows if predicate(row)), key=lambda row: row["box_area"], reverse=True)
            for row in choices[:4]:
                key = (pipeline, f"{row['image_id']}::{row['human_person_id']}")
                if key not in seen:
                    selected.append(row); seen.add(key)

    paths = []
    for pipeline in ["p2", "p4"]:
        rows = [row for row in selected if row["pipeline"] == pipeline][:12]
        if not rows:
            continue
        cols = 3; card_w, card_h = 560, 500
        board = Image.new("RGB", (cols * card_w, math.ceil(len(rows) / cols) * card_h + 70), "#e9edf2")
        draw = ImageDraw.Draw(board)
        draw.text((18, 15), f"{pipeline.upper()} expression-legibility critical cases", fill="#111111", font=FONTS["title"])
        for index, row in enumerate(rows):
            board.paste(expression_card(row, card_w, card_h), ((index % cols) * card_w, 70 + (index // cols) * card_h))
        path = out_dir / f"{pipeline}_expression_cases.jpg"
        board.save(path, quality=94)
        paths.append(path)
    return paths


def group_expression_cases(gold: dict[str, dict[str, Any]], predictions: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for pipeline in ["p2", "p4"]:
        for image_id in sorted(set(gold) & set(predictions[pipeline])):
            for pair in matched_entity_pairs(image_id, gold[image_id], predictions[pipeline][image_id], "groups", "strict"):
                h, p = pair.human.data, pair.predicted.data
                fields = ["expression_legibility_distribution", "dominant_gaze", "smile_prevalence", "dominant_smile_intensity"]
                if all(ev.exact_label(field, h.get(field), p.get(field)) for field in fields):
                    continue
                rows.append({
                    "pipeline": pipeline,
                    "image_id": image_id,
                    "human_group_id": pair.human.item_id,
                    "predicted_group_id": pair.predicted.item_id,
                    "human_box": pair.human.box,
                    "predicted_box": pair.predicted.box,
                    "box_area": ev.area(pair.human.box),
                    "iou": pair.iou,
                    **{f"human_{field}": h.get(field) for field in fields},
                    **{f"predicted_{field}": p.get(field) for field in fields},
                })
    return rows


def group_expression_card(case: dict[str, Any], width: int = 720, height: int = 570) -> Image.Image:
    source = Image.open(IMAGE_DIR / f"{case['image_id']}.jpg").convert("RGB")
    crop = crop_from_box(source, case["human_box"], .18)
    visual_h = 365
    canvas = Image.new("RGB", (width, height), "#fafafa")
    fitted = ImageOps.contain(crop, (width - 20, visual_h - 20), Image.Resampling.LANCZOS)
    canvas.paste(fitted, ((width - fitted.width) // 2, 10 + (visual_h - 20 - fitted.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width - 1, height - 1), outline="#888888", width=2)
    draw.text((12, visual_h), f"{case['image_id']}  {case['pipeline'].upper()}  IoU={case['iou']:.2f}", fill="#111111", font=FONTS["label"])
    human_text = (f"Gold: leg={case['human_expression_legibility_distribution']}  gaze={case['human_dominant_gaze']}  "
                  f"smiles={case['human_smile_prevalence']}/{case['human_dominant_smile_intensity']}")
    pred_text = (f"Model: leg={case['predicted_expression_legibility_distribution']}  gaze={case['predicted_dominant_gaze']}  "
                 f"smiles={case['predicted_smile_prevalence']}/{case['predicted_dominant_smile_intensity']}")
    y = visual_h + 38
    for line, color in [(human_text, "#145a32"), (pred_text, "#8e2b2b")]:
        for wrapped in textwrap.wrap(line, width=78):
            draw.text((12, y), wrapped, fill=color, font=FONTS["body"])
            y += 23
        y += 3
    return canvas


def group_expression_boards(cases: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    paths = []
    for pipeline in ["p2", "p4"]:
        rows = sorted((row for row in cases if row["pipeline"] == pipeline), key=lambda row: (row["box_area"], row["iou"]), reverse=True)[:10]
        if not rows:
            continue
        cols = 2; card_w, card_h = 720, 570
        board = Image.new("RGB", (cols * card_w, math.ceil(len(rows) / cols) * card_h + 70), "#e9edf2")
        draw = ImageDraw.Draw(board)
        draw.text((18, 15), f"{pipeline.upper()} group-expression disagreements", fill="#111111", font=FONTS["title"])
        for index, row in enumerate(rows):
            board.paste(group_expression_card(row, card_w, card_h), ((index % cols) * card_w, 70 + (index // cols) * card_h))
        path = out_dir / f"{pipeline}_group_expression_cases.jpg"
        board.save(path, quality=94)
        paths.append(path)
    return paths


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="first100_frozen_v1")
    parser.add_argument("--all-group-boards", action="store_true")
    args = parser.parse_args()
    out_dir = HERE / "evaluation" / args.run_name / "visual_audit"
    gold, predictions = annotations(args.run_name)

    group_cases = build_group_cases(gold, predictions)
    write_csv(out_dir / "group_cases.csv", [{**row, "kinds": ";".join(row["kinds"])} for row in group_cases])
    # Always render P4 errors and pages on which P4 corrects a cheaper route.
    selected_group = [row for row in group_cases if any(kind.startswith("p4_") or kind.startswith("p2_recovers_p4") for kind in row["kinds"])]
    if args.all_group_boards:
        selected_group = group_cases
    group_paths = [group_board(row, gold, predictions, out_dir / "group_boards") for row in selected_group]

    expr = expression_cases(gold, predictions)
    serializable = [{**row, "human_box": json.dumps(row["human_box"]), "predicted_box": json.dumps(row["predicted_box"])} for row in expr]
    write_csv(out_dir / "expression_cases.csv", serializable)
    expression_paths = expression_boards(expr, out_dir)

    group_expr = group_expression_cases(gold, predictions)
    group_expr_serializable = [{**row, "human_box": json.dumps(row["human_box"]), "predicted_box": json.dumps(row["predicted_box"])} for row in group_expr]
    write_csv(out_dir / "group_expression_cases.csv", group_expr_serializable)
    group_expression_paths = group_expression_boards(group_expr, out_dir)

    summary = {
        "schema_version": "first100_visual_audit_v1",
        "run_name": args.run_name,
        "group_disagreement_pages": len(group_cases),
        "rendered_group_boards": [str(path) for path in group_paths],
        "expression_disagreements": len(expr),
        "rendered_expression_boards": [str(path) for path in expression_paths],
        "group_expression_disagreements": len(group_expr),
        "rendered_group_expression_boards": [str(path) for path in group_expression_paths],
        "gold_used_for_evaluation_only": True,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

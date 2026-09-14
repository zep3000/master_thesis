from __future__ import annotations

import argparse
import csv
import html
import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from core import iter_jsonl, read_json, write_json


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local pilot review artifacts.")
    parser.add_argument("--manifest", type=Path, default=HERE / "local" / "manifest.json")
    parser.add_argument("--local-root", type=Path, default=HERE / "local")
    parser.add_argument("--per-sheet", type=int, default=10)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def label_lines(face: dict[str, Any], annotation: dict[str, Any]) -> list[str]:
    return [
        f"{face['face_id']} | {face['decade']}s | {face['detection_status']} | analyzable={face['analyzable']}",
        f"gender={annotation.get('perceived_gender_presentation')} | age={annotation.get('perceived_age')} | smile={annotation.get('smile_present')} ({annotation.get('smile_intensity')})",
        f"legibility={annotation.get('face_expression_legibility')} | depiction={annotation.get('depiction_type')} | orientation={annotation.get('face_orientation')}",
        f"gaze={annotation.get('gaze_target')} | mouth={annotation.get('mouth_covered')} | confidence={annotation.get('confidence')}",
        f"flags={', '.join(str(value) for value in annotation.get('review_flags') or []) or 'none'}",
    ]


def make_sheet(rows: list[tuple[dict[str, Any], dict[str, Any], Path]], output: Path) -> None:
    columns = 2
    tile_width = 790
    image_width = 750
    image_height = round(554 * image_width / 1536)
    text_height = 112
    tile_height = image_height + text_height + 20
    row_count = (len(rows) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, row_count * tile_height + 55), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((20, 18), "Full-issue forced-box attribute pilot", fill="black", font=font)
    for index, (face, annotation, crop_path) in enumerate(rows):
        column = index % columns
        row = index // columns
        x = column * tile_width + 20
        y = 55 + row * tile_height
        with Image.open(crop_path) as source:
            composite = source.convert("RGB")
        composite.thumbnail((image_width, image_height), Image.Resampling.LANCZOS)
        canvas.paste(composite, (x, y))
        text_y = y + image_height + 7
        for line in label_lines(face, annotation):
            for wrapped in textwrap.wrap(line, width=115) or [""]:
                draw.text((x, text_y), wrapped, fill="black", font=font)
                text_y += 14
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, "JPEG", quality=92, optimize=True)


def main() -> int:
    args = parse_args()
    manifest = read_json(args.manifest.resolve())
    local_root = args.local_root.resolve()
    results_path = local_root / "output" / "entities.jsonl"
    successful = {
        row["task_key"]: row for row in iter_jsonl(results_path) if row.get("ok")
    }
    by_id = {face["face_id"]: face for face in manifest["faces"]}
    pilot_ids = manifest["pilot"]["face_ids"]
    missing = [face_id for face_id in pilot_ids if face_id not in successful]
    if missing and not args.allow_partial:
        raise ValueError(f"pilot results are incomplete: {missing}")
    review_ids = [face_id for face_id in pilot_ids if face_id in successful]
    if not review_ids:
        raise ValueError("no successful pilot results are available")
    review_dir = local_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    joined_rows = []
    sheet_rows = []
    for face_id in review_ids:
        face = by_id[face_id]
        result = successful[face_id]
        annotation = result["model_annotation"]
        crop_path = local_root / "crops" / f"{face_id.replace('::', '__')}.jpg"
        if not crop_path.is_file():
            raise FileNotFoundError(crop_path)
        joined = {
            "face_id": face_id,
            "issue_page_identifier": face["issue_page_identifier"],
            "decade": face["decade"],
            "detection_status": face["detection_status"],
            "analyzable": face["analyzable"],
            "bbox_area_rel": face["bbox_area_rel"],
            "match_iou": face["match_iou"],
            **{key: annotation.get(key) for key in [
                "depiction_type",
                "perceived_age",
                "perceived_gender_presentation",
                "face_expression_legibility",
                "face_orientation",
                "gaze_target",
                "mouth_covered",
                "mouth_covering",
                "smile_present",
                "smile_intensity",
                "confidence",
            ]},
            "review_flags": " | ".join(str(value) for value in annotation.get("review_flags") or []),
            "crop_path": str(crop_path),
        }
        joined_rows.append(joined)
        sheet_rows.append((face, annotation, crop_path))
    csv_path = review_dir / "pilot_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(joined_rows[0]))
        writer.writeheader()
        writer.writerows(joined_rows)
    sheet_paths = []
    for offset in range(0, len(sheet_rows), args.per_sheet):
        sheet_path = review_dir / f"pilot_review_{offset // args.per_sheet + 1:02d}.jpg"
        make_sheet(sheet_rows[offset : offset + args.per_sheet], sheet_path)
        sheet_paths.append(sheet_path)
    category_counts: dict[str, dict[str, dict[str, int]]] = {}
    for field in [
        "perceived_gender_presentation",
        "perceived_age",
        "smile_present",
        "face_expression_legibility",
        "depiction_type",
        "face_orientation",
    ]:
        by_status: dict[str, Counter[str]] = defaultdict(Counter)
        for row in joined_rows:
            by_status[str(row["detection_status"])][str(row[field])] += 1
        category_counts[field] = {
            status: dict(sorted(counts.items())) for status, counts in sorted(by_status.items())
        }
    summary = {
        "schema_version": "full_issue_face_audit_pilot_review_v1",
        "pilot_size": len(joined_rows),
        "planned_pilot_size": len(pilot_ids),
        "partial": bool(missing),
        "missing_face_ids": missing,
        "csv": str(csv_path),
        "sheets": [str(path) for path in sheet_paths],
        "counts_by_detection_status": dict(Counter(row["detection_status"] for row in joined_rows)),
        "category_counts": category_counts,
        "mean_confidence": sum(float(row["confidence"]) for row in joined_rows) / len(joined_rows),
        "review_flagged": sum(bool(row["review_flags"]) for row in joined_rows),
    }
    write_json(review_dir / "pilot_summary.json", summary)
    html_path = review_dir / "pilot_review.html"
    html_rows = []
    for row in joined_rows:
        relative_crop = Path(row["crop_path"]).relative_to(review_dir.parent).as_posix()
        html_rows.append(
            "<tr>"
            f"<td><img src='../{html.escape(relative_crop)}' alt='face composite'></td>"
            + "".join(
                f"<td>{html.escape(str(row[column]))}</td>"
                for column in [
                    "face_id",
                    "detection_status",
                    "analyzable",
                    "perceived_gender_presentation",
                    "perceived_age",
                    "smile_present",
                    "face_expression_legibility",
                    "depiction_type",
                    "face_orientation",
                    "confidence",
                    "review_flags",
                ]
            )
            + "</tr>"
        )
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Full-issue face audit pilot</title>"
        "<style>body{font:14px system-ui;margin:20px}table{border-collapse:collapse}"
        "th,td{border:1px solid #bbb;padding:6px;vertical-align:top}"
        "th{position:sticky;top:0;background:#fff}img{width:620px;height:auto}</style>"
        "<h1>Full-issue forced-box attribute pilot</h1><table><thead><tr>"
        + "".join(
            f"<th>{html.escape(column)}</th>"
            for column in [
                "composite",
                "face_id",
                "detection_status",
                "analyzable",
                "gender",
                "age",
                "smile",
                "legibility",
                "depiction",
                "orientation",
                "confidence",
                "flags",
            ]
        )
        + "</tr></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table>",
        encoding="utf-8",
    )
    print(json.dumps({**summary, "html": str(html_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

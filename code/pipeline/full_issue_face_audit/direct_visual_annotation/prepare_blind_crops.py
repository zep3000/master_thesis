from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
AUDIT_DIR = HERE.parent
sys.path.insert(0, str(AUDIT_DIR))

from core import expand_box, read_json, render_pdf_page, to_pixels, write_json  # noqa: E402


VISIBLE_FIELDS = [
    "blind_id",
    "crop_path",
    "sheet_path",
    "sheet_index",
    "sheet_position",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare blind extended face crops and contact sheets for local visual annotation."
    )
    parser.add_argument("--manifest", type=Path, default=AUDIT_DIR / "local" / "manifest.json")
    parser.add_argument("--pdftoppm", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--crop-scale", type=float, default=2.35)
    parser.add_argument("--tile-size", type=int, default=320)
    parser.add_argument("--sheet-cols", type=int, default=4)
    parser.add_argument("--sheet-rows", type=int, default=5)
    parser.add_argument("--local-root", type=Path, default=AUDIT_DIR / "local" / "direct_visual_annotation")
    return parser.parse_args()


def load_faces(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(manifest_path.resolve())
    if manifest.get("schema_version") != "full_issue_face_audit_manifest_v1":
        raise ValueError("unsupported manifest schema")
    faces = sorted(manifest["faces"], key=lambda face: face["face_id"])
    if len(faces) != 1599:
        raise ValueError(f"expected 1599 faces, got {len(faces)}")
    return manifest, faces


def crop_extended_face(page_path: Path, box_1000: list[int], output_path: Path, scale: float) -> None:
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(page_path) as source:
        page = source.convert("RGB")
    face_pixels = to_pixels(box_1000, page.size)
    crop = page.crop(expand_box(face_pixels, page.size, scale))
    crop.save(output_path, "JPEG", quality=94, optimize=True)


def make_tile(crop_path: Path, blind_id: str, tile_size: int, font: ImageFont.ImageFont) -> Image.Image:
    label_height = 30
    tile = Image.new("RGB", (tile_size, tile_size + label_height), "white")
    with Image.open(crop_path) as source:
        image = source.convert("RGB")
    image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
    x = (tile_size - image.width) // 2
    y = label_height + (tile_size - image.height) // 2
    tile.paste(image, (x, y))
    draw = ImageDraw.Draw(tile)
    draw.text((8, 8), blind_id, fill="black", font=font)
    draw.rectangle((0, 0, tile.width - 1, tile.height - 1), outline=(210, 210, 210), width=1)
    return tile


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.crop_scale <= 1:
        raise ValueError("--crop-scale must be greater than 1")
    if args.sheet_cols <= 0 or args.sheet_rows <= 0:
        raise ValueError("--sheet-cols and --sheet-rows must be positive")
    manifest, faces = load_faces(args.manifest)
    local_root = args.local_root.resolve()
    page_root = local_root / "pages"
    crop_root = local_root / "crops"
    sheet_root = local_root / "sheets"
    sheet_root.mkdir(parents=True, exist_ok=True)
    per_sheet = args.sheet_cols * args.sheet_rows
    font = ImageFont.load_default()
    blind_rows: list[dict[str, Any]] = []
    visible_rows: list[dict[str, Any]] = []

    for index, face in enumerate(faces, start=1):
        blind_id = f"blind_{index:06d}"
        page_path = page_root / f"page_{index:06d}.jpg"
        crop_path = crop_root / f"{blind_id}.jpg"
        pdf_path = Path(manifest["pdfs"][face["issue_id"]]["path"])
        render_pdf_page(args.pdftoppm.resolve(), pdf_path, face["pdf_page_index"], page_path, args.dpi)
        crop_extended_face(page_path, face["bbox_1000"], crop_path, args.crop_scale)
        sheet_index = math.ceil(index / per_sheet)
        sheet_path = sheet_root / f"sheet_{sheet_index:03d}.jpg"
        sheet_position = ((index - 1) % per_sheet) + 1
        row = {
            "blind_id": blind_id,
            "face_id": face["face_id"],
            "crop_path": str(crop_path),
            "sheet_path": str(sheet_path),
            "sheet_index": sheet_index,
            "sheet_position": sheet_position,
        }
        blind_rows.append(row)
        visible_rows.append({field: row[field] for field in VISIBLE_FIELDS})

    sheet_count = math.ceil(len(blind_rows) / per_sheet)
    for sheet_index in range(1, sheet_count + 1):
        rows = [
            row for row in blind_rows
            if row["sheet_index"] == sheet_index
        ]
        sheet = Image.new(
            "RGB",
            (args.sheet_cols * args.tile_size, args.sheet_rows * (args.tile_size + 30)),
            "white",
        )
        for row in rows:
            offset = row["sheet_position"] - 1
            col = offset % args.sheet_cols
            sheet_row = offset // args.sheet_cols
            tile = make_tile(Path(row["crop_path"]), row["blind_id"], args.tile_size, font)
            sheet.paste(tile, (col * args.tile_size, sheet_row * (args.tile_size + 30)))
        sheet.save(sheet_root / f"sheet_{sheet_index:03d}.jpg", "JPEG", quality=94, optimize=True)

    write_json(
        local_root / "blind_manifest.json",
        {
            "schema_version": "direct_visual_annotation_blind_manifest_v1",
            "source_manifest": str(args.manifest.resolve()),
            "face_count": len(blind_rows),
            "crop_scale": args.crop_scale,
            "dpi": args.dpi,
            "per_sheet": per_sheet,
            "sheet_count": sheet_count,
            "rows": blind_rows,
        },
    )
    write_csv(local_root / "visible_sheet_manifest.csv", visible_rows, VISIBLE_FIELDS)
    raw_path = local_root / "raw_annotations.jsonl"
    raw_path.touch(exist_ok=True)
    print(f"prepared {len(blind_rows)} blind crops")
    print(f"wrote {sheet_count} sheets to {sheet_root}")
    print(f"raw annotation target: {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

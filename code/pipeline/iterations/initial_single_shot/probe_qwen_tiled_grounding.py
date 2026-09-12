#!/usr/bin/env python3
"""Test one-request multi-image tiled grounding with Qwen through OpenRouter."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import time
from pathlib import Path
from typing import Any

from PIL import Image

from probe_face_grounding_providers import greedy_ious, load_gold_boxes, median, normalize_box
from run_single_shot_first20 import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_KEY_FILE,
    DEFAULT_MODEL,
    build_openrouter_client,
    iso_now,
    load_api_key,
    resolve_path,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD = ROOT / "annotation_results" / "test_collection_200_difficult_joined_v1_2026-08-02.json"
DEFAULT_OUTPUT = ROOT / "qwen_iteration" / "output" / "tiled_grounding_probe_1955-0618-0040.json"
DEFAULT_PROVIDERS = ["deepinfra/bf16", "parasail/bf16", "venice/fp8"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", default="1955-0618-0040")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider-tag", action="append", dest="provider_tags")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--request-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--tile-only", action="store_true", help="Test tiles only; default tests tiles and overview+tiles.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = resolve_path(args.image_dir) / f"{args.image_id}.jpg"
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    tiles = make_tiles(image, args.rows, args.columns, args.overlap)
    gold_boxes = load_gold_boxes(resolve_path(args.gold), args.image_id)
    client = build_openrouter_client(
        api_key=load_api_key(resolve_path(args.api_key_file)),
        base_url="https://openrouter.ai/api/v1",
        app_name="economist-ad-face-qwen-tiled-grounding-probe",
        http_referer=None,
        timeout=args.request_timeout_seconds,
    )
    configurations = [False] if args.tile_only else [False, True]
    results = []
    for provider_tag in args.provider_tags or DEFAULT_PROVIDERS:
        for include_overview in configurations:
            results.append(
                run_probe(
                    client=client,
                    model=args.model,
                    provider_tag=provider_tag,
                    image=image,
                    tiles=tiles,
                    include_overview=include_overview,
                    gold_boxes=gold_boxes,
                    page_size=(width, height),
                )
            )
    payload = {
        "schema_version": "qwen_tiled_grounding_probe_v1",
        "generated_at": iso_now(),
        "image_id": args.image_id,
        "image_path": str(image_path),
        "page_size": [width, height],
        "rows": args.rows,
        "columns": args.columns,
        "overlap": args.overlap,
        "tile_bounds_pixels": {tile["tile_id"]: tile["bounds"] for tile in tiles},
        "gold_boxes": gold_boxes,
        "results": results,
    }
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_results(results)
    print(f"wrote {output}")
    return 0


def make_tiles(image: Image.Image, rows: int, columns: int, overlap: float) -> list[dict[str, Any]]:
    if rows < 1 or columns < 1 or not 0 <= overlap < 0.5:
        raise ValueError("rows/columns must be positive and overlap must be in [0, 0.5)")
    width, height = image.size
    cell_width = math.ceil(width / columns)
    cell_height = math.ceil(height / rows)
    pad_x = round(cell_width * overlap / 2)
    pad_y = round(cell_height * overlap / 2)
    tiles = []
    for row in range(rows):
        for column in range(columns):
            x1 = max(0, column * cell_width - pad_x)
            y1 = max(0, row * cell_height - pad_y)
            x2 = min(width, (column + 1) * cell_width + pad_x)
            y2 = min(height, (row + 1) * cell_height + pad_y)
            tile_id = f"r{row + 1}c{column + 1}"
            tiles.append({"tile_id": tile_id, "bounds": [x1, y1, x2, y2], "image": image.crop((x1, y1, x2, y2))})
    return tiles


def run_probe(
    client: Any,
    model: str,
    provider_tag: str,
    image: Image.Image,
    tiles: list[dict[str, Any]],
    include_overview: bool,
    gold_boxes: list[list[float]],
    page_size: tuple[int, int],
) -> dict[str, Any]:
    started = time.time()
    result: dict[str, Any] = {
        "provider_tag": provider_tag,
        "configuration": "overview_plus_tiles" if include_overview else "tiles_only",
    }
    try:
        messages = build_tiled_messages(image, tiles, include_overview)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2500,
            temperature=0,
            response_format={"type": "json_object"},
            extra_body={
                "provider": {"order": [provider_tag], "allow_fallbacks": False, "require_parameters": True},
                "reasoning": {"effort": "none", "exclude": True},
            },
        )
        response_text = response.choices[0].message.content
        parsed = json.loads(response_text)
        mapped = map_tile_detections(parsed, tiles, page_size)
        merged = merge_detections(mapped, threshold=0.30)
        raw_scores = greedy_ious(gold_boxes, [item["box"] for item in mapped])
        merged_scores = greedy_ious(gold_boxes, [item["box"] for item in merged])
        result.update(
            {
                "ok": True,
                "response_text": response_text,
                "raw_detections": mapped,
                "merged_detections": merged,
                "raw_metrics": metrics(raw_scores, len(mapped)),
                "merged_metrics": metrics(merged_scores, len(merged)),
                "usage": response.usage.model_dump() if response.usage else None,
            }
        )
    except Exception as exc:
        result.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    result["elapsed_seconds"] = round(time.time() - started, 3)
    return result


def build_tiled_messages(image: Image.Image, tiles: list[dict[str, Any]], include_overview: bool) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if include_overview:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(image)}})
        content.append(
            {
                "type": "text",
                "text": "The preceding image is the complete page overview for context only. Do not return coordinates for it.",
            }
        )
    for tile in tiles:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(tile["image"])}})
        content.append({"type": "text", "text": f"The preceding image is tile {tile['tile_id']}."})
    tile_ids = ", ".join(tile["tile_id"] for tile in tiles)
    content.append(
        {
            "type": "text",
            "text": (
                "Locate every visible human face independently in each named tile. Return only one JSON object: "
                '{"tiles":[{"tile_id":"r1c1","faces":[{"bbox_2d":[x1,y1,x2,y2]}]}]}. '
                "Coordinates must be integers normalized 0..1000 relative to that tile, not the complete page. "
                "For every face use the smallest box covering visible hair, ears, forehead, cheeks, chin, beard or "
                "moustache, and glasses. Exclude neck, shoulders, captions, and empty background. Include only faces "
                "visibly present; do not infer regular patterns or face-like texture. Overlap between tiles is expected, "
                f"so the same face may appear in more than one tile. Return exactly these tile ids: {tile_ids}."
            ),
        }
    )
    return [
        {"role": "system", "content": "You are a precise visual grounding system. Return only the requested JSON."},
        {"role": "user", "content": content},
    ]


def image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def map_tile_detections(parsed: dict[str, Any], tiles: list[dict[str, Any]], page_size: tuple[int, int]) -> list[dict[str, Any]]:
    tile_map = {tile["tile_id"]: tile for tile in tiles}
    page_width, page_height = page_size
    output = []
    for tile_result in parsed.get("tiles", []):
        if not isinstance(tile_result, dict):
            continue
        tile_id = str(tile_result.get("tile_id") or "")
        tile = tile_map.get(tile_id)
        if not tile:
            continue
        x1, y1, x2, y2 = tile["bounds"]
        tile_width, tile_height = x2 - x1, y2 - y1
        for face in tile_result.get("faces", []):
            raw_box = face.get("bbox_2d") if isinstance(face, dict) else face
            local = normalize_box(raw_box)
            if not local:
                continue
            full = [
                (x1 + local[0] * tile_width) / page_width,
                (y1 + local[1] * tile_height) / page_height,
                (x1 + local[2] * tile_width) / page_width,
                (y1 + local[3] * tile_height) / page_height,
            ]
            center_x = (local[0] + local[2]) / 2
            center_y = (local[1] + local[3]) / 2
            margin = min(center_x, center_y, 1 - center_x, 1 - center_y)
            output.append({"tile_id": tile_id, "tile_box": local, "box": full, "tile_center_margin": margin})
    return output


def merge_detections(detections: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    remaining = sorted(detections, key=lambda item: item["tile_center_margin"], reverse=True)
    kept = []
    while remaining:
        best = remaining.pop(0)
        cluster = [best]
        others = []
        for candidate in remaining:
            if box_iou(best["box"], candidate["box"]) >= threshold:
                cluster.append(candidate)
            else:
                others.append(candidate)
        remaining = others
        weight_sum = sum(max(0.01, item["tile_center_margin"]) for item in cluster)
        fused = [
            sum(item["box"][coordinate] * max(0.01, item["tile_center_margin"]) for item in cluster) / weight_sum
            for coordinate in range(4)
        ]
        kept.append(
            {
                "box": fused,
                "source_tiles": [item["tile_id"] for item in cluster],
                "source_count": len(cluster),
            }
        )
    return kept


def box_iou(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def metrics(scores: list[float], prediction_count: int) -> dict[str, Any]:
    return {
        "prediction_count": prediction_count,
        "matched_count": len(scores),
        "iou_ge_0_25": sum(score >= 0.25 for score in scores),
        "iou_ge_0_50": sum(score >= 0.50 for score in scores),
        "iou_ge_0_75": sum(score >= 0.75 for score in scores),
        "median_iou": median(scores),
        "ious": scores,
    }


def print_results(results: list[dict[str, Any]]) -> None:
    for result in results:
        usage = result.get("usage") or {}
        merged = result.get("merged_metrics") or {}
        print(
            result["provider_tag"],
            result["configuration"],
            f"ok={result['ok']}",
            f"tokens={usage.get('prompt_tokens')}",
            f"boxes={merged.get('prediction_count')}",
            f"iou>=.5={merged.get('iou_ge_0_50')}",
            f"median_iou={merged.get('median_iou')}",
            f"elapsed={result['elapsed_seconds']}",
        )


if __name__ == "__main__":
    raise SystemExit(main())

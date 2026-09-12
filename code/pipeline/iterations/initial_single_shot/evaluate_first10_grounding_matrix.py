#!/usr/bin/env python3
"""Evaluate full-page and tiled Qwen face grounding on the first ten pages."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from probe_face_grounding_providers import normalize_box
from probe_qwen_tiled_grounding import (
    box_iou,
    build_tiled_messages,
    image_data_url,
    make_tiles,
    map_tile_detections,
    merge_detections,
)
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
DEFAULT_MANIFEST = ROOT / "annotation_app_v2" / "data" / "test_collection_200_difficult_joined_hosted_manifest.json"
DEFAULT_GOLD = ROOT / "annotation_results" / "test_collection_200_difficult_joined_v1_2026-08-02.json"
DEFAULT_OUTPUT = ROOT / "qwen_iteration" / "output" / "first10_grounding_matrix.json"
FULL_PROVIDERS = ["deepinfra/bf16", "parasail/bf16", "venice/fp8"]
TILED_PROVIDERS = ["deepinfra/bf16", "venice/fp8"]
THRESHOLDS = (0.25, 0.50, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = resolve_path(args.manifest)
    gold_path = resolve_path(args.gold)
    image_dir = resolve_path(args.image_dir)
    output_path = resolve_path(args.output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_rows = manifest["images"][: args.limit]
    gold_payload = json.loads(gold_path.read_text(encoding="utf-8"))
    gold_by_image = {row["image_id"]: gold_for_image(gold_payload, row["image_id"]) for row in image_rows}

    if args.resume and output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        results = payload.get("results", [])
    else:
        results = []
    refresh_metrics(results, gold_by_image)
    if args.summarize_only:
        payload = build_payload(args, image_rows, gold_by_image, results)
        payload["summary"] = summarize(results, gold_by_image)
        payload["summary_individual_only"] = summarize(
            results,
            gold_by_image,
            {image_id for image_id, item in gold_by_image.items() if item["evaluation_class"] == "individual_only"},
        )
        write_payload(output_path, payload)
        print_summary(payload["summary"])
        return 0
    completed = {(row["image_id"], row["provider_tag"], row["configuration"]) for row in results if row.get("ok")}

    client = build_openrouter_client(
        api_key=load_api_key(resolve_path(args.api_key_file)),
        base_url="https://openrouter.ai/api/v1",
        app_name="economist-ad-face-first10-grounding-matrix",
        http_referer=None,
        timeout=args.request_timeout_seconds,
    )
    configurations = [
        ("full_page", FULL_PROVIDERS),
        ("tiles_only", TILED_PROVIDERS),
        ("overview_plus_tiles", TILED_PROVIDERS),
    ]
    for image_row in image_rows:
        image_id = image_row["image_id"]
        image_path = image_dir / image_row["filename"]
        image = Image.open(image_path).convert("RGB")
        tiles = make_tiles(image, args.rows, args.columns, args.overlap)
        for configuration, providers in configurations:
            for provider_tag in providers:
                key = (image_id, provider_tag, configuration)
                if key in completed:
                    continue
                print(f"START {image_id} {provider_tag} {configuration}", flush=True)
                result = run_condition(
                    client=client,
                    model=args.model,
                    image_id=image_id,
                    image=image,
                    tiles=tiles,
                    provider_tag=provider_tag,
                    configuration=configuration,
                    gold_boxes=gold_by_image[image_id]["boxes"],
                )
                results.append(result)
                payload = build_payload(args, image_rows, gold_by_image, results)
                write_payload(output_path, payload)
                print_result(result)

    payload = build_payload(args, image_rows, gold_by_image, results)
    payload["summary"] = summarize(results, gold_by_image)
    payload["summary_individual_only"] = summarize(
        results,
        gold_by_image,
        {image_id for image_id, item in gold_by_image.items() if item["evaluation_class"] == "individual_only"},
    )
    write_payload(output_path, payload)
    print_summary(payload["summary"])
    print(f"wrote {output_path}")
    return 0


def run_condition(
    client: Any,
    model: str,
    image_id: str,
    image: Image.Image,
    tiles: list[dict[str, Any]],
    provider_tag: str,
    configuration: str,
    gold_boxes: list[list[float]],
) -> dict[str, Any]:
    started = time.time()
    result: dict[str, Any] = {
        "image_id": image_id,
        "provider_tag": provider_tag,
        "configuration": configuration,
        "gold_count": len(gold_boxes),
    }
    try:
        if configuration == "full_page":
            messages = build_full_page_messages(image)
        else:
            messages = build_tiled_messages(image, tiles, configuration == "overview_plus_tiles")
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
        if configuration == "full_page":
            raw_detections = [{"box": box} for box in parse_full_page_boxes(parsed)]
            variants = {"predictions": [item["box"] for item in raw_detections]}
        else:
            raw_detections = map_tile_detections(parsed, tiles, image.size)
            variants = {
                "raw": [item["box"] for item in raw_detections],
                "merged_iou_0.30": [item["box"] for item in merge_detections(raw_detections, 0.30)],
                "merged_iou_0.50": [item["box"] for item in merge_detections(raw_detections, 0.50)],
            }
        result.update(
            {
                "ok": True,
                "response_text": response_text,
                "raw_detections": raw_detections,
                "variants": {
                    name: {"boxes": boxes, "metrics": score_boxes(gold_boxes, boxes)}
                    for name, boxes in variants.items()
                },
                "usage": response.usage.model_dump() if response.usage else None,
            }
        )
    except Exception as exc:
        result.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    result["elapsed_seconds"] = round(time.time() - started, 3)
    return result


def build_full_page_messages(image: Image.Image) -> list[dict[str, Any]]:
    prompt = (
        "Locate every visible human face in this complete page. Return only one JSON object of the form "
        '{"faces":[{"bbox_2d":[x1,y1,x2,y2]}]}. Coordinates must be integers normalized to a 0..1000 '
        "grid relative to the complete image. For every face use the smallest box covering visible hair, ears, "
        "forehead, cheeks, chin, beard or moustache, and glasses. Exclude neck, shoulders, captions, and empty "
        "background. Include only faces visibly present; do not infer regular patterns or face-like texture."
    )
    return [
        {"role": "system", "content": "You are a precise visual grounding system. Return only the requested JSON."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url(image)}},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def parse_full_page_boxes(parsed: Any) -> list[list[float]]:
    if isinstance(parsed, list):
        candidates = parsed
    elif isinstance(parsed, dict):
        candidates = parsed.get("faces", [])
    else:
        candidates = []
    boxes = []
    for item in candidates:
        raw = item.get("bbox_2d") if isinstance(item, dict) else item
        box = normalize_box(raw)
        if box:
            boxes.append(box)
    return boxes


def gold_for_image(payload: dict[str, Any], image_id: str) -> dict[str, Any]:
    candidates = [
        row
        for row in payload["annotations"]
        if row.get("image_id") == image_id and row.get("assignment_status") == "done"
    ]
    if not candidates:
        raise ValueError(f"No completed human row for {image_id}")
    row = next((item for item in candidates if item.get("status") == "complete"), candidates[-1])
    advertisements = (row.get("payload") or {}).get("advertisements", [])
    boxes = [
        person["face_bbox"]
        for ad in advertisements
        for person in ad.get("people", [])
        if normalize_box(person.get("face_bbox"))
    ]
    group_count = sum(len(ad.get("groups", [])) for ad in advertisements)
    if row.get("status") != "complete":
        evaluation_class = "ineligible"
    elif group_count and boxes:
        evaluation_class = "mixed_individual_and_group"
    elif group_count:
        evaluation_class = "group_only"
    else:
        evaluation_class = "individual_only"
    return {
        "status": row.get("status"),
        "evaluation_class": evaluation_class,
        "boxes": boxes,
        "individual_count": len(boxes),
        "group_count": group_count,
    }


def score_boxes(gold: list[list[float]], predicted: list[list[float]]) -> dict[str, Any]:
    pairs = sorted(
        (
            (box_iou(gold_box, predicted_box), gold_index, predicted_index)
            for gold_index, gold_box in enumerate(gold)
            for predicted_index, predicted_box in enumerate(predicted)
        ),
        reverse=True,
    )
    used_gold: set[int] = set()
    used_predicted: set[int] = set()
    matched_ious = []
    for iou, gold_index, predicted_index in pairs:
        if gold_index in used_gold or predicted_index in used_predicted:
            continue
        used_gold.add(gold_index)
        used_predicted.add(predicted_index)
        matched_ious.append(iou)
    metrics: dict[str, Any] = {
        "gold_count": len(gold),
        "prediction_count": len(predicted),
        "count_error": len(predicted) - len(gold),
        "matched_ious": matched_ious,
    }
    for threshold in THRESHOLDS:
        tp = sum(iou >= threshold for iou in matched_ious)
        suffix = threshold_suffix(threshold)
        metrics[f"tp_{suffix}"] = tp
        metrics[f"precision_{suffix}"] = round(tp / len(predicted), 4) if predicted else (1.0 if not gold else 0.0)
        metrics[f"recall_{suffix}"] = round(tp / len(gold), 4) if gold else (1.0 if not predicted else 0.0)
    return metrics


def selected_variant(result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    name = "predictions" if result["configuration"] == "full_page" else "merged_iou_0.30"
    return name, result["variants"][name]["metrics"]


def summarize(
    results: list[dict[str, Any]],
    gold_by_image: dict[str, dict[str, Any]],
    image_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        if result.get("ok") and (image_ids is None or result["image_id"] in image_ids):
            groups.setdefault((result["provider_tag"], result["configuration"]), []).append(result)
    output = []
    for (provider_tag, configuration), rows in sorted(groups.items()):
        variant_name = "predictions" if configuration == "full_page" else "merged_iou_0.30"
        metrics = [row["variants"][variant_name]["metrics"] for row in rows]
        total_gold = sum(len(gold_by_image[row["image_id"]]["boxes"]) for row in rows)
        predictions = sum(item["prediction_count"] for item in metrics)
        summary: dict[str, Any] = {
            "provider_tag": provider_tag,
            "configuration": configuration,
            "variant": variant_name,
            "pages_ok": len(rows),
            "gold_count": total_gold,
            "prediction_count": predictions,
            "mean_absolute_count_error": round(sum(abs(item["count_error"]) for item in metrics) / len(metrics), 3),
            "prompt_tokens": sum((row.get("usage") or {}).get("prompt_tokens") or 0 for row in rows),
            "completion_tokens": sum((row.get("usage") or {}).get("completion_tokens") or 0 for row in rows),
            "cost": round(sum(float((row.get("usage") or {}).get("cost") or 0) for row in rows), 6),
        }
        for threshold in THRESHOLDS:
            suffix = threshold_suffix(threshold)
            tp = sum(item[f"tp_{suffix}"] for item in metrics)
            summary[f"tp_{suffix}"] = tp
            summary[f"precision_{suffix}"] = round(tp / predictions, 4) if predictions else 0.0
            summary[f"recall_{suffix}"] = round(tp / total_gold, 4) if total_gold else 0.0
        output.append(summary)
    return output


def threshold_suffix(threshold: float) -> str:
    return str(round(threshold * 100))


def refresh_metrics(results: list[dict[str, Any]], gold_by_image: dict[str, dict[str, Any]]) -> None:
    """Re-score stored boxes so interrupted runs remain compatible with metric changes."""
    for result in results:
        if not result.get("ok") or result.get("image_id") not in gold_by_image:
            continue
        gold_boxes = gold_by_image[result["image_id"]]["boxes"]
        for variant in (result.get("variants") or {}).values():
            boxes = variant.get("boxes") or []
            variant["metrics"] = score_boxes(gold_boxes, boxes)


def build_payload(
    args: argparse.Namespace,
    image_rows: list[dict[str, Any]],
    gold_by_image: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "qwen_first10_grounding_matrix_v1",
        "updated_at": iso_now(),
        "model": args.model,
        "image_ids": [row["image_id"] for row in image_rows],
        "gold": gold_by_image,
        "tiling": {"rows": args.rows, "columns": args.columns, "overlap": args.overlap},
        "full_providers": FULL_PROVIDERS,
        "tiled_providers": TILED_PROVIDERS,
        "results": results,
    }


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_result(result: dict[str, Any]) -> None:
    if not result.get("ok"):
        print(f"ERROR {result['image_id']} {result['provider_tag']} {result['configuration']}: {result['error']}", flush=True)
        return
    name, metrics = selected_variant(result)
    usage = result.get("usage") or {}
    print(
        f"DONE {result['image_id']} {result['provider_tag']} {result['configuration']} {name} "
        f"gold={metrics['gold_count']} pred={metrics['prediction_count']} tp50={metrics['tp_50']} "
        f"tokens={usage.get('prompt_tokens')} elapsed={result['elapsed_seconds']}",
        flush=True,
    )


def print_summary(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(
            row["provider_tag"],
            row["configuration"],
            f"pages={row['pages_ok']}",
            f"pred={row['prediction_count']}",
            f"P50={row['precision_50']}",
            f"R50={row['recall_50']}",
            f"MAE={row['mean_absolute_count_error']}",
            f"tokens={row['prompt_tokens']}",
            f"cost={row['cost']}",
        )


if __name__ == "__main__":
    raise SystemExit(main())

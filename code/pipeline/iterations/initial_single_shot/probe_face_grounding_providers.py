#!/usr/bin/env python3
"""Compare provider preprocessing and Qwen face grounding on one gold page."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from run_single_shot_first20 import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_KEY_FILE,
    DEFAULT_MODEL,
    build_messages,
    build_openrouter_client,
    image_to_data_url,
    iso_now,
    load_api_key,
    resolve_path,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD = ROOT / "annotation_results" / "test_collection_200_difficult_joined_v1_2026-08-02.json"
DEFAULT_OUTPUT = ROOT / "qwen_iteration" / "output" / "provider_face_grounding_probe_1955-0618-0040.json"
DEFAULT_PROVIDERS = ["deepinfra/bf16", "parasail/bf16", "venice/fp8", "siliconflow/fp8"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", default="1955-0618-0040")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider-tag", action="append", dest="provider_tags")
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--reparse-only",
        action="store_true",
        help="Reparse response_text already stored in --output without making API calls.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = resolve_path(args.output)
    if args.reparse_only:
        payload = json.loads(output.read_text(encoding="utf-8"))
        gold_boxes = payload["gold_boxes"]
        for result in payload.get("results", []):
            if not result.get("response_text"):
                continue
            parsed = json.loads(result["response_text"])
            boxes = parse_face_boxes(parsed)
            scores = greedy_ious(gold_boxes, boxes)
            result.update(
                {
                    "boxes": boxes,
                    "ious": scores,
                    "iou_ge_0_25": sum(value >= 0.25 for value in scores),
                    "iou_ge_0_50": sum(value >= 0.50 for value in scores),
                    "median_iou": median(scores),
                }
            )
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print_results(payload["results"])
        print(f"reparsed {output}")
        return 0

    image_path = resolve_path(args.image_dir) / f"{args.image_id}.jpg"
    gold_boxes = load_gold_boxes(resolve_path(args.gold), args.image_id)
    data_url, payload_info = image_to_data_url(image_path, max_image_mb=0)
    client = build_openrouter_client(
        api_key=load_api_key(resolve_path(args.api_key_file)),
        base_url="https://openrouter.ai/api/v1",
        app_name="economist-ad-face-provider-grounding-probe",
        http_referer=None,
        timeout=args.request_timeout_seconds,
    )
    prompt = (
        "Locate every visible human face in this image. Return only one JSON object of the form "
        "{\"faces\":[{\"bbox_2d\":[x1,y1,x2,y2]}]}. Coordinates must be integers normalized "
        "to a 0..1000 grid relative to the complete image. Each box must tightly enclose only the "
        "visible face from hairline/top of head to chin and cheek to cheek; never include neck, torso, "
        "or a whole person. Include only faces that are visibly present and do not infer regularly "
        "spaced faces."
    )
    results = []
    for provider_tag in args.provider_tags or DEFAULT_PROVIDERS:
        results.append(run_provider(client, args.model, provider_tag, prompt, data_url, gold_boxes))
    payload = {
        "schema_version": "qwen_provider_face_grounding_probe_v1",
        "generated_at": iso_now(),
        "image_id": args.image_id,
        "image_path": str(image_path),
        "payload_info": payload_info,
        "gold_boxes": gold_boxes,
        "prompt": prompt,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_results(results)
    print(f"wrote {output}")
    return 0


def run_provider(client: Any, model: str, provider_tag: str, prompt: str, data_url: str, gold_boxes: list[list[float]]) -> dict[str, Any]:
    started = time.time()
    result: dict[str, Any] = {"provider_tag": provider_tag}
    try:
        response = client.chat.completions.create(
            model=model,
            messages=build_messages(prompt, data_url),
            max_tokens=1200,
            temperature=0,
            response_format={"type": "json_object"},
            extra_body={
                "provider": {"order": [provider_tag], "allow_fallbacks": False, "require_parameters": True},
                "reasoning": {"effort": "none", "exclude": True},
            },
        )
        text = response.choices[0].message.content
        parsed = json.loads(text)
        boxes = parse_face_boxes(parsed)
        scores = greedy_ious(gold_boxes, boxes)
        result.update(
            {
                "ok": True,
                "response_text": text,
                "boxes": boxes,
                "ious": scores,
                "iou_ge_0_25": sum(value >= 0.25 for value in scores),
                "iou_ge_0_50": sum(value >= 0.50 for value in scores),
                "median_iou": median(scores),
                "usage": response.usage.model_dump() if response.usage else None,
            }
        )
    except Exception as exc:
        result.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    result["elapsed_seconds"] = round(time.time() - started, 3)
    return result


def parse_face_boxes(parsed: dict[str, Any]) -> list[list[float]]:
    boxes = []
    for item in parsed.get("faces", []):
        raw = item.get("bbox_2d") if isinstance(item, dict) else item
        normalized = normalize_box(raw)
        if normalized:
            boxes.append(normalized)
    return boxes


def print_results(results: list[dict[str, Any]]) -> None:
    for result in results:
        usage = result.get("usage") or {}
        print(
            result["provider_tag"],
            f"ok={result['ok']}",
            f"prompt_tokens={usage.get('prompt_tokens')}",
            f"faces={len(result.get('boxes') or [])}",
            f"iou>=.5={result.get('iou_ge_0_50')}",
            f"median_iou={result.get('median_iou')}",
        )


def load_gold_boxes(path: Path, image_id: str) -> list[list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = next(
        item
        for item in payload["annotations"]
        if item.get("image_id") == image_id and item.get("assignment_status") == "done" and item.get("status") == "complete"
    )
    return [person["face_bbox"] for ad in row["payload"].get("advertisements", []) for person in ad.get("people", [])]


def normalize_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    box = [float(item) for item in value]
    if any(item > 1 for item in box):
        box = [item / 1000 for item in box]
    return box


def box_iou(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union else 0.0


def greedy_ious(gold: list[list[float]], predicted: list[list[float]]) -> list[float]:
    candidates = sorted(
        ((box_iou(gold_box, predicted_box), gold_index, predicted_index) for gold_index, gold_box in enumerate(gold) for predicted_index, predicted_box in enumerate(predicted)),
        reverse=True,
    )
    used_gold: set[int] = set()
    used_predicted: set[int] = set()
    scores = []
    for score, gold_index, predicted_index in candidates:
        if gold_index in used_gold or predicted_index in used_predicted:
            continue
        used_gold.add(gold_index)
        used_predicted.add(predicted_index)
        scores.append(score)
    return scores


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    value = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return round(value, 4)


if __name__ == "__main__":
    raise SystemExit(main())

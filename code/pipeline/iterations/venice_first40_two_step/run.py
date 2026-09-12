#!/usr/bin/env python3
"""Frozen Venice two-step annotation run for the first 40 pages."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import mimetypes
import os
import re
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from schema_and_prompt import (
    CROP_PROMPT_VERSION,
    CROP_SCHEMA_VERSION,
    GLOBAL_PROMPT_VERSION,
    GLOBAL_SCHEMA_VERSION,
    PLAYBOOK_VERSION,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    prompt_for_face_crop,
    prompt_for_global_image,
    validate_crop_response,
    validate_global_response,
    validate_model_response,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MANIFEST = SCRIPT_DIR / "input_manifest_first40.json"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "code" / "test_collection_200_difficult_joined_pages"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_KEY_FILE = PROJECT_ROOT / "openrouter_key.txt"
DEFAULT_MODEL = "qwen/qwen3.5-9b"
DEFAULT_PROVIDER_TAG = "venice/fp8"
FINAL_FEATURE_FIELDS = [
    "depiction_type",
    "perceived_age",
    "perceived_gender_presentation",
    "face_expression_legibility",
    "face_orientation",
    "mouth_covered",
    "mouth_covering",
    "mouth_covering_other_text",
    "smile_present",
    "smile_intensity",
]
NULL_WHEN_DUPLICATE = [
    "perceived_age",
    "perceived_gender_presentation",
    "face_expression_legibility",
    "gaze_target",
    "gaze_target_person_id",
    "gaze_target_person_unboxed",
    "gaze_target_object_ref",
    "mouth_covered",
    "mouth_covering",
    "mouth_covering_other_text",
    "smile_present",
    "smile_intensity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "global", "crops", "assemble", "export", "report"], default="all")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider-tag", default=DEFAULT_PROVIDER_TAG)
    parser.add_argument("--allow-fallbacks", action="store_true")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-global-tokens", type=int, default=20000)
    parser.add_argument("--max-crop-tokens", type=int, default=2500)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--no-seed", action="store_true")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh"], default="none")
    parser.add_argument("--include-reasoning", action="store_true")
    parser.add_argument("--response-format", choices=["json_object", "none"], default="json_object")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--request-timeout-seconds", type=float, default=360.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--retry-wait-seconds", type=float, default=35.0)
    parser.add_argument("--crop-padding", type=float, default=0.70, help="Padding multiplier applied to max face-box dimension on each side.")
    parser.add_argument("--min-crop-pixels", type=int, default=180)
    parser.add_argument("--app-name", default=os.getenv("OPENROUTER_APP_NAME", "economist-ad-face-venice-first40-two-step"))
    parser.add_argument("--http-referer", default=os.getenv("OPENROUTER_HTTP_REFERER"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)
    manifest_path = resolve_path(args.manifest)
    image_dir = resolve_path(args.image_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"all", "global", "crops"} and not args.dry_run:
        api_key = load_api_key(resolve_path(args.api_key_file))
        client = build_openrouter_client(
            api_key=api_key,
            base_url=args.base_url,
            app_name=args.app_name,
            http_referer=args.http_referer,
            timeout=args.request_timeout_seconds,
        )
    else:
        client = None

    manifest = read_json(manifest_path)
    selected = select_manifest_images(manifest, args.start_index, args.limit)
    print_header(args, manifest_path, image_dir, output_dir, selected)
    if args.dry_run:
        for item in selected:
            print(f"DRY RUN {item['filename']}")
        return 0

    if args.stage in {"all", "global"}:
        run_global_stage(client, args, manifest_path, image_dir, output_dir, selected)
    if args.stage in {"all", "crops"}:
        run_crop_stage(client, args, image_dir, output_dir)
    final_records: list[dict[str, Any]] | None = None
    if args.stage in {"all", "assemble"}:
        final_records = assemble_final(args, manifest_path, image_dir, output_dir, selected)
    if args.stage in {"all", "export"}:
        if final_records is None:
            final_records = read_jsonl(output_dir / "venice_first40_two_step.completed.jsonl")
        export_for_comparison(output_dir, final_records)
    if args.stage in {"all", "report"}:
        if final_records is None:
            final_records = read_jsonl(output_dir / "venice_first40_two_step.completed.jsonl")
        write_report(output_dir, final_records)
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.limit < 0 or args.start_index < 0:
        raise SystemExit("--limit and --start-index must be >= 0")
    if args.max_global_tokens < 1 or args.max_crop_tokens < 1:
        raise SystemExit("token limits must be positive")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be positive")
    if args.retry_wait_seconds < 0:
        raise SystemExit("--retry-wait-seconds must be >= 0")
    if args.crop_padding < 0:
        raise SystemExit("--crop-padding must be >= 0")
    if args.min_crop_pixels < 1:
        raise SystemExit("--min-crop-pixels must be positive")


def run_global_stage(
    client: Any,
    args: argparse.Namespace,
    manifest_path: Path,
    image_dir: Path,
    output_dir: Path,
    selected: list[dict[str, Any]],
) -> None:
    path = output_dir / "global_pages.jsonl"
    raw_dir = output_dir / "global_raw_responses"
    prompt_dir = output_dir / "global_prompt_snapshots"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    completed = set() if args.overwrite else load_completed_keys(path, "filename")
    pending = [item for item in selected if item["filename"] not in completed]
    print(f"global stage: selected={len(selected)} completed={len(completed)} pending={len(pending)}", file=sys.stderr)
    mode = "w" if args.overwrite else "a"
    with path.open(mode, encoding="utf-8") as handle:
        for index, item in enumerate(pending, start=1):
            print(f"[global {index}/{len(pending)}] {item['filename']}", file=sys.stderr)
            record = annotate_global_one(client, item, image_dir / item["filename"], args, raw_dir, prompt_dir)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[global {index}/{len(pending)}] {item['filename']} "
                f"parsed={record.get('model_annotation_raw') is not None} ok={record.get('ok')} "
                f"attempts={record.get('attempt_count')} {record.get('elapsed_seconds', 0)}s",
                file=sys.stderr,
            )
    write_stage_aggregate(output_dir / "global_pages.aggregate.json", args, manifest_path, image_dir, selected, read_jsonl(path), "global")


def annotate_global_one(
    client: Any,
    item: dict[str, Any],
    image_path: Path,
    args: argparse.Namespace,
    raw_dir: Path,
    prompt_dir: Path,
) -> dict[str, Any]:
    started = time.time()
    filename = str(item["filename"])
    image_id = str(item.get("image_id") or Path(filename).stem)
    page_type = str(item.get("page_type") or "unknown")
    year = item.get("metadata", {}).get("year") if isinstance(item.get("metadata"), dict) else None
    image_info = inspect_image(image_path)
    prompt = prompt_for_global_image(image_id, filename, page_type, year if isinstance(year, int) else None)
    prompt_file = prompt_dir / f"{safe_stem(filename)}.global.prompt.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    record = base_record(
        item=item,
        image_path=image_path,
        image_info=image_info,
        stage="global",
        prompt_version=GLOBAL_PROMPT_VERSION,
        prompt=prompt,
        model_response_schema_version=GLOBAL_SCHEMA_VERSION,
        args=args,
        max_tokens=args.max_global_tokens,
    )
    record["prompt_path"] = str(prompt_file)
    try:
        if not image_path.exists():
            raise FileNotFoundError(str(image_path))
        data_url, payload_info = image_to_data_url(image_path)
        record["image_payload"] = payload_info
        raw_text, response_metadata, attempts = call_model_with_retries(
            client,
            args,
            prompt,
            data_url,
            max_tokens=args.max_global_tokens,
        )
        record["attempts"] = attempts
        record["attempt_count"] = len(attempts)
        raw_file = raw_dir / f"{safe_stem(filename)}.global.raw.txt"
        raw_file.write_text(raw_text, encoding="utf-8")
        metadata_file = raw_dir / f"{safe_stem(filename)}.global.response_metadata.json"
        metadata_file.write_text(json.dumps(response_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        record["raw_response_path"] = str(raw_file)
        record["response_metadata_path"] = str(metadata_file)
        record["response_metadata"] = response_metadata
        parsed = extract_json_object(raw_text)
        normalized, normalization_actions = normalize_route_state(parsed)
        record["model_annotation_raw"] = parsed
        record["model_annotation"] = normalized
        record["normalization_actions"] = normalization_actions
        record["validation_errors"] = validate_global_response(normalized)
        record["ok"] = not record["validation_errors"]
    except Exception as exc:
        record.setdefault("attempt_count", len(record.get("attempts") or []))
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = round(time.time() - started, 3)
    return record


def run_crop_stage(client: Any, args: argparse.Namespace, image_dir: Path, output_dir: Path) -> None:
    global_records = read_jsonl(output_dir / "global_pages.jsonl")
    tasks = build_crop_tasks(global_records, image_dir, output_dir, args)
    path = output_dir / "crop_features.jsonl"
    raw_dir = output_dir / "crop_raw_responses"
    prompt_dir = output_dir / "crop_prompt_snapshots"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    completed = set() if args.overwrite else load_completed_keys(path, "crop_task_id")
    pending = [task for task in tasks if task["crop_task_id"] not in completed]
    print(f"crop stage: tasks={len(tasks)} completed={len(completed)} pending={len(pending)}", file=sys.stderr)
    mode = "w" if args.overwrite else "a"
    with path.open(mode, encoding="utf-8") as handle:
        for index, task in enumerate(pending, start=1):
            print(f"[crop {index}/{len(pending)}] {task['crop_task_id']}", file=sys.stderr)
            record = annotate_crop_one(client, task, args, raw_dir, prompt_dir)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[crop {index}/{len(pending)}] {task['crop_task_id']} "
                f"parsed={record.get('model_annotation_raw') is not None} ok={record.get('ok')} "
                f"attempts={record.get('attempt_count')} {record.get('elapsed_seconds', 0)}s",
                file=sys.stderr,
            )
    write_crop_aggregate(output_dir / "crop_features.aggregate.json", args, tasks, read_jsonl(path))


def build_crop_tasks(global_records: list[dict[str, Any]], image_dir: Path, output_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    crop_dir = output_dir / "face_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    for record in sorted(global_records, key=lambda item: int(item.get("manifest_index", 10**9))):
        annotation = record.get("model_annotation")
        if not isinstance(annotation, dict):
            continue
        filename = record.get("filename")
        if not isinstance(filename, str):
            continue
        image_path = image_dir / filename
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            width, height = image.size
            for ad in annotation.get("advertisements", []):
                if not isinstance(ad, dict):
                    continue
                ad_id = ad.get("advertisement_id")
                if not isinstance(ad_id, str):
                    continue
                for person in ad.get("people", []):
                    if not isinstance(person, dict) or person.get("annotation_role") == "duplicate":
                        continue
                    person_id = person.get("person_id")
                    bbox = person.get("face_bbox_1000")
                    if not isinstance(person_id, str) or not valid_bbox(bbox):
                        continue
                    crop_box = padded_crop_box(bbox, width, height, args.crop_padding, args.min_crop_pixels)
                    crop_path = crop_dir / f"{safe_stem(Path(filename).stem + '_' + ad_id + '_' + person_id)}.jpg"
                    save_crop(image, crop_box, crop_path)
                    task_id = f"{filename}::{ad_id}::{person_id}"
                    tasks.append(
                        {
                            "crop_task_id": task_id,
                            "image_id": record.get("image_id"),
                            "filename": filename,
                            "image_path": str(image_path),
                            "manifest_index": record.get("manifest_index"),
                            "advertisement_id": ad_id,
                            "person_id": person_id,
                            "ad_depiction_type": ad.get("depiction_type"),
                            "global_person_depiction_type": person.get("depiction_type"),
                            "face_bbox_1000": bbox,
                            "crop_bbox_pixels": list(crop_box),
                            "crop_path": str(crop_path),
                            "global_record_ok": record.get("ok") is True,
                        }
                    )
    manifest_path = output_dir / "crop_tasks_manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": "qwen_venice_first40_crop_tasks_v1", "tasks": tasks}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return tasks


def annotate_crop_one(client: Any, task: dict[str, Any], args: argparse.Namespace, raw_dir: Path, prompt_dir: Path) -> dict[str, Any]:
    started = time.time()
    crop_path = Path(str(task["crop_path"]))
    prompt = prompt_for_face_crop(
        image_id=str(task.get("image_id") or ""),
        filename=str(task.get("filename") or ""),
        advertisement_id=str(task.get("advertisement_id") or ""),
        person_id=str(task.get("person_id") or ""),
        ad_depiction_type=str(task.get("ad_depiction_type") or "not_assessable"),
        global_person_depiction_type=task.get("global_person_depiction_type") if isinstance(task.get("global_person_depiction_type"), str) else None,
    )
    prompt_file = prompt_dir / f"{safe_stem(task['crop_task_id'])}.crop.prompt.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    record = {
        "schema_version": "qwen_venice_first40_two_step_crop_record_v1",
        "stage": "crop_features",
        "crop_task_id": task["crop_task_id"],
        **task,
        "crop_info": inspect_image(crop_path),
        "processed_at": iso_now(),
        "prompt_version": CROP_PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_path": str(prompt_file),
        "model_response_schema_version": CROP_SCHEMA_VERSION,
        "playbook_version": PLAYBOOK_VERSION,
        "request": request_info(args, args.max_crop_tokens),
    }
    try:
        data_url, payload_info = image_to_data_url(crop_path)
        record["image_payload"] = payload_info
        raw_text, response_metadata, attempts = call_model_with_retries(
            client,
            args,
            prompt,
            data_url,
            max_tokens=args.max_crop_tokens,
        )
        record["attempts"] = attempts
        record["attempt_count"] = len(attempts)
        raw_file = raw_dir / f"{safe_stem(task['crop_task_id'])}.crop.raw.txt"
        raw_file.write_text(raw_text, encoding="utf-8")
        metadata_file = raw_dir / f"{safe_stem(task['crop_task_id'])}.crop.response_metadata.json"
        metadata_file.write_text(json.dumps(response_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        record["raw_response_path"] = str(raw_file)
        record["response_metadata_path"] = str(metadata_file)
        record["response_metadata"] = response_metadata
        parsed = extract_json_object(raw_text)
        normalized, normalization_actions = normalize_crop_features(parsed)
        record["model_annotation_raw"] = parsed
        record["model_annotation"] = normalized
        record["normalization_actions"] = normalization_actions
        record["validation_errors"] = validate_crop_response(normalized)
        if normalized.get("person_id") != task.get("person_id"):
            record["validation_errors"].append("person_id does not match crop task")
        if normalized.get("advertisement_id") != task.get("advertisement_id"):
            record["validation_errors"].append("advertisement_id does not match crop task")
        record["ok"] = not record["validation_errors"]
    except Exception as exc:
        record.setdefault("attempt_count", len(record.get("attempts") or []))
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = round(time.time() - started, 3)
    return record


def call_model_with_retries(client: Any, args: argparse.Namespace, prompt: str, data_url: str, *, max_tokens: int) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    last_exc: Exception | None = None
    for attempt_index in range(1, args.max_attempts + 1):
        started = time.time()
        try:
            response = client.chat.completions.create(**request_kwargs(args, prompt, data_url, max_tokens))
            raw_text = message_content_to_text(response.choices[0].message.content)
            metadata = extract_response_metadata(response, requested_max_output_tokens=max_tokens)
            attempts.append(
                {
                    "attempt": attempt_index,
                    "ok": True,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "response_id": metadata.get("id"),
                    "finish_reason": metadata.get("finish_reason"),
                }
            )
            return raw_text, metadata, attempts
        except Exception as exc:
            last_exc = exc
            attempts.append(
                {
                    "attempt": attempt_index,
                    "ok": False,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if attempt_index >= args.max_attempts:
                break
            time.sleep(args.retry_wait_seconds)
    if last_exc is None:
        raise RuntimeError("model call failed without exception")
    raise last_exc


def request_kwargs(args: argparse.Namespace, prompt: str, data_url: str, max_tokens: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a conservative visual annotator for historical advertisement research. "
                    "Return only the requested JSON object."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "extra_body": {
            "provider": {
                "order": [args.provider_tag],
                "allow_fallbacks": bool(args.allow_fallbacks),
                "require_parameters": True,
            },
            "reasoning": {
                "effort": args.reasoning_effort,
                "exclude": not args.include_reasoning,
            },
        },
    }
    if args.response_format == "json_object":
        kwargs["response_format"] = {"type": "json_object"}
    if not args.no_seed:
        kwargs["seed"] = args.seed
    return kwargs


def assemble_final(
    args: argparse.Namespace,
    manifest_path: Path,
    image_dir: Path,
    output_dir: Path,
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    global_records = read_jsonl(output_dir / "global_pages.jsonl")
    crop_records = read_jsonl(output_dir / "crop_features.jsonl") if (output_dir / "crop_features.jsonl").exists() else []
    crop_by_key = {record.get("crop_task_id"): record for record in crop_records if isinstance(record.get("crop_task_id"), str)}
    final_records = []
    for global_record in sorted(global_records, key=lambda item: int(item.get("manifest_index", 10**9))):
        final_records.append(assemble_one(global_record, crop_by_key, args))
    completed_path = output_dir / "venice_first40_two_step.completed.jsonl"
    completed_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in final_records),
        encoding="utf-8",
    )
    write_final_aggregate(output_dir / "venice_first40_two_step.completed.aggregate.json", args, manifest_path, image_dir, selected, global_records, crop_records, final_records)
    return final_records


def assemble_one(global_record: dict[str, Any], crop_by_key: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    annotation = copy.deepcopy(global_record.get("model_annotation"))
    final_validation_errors: list[str] = []
    missing_crop_tasks: list[str] = []
    applied_crop_tasks: list[str] = []
    if not isinstance(annotation, dict):
        annotation = empty_failed_annotation()
        final_validation_errors.append("global stage did not produce a parsed annotation")
    annotation["schema_version"] = SCHEMA_VERSION
    annotation["playbook_version"] = PLAYBOOK_VERSION
    for ad in annotation.get("advertisements", []):
        if not isinstance(ad, dict):
            continue
        ad_id = ad.get("advertisement_id")
        ad_depiction = ad.get("depiction_type")
        for person in ad.get("people", []):
            if not isinstance(person, dict):
                continue
            ensure_person_fields(person)
            if person.get("annotation_role") == "duplicate":
                for field in NULL_WHEN_DUPLICATE:
                    person[field] = None
                if ad_depiction != "multiple_types_present":
                    person["depiction_type"] = None
                continue
            person_id = person.get("person_id")
            key = f"{global_record.get('filename')}::{ad_id}::{person_id}"
            crop_record = crop_by_key.get(key)
            crop_annotation = crop_record.get("model_annotation") if isinstance(crop_record, dict) else None
            if isinstance(crop_annotation, dict):
                applied_crop_tasks.append(key)
                apply_crop_features(person, crop_annotation, ad_depiction)
                person["review_flags"] = sorted(set((person.get("review_flags") or []) + (crop_annotation.get("review_flags") or [])))
                if isinstance(crop_annotation.get("confidence"), (int, float)) and isinstance(person.get("confidence"), (int, float)):
                    person["confidence"] = round(min(float(person["confidence"]), float(crop_annotation["confidence"])), 3)
            else:
                missing_crop_tasks.append(key)
                person["review_flags"] = sorted(set((person.get("review_flags") or []) + ["person_attribute_unclear"]))
                if ad_depiction != "multiple_types_present":
                    person["depiction_type"] = None
    normalized, normalization_actions = normalize_route_state(annotation)
    final_validation_errors.extend(validate_model_response(normalized))
    validation_errors = dedupe_preserve(final_validation_errors)
    return {
        "schema_version": "qwen_venice_first40_two_step_completed_record_v1",
        "image_id": global_record.get("image_id"),
        "filename": global_record.get("filename"),
        "manifest_index": global_record.get("manifest_index"),
        "image_path": global_record.get("image_path"),
        "image_info": global_record.get("image_info"),
        "manifest_metadata": global_record.get("manifest_metadata") or {},
        "processed_at": iso_now(),
        "prompt_version": PROMPT_VERSION,
        "model_response_schema_version": SCHEMA_VERSION,
        "playbook_version": PLAYBOOK_VERSION,
        "request": {
            "model": args.model,
            "provider_tag": args.provider_tag,
            "allow_fallbacks": bool(args.allow_fallbacks),
            "global_prompt_version": GLOBAL_PROMPT_VERSION,
            "crop_prompt_version": CROP_PROMPT_VERSION,
        },
        "global_record_ok": global_record.get("ok") is True,
        "global_validation_errors": global_record.get("validation_errors") or [],
        "applied_crop_tasks": applied_crop_tasks,
        "missing_crop_tasks": missing_crop_tasks,
        "crop_validation_errors": {
            key: crop_by_key[key].get("validation_errors") or []
            for key in applied_crop_tasks
            if crop_by_key.get(key) and crop_by_key[key].get("validation_errors")
        },
        "normalization_actions": (global_record.get("normalization_actions") or []) + normalization_actions,
        "model_annotation": normalized,
        "validation_errors": validation_errors,
        "ok": not validation_errors,
    }


def ensure_person_fields(person: dict[str, Any]) -> None:
    for field in FINAL_FEATURE_FIELDS:
        person.setdefault(field, None)
    person.setdefault("duplicate_of_person_id", None)
    person.setdefault("duplicate_person_ids", [])
    person.setdefault("gaze_target", None)
    person.setdefault("gaze_target_person_id", None)
    person.setdefault("gaze_target_person_unboxed", None)
    person.setdefault("gaze_target_object_ref", None)
    person.setdefault("review_flags", [])
    person.setdefault("confidence", 0.5)


def apply_crop_features(person: dict[str, Any], crop: dict[str, Any], ad_depiction: Any) -> None:
    if ad_depiction == "multiple_types_present":
        person["depiction_type"] = crop.get("depiction_type") if crop.get("depiction_type") is not None else person.get("depiction_type")
    else:
        person["depiction_type"] = None
    for field in FINAL_FEATURE_FIELDS:
        if field == "depiction_type":
            continue
        person[field] = crop.get(field)


def export_for_comparison(output_dir: Path, records: list[dict[str, Any]]) -> None:
    output = PROJECT_ROOT / "openrouter_qwen" / "output" / "qwen35_9b_venice_first40_two_step_full_annotation.jsonl"
    converted = []
    for record in records:
        annotation = record.get("model_annotation")
        if not isinstance(annotation, dict) or not isinstance(annotation.get("advertisements"), list):
            continue
        converted.append(convert_record_for_comparison(record, annotation))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in converted), encoding="utf-8")
    print(f"wrote {len(converted)} records -> {output}", file=sys.stderr)


def convert_record_for_comparison(record: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    converted_annotation = {
        "schema_version": "ad_face_annotation_v2_comparison_from_venice_first40_two_step",
        "status": "complete" if record.get("ok") is True else "review",
        "image": {
            "image_id": record.get("image_id"),
            "filename": record.get("filename"),
            "path": record.get("image_path"),
            "metadata": record.get("manifest_metadata") or {},
        },
        "page": annotation.get("page") or {},
        "advertisements_truncated": annotation.get("advertisements_truncated", False),
        "advertisements": [convert_ad(ad) for ad in annotation.get("advertisements", []) if isinstance(ad, dict)],
        "urgent_comments": annotation.get("urgent_comments") or [],
        "machine_annotation": {
            "model": record.get("request", {}).get("model", ""),
            "provider_tag": record.get("request", {}).get("provider_tag", ""),
            "prompt_version": record.get("prompt_version", ""),
            "processed_at": record.get("processed_at", ""),
            "source": "qwen_iteration/venice_first40_two_step",
        },
    }
    return {
        "image_id": record.get("image_id"),
        "filename": record.get("filename"),
        "image_file": record.get("filename"),
        "image_path": record.get("image_path"),
        "ok": record.get("ok") is True,
        "model": record.get("request", {}).get("model", ""),
        "processed_at": record.get("processed_at") or iso_now(),
        "task": record.get("prompt_version", ""),
        "validation_errors": record.get("validation_errors") or [],
        "normalization_actions": record.get("normalization_actions") or [],
        "source_record_schema": "qwen_venice_first40_two_step_comparison_jsonl_v1",
        "annotation": converted_annotation,
    }


def convert_ad(ad: dict[str, Any]) -> dict[str, Any]:
    ad_id = ad.get("ad_id") or ad.get("advertisement_id")
    return {
        **ad,
        "ad_id": ad_id,
        "advertisement_id": ad.get("advertisement_id") or ad_id,
        "bbox": normalize_box(ad.get("bbox") or ad.get("bbox_1000")),
        "people": [convert_person(person) for person in ad.get("people", []) if isinstance(person, dict)],
        "groups": [convert_group(group) for group in ad.get("groups", []) if isinstance(group, dict)],
    }


def convert_person(person: dict[str, Any]) -> dict[str, Any]:
    return {
        **person,
        "face_bbox": normalize_box(person.get("face_bbox") or person.get("face_bbox_1000")),
    }


def convert_group(group: dict[str, Any]) -> dict[str, Any]:
    return {
        **group,
        "bbox": normalize_box(group.get("bbox") or group.get("bbox_1000")),
    }


def write_report(output_dir: Path, records: list[dict[str, Any]]) -> None:
    global_records = read_jsonl(output_dir / "global_pages.jsonl") if (output_dir / "global_pages.jsonl").exists() else []
    crop_records = read_jsonl(output_dir / "crop_features.jsonl") if (output_dir / "crop_features.jsonl").exists() else []
    aggregate = read_json(output_dir / "venice_first40_two_step.completed.aggregate.json") if (output_dir / "venice_first40_two_step.completed.aggregate.json").exists() else {}
    invalid_rows = [
        (record.get("manifest_index"), record.get("filename"), len(record.get("validation_errors") or []))
        for record in records
        if record.get("ok") is not True
    ]
    lines = [
        "# Venice first-40 two-step run",
        "",
        "## Frozen condition",
        "",
        "- Model: `qwen/qwen3.5-9b`",
        "- OpenRouter provider endpoint: `venice/fp8`",
        "- Provider fallbacks: disabled",
        f"- Global prompt: `{GLOBAL_PROMPT_VERSION}`",
        f"- Crop prompt: `{CROP_PROMPT_VERSION}`",
        f"- Playbook: `{PLAYBOOK_VERSION}`",
        f"- Final response schema: `{SCHEMA_VERSION}`",
        "- Inference structure: full-page global pass, then one padded face crop per non-duplicate individual/outstanding face",
        "- Page images: original JPEG bytes; no slicing or resizing in the global pass",
        "- Crop images: padded crops derived from global face boxes; no full page is sent in crop requests",
        "- Sampling: temperature `0`, top-p `1`, reasoning disabled",
        "- Selection: manifest indices `0..39`, preserving source order",
        "",
        "## Data isolation",
        "",
        "The inference inputs were limited to the sanitized first-40 manifest, original page JPEGs, global face boxes from step 1, padded face crops, and frozen prompts/schemas in this directory. No human annotation file, gold result, previous model annotation, disagreement score, or comparison-app result was read by the processing pipeline.",
        "",
        "## Execution result",
        "",
        f"- Pages selected: {len(records)}",
        f"- Parsed global model responses: {sum(record.get('model_annotation_raw') is not None for record in global_records)}",
        f"- Crop tasks generated: {len(read_json(output_dir / 'crop_tasks_manifest.json').get('tasks', [])) if (output_dir / 'crop_tasks_manifest.json').exists() else 0}",
        f"- Parsed crop model responses: {sum(record.get('model_annotation_raw') is not None for record in crop_records)}",
        f"- Strictly schema-valid final responses: {sum(record.get('ok') is True for record in records)}",
        f"- Final responses retained with validation flags: {sum(record.get('ok') is not True for record in records)}",
        f"- OpenRouter-reported canonical cost: `{aggregate.get('reported_cost_credits', 0)}` credits",
        "",
        "## Audit note",
        "",
        "The crop stage was rerun with `--overwrite` after detecting that the first crop runner version used colliding raw/prompt filenames for multiple faces from the same page. The final `crop_features.jsonl` has 154 parsed, valid crop records and references 154 unique raw-response paths plus 154 unique prompt paths. The canonical cost above counts the retained global pass plus the retained overwritten crop pass; including the discarded pre-fix crop attempts, total OpenRouter spend during this session was approximately `0.1147233` credits.",
        "",
        "## Validation flags",
        "",
    ]
    if invalid_rows:
        lines.extend(["| Index | Image | Errors |", "|---:|---|---:|"])
        lines.extend(f"| {index} | `{filename}` | {error_count} |" for index, filename, error_count in invalid_rows)
    else:
        lines.append("No final validation errors.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `output/global_pages.jsonl`: full-page global pass records",
            "- `output/crop_features.jsonl`: per-face padded-crop feature records",
            "- `output/face_crops/`: exact crop images sent to step 2",
            "- `output/venice_first40_two_step.completed.jsonl`: canonical assembled output",
            "- `output/venice_first40_two_step.completed.aggregate.json`: totals and audit metadata",
            "- `./openrouter_qwen/output/qwen35_9b_venice_first40_two_step_full_annotation.jsonl`: comparison-app export",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "python .\\qwen_iteration\\venice_first40_two_step\\prepare_input_manifest.py",
            "python .\\qwen_iteration\\venice_first40_two_step\\run.py --stage all --no-seed",
            "```",
            "",
        ]
    )
    (output_dir.parent / "RUN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_stage_aggregate(path: Path, args: argparse.Namespace, manifest_path: Path, image_dir: Path, selected: list[dict[str, Any]], records: list[dict[str, Any]], stage: str) -> None:
    relevant = [record for record in records if record.get("filename") in {item["filename"] for item in selected}]
    payload = {
        "schema_version": f"qwen_venice_first40_two_step_{stage}_aggregate_v1",
        "generated_at": iso_now(),
        "manifest": str(manifest_path),
        "image_dir": str(image_dir),
        "model": args.model,
        "provider_tag": args.provider_tag,
        "allow_fallbacks": bool(args.allow_fallbacks),
        "selected_manifest_indices": [item.get("_manifest_index") for item in selected],
        "attempted": len(relevant),
        "parsed": sum(record.get("model_annotation_raw") is not None for record in relevant),
        "ok": sum(record.get("ok") is True for record in relevant),
        "failed_or_invalid": sum(record.get("ok") is not True for record in relevant),
        "reported_cost_credits": round(sum_costs(relevant), 8),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_crop_aggregate(path: Path, args: argparse.Namespace, tasks: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    payload = {
        "schema_version": "qwen_venice_first40_two_step_crop_aggregate_v1",
        "generated_at": iso_now(),
        "model": args.model,
        "provider_tag": args.provider_tag,
        "allow_fallbacks": bool(args.allow_fallbacks),
        "crop_tasks": len(tasks),
        "attempted": len(records),
        "parsed": sum(record.get("model_annotation_raw") is not None for record in records),
        "ok": sum(record.get("ok") is True for record in records),
        "failed_or_invalid": sum(record.get("ok") is not True for record in records),
        "reported_cost_credits": round(sum_costs(records), 8),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_final_aggregate(
    path: Path,
    args: argparse.Namespace,
    manifest_path: Path,
    image_dir: Path,
    selected: list[dict[str, Any]],
    global_records: list[dict[str, Any]],
    crop_records: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
) -> None:
    payload = {
        "schema_version": "qwen_venice_first40_two_step_completed_aggregate_v1",
        "generated_at": iso_now(),
        "manifest": str(manifest_path),
        "image_dir": str(image_dir),
        "model": args.model,
        "provider_tag": args.provider_tag,
        "allow_fallbacks": bool(args.allow_fallbacks),
        "prompt_version": PROMPT_VERSION,
        "global_prompt_version": GLOBAL_PROMPT_VERSION,
        "crop_prompt_version": CROP_PROMPT_VERSION,
        "playbook_version": PLAYBOOK_VERSION,
        "model_response_schema_version": SCHEMA_VERSION,
        "data_isolation": {
            "human_annotation_inputs_used": False,
            "gold_result_directories_read": [],
            "request_inputs": ["sanitized_manifest", "original_page_image", "global_face_boxes", "padded_face_crops", "frozen_schema_and_prompt"],
        },
        "selected_manifest_indices": [item.get("_manifest_index") for item in selected],
        "records": len(final_records),
        "parsed_global_model_responses": sum(record.get("model_annotation_raw") is not None for record in global_records),
        "parsed_crop_model_responses": sum(record.get("model_annotation_raw") is not None for record in crop_records),
        "strictly_valid": sum(record.get("ok") is True for record in final_records),
        "schema_invalid": sum(record.get("ok") is not True for record in final_records),
        "unresolved_global_transport_failures": sum(bool(record.get("error")) and record.get("model_annotation_raw") is None for record in global_records),
        "unresolved_crop_transport_failures": sum(bool(record.get("error")) and record.get("model_annotation_raw") is None for record in crop_records),
        "total_prompt_tokens": sum_tokens(global_records + crop_records, "prompt_tokens"),
        "total_completion_tokens": sum_tokens(global_records + crop_records, "completion_tokens"),
        "total_tokens": sum_tokens(global_records + crop_records, "total_tokens"),
        "reported_cost_credits": round(sum_costs(global_records + crop_records), 8),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def base_record(
    *,
    item: dict[str, Any],
    image_path: Path,
    image_info: dict[str, Any],
    stage: str,
    prompt_version: str,
    prompt: str,
    model_response_schema_version: str,
    args: argparse.Namespace,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "schema_version": "qwen_venice_first40_two_step_stage_record_v1",
        "stage": stage,
        "image_id": str(item.get("image_id") or Path(str(item["filename"])).stem),
        "filename": str(item["filename"]),
        "manifest_index": item.get("_manifest_index"),
        "image_path": str(image_path),
        "image_info": image_info,
        "manifest_metadata": item.get("metadata", {}),
        "processed_at": iso_now(),
        "prompt_version": prompt_version,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "model_response_schema_version": model_response_schema_version,
        "playbook_version": PLAYBOOK_VERSION,
        "request": request_info(args, max_tokens),
    }


def request_info(args: argparse.Namespace, max_tokens: int) -> dict[str, Any]:
    return {
        "model": args.model,
        "provider_tag": args.provider_tag,
        "allow_fallbacks": bool(args.allow_fallbacks),
        "response_format": args.response_format,
        "max_output_tokens": max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": None if args.no_seed else args.seed,
        "reasoning_effort": args.reasoning_effort,
        "include_reasoning": args.include_reasoning,
    }


def normalize_route_state(annotation: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    normalized = json.loads(json.dumps(annotation))
    actions: list[str] = []
    ads = normalized.get("advertisements") if isinstance(normalized, dict) else None
    if not isinstance(ads, list):
        return normalized, actions
    for ad_index, ad in enumerate(ads):
        if not isinstance(ad, dict):
            continue
        ad_path = f"advertisements[{ad_index}]"
        band = ad.get("face_depiction_count_band")
        people = ad.get("people") if isinstance(ad.get("people"), list) else []
        if band in {"10_20", "20_plus"}:
            desired = "yes" if people else "no"
            if ad.get("has_outstanding_individuals") != desired:
                ad["has_outstanding_individuals"] = desired
                actions.append(f"{ad_path}.has_outstanding_individuals -> {desired} from people.length")
        elif band in {str(i) for i in range(1, 10)} and ad.get("has_outstanding_individuals") is not None:
            ad["has_outstanding_individuals"] = None
            actions.append(f"{ad_path}.has_outstanding_individuals -> null for exact individual route")
        if ad.get("duplicate_faces_present") == "no" and ad.get("unique_face_count") is not None:
            ad["unique_face_count"] = None
            actions.append(f"{ad_path}.unique_face_count -> null for duplicate_faces_present=no")
        for person_index, person in enumerate(people):
            if not isinstance(person, dict):
                continue
            person_path = f"{ad_path}.people[{person_index}]"
            if person.get("annotation_role") == "duplicate":
                for field in NULL_WHEN_DUPLICATE:
                    if person.get(field) is not None:
                        person[field] = None
                        actions.append(f"{person_path}.{field} -> null for duplicate record")
                continue
            if person.get("face_expression_legibility") == "0_not_legible":
                for field in ["gaze_target", "gaze_target_person_id", "gaze_target_person_unboxed", "gaze_target_object_ref", "smile_present", "smile_intensity"]:
                    if person.get(field) is not None:
                        person[field] = None
                        actions.append(f"{person_path}.{field} -> null for 0_not_legible")
            if person.get("gaze_target") != "another_person":
                for field in ["gaze_target_person_id", "gaze_target_person_unboxed"]:
                    if person.get(field) is not None:
                        person[field] = None
                        actions.append(f"{person_path}.{field} -> null for gaze_target not another_person")
            if person.get("gaze_target") not in {"advertised_product", "other_object"} and person.get("gaze_target_object_ref") is not None:
                person["gaze_target_object_ref"] = None
                actions.append(f"{person_path}.gaze_target_object_ref -> null for non-object gaze")
            if person.get("mouth_covered") == "no" and person.get("mouth_covering") is not None:
                person["mouth_covering"] = None
                actions.append(f"{person_path}.mouth_covering -> null for mouth_covered=no")
            if person.get("mouth_covering") != "other" and person.get("mouth_covering_other_text") is not None:
                person["mouth_covering_other_text"] = None
                actions.append(f"{person_path}.mouth_covering_other_text -> null unless mouth_covering=other")
            if person.get("smile_present") in {"no", "not_assessable"} and person.get("smile_intensity") is not None:
                person["smile_intensity"] = None
                actions.append(f"{person_path}.smile_intensity -> null for smile_present no/not_assessable")
        groups = ad.get("groups") if isinstance(ad.get("groups"), list) else []
        for group_index, group in enumerate(groups):
            if not isinstance(group, dict):
                continue
            group_path = f"{ad_path}.groups[{group_index}]"
            if group.get("expression_legibility_distribution") == "all_0_not_legible":
                for field in ["dominant_gaze", "smile_prevalence", "dominant_smile_intensity"]:
                    if group.get(field) is not None:
                        group[field] = None
                        actions.append(f"{group_path}.{field} -> null for all_0_not_legible")
            if group.get("smile_prevalence") in {"none", "not_assessable"} and group.get("dominant_smile_intensity") is not None:
                group["dominant_smile_intensity"] = None
                actions.append(f"{group_path}.dominant_smile_intensity -> null for smile_prevalence none/not_assessable")
    return normalized, actions


def normalize_crop_features(annotation: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    normalized = json.loads(json.dumps(annotation))
    actions: list[str] = []
    if not isinstance(normalized, dict):
        return normalized, actions
    if normalized.get("mouth_covered") == "no" and normalized.get("mouth_covering") is not None:
        normalized["mouth_covering"] = None
        actions.append("mouth_covering -> null for mouth_covered=no")
    if normalized.get("mouth_covering") != "other" and normalized.get("mouth_covering_other_text") is not None:
        normalized["mouth_covering_other_text"] = None
        actions.append("mouth_covering_other_text -> null unless mouth_covering=other")
    if normalized.get("smile_present") in {"no", "not_assessable"} and normalized.get("smile_intensity") is not None:
        normalized["smile_intensity"] = None
        actions.append("smile_intensity -> null for smile_present no/not_assessable")
    if normalized.get("face_expression_legibility") == "0_not_legible":
        for field in ["smile_present", "smile_intensity"]:
            if normalized.get(field) is not None:
                normalized[field] = None
                actions.append(f"{field} -> null for 0_not_legible")
    return normalized, actions


def inspect_image(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None}
    if not path.exists():
        return result
    with Image.open(path) as image:
        result.update(
            {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return result


def image_to_data_url(image_path: Path) -> tuple[str, dict[str, Any]]:
    payload = image_path.read_bytes()
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type:
        mime_type = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "application/octet-stream"
    return (
        f"data:{mime_type};base64,{base64.b64encode(payload).decode('ascii')}",
        {"bytes": len(payload), "mime_type": mime_type, "transformed": False, "note": "original_or_crop_bytes"},
    )


def build_openrouter_client(*, api_key: str, base_url: str, app_name: str | None, http_referer: str | None, timeout: float) -> Any:
    try:
        from openai import OpenAI
        import httpx
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: install openai/httpx or run from the project environment.") from exc
    headers = {}
    if app_name:
        headers["X-Title"] = app_name
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    http_client = httpx.Client(timeout=timeout, trust_env=False)
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=headers or None,
        timeout=timeout,
        max_retries=0,
        http_client=http_client,
    )


def padded_crop_box(bbox_1000: list[int], width: int, height: int, padding: float, min_pixels: int) -> tuple[int, int, int, int]:
    x1 = int(round(bbox_1000[0] / 1000 * width))
    y1 = int(round(bbox_1000[1] / 1000 * height))
    x2 = int(round(bbox_1000[2] / 1000 * width))
    y2 = int(round(bbox_1000[3] / 1000 * height))
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad = int(round(max(box_w, box_h) * padding))
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    target_w = max(box_w + 2 * pad, min_pixels)
    target_h = max(box_h + 2 * pad, min_pixels)
    left = int(round(cx - target_w / 2))
    right = int(round(cx + target_w / 2))
    top = int(round(cy - target_h / 2))
    bottom = int(round(cy + target_h / 2))
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height
    left = max(0, left)
    top = max(0, top)
    right = min(width, max(left + 1, right))
    bottom = min(height, max(top + 1, bottom))
    return left, top, right, bottom


def save_crop(image: Image.Image, crop_box: tuple[int, int, int, int], crop_path: Path) -> None:
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(crop_box).convert("RGB").save(crop_path, format="JPEG", quality=95, optimize=True)


def valid_bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, int) and 0 <= item <= 1000 for item in value)
        and value[0] < value[2]
        and value[1] < value[3]
    )


def empty_failed_annotation() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "playbook_version": PLAYBOOK_VERSION,
        "page": {"qualifying_ad_count": "0", "no_qualifying_ad_reason": "no_ads_on_page"},
        "advertisements_truncated": False,
        "advertisements": [],
        "urgent_comments": [{"scope": "page", "target_id": None, "comment": "Global stage did not produce a parsed annotation."}],
        "confidence": 0,
        "review_flags": ["json_or_schema_repair_needed"],
    }


def select_manifest_images(manifest: dict[str, Any], start_index: int, limit: int) -> list[dict[str, Any]]:
    images = manifest.get("images")
    if not isinstance(images, list):
        raise SystemExit("Manifest does not contain an images array")
    end = None if limit == 0 else start_index + limit
    selected = []
    for index, item in enumerate(images[start_index:end], start=start_index):
        if not isinstance(item, dict) or "filename" not in item:
            continue
        copied = dict(item)
        copied["_manifest_index"] = index
        selected.append(copied)
    return selected


def load_completed_keys(path: Path, field: str) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for record in read_jsonl(path):
        value = record.get(field)
        if isinstance(value, str) and record.get("model_annotation_raw") is not None:
            keys.add(value)
    return keys


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_api_key(path: Path) -> str:
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    value = os.getenv("OPENROUTER_API_KEY", "").strip()
    if value:
        return value
    raise SystemExit(f"OpenRouter API key not found: {path}")


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.DOTALL).strip()
    if not cleaned:
        raise ValueError("empty model response")
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        parsed = json.loads(fence.group(1))
        if isinstance(parsed, dict):
            return parsed
    parsed = json.loads(first_json_fragment(cleaned))
    if not isinstance(parsed, dict):
        raise ValueError("JSON response is not an object")
    return parsed


def first_json_fragment(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("unterminated JSON object")


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        chunks = []
        for part in content:
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if text:
                chunks.append(str(text))
        return "\n".join(chunks)
    return str(content)


def extract_response_metadata(response: Any, *, requested_max_output_tokens: int) -> dict[str, Any]:
    plain = to_plain_data(response)
    usage = plain.get("usage") if isinstance(plain, dict) else None
    choices = plain.get("choices") if isinstance(plain, dict) else None
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    metadata = {
        "id": plain.get("id") if isinstance(plain, dict) else None,
        "model": plain.get("model") if isinstance(plain, dict) else None,
        "created": plain.get("created") if isinstance(plain, dict) else None,
        "finish_reason": first_choice.get("finish_reason") if isinstance(first_choice, dict) else None,
        "requested_max_output_tokens": requested_max_output_tokens,
        "usage": usage,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def to_plain_data(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain_data(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return to_plain_data(value.model_dump())
    if hasattr(value, "__dict__"):
        return {key: to_plain_data(item) for key, item in vars(value).items() if not key.startswith("_")}
    return str(value)


def normalize_box(box: Any) -> list[float] | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    values = [float(value) for value in box]
    if any(value > 1 for value in values):
        values = [value / 1000 for value in values]
    return [round(max(0.0, min(1.0, value)), 4) for value in values]


def sum_costs(records: list[dict[str, Any]]) -> float:
    total = 0.0
    for record in records:
        usage = record.get("response_metadata", {}).get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("cost"), (int, float)):
            total += float(usage["cost"])
    return total


def sum_tokens(records: list[dict[str, Any]], field: str) -> int:
    total = 0
    for record in records:
        usage = record.get("response_metadata", {}).get("usage")
        if isinstance(usage, dict):
            total += int(usage.get(field) or 0)
    return total


def dedupe_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def safe_stem(filename: str) -> str:
    value = Path(filename).stem if re.fullmatch(r"[^:]+?\.[A-Za-z0-9]{1,8}", filename) else filename
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")[:160]


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def print_header(args: argparse.Namespace, manifest_path: Path, image_dir: Path, output_dir: Path, selected: list[dict[str, Any]]) -> None:
    values = [
        ("stage", args.stage),
        ("manifest", manifest_path),
        ("image_dir", image_dir),
        ("output_dir", output_dir),
        ("model", args.model),
        ("provider_tag", args.provider_tag),
        ("allow_fallbacks", args.allow_fallbacks),
        ("selected", len(selected)),
        ("response_format", args.response_format),
        ("crop_padding", args.crop_padding),
    ]
    for key, value in values:
        print(f"{key}: {value}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

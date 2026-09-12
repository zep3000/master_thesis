#!/usr/bin/env python3
"""Single-shot OpenRouter/VLM annotation runner for the first manifest pages."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from schema_and_prompt import (
    PLAYBOOK_VERSION,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    prompt_for_image,
    validate_model_response,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "annotation_app_v2" / "data" / "test_collection_200_difficult_joined_hosted_manifest.json"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "code" / "test_collection_200_difficult_joined_pages"
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "first20_single_shot.jsonl"
DEFAULT_AGGREGATE = SCRIPT_DIR / "output" / "first20_single_shot.aggregate.json"
DEFAULT_RAW_DIR = SCRIPT_DIR / "raw_responses"
DEFAULT_KEY_FILE = PROJECT_ROOT / "openrouter_key.txt"
DEFAULT_MODEL = "qwen/qwen3.5-9b"
DEFAULT_PROVIDER_TAG = "deepinfra/bf16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--aggregate-output", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider-tag", default=DEFAULT_PROVIDER_TAG, help="Exact OpenRouter endpoint tag, e.g. deepinfra/bf16.")
    parser.add_argument("--allow-fallbacks", action="store_true", help="Allow OpenRouter provider fallback. Off by default for reproducibility.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0, help="0-based manifest index.")
    parser.add_argument("--max-output-tokens", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--no-seed", action="store_true")
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh"], default="none")
    parser.add_argument("--include-reasoning", action="store_true")
    parser.add_argument("--response-format", choices=["json_object", "none"], default="json_object")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--request-timeout-seconds", type=float, default=360.0)
    parser.add_argument("--app-name", default=os.getenv("OPENROUTER_APP_NAME", "economist-ad-face-qwen-iteration"))
    parser.add_argument("--http-referer", default=os.getenv("OPENROUTER_HTTP_REFERER"))
    parser.add_argument("--max-image-mb", type=float, default=0.0, help="Disabled by default. Only compress if set and an image exceeds this size.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-attempted", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)
    manifest_path = resolve_path(args.manifest)
    image_dir = resolve_path(args.image_dir)
    output_path = resolve_path(args.output)
    aggregate_path = resolve_path(args.aggregate_output)
    raw_dir = resolve_path(args.raw_dir)

    manifest = read_json(manifest_path)
    selected = select_manifest_images(manifest, args.start_index, args.limit)
    completed = set() if args.overwrite else load_completed(output_path, include_failures=args.skip_attempted)
    pending = [item for item in selected if item.get("filename") not in completed]
    print_header(args, manifest_path, image_dir, output_path, selected, pending, completed)
    if args.dry_run:
        for item in pending:
            print(f"DRY RUN {item['filename']}")
        return 0

    api_key = load_api_key(resolve_path(args.api_key_file))
    client = build_openrouter_client(
        api_key=api_key,
        base_url=args.base_url,
        app_name=args.app_name,
        http_referer=args.http_referer,
        timeout=args.request_timeout_seconds,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    mode = "w" if args.overwrite else "a"
    records: list[dict[str, Any]] = []
    with output_path.open(mode, encoding="utf-8") as handle:
        for run_index, item in enumerate(pending, start=1):
            image_path = image_dir / item["filename"]
            print(f"[{run_index}/{len(pending)}] {item['filename']}", file=sys.stderr)
            record = annotate_one(client, item, image_path, args, raw_dir)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
            status = "ok" if record.get("ok") else "error"
            print(
                f"[{run_index}/{len(pending)}] {item['filename']} {status} "
                f"{record.get('elapsed_seconds', 0)}s",
                file=sys.stderr,
            )

    all_records = load_records(output_path)
    write_aggregate(aggregate_path, args, manifest_path, image_dir, selected, all_records)
    return 0 if all(record.get("ok") for record in records) else 2


def validate_args(args: argparse.Namespace) -> None:
    if args.limit < 0 or args.start_index < 0:
        raise SystemExit("--limit and --start-index must be >= 0")
    if args.max_output_tokens < 1:
        raise SystemExit("--max-output-tokens must be positive")
    if args.request_timeout_seconds <= 0:
        raise SystemExit("--request-timeout-seconds must be positive")
    if args.max_image_mb < 0:
        raise SystemExit("--max-image-mb must be >= 0")


def annotate_one(client: Any, item: dict[str, Any], image_path: Path, args: argparse.Namespace, raw_dir: Path) -> dict[str, Any]:
    started = time.time()
    processed_at = iso_now()
    image_id = str(item.get("image_id") or Path(item["filename"]).stem)
    filename = str(item["filename"])
    page_type = str(item.get("page_type") or "unknown")
    year = item.get("metadata", {}).get("year") if isinstance(item.get("metadata"), dict) else None
    image_info = inspect_image(image_path)
    prompt = prompt_for_image(image_id=image_id, filename=filename, page_type=page_type, year=year if isinstance(year, int) else None)
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    request_info = {
        "model": args.model,
        "provider_tag": args.provider_tag,
        "allow_fallbacks": args.allow_fallbacks,
        "response_format": args.response_format,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": None if args.no_seed else args.seed,
        "reasoning_effort": args.reasoning_effort,
        "include_reasoning": args.include_reasoning,
        "max_image_mb": args.max_image_mb,
    }
    record: dict[str, Any] = {
        "schema_version": "qwen_iteration_record_v1",
        "image_id": image_id,
        "filename": filename,
        "manifest_index": item.get("_manifest_index"),
        "image_path": str(image_path),
        "image_info": image_info,
        "manifest_metadata": item.get("metadata", {}),
        "processed_at": processed_at,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha,
        "model_response_schema_version": SCHEMA_VERSION,
        "playbook_version": PLAYBOOK_VERSION,
        "request": request_info,
    }
    try:
        if not image_path.exists():
            raise FileNotFoundError(str(image_path))
        data_url, image_payload_info = image_to_data_url(image_path, max_image_mb=args.max_image_mb)
        record["image_payload"] = image_payload_info
        request_kwargs: dict[str, Any] = {
            "model": args.model,
            "messages": build_messages(prompt, data_url),
            "max_tokens": args.max_output_tokens,
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
            request_kwargs["response_format"] = {"type": "json_object"}
        if not args.no_seed:
            request_kwargs["seed"] = args.seed

        response = client.chat.completions.create(**request_kwargs)
        raw_text = message_content_to_text(response.choices[0].message.content)
        response_metadata = extract_response_metadata(response, requested_max_output_tokens=args.max_output_tokens)
        raw_file = raw_dir / f"{safe_stem(filename)}.raw.txt"
        raw_file.write_text(raw_text, encoding="utf-8")
        metadata_file = raw_dir / f"{safe_stem(filename)}.response_metadata.json"
        metadata_file.write_text(json.dumps(response_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        record["raw_response_path"] = str(raw_file)
        record["response_metadata_path"] = str(metadata_file)
        record["response_metadata"] = response_metadata
        parsed = extract_json_object(raw_text)
        normalized, normalization_actions = normalize_route_state(parsed)
        record["model_annotation_raw"] = parsed
        record["model_annotation"] = normalized
        record["normalization_actions"] = normalization_actions
        validation_errors = validate_model_response(normalized)
        record["validation_errors"] = validation_errors
        record["ok"] = not validation_errors
    except Exception as exc:
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = round(time.time() - started, 3)
    return record


def build_messages(prompt: str, data_url: str) -> list[dict[str, Any]]:
    return [
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
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]


def normalize_route_state(annotation: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Normalize only playbook routing sentinels; never invent visual judgments."""
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
                for field in [
                    "perceived_age",
                    "perceived_gender_presentation",
                    "face_expression_legibility",
                    "gaze_target",
                    "gaze_target_person_id",
                    "mouth_covered",
                    "mouth_covering_cause",
                    "mouth_covering_cause_other_text",
                    "smile_present",
                    "smile_intensity",
                ]:
                    if person.get(field) is not None:
                        person[field] = None
                        actions.append(f"{person_path}.{field} -> null for duplicate record")
                continue
            if person.get("face_expression_legibility") == "0_not_legible":
                for field in ["gaze_target", "gaze_target_person_id", "smile_present", "smile_intensity"]:
                    if person.get(field) is not None:
                        person[field] = None
                        actions.append(f"{person_path}.{field} -> null for 0_not_legible")
            if person.get("mouth_covered") == "no" and person.get("mouth_covering_cause") is not None:
                person["mouth_covering_cause"] = None
                actions.append(f"{person_path}.mouth_covering_cause -> null for mouth_covered=no")
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


def inspect_image(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
    }
    if not path.exists():
        return result
    with Image.open(path) as image:
        result.update(
            {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format,
            }
        )
    return result


def image_to_data_url(image_path: Path, *, max_image_mb: float) -> tuple[str, dict[str, Any]]:
    original_bytes = image_path.read_bytes()
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type:
        mime_type = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "application/octet-stream"
    payload = original_bytes
    transformed = False
    note = "original"
    if max_image_mb and len(original_bytes) > max_image_mb * 1024 * 1024:
        # This is intentionally only a size-limit fallback. The default run does not use it.
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            quality = 90
            from io import BytesIO

            while quality >= 65:
                buffer = BytesIO()
                rgb.save(buffer, format="JPEG", quality=quality, optimize=True)
                candidate = buffer.getvalue()
                if len(candidate) <= max_image_mb * 1024 * 1024:
                    payload = candidate
                    transformed = True
                    note = f"jpeg_reencoded_quality_{quality}"
                    mime_type = "image/jpeg"
                    break
                quality -= 5
            if not transformed:
                payload = candidate
                transformed = True
                note = "jpeg_reencoded_quality_65_still_above_limit"
                mime_type = "image/jpeg"
    encoded = base64.b64encode(payload).decode("ascii")
    return (
        f"data:{mime_type};base64,{encoded}",
        {
            "original_bytes": len(original_bytes),
            "sent_bytes": len(payload),
            "mime_type": mime_type,
            "transformed": transformed,
            "note": note,
        },
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


def load_completed(path: Path, *, include_failures: bool) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not include_failures and record.get("ok") is not True:
                continue
            filename = record.get("filename")
            if isinstance(filename, str):
                completed.add(filename)
    return completed


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def write_aggregate(
    path: Path,
    args: argparse.Namespace,
    manifest_path: Path,
    image_dir: Path,
    selected: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> None:
    selected_filenames = {item["filename"] for item in selected}
    relevant = [record for record in records if record.get("filename") in selected_filenames]
    relevant.sort(key=lambda record: record.get("manifest_index", 10**9))
    payload = {
        "schema_version": "qwen_iteration_aggregate_v1",
        "generated_at": iso_now(),
        "manifest": str(manifest_path),
        "image_dir": str(image_dir),
        "model": args.model,
        "provider_tag": args.provider_tag,
        "allow_fallbacks": bool(args.allow_fallbacks),
        "prompt_version": PROMPT_VERSION,
        "playbook_version": PLAYBOOK_VERSION,
        "model_response_schema_version": SCHEMA_VERSION,
        "selected_manifest_indices": [item.get("_manifest_index") for item in selected],
        "attempted": len(relevant),
        "ok": sum(record.get("ok") is True for record in relevant),
        "failed": sum(record.get("ok") is not True for record in relevant),
        "reported_cost_credits": round(sum_costs(relevant), 8),
        "records": relevant,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sum_costs(records: list[dict[str, Any]]) -> float:
    total = 0.0
    for record in records:
        usage = record.get("response_metadata", {}).get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("cost"), (int, float)):
            total += float(usage["cost"])
    return total


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
    fragment = first_json_fragment(cleaned)
    parsed = json.loads(fragment)
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
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
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


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def safe_stem(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(filename).stem)[:120]


def print_header(
    args: argparse.Namespace,
    manifest_path: Path,
    image_dir: Path,
    output_path: Path,
    selected: list[dict[str, Any]],
    pending: list[dict[str, Any]],
    completed: set[str],
) -> None:
    values = [
        ("manifest", manifest_path),
        ("image_dir", image_dir),
        ("output", output_path),
        ("model", args.model),
        ("provider_tag", args.provider_tag),
        ("allow_fallbacks", args.allow_fallbacks),
        ("selected", len(selected)),
        ("completed_or_skipped", len(completed)),
        ("pending", len(pending)),
        ("response_format", args.response_format),
        ("max_image_mb", args.max_image_mb or "disabled"),
    ]
    for key, value in values:
        print(f"{key}: {value}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

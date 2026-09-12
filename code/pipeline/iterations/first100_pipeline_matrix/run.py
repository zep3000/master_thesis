"""Run one gold-isolated pipeline in the first-100 Qwen pipeline matrix.

P1: one monolithic call per page.
P2: one page structural router, then group/person attribute calls.
P3: page ad locator, per-ad structural crop call, then entity calls.
P4: P3 plus four face tiles, face-guided resolver, and group composition audit.

The module never imports or reads an annotation result or evaluation file.
All OpenRouter attempts across run names/pipelines share one hard request ledger.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import mimetypes
import os
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from schema_and_prompts import (
    AD_STRUCTURE_SCHEMA, AGE_VALUES, COUNT_BANDS, DEPICTION_TYPES, GAZE_VALUES,
    GENDER_VALUES, GROUP_AGES, GROUP_ATTR_FIELDS, GROUP_GAZE, GROUP_GENDERS,
    GROUP_INTENSITY, GROUP_LEGIBILITY, GROUP_SMILE, GROUP_TYPES, INTENSITY_VALUES,
    LEGIBILITY_VALUES, MONOLITHIC_SCHEMA, MOUTH_CAUSES, MOUTH_VALUES,
    ORIENTATION_VALUES, PAGE_LOCATOR_SCHEMA, PAGE_ROUTER_SCHEMA, PERSON_ATTR_FIELDS,
    PLAYBOOK_VERSION, PROMPT_VERSION, SMILE_VALUES, ad_structure_prompt,
    composition_prompt, group_prompt, monolithic_prompt, page_locator_prompt,
    page_router_prompt, person_prompt, resolver_prompt, tile_prompt,
)

ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "first100_pipeline_matrix"
MANIFEST = HERE / "data" / "input_manifest_first100.json"
IMAGE_DIR = ROOT / "code" / "test_collection_200_difficult_joined_pages"
API_KEY_PATH = ROOT / "openrouter_key.txt"
BASE_URL = "https://openrouter.ai/api/v1"
SHARED_LEDGER = HERE / "output" / "shared_request_ledger.jsonl"
PIPELINE_LOGIC_VERSION = "first100_matrix_logic_v3"

STAGE_FILES = {
    "monolithic": "monolithic.jsonl", "router": "router.jsonl", "locator": "locator.jsonl",
    "ad_structure": "ad_structure.jsonl", "tiles": "tiles.jsonl", "resolver": "resolver.jsonl",
    "groups": "groups.jsonl", "persons": "persons.jsonl", "composition": "composition.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["p1", "p2", "p3", "p4"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", choices=["all", "structure", "tiles", "resolver", "entities", "assemble"], default="all")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument("--provider-tag", default="venice/fp8")
    parser.add_argument("--max-requests", type=int, default=3000)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--force-stage", action="append", choices=list(STAGE_FILES), default=[], help="Rerun every task in the named stage; repeatable. Use after an upstream retry changes inputs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--p1-max-tokens", type=int, default=14000, help="Use a larger value only when retrying a P1 response truncated at the normal 14k ceiling.")
    parser.add_argument("--locator-max-tokens", type=int, default=3000, help="Use a larger value only when retrying a truncated locator response.")
    parser.add_argument("--composition-max-tokens", type=int, default=7500, help="Use a larger value only when retrying a truncated P4 group face-inventory response.")
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, payload: dict[str, Any], lock: threading.Lock | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    if lock:
        with lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
    else:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def latest_records(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        latest[str(row["task_key"])] = row
    return latest


def latest_ok(path: Path) -> dict[str, dict[str, Any]]:
    return {key: row for key, row in latest_records(path).items() if row.get("ok") is True}


class RequestBudget:
    def __init__(self, limit: int):
        self.limit = limit
        self.lock = threading.Lock()
        rows = read_jsonl(SHARED_LEDGER)
        self.used = len(rows)
        self.next_sequence = max((int(row.get("request_sequence", 0)) for row in rows), default=0) + 1

    def reserve(self) -> int:
        with self.lock:
            if self.used >= self.limit:
                raise RuntimeError(f"Shared OpenRouter budget exhausted: {self.used}/{self.limit}")
            sequence = self.next_sequence
            self.next_sequence += 1
            self.used += 1
            return sequence

    def record(self, payload: dict[str, Any]) -> None:
        append_jsonl(SHARED_LEDGER, payload, self.lock)


def load_key() -> str:
    if API_KEY_PATH.exists() and API_KEY_PATH.read_text(encoding="utf-8").strip():
        return API_KEY_PATH.read_text(encoding="utf-8").strip()
    if os.getenv("OPENROUTER_API_KEY"):
        return str(os.getenv("OPENROUTER_API_KEY")).strip()
    raise SystemExit("OpenRouter key missing")


def build_client(args: argparse.Namespace) -> Any:
    import httpx
    from openai import OpenAI
    return OpenAI(
        api_key=load_key(), base_url=BASE_URL, timeout=args.timeout, max_retries=0,
        http_client=httpx.Client(timeout=args.timeout, trust_env=False),
        default_headers={"X-Title": "Qwen First100 Pipeline Matrix"},
    )


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def response_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(item.get("text", "")) for item in value if isinstance(item, dict))
    return "" if value is None else str(value)


def plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return plain(value.model_dump())
    return str(value)


def extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.DOTALL).strip()
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    depth = 0
    quoted = escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif not quoted and char == "{":
            depth += 1
        elif not quoted and char == "}":
            depth -= 1
            if depth == 0:
                value = json.loads(cleaned[start : index + 1])
                if isinstance(value, dict):
                    return value
    raise ValueError("unterminated JSON")


def bbox(value: Any) -> list[int] | None:
    if isinstance(value, str):
        parts = re.findall(r"-?\d+(?:\.\d+)?", value)
        value = [float(part) for part in parts] if len(parts) == 4 else value
    if not isinstance(value, list) or len(value) != 4 or not all(isinstance(part, (int, float)) for part in value):
        return None
    result = [max(0, min(1000, int(round(part)))) for part in value]
    return result if result[0] < result[2] and result[1] < result[3] else None


def normalize_bboxes(value: Any, actions: list[str], path: str = "") -> Any:
    if isinstance(value, list):
        return [normalize_bboxes(item, actions, f"{path}[{index}]") for index, item in enumerate(value)]
    if not isinstance(value, dict):
        return value
    output = {}
    for key, item in value.items():
        item_path = f"{path}.{key}" if path else key
        if key in {"bbox_1000", "face_bbox_1000"}:
            repaired = bbox(item)
            output[key] = repaired if repaired is not None else item
            if repaired is not None and repaired != item:
                actions.append(f"{item_path} normalized")
        else:
            output[key] = normalize_bboxes(item, actions, item_path)
    return output


def force_id(annotation: dict[str, Any], field: str, expected: str, actions: list[str]) -> None:
    if annotation.get(field) != expected:
        annotation[field] = expected
        actions.append(f"{field} assigned from local task identity")


def normalize_person_attributes(person: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    if person.get("face_expression_legibility") == "0_not_legible":
        for field in ["gaze_target", "gaze_target_person_unboxed", "smile_present", "smile_intensity"]:
            if person.get(field) is not None:
                person[field] = None; actions.append(f"{field}=null for 0_not_legible")
    if person.get("mouth_covered") == "no" and person.get("mouth_covering") is not None:
        person["mouth_covering"] = None; actions.append("mouth_covering=null for uncovered mouth")
    if person.get("smile_present") != "yes" and person.get("smile_intensity") is not None:
        person["smile_intensity"] = None; actions.append("smile_intensity=null without visible smile")
    if person.get("gaze_target") != "another_person" and person.get("gaze_target_person_unboxed") is not None:
        person["gaze_target_person_unboxed"] = None; actions.append("gaze_target_person_unboxed=null without another-person gaze")
    return actions


def normalize_group_attributes(group: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    age_repair = {"young": "young_only", "middle": "middle_only", "older": "older_only"}
    gender_repair = {"feminine": "feminine_only", "masculine": "masculine_only"}
    if group.get("age_composition") in age_repair:
        group["age_composition"] = age_repair[group["age_composition"]]; actions.append("expanded group age shorthand")
    if group.get("gender_presentation_composition") in gender_repair:
        group["gender_presentation_composition"] = gender_repair[group["gender_presentation_composition"]]; actions.append("expanded group gender shorthand")
    if group.get("dominant_smile_intensity") in {"not_applicable", "not_assessable"}:
        group["dominant_smile_intensity"] = None; actions.append("group not_applicable intensity normalized to null")
    if group.get("expression_legibility_distribution") == "all_0_not_legible":
        for field in ["dominant_gaze", "smile_prevalence", "dominant_smile_intensity"]:
            if group.get(field) is not None:
                group[field] = None; actions.append(f"{field}=null for all_0_not_legible")
    if group.get("smile_prevalence") in {"none", "not_assessable", None} and group.get("dominant_smile_intensity") is not None:
        group["dominant_smile_intensity"] = None; actions.append("group intensity=null without positive prevalence")
    return actions


def normalize_routes(annotation: dict[str, Any], *, page_level: bool) -> list[str]:
    actions: list[str] = []
    ads = annotation.get("advertisements") if page_level else [annotation]
    if not isinstance(ads, list):
        return actions
    for ad in ads:
        if not isinstance(ad, dict):
            continue
        band = str(ad.get("face_depiction_count_band") or "")
        people = [person for person in ad.get("people") or [] if isinstance(person, dict) and bbox(person.get("face_bbox_1000"))]
        groups = [group for group in ad.get("groups") or [] if isinstance(group, dict) and bbox(group.get("bbox_1000"))]
        if len(groups) > 6:
            groups = sorted(groups, key=lambda item: float(item.get("confidence") or 0), reverse=True)[:6]
            actions.append("capped people areas at 6")
        for person in people:
            actions.extend(normalize_person_attributes(person))
        for group in groups:
            actions.extend(normalize_group_attributes(group))
        if band == "0":
            ad["people"] = []; ad["groups"] = []; ad["unique_face_count"] = 0
            actions.append("marked zero-face ad candidate for discard")
            continue
        if band in {str(i) for i in range(1, 10)}:
            people = people[:9]
            for person in people:
                person["annotation_role"] = "individual"
                person["prominence_reason"] = None
            if people:
                ad["face_depiction_count_band"] = str(len(people))
            ad["groups"] = []
            if groups:
                actions.append("removed groups from exact 1-9 route")
        elif band in {"10_20", "20_plus"}:
            for person in people:
                person["annotation_role"] = "outstanding_individual"
            if len(people) > 3:
                people = sorted(people, key=lambda item: float(item.get("confidence") or 0), reverse=True)[:3]
                actions.append("capped crowd outstanding individuals at 3")
            ad["groups"] = groups
        ad["people"] = people
    return actions


def enum_valid(value: Any, allowed: list[Any], nullable: bool = False) -> bool:
    return (nullable and value is None) or value in allowed


def validate_annotation(stage: str, annotation: dict[str, Any], expected_id: str) -> list[str]:
    errors: list[str] = []
    id_fields = {"monolithic": "image_id", "router": "image_id", "locator": "image_id", "ad_structure": "ad_task_id", "tiles": "tile_task_id", "resolver": "ad_task_id", "groups": "group_task_id", "persons": "person_task_id", "composition": "group_task_id"}
    field = id_fields[stage]
    if annotation.get(field) != expected_id:
        errors.append(f"{field} mismatch")
    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"bbox_1000", "face_bbox_1000"} and bbox(item) is None:
                    errors.append(f"invalid {path + '.' if path else ''}{key}")
                else:
                    walk(item, f"{path}.{key}" if path else key)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
    walk(annotation)
    if stage == "groups":
        checks = {
            "group_type": GROUP_TYPES, "age_composition": GROUP_AGES,
            "gender_presentation_composition": GROUP_GENDERS,
            "expression_legibility_distribution": GROUP_LEGIBILITY,
            "dominant_gaze": [*GROUP_GAZE, None], "smile_prevalence": [*GROUP_SMILE, None],
            "dominant_smile_intensity": [*GROUP_INTENSITY, None],
        }
        errors.extend(f"invalid {key}" for key, allowed in checks.items() if annotation.get(key) not in allowed)
    if stage == "persons":
        checks = {
            "depiction_type": [*DEPICTION_TYPES, None], "perceived_age": AGE_VALUES,
            "perceived_gender_presentation": GENDER_VALUES, "face_expression_legibility": LEGIBILITY_VALUES,
            "face_orientation": ORIENTATION_VALUES, "gaze_target": [*GAZE_VALUES, None],
            "mouth_covered": MOUTH_VALUES, "mouth_covering": [*MOUTH_CAUSES, None],
            "smile_present": [*SMILE_VALUES, None], "smile_intensity": [*INTENSITY_VALUES, None],
        }
        errors.extend(f"invalid {key}" for key, allowed in checks.items() if annotation.get(key) not in allowed)
    route_ads = annotation.get("advertisements") if stage in {"monolithic", "router"} else [annotation] if stage == "ad_structure" else []
    for ad_index, ad in enumerate(route_ads or []):
        prefix = f"advertisements[{ad_index}]" if stage in {"monolithic", "router"} else "ad"
        allowed_bands = ["0", *COUNT_BANDS] if stage == "ad_structure" else COUNT_BANDS
        if ad.get("face_depiction_count_band") not in allowed_bands:
            errors.append(f"invalid {prefix}.face_depiction_count_band")
        if ad.get("depiction_type") not in DEPICTION_TYPES:
            errors.append(f"invalid {prefix}.depiction_type")
        for person_index, person in enumerate(ad.get("people") or []):
            if person.get("annotation_role") not in {"individual", "outstanding_individual", "duplicate"}:
                errors.append(f"invalid {prefix}.people[{person_index}].annotation_role")
            if stage == "monolithic":
                checks = {"perceived_age": AGE_VALUES, "perceived_gender_presentation": GENDER_VALUES, "face_expression_legibility": LEGIBILITY_VALUES, "face_orientation": ORIENTATION_VALUES, "gaze_target": [*GAZE_VALUES, None], "mouth_covered": MOUTH_VALUES, "mouth_covering": [*MOUTH_CAUSES, None], "smile_present": [*SMILE_VALUES, None], "smile_intensity": [*INTENSITY_VALUES, None]}
                errors.extend(f"invalid {prefix}.people[{person_index}].{key}" for key, allowed in checks.items() if person.get(key) not in allowed)
        if stage == "monolithic":
            for group_index, group in enumerate(ad.get("groups") or []):
                checks = {"group_type": GROUP_TYPES, "age_composition": GROUP_AGES, "gender_presentation_composition": GROUP_GENDERS, "expression_legibility_distribution": GROUP_LEGIBILITY, "dominant_gaze": [*GROUP_GAZE, None], "smile_prevalence": [*GROUP_SMILE, None], "dominant_smile_intensity": [*GROUP_INTENSITY, None]}
                errors.extend(f"invalid {prefix}.groups[{group_index}].{key}" for key, allowed in checks.items() if group.get(key) not in allowed)
    for required in {
        "monolithic": ["advertisements", "qualifying_ad_count"], "router": ["advertisements", "qualifying_ad_count"],
        "locator": ["qualifying_ads"], "ad_structure": ["people", "groups", "face_depiction_count_band"],
        "tiles": ["faces"], "resolver": ["people", "groups", "accepted_face_ids"],
        "groups": list(GROUP_ATTR_FIELDS), "persons": list(PERSON_ATTR_FIELDS), "composition": ["faces", "faces_truncated"],
    }[stage]:
        if required not in annotation:
            errors.append(f"missing {required}")
    return errors


def request_kwargs(args: argparse.Namespace, prompt: str, image_path: Path, max_tokens: int) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a conservative visual annotation researcher. Return only the requested JSON object."},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(image_path)}}, {"type": "text", "text": prompt}]},
        ],
        "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0,
        "response_format": {"type": "json_object"},
        "extra_body": {"provider": {"order": [args.provider_tag], "allow_fallbacks": False, "require_parameters": True}, "reasoning": {"effort": "none", "exclude": True}},
    }


def call_one(client: Any, budget: RequestBudget, args: argparse.Namespace, output_dir: Path, stage: str, task_key: str, expected_id: str, prompt: str, image_path: Path, max_tokens: int, extra_normalize: Callable[[dict[str, Any]], list[str]] | None = None) -> dict[str, Any]:
    sequence = budget.reserve()
    started = time.time()
    record: dict[str, Any] = {
        "record_schema": "qwen_first100_matrix_call_v1", "run_name": args.run_name, "pipeline": args.pipeline,
        "stage": stage, "task_key": task_key, "request_sequence": sequence, "started_at": iso_now(),
        "prompt_version": PROMPT_VERSION, "pipeline_logic_version": PIPELINE_LOGIC_VERSION,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "image_path": str(image_path), "request": {"model": args.model, "provider": args.provider_tag, "max_tokens": max_tokens},
    }
    raw_dir = output_dir / "raw" / stage
    prompt_dir = output_dir / "prompts" / stage
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / f"{safe(task_key)}.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    record["prompt_path"] = str(prompt_path)
    ledger = {"request_sequence": sequence, "run_name": args.run_name, "pipeline": args.pipeline, "stage": stage, "task_key": task_key, "started_at": record["started_at"]}
    try:
        response = client.chat.completions.create(**request_kwargs(args, prompt, image_path, max_tokens))
        metadata = plain(response)
        raw = response_text(response.choices[0].message.content)
        raw_path = raw_dir / f"{safe(task_key)}.raw.txt"
        meta_path = raw_dir / f"{safe(task_key)}.response.json"
        raw_path.write_text(raw, encoding="utf-8")
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        parsed = extract_json(raw)
        actions: list[str] = []
        normalized = normalize_bboxes(parsed, actions)
        force_id(normalized, {"monolithic": "image_id", "router": "image_id", "locator": "image_id", "ad_structure": "ad_task_id", "tiles": "tile_task_id", "resolver": "ad_task_id", "groups": "group_task_id", "persons": "person_task_id", "composition": "group_task_id"}[stage], expected_id, actions)
        if extra_normalize:
            actions.extend(extra_normalize(normalized))
        errors = validate_annotation(stage, normalized, expected_id)
        record.update({"raw_response_path": str(raw_path), "response_metadata_path": str(meta_path), "usage": metadata.get("usage"), "model_annotation_raw": parsed, "model_annotation": normalized, "normalization_actions": actions, "validation_errors": errors, "ok": not errors})
        ledger.update({"ok": not errors, "usage": metadata.get("usage"), "response_id": metadata.get("id")})
    except Exception as exc:
        record.update({"ok": False, "error": f"{type(exc).__name__}: {exc}", "validation_errors": []})
        ledger.update({"ok": False, "error": record["error"]})
    record["elapsed_seconds"] = round(time.time() - started, 3)
    ledger.update({"elapsed_seconds": record["elapsed_seconds"], "finished_at": iso_now()})
    budget.record(ledger)
    return record


def run_tasks(client: Any, budget: RequestBudget, args: argparse.Namespace, output_dir: Path, stage: str, tasks: list[dict[str, Any]], worker: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    path = output_dir / STAGE_FILES[stage]
    latest = latest_records(path)
    done = set() if stage in args.force_stage else ({key for key, row in latest.items() if row.get("ok") is True} if args.retry_failures else set(latest))
    pending = [task for task in tasks if task["task_key"] not in done]
    print(f"{args.pipeline}/{stage}: tasks={len(tasks)} completed={len(done)} pending={len(pending)} shared={budget.used}/{budget.limit}")
    if args.dry_run or not pending:
        return
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(worker, task): task for task in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {"run_name": args.run_name, "pipeline": args.pipeline, "stage": stage, "task_key": task["task_key"], "ok": False, "error": f"worker: {type(exc).__name__}: {exc}"}
            append_jsonl(path, record, lock)
            print(f"[{stage} {index}/{len(pending)}] {task['task_key']} ok={record.get('ok')} requests={budget.used}/{budget.limit}")


def crop_from_box(image_path: Path, box: list[int], output_path: Path) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    x1, y1, x2, y2 = box
    pixels = (int(x1 / 1000 * image.width), int(y1 / 1000 * image.height), max(1, int(math.ceil(x2 / 1000 * image.width))), max(1, int(math.ceil(y2 / 1000 * image.height))))
    crop = image.crop(pixels)
    if min(crop.size) < 64:
        scale = 64 / min(crop.size)
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, "JPEG", quality=95, optimize=True)


def letterbox(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, tuple[float, int, int]]:
    scale = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    offsets = ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2)
    canvas.paste(resized, offsets)
    return canvas, (scale, offsets[0], offsets[1])


def person_composite(page_path: Path, face_box: list[int], context_box: list[int], output_path: Path, label: str) -> None:
    with Image.open(page_path) as source:
        page = source.convert("RGB")
    def pixels(box_: list[int]) -> list[float]:
        return [box_[0] / 1000 * page.width, box_[1] / 1000 * page.height, box_[2] / 1000 * page.width, box_[3] / 1000 * page.height]
    fp = pixels(face_box); cp = pixels(context_box)
    fw, fh = fp[2] - fp[0], fp[3] - fp[1]
    face_crop = page.crop((max(0, int(fp[0] - .45 * fw)), max(0, int(fp[1] - .45 * fh)), min(page.width, int(fp[2] + .45 * fw)), min(page.height, int(fp[3] + .45 * fh))))
    context = page.crop(tuple(map(int, cp)))
    left, _ = letterbox(face_crop, (700, 700)); right, transform = letterbox(context, (700, 700))
    scale, ox, oy = transform
    rel = [fp[0] - cp[0], fp[1] - cp[1], fp[2] - cp[0], fp[3] - cp[1]]
    draw = ImageDraw.Draw(right); draw.rectangle([ox + rel[0] * scale, oy + rel[1] * scale, ox + rel[2] * scale, oy + rel[3] * scale], outline="red", width=6)
    canvas = Image.new("RGB", (1400, 740), "white"); canvas.paste(left, (0, 40)); canvas.paste(right, (700, 40))
    draw = ImageDraw.Draw(canvas); font = ImageFont.load_default(); draw.text((10, 10), f"TARGET {label}", fill="black", font=font); draw.text((710, 10), "AD CONTEXT (target red)", fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True); canvas.save(output_path, "JPEG", quality=95, optimize=True)


def map_child_box(child: list[int], parent: list[int]) -> list[int]:
    pw, ph = parent[2] - parent[0], parent[3] - parent[1]
    return [round(parent[0] + child[0] / 1000 * pw), round(parent[1] + child[1] / 1000 * ph), round(parent[0] + child[2] / 1000 * pw), round(parent[1] + child[3] / 1000 * ph)]


def normalize_structural_page(annotation: dict[str, Any]) -> list[str]:
    actions = normalize_routes(annotation, page_level=True)
    ads = annotation.get("advertisements") if isinstance(annotation.get("advertisements"), list) else []
    dropped = sum(str(ad.get("face_depiction_count_band")) == "0" for ad in ads if isinstance(ad, dict))
    ads = [ad for ad in ads if isinstance(ad, dict) and str(ad.get("face_depiction_count_band")) != "0"]
    if dropped:
        actions.append(f"removed {dropped} nonqualifying zero-face advertisements")
    annotation["advertisements"] = ads
    annotation["qualifying_ad_count"] = len(ads)
    annotation["no_qualifying_ad_reason"] = None if ads else annotation.get("no_qualifying_ad_reason") or "ads_present_no_visible_faces"
    return actions


def normalize_ad_structure(annotation: dict[str, Any]) -> list[str]:
    return normalize_routes(annotation, page_level=False)


def dedupe_locator_ads(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    candidates = annotation.get("qualifying_ads") if isinstance(annotation.get("qualifying_ads"), list) else []
    for candidate in sorted((item for item in candidates if isinstance(item, dict) and bbox(item.get("bbox_1000"))), key=lambda item: float(item.get("confidence") or 0), reverse=True):
        candidate_box = bbox(candidate.get("bbox_1000"))
        if candidate_box and not any(box_iou(candidate_box, bbox(other.get("bbox_1000")) or [0, 0, 1, 1]) >= .85 for other in kept):
            kept.append(candidate)
        if len(kept) == 6:
            break
    return kept


def normalize_page_locator(annotation: dict[str, Any]) -> list[str]:
    before = len(annotation.get("qualifying_ads") or [])
    annotation["qualifying_ads"] = dedupe_locator_ads(annotation)
    after = len(annotation["qualifying_ads"])
    annotation["no_qualifying_ad_reason"] = None if after else annotation.get("no_qualifying_ad_reason") or "ads_present_no_visible_faces"
    return [f"deduplicated/capped locator ads {before}->{after}"] if before != after else []


def selected_images(args: argparse.Namespace) -> list[dict[str, Any]]:
    images = read_json(MANIFEST)["images"]
    return [item for item in images if args.start_index <= int(item["manifest_index"]) < min(100, args.start_index + args.limit)]


def stage_structure(client: Any, budget: RequestBudget, args: argparse.Namespace, output_dir: Path, images: list[dict[str, Any]]) -> None:
    if args.pipeline == "p1":
        tasks = [{"task_key": item["image_id"], **item} for item in images]
        run_tasks(client, budget, args, output_dir, "monolithic", tasks, lambda t: call_one(client, budget, args, output_dir, "monolithic", t["task_key"], t["image_id"], monolithic_prompt(t["image_id"], t["metadata"]), IMAGE_DIR / t["filename"], args.p1_max_tokens, normalize_structural_page))
        return
    if args.pipeline == "p2":
        tasks = [{"task_key": item["image_id"], **item} for item in images]
        run_tasks(client, budget, args, output_dir, "router", tasks, lambda t: call_one(client, budget, args, output_dir, "router", t["task_key"], t["image_id"], page_router_prompt(t["image_id"], t["metadata"]), IMAGE_DIR / t["filename"], 8500, normalize_structural_page))
        return
    tasks = [{"task_key": item["image_id"], **item} for item in images]
    run_tasks(client, budget, args, output_dir, "locator", tasks, lambda t: call_one(client, budget, args, output_dir, "locator", t["task_key"], t["image_id"], page_locator_prompt(t["image_id"], t["metadata"]), IMAGE_DIR / t["filename"], args.locator_max_tokens, normalize_page_locator))
    locators = latest_ok(output_dir / STAGE_FILES["locator"])
    ad_tasks = []
    image_lookup = {item["image_id"]: item for item in images}
    for image_id, record in locators.items():
        item = image_lookup.get(image_id)
        if not item:
            continue
        for index, ad in enumerate(dedupe_locator_ads(record["model_annotation"]), start=1):
            ad_box = bbox(ad.get("bbox_1000"))
            if not ad_box:
                continue
            task_key = f"{image_id}::ad{index}"
            crop_path = output_dir / "crops" / "ads" / f"{safe(task_key)}.jpg"
            crop_from_box(IMAGE_DIR / item["filename"], ad_box, crop_path)
            ad_tasks.append({"task_key": task_key, "ad_task_id": task_key, "image_id": image_id, "filename": item["filename"], "ad_index": index, "ad_bbox_page": ad_box, "extent": ad.get("extent") or "partial_page", "crop_path": str(crop_path)})
    (output_dir / "ad_tasks.json").write_text(json.dumps({"tasks": ad_tasks}, indent=2) + "\n", encoding="utf-8")
    run_tasks(client, budget, args, output_dir, "ad_structure", ad_tasks, lambda t: call_one(client, budget, args, output_dir, "ad_structure", t["task_key"], t["ad_task_id"], ad_structure_prompt(t["ad_task_id"]), Path(t["crop_path"]), 7000, normalize_ad_structure))


def tile_boxes() -> list[tuple[str, list[int]]]:
    return [("r1c1", [0, 0, 600, 600]), ("r1c2", [400, 0, 1000, 600]), ("r2c1", [0, 400, 600, 1000]), ("r2c2", [400, 400, 1000, 1000])]


def stage_tiles(client: Any, budget: RequestBudget, args: argparse.Namespace, output_dir: Path) -> None:
    if args.pipeline != "p4":
        return
    ad_tasks = read_json(output_dir / "ad_tasks.json")["tasks"]
    tasks = []
    for ad in ad_tasks:
        with Image.open(ad["crop_path"]) as source:
            image = source.convert("RGB")
        for tile_id, box1000 in tile_boxes():
            px = (round(box1000[0] / 1000 * image.width), round(box1000[1] / 1000 * image.height), round(box1000[2] / 1000 * image.width), round(box1000[3] / 1000 * image.height))
            tile = image.crop(px)
            if min(tile.size) < 64:
                scale = 64 / min(tile.size); tile = tile.resize((round(tile.width * scale), round(tile.height * scale)), Image.Resampling.LANCZOS)
            task_key = f"{ad['task_key']}::{tile_id}"
            path = output_dir / "crops" / "tiles" / f"{safe(task_key)}.jpg"; path.parent.mkdir(parents=True, exist_ok=True); tile.save(path, "JPEG", quality=95, optimize=True)
            tasks.append({"task_key": task_key, "tile_task_id": task_key, "ad_task_id": ad["task_key"], "tile_id": tile_id, "tile_bbox_ad": box1000, "path": str(path)})
    (output_dir / "tile_tasks.json").write_text(json.dumps({"tasks": tasks}, indent=2) + "\n", encoding="utf-8")
    run_tasks(client, budget, args, output_dir, "tiles", tasks, lambda t: call_one(client, budget, args, output_dir, "tiles", t["task_key"], t["tile_task_id"], tile_prompt(t["tile_task_id"]), Path(t["path"]), 3200))


def box_iou(a: list[int], b: list[int]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0


def merge_boxes(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: list[list[dict[str, Any]]] = []
    for candidate in sorted(candidates, key=lambda item: float(item.get("confidence") or 0), reverse=True):
        match = next((cluster for cluster in clusters if any(box_iou(candidate["bbox_1000"], other["bbox_1000"]) >= .20 for other in cluster)), None)
        if match is None:
            clusters.append([candidate])
        else:
            match.append(candidate)
    merged = []
    for index, cluster in enumerate(clusters, start=1):
        weights = [max(.1, float(item.get("confidence") or .5)) for item in cluster]; total = sum(weights)
        box_ = [round(sum(item["bbox_1000"][axis] * weight for item, weight in zip(cluster, weights)) / total) for axis in range(4)]
        merged.append({"face_id": f"face_{index}", "bbox_1000": box_, "confidence": max(weights), "sources": len(cluster)})
    return sorted(merged, key=lambda item: ((item["bbox_1000"][1] + item["bbox_1000"][3]) / 2, (item["bbox_1000"][0] + item["bbox_1000"][2]) / 2))[:40]


def stage_resolver(client: Any, budget: RequestBudget, args: argparse.Namespace, output_dir: Path) -> None:
    if args.pipeline != "p4":
        return
    ad_tasks = {task["task_key"]: task for task in read_json(output_dir / "ad_tasks.json")["tasks"]}
    tile_tasks = read_json(output_dir / "tile_tasks.json")["tasks"]
    tiles = latest_ok(output_dir / STAGE_FILES["tiles"]); structures = latest_ok(output_dir / STAGE_FILES["ad_structure"])
    by_ad: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tile_tasks:
        record = tiles.get(task["task_key"])
        if not record:
            continue
        parent = task["tile_bbox_ad"]
        for face in record["model_annotation"].get("faces") or []:
            face_box = bbox(face.get("bbox_1000"))
            if face_box:
                by_ad[task["ad_task_id"]].append({"bbox_1000": map_child_box(face_box, parent), "confidence": face.get("confidence", .5)})
    tasks = []
    for ad_key, ad in ad_tasks.items():
        proposals = merge_boxes(by_ad.get(ad_key, []))
        with Image.open(ad["crop_path"]) as source:
            image = source.convert("RGB")
        panel, transform = letterbox(image, (1200, 1200)); scale, ox, oy = transform; draw = ImageDraw.Draw(panel); font = ImageFont.load_default()
        for proposal in proposals:
            b = proposal["bbox_1000"]
            rect = [ox + b[0] / 1000 * image.width * scale, oy + b[1] / 1000 * image.height * scale, ox + b[2] / 1000 * image.width * scale, oy + b[3] / 1000 * image.height * scale]
            draw.rectangle(rect, outline="red", width=4); draw.text((rect[0], max(0, rect[1] - 12)), proposal["face_id"], fill="red", font=font)
        overlay = output_dir / "crops" / "resolver" / f"{safe(ad_key)}.jpg"; overlay.parent.mkdir(parents=True, exist_ok=True); panel.save(overlay, "JPEG", quality=95)
        preliminary = (structures.get(ad_key) or {}).get("model_annotation") or {}
        tasks.append({"task_key": ad_key, "ad_task_id": ad_key, "overlay_path": str(overlay), "proposals": proposals, "preliminary": preliminary})
    (output_dir / "resolver_tasks.json").write_text(json.dumps({"tasks": tasks}, indent=2) + "\n", encoding="utf-8")
    run_tasks(client, budget, args, output_dir, "resolver", tasks, lambda t: call_one(client, budget, args, output_dir, "resolver", t["task_key"], t["ad_task_id"], resolver_prompt(t["ad_task_id"], t["preliminary"], t["proposals"]), Path(t["overlay_path"]), 5500))


def canonical_routes(args: argparse.Namespace, output_dir: Path, images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    image_lookup = {item["image_id"]: item for item in images}
    pages = []
    if args.pipeline == "p2":
        for image_id, record in latest_ok(output_dir / STAGE_FILES["router"]).items():
            if image_id not in image_lookup:
                continue
            annotation = json.loads(json.dumps(record["model_annotation"]))
            normalize_routes(annotation, page_level=True)
            pages.append({"image": image_lookup[image_id], "no_reason": annotation.get("no_qualifying_ad_reason"), "ads": annotation.get("advertisements") or []})
        return pages
    locators = latest_ok(output_dir / STAGE_FILES["locator"]); structures = latest_ok(output_dir / STAGE_FILES["ad_structure"])
    ad_tasks = read_json(output_dir / "ad_tasks.json")["tasks"] if (output_dir / "ad_tasks.json").exists() else []
    resolvers = latest_ok(output_dir / STAGE_FILES["resolver"]) if args.pipeline == "p4" else {}
    resolver_tasks = {task["ad_task_id"]: task for task in read_json(output_dir / "resolver_tasks.json")["tasks"]} if args.pipeline == "p4" and (output_dir / "resolver_tasks.json").exists() else {}
    ads_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in ad_tasks:
        structure = (structures.get(task["task_key"]) or {}).get("model_annotation")
        if not structure or str(structure.get("face_depiction_count_band")) == "0":
            continue
        route = json.loads(json.dumps(structure))
        if args.pipeline == "p4" and task["task_key"] in resolvers:
            resolved = resolvers[task["task_key"]]["model_annotation"]
            route["face_depiction_count_band"] = resolved.get("face_depiction_count_band", route.get("face_depiction_count_band"))
            proposal_lookup = {p["face_id"]: p for p in resolver_tasks[task["task_key"]]["proposals"]}
            route["people"] = []
            for item in resolved.get("people") or []:
                proposal = proposal_lookup.get(item.get("face_id"))
                if proposal:
                    route["people"].append({"person_id": item["face_id"], "face_bbox_1000": proposal["bbox_1000"], "annotation_role": item.get("annotation_role"), "prominence_reason": item.get("prominence_reason"), "confidence": proposal.get("confidence", .5)})
            route["groups"] = resolved.get("groups") or []
        normalize_routes(route, page_level=False)
        page_box = task["ad_bbox_page"]
        route["advertisement_id"] = f"ad_{task['ad_index']}"; route["extent"] = task["extent"]; route["bbox_1000"] = page_box
        for person in route.get("people") or []:
            person["face_bbox_1000"] = map_child_box(person["face_bbox_1000"], page_box)
        for group in route.get("groups") or []:
            group["bbox_1000"] = map_child_box(group["bbox_1000"], page_box)
        ads_by_image[task["image_id"]].append(route)
    for image_id, locator in locators.items():
        if image_id in image_lookup:
            pages.append({"image": image_lookup[image_id], "no_reason": locator["model_annotation"].get("no_qualifying_ad_reason"), "ads": sorted(ads_by_image.get(image_id, []), key=lambda ad: ad["bbox_1000"][1])})
    return pages


def prepare_entity_tasks(args: argparse.Namespace, output_dir: Path, images: list[dict[str, Any]]) -> dict[str, Any]:
    pages = canonical_routes(args, output_dir, images); groups = []; persons = []
    for page in pages:
        image = page["image"]; page_path = IMAGE_DIR / image["filename"]
        for ad_index, ad in enumerate(page["ads"], start=1):
            ad_id = f"ad_{ad_index}"; ad["advertisement_id"] = ad_id
            for group_index, group in enumerate(ad.get("groups") or [], start=1):
                box_ = bbox(group.get("bbox_1000"))
                if not box_: continue
                group_id = f"{ad_id}_group_{group_index}"; task_key = f"{image['image_id']}::{group_id}"; path = output_dir / "crops" / "groups" / f"{safe(task_key)}.jpg"; crop_from_box(page_path, box_, path)
                group["group_id"] = group_id
                groups.append({"task_key": task_key, "group_task_id": task_key, "image_id": image["image_id"], "ad_id": ad_id, "group_id": group_id, "bbox_page": box_, "crop_path": str(path)})
            for person_index, person in enumerate(ad.get("people") or [], start=1):
                box_ = bbox(person.get("face_bbox_1000"))
                if not box_: continue
                person_id = f"{ad_id}_person_{person_index}"; task_key = f"{image['image_id']}::{person_id}"; path = output_dir / "crops" / "persons" / f"{safe(task_key)}.jpg"; person_composite(page_path, box_, ad["bbox_1000"], path, task_key)
                person["person_id"] = person_id
                persons.append({"task_key": task_key, "person_task_id": task_key, "image_id": image["image_id"], "ad_id": ad_id, "person_id": person_id, "role": person.get("annotation_role") or "individual", "bbox_page": box_, "ad_depiction_type": ad.get("depiction_type") or "multiple_types_present", "composite_path": str(path)})
    payload = {"pages": pages, "groups": groups, "persons": persons}
    (output_dir / "entity_tasks.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def stage_entities(client: Any, budget: RequestBudget, args: argparse.Namespace, output_dir: Path, images: list[dict[str, Any]]) -> None:
    if args.pipeline == "p1": return
    tasks = prepare_entity_tasks(args, output_dir, images)
    run_tasks(client, budget, args, output_dir, "groups", tasks["groups"], lambda t: call_one(client, budget, args, output_dir, "groups", t["task_key"], t["group_task_id"], group_prompt(t["group_task_id"]), Path(t["crop_path"]), 2800, normalize_group_attributes))
    run_tasks(client, budget, args, output_dir, "persons", tasks["persons"], lambda t: call_one(client, budget, args, output_dir, "persons", t["task_key"], t["person_task_id"], person_prompt(t["person_task_id"], t["ad_depiction_type"]), Path(t["composite_path"]), 2600, normalize_person_attributes))
    if args.pipeline == "p4":
        run_tasks(client, budget, args, output_dir, "composition", tasks["groups"], lambda t: call_one(client, budget, args, output_dir, "composition", t["task_key"], t["group_task_id"], composition_prompt(t["group_task_id"]), Path(t["crop_path"]), args.composition_max_tokens))


def aggregate_composition(annotation: dict[str, Any]) -> dict[str, str] | None:
    faces = annotation.get("faces") or []
    if annotation.get("faces_truncated") or len(faces) < 2:
        return None
    age_map = {"infant": "young", "child": "young", "adolescent": "young", "young_adult": "young", "middle_adult": "middle", "older_adult": "older"}
    ages = [age_map[f.get("perceived_age")] for f in faces if f.get("perceived_age") in age_map]
    genders = [f.get("perceived_gender_presentation") for f in faces if f.get("perceived_gender_presentation") in {"feminine", "masculine", "ambiguous_or_androgynous"}]
    if len(ages) / len(faces) < .5 or len(genders) / len(faces) < .5:
        return None
    age, age_n = Counter(ages).most_common(1)[0]; age_share = age_n / len(ages)
    age_value = f"{age}_only" if age_share == 1 else f"mostly_{age}" if age_share > .5 else "mixed"
    ambiguous = [f for f in faces if f.get("perceived_gender_presentation") == "ambiguous_or_androgynous" and float(f.get("confidence") or 0) >= .8]
    if ambiguous:
        gender_value = "ambiguous_or_androgynous_present"
    else:
        substantive = [g for g in genders if g in {"feminine", "masculine"}]
        if not substantive: return None
        gender, gender_n = Counter(substantive).most_common(1)[0]; share = gender_n / len(substantive)
        gender_value = f"{gender}_only" if share == 1 else f"mostly_{gender}" if share > .5 else "mixed"
    return {"age_composition": age_value, "gender_presentation_composition": gender_value}


def assemble(args: argparse.Namespace, output_dir: Path, images: list[dict[str, Any]]) -> None:
    records = []
    if args.pipeline == "p1":
        source = latest_ok(output_dir / STAGE_FILES["monolithic"])
        for image in images:
            row = source.get(image["image_id"]); raw = (row or {}).get("model_annotation")
            annotation = None
            if raw is not None:
                ads = json.loads(json.dumps(raw.get("advertisements") or []))
                for ad in ads:
                    ad["groups"] = (ad.get("groups") or [])[:6]
                annotation = {
                    "schema_version": "qwen_first100_canonical_v1",
                    "playbook_version": PLAYBOOK_VERSION,
                    "page": {
                        "qualifying_ad_count": str(len(ads)),
                        "no_qualifying_ad_reason": None if ads else raw.get("no_qualifying_ad_reason") or "ads_present_no_visible_faces",
                    },
                    "advertisements": ads,
                    "urgent_comments": [],
                    "confidence": None,
                    "review_flags": raw.get("review_flags") or [],
                }
            records.append({"pipeline": args.pipeline, "run_name": args.run_name, "manifest_index": image["manifest_index"], "image_id": image["image_id"], "filename": image["filename"], "ok": annotation is not None, "annotation": annotation})
    else:
        tasks = read_json(output_dir / "entity_tasks.json")
        groups = latest_ok(output_dir / STAGE_FILES["groups"]); persons = latest_ok(output_dir / STAGE_FILES["persons"]); compositions = latest_ok(output_dir / STAGE_FILES["composition"]) if args.pipeline == "p4" else {}
        pages_by_id = {page["image"]["image_id"]: page for page in tasks["pages"]}
        group_tasks = {(t["image_id"], t["ad_id"], t["group_id"]): t for t in tasks["groups"]}; person_tasks = {(t["image_id"], t["ad_id"], t["person_id"]): t for t in tasks["persons"]}
        for image in images:
            page = pages_by_id.get(image["image_id"], {"ads": [], "no_reason": "ads_present_no_visible_faces"}); ads = []
            for ad_index, route_ad in enumerate(page["ads"], start=1):
                ad_id = f"ad_{ad_index}"; out_people = []; out_groups = []
                for person in route_ad.get("people") or []:
                    person_id = person.get("person_id"); task = person_tasks.get((image["image_id"], ad_id, person_id)); call = persons.get(task["task_key"]) if task else None
                    if not call or call["model_annotation"].get("eligible_face_visible") is not True: continue
                    attrs = call["model_annotation"]
                    out_people.append({"person_id": person_id, "annotation_role": task["role"], "face_bbox_1000": task["bbox_page"], "duplicate_of_person_id": None, "duplicate_person_ids": [], "depiction_type": attrs.get("depiction_type") if route_ad.get("depiction_type") == "multiple_types_present" else None, **{key: attrs.get(key) for key in PERSON_ATTR_FIELDS if key != "depiction_type"}, "gaze_target_person_id": None, "gaze_target_object_ref": None, "mouth_covering_other_text": None, "confidence": attrs.get("confidence"), "review_flags": attrs.get("review_flags") or []})
                for group in route_ad.get("groups") or []:
                    group_id = group.get("group_id"); task = group_tasks.get((image["image_id"], ad_id, group_id)); call = groups.get(task["task_key"]) if task else None
                    if not call: continue
                    attrs = dict(call["model_annotation"])
                    if args.pipeline == "p4" and task["task_key"] in compositions:
                        reduced = aggregate_composition(compositions[task["task_key"]]["model_annotation"])
                        if reduced: attrs.update(reduced)
                    out_groups.append({"group_id": group_id, "bbox_1000": task["bbox_page"], **{key: attrs.get(key) for key in GROUP_ATTR_FIELDS}, "confidence": attrs.get("confidence"), "review_flags": attrs.get("review_flags") or []})
                band = str(route_ad.get("face_depiction_count_band")); crowd = band in {"10_20", "20_plus"}
                ads.append({"advertisement_id": ad_id, "extent": route_ad.get("extent"), "bbox_1000": route_ad.get("bbox_1000"), "depiction_type": route_ad.get("depiction_type"), "face_depiction_count_band": band, "has_outstanding_individuals": ("yes" if out_people else "no") if crowd else None, "duplicate_faces_present": route_ad.get("duplicate_faces_present") if not crowd else None, "unique_face_count": len(out_people) if not crowd and out_people else None, "people": out_people, "identity_groups": [], "groups": out_groups, "confidence": route_ad.get("confidence"), "review_flags": route_ad.get("review_flags") or []})
            annotation = {"schema_version": "qwen_first100_canonical_v1", "playbook_version": PLAYBOOK_VERSION, "page": {"qualifying_ad_count": str(len(ads)), "no_qualifying_ad_reason": None if ads else page.get("no_reason") or "ads_present_no_visible_faces"}, "advertisements": ads, "urgent_comments": [], "confidence": None, "review_flags": []}
            records.append({"pipeline": args.pipeline, "run_name": args.run_name, "manifest_index": image["manifest_index"], "image_id": image["image_id"], "filename": image["filename"], "ok": True, "annotation": annotation})
    path = output_dir / "completed.jsonl"; path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    aggregate = {"schema_version": "qwen_first100_matrix_completed_aggregate_v1", "run_name": args.run_name, "pipeline": args.pipeline, "generated_at": iso_now(), "pages": len(records), "valid_pages": sum(record["ok"] for record in records), "shared_request_ledger": str(SHARED_LEDGER), "shared_requests_recorded": len(read_jsonl(SHARED_LEDGER)), "data_isolation": {"human_gold_read": False, "annotation_results_read": False, "inputs": [str(MANIFEST), str(IMAGE_DIR), "model outputs from same pipeline"]}}
    (output_dir / "completed.aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed": str(path), **aggregate}, indent=2))


def main() -> None:
    args = parse_args(); images = selected_images(args)
    output_dir = HERE / "output" / args.run_name / args.pipeline; output_dir.mkdir(parents=True, exist_ok=True)
    budget = RequestBudget(args.max_requests)
    client = None if args.dry_run or args.stage == "assemble" else build_client(args)
    if args.stage in {"all", "structure"}: stage_structure(client, budget, args, output_dir, images)
    if args.stage in {"all", "tiles"}: stage_tiles(client, budget, args, output_dir)
    if args.stage in {"all", "resolver"}: stage_resolver(client, budget, args, output_dir)
    if args.stage in {"all", "entities"}: stage_entities(client, budget, args, output_dir, images)
    if args.stage in {"all", "assemble"}:
        if args.pipeline != "p1" and not (output_dir / "entity_tasks.json").exists(): prepare_entity_tasks(args, output_dir, images)
        assemble(args, output_dir, images)


if __name__ == "__main__":
    main()

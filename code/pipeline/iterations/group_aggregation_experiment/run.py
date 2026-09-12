"""Run the bounded first-50 Qwen group aggregation experiment.

Stages:
  direct         one direct aggregate judgment per oracle people-area crop
  inventory      one full-crop individual inventory per people area
  tiles          four independent localization calls per people area
  prepare_faces  merge full/tile boxes and create up to N face/context composites
  faces          one attribute call per selected individual face composite
  reducers       one text-only LLM reduction of face attributes per people area

The runner reads only the sanitized input manifest.  It does not import or open
the gold-label file.  Every HTTP attempt is recorded in call_ledger.jsonl and a
hard shared budget prevents more than --max-calls attempts across resumptions.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from schema_and_prompt import (
    DIRECT_PROMPT_VERSION,
    EXPERIMENT_VERSION,
    FACE_PROMPT_VERSION,
    GROUP_TYPE_VALUES,
    INVENTORY_PROMPT_VERSION,
    REDUCER_PROMPT_VERSION,
    TALLY_PROMPT_VERSION,
    TILE_PROMPT_VERSION,
    direct_prompt,
    face_prompt,
    inventory_prompt,
    reducer_prompt,
    tally_prompt,
    tile_prompt,
    validate_bbox,
    validate_response,
)


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "group_aggregation_experiment"
INPUT_MANIFEST = HERE / "data" / "input_manifest_first50_groups.json"
OUTPUT_DIR = HERE / "output"
LEDGER_PATH = OUTPUT_DIR / "call_ledger.jsonl"
API_KEY_PATH = ROOT / "openrouter_key.txt"
BASE_URL = "https://openrouter.ai/api/v1"

STAGE_FILES = {
    "direct": OUTPUT_DIR / "direct_groups.jsonl",
    "inventory": OUTPUT_DIR / "individual_inventory.jsonl",
    "tiles": OUTPUT_DIR / "tile_localization.jsonl",
    "faces": OUTPUT_DIR / "face_attributes.jsonl",
    "reducers": OUTPUT_DIR / "llm_reducers.jsonl",
    "tally": OUTPUT_DIR / "distribution_tallies.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "direct", "inventory", "tiles", "prepare_faces", "faces", "reducers", "tally"],
        default="all",
    )
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument("--provider-tag", default="venice/fp8")
    parser.add_argument("--max-calls", type=int, default=500)
    parser.add_argument("--max-faces-per-group", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--retry-failures", action="store_true", help="Retry tasks whose latest record is not ok.")
    parser.add_argument("--dry-run", action="store_true")
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
    if lock is None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
        return
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


class CallBudget:
    def __init__(self, path: Path, limit: int):
        self.path = path
        self.limit = limit
        self.lock = threading.Lock()
        self.used = len(read_jsonl(path))

    def reserve(self) -> int:
        with self.lock:
            if self.used >= self.limit:
                raise RuntimeError(f"Qwen call budget exhausted: {self.used}/{self.limit}")
            self.used += 1
            return self.used

    def record(self, payload: dict[str, Any]) -> None:
        append_jsonl(self.path, payload, self.lock)


def load_api_key() -> str:
    if API_KEY_PATH.exists():
        value = API_KEY_PATH.read_text(encoding="utf-8").strip()
        if value:
            return value
    value = os.getenv("OPENROUTER_API_KEY", "").strip()
    if value:
        return value
    raise SystemExit(f"OpenRouter API key missing: {API_KEY_PATH}")


def build_client(args: argparse.Namespace) -> Any:
    try:
        import httpx
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit("The existing project environment needs openai and httpx.") from exc
    return OpenAI(
        api_key=load_api_key(),
        base_url=BASE_URL,
        timeout=args.timeout,
        max_retries=0,
        http_client=httpx.Client(timeout=args.timeout, trust_env=False),
        default_headers={"X-Title": "Qwen Group Aggregation First50"},
    )


def data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def response_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        chunks = []
        for item in content:
            value = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if value:
                chunks.append(str(value))
        return "\n".join(chunks)
    return str(content)


def plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return plain(value.model_dump())
    if hasattr(value, "dict"):
        return plain(value.dict())
    return str(value)


def extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.DOTALL).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("no JSON object in response")
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            quoted = not quoted
        elif not quoted and char == "{":
            depth += 1
        elif not quoted and char == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(cleaned[start : index + 1])
                if not isinstance(parsed, dict):
                    raise ValueError("response fragment is not an object")
                return parsed
    raise ValueError("unterminated JSON object")


def normalize(stage: str, annotation: dict[str, Any], expected_key: str | None = None) -> tuple[dict[str, Any], list[str]]:
    output = json.loads(json.dumps(annotation))
    actions: list[str] = []
    if stage == "tile" and expected_key is not None:
        supplied = output.get("group_key")
        if isinstance(supplied, str):
            repaired = supplied
            for prefix in ("people_area_", "people area "):
                if repaired.startswith(prefix):
                    repaired = repaired[len(prefix) :]
                    break
            if repaired == expected_key and repaired != supplied:
                output["group_key"] = repaired
                actions.append("group_key removed model-added people-area prefix")
    faces = output.get("faces") if stage in {"inventory", "tile"} else [output] if stage == "face" else []
    if isinstance(faces, list):
        for index, face in enumerate(faces):
            if not isinstance(face, dict):
                continue
            bbox = face.get("bbox_1000")
            if stage in {"inventory", "tile"} and isinstance(bbox, str):
                numbers = re.findall(r"-?\d+(?:\.\d+)?", bbox)
                if len(numbers) == 4:
                    bbox = [float(value) for value in numbers]
                    face["bbox_1000"] = bbox
                    actions.append(f"faces[{index}].bbox_1000 parsed from four-number string")
            if stage in {"inventory", "tile"} and isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(value, (int, float)) for value in bbox):
                repaired = [max(0, min(1000, int(round(value)))) for value in bbox]
                if repaired != bbox:
                    face["bbox_1000"] = repaired
                    actions.append(f"faces[{index}].bbox_1000 clamped to 0..1000")
            if face.get("face_expression_legibility") == "0_not_legible":
                for field in ("gaze_target", "smile_present", "smile_intensity"):
                    if face.get(field) is not None:
                        face[field] = None
                        actions.append(f"faces[{index}].{field}->null for zero legibility")
            if face.get("smile_present") != "yes" and face.get("smile_intensity") is not None:
                face["smile_intensity"] = None
                actions.append(f"faces[{index}].smile_intensity->null for non-positive smile")
    if stage in {"direct", "reducer"}:
        if output.get("expression_legibility_distribution") == "all_0_not_legible":
            for field in ("dominant_gaze", "smile_prevalence", "dominant_smile_intensity"):
                if output.get(field) is not None:
                    output[field] = None
                    actions.append(f"{field}->null for all zero legibility")
        if output.get("smile_prevalence") in {"none", "not_assessable", None} and output.get("dominant_smile_intensity") is not None:
            output["dominant_smile_intensity"] = None
            actions.append("dominant_smile_intensity->null for non-positive prevalence")
    return output, actions


def request_kwargs(args: argparse.Namespace, prompt: str, image_path: Path | None, max_tokens: int) -> dict[str, Any]:
    user_content: list[dict[str, Any]] = []
    if image_path is not None:
        user_content.append({"type": "image_url", "image_url": {"url": data_url(image_path)}})
    user_content.append({"type": "text", "text": prompt})
    return {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "You are a conservative visual annotation researcher. Return only the requested JSON object.",
            },
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "response_format": {"type": "json_object"},
        "extra_body": {
            "provider": {"order": [args.provider_tag], "allow_fallbacks": False, "require_parameters": True},
            "reasoning": {"effort": "none", "exclude": True},
        },
    }


def call_one(
    client: Any,
    budget: CallBudget,
    args: argparse.Namespace,
    stage: str,
    task_key: str,
    expected_key: str,
    prompt: str,
    prompt_version: str,
    image_path: Path | None,
    max_tokens: int,
    extra_validation: Callable[[dict[str, Any]], list[str]] | None = None,
) -> dict[str, Any]:
    sequence = budget.reserve()
    started = time.time()
    record: dict[str, Any] = {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": stage,
        "task_key": task_key,
        "group_key": expected_key if stage != "face" else task_key.rsplit("::", 1)[0],
        "call_sequence": sequence,
        "started_at": iso_now(),
        "prompt_version": prompt_version,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "image_path": str(image_path) if image_path else None,
        "request": {
            "model": args.model,
            "provider_tag": args.provider_tag,
            "max_tokens": max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    }
    raw_dir = OUTPUT_DIR / "raw" / stage
    prompt_dir = OUTPUT_DIR / "prompts" / stage
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / f"{safe(task_key)}.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    record["prompt_path"] = str(prompt_path)
    ledger: dict[str, Any] = {"call_sequence": sequence, "stage": stage, "task_key": task_key, "started_at": record["started_at"]}
    try:
        response = client.chat.completions.create(**request_kwargs(args, prompt, image_path, max_tokens))
        raw = response_text(response.choices[0].message.content)
        raw_path = raw_dir / f"{safe(task_key)}.raw.txt"
        raw_path.write_text(raw, encoding="utf-8")
        metadata = plain(response)
        metadata_path = raw_dir / f"{safe(task_key)}.response.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        parsed = extract_json(raw)
        normalized, actions = normalize(stage, parsed, expected_key)
        errors = validate_response(stage, normalized, expected_key)
        if extra_validation:
            errors.extend(extra_validation(normalized))
        record.update(
            {
                "raw_response_path": str(raw_path),
                "response_metadata_path": str(metadata_path),
                "response_id": metadata.get("id") if isinstance(metadata, dict) else None,
                "usage": metadata.get("usage") if isinstance(metadata, dict) else None,
                "model_annotation_raw": parsed,
                "model_annotation": normalized,
                "normalization_actions": actions,
                "validation_errors": errors,
                "ok": not errors,
            }
        )
        ledger.update({"ok": record["ok"], "response_id": record.get("response_id"), "usage": record.get("usage")})
    except Exception as exc:
        record.update({"ok": False, "error": f"{type(exc).__name__}: {exc}", "validation_errors": []})
        ledger.update({"ok": False, "error": record["error"]})
    record["elapsed_seconds"] = round(time.time() - started, 3)
    ledger["elapsed_seconds"] = record["elapsed_seconds"]
    ledger["finished_at"] = iso_now()
    budget.record(ledger)
    return record


def completed_keys(path: Path, retry_failures: bool) -> set[str]:
    rows = read_jsonl(path)
    latest = {row["task_key"]: row for row in rows}
    if retry_failures:
        return {key for key, row in latest.items() if row.get("ok") is True}
    return set(latest)


def validate_tally(annotation: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if annotation.get("group_type") not in GROUP_TYPE_VALUES:
        errors.append("invalid group_type")
    expected = {
        "age_percent": {"young", "middle", "older", "not_assessable"},
        "gender_percent": {"feminine", "masculine", "ambiguous_or_androgynous", "not_assessable"},
        "legibility_percent": {"0_not_legible", "1_low_legibility", "2_moderate_legibility", "3_high_legibility"},
        "gaze_percent": {"toward_viewer_camera", "toward_each_other", "toward_object", "off_frame_or_scene_direction", "not_assessable"},
        "smile_percent": {"yes", "no", "not_assessable"},
        "smiling_face_intensity_percent": {"slight", "clear", "broad_or_laughter_like", "not_assessable"},
    }
    for field, keys in expected.items():
        values = annotation.get(field)
        if not isinstance(values, dict) or set(values) != keys:
            errors.append(f"{field} keys mismatch")
            continue
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 100 for value in values.values()):
            errors.append(f"{field} has invalid percentage")
            continue
        total = sum(float(value) for value in values.values())
        allowed = {0.0, 100.0} if field == "smiling_face_intensity_percent" else {100.0}
        if not any(abs(total - target) <= 1.0 for target in allowed):
            errors.append(f"{field} sums to {total}, expected 100")
    return errors


def run_tasks(
    client: Any,
    budget: CallBudget,
    args: argparse.Namespace,
    stage: str,
    tasks: list[dict[str, Any]],
    worker: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    output_path = STAGE_FILES[stage]
    done = completed_keys(output_path, args.retry_failures)
    pending = [task for task in tasks if task["task_key"] not in done]
    print(f"{stage}: tasks={len(tasks)} completed={len(done)} pending={len(pending)} budget={budget.used}/{budget.limit}")
    if args.dry_run or not pending:
        return
    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(worker, task): task for task in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    "experiment_version": EXPERIMENT_VERSION,
                    "stage": stage,
                    "task_key": task["task_key"],
                    "group_key": task.get("group_key"),
                    "ok": False,
                    "error": f"worker failure: {type(exc).__name__}: {exc}",
                }
            append_jsonl(output_path, record, write_lock)
            print(f"[{stage} {index}/{len(pending)}] {task['task_key']} ok={record.get('ok')} calls={budget.used}/{budget.limit}")


def latest_ok(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        latest[str(row["task_key"])] = row
    return {key: row for key, row in latest.items() if row.get("ok") is True}


def box_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def same_face(a: list[float], b: list[float]) -> bool:
    if box_iou(a, b) >= 0.20:
        return True
    ac = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    bc = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    distance = math.dist(ac, bc)
    scale = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1], 1.0)
    return distance <= 0.42 * scale


def merge_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: list[list[dict[str, Any]]] = []
    for candidate in sorted(candidates, key=lambda item: float(item.get("confidence") or 0), reverse=True):
        target = next((cluster for cluster in clusters if any(same_face(candidate["bbox_1000"], item["bbox_1000"]) for item in cluster)), None)
        if target is None:
            clusters.append([candidate])
        else:
            target.append(candidate)
    merged = []
    for cluster in clusters:
        weights = [max(0.1, float(item.get("confidence") or 0.5)) for item in cluster]
        total = sum(weights)
        box = [int(round(sum(item["bbox_1000"][index] * weight for item, weight in zip(cluster, weights)) / total)) for index in range(4)]
        if not validate_bbox(box):
            box = list(cluster[0]["bbox_1000"])
        merged.append(
            {
                "bbox_1000": box,
                "confidence": max(weights),
                "sources": sorted({str(item["source"]) for item in cluster}),
                "source_count": len(cluster),
            }
        )
    return sorted(merged, key=lambda item: ((item["bbox_1000"][1] + item["bbox_1000"][3]) / 2, (item["bbox_1000"][0] + item["bbox_1000"][2]) / 2))


def tile_box_to_group(box: list[int], tile_box: list[int], group_width: int, group_height: int) -> list[int]:
    left, top, right, bottom = tile_box
    tile_width, tile_height = right - left, bottom - top
    x1 = left + box[0] / 1000 * tile_width
    y1 = top + box[1] / 1000 * tile_height
    x2 = left + box[2] / 1000 * tile_width
    y2 = top + box[3] / 1000 * tile_height
    return [
        max(0, min(999, int(round(x1 / group_width * 1000)))),
        max(0, min(999, int(round(y1 / group_height * 1000)))),
        max(1, min(1000, int(round(x2 / group_width * 1000)))),
        max(1, min(1000, int(round(y2 / group_height * 1000)))),
    ]


def select_spatial(proposals: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if len(proposals) <= maximum:
        return proposals
    indices = sorted({int(round(index * (len(proposals) - 1) / (maximum - 1))) for index in range(maximum)})
    return [proposals[index] for index in indices]


def letterbox(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, tuple[float, int, int]]:
    target_w, target_h = size
    scale = min(target_w / image.width, target_h / image.height)
    resized = image.resize((max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    offset_x = (target_w - resized.width) // 2
    offset_y = (target_h - resized.height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas, (scale, offset_x, offset_y)


def create_face_composite(group_path: Path, bbox: list[int], output_path: Path, label: str) -> None:
    with Image.open(group_path) as source:
        group = source.convert("RGB")
    x1, y1, x2, y2 = [value / 1000 for value in bbox]
    px = [x1 * group.width, y1 * group.height, x2 * group.width, y2 * group.height]
    width, height = px[2] - px[0], px[3] - px[1]
    padding = 0.45
    crop_box = (
        max(0, int(math.floor(px[0] - width * padding))),
        max(0, int(math.floor(px[1] - height * padding))),
        min(group.width, int(math.ceil(px[2] + width * padding))),
        min(group.height, int(math.ceil(px[3] + height * padding))),
    )
    target = group.crop(crop_box)
    target_panel, _ = letterbox(target, (700, 700))
    context_panel, transform = letterbox(group, (700, 700))
    scale, ox, oy = transform
    draw = ImageDraw.Draw(context_panel)
    draw.rectangle(
        [ox + px[0] * scale, oy + px[1] * scale, ox + px[2] * scale, oy + px[3] * scale],
        outline=(220, 20, 20),
        width=6,
    )
    canvas = Image.new("RGB", (1400, 740), "white")
    canvas.paste(target_panel, (0, 40))
    canvas.paste(context_panel, (700, 40))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 10), f"TARGET FACE: {label}", fill="black", font=font)
    draw.text((710, 10), "PEOPLE-AREA CONTEXT (target in red)", fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=95, optimize=True)


def prepare_face_tasks(groups: list[dict[str, Any]], args: argparse.Namespace) -> None:
    inventories = latest_ok(STAGE_FILES["inventory"])
    tiles = latest_ok(STAGE_FILES["tiles"])
    tasks = []
    summary = []
    for group in groups:
        group_key = str(group["group_key"])
        candidates: list[dict[str, Any]] = []
        inventory = inventories.get(group_key)
        if inventory:
            for face in inventory["model_annotation"].get("faces") or []:
                if validate_bbox(face.get("bbox_1000")):
                    candidates.append(
                        {
                            "bbox_1000": face["bbox_1000"],
                            "confidence": face.get("confidence", 0.5),
                            "source": "full_inventory",
                        }
                    )
        tile_lookup = {tile["tile_id"]: tile for tile in group["tiles"]}
        for tile_id, tile in tile_lookup.items():
            record = tiles.get(f"{group_key}::{tile_id}")
            if not record:
                continue
            for face in record["model_annotation"].get("faces") or []:
                if not validate_bbox(face.get("bbox_1000")):
                    continue
                mapped = tile_box_to_group(
                    face["bbox_1000"],
                    tile["tile_bbox_pixels_in_group_crop"],
                    int(group["group_crop_width"]),
                    int(group["group_crop_height"]),
                )
                if validate_bbox(mapped):
                    candidates.append(
                        {
                            "bbox_1000": mapped,
                            "confidence": face.get("confidence", 0.5),
                            "source": f"tile_{tile_id}",
                        }
                    )
        merged = merge_candidates(candidates)
        selected = select_spatial(merged, args.max_faces_per_group)
        for index, proposal in enumerate(selected, start=1):
            face_task_id = f"{group_key}::face_{index:02d}"
            composite_path = HERE / "crops" / "face_composites" / f"{safe(face_task_id)}.jpg"
            create_face_composite(Path(group["group_crop_path"]), proposal["bbox_1000"], composite_path, face_task_id)
            tasks.append(
                {
                    "task_key": face_task_id,
                    "face_task_id": face_task_id,
                    "group_key": group_key,
                    "bbox_1000": proposal["bbox_1000"],
                    "proposal_confidence": proposal["confidence"],
                    "proposal_sources": proposal["sources"],
                    "proposal_source_count": proposal["source_count"],
                    "composite_path": str(composite_path),
                }
            )
        summary.append(
            {
                "group_key": group_key,
                "raw_candidates": len(candidates),
                "merged_candidates": len(merged),
                "selected_face_tasks": len(selected),
                "truncated_by_face_cap": len(merged) > len(selected),
            }
        )
    payload = {
        "schema_version": "qwen_group_face_tasks_v1",
        "generated_at": iso_now(),
        "max_faces_per_group": args.max_faces_per_group,
        "tasks": tasks,
        "group_summary": summary,
    }
    path = OUTPUT_DIR / "face_tasks_manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"prepare_faces: groups={len(groups)} face_tasks={len(tasks)} -> {path}")


def main() -> None:
    args = parse_args()
    manifest = read_json(INPUT_MANIFEST)
    groups = list(manifest["groups"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    budget = CallBudget(LEDGER_PATH, args.max_calls)
    client = None if args.dry_run or args.stage == "prepare_faces" else build_client(args)

    if args.stage in {"all", "direct"}:
        tasks = [{"task_key": group["group_key"], **group} for group in groups]
        run_tasks(
            client,
            budget,
            args,
            "direct",
            tasks,
            lambda task: call_one(
                client,
                budget,
                args,
                "direct",
                task["task_key"],
                task["group_key"],
                direct_prompt(task["group_key"]),
                DIRECT_PROMPT_VERSION,
                Path(task["group_crop_path"]),
                2600,
            ),
        )

    if args.stage in {"all", "inventory"}:
        tasks = [{"task_key": group["group_key"], **group} for group in groups]
        run_tasks(
            client,
            budget,
            args,
            "inventory",
            tasks,
            lambda task: call_one(
                client,
                budget,
                args,
                "inventory",
                task["task_key"],
                task["group_key"],
                inventory_prompt(task["group_key"]),
                INVENTORY_PROMPT_VERSION,
                Path(task["group_crop_path"]),
                12000,
            ),
        )

    if args.stage in {"all", "tiles"}:
        tasks = []
        for group in groups:
            for tile in group["tiles"]:
                tasks.append(
                    {
                        "task_key": f"{group['group_key']}::{tile['tile_id']}",
                        "group_key": group["group_key"],
                        "tile_id": tile["tile_id"],
                        "tile_path": tile["tile_path"],
                    }
                )

        def tile_worker(task: dict[str, Any]) -> dict[str, Any]:
            def extra(annotation: dict[str, Any]) -> list[str]:
                return [] if annotation.get("tile_id") == task["tile_id"] else ["tile_id mismatch"]

            return call_one(
                client,
                budget,
                args,
                "tile",
                task["task_key"],
                task["group_key"],
                tile_prompt(task["group_key"], task["tile_id"]),
                TILE_PROMPT_VERSION,
                Path(task["tile_path"]),
                3500,
                extra,
            )

        run_tasks(client, budget, args, "tiles", tasks, tile_worker)

    if args.stage in {"all", "prepare_faces"}:
        prepare_face_tasks(groups, args)

    if args.stage in {"all", "faces"}:
        face_manifest_path = OUTPUT_DIR / "face_tasks_manifest.json"
        if not face_manifest_path.exists():
            raise SystemExit("Run --stage prepare_faces after inventory/tiles first.")
        face_tasks = list(read_json(face_manifest_path)["tasks"])
        run_tasks(
            client,
            budget,
            args,
            "faces",
            face_tasks,
            lambda task: call_one(
                client,
                budget,
                args,
                "face",
                task["task_key"],
                task["face_task_id"],
                face_prompt(task["face_task_id"], task["group_key"]),
                FACE_PROMPT_VERSION,
                Path(task["composite_path"]),
                2200,
            ),
        )

    if args.stage in {"all", "reducers"}:
        inventories = latest_ok(STAGE_FILES["inventory"])
        faces = latest_ok(STAGE_FILES["faces"])
        face_manifest = read_json(OUTPUT_DIR / "face_tasks_manifest.json")
        tasks_by_group: dict[str, list[dict[str, Any]]] = {}
        for task in face_manifest["tasks"]:
            record = faces.get(task["face_task_id"])
            if not record or record["model_annotation"].get("eligible_face_visible") is not True:
                continue
            annotation = dict(record["model_annotation"])
            for key in ("schema_version", "face_task_id", "eligible_face_visible", "visual_blockers", "confidence"):
                annotation.pop(key, None)
            tasks_by_group.setdefault(task["group_key"], []).append(annotation)
        reducer_tasks = []
        for group in groups:
            group_key = group["group_key"]
            inventory = inventories.get(group_key)
            group_type = "other_group"
            if inventory:
                group_type = str(inventory["model_annotation"].get("group_type") or group_type)
            reducer_tasks.append(
                {
                    "task_key": group_key,
                    "group_key": group_key,
                    "group_type": group_type,
                    "individuals": tasks_by_group.get(group_key, []),
                }
            )
        run_tasks(
            client,
            budget,
            args,
            "reducers",
            reducer_tasks,
            lambda task: call_one(
                client,
                budget,
                args,
                "reducer",
                task["task_key"],
                task["group_key"],
                reducer_prompt(task["group_key"], task["group_type"], task["individuals"]),
                REDUCER_PROMPT_VERSION,
                None,
                2600,
            ),
        )

    if args.stage in {"all", "tally"}:
        tasks = [{"task_key": group["group_key"], **group} for group in groups]
        run_tasks(
            client,
            budget,
            args,
            "tally",
            tasks,
            lambda task: call_one(
                client,
                budget,
                args,
                "tally",
                task["task_key"],
                task["group_key"],
                tally_prompt(task["group_key"]),
                TALLY_PROMPT_VERSION,
                Path(task["group_crop_path"]),
                3200,
                validate_tally,
            ),
        )

    print(json.dumps({"budget_used": budget.used, "budget_limit": budget.limit, "output_dir": str(OUTPUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()

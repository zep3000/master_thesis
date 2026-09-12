#!/usr/bin/env python3
"""Resumable gold-free runner for the lean 140+140 experiment.

Stages are deliberately separable so pilot decisions can be made before scale:

  python run.py structure --cohort difficult140 --variant s1 --pilot-only
  python run.py entities  --cohort difficult140 --variant s1 --style direct --gaze no --pilot-only
  python run.py assemble  --cohort difficult140 --variant s1 --style direct --gaze no

All OpenRouter attempts, including failures, are counted in one package-local
ledger. Inference only reads neutral manifests produced by prepare_manifests.py.
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import math
import mimetypes
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

import prompts


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "lean_final_280"
OUTPUT = HERE / "output"
LEDGER = OUTPUT / "request_ledger.jsonl"
KEY = ROOT / "openrouter_key.txt"
MODEL = "qwen/qwen3.5-9b"
PROVIDER = "venice/fp8"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


WRITE_LOCK = threading.Lock()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("response is not a JSON object")
    return value


def data_url_from_bytes(payload: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return data_url_from_bytes(path.read_bytes(), mime)


def jpeg_data_url(image: Image.Image, quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality, optimize=True)
    return data_url_from_bytes(buffer.getvalue())


class Budget:
    def __init__(self, limit: int):
        old = jsonl(LEDGER)
        self.limit = limit
        self.used = len(old)
        self.sequence = max([int(row.get("request_sequence", 0)) for row in old] or [0]) + 1
        self.lock = threading.Lock()

    def reserve(self) -> int:
        with self.lock:
            if self.used >= self.limit:
                raise RuntimeError(f"package request budget exhausted ({self.used}/{self.limit})")
            sequence = self.sequence
            self.sequence += 1
            self.used += 1
            return sequence

    def record(self, row: dict[str, Any]) -> None:
        append_jsonl(LEDGER, row)


def load_manifest(cohort: str) -> dict[str, Any]:
    return json.loads((HERE / "data" / f"manifest_{cohort}.json").read_text(encoding="utf-8"))


def selected_images(cohort: str, pilot_only: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_manifest(cohort)
    images = manifest["images"]
    if pilot_only:
        pilots = json.loads((HERE / "data" / "pilot_ids.json").read_text(encoding="utf-8"))["ids"][cohort]
        by_id = {item["image_id"]: item for item in images}
        images = [by_id[image_id] for image_id in pilots]
    return manifest, images


def clamp_box(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = [max(0, min(1000, round(float(item)))) for item in value]
    except (TypeError, ValueError):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def enum(value: Any, allowed: list[Any], fallback: Any) -> Any:
    # Some Qwen JSON responses wrap a scalar enum in a one-element array even
    # when explicitly asked for a string.  This is a serialization repair, not
    # a label decision: unwrap one value; for multi-valued depiction output use
    # the schema's explicit multiple-types label, otherwise take the first valid
    # displayed option in model order.
    if isinstance(value, list):
        if not value:
            value = None
        elif len(value) == 1:
            value = value[0]
        elif "multiple_types_present" in allowed:
            value = "multiple_types_present"
        else:
            value = next((item for item in value if item in allowed), fallback)
    return value if value in allowed else fallback


def confidence(value: Any, fallback: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return fallback


def normalize_structure(raw: dict[str, Any], image_id: str) -> dict[str, Any]:
    ads = []
    raw_ads = raw.get("advertisements") if isinstance(raw.get("advertisements"), list) else []
    for ad_index, source in enumerate(raw_ads, 1):
        if not isinstance(source, dict):
            continue
        ad_box = clamp_box(source.get("bbox_1000"))
        if not ad_box:
            continue
        ad_id = f"ad_{ad_index}"
        people = []
        source_people = source.get("people") if isinstance(source.get("people"), list) else []
        for person_index, item in enumerate(source_people, 1):
            if not isinstance(item, dict):
                continue
            box = clamp_box(item.get("face_bbox_1000"))
            if not box:
                continue
            people.append({
                "person_id": f"{ad_id}_person_{person_index}",
                "annotation_role": enum(item.get("annotation_role"), ["individual", "outstanding_individual"], "individual"),
                "face_bbox_1000": box,
                "prominence_reason": enum(item.get("prominence_reason"), ["dramatically_larger_or_clearer", "separate_panel_or_scene", "spatially_separate_from_people_area", None], None),
                "confidence": confidence(item.get("confidence")),
            })
        groups = []
        source_groups = source.get("groups") if isinstance(source.get("groups"), list) else []
        for group_index, item in enumerate(source_groups, 1):
            if not isinstance(item, dict):
                continue
            box = clamp_box(item.get("bbox_1000"))
            if box:
                groups.append({
                    "group_id": f"{ad_id}_group_{group_index}",
                    "bbox_1000": box,
                    "confidence": confidence(item.get("confidence")),
                })
        band = enum(source.get("face_depiction_count_band"), prompts.COUNT_BANDS, None)
        if band is None:
            band = str(min(9, max(1, len(people)))) if people else ("10_20" if groups else "1")
        is_group = band in {"10_20", "20_plus"}
        if is_group:
            people = people[:3]
            for person in people:
                person["annotation_role"] = "outstanding_individual"
            if not groups:
                groups = [{"group_id": f"{ad_id}_group_1", "bbox_1000": ad_box.copy(), "confidence": confidence(source.get("confidence"), 0.35)}]
            unique_count = None
            duplicate = None
        else:
            groups = []
            people = people[:9]
            for person in people:
                person["annotation_role"] = "individual"
                person["prominence_reason"] = None
            # Keep the model's band because missing boxes must remain an observable
            # structural error; do not silently rewrite it from the returned list.
            unique_count = source.get("unique_face_count")
            try:
                unique_count = int(unique_count) if unique_count is not None else len(people)
            except (TypeError, ValueError):
                unique_count = len(people)
            duplicate = enum(source.get("duplicate_faces_present"), ["yes", "no", None], None)
        category = enum(source.get("ad_category"), prompts.AD_CATEGORIES, "other")
        brand = source.get("brand_or_advertiser")
        if not isinstance(brand, str) or not brand.strip():
            brand = None
        ads.append({
            "advertisement_id": ad_id,
            "extent": enum(source.get("extent"), ["full_page", "partial_page"], "partial_page"),
            "bbox_1000": ad_box,
            "ad_category": category,
            "ad_category_confidence": confidence(source.get("ad_category_confidence")),
            "brand_or_advertiser": brand.strip() if brand else None,
            "brand_confidence": confidence(source.get("brand_confidence"), 0.0) if brand else 0.0,
            "depiction_type": enum(source.get("depiction_type"), prompts.DEPICTION_TYPES, "photo_of_person"),
            "face_depiction_count_band": band,
            "duplicate_faces_present": duplicate,
            "unique_face_count": unique_count,
            "people": people,
            "groups": groups,
            "has_outstanding_individuals": "yes" if is_group and people else ("no" if is_group else None),
            "confidence": confidence(source.get("confidence")),
            "review_flags": source.get("review_flags") if isinstance(source.get("review_flags"), list) else [],
        })
    page = {
        # Canonical annotation-app schema stores this categorical count as a
        # string even though the model-facing structure schema requests an
        # integer.
        "qualifying_ad_count": str(len(ads)),
        "no_qualifying_ad_reason": None if ads else enum(raw.get("no_qualifying_ad_reason"), ["no_ads_on_page", "ads_present_no_visible_faces"], "no_ads_on_page"),
    }
    return {
        "schema_version": "lean_page_structure_v1",
        "image_id": image_id,
        "page": page,
        # Root aliases retain fidelity to the model-facing schema while the
        # canonical page object makes the annotation-app/evaluator contract
        # explicit.
        **page,
        "advertisements": ads,
        "review_flags": raw.get("review_flags") if isinstance(raw.get("review_flags"), list) else [],
    }


def normalize_person(raw: dict[str, Any], task_id: str, style: str, gaze: bool) -> dict[str, Any]:
    if style == "ordinal":
        low = bool(raw.get("at_least_low"))
        moderate = low and bool(raw.get("at_least_moderate"))
        high = moderate and bool(raw.get("at_least_high"))
        legibility = "3_high_legibility" if high else "2_moderate_legibility" if moderate else "1_low_legibility" if low else "0_not_legible"
    else:
        legibility = enum(raw.get("face_expression_legibility"), prompts.LEGIBILITY, "0_not_legible")
    mouth = enum(raw.get("mouth_covered"), prompts.MOUTH, "not_assessable")
    smile = enum(raw.get("smile_present"), prompts.SMILE, "not_assessable")
    result = {
        "person_task_id": task_id,
        "depiction_type": enum(raw.get("depiction_type"), prompts.DEPICTION_TYPES + [None], None),
        "perceived_age": enum(raw.get("perceived_age"), prompts.AGES, "not_assessable"),
        "perceived_gender_presentation": enum(raw.get("perceived_gender_presentation"), prompts.GENDERS, "not_assessable"),
        "face_expression_legibility": legibility,
        "face_orientation": enum(raw.get("face_orientation"), prompts.ORIENTATION, "not_assessable"),
        "gaze_target": enum(raw.get("gaze_target"), prompts.GAZE, "not_assessable") if gaze else None,
        "gaze_target_person_unboxed": bool(raw.get("gaze_target_person_unboxed")) if gaze and raw.get("gaze_target") == "another_person" else None,
        "mouth_covered": mouth,
        "mouth_covering": enum(raw.get("mouth_covering"), prompts.MOUTH_CAUSES + [None], None) if mouth in {"yes", "partly"} else None,
        "smile_present": smile,
        "smile_intensity": enum(raw.get("smile_intensity"), prompts.INTENSITY + [None], None) if smile == "yes" else None,
        "confidence": confidence(raw.get("confidence")),
        "review_flags": raw.get("review_flags") if isinstance(raw.get("review_flags"), list) else [],
    }
    if legibility == "0_not_legible":
        result["smile_present"] = "not_assessable"
        result["smile_intensity"] = None
        result["gaze_target"] = "not_assessable" if gaze else None
        result["gaze_target_person_unboxed"] = None
    return result


def normalize_group(raw: dict[str, Any], task_id: str, gaze: bool) -> dict[str, Any]:
    legibility = enum(raw.get("expression_legibility_distribution"), prompts.GROUP_LEGIBILITY, "mixed_legibility")
    smile = enum(raw.get("smile_prevalence"), prompts.GROUP_SMILE, "not_assessable")
    result = {
        "group_task_id": task_id,
        "group_type": enum(raw.get("group_type"), prompts.GROUP_TYPES, "other_group"),
        "age_composition": enum(raw.get("age_composition"), prompts.GROUP_AGES, "not_assessable"),
        "gender_presentation_composition": enum(raw.get("gender_presentation_composition"), prompts.GROUP_GENDERS, "not_assessable"),
        "expression_legibility_distribution": legibility,
        "dominant_gaze": enum(raw.get("dominant_gaze"), prompts.GROUP_GAZE, "not_assessable") if gaze else None,
        "smile_prevalence": smile,
        "dominant_smile_intensity": enum(raw.get("dominant_smile_intensity"), prompts.GROUP_INTENSITY + [None], None) if smile not in {"none", "not_assessable"} else None,
        "confidence": confidence(raw.get("confidence")),
        "review_flags": raw.get("review_flags") if isinstance(raw.get("review_flags"), list) else [],
    }
    if legibility == "all_0_not_legible":
        result["smile_prevalence"] = "not_assessable"
        result["dominant_smile_intensity"] = None
        result["dominant_gaze"] = "not_assessable" if gaze else None
    return result


def structure_content(page_path: Path, prompt: str, variant: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.append({"type": "text", "text": "IMAGE 1 — complete page, coordinates [0,0,1000,1000]"})
    content.append({"type": "image_url", "image_url": {"url": image_data_url(page_path)}})
    if variant == "s2":
        image = Image.open(page_path).convert("RGB")
        width, height = image.size
        regions = [(0, 0, 600, 600), (400, 0, 1000, 600), (0, 400, 600, 1000), (400, 400, 1000, 1000)]
        for index, (x1, y1, x2, y2) in enumerate(regions, 2):
            crop = image.crop((round(x1 * width / 1000), round(y1 * height / 1000), round(x2 * width / 1000), round(y2 * height / 1000)))
            content.append({"type": "text", "text": f"IMAGE {index} — detail crop full-page range [{x1},{y1},{x2},{y2}]"})
            content.append({"type": "image_url", "image_url": {"url": jpeg_data_url(crop)}})
    return content


def to_pixels(box: list[int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    return tuple(round(value * (width if index % 2 == 0 else height) / 1000) for index, value in enumerate(box))  # type: ignore[return-value]


def expand_box(box: tuple[int, int, int, int], image_size: tuple[int, int], scale: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * scale
    width, height = image_size
    return (max(0, round(cx - side / 2)), max(0, round(cy - side / 2)), min(width, round(cx + side / 2)), min(height, round(cy + side / 2)))


def letterbox(image: Image.Image, size: int, label: str) -> Image.Image:
    panel = Image.new("RGB", (size, size + 42), "white")
    work = image.copy()
    work.thumbnail((size, size), Image.Resampling.LANCZOS)
    panel.paste(work, ((size - work.width) // 2, 42 + (size - work.height) // 2))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 12), label, fill="black", font=ImageFont.load_default())
    return panel


def person_composite(page_path: Path, ad_box: list[int], face_box: list[int], out_path: Path) -> Path:
    if out_path.exists():
        return out_path
    page = Image.open(page_path).convert("RGB")
    face_px = to_pixels(face_box, page.size)
    ad_px = to_pixels(ad_box, page.size)
    tight = page.crop(expand_box(face_px, page.size, 1.65))
    medium = page.crop(expand_box(face_px, page.size, 4.5))
    context = page.crop(ad_px)
    draw = ImageDraw.Draw(context)
    fx1, fy1, fx2, fy2 = face_px
    ax1, ay1, _, _ = ad_px
    draw.rectangle((fx1 - ax1, fy1 - ay1, fx2 - ax1, fy2 - ay1), outline=(255, 0, 0), width=max(3, round(min(page.size) / 350)))
    panels = [letterbox(tight, 512, "TARGET FACE — enlarged"), letterbox(medium, 512, "LOCAL CONTEXT"), letterbox(context, 512, "ADVERTISEMENT — target in red")]
    canvas = Image.new("RGB", (1536, 554), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * 512, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "JPEG", quality=92, optimize=True)
    return out_path


def group_crop(page_path: Path, box: list[int], out_path: Path) -> Path:
    if out_path.exists():
        return out_path
    page = Image.open(page_path).convert("RGB")
    crop = page.crop(to_pixels(box, page.size))
    crop.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path, "JPEG", quality=92, optimize=True)
    return out_path


def make_client():
    import httpx
    from openai import OpenAI
    return OpenAI(
        api_key=KEY.read_text(encoding="utf-8").strip(),
        base_url="https://openrouter.ai/api/v1",
        timeout=300,
        max_retries=0,
        http_client=httpx.Client(timeout=300, trust_env=False),
        default_headers={"X-Title": "Qwen Lean Final 280"},
    )


def one_request(
    client: Any,
    budget: Budget,
    run_meta: dict[str, Any],
    task_key: str,
    content: list[dict[str, Any]],
    max_tokens: int,
    normalize: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    last_error = ""
    for attempt in range(1, 4):
        sequence = budget.reserve()
        started = time.time()
        usage = None
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=0,
                response_format={"type": "json_object"},
                extra_body={"provider": {"order": [PROVIDER], "allow_fallbacks": False}, "reasoning": {"effort": "none", "exclude": True}},
            )
            metadata = plain(response)
            usage = metadata.get("usage")
            raw = extract_json(str(response.choices[0].message.content))
            annotation = normalize(raw)
            ok = True
            last_error = ""
        except Exception as exc:  # request/parse failures are ledgered and resumable
            raw = annotation = None
            ok = False
            last_error = f"{type(exc).__name__}: {exc}"
        budget.record({
            "request_sequence": sequence,
            **run_meta,
            "task_key": task_key,
            "attempt": attempt,
            "ok": ok,
            "usage": usage,
            "error": None if ok else last_error,
            "elapsed_seconds": round(time.time() - started, 3),
            "finished_at": now(),
        })
        if ok:
            return {"ok": True, "task_key": task_key, "model_annotation_raw": raw, "model_annotation": annotation, "usage": usage}
        if attempt < 3:
            time.sleep(32 if "429" in last_error else 3)
    return {"ok": False, "task_key": task_key, "error": last_error}


def execute_jobs(jobs: list[dict[str, Any]], result_path: Path, budget: Budget, workers: int) -> None:
    done = {row["task_key"] for row in jsonl(result_path) if row.get("ok")}
    pending = [job for job in jobs if job["task_key"] not in done]
    if not pending:
        print(f"nothing pending: {result_path}")
        return
    local = threading.local()

    def run(job: dict[str, Any]) -> dict[str, Any]:
        if not hasattr(local, "client"):
            local.client = make_client()
        return one_request(local.client, budget, job["meta"], job["task_key"], job["content"], job["max_tokens"], job["normalize"])

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(run, job): job for job in pending}
        for future in as_completed(future_map):
            row = future.result()
            append_jsonl(result_path, row)
            completed += 1
            print(f"[{completed}/{len(pending)}] {row['task_key']} {'ok' if row.get('ok') else 'FAILED'}", flush=True)


def structure_stage(args: argparse.Namespace, budget: Budget) -> None:
    manifest, images = selected_images(args.cohort, args.pilot_only)
    if args.limit:
        images = images[: args.limit]
    result_path = OUTPUT / args.cohort / f"structure_{args.variant}.jsonl"
    jobs = []
    for item in images:
        image_id = item["image_id"]
        page_path = Path(manifest["image_dir"]) / item["filename"]
        tiled = args.variant.startswith("s2")
        prompt = prompts.structure_prompt(image_id, item.get("metadata", {}).get("year"), tiled, include_business=args.variant != "s2n")
        jobs.append({
            "task_key": image_id,
            "content": structure_content(page_path, prompt, "s2" if tiled else "s1"),
            "max_tokens": 8000,
            "normalize": lambda raw, image_id=image_id: normalize_structure(raw, image_id),
            "meta": {"cohort": args.cohort, "stage": "structure", "variant": args.variant, "prompt_version": prompts.PROMPT_VERSION, "model": MODEL},
        })
    execute_jobs(jobs, result_path, budget, args.workers)


def structure_map(cohort: str, variant: str) -> dict[str, dict[str, Any]]:
    path = OUTPUT / cohort / f"structure_{variant}.jsonl"
    return {row["task_key"]: row["model_annotation"] for row in jsonl(path) if row.get("ok")}


def entity_stage(args: argparse.Namespace, budget: Budget) -> None:
    manifest, images = selected_images(args.cohort, args.pilot_only)
    if args.limit:
        images = images[: args.limit]
    structures = structure_map(args.cohort, args.variant)
    gaze = args.gaze == "yes"
    suffix = f"{args.style}_{'gaze' if gaze else 'nogaze'}"
    result_path = OUTPUT / args.cohort / f"entities_{args.variant}_{suffix}.jsonl"
    cache = OUTPUT / args.cohort / "crops" / args.variant
    jobs = []
    for item in images:
        image_id = item["image_id"]
        annotation = structures.get(image_id)
        if annotation is None:
            continue
        page_path = Path(manifest["image_dir"]) / item["filename"]
        for ad in annotation.get("advertisements") or []:
            for person in ad.get("people") or []:
                task_id = f"{image_id}::{person['person_id']}"
                composite = person_composite(page_path, ad["bbox_1000"], person["face_bbox_1000"], cache / f"{task_id.replace('::', '__')}_person.jpg")
                content = [{"type": "text", "text": prompts.person_prompt(task_id, args.style, gaze)}, {"type": "image_url", "image_url": {"url": image_data_url(composite)}}]
                jobs.append({
                    "task_key": task_id,
                    "content": content,
                    "max_tokens": 1500,
                    "normalize": lambda raw, task_id=task_id: normalize_person(raw, task_id, args.style, gaze),
                    "meta": {"cohort": args.cohort, "stage": "person", "variant": args.variant, "style": args.style, "gaze": gaze, "prompt_version": prompts.PROMPT_VERSION, "model": MODEL},
                })
            for group in ad.get("groups") or []:
                task_id = f"{image_id}::{group['group_id']}"
                crop = group_crop(page_path, group["bbox_1000"], cache / f"{task_id.replace('::', '__')}_group.jpg")
                content = [{"type": "text", "text": prompts.group_prompt(task_id, gaze)}, {"type": "image_url", "image_url": {"url": image_data_url(crop)}}]
                jobs.append({
                    "task_key": task_id,
                    "content": content,
                    "max_tokens": 1300,
                    "normalize": lambda raw, task_id=task_id: normalize_group(raw, task_id, gaze),
                    "meta": {"cohort": args.cohort, "stage": "group", "variant": args.variant, "style": args.style, "gaze": gaze, "prompt_version": prompts.PROMPT_VERSION, "model": MODEL},
                })
    execute_jobs(jobs, result_path, budget, args.workers)


def assemble_stage(args: argparse.Namespace) -> None:
    gaze = args.gaze == "yes"
    suffix = f"{args.style}_{'hybrid' if args.gaze == 'hybrid' else 'gaze' if gaze else 'nogaze'}"
    structures = structure_map(args.cohort, args.variant)
    if args.gaze == "hybrid":
        no_path = OUTPUT / args.cohort / f"entities_{args.variant}_{args.style}_nogaze.jsonl"
        yes_path = OUTPUT / args.cohort / f"entities_{args.variant}_{args.style}_gaze.jsonl"
        entities = {row["task_key"]: row["model_annotation"] for row in jsonl(no_path) if row.get("ok") and "_group_" not in row["task_key"]}
        entities.update({row["task_key"]: row["model_annotation"] for row in jsonl(yes_path) if row.get("ok") and "_group_" in row["task_key"]})
    else:
        entity_path = OUTPUT / args.cohort / f"entities_{args.variant}_{suffix}.jsonl"
        entities = {row["task_key"]: row["model_annotation"] for row in jsonl(entity_path) if row.get("ok")}
    manifest = load_manifest(args.cohort)
    output = []
    for item in manifest["images"]:
        image_id = item["image_id"]
        structure = structures.get(image_id)
        if structure is None:
            continue
        annotation = copy.deepcopy(structure)
        complete = True
        for ad in annotation.get("advertisements") or []:
            for person in ad.get("people") or []:
                attrs = entities.get(f"{image_id}::{person['person_id']}")
                if attrs is None:
                    complete = False
                    continue
                person.update({key: value for key, value in attrs.items() if key != "person_task_id"})
                person.setdefault("duplicate_of_person_id", None)
                person.setdefault("duplicate_person_ids", [])
                person.setdefault("gaze_target_person_id", None)
                person.setdefault("gaze_target_object_ref", None)
                person.setdefault("mouth_covering_other_text", None)
            for group in ad.get("groups") or []:
                attrs = entities.get(f"{image_id}::{group['group_id']}")
                if attrs is None:
                    complete = False
                    continue
                group.update({key: value for key, value in attrs.items() if key != "group_task_id"})
        output.append({
            "image_id": image_id,
            "filename": item["filename"],
            "cohort": args.cohort,
            "route": f"lean_{args.variant}_{suffix}",
            "ok": complete,
            "annotation": annotation,
        })
    out_path = OUTPUT / args.cohort / "assembled" / f"lean_{args.variant}_{suffix}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({"path": str(out_path), "pages": len(output), "complete": sum(bool(row["ok"]) for row in output)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ["structure", "entities", "assemble"]:
        child = sub.add_parser(name)
        child.add_argument("--cohort", choices=["difficult140", "stratified140"], required=True)
        child.add_argument("--variant", choices=["s1", "s2", "s2n"], required=True)
        child.add_argument("--max-requests", type=int, default=3000)
        child.add_argument("--workers", type=int, default=3)
        child.add_argument("--pilot-only", action="store_true")
        child.add_argument("--limit", type=int)
        if name in {"entities", "assemble"}:
            child.add_argument("--style", choices=["direct", "ordinal"], required=True)
            child.add_argument("--gaze", choices=["yes", "no", "hybrid"] if name == "assemble" else ["yes", "no"], required=True)
    args = parser.parse_args()
    if args.stage == "assemble":
        assemble_stage(args)
        return 0
    budget = Budget(args.max_requests)
    if args.stage == "structure":
        structure_stage(args, budget)
    else:
        entity_stage(args, budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

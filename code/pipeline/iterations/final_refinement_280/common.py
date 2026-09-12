"""Shared gold-free utilities for final_refinement_280 inference."""

from __future__ import annotations

import base64
import copy
import io
import json
import math
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
HERE = ROOT / "qwen_iteration" / "final_refinement_280"
BASE = ROOT / "qwen_iteration" / "lean_final_280"
OUTPUT = HERE / "output"
LEDGER = OUTPUT / "request_ledger.jsonl"
KEY = ROOT / "openrouter_key.txt"
MODEL = "qwen/qwen3.5-9b"
PROVIDER = "venice/fp8"
WRITE_LOCK = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return plain(value.model_dump())
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain(item) for item in value]
    return value


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            value = json.loads(text[start:end + 1])
            return value if isinstance(value, dict) else {}
        raise


def scalar(value: Any) -> Any:
    return value[0] if isinstance(value, list) and len(value) == 1 else value


def enum(value: Any, allowed: list[Any], fallback: Any = None) -> Any:
    value = scalar(value)
    return value if value in allowed else fallback


def confidence(value: Any, fallback: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(scalar(value))))
    except (TypeError, ValueError):
        return fallback


def clamp_box(value: Any) -> list[int] | None:
    if isinstance(value, str):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
        value = numbers if len(numbers) == 4 else None
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = [max(0, min(1000, round(float(item)))) for item in value]
    except (TypeError, ValueError):
        return None
    return box if box[2] > box[0] and box[3] > box[1] else None


def manifest(cohort: str) -> dict[str, Any]:
    return json.loads((BASE / "data" / f"manifest_{cohort}.json").read_text(encoding="utf-8"))


def pilot_ids(cohort: str) -> list[str]:
    payload = json.loads((HERE / "data" / "pilot_ids.json").read_text(encoding="utf-8"))
    return payload["ids"][cohort]


def selected_images(cohort: str, scope: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = manifest(cohort)
    images = data["images"]
    if scope == "pilot":
        wanted = set(pilot_ids(cohort))
        images = [item for item in images if item["image_id"] in wanted]
    return data, images


def image_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def jpeg_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=92, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def to_pixels(box: list[int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    return tuple(round(item * (width if index % 2 == 0 else height) / 1000) for index, item in enumerate(box))  # type: ignore[return-value]


def expand_box(box: tuple[int, int, int, int], size: tuple[int, int], scale: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * scale
    width, height = size
    return max(0, round(cx - side / 2)), max(0, round(cy - side / 2)), min(width, round(cx + side / 2)), min(height, round(cy + side / 2))


def letterbox(image: Image.Image, size: int, label: str, outline: str | None = None) -> Image.Image:
    panel = Image.new("RGB", (size, size + 42), "white")
    work = image.copy()
    work.thumbnail((size, size), Image.Resampling.LANCZOS)
    panel.paste(work, ((size - work.width) // 2, 42 + (size - work.height) // 2))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 12), label, fill="black", font=ImageFont.load_default())
    if outline:
        draw.rectangle((1, 1, panel.width - 2, panel.height - 2), outline=outline, width=3)
    return panel


def person_composite(page_path: Path, ad_box: list[int], face_box: list[int], out_path: Path) -> Path:
    if out_path.exists():
        return out_path
    page = Image.open(page_path).convert("RGB")
    face_px, ad_px = to_pixels(face_box, page.size), to_pixels(ad_box, page.size)
    tight = page.crop(expand_box(face_px, page.size, 1.65))
    medium = page.crop(expand_box(face_px, page.size, 4.5))
    context = page.crop(ad_px)
    draw = ImageDraw.Draw(context)
    fx1, fy1, fx2, fy2 = face_px
    ax1, ay1, _, _ = ad_px
    draw.rectangle((fx1 - ax1, fy1 - ay1, fx2 - ax1, fy2 - ay1), outline=(255, 0, 0), width=max(3, round(min(page.size) / 350)))
    panels = [letterbox(tight, 512, "TARGET FACE - enlarged"), letterbox(medium, 512, "LOCAL CONTEXT"), letterbox(context, 512, "ADVERTISEMENT - target red")]
    canvas = Image.new("RGB", (1536, 554), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * 512, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "JPEG", quality=92, optimize=True)
    return out_path


def group_composite(page_path: Path, ad_box: list[int], group_box: list[int], out_path: Path, include_context: bool) -> Path:
    if out_path.exists():
        return out_path
    page = Image.open(page_path).convert("RGB")
    group_px, ad_px = to_pixels(group_box, page.size), to_pixels(ad_box, page.size)
    group = page.crop(group_px)
    width, height = group.size
    regions = [(0, 0, .6, .6), (.4, 0, 1, .6), (0, .4, .6, 1), (.4, .4, 1, 1)]
    views = [letterbox(group, 512, "COMPLETE PEOPLE AREA", "#444444")]
    for index, (x1, y1, x2, y2) in enumerate(regions, 1):
        crop = group.crop((round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)))
        views.append(letterbox(crop, 512, f"DETAIL TILE {index}"))
    if include_context:
        context = page.crop(ad_px)
        draw = ImageDraw.Draw(context)
        gx1, gy1, gx2, gy2 = group_px
        ax1, ay1, _, _ = ad_px
        draw.rectangle((gx1 - ax1, gy1 - ay1, gx2 - ax1, gy2 - ay1), outline=(255, 0, 255), width=max(3, round(min(page.size) / 350)))
        views.append(letterbox(context, 512, "ADVERTISEMENT - group magenta", "#ff00ff"))
    columns = 3
    rows_count = math.ceil(len(views) / columns)
    canvas = Image.new("RGB", (columns * 512, rows_count * 554), "white")
    for index, panel in enumerate(views):
        canvas.paste(panel, ((index % columns) * 512, (index // columns) * 554))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "JPEG", quality=92, optimize=True)
    return out_path


def normalize_structure(raw: dict[str, Any], image_id: str) -> dict[str, Any]:
    advertisements = []
    sources = raw.get("advertisements") if isinstance(raw.get("advertisements"), list) else []
    for ad_index, source in enumerate(sources, 1):
        if not isinstance(source, dict):
            continue
        ad_box = clamp_box(source.get("bbox_1000"))
        if not ad_box:
            continue
        ad_id = f"ad_{ad_index}"
        people = []
        for item in source.get("people") if isinstance(source.get("people"), list) else []:
            if not isinstance(item, dict):
                continue
            box = clamp_box(item.get("face_bbox_1000"))
            if not box:
                continue
            people.append({
                "person_id": f"{ad_id}_person_{len(people) + 1}",
                "annotation_role": enum(item.get("annotation_role"), ["individual", "outstanding_individual"], "individual"),
                "face_bbox_1000": box,
                "prominence_reason": enum(item.get("prominence_reason"), ["dramatically_larger_or_clearer", "separate_panel_or_scene", "spatially_separate_from_people_area", None], None),
                "confidence": confidence(item.get("confidence")),
            })
        groups = []
        for item in source.get("groups") if isinstance(source.get("groups"), list) else []:
            if not isinstance(item, dict):
                continue
            box = clamp_box(item.get("bbox_1000"))
            if box:
                groups.append({"group_id": f"{ad_id}_group_{len(groups) + 1}", "bbox_1000": box, "confidence": confidence(item.get("confidence"))})
        band = enum(source.get("face_depiction_count_band"), prompts.COUNT_BANDS)
        if band is None:
            band = str(min(9, max(1, len(people)))) if people else ("10_20" if groups else "1")
        is_group = band in {"10_20", "20_plus"}
        if is_group:
            people = people[:3]
            for person in people:
                person["annotation_role"] = "outstanding_individual"
            if not groups:
                groups = [{"group_id": f"{ad_id}_group_1", "bbox_1000": ad_box.copy(), "confidence": confidence(source.get("confidence"), .35)}]
            unique_count, duplicate = None, None
        else:
            groups, people = [], people[:9]
            for person in people:
                person["annotation_role"], person["prominence_reason"] = "individual", None
            try:
                unique_count = int(source.get("unique_face_count")) if source.get("unique_face_count") is not None else len(people)
            except (TypeError, ValueError):
                unique_count = len(people)
            duplicate = enum(source.get("duplicate_faces_present"), ["yes", "no", None], None)
        brand = scalar(source.get("brand_or_advertiser"))
        brand = brand.strip() if isinstance(brand, str) and brand.strip() else None
        advertisements.append({
            "advertisement_id": ad_id,
            "extent": enum(source.get("extent"), ["full_page", "partial_page"], "partial_page"),
            "bbox_1000": ad_box,
            "ad_category": enum(source.get("ad_category"), prompts.WARC_CATEGORIES + [None], None),
            "brand_or_advertiser": brand,
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
        "qualifying_ad_count": str(len(advertisements)),
        "no_qualifying_ad_reason": None if advertisements else enum(raw.get("no_qualifying_ad_reason"), ["no_ads_on_page", "ads_present_no_visible_faces"], "no_ads_on_page"),
    }
    return {
        "schema_version": "final_refinement_structure_v1", "image_id": image_id,
        "page": page, **page, "advertisements": advertisements,
        "review_flags": raw.get("review_flags") if isinstance(raw.get("review_flags"), list) else [],
    }


def normalize_person(raw: dict[str, Any], task_id: str, gate_zero: bool) -> dict[str, Any]:
    legibility = enum(raw.get("face_expression_legibility"), prompts.LEGIBILITY, "0_not_legible")
    gaze = enum(raw.get("gaze_target"), prompts.GAZE, "not_assessable")
    smile = enum(raw.get("smile_present"), prompts.SMILE, "not_assessable")
    mouth = enum(raw.get("mouth_covered"), prompts.MOUTH, "not_assessable")
    result = {
        "person_task_id": task_id,
        "depiction_type": enum(raw.get("depiction_type"), prompts.DEPICTION_TYPES + [None], None),
        "perceived_age": enum(raw.get("perceived_age"), prompts.AGES, "not_assessable"),
        "perceived_gender_presentation": enum(raw.get("perceived_gender_presentation"), prompts.GENDERS, "not_assessable"),
        "face_expression_legibility": legibility,
        "face_orientation": enum(raw.get("face_orientation"), prompts.ORIENTATION, "not_assessable"),
        "gaze_target": gaze,
        "gaze_target_person_unboxed": bool(scalar(raw.get("gaze_target_person_unboxed"))) if gaze == "another_person" else None,
        "mouth_covered": mouth,
        "mouth_covering": enum(raw.get("mouth_covering"), prompts.MOUTH_CAUSES + [None], None) if mouth in {"yes", "partly"} else None,
        "smile_present": smile,
        "smile_intensity": enum(raw.get("smile_intensity"), prompts.INTENSITY + [None], None) if smile == "yes" else None,
        "confidence": confidence(raw.get("confidence")),
        "review_flags": raw.get("review_flags") if isinstance(raw.get("review_flags"), list) else [],
    }
    if gate_zero and legibility == "0_not_legible":
        result["gaze_target"] = "not_assessable"
        result["gaze_target_person_unboxed"] = None
        result["smile_present"] = "not_assessable"
        result["smile_intensity"] = None
    return result


def normalize_group(raw: dict[str, Any], task_id: str, gate_zero: bool) -> dict[str, Any]:
    legibility = enum(raw.get("expression_legibility_distribution"), prompts.GROUP_LEGIBILITY, "mixed_legibility")
    gaze = enum(raw.get("dominant_gaze"), prompts.GROUP_GAZE, "not_assessable")
    smile = enum(raw.get("smile_prevalence"), prompts.GROUP_SMILE, "not_assessable")
    result = {
        "group_task_id": task_id,
        "group_type": enum(raw.get("group_type"), prompts.GROUP_TYPES, "other_group"),
        "age_composition": enum(raw.get("age_composition"), prompts.GROUP_AGES, "not_assessable"),
        "gender_presentation_composition": enum(raw.get("gender_presentation_composition"), prompts.GROUP_GENDERS, "not_assessable"),
        "expression_legibility_distribution": legibility,
        "dominant_gaze": gaze,
        "smile_prevalence": smile,
        "dominant_smile_intensity": enum(raw.get("dominant_smile_intensity"), prompts.GROUP_INTENSITY + [None], None) if smile not in {"none", "not_assessable"} else None,
        "confidence": confidence(raw.get("confidence")),
        "review_flags": raw.get("review_flags") if isinstance(raw.get("review_flags"), list) else [],
    }
    if gate_zero and legibility == "all_0_not_legible":
        result["dominant_gaze"] = "not_assessable"
        result["smile_prevalence"] = "not_assessable"
        result["dominant_smile_intensity"] = None
    return result


def normalize_threshold(raw: dict[str, Any], task_id: str) -> dict[str, Any]:
    def boolean(value: Any) -> bool:
        value = scalar(value)
        return value is True or str(value).strip().lower() in {"true", "yes", "1"}
    low, moderate, high = (boolean(raw.get(field)) for field in ("at_least_low", "at_least_moderate", "at_least_high"))
    if high:
        moderate = low = True
    if moderate:
        low = True
    legibility = prompts.LEGIBILITY[3 if high else 2 if moderate else 1 if low else 0]
    return {
        "person_task_id": task_id, "at_least_low": low,
        "at_least_moderate": moderate, "at_least_high": high,
        "face_expression_legibility": legibility,
        "review_flags": raw.get("review_flags") if isinstance(raw.get("review_flags"), list) else [],
    }


def normalize_moderate_audit(raw: dict[str, Any], task_id: str) -> dict[str, Any]:
    legibility = enum(raw.get("face_expression_legibility"), prompts.LEGIBILITY, "2_moderate_legibility")
    decision = "keep_moderate" if legibility == "2_moderate_legibility" else "override_up" if legibility == "3_high_legibility" else "override_down"
    return {
        "person_task_id": task_id, "face_expression_legibility": legibility, "decision": decision,
        "visible_regions": raw.get("visible_regions") if isinstance(raw.get("visible_regions"), list) else [],
        "review_flags": raw.get("review_flags") if isinstance(raw.get("review_flags"), list) else [],
    }


def clean_structure(annotation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    result = copy.deepcopy(annotation)
    stats = {"people_removed": 0, "groups_removed": 0, "ads_removed": 0}
    ads = []
    for ad in result.get("advertisements") or []:
        parent = ad["bbox_1000"]
        def inside(box: list[int]) -> bool:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            return parent[0] <= cx <= parent[2] and parent[1] <= cy <= parent[3]
        old_people, old_groups = ad.get("people") or [], ad.get("groups") or []
        ad["people"] = [person for person in old_people if inside(person["face_bbox_1000"])]
        ad["groups"] = [group for group in old_groups if inside(group["bbox_1000"])]
        stats["people_removed"] += len(old_people) - len(ad["people"])
        stats["groups_removed"] += len(old_groups) - len(ad["groups"])
        if not ad["people"] and not ad["groups"]:
            stats["ads_removed"] += 1
            continue
        if ad["face_depiction_count_band"] in {"10_20", "20_plus"}:
            ad["has_outstanding_individuals"] = "yes" if ad["people"] else "no"
        ads.append(ad)
    result["advertisements"] = ads
    result["page"]["qualifying_ad_count"] = result["qualifying_ad_count"] = str(len(ads))
    if ads:
        result["page"]["no_qualifying_ad_reason"] = result["no_qualifying_ad_reason"] = None
    return result, stats


class Budget:
    def __init__(self, maximum: int):
        self.maximum = maximum
        self.lock = threading.Lock()
        self.used = len(rows(LEDGER))

    def reserve(self) -> int:
        with self.lock:
            if self.used >= self.maximum:
                raise RuntimeError(f"OpenRouter attempt ceiling reached: {self.used}/{self.maximum}")
            self.used += 1
            return self.used


def client() -> Any:
    import httpx
    from openai import OpenAI
    return OpenAI(
        api_key=KEY.read_text(encoding="utf-8").strip(), base_url="https://openrouter.ai/api/v1",
        timeout=300, max_retries=0, http_client=httpx.Client(timeout=300, trust_env=False),
        default_headers={"X-Title": "Qwen Final Refinement 280"},
    )


def one_request(api: Any, budget: Budget, job: dict[str, Any]) -> dict[str, Any]:
    last_error = ""
    for attempt in range(1, 4):
        sequence = budget.reserve()
        started, usage = time.time(), None
        try:
            response = api.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": job["content"]}],
                max_tokens=job["max_tokens"], temperature=0,
                response_format={"type": "json_object"},
                extra_body={"provider": {"order": [PROVIDER], "allow_fallbacks": False}, "reasoning": {"effort": "none", "exclude": True}},
            )
            metadata = plain(response)
            usage = metadata.get("usage")
            raw = extract_json(str(response.choices[0].message.content))
            annotation = job["normalize"](raw)
            ok, last_error = True, ""
        except Exception as exc:
            raw = annotation = None
            ok, last_error = False, f"{type(exc).__name__}: {exc}"
        append_row(LEDGER, {
            "request_sequence": sequence, **job["meta"], "task_key": job["task_key"],
            "attempt": attempt, "ok": ok, "usage": usage,
            "error": None if ok else last_error,
            "elapsed_seconds": round(time.time() - started, 3), "finished_at": now(),
        })
        if ok:
            return {"ok": True, "task_key": job["task_key"], "model_annotation_raw": raw, "model_annotation": annotation, "usage": usage}
        if attempt < 3:
            time.sleep(32 if "429" in last_error else 3)
    return {"ok": False, "task_key": job["task_key"], "error": last_error}


def execute(jobs: list[dict[str, Any]], result_path: Path, maximum: int, workers: int) -> None:
    completed = {row["task_key"] for row in rows(result_path) if row.get("ok")}
    pending = [job for job in jobs if job["task_key"] not in completed]
    print(json.dumps({"result_path": str(result_path), "jobs": len(jobs), "pending": len(pending), "ledger_attempts": len(rows(LEDGER)), "ceiling": maximum}))
    if not pending:
        return
    local = threading.local()
    budget = Budget(maximum)
    def run(job: dict[str, Any]) -> dict[str, Any]:
        if not hasattr(local, "api"):
            local.api = client()
        return one_request(local.api, budget, job)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, job) for job in pending]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            append_row(result_path, row)
            print(f"[{index}/{len(pending)}] {row['task_key']} {'ok' if row.get('ok') else 'FAILED'}", flush=True)

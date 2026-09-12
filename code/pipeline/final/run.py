#!/usr/bin/env python3
"""Resumable gold-free runner for validation and directory-scale production.

Stages are deliberately separable so pilot decisions can be made before scale:

  python run.py structure --cohort difficult140 --variant s1 --pilot-only
  python run.py entities  --cohort difficult140 --variant s1 --style direct --gaze no --pilot-only
  python run.py assemble  --cohort difficult140 --variant s1 --style direct --gaze no

All OpenRouter attempts, including failures, are counted in one package-local
ledger. Inference only reads neutral manifests. ``run-all`` freezes a directory
inventory, streams bounded batches through every stage, retries, assembles, and
audits completeness without human intervention.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import math
import mimetypes
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image, ImageDraw, ImageFont

import prompts


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PIPELINE_PROJECT_ROOT", Path.cwd())).resolve()
BASE_OUTPUT = HERE / "output"
OUTPUT = BASE_OUTPUT
LEDGER = OUTPUT / "request_ledger.jsonl"
KEY = Path(os.environ.get("OPENROUTER_KEY_FILE", ROOT / "openrouter_key.txt"))
MODEL = "qwen/qwen3.5-9b"
PROVIDER = "venice/fp8"
DEFAULT_WORKERS = 20
DEFAULT_FULL_PAGE_DIR = Path(os.environ.get("PIPELINE_IMAGE_DIR", ROOT / "data" / "images"))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ACTIVE_MANIFEST: Path | None = None
DEFAULT_TARGET_TOKENS_PER_MINUTE = 850_000
PACING_BACKOFF_HALF_LIFE_SECONDS = 60.0
PACING_BACKOFF_QUIET_GRACE_SECONDS = 60.0
RATE_LIMIT_EPISODE_QUIET_SECONDS = 120.0
RECOVERY_MAX_TOKENS = 16000
INITIAL_TOKENS_PER_ATTEMPT = {
    # Empirical means from the completed 2,000-page Venice run.  They are only
    # launch-pacing estimates and update from returned usage during a run.
    "structure": 21_350.0,
    "entities": 1_800.0,
    "duplicates": 2_000.0,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A process kill can leave only the last append incomplete.
                # Successful earlier rows remain resumable.
                continue


def jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


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


class BudgetExhausted(RuntimeError):
    """The explicit physical-attempt allowance for this invocation was used."""


class Budget:
    def __init__(
        self,
        limit: int,
        ledger_path: Path | None = None,
        target_tokens_per_minute: int | None = None,
    ):
        self.ledger_path = ledger_path or LEDGER
        used = 0
        maximum_sequence = 0
        # Only truncation history is needed to choose the compact recovery
        # prompt on a resumed process. Keeping the complete 100k-request ledger
        # in RAM would defeat the production runner's bounded-memory contract.
        self.history = []
        for row in iter_jsonl(self.ledger_path):
            used += 1
            maximum_sequence = max(maximum_sequence, int(row.get("request_sequence", 0)))
            if row.get("error_category") == "response_truncated":
                self.history.append(row)
        self.limit = limit
        self.used = used
        self.starting_used = used
        self.sequence = maximum_sequence + 1
        self.lock = threading.Lock()
        self.blocked_until = 0.0
        self.next_launch_at = 0.0
        self.launch_spacing = 0.0
        self.target_tokens_per_minute = int(target_tokens_per_minute) if target_tokens_per_minute else None
        self.token_estimates = dict(INITIAL_TOKENS_PER_ATTEMPT)
        self.token_launches: deque[tuple[float, float]] = deque()
        self.estimated_tokens_in_window = 0.0
        self.pacing_multiplier = 1.0
        self.pacing_decay_at = time.monotonic()
        self.last_rate_limit_at: float | None = None
        self.successes_since_rate_limit = 0
        self.execution_records: list[dict[str, Any]] = []

    def _decay_pacing_locked(self, current: float) -> None:
        """Relax token-target backoff after the endpoint has stayed quiet."""
        decay_start = self.pacing_decay_at
        if self.last_rate_limit_at is not None:
            decay_start = max(decay_start, self.last_rate_limit_at + PACING_BACKOFF_QUIET_GRACE_SECONDS)
        elapsed = max(0.0, current - decay_start)
        if self.target_tokens_per_minute is not None and self.pacing_multiplier > 1.0 and elapsed > 0:
            excess = self.pacing_multiplier - 1.0
            excess *= math.pow(0.5, elapsed / PACING_BACKOFF_HALF_LIFE_SECONDS)
            self.pacing_multiplier = 1.0 if excess < .01 else 1.0 + excess
        if current >= decay_start:
            self.pacing_decay_at = current

    def reserve(self) -> int:
        with self.lock:
            if self.used >= self.limit:
                raise BudgetExhausted(f"package request budget exhausted ({self.used}/{self.limit})")
            sequence = self.sequence
            self.sequence += 1
            self.used += 1
            return sequence

    def record(self, row: dict[str, Any]) -> None:
        append_jsonl(self.ledger_path, row)
        with self.lock:
            self.execution_records.append(row)
            stage = str(row.get("stage") or "")
            total_tokens = int((row.get("usage") or {}).get("total_tokens") or 0)
            if stage in self.token_estimates and total_tokens > 0:
                old = self.token_estimates[stage]
                self.token_estimates[stage] = old * .95 + total_tokens * .05
            if row.get("ok"):
                self.successes_since_rate_limit += 1
                if self.target_tokens_per_minute is not None:
                    self._decay_pacing_locked(time.monotonic())
                elif self.successes_since_rate_limit >= 50:
                    if self.target_tokens_per_minute is None:
                        self.launch_spacing = max(0.0, self.launch_spacing * .75)
                        if self.launch_spacing < .02:
                            self.launch_spacing = 0.0
                    self.successes_since_rate_limit = 0

    def begin_execution(self) -> int:
        with self.lock:
            if self.execution_records:
                raise RuntimeError("internal error: endpoint records were not consumed")
            return self.sequence

    def finish_execution(self) -> list[dict[str, Any]]:
        with self.lock:
            records = self.execution_records
            self.execution_records = []
            return records

    def wait_for_endpoint(self, stage: str | None = None) -> None:
        while True:
            with self.lock:
                current = time.monotonic()
                self._decay_pacing_locked(current)
                token_ready_at = current
                spacing = self.launch_spacing
                if self.target_tokens_per_minute is not None:
                    while self.token_launches and self.token_launches[0][0] <= current - 60.0:
                        _, expired = self.token_launches.popleft()
                        self.estimated_tokens_in_window = max(0.0, self.estimated_tokens_in_window - expired)
                    estimate = self.token_estimates.get(str(stage), 2_000.0)
                    effective_target = self.target_tokens_per_minute / self.pacing_multiplier
                    spacing = estimate * 60.0 / effective_target
                    if self.token_launches and self.estimated_tokens_in_window + estimate > effective_target:
                        token_ready_at = self.token_launches[0][0] + 60.0
                target = max(self.blocked_until, self.next_launch_at, token_ready_at)
                remaining = target - current
                if remaining <= 0:
                    self.next_launch_at = max(current, self.next_launch_at) + spacing
                    if self.target_tokens_per_minute is not None:
                        estimate = self.token_estimates.get(str(stage), 2_000.0)
                        self.token_launches.append((current, estimate))
                        self.estimated_tokens_in_window += estimate
                    return
            time.sleep(min(remaining, 1.0))

    def cooldown(self, seconds: float) -> None:
        with self.lock:
            current = time.monotonic()
            self._decay_pacing_locked(current)
            new_rate_limit_wave = current >= self.blocked_until
            new_rate_limit_episode = (
                self.last_rate_limit_at is None
                or current - self.last_rate_limit_at >= RATE_LIMIT_EPISODE_QUIET_SECONDS
            )
            self.last_rate_limit_at = current
            self.blocked_until = max(self.blocked_until, current + seconds)
            if new_rate_limit_wave:
                self.successes_since_rate_limit = 0
                if self.target_tokens_per_minute is not None and new_rate_limit_episode:
                    self.pacing_multiplier = min(3.0, self.pacing_multiplier * 1.25)
                    self.pacing_decay_at = current
                else:
                    # Validation commands without token pacing retain the old
                    # stagger, but it now decays after clean successes.
                    self.launch_spacing = max(self.launch_spacing, .25)
            self.next_launch_at = max(self.next_launch_at, self.blocked_until)

    def pacing_snapshot(self) -> dict[str, Any]:
        with self.lock:
            self._decay_pacing_locked(time.monotonic())
            target = self.target_tokens_per_minute
            return {
                "target_tokens_per_minute": target,
                "effective_target_after_429_backoff": (target / self.pacing_multiplier) if target else None,
                "rate_backoff_multiplier": self.pacing_multiplier,
                "rate_backoff_half_life_seconds": PACING_BACKOFF_HALF_LIFE_SECONDS if target else None,
                "rate_backoff_quiet_grace_seconds": PACING_BACKOFF_QUIET_GRACE_SECONDS if target else None,
                "rate_limit_episode_quiet_seconds": RATE_LIMIT_EPISODE_QUIET_SECONDS if target else None,
                "estimated_tokens_per_attempt": {key: round(value, 1) for key, value in self.token_estimates.items()},
                "estimated_tokens_in_rolling_window": round(self.estimated_tokens_in_window, 1),
            }

    def marker(self) -> int:
        with self.lock:
            return self.sequence

    def had_failure(self, run_meta: dict[str, Any], task_key: str, category: str) -> bool:
        return any(
            row.get("task_key") == task_key
            and row.get("error_category") == category
            and all(row.get(key) == value for key, value in run_meta.items())
            for row in self.history
        )


def error_details(exc: Exception) -> tuple[str, int | None, float | None, bool]:
    text = f"{type(exc).__name__}: {exc}".lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    retry_after = None
    try:
        headers = getattr(exc, "headers", None) or (getattr(response, "headers", None) if response is not None else None)
        header = headers.get("retry-after") if headers is not None else None
        retry_after = float(header) if header else None
    except (TypeError, ValueError, AttributeError):
        pass
    if status == 429 or "rate limit" in text or "429" in text:
        return "rate_limit", status or 429, retry_after, True
    if status and int(status) >= 500:
        return "server_5xx", int(status), retry_after, True
    if "timeout" in text:
        return "timeout", status, retry_after, True
    if any(token in text for token in ["connection", "network", "temporarily unavailable"]):
        return "connection", status, retry_after, True
    if isinstance(exc, (json.JSONDecodeError, ValueError)):
        return "response_parse_or_schema", status, retry_after, True
    if "provider" in text or "upstream" in text:
        return "provider", status, retry_after, True
    return "nonretryable", status, retry_after, False


def retry_delay(category: str, attempt: int, retry_after: float | None) -> float:
    base = 5.0 if category == "rate_limit" else 2.0 if category in {"server_5xx", "provider", "timeout", "connection"} else .5 if category == "response_truncated" else 1.0
    return min(60.0, max(retry_after or 0.0, base * (2 ** (attempt - 1))) + random.uniform(0.0, 1.5))


def terminal_failure_reason(row: dict[str, Any] | None) -> str | None:
    """Classify bounded failures that cannot improve through identical retries."""
    if not row or row.get("ok"):
        return None
    status = row.get("http_status")
    evidence = " ".join([
        str(row.get("error") or ""),
        json.dumps(row.get("error_metadata"), ensure_ascii=False, default=str),
    ]).lower()
    if int(status or 0) == 400 and "supplied image did not pass validation checks" in evidence:
        return "provider_input_image_validation"
    if (
        row.get("error_category") == "response_truncated"
        and int(row.get("requested_max_tokens") or 0) >= RECOVERY_MAX_TOKENS
    ):
        return "output_limit_exhausted"
    return None


def load_manifest(cohort: str) -> dict[str, Any]:
    path = ACTIVE_MANIFEST or (HERE / "data" / f"manifest_{cohort}.json")
    return json.loads(path.read_text(encoding="utf-8"))


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not cleaned:
        raise ValueError("run name must contain at least one letter or digit")
    return cleaned


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def freeze_directory_manifest(input_dir: Path, manifest_path: Path) -> dict[str, Any]:
    """Create once, then reuse a stable, label-free image inventory.

    Resumption intentionally does not rescan the source directory: a file added
    midway through a month-long run must not shift chunk boundaries.
    """
    resolved = input_dir.resolve()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if Path(manifest["image_dir"]).resolve() != resolved:
            raise ValueError(
                f"frozen manifest points to {manifest['image_dir']}, not {resolved}; "
                "use a new --run-name for a different source"
            )
        return manifest
    if not resolved.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {resolved}")
    files = sorted(
        (path for path in resolved.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )
    if not files:
        raise ValueError(f"no supported images found in {resolved}")
    seen: set[str] = set()
    images = []
    for index, path in enumerate(files):
        image_id = path.stem
        if image_id in seen:
            raise ValueError(f"duplicate image stem in source directory: {image_id}")
        seen.add(image_id)
        year_match = re.match(r"(\d{4})", image_id)
        images.append({
            "index": index,
            "image_id": image_id,
            "filename": path.name,
            "year": int(year_match.group(1)) if year_match else None,
            "source_size_bytes": path.stat().st_size,
        })
    manifest = {
        "schema_version": "qwen_final_directory_manifest_v1",
        "created_at": now(),
        "image_dir": str(resolved),
        "image_count": len(images),
        "sort_order": "casefolded_filename",
        "images": images,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def stable_manifest_hash(manifest: dict[str, Any]) -> str:
    identity = [
        (row["image_id"], row["filename"], row.get("year"), row.get("source_size_bytes"))
        for row in manifest["images"]
    ]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def ordered_manifest_images(manifest: dict[str, Any], order_path: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if order_path is None:
        return list(manifest["images"]), {
            "order_id": "source_casefolded_filename_order",
            "order_sha256": None,
            "path": None,
        }
    resolved = order_path.resolve()
    payload_bytes = resolved.read_bytes()
    payload = json.loads(payload_bytes.decode("utf-8"))
    image_ids = payload.get("image_ids")
    if not isinstance(image_ids, list) or not all(isinstance(value, str) for value in image_ids):
        raise ValueError("processing order must contain an image_ids string array")
    source = manifest["images"]
    source_ids = [row["image_id"] for row in source]
    if len(image_ids) != len(source_ids) or len(set(image_ids)) != len(image_ids) or set(image_ids) != set(source_ids):
        raise ValueError("processing order must be a complete, duplicate-free permutation of the frozen manifest")
    expected_hash = payload.get("source_manifest_sha256")
    actual_hash = stable_manifest_hash(manifest)
    if expected_hash is not None and expected_hash != actual_hash:
        raise ValueError("processing order was generated from a different source manifest")
    by_id = {row["image_id"]: row for row in source}
    return [by_id[image_id] for image_id in image_ids], {
        "order_id": str(payload.get("order_id") or resolved.stem),
        "order_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "path": str(resolved),
        "strategy": payload.get("strategy"),
        "seed": payload.get("seed"),
        "preserved_prefix_count": payload.get("preserved_prefix_count"),
    }


def decade_counts(images: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in images:
        year = row.get("year")
        key = str((int(year) // 10) * 10) if year is not None else "unknown"
        counts[key] += 1
    return dict(sorted(counts.items()))


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


WARC_CATEGORY_ALIASES = {
    "travel & tourism": "Transport & tourism",
    "education": "Non-profit, public sector & education",
}


def warc_category(value: Any) -> str | None:
    """Repair only unambiguous shape/casing variants of the frozen taxonomy."""
    if isinstance(value, list):
        if len(value) == 1:
            value = value[0]
        else:
            candidates = [warc_category(item) for item in value]
            return next((item for item in candidates if item is not None), None)
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    direct = {item.casefold(): item for item in prompts.AD_CATEGORIES}
    return direct.get(cleaned.casefold()) or WARC_CATEGORY_ALIASES.get(cleaned.casefold())


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
        else:
            groups = []
            # For individual-route ads, model array order is not an
            # assignment-quality ranking.  Remove children associated with
            # another ad before applying the nine-person safety cap.
            people = [person for person in people if center_inside(person["face_bbox_1000"], ad_box)]
            people = people[:9]
            for person in people:
                person["annotation_role"] = "individual"
                person["prominence_reason"] = None
            # Keep the model's band because missing boxes must remain an observable
            # structural error; do not silently rewrite it from the returned list.
        # Do not reject a response over metadata belonging to an ad that will
        # be removed for lacking eligible faces.  Keep the temporary empty ad
        # until clean_structure so the established page-level reason is stable.
        has_eligible_entities = bool(people or groups)
        category = warc_category(source.get("ad_category")) if has_eligible_entities else None
        review_flags = list(source.get("review_flags")) if isinstance(source.get("review_flags"), list) else []
        if has_eligible_entities and category is None and "ad_category_unresolved" not in review_flags:
            # Preserve the useful structure/faces.  The untouched model value is
            # retained in model_annotation_raw for later review.
            review_flags.append("ad_category_unresolved")
        brand = source.get("brand_or_advertiser")
        if not isinstance(brand, str) or not brand.strip():
            brand = None
        ads.append({
            "advertisement_id": ad_id,
            "extent": enum(source.get("extent"), ["full_page", "partial_page"], "partial_page"),
            "bbox_1000": ad_box,
            "ad_category": category,
            "brand_or_advertiser": brand.strip() if brand else None,
            "depiction_type": enum(source.get("depiction_type"), prompts.DEPICTION_TYPES, "photo_of_person"),
            "face_depiction_count_band": band,
            "duplicate_faces_present": None,
            "unique_face_count": None,
            "people": people,
            "groups": groups,
            "has_outstanding_individuals": "yes" if is_group and people else ("no" if is_group else None),
            "confidence": confidence(source.get("confidence")),
            "review_flags": review_flags,
        })
    page = {
        # Canonical annotation-app schema stores this categorical count as a
        # string even though the model-facing structure schema requests an
        # integer.
        "qualifying_ad_count": str(len(ads)),
        "no_qualifying_ad_reason": None if ads else enum(raw.get("no_qualifying_ad_reason"), ["no_ads_on_page", "ads_present_no_visible_faces"], "no_ads_on_page"),
    }
    annotation = {
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
    return clean_structure(annotation)


def center_inside(child: list[int], parent: list[int]) -> bool:
    cx, cy = (child[0] + child[2]) / 2, (child[1] + child[3]) / 2
    return parent[0] <= cx <= parent[2] and parent[1] <= cy <= parent[3]


def clean_structure(annotation: dict[str, Any]) -> dict[str, Any]:
    """Gold-free contract check: children must be inside a nonempty parent ad."""
    ads = []
    for ad in annotation.get("advertisements") or []:
        parent = ad["bbox_1000"]
        ad["people"] = [person for person in ad.get("people") or [] if center_inside(person["face_bbox_1000"], parent)]
        ad["groups"] = [group for group in ad.get("groups") or [] if center_inside(group["bbox_1000"], parent)]
        if not ad["people"] and not ad["groups"]:
            continue
        if ad["face_depiction_count_band"] in {"10_20", "20_plus"}:
            ad["has_outstanding_individuals"] = "yes" if ad["people"] else "no"
        ads.append(ad)
    annotation["advertisements"] = ads
    annotation["page"]["qualifying_ad_count"] = str(len(ads))
    annotation["qualifying_ad_count"] = str(len(ads))
    annotation["page"]["no_qualifying_ad_reason"] = None if ads else annotation["page"].get("no_qualifying_ad_reason") or "no_ads_on_page"
    annotation["no_qualifying_ad_reason"] = annotation["page"]["no_qualifying_ad_reason"]
    return annotation


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


def make_person_composite(page_path: Path, ad_box: list[int], face_box: list[int]) -> Image.Image:
    with Image.open(page_path) as source:
        page = source.convert("RGB")
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
    return canvas


def person_composite(page_path: Path, ad_box: list[int], face_box: list[int], out_path: Path) -> Path:
    if out_path.exists():
        return out_path
    canvas = make_person_composite(page_path, ad_box, face_box)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "JPEG", quality=92, optimize=True)
    return out_path


def make_group_crop(page_path: Path, box: list[int]) -> Image.Image:
    with Image.open(page_path) as source:
        page = source.convert("RGB")
    crop = page.crop(to_pixels(box, page.size))
    crop.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
    return crop


def group_crop(page_path: Path, box: list[int], out_path: Path) -> Path:
    if out_path.exists():
        return out_path
    crop = make_group_crop(page_path, box)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path, "JPEG", quality=92, optimize=True)
    return out_path


def person_request_content(
    page_path: Path,
    ad_box: list[int],
    face_box: list[int],
    prompt: str,
    cache_path: Path | None,
) -> list[dict[str, Any]]:
    if cache_path is not None:
        image_url = image_data_url(person_composite(page_path, ad_box, face_box, cache_path))
    else:
        image_url = jpeg_data_url(make_person_composite(page_path, ad_box, face_box), quality=92)
    return [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]


def group_request_content(
    page_path: Path,
    box: list[int],
    prompt: str,
    cache_path: Path | None,
) -> list[dict[str, Any]]:
    if cache_path is not None:
        image_url = image_data_url(group_crop(page_path, box, cache_path))
    else:
        image_url = jpeg_data_url(make_group_crop(page_path, box), quality=92)
    return [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]


class OpenRouterResponseError(RuntimeError):
    def __init__(self, payload: Any):
        error = payload.get("error") if isinstance(payload, dict) else None
        code = error.get("code") if isinstance(error, dict) else None
        message = error.get("message") if isinstance(error, dict) else str(payload)
        super().__init__(f"OpenRouter response error {code}: {message}")
        self.status_code = int(code) if str(code).isdigit() else None
        self.error_metadata = plain(error.get("metadata")) if isinstance(error, dict) else None
        self.response_headers = None


class OpenRouterHTTPError(RuntimeError):
    """HTTP failure retaining OpenRouter/Venice diagnostics for the ledger."""

    def __init__(self, exc: urllib.error.HTTPError, body: bytes, request_body_bytes: int):
        text = body.decode("utf-8", errors="replace")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        error = payload.get("error") if isinstance(payload, dict) else None
        message = error.get("message") if isinstance(error, dict) else text[:1200]
        super().__init__(f"OpenRouter HTTP {exc.code}: {message}")
        self.status_code = int(exc.code)
        self.headers = exc.headers
        self.request_body_bytes = request_body_bytes
        self.response_body_bytes = len(body)
        self.error_metadata = plain(error.get("metadata")) if isinstance(error, dict) else None
        self.response_headers = {
            str(key): str(value)
            for key, value in (exc.headers.items() if exc.headers is not None else [])
            if "rate" in str(key).lower() or "retry" in str(key).lower() or "provider" in str(key).lower()
        }


class OpenRouterClient:
    """Tiny dependency-free client; retry policy deliberately lives above it."""

    def __init__(self) -> None:
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self.api_key:
            self.api_key = KEY.read_text(encoding="utf-8").strip()

    def create(self, content: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        payload = json.dumps(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "provider": {"order": [PROVIDER], "allow_fallbacks": False},
                "reasoning": {"effort": "none", "exclude": True},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Title": "Qwen Standalone Production Pipeline",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                response_bytes = response.read()
                result = json.loads(response_bytes.decode("utf-8"))
                if isinstance(result, dict):
                    result["_client_request_body_bytes"] = len(payload)
                    result["_client_response_body_bytes"] = len(response_bytes)
                return result
        except urllib.error.HTTPError as exc:
            raise OpenRouterHTTPError(exc, exc.read(), len(payload)) from exc


def make_client() -> OpenRouterClient:
    return OpenRouterClient()


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
    last_category = ""
    compact_retry = budget.had_failure(run_meta, task_key, "response_truncated")
    token_limit = min(RECOVERY_MAX_TOKENS, math.ceil(max_tokens * 1.25)) if compact_retry else max_tokens
    for attempt in range(1, 6):
        budget.wait_for_endpoint(str(run_meta.get("stage") or ""))
        sequence = budget.reserve()
        started = time.time()
        usage = None
        request_body_bytes = None
        response_body_bytes = None
        error_metadata = None
        error_response_headers = None
        status = None
        planned_delay = 0.0
        attempt_token_limit = token_limit
        attempt_content = content
        if compact_retry:
            attempt_content = copy.deepcopy(content)
            if run_meta.get("stage") == "structure":
                # Dense or looping page outputs can be caused by repeated tile
                # enumeration.  The recovery request uses the same full page
                # but removes detail tiles; it does not change label criteria.
                attempt_content = attempt_content[:3]
            repair = (
                "\nRETRY AFTER OUTPUT-LIMIT TRUNCATION: return an aggressively compact JSON object with no prose and minimal review_flags. "
                "For page structure, a 10+ face advertisement MUST use people-area group boxes and MUST NOT enumerate ordinary faces; "
                "people[] may contain at most three truly outstanding faces. Merge adjacent ordinary faces into one broad people area. "
                "This retry supplies only the complete page; do not transcribe page text."
            )
            if attempt_content and attempt_content[0].get("type") == "text":
                attempt_content[0]["text"] = str(attempt_content[0].get("text") or "") + repair
        try:
            metadata = client.create(attempt_content, attempt_token_limit)
            request_body_bytes = metadata.pop("_client_request_body_bytes", None)
            response_body_bytes = metadata.pop("_client_response_body_bytes", None)
            usage = metadata.get("usage")
            if metadata.get("error"):
                raise OpenRouterResponseError(metadata)
            if not isinstance(metadata.get("choices"), list) or not metadata["choices"]:
                raise ValueError(f"provider response missing choices: {json.dumps(metadata, ensure_ascii=False)[:1200]}")
            raw = extract_json(str(metadata["choices"][0]["message"]["content"]))
            annotation = normalize(raw)
            ok = True
            last_error = ""
            last_category = "success"
        except Exception as exc:  # request/parse failures are ledgered and resumable
            raw = annotation = None
            ok = False
            request_body_bytes = request_body_bytes or getattr(exc, "request_body_bytes", None)
            response_body_bytes = response_body_bytes or getattr(exc, "response_body_bytes", None)
            last_error = f"{type(exc).__name__}: {exc}"
            error_metadata = plain(getattr(exc, "error_metadata", None))
            error_response_headers = plain(getattr(exc, "response_headers", None))
            last_category, status, header_delay, retryable = error_details(exc)
            completion_tokens = int((usage or {}).get("completion_tokens") or 0)
            if last_category == "response_parse_or_schema" and completion_tokens >= int(attempt_token_limit * .98):
                last_category = "response_truncated"
                compact_retry = True
                token_limit = min(RECOVERY_MAX_TOKENS, max(max_tokens, math.ceil(attempt_token_limit * 1.25)))
            if retryable and attempt < 5:
                planned_delay = retry_delay(last_category, attempt, header_delay)
                if last_category == "rate_limit":
                    budget.cooldown(planned_delay)
        budget.record({
            "request_sequence": sequence,
            **run_meta,
            "task_key": task_key,
            "attempt": attempt,
            "requested_max_tokens": attempt_token_limit,
            "ok": ok,
            "usage": usage,
            "request_body_bytes": request_body_bytes,
            "response_body_bytes": response_body_bytes,
            "error": None if ok else last_error,
            "error_category": None if ok else last_category,
            "http_status": status,
            "error_metadata": error_metadata,
            "error_response_headers": error_response_headers,
            "retry_delay_seconds": round(planned_delay, 3),
            "elapsed_seconds": round(time.time() - started, 3),
            "finished_at": now(),
        })
        if ok:
            return {"ok": True, "task_key": task_key, "model_annotation_raw": raw, "model_annotation": annotation, "usage": usage}
        if attempt < 5 and planned_delay > 0:
            time.sleep(planned_delay)
        else:
            break
    return {"ok": False, "task_key": task_key, "error": last_error, "error_category": last_category}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def endpoint_summary(meta: dict[str, Any], workers: int, elapsed: float, logical_jobs: int, records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row.get("elapsed_seconds") or 0) for row in records]
    errors = Counter(str(row.get("error_category")) for row in records if not row.get("ok"))
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_task.setdefault(str(row.get("task_key")), []).append(row)
    recovered = sum(any(not x.get("ok") for x in values) and any(x.get("ok") for x in values) for values in by_task.values())
    unrecovered = sum(not any(x.get("ok") for x in values) for values in by_task.values())
    usage = Counter()
    transport = Counter()
    for row in records:
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            usage[key] += int((row.get("usage") or {}).get(key) or 0)
        try:
            usage["cost_usd"] += float((row.get("usage") or {}).get("cost") or 0)
        except (TypeError, ValueError):
            pass
        transport["request_body_bytes"] += int(row.get("request_body_bytes") or 0)
        transport["response_body_bytes"] += int(row.get("response_body_bytes") or 0)
    total_tokens = int(usage.get("total_tokens") or 0)
    return {
        **meta,
        "configured_concurrency": workers,
        "logical_jobs": logical_jobs,
        "physical_attempts": len(records),
        "successful_attempts": sum(bool(row.get("ok")) for row in records),
        "failed_attempts": sum(not bool(row.get("ok")) for row in records),
        "recovered_tasks": recovered,
        "unrecovered_tasks": unrecovered,
        "attempts_per_logical_job": len(records) / logical_jobs if logical_jobs else None,
        "error_categories": dict(errors),
        "wall_seconds_this_execution": elapsed,
        "completed_jobs_per_minute": logical_jobs * 60 / elapsed if elapsed else None,
        "reported_tokens_per_minute": total_tokens * 60 / elapsed if elapsed else None,
        "latency_seconds": {"p50": percentile(latencies, .50), "p90": percentile(latencies, .90), "p95": percentile(latencies, .95), "max": max(latencies) if latencies else None},
        "usage": dict(usage),
        "transport": dict(transport),
    }


def execute_jobs(
    jobs,
    result_path: Path,
    budget: Budget,
    workers: int,
    max_in_flight: int | None = None,
) -> dict[str, Any] | None:
    """Consume jobs lazily and keep only a bounded number of image payloads live."""
    done = {row["task_key"] for row in iter_jsonl(result_path) if row.get("ok")}
    local = threading.local()

    def run(job: dict[str, Any]) -> dict[str, Any]:
        if not hasattr(local, "client"):
            local.client = make_client()
        try:
            content = job["content_factory"]() if "content_factory" in job else job["content"]
        except Exception as exc:
            return {
                "ok": False,
                "task_key": job["task_key"],
                "error": f"{type(exc).__name__}: {exc}",
                "error_category": "local_content_build",
            }
        return one_request(local.client, budget, job["meta"], job["task_key"], content, job["max_tokens"], job["normalize"])

    completed = 0
    skipped = 0
    discovered = 0
    selected_meta: dict[str, Any] | None = None
    start_sequence = budget.begin_execution()
    started = time.time()
    iterator = iter(jobs)
    capacity = max(workers, max_in_flight or workers * 2)
    source_exhausted = False
    fatal_exception: Exception | None = None

    def next_pending():
        nonlocal skipped, discovered, selected_meta, source_exhausted
        while not source_exhausted:
            try:
                job = next(iterator)
            except StopIteration:
                source_exhausted = True
                return None
            discovered += 1
            if selected_meta is None:
                selected_meta = {key: job["meta"].get(key) for key in ["run_id", "cohort", "stage"]}
            if job["task_key"] in done:
                skipped += 1
                continue
            return job
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {}
        while len(future_map) < capacity:
            job = next_pending()
            if job is None:
                break
            future_map[pool.submit(run, job)] = job
        while future_map:
            finished, _ = wait(future_map, return_when=FIRST_COMPLETED)
            for future in finished:
                future_map.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    # Most notably request-budget exhaustion. Stop feeding new
                    # work, but drain already-running futures so their paid
                    # successes are checkpointed rather than lost.
                    fatal_exception = fatal_exception or exc
                    source_exhausted = True
                    continue
                append_jsonl(result_path, row)
                completed += 1
                print(f"[{completed} complete; {skipped} resumed] {row['task_key']} {'ok' if row.get('ok') else 'FAILED'}", flush=True)
            while fatal_exception is None and len(future_map) < capacity:
                job = next_pending()
                if job is None:
                    break
                future_map[pool.submit(run, job)] = job
    elapsed = time.time() - started
    execution_records = budget.finish_execution()
    if completed == 0 and fatal_exception is None:
        print(f"nothing pending: {result_path} ({skipped} successful jobs already present)")
        return None
    if selected_meta is not None:
        summary = endpoint_summary(selected_meta, workers, elapsed, completed, execution_records)
        local_failures = sum(
            1
            for row in iter_jsonl(result_path)
            if not row.get("ok") and row.get("error_category") == "local_content_build"
        )
        summary["local_content_build_failures_in_result_file"] = local_failures
        summary["request_sequence_start"] = start_sequence
        summary["request_sequence_end"] = max([int(row.get("request_sequence") or 0) for row in execution_records] or [start_sequence - 1])
        summary["already_complete_jobs_skipped"] = skipped
        summary["jobs_discovered"] = discovered
        summary["max_in_flight"] = capacity
        summary["execution_complete"] = fatal_exception is None
        summary["fatal_error"] = None if fatal_exception is None else f"{type(fatal_exception).__name__}: {fatal_exception}"
        summary["pacing"] = budget.pacing_snapshot()
        path = OUTPUT / "endpoint_behavior" / f"{selected_meta['cohort']}_{selected_meta['stage']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        history = OUTPUT / "endpoint_behavior" / "history" / f"{selected_meta['cohort']}_{selected_meta['stage']}_{start_sequence:05d}.json"
        history.parent.mkdir(parents=True, exist_ok=True)
        history.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"endpoint_summary": str(path), **summary}, indent=2), flush=True)
        if fatal_exception is not None:
            raise fatal_exception
        return summary
    if fatal_exception is not None:
        raise fatal_exception
    return None


def execute_mixed_jobs(groups: list[dict[str, Any]], budget: Budget) -> dict[str, dict[str, Any] | None]:
    """Run independent stage queues concurrently under one shared token budget.

    Each queue retains its own worker and in-flight limits and writes to its
    established result file.  Only launch timing is shared, so mixing cannot
    change prompts, inputs, normalization, task keys, or resumability.
    """
    local = threading.local()

    def run_job(job: dict[str, Any]) -> dict[str, Any]:
        if not hasattr(local, "client"):
            local.client = make_client()
        try:
            content = job["content_factory"]() if "content_factory" in job else job["content"]
        except Exception as exc:
            return {
                "ok": False,
                "task_key": job["task_key"],
                "error": f"{type(exc).__name__}: {exc}",
                "error_category": "local_content_build",
            }
        return one_request(
            local.client,
            budget,
            job["meta"],
            job["task_key"],
            content,
            job["max_tokens"],
            job["normalize"],
        )

    states = []
    for spec in groups:
        result_path = Path(spec["result_path"])
        jobs = list(spec["jobs"])
        done = {row["task_key"] for row in iter_jsonl(result_path) if row.get("ok")}
        pending = [job for job in jobs if job["task_key"] not in done]
        states.append({
            **spec,
            "jobs": jobs,
            "pending": iter(pending),
            "discovered": len(jobs),
            "skipped": len(jobs) - len(pending),
            "completed": 0,
            "source_exhausted": not pending,
            "meta": ({key: jobs[0]["meta"].get(key) for key in ["run_id", "cohort", "stage"]} if jobs else None),
            "capacity": max(int(spec["workers"]), int(spec.get("max_in_flight") or int(spec["workers"]) * 2)),
        })

    if not any(not state["source_exhausted"] for state in states):
        for state in states:
            print(
                f"nothing pending: {state['result_path']} ({state['skipped']} successful jobs already present)",
                flush=True,
            )
        return {str(state["name"]): None for state in states}

    start_sequence = budget.begin_execution()
    started = time.time()
    fatal_exception: Exception | None = None
    pools = [ThreadPoolExecutor(max_workers=int(state["workers"])) for state in states]
    future_map: dict[Any, tuple[dict[str, Any], dict[str, Any]]] = {}

    def fill(state: dict[str, Any], pool: ThreadPoolExecutor) -> None:
        while fatal_exception is None and not state["source_exhausted"]:
            active = sum(1 for queued_state, _ in future_map.values() if queued_state is state)
            if active >= state["capacity"]:
                return
            try:
                job = next(state["pending"])
            except StopIteration:
                state["source_exhausted"] = True
                return
            future_map[pool.submit(run_job, job)] = (state, job)

    try:
        for state, pool in zip(states, pools):
            fill(state, pool)
        while future_map:
            finished, _ = wait(future_map, return_when=FIRST_COMPLETED)
            for future in finished:
                state, job = future_map.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    fatal_exception = fatal_exception or exc
                    for queued_state in states:
                        queued_state["source_exhausted"] = True
                    continue
                append_jsonl(Path(state["result_path"]), row)
                state["completed"] += 1
                print(
                    f"[{state['name']} {state['completed']} complete; {state['skipped']} resumed] "
                    f"{row['task_key']} {'ok' if row.get('ok') else 'FAILED'}",
                    flush=True,
                )
            if fatal_exception is None:
                for state, pool in zip(states, pools):
                    fill(state, pool)
    finally:
        for pool in pools:
            pool.shutdown(wait=True)

    elapsed = time.time() - started
    records = budget.finish_execution()
    summaries: dict[str, dict[str, Any] | None] = {}
    for state in states:
        meta = state["meta"]
        name = str(state["name"])
        if meta is None or state["completed"] == 0:
            summaries[name] = None
            continue
        stage_records = [row for row in records if row.get("stage") == meta["stage"]]
        summary = endpoint_summary(meta, int(state["workers"]), elapsed, int(state["completed"]), stage_records)
        result_path = Path(state["result_path"])
        summary["local_content_build_failures_in_result_file"] = sum(
            1 for row in iter_jsonl(result_path)
            if not row.get("ok") and row.get("error_category") == "local_content_build"
        )
        summary["request_sequence_start"] = start_sequence
        summary["request_sequence_end"] = max(
            [int(row.get("request_sequence") or 0) for row in stage_records] or [start_sequence - 1]
        )
        summary["already_complete_jobs_skipped"] = state["skipped"]
        summary["jobs_discovered"] = state["discovered"]
        summary["max_in_flight"] = state["capacity"]
        summary["execution_complete"] = fatal_exception is None
        summary["fatal_error"] = None if fatal_exception is None else f"{type(fatal_exception).__name__}: {fatal_exception}"
        summary["mixed_execution"] = True
        summary["mixed_stages"] = [str(other["name"]) for other in states]
        summary["pacing"] = budget.pacing_snapshot()
        path = OUTPUT / "endpoint_behavior" / f"{meta['cohort']}_{meta['stage']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        history = OUTPUT / "endpoint_behavior" / "history" / f"{meta['cohort']}_{meta['stage']}_{start_sequence:05d}.json"
        history.parent.mkdir(parents=True, exist_ok=True)
        history.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"endpoint_summary": str(path), **summary}, indent=2), flush=True)
        summaries[name] = summary
    if fatal_exception is not None:
        raise fatal_exception
    return summaries


def stage_selection(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if hasattr(args, "_manifest") and hasattr(args, "_images"):
        return args._manifest, args._images
    manifest, images = selected_images(args.cohort, args.pilot_only)
    start = max(0, int(getattr(args, "start", 0) or 0))
    limit = getattr(args, "limit", None)
    return manifest, images[start : start + limit if limit is not None else None]


def inference_meta(args: argparse.Namespace, stage: str, **extra: Any) -> dict[str, Any]:
    return {
        "run_id": getattr(args, "run_id", "standalone_final_398_v1"),
        "cohort": args.cohort,
        "stage": stage,
        "variant": args.variant,
        "prompt_version": prompts.PROMPT_VERSION,
        "model": MODEL,
        "provider": PROVIDER,
        **extra,
    }


def structure_jobs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path]:
    manifest, images = stage_selection(args)
    result_path = OUTPUT / args.cohort / f"structure_{args.variant}.jsonl"
    jobs = []
    for item in images:
        image_id = item["image_id"]
        page_path = Path(manifest["image_dir"]) / item["filename"]
        tiled = args.variant.startswith("s2")
        prompt = prompts.structure_prompt(image_id, item.get("year"), tiled, include_business=args.variant != "s2n")
        jobs.append({
            "task_key": image_id,
            "content_factory": lambda page_path=page_path, prompt=prompt, tiled=tiled: structure_content(page_path, prompt, "s2" if tiled else "s1"),
            "max_tokens": 12000,
            "normalize": lambda raw, image_id=image_id: normalize_structure(raw, image_id),
            "meta": inference_meta(args, "structure"),
        })
    return jobs, result_path


def structure_stage(args: argparse.Namespace, budget: Budget) -> dict[str, Any] | None:
    jobs, result_path = structure_jobs(args)
    return execute_jobs(jobs, result_path, budget, args.workers, getattr(args, "max_in_flight", None))


def structure_map(cohort: str, variant: str, wanted: set[str] | None = None) -> dict[str, dict[str, Any]]:
    path = OUTPUT / cohort / f"structure_{variant}.jsonl"
    return {
        row["task_key"]: row["model_annotation"]
        for row in iter_jsonl(path)
        if row.get("ok") and (wanted is None or row.get("task_key") in wanted)
    }


def entity_jobs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path]:
    manifest, images = stage_selection(args)
    wanted = {item["image_id"] for item in images}
    structures = structure_map(args.cohort, args.variant, wanted)
    gaze = args.gaze == "yes"
    suffix = f"{args.style}_{'gaze' if gaze else 'nogaze'}"
    result_path = OUTPUT / args.cohort / f"entities_{args.variant}_{suffix}.jsonl"
    cache = OUTPUT / args.cohort / "crops" / args.variant
    use_cache = getattr(args, "crop_cache", "disk") == "disk"
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
                prompt = prompts.person_prompt(task_id, args.style, gaze)
                cache_path = cache / f"{task_id.replace('::', '__')}_person.jpg" if use_cache else None
                jobs.append({
                    "task_key": task_id,
                    "content_factory": lambda page_path=page_path, ad_box=ad["bbox_1000"], face_box=person["face_bbox_1000"], prompt=prompt, cache_path=cache_path: person_request_content(page_path, ad_box, face_box, prompt, cache_path),
                    "max_tokens": 1500,
                    "normalize": lambda raw, task_id=task_id: normalize_person(raw, task_id, args.style, gaze),
                    "meta": inference_meta(args, "entities", entity_type="person", style=args.style, gaze=gaze),
                })
            for group in ad.get("groups") or []:
                task_id = f"{image_id}::{group['group_id']}"
                prompt = prompts.group_prompt(task_id, gaze)
                cache_path = cache / f"{task_id.replace('::', '__')}_group.jpg" if use_cache else None
                jobs.append({
                    "task_key": task_id,
                    "content_factory": lambda page_path=page_path, box=group["bbox_1000"], prompt=prompt, cache_path=cache_path: group_request_content(page_path, box, prompt, cache_path),
                    "max_tokens": 1300,
                    "normalize": lambda raw, task_id=task_id: normalize_group(raw, task_id, gaze),
                    "meta": inference_meta(args, "entities", entity_type="group", style=args.style, gaze=gaze),
                })
    return jobs, result_path


def entity_stage(args: argparse.Namespace, budget: Budget) -> dict[str, Any] | None:
    jobs, result_path = entity_jobs(args)
    return execute_jobs(jobs, result_path, budget, args.workers, getattr(args, "max_in_flight", None))


def assemble_stage(args: argparse.Namespace) -> None:
    gaze = args.gaze == "yes"
    suffix = f"{args.style}_{'gaze' if gaze else 'nogaze'}"
    structures = structure_map(args.cohort, args.variant)
    entity_path = OUTPUT / args.cohort / f"entities_{args.variant}_{suffix}.jsonl"
    entity_records = {
        row["task_key"]: (row["model_annotation"], row.get("model_annotation_raw"))
        for row in iter_jsonl(entity_path)
        if row.get("ok")
    }
    manifest = load_manifest(args.cohort)
    out_path = OUTPUT / args.cohort / "assembled" / "final.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT / "raw_values" / f"{args.cohort}.jsonl"
    norm_path = OUTPUT / "normalization" / f"{args.cohort}.jsonl"
    for path in [raw_path, norm_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = [path.with_name(path.name + ".tmp") for path in [out_path, raw_path, norm_path]]
    pages = complete_pages = raw_count = norm_count = 0
    with (
        temporary_paths[0].open("w", encoding="utf-8") as out_handle,
        temporary_paths[1].open("w", encoding="utf-8") as raw_handle,
        temporary_paths[2].open("w", encoding="utf-8") as norm_handle,
    ):
        for item in manifest["images"]:
            image_id = item["image_id"]
            structure = structures.get(image_id)
            if structure is None:
                continue
            annotation = copy.deepcopy(structure)
            annotation["schema_version"] = "qwen_final_standalone_v1"
            complete = True

            def write_normalization(row: dict[str, Any]) -> None:
                nonlocal norm_count
                norm_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                norm_count += 1

            for ad in annotation.get("advertisements") or []:
                for person in ad.get("people") or []:
                    key = f"{image_id}::{person['person_id']}"
                    record = entity_records.get(key)
                    if record is None:
                        complete = False
                        continue
                    attrs, raw_response = record
                    raw_handle.write(json.dumps({"task_key": key, "image_id": image_id, "entity_type": "person", "raw_model_response": raw_response, "raw_normalized_values": copy.deepcopy(attrs)}, ensure_ascii=False) + "\n")
                    raw_count += 1
                    person.update({field: value for field, value in attrs.items() if field != "person_task_id"})
                    person.update({"duplicate_of_person_id": None, "duplicate_person_ids": [], "gaze_target_person_id": None, "gaze_target_object_ref": None, "mouth_covering_other_text": None})
                    if person.get("face_expression_legibility") == "0_not_legible":
                        for field in ["gaze_target", "gaze_target_person_unboxed", "smile_present", "smile_intensity"]:
                            if person.get(field) is not None:
                                write_normalization({"task_key": key, "field": field, "from": person.get(field), "to": None, "rule": "v1.17_zero_legibility_null"})
                            person[field] = None
                    elif person.get("smile_present") != "yes":
                        if person.get("smile_intensity") is not None:
                            write_normalization({"task_key": key, "field": "smile_intensity", "from": person.get("smile_intensity"), "to": None, "rule": "v1.17_no_smile_intensity_null"})
                        person["smile_intensity"] = None
                for group in ad.get("groups") or []:
                    key = f"{image_id}::{group['group_id']}"
                    record = entity_records.get(key)
                    if record is None:
                        complete = False
                        continue
                    attrs, raw_response = record
                    raw_handle.write(json.dumps({"task_key": key, "image_id": image_id, "entity_type": "group", "raw_model_response": raw_response, "raw_normalized_values": copy.deepcopy(attrs)}, ensure_ascii=False) + "\n")
                    raw_count += 1
                    group.update({field: value for field, value in attrs.items() if field != "group_task_id"})
                    if group.get("expression_legibility_distribution") == "all_0_not_legible":
                        for field in ["dominant_gaze", "smile_prevalence", "dominant_smile_intensity"]:
                            if group.get(field) is not None:
                                write_normalization({"task_key": key, "field": field, "from": group.get(field), "to": None, "rule": "v1.17_group_zero_legibility_null"})
                            group[field] = None
                    elif group.get("smile_prevalence") in {"none", "not_assessable", None}:
                        if group.get("dominant_smile_intensity") is not None:
                            write_normalization({"task_key": key, "field": "dominant_smile_intensity", "from": group.get("dominant_smile_intensity"), "to": None, "rule": "v1.17_group_no_smile_intensity_null"})
                        group["dominant_smile_intensity"] = None
            row = {"image_id": image_id, "filename": item["filename"], "cohort": args.cohort, "route": "standalone_final_v1", "ok": complete, "annotation": annotation}
            out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            pages += 1
            complete_pages += int(complete)
    for temporary, target in zip(temporary_paths, [out_path, raw_path, norm_path]):
        os.replace(temporary, target)
    print(json.dumps({"path": str(out_path), "pages": pages, "complete": complete_pages, "raw_entities": raw_count, "normalization_events": norm_count}, indent=2))


DUPLICATE_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#17becf", "#bcbd22"]


def duplicate_normalize(raw: dict[str, Any], task_key: str, allowed_ids: list[str]) -> dict[str, Any]:
    allowed = set(allowed_ids); clusters = []
    source = raw.get("candidate_clusters") if isinstance(raw.get("candidate_clusters"), list) else []
    for item in source:
        if not isinstance(item, dict):
            continue
        ids = sorted({str(value) for value in item.get("person_ids", []) if str(value) in allowed}) if isinstance(item.get("person_ids"), list) else []
        strength = item.get("candidate_strength")
        if len(ids) >= 2 and strength in {"high", "medium"}:
            clusters.append({"person_ids": ids, "candidate_strength": strength})
    merged = []
    for cluster in clusters:
        overlaps = [old for old in merged if set(old["person_ids"]) & set(cluster["person_ids"])]
        if not overlaps:
            merged.append(cluster); continue
        ids = set(cluster["person_ids"]); strength = cluster["candidate_strength"]
        for old in overlaps:
            ids.update(old["person_ids"]); strength = "medium" if "medium" in {strength, old["candidate_strength"]} else "high"; merged.remove(old)
        merged.append({"person_ids": sorted(ids), "candidate_strength": strength})
    return {"task_key": task_key, "candidate_clusters": sorted(merged, key=lambda item: item["person_ids"])}


def fit_panel(image: Image.Image, width: int, height: int, label: str) -> Image.Image:
    panel = Image.new("RGB", (width, height), "white"); work = image.copy(); work.thumbnail((width - 16, height - 46), Image.Resampling.LANCZOS)
    panel.paste(work, ((width - work.width) // 2, 40 + (height - 40 - work.height) // 2)); ImageDraw.Draw(panel).text((10, 12), label, fill="black", font=ImageFont.load_default()); return panel


def duplicate_montage(page_path: Path, ad: dict[str, Any], out_path: Path) -> Path:
    if out_path.exists():
        return out_path
    page = Image.open(page_path).convert("RGB"); ad_px = to_pixels(ad["bbox_1000"], page.size); ad_image = page.crop(ad_px); draw = ImageDraw.Draw(ad_image); ax1, ay1, _, _ = ad_px
    people = ad.get("people") or []
    for index, person in enumerate(people):
        x1, y1, x2, y2 = to_pixels(person["face_bbox_1000"], page.size); color = DUPLICATE_COLORS[index % len(DUPLICATE_COLORS)]
        draw.rectangle((x1 - ax1, y1 - ay1, x2 - ax1, y2 - ay1), outline=color, width=max(3, round(min(page.size) / 350))); draw.text((x1 - ax1 + 2, max(0, y1 - ay1 - 16)), f"P{index+1}", fill=color, font=ImageFont.load_default())
    panels = [fit_panel(ad_image, 760, 620, "FROZEN ADVERTISEMENT CONTEXT")]
    for index, person in enumerate(people):
        face = page.crop(expand_box(to_pixels(person["face_bbox_1000"], page.size), page.size, 1.4)); panels.append(fit_panel(face, 300, 300, f"P{index+1}: {person['person_id']}"))
    columns = 3; rows_count = math.ceil(len(people) / columns); canvas = Image.new("RGB", (760 + columns * 300, max(620, rows_count * 300)), "white"); canvas.paste(panels[0], (0, 0))
    for index, panel in enumerate(panels[1:]):
        canvas.paste(panel, (760 + (index % columns) * 300, (index // columns) * 300))
    out_path.parent.mkdir(parents=True, exist_ok=True); canvas.save(out_path, "JPEG", quality=92, optimize=True); return out_path


def duplicate_stage(args: argparse.Namespace, budget: Budget) -> None:
    man = load_manifest(args.cohort); by_id = {item["image_id"]: item for item in man["images"]}; jobs = []
    final_rows = [row for row in jsonl(OUTPUT / args.cohort / "assembled" / "final.jsonl") if row.get("ok")]
    for row in final_rows:
        page_path = Path(man["image_dir"]) / by_id[row["image_id"]]["filename"]
        for ad in row["annotation"].get("advertisements") or []:
            people = ad.get("people") or []
            if len(people) < 2:
                continue
            task_key = f"{row['image_id']}::{ad['advertisement_id']}"; ids = [person["person_id"] for person in people]
            montage = duplicate_montage(page_path, ad, OUTPUT / args.cohort / "duplicate_montages" / f"{task_key.replace('::', '__')}.jpg")
            jobs.append({
                "task_key": task_key,
                "content": [{"type": "text", "text": prompts.duplicate_prompt(task_key, ids)}, {"type": "image_url", "image_url": {"url": image_data_url(montage)}}],
                "max_tokens": 700,
                "normalize": lambda raw, task_key=task_key, ids=ids: duplicate_normalize(raw, task_key, ids),
                "meta": {"run_id": "standalone_final_398_v1", "cohort": args.cohort, "stage": "duplicates", "variant": "conservative_sidecar_v1", "prompt_version": prompts.PROMPT_VERSION, "model": MODEL, "provider": PROVIDER},
            })
    execute_jobs(jobs, OUTPUT / "duplicate_candidates" / f"{args.cohort}.jsonl", budget, args.workers)


def configure_production_run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    global OUTPUT, LEDGER, ACTIVE_MANIFEST
    run_name = safe_name(args.run_name)
    run_root = BASE_OUTPUT / "production" / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    OUTPUT = run_root
    LEDGER = run_root / "request_ledger.jsonl"
    ACTIVE_MANIFEST = run_root / "manifest.json"
    manifest = freeze_directory_manifest(Path(args.input_dir), ACTIVE_MANIFEST)
    specification = {
        "schema_version": "qwen_final_production_spec_v1",
        "run_name": run_name,
        "image_dir": manifest["image_dir"],
        "manifest_image_count": len(manifest["images"]),
        "model": MODEL,
        "provider": PROVIDER,
        "prompt_version": prompts.PROMPT_VERSION,
        "variant": args.variant,
        "style": args.style,
        "gaze": args.gaze,
        "response_format": "json_object",
        "temperature": 0,
        "reasoning": "none",
        "duplicate_stage": "disabled_after_failed_validation",
        "crop_cache": args.crop_cache,
    }
    config_path = run_root / "run_config.json"
    if config_path.exists():
        frozen = json.loads(config_path.read_text(encoding="utf-8"))
        differences = {
            key: {"frozen": frozen.get(key), "requested": value}
            for key, value in specification.items()
            if frozen.get(key) != value
        }
        if differences:
            raise ValueError(
                "run specification changed; resumption would mix incompatible outputs. "
                f"Use a new --run-name. Differences: {json.dumps(differences, ensure_ascii=False)}"
            )
    else:
        atomic_write_json(config_path, {**specification, "created_at": now()})
    return manifest, run_root


def call_estimate(image_count: int) -> dict[str, Any]:
    # Empirical ratios from the frozen 398-page validation run. These estimates
    # size a safety budget; the actual entity count is determined by structure.
    entities_per_page = 732 / 398
    attempts_per_logical_call = 1218 / 1130
    logical = image_count * (1 + entities_per_page)
    physical = logical * attempts_per_logical_call
    return {
        "pages": image_count,
        "estimated_entity_calls": round(image_count * entities_per_page),
        "estimated_logical_calls": round(logical),
        "estimated_physical_attempts": round(physical),
        "suggested_request_budget_with_12pct_margin": math.ceil(physical * 1.12),
        "estimated_main_cost_usd": round(image_count * 0.00260, 2),
    }


def production_audit(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    run_root: Path,
    budget: Budget,
    stopped_reason: str | None = None,
) -> dict[str, Any]:
    selected_ids = {item["image_id"] for item in selected}
    structures = structure_map(args.cohort, args.variant, selected_ids)
    expected_entities = set()
    for image_id, annotation in structures.items():
        for ad in annotation.get("advertisements") or []:
            expected_entities.update(f"{image_id}::{person['person_id']}" for person in ad.get("people") or [])
            expected_entities.update(f"{image_id}::{group['group_id']}" for group in ad.get("groups") or [])
    gaze = args.gaze == "yes"
    suffix = f"{args.style}_{'gaze' if gaze else 'nogaze'}"
    entity_path = OUTPUT / args.cohort / f"entities_{args.variant}_{suffix}.jsonl"
    successful_entities = {row["task_key"] for row in iter_jsonl(entity_path) if row.get("ok")}
    assembled_path = OUTPUT / args.cohort / "assembled" / "final.jsonl"
    assembled_all = {
        row["image_id"]: bool(row.get("ok"))
        for row in iter_jsonl(assembled_path)
    }
    assembled = {image_id: assembled_all.get(image_id, False) for image_id in selected_ids}
    usage = Counter()
    attempts = failures = 0
    stage_stats: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(budget.ledger_path):
        attempts += 1
        failures += int(not row.get("ok"))
        stage = str(row.get("stage") or "unknown")
        stats = stage_stats.setdefault(stage, {"attempts": 0, "successful_attempts": 0, "rate_limit_attempts": 0, "task_keys": set(), "elapsed_seconds": [], "request_body_bytes": 0, "response_body_bytes": 0})
        stats["attempts"] += 1
        stats["successful_attempts"] += int(bool(row.get("ok")))
        stats["rate_limit_attempts"] += int(row.get("error_category") == "rate_limit")
        stats["task_keys"].add(str(row.get("task_key")))
        stats["elapsed_seconds"].append(float(row.get("elapsed_seconds") or 0))
        stats["request_body_bytes"] += int(row.get("request_body_bytes") or 0)
        stats["response_body_bytes"] += int(row.get("response_body_bytes") or 0)
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            usage[key] += int((row.get("usage") or {}).get(key) or 0)
        try:
            usage["cost_usd"] += float((row.get("usage") or {}).get("cost") or 0)
        except (TypeError, ValueError):
            pass
    missing_structures = sorted(selected_ids - structures.keys())
    missing_entities = sorted(expected_entities - successful_entities)
    incomplete_assembled = sorted(image_id for image_id in selected_ids if not assembled.get(image_id, False))
    unresolved_tasks = set(missing_structures) | set(missing_entities)
    latest_unresolved_attempts = {}
    if unresolved_tasks:
        for row in iter_jsonl(budget.ledger_path):
            task_key = str(row.get("task_key") or "")
            if task_key in unresolved_tasks:
                latest_unresolved_attempts[task_key] = row
    terminal_reasons = {
        task_key: reason
        for task_key in sorted(unresolved_tasks)
        if (reason := terminal_failure_reason(latest_unresolved_attempts.get(task_key))) is not None
    }
    retryable_unresolved = sorted(unresolved_tasks - set(terminal_reasons))
    terminal_pages = sorted({task_key.split("::", 1)[0] for task_key in terminal_reasons})
    terminal_reason_counts = Counter(terminal_reasons.values())
    all_selected_complete = not missing_structures and not missing_entities and not incomplete_assembled
    supervisor_terminal_state = bool(
        stopped_reason is None
        and not retryable_unresolved
        and set(incomplete_assembled).issubset(terminal_pages)
    )
    cumulative_structure_ids = {
        row["task_key"]
        for row in iter_jsonl(OUTPUT / args.cohort / f"structure_{args.variant}.jsonl")
        if row.get("ok")
    }
    endpoint_stages = {}
    for stage, stats in stage_stats.items():
        stage_attempts = stats["attempts"]
        endpoint_stages[stage] = {
            "logical_task_keys_seen": len(stats["task_keys"]),
            "physical_attempts": stage_attempts,
            "successful_attempts": stats["successful_attempts"],
            "rate_limit_attempts": stats["rate_limit_attempts"],
            "rate_limit_attempt_fraction": stats["rate_limit_attempts"] / stage_attempts if stage_attempts else 0,
            "request_body_bytes": stats["request_body_bytes"],
            "response_body_bytes": stats["response_body_bytes"],
            "attempt_latency_seconds": {
                "p50": percentile(stats["elapsed_seconds"], .50),
                "p95": percentile(stats["elapsed_seconds"], .95),
            },
        }
    report = {
        "schema_version": "qwen_final_production_status_v1",
        "run_name": args.run_name,
        "updated_at": now(),
        "selected_range": {"start": args.start, "count": len(selected), "end_exclusive": args.start + len(selected)},
        "processing_order": getattr(args, "_processing_order", None),
        "selected_decade_counts": decade_counts(selected),
        "selected_pages": len(selected),
        "successful_structures": len(structures),
        "expected_entities": len(expected_entities),
        "successful_expected_entities": len(expected_entities & successful_entities),
        "complete_assembled_pages": sum(bool(assembled.get(image_id)) for image_id in selected_ids),
        "all_selected_complete": all_selected_complete,
        "supervisor_terminal_state": supervisor_terminal_state,
        "stopped_reason": stopped_reason,
        "cumulative": {
            "manifest_pages": len(load_manifest(args.cohort)["images"]),
            "successful_structure_pages": len(cumulative_structure_ids),
            "complete_assembled_pages": sum(bool(value) for value in assembled_all.values()),
            "assembled_pages_any_status": len(assembled_all),
        },
        "unrecovered": {
            "structure_count": len(missing_structures),
            "entity_count": len(missing_entities),
            "assembled_page_count": len(incomplete_assembled),
            "structure_sample": missing_structures[:100],
            "entity_sample": missing_entities[:100],
            "assembled_page_sample": incomplete_assembled[:100],
        },
        "terminal_unrecovered": {
            "task_count": len(terminal_reasons),
            "page_count": len(terminal_pages),
            "reason_counts": dict(terminal_reason_counts),
            "task_sample": list(terminal_reasons)[:100],
            "page_sample": terminal_pages[:100],
            "retryable_task_count": len(retryable_unresolved),
            "retryable_task_sample": retryable_unresolved[:100],
        },
        "request_ledger": {
            "physical_attempts_all_invocations": attempts,
            "failed_attempts_all_invocations": failures,
            "physical_attempts_this_invocation": budget.used - budget.starting_used,
            "additional_request_budget_this_invocation": budget.limit - budget.starting_used,
            "absolute_run_ledger_ceiling_this_invocation": budget.limit,
            "usage": dict(usage),
            "by_stage": endpoint_stages,
        },
        "paths": {
            "run_root": str(run_root),
            "manifest": str(ACTIVE_MANIFEST),
            "assembled": str(assembled_path),
            "ledger": str(budget.ledger_path),
        },
    }
    atomic_write_json(run_root / "run_status.json", report)
    append_jsonl(run_root / "range_history.jsonl", {
        "finished_at": now(),
        "start": args.start,
        "count": len(selected),
        "processing_order_id": (getattr(args, "_processing_order", None) or {}).get("order_id"),
        "processing_order_sha256": (getattr(args, "_processing_order", None) or {}).get("order_sha256"),
        "selected_decade_counts": decade_counts(selected),
        "all_selected_complete": report["all_selected_complete"],
        "supervisor_terminal_state": report["supervisor_terminal_state"],
        "physical_attempts_all_invocations": attempts,
    })
    return report


def run_all_stage(args: argparse.Namespace) -> int:
    manifest, run_root = configure_production_run(args)
    order_path = getattr(args, "processing_order", None)
    images, processing_order = ordered_manifest_images(manifest, order_path)
    if order_path is not None:
        order_payload = json.loads(Path(order_path).read_text(encoding="utf-8"))
        archive_dir = run_root / "processing_orders"
        archive_path = archive_dir / f"{safe_name(processing_order['order_id'])}_{processing_order['order_sha256'][:12]}.json"
        if not archive_path.exists():
            atomic_write_json(archive_path, order_payload)
        processing_order["archived_path"] = str(archive_path)
    args._processing_order = processing_order
    if args.start < 0 or args.start >= len(images):
        raise ValueError(f"--start must be between 0 and {len(images) - 1}")
    end = min(len(images), args.start + args.limit) if args.limit is not None else len(images)
    selected = images[args.start:end]
    if not selected:
        raise ValueError("selected range contains no images")
    image_root = Path(manifest["image_dir"])
    missing = []
    size_changed = []
    for item in selected:
        path = image_root / item["filename"]
        if not path.is_file():
            missing.append(item["filename"])
        elif item.get("source_size_bytes") is not None and path.stat().st_size != item["source_size_bytes"]:
            size_changed.append(item["filename"])
    if missing or size_changed:
        raise ValueError(
            "source no longer matches frozen manifest: "
            f"missing={missing[:20]}, size_changed={size_changed[:20]}"
        )
    estimate = call_estimate(len(selected))
    plan = {
        "run_root": str(run_root),
        "input_dir": manifest["image_dir"],
        "manifest_images": len(images),
        "processing_order": processing_order,
        "selected_start": args.start,
        "selected_end_exclusive": end,
        "selected_images": len(selected),
        "selected_decade_counts": decade_counts(selected),
        "page_batch_size": args.batch_pages,
        "page_workers": args.page_workers,
        "entity_workers": args.entity_workers,
        "crop_cache": args.crop_cache,
        "recovery_passes": args.recovery_passes,
        "target_tokens_per_minute": getattr(args, "target_tokens_per_minute", None),
        "pacing_backoff_half_life_seconds": PACING_BACKOFF_HALF_LIFE_SECONDS,
        "pacing_backoff_quiet_grace_seconds": PACING_BACKOFF_QUIET_GRACE_SECONDS,
        "scheduling": "adjacent_batch_mixed_structure_entity",
        "estimate": estimate,
    }
    print(json.dumps({"production_plan": plan}, indent=2), flush=True)
    if args.dry_run:
        atomic_write_json(run_root / "dry_run_plan.json", plan)
        return 0
    if args.request_budget <= 0:
        raise ValueError("--request-budget must be positive")
    if not os.environ.get("OPENROUTER_API_KEY") and not KEY.exists():
        raise FileNotFoundError(f"set OPENROUTER_API_KEY or create {KEY}")
    existing_attempts = sum(1 for _ in iter_jsonl(LEDGER))
    budget = Budget(
        existing_attempts + args.request_budget,
        LEDGER,
        target_tokens_per_minute=getattr(args, "target_tokens_per_minute", None),
    )
    args._manifest = manifest
    args.cohort = "pages"
    args.run_id = f"standalone_production::{safe_name(args.run_name)}"
    args.pilot_only = False
    stopped_reason = None
    try:
        batches = [
            selected[relative_start : relative_start + args.batch_pages]
            for relative_start in range(0, len(selected), args.batch_pages)
        ]

        def write_checkpoint(pass_name: str, batch_index: int, phase: str) -> dict[str, Any]:
            batch = batches[batch_index]
            absolute_start = args.start + batch_index * args.batch_pages
            checkpoint = {
                "updated_at": now(),
                "pass": pass_name,
                "phase": phase,
                "processing_order_id": processing_order["order_id"],
                "processing_order_sha256": processing_order["order_sha256"],
                "batch_start": absolute_start,
                "batch_end_exclusive": absolute_start + len(batch),
                "request_attempts_before_batch": budget.used,
            }
            atomic_write_json(run_root / "checkpoint.json", checkpoint)
            return checkpoint

        for pass_index in range(args.recovery_passes + 1):
            pass_name = "initial" if pass_index == 0 else f"recovery_{pass_index}"
            first = batches[0]
            first_start = args.start
            args._images = first
            checkpoint = write_checkpoint(pass_name, 0, "structure")
            print(f"\n=== {pass_name}: pages {first_start}:{first_start + len(first)} structure ===", flush=True)
            args.workers = args.page_workers
            args.max_in_flight = args.page_max_in_flight or args.page_workers * 2
            structure_stage(args, budget)

            for batch_index in range(1, len(batches)):
                previous = batches[batch_index - 1]
                current = batches[batch_index]
                previous_start = args.start + (batch_index - 1) * args.batch_pages
                current_start = args.start + batch_index * args.batch_pages

                args._images = previous
                entity_job_list, entity_result_path = entity_jobs(args)
                args._images = current
                structure_job_list, structure_result_path = structure_jobs(args)
                checkpoint = write_checkpoint(pass_name, batch_index, "mixed")
                print(
                    f"\n=== {pass_name}: mixed entities {previous_start}:{previous_start + len(previous)} "
                    f"+ structure {current_start}:{current_start + len(current)} ===",
                    flush=True,
                )
                execute_mixed_jobs([
                    {
                        "name": "entities",
                        "jobs": entity_job_list,
                        "result_path": entity_result_path,
                        "workers": args.entity_workers,
                        "max_in_flight": args.entity_max_in_flight or args.entity_workers * 2,
                    },
                    {
                        "name": "structure",
                        "jobs": structure_job_list,
                        "result_path": structure_result_path,
                        "workers": args.page_workers,
                        "max_in_flight": args.page_max_in_flight or args.page_workers * 2,
                    },
                ], budget)
                checkpoint.update({"completed_at": now(), "request_attempts_after_batch": budget.used})
                atomic_write_json(run_root / "checkpoint.json", checkpoint)

            last = batches[-1]
            last_start = args.start + (len(batches) - 1) * args.batch_pages
            args._images = last
            checkpoint = write_checkpoint(pass_name, len(batches) - 1, "entities")
            print(f"\n=== {pass_name}: pages {last_start}:{last_start + len(last)} entities ===", flush=True)
            args.workers = args.entity_workers
            args.max_in_flight = args.entity_max_in_flight or args.entity_workers * 2
            entity_stage(args, budget)
            checkpoint.update({"completed_at": now(), "request_attempts_after_batch": budget.used})
            atomic_write_json(run_root / "checkpoint.json", checkpoint)
    except BudgetExhausted as exc:
        stopped_reason = str(exc)
        print(f"\nREQUEST BUDGET EXHAUSTED: {stopped_reason}; assembling and auditing resumable partial output.", flush=True)
    assemble_stage(args)
    report = production_audit(args, selected, run_root, budget, stopped_reason)
    print(json.dumps({"production_status": report}, indent=2), flush=True)
    return 0 if report["all_selected_complete"] and stopped_reason is None else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ["structure", "entities", "assemble", "duplicates"]:
        child = sub.add_parser(name)
        child.add_argument("--cohort", choices=["difficult", "stratified"], required=True)
        child.add_argument("--variant", choices=["s1", "s2", "s2n"], default="s2")
        child.add_argument("--max-requests", type=int, default=3000)
        child.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
        child.add_argument("--pilot-only", action="store_true")
        child.add_argument("--start", type=int, default=0)
        child.add_argument("--limit", type=int)
        if name in {"entities", "assemble"}:
            child.add_argument("--style", choices=["direct", "ordinal"], default="direct")
            child.add_argument("--gaze", choices=["yes", "no"], default="yes")
        if name == "entities":
            child.add_argument("--crop-cache", choices=["disk", "none"], default="disk")
            child.add_argument("--max-in-flight", type=int)
        if name == "structure":
            child.add_argument("--max-in-flight", type=int)
    production = sub.add_parser("run-all", help="self-supervising, resumable directory-scale production run")
    production.add_argument("--input-dir", type=Path, default=DEFAULT_FULL_PAGE_DIR)
    production.add_argument("--run-name", default="full_pages_joined_v1")
    production.add_argument(
        "--processing-order",
        type=Path,
        help="optional complete image-id permutation; changes scheduling only, not the frozen source manifest",
    )
    production.add_argument("--start", type=int, default=0)
    production.add_argument("--limit", type=int)
    production.add_argument("--batch-pages", type=int, default=500)
    production.add_argument("--page-workers", type=int, default=24)
    production.add_argument("--entity-workers", type=int, default=32)
    production.add_argument("--page-max-in-flight", type=int)
    production.add_argument("--entity-max-in-flight", type=int)
    production.add_argument("--request-budget", type=int, default=0, help="maximum additional physical attempts allowed in this invocation")
    production.add_argument(
        "--target-tokens-per-minute",
        type=int,
        default=DEFAULT_TARGET_TOKENS_PER_MINUTE,
        help="shared proactive Venice pacing target across page and entity stages; use 0 to disable",
    )
    production.add_argument("--recovery-passes", type=int, default=1)
    production.add_argument("--variant", choices=["s1", "s2", "s2n"], default="s2")
    production.add_argument("--style", choices=["direct", "ordinal"], default="direct")
    production.add_argument("--gaze", choices=["yes", "no"], default="yes")
    production.add_argument("--crop-cache", choices=["none", "disk"], default="none")
    production.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.stage == "run-all":
        if args.batch_pages <= 0 or args.page_workers <= 0 or args.entity_workers <= 0 or args.recovery_passes < 0 or args.target_tokens_per_minute < 0:
            parser.error("batch size/workers must be positive; recovery passes and token target cannot be negative")
        return run_all_stage(args)
    if args.stage == "assemble":
        assemble_stage(args)
        return 0
    budget = Budget(args.max_requests)
    if args.stage == "structure":
        structure_stage(args, budget)
    elif args.stage == "duplicates":
        duplicate_stage(args, budget)
    else:
        entity_stage(args, budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

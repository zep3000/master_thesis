"""Gold-blind utilities for the canonical 280-page candidate package.

Inference-side modules in this directory must not import an evaluator or read a
human annotation export.  Evaluation is deliberately isolated in evaluate.py.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "canonical_candidate_280"
LEAN = ROOT / "qwen_iteration" / "lean_final_280"
REFINEMENT = ROOT / "qwen_iteration" / "final_refinement_280"
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
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in values), encoding="utf-8")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def manifest(cohort: str) -> dict[str, Any]:
    return json.loads((LEAN / "data" / f"manifest_{cohort}.json").read_text(encoding="utf-8"))


def baseline_rows(cohort: str) -> list[dict[str, Any]]:
    path = REFINEMENT / "output" / "assembled" / cohort / "current_direct_ungated.jsonl"
    return [row for row in rows(path) if row.get("ok")]


def image_path(cohort: str, filename: str) -> Path:
    return Path(manifest(cohort)["image_dir"]) / filename


def image_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def jpeg_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=92, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def to_pixels(box: list[int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    return tuple(round(v * (width if i % 2 == 0 else height) / 1000) for i, v in enumerate(box))  # type: ignore[return-value]


def expand_box(box: tuple[int, int, int, int], size: tuple[int, int], scale: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * scale
    width, height = size
    return max(0, round(cx - side / 2)), max(0, round(cy - side / 2)), min(width, round(cx + side / 2)), min(height, round(cy + side / 2))


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    return value if isinstance(value, dict) else {}


class Budget:
    def __init__(self, maximum: int):
        self.maximum = maximum
        self.used = len(rows(LEDGER))
        self.lock = threading.Lock()

    def reserve(self) -> int:
        with self.lock:
            if self.used >= self.maximum:
                raise RuntimeError(f"OpenRouter attempt ceiling reached: {self.used}/{self.maximum}")
            self.used += 1
            return self.used


def make_client() -> Any:
    import httpx
    from openai import OpenAI

    return OpenAI(
        api_key=KEY.read_text(encoding="utf-8").strip(),
        base_url="https://openrouter.ai/api/v1",
        timeout=300,
        max_retries=0,
        http_client=httpx.Client(timeout=300, trust_env=False),
        default_headers={"X-Title": "Qwen Canonical Candidate 280"},
    )


def one_request(api: Any, budget: Budget, job: dict[str, Any]) -> dict[str, Any]:
    last_error = ""
    for attempt in range(1, 4):
        sequence = budget.reserve()
        started = time.time()
        usage = None
        try:
            response = api.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": job["content"]}],
                max_tokens=job["max_tokens"],
                temperature=0,
                response_format={"type": "json_object"},
                extra_body={
                    "provider": {"order": [PROVIDER], "allow_fallbacks": False},
                    "reasoning": {"effort": "none", "exclude": True},
                },
            )
            metadata = plain(response)
            usage = metadata.get("usage")
            raw = extract_json(str(response.choices[0].message.content))
            normalized = job["normalize"](raw)
            ok, last_error = True, ""
        except Exception as exc:
            raw = normalized = None
            ok, last_error = False, f"{type(exc).__name__}: {exc}"
        append_jsonl(
            LEDGER,
            {
                "request_sequence": sequence,
                **job["meta"],
                "task_key": job["task_key"],
                "attempt": attempt,
                "ok": ok,
                "usage": usage,
                "error": None if ok else last_error,
                "elapsed_seconds": round(time.time() - started, 3),
                "finished_at": now(),
            },
        )
        if ok:
            return {
                "ok": True,
                "task_key": job["task_key"],
                "model_annotation_raw": raw,
                "model_annotation": normalized,
                "usage": usage,
                **job.get("result_meta", {}),
            }
        if attempt < 3:
            time.sleep(32 if "429" in last_error else 3)
    return {"ok": False, "task_key": job["task_key"], "error": last_error, **job.get("result_meta", {})}


def execute(jobs: list[dict[str, Any]], path: Path, maximum: int, workers: int) -> None:
    completed = {row["task_key"] for row in rows(path) if row.get("ok")}
    pending = [job for job in jobs if job["task_key"] not in completed]
    print(json.dumps({"output": str(path), "jobs": len(jobs), "pending": len(pending), "ledger_attempts": len(rows(LEDGER)), "ceiling": maximum}))
    if not pending:
        return
    local = threading.local()

    def run(job: dict[str, Any]) -> dict[str, Any]:
        if not hasattr(local, "api"):
            local.api = make_client()
        return one_request(local.api, BudgetProxy(budget), job)

    budget = Budget(maximum)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, job) for job in pending]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            append_jsonl(path, row)
            print(f"[{index}/{len(pending)}] {row['task_key']} {'ok' if row.get('ok') else 'FAILED'}", flush=True)


class BudgetProxy:
    """Share one Budget instance without exposing any other runner state."""

    def __init__(self, budget: Budget):
        self.budget = budget

    def reserve(self) -> int:
        return self.budget.reserve()

#!/usr/bin/env python3
r"""Compare three one-call confidence prompts on the fixed 50-ad audit.

This experiment is deliberately isolated from the annotation pipeline.  It
uses only source pixels plus a published taxonomy; no human gold annotations
are available to inference.  All variants return the same four fields and use
the same model, image crop, decoding settings, category definitions, and token
limit.  The only manipulated text is the confidence instruction.

Usage:
    .\.venv\Scripts\python.exe run_confidence_prompt_pilot.py --workers 3

The runner is resumable by (variant, image_id, ad_id), records every API
attempt, and refuses to exceed its local request-attempt budget.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "category_taxonomy_reuse"
MANIFEST = HERE / "audit_50" / "audit_manifest.json"
OUTPUT = HERE / "confidence_pilot"
RESULTS = OUTPUT / "results.jsonl"
LEDGER = OUTPUT / "request_ledger.jsonl"
KEY = ROOT / "openrouter_key.txt"
MODEL = "qwen/qwen3.5-9b"
PROVIDER = "venice/fp8"

CATEGORIES = [
    "Alcoholic drinks", "Automotive", "Business & industrial", "Clothing & accessories",
    "Financial services", "Food", "Household & domestic", "Leisure & entertainment",
    "Media & publishing", "Non-profit, public sector & education", "Pharma & healthcare",
    "Politics", "Retail", "Soft drinks", "Technology & electronics", "Telecoms & utilities",
    "Tobacco", "Toiletries & cosmetics", "Transport & tourism",
]

COMMON = """Return one JSON object only for this single advertisement crop.
Choose exactly one category from this WARC/Nielsen 19-category list:
{categories}

Classify the primary advertised product, service, event, cause, or organization—not incidental imagery. For a recruitment or pure corporate-image ad with no direct product category, use the advertiser's main sector. Business & industrial includes B2B services, manufacturing, industrial materials, construction and commercial equipment. Non-profit, public sector & education includes charities, government/public information and education. Financial services includes banking, investment and insurance. Transport & tourism includes airlines, hotels, tourism and postal/logistics services. Telecoms & utilities is separate from Technology & electronics. Media & publishing includes books, periodicals, news and paid research publications; Leisure & entertainment includes film and entertainment events. Retail means a general retailer rather than the specific product category of an item it sells.

`brand_or_advertiser` may be a product brand, company, sponsor, public body, or other visible/inferable advertiser; this deliberately does not distinguish those roles. Use null if no name is adequately supported. Return no explanation and no extra fields.
"""

VARIANT_TEXT = {
    "unanchored": """Return ad_category, ad_category_confidence (any number 0..1), brand_or_advertiser, and brand_confidence (0..1; 0 if brand is null).""",
    "anchored": """Confidence is your probability that the returned value is correct from this crop, not a general impression of image quality. Use the full range: 0.99 only when directly printed/depicted and unequivocal; 0.90 for very strong evidence; 0.75 when it is the best answer but a plausible alternative or incomplete text remains; 0.60 for weak evidence; 0.40 or lower for a guess. Return ad_category, ad_category_confidence, brand_or_advertiser, and brand_confidence (0 if brand is null).""",
    "anchored_check": """Silently do this before output: identify the decisive visible wording/product; identify the strongest alternative category/name; lower confidence when that alternative is materially plausible. Confidence is the probability that the returned value is correct from this crop, not image quality. Use the full range: 0.99 only for direct unequivocal evidence; 0.90 for very strong evidence; 0.75 when a plausible alternative or incomplete text remains; 0.60 for weak evidence; 0.40 or lower for a guess. A broad fallback for an offering missing from the taxonomy must not exceed 0.60. Return only ad_category, ad_category_confidence, brand_or_advertiser, and brand_confidence (0 if brand is null).""",
    "decision_rubric": """Use only these confidence values: 0.95, 0.80, 0.65, 0.50. For category confidence: 0.95 only when the advertised offering is explicit and exactly one taxonomy category fits; 0.80 when the best category is clear but some inference is needed; 0.65 when two categories are genuinely plausible or this is a corporate-image/recruitment fallback; 0.50 when the offering or category is a guess. For brand confidence: 0.95 when the returned name is visibly printed and unambiguous; 0.80 when the organization is clear but exact wording/expansion is partly inferred; 0.65 when multiple visible organizations could count as advertiser; 0.50 for weak identification. Use 0 when brand is null. Return only ad_category, ad_category_confidence, brand_or_advertiser, and brand_confidence.""",
}

SCHEMA = {
    "ad_category": CATEGORIES,
    "ad_category_confidence": "number 0..1",
    "brand_or_advertiser": "string or null",
    "brand_confidence": "number 0..1; 0 if null",
}

LOCK = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = json.loads(text[text.find("{") : text.rfind("}") + 1])
    if not isinstance(value, dict):
        raise ValueError("response is not an object")
    return value


def normalize(raw: dict[str, Any]) -> dict[str, Any]:
    category = raw.get("ad_category")
    if category not in CATEGORIES:
        raise ValueError(f"invalid category: {category!r}")
    brand = raw.get("brand_or_advertiser")
    if not isinstance(brand, str) or not brand.strip():
        brand = None
    def conf(value: Any) -> float:
        return max(0.0, min(1.0, float(value)))
    return {
        "ad_category": category,
        "ad_category_confidence": conf(raw.get("ad_category_confidence")),
        "brand_or_advertiser": brand.strip() if brand else None,
        "brand_confidence": conf(raw.get("brand_confidence")) if brand else 0.0,
    }


def crop_data_url(row: dict[str, Any]) -> str:
    page = Image.open(row["image_path"]).convert("RGB")
    width, height = page.size
    box = row["bbox_1000"]
    pixels = tuple(round(float(v) * (width if i % 2 == 0 else height) / 1000) for i, v in enumerate(box))
    crop = page.crop(pixels)
    crop.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    crop.save(buffer, "JPEG", quality=92, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def prompt(variant: str) -> str:
    common = COMMON.format(categories=json.dumps(CATEGORIES, ensure_ascii=False))
    return common + "\n" + VARIANT_TEXT[variant] + "\nSchema: " + json.dumps(SCHEMA, ensure_ascii=False)


class Budget:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = len(jsonl(LEDGER))
        self.sequence = max([int(r.get("request_sequence", 0)) for r in jsonl(LEDGER)] or [0]) + 1

    def reserve(self) -> int:
        with LOCK:
            if self.used >= self.limit:
                raise RuntimeError(f"local attempt budget exhausted: {self.used}/{self.limit}")
            self.used += 1
            value = self.sequence
            self.sequence += 1
            return value


def client():
    import httpx
    from openai import OpenAI
    return OpenAI(
        api_key=KEY.read_text(encoding="utf-8").strip(),
        base_url="https://openrouter.ai/api/v1",
        timeout=240,
        max_retries=0,
        http_client=httpx.Client(timeout=240, trust_env=False),
        default_headers={"X-Title": "WARC taxonomy confidence prompt pilot"},
    )


def call_one(api: Any, budget: Budget, row: dict[str, Any], variant: str) -> dict[str, Any]:
    task_key = f"{variant}::{row['image_id']}::{row['ad_id']}"
    content = [
        {"type": "text", "text": prompt(variant)},
        {"type": "image_url", "image_url": {"url": crop_data_url(row)}},
    ]
    last_error = ""
    for attempt in range(1, 3):
        sequence = budget.reserve()
        started = time.time()
        usage = None
        try:
            response = api.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": content}],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"},
                extra_body={"provider": {"order": [PROVIDER], "allow_fallbacks": False}, "reasoning": {"effort": "none", "exclude": True}},
            )
            usage_obj = response.usage
            usage = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else str(usage_obj)
            result = normalize(extract_json(str(response.choices[0].message.content)))
            ok = True
        except Exception as exc:
            result = None
            ok = False
            last_error = f"{type(exc).__name__}: {exc}"
        append(LEDGER, {
            "request_sequence": sequence,
            "task_key": task_key,
            "variant": variant,
            "image_id": row["image_id"],
            "ad_id": row["ad_id"],
            "model": MODEL,
            "provider": PROVIDER,
            "attempt": attempt,
            "ok": ok,
            "usage": usage,
            "error": None if ok else last_error,
            "elapsed_seconds": round(time.time() - started, 3),
            "finished_at": now(),
        })
        if ok:
            return {"task_key": task_key, "variant": variant, "image_id": row["image_id"], "ad_id": row["ad_id"], "result": result, "usage": usage, "ok": True}
        if attempt < 2:
            time.sleep(3)
    return {"task_key": task_key, "variant": variant, "image_id": row["image_id"], "ad_id": row["ad_id"], "ok": False, "error": last_error}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=300)
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANT_TEXT), default=list(VARIANT_TEXT))
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    completed = {r["task_key"] for r in jsonl(RESULTS) if r.get("ok")}
    jobs = [(row, variant) for variant in args.variants for row in manifest["ads"] if f"{variant}::{row['image_id']}::{row['ad_id']}" not in completed]
    budget = Budget(args.max_attempts)
    print(json.dumps({"pending": len(jobs), "completed": len(completed), "attempts_used": budget.used, "attempt_limit": budget.limit}, indent=2))
    if not jobs:
        return 0
    local = threading.local()
    def run(job: tuple[dict[str, Any], str]) -> dict[str, Any]:
        if not hasattr(local, "api"):
            local.api = client()
        return call_one(local.api, budget, job[0], job[1])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            append(RESULTS, row)
            print(f"[{index}/{len(jobs)}] {row['task_key']} {'ok' if row.get('ok') else 'FAILED'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

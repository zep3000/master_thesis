#!/usr/bin/env python3
"""Gold-isolated, resumable runner for expression-legibility strategies."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from strategies import (
    ALL_STRATEGIES, DISCOVERY_STRATEGIES, LOGIC_VERSION, PROMPT_VERSION, STRATEGIES, normalize,
    threshold_prompt, verifier_trigger,
)


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "expression_legibility_research"
MANIFEST = HERE / "data" / "cases_manifest.json"
PRIOR_OUTPUTS = HERE / "data" / "frozen_p2_outputs.json"
OUTPUT = HERE / "output"
API_KEY_PATH = ROOT / "openrouter_key.txt"
BASE_URL = "https://openrouter.ai/api/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument("--provider-tag", default="")
    parser.add_argument("--strategies", nargs="+", choices=ALL_STRATEGIES, default=DISCOVERY_STRATEGIES)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-requests", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--omit-reasoning", action="store_true", help="Do not send OpenRouter's reasoning control to models/providers that do not advertise it.")
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def append_jsonl(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_results(path: Path) -> dict[str, dict[str, Any]]:
    latest = {}
    for row in read_jsonl(path):
        latest[str(row["logical_key"])] = row
    return latest


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


def response_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(item.get("text", "")) for item in value if isinstance(item, dict))
    return "" if value is None else str(value)


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
        raise ValueError("no JSON object")
    depth = 0; quoted = False; escaped = False
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
                parsed = json.loads(cleaned[start:index + 1])
                if isinstance(parsed, dict):
                    return parsed
    raise ValueError("unterminated JSON")


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


class Budget:
    def __init__(self, ledger: Path, limit: int):
        self.ledger = ledger
        self.limit = limit
        self.lock = threading.Lock()
        existing = read_jsonl(ledger)
        self.used = len(existing)
        self.next_sequence = max((int(row.get("request_sequence", 0)) for row in existing), default=0) + 1

    def reserve(self) -> int:
        with self.lock:
            if self.used >= self.limit:
                raise RuntimeError(f"Run request ceiling exhausted: {self.used}/{self.limit}")
            value = self.next_sequence
            self.next_sequence += 1
            self.used += 1
            return value

    def record(self, payload: dict[str, Any]) -> None:
        append_jsonl(self.ledger, payload, self.lock)


def load_key() -> str:
    value = API_KEY_PATH.read_text(encoding="utf-8").strip() if API_KEY_PATH.exists() else ""
    if not value:
        raise SystemExit("OpenRouter key missing")
    return value


def make_client(args: argparse.Namespace) -> Any:
    import httpx
    from openai import OpenAI
    return OpenAI(
        api_key=load_key(), base_url=BASE_URL, timeout=args.timeout, max_retries=0,
        http_client=httpx.Client(timeout=args.timeout, trust_env=False),
        default_headers={"X-Title": "Expression Legibility Research"},
    )


def request_kwargs(args: argparse.Namespace, prompt: str, image_path: Path, max_tokens: int) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if not args.omit_reasoning:
        extra["reasoning"] = {"effort": "none", "exclude": True}
    if args.provider_tag:
        extra["provider"] = {"order": [args.provider_tag], "allow_fallbacks": False}
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url(image_path)}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "response_format": {"type": "json_object"},
        "extra_body": extra,
    }


def call_with_retries(
    client: Any, budget: Budget, args: argparse.Namespace, run_dir: Path,
    logical_key: str, strategy: str, stage: str, case_id: str,
    prompt: str, image_path: Path, max_tokens: int,
) -> dict[str, Any]:
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    last_error = ""
    for local_attempt in range(1, 4):
        sequence = budget.reserve()
        started = time.time()
        ledger = {
            "request_sequence": sequence, "run_name": args.run_name, "model": args.model,
            "provider_tag": args.provider_tag or None, "logical_key": logical_key,
            "strategy": strategy, "stage": stage, "case_id": case_id,
            "attempt": local_attempt, "started_at": iso_now(), "prompt_version": PROMPT_VERSION,
            "logic_version": LOGIC_VERSION, "prompt_sha256": prompt_hash, "image_path": str(image_path),
        }
        usage = None; raw = ""; metadata: dict[str, Any] = {}
        try:
            response = client.chat.completions.create(**request_kwargs(args, prompt, image_path, max_tokens))
            metadata = plain(response)
            usage = metadata.get("usage")
            raw = response_text(response.choices[0].message.content)
            parsed = extract_json(raw)
            normalized, actions = normalize(strategy, parsed, case_id)
            if normalized["case_id"] != case_id:
                raise ValueError("case ID normalization failed")
            ok = True; last_error = ""
        except Exception as exc:
            ok = False; parsed = None; normalized = None; actions = []
            last_error = f"{type(exc).__name__}: {exc}"
        elapsed = round(time.time() - started, 3)
        raw_dir = run_dir / "raw" / safe(args.model) / strategy / stage
        raw_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{safe(case_id)}_a{local_attempt}_r{sequence}"
        if raw:
            (raw_dir / f"{stem}.txt").write_text(raw, encoding="utf-8")
        if metadata:
            (raw_dir / f"{stem}.response.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        ledger.update({"ok": ok, "usage": usage, "error": None if ok else last_error, "elapsed_seconds": elapsed, "finished_at": iso_now()})
        budget.record(ledger)
        if ok:
            return {
                "logical_key": logical_key, "run_name": args.run_name, "model": args.model,
                "strategy": strategy, "stage": stage, "case_id": case_id, "ok": True,
                "model_annotation_raw": parsed, "model_annotation": normalized,
                "normalization_actions": actions, "usage": usage, "physical_request_sequence": sequence,
                "prompt_sha256": prompt_hash, "image_path": str(image_path),
            }
        if local_attempt < 3:
            time.sleep(2 ** local_attempt)
    return {
        "logical_key": logical_key, "run_name": args.run_name, "model": args.model,
        "strategy": strategy, "stage": stage, "case_id": case_id, "ok": False,
        "error": last_error, "prompt_sha256": prompt_hash, "image_path": str(image_path),
    }


def run_jobs(client: Any, budget: Budget, args: argparse.Namespace, run_dir: Path, results_path: Path, jobs: list[dict[str, Any]]) -> None:
    latest = latest_results(results_path)
    pending = [job for job in jobs if job["logical_key"] not in latest or (args.retry_failures and latest[job["logical_key"]].get("ok") is not True)]
    print(f"jobs={len(jobs)} pending={len(pending)} requests={budget.used}/{budget.limit}")
    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(call_with_retries, client, budget, args, run_dir, **job): job for job in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            append_jsonl(results_path, record, write_lock)
            print(f"[{index}/{len(pending)}] {record['logical_key']} ok={record['ok']} requests={budget.used}/{budget.limit}")


def main() -> None:
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    if args.pilot:
        pilot = set(manifest["pilot_case_ids"])
        cases = [case for case in cases if case["case_id"] in pilot]
    run_dir = OUTPUT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps({
        "run_name": args.run_name, "model": args.model, "provider_tag": args.provider_tag or None,
        "strategies": args.strategies, "pilot": args.pilot, "prompt_version": PROMPT_VERSION,
        "logic_version": LOGIC_VERSION, "case_count": len(cases), "omit_reasoning": args.omit_reasoning,
    }, indent=2) + "\n", encoding="utf-8")
    results_path = run_dir / "results.jsonl"
    budget = Budget(run_dir / "request_ledger.jsonl", args.max_requests)
    client = make_client(args)

    base_strategies = [strategy for strategy in args.strategies if strategy in STRATEGIES]
    jobs = []
    for strategy in base_strategies:
        spec = STRATEGIES[strategy]
        for case in cases:
            case_id = case["case_id"]
            jobs.append({
                "logical_key": f"{args.model}::{strategy}::primary::{case_id}",
                "strategy": strategy, "stage": "primary", "case_id": case_id,
                "prompt": spec["prompt"](case_id), "image_path": Path(case[spec["image_field"]]),
                "max_tokens": int(spec["max_tokens"]),
            })
    run_jobs(client, budget, args, run_dir, results_path, jobs)

    if "p2_zero_verify" in args.strategies:
        prior_outputs = json.loads(PRIOR_OUTPUTS.read_text(encoding="utf-8"))["cases"]
        latest = latest_results(results_path)
        verifier_jobs = []
        for case in cases:
            case_id = case["case_id"]
            primary = prior_outputs[case_id]
            trigger, reasons = verifier_trigger(primary["model_annotation"])
            decision_key = f"{args.model}::p2_zero_verify::decision::{case_id}"
            if not trigger:
                if decision_key not in latest:
                    append_jsonl(results_path, {
                        "logical_key": decision_key, "run_name": args.run_name, "model": args.model,
                        "strategy": "p2_zero_verify", "stage": "decision", "case_id": case_id,
                        "ok": True, "verifier_triggered": False, "trigger_reasons": [],
                        "model_annotation": primary["model_annotation"], "primary_source_task_key": primary["source_task_key"],
                    }, threading.Lock())
                continue
            verifier_jobs.append({
                "logical_key": f"{args.model}::p2_zero_verify::verifier::{case_id}",
                "strategy": "p2_zero_verify", "stage": "verifier", "case_id": case_id,
                "prompt": threshold_prompt(case_id, verifier=True), "image_path": Path(case["face_only_path"]),
                "max_tokens": 750,
            })
        run_jobs(client, budget, args, run_dir, results_path, verifier_jobs)
        latest = latest_results(results_path)
        for case in cases:
            case_id = case["case_id"]
            decision_key = f"{args.model}::p2_zero_verify::decision::{case_id}"
            if decision_key in latest:
                continue
            primary = prior_outputs[case_id]
            verifier_key = f"{args.model}::p2_zero_verify::verifier::{case_id}"
            verifier = latest.get(verifier_key)
            trigger, reasons = verifier_trigger(primary["model_annotation"])
            chosen = verifier["model_annotation"] if verifier and verifier.get("ok") is True else primary["model_annotation"]
            append_jsonl(results_path, {
                "logical_key": decision_key, "run_name": args.run_name, "model": args.model,
                "strategy": "p2_zero_verify", "stage": "decision", "case_id": case_id,
                "ok": True, "verifier_triggered": trigger, "trigger_reasons": reasons,
                "verifier_succeeded": bool(verifier and verifier.get("ok") is True),
                "model_annotation": chosen, "primary_source_task_key": primary["source_task_key"],
                "verifier_logical_key": verifier_key if trigger else None,
            }, threading.Lock())
    print(f"Complete: {run_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run one pinned OpenRouter provider on a frozen 50-page comparison sample.

This wrapper leaves the frozen production runner unchanged.  It redirects all
artifacts to a provider-specific experiment directory, pins fallback-free
routing to the requested endpoint, and records the literal assistant response
text in addition to the parsed/normalized annotations written by ``run.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run


RUN_ID = "standalone_provider_comparison_first50_v1"
SAMPLE_COHORT = "sample50"
SAMPLE_SIZE = 50
OUTPUT_ROOT = run.BASE_OUTPUT / "experiments" / "provider_comparison_first50"
SAMPLE_MANIFEST = OUTPUT_ROOT / "sample_manifest.json"
GOLD_EXPORT = run.ROOT / "annotation_results" / "test_collection_200_difficult_joined_v1_2026-08-02.json"
PROVIDERS = ("deepinfra/bf16", "parasail/bf16", "venice/fp8")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def provider_slug(provider: str) -> str:
    return provider.replace("/", "_").replace(".", "_")


def prepare_sample() -> dict[str, Any]:
    source = json.loads((run.HERE / "data" / "manifest_difficult.json").read_text(encoding="utf-8"))
    gold = json.loads(GOLD_EXPORT.read_text(encoding="utf-8"))
    certified = {
        str(row["image_id"]): row
        for row in gold.get("annotations") or []
        if row.get("assignment_code") == "79201188"
        and row.get("assignment_status") == "done"
        and row.get("status") in {"complete", "ineligible"}
        and isinstance(row.get("payload"), dict)
    }
    images = source["images"][:SAMPLE_SIZE]
    missing = [row["image_id"] for row in images if row["image_id"] not in certified]
    if missing:
        raise ValueError(f"sample contains images without completed human certification: {missing}")
    manifest = {
        "schema_version": "qwen_provider_comparison_manifest_v1",
        "cohort": SAMPLE_COHORT,
        "selection": "first 50 images in the frozen difficult manifest; all have completed assignment 79201188",
        "source_manifest": str(run.HERE / "data" / "manifest_difficult.json"),
        "gold_export": str(GOLD_EXPORT),
        "image_dir": source["image_dir"],
        "images": images,
    }
    if SAMPLE_MANIFEST.exists():
        frozen = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))
        if frozen != manifest:
            raise ValueError("existing provider-comparison sample differs from the requested frozen sample")
    else:
        run.atomic_write_json(SAMPLE_MANIFEST, manifest)
    return manifest


class CapturingOpenRouterClient(run.OpenRouterClient):
    """Production client plus last-response metadata for exact text auditing."""

    def __init__(self) -> None:
        super().__init__()
        self.last_response_text: str | None = None
        self.last_response_provider: str | None = None
        self.last_response_model: str | None = None
        self.last_finish_reason: str | None = None

    def create(self, content: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        self.last_response_text = None
        self.last_response_provider = None
        self.last_response_model = None
        self.last_finish_reason = None
        result = super().create(content, max_tokens)
        choices = result.get("choices") if isinstance(result, dict) else None
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict) and message.get("content") is not None:
                self.last_response_text = str(message["content"])
            if isinstance(choices[0], dict) and choices[0].get("finish_reason") is not None:
                self.last_finish_reason = str(choices[0]["finish_reason"])
        if isinstance(result, dict):
            if result.get("provider") is not None:
                self.last_response_provider = str(result["provider"])
            if result.get("model") is not None:
                self.last_response_model = str(result["model"])
        return result


ORIGINAL_ONE_REQUEST = run.one_request


def capturing_one_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
    client = args[0] if args else kwargs["client"]
    result = ORIGINAL_ONE_REQUEST(*args, **kwargs)
    result["response_text"] = getattr(client, "last_response_text", None)
    result["response_provider"] = getattr(client, "last_response_provider", None)
    result["response_model"] = getattr(client, "last_response_model", None)
    result["finish_reason"] = getattr(client, "last_finish_reason", None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PROVIDERS)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-in-flight", type=int)
    parser.add_argument("--request-budget", type=int, default=1000, help="additional physical-attempt ceiling")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.request_budget <= 0:
        parser.error("workers and request budget must be positive")

    manifest = prepare_sample()
    if args.prepare_only:
        print(json.dumps({"sample_manifest": str(SAMPLE_MANIFEST), "images": len(manifest["images"])}, indent=2))
        return 0
    if not args.provider:
        parser.error("--provider is required unless --prepare-only is used")

    output = OUTPUT_ROOT / provider_slug(args.provider)
    config = {
        "schema_version": "qwen_provider_comparison_spec_v1",
        "run_id": RUN_ID,
        "sample_manifest": str(SAMPLE_MANIFEST),
        "sample_size": len(manifest["images"]),
        "selection": manifest["selection"],
        "model": run.MODEL,
        "provider": args.provider,
        "fallbacks": False,
        "variant": "s2",
        "style": "direct",
        "gaze": "yes",
        "temperature": 0,
        "response_format": "json_object",
        "reasoning": "none",
        "prompt_version": run.prompts.PROMPT_VERSION,
        "literal_response_text_captured": True,
    }
    config_path = output / "experiment_config.json"
    if config_path.exists():
        frozen = json.loads(config_path.read_text(encoding="utf-8"))
        if frozen != config:
            raise ValueError("existing provider experiment has an incompatible frozen configuration")
    else:
        run.atomic_write_json(config_path, config)

    run.OUTPUT = output
    run.LEDGER = output / "request_ledger.jsonl"
    run.ACTIVE_MANIFEST = SAMPLE_MANIFEST
    run.PROVIDER = args.provider
    run.make_client = CapturingOpenRouterClient
    run.one_request = capturing_one_request

    existing = sum(1 for _ in run.iter_jsonl(run.LEDGER))
    budget = run.Budget(existing + args.request_budget, run.LEDGER)
    stage_args = Namespace(
        cohort=SAMPLE_COHORT,
        variant="s2",
        style="direct",
        gaze="yes",
        workers=args.workers,
        max_in_flight=args.max_in_flight or args.workers * 2,
        pilot_only=False,
        start=0,
        limit=None,
        crop_cache="none",
        run_id=f"{RUN_ID}::{provider_slug(args.provider)}",
    )

    started_at = iso_now()
    wall_started = time.time()
    print(f"=== {args.provider}: 50-page structure stage ===", flush=True)
    structure_summary = run.structure_stage(stage_args, budget)
    print(f"=== {args.provider}: entity stage ===", flush=True)
    entity_summary = run.entity_stage(stage_args, budget)
    run.assemble_stage(stage_args)
    summary = {
        "schema_version": "qwen_provider_comparison_run_summary_v1",
        "provider": args.provider,
        "started_at": started_at,
        "finished_at": iso_now(),
        "wall_seconds": round(time.time() - wall_started, 3),
        "attempts_before_invocation": budget.starting_used,
        "attempts_this_invocation": budget.used - budget.starting_used,
        "attempts_all_invocations": budget.used,
        "structure": structure_summary,
        "entities": entity_summary,
        "output": str(output),
    }
    run.atomic_write_json(output / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

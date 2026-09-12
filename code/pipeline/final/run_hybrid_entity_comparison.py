#!/usr/bin/env python3
"""Run alternate providers on the exact entity jobs produced by Venice structure."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import run
from run_provider_comparison import (
    CapturingOpenRouterClient,
    OUTPUT_ROOT,
    RUN_ID,
    SAMPLE_COHORT,
    SAMPLE_MANIFEST,
    capturing_one_request,
    iso_now,
    prepare_sample,
    provider_slug,
)


SOURCE_PROVIDER = "venice/fp8"
SOURCE_ROOT = OUTPUT_ROOT / provider_slug(SOURCE_PROVIDER)
SOURCE_STRUCTURE = SOURCE_ROOT / SAMPLE_COHORT / "structure_s2.jsonl"
PROVIDERS = ("deepinfra/bf16", "parasail/bf16")


def source_structure_map(cohort: str, variant: str, wanted: set[str] | None = None) -> dict[str, dict[str, Any]]:
    if cohort != SAMPLE_COHORT or variant != "s2":
        raise ValueError(f"unexpected hybrid structure request: {cohort}/{variant}")
    return {
        row["task_key"]: row["model_annotation"]
        for row in run.iter_jsonl(SOURCE_STRUCTURE)
        if row.get("ok") and (wanted is None or row.get("task_key") in wanted)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=PROVIDERS)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-in-flight", type=int)
    parser.add_argument("--request-budget", type=int, default=600)
    args = parser.parse_args()
    if args.workers <= 0 or args.request_budget <= 0:
        parser.error("workers and request budget must be positive")
    prepare_sample()
    if not SOURCE_STRUCTURE.exists():
        raise FileNotFoundError(f"Venice control structure output is missing: {SOURCE_STRUCTURE}")
    source_rows = [row for row in run.iter_jsonl(SOURCE_STRUCTURE) if row.get("ok")]
    if len(source_rows) != 50:
        raise ValueError(f"expected 50 successful Venice structure pages, found {len(source_rows)}")

    output = OUTPUT_ROOT / f"hybrid_venice_structure_{provider_slug(args.provider)}"
    config = {
        "schema_version": "qwen_hybrid_entity_provider_comparison_spec_v1",
        "run_id": RUN_ID,
        "sample_manifest": str(SAMPLE_MANIFEST),
        "model": run.MODEL,
        "structure_provider": SOURCE_PROVIDER,
        "structure_result": str(SOURCE_STRUCTURE),
        "structure_result_sha256": hashlib.sha256(SOURCE_STRUCTURE.read_bytes()).hexdigest(),
        "entity_provider": args.provider,
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
            raise ValueError("existing hybrid experiment has an incompatible frozen configuration")
    else:
        run.atomic_write_json(config_path, config)

    run.OUTPUT = output
    run.LEDGER = output / "request_ledger.jsonl"
    run.ACTIVE_MANIFEST = SAMPLE_MANIFEST
    run.PROVIDER = args.provider
    run.make_client = CapturingOpenRouterClient
    run.one_request = capturing_one_request
    run.structure_map = source_structure_map

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
        run_id=f"{RUN_ID}::venice_structure::{provider_slug(args.provider)}_entities",
    )
    started_at = iso_now()
    wall_started = time.time()
    print(f"=== Venice structure + {args.provider} entities ===", flush=True)
    entity_summary = run.entity_stage(stage_args, budget)
    run.assemble_stage(stage_args)
    summary = {
        "schema_version": "qwen_hybrid_entity_provider_comparison_run_summary_v1",
        "structure_provider": SOURCE_PROVIDER,
        "entity_provider": args.provider,
        "started_at": started_at,
        "finished_at": iso_now(),
        "wall_seconds": round(time.time() - wall_started, 3),
        "attempts_before_invocation": budget.starting_used,
        "attempts_this_invocation": budget.used - budget.starting_used,
        "attempts_all_invocations": budget.used,
        "entities": entity_summary,
        "output": str(output),
    }
    run.atomic_write_json(output / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Isolated full-page-only ablation of the frozen s2 candidate.

The only model-facing change is structure variant s1: the complete page is sent
without the four overlapping detail tiles. Entity prompts/composites, model,
provider, taxonomy, normalization, retry policy, and assembly remain unchanged.
Outputs never overwrite the frozen s2 result set.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import run


RUN_ID = "standalone_ablation_no_tiles_398_v1"
OUTPUT = run.BASE_OUTPUT / "experiments" / "no_tiles_s1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", nargs="+", choices=["difficult", "stratified"], default=["difficult", "stratified"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--max-in-flight", type=int)
    parser.add_argument("--request-budget", type=int, default=1500, help="additional physical-attempt ceiling for this invocation")
    args = parser.parse_args()
    if args.workers <= 0 or args.request_budget <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("workers, request budget, and optional limit must be positive")

    run.OUTPUT = OUTPUT
    run.LEDGER = OUTPUT / "request_ledger.jsonl"
    run.ACTIVE_MANIFEST = None
    existing = sum(1 for _ in run.iter_jsonl(run.LEDGER))
    budget = run.Budget(existing + args.request_budget, run.LEDGER)
    configuration = {
        "schema_version": "standalone_ablation_spec_v1",
        "run_id": RUN_ID,
        "change_from_final": "structure full page only; omit four s2 tiles",
        "variant": "s1",
        "style": "direct",
        "gaze": "yes",
        "model": run.MODEL,
        "provider": run.PROVIDER,
        "prompt_version": run.prompts.PROMPT_VERSION,
        "workers": args.workers,
    }
    config_path = OUTPUT / "experiment_config.json"
    if config_path.exists():
        frozen = json.loads(config_path.read_text(encoding="utf-8"))
        comparable = {key: frozen.get(key) for key in configuration if key != "workers"}
        requested = {key: value for key, value in configuration.items() if key != "workers"}
        if comparable != requested:
            raise ValueError("existing ablation configuration is incompatible; use a new output directory")
    else:
        run.atomic_write_json(config_path, configuration)

    for cohort in args.cohorts:
        stage_args = Namespace(
            cohort=cohort,
            variant="s1",
            style="direct",
            gaze="yes",
            workers=args.workers,
            max_in_flight=args.max_in_flight or args.workers * 2,
            pilot_only=False,
            start=0,
            limit=args.limit,
            crop_cache="none",
            run_id=RUN_ID,
        )
        print(f"\n=== {cohort}: full-page-only structure ===", flush=True)
        run.structure_stage(stage_args, budget)
        print(f"\n=== {cohort}: unchanged entity analysis ===", flush=True)
        run.entity_stage(stage_args, budget)
        run.assemble_stage(stage_args)
    print(json.dumps({
        "output": str(OUTPUT),
        "attempts_before_invocation": budget.starting_used,
        "attempts_this_invocation": budget.used - budget.starting_used,
        "attempts_all_invocations": budget.used,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

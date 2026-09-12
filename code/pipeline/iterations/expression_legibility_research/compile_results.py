#!/usr/bin/env python3
"""Compile cross-run metrics and the auditable stronger-model cost screen."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "expression_legibility_research"
EVAL = HERE / "evaluation"
OUTPUT = HERE / "output"
MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"

RUNS = {
    "qwen9_discovery": "frozen_qwen_v1",
    "qwen9_integration": "integration_qwen_v1",
    "gemma4": "full_gemma4_31b",
    "qwen37": "full_qwen37_flash",
    "qwen35_27b": "full_qwen35_27b",
    "qwen3vl_235b": "full_qwen3vl_235b",
    "llama4": "full_llama4_maverick",
}
PILOTS = {
    "google/gemma-4-31b-it": "pilot_gemma4_31b",
    "qwen/qwen3.7-flash": "pilot_qwen37_flash",
    "qwen/qwen3.5-27b": "pilot_qwen35_27b",
    "qwen/qwen3-vl-235b-a22b-instruct": "pilot_qwen3vl_235b",
    "meta-llama/llama-4-maverick": "pilot_llama4_maverick",
}


def summary(run: str) -> dict[str, Any]:
    return json.loads((EVAL / run / "summary.json").read_text(encoding="utf-8"))


def main() -> None:
    metric_rows = []
    for label, run in RUNS.items():
        report = summary(run)
        for strategy, metrics in report["metrics"].items():
            if strategy == "frozen_p2_baseline" and label != "qwen9_discovery":
                continue
            metric_rows.append({
                "run_label": label, "run_name": run, "model": report["model"], "strategy": strategy,
                "balanced_accuracy": metrics["balanced_accuracy"], "exact_accuracy": metrics["exact_accuracy"],
                "within_one_accuracy": metrics["within_one_accuracy"], "mean_absolute_error": metrics["mean_absolute_error"],
                "mean_signed_error": metrics["mean_signed_error"], "zero_specificity": metrics["zero_specificity"],
                "nonzero_recall": metrics["nonzero_recall"], "moderate_plus_recall": metrics["moderate_plus_recall"],
                "gaze_exact": metrics["gaze_exact"], "smile_exact": metrics["smile_exact"],
                "calls_per_case": metrics["cost"]["calls_per_case"],
                "reported_cost_per_case_usd": metrics["cost"]["reported_cost_per_case_usd"],
            })
    with (EVAL / "consolidated_model_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader(); writer.writerows(metric_rows)

    import httpx
    with httpx.Client(timeout=30, trust_env=False) as client:
        catalog = client.get(MODELS_ENDPOINT).raise_for_status().json()
    models = {item["id"]: item for item in catalog["data"]}
    qwen = summary("frozen_qwen_v1")["metrics"]
    universal_ceiling = 3 * qwen["ordinal_thresholds"]["cost"]["reported_cost_per_case_usd"]
    conditional_ceiling = 3 * qwen["p2_zero_verify"]["cost"]["reported_cost_per_case_usd"]
    screen = []
    for model, pilot_run in PILOTS.items():
        item = models[model]; pilot = summary(pilot_run)
        universal = pilot["metrics"]["ordinal_thresholds"]["cost"]["reported_cost_per_case_usd"]
        conditional = pilot["metrics"]["p2_zero_verify"]["cost"]["reported_cost_per_case_usd"]
        screen.append({
            "model": model, "name": item["name"],
            "api_prompt_usd_per_million": float(item["pricing"]["prompt"]) * 1_000_000,
            "api_completion_usd_per_million": float(item["pricing"]["completion"]) * 1_000_000,
            "input_modalities": item["architecture"]["input_modalities"],
            "pilot_universal_cost_per_case_usd": universal,
            "pilot_universal_within_3x": universal <= universal_ceiling,
            "pilot_conditional_cost_per_case_usd": conditional,
            "pilot_conditional_within_3x": conditional <= conditional_ceiling,
        })

    attempts = billed_cost = 0.0
    run_accounting = []
    for ledger in sorted(OUTPUT.glob("*/request_ledger.jsonl")):
        rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
        cost = sum(float((row.get("usage") or {}).get("cost") or 0) for row in rows)
        attempts += len(rows); billed_cost += cost
        run_accounting.append({"run_name": ledger.parent.name, "physical_attempts": len(rows), "successful_attempts": sum(row.get("ok") is True for row in rows), "reported_cost_usd": cost})

    cost_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "model_catalog_source": MODELS_ENDPOINT,
        "qwen_universal_cost_per_case_usd": qwen["ordinal_thresholds"]["cost"]["reported_cost_per_case_usd"],
        "universal_3x_ceiling_usd": universal_ceiling,
        "qwen_conditional_cost_per_case_usd": qwen["p2_zero_verify"]["cost"]["reported_cost_per_case_usd"],
        "conditional_3x_ceiling_usd": conditional_ceiling,
        "model_screen": screen, "run_accounting": run_accounting,
        "total_physical_attempts": int(attempts), "total_reported_cost_usd": billed_cost,
    }
    (EVAL / "model_cost_screen.json").write_text(json.dumps(cost_report, indent=2) + "\n", encoding="utf-8")
    print(EVAL / "consolidated_model_metrics.csv")
    print(EVAL / "model_cost_screen.json")


if __name__ == "__main__":
    main()

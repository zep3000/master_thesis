#!/usr/bin/env python3
"""Audit token use and reported OpenRouter cost for the frozen first-100 run.

This script is deliberately separate from inference and evaluation.  It reads the
append-only request ledger, never reads gold annotations, and writes reproducible
cost tables.  The provider reports prompt tokens as one combined number; it does
not expose a separate image-token count, so the report does not pretend to split
text and vision tokens exactly.

Usage:
    python analyze_costs.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "output" / "shared_request_ledger.jsonl"
OUT_DIR = ROOT / "evaluation" / "first100_frozen_v1" / "cost_analysis"
RUN_NAME = "first100_frozen_v1"
PAGES = 100


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def usage_cost(usage: dict[str, Any] | None) -> float:
    if not usage:
        return 0.0
    if isinstance(usage.get("cost"), (int, float)):
        return float(usage["cost"])
    details = usage.get("cost_details") or {}
    return float(details.get("upstream_inference_cost") or 0.0)


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    billed = [row for row in rows if isinstance(row.get("usage"), dict)]
    prompt = [float(row["usage"].get("prompt_tokens") or 0) for row in billed]
    completion = [float(row["usage"].get("completion_tokens") or 0) for row in billed]
    costs = [usage_cost(row["usage"]) for row in billed]
    return {
        "attempts": len(rows),
        "ok": sum(row.get("ok") is True for row in rows),
        "failed": sum(row.get("ok") is not True for row in rows),
        "attempts_with_usage": len(billed),
        "failed_without_usage": sum(row.get("ok") is not True and not row.get("usage") for row in rows),
        "prompt_tokens": int(sum(prompt)),
        "completion_tokens": int(sum(completion)),
        "total_tokens": int(sum(prompt) + sum(completion)),
        "reported_cost_usd": sum(costs),
        "prompt_mean": mean(prompt) if prompt else 0.0,
        "prompt_p50": median(prompt) if prompt else 0.0,
        "prompt_p95": percentile(prompt, 0.95),
        "completion_mean": mean(completion) if completion else 0.0,
        "cost_mean_billed_call_usd": mean(costs) if costs else 0.0,
        "cost_p50_billed_call_usd": median(costs) if costs else 0.0,
        "cost_p95_billed_call_usd": percentile(costs, 0.95),
    }


def money(value: float, digits: int = 6) -> str:
    return f"${value:.{digits}f}"


def main() -> None:
    rows = []
    with LEDGER.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("run_name") == RUN_NAME:
                rows.append(row)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_pipeline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pipeline = str(row["pipeline"])
        stage = str(row["stage"])
        grouped[pipeline, stage].append(row)
        by_pipeline[pipeline].append(row)

    stages = {
        pipeline: {stage: summarize(group_rows) for (pipe, stage), group_rows in grouped.items() if pipe == pipeline}
        for pipeline in sorted(by_pipeline)
    }
    pipelines = {pipeline: summarize(group_rows) for pipeline, group_rows in sorted(by_pipeline.items())}
    for stats in pipelines.values():
        stats["reported_cost_per_page_usd"] = stats["reported_cost_usd"] / PAGES
        stats["reported_cost_per_1000_pages_usd"] = stats["reported_cost_usd"] / PAGES * 1000
        stats["prompt_tokens_per_page"] = stats["prompt_tokens"] / PAGES
        stats["completion_tokens_per_page"] = stats["completion_tokens"] / PAGES

    # Cost of the practical conditional design can be computed directly from the
    # observed P2 stage means.  A verifier is conservatively priced like a router;
    # an expression montage is priced like one person-attribute call.
    p2_router = stages["p2"]["router"]["cost_mean_billed_call_usd"]
    p2_person = stages["p2"]["persons"]["cost_mean_billed_call_usd"]
    p2_base = pipelines["p2"]["reported_cost_per_page_usd"]
    hybrid_scenarios = []
    for ambiguous_share, montage_share in ((0.10, 0.10), (0.20, 0.20), (0.30, 0.30)):
        estimated = p2_base + ambiguous_share * p2_router + montage_share * p2_person
        hybrid_scenarios.append({
            "ambiguous_page_share": ambiguous_share,
            "montage_page_share": montage_share,
            "estimated_cost_per_page_usd": estimated,
            "estimated_cost_per_1000_pages_usd": estimated * 1000,
        })

    report = {
        "run_name": RUN_NAME,
        "pages": PAGES,
        "accounting_note": (
            "Reported cost is the sum of usage.cost in the request ledger. Failures raised after a response "
            "but before usage was copied into the ledger are not recoverable here, so totals are observed "
            "minimums, not guaranteed invoice totals. Prompt tokens combine text and image tokens."
        ),
        "pipelines": pipelines,
        "stages": stages,
        "conditional_p2_scenarios": hybrid_scenarios,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cost_analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Frozen first-100 cost and token audit",
        "",
        "All dollar figures below are sums of the OpenRouter `usage.cost` values stored in the shared ledger.",
        "They are observed minimums: attempts that returned a response but then failed JSON parsing lost their",
        "usage object in the original runner. The provider's prompt-token field combines prompt text and image",
        "tokens; it does not expose an image-only subtotal for these calls.",
        "",
        "## Pipeline totals",
        "",
        "| Pipeline | Attempts | Successful | Usage recorded | Prompt tok/page | Completion tok/page | Reported $/page | Reported $/1k pages |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pipeline, stats in pipelines.items():
        lines.append(
            f"| {pipeline.upper()} | {stats['attempts']} | {stats['ok']} | {stats['attempts_with_usage']} | "
            f"{stats['prompt_tokens_per_page']:.0f} | {stats['completion_tokens_per_page']:.0f} | "
            f"{money(stats['reported_cost_per_page_usd'])} | {money(stats['reported_cost_per_1000_pages_usd'], 3)} |"
        )

    lines += [
        "",
        "## Stage distributions",
        "",
        "| Pipeline/stage | Calls with usage | Prompt mean | Prompt p50 | Prompt p95 | Completion mean | Mean $/call | Total reported $ | Missing-usage failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pipeline in sorted(stages):
        for stage, stats in sorted(stages[pipeline].items()):
            lines.append(
                f"| {pipeline.upper()}/{stage} | {stats['attempts_with_usage']} | {stats['prompt_mean']:.0f} | "
                f"{stats['prompt_p50']:.0f} | {stats['prompt_p95']:.0f} | {stats['completion_mean']:.0f} | "
                f"{money(stats['cost_mean_billed_call_usd'])} | {money(stats['reported_cost_usd'])} | "
                f"{stats['failed_without_usage']} |"
            )

    lines += [
        "",
        "## Conditional P2 estimate",
        "",
        "This is a transparent sensitivity estimate, not another measured run. It starts from observed P2 cost",
        "and assumes each flagged page receives one verifier priced like a P2 router call and/or one batched",
        "expression montage priced like a P2 person call.",
        "",
        "| Pages with verifier | Pages with montage | Estimated $/page | Estimated $/1k pages |",
        "|---:|---:|---:|---:|",
    ]
    for scenario in hybrid_scenarios:
        lines.append(
            f"| {scenario['ambiguous_page_share']:.0%} | {scenario['montage_page_share']:.0%} | "
            f"{money(scenario['estimated_cost_per_page_usd'])} | "
            f"{money(scenario['estimated_cost_per_1000_pages_usd'], 3)} |"
        )
    lines.append("")
    (OUT_DIR / "cost_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUT_DIR / "cost_analysis.md")


if __name__ == "__main__":
    main()

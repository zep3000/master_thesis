#!/usr/bin/env python3
"""Evaluate the separate post-pipeline legacy-box matching experiment."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANONICAL = ROOT / "qwen_iteration" / "canonical_candidate_280"
sys.path.insert(0, str(CANONICAL))

import evaluate  # noqa: E402
from common import rows  # noqa: E402


def predictions(cohort: str, route: str) -> dict:
    if route == "canonical":
        return evaluate.predictions(cohort, route)
    path = HERE / "output" / "assembled" / cohort / f"{route}.jsonl"
    return {row["image_id"]: row["annotation"] for row in rows(path) if row.get("ok")}


def main() -> int:
    ev = evaluate.frozen()
    cohort = "difficult140"
    human, ids = evaluate.gold(cohort), evaluate.selected_ids(cohort)
    routes = {name: predictions(cohort, name) for name in ["canonical", "detector_half_center", "detector_full_center"]}
    report = {}
    for mode in ["legacy_strict", "iou_0.5"]:
        page = {}
        for image_id in ids:
            if image_id not in human:
                continue
            page[image_id] = {}
            for route, pred in routes.items():
                if image_id not in pred:
                    continue
                pairs, totals = evaluate.page_pairs(ev, image_id, human[image_id], pred[image_id], mode)
                page[image_id][route] = (totals["people"][0], totals["people"][1], len(pairs["people"]))
        values = list(page)
        rng = random.Random(20260818 + len(mode))
        route_report = {}
        for route in ["detector_half_center", "detector_full_center"]:
            deltas = []
            for _ in range(10000):
                sample = [rng.choice(values) for _ in values]
                scores = {}
                for name in ["canonical", route]:
                    h = sum(page[i][name][0] for i in sample); p = sum(page[i][name][1] for i in sample); m = sum(page[i][name][2] for i in sample)
                    precision, recall = evaluate.safe_div(m, p), evaluate.safe_div(m, h)
                    scores[name] = evaluate.f1(precision, recall) or 0.0
                deltas.append(scores[route] - scores["canonical"])
            route_report[route] = {"f1_delta": statistics_mean(deltas), "paired_page_bootstrap_ci95": evaluate.ci(deltas), "probability_delta_positive": sum(value > 0 for value in deltas) / len(deltas)}
        report[mode] = route_report
    path = HERE / "evaluation" / "detector_center_paired_bootstrap.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def statistics_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())

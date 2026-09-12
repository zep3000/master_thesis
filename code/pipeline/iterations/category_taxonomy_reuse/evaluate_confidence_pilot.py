#!/usr/bin/env python3
"""Evaluate WARC-label accuracy and confidence calibration on manual audit 50."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "category_taxonomy_reuse"
MANUAL = HERE / "audit_50_manual.json"
RESULTS = HERE / "confidence_pilot" / "results.jsonl"
OUT_JSON = HERE / "confidence_pilot" / "evaluation.json"
OUT_MD = HERE / "confidence_pilot" / "evaluation.md"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm_name(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def name_match(predicted: str | None, references: list[str]) -> bool:
    pred = norm_name(predicted)
    if not pred:
        return False
    for reference in references:
        ref = norm_name(reference)
        if not ref:
            continue
        if (pred in ref or ref in pred) and min(len(pred), len(ref)) >= 4:
            return True
        if SequenceMatcher(None, pred, ref).ratio() >= 0.76:
            return True
    return False


def calibration(rows: list[dict[str, Any]], confidence_key: str, correct_key: str) -> dict[str, Any]:
    if not rows:
        return {}
    values = [(float(r[confidence_key]), int(bool(r[correct_key]))) for r in rows]
    brier = sum((p - y) ** 2 for p, y in values) / len(values)
    bins = []
    ece = 0.0
    for lower in [0.0, 0.2, 0.4, 0.6, 0.8]:
        upper = lower + 0.2
        subset = [(p, y) for p, y in values if lower <= p <= upper] if upper == 1.0 else [(p, y) for p, y in values if lower <= p < upper]
        if not subset:
            continue
        mean_p = sum(p for p, _ in subset) / len(subset)
        accuracy = sum(y for _, y in subset) / len(subset)
        ece += len(subset) / len(values) * abs(mean_p - accuracy)
        bins.append({"lower": lower, "upper": upper, "n": len(subset), "mean_confidence": mean_p, "accuracy": accuracy})
    correct_conf = [p for p, y in values if y]
    wrong_conf = [p for p, y in values if not y]
    return {
        "n": len(values),
        "accuracy": sum(y for _, y in values) / len(values),
        "mean_confidence": sum(p for p, _ in values) / len(values),
        "min_confidence": min(p for p, _ in values),
        "max_confidence": max(p for p, _ in values),
        "stdev_confidence": statistics.pstdev(p for p, _ in values),
        "mean_confidence_correct": sum(correct_conf) / len(correct_conf) if correct_conf else None,
        "mean_confidence_incorrect": sum(wrong_conf) / len(wrong_conf) if wrong_conf else None,
        "brier": brier,
        "ece_5_fixed_bins": ece,
        "high_confidence_errors_0_90_plus": sum(1 for p, y in values if p >= 0.9 and not y),
        "unique_confidences": sorted(set(p for p, _ in values)),
        "bins": bins,
    }


def main() -> int:
    manual = json.loads(MANUAL.read_text(encoding="utf-8"))
    truth = {(r["image_id"], r["ad_id"]): r for r in manual["ads"]}
    raw_results = [r for r in load_jsonl(RESULTS) if r.get("ok")]
    evaluated = []
    for row in raw_results:
        ref = truth[(row["image_id"], row["ad_id"])]
        result = row["result"]
        accepted = ref["accepted_categories"]
        direct = ref["fit_status"] != "no_direct_category"
        category_correct = result["ad_category"] in accepted if direct else None
        preferred_correct = result["ad_category"] == accepted[0] if direct else None
        references = [ref["reference_brand"], *(ref.get("reference_brand_alternatives") or [])]
        brand_correct = name_match(result.get("brand_or_advertiser"), references)
        evaluated.append({
            "variant": row["variant"],
            "index": ref["index"],
            "image_id": row["image_id"],
            "ad_id": row["ad_id"],
            "fit_status": ref["fit_status"],
            "accepted_categories": accepted,
            "predicted_category": result["ad_category"],
            "category_confidence": result["ad_category_confidence"],
            "category_correct": category_correct,
            "preferred_correct": preferred_correct,
            "reference_brand": ref["reference_brand"],
            "predicted_brand": result.get("brand_or_advertiser"),
            "brand_confidence": result["brand_confidence"],
            "brand_correct_auto": brand_correct,
            "note": ref["note"],
        })

    report: dict[str, Any] = {
        "audit_fit": {
            "n": len(manual["ads"]),
            "clear": sum(r["fit_status"] == "clear" for r in manual["ads"]),
            "boundary": sum(r["fit_status"] == "boundary" for r in manual["ads"]),
            "no_direct_category": sum(r["fit_status"] == "no_direct_category" for r in manual["ads"]),
            "category_present_rate": sum(r["fit_status"] != "no_direct_category" for r in manual["ads"]) / len(manual["ads"]),
            "single_unambiguous_rate": sum(r["fit_status"] == "clear" for r in manual["ads"]) / len(manual["ads"]),
        },
        "variants": {},
        "rows": evaluated,
    }
    for variant in sorted(set(r["variant"] for r in evaluated)):
        rows = [r for r in evaluated if r["variant"] == variant]
        direct = [r for r in rows if r["category_correct"] is not None]
        clear = [r for r in direct if r["fit_status"] == "clear"]
        boundary = [r for r in direct if r["fit_status"] == "boundary"]
        no_direct = [r for r in rows if r["fit_status"] == "no_direct_category"]
        usage = [r["usage"] for r in raw_results if r["variant"] == variant and isinstance(r.get("usage"), dict)]
        report["variants"][variant] = {
            "category": calibration(direct, "category_confidence", "category_correct"),
            "category_preferred_accuracy": sum(bool(r["preferred_correct"]) for r in direct) / len(direct),
            "clear_accuracy": sum(bool(r["category_correct"]) for r in clear) / len(clear),
            "boundary_accepted_accuracy": sum(bool(r["category_correct"]) for r in boundary) / len(boundary),
            "no_direct_category_mean_confidence": sum(float(r["category_confidence"]) for r in no_direct) / len(no_direct),
            "no_direct_category_predictions": [{"image_id": r["image_id"], "prediction": r["predicted_category"], "confidence": r["category_confidence"]} for r in no_direct],
            "brand": calibration(rows, "brand_confidence", "brand_correct_auto"),
            "usage": {
                "calls": len(usage),
                "prompt_tokens": sum(int(u.get("prompt_tokens") or 0) for u in usage),
                "completion_tokens": sum(int(u.get("completion_tokens") or 0) for u in usage),
                "total_tokens": sum(int(u.get("total_tokens") or 0) for u in usage),
                "cost_usd": sum(float(u.get("cost") or 0) for u in usage),
                "mean_cost_usd": sum(float(u.get("cost") or 0) for u in usage) / len(usage),
            },
            "category_errors": [r for r in direct if not r["category_correct"]],
            "brand_errors_auto": [r for r in rows if not r["brand_correct_auto"]],
        }

    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Confidence-prompt pilot",
        "",
        f"Manual WARC fit: {report['audit_fit']['clear']}/50 clear, {report['audit_fit']['boundary']}/50 boundary, {report['audit_fit']['no_direct_category']}/50 no direct category.",
        "",
        "| Variant | Accepted accuracy | Clear accuracy | Boundary accuracy | Mean conf. | Wrong conf. | Brier | ECE | >=.90 errors | Brand auto acc. | Cost/call |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, stats in report["variants"].items():
        cat, brand, usage = stats["category"], stats["brand"], stats["usage"]
        lines.append(
            f"| {variant} | {cat['accuracy']:.1%} | {stats['clear_accuracy']:.1%} | {stats['boundary_accepted_accuracy']:.1%} | "
            f"{cat['mean_confidence']:.3f} | {cat['mean_confidence_incorrect'] if cat['mean_confidence_incorrect'] is not None else 'n/a'} | "
            f"{cat['brier']:.3f} | {cat['ece_5_fixed_bins']:.3f} | {cat['high_confidence_errors_0_90_plus']} | {brand['accuracy']:.1%} | ${usage['mean_cost_usd']:.6f} |"
        )
    lines.extend(["", "Automatic brand matching is deliberately permissive and must be visually adjudicated for reported errors.", ""])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))
    print(OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

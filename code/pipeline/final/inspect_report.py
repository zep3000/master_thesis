"""Print a compact, read-only extraction from evaluation/final_report.json."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPORT = json.loads((HERE / "evaluation" / "final_report.json").read_text(encoding="utf-8"))
FIELDS = [
    "perceived_age", "perceived_gender_presentation", "face_expression_legibility",
    "face_orientation", "gaze_target", "smile_present", "smile_intensity",
]
GROUP_FIELDS = ["expression_legibility_distribution", "dominant_gaze", "smile_prevalence", "dominant_smile_intensity"]


def modes(value: dict) -> list[str]:
    preferred = ["optimal_strict", "optimal_lenient", "optimal_iou_0.5", "legacy_strict", "legacy_lenient", "legacy_iou_0.5"]
    historical = ["legacy_strict", "legacy_lenient", "iou_0.5"]
    return [mode for mode in preferred if mode in value] or [mode for mode in historical if mode in value]


def slim(value: dict) -> dict:
    return {
        "pages": value["pages"],
        "detection": {entity: {key: value["detection"][entity].get(key) for key in ["human", "predicted", "matched", "f1", "f1_ci95_page_bootstrap"]} for entity in ["ads", "people", "groups"]},
        "people": {field: {key: value["fields"]["people"][field].get(key) for key in ["gold_observed_in_spatial_pairs", "prediction_coverage", "exact_accuracy", "lenient_accuracy", "majority_baseline_accuracy", "gain_over_majority", "cohen_kappa_nonnull", "qwk_nonnull", "exact_ci95_page_bootstrap"]} for field in FIELDS},
        "groups": {field: {key: value["fields"]["groups"][field].get(key) for key in ["gold_observed_in_spatial_pairs", "prediction_coverage", "exact_accuracy", "lenient_accuracy", "majority_baseline_accuracy", "cohen_kappa_nonnull", "qwk_nonnull", "exact_ci95_page_bootstrap"]} for field in GROUP_FIELDS},
        "no_group": value["human_no_group"],
        "group_graph": value.get("group_overlap_graph"),
        "ad_group_smile": value.get("ad_level_group_smile_sensitivity"),
    }


def main() -> int:
    out = {"cohorts": {}, "pooled": {}, "duplicates": {}, "costs": REPORT["costs"]}
    for cohort in ["difficult", "stratified"]:
        final = REPORT["cohorts"][cohort]["final"]
        out["cohorts"][cohort] = {
            scope: {mode: slim(final[scope][mode]) for mode in modes(final[scope])}
            for scope in final
        }
        out["duplicates"][cohort] = REPORT["cohorts"][cohort]["duplicate_candidates"]
    pooled = REPORT["pooled"]["final"]["scopes"]
    out["pooled"] = {scope: {mode: slim(value[mode]) for mode in modes(value)} for scope, value in pooled.items()}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

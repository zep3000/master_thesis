"""Print compact Markdown tables from the frozen final evaluation (read-only)."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "evaluation" / "final_report.json").read_text(encoding="utf-8"))


def pct(value):
    return "—" if value is None else f"{100 * value:.1f}"


def scope(cohort, name, mode=None):
    available = R["pooled"]["final"]["scopes"][name] if cohort == "pooled" else R["cohorts"][cohort]["final"][name]
    mode = mode or ("optimal_strict" if "optimal_strict" in available else "legacy_strict")
    if cohort == "pooled":
        return available[mode]
    return available[mode]


def main() -> int:
    print("DETECTION")
    print("cohort\tscope\tmode\tpages\tadF1\tpersonF1\tgroupF1\tnoGroupSpec\tnoGroupPersonF1")
    selections = [
        ("difficult", "all"), ("stratified", "all"),
        ("difficult", "development140"), ("difficult", "reserve"),
        ("stratified", "development140"), ("stratified", "reserve"),
        ("pooled", "development280"), ("pooled", "reserve118"),
        ("difficult", "first60"), ("difficult", "historical_next40_indices60_99"), ("difficult", "reserve_first40_indices140_179"),
        ("stratified", "first60"), ("stratified", "historical_next40_indices60_99"), ("stratified", "reserve_first40_indices140_179"),
    ]
    for cohort, name in selections:
        available = R["pooled"]["final"]["scopes"][name] if cohort == "pooled" else R["cohorts"][cohort]["final"][name]
        modes = [mode for mode in ["optimal_strict", "optimal_lenient", "optimal_iou_0.5", "legacy_strict", "legacy_lenient", "legacy_iou_0.5", "iou_0.5"] if mode in available]
        for mode in modes:
            x = scope(cohort, name, mode)
            print("\t".join(map(str, [cohort, name, mode, x["pages"], pct(x["detection"]["ads"]["f1"]), pct(x["detection"]["people"]["f1"]), pct(x["detection"]["groups"]["f1"]), pct(x["human_no_group"]["page_group_free_specificity"]), pct(x["human_no_group"]["person_detection"]["f1"])])))

    print("\nLABELS_STRICT_SPATIAL")
    print("cohort\tscope\tfield\tn\texact\tlenient\tmajority\tkappa\tqwk\tCI95")
    fields = ["perceived_age", "perceived_gender_presentation", "face_expression_legibility", "face_orientation", "gaze_target", "smile_present", "smile_intensity"]
    for cohort, name in [selections[i] for i in [0, 1, 2, 3, 4, 5, 6, 7]]:
        x = scope(cohort, name)
        for field in fields:
            f = x["fields"]["people"][field]
            print("\t".join(map(str, [cohort, name, field, f["gold_observed_in_spatial_pairs"], pct(f["exact_accuracy"]), pct(f["lenient_accuracy"]), pct(f["majority_baseline_accuracy"]), f.get("cohen_kappa_nonnull"), f.get("qwk_nonnull"), f.get("exact_ci95_page_bootstrap")])))

    print("\nGROUP_LABELS_STRICT_SPATIAL")
    print("cohort\tfield\tn\texact\tlenient\tmajority\tkappa")
    for cohort in ["difficult", "stratified"]:
        x = scope(cohort, "all")
        for field in ["expression_legibility_distribution", "dominant_gaze", "smile_prevalence", "dominant_smile_intensity"]:
            f = x["fields"]["groups"][field]
            print("\t".join(map(str, [cohort, field, f["gold_observed_in_spatial_pairs"], pct(f["exact_accuracy"]), pct(f["lenient_accuracy"]), pct(f["majority_baseline_accuracy"]), f.get("cohen_kappa_nonnull")])))

    print("\nGROUP_DIAGNOSTICS_STRICT")
    for cohort in ["difficult", "stratified"]:
        x = scope(cohort, "all")
        print(cohort, json.dumps({"graph": x["group_overlap_graph"], "ad_smile": x["ad_level_group_smile_sensitivity"]}, separators=(",", ":")))

    print("\nOVERFIT_KEY_METRICS")
    print("scope\tadF1\tpersonF1\tgroupF1\tlegExact\torient\tgaze\tsmile\tlegCI")
    for name in ["development280", "reserve118"]:
        x = scope("pooled", name)
        p = x["fields"]["people"]
        print("\t".join(map(str, [name, pct(x["detection"]["ads"]["f1"]), pct(x["detection"]["people"]["f1"]), pct(x["detection"]["groups"]["f1"]), pct(p["face_expression_legibility"]["exact_accuracy"]), pct(p["face_orientation"]["exact_accuracy"]), pct(p["gaze_target"]["exact_accuracy"]), pct(p["smile_present"]["exact_accuracy"]), p["face_expression_legibility"]["exact_ci95_page_bootstrap"]])))

    print("\nSMALL_SCOPE_LABELS_STRICT")
    print("cohort\tscope\tlegN\tlegExact\tlegLenient\torientation\tgaze\tsmile")
    for cohort in ["difficult", "stratified"]:
        for name in ["first60", "historical_next40_indices60_99", "reserve_first40_indices140_179"]:
            p = scope(cohort, name)["fields"]["people"]
            leg = p["face_expression_legibility"]
            print("\t".join(map(str, [cohort, name, leg["gold_observed_in_spatial_pairs"], pct(leg["exact_accuracy"]), pct(leg["lenient_accuracy"]), pct(p["face_orientation"]["exact_accuracy"]), pct(p["gaze_target"]["exact_accuracy"]), pct(p["smile_present"]["exact_accuracy"])])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluation-only reporting for the canonical candidate package.

This is the only module in the package that reads human gold.  It reports the
legacy spatial conventions for comparability, conventional IoU >= .50, field
coverage, class metrics, Cohen's kappa / QWK, majority baselines, page-bootstrap
confidence intervals, duplicate-candidate performance, and physical costs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matching as local_matching


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
HUMAN_GOLD = (
    REPOSITORY_ROOT / "artifacts" / "human-validation" / "raw" / "pipeline-development"
)
OUTPUT = HERE / "output"
COST_RUN_ID = "standalone_final_398_v1"
GOLD = {
    "difficult": HUMAN_GOLD / "difficult-198.json",
    "stratified": HUMAN_GOLD / "stratified-200.json",
}
MODES = [
    "optimal_strict", "optimal_lenient", "optimal_iou_0.5",
    "legacy_strict", "legacy_lenient", "legacy_iou_0.5",
]
ENTITY_FIELDS = {
    "ads": ["extent", "depiction_type", "face_depiction_count_band", "duplicate_faces_present", "unique_face_count", "has_outstanding_individuals"],
    "people": ["annotation_role", "depiction_type", "perceived_age", "perceived_gender_presentation", "face_expression_legibility", "face_orientation", "gaze_target", "gaze_target_person_unboxed", "mouth_covered", "mouth_covering", "smile_present", "smile_intensity"],
    "groups": ["group_type", "age_composition", "gender_presentation_composition", "expression_legibility_distribution", "dominant_gaze", "smile_prevalence", "dominant_smile_intensity"],
}
ORDERS = {
    "face_depiction_count_band": [str(i) for i in range(1, 10)] + ["10_20", "20_plus"],
    "perceived_age": ["infant", "child", "adolescent", "young_adult", "middle_adult", "older_adult"],
    "face_expression_legibility": ["0_not_legible", "1_low_legibility", "2_moderate_legibility", "3_high_legibility"],
    "smile_intensity": ["1_slight", "2_clear", "3_broad", "4_laughter_like"],
    "smile_prevalence": ["none", "minority", "about_half", "majority", "all"],
    "dominant_smile_intensity": ["slight", "clear", "broad_or_laughter_like"],
}
NOT_REQUESTED: dict[str, Any] = {}


def rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def manifest(cohort: str) -> dict[str, Any]:
    return json.loads((HERE / "data" / f"manifest_{cohort}.json").read_text(encoding="utf-8"))


def frozen() -> Any:
    return local_matching


def gold(cohort: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(GOLD[cohort].read_text(encoding="utf-8"))
    result = {}
    for row in payload.get("annotations") or []:
        if isinstance(row.get("payload"), dict):
            result[str(row["image_id"])] = row["payload"]
    return result


def selected_ids(cohort: str) -> list[str]:
    return [row["image_id"] for row in manifest(cohort)["images"]]


def apply_raw(cohort: str, predictions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    raw = {row["task_key"]: row["raw_normalized_values"] for row in rows(OUTPUT / "raw_values" / f"{cohort}.jsonl")}
    output = copy.deepcopy(predictions)
    for image_id, ann in output.items():
        for ad in ann.get("advertisements") or []:
            for kind, id_field in [("people", "person_id"), ("groups", "group_id")]:
                for entity in ad.get(kind) or []:
                    values = raw.get(f"{image_id}::{entity[id_field]}")
                    if values:
                        entity.update(copy.deepcopy(values))
    return output


def predictions(cohort: str, route: str) -> dict[str, dict[str, Any]]:
    if route in {"final", "raw_analysis"}:
        path = OUTPUT / cohort / "assembled" / "final.jsonl"
        result = {row["image_id"]: row["annotation"] for row in rows(path) if row.get("ok")}
        return apply_raw(cohort, result) if route == "raw_analysis" else result
    raise ValueError(route)


def pair_items(ev: Any, human: list[Any], predicted: list[Any], entity: str, mode: str) -> list[Any]:
    if mode == "legacy_strict":
        return ev.pair_items(human, predicted, entity, "strict")
    if mode == "legacy_lenient":
        return ev.pair_items(human, predicted, entity, "lenient")
    if mode == "optimal_strict":
        return ev.optimal_pair_items(human, predicted, entity, "strict")
    if mode == "optimal_lenient":
        return ev.optimal_pair_items(human, predicted, entity, "lenient")
    if mode == "optimal_iou_0.5":
        return ev.optimal_pair_items(human, predicted, entity, "strict", threshold=.50)
    if mode != "legacy_iou_0.5":
        raise ValueError(f"unknown spatial matching mode: {mode}")
    candidates = []
    for hi, h in enumerate(human):
        for pi, p in enumerate(predicted):
            iou, contained = ev.overlap(h.box, p.box)
            if iou >= 0.50:
                candidates.append((iou, contained, hi, pi, h, p))
    candidates.sort(key=lambda value: value[0], reverse=True)
    used_h: set[int] = set(); used_p: set[int] = set(); result = []
    for iou, contained, hi, pi, h, p in candidates:
        if hi in used_h or pi in used_p:
            continue
        used_h.add(hi); used_p.add(pi)
        result.append(ev.Pair(h, p, iou, contained))
    return result


def page_pairs(ev: Any, image_id: str, human: dict[str, Any], pred: dict[str, Any], mode: str) -> tuple[dict[str, list[Any]], dict[str, tuple[int, int]]]:
    pairs = {entity: [] for entity in ["ads", "people", "groups"]}
    totals = {entity: (len(ev.all_items(image_id, human, entity)), len(ev.all_items(image_id, pred, entity))) for entity in pairs}
    h_ads, p_ads = ev.all_items(image_id, human, "ads"), ev.all_items(image_id, pred, "ads")
    pairs["ads"] = pair_items(ev, h_ads, p_ads, "ads", mode)
    for ad_pair in pairs["ads"]:
        pairs["people"].extend(pair_items(ev, ev.person_items(image_id, ad_pair.human.data), ev.person_items(image_id, ad_pair.predicted.data), "people", mode))
        pairs["groups"].extend(pair_items(ev, ev.group_items(image_id, ad_pair.human.data), ev.group_items(image_id, ad_pair.predicted.data), "groups", mode))
    return pairs, totals


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def f1(precision: float | None, recall: float | None) -> float | None:
    return 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else 0.0 if precision is not None and recall is not None else None


def cohen_kappa(human: list[str], predicted: list[str]) -> float | None:
    if not human or len(human) != len(predicted):
        return None
    labels = sorted(set(human) | set(predicted))
    n = len(human)
    observed = sum(h == p for h, p in zip(human, predicted)) / n
    expected = sum(human.count(label) * predicted.count(label) for label in labels) / (n * n)
    return (observed - expected) / (1 - expected) if expected < 1 else None


def qwk(human: list[str], predicted: list[str], order: list[str]) -> float | None:
    pairs = [(order.index(h), order.index(p)) for h, p in zip(human, predicted) if h in order and p in order]
    if not pairs:
        return None
    size, n = len(order), len(pairs)
    matrix = [[0] * size for _ in range(size)]
    for h, p in pairs:
        matrix[h][p] += 1
    hrows = [sum(row) for row in matrix]
    pcols = [sum(matrix[i][j] for i in range(size)) for j in range(size)]
    denominator = max(1, (size - 1) ** 2)
    observed = sum(((i - j) ** 2 / denominator) * matrix[i][j] for i in range(size) for j in range(size)) / n
    expected = sum(((i - j) ** 2 / denominator) * hrows[i] * pcols[j] / n for i in range(size) for j in range(size)) / n
    return 1 - observed / expected if expected else None


def field_counts(ev: Any, pairs: list[Any], gold_total: int, field: str) -> tuple[dict[str, Any], dict[str, float]]:
    gold_values: list[str] = []
    pred_values_nonnull: list[str] = []
    paired_h: list[str] = []
    paired_p: list[str] = []
    exact = lenient = predicted_nonnull = 0
    for pair in pairs:
        h = ev.normalize_value(pair.human.data.get(field), field)
        p = ev.normalize_value(pair.predicted.data.get(field), field)
        if h is None:
            continue
        hs = str(h); gold_values.append(hs)
        if p is not None:
            ps = str(p); predicted_nonnull += 1; pred_values_nonnull.append(ps); paired_h.append(hs); paired_p.append(ps)
        exact += int(p is not None and ev.exact_label(field, h, p))
        lenient += int(p is not None and ev.lenient_label(field, h, p))
    gold_n = len(gold_values)
    labels = sorted(set(gold_values) | set(pred_values_nonnull))
    class_metrics: dict[str, Any] = {}
    for label in labels:
        tp = sum(h == label and p == label for h, p in zip(paired_h, paired_p))
        fp = sum(h != label and p == label for h, p in zip(paired_h, paired_p))
        fn = sum(h == label for h in gold_values) - tp
        precision, recall = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
        class_metrics[label] = {"support": gold_values.count(label), "precision": precision, "recall": recall, "f1": f1(precision, recall)}
    recalls = [value["recall"] for value in class_metrics.values() if value["recall"] is not None]
    majority = max(Counter(gold_values).values()) / gold_n if gold_n else None
    report = {
        "gold_observed_in_spatial_pairs": gold_n,
        "predicted_nonnull": predicted_nonnull,
        "prediction_coverage": safe_div(predicted_nonnull, gold_n),
        "exact_accuracy": safe_div(exact, gold_n),
        "lenient_accuracy": safe_div(lenient, gold_n),
        "exact_when_nonnull": safe_div(exact, predicted_nonnull),
        "gold_observed_total": gold_total,
        "exact_end_to_end_recall": safe_div(exact, gold_total),
        "majority_baseline_accuracy": majority,
        "gain_over_majority": (safe_div(exact, gold_n) - majority) if gold_n and majority is not None else None,
        "cohen_kappa_nonnull": cohen_kappa(paired_h, paired_p),
        "qwk_nonnull": qwk(paired_h, paired_p, ORDERS[field]) if field in ORDERS else None,
        "macro_recall": statistics.fmean(recalls) if recalls else None,
        "class_metrics": class_metrics,
    }
    return report, {"gold": gold_n, "pred": predicted_nonnull, "exact": exact, "lenient": lenient, "gold_total": gold_total}


def simple_field_counts(ev: Any, pairs: list[Any], field: str) -> tuple[int, int, int]:
    gold_n = exact = lenient = 0
    for pair in pairs:
        h = ev.normalize_value(pair.human.data.get(field), field)
        p = ev.normalize_value(pair.predicted.data.get(field), field)
        if h is None:
            continue
        gold_n += 1
        exact += int(p is not None and ev.exact_label(field, h, p))
        lenient += int(p is not None and ev.lenient_label(field, h, p))
    return gold_n, exact, lenient


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def ci(values: list[float]) -> list[float | None]:
    return [percentile(values, .025), percentile(values, .975)]


def route_mode(ev: Any, human: dict[str, dict[str, Any]], pred: dict[str, dict[str, Any]], ids: list[str], mode: str, route: str, bootstrap: int) -> dict[str, Any]:
    overlap = [image_id for image_id in ids if image_id in human and image_id in pred]
    per_page: dict[str, Any] = {}
    all_pairs = {entity: [] for entity in ENTITY_FIELDS}
    totals_h = Counter(); totals_p = Counter(); gold_items = {entity: [] for entity in ENTITY_FIELDS}
    ious = {entity: [] for entity in ENTITY_FIELDS}
    no_group_ids = []
    for image_id in overlap:
        pairs, totals = page_pairs(ev, image_id, human[image_id], pred[image_id], mode)
        per_page[image_id] = {"pairs": pairs, "totals": totals}
        if totals["groups"][0] == 0:
            no_group_ids.append(image_id)
        for entity in ENTITY_FIELDS:
            all_pairs[entity].extend(pairs[entity]); totals_h[entity] += totals[entity][0]; totals_p[entity] += totals[entity][1]
            gold_items[entity].extend(ev.all_items(image_id, human[image_id], entity)); ious[entity].extend(pair.iou for pair in pairs[entity])
    detection = {}
    for entity in ENTITY_FIELDS:
        precision = safe_div(len(all_pairs[entity]), totals_p[entity]); recall = safe_div(len(all_pairs[entity]), totals_h[entity])
        detection[entity] = {"human": totals_h[entity], "predicted": totals_p[entity], "matched": len(all_pairs[entity]), "precision": precision, "recall": recall, "f1": f1(precision, recall), "mean_iou": statistics.fmean(ious[entity]) if ious[entity] else None}
    fields: dict[str, Any] = {}
    contribution: dict[str, Any] = {}
    not_requested = NOT_REQUESTED.get(route, {})
    for entity, field_names in ENTITY_FIELDS.items():
        fields[entity] = {}; contribution[entity] = {}
        for field in field_names:
            if field in not_requested.get(entity, set()):
                fields[entity][field] = {"status": "not_requested"}
                continue
            gold_total = sum(ev.normalize_value(item.data.get(field), field) is not None for item in gold_items[entity])
            fields[entity][field], contribution[entity][field] = field_counts(ev, all_pairs[entity], gold_total, field)
    pred_false_groups = sum(per_page[image_id]["totals"]["groups"][1] for image_id in no_group_ids)
    no_group_person_h = sum(per_page[i]["totals"]["people"][0] for i in no_group_ids)
    no_group_person_p = sum(per_page[i]["totals"]["people"][1] for i in no_group_ids)
    no_group_person_m = sum(len(per_page[i]["pairs"]["people"]) for i in no_group_ids)
    ng_precision, ng_recall = safe_div(no_group_person_m, no_group_person_p), safe_div(no_group_person_m, no_group_person_h)
    no_group = {
        "pages": len(no_group_ids), "pages_predicted_group_free": sum(per_page[i]["totals"]["groups"][1] == 0 for i in no_group_ids),
        "page_group_free_specificity": safe_div(sum(per_page[i]["totals"]["groups"][1] == 0 for i in no_group_ids), len(no_group_ids)),
        "false_predicted_groups": pred_false_groups,
        "person_detection": {"human": no_group_person_h, "predicted": no_group_person_p, "matched": no_group_person_m, "precision": ng_precision, "recall": ng_recall, "f1": f1(ng_precision, ng_recall)},
    }
    rng = random.Random(20260818 + MODES.index(mode) * 100 + sum(ord(x) for x in route))
    boot_detection = {entity: [] for entity in ENTITY_FIELDS}
    boot_fields = {(entity, field): {"exact": [], "lenient": []} for entity in ENTITY_FIELDS for field in fields[entity] if fields[entity][field].get("status") != "not_requested"}
    if overlap and bootstrap:
        page_field_counts = {
            (image_id, entity, field): simple_field_counts(ev, per_page[image_id]["pairs"][entity], field)
            for image_id in overlap for entity, field in boot_fields
        }
        for _ in range(bootstrap):
            sample = [rng.choice(overlap) for _ in overlap]
            for entity in ENTITY_FIELDS:
                h = sum(per_page[i]["totals"][entity][0] for i in sample); p = sum(per_page[i]["totals"][entity][1] for i in sample); m = sum(len(per_page[i]["pairs"][entity]) for i in sample)
                precision, recall = safe_div(m, p), safe_div(m, h)
                value = f1(precision, recall)
                if value is not None: boot_detection[entity].append(value)
            for entity, field in boot_fields:
                gold_n = exact_n = lenient_n = 0
                for image_id in sample:
                    page_gold, page_exact, page_lenient = page_field_counts[(image_id, entity, field)]
                    gold_n += page_gold; exact_n += page_exact; lenient_n += page_lenient
                if gold_n:
                    boot_fields[(entity, field)]["exact"].append(exact_n / gold_n)
                    boot_fields[(entity, field)]["lenient"].append(lenient_n / gold_n)
        for entity in detection:
            detection[entity]["f1_ci95_page_bootstrap"] = ci(boot_detection[entity])
        for (entity, field), values in boot_fields.items():
            fields[entity][field]["exact_ci95_page_bootstrap"] = ci(values["exact"])
            fields[entity][field]["lenient_ci95_page_bootstrap"] = ci(values["lenient"])
    return {"pages": len(overlap), "missing_prediction": [i for i in ids if i not in pred], "detection": detection, "fields": fields, "human_no_group": no_group}


def duplicate_groups(ad: dict[str, Any]) -> list[set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    people = ad.get("people") or []
    for person in people:
        pid = str(person.get("person_id"))
        for other in person.get("duplicate_person_ids") or []:
            graph[pid].add(str(other)); graph[str(other)].add(pid)
        other = person.get("duplicate_of_person_id")
        if other:
            graph[pid].add(str(other)); graph[str(other)].add(pid)
    result, seen = [], set()
    for node in graph:
        if node in seen:
            continue
        stack, component = [node], set()
        while stack:
            current = stack.pop()
            if current in component: continue
            component.add(current); stack.extend(graph[current] - component)
        seen.update(component)
        if len(component) >= 2: result.append(component)
    return result


def pair_set(clusters: list[set[str]]) -> set[frozenset[str]]:
    return {frozenset((a, b)) for cluster in clusters for i, a in enumerate(sorted(cluster)) for b in sorted(cluster)[i + 1:]}


def duplicate_evaluation(ev: Any, cohort: str, human: dict[str, dict[str, Any]], pred: dict[str, dict[str, Any]], mode: str) -> dict[str, Any]:
    sidecar = [row for row in rows(OUTPUT / "duplicate_candidates" / f"{cohort}.jsonl") if row.get("ok")]
    by_task = {row["task_key"]: row for row in sidecar}
    totals = {scope: Counter() for scope in ["all", "high"]}
    d0 = {scope: Counter() for scope in ["all", "high"]}
    representable_positive = 0; total_gold_positive = 0; matched_ads = 0
    for image_id in selected_ids(cohort):
        if image_id not in human or image_id not in pred:
            continue
        ad_pairs = pair_items(ev, ev.all_items(image_id, human[image_id], "ads"), ev.all_items(image_id, pred[image_id], "ads"), "ads", mode)
        for ad_pair in ad_pairs:
            task_key = f"{image_id}::{ad_pair.predicted.item_id}"
            row = by_task.get(task_key)
            if not row:
                continue
            matched_ads += 1
            ppairs = pair_items(ev, ev.person_items(image_id, ad_pair.human.data), ev.person_items(image_id, ad_pair.predicted.data), "people", mode)
            pred_to_gold = {str(pair.predicted.item_id): str(pair.human.item_id) for pair in ppairs}
            gold_pairs = pair_set(duplicate_groups(ad_pair.human.data))
            total_gold_positive += len(gold_pairs)
            representable = {pair for pair in gold_pairs if all(value in set(pred_to_gold.values()) for value in pair)}
            representable_positive += len(representable)
            for scope in ["all", "high"]:
                clusters = []
                for cluster in row["model_annotation"].get("candidate_clusters") or []:
                    if scope == "high" and cluster.get("candidate_strength") != "high": continue
                    clusters.append(set(map(str, cluster.get("person_ids") or [])))
                proposed = pair_set(clusters)
                mapped: set[frozenset[str]] = set()
                unmapped = 0
                for pair in proposed:
                    values = [pred_to_gold.get(value) for value in pair]
                    if all(values): mapped.add(frozenset(values))
                    else: unmapped += 1
                tp = len(mapped & gold_pairs); fp = len(mapped - gold_pairs) + unmapped; fn = len(representable - mapped)
                totals[scope].update(tp=tp, fp=fp, fn=fn, proposed=len(proposed))
                gold_d0, pred_d0 = bool(gold_pairs), bool(proposed)
                d0[scope].update(tp=int(gold_d0 and pred_d0), fp=int(not gold_d0 and pred_d0), fn=int(gold_d0 and not pred_d0), tn=int(not gold_d0 and not pred_d0))
    output = {"eligible_predicted_ads": sum(1 for image_id, ann in pred.items() for ad in ann.get("advertisements") or [] if len(ad.get("people") or []) >= 2), "sidecar_tasks": len(sidecar), "matched_sidecar_ads": matched_ads, "representable_gold_positive_pairs": representable_positive, "gold_positive_pairs_in_matched_ads": total_gold_positive, "scopes": {}}
    for scope in ["all", "high"]:
        c = totals[scope]; precision, recall = safe_div(c["tp"], c["tp"] + c["fp"]), safe_div(c["tp"], c["tp"] + c["fn"])
        dc = d0[scope]; dp, dr = safe_div(dc["tp"], dc["tp"] + dc["fp"]), safe_div(dc["tp"], dc["tp"] + dc["fn"])
        output["scopes"][scope] = {"pairwise": {**dict(c), "precision": precision, "recall_representable": recall, "f1": f1(precision, recall), "false_merges": c["fp"], "end_to_end_recall_in_matched_ads": safe_div(c["tp"], total_gold_positive)}, "ad_d0": {**dict(dc), "precision": dp, "recall": dr, "f1": f1(dp, dr)}}
    return output


def intersection_fraction(a: list[float] | None, b: list[float] | None, denominator: str) -> float:
    if not a or not b:
        return 0.0
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    base = local_matching.area(a if denominator == "a" else b)
    return inter / base if base else 0.0


def group_overlap_graph(ev: Any, human: dict[str, dict[str, Any]], pred: dict[str, dict[str, Any]], ids: list[str], mode: str) -> dict[str, Any]:
    """Diagnose splits/merges without changing the one-to-one score matching."""
    component_counts = Counter(); human_coverage = []; predicted_purity = []
    matched_ads = human_groups = predicted_groups = graph_edges = 0
    for image_id in ids:
        if image_id not in human or image_id not in pred:
            continue
        ad_pairs = pair_items(ev, ev.all_items(image_id, human[image_id], "ads"), ev.all_items(image_id, pred[image_id], "ads"), "ads", mode)
        for ad_pair in ad_pairs:
            hgroups = ev.group_items(image_id, ad_pair.human.data); pgroups = ev.group_items(image_id, ad_pair.predicted.data)
            if not hgroups and not pgroups:
                continue
            matched_ads += 1; human_groups += len(hgroups); predicted_groups += len(pgroups)
            adjacency: dict[tuple[str, int], set[tuple[str, int]]] = {}
            for hi, hgroup in enumerate(hgroups): adjacency[("h", hi)] = set()
            for pi, pgroup in enumerate(pgroups): adjacency[("p", pi)] = set()
            for hi, hgroup in enumerate(hgroups):
                overlaps = []
                for pi, pgroup in enumerate(pgroups):
                    iou, contained = ev.overlap(hgroup.box, pgroup.box)
                    hfrac = intersection_fraction(hgroup.box, pgroup.box, "a")
                    pfrac = intersection_fraction(hgroup.box, pgroup.box, "b")
                    overlaps.append(hfrac)
                    if iou >= .02 or contained >= .25 or hfrac >= .25 or pfrac >= .25:
                        adjacency[("h", hi)].add(("p", pi)); adjacency[("p", pi)].add(("h", hi)); graph_edges += 1
                human_coverage.append(max(overlaps) if overlaps else 0.0)
            for pi, pgroup in enumerate(pgroups):
                overlaps = [intersection_fraction(hgroup.box, pgroup.box, "b") for hgroup in hgroups]
                predicted_purity.append(max(overlaps) if overlaps else 0.0)
            seen = set()
            for node in adjacency:
                if node in seen:
                    continue
                stack = [node]; component = set()
                while stack:
                    current = stack.pop()
                    if current in component: continue
                    component.add(current); stack.extend(adjacency[current] - component)
                seen.update(component)
                hn = sum(kind == "h" for kind, _ in component); pn = sum(kind == "p" for kind, _ in component)
                if hn == 1 and pn == 1: label = "one_to_one"
                elif hn == 1 and pn > 1: label = "split"
                elif hn > 1 and pn == 1: label = "merge"
                elif hn > 1 and pn > 1: label = "complex"
                elif hn == 1: label = "uncovered_human"
                else: label = "unmatched_predicted"
                component_counts[label] += 1
    return {
        "edge_rule": "IoU>=.02 OR containment/intersection fraction>=.25",
        "matched_ads_with_any_group": matched_ads,
        "human_groups": human_groups, "predicted_groups": predicted_groups, "graph_edges": graph_edges,
        "components": dict(component_counts),
        "mean_best_human_area_coverage": statistics.fmean(human_coverage) if human_coverage else None,
        "mean_best_predicted_area_purity": statistics.fmean(predicted_purity) if predicted_purity else None,
    }


def collapsed_group_smile(ad: dict[str, Any]) -> str:
    groups = ad.get("groups") or []
    if not groups:
        return "no_group"
    values = [local_matching.normalize_value(group.get("smile_prevalence"), "smile_prevalence") for group in groups]
    if any(value in {"minority", "about_half", "majority", "all"} for value in values):
        return "any_smile"
    if any(value == "none" for value in values):
        return "no_smile"
    return "not_assessable"


def ad_group_smile_sensitivity(ev: Any, human: dict[str, dict[str, Any]], pred: dict[str, dict[str, Any]], ids: list[str], mode: str) -> dict[str, Any]:
    confusion = Counter(); assessed = exact = 0
    h_binary = []; p_binary = []
    for image_id in ids:
        if image_id not in human or image_id not in pred: continue
        ad_pairs = pair_items(ev, ev.all_items(image_id, human[image_id], "ads"), ev.all_items(image_id, pred[image_id], "ads"), "ads", mode)
        for pair in ad_pairs:
            h = collapsed_group_smile(pair.human.data); p = collapsed_group_smile(pair.predicted.data)
            confusion[(h, p)] += 1; assessed += 1; exact += int(h == p)
            if h in {"any_smile", "no_smile"} and p in {"any_smile", "no_smile"}:
                h_binary.append(h == "any_smile"); p_binary.append(p == "any_smile")
    tp = sum(h and p for h, p in zip(h_binary, p_binary)); fp = sum(not h and p for h, p in zip(h_binary, p_binary)); fn = sum(h and not p for h, p in zip(h_binary, p_binary))
    precision, recall = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
    return {
        "matched_ads": assessed, "exact_accuracy_four_state": safe_div(exact, assessed),
        "confusion": [{"human": h, "predicted": p, "n": n} for (h, p), n in confusion.most_common()],
        "binary_among_assessable_group_ads": {"n": len(h_binary), "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1(precision, recall)},
    }


def usage_summary(values: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in values if row.get("ok")]
    usage = [row.get("usage") or {} for row in values]
    failed = [row for row in values if not row.get("ok")]
    failed_keys = {(row.get("cohort"), row.get("stage"), row.get("variant"), row.get("task_key")) for row in failed}
    successful_keys = {(row.get("cohort"), row.get("stage"), row.get("variant"), row.get("task_key")) for row in successful}
    format_markers = ["json", "decode", "schema", "invalid", "truncat"]
    return {
        "logical_successful_calls": len(successful_keys),
        "physical_attempts": len(values), "failed_attempts": len(values) - len(successful),
        "tasks_recovered_after_retry": len(failed_keys & successful_keys),
        "unrecovered_failed_tasks": len(failed_keys - successful_keys),
        "response_format_failure_attempts": sum(any(marker in str(row.get("error") or "").lower() for marker in format_markers) for row in failed),
        "failed_attempt_cost_usd": sum(float((row.get("usage") or {}).get("cost") or 0) for row in failed),
        "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in usage),
        "completion_tokens": sum(int(item.get("completion_tokens") or 0) for item in usage),
        "total_tokens": sum(int(item.get("total_tokens") or 0) for item in usage),
        "cost_usd": sum(float(item.get("cost") or 0) for item in usage),
    }


def costs() -> dict[str, Any]:
    package_rows = [row for row in rows(OUTPUT / "request_ledger.jsonl") if row.get("run_id") == COST_RUN_ID]
    stage_values = {stage: [row for row in package_rows if row.get("stage") == stage] for stage in ["structure", "entities", "duplicates"]}
    stages = {stage: usage_summary(values) for stage, values in stage_values.items()}
    production = usage_summary(stage_values["structure"] + stage_values["entities"])
    with_duplicates = usage_summary(package_rows)
    image_count = sum(len(manifest(cohort)["images"]) for cohort in GOLD)
    for summary in [production, with_duplicates]:
        summary["denominator_images"] = image_count
        summary["per_image"] = {key: summary[key] / image_count for key in ["logical_successful_calls", "physical_attempts", "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"]}
    endpoint_files = sorted((OUTPUT / "endpoint_behavior").glob("*.json")) if (OUTPUT / "endpoint_behavior").exists() else []
    endpoint_behavior = {}
    for path in endpoint_files:
        value = json.loads(path.read_text(encoding="utf-8"))
        wall = value.get("wall_seconds_this_execution")
        maximum = (value.get("latency_seconds") or {}).get("max")
        if wall is not None and maximum is not None and wall < maximum:
            value["telemetry_warning"] = "pre-fix file mixed cumulative ledger records with the final resumable execution wall time; throughput/wall are invalid"
            value["original_invalid_wall_seconds"] = wall
            value["original_invalid_completed_jobs_per_minute"] = value.get("completed_jobs_per_minute")
            value["wall_seconds_this_execution"] = None
            value["completed_jobs_per_minute"] = None
        endpoint_behavior[path.stem] = value
    return {"selected_stages": stages, "production_annotations": production, "production_plus_isolated_duplicate_candidates": with_duplicates, "endpoint_behavior": endpoint_behavior}


def compact(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "detection_f1": {entity: route["detection"][entity]["f1"] for entity in ENTITY_FIELDS},
        "person_exact": {field: value.get("exact_accuracy") for field, value in route["fields"]["people"].items()},
        "person_kappa": {field: value.get("cohen_kappa_nonnull") for field, value in route["fields"]["people"].items()},
        "no_group_specificity": route["human_no_group"]["page_group_free_specificity"],
    }


def main() -> int:
    global OUTPUT, COST_RUN_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", nargs="+", default=["final", "raw_analysis"])
    parser.add_argument("--cohorts", nargs="+", default=["difficult", "stratified"])
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--tag", default="final_report")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--cost-run-id", default=COST_RUN_ID)
    args = parser.parse_args()
    OUTPUT = args.output_root.resolve()
    COST_RUN_ID = args.cost_run_id
    ev = frozen()
    report: dict[str, Any] = {"schema_version": "standalone_final_robust_evaluation_v1", "routes": args.routes, "cohorts": {}, "pooled": {}, "costs": costs()}
    for cohort in args.cohorts:
        human, ids = gold(cohort), selected_ids(cohort)
        report["cohorts"][cohort] = {}
        for route in args.routes:
            pred = predictions(cohort, route)
            if not pred:
                report["cohorts"][cohort][route] = {"status": "not_available_for_cohort"}
                continue
            scopes = {
                "all": ids,
                "development140": ids[:140],
                "reserve": ids[140:],
                "first60": ids[:60],
                "historical_next40_indices60_99": ids[60:100],
                "reserve_first40_indices140_179": ids[140:180],
            }
            route_report = {}
            for scope_name, scope_ids in scopes.items():
                route_report[scope_name] = {mode: route_mode(ev, human, pred, scope_ids, mode, route, args.bootstrap) for mode in MODES}
                if route == "final":
                    for mode in MODES:
                        route_report[scope_name][mode]["group_overlap_graph"] = group_overlap_graph(ev, human, pred, scope_ids, mode)
                        route_report[scope_name][mode]["ad_level_group_smile_sensitivity"] = ad_group_smile_sensitivity(ev, human, pred, scope_ids, mode)
            report["cohorts"][cohort][route] = route_report
            print(json.dumps({"cohort": cohort, "route": route, "all_legacy_strict": compact(route_report["all"]["legacy_strict"]), "reserve_legacy_strict": compact(route_report["reserve"]["legacy_strict"])}, ensure_ascii=False))
        if "final" in args.routes:
            pred = predictions(cohort, "final")
            report["cohorts"][cohort]["duplicate_candidates"] = {mode: duplicate_evaluation(ev, cohort, human, pred, mode) for mode in MODES}
    for route in args.routes:
        pooled_human: dict[str, dict[str, Any]] = {}; pooled_pred: dict[str, dict[str, Any]] = {}
        pooled_scopes: dict[str, list[str]] = {"all398": [], "development280": [], "reserve118": []}; included = []
        for cohort in args.cohorts:
            human, pred = gold(cohort), predictions(cohort, route)
            if not pred:
                continue
            included.append(cohort)
            for index, image_id in enumerate(selected_ids(cohort)):
                key = f"{cohort}::{image_id}"
                pooled_scopes["all398"].append(key)
                pooled_scopes["development280" if index < 140 else "reserve118"].append(key)
                if image_id in human: pooled_human[key] = human[image_id]
                if image_id in pred: pooled_pred[key] = pred[image_id]
        if pooled_pred:
            report["pooled"][route] = {"included_cohorts": included, "scopes": {scope: {mode: route_mode(ev, pooled_human, pooled_pred, ids, mode, route, args.bootstrap) for mode in MODES} for scope, ids in pooled_scopes.items()}}
    path = HERE / "evaluation" / f"{args.tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

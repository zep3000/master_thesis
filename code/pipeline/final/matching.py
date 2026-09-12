"""Standalone spatial and label-matching primitives.

The inference runner cannot import this file and never sees human gold.
Evaluation preserves the established Venice-comparison conventions:

* strict spatial matching: ad/person IoU >= .20, group IoU >= .18;
* lenient spatial matching: ad/person IoU >= .10, group IoU >= .08, or >=95%
  containment of the smaller box with IoU >= .02;
* legacy one-to-one greedy matching for historical comparability;
* optimal one-to-one matching primitives for publication reporting, maximizing
  accepted match count first and spatial score second.

Both exact-label and lenient-label agreement are reported for either spatial
mode.  The report also isolates pages and matched ads where the human annotator
used no group.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STRICT_THRESHOLDS = {"ads": .20, "people": .20, "groups": .18}
LENIENT_THRESHOLDS = {"ads": .10, "people": .10, "groups": .08}
CONTAINMENT_THRESHOLD = .95
MIN_CONTAINED_IOU = .02

PAGE_FIELDS = ["qualifying_ad_count", "no_qualifying_ad_reason"]
AD_FIELDS = ["extent", "depiction_type", "face_depiction_count_band", "duplicate_faces_present", "unique_face_count", "has_outstanding_individuals"]
PERSON_FIELDS = ["annotation_role", "depiction_type", "perceived_age", "perceived_gender_presentation", "face_expression_legibility", "face_orientation", "gaze_target", "gaze_target_person_unboxed", "mouth_covered", "mouth_covering", "smile_present", "smile_intensity"]
GROUP_FIELDS = ["group_type", "age_composition", "gender_presentation_composition", "expression_legibility_distribution", "dominant_gaze", "smile_prevalence", "dominant_smile_intensity"]

ORDERS = {
    "face_depiction_count_band": [str(i) for i in range(1, 10)] + ["10_20", "20_plus"],
    "perceived_age": ["infant", "child", "adolescent", "young_adult", "middle_adult", "older_adult"],
    "face_expression_legibility": ["0_not_legible", "1_low_legibility", "2_moderate_legibility", "3_high_legibility"],
    "smile_intensity": ["1_slight", "2_clear", "3_broad", "4_laughter_like"],
    "smile_prevalence": ["none", "minority", "about_half", "majority", "all"],
    "dominant_smile_intensity": ["slight", "clear", "broad_or_laughter_like"],
}


@dataclass
class Item:
    image_id: str
    ad_id: str | None
    item_id: str | None
    data: dict[str, Any]
    box: list[float] | None


@dataclass
class Pair:
    human: Item
    predicted: Item
    iou: float
    containment: float


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_value(value: Any, field: str) -> Any:
    if value == [] or (isinstance(value, str) and value in {"", "not_applicable"}):
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        generic = {
            "general_crowd": "other_group",
            "1_not_legible": "0_not_legible", "2_low_legibility": "1_low_legibility",
            "3_moderate_legibility": "2_moderate_legibility", "4_high_legibility": "3_high_legibility",
        }
        per_field = {
            "depiction_type": {"naturalistic_illustration": "illustration", "stylized_illustration": "illustration"},
            "gaze_target": {"off_frame": "off_frame_or_scene_direction", "scene_direction": "off_frame_or_scene_direction"},
            "dominant_gaze": {"off_frame": "off_frame_or_scene_direction", "scene_direction": "off_frame_or_scene_direction"},
            "mouth_covering": {"own_body_part": "other_body_part", "another_person": "part_of_another_person"},
        }
        return per_field.get(field, {}).get(value, generic.get(value, value))
    return value


def exact_label(field: str, human: Any, predicted: Any) -> bool:
    return normalize_value(human, field) == normalize_value(predicted, field)


def composition_family(value: Any) -> str | None:
    value = str(value)
    for family in ["young", "middle", "older", "feminine", "masculine"]:
        if family in value:
            return family
    return value if value in {"mixed", "ambiguous_or_androgynous_present", "not_assessable"} else None


def group_legibility_level(value: Any) -> int | None:
    value = str(value)
    if value == "mixed_legibility":
        return None
    for level in range(4):
        if f"_{level}_" in value:
            return level
    return None


def lenient_label(field: str, human: Any, predicted: Any) -> bool:
    if exact_label(field, human, predicted):
        return True
    h = normalize_value(human, field); p = normalize_value(predicted, field)
    if h is None or p is None:
        return False
    if field in ORDERS and h in ORDERS[field] and p in ORDERS[field]:
        return abs(ORDERS[field].index(h) - ORDERS[field].index(p)) <= 1
    if field == "unique_face_count":
        try:
            return abs(int(h) - int(p)) <= 1
        except (TypeError, ValueError):
            return False
    if field in {"age_composition", "gender_presentation_composition"}:
        return composition_family(h) == composition_family(p)
    if field == "expression_legibility_distribution":
        hl = group_legibility_level(h); pl = group_legibility_level(p)
        return hl is not None and pl is not None and abs(hl - pl) <= 1
    return False


def normalized_box(entity: dict[str, Any], *, face: bool = False) -> list[float] | None:
    key = "face_bbox" if face else "bbox"; key1000 = "face_bbox_1000" if face else "bbox_1000"
    raw = entity.get(key)
    scale = 1.0
    if not (isinstance(raw, list) and len(raw) == 4):
        raw = entity.get(key1000); scale = 1000.0
    if not (isinstance(raw, list) and len(raw) == 4):
        return None
    try:
        box = [float(value) / scale for value in raw]
    except (TypeError, ValueError):
        return None
    return box if box[0] < box[2] and box[1] < box[3] else None


def area(box: list[float] | None) -> float:
    return 0.0 if not box else max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def overlap(a: list[float] | None, b: list[float] | None) -> tuple[float, float]:
    if not a or not b:
        return 0.0, 0.0
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = area(a) + area(b) - inter; smaller = min(area(a), area(b))
    return (inter / union if union else 0.0, inter / smaller if smaller else 0.0)


def ad_item(image_id: str, ad: dict[str, Any]) -> Item:
    ad_id = ad.get("ad_id") or ad.get("advertisement_id")
    return Item(image_id, str(ad_id) if ad_id else None, str(ad_id) if ad_id else None, ad, normalized_box(ad))


def person_items(image_id: str, ad: dict[str, Any]) -> list[Item]:
    ad_id = ad.get("ad_id") or ad.get("advertisement_id"); out = []
    for person in ad.get("people") or []:
        display = dict(person)
        if display.get("depiction_type") is None:
            display["depiction_type"] = ad.get("depiction_type")
        out.append(Item(image_id, str(ad_id), person.get("person_id"), display, normalized_box(person, face=True)))
    return out


def group_items(image_id: str, ad: dict[str, Any]) -> list[Item]:
    ad_id = ad.get("ad_id") or ad.get("advertisement_id")
    return [Item(image_id, str(ad_id), group.get("group_id"), group, normalized_box(group)) for group in ad.get("groups") or []]


def all_items(image_id: str, annotation: dict[str, Any], entity: str) -> list[Item]:
    ads = annotation.get("advertisements") or []
    if entity == "ads":
        return [ad_item(image_id, ad) for ad in ads]
    fn = person_items if entity == "people" else group_items
    return [item for ad in ads for item in fn(image_id, ad)]


def pair_items(human: list[Item], predicted: list[Item], entity: str, mode: str) -> list[Pair]:
    candidates = []
    threshold = (STRICT_THRESHOLDS if mode == "strict" else LENIENT_THRESHOLDS)[entity]
    for hi, h in enumerate(human):
        for pi, p in enumerate(predicted):
            iou, contained = overlap(h.box, p.box)
            accepted = iou >= threshold or (mode == "lenient" and contained >= CONTAINMENT_THRESHOLD and iou >= MIN_CONTAINED_IOU)
            if accepted:
                score = iou if mode == "strict" else max(iou, contained * .19)
                candidates.append((score, iou, contained, hi, pi, h, p))
    candidates.sort(key=lambda row: row[0], reverse=True)
    used_h: set[int] = set(); used_p: set[int] = set(); pairs = []
    for _score, iou, contained, hi, pi, h, p in candidates:
        if hi in used_h or pi in used_p:
            continue
        used_h.add(hi); used_p.add(pi); pairs.append(Pair(h, p, iou, contained))
    return pairs


def optimal_pair_items(
    human: list[Item],
    predicted: list[Item],
    entity: str,
    mode: str,
    threshold: float | None = None,
) -> list[Pair]:
    """Maximum-cardinality, then maximum-spatial-score bipartite matching.

    A small dependency-free min-cost max-flow solver is used because the
    standalone runtime does not require SciPy.  Semantic labels are never part
    of the assignment objective.
    """
    candidates: list[tuple[int, int, int, float, float]] = []
    accepted_threshold = threshold if threshold is not None else (STRICT_THRESHOLDS if mode == "strict" else LENIENT_THRESHOLDS)[entity]
    for hi, h in enumerate(human):
        for pi, p in enumerate(predicted):
            iou, contained = overlap(h.box, p.box)
            accepted = iou >= accepted_threshold
            if threshold is None and mode == "lenient":
                accepted = accepted or (contained >= CONTAINMENT_THRESHOLD and iou >= MIN_CONTAINED_IOU)
            if not accepted:
                continue
            score = iou if threshold is not None or mode == "strict" else max(iou, contained * .19)
            # Integer lexicographic weight: established spatial score, then IoU,
            # then containment. Cardinality is handled by sending maximum flow.
            weight = round(score * 1_000_000_000) * 1_000_000_000 + round(iou * 1_000_000) * 1_000 + round(contained * 1_000)
            candidates.append((hi, pi, weight, iou, contained))
    if not candidates:
        return []

    source = 0
    human_start = 1
    predicted_start = human_start + len(human)
    sink = predicted_start + len(predicted)
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def add_edge(start: int, end: int, capacity: int, cost: int) -> int:
        forward_index = len(graph[start])
        reverse_index = len(graph[end])
        graph[start].append([end, reverse_index, capacity, cost])
        graph[end].append([start, forward_index, 0, -cost])
        return forward_index

    for hi in range(len(human)):
        add_edge(source, human_start + hi, 1, 0)
    for pi in range(len(predicted)):
        add_edge(predicted_start + pi, sink, 1, 0)
    references = []
    for hi, pi, weight, iou, contained in candidates:
        node = human_start + hi
        edge_index = add_edge(node, predicted_start + pi, 1, -weight)
        references.append((node, edge_index, hi, pi, iou, contained))

    # Successive shortest augmenting paths with Bellman-Ford support the
    # negative forward costs and residual reassignment edges.
    while True:
        distance: list[int | None] = [None] * len(graph)
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        distance[source] = 0
        for _ in range(len(graph) - 1):
            changed = False
            for node, edges in enumerate(graph):
                if distance[node] is None:
                    continue
                for edge_index, edge in enumerate(edges):
                    end, _reverse, capacity, cost = edge
                    if capacity <= 0:
                        continue
                    proposed = distance[node] + cost
                    if distance[end] is None or proposed < distance[end]:
                        distance[end] = proposed
                        previous[end] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            break
        node = sink
        while node != source:
            start, edge_index = previous[node]  # type: ignore[misc]
            edge = graph[start][edge_index]
            edge[2] -= 1
            graph[node][edge[1]][2] += 1
            node = start

    pairs = []
    for node, edge_index, hi, pi, iou, contained in references:
        if graph[node][edge_index][2] == 0:
            pairs.append((hi, Pair(human[hi], predicted[pi], iou, contained)))
    return [pair for _hi, pair in sorted(pairs, key=lambda value: value[0])]


def field_summary(pairs: list[Pair], fields: list[str], human_total_items: list[Item]) -> dict[str, Any]:
    output = {}
    for field in fields:
        exact = lenient = assessed = missing_one = 0; confusion: Counter[tuple[str, str]] = Counter()
        for pair in pairs:
            h = pair.human.data.get(field); p = pair.predicted.data.get(field)
            hn = normalize_value(h, field); pn = normalize_value(p, field)
            if hn is None and pn is None:
                continue
            assessed += 1; missing_one += int((hn is None) != (pn is None))
            exact += int(exact_label(field, h, p)); lenient += int(lenient_label(field, h, p))
            confusion[(str(hn), str(pn))] += 1
        gold_assessed = sum(normalize_value(item.data.get(field), field) is not None for item in human_total_items)
        output[field] = {
            "matched_assessed": assessed, "exact_matches": exact, "exact_accuracy": exact / assessed if assessed else None,
            "lenient_matches": lenient, "lenient_accuracy": lenient / assessed if assessed else None,
            "gold_assessed_total": gold_assessed,
            "exact_end_to_end_recall": exact / gold_assessed if gold_assessed else None,
            "lenient_end_to_end_recall": lenient / gold_assessed if gold_assessed else None,
            "one_missing": missing_one,
            "top_confusions": [{"human": h, "predicted": p, "n": n} for (h, p), n in confusion.most_common(6) if h != p],
        }
    return output


def detection_summary(human_total: int, predicted_total: int, matched: int) -> dict[str, Any]:
    if human_total == 0 and predicted_total == 0:
        return {"human": 0, "predicted": 0, "matched": 0, "precision": None, "recall": None, "f1": None}
    precision = matched / predicted_total if predicted_total else 0.0
    recall = matched / human_total if human_total else 0.0
    return {"human": human_total, "predicted": predicted_total, "matched": matched, "precision": precision, "recall": recall, "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0}


def evaluate_scope(human: dict[str, dict[str, Any]], predicted: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    overlap_ids = [image_id for image_id in ids if image_id in human and image_id in predicted]
    result: dict[str, Any] = {"requested_pages": len(ids), "evaluated_pages": len(overlap_ids), "missing_gold": [i for i in ids if i not in human], "missing_prediction": [i for i in ids if i not in predicted], "match_modes": {}}
    page_pairs = []
    for image_id in overlap_ids:
        hpage = human[image_id].get("page") or {}; ppage = predicted[image_id].get("page") or {}
        page_pairs.append(Pair(Item(image_id, None, image_id, hpage, None), Item(image_id, None, image_id, ppage, None), 0, 0))
    result["page_fields"] = field_summary(page_pairs, PAGE_FIELDS, [pair.human for pair in page_pairs])
    for mode in ["strict", "lenient"]:
        pairs: dict[str, list[Pair]] = {"ads": [], "people": [], "groups": []}
        totals_h = Counter(); totals_p = Counter(); all_h: dict[str, list[Item]] = defaultdict(list)
        for image_id in overlap_ids:
            h_ann = human[image_id]; p_ann = predicted[image_id]
            h_ads = all_items(image_id, h_ann, "ads"); p_ads = all_items(image_id, p_ann, "ads")
            ad_pairs = pair_items(h_ads, p_ads, "ads", mode); pairs["ads"].extend(ad_pairs)
            for entity in ["ads", "people", "groups"]:
                h_items = all_items(image_id, h_ann, entity); p_items = all_items(image_id, p_ann, entity)
                totals_h[entity] += len(h_items); totals_p[entity] += len(p_items); all_h[entity].extend(h_items)
            for ad_pair in ad_pairs:
                pairs["people"].extend(pair_items(person_items(image_id, ad_pair.human.data), person_items(image_id, ad_pair.predicted.data), "people", mode))
                pairs["groups"].extend(pair_items(group_items(image_id, ad_pair.human.data), group_items(image_id, ad_pair.predicted.data), "groups", mode))
        mode_result = {
            "detection": {entity: detection_summary(totals_h[entity], totals_p[entity], len(pairs[entity])) for entity in ["ads", "people", "groups"]},
            "mean_matched_iou": {entity: (statistics.fmean(pair.iou for pair in pairs[entity]) if pairs[entity] else None) for entity in ["ads", "people", "groups"]},
            "fields": {
                "ads": field_summary(pairs["ads"], AD_FIELDS, all_h["ads"]),
                "people": field_summary(pairs["people"], PERSON_FIELDS, all_h["people"]),
                "groups": field_summary(pairs["groups"], GROUP_FIELDS, all_h["groups"]),
            },
        }
        no_group_ids = [image_id for image_id in overlap_ids if len(all_items(image_id, human[image_id], "groups")) == 0]
        no_group_set = set(no_group_ids)
        pred_group_counts = {image_id: len(all_items(image_id, predicted[image_id], "groups")) for image_id in no_group_ids}
        h_people = sum(len(all_items(i, human[i], "people")) for i in no_group_ids)
        p_people = sum(len(all_items(i, predicted[i], "people")) for i in no_group_ids)
        matched_people = sum(pair.human.image_id in no_group_set for pair in pairs["people"])
        matched_no_group_ads = [pair for pair in pairs["ads"] if not group_items(pair.human.image_id, pair.human.data)]
        no_group_pairs = {
            entity: [pair for pair in pairs[entity] if pair.human.image_id in no_group_set]
            for entity in ["ads", "people"]
        }
        no_group_h = {
            entity: [item for image_id in no_group_ids for item in all_items(image_id, human[image_id], entity)]
            for entity in ["ads", "people"]
        }
        no_group_p = {
            entity: [item for image_id in no_group_ids for item in all_items(image_id, predicted[image_id], entity)]
            for entity in ["ads", "people"]
        }
        no_group_page_pairs = [pair for pair in page_pairs if pair.human.image_id in no_group_set]
        mode_result["human_no_group"] = {
            "pages": len(no_group_ids),
            "pages_predicted_group_free": sum(count == 0 for count in pred_group_counts.values()),
            "page_group_free_specificity": sum(count == 0 for count in pred_group_counts.values()) / len(no_group_ids) if no_group_ids else None,
            "false_predicted_groups": sum(pred_group_counts.values()),
            "pages_with_false_group": [image_id for image_id, count in pred_group_counts.items() if count],
            "person_detection": detection_summary(h_people, p_people, matched_people),
            "matched_gold_group_free_ads": len(matched_no_group_ads),
            "matched_ads_predicted_group_free": sum(not group_items(pair.predicted.image_id, pair.predicted.data) for pair in matched_no_group_ads),
            "false_groups_in_matched_gold_group_free_ads": sum(len(group_items(pair.predicted.image_id, pair.predicted.data)) for pair in matched_no_group_ads),
            "standalone": {
                "detection": {
                    entity: detection_summary(len(no_group_h[entity]), len(no_group_p[entity]), len(no_group_pairs[entity]))
                    for entity in ["ads", "people"]
                },
                "mean_matched_iou": {
                    entity: (statistics.fmean(pair.iou for pair in no_group_pairs[entity]) if no_group_pairs[entity] else None)
                    for entity in ["ads", "people"]
                },
                "page_fields": field_summary(no_group_page_pairs, PAGE_FIELDS, [pair.human for pair in no_group_page_pairs]),
                "fields": {
                    "ads": field_summary(no_group_pairs["ads"], AD_FIELDS, no_group_h["ads"]),
                    "people": field_summary(no_group_pairs["people"], PERSON_FIELDS, no_group_h["people"]),
                },
            },
        }
        group_positive_ids = [image_id for image_id in overlap_ids if len(all_items(image_id, human[image_id], "groups")) > 0]
        group_positive_set = set(group_positive_ids)
        group_positive_pairs = {
            entity: [pair for pair in pairs[entity] if pair.human.image_id in group_positive_set]
            for entity in ["ads", "people", "groups"]
        }
        group_positive_h = {
            entity: [item for image_id in group_positive_ids for item in all_items(image_id, human[image_id], entity)]
            for entity in ["ads", "people", "groups"]
        }
        group_positive_p = {
            entity: [item for image_id in group_positive_ids for item in all_items(image_id, predicted[image_id], entity)]
            for entity in ["ads", "people", "groups"]
        }
        mode_result["human_group_positive"] = {
            "pages": len(group_positive_ids),
            "pages_with_any_predicted_group": sum(bool(all_items(i, predicted[i], "groups")) for i in group_positive_ids),
            "page_group_presence_recall": sum(bool(all_items(i, predicted[i], "groups")) for i in group_positive_ids) / len(group_positive_ids) if group_positive_ids else None,
            "standalone_detection": {
                entity: detection_summary(len(group_positive_h[entity]), len(group_positive_p[entity]), len(group_positive_pairs[entity]))
                for entity in ["ads", "people", "groups"]
            },
        }
        result["match_modes"][mode] = mode_result
    return result

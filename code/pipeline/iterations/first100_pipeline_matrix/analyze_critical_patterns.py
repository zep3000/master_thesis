"""Quantify patterns surfaced by the manual visual audit.

This script is evaluation-only. It measures routing performance conditional on
advertisement/face retention and decomposes expression errors into legibility,
missing-value, size, and depiction-type effects. It never participates in
inference.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import evaluate as ev


HERE = Path(__file__).resolve().parent
RUN_NAME = "first100_frozen_v1"
OUT = HERE / "evaluation" / RUN_NAME / "visual_audit"
PIPELINES = ["p1", "p2", "p3", "p4"]
LEG_ORDER = ev.ORDERS["face_expression_legibility"]


def pct(n: int | float, d: int | float) -> str:
    return "n/a" if not d else f"{100 * n / d:.1f}%"


def md_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def matched_pairs(image_id: str, human: dict[str, Any], predicted: dict[str, Any], entity: str) -> tuple[list[ev.Pair], list[ev.Pair]]:
    h_ads = ev.all_items(image_id, human, "ads")
    p_ads = ev.all_items(image_id, predicted, "ads")
    ad_pairs = ev.pair_items(h_ads, p_ads, "ads", "strict")
    pairs: list[ev.Pair] = []
    getter = ev.person_items if entity == "people" else ev.group_items
    for ad_pair in ad_pairs:
        pairs.extend(ev.pair_items(getter(image_id, ad_pair.human.data), getter(image_id, ad_pair.predicted.data), entity, "strict"))
    return ad_pairs, pairs


def page_gold_count_band(annotation: dict[str, Any]) -> str:
    bands = [str(ad.get("face_depiction_count_band")) for ad in annotation.get("advertisements") or []]
    return "+".join(bands) if bands else "no_ad"


def group_route_rows(gold: dict[str, dict[str, Any]], predictions: dict[str, dict[str, dict[str, Any]]], ids: list[str]) -> list[dict[str, Any]]:
    rows = []
    for pipeline in PIPELINES:
        for image_id in ids:
            human = gold[image_id]; predicted = predictions[pipeline][image_id]
            ad_pairs, person_pairs = matched_pairs(image_id, human, predicted, "people")
            _ads, group_pairs = matched_pairs(image_id, human, predicted, "groups")
            gold_people = len(ev.all_items(image_id, human, "people"))
            pred_people = len(ev.all_items(image_id, predicted, "people"))
            gold_groups = len(ev.all_items(image_id, human, "groups"))
            pred_groups = len(ev.all_items(image_id, predicted, "groups"))
            rows.append({
                "pipeline": pipeline,
                "image_id": image_id,
                "gold_count_band": page_gold_count_band(human),
                "gold_ads": len(ev.all_items(image_id, human, "ads")),
                "predicted_ads": len(ev.all_items(image_id, predicted, "ads")),
                "matched_ads": len(ad_pairs),
                "gold_people": gold_people,
                "predicted_people": pred_people,
                "matched_people": len(person_pairs),
                "gold_groups": gold_groups,
                "predicted_groups": pred_groups,
                "matched_groups": len(group_pairs),
                "human_group_free": gold_groups == 0,
                "predicted_group_free": pred_groups == 0,
            })
    return rows


def face_size_bin(box: list[float] | None) -> str:
    if not box:
        return "missing"
    height = box[3] - box[1]
    if height < .035:
        return "tiny_<3.5%h"
    if height < .07:
        return "small_3.5-7%h"
    if height < .14:
        return "medium_7-14%h"
    return "large_>=14%h"


def relation(human: Any, predicted: Any) -> str:
    h = ev.normalize_value(human, "face_expression_legibility")
    p = ev.normalize_value(predicted, "face_expression_legibility")
    if h not in LEG_ORDER:
        return "human_unordered"
    if p not in LEG_ORDER:
        return "predicted_missing"
    delta = LEG_ORDER.index(p) - LEG_ORDER.index(h)
    return "equal" if delta == 0 else ("under" if delta < 0 else "over")


def output_state(value: Any) -> str:
    if value is None:
        return "null"
    if value == "not_assessable":
        return "not_assessable"
    return "substantive"


def expression_rows(gold: dict[str, dict[str, Any]], predictions: dict[str, dict[str, dict[str, Any]]], ids: list[str]) -> list[dict[str, Any]]:
    rows = []
    for pipeline in PIPELINES:
        for image_id in ids:
            _ad_pairs, pairs = matched_pairs(image_id, gold[image_id], predictions[pipeline][image_id], "people")
            for pair in pairs:
                h, p = pair.human.data, pair.predicted.data
                rows.append({
                    "pipeline": pipeline,
                    "image_id": image_id,
                    "human_person_id": pair.human.item_id,
                    "predicted_person_id": pair.predicted.item_id,
                    "iou": pair.iou,
                    "face_size_bin": face_size_bin(pair.human.box),
                    "human_depiction_type": h.get("depiction_type"),
                    "human_orientation": h.get("face_orientation"),
                    "human_legibility": ev.normalize_value(h.get("face_expression_legibility"), "face_expression_legibility"),
                    "predicted_legibility": ev.normalize_value(p.get("face_expression_legibility"), "face_expression_legibility"),
                    "legibility_relation": relation(h.get("face_expression_legibility"), p.get("face_expression_legibility")),
                    "human_gaze": h.get("gaze_target"),
                    "predicted_gaze": p.get("gaze_target"),
                    "gaze_output_state": output_state(p.get("gaze_target")),
                    "gaze_exact": ev.exact_label("gaze_target", h.get("gaze_target"), p.get("gaze_target")),
                    "human_smile": h.get("smile_present"),
                    "predicted_smile": p.get("smile_present"),
                    "smile_output_state": output_state(p.get("smile_present")),
                    "smile_exact": ev.exact_label("smile_present", h.get("smile_present"), p.get("smile_present")),
                    "human_intensity": h.get("smile_intensity"),
                    "predicted_intensity": p.get("smile_intensity"),
                })
    return rows


def group_expression_rows(gold: dict[str, dict[str, Any]], predictions: dict[str, dict[str, dict[str, Any]]], ids: list[str]) -> list[dict[str, Any]]:
    rows = []
    for pipeline in PIPELINES:
        for image_id in ids:
            _ad_pairs, pairs = matched_pairs(image_id, gold[image_id], predictions[pipeline][image_id], "groups")
            for pair in pairs:
                h, p = pair.human.data, pair.predicted.data
                hleg = h.get("expression_legibility_distribution"); pleg = p.get("expression_legibility_distribution")
                hl = ev.group_legibility_level(hleg); pl = ev.group_legibility_level(pleg)
                rel = "unranked"
                if hl is not None and pl is not None:
                    rel = "equal" if hl == pl else ("under" if pl < hl else "over")
                rows.append({
                    "pipeline": pipeline,
                    "image_id": image_id,
                    "human_group_id": pair.human.item_id,
                    "predicted_group_id": pair.predicted.item_id,
                    "iou": pair.iou,
                    "human_group_type": h.get("group_type"),
                    "human_legibility": hleg,
                    "predicted_legibility": pleg,
                    "legibility_level_relation": rel,
                    "human_gaze": h.get("dominant_gaze"),
                    "predicted_gaze": p.get("dominant_gaze"),
                    "human_smile_prevalence": h.get("smile_prevalence"),
                    "predicted_smile_prevalence": p.get("smile_prevalence"),
                    "human_smile_intensity": h.get("dominant_smile_intensity"),
                    "predicted_smile_intensity": p.get("dominant_smile_intensity"),
                })
    return rows


def count_table(rows: list[dict[str, Any]], pipeline: str, field: str) -> Counter[str]:
    return Counter(str(row[field]) for row in rows if row["pipeline"] == pipeline)


def report_markdown(route_rows: list[dict[str, Any]], expr_rows: list[dict[str, Any]], group_expr_rows: list[dict[str, Any]]) -> str:
    lines = ["# Quantitative patterns behind the visual audit", "", "All metrics are evaluation-only and use strict spatial matches.", ""]

    lines.extend(["## Group routing with retention controls", ""])
    ng_rows = []
    for pipeline in PIPELINES:
        rows = [row for row in route_rows if row["pipeline"] == pipeline and row["human_group_free"]]
        gold_ad_rows = [row for row in rows if row["gold_ads"] > 0]
        matched_ad_rows = [row for row in rows if row["matched_ads"] > 0]
        ng_rows.append([
            pipeline.upper(), len(rows),
            pct(sum(row["predicted_group_free"] for row in rows), len(rows)),
            pct(sum(row["matched_ads"] > 0 for row in gold_ad_rows), len(gold_ad_rows)),
            pct(sum(row["predicted_group_free"] for row in matched_ad_rows), len(matched_ad_rows)),
            pct(sum(row["matched_people"] for row in rows), sum(row["gold_people"] for row in rows)),
        ])
    lines.extend(md_table(["Pipeline", "No-group pages", "Raw group-free", "Ad retained on gold-ad pages", "Group-free if ad retained", "Person recall"], ng_rows))
    lines.extend(["", "Raw group-free specificity should not be read alone: a missing ad is automatically group-free.", ""])

    boundary_rows = []
    for pipeline in PIPELINES:
        rows = [row for row in route_rows if row["pipeline"] == pipeline and row["human_group_free"] and row["gold_count_band"] in {"7", "8", "9"}]
        boundary_rows.append([
            pipeline.upper(), len(rows), sum(not row["predicted_group_free"] for row in rows),
            pct(sum(row["matched_ads"] > 0 for row in rows), len(rows)),
            pct(sum(row["matched_people"] for row in rows), sum(row["gold_people"] for row in rows)),
        ])
    lines.extend(["## The 7–9-face boundary", ""])
    lines.extend(md_table(["Pipeline", "Gold pages", "False-group pages", "Ad retained", "Person recall"], boundary_rows))
    lines.append("")

    positive_rows = []
    for pipeline in PIPELINES:
        rows = [row for row in route_rows if row["pipeline"] == pipeline and not row["human_group_free"]]
        positive_rows.append([
            pipeline.upper(), len(rows),
            pct(sum(row["predicted_groups"] > 0 for row in rows), len(rows)),
            pct(sum(row["matched_ads"] > 0 for row in rows), len(rows)),
            pct(sum(row["matched_groups"] for row in rows), sum(row["gold_groups"] for row in rows)),
        ])
    lines.extend(["## Human group-positive pages", ""])
    lines.extend(md_table(["Pipeline", "Pages", "Any group predicted", "Ad retained", "Strict group-box recall"], positive_rows))
    lines.append("")

    lines.extend(["## Individual expression legibility direction", ""])
    rel_rows = []
    for pipeline in PIPELINES:
        counts = count_table(expr_rows, pipeline, "legibility_relation")
        total = sum(counts.values())
        rel_rows.append([pipeline.upper(), total, pct(counts["under"], total), pct(counts["equal"], total), pct(counts["over"], total), pct(counts["predicted_missing"], total)])
    lines.extend(md_table(["Pipeline", "Matched people", "Under", "Exact", "Over", "Missing"], rel_rows))
    lines.append("")

    for pipeline in ["p2", "p4"]:
        lines.extend([f"### {pipeline.upper()} by human legibility", ""])
        rows = [row for row in expr_rows if row["pipeline"] == pipeline]
        matrix = Counter((row["human_legibility"], row["predicted_legibility"]) for row in rows)
        matrix_rows = []
        for human_leg in LEG_ORDER:
            total = sum(matrix[(human_leg, predicted_leg)] for predicted_leg in LEG_ORDER)
            matrix_rows.append([human_leg, total] + [matrix[(human_leg, predicted_leg)] for predicted_leg in LEG_ORDER])
        lines.extend(md_table(["Human", "n", *LEG_ORDER], matrix_rows)); lines.append("")

    lines.extend(["## Legibility by face size", ""])
    size_order = ["tiny_<3.5%h", "small_3.5-7%h", "medium_7-14%h", "large_>=14%h"]
    size_rows = []
    for pipeline in ["p2", "p4"]:
        for size in size_order:
            rows = [row for row in expr_rows if row["pipeline"] == pipeline and row["face_size_bin"] == size]
            size_rows.append([pipeline.upper(), size, len(rows), pct(sum(row["legibility_relation"] == "under" for row in rows), len(rows)), pct(sum(row["legibility_relation"] == "equal" for row in rows), len(rows))])
    lines.extend(md_table(["Pipeline", "Gold face height", "n", "Under", "Exact"], size_rows)); lines.append("")

    lines.extend(["## Legibility by depiction type", ""])
    depiction_rows = []
    for pipeline in ["p2", "p4"]:
        depictions = Counter(row["human_depiction_type"] for row in expr_rows if row["pipeline"] == pipeline)
        for depiction, count in depictions.most_common():
            rows = [row for row in expr_rows if row["pipeline"] == pipeline and row["human_depiction_type"] == depiction]
            depiction_rows.append([pipeline.upper(), depiction, count, pct(sum(row["legibility_relation"] == "under" for row in rows), count), pct(sum(row["legibility_relation"] == "equal" for row in rows), count)])
    lines.extend(md_table(["Pipeline", "Human depiction", "n", "Under", "Exact"], depiction_rows)); lines.append("")

    lines.extend(["## Legibility by face orientation", ""])
    orientation_rows = []
    for pipeline in ["p2", "p4"]:
        orientations = Counter(row["human_orientation"] for row in expr_rows if row["pipeline"] == pipeline)
        for orientation, count in orientations.most_common():
            rows = [row for row in expr_rows if row["pipeline"] == pipeline and row["human_orientation"] == orientation]
            orientation_rows.append([pipeline.upper(), orientation, count, pct(sum(row["legibility_relation"] == "under" for row in rows), count), pct(sum(row["legibility_relation"] == "equal" for row in rows), count)])
    lines.extend(md_table(["Pipeline", "Human orientation", "n", "Under", "Exact"], orientation_rows)); lines.append("")

    lines.extend(["## Missing-value cascade", ""])
    cascade_rows = []
    for pipeline in PIPELINES:
        rows = [row for row in expr_rows if row["pipeline"] == pipeline]
        gold_gaze = [row for row in rows if row["human_gaze"] not in {None, "not_assessable"}]
        gold_smile = [row for row in rows if row["human_smile"] in {"yes", "no"}]
        cascade_rows.append([
            pipeline.upper(),
            len(gold_gaze), pct(sum(row["gaze_output_state"] == "null" for row in gold_gaze), len(gold_gaze)), pct(sum(row["gaze_output_state"] == "not_assessable" for row in gold_gaze), len(gold_gaze)), pct(sum(row["gaze_exact"] for row in gold_gaze), len(gold_gaze)),
            len(gold_smile), pct(sum(row["smile_output_state"] == "null" for row in gold_smile), len(gold_smile)), pct(sum(row["smile_output_state"] == "not_assessable" for row in gold_smile), len(gold_smile)), pct(sum(row["smile_exact"] for row in gold_smile), len(gold_smile)),
        ])
    lines.extend(md_table(["Pipeline", "Gold gaze n", "Gaze null", "Gaze N/A", "Gaze exact", "Gold smile n", "Smile null", "Smile N/A", "Smile exact"], cascade_rows)); lines.append("")

    lines.extend(["## Downstream fields by predicted legibility", ""])
    gate_rows = []
    for pipeline in ["p2", "p4"]:
        for predicted_legibility in LEG_ORDER:
            rows = [row for row in expr_rows if row["pipeline"] == pipeline and row["predicted_legibility"] == predicted_legibility]
            gaze_rows = [row for row in rows if row["human_gaze"] not in {None, "not_assessable"}]
            smile_rows = [row for row in rows if row["human_smile"] in {"yes", "no"}]
            gate_rows.append([
                pipeline.upper(), predicted_legibility, len(rows),
                pct(sum(row["gaze_output_state"] == "null" for row in gaze_rows), len(gaze_rows)), pct(sum(row["gaze_exact"] for row in gaze_rows), len(gaze_rows)),
                pct(sum(row["smile_output_state"] == "null" for row in smile_rows), len(smile_rows)), pct(sum(row["smile_output_state"] == "not_assessable" for row in smile_rows), len(smile_rows)), pct(sum(row["smile_exact"] for row in smile_rows), len(smile_rows)),
            ])
    lines.extend(md_table(["Pipeline", "Predicted legibility", "n", "Gaze null", "Gaze exact", "Smile null", "Smile N/A", "Smile exact"], gate_rows)); lines.append("")

    lines.extend(["## Group expression direction", ""])
    group_rows = []
    for pipeline in PIPELINES:
        rows = [row for row in group_expr_rows if row["pipeline"] == pipeline]
        relations = Counter(row["legibility_level_relation"] for row in rows)
        group_rows.append([pipeline.upper(), len(rows), pct(relations["under"], len(rows)), pct(relations["equal"], len(rows)), pct(relations["over"], len(rows)), pct(relations["unranked"], len(rows))])
    lines.extend(md_table(["Pipeline", "Matched groups", "Under", "Same level", "Over", "Unranked mixed"], group_rows)); lines.append("")
    return "\n".join(lines)


def main() -> None:
    manifest = json.loads(ev.MANIFEST.read_text(encoding="utf-8"))
    ids = [row["image_id"] for row in manifest["images"][:100]]
    gold = ev.load_gold()
    predictions = {pipeline: ev.load_predictions(RUN_NAME, pipeline) for pipeline in PIPELINES}

    route = group_route_rows(gold, predictions, ids)
    expression = expression_rows(gold, predictions, ids)
    group_expression = group_expression_rows(gold, predictions, ids)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "group_routing_by_page.csv", route)
    write_csv(OUT / "person_expression_matched.csv", expression)
    write_csv(OUT / "group_expression_matched.csv", group_expression)
    report = report_markdown(route, expression, group_expression)
    (OUT / "quantitative_patterns.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(OUT / "quantitative_patterns.md"), "route_rows": len(route), "person_expression_rows": len(expression), "group_expression_rows": len(group_expression)}, indent=2))


if __name__ == "__main__":
    main()

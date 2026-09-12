"""Independent page-bootstrap deltas between development and reserve scopes."""

from __future__ import annotations

import json
import random
from pathlib import Path

import evaluate


HERE = Path(__file__).resolve().parent
EV = evaluate.frozen()
FIELDS = ["face_expression_legibility", "face_orientation", "gaze_target", "smile_present"]


def page_record(cohort: str, image_id: str, human: dict, predicted: dict) -> dict:
    pairs, totals = evaluate.page_pairs(EV, image_id, human[image_id], predicted[image_id], "legacy_strict")
    out = {"detection": {entity: [totals[entity][0], totals[entity][1], len(pairs[entity])] for entity in ["ads", "people", "groups"]}, "fields": {}}
    for field in FIELDS:
        gold = exact = 0
        for pair in pairs["people"]:
            h = EV.normalize_value(pair.human.data.get(field), field); p = EV.normalize_value(pair.predicted.data.get(field), field)
            if h is None: continue
            gold += 1; exact += int(p is not None and EV.exact_label(field, h, p))
        out["fields"][field] = [gold, exact]
    return out


def aggregate(records: list[dict]) -> dict[str, float | None]:
    output = {}
    for entity in ["ads", "people", "groups"]:
        h = sum(r["detection"][entity][0] for r in records); p = sum(r["detection"][entity][1] for r in records); m = sum(r["detection"][entity][2] for r in records)
        precision, recall = evaluate.safe_div(m, p), evaluate.safe_div(m, h)
        output[f"{entity}_f1"] = evaluate.f1(precision, recall)
    for field in FIELDS:
        gold = sum(r["fields"][field][0] for r in records); exact = sum(r["fields"][field][1] for r in records)
        output[field] = evaluate.safe_div(exact, gold)
    return output


def compare(dev: list[dict], reserve: list[dict], repetitions: int, rng: random.Random) -> dict:
    observed_dev, observed_reserve = aggregate(dev), aggregate(reserve)
    distributions = {key: [] for key in observed_dev}
    for _ in range(repetitions):
        d = aggregate([rng.choice(dev) for _ in dev]); r = aggregate([rng.choice(reserve) for _ in reserve])
        for key in distributions:
            if d[key] is not None and r[key] is not None: distributions[key].append(d[key] - r[key])
    output = {}
    for key, values in distributions.items():
        values.sort(); n = len(values)
        output[key] = {
            "development": observed_dev[key], "reserve": observed_reserve[key],
            "development_minus_reserve": (observed_dev[key] - observed_reserve[key]) if observed_dev[key] is not None and observed_reserve[key] is not None else None,
            "delta_ci95": [values[round((n - 1) * .025)], values[round((n - 1) * .975)]] if values else None,
            "bootstrap_probability_development_gt_reserve": sum(value > 0 for value in values) / n if n else None,
        }
    return output


def main() -> int:
    rng = random.Random(20260819); repetitions = 5000
    records = {}
    for cohort in ["difficult", "stratified"]:
        human, predicted = evaluate.gold(cohort), evaluate.predictions(cohort, "final")
        ids = evaluate.selected_ids(cohort)
        records[cohort] = [page_record(cohort, image_id, human, predicted) for image_id in ids]
    result = {
        "repetitions": repetitions,
        "pooled": compare(records["difficult"][:140] + records["stratified"][:140], records["difficult"][140:] + records["stratified"][140:], repetitions, rng),
        "difficult": compare(records["difficult"][:140], records["difficult"][140:], repetitions, rng),
        "stratified": compare(records["stratified"][:140], records["stratified"][140:], repetitions, rng),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

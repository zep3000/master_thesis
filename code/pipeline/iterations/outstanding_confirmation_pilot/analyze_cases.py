#!/usr/bin/env python3
"""Gold-reading case audit for the failed confirmation gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "outstanding_confirmation_pilot"
CANONICAL = ROOT / "qwen_iteration" / "canonical_candidate_280"
sys.path.insert(0, str(CANONICAL))


def load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path); assert spec and spec.loader
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value); return value


ce = load(CANONICAL / "evaluate.py", "confirmation_case_eval")
common = load(CANONICAL / "common.py", "confirmation_case_common")


def main() -> int:
    ev = ce.frozen(); cases = []
    for cohort in ["difficult140", "stratified140"]:
        gold = ce.gold(cohort)
        base_path = CANONICAL / "output" / "assembled" / cohort / "canonical_candidate_v1.jsonl"
        base = {row["image_id"]: row["annotation"] for row in common.rows(base_path) if row.get("ok")}
        audits = common.rows(HERE / "output" / "audit" / f"{cohort}.jsonl")
        audit_by_key = {row["task_key"]: row for row in audits}
        matched = {}
        for image_id in ce.selected_ids(cohort):
            if image_id not in gold or image_id not in base:
                continue
            pairs, _ = ce.page_pairs(ev, image_id, gold[image_id], base[image_id], "legacy_strict")
            for pair in pairs["people"]:
                person_id = pair.predicted.data.get("person_id")
                if person_id:
                    matched[f"{image_id}::{person_id}"] = {
                        "human_role": pair.human.data.get("annotation_role"),
                        "human_box": pair.human.data.get("face_bbox") or pair.human.data.get("face_bbox_1000"),
                        "iou": pair.iou,
                    }
        for key, audit in audit_by_key.items():
            item = {"cohort": cohort, **audit, **matched.get(key, {"human_role": None, "human_box": None, "iou": None})}
            item["composite"] = str(HERE / "output" / "crops" / cohort / f"{key.replace('::', '__')}.jpg")
            cases.append(item)
    counts = Counter((str(case["human_role"]), case["action"]) for case in cases)
    report = {
        "counts_by_matched_human_role_and_action": {f"{role}|{action}": n for (role, action), n in sorted(counts.items())},
        "lost_matched_outstanding": [case for case in cases if case["human_role"] == "outstanding_individual" and case["action"] == "absorbed"],
        "retained_matched_outstanding": [case for case in cases if case["human_role"] == "outstanding_individual" and case["action"] == "retained"],
        "all_cases": cases,
    }
    path = HERE / "evaluation" / "case_audit.json"; path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts_by_matched_human_role_and_action"], "lost": [{k: x[k] for k in ["cohort", "image_id", "person_id", "role", "iou", "composite"]} for x in report["lost_matched_outstanding"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

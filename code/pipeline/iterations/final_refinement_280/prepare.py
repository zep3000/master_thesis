#!/usr/bin/env python3
"""Prepare model-output-derived pilot lists; never reads gold annotations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from common import BASE, HERE, rows


COHORTS = ["difficult140", "stratified140"]


def stable(items: list[dict], salt: str) -> list[dict]:
    return sorted(items, key=lambda x: hashlib.sha256(f"{salt}|{x['task_id']}".encode()).hexdigest())


def main() -> int:
    data_dir = HERE / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    old = json.loads((BASE / "data" / "pilot_ids.json").read_text(encoding="utf-8"))
    (data_dir / "pilot_ids.json").write_text(json.dumps({
        "schema_version": "final_refinement_pilot_ids_v1", "selection": "copied pre-existing critical 30+30; no gold read", "ids": old["ids"]
    }, indent=2) + "\n", encoding="utf-8")

    fer, groups = [], []
    for cohort in COHORTS:
        path = BASE / "output" / cohort / "assembled" / "lean_s2c_direct_gaze.jsonl"
        candidates = []
        for row in rows(path):
            ann = row.get("annotation") or {}
            for ad in ann.get("advertisements") or []:
                for person in ad.get("people") or []:
                    candidates.append({
                        "cohort": cohort, "image_id": row["image_id"],
                        "task_id": f"{row['image_id']}::{person['person_id']}",
                        "ad_id": ad["advertisement_id"], "person_id": person["person_id"],
                        "ad_box": ad["bbox_1000"], "face_box": person["face_bbox_1000"],
                        "direct_legibility": person.get("face_expression_legibility"),
                    })
                for group in ad.get("groups") or []:
                    groups.append({
                        "cohort": cohort, "image_id": row["image_id"],
                        "task_id": f"{row['image_id']}::{group['group_id']}",
                        "ad_id": ad["advertisement_id"], "group_id": group["group_id"],
                        "ad_box": ad["bbox_1000"], "group_box": group["bbox_1000"],
                    })
        moderate = stable([x for x in candidates if x["direct_legibility"] == "2_moderate_legibility"], f"fer-moderate-{cohort}")[:40]
        other = stable([x for x in candidates if x["direct_legibility"] != "2_moderate_legibility"], f"fer-other-{cohort}")[:10]
        fer.extend(moderate + other)
    (data_dir / "fer_pilot.json").write_text(json.dumps({
        "schema_version": "fer_pilot_v1", "selection": "40 direct-moderate + 10 other per cohort by stable hash; no gold read", "tasks": fer
    }, indent=2) + "\n", encoding="utf-8")
    (data_dir / "current_groups.json").write_text(json.dumps({
        "schema_version": "current_group_tasks_v1", "selection": "all groups in current development predictions", "tasks": groups
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"page_pilot": sum(len(v) for v in old["ids"].values()), "fer_pilot": len(fer), "groups": len(groups)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

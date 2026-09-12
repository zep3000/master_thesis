#!/usr/bin/env python3
"""Apply one gold-free geometric/schema validity filter to assembled routes.

Rules:
1. Keep a child entity only when its box center lies inside its parent ad.
2. A qualifying ad must contain at least one returned entity box after rule 1.

No thresholds were tuned against gold; these are direct consequences of the
model-facing contract. IDs and all remaining labels/boxes are unchanged.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "lean_final_280"
ROUTES = ["lean_s2_direct_nogaze", "lean_s2_direct_gaze", "lean_s2_direct_hybrid"]


def inside(child: list[int], parent: list[int]) -> bool:
    cx, cy = (child[0] + child[2]) / 2, (child[1] + child[3]) / 2
    return parent[0] <= cx <= parent[2] and parent[1] <= cy <= parent[3]


def clean(annotation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    result = copy.deepcopy(annotation)
    stats = {"people_removed": 0, "groups_removed": 0, "ads_removed": 0}
    ads = []
    for ad in result.get("advertisements") or []:
        parent = ad["bbox_1000"]
        old_people = ad.get("people") or []
        old_groups = ad.get("groups") or []
        ad["people"] = [person for person in old_people if inside(person["face_bbox_1000"], parent)]
        ad["groups"] = [group for group in old_groups if inside(group["bbox_1000"], parent)]
        stats["people_removed"] += len(old_people) - len(ad["people"])
        stats["groups_removed"] += len(old_groups) - len(ad["groups"])
        if not ad["people"] and not ad["groups"]:
            stats["ads_removed"] += 1
            continue
        if ad["face_depiction_count_band"] in {"10_20", "20_plus"}:
            ad["has_outstanding_individuals"] = "yes" if ad["people"] else "no"
        ads.append(ad)
    result["advertisements"] = ads
    result["page"]["qualifying_ad_count"] = str(len(ads))
    result["qualifying_ad_count"] = str(len(ads))
    if ads:
        result["page"]["no_qualifying_ad_reason"] = None
        result["no_qualifying_ad_reason"] = None
    return result, stats


def main() -> int:
    total = {"people_removed": 0, "groups_removed": 0, "ads_removed": 0}
    for cohort in ["difficult140", "stratified140"]:
        assembled = HERE / "output" / cohort / "assembled"
        for route in ROUTES:
            source = assembled / f"{route}.jsonl"
            output = []
            for line in source.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                row["annotation"], stats = clean(row["annotation"])
                for key, value in stats.items():
                    total[key] += value
                row["route"] = route.replace("lean_s2_", "lean_s2c_")
                row["consistency_filter"] = "center_inside_parent_and_nonempty_ad_v1"
                output.append(row)
            target = assembled / f"{route.replace('lean_s2_', 'lean_s2c_')}.jsonl"
            target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
            print(f"wrote {len(output)} -> {target}")
    print(json.dumps(total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

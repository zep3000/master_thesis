#!/usr/bin/env python3
"""Reapply deterministic validators to saved raw model responses.

This exists so normalization bugs can be repaired without spending API calls.
It never reads human labels and never changes `model_annotation_raw` or usage.
"""

from __future__ import annotations

import json
from pathlib import Path

import run


def rewrite(path: Path, kind: str, style: str | None = None, gaze: bool = False) -> None:
    output = []
    for row in run.jsonl(path):
        if row.get("ok") and isinstance(row.get("model_annotation_raw"), dict):
            raw = row["model_annotation_raw"]
            task = str(row["task_key"])
            if kind == "structure":
                row["model_annotation"] = run.normalize_structure(raw, task)
            elif "group_task_id" in raw:
                row["model_annotation"] = run.normalize_group(raw, task, gaze)
            else:
                row["model_annotation"] = run.normalize_person(raw, task, str(style), gaze)
        output.append(row)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    print(f"renormalized {len(output)} rows: {path}")


def main() -> int:
    for path in run.OUTPUT.glob("*140/structure_*.jsonl"):
        rewrite(path, "structure")
    for path in run.OUTPUT.glob("*140/entities_*.jsonl"):
        name = path.stem
        style = "ordinal" if "_ordinal_" in name else "direct"
        gaze = name.endswith("_gaze") and not name.endswith("_nogaze")
        rewrite(path, "entity", style, gaze)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

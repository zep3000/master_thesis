"""Revalidate saved raw annotations after non-semantic parser repairs.

This performs no model calls and never reads gold. It appends derived records
to the stage JSONL, leaving the original call record intact and auditable.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from run import OUTPUT_DIR, append_jsonl, normalize, read_jsonl
from schema_and_prompt import validate_response


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["tiles", "reducers"], default="tiles")
    args = parser.parse_args()
    path = OUTPUT_DIR / ("tile_localization.jsonl" if args.stage == "tiles" else "llm_reducers.jsonl")
    validation_stage = "tile" if args.stage == "tiles" else "reducer"
    appended = 0
    already = {
        row.get("derived_from_call_sequence")
        for row in read_jsonl(path)
        if row.get("record_kind") == "offline_revalidation"
    }
    for row in read_jsonl(path):
        sequence = row.get("call_sequence")
        raw: dict[str, Any] | None = row.get("model_annotation_raw")
        if not isinstance(raw, dict) or sequence in already:
            continue
        expected_key = str(row["task_key"]).rsplit("::", 1)[0] if args.stage == "tiles" else str(row["task_key"])
        tile_id = str(row["task_key"]).rsplit("::", 1)[1] if args.stage == "tiles" else None
        annotation, actions = normalize(validation_stage, raw, expected_key)
        errors = validate_response(validation_stage, annotation, expected_key)
        if tile_id is not None and annotation.get("tile_id") != tile_id:
            errors.append("tile_id mismatch")
        derived = {
            **row,
            "record_kind": "offline_revalidation",
            "derived_from_call_sequence": sequence,
            "revalidated_at": iso_now(),
            "model_annotation": annotation,
            "normalization_actions": actions,
            "validation_errors": errors,
            "ok": not errors,
        }
        append_jsonl(path, derived)
        appended += 1
    print(json.dumps({"stage": args.stage, "appended": appended, "path": str(path)}, indent=2))


if __name__ == "__main__":
    main()

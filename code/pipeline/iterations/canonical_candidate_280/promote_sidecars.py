#!/usr/bin/env python3
"""Materialize clean, successful-only sidecars without request-failure rows."""

from __future__ import annotations

from common import OUTPUT, rows, write_jsonl


def main() -> int:
    for cohort in ["difficult140", "stratified140"]:
        source = rows(OUTPUT / "duplicate_candidates" / f"{cohort}_conservative_v2.jsonl")
        successful = {row["task_key"]: row for row in source if row.get("ok")}
        values = [successful[key] for key in sorted(successful)]
        write_jsonl(OUTPUT / "duplicate_candidates_promoted" / f"{cohort}.jsonl", values)
        print(cohort, len(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

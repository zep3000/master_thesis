#!/usr/bin/env python3
"""Remove the three known incomplete pages from analysis-bearing run results.

The frozen manifest and operational request ledger remain untouched.  This
keeps the original processing inventory and accounting evidence available
while making the stage, raw-value, normalization, and assembled JSONL files a
complete-case baseline for downstream analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = Path(
    os.environ.get(
        "PIPELINE_RUN_ROOT",
        HERE / "output" / "production" / "full_pages_joined_v1",
    )
)
EXPECTED_MANIFEST_PAGES = 33_047
EXPECTED_BASELINE_PAGES = 33_044
EXPECTED_INCOMPLETE_PAGES = 3
RESULT_GLOBS = (
    "pages/**/*.jsonl",
    "raw_values/**/*.jsonl",
    "normalization/**/*.jsonl",
)
REQUIRED_RESULT_PATHS = {
    "pages/structure_s2.jsonl",
    "pages/entities_s2_direct_gaze.jsonl",
    "pages/assembled/final.jsonl",
    "raw_values/pages.jsonl",
    "normalization/pages.jsonl",
}
PRESERVED_OPERATIONAL_PATHS = (
    "manifest.json",
    "request_ledger.jsonl",
    "range_history.jsonl",
    "run_status.json",
    "checkpoint.json",
)
BASELINE_AUDIT = "analysis_baseline.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL record") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield record


def record_page_id(record: dict[str, Any]) -> str | None:
    value = record.get("image_id") or record.get("task_key")
    if not isinstance(value, str) or not value:
        return None
    return value.split("::", 1)[0]


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def result_paths(run_root: Path) -> list[Path]:
    paths = {
        path.resolve()
        for pattern in RESULT_GLOBS
        for path in run_root.glob(pattern)
        if path.is_file()
    }
    relative = {relative_posix(path, run_root.resolve()) for path in paths}
    missing = sorted(REQUIRED_RESULT_PATHS - relative)
    if missing:
        raise FileNotFoundError(f"required result files are missing: {missing}")
    return sorted(paths, key=lambda path: relative_posix(path, run_root.resolve()))


def manifest_ids(run_root: Path) -> set[str]:
    payload = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    images = payload.get("images")
    if not isinstance(images, list):
        raise ValueError("manifest.json must contain an images array")
    identifiers = [item.get("image_id") for item in images if isinstance(item, dict)]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("every manifest image must have a non-empty string image_id")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("manifest image_id values must be unique")
    return set(identifiers)


def detect_incomplete_pages(run_root: Path, all_ids: set[str]) -> dict[str, str]:
    structures = run_root / "pages" / "structure_s2.jsonl"
    final = run_root / "pages" / "assembled" / "final.jsonl"
    successful_structures = {
        record["task_key"]
        for record in read_jsonl(structures)
        if record.get("ok") and isinstance(record.get("task_key"), str)
    }
    assembled: dict[str, bool] = {}
    for record in read_jsonl(final):
        image_id = record.get("image_id")
        if not isinstance(image_id, str):
            raise ValueError(f"{final}: assembled record lacks image_id")
        if image_id in assembled:
            raise ValueError(f"{final}: duplicate assembled image_id {image_id}")
        assembled[image_id] = bool(record.get("ok"))
    missing_structures = all_ids - successful_structures
    incomplete_assembly = {
        image_id for image_id in all_ids if not assembled.get(image_id, False)
    }
    return {
        image_id: (
            "missing completed structure and assembly"
            if image_id in missing_structures
            else "incomplete entity processing and assembly"
        )
        for image_id in missing_structures | incomplete_assembly
    }


def scan_result(path: Path, excluded: set[str]) -> dict[str, Any]:
    records = removed = unidentified = 0
    removed_by_page = {image_id: 0 for image_id in sorted(excluded)}
    for record in read_jsonl(path):
        records += 1
        image_id = record_page_id(record)
        if image_id is None:
            unidentified += 1
        elif image_id in excluded:
            removed += 1
            removed_by_page[image_id] += 1
    return {
        "records_before": records,
        "records_removed": removed,
        "records_after": records - removed,
        "records_without_page_id": unidentified,
        "removed_by_page": removed_by_page,
        "sha256_before": sha256(path),
    }


def rewrite_result(path: Path, excluded: set[str]) -> None:
    temporary = path.with_name(f".{path.name}.incomplete-page-cleanup.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale cleanup temporary file exists: {temporary}")
    try:
        with path.open("r", encoding="utf-8", newline="") as source, temporary.open(
            "x", encoding="utf-8", newline=""
        ) as destination:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    destination.write(line)
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL record") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                if record_page_id(record) not in excluded:
                    destination.write(line)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_baseline(
    run_root: Path,
    all_ids: set[str],
    excluded: set[str],
    expected_baseline_pages: int,
) -> None:
    retained = all_ids - excluded
    if len(retained) != expected_baseline_pages:
        raise AssertionError(
            f"expected {expected_baseline_pages} retained pages, found {len(retained)}"
        )
    for path in result_paths(run_root):
        remaining = {
            image_id
            for record in read_jsonl(path)
            if (image_id := record_page_id(record)) in excluded
        }
        if remaining:
            raise AssertionError(f"{path}: excluded page records remain: {sorted(remaining)}")

    final = run_root / "pages" / "assembled" / "final.jsonl"
    rows = list(read_jsonl(final))
    final_ids = [record.get("image_id") for record in rows]
    if len(final_ids) != len(set(final_ids)):
        raise AssertionError("assembled baseline contains duplicate image_id values")
    if set(final_ids) != retained:
        missing = sorted(retained - set(final_ids))
        unexpected = sorted(set(final_ids) - retained)
        raise AssertionError(
            f"assembled baseline page mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    if any(not record.get("ok") for record in rows):
        raise AssertionError("assembled baseline contains an unsuccessful page")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale metadata temporary file exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def clean_run(
    run_root: Path,
    *,
    apply: bool,
    expected_manifest_pages: int = EXPECTED_MANIFEST_PAGES,
    expected_baseline_pages: int = EXPECTED_BASELINE_PAGES,
    expected_incomplete_pages: int = EXPECTED_INCOMPLETE_PAGES,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"run root does not exist: {run_root}")

    all_ids = manifest_ids(run_root)
    if len(all_ids) != expected_manifest_pages:
        raise AssertionError(
            f"expected {expected_manifest_pages} manifest pages, found {len(all_ids)}"
        )
    excluded_pages = detect_incomplete_pages(run_root, all_ids)
    excluded_ids = set(excluded_pages)
    if len(excluded_ids) != expected_incomplete_pages:
        raise AssertionError(
            f"expected {expected_incomplete_pages} incomplete pages, "
            f"detected {len(excluded_ids)}"
        )

    paths = result_paths(run_root)
    plans = []
    for path in paths:
        plan = scan_result(path, excluded_ids)
        plan["path"] = relative_posix(path, run_root)
        plans.append(plan)

    if apply:
        for path, plan in zip(paths, plans):
            if plan["records_removed"]:
                rewrite_result(path, excluded_ids)
        verify_baseline(run_root, all_ids, excluded_ids, expected_baseline_pages)
        for path, plan in zip(paths, plans):
            plan["sha256_after"] = sha256(path)
            plan["bytes_after"] = path.stat().st_size

    operational = []
    for relative in PRESERVED_OPERATIONAL_PATHS:
        path = run_root / relative
        if path.exists():
            operational.append(
                {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )

    report = {
        "schema_version": "complete_case_analysis_baseline_v1",
        "applied": apply,
        "source_manifest_pages": len(all_ids),
        "analysis_pages": expected_baseline_pages,
        "excluded_pages": [
            {"image_id": image_id, "reason": excluded_pages[image_id]}
            for image_id in sorted(excluded_ids)
        ],
        "result_files": plans,
        "preserved_operational_files": operational,
    }
    if apply:
        audit_path = run_root / BASELINE_AUDIT
        if any(plan["records_removed"] for plan in plans) or not audit_path.exists():
            write_json_atomic(audit_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="production run root (or set PIPELINE_RUN_ROOT)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="atomically rewrite result JSONL files; omission performs a dry run",
    )
    args = parser.parse_args()
    report = clean_run(args.run_root, apply=args.apply)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

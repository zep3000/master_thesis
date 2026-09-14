"""Build the reviewed public export of row-level human annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PureWindowsPath
from typing import Any


COMMENT_KEYS = {"note", "notes", "identification_note"}
FULL_ISSUE_COLUMNS = [
    "issue_page_identifier",
    "x1_rel",
    "y1_rel",
    "x2_rel",
    "y2_rel",
    "analyzable",
]
ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|/Users/|/home/)", re.IGNORECASE)
ABSOLUTE_PATH_ANYWHERE = re.compile(
    r"(?:(?<![A-Za-z])[A-Za-z]:[\\/](?![\\/])|/Users/|/home/)", re.IGNORECASE
)


def is_comment_key(key: str) -> bool:
    return "comment" in key.lower() or key.lower() in COMMENT_KEYS


def portable_path(value: str, key: str) -> str:
    if not ABSOLUTE_PATH.match(value):
        return value.replace("\\", "/")
    name = PureWindowsPath(value).name
    if key == "manifest_path":
        return f"manifests/{name}"
    return f"source-images/{name}"


def sanitize(value: Any, label: str | None, counts: Counter[str]) -> Any:
    if isinstance(value, list):
        return [sanitize(item, label, counts) for item in value]
    if not isinstance(value, dict):
        return value

    result: dict[str, Any] = {}
    for key, item in value.items():
        if key == "session_code" or key.endswith("assignment_code"):
            counts["assignment_code_removed"] += 1
            continue
        if is_comment_key(key):
            counts["comment_fields_removed"] += 1
            continue
        if key == "assignee_name" and label is not None:
            result[key] = label
            counts["assignee_name_replaced"] += 1
            continue
        if key in {"path", "manifest_path"} and isinstance(item, str):
            replacement = portable_path(item, key)
            counts["paths_made_portable"] += replacement != item
            result[key] = replacement
            continue
        result[key] = sanitize(item, label, counts)
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    path.write_text(content, encoding="utf-8", newline="\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def publish_coder(source: Path, destination: Path, label: str) -> dict[str, Any]:
    records = read_jsonl(source)
    if len(records) != 300:
        raise ValueError(f"Expected 300 records for coder {label}, found {len(records)}")
    original_names = {
        value
        for record in records
        for value in collect_values(record, "assignee_name")
        if isinstance(value, str) and value
    }
    if len(original_names) != 1:
        raise ValueError(f"Coder {label} must have exactly one assignee_name")

    counts: Counter[str] = Counter()
    published = [sanitize(record, label, counts) for record in records]
    validate_sanitized(published, label)
    write_jsonl(destination, published)
    return file_entry(destination, len(published), dict(counts))


def publish_brand(source: Path, destination: Path) -> dict[str, Any]:
    records = read_jsonl(source)
    counts: Counter[str] = Counter()
    published = [sanitize(record, None, counts) for record in records]
    validate_sanitized(published, None)
    write_jsonl(destination, published)
    return file_entry(destination, len(published), dict(counts))


def publish_development_export(
    source: Path,
    destination: Path,
    expected_records: int,
    label: str | None,
    assignment_code: str | None = None,
) -> dict[str, Any]:
    export = json.loads(source.read_text(encoding="utf-8"))
    records = export.get("annotations")
    if not isinstance(records, list):
        raise ValueError(f"Expected an annotations list in {source}")

    counts: Counter[str] = Counter()
    if assignment_code is not None:
        selected = [
            record
            for record in records
            if record.get("assignment_code") == assignment_code
        ]
        counts["other_assignment_records_excluded"] = len(records) - len(selected)
        export["annotations"] = selected
        records = selected
    if len(records) != expected_records:
        raise ValueError(
            f"Expected {expected_records} records in {source}, found {len(records)}"
        )
    if len({str(record.get("image_id")) for record in records}) != expected_records:
        raise ValueError(f"Expected {expected_records} unique image IDs in {source}")

    published = sanitize(export, label, counts)
    validate_sanitized([published], label)
    write_json(destination, published)
    return file_entry(destination, len(records), dict(counts))


def collect_values(value: Any, target_key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == target_key:
                found.append(item)
            found.extend(collect_values(item, target_key))
    elif isinstance(value, list):
        for item in value:
            found.extend(collect_values(item, target_key))
    return found


def validate_sanitized(records: list[dict[str, Any]], label: str | None) -> None:
    serialized = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    if re.search(r'"[^"\\]*assignment_code"\s*:', serialized):
        raise ValueError("assignment_code remains in sanitized output")
    if re.search(r'"session_code"\s*:', serialized):
        raise ValueError("session_code remains in sanitized output")
    if re.search(r'"[^"\\]*comment[^"\\]*"\s*:', serialized, re.IGNORECASE):
        raise ValueError("comment field remains in sanitized output")
    if re.search(r'"(?:notes|identification_note)"\s*:', serialized):
        raise ValueError("note field remains in sanitized output")
    if ABSOLUTE_PATH_ANYWHERE.search(serialized):
        raise ValueError("absolute path remains in sanitized output")
    if label is not None:
        names = set(collect_values(records, "assignee_name"))
        if names != {label}:
            raise ValueError(f"Unexpected assignee_name values for coder {label}")
    elif collect_values(records, "assignee_name"):
        raise ValueError("assignee_name remains in sanitized output")


def publish_full_issue_scans(
    source_dir: Path, destination_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sources = sorted(source_dir.glob("ECON-*_annotations.csv"))
    if not sources:
        raise ValueError(f"No full-issue annotation CSVs found in {source_dir}")

    entries: list[dict[str, Any]] = []
    page_ids: set[str] = set()
    total_rows = 0
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        with source.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FULL_ISSUE_COLUMNS:
                raise ValueError(f"Unexpected columns in {source.name}: {reader.fieldnames}")
            rows = list(reader)
        for row in rows:
            x1, y1, x2, y2 = (float(row[name]) for name in FULL_ISSUE_COLUMNS[1:5])
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f"Invalid relative bounding box in {source.name}")
            if row["analyzable"].strip().lower() not in {
                "true",
                "false",
                "1",
                "0",
                "yes",
                "no",
            }:
                raise ValueError(f"Invalid analyzable value in {source.name}")
            page_ids.add(row["issue_page_identifier"])

        destination = destination_dir / source.name
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=FULL_ISSUE_COLUMNS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        total_rows += len(rows)
        entries.append(file_entry(destination, len(rows), {}))

    if len(sources) != 35 or total_rows != 1599:
        raise ValueError(
            f"Expected 35 files and 1,599 rows, found {len(sources)} files "
            f"and {total_rows} rows"
        )
    return entries, {
        "file_count": len(sources),
        "records": total_rows,
        "unique_page_identifiers": len(page_ids),
    }


def file_entry(path: Path, records: int, transformations: dict[str, int]) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.as_posix(),
        "records": records,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "transformations": transformations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coder-a", type=Path, required=True)
    parser.add_argument("--coder-b", type=Path, required=True)
    parser.add_argument("--coder-c", type=Path, required=True)
    parser.add_argument("--development-difficult", type=Path, required=True)
    parser.add_argument("--development-difficult-code", required=True)
    parser.add_argument("--development-stratified", type=Path, required=True)
    parser.add_argument("--brand", type=Path, required=True)
    parser.add_argument("--full-issue-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    coder_entries = []
    for label, source in (
        ("A", args.coder_a),
        ("B", args.coder_b),
        ("C", args.coder_c),
    ):
        destination = output / "three-coder" / f"coder-{label.lower()}.jsonl"
        entry = publish_coder(source, destination, label)
        entry["path"] = destination.relative_to(output).as_posix()
        coder_entries.append(entry)

    development_entries = []
    for source, filename, expected, label, assignment_code in (
        (
            args.development_difficult,
            "difficult-198.json",
            198,
            None,
            args.development_difficult_code,
        ),
        (args.development_stratified, "stratified-200.json", 200, "D", None),
    ):
        destination = output / "pipeline-development" / filename
        entry = publish_development_export(
            source,
            destination,
            expected_records=expected,
            label=label,
            assignment_code=assignment_code,
        )
        entry["path"] = destination.relative_to(output).as_posix()
        development_entries.append(entry)

    brand_destination = output / "brand-industry" / "annotations.jsonl"
    brand_entry = publish_brand(args.brand, brand_destination)
    brand_entry["path"] = brand_destination.relative_to(output).as_posix()

    full_issue_entries, full_issue_scope = publish_full_issue_scans(
        args.full_issue_dir, output / "full-issue-scans"
    )
    for entry in full_issue_entries:
        entry["path"] = Path(entry["path"]).relative_to(output).as_posix()

    manifest = {
        "schema_version": "human_annotation_publication_v2",
        "sanitization": {
            "assignee_name": (
                "replaced with A, B, or C in the three-coder exports and D in "
                "the named pipeline-development export"
            ),
            "assignment_and_session_codes": "removed",
            "comments_and_notes": "removed",
            "absolute_paths": "replaced with portable source-images/ or manifests/ paths",
            "other_fields": "retained",
        },
        "datasets": [
            {"name": "three-coder validation", "records": 900, "files": coder_entries},
            {
                "name": "pipeline development and evaluation",
                "records": sum(entry["records"] for entry in development_entries),
                "nominal_sample_size": 400,
                "distinct_annotated_pages": 398,
                "development_pages": 280,
                "reserve_pages": 118,
                "files": development_entries,
            },
            {
                "name": "brand and industry verification",
                "records": brand_entry["records"],
                "files": [brand_entry],
            },
            {
                "name": "full-issue face boxes",
                **full_issue_scope,
                "files": full_issue_entries,
            },
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()

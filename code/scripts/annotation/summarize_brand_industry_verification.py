"""Build aggregate results from the reviewed brand/industry annotation export.

The summaries contain no item, image, session, timestamp, note, or bounding-box
fields. Error inventories are grouped label transitions without row IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_LIMIT = 200


def load_records(source: Path) -> list[dict[str, Any]]:
    if source.is_dir():
        paths = sorted(path for path in source.glob("*.json") if path.name != "session.json")
        return [json.loads(path.read_text(encoding="utf-8")) for path in paths]

    if source.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]

    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("annotations"), list):
        return payload["annotations"]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported JSON structure in {source}")


def select_declared_scope(records: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for record in records:
        if record.get("status") != "complete":
            continue
        position = record.get("item_position")
        if not isinstance(position, int):
            raise ValueError("Every completed record must have an integer item_position")
        if position in indexed:
            raise ValueError(f"Duplicate item_position: {position}")
        indexed[position] = record

    missing = sorted(set(range(limit)) - indexed.keys())
    if missing:
        raise ValueError(f"Missing completed records in declared scope: {missing}")
    return [indexed[position] for position in range(limit)]


def metric(name: str, successes: int, denominator: int, basis: str) -> dict[str, Any]:
    return {
        "metric": name,
        "basis": basis,
        "successes": successes,
        "denominator": denominator,
        "percent": round(100 * successes / denominator, 1),
    }


def summarize(records: Iterable[dict[str, Any]], limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    records = list(records)
    scoped = select_declared_scope(records, limit)
    scorable = [record for record in scoped if not record.get("identification_wrong", False)]

    for record in scorable:
        if record.get("category_review") not in {"correct", "incorrect"}:
            raise ValueError("A scorable record lacks a category verdict")
        if record.get("brand_name_review") not in {"correct", "incorrect"}:
            raise ValueError("A scorable record lacks a brand-name verdict")

    category_correct = sum(record["category_review"] == "correct" for record in scorable)
    brand_correct = sum(record["brand_name_review"] == "correct" for record in scorable)
    both_correct = sum(
        record["category_review"] == "correct" and record["brand_name_review"] == "correct"
        for record in scorable
    )

    metrics = [
        metric("Field-scorable identification", len(scorable), len(scoped), "all reviewed items"),
        metric("Industry category agreement", category_correct, len(scorable), "conditional on scorable identification"),
        metric("Brand-name agreement", brand_correct, len(scorable), "conditional on scorable identification"),
        metric("Both fields agree", both_correct, len(scorable), "conditional on scorable identification"),
        metric("Industry category end-to-end", category_correct, len(scoped), "identification failures count as failures"),
        metric("Brand-name end-to-end", brand_correct, len(scoped), "identification failures count as failures"),
        metric("Both fields end-to-end", both_correct, len(scoped), "identification failures count as failures"),
    ]

    industry_errors = Counter()
    brand_errors = Counter()
    for record in scorable:
        original = record.get("original") or {}
        if record["category_review"] == "incorrect":
            key = (
                str(original.get("brand_name") or "[unidentified]"),
                str(original.get("brand_category") or "[missing]"),
                str(record.get("corrected_brand_category") or "[missing]"),
            )
            industry_errors[key] += 1
        if record["brand_name_review"] == "incorrect":
            key = (
                str(original.get("brand_name") or "[unidentified]"),
                str(record.get("corrected_brand_name") or "[missing]"),
            )
            brand_errors[key] += 1

    return {
        "scope": {
            "completed_input_records": sum(record.get("status") == "complete" for record in records),
            "declared_first_n_positions": limit,
            "included_positions": f"0-{limit - 1}",
            "scorable_records": len(scorable),
            "identification_failures": len(scoped) - len(scorable),
            "excluded_completed_records_beyond_scope": sum(
                record.get("status") == "complete"
                and isinstance(record.get("item_position"), int)
                and record["item_position"] >= limit
                for record in records
            ),
        },
        "metrics": metrics,
        "industry_errors": [
            {
                "brand_or_advertiser": key[0],
                "model_category": key[1],
                "verified_category": key[2],
                "count": count,
            }
            for key, count in sorted(industry_errors.items())
        ],
        "brand_errors": [
            {"model_name": key[0], "verified_name": key[1], "count": count}
            for key, count in sorted(brand_errors.items())
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[str]], alignments: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(alignments) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_qmd(path: Path, result: dict[str, Any]) -> None:
    primary = result["metrics"][:4]
    summary_table = markdown_table(
        ["Outcome", "Correct / N", "Agreement"],
        [
            [
                row["metric"],
                f'{row["successes"]} / {row["denominator"]}',
                f'{row["percent"]:.1f}%',
            ]
            for row in primary
        ],
        [":---", "---:", "---:"],
    )
    industry_table = markdown_table(
        ["Brand or advertiser", "Model category", "Corrected category"],
        [
            [row["brand_or_advertiser"], row["model_category"], row["verified_category"]]
            for row in result["industry_errors"]
        ],
        [":---", ":---", ":---"],
    )
    brand_table = markdown_table(
        ["Model-assigned name", "Corrected name"],
        [[row["model_name"], row["verified_name"]] for row in result["brand_errors"]],
        [":---", ":---"],
    )
    content = f"""{summary_table}

: Human verification of the model's advertisement identification, industry category, and brand name. {{#tbl-brand-industry-verification}}

{industry_table}

: All industry-category corrections in the 196 scorable records. {{#tbl-brand-industry-errors}}

{brand_table}

: All brand-name corrections in the 196 scorable records. {{#tbl-brand-name-errors}}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local annotation directory, JSON, or JSONL export")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for public aggregate CSV/JSON files")
    parser.add_argument("--qmd-output", type=Path, help="Optional generated Quarto table fragment")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of zero-based item positions to include")
    args = parser.parse_args()

    result = summarize(load_records(args.source), args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "summary.csv",
        result["metrics"],
        ["metric", "basis", "successes", "denominator", "percent"],
    )
    write_csv(
        args.output_dir / "industry-category-corrections.csv",
        result["industry_errors"],
        ["brand_or_advertiser", "model_category", "verified_category", "count"],
    )
    write_csv(
        args.output_dir / "brand-name-corrections.csv",
        result["brand_errors"],
        ["model_name", "verified_name", "count"],
    )
    (args.output_dir / "scope.json").write_text(
        json.dumps(result["scope"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.qmd_output:
        write_qmd(args.qmd_output, result)


if __name__ == "__main__":
    main()

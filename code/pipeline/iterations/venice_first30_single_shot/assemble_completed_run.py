"""Assemble the immutable initial run and transport-only retries.

Only records that failed before receiving a model response may be replaced. The
replacement must refer to the same manifest entry, image hash, and prompt hash.
The source files are never modified.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"
INITIAL_PATH = OUTPUT_DIR / "venice_first30_single_shot.jsonl"
RETRY_DIR = OUTPUT_DIR / "provider_retries"
COMPLETED_PATH = OUTPUT_DIR / "venice_first30_single_shot.completed.jsonl"
AGGREGATE_PATH = OUTPUT_DIR / "venice_first30_single_shot.completed.aggregate.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    initial = read_jsonl(INITIAL_PATH)
    if len(initial) != 30:
        raise ValueError(f"Expected 30 initial records, found {len(initial)}")

    retries: dict[int, tuple[dict, Path]] = {}
    for path in sorted(RETRY_DIR.glob("retry_index_*.jsonl")):
        records = read_jsonl(path)
        if len(records) != 1:
            raise ValueError(f"Expected one retry record in {path}, found {len(records)}")
        record = records[0]
        index = int(record["manifest_index"])
        if index in retries:
            raise ValueError(f"Duplicate retry for manifest index {index}")
        retries[index] = (record, path)

    assembled: list[dict] = []
    replacements: list[dict] = []
    for original in initial:
        index = int(original["manifest_index"])
        retry_entry = retries.get(index)
        if retry_entry is None:
            assembled.append(original)
            continue

        retry, retry_path = retry_entry
        if original.get("model_annotation_raw") is not None or not original.get("error"):
            raise ValueError(f"Refusing to replace non-transport failure at index {index}")
        if retry.get("model_annotation_raw") is None:
            raise ValueError(f"Retry at index {index} has no parsed model response")

        identity_fields = ("manifest_index", "image_id", "filename", "prompt_sha256")
        for field in identity_fields:
            if original.get(field) != retry.get(field):
                raise ValueError(f"Retry identity mismatch at index {index}: {field}")
        if original["image_info"]["sha256"] != retry["image_info"]["sha256"]:
            raise ValueError(f"Retry image hash mismatch at index {index}")

        assembled.append(retry)
        replacements.append(
            {
                "manifest_index": index,
                "filename": retry["filename"],
                "initial_error": original["error"],
                "replacement_source": str(retry_path),
                "replacement_ok": bool(retry.get("ok")),
            }
        )

    assembled.sort(key=lambda record: int(record["manifest_index"]))
    expected_indices = list(range(30))
    actual_indices = [int(record["manifest_index"]) for record in assembled]
    if actual_indices != expected_indices:
        raise ValueError(f"Expected manifest indices {expected_indices}, found {actual_indices}")

    COMPLETED_PATH.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in assembled),
        encoding="utf-8",
    )

    usage_records = [
        record["response_metadata"]["usage"]
        for record in assembled
        if record.get("response_metadata", {}).get("usage")
    ]
    aggregate = {
        "schema_version": "qwen_venice_first30_completed_aggregate_v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "assembly_policy": "Replace only initial transport failures with matching parsed retries",
        "data_isolation": {
            "human_annotation_inputs_used": False,
            "gold_result_directories_read": [],
            "inputs": [
                str(INITIAL_PATH),
                *[str(path) for _, path in sorted(retries.values(), key=lambda item: item[0]["manifest_index"])],
            ],
        },
        "source_sha256": {
            str(INITIAL_PATH): sha256(INITIAL_PATH),
            **{str(path): sha256(path) for _, path in retries.values()},
        },
        "records": len(assembled),
        "parsed_model_responses": sum(record.get("model_annotation_raw") is not None for record in assembled),
        "strictly_valid": sum(bool(record.get("ok")) for record in assembled),
        "schema_invalid": sum(not bool(record.get("ok")) for record in assembled),
        "unresolved_transport_failures": sum(bool(record.get("error")) for record in assembled),
        "total_prompt_tokens": sum(int(usage.get("prompt_tokens", 0)) for usage in usage_records),
        "total_completion_tokens": sum(int(usage.get("completion_tokens", 0)) for usage in usage_records),
        "total_tokens": sum(int(usage.get("total_tokens", 0)) for usage in usage_records),
        "reported_cost_credits": sum(float(usage.get("cost", 0.0)) for usage in usage_records),
        "transport_replacements": replacements,
        "completed_jsonl": str(COMPLETED_PATH),
    }
    AGGREGATE_PATH.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

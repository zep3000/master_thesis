#!/usr/bin/env python3
"""Read-only live metrics for the prioritized 10k production tranche."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE / "output" / "production" / "full_pages_joined_v1"


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=Path, default=HERE / "data" / "full_pages_main_evaluation_600_priority_after_11000.json")
    parser.add_argument("--evaluation-export", type=Path, default=HERE.parent.parent / "annotation_results" / "karin_economist_decade_face_count_stratified_600_v3_60109424_2026-08-10.json")
    parser.add_argument("--start", type=int, default=11_000)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--resume-sequence", type=int, default=37_765)
    parser.add_argument("--baseline-complete", type=int, default=10_999)
    parser.add_argument("--recent-minutes", type=int, default=45)
    args = parser.parse_args()

    order = json.loads(args.order.read_text(encoding="utf-8"))["image_ids"]
    selected = order[args.start : args.start + args.limit]
    selected_set = set(selected)

    structures = {}
    for row in iter_jsonl(RUN_ROOT / "pages" / "structure_s2.jsonl"):
        if row.get("ok") and row.get("task_key") in selected_set:
            structures[row["task_key"]] = row["model_annotation"]
    successful_entities = {
        row["task_key"]
        for row in iter_jsonl(RUN_ROOT / "pages" / "entities_s2_direct_gaze.jsonl")
        if row.get("ok")
    }

    attempts = []
    success_times = {}
    for row in iter_jsonl(RUN_ROOT / "request_ledger.jsonl"):
        if int(row.get("request_sequence") or 0) < args.resume_sequence:
            continue
        attempts.append(row)
        if row.get("ok"):
            stamp = parse_time(row.get("finished_at"))
            if stamp is not None:
                success_times[row["task_key"]] = stamp

    complete_times = {}
    expected_entity_count = 0
    for image_id, annotation in structures.items():
        expected = []
        for ad in annotation.get("advertisements") or []:
            expected.extend(f"{image_id}::{person['person_id']}" for person in ad.get("people") or [])
            expected.extend(f"{image_id}::{group['group_id']}" for group in ad.get("groups") or [])
        expected_entity_count += len(expected)
        if all(key in successful_entities for key in expected):
            stamps = [success_times.get(image_id)] + [success_times.get(key) for key in expected]
            stamps = [stamp for stamp in stamps if stamp is not None]
            if stamps:
                complete_times[image_id] = max(stamps)

    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(ZoneInfo("Europe/Berlin"))
    first_success = min(success_times.values()) if success_times else None
    elapsed_minutes = max((now_utc - first_success).total_seconds() / 60, 1 / 60) if first_success else 0
    complete = len(complete_times)
    total_rate = complete / elapsed_minutes if elapsed_minutes else 0
    cutoff = now_utc - timedelta(minutes=args.recent_minutes)
    recent_complete = sum(stamp >= cutoff for stamp in complete_times.values())
    recent_rate = recent_complete / args.recent_minutes

    recent_attempts = [row for row in attempts if (parse_time(row.get("finished_at")) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff]
    usage = Counter()
    recent_usage = Counter()
    for row in attempts:
        for key in ["total_tokens", "prompt_tokens", "completion_tokens"]:
            usage[key] += int((row.get("usage") or {}).get(key) or 0)
        usage["cost_microusd"] += round(float((row.get("usage") or {}).get("cost") or 0) * 1_000_000)
        usage["request_body_bytes"] += int(row.get("request_body_bytes") or 0)
        usage["response_body_bytes"] += int(row.get("response_body_bytes") or 0)
    for row in recent_attempts:
        recent_usage["total_tokens"] += int((row.get("usage") or {}).get("total_tokens") or 0)

    eval_export = json.loads(args.evaluation_export.read_text(encoding="utf-8"))
    eval_ids = {row["image_id"] for row in eval_export["annotation_set"]["manifest"]["images"]}
    completed_eval_in_tranche = len(eval_ids & set(complete_times))
    evaluation_complete = 192 + completed_eval_in_tranche

    remaining = args.limit - complete
    eta_hours = remaining / total_rate / 60 if total_rate else None
    finish_local = now_local + timedelta(hours=eta_hours) if eta_hours is not None else None
    checkpoint = json.loads((RUN_ROOT / "checkpoint.json").read_text(encoding="utf-8"))
    errors = Counter(row.get("error_category") or "success" for row in attempts)
    recent_errors = Counter(row.get("error_category") or "success" for row in recent_attempts)

    report = {
        "current_time_berlin": now_local.isoformat(timespec="seconds"),
        "phase": {
            "pass": checkpoint.get("pass"),
            "phase": checkpoint.get("phase"),
            "batch_start": checkpoint.get("batch_start"),
            "batch_end_exclusive": checkpoint.get("batch_end_exclusive"),
        },
        "coverage": {
            "complete_in_tranche": complete,
            "tranche_total": args.limit,
            "structures_available": len(structures),
            "expected_entities_for_available_structures": expected_entity_count,
            "main_evaluation_complete": evaluation_complete,
            "main_evaluation_total": 600,
            "cumulative_corpus_complete": args.baseline_complete + complete,
            "corpus_total": 33_047,
        },
        "throughput": {
            "complete_images_per_minute_total": round(total_rate, 3),
            f"complete_images_per_minute_recent_{args.recent_minutes}m": round(recent_rate, 3),
            "tokens_per_minute_total": round(usage["total_tokens"] / elapsed_minutes) if elapsed_minutes else 0,
            f"tokens_per_minute_recent_{args.recent_minutes}m": round(recent_usage["total_tokens"] / args.recent_minutes),
        },
        "projection": {
            "remaining_tranche_images": remaining,
            "remaining_runtime_hours": round(eta_hours, 2) if eta_hours is not None else None,
            "predicted_finish_berlin": finish_local.isoformat(timespec="minutes") if finish_local else None,
        },
        "endpoint": {
            "attempts_since_healthy_resume": len(attempts),
            "failed_attempts": sum(not row.get("ok") for row in attempts),
            "retry_fraction": round(sum(not row.get("ok") for row in attempts) / len(attempts), 4) if attempts else 0,
            "rate_limits": errors["rate_limit"],
            "rate_limit_fraction": round(errors["rate_limit"] / len(attempts), 4) if attempts else 0,
            f"recent_{args.recent_minutes}m_attempts": len(recent_attempts),
            f"recent_{args.recent_minutes}m_rate_limit_fraction": round(recent_errors["rate_limit"] / len(recent_attempts), 4) if recent_attempts else 0,
            "error_categories": dict(errors),
        },
        "resources": {
            "cost_usd_since_healthy_resume": round(usage["cost_microusd"] / 1_000_000, 4),
            "upload_gb": round(usage["request_body_bytes"] / 1_000_000_000, 3),
            "download_mb": round(usage["response_body_bytes"] / 1_000_000, 3),
            "average_upload_mbps": round(usage["request_body_bytes"] * 8 / elapsed_minutes / 60 / 1_000_000, 3) if elapsed_minutes else 0,
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

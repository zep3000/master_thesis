from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from core import iter_jsonl, read_json, render_pdf_page, save_face_composite, write_json


HERE = Path(__file__).resolve().parent
FINAL_PIPELINE_DIR = HERE.parent / "final"


def load_frozen_pipeline():
    sys.path.insert(0, str(FINAL_PIPELINE_DIR))
    try:
        return importlib.import_module("run")
    finally:
        sys.path.pop(0)


def page_context_prompt(prompts: Any, task_id: str) -> str:
    prompt = prompts.person_prompt(task_id, "direct", True)
    return prompt.replace(
        "complete advertisement context with the target marked red",
        "complete issue-page context with the target marked red",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen person attributes on human face boxes.")
    parser.add_argument("--manifest", type=Path, default=HERE / "local" / "manifest.json")
    parser.add_argument("--scope", choices=["pilot", "all"], default="pilot")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N deterministically ordered faces in the selected scope.",
    )
    parser.add_argument("--key-file", type=Path, default=None)
    parser.add_argument("--pdftoppm", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--request-budget", type=int, default=60)
    parser.add_argument("--target-tokens-per-minute", type=int, default=120000)
    parser.add_argument("--local-root", type=Path, default=HERE / "local")
    parser.add_argument("--render-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = read_json(args.manifest.resolve())
    if manifest.get("schema_version") != "full_issue_face_audit_manifest_v1":
        raise ValueError("unsupported manifest schema")
    by_id = {face["face_id"]: face for face in manifest["faces"]}
    selected_ids = manifest["pilot"]["face_ids"] if args.scope == "pilot" else sorted(by_id)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be a positive integer")
        selected_ids = selected_ids[: args.limit]
    selected = [by_id[face_id] for face_id in selected_ids]
    local_root = args.local_root.resolve()
    page_root = local_root / "pages"
    crop_root = local_root / "crops"
    output_root = local_root / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen_pipeline()
    frozen.OUTPUT = output_root
    frozen.LEDGER = output_root / "request_ledger.jsonl"
    results_path = output_root / "entities.jsonl"
    prompt_hashes: set[str] = set()
    jobs = []
    for face in selected:
        page_path = page_root / f"{face['issue_page_identifier']}.jpg"
        crop_path = crop_root / f"{face['face_id'].replace('::', '__')}.jpg"
        pdf_path = Path(manifest["pdfs"][face["issue_id"]]["path"])
        prompt = page_context_prompt(frozen.prompts, face["face_id"])
        prompt_hashes.add(frozen.hashlib.sha256(prompt.encode("utf-8")).hexdigest())

        def content_factory(
            face=face,
            page_path=page_path,
            crop_path=crop_path,
            pdf_path=pdf_path,
            prompt=prompt,
        ):
            render_pdf_page(
                args.pdftoppm.resolve(),
                pdf_path,
                face["pdf_page_index"],
                page_path,
                args.dpi,
            )
            save_face_composite(page_path, face["bbox_1000"], crop_path)
            return [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": frozen.image_data_url(crop_path)}},
            ]

        jobs.append({
            "task_key": face["face_id"],
            "content_factory": content_factory,
            "max_tokens": 1500,
            "normalize": lambda raw, task_id=face["face_id"]: frozen.normalize_person(
                raw, task_id, "direct", True
            ),
            "meta": {
                "run_id": f"full_issue_face_audit_{args.scope}_v1",
                "cohort": "full_issue_face_audit",
                "stage": "entities",
                "variant": "forced_human_boxes_full_page_context_v1",
                "prompt_version": f"{frozen.prompts.PROMPT_VERSION}+full_issue_page_context_v1",
                "model": frozen.MODEL,
                "provider": frozen.PROVIDER,
                "entity_type": "person",
                "style": "direct",
                "gaze": True,
            },
        })
    if args.render_only:
        for index, job in enumerate(jobs, start=1):
            job["content_factory"]()
            print(f"[{index}/{len(jobs)}] rendered {job['task_key']}")
        write_json(
            output_root / f"render_summary_{args.scope}.json",
            {
                "schema_version": "full_issue_face_audit_render_summary_v1",
                "scope": args.scope,
                "rendered_faces": len(jobs),
                "dpi": args.dpi,
                "page_root": str(page_root),
                "crop_root": str(crop_root),
            },
        )
        return 0
    if args.key_file is not None:
        frozen.KEY = args.key_file.resolve()
    elif not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("supply --key-file or OPENROUTER_API_KEY")
    if not os.environ.get("OPENROUTER_API_KEY") and not frozen.KEY.is_file():
        raise FileNotFoundError(f"OpenRouter key file not found: {frozen.KEY}")
    existing_attempts = sum(1 for _ in iter_jsonl(frozen.LEDGER))
    budget = frozen.Budget(
        existing_attempts + args.request_budget,
        frozen.LEDGER,
        target_tokens_per_minute=args.target_tokens_per_minute,
    )
    summary = frozen.execute_jobs(jobs, results_path, budget, args.workers)
    successful = {
        row["task_key"]: row for row in iter_jsonl(results_path) if row.get("ok")
    }
    missing = [face_id for face_id in selected_ids if face_id not in successful]
    joined_counts = Counter(
        (by_id[face_id]["detection_status"], str(by_id[face_id]["analyzable"]).lower())
        for face_id in selected_ids
        if face_id in successful
    )
    run_summary = {
        "schema_version": "full_issue_face_audit_run_summary_v1",
        "scope": args.scope,
        "limit": args.limit,
        "selected_faces": len(selected),
        "successful_faces": len(selected) - len(missing),
        "missing_face_ids": missing,
        "model": frozen.MODEL,
        "provider": frozen.PROVIDER,
        "dpi": args.dpi,
        "prompt_hashes": sorted(prompt_hashes),
        "result_path": str(results_path),
        "joint_status_analyzable_counts": {
            f"{status}|{analyzable}": count
            for (status, analyzable), count in sorted(joined_counts.items())
        },
        "endpoint_summary": summary,
    }
    write_json(output_root / f"run_summary_{args.scope}.json", run_summary)
    print(json.dumps(run_summary, indent=2))
    if missing:
        raise RuntimeError(f"{len(missing)} selected faces lack successful results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

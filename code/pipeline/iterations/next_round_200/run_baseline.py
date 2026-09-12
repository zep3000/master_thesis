"""Run the proven P2 baseline on a neutral cohort manifest.

The implementation delegates to the frozen first100 P2 runner after replacing
only its path globals. This preserves the exact prompt, schema, normalization,
retry, and ledger behavior used for the earlier difficult100 baseline.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "next_round_200"
SOURCE_DIR = ROOT / "qwen_iteration" / "first100_pipeline_matrix"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", choices=["difficult100", "stratified100"], required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--stage", choices=["all", "structure", "entities", "assemble"], default="all")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--max-requests", type=int, default=3000)
    p.add_argument("--retry-failures", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(SOURCE_DIR))
    spec = importlib.util.spec_from_file_location("frozen_first100_runner", SOURCE_DIR / "run.py")
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.HERE = HERE
    runner.MANIFEST = HERE / "data" / f"manifest_{args.cohort}.json"
    runner.IMAGE_DIR = (ROOT / "code" / "test_collection_200_difficult_joined_pages") if args.cohort == "difficult100" else (ROOT / "master_thesis" / "data" / "images" / "full_pages_1940_2007_joined")
    runner.SHARED_LEDGER = HERE / "output" / "shared_request_ledger.jsonl"
    sys.argv = [
        "run_baseline.py", "--pipeline", "p2", "--run-name", args.run_name,
        "--stage", args.stage, "--start-index", str(args.start_index), "--limit", str(args.limit),
        "--workers", str(args.workers), "--max-requests", str(args.max_requests),
    ]
    if args.retry_failures:
        sys.argv.append("--retry-failures")
    if args.dry_run:
        sys.argv.append("--dry-run")
    runner.main()


if __name__ == "__main__":
    main()

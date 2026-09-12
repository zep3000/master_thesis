#!/usr/bin/env python3
"""Offline measurement of JPEG/base64 request payload overhead.

No API request is made. The script builds the exact s2 page input for an
evenly spaced sample and reports JPEG bytes versus JSON transport bytes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import prompts
import run


def mean(values):
    return statistics.mean(values) if values else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=run.DEFAULT_FULL_PAGE_DIR)
    parser.add_argument("--sample", type=int, default=12)
    args = parser.parse_args()
    files = sorted(
        (path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() in run.IMAGE_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )
    if not files or args.sample <= 0:
        raise ValueError("no images or invalid sample size")
    count = min(args.sample, len(files))
    selected = [files[round(index * (len(files) - 1) / max(1, count - 1))] for index in range(count)]
    rows = []
    for path in selected:
        started = time.perf_counter()
        prompt = prompts.structure_prompt(path.stem, int(path.stem[:4]) if path.stem[:4].isdigit() else None, True, True)
        content = run.structure_content(path, prompt, "s2")
        build_seconds = time.perf_counter() - started
        urls = [part["image_url"]["url"] for part in content if part.get("type") == "image_url"]
        base64_chars = sum(len(url.split(",", 1)[1]) for url in urls)
        encoded_image_bytes = sum((len(url.split(",", 1)[1]) * 3 // 4) - url.split(",", 1)[1].count("=") for url in urls)
        request = {
            "model": run.MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 12000,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "provider": {"order": [run.PROVIDER], "allow_fallbacks": False},
            "reasoning": {"effort": "none", "exclude": True},
        }
        json_bytes = len(json.dumps(request).encode("utf-8"))
        rows.append({
            "file": path.name,
            "source_jpeg_bytes": path.stat().st_size,
            "all_five_jpeg_bytes": encoded_image_bytes,
            "base64_characters": base64_chars,
            "json_request_bytes": json_bytes,
            "build_seconds": build_seconds,
        })
    report = {
        "sample": len(rows),
        "mean_source_jpeg_mb": mean([row["source_jpeg_bytes"] for row in rows]) / 1_000_000,
        "mean_all_five_jpegs_mb": mean([row["all_five_jpeg_bytes"] for row in rows]) / 1_000_000,
        "mean_json_request_mb": mean([row["json_request_bytes"] for row in rows]) / 1_000_000,
        "mean_base64_expansion_over_jpeg": mean([row["base64_characters"] / row["all_five_jpeg_bytes"] for row in rows]),
        "mean_build_seconds": mean([row["build_seconds"] for row in rows]),
        "estimated_live_request_json_mb_at_24_workers": mean([row["json_request_bytes"] for row in rows]) * 24 / 1_000_000,
        "rows": rows,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluation-only visual boards and error-pattern tables for final strategies."""

from __future__ import annotations

import csv
import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "expression_legibility_research"
DATA = HERE / "data"
EVAL = HERE / "evaluation"
OUT = EVAL / "final_visual_audit"


def prediction_rows(run: str, strategy: str) -> dict[str, dict[str, str]]:
    path = EVAL / run / "predictions.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle) if row["strategy"] == strategy}


def short(value: str | None) -> str:
    mapping = {
        "0_not_legible": "0", "1_low_legibility": "1", "2_moderate_legibility": "2", "3_high_legibility": "3",
    }
    return mapping.get(value or "", "-")


def board(name: str, case_ids: list[str], manifest: dict[str, Any], labels: dict[str, Any], qwen: dict[str, Any], gemma: dict[str, Any], columns: int = 4) -> None:
    tile_w, tile_h = 360, 410
    rows = (len(case_ids) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_w, max(1, rows) * tile_h), "white")
    draw = ImageDraw.Draw(canvas); font = ImageFont.load_default()
    cases = {case["case_id"]: case for case in manifest["cases"]}
    for index, case_id in enumerate(case_ids):
        x = (index % columns) * tile_w; y = (index // columns) * tile_h
        with Image.open(cases[case_id]["face_only_path"]) as opened:
            image = opened.convert("RGB")
        image.thumbnail((340, 330), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + (tile_w - image.width) // 2, y + 48))
        item = labels[case_id]
        caption = (
            f"{case_id} / {item['source_image_id']}   gold={short(item['gold']['face_expression_legibility'])} "
            f"P2={short(item['frozen_p2_baseline']['face_expression_legibility'])} "
            f"Q9={short(qwen[case_id]['pred_legibility'])} G4={short(gemma[case_id]['pred_legibility'])}\n"
            f"{item['face_size_bin']} | {item['human_depiction_type']} | {item['human_orientation']}"
        )
        draw.multiline_text((x + 8, y + 7), textwrap.fill(caption, 70), fill="black", font=font, spacing=2)
        draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline="#999999", width=1)
    canvas.save(OUT / f"{name}.jpg", "JPEG", quality=92, optimize=True)


def grouped_accuracy(rows: dict[str, dict[str, str]], labels: dict[str, Any], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for case_id, item in labels.items():
        groups[str(item[field])].append(case_id)
    result = []
    for value, case_ids in sorted(groups.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        exact = sum(rows[case_id]["pred_legibility"] == rows[case_id]["gold_legibility"] for case_id in case_ids)
        under = sum(int(rows[case_id]["pred_legibility"][0]) < int(rows[case_id]["gold_legibility"][0]) for case_id in case_ids)
        over = sum(int(rows[case_id]["pred_legibility"][0]) > int(rows[case_id]["gold_legibility"][0]) for case_id in case_ids)
        result.append({"field": field, "value": value, "n": len(case_ids), "exact": exact / len(case_ids), "under": under / len(case_ids), "over": over / len(case_ids)})
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA / "cases_manifest.json").read_text(encoding="utf-8"))
    labels = json.loads((DATA / "gold_labels.json").read_text(encoding="utf-8"))["cases"]
    qwen = prediction_rows("frozen_qwen_v1", "ordinal_thresholds")
    gemma = prediction_rows("full_gemma4_31b", "ordinal_thresholds")
    errors = [case_id for case_id in qwen if qwen[case_id]["pred_legibility"] != qwen[case_id]["gold_legibility"]]
    errors.sort(key=lambda case_id: (qwen[case_id]["gold_legibility"], case_id))
    for part, start in enumerate(range(0, len(errors), 12), start=1):
        board(f"qwen_ordinal_errors_{part}", errors[start:start + 12], manifest, labels, qwen, gemma)
    zero_cases = [case_id for case_id in qwen if qwen[case_id]["gold_legibility"] == "0_not_legible"]
    high_cases = [case_id for case_id in qwen if qwen[case_id]["gold_legibility"] == "3_high_legibility"]
    disagreements = [case_id for case_id in qwen if qwen[case_id]["pred_legibility"] != gemma[case_id]["pred_legibility"]]
    board("all_gold_zero_cases", zero_cases, manifest, labels, qwen, gemma)
    board("all_gold_high_cases", high_cases, manifest, labels, qwen, gemma)
    for part, start in enumerate(range(0, len(disagreements), 12), start=1):
        board(f"qwen_gemma_disagreements_{part}", disagreements[start:start + 12], manifest, labels, qwen, gemma)

    patterns = []
    for field in ("face_size_bin", "human_depiction_type", "human_orientation"):
        patterns.extend(grouped_accuracy(qwen, labels, field))
    with (OUT / "qwen_patterns.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patterns[0]))
        writer.writeheader(); writer.writerows(patterns)
    summary = {
        "qwen_error_count": len(errors), "qwen_gemma_disagreement_count": len(disagreements),
        "qwen_error_direction": dict(Counter(
            "under" if int(qwen[c]["pred_legibility"][0]) < int(qwen[c]["gold_legibility"][0]) else "over"
            for c in errors
        )),
        "error_case_ids": errors, "disagreement_case_ids": disagreements,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()

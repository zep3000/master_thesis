#!/usr/bin/env python3
"""Build an evaluation-only visual review of potentially disputable gold labels.

This script is deliberately separate from inference. It reads human gold labels
and model predictions only to create a post-hoc review board; nothing produced
here is an input to any annotation pipeline.
"""

from __future__ import annotations

import base64
import csv
import html
import io
import json
from pathlib import Path

from PIL import Image


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "expression_legibility_research"
DATA = HERE / "data"
EVAL = HERE / "evaluation"

CASES = [
    {
        "id": "expr_044",
        "confidence": "strong",
        "review": "Gold 1 appears too low; eyes, brows, and mouth are all distinct. Review as 2–3, probably 3.",
    },
    {
        "id": "expr_025",
        "confidence": "strong",
        "review": "Gold 0 conflicts with a visible eye, mouth line, and profile configuration. Review as at least 1.",
    },
    {
        "id": "expr_017",
        "confidence": "medium",
        "review": "The profile illustration retains a clear eye/nose/mouth configuration. Gold 0 should be reviewed against 1.",
    },
    {
        "id": "expr_018",
        "confidence": "medium",
        "review": "The driver’s profile, eye area, and mouth contour remain visible. Gold 0 versus 1 is debatable.",
    },
    {
        "id": "expr_006",
        "confidence": "medium",
        "review": "Only part of the face is available, but nose and mouth configuration survive. Review 0 versus 1.",
    },
    {
        "id": "expr_008",
        "confidence": "medium",
        "review": "The tiny frontal illustration has coarse eye and mouth evidence. Gold 0 may be too strict, although exact smile is unsafe.",
    },
    {
        "id": "expr_021",
        "confidence": "medium",
        "review": "Rotation and clutter make the face difficult, but coarse facial configuration survives. Review 0 versus 1.",
    },
    {
        "id": "expr_023",
        "confidence": "medium",
        "review": "Smile and gaze are clear, but scan resolution may not support fine/subtle cues. Gold 3 should be reviewed against 2.",
    },
    {
        "id": "expr_032",
        "confidence": "medium",
        "review": "The face is readable but small, dark, and partly crowded. Gold 3 may be better treated as 2.",
    },
]


def prediction_rows(run: str, strategy: str) -> dict[str, dict[str, str]]:
    path = EVAL / run / "predictions.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["case_id"]: row
            for row in csv.DictReader(handle)
            if row["strategy"] == strategy
        }


def short(value: str | None) -> str:
    return {
        "0_not_legible": "0",
        "1_low_legibility": "1",
        "2_moderate_legibility": "2",
        "3_high_legibility": "3",
        "": "–",
        None: "–",
    }.get(value, value or "–")


def image_data_uri(path: Path) -> str:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    image.thumbnail((720, 390), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=58, optimize=True, progressive=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def build(output: Path) -> None:
    manifest = json.loads((DATA / "cases_manifest.json").read_text(encoding="utf-8"))
    labels = json.loads((DATA / "gold_labels.json").read_text(encoding="utf-8"))["cases"]
    manifest_by_id = {item["case_id"]: item for item in manifest["cases"]}

    strategies = {
        "P2": prediction_rows("frozen_qwen_v1", "frozen_p2_baseline"),
        "direct": prediction_rows("frozen_qwen_v1", "anchors_direct"),
        "evidence": prediction_rows("frozen_qwen_v1", "evidence_derived"),
        "ordinal": prediction_rows("frozen_qwen_v1", "ordinal_thresholds"),
        "dual": prediction_rows("frozen_qwen_v1", "dualview_anchors"),
        "Gemma ordinal": prediction_rows("full_gemma4_31b", "ordinal_thresholds"),
    }

    tiles: list[str] = []
    for case in CASES:
        case_id = case["id"]
        label = labels[case_id]
        gold = label["gold"]
        predictions = " · ".join(
            f"{name} {short(rows[case_id]['pred_legibility'])}"
            for name, rows in strategies.items()
        )
        gold_details = (
            f"Gold {short(gold['face_expression_legibility'])} · "
            f"gaze {gold['gaze_target'] or 'null'} · "
            f"smile {gold['smile_present'] or 'null'}"
        )
        if gold.get("smile_intensity"):
            gold_details += f" / {gold['smile_intensity']}"
        metadata = (
            f"{label['source_image_id']} · {label['human_depiction_type']} · "
            f"{label['human_orientation']} · {label['face_size_bin']}"
        )
        uri = image_data_uri(Path(manifest_by_id[case_id]["source_composite_path"]))
        tiles.append(
            f"""
            <section class="audit-case" aria-label="{esc(case_id)} gold-label review">
              <div class="case-heading viz-row">
                <h2>{esc(case_id)}</h2>
                <span class="viz-badge">{esc(case['confidence'])} review candidate</span>
              </div>
              <div class="text-small text-muted">{esc(metadata)}</div>
              <img class="case-image" src="{uri}" alt="Enlarged target face at left and source advertisement with target marked at right for {esc(case_id)}">
              <div class="gold-line"><strong>{esc(gold_details)}</strong></div>
              <div class="text-small prediction-line">{esc(predictions)}</div>
              <div class="review-line"><strong>Independent audit:</strong> {esc(case['review'])}</div>
            </section>
            """
        )

    fragment = f"""
<div id="expression-gold-review">
  <style>
    #expression-gold-review {{ color: var(--foreground); }}
    #expression-gold-review .legend {{ margin-block: 0.5rem 1rem; }}
    #expression-gold-review .case-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.5rem; }}
    #expression-gold-review .audit-case {{ min-width: 0; }}
    #expression-gold-review .case-heading {{ justify-content: space-between; align-items: baseline; margin-bottom: 0.25rem; }}
    #expression-gold-review .case-heading h2 {{ margin: 0; }}
    #expression-gold-review .case-image {{ display: block; width: 100%; height: auto; margin-block: 0.6rem; border: 1px solid var(--border); }}
    #expression-gold-review .gold-line {{ margin-bottom: 0.35rem; }}
    #expression-gold-review .prediction-line {{ overflow-wrap: anywhere; }}
    #expression-gold-review .review-line {{ margin-top: 0.45rem; }}
    @media (max-width: 640px) {{
      #expression-gold-review .case-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
  <h1>Gold-label review candidates</h1>
  <div class="legend text-small text-muted">Legibility: 0 not legible · 1 low · 2 moderate · 3 high. “Strong” means the current gold looks inconsistent with the written rubric; “medium” means blinded re-annotation is warranted.</div>
  <div class="case-grid">
    {''.join(tiles)}
  </div>
</div>
""".strip() + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment, encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.output)

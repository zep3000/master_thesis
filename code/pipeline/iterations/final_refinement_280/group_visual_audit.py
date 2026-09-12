#!/usr/bin/env python3
"""Evaluation-only visual audit of fields changed by the group context composite."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import evaluate
from common import HERE


FIELDS = ["group_type", "age_composition", "gender_presentation_composition", "expression_legibility_distribution", "dominant_gaze", "smile_prevalence", "dominant_smile_intensity"]


def main() -> int:
    ev = evaluate.evaluator(); records = []
    for cohort in ["difficult140", "stratified140"]:
        human = evaluate.gold(cohort)
        direct = evaluate.predictions(cohort, "current_direct_ungated")
        context = evaluate.predictions(cohort, "current_group_multiscale_context_ungated")
        for image_id in sorted(set(human) & set(direct)):
            ad_pairs = ev.pair_items(ev.all_items(image_id, human[image_id], "ads"), ev.all_items(image_id, direct[image_id], "ads"), "ads", "strict")
            for ad_pair in ad_pairs:
                pairs = ev.pair_items(ev.group_items(image_id, ad_pair.human.data), ev.group_items(image_id, ad_pair.predicted.data), "groups", "strict")
                for pair in pairs:
                    group_id = pair.predicted.item_id
                    replacement = next((g for ad in context[image_id].get("advertisements") or [] for g in ad.get("groups") or [] if g.get("group_id") == group_id), None)
                    if not replacement:
                        continue
                    changes = []
                    for field in FIELDS:
                        h, d, c = pair.human.data.get(field), pair.predicted.data.get(field), replacement.get(field)
                        before, after = ev.exact_label(field, h, d), ev.exact_label(field, h, c)
                        if before != after:
                            changes.append({"field": field, "gold": h, "direct": d, "context": c, "direction": "improved" if after else "worsened"})
                    if changes:
                        records.append({"cohort": cohort, "image_id": image_id, "group_id": group_id, "changes": changes})
    records.sort(key=lambda x: (-sum(c["direction"] == "improved" for c in x["changes"]), x["image_id"]))
    out_dir = HERE / "evaluation"; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "group_visual_audit.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    font = ImageFont.load_default(); width, cell_h = 1500, 500
    shown = records[:12]; canvas = Image.new("RGB", (width, max(1, len(shown)) * cell_h), "white"); draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(shown):
        y = index * cell_h
        crop_path = HERE / "output" / "crops" / "groups" / "multiscale_context" / f"{record['image_id']}__{record['group_id']}.jpg"
        if crop_path.exists():
            crop = Image.open(crop_path).convert("RGB"); crop.thumbnail((850, 450), Image.Resampling.LANCZOS); canvas.paste(crop, (0, y))
        lines = [f"{record['image_id']} / {record['group_id']}"]
        for change in record["changes"]:
            lines.append(f"{change['direction'].upper()} {change['field']}")
            lines.append(f"gold={change['gold']} | direct={change['direct']} | context={change['context']}")
        text = "\n".join(textwrap.fill(line, 82) for line in lines)
        draw.multiline_text((870, y + 15), text, fill="black", font=font, spacing=7)
        draw.line((0, y + cell_h - 1, width, y + cell_h - 1), fill="#999999", width=2)
    canvas.save(out_dir / "group_visual_audit.jpg", quality=92)
    print(json.dumps({"changed_cases": len(records), "shown": len(shown), "image": str(out_dir / 'group_visual_audit.jpg')}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

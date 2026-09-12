#!/usr/bin/env python3
"""Gold-blind outstanding-person confirmation pilot."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r".")
HERE = ROOT / "qwen_iteration" / "outstanding_confirmation_pilot"
CANONICAL = ROOT / "qwen_iteration" / "canonical_candidate_280"
LEAN = ROOT / "qwen_iteration" / "lean_final_280"
OUTPUT = HERE / "output"

sys.path.insert(0, str(CANONICAL))
import common  # noqa: E402

common.LEDGER = OUTPUT / "request_ledger.jsonl"


def load_prompts() -> Any:
    spec = importlib.util.spec_from_file_location("pilot_lean_prompts", LEAN / "prompts.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = load_prompts()
ROLES = [
    "ordinary_group_member",
    "dramatically_larger_or_clearer",
    "separate_panel_or_scene",
    "spatially_separate_from_people_area",
]


def baseline(cohort: str) -> list[dict[str, Any]]:
    path = CANONICAL / "output" / "assembled" / cohort / "canonical_candidate_v1.jsonl"
    return [row for row in common.rows(path) if row.get("ok")]


def manifest(cohort: str) -> dict[str, Any]:
    return json.loads((LEAN / "data" / f"manifest_{cohort}.json").read_text(encoding="utf-8"))


def clamp_enum(value: Any, allowed: list[Any], fallback: Any) -> Any:
    if isinstance(value, list):
        value = value[0] if len(value) == 1 else None
    return value if value in allowed else fallback


def confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def normalize(raw: dict[str, Any], task_id: str) -> dict[str, Any]:
    legibility = clamp_enum(raw.get("face_expression_legibility"), P.LEGIBILITY, "0_not_legible")
    mouth = clamp_enum(raw.get("mouth_covered"), P.MOUTH, "not_assessable")
    smile = clamp_enum(raw.get("smile_present"), P.SMILE, "not_assessable")
    gaze = clamp_enum(raw.get("gaze_target"), P.GAZE, "not_assessable")
    result = {
        "person_task_id": task_id,
        "group_membership_role": clamp_enum(raw.get("group_membership_role"), ROLES, "ordinary_group_member"),
        "depiction_type": clamp_enum(raw.get("depiction_type"), P.DEPICTION_TYPES + [None], None),
        "perceived_age": clamp_enum(raw.get("perceived_age"), P.AGES, "not_assessable"),
        "perceived_gender_presentation": clamp_enum(raw.get("perceived_gender_presentation"), P.GENDERS, "not_assessable"),
        "face_expression_legibility": legibility,
        "face_orientation": clamp_enum(raw.get("face_orientation"), P.ORIENTATION, "not_assessable"),
        "gaze_target": gaze,
        "gaze_target_person_unboxed": bool(raw.get("gaze_target_person_unboxed")) if gaze == "another_person" else None,
        "mouth_covered": mouth,
        "mouth_covering": clamp_enum(raw.get("mouth_covering"), P.MOUTH_CAUSES + [None], None) if mouth in {"yes", "partly"} else None,
        "smile_present": smile,
        "smile_intensity": clamp_enum(raw.get("smile_intensity"), P.INTENSITY + [None], None) if smile == "yes" else None,
        "confidence": confidence(raw.get("confidence")),
        "review_flags": raw.get("review_flags") if isinstance(raw.get("review_flags"), list) else [],
    }
    if legibility == "0_not_legible":
        result.update({"gaze_target": None, "gaze_target_person_unboxed": None, "smile_present": None, "smile_intensity": None})
    return result


def schema() -> dict[str, Any]:
    value = P.person_schema("direct", True)
    value["group_membership_role"] = ROLES
    return value


def prompt(task_id: str) -> str:
    return f"""Return one JSON object only. Apply annotation_playbook_v1.17. person_task_id={task_id}. Every enum is one scalar string from the displayed options.

The composite contains an enlarged target face, medium context, and the complete advertisement. The target is marked red; predicted people-area compression units are marked blue. Code only the red target. The blue area is an annotation-compression unit and need not be a social group.

In addition to the full individual taxonomy, decide `group_membership_role`. Use `ordinary_group_member` when the target is adequately represented by the people-area aggregate, including a merely front-row, central, or somewhat larger member. Use an outstanding category only for a dramatic visibility advantage, a genuinely separate panel/scene, or real spatial separation. Spatial overlap with a blue box does not by itself make the face ordinary or outstanding.

Judge all facial attributes independently of the membership decision. Halftone/grain alone is not illegible. Legibility is expression codability: 0=no usable expression evidence; 1=one coarse cue; 2=stable mouth plus eye/brow or equivalent configuration; 3=fine/subtle configuration. Gender is visible presentation; ambiguous_or_androgynous is substantive, not uncertainty. Orientation is horizontal pose only. Frontal pose is not proof of viewer gaze. Use viewer_camera only with positive pupil/eye-axis evidence. Smile intensity is null unless smile_present=yes; mouth covering is null unless mouth_covered is yes/partly.

Schema: {json.dumps(schema(), separators=(',', ':'), ensure_ascii=True)}"""


def letterbox(image: Image.Image, size: int, label: str) -> Image.Image:
    panel = Image.new("RGB", (size, size + 42), "white")
    work = image.copy(); work.thumbnail((size, size), Image.Resampling.LANCZOS)
    panel.paste(work, ((size - work.width) // 2, 42 + (size - work.height) // 2))
    ImageDraw.Draw(panel).text((12, 12), label, fill="black", font=ImageFont.load_default())
    return panel


def composite(page_path: Path, ad: dict[str, Any], person: dict[str, Any], out_path: Path) -> Path:
    if out_path.exists():
        return out_path
    page = Image.open(page_path).convert("RGB")
    face_px = common.to_pixels(person["face_bbox_1000"], page.size)
    ad_px = common.to_pixels(ad["bbox_1000"], page.size)
    tight = page.crop(common.expand_box(face_px, page.size, 1.65))
    medium = page.crop(common.expand_box(face_px, page.size, 4.5))
    context = page.crop(ad_px)
    draw = ImageDraw.Draw(context)
    ax1, ay1, _, _ = ad_px
    for group in ad.get("groups") or []:
        gx1, gy1, gx2, gy2 = common.to_pixels(group["bbox_1000"], page.size)
        draw.rectangle((gx1 - ax1, gy1 - ay1, gx2 - ax1, gy2 - ay1), outline=(0, 110, 255), width=5)
    fx1, fy1, fx2, fy2 = face_px
    draw.rectangle((fx1 - ax1, fy1 - ay1, fx2 - ax1, fy2 - ay1), outline=(255, 0, 0), width=6)
    panels = [letterbox(tight, 512, "TARGET FACE — red"), letterbox(medium, 512, "LOCAL CONTEXT"), letterbox(context, 512, "AD: target red; people areas blue")]
    canvas = Image.new("RGB", (1536, 554), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * 512, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "JPEG", quality=92, optimize=True)
    return out_path


def infer(cohort: str, workers: int, maximum: int) -> None:
    man = manifest(cohort); by_id = {item["image_id"]: item for item in man["images"]}
    jobs = []
    for row in baseline(cohort):
        item = by_id[row["image_id"]]
        page_path = Path(man["image_dir"]) / item["filename"]
        for ad in row["annotation"].get("advertisements") or []:
            if not (ad.get("groups") and ad.get("people")):
                continue
            for person in ad["people"]:
                task_id = f"{row['image_id']}::{person['person_id']}"
                path = composite(page_path, ad, person, OUTPUT / "crops" / cohort / f"{task_id.replace('::', '__')}.jpg")
                jobs.append({
                    "task_key": task_id,
                    "content": [{"type": "text", "text": prompt(task_id)}, {"type": "image_url", "image_url": {"url": common.image_data_url(path)}}],
                    "max_tokens": 1800,
                    "normalize": lambda raw, task_id=task_id: normalize(raw, task_id),
                    "meta": {"stage": "outstanding_confirmation", "cohort": cohort, "variant": "combined_person_v1", "model": common.MODEL},
                    "result_meta": {"image_id": row["image_id"], "advertisement_id": ad["advertisement_id"], "person_id": person["person_id"]},
                })
    common.execute(jobs, OUTPUT / f"{cohort}_responses.jsonl", maximum, workers)


def center_inside(person_box: list[int], group_box: list[int]) -> bool:
    x = (person_box[0] + person_box[2]) / 2; y = (person_box[1] + person_box[3]) / 2
    return group_box[0] <= x <= group_box[2] and group_box[1] <= y <= group_box[3]


def assemble(cohort: str) -> None:
    responses = {row["task_key"]: row["model_annotation"] for row in common.rows(OUTPUT / f"{cohort}_responses.jsonl") if row.get("ok")}
    output = []; audit = []
    for source in baseline(cohort):
        row = copy.deepcopy(source); image_id = row["image_id"]
        complete = True
        for ad in row["annotation"].get("advertisements") or []:
            if not ad.get("groups"):
                continue
            retained = []
            for person in ad.get("people") or []:
                key = f"{image_id}::{person['person_id']}"; attrs = responses.get(key)
                if attrs is None:
                    complete = False; retained.append(person); continue
                role = attrs["group_membership_role"]
                inside = any(center_inside(person["face_bbox_1000"], group["bbox_1000"]) for group in ad["groups"])
                absorb = role == "ordinary_group_member" and inside
                audit.append({"task_key": key, "image_id": image_id, "advertisement_id": ad["advertisement_id"], "person_id": person["person_id"], "role": role, "center_inside_people_area": inside, "action": "absorbed" if absorb else "retained"})
                if absorb:
                    continue
                person.update({k: copy.deepcopy(v) for k, v in attrs.items() if k not in {"person_task_id", "group_membership_role"}})
                person["annotation_role"] = "outstanding_individual"
                if role != "ordinary_group_member":
                    person["prominence_reason"] = role
                retained.append(person)
            ad["people"] = retained
            ad["has_outstanding_individuals"] = "yes" if retained else "no"
        row["route"] = "outstanding_confirmation_pilot_v1"; row["ok"] = bool(row.get("ok")) and complete
        output.append(row)
    common.write_jsonl(OUTPUT / "assembled" / f"{cohort}.jsonl", output)
    common.write_jsonl(OUTPUT / "audit" / f"{cohort}.jsonl", audit)
    print(json.dumps({"cohort": cohort, "pages": len(output), "complete": sum(bool(x["ok"]) for x in output), "candidates": len(audit), "absorbed": sum(x["action"] == "absorbed" for x in audit)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ["infer", "assemble"]:
        child = sub.add_parser(stage); child.add_argument("--cohort", choices=["difficult140", "stratified140"], required=True)
        child.add_argument("--workers", type=int, default=4); child.add_argument("--maximum", type=int, default=3000)
    args = parser.parse_args()
    if args.stage == "infer": infer(args.cohort, args.workers, args.maximum)
    else: assemble(args.cohort)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

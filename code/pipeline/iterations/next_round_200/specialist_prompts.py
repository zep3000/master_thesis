"""Gold-free specialist prompts and deterministic expression derivations."""

from __future__ import annotations

import json
from typing import Any

LEG = ["0_not_legible", "1_low_legibility", "2_moderate_legibility", "3_high_legibility"]
GAZE = ["viewer_camera", "another_person", "advertised_product", "other_object", "off_frame_or_scene_direction", "eyes_covered", "closed_eyes", "not_assessable"]
SMILE = ["yes", "no", "not_assessable"]
INTENSITY = ["1_slight", "2_clear", "3_broad", "4_laughter_like"]

ANCHORS = """Rate visible information, not emotional intensity or scan aesthetics.
0: no expression evidence at all; even smile versus non-smile is impossible.
1: at least one coarse cue is readable, but stable multi-region configuration is not.
2: mouth plus at least one eye/brow region support a stable coarse judgment.
3: eyes/brows and mouth support fine or subtle expression judgments.
Neutral, stern, profile, grainy, illustrated, or weakly expressive does not itself mean unreadable."""


def schema(extra: dict[str, Any]) -> str:
    base = {
        "task_id": "copy supplied ID", "at_least_low": "boolean",
        "at_least_moderate": "boolean", "at_least_high": "boolean",
        "gaze_target": [*GAZE, None], "smile_present": [*SMILE, None],
        "smile_intensity": [*INTENSITY, None], "confidence": "0..1 number",
        "review_flags": "array of short strings",
    }
    base.update(extra)
    return json.dumps(base, separators=(",", ":"))


def person_prompt(task_id: str, strategy: str) -> str:
    extra: dict[str, Any] = {}
    method = "Return direct substantive gaze and smile categories wherever their evidence is usable."
    if strategy in {"f1_hierarchical_gaze", "f2_hierarchical_gaze_smile"}:
        extra.update({
            "eye_state": ["open_assessable", "closed", "covered", "unreadable"],
            "positive_direct_eye_contact_evidence": "boolean; visible pupils/eye axes positively support looking at viewer",
            "external_gaze_target": ["another_person", "advertised_product", "other_object", "off_frame_or_scene_direction", "not_assessable", None],
        })
        method = """Judge gaze hierarchically. First code eye_state. Use viewer only when visible pupils/eye axes give positive direct-eye-contact evidence; frontal pose or centered composition is insufficient. Otherwise choose the best external target from another person, advertised product, other object, off-frame/scene direction, or not_assessable."""
    if strategy == "f2_hierarchical_gaze_smile":
        extra.update({
            "mouth_shape_assessable": "boolean; yes/no smile polarity can be visually judged",
            "at_least_slight_smile": "boolean", "at_least_clear_smile": "boolean",
            "at_least_broad_smile": "boolean", "laughter_like": "boolean",
        })
        method += " Judge smile through nested visible-mouth thresholds. Do not infer non-smile when mouth shape is unresolved."
    return f"""Return one JSON object only. task_id={task_id}. The image has an enlarged target face on the left and its ad context, target boxed red, on the right. Code only the target.
{ANCHORS}
Answer nested legibility thresholds: high implies moderate implies low; the pipeline repairs monotonic violations and derives the category. {method}
At derived zero, gaze and smile become null. Smile intensity is non-null only for smile=yes.
Schema: {schema(extra)}"""


GROUP_SCHEMA = {
    "task_id": "copy supplied ID",
    "group_type": ["interacting_group", "posed_group", "audience", "background_population", "separate_portraits_or_composite", "other_group"],
    "age_composition": ["young_only", "middle_only", "older_only", "mostly_young", "mostly_middle", "mostly_older", "mixed", "not_assessable"],
    "gender_presentation_composition": ["feminine_only", "masculine_only", "mostly_feminine", "mostly_masculine", "mixed", "ambiguous_or_androgynous_present", "not_assessable"],
    "legibility_percent_0": "integer 0..100", "legibility_percent_1": "integer 0..100",
    "legibility_percent_2": "integer 0..100", "legibility_percent_3": "integer 0..100",
    "dominant_gaze": ["toward_viewer_camera", "toward_each_other", "toward_object", "off_frame_or_scene_direction", "mixed", "not_assessable", None],
    "smile_percent": "integer 0..100 or null when faces are not assessable",
    "dominant_smile_intensity": ["slight", "clear", "broad_or_laughter_like", "mixed", None],
    "confidence": "0..1 number", "review_flags": "array",
}


def group_prompt(task_id: str) -> str:
    return f"""Return one JSON object only. task_id={task_id}. This is one proposed people-area crop. Treat it as a unified analytical expression unit; never enumerate faces.
Estimate the percentage of eligible faces at each expression-legibility level 0..3; the four integer percentages must sum to 100. Level anchors: 0 no expression evidence, 1 only coarse cue, 2 stable coarse multi-region configuration, 3 fine/subtle cues. Then code the ensemble's dominant gaze and smiling percentage. For age/gender, 'only' means all or nearly all; 'mostly' means strict majority; mixed means no strict majority.
Do not call viewer gaze from frontal pose alone. Intensity is null if smile_percent is 0/unassessable. Schema: {json.dumps(GROUP_SCHEMA,separators=(',',':'))}"""


def boolish(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "yes", "1"}


def enum(value: Any, allowed: list[str]) -> str | None:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    return str(value) if value in allowed else None


def normalize_person(raw: dict[str, Any], task_id: str, strategy: str) -> dict[str, Any]:
    low, moderate, high = (boolish(raw.get(k)) for k in ("at_least_low", "at_least_moderate", "at_least_high"))
    if high: moderate = True
    if moderate: low = True
    leg = LEG[3 if high else 2 if moderate else 1 if low else 0]
    gaze = enum(raw.get("gaze_target"), GAZE)
    smile = enum(raw.get("smile_present"), SMILE)
    intensity = enum(raw.get("smile_intensity"), INTENSITY)
    if strategy in {"f1_hierarchical_gaze", "f2_hierarchical_gaze_smile"}:
        state = raw.get("eye_state")
        if state == "closed": gaze = "closed_eyes"
        elif state == "covered": gaze = "eyes_covered"
        elif state == "unreadable": gaze = "not_assessable"
        elif boolish(raw.get("positive_direct_eye_contact_evidence")): gaze = "viewer_camera"
        else: gaze = enum(raw.get("external_gaze_target"), ["another_person", "advertised_product", "other_object", "off_frame_or_scene_direction", "not_assessable"]) or "not_assessable"
    if strategy == "f2_hierarchical_gaze_smile":
        assessable = boolish(raw.get("mouth_shape_assessable"))
        slight = boolish(raw.get("at_least_slight_smile")); clear = boolish(raw.get("at_least_clear_smile")); broad = boolish(raw.get("at_least_broad_smile")); laugh = boolish(raw.get("laughter_like"))
        if laugh: broad = clear = slight = True
        if broad: clear = slight = True
        if clear: slight = True
        smile = "yes" if slight else "no" if assessable else "not_assessable"
        intensity = ("4_laughter_like" if laugh else "3_broad" if broad else "2_clear" if clear else "1_slight") if smile == "yes" else None
    if leg == LEG[0]: gaze = smile = intensity = None
    elif smile != "yes": intensity = None
    return {"task_id": task_id, "face_expression_legibility": leg, "gaze_target": gaze, "smile_present": smile, "smile_intensity": intensity, "confidence": float(raw.get("confidence") or .5), "review_flags": raw.get("review_flags") if isinstance(raw.get("review_flags"), list) else [], "audit": {k: raw.get(k) for k in raw if k.startswith("at_least") or k in {"eye_state","positive_direct_eye_contact_evidence","external_gaze_target","mouth_shape_assessable","laughter_like"}}}


def normalize_group(raw: dict[str, Any], task_id: str) -> dict[str, Any]:
    perc = []
    for i in range(4):
        try: perc.append(max(0, min(100, int(raw.get(f"legibility_percent_{i}") or 0))))
        except (TypeError, ValueError): perc.append(0)
    if sum(perc) == 0: perc = [100, 0, 0, 0]
    total = sum(perc); perc = [round(v * 100 / total) for v in perc]; perc[max(range(4), key=perc.__getitem__)] += 100 - sum(perc)
    top = max(range(4), key=perc.__getitem__); peak = perc[top]
    leg = f"all_{top}_{['not_legible','low_legibility','moderate_legibility','high_legibility'][top]}" if peak >= 90 else f"mostly_{top}_{['not_legible','low_legibility','moderate_legibility','high_legibility'][top]}" if peak > 50 else "mixed_legibility"
    try: smile_pct = int(raw.get("smile_percent")) if raw.get("smile_percent") is not None else None
    except (TypeError, ValueError): smile_pct = None
    prevalence = None if smile_pct is None else "none" if smile_pct == 0 else "minority" if smile_pct < 40 else "about_half" if smile_pct <= 60 else "majority" if smile_pct < 90 else "all"
    def scalar(value: Any) -> Any:
        return value[0] if isinstance(value, list) and len(value) == 1 else value
    dominant_gaze = scalar(raw.get("dominant_gaze"))
    intensity = scalar(raw.get("dominant_smile_intensity")) if prevalence not in {None, "none"} else None
    if leg == "all_0_not_legible": dominant_gaze = prevalence = intensity = None
    return {"task_id": task_id, "group_type": scalar(raw.get("group_type")), "age_composition": scalar(raw.get("age_composition")), "gender_presentation_composition": scalar(raw.get("gender_presentation_composition")), "expression_legibility_distribution": leg, "dominant_gaze": dominant_gaze, "smile_prevalence": prevalence, "dominant_smile_intensity": intensity, "confidence": float(raw.get("confidence") or .5), "review_flags": raw.get("review_flags") if isinstance(raw.get("review_flags"), list) else [], "audit_percentages": perc}

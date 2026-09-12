"""Frozen prompts and gold-free normalization rules for the strategy matrix."""

from __future__ import annotations

import json
from typing import Any


PROMPT_VERSION = "expression_legibility_strategies_frozen_v2"
LOGIC_VERSION = "expression_legibility_logic_v2"

LEGIBILITY = ["0_not_legible", "1_low_legibility", "2_moderate_legibility", "3_high_legibility"]
GAZE = ["viewer_camera", "another_person", "advertised_product", "other_object", "off_frame_or_scene_direction", "eyes_covered", "closed_eyes", "not_assessable"]
SMILE = ["yes", "no", "not_assessable"]
INTENSITY = ["1_slight", "2_clear", "3_broad", "4_laughter_like"]
DEPICTION = ["photo_of_person", "illustration", "cartoon_or_caricature", "generic_human_figure", "photo_of_artwork_or_statue", "drawing_of_statue_monument_or_public_symbol", "mask_mannequin_doll_or_puppet", "personified_object", "nonhuman_creature_with_face", "schematic_icon_or_logo_face", "multiple_types_present"]
AGE = ["infant", "child", "adolescent", "young_adult", "middle_adult", "older_adult", "not_assessable"]
GENDER = ["feminine", "masculine", "ambiguous_or_androgynous", "not_assessable"]
ORIENTATION = ["beyond_profile", "profile", "three_quarter", "frontal", "tilted_down", "tilted_up", "other", "not_assessable"]
MOUTH = ["no", "yes", "partly", "not_assessable"]
MOUTH_CAUSE = ["hand", "beard", "other_body_part", "part_of_another_person", "object", "object_in_mouth", "text_or_graphic_overlay", "cropped_by_page_edge", "other", "not_assessable"]

BASE_FIELDS = {
    "case_id": "copy the supplied case ID",
    "face_expression_legibility": LEGIBILITY,
    "gaze_target": [*GAZE, None],
    "smile_present": [*SMILE, None],
    "smile_intensity": [*INTENSITY, None],
    "confidence": "number from 0 through 1",
    "review_flags": "array of short strings",
}

ANCHORS = """Rate information availability, not emotional intensity, attractiveness, certainty about identity, or scan aesthetics.
The target may be a photograph, illustration, cartoon, statue, mask, or other eligible facial depiction.
Use these operational anchors:
- 0_not_legible: no facial-expression evidence can be interpreted at all. Even a smile-versus-non-smile or broad-affect judgment is impossible. A merely neutral, stern, subtle, profile, grainy, or non-photographic face is NOT zero when facial configuration remains interpretable.
- 1_low_legibility: at least one coarse cue is readable, such as smile polarity or broad affect, but fine expression configuration is not stable.
- 2_moderate_legibility: mouth plus at least one eye/brow region support a stable coarse expression judgment, while degradation prevents confident fine-cue reading.
- 3_high_legibility: eyes/brows and mouth are clear enough for fine or subtle expression cues. A completely neutral expression can be high-legibility.
Judge what a careful human can visibly distinguish, not how emotionally expressive the person is. Select substantive gaze and smile categories whenever their relevant evidence is usable. If smile polarity is judgeable, smile_present must be yes or no and legibility cannot be zero. If gaze direction is judgeable, gaze_target must be substantive and legibility cannot be zero."""


def compact_schema(extra: dict[str, Any] | None = None) -> str:
    fields = dict(BASE_FIELDS)
    if extra:
        fields.update(extra)
    return json.dumps(fields, ensure_ascii=True, separators=(",", ":"))


def direct_prompt(case_id: str, dual_view: bool = False) -> str:
    image_note = (
        "The image contains two views of the SAME target face: original on the left and a deterministic contrast/sharpness view on the right. The second view is not another person and must not be allowed to invent evidence absent from the original."
        if dual_view else
        "The image contains an enlarged target face on the left and advertisement context with that same target marked in red on the right. Code only the target face."
    )
    return f"""Return one JSON object only. case_id={case_id}. {image_note}
{ANCHORS}
Use null for gaze and smile only at 0_not_legible. At legibility 1 or above, use not_assessable only when that specific cue truly cannot be judged. smile_intensity is non-null only when smile_present=yes.
Output schema: {compact_schema()}"""


def evidence_prompt(case_id: str) -> str:
    extra = {
        "facial_surface_locatable": "boolean",
        "smile_polarity_judgeable": "boolean; can yes versus no be distinguished?",
        "broad_affect_judgeable": "boolean; is any broad expression tendency interpretable?",
        "mouth_region_readability": ["unreadable", "partial", "clear"],
        "eye_region_readability": ["unreadable", "one_or_partial", "both_clear"],
        "brow_cue_readability": ["unreadable", "partial", "clear"],
        "stable_coarse_configuration": "boolean",
        "fine_expression_cues_judgeable": "boolean",
    }
    return f"""Return one JSON object only. case_id={case_id}. The image contains an enlarged target face on the left and advertisement context with the same target marked in red on the right. Inspect only that target.
Do not choose the legibility category directly. First record the requested observable judgments. Neutral or subtle does not mean unreadable. Grain, profile, illustration, age, and weak emotional intensity are not themselves reasons to mark evidence absent. smile_polarity_judgeable means a careful human can decide smile yes versus no; it does not require a smile to be present. stable_coarse_configuration means the visible mouth and eye/brow information supports a stable coarse expression judgment. fine_expression_cues_judgeable requires subtle configuration to be clear.
Still return gaze/smile fields wherever their own evidence is usable. The pipeline will derive legibility deterministically and will ignore your face_expression_legibility field, so set that field to your best audit recommendation only.
Output schema: {compact_schema(extra)}"""


def threshold_prompt(case_id: str, verifier: bool = False) -> str:
    extra = {
        "at_least_low": "boolean: any expression evidence, smile polarity, or broad affect is visibly interpretable",
        "at_least_moderate": "boolean: multiple facial regions are resolved enough to support a stable coarse expression judgment, beyond merely locating them",
        "at_least_high": "boolean: fine/subtle expression cues are clearly readable",
    }
    image_note = (
        "This is a focused enlarged target-face crop. Independently audit this borderline case."
        if verifier else
        "The image contains an enlarged target face on the left and advertisement context with that same target marked in red on the right."
    )
    return f"""Return one JSON object only. case_id={case_id}. {image_note} Code only the target.
Answer three nested thresholds rather than making one immediate four-way choice:
- at_least_low: any facial-expression evidence can be visibly interpreted; smile polarity or broad affect alone is sufficient. Do not infer "not smiling" merely because no smile is obvious when the mouth shape itself is unresolved.
- at_least_moderate: mouth plus at least one eye/brow region are resolved enough to support a stable coarse expression judgment. Merely locating those regions or guessing gaze does not satisfy moderate; if only smile polarity or broad affect survives, select low.
- at_least_high: eyes/brows and mouth are clear enough for fine or subtle expression cues.
High implies moderate and low; moderate implies low. Neutral/stern/subtle may satisfy high. Grain, profile, or illustration does not imply zero. Zero is reserved for cases where even coarse expression or smile-versus-non-smile is impossible.
Return substantive gaze/smile fields whenever their specific cue is usable. The pipeline derives face_expression_legibility from the booleans and ignores that categorical field, so put your best audit recommendation there only.
Output schema: {compact_schema(extra)}"""


def integrated_threshold_prompt(case_id: str) -> str:
    extra = {
        "eligible_face_visible": "boolean", "depiction_type": DEPICTION,
        "perceived_age": AGE, "perceived_gender_presentation": GENDER,
        "face_orientation": ORIENTATION, "mouth_covered": MOUTH,
        "mouth_covering": [*MOUTH_CAUSE, None],
        "at_least_low": "boolean: any expression evidence, smile polarity, or broad affect is visibly interpretable",
        "at_least_moderate": "boolean: multiple facial regions are resolved enough to support a stable coarse expression judgment, beyond merely locating them",
        "at_least_high": "boolean: fine/subtle expression cues are clearly readable",
    }
    return f"""Return one JSON object only. case_id={case_id}. The image contains an enlarged target face on the left and advertisement context with the same target marked in red on the right. Code only that target.
Code the ordinary person attributes, but determine expression legibility through three nested thresholds before assigning the category:
- at_least_low: any facial-expression evidence can be visibly interpreted; smile polarity or broad affect alone is sufficient. Do not infer non-smile when mouth shape is unresolved.
- at_least_moderate: mouth plus at least one eye/brow region are resolved enough for a stable coarse expression judgment. Merely locating them or guessing gaze is insufficient.
- at_least_high: eyes/brows and mouth are clear enough for fine/subtle cues. Neutral, stern, profile, grainy, or illustrated faces may be high when configuration is clear.
High implies moderate and low; moderate implies low. The pipeline derives face_expression_legibility from these booleans and ignores the categorical recommendation. Return substantive gaze/smile fields whenever their specific evidence is usable. gaze/smile are null only if derived legibility is zero. smile_intensity is non-null only for smile=yes; mouth_covering is non-null only for covered/partly.
Output schema: {compact_schema(extra)}"""


STRATEGIES = {
    "anchors_direct": {"image_field": "source_composite_path", "prompt": direct_prompt, "max_tokens": 700},
    "evidence_derived": {"image_field": "source_composite_path", "prompt": evidence_prompt, "max_tokens": 900},
    "ordinal_thresholds": {"image_field": "source_composite_path", "prompt": threshold_prompt, "max_tokens": 750},
    "dualview_anchors": {"image_field": "dual_view_path", "prompt": lambda case_id: direct_prompt(case_id, dual_view=True), "max_tokens": 700},
    "integrated_thresholds": {"image_field": "source_composite_path", "prompt": integrated_threshold_prompt, "max_tokens": 1200},
}
DISCOVERY_STRATEGIES = ["anchors_direct", "evidence_derived", "ordinal_thresholds", "dualview_anchors", "p2_zero_verify"]
ALL_STRATEGIES = [*DISCOVERY_STRATEGIES, "integrated_thresholds"]


def legibility_value(value: Any) -> str | None:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if value in LEGIBILITY:
        return str(value)
    text = str(value or "").strip().lower()
    aliases = {
        "0": LEGIBILITY[0], "zero": LEGIBILITY[0], "not_legible": LEGIBILITY[0],
        "1": LEGIBILITY[1], "low": LEGIBILITY[1], "low_legibility": LEGIBILITY[1],
        "2": LEGIBILITY[2], "moderate": LEGIBILITY[2], "moderate_legibility": LEGIBILITY[2],
        "3": LEGIBILITY[3], "high": LEGIBILITY[3], "high_legibility": LEGIBILITY[3],
    }
    return aliases.get(text)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1"}


def enum_scalar(value: Any, allowed: set[str]) -> str | None:
    """Tolerate models that wrap one requested enum in a one-item list."""
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    return str(value) if isinstance(value, str) and value in allowed else None


def normalize_common(raw: dict[str, Any], case_id: str) -> tuple[dict[str, Any], list[str]]:
    actions: list[str] = []
    legibility = legibility_value(raw.get("face_expression_legibility")) or LEGIBILITY[0]
    if legibility_value(raw.get("face_expression_legibility")) is None:
        actions.append("invalid/missing legibility defaulted to 0")
    gaze = enum_scalar(raw.get("gaze_target"), set(GAZE))
    smile = enum_scalar(raw.get("smile_present"), set(SMILE))
    intensity = enum_scalar(raw.get("smile_intensity"), set(INTENSITY))
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    result = {
        "case_id": case_id,
        "face_expression_legibility": legibility,
        "gaze_target": gaze,
        "smile_present": smile,
        "smile_intensity": intensity,
        "confidence": confidence,
        "review_flags": raw.get("review_flags") if isinstance(raw.get("review_flags"), list) else [],
    }
    return result, actions


def finalize_conditionals(result: dict[str, Any], actions: list[str]) -> None:
    if result["face_expression_legibility"] == LEGIBILITY[0]:
        for field in ("gaze_target", "smile_present", "smile_intensity"):
            if result.get(field) is not None:
                result[field] = None
                actions.append(f"{field}=null at legibility 0")
    elif result.get("smile_present") != "yes" and result.get("smile_intensity") is not None:
        result["smile_intensity"] = None
        actions.append("smile_intensity=null without smile=yes")


def normalize(strategy: str, raw: dict[str, Any], case_id: str) -> tuple[dict[str, Any], list[str]]:
    result, actions = normalize_common(raw, case_id)
    if strategy == "evidence_derived":
        locatable = as_bool(raw.get("facial_surface_locatable"))
        coarse = as_bool(raw.get("smile_polarity_judgeable")) or as_bool(raw.get("broad_affect_judgeable"))
        mouth = enum_scalar(raw.get("mouth_region_readability"), {"unreadable", "partial", "clear"})
        eyes = enum_scalar(raw.get("eye_region_readability"), {"unreadable", "one_or_partial", "both_clear"})
        brows = enum_scalar(raw.get("brow_cue_readability"), {"unreadable", "partial", "clear"})
        stable = as_bool(raw.get("stable_coarse_configuration"))
        fine = as_bool(raw.get("fine_expression_cues_judgeable"))
        if not locatable or not coarse:
            derived = LEGIBILITY[0]
        elif fine and mouth == "clear" and eyes == "both_clear" and brows in {"partial", "clear"}:
            derived = LEGIBILITY[3]
        elif stable and mouth in {"partial", "clear"} and eyes in {"one_or_partial", "both_clear"}:
            derived = LEGIBILITY[2]
        else:
            derived = LEGIBILITY[1]
        if result["face_expression_legibility"] != derived:
            actions.append(f"legibility derived from evidence: {result['face_expression_legibility']}->{derived}")
        result["face_expression_legibility"] = derived
        result["audit_evidence"] = {key: raw.get(key) for key in (
            "facial_surface_locatable", "smile_polarity_judgeable", "broad_affect_judgeable",
            "mouth_region_readability", "eye_region_readability", "brow_cue_readability",
            "stable_coarse_configuration", "fine_expression_cues_judgeable",
        )}
    elif strategy in {"ordinal_thresholds", "p2_zero_verify", "integrated_thresholds"}:
        low = as_bool(raw.get("at_least_low"))
        moderate = as_bool(raw.get("at_least_moderate"))
        high = as_bool(raw.get("at_least_high"))
        if high and not moderate:
            moderate = True; actions.append("monotonic repair: high=>moderate")
        if moderate and not low:
            low = True; actions.append("monotonic repair: moderate=>low")
        derived = LEGIBILITY[3] if high else LEGIBILITY[2] if moderate else LEGIBILITY[1] if low else LEGIBILITY[0]
        if result["face_expression_legibility"] != derived:
            actions.append(f"legibility derived from thresholds: {result['face_expression_legibility']}->{derived}")
        result["face_expression_legibility"] = derived
        result["audit_thresholds"] = {"at_least_low": low, "at_least_moderate": moderate, "at_least_high": high}
    finalize_conditionals(result, actions)
    return result, actions


def verifier_trigger(primary: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    if primary.get("face_expression_legibility") == LEGIBILITY[0]:
        reasons.append("primary_zero")
    if float(primary.get("confidence") or 0) < 0.75:
        reasons.append("low_confidence")
    if primary.get("face_expression_legibility") != LEGIBILITY[0] and primary.get("smile_present") in {None, "not_assessable"} and primary.get("gaze_target") in {None, "not_assessable"}:
        reasons.append("nonzero_but_both_downstream_unassessable")
    return bool(reasons), reasons

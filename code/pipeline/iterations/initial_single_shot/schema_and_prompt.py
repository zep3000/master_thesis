"""App v2 single-shot schema and prompt for OpenRouter Qwen experiments."""

from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft202012Validator


SCHEMA_VERSION = "qwen_iteration_app_v2_single_shot_v1"
PLAYBOOK_VERSION = "annotation_playbook_v1.15"
PROMPT_VERSION = "app_v2_single_shot_prompt_v1"


DEPICTION_TYPES = [
    "photo_of_person",
    "naturalistic_illustration",
    "stylized_illustration",
    "cartoon_or_caricature",
    "generic_human_figure",
    "photo_of_artwork_or_statue",
    "drawing_of_statue_monument_or_public_symbol",
    "mask_mannequin_doll_or_puppet",
    "personified_object",
    "nonhuman_creature_with_face",
    "schematic_icon_or_logo_face",
    "multiple_types_present",
]
AGE_VALUES = [
    "infant",
    "child",
    "adolescent",
    "young_adult",
    "middle_adult",
    "older_adult",
    "not_assessable",
]
GENDER_VALUES = [
    "feminine",
    "masculine",
    "ambiguous_or_androgynous",
    "not_assessable",
]
EXPRESSION_LEGIBILITY_VALUES = [
    "0_not_legible",
    "1_low_legibility",
    "2_moderate_legibility",
    "3_high_legibility",
]
FACE_ORIENTATION_VALUES = [
    "beyond_profile",
    "profile",
    "three_quarter",
    "frontal",
    "tilted_down",
    "tilted_up",
    "other",
    "not_assessable",
]
GAZE_VALUES = [
    "viewer_camera",
    "another_person",
    "advertised_product",
    "other_object",
    "off_frame_or_scene_direction",
    "eyes_covered",
    "closed_eyes",
    "not_assessable",
]
MOUTH_COVERED_VALUES = ["no", "yes", "partly", "not_assessable"]
MOUTH_COVERING_CAUSES = [
    "hand",
    "beard",
    "other_body_part",
    "part_of_another_person",
    "object",
    "object_in_mouth",
    "text_or_graphic_overlay",
    "cropped_by_page_edge",
    "other",
    "not_assessable",
]
SMILE_PRESENT_VALUES = ["yes", "no", "not_assessable"]
SMILE_INTENSITY_VALUES = ["1_slight", "2_clear", "3_broad", "4_laughter_like"]
GROUP_TYPE_VALUES = [
    "interacting_group",
    "posed_group",
    "audience",
    "background_population",
    "separate_portraits_or_composite",
    "other_group",
]
GROUP_AGE_VALUES = [
    "young_only",
    "middle_only",
    "older_only",
    "mostly_young",
    "mostly_middle",
    "mostly_older",
    "mixed",
    "not_assessable",
]
GROUP_GENDER_VALUES = [
    "feminine_only",
    "masculine_only",
    "mostly_feminine",
    "mostly_masculine",
    "mixed",
    "ambiguous_or_androgynous_present",
    "not_assessable",
]
GROUP_LEGIBILITY_VALUES = [
    "all_0_not_legible",
    "mostly_0_not_legible",
    "all_1_low_legibility",
    "mostly_1_low_legibility",
    "all_2_moderate_legibility",
    "mostly_2_moderate_legibility",
    "all_3_high_legibility",
    "mostly_3_high_legibility",
    "mixed_legibility",
]
GROUP_GAZE_VALUES = [
    "toward_viewer_camera",
    "toward_each_other",
    "toward_object",
    "off_frame_or_scene_direction",
    "mixed",
    "not_assessable",
]
GROUP_SMILE_PREVALENCE_VALUES = [
    "none",
    "minority",
    "about_half",
    "majority",
    "all",
    "not_assessable",
]
GROUP_SMILE_INTENSITY_VALUES = ["slight", "clear", "broad_or_laughter_like", "mixed"]
REVIEW_FLAG_VALUES = [
    "ad_boundary_unclear",
    "ad_eligibility_unclear",
    "face_count_unclear",
    "face_box_unclear",
    "duplicate_identity_unclear",
    "tiny_or_low_quality_faces",
    "person_attribute_unclear",
    "group_composition_unclear",
    "strange_or_missing_label",
    "json_or_schema_repair_needed",
]


def enum(values: list[str]) -> dict[str, Any]:
    return {"type": "string", "enum": values}


def nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required or list(properties),
    }


BBOX_1000_SCHEMA = {
    "type": "array",
    "items": {"type": "integer", "minimum": 0, "maximum": 1000},
    "minItems": 4,
    "maxItems": 4,
}
CONFIDENCE_SCHEMA = {"type": "number", "minimum": 0, "maximum": 1}
REVIEW_FLAGS_SCHEMA = {
    "type": "array",
    "items": enum(REVIEW_FLAG_VALUES),
    "uniqueItems": True,
}

PERSON_SCHEMA = obj(
    {
        "person_id": {"type": "string", "pattern": "^ad_[1-9][0-9]*_person_[1-9][0-9]*$"},
        "annotation_role": enum(["individual", "outstanding_individual", "duplicate"]),
        "face_bbox_1000": BBOX_1000_SCHEMA,
        "duplicate_of_person_id": nullable({"type": "string", "pattern": "^ad_[1-9][0-9]*_person_[1-9][0-9]*$"}),
        "duplicate_person_ids": {
            "type": "array",
            "items": {"type": "string", "pattern": "^ad_[1-9][0-9]*_person_[1-9][0-9]*$"},
            "uniqueItems": True,
        },
        "depiction_type": nullable(enum(DEPICTION_TYPES)),
        "perceived_age": nullable(enum(AGE_VALUES)),
        "perceived_gender_presentation": nullable(enum(GENDER_VALUES)),
        "face_expression_legibility": nullable(enum(EXPRESSION_LEGIBILITY_VALUES)),
        "face_orientation": nullable(enum(FACE_ORIENTATION_VALUES)),
        "gaze_target": nullable(enum(GAZE_VALUES)),
        "gaze_target_person_id": nullable({"type": "string", "maxLength": 80}),
        "mouth_covered": nullable(enum(MOUTH_COVERED_VALUES)),
        "mouth_covering_cause": nullable(enum(MOUTH_COVERING_CAUSES)),
        "mouth_covering_cause_other_text": nullable({"type": "string", "maxLength": 160}),
        "smile_present": nullable(enum(SMILE_PRESENT_VALUES)),
        "smile_intensity": nullable(enum(SMILE_INTENSITY_VALUES)),
        "confidence": CONFIDENCE_SCHEMA,
        "review_flags": REVIEW_FLAGS_SCHEMA,
    }
)

IDENTITY_GROUP_SCHEMA = obj(
    {
        "identity_group_id": {"type": "string", "pattern": "^ad_[1-9][0-9]*_identity_[1-9][0-9]*$"},
        "main_person_id": {"type": "string", "pattern": "^ad_[1-9][0-9]*_person_[1-9][0-9]*$"},
        "person_ids": {
            "type": "array",
            "items": {"type": "string", "pattern": "^ad_[1-9][0-9]*_person_[1-9][0-9]*$"},
            "minItems": 1,
            "uniqueItems": True,
        },
    }
)

GROUP_SCHEMA = obj(
    {
        "group_id": {"type": "string", "pattern": "^ad_[1-9][0-9]*_group_[1-9][0-9]*$"},
        "bbox_1000": BBOX_1000_SCHEMA,
        "group_type": enum(GROUP_TYPE_VALUES),
        "age_composition": enum(GROUP_AGE_VALUES),
        "gender_presentation_composition": enum(GROUP_GENDER_VALUES),
        "expression_legibility_distribution": enum(GROUP_LEGIBILITY_VALUES),
        "dominant_gaze": nullable(enum(GROUP_GAZE_VALUES)),
        "smile_prevalence": nullable(enum(GROUP_SMILE_PREVALENCE_VALUES)),
        "dominant_smile_intensity": nullable(enum(GROUP_SMILE_INTENSITY_VALUES)),
        "confidence": CONFIDENCE_SCHEMA,
        "review_flags": REVIEW_FLAGS_SCHEMA,
    }
)

ADVERTISEMENT_SCHEMA = obj(
    {
        "advertisement_id": {"type": "string", "pattern": "^ad_[1-9][0-9]*$"},
        "extent": enum(["full_page", "partial_page"]),
        "bbox_1000": BBOX_1000_SCHEMA,
        "depiction_type": enum(DEPICTION_TYPES),
        "face_depiction_count_band": enum([str(i) for i in range(1, 10)] + ["10_20", "20_plus"]),
        "has_outstanding_individuals": nullable(enum(["yes", "no"])),
        "duplicate_faces_present": nullable(enum(["yes", "no"])),
        "unique_face_count": nullable({"type": "integer", "minimum": 1, "maximum": 99}),
        "people": {"type": "array", "items": PERSON_SCHEMA, "maxItems": 12},
        "identity_groups": {"type": "array", "items": IDENTITY_GROUP_SCHEMA, "maxItems": 12},
        "groups": {"type": "array", "items": GROUP_SCHEMA, "maxItems": 12},
        "confidence": CONFIDENCE_SCHEMA,
        "review_flags": REVIEW_FLAGS_SCHEMA,
    }
)

MODEL_RESPONSE_SCHEMA = obj(
    {
        "schema_version": {"type": "string", "enum": [SCHEMA_VERSION]},
        "playbook_version": {"type": "string", "enum": [PLAYBOOK_VERSION]},
        "page": obj(
            {
                "qualifying_ad_count": {"type": "string", "pattern": "^(0|[1-9][0-9]?)$"},
                "no_qualifying_ad_reason": nullable(enum(["no_ads_on_page", "ads_present_no_visible_faces"])),
            }
        ),
        "advertisements_truncated": {"type": "boolean"},
        "advertisements": {"type": "array", "items": ADVERTISEMENT_SCHEMA, "maxItems": 20},
        "urgent_comments": {
            "type": "array",
            "items": obj(
                {
                    "scope": enum(["page", "advertisement", "person", "group"]),
                    "target_id": nullable({"type": "string", "maxLength": 80}),
                    "comment": {"type": "string", "minLength": 1, "maxLength": 240},
                }
            ),
            "maxItems": 12,
        },
        "confidence": CONFIDENCE_SCHEMA,
        "review_flags": REVIEW_FLAGS_SCHEMA,
    }
)

VALIDATOR = Draft202012Validator(MODEL_RESPONSE_SCHEMA)


def prompt_for_image(image_id: str, filename: str, page_type: str, year: int | None) -> str:
    schema_json = json.dumps(MODEL_RESPONSE_SCHEMA, ensure_ascii=True, separators=(",", ":"))
    year_text = "unknown" if year is None else str(year)
    return f"""
Return ONLY one valid compact JSON object. Do not use markdown fences, prose, comments, or hidden reasoning.
The first character must be {{ and the last character must be }}.

You are annotating one full historical Economist magazine page for a research project on faces and smiling in advertisements.

IMAGE METADATA
- image_id: {image_id}
- filename: {filename}
- page_type: {page_type}
- year: {year_text}

TASK
Apply annotation_playbook_v1.15 exactly. Use only visible evidence. Do not annotate brand, product category, ad topic, appeals, tone, emotion, race, ethnicity, actual identity, or authenticity of smiles.

MANDATORY DECISION RULE
- There is no unclear label. When visual evidence is available, choose the single best-fitting substantive category.
- Use not_assessable only when missing, occluded, too-small, cropped, blurred, low-contrast, or degraded evidence prevents a meaningful visual estimate.
- Use urgent_comments only for strange cases, missing label options, or decisions that need human review. Comments do not replace required fields.

ELIGIBLE FACE DEPICTIONS
- A face depiction is eligible when more than an ear or the back of a head is visible and the face can be located with a bounding box.
- Include eligible faces even when the facial expression is not readable.
- Include photographs, naturalistic/stylized illustrations, cartoons, artwork/statues, masks/mannequins/dolls/puppets, personified objects, nonhuman creatures with faces, and schematic/logo faces.
- Exclude ear-only cases, backs of heads without facial surface, and marks too small or degraded to confirm that a face is present.
- Count visible depictions, not unique real people. Repeated portraits, mirrors, collage repetitions, and repeated product shots get separate face boxes. Resolve exact duplicate face depictions through duplicate fields.

PAGE AND ADVERTISEMENT CODING
- qualifying_ad_count counts advertisements containing at least one eligible face depiction.
- If no ads are present: page.qualifying_ad_count="0", no_qualifying_ad_reason="no_ads_on_page", advertisements=[].
- If ads are present but none has an eligible face: page.qualifying_ad_count="0", no_qualifying_ad_reason="ads_present_no_visible_faces", advertisements=[].
- If qualifying ads are present: no_qualifying_ad_reason=null, and advertisements contains those ads in reading order.
- For a full-page single ad, extent="full_page" and bbox_1000=[0,0,1000,1000]. Otherwise bbox_1000 covers the complete advertising unit, including image and copy.
- Coordinates are integer [x1,y1,x2,y2] on a 0..1000 grid relative to the full page.

FACE COUNT ROUTE
- For 1 through 9 eligible faces in an ad: face_depiction_count_band is the exact string "1".."9"; draw and code every eligible individual face; groups=[]; has_outstanding_individuals=null.
- For 10 or more eligible faces: use face_depiction_count_band "10_20" or "20_plus"; code only prominent outstanding individuals if present; draw group boxes for the remaining crowd or group faces.
- has_outstanding_individuals is null for exact 1..9 ads, "yes" for crowd ads with outstanding individuals, and "no" for crowd ads without them.
- group boxes should include all remaining faces in a visually distinct group; separate groups only for distinct clusters, panels, scenes, or portrait sets.

DUPLICATES
- duplicate_faces_present is "yes" only when two or more individual boxes repeat the exact same represented face/identity with the same expression, for example a mirror, collage repetition, or repeated product shot.
- Do not merge merely similar-looking faces, or the same person in a different pose/expression.
- Every duplicated face still gets a person record. The clearest main record stores duplicate_person_ids. Duplicate records set duplicate_of_person_id to the main person and set detailed coding fields to null.
- identity_groups lists the main and duplicate records only when duplicate_faces_present="yes"; otherwise use [] and unique_face_count=null.

INDIVIDUAL FACE CODING
- If ad depiction_type is multiple_types_present, set each main person's depiction_type. Otherwise set person depiction_type=null.
- perceived_age values: {", ".join(AGE_VALUES)}.
- perceived_gender_presentation values: {", ".join(GENDER_VALUES)}. Code visible presentation, not identity.
- face_expression_legibility values: {", ".join(EXPRESSION_LEGIBILITY_VALUES)}. This measures how well the visible face supports expression coding, not emotional intensity.
- If face_expression_legibility is "0_not_legible", set gaze_target=null, gaze_target_person_id=null, smile_present=null, and smile_intensity=null. Still code face_orientation and mouth_covered if possible.
- face_orientation values: {", ".join(FACE_ORIENTATION_VALUES)}.
- gaze_target values: {", ".join(GAZE_VALUES)}. Code visible gaze direction. If gaze_target="another_person", set gaze_target_person_id to a visible person id or "person_without_bounding_box".
- mouth_covered values: {", ".join(MOUTH_COVERED_VALUES)}.
- mouth_covering_cause is null when mouth_covered="no". When mouth_covered is "yes" or "partly", choose one of: {", ".join(MOUTH_COVERING_CAUSES)}.
- smile_present values: {", ".join(SMILE_PRESENT_VALUES)}. Code visible facial configuration only.
- smile_intensity is null unless smile_present="yes"; then use one of: {", ".join(SMILE_INTENSITY_VALUES)}.

GROUP CODING
- group_type values: {", ".join(GROUP_TYPE_VALUES)}.
- age_composition values: {", ".join(GROUP_AGE_VALUES)}.
- gender_presentation_composition values: {", ".join(GROUP_GENDER_VALUES)}.
- expression_legibility_distribution values: {", ".join(GROUP_LEGIBILITY_VALUES)}.
- If expression_legibility_distribution="all_0_not_legible", set dominant_gaze=null, smile_prevalence=null, and dominant_smile_intensity=null.
- dominant_gaze values otherwise: {", ".join(GROUP_GAZE_VALUES)}.
- smile_prevalence values otherwise: {", ".join(GROUP_SMILE_PREVALENCE_VALUES)}.
- dominant_smile_intensity is null when smile_prevalence is "none" or "not_assessable"; otherwise use {", ".join(GROUP_SMILE_INTENSITY_VALUES)}.

ID AND OUTPUT RULES
- Use advertisement ids ad_1, ad_2, ...
- Use person ids ad_1_person_1, ad_1_person_2, ...
- Use group ids ad_1_group_1, ad_1_group_2, ...
- Use identity ids ad_1_identity_1, ad_1_identity_2, ...
- Confidence values are 0..1 visible-evidence confidence.
- review_flags can use only: {", ".join(REVIEW_FLAG_VALUES)}.
- Return exactly one object conforming to this JSON Schema. Do not output the schema itself.

JSON SCHEMA
{schema_json}
""".strip()


def validate_model_response(annotation: dict[str, Any]) -> list[str]:
    errors = [format_schema_error(error) for error in sorted(VALIDATOR.iter_errors(annotation), key=str)]
    errors.extend(validate_semantics(annotation))
    return errors


def format_schema_error(error: Any) -> str:
    path = ".".join(str(part) for part in error.absolute_path) or "<root>"
    return f"{path}: {error.message}"


def validate_semantics(annotation: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    page = annotation.get("page") if isinstance(annotation, dict) else None
    ads = annotation.get("advertisements") if isinstance(annotation, dict) else None
    if not isinstance(page, dict) or not isinstance(ads, list):
        return errors

    count = page.get("qualifying_ad_count")
    try:
        count_int = int(count)
    except Exception:
        count_int = None
    reason = page.get("no_qualifying_ad_reason")
    if count_int == 0:
        if reason not in {"no_ads_on_page", "ads_present_no_visible_faces"}:
            errors.append("page.no_qualifying_ad_reason must be set when qualifying_ad_count is 0")
        if ads:
            errors.append("advertisements must be empty when qualifying_ad_count is 0")
    elif isinstance(count_int, int):
        if reason is not None:
            errors.append("page.no_qualifying_ad_reason must be null when qualifying_ad_count is positive")
        if annotation.get("advertisements_truncated") is not True and len(ads) != count_int:
            errors.append("advertisements length must equal page.qualifying_ad_count unless advertisements_truncated is true")

    for ad_index, ad in enumerate(ads):
        if not isinstance(ad, dict):
            continue
        prefix = f"advertisements[{ad_index}]"
        bbox = ad.get("bbox_1000")
        if bbox == [0, 0, 1000, 1000] and ad.get("extent") != "full_page":
            errors.append(f"{prefix}: full-page bbox should use extent=full_page")
        check_bbox(errors, f"{prefix}.bbox_1000", bbox)
        people = ad.get("people") if isinstance(ad.get("people"), list) else []
        groups = ad.get("groups") if isinstance(ad.get("groups"), list) else []
        band = ad.get("face_depiction_count_band")
        if band in {str(i) for i in range(1, 10)}:
            if len(people) != int(band):
                errors.append(f"{prefix}: exact face band {band} requires exactly {band} people")
            if groups:
                errors.append(f"{prefix}: exact 1..9 face route requires groups=[]")
            if ad.get("has_outstanding_individuals") is not None:
                errors.append(f"{prefix}: exact 1..9 face route requires has_outstanding_individuals=null")
        elif band in {"10_20", "20_plus"}:
            has_outstanding = ad.get("has_outstanding_individuals")
            if has_outstanding == "yes" and not people:
                errors.append(f"{prefix}: has_outstanding_individuals=yes requires at least one person")
            if has_outstanding == "no" and people:
                errors.append(f"{prefix}: has_outstanding_individuals=no requires people=[]")
            if not groups:
                errors.append(f"{prefix}: crowd route requires at least one group")
        if ad.get("duplicate_faces_present") == "no":
            if ad.get("identity_groups"):
                errors.append(f"{prefix}: duplicate_faces_present=no requires identity_groups=[]")
            if ad.get("unique_face_count") is not None:
                errors.append(f"{prefix}: duplicate_faces_present=no requires unique_face_count=null")

        person_ids = {p.get("person_id") for p in people if isinstance(p, dict)}
        for person_index, person in enumerate(people):
            if not isinstance(person, dict):
                continue
            person_prefix = f"{prefix}.people[{person_index}]"
            check_bbox(errors, f"{person_prefix}.face_bbox_1000", person.get("face_bbox_1000"))
            is_duplicate = person.get("annotation_role") == "duplicate"
            if is_duplicate:
                if not person.get("duplicate_of_person_id"):
                    errors.append(f"{person_prefix}: duplicate role requires duplicate_of_person_id")
                for field in [
                    "perceived_age",
                    "perceived_gender_presentation",
                    "face_expression_legibility",
                    "gaze_target",
                    "mouth_covered",
                    "smile_present",
                    "smile_intensity",
                ]:
                    if person.get(field) is not None:
                        errors.append(f"{person_prefix}: duplicate role requires {field}=null")
                continue
            if person.get("face_expression_legibility") == "0_not_legible":
                for field in ["gaze_target", "gaze_target_person_id", "smile_present", "smile_intensity"]:
                    if person.get(field) is not None:
                        errors.append(f"{person_prefix}: 0_not_legible requires {field}=null")
            if person.get("mouth_covered") == "no" and person.get("mouth_covering_cause") is not None:
                errors.append(f"{person_prefix}: mouth_covered=no requires mouth_covering_cause=null")
            if person.get("mouth_covered") in {"yes", "partly"} and person.get("mouth_covering_cause") is None:
                errors.append(f"{person_prefix}: mouth covering yes/partly requires a cause")
            if person.get("smile_present") == "yes" and person.get("smile_intensity") is None:
                errors.append(f"{person_prefix}: smile_present=yes requires smile_intensity")
            if person.get("smile_present") in {"no", "not_assessable"} and person.get("smile_intensity") is not None:
                errors.append(f"{person_prefix}: smile_present no/not_assessable requires smile_intensity=null")
            if person.get("gaze_target") == "another_person":
                target = person.get("gaze_target_person_id")
                if not target:
                    errors.append(f"{person_prefix}: gaze_target=another_person requires gaze_target_person_id")
                elif target not in person_ids and target != "person_without_bounding_box":
                    errors.append(f"{person_prefix}: gaze_target_person_id does not refer to an existing person")

        for group_index, group in enumerate(groups):
            if not isinstance(group, dict):
                continue
            group_prefix = f"{prefix}.groups[{group_index}]"
            check_bbox(errors, f"{group_prefix}.bbox_1000", group.get("bbox_1000"))
            if group.get("expression_legibility_distribution") == "all_0_not_legible":
                for field in ["dominant_gaze", "smile_prevalence", "dominant_smile_intensity"]:
                    if group.get(field) is not None:
                        errors.append(f"{group_prefix}: all_0_not_legible requires {field}=null")
            if group.get("smile_prevalence") in {"none", "not_assessable"} and group.get("dominant_smile_intensity") is not None:
                errors.append(f"{group_prefix}: smile_prevalence none/not_assessable requires dominant_smile_intensity=null")
            if group.get("smile_prevalence") in {"minority", "about_half", "majority", "all"} and group.get("dominant_smile_intensity") is None:
                errors.append(f"{group_prefix}: positive smile_prevalence requires dominant_smile_intensity")
    return errors


def check_bbox(errors: list[str], path: str, bbox: Any) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return
    if not all(isinstance(value, int) for value in bbox):
        return
    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2:
        errors.append(f"{path}: expected x1<x2 and y1<y2")

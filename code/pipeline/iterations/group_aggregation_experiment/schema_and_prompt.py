"""Frozen schemas and prompts for the group aggregation experiment.

The experiment deliberately compares direct group judgments with two forms of
individual-first inference.  Prompts are kept here so their exact text is
versioned independently of the runners and evaluation code.
"""

from __future__ import annotations

import json
from typing import Any


PLAYBOOK_VERSION = "annotation_playbook_v1.17"
EXPERIMENT_VERSION = "qwen_group_aggregation_first50_v1"
DIRECT_PROMPT_VERSION = "qwen_group_direct_oracle_crop_v1"
INVENTORY_PROMPT_VERSION = "qwen_group_individual_inventory_oracle_crop_v1"
TILE_PROMPT_VERSION = "qwen_group_individual_tile_localization_v1"
FACE_PROMPT_VERSION = "qwen_group_individual_face_attributes_v1"
REDUCER_PROMPT_VERSION = "qwen_group_individual_text_reducer_v1"
TALLY_PROMPT_VERSION = "qwen_group_distribution_tally_oracle_crop_v1"

AGE_VALUES = [
    "infant",
    "child",
    "adolescent",
    "young_adult",
    "middle_adult",
    "older_adult",
    "not_assessable",
]
GENDER_VALUES = ["feminine", "masculine", "ambiguous_or_androgynous", "not_assessable"]
LEGIBILITY_VALUES = ["0_not_legible", "1_low_legibility", "2_moderate_legibility", "3_high_legibility"]
ORIENTATION_VALUES = ["beyond_profile", "profile", "three_quarter", "frontal", "other", "not_assessable"]
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
SMILE_VALUES = ["yes", "no", "not_assessable"]
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
GROUP_SMILE_VALUES = ["none", "minority", "about_half", "majority", "all", "not_assessable"]
GROUP_INTENSITY_VALUES = ["slight", "clear", "broad_or_laughter_like", "mixed"]


def enum(values: list[str]) -> dict[str, Any]:
    return {"type": "string", "enum": values}


def nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


GROUP_FIELDS = {
    "group_type": enum(GROUP_TYPE_VALUES),
    "age_composition": enum(GROUP_AGE_VALUES),
    "gender_presentation_composition": enum(GROUP_GENDER_VALUES),
    "expression_legibility_distribution": enum(GROUP_LEGIBILITY_VALUES),
    "dominant_gaze": nullable(enum(GROUP_GAZE_VALUES)),
    "smile_prevalence": nullable(enum(GROUP_SMILE_VALUES)),
    "dominant_smile_intensity": nullable(enum(GROUP_INTENSITY_VALUES)),
}

DIRECT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string", "enum": ["qwen_group_direct_v1"]},
        "group_key": {"type": "string"},
        "estimated_eligible_face_count": {"type": "integer", "minimum": 0, "maximum": 99},
        **GROUP_FIELDS,
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "review_flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "schema_version",
        "group_key",
        "estimated_eligible_face_count",
        *GROUP_FIELDS.keys(),
        "confidence",
        "review_flags",
    ],
}

INDIVIDUAL_PROPERTIES = {
    "perceived_age": enum(AGE_VALUES),
    "perceived_gender_presentation": enum(GENDER_VALUES),
    "face_expression_legibility": enum(LEGIBILITY_VALUES),
    "face_orientation": enum(ORIENTATION_VALUES),
    "gaze_target": nullable(enum(GAZE_VALUES)),
    "smile_present": nullable(enum(SMILE_VALUES)),
    "smile_intensity": nullable(enum(SMILE_INTENSITY_VALUES)),
}

INVENTORY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string", "enum": ["qwen_group_individual_inventory_v1"]},
        "group_key": {"type": "string"},
        "group_type": enum(GROUP_TYPE_VALUES),
        "faces_truncated": {"type": "boolean"},
        "faces": {
            "type": "array",
            "maxItems": 40,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "face_local_id": {"type": "string"},
                    "bbox_1000": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                    },
                    **INDIVIDUAL_PROPERTIES,
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["face_local_id", "bbox_1000", *INDIVIDUAL_PROPERTIES.keys(), "confidence"],
            },
        },
    },
    "required": ["schema_version", "group_key", "group_type", "faces_truncated", "faces"],
}

TILE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string", "enum": ["qwen_group_tile_faces_v1"]},
        "group_key": {"type": "string"},
        "tile_id": {"type": "string"},
        "faces": {
            "type": "array",
            "maxItems": 30,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "face_local_id": {"type": "string"},
                    "bbox_1000": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["face_local_id", "bbox_1000", "confidence"],
            },
        },
    },
    "required": ["schema_version", "group_key", "tile_id", "faces"],
}

FACE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string", "enum": ["qwen_group_face_attributes_v1"]},
        "face_task_id": {"type": "string"},
        "eligible_face_visible": {"type": "boolean"},
        **INDIVIDUAL_PROPERTIES,
        "visual_blockers": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "schema_version",
        "face_task_id",
        "eligible_face_visible",
        *INDIVIDUAL_PROPERTIES.keys(),
        "visual_blockers",
        "confidence",
    ],
}

REDUCER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string", "enum": ["qwen_group_text_reducer_v1"]},
        "group_key": {"type": "string"},
        **GROUP_FIELDS,
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "review_flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["schema_version", "group_key", *GROUP_FIELDS.keys(), "confidence", "review_flags"],
}

PERCENT = {"type": "number", "minimum": 0, "maximum": 100}
TALLY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string", "enum": ["qwen_group_distribution_tally_v1"]},
        "group_key": {"type": "string"},
        "group_type": enum(GROUP_TYPE_VALUES),
        "estimated_eligible_face_count": {"type": "integer", "minimum": 0, "maximum": 99},
        "age_percent": {
            "type": "object",
            "properties": {key: PERCENT for key in ("young", "middle", "older", "not_assessable")},
            "required": ["young", "middle", "older", "not_assessable"],
        },
        "gender_percent": {
            "type": "object",
            "properties": {key: PERCENT for key in ("feminine", "masculine", "ambiguous_or_androgynous", "not_assessable")},
            "required": ["feminine", "masculine", "ambiguous_or_androgynous", "not_assessable"],
        },
        "legibility_percent": {
            "type": "object",
            "properties": {key: PERCENT for key in LEGIBILITY_VALUES},
            "required": LEGIBILITY_VALUES,
        },
        "gaze_percent": {
            "type": "object",
            "properties": {key: PERCENT for key in ("toward_viewer_camera", "toward_each_other", "toward_object", "off_frame_or_scene_direction", "not_assessable")},
            "required": ["toward_viewer_camera", "toward_each_other", "toward_object", "off_frame_or_scene_direction", "not_assessable"],
        },
        "smile_percent": {
            "type": "object",
            "properties": {key: PERCENT for key in ("yes", "no", "not_assessable")},
            "required": ["yes", "no", "not_assessable"],
        },
        "smiling_face_intensity_percent": {
            "type": "object",
            "properties": {key: PERCENT for key in ("slight", "clear", "broad_or_laughter_like", "not_assessable")},
            "required": ["slight", "clear", "broad_or_laughter_like", "not_assessable"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "review_flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "schema_version", "group_key", "group_type", "estimated_eligible_face_count",
        "age_percent", "gender_percent", "legibility_percent", "gaze_percent",
        "smile_percent", "smiling_face_intensity_percent", "confidence", "review_flags",
    ],
}


GROUP_RULES = f"""
People-area labels:
- group_type: {', '.join(GROUP_TYPE_VALUES)}.
- age_composition: {', '.join(GROUP_AGE_VALUES)}.
- gender_presentation_composition: {', '.join(GROUP_GENDER_VALUES)}.
- expression_legibility_distribution: {', '.join(GROUP_LEGIBILITY_VALUES)}.
- dominant_gaze: {', '.join(GROUP_GAZE_VALUES)}.
- smile_prevalence: {', '.join(GROUP_SMILE_VALUES)}.
- dominant_smile_intensity: {', '.join(GROUP_INTENSITY_VALUES)}.
Use an all/only label when all or nearly all assessable faces share it; use mostly when one category clearly predominates; use mixed when no category clearly predominates. Use not_assessable only when visual evidence is genuinely unusable, not merely difficult. If expression legibility is all_0_not_legible, set gaze, smile prevalence, and smile intensity to null. If smile prevalence is none or not_assessable, set dominant smile intensity to null.
""".strip()


def direct_prompt(group_key: str) -> str:
    return f"""
Return only one compact JSON object. You are annotating one pre-cropped people area from a historical advertisement. The crop boundary is given; do not split the area and do not enumerate individual people. Judge the unified/tendency-level properties of the eligible faces in the area.

group_key: {group_key}

{GROUP_RULES}

Use visible facial configuration, not inferred emotion or narrative intent. Count only eligible faces for which more than an ear or the back of a head is visible and a facial surface can be located.

JSON schema:
{json.dumps(DIRECT_SCHEMA, separators=(',', ':'))}
""".strip()


def inventory_prompt(group_key: str) -> str:
    return f"""
Return only one compact JSON object. This is an individual-first condition for one pre-cropped people area from a historical advertisement.

group_key: {group_key}

Locate every eligible face visible in the crop and return one record per face. Count a face when more than an ear or the back of a head is visible and a facial surface can be located. Do not replace several people with a group record. Coordinates use one 0..1000 grid relative to this crop. Use the smallest box containing the visible face/head features; exclude torso, captions, and empty background. Set faces_truncated=true only if output limits prevent listing all faces.

Code each face independently. Age values: {', '.join(AGE_VALUES)}. Gender-presentation values: {', '.join(GENDER_VALUES)}. Expression-legibility values: {', '.join(LEGIBILITY_VALUES)}. Orientation values: {', '.join(ORIENTATION_VALUES)}. Gaze values: {', '.join(GAZE_VALUES)}. Smile values: {', '.join(SMILE_VALUES)}. Smile-intensity values: {', '.join(SMILE_INTENSITY_VALUES)}. When legibility is 0_not_legible, gaze and smile fields must be null. Smile intensity is non-null only when smile_present=yes.

Also classify only the visual arrangement as group_type; the other aggregate labels will be derived later.

JSON schema:
{json.dumps(INVENTORY_SCHEMA, separators=(',', ':'))}
""".strip()


def tile_prompt(group_key: str, tile_id: str) -> str:
    return f"""
Return only one compact JSON object. Locate every eligible face visible in this one tile from people area {group_key}. Tile id: {tile_id}. This request contains one image and one 0..1000 coordinate system relative to the tile. Include partial faces at tile borders when a facial surface is visible. Return tight face/head boxes, not full bodies. Do not code attributes.

JSON schema:
{json.dumps(TILE_SCHEMA, separators=(',', ':'))}
""".strip()


def face_prompt(face_task_id: str, group_key: str) -> str:
    return f"""
Return only one compact JSON object. The supplied single composite image has a large target-face panel and a wider people-area context panel with the same target outlined in red. Code only that target.

face_task_id: {face_task_id}
group_key: {group_key}

First decide whether an eligible face is actually visible. If not, set eligible_face_visible=false and use not_assessable/null values. Otherwise code: perceived age ({', '.join(AGE_VALUES)}); gender presentation ({', '.join(GENDER_VALUES)}); expression legibility ({', '.join(LEGIBILITY_VALUES)}); horizontal face orientation ({', '.join(ORIENTATION_VALUES)}); gaze ({', '.join(GAZE_VALUES)}); smile presence ({', '.join(SMILE_VALUES)}); and smile intensity ({', '.join(SMILE_INTENSITY_VALUES)}). Use visible facial configuration only. If legibility is 0_not_legible, gaze and smile fields must be null. If legibility is 1-3 and facial configuration is visible, prefer a substantive yes/no smile judgment over not_assessable. Smile intensity is non-null only for smile_present=yes.

JSON schema:
{json.dumps(FACE_SCHEMA, separators=(',', ':'))}
""".strip()


def reducer_prompt(group_key: str, group_type: str, individuals: list[dict[str, Any]]) -> str:
    compact = json.dumps(individuals, ensure_ascii=True, separators=(",", ":"))
    return f"""
Return only one compact JSON object. You are reducing individual face annotations into one people-area annotation. No image is supplied in this condition. Use only the individual evidence below and the separately observed visual arrangement.

group_key: {group_key}
group_type: {group_type}
individual face annotations: {compact}

{GROUP_RULES}

Do not invent people or visual evidence absent from the individual list. Preserve group_type exactly as supplied. Return one aggregate record.

JSON schema:
{json.dumps(REDUCER_SCHEMA, separators=(',', ':'))}
""".strip()


def tally_prompt(group_key: str) -> str:
    return f"""
Return only one compact JSON object. This is a distribution-measurement follow-up for one pre-cropped people area from a historical advertisement.

group_key: {group_key}

Do not return face boxes or individual records. Inspect the area holistically and estimate percentage distributions over all eligible visible faces. Each percentage object must sum to 100, except smiling_face_intensity_percent: it must sum to 100 among estimated smiling faces, or contain all zeros when no smiling face is visible. Use multiples of 5 when the visual evidence is approximate. Count a face only when more than an ear or the back of a head is visible and a facial surface can be located.

Age families: young includes infant through young adult; then middle and older. Gender is visible presentation only. Legibility levels use the playbook 0–3 scale. For gaze, classify each assessable face into viewer/camera, another person, an object/product, or off-frame/scene direction. For smile, prefer substantive yes/no when expression legibility permits; reserve not_assessable for unusable facial evidence. Intensity is conditional on smiling faces only.

Also classify the visual arrangement as group_type. Do not directly choose only/mostly/mixed or prevalence labels; those will be mapped deterministically from the percentages.

JSON schema:
{json.dumps(TALLY_SCHEMA, separators=(',', ':'))}
""".strip()


def validate_bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, int) and 0 <= item <= 1000 for item in value)
        and value[0] < value[2]
        and value[1] < value[3]
    )


def validate_response(stage: str, annotation: dict[str, Any], expected_key: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(annotation, dict):
        return ["response is not an object"]
    id_field = "face_task_id" if stage == "face" else "group_key"
    if annotation.get(id_field) != expected_key:
        errors.append(f"{id_field} mismatch")
    if stage == "tile" and not isinstance(annotation.get("tile_id"), str):
        errors.append("tile_id missing")
    if stage in {"inventory", "tile"}:
        faces = annotation.get("faces")
        if not isinstance(faces, list):
            errors.append("faces is not an array")
        else:
            for index, face in enumerate(faces):
                if not isinstance(face, dict) or not validate_bbox(face.get("bbox_1000")):
                    errors.append(f"faces[{index}] invalid bbox")
    if stage in {"direct", "reducer"}:
        allowed_group = {
            "group_type": GROUP_TYPE_VALUES,
            "age_composition": GROUP_AGE_VALUES,
            "gender_presentation_composition": GROUP_GENDER_VALUES,
            "expression_legibility_distribution": GROUP_LEGIBILITY_VALUES,
            "dominant_gaze": [*GROUP_GAZE_VALUES, None],
            "smile_prevalence": [*GROUP_SMILE_VALUES, None],
            "dominant_smile_intensity": [*GROUP_INTENSITY_VALUES, None],
        }
        for field in GROUP_FIELDS:
            if field not in annotation:
                errors.append(f"missing {field}")
            elif annotation.get(field) not in allowed_group[field]:
                errors.append(f"invalid {field}: {annotation.get(field)!r}")
    if stage in {"inventory", "face"}:
        allowed_individual = {
            "perceived_age": AGE_VALUES,
            "perceived_gender_presentation": GENDER_VALUES,
            "face_expression_legibility": LEGIBILITY_VALUES,
            "face_orientation": ORIENTATION_VALUES,
            "gaze_target": [*GAZE_VALUES, None],
            "smile_present": [*SMILE_VALUES, None],
            "smile_intensity": [*SMILE_INTENSITY_VALUES, None],
        }
        faces = annotation.get("faces") if stage == "inventory" else [annotation]
        if isinstance(faces, list):
            for index, face in enumerate(faces):
                if not isinstance(face, dict):
                    continue
                for field, values in allowed_individual.items():
                    if face.get(field) not in values:
                        errors.append(f"face[{index}] invalid {field}: {face.get(field)!r}")
                legibility = face.get("face_expression_legibility")
                if legibility == "0_not_legible" and any(face.get(field) is not None for field in ("gaze_target", "smile_present", "smile_intensity")):
                    errors.append(f"face[{index}] zero-legibility route violation")
                if face.get("smile_present") != "yes" and face.get("smile_intensity") is not None:
                    errors.append(f"face[{index}] smile intensity route violation")
    if stage == "inventory" and annotation.get("group_type") not in GROUP_TYPE_VALUES:
        errors.append(f"invalid group_type: {annotation.get('group_type')!r}")
    return errors

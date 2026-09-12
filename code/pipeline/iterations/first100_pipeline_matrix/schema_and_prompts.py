"""Frozen schemas and prompts for the first-100 four-pipeline matrix."""

from __future__ import annotations

import json
from typing import Any

PLAYBOOK_VERSION = "annotation_playbook_v1.17"
PROMPT_VERSION = "first100_matrix_frozen_v1_2"

DEPICTION_TYPES = [
    "photo_of_person", "illustration", "cartoon_or_caricature", "generic_human_figure",
    "photo_of_artwork_or_statue", "drawing_of_statue_monument_or_public_symbol",
    "mask_mannequin_doll_or_puppet", "personified_object", "nonhuman_creature_with_face",
    "schematic_icon_or_logo_face", "multiple_types_present",
]
AGE_VALUES = ["infant", "child", "adolescent", "young_adult", "middle_adult", "older_adult", "not_assessable"]
GENDER_VALUES = ["feminine", "masculine", "ambiguous_or_androgynous", "not_assessable"]
LEGIBILITY_VALUES = ["0_not_legible", "1_low_legibility", "2_moderate_legibility", "3_high_legibility"]
ORIENTATION_VALUES = ["beyond_profile", "profile", "three_quarter", "frontal", "tilted_down", "tilted_up", "other", "not_assessable"]
GAZE_VALUES = ["viewer_camera", "another_person", "advertised_product", "other_object", "off_frame_or_scene_direction", "eyes_covered", "closed_eyes", "not_assessable"]
MOUTH_VALUES = ["no", "yes", "partly", "not_assessable"]
MOUTH_CAUSES = ["hand", "beard", "other_body_part", "part_of_another_person", "object", "object_in_mouth", "text_or_graphic_overlay", "cropped_by_page_edge", "other", "not_assessable"]
SMILE_VALUES = ["yes", "no", "not_assessable"]
INTENSITY_VALUES = ["1_slight", "2_clear", "3_broad", "4_laughter_like"]
GROUP_TYPES = ["interacting_group", "posed_group", "audience", "background_population", "separate_portraits_or_composite", "other_group"]
GROUP_AGES = ["young_only", "middle_only", "older_only", "mostly_young", "mostly_middle", "mostly_older", "mixed", "not_assessable"]
GROUP_GENDERS = ["feminine_only", "masculine_only", "mostly_feminine", "mostly_masculine", "mixed", "ambiguous_or_androgynous_present", "not_assessable"]
GROUP_LEGIBILITY = ["all_0_not_legible", "mostly_0_not_legible", "all_1_low_legibility", "mostly_1_low_legibility", "all_2_moderate_legibility", "mostly_2_moderate_legibility", "all_3_high_legibility", "mostly_3_high_legibility", "mixed_legibility"]
GROUP_GAZE = ["toward_viewer_camera", "toward_each_other", "toward_object", "off_frame_or_scene_direction", "mixed", "not_assessable"]
GROUP_SMILE = ["none", "minority", "about_half", "majority", "all", "not_assessable"]
GROUP_INTENSITY = ["slight", "clear", "broad_or_laughter_like", "mixed"]
COUNT_BANDS = [str(i) for i in range(1, 10)] + ["10_20", "20_plus"]


def en(values: list[str]) -> dict[str, Any]:
    return {"type": "string", "enum": values}


def nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


BBOX = {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000}, "minItems": 4, "maxItems": 4}
PERSON_ATTR_FIELDS = {
    "depiction_type": nullable(en(DEPICTION_TYPES)),
    "perceived_age": en(AGE_VALUES),
    "perceived_gender_presentation": en(GENDER_VALUES),
    "face_expression_legibility": en(LEGIBILITY_VALUES),
    "face_orientation": en(ORIENTATION_VALUES),
    "gaze_target": nullable(en(GAZE_VALUES)),
    "gaze_target_person_unboxed": nullable({"type": "boolean"}),
    "mouth_covered": en(MOUTH_VALUES),
    "mouth_covering": nullable(en(MOUTH_CAUSES)),
    "smile_present": nullable(en(SMILE_VALUES)),
    "smile_intensity": nullable(en(INTENSITY_VALUES)),
}
GROUP_ATTR_FIELDS = {
    "group_type": en(GROUP_TYPES), "age_composition": en(GROUP_AGES),
    "gender_presentation_composition": en(GROUP_GENDERS),
    "expression_legibility_distribution": en(GROUP_LEGIBILITY),
    "dominant_gaze": nullable(en(GROUP_GAZE)), "smile_prevalence": nullable(en(GROUP_SMILE)),
    "dominant_smile_intensity": nullable(en(GROUP_INTENSITY)),
}

PERSON_FULL = {
    "type": "object", "properties": {
        "person_id": {"type": "string"}, "annotation_role": en(["individual", "outstanding_individual", "duplicate"]),
        "face_bbox_1000": BBOX, **PERSON_ATTR_FIELDS, "confidence": {"type": "number"}, "review_flags": {"type": "array"},
    }, "required": ["person_id", "annotation_role", "face_bbox_1000", *PERSON_ATTR_FIELDS, "confidence", "review_flags"],
}
GROUP_FULL = {
    "type": "object", "properties": {
        "group_id": {"type": "string"}, "bbox_1000": BBOX, **GROUP_ATTR_FIELDS,
        "confidence": {"type": "number"}, "review_flags": {"type": "array"},
    }, "required": ["group_id", "bbox_1000", *GROUP_ATTR_FIELDS, "confidence", "review_flags"],
}
AD_FULL = {
    "type": "object", "properties": {
        "advertisement_id": {"type": "string"}, "extent": en(["full_page", "partial_page"]), "bbox_1000": BBOX,
        "depiction_type": en(DEPICTION_TYPES), "face_depiction_count_band": en(COUNT_BANDS),
        "has_outstanding_individuals": nullable(en(["yes", "no"])), "duplicate_faces_present": nullable(en(["yes", "no"])),
        "unique_face_count": nullable({"type": "integer"}), "people": {"type": "array", "items": PERSON_FULL, "maxItems": 9},
        "groups": {"type": "array", "items": GROUP_FULL, "maxItems": 6}, "confidence": {"type": "number"}, "review_flags": {"type": "array"},
    }, "required": ["advertisement_id", "extent", "bbox_1000", "depiction_type", "face_depiction_count_band", "has_outstanding_individuals", "duplicate_faces_present", "unique_face_count", "people", "groups", "confidence", "review_flags"],
}
MONOLITHIC_SCHEMA = {
    "type": "object", "properties": {
        "schema_version": {"type": "string", "enum": ["qwen_first100_monolithic_v1"]},
        "image_id": {"type": "string"}, "qualifying_ad_count": {"type": "integer"},
        "no_qualifying_ad_reason": nullable(en(["no_ads_on_page", "ads_present_no_visible_faces"])),
        "advertisements": {"type": "array", "items": AD_FULL}, "review_flags": {"type": "array"},
    }, "required": ["schema_version", "image_id", "qualifying_ad_count", "no_qualifying_ad_reason", "advertisements", "review_flags"],
}

ROUTE_PERSON = {"type": "object", "properties": {"person_id": {"type": "string"}, "face_bbox_1000": BBOX, "annotation_role": en(["individual", "outstanding_individual"]), "prominence_reason": nullable({"type": "string"}), "confidence": {"type": "number"}}, "required": ["person_id", "face_bbox_1000", "annotation_role", "prominence_reason", "confidence"]}
ROUTE_GROUP = {"type": "object", "properties": {"group_id": {"type": "string"}, "bbox_1000": BBOX, "group_type_hint": nullable(en(GROUP_TYPES)), "confidence": {"type": "number"}}, "required": ["group_id", "bbox_1000", "group_type_hint", "confidence"]}
ROUTE_AD = {"type": "object", "properties": {
    "advertisement_id": {"type": "string"}, "extent": en(["full_page", "partial_page"]), "bbox_1000": BBOX,
    "depiction_type": en(DEPICTION_TYPES), "face_depiction_count_band": en(COUNT_BANDS),
    "duplicate_faces_present": nullable(en(["yes", "no"])), "unique_face_count": nullable({"type": "integer"}),
    "people": {"type": "array", "items": ROUTE_PERSON, "maxItems": 9}, "groups": {"type": "array", "items": ROUTE_GROUP, "maxItems": 6},
    "confidence": {"type": "number"}, "review_flags": {"type": "array"},
}, "required": ["advertisement_id", "extent", "bbox_1000", "depiction_type", "face_depiction_count_band", "duplicate_faces_present", "unique_face_count", "people", "groups", "confidence", "review_flags"]}
PAGE_ROUTER_SCHEMA = {"type": "object", "properties": {"schema_version": {"type": "string", "enum": ["qwen_first100_page_router_v1"]}, "image_id": {"type": "string"}, "qualifying_ad_count": {"type": "integer"}, "no_qualifying_ad_reason": nullable(en(["no_ads_on_page", "ads_present_no_visible_faces"])), "advertisements": {"type": "array", "items": ROUTE_AD}, "review_flags": {"type": "array"}}, "required": ["schema_version", "image_id", "qualifying_ad_count", "no_qualifying_ad_reason", "advertisements", "review_flags"]}
PAGE_LOCATOR_SCHEMA = {"type": "object", "properties": {"schema_version": {"type": "string", "enum": ["qwen_first100_page_locator_v1"]}, "image_id": {"type": "string"}, "qualifying_ads": {"type": "array", "maxItems": 6, "items": {"type": "object", "properties": {"advertisement_id": {"type": "string"}, "bbox_1000": BBOX, "extent": en(["full_page", "partial_page"]), "confidence": {"type": "number"}, "review_flags": {"type": "array"}}, "required": ["advertisement_id", "bbox_1000", "extent", "confidence", "review_flags"]}}, "no_qualifying_ad_reason": nullable(en(["no_ads_on_page", "ads_present_no_visible_faces"])), "review_flags": {"type": "array"}}, "required": ["schema_version", "image_id", "qualifying_ads", "no_qualifying_ad_reason", "review_flags"]}
AD_STRUCTURE_SCHEMA = {"type": "object", "properties": {"schema_version": {"type": "string", "enum": ["qwen_first100_ad_structure_v1"]}, "ad_task_id": {"type": "string"}, "depiction_type": en(DEPICTION_TYPES), "face_depiction_count_band": en(["0", *COUNT_BANDS]), "duplicate_faces_present": nullable(en(["yes", "no"])), "unique_face_count": nullable({"type": "integer"}), "people": {"type": "array", "items": ROUTE_PERSON, "maxItems": 9}, "groups": {"type": "array", "items": ROUTE_GROUP, "maxItems": 6}, "confidence": {"type": "number"}, "review_flags": {"type": "array"}}, "required": ["schema_version", "ad_task_id", "depiction_type", "face_depiction_count_band", "duplicate_faces_present", "unique_face_count", "people", "groups", "confidence", "review_flags"]}
PERSON_ATTR_SCHEMA = {"type": "object", "properties": {"schema_version": {"type": "string", "enum": ["qwen_first100_person_attributes_v1"]}, "person_task_id": {"type": "string"}, "eligible_face_visible": {"type": "boolean"}, **PERSON_ATTR_FIELDS, "confidence": {"type": "number"}, "review_flags": {"type": "array"}}, "required": ["schema_version", "person_task_id", "eligible_face_visible", *PERSON_ATTR_FIELDS, "confidence", "review_flags"]}
GROUP_ATTR_SCHEMA = {"type": "object", "properties": {"schema_version": {"type": "string", "enum": ["qwen_first100_group_attributes_v1"]}, "group_task_id": {"type": "string"}, **GROUP_ATTR_FIELDS, "confidence": {"type": "number"}, "review_flags": {"type": "array"}}, "required": ["schema_version", "group_task_id", *GROUP_ATTR_FIELDS, "confidence", "review_flags"]}
TILE_SCHEMA = {"type": "object", "properties": {"schema_version": {"type": "string", "enum": ["qwen_first100_tile_faces_v1"]}, "tile_task_id": {"type": "string"}, "faces": {"type": "array", "maxItems": 40, "items": {"type": "object", "properties": {"face_id": {"type": "string"}, "bbox_1000": BBOX, "confidence": {"type": "number"}}, "required": ["face_id", "bbox_1000", "confidence"]}}}, "required": ["schema_version", "tile_task_id", "faces"]}
RESOLVER_SCHEMA = {"type": "object", "properties": {"schema_version": {"type": "string", "enum": ["qwen_first100_face_guided_resolver_v1"]}, "ad_task_id": {"type": "string"}, "face_depiction_count_band": en(COUNT_BANDS), "accepted_face_ids": {"type": "array", "maxItems": 40, "items": {"type": "string"}}, "people": {"type": "array", "maxItems": 9, "items": {"type": "object", "properties": {"face_id": {"type": "string"}, "annotation_role": en(["individual", "outstanding_individual"]), "prominence_reason": nullable({"type": "string"})}, "required": ["face_id", "annotation_role", "prominence_reason"]}}, "groups": {"type": "array", "maxItems": 6, "items": ROUTE_GROUP}, "confidence": {"type": "number"}, "review_flags": {"type": "array"}}, "required": ["schema_version", "ad_task_id", "face_depiction_count_band", "accepted_face_ids", "people", "groups", "confidence", "review_flags"]}
COMPOSITION_SCHEMA = {"type": "object", "properties": {"schema_version": {"type": "string", "enum": ["qwen_first100_group_composition_inventory_v1"]}, "group_task_id": {"type": "string"}, "faces_truncated": {"type": "boolean"}, "faces": {"type": "array", "items": {"type": "object", "properties": {"bbox_1000": BBOX, "perceived_age": en(AGE_VALUES), "perceived_gender_presentation": en(GENDER_VALUES), "confidence": {"type": "number"}}, "required": ["bbox_1000", "perceived_age", "perceived_gender_presentation", "confidence"]}}}, "required": ["schema_version", "group_task_id", "faces_truncated", "faces"]}

COMMON = f"""Apply {PLAYBOOK_VERSION}. An eligible face has more than an ear/back of head visible and a locatable facial surface. Include photographs, illustrations, cartoons, statues, masks, personified objects, creatures, and logo faces. An advertisement with zero eligible faces is not qualifying and must not be returned. Select a substantive category when evidence exists; not_assessable is only for unusable evidence. Legibility 0 means no expression evidence can be interpreted at all; if smile versus non-smile can be judged, use legibility 1 or higher. Conditional null rules: if face_expression_legibility is 0_not_legible, gaze_target, gaze_target_person_unboxed, smile_present, and smile_intensity are null; mouth_covered=no whenever an unobstructed mouth region is visible, and mouth_covering is then null; smile_intensity is null unless smile_present is yes. For a group with all_0_not_legible, dominant_gaze, smile_prevalence, and dominant_smile_intensity are null; group intensity is null when prevalence is none or not_assessable. Coordinates are integer [x1,y1,x2,y2] on one 0..1000 grid for the supplied image."""
ROUTE = """For 1-9 eligible faces, return every face as an individual and no people areas. For 10+ faces, return people-area boxes for all ordinary remaining faces and only genuinely outstanding individuals that are much larger, clearer, more central, in a separate panel, or spatially separate. HARD CAP: in a 10+ route the people array contains at most 3 outstanding individuals; stop after the three strongest and never enumerate ordinary crowd members or candidate faces. A people area is an analytical aggregate: prefer one broad area whenever one unified expression tendency adequately describes the ordinary faces. Use at most 6 areas and split only analytically distinct ensembles/scenes; layout panels alone do not require separate areas."""


def schema_text(schema: dict[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=True, separators=(",", ":"))


def monolithic_prompt(image_id: str, metadata: dict[str, Any]) -> str:
    return f"Return only JSON. Annotate the complete page in one pass. image_id={image_id}; year={metadata.get('year')}. {COMMON} {ROUTE} Code all advertisement, individual, and people-area fields in the schema. Human or prior-model annotations are unavailable. Schema: {schema_text(MONOLITHIC_SCHEMA)}"


def page_router_prompt(image_id: str, metadata: dict[str, Any]) -> str:
    return f"Return only JSON. Perform structural routing for the complete page; do not code facial attributes. image_id={image_id}; year={metadata.get('year')}. Find only advertisements containing an eligible face, then apply the route independently per ad. {COMMON} {ROUTE} Human or prior-model annotations are unavailable. Schema: {schema_text(PAGE_ROUTER_SCHEMA)}"


def page_locator_prompt(image_id: str, metadata: dict[str, Any]) -> str:
    return f"Return only JSON. Locate only complete advertisement units on this page that contain at least one eligible face. Do not locate people or code attributes. image_id={image_id}; year={metadata.get('year')}. {COMMON} For one full-page ad use [0,0,1000,1000]. HARD CAP: return at most 6 distinct advertisements and never repeat or tile the same ad box. Human or prior-model annotations are unavailable. Schema: {schema_text(PAGE_LOCATOR_SCHEMA)}"


def ad_structure_prompt(task_id: str) -> str:
    return f"Return only JSON. This is one advertisement candidate crop. ad_task_id={task_id}. Verify that it contains an eligible face, then detect and route people only; no facial attributes. If it contains zero eligible faces, use face_depiction_count_band=0 with empty people and groups so the false-positive candidate can be discarded. {COMMON} {ROUTE} All coordinates are relative to this ad crop. Schema: {schema_text(AD_STRUCTURE_SCHEMA)}"


def person_prompt(task_id: str, ad_depiction_type: str) -> str:
    return f"Return only JSON. One composite shows an enlarged target face and wider ad context with the target outlined. Code only the target. person_task_id={task_id}; ad depiction type={ad_depiction_type}. {COMMON} If legibility is 0, gaze and smile fields are null. Smile intensity is non-null only for smile=yes. Mouth covering cause is non-null only for yes/partly. Schema: {schema_text(PERSON_ATTR_SCHEMA)}"


def group_prompt(task_id: str) -> str:
    return f"Return only JSON. This is one exact people-area crop. Treat it as one aggregate annotation unit and do not enumerate faces. group_task_id={task_id}. {COMMON} Judge the visible ensemble tendency. Age buckets are young=infant through young_adult, middle=middle_adult, older=older_adult. For age and feminine/masculine composition, only means all or nearly all, mostly means a strict majority over 50 percent, and mixed means no strict majority; do not choose mixed merely because a minority category is present. For legibility, all means all/nearly all, mostly means one level is a strict majority, and mixed means no level is a strict majority. If all_0 legibility, gaze and smile fields are null; intensity is null for none/not_assessable prevalence. Schema: {schema_text(GROUP_ATTR_SCHEMA)}"


def tile_prompt(task_id: str) -> str:
    return f"Return only JSON. Locate every eligible face in this one advertisement tile. tile_task_id={task_id}. {COMMON} Include partial faces at tile borders. Tight face/head boxes only; no attributes. Schema: {schema_text(TILE_SCHEMA)}"


def resolver_prompt(task_id: str, preliminary: dict[str, Any], proposals: list[dict[str, Any]]) -> str:
    return f"Return only JSON. The image is an ad crop with candidate faces outlined and labelled. ad_task_id={task_id}. Reject false candidates, then apply the group-versus-individual route. {COMMON} {ROUTE} Candidate proposals={json.dumps(proposals,separators=(',',':'))}. Preliminary independent structure={json.dumps(preliminary,separators=(',',':'))}. People must reference accepted candidate face_id values. Group boxes use the ad-crop 0..1000 grid. Schema: {schema_text(RESOLVER_SCHEMA)}"


def composition_prompt(task_id: str) -> str:
    return f"Return only JSON. This is one people-area crop. Enumerate eligible faces solely for an age/gender composition audit. group_task_id={task_id}. {COMMON} Return every face with a tight box, age, gender presentation, and confidence; no expression fields. Set faces_truncated only if output limits prevent completion. Schema: {schema_text(COMPOSITION_SCHEMA)}"

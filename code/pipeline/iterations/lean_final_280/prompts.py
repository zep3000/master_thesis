"""Schemas and prompts for the deliberately lean final-pipeline experiment."""

from __future__ import annotations

import json
import copy
from typing import Any


PLAYBOOK_VERSION = "annotation_playbook_v1.17"
PROMPT_VERSION = "lean_final_v1"

AD_CATEGORIES = [
    "banking_and_financial_services",
    "insurance",
    "automotive_and_transportation",
    "travel_hospitality_and_tourism",
    "industrial_manufacturing_and_engineering",
    "energy_utilities_and_natural_resources",
    "technology_telecommunications_and_electronics",
    "food_beverage_and_tobacco",
    "consumer_goods_and_personal_care",
    "retail_and_ecommerce",
    "healthcare_pharmaceuticals_and_biotechnology",
    "education_and_training",
    "employment_and_recruitment",
    "professional_and_business_services",
    "real_estate_and_construction",
    "media_entertainment_and_publishing",
    "government_public_sector_and_political",
    "nonprofit_public_interest_and_civic",
    "other",
]

DEPICTION_TYPES = [
    "photo_of_person", "illustration", "cartoon_or_caricature", "generic_human_figure",
    "photo_of_artwork_or_statue", "drawing_of_statue_monument_or_public_symbol",
    "mask_mannequin_doll_or_puppet", "personified_object", "nonhuman_creature_with_face",
    "schematic_icon_or_logo_face", "multiple_types_present",
]
COUNT_BANDS = [str(index) for index in range(1, 10)] + ["10_20", "20_plus"]
AGES = ["infant", "child", "adolescent", "young_adult", "middle_adult", "older_adult", "not_assessable"]
GENDERS = ["feminine", "masculine", "ambiguous_or_androgynous", "not_assessable"]
LEGIBILITY = ["0_not_legible", "1_low_legibility", "2_moderate_legibility", "3_high_legibility"]
ORIENTATION = ["beyond_profile", "profile", "three_quarter", "frontal", "other", "not_assessable"]
GAZE = ["viewer_camera", "another_person", "advertised_product", "other_object", "off_frame_or_scene_direction", "eyes_covered", "closed_eyes", "not_assessable"]
MOUTH = ["no", "yes", "partly", "not_assessable"]
MOUTH_CAUSES = ["hand", "beard", "other_body_part", "part_of_another_person", "object", "object_in_mouth", "text_or_graphic_overlay", "cropped_by_page_edge", "other", "not_assessable"]
SMILE = ["yes", "no", "not_assessable"]
INTENSITY = ["1_slight", "2_clear", "3_broad", "4_laughter_like"]
GROUP_TYPES = ["interacting_group", "posed_group", "audience", "background_population", "separate_portraits_or_composite", "other_group"]
GROUP_AGES = ["young_only", "middle_only", "older_only", "mostly_young", "mostly_middle", "mostly_older", "mixed", "not_assessable"]
GROUP_GENDERS = ["feminine_only", "masculine_only", "mostly_feminine", "mostly_masculine", "mixed", "ambiguous_or_androgynous_present", "not_assessable"]
GROUP_LEGIBILITY = ["all_0_not_legible", "mostly_0_not_legible", "all_1_low_legibility", "mostly_1_low_legibility", "all_2_moderate_legibility", "mostly_2_moderate_legibility", "all_3_high_legibility", "mostly_3_high_legibility", "mixed_legibility"]
GROUP_GAZE = ["toward_viewer_camera", "toward_each_other", "toward_object", "off_frame_or_scene_direction", "mixed", "not_assessable"]
GROUP_SMILE = ["none", "minority", "about_half", "majority", "all", "not_assessable"]
GROUP_INTENSITY = ["slight", "clear", "broad_or_laughter_like", "mixed"]


def compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


STRUCTURE_SCHEMA = {
    "schema_version": "lean_page_structure_v1",
    "image_id": "copy supplied ID",
    "qualifying_ad_count": "integer",
    "no_qualifying_ad_reason": ["no_ads_on_page", "ads_present_no_visible_faces", None],
    "advertisements": [{
        "advertisement_id": "ad_1 etc in reading order",
        "extent": ["full_page", "partial_page"],
        "bbox_1000": "[x1,y1,x2,y2] on full page",
        "ad_category": AD_CATEGORIES,
        "ad_category_confidence": "0..1",
        "brand_or_advertiser": "visible/inferable canonical name or null",
        "brand_confidence": "0..1; 0 when null",
        "depiction_type": DEPICTION_TYPES,
        "face_depiction_count_band": COUNT_BANDS,
        "duplicate_faces_present": ["yes", "no", None],
        "unique_face_count": "integer or null",
        "people": [{
            "person_id": "ad_1_person_1 etc",
            "annotation_role": ["individual", "outstanding_individual"],
            "face_bbox_1000": "tight full-page face/head box",
            "prominence_reason": ["dramatically_larger_or_clearer", "separate_panel_or_scene", "spatially_separate_from_people_area", None],
            "confidence": "0..1",
        }],
        "groups": [{
            "group_id": "ad_1_group_1 etc",
            "bbox_1000": "full-page people-area box",
            "confidence": "0..1",
        }],
        "confidence": "0..1",
        "review_flags": "array of short strings",
    }],
    "review_flags": "array of short strings",
}


def structure_prompt(image_id: str, year: Any, tiled: bool, include_business: bool = True) -> str:
    detail = (
        "IMAGE 1 is the complete page and defines the output coordinate grid. Images 2-5 are overlapping detail crops "
        "labelled with their full-page coordinate ranges; use them only to avoid missing small ads/faces, and always "
        "return full-page coordinates."
        if tiled else
        "The supplied image is the complete page and defines the output coordinate grid."
    )
    schema = copy.deepcopy(STRUCTURE_SCHEMA)
    if not include_business:
        for ad in schema["advertisements"]:
            for field in ["ad_category", "ad_category_confidence", "brand_or_advertiser", "brand_confidence"]:
                ad.pop(field, None)
    business = (
        "Ad category means the primary advertised offering/organization. Choose exactly one of the supplied broad labels. `brand_or_advertiser` is the canonical visible organization/product brand; use null rather than guessing. Category and brand inference must not change ad segmentation, face count, routing, or boxes."
        if include_business else
        "This is the category/brand ablation: do not infer or mention business category, brand, or advertiser."
    )
    return f"""Return one JSON object only. Apply {PLAYBOOK_VERSION}. image_id={image_id}; year={year}. Every enum field is one scalar JSON string chosen from the displayed options, never an array. {detail}
Find every distinct advertisement containing at least one eligible face. An eligible face shows more than only an ear or back of head and has a locatable facial surface; include photos, drawings, cartoons, statues, creatures and logo faces. Segment advertisement units first in reading order, then test every unit for eligible faces. Recheck lower-page/lower-right ads, inset portraits/posters, repeated portraits, and the largest face in every ad before finalizing. Do not return editorial matter or ads with zero eligible faces.
Count eligible faces, not bodies, silhouettes, or the general impression of a crowd. For 1-9 faces: return every face in people[], role=individual, and groups=[]. For 10+ faces: return one or more people-area boxes for all ordinary faces; people[] contains only genuinely outstanding faces. A front-row, central, or somewhat larger face is still ordinary. Outstanding requires dramatic visibility advantage or genuine spatial/panel separation; if adequately represented by the people area, do not duplicate it individually. Usually return zero outstanding people; hard cap three.
Boxes are integer [x1,y1,x2,y2] on the complete-page 0..1000 grid. Face boxes include head/visible hair but not shoulders. One broad people area is preferred unless ensembles/scenes are spatially distinct.
{business}
Schema: {compact(schema)}"""


PERSON_BASE = {
    "person_task_id": "copy supplied ID",
    "depiction_type": DEPICTION_TYPES + [None],
    "perceived_age": AGES,
    "perceived_gender_presentation": GENDERS,
    "face_orientation": ORIENTATION,
    "mouth_covered": MOUTH,
    "mouth_covering": MOUTH_CAUSES + [None],
    "smile_present": SMILE,
    "smile_intensity": INTENSITY + [None],
    "confidence": "0..1",
    "review_flags": "array of short strings",
}


def person_schema(style: str, gaze: bool) -> dict[str, Any]:
    schema = dict(PERSON_BASE)
    if style == "direct":
        schema["face_expression_legibility"] = LEGIBILITY
    else:
        schema.update({
            "at_least_low": "boolean",
            "at_least_moderate": "boolean",
            "at_least_high": "boolean",
        })
    if gaze:
        schema["gaze_target"] = GAZE
        schema["gaze_target_person_unboxed"] = "boolean or null"
    return schema


def person_prompt(task_id: str, style: str, gaze: bool) -> str:
    legibility_task = (
        "Choose the single nearest legibility category directly."
        if style == "direct" else
        "Answer nested legibility thresholds: high implies moderate implies low; the pipeline derives the category and repairs monotonic violations."
    )
    gaze_task = (
        "For gaze, frontal head pose is not viewer gaze. Use viewer_camera only with positive visible pupil/eye-axis evidence; otherwise choose the best scene target."
        if gaze else
        "Do not analyze or mention gaze; it is intentionally excluded to test whether removing it improves the other judgments."
    )
    return f"""Return one JSON object only. Apply {PLAYBOOK_VERSION}. person_task_id={task_id}. Every enum field is one scalar JSON string chosen from the displayed options, never an array. The composite contains: enlarged target face; medium local context; complete advertisement context with the target marked red. Code only the target.
Use all three views. Magnified halftone dots, print grain, or pixelation do not by themselves make an expression illegible. Judge whether facial configuration remains readable in the source/context view. Legibility is expression codability, not emotional intensity: 0=no expression evidence and even smile-versus-nonsmile is impossible; 1=one coarse cue is readable but no stable multi-region configuration; 2=mouth plus eye/brow or equivalent regions support a stable coarse judgment; 3=fine/subtle configuration is readable. {legibility_task}
Gender is visible presentation, not identity. ambiguous_or_androgynous is a positive substantive presentation, never an uncertainty response. If masculine or feminine is even slightly better supported, choose it; use not_assessable only when evidence is unusable.
Orientation is horizontal facial pose only. frontal=rough symmetry/nose centered; three_quarter=both eyes may remain visible but far side is foreshortened/nose displaced; profile=one side dominates; beyond_profile=less facial surface than profile. Ignore looking up/down; vertical tilt is never a category. {gaze_task}
Choose substantive age/gender/orientation/smile labels whenever evidence permits. Smile intensity is null unless smile_present=yes. Mouth covering is null unless mouth_covered is yes/partly. If derived legibility is 0, smile and any gaze fields are later nulled deterministically.
Schema: {compact(person_schema(style, gaze))}"""


def group_schema(gaze: bool) -> dict[str, Any]:
    schema = {
        "group_task_id": "copy supplied ID",
        "group_type": GROUP_TYPES,
        "age_composition": GROUP_AGES,
        "gender_presentation_composition": GROUP_GENDERS,
        "expression_legibility_distribution": GROUP_LEGIBILITY,
        "smile_prevalence": GROUP_SMILE,
        "dominant_smile_intensity": GROUP_INTENSITY + [None],
        "confidence": "0..1",
        "review_flags": "array",
    }
    if gaze:
        schema["dominant_gaze"] = GROUP_GAZE
    return schema


def group_prompt(task_id: str, gaze: bool) -> str:
    gaze_task = "Code dominant gaze from visible ensemble evidence." if gaze else "Do not analyze or mention gaze."
    return f"""Return one JSON object only. Apply {PLAYBOOK_VERSION}. group_task_id={task_id}. Every enum field is one scalar JSON string chosen from the displayed options, never an array. This is one exact people-area crop. Treat it as one aggregate analytical unit; never enumerate or invent outstanding individuals.
Use the same 0-3 expression-legibility anchors as individual faces. Halftone/grain alone is not zero if visible configuration remains interpretable. all means all/nearly all; mostly means one level or category is a strict majority; mixed means no strict majority. `ambiguous_or_androgynous_present` requires a positively visible presentation and is not uncertainty. {gaze_task} Intensity is null when smile prevalence is none/not_assessable. For all_0_not_legible, smile and any gaze fields are later nulled.
Schema: {compact(group_schema(gaze))}"""

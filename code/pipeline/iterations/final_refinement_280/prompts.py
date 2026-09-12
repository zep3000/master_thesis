"""Frozen schemas and prompts for the final 140+140 refinement experiment."""

from __future__ import annotations

import json
from typing import Any


PLAYBOOK_VERSION = "annotation_playbook_v1.17"
PROMPT_VERSION = "final_refinement_v1"

WARC_CATEGORIES = [
    "Alcoholic drinks", "Automotive", "Business & industrial",
    "Clothing & accessories", "Financial services", "Food",
    "Household & domestic", "Leisure & entertainment", "Media & publishing",
    "Non-profit, public sector & education", "Pharma & healthcare", "Politics",
    "Retail", "Soft drinks", "Technology & electronics", "Telecoms & utilities",
    "Tobacco", "Toiletries & cosmetics", "Transport & tourism",
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
    "schema_version": "final_refinement_structure_v1",
    "image_id": "copy supplied ID",
    "qualifying_ad_count": "integer",
    "no_qualifying_ad_reason": ["no_ads_on_page", "ads_present_no_visible_faces", None],
    "advertisements": [{
        "advertisement_id": "ad_1 etc in reading order",
        "extent": ["full_page", "partial_page"],
        "bbox_1000": "[x1,y1,x2,y2] on full page",
        "ad_category": WARC_CATEGORIES + [None],
        "brand_or_advertiser": "visible/inferable printed historical name or null",
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


BUSINESS_RULES = """Classify the primary advertised product, service, event, cause, or organization, not incidental imagery. Business & industrial includes B2B services, manufacturing, industrial materials, construction, and commercial equipment. Non-profit, public sector & education includes charities, government/public information, and education. Financial services includes insurance. Transport & tourism covers transport/travel services; industrial vehicle or equipment manufacture is Business & industrial except passenger automotive. Telecoms & utilities is separate from Technology & electronics. Media & publishing applies when a publication/media product is offered; public-interest messages follow the message. Retail means a general retailer rather than the product category of one item. For recruitment or pure corporate-image advertising with no direct offering category, use the advertiser's main sector. Prefer the advertiser/product name exactly as visibly printed; preserve historical names and do not modernize or expand an acronym unless printed. Use null rather than guessing. Category and brand inference must not change segmentation, face count, routing, or boxes."""


ORIGINAL_PARTITION = "One broad people area is preferred unless ensembles/scenes are spatially distinct."
REVISED_PARTITION = """The 10+ decision applies to the advertisement as a whole, not separately to each cluster. After an advertisement crosses 10+, cover ordinary faces with the minimum number of spatially coherent people-area boxes. Use one area for a continuous crowd, row, audience, or scene. Split only across clearly disconnected panels/scenes, a page gutter, substantial blank space, or spatially separate ensembles that cannot be represented by one coherent box without covering a large unrelated region. Cover every ordinary eligible face exactly once as far as visible; do not create semantic micro-groups or overlapping duplicate areas."""


def structure_prompt(image_id: str, year: Any, partition: str) -> str:
    partition_text = REVISED_PARTITION if partition == "revised" else ORIGINAL_PARTITION
    return f"""Return one JSON object only. Apply {PLAYBOOK_VERSION}. image_id={image_id}; year={year}. Every enum is one scalar JSON string, never an array. IMAGE 1 is the complete page and defines the 0..1000 coordinate grid. Remaining images are labelled overlapping detail crops; use them to avoid missing small ads and faces, but always return full-page coordinates.
Find every distinct advertisement containing at least one eligible face. An eligible face shows more than only an ear or back of head and has a locatable facial surface; include photos, drawings, cartoons, statues, creatures, and logo faces. Segment advertisement units first in reading order, then test every unit for eligible faces. Recheck lower-page/lower-right ads, inset portraits/posters, repeated portraits, the largest face in every ad, and faces near page gutters. Do not return editorial matter or ads with zero eligible faces.
Count eligible faces, not bodies, silhouettes, or the impression of a crowd. For 1-9 faces: return every face in people[], role=individual, and groups=[]. For 10+ faces: return people-area boxes for all ordinary faces; people[] contains only genuinely outstanding faces. A front-row, central, or somewhat larger face is ordinary. Outstanding requires dramatic visibility advantage or genuine spatial/panel separation; if represented adequately by the people area, do not duplicate it. Usually return zero outstanding people; hard cap three. {partition_text}
Boxes are integer [x1,y1,x2,y2] on the complete-page grid. Face boxes include head/visible hair but not shoulders.
{BUSINESS_RULES}
Schema: {compact(STRUCTURE_SCHEMA)}"""


PERSON_SCHEMA = {
    "person_task_id": "copy supplied ID",
    "depiction_type": DEPICTION_TYPES + [None],
    "perceived_age": AGES,
    "perceived_gender_presentation": GENDERS,
    "face_expression_legibility": LEGIBILITY,
    "face_orientation": ORIENTATION,
    "gaze_target": GAZE,
    "gaze_target_person_unboxed": "boolean or null",
    "mouth_covered": MOUTH,
    "mouth_covering": MOUTH_CAUSES + [None],
    "smile_present": SMILE,
    "smile_intensity": INTENSITY + [None],
    "confidence": "0..1",
    "review_flags": "array of short strings",
}


def person_prompt(task_id: str) -> str:
    return f"""Return one JSON object only. Apply {PLAYBOOK_VERSION}. person_task_id={task_id}. Every enum is one scalar JSON string, never an array. The composite contains an enlarged target face, medium local context, and complete advertisement context with the target marked red. Code only the target.
Use all views. Magnified halftone dots, print grain, or pixelation do not alone make an expression illegible. Legibility is expression codability, not intensity: 0=no visible expression evidence and even smile-versus-nonsmile is impossible; 1=one coarse cue is readable but no stable multi-region configuration; 2=mouth plus eye/brow or equivalent regions support a stable coarse judgment; 3=fine/subtle configuration is readable.
Gender is visible presentation, not identity. ambiguous_or_androgynous is a positive substantive presentation, never uncertainty. Orientation is horizontal pose only: frontal=rough symmetry; three_quarter=both eyes may remain but the far side is foreshortened; profile=one side dominates; beyond_profile=less facial surface than profile. Ignore vertical tilt. Frontal pose is not viewer gaze: viewer_camera requires positive visible eye-axis/pupil evidence. Choose substantive labels whenever evidence permits and not_assessable only when evidence is unusable. Always return gaze and smile independently; downstream code, not this response, tests conditional-null policies. Intensity is null unless smile=yes; mouth_covering is null unless mouth_covered=yes/partly.
Schema: {compact(PERSON_SCHEMA)}"""


GROUP_SCHEMA = {
    "group_task_id": "copy supplied ID",
    "group_type": GROUP_TYPES,
    "age_composition": GROUP_AGES,
    "gender_presentation_composition": GROUP_GENDERS,
    "expression_legibility_distribution": GROUP_LEGIBILITY,
    "dominant_gaze": GROUP_GAZE,
    "smile_prevalence": GROUP_SMILE,
    "dominant_smile_intensity": GROUP_INTENSITY + [None],
    "confidence": "0..1",
    "review_flags": "array of short strings",
}


def group_prompt(task_id: str, context: bool) -> str:
    context_text = (
        "The composite also includes the complete advertisement with this people area marked magenta; use it for group type and gaze-target context."
        if context else "No surrounding advertisement panel is supplied; use the complete people-area panel for spatial context."
    )
    return f"""Return one JSON object only. Apply {PLAYBOOK_VERSION}. group_task_id={task_id}. Every enum is one scalar JSON string, never an array. This composite represents one people-area analytical unit. It contains the complete people area and four labelled overlapping detail tiles. {context_text}
Systematically inspect the complete area and every tile before deciding. Tiles overlap and repeat some faces; never count repeated appearances as different people. Internally consider visible faces anonymously, but do not enumerate, box, list, or output individuals or percentages. Return exactly one aggregate record with the existing categorical granularity.
For age/gender, only means all/nearly all; mostly means a strict majority; mixed means no strict majority. Legibility uses the individual 0-3 anchors; all means all/nearly all, mostly means strict majority, mixed means none has a strict majority. Halftone alone is not zero. Do not infer viewer gaze from frontal head pose. For smile prevalence, inspect minority smiles rather than defaulting to none. Intensity is null only when prevalence is none/not_assessable. Always return gaze and smile independently; downstream code tests conditional-null policies.
Schema: {compact(GROUP_SCHEMA)}"""


FER_SCHEMA = {
    "person_task_id": "copy supplied ID",
    "at_least_low": "boolean",
    "at_least_moderate": "boolean",
    "at_least_high": "boolean",
    "review_flags": "array of short strings",
}


def fer_threshold_prompt(task_id: str) -> str:
    return f"""Return one JSON object only. Apply {PLAYBOOK_VERSION}. person_task_id={task_id}. This is a legibility-only visibility task. The composite contains the enlarged target, local source context, and marked advertisement context. Code only the target.
Answer three nested visual thresholds independently, then ensure high implies moderate implies low.
- at_least_low: at least one coarse expression-relevant configuration is visibly resolved. A semantic guess such as neutral or nonsmiling is not evidence.
- at_least_moderate: at least two expression-relevant regions, such as mouth plus eyes/brows, visibly support a stable coarse facial-configuration judgment.
- at_least_high: fine or subtle facial configuration is genuinely resolved at source quality, not merely enlarged interpolation.
Judge visible facial configuration only. Do not infer legibility from guessed emotion, neutrality, smile, gaze, pose, age, gender, depiction type, or contextual expectations. Halftone/grain alone does not imply zero, but magnification does not restore absent source detail. Schema: {compact(FER_SCHEMA)}"""


FER_MODERATE_AUDIT_SCHEMA = {
    "person_task_id": "copy supplied ID",
    "face_expression_legibility": LEGIBILITY,
    "decision": ["keep_moderate", "override_down", "override_up"],
    "visible_regions": ["short list of visibly resolved expression-relevant regions"],
    "review_flags": ["short strings"],
}


def fer_moderate_audit_prompt(task_id: str) -> str:
    return f"""Return one JSON object only. Apply {PLAYBOOK_VERSION}. person_task_id={task_id}. A preceding complete annotation call provisionally labeled this target 2_moderate_legibility. Audit that boundary only; do not infer any other person attribute.
Use source visibility, not semantic guesses. 0 means no expression-relevant facial configuration is visibly resolved and even smile-versus-nonsmile cannot be coded. 1 means exactly one coarse expression-relevant cue or region is resolved but no stable multi-region configuration. 2 means mouth plus eyes/brows, or equivalent multiple regions, visibly support a stable coarse configuration. 3 requires fine/subtle configuration genuinely resolved at source quality. Halftone dots, grain, illustration, neutral appearance, weak emotion, non-smiling, gaze, or pose do not by themselves justify a downgrade.
Keep moderate unless the images provide positive visual evidence for another anchor. Set decision=keep_moderate for 2, override_down for 0/1, or override_up for 3. visible_regions is descriptive audit evidence only and does not change the taxonomy. Schema: {compact(FER_MODERATE_AUDIT_SCHEMA)}"""

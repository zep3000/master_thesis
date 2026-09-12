"""Frozen prompts introduced by the canonical candidate package."""

from __future__ import annotations

import json


WARC_CATEGORIES = [
    "Alcoholic drinks", "Automotive", "Business & industrial", "Clothing & accessories",
    "Financial services", "Food", "Household & domestic", "Leisure & entertainment",
    "Media & publishing", "Non-profit, public sector & education", "Pharma & healthcare",
    "Politics", "Retail", "Soft drinks", "Technology & electronics",
    "Telecoms & utilities", "Tobacco", "Toiletries & cosmetics", "Transport & tourism",
]


def business_prompt(task_key: str) -> str:
    schema = {"task_key": task_key, "ad_category": WARC_CATEGORIES, "brand_or_advertiser": "string or null"}
    return f"""Classify only the already-frozen advertisement shown in the image. Do not locate or recount ads or faces and do not output any other annotation.

Return one JSON object matching this schema (the category value is one member of the displayed list, not the list itself):
{json.dumps(schema, ensure_ascii=False)}

Classify the primary advertised product, service, event, cause, or organization, not incidental imagery. Business & industrial includes B2B services, manufacturing, industrial materials, construction, and commercial equipment. Non-profit, public sector & education includes charities, government/public information, and education. Financial services includes insurance. Transport & tourism covers transport/travel services; industrial vehicle or equipment manufacture is Business & industrial except passenger automotive. Telecoms & utilities is separate from Technology & electronics. Media & publishing applies when a publication/media product is offered; a public-interest message sponsored by a media organization follows the message. Retail means a general retailer rather than the specific product category of an item it sells. For recruitment or a pure corporate-image ad with no direct offering category, use the advertiser's main sector.

For brand_or_advertiser, a product brand, company, sponsor, public body, or other visible advertiser is acceptable. Prefer the name exactly as visibly printed. Preserve historical names; do not modernize a company name or expand an acronym unless printed. Use null rather than guessing.

No confidence or evidence fields."""


def duplicate_prompt(task_key: str, person_ids: list[str], conservative: bool = False) -> str:
    schema = {
        "task_key": task_key,
        "candidate_clusters": [{"person_ids": ["two or more IDs from the supplied list"], "candidate_strength": "high or medium"}],
    }
    conservative_rule = """
This is a literal repeated-depiction check, not face recognition. Propose a link only when the two boxes appear to reproduce the same underlying printed source crop/artwork: the expression, pose/angle, feature geometry, hair/occlusion pattern, and local image pattern must agree. The same represented person in a different photograph, film frame, pose, angle, expression, or moment is NOT a duplicate. Adjacent people visible at the edge of a target tile are context and must not be compared as that tile's target. When in doubt, return no candidate.
""" if conservative else ""
    return f"""This is an isolated duplicate-candidate task over frozen face boxes and IDs.

The only permitted decision is whether two or more supplied boxes repeat the exact same face depiction within this advertisement. Exact duplicates include a mirror, collage repetition, repeated portrait, or repeated product shot showing the same represented identity with the same face/expression. For nonhuman or schematic depictions, identity means the same represented character, object, or symbol. Do not link merely similar-looking faces, or the same person shown with a different expression, pose, angle, or moment.

The boxes, IDs, and all main annotations are frozen. Do not assess, correct, or output boxes, groups, counts, age, gender, expression, orientation, gaze, mouth, smile, advertisement category, or brand. Every supplied face remains a separately annotated depiction regardless of your candidate decision. Your answer cannot change the main annotations.
{conservative_rule}

Supplied IDs: {json.dumps(person_ids)}

Return JSON matching:
{json.dumps(schema)}

Use high only for visually compelling exact repetition. Use medium for a plausible exact repetition worth human review. If no pair is a reasonable exact-duplicate candidate, return an empty candidate_clusters array."""

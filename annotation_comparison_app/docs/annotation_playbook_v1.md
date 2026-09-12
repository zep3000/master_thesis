# Advertisement Face Annotation Playbook

Version: 1.12

This playbook defines the human annotation procedure and the behavior expected from an automated or LLM annotator. The application presents one decision at a time and stores bounding boxes as normalized page coordinates in `[x1, y1, x2, y2]` format.

## 1. Scope and eligibility

The unit screened at the beginning is the page. A qualifying advertisement contains at least one eligible face depiction.

A face depiction is eligible when more than an ear or the back of a head is visible and the face can be located with a bounding box. Include photographs, illustrations, artwork or statues, masks or mannequins, personified objects, nonhuman creatures, and schematic or logo faces. Exclude ear-only cases, backs of heads without facial surface, and marks too small or degraded to confirm that a face is present.

This pass does not annotate brands or product categories. It also does not provide a denominator for estimating face prevalence among all advertisements unless the input manifest itself represents all relevant pages or advertisements.

## 2. Mandatory decision rule

There is no `unclear` or `ambiguous` answer.

When visual evidence is available, choose the single best-fitting substantive category. If two categories appear plausible, choose the one that is slightly better supported. A difficult boundary is not a reason to avoid the decision.

Use `not_assessable` only when the visual evidence required for that field is unavailable or unusable because of factors such as crop, size, occlusion, orientation, blur, or poor reproduction. Do not use it merely because the choice is difficult.

Use `other` only where it is explicitly offered and the observation is visible and interpretable but falls outside the supplied taxonomy. It is not an uncertainty response.

Use an urgent comment for an exceptional problem that may matter during review. The comment does not replace a required answer.

### LLM annotation instruction

Use the following rule in one-shot or two-shot LLM prompts:

> Select exactly one best-fitting substantive category whenever the relevant visual evidence is available. When a case lies near a category boundary, choose the category with slightly stronger visual support. Do not return an uncertainty label. Use `not_assessable` only when missing, occluded, too-small, cropped, or degraded evidence prevents a meaningful best estimate. Use `other` only when the visible observation is clear but absent from the provided taxonomy.

For `face_expression_legibility`, judge the overall ability to code the visible expression. Equivalent scores may arise from different causes, including limited facial detail, covering or occlusion, face orientation, small face size, blur, low contrast, or poor reproduction quality.

If individual `face_expression_legibility` is `0_not_legible`, do not infer gaze, smile presence, or smile intensity. Leave those fields null and continue only with orientation and mouth-covering fields. If group `expression_legibility_distribution` is `all_0_not_legible`, do not infer group gaze, smile prevalence, or smile intensity.

## 3. Page setup

### P1. Qualifying advertisement count

Count advertisements containing at least one eligible face depiction.

Input: one zero-case reason, `1` through `5` as direct buttons, or an exact integer from `6` through `99`.

- `no_ads_on_page` finishes the page as ineligible and records that no advertisement is present.
- `ads_present_no_visible_faces` finishes the page as ineligible and records that advertisements are present, but none contains an eligible visible face depiction.
- `1` continues to P2.
- More than one continues to drawing all advertisement boxes.

For both zero-case reasons, store `page.qualifying_ad_count` as `"0"` and store the reason in `page.no_qualifying_ad_reason`. Legacy annotations with `page.qualifying_ad_count` equal to `"0"` but no `page.no_qualifying_ad_reason` are incomplete for the current flow and must be reopened at P1.

Count advertisements, not faces. When an advertisement boundary is difficult, draw and count the best defensible advertising unit and add an urgent comment only if the boundary could materially affect analysis.

### P2. Single advertisement extent

Input: `full_page` or `partial_page`.

- `full_page` creates an automatic page-sized advertisement box.
- `partial_page` requires a manually drawn advertisement box.

## 4. Advertisement preparation

### A1. Advertisement bounding boxes

For a partial-page advertisement, draw the complete advertising unit, including its associated image and copy. Exclude neighboring advertisements and editorial material. For multiple qualifying advertisements, draw all advertisement boxes before continuing.

### A2. Advertisement depiction type

Choose one value for each advertisement when the same depiction type applies to every individually coded face in that advertisement. Choose `multiple_types_present` only when two or more types occur within the current advertisement; the app will then ask for the depiction type of every canonical person in that advertisement.

Values:

- `photo_of_person`: direct photograph of a living or historically photographed person
- `naturalistic_illustration`: realistic drawing, painting, engraving, or rendered illustration of a person
- `stylized_illustration`: recognizably human illustration with deliberate simplification or stylization
- `cartoon_or_caricature`: cartoon convention or conspicuous comic exaggeration
- `generic_human_figure`: anonymous or generic human figure rather than a represented individual
- `photo_of_artwork_or_statue`: photograph of physical artwork, sculpture, or statue containing a face
- `drawing_of_statue_monument_or_public_symbol`: drawn depiction of a statue, monument, or established public symbol
- `mask_mannequin_doll_or_puppet`: face belonging to one of these physical or depicted objects
- `personified_object`: nonliving object given a face or human facial organization
- `nonhuman_creature_with_face`: animal, fictional creature, or other nonhuman being with a face
- `schematic_icon_or_logo_face`: highly schematic face, icon, emblem, or logo
- `multiple_types_present`: more than one of the preceding types occurs within this advertisement

Classify the form of the depicted face, not the printing process. For example, a photograph of a statue is `photo_of_artwork_or_statue`, not `photo_of_person`.

### A3. Draw faces or choose a crowd band

Count every eligible face depiction, including mirrors, repeated portraits, collage repetitions, and repeated product shots of the same person.

For one through nine faces, keep **only individuals** selected and begin drawing immediately. Draw every eligible face box and choose **Done** only when every eligible face depiction in the current advertisement has a box. The app derives and stores the exact count from the number of boxes; no separate count selection is made.

For ten or more faces, do not draw all individual boxes. Choose `10_20` or `20_plus` to enter the crowd route. This route can be changed back to **only individuals** before continuing. If a crowd band is selected after provisional boxes have been drawn on this screen, those provisional individual boxes are removed.

Draw the smallest box that covers all visible face and head features needed for coding. Include visible hair, ears, forehead, cheeks, chin, beard or moustache, and face-worn items such as glasses. Do not include neck, shoulders, captions, labels, or empty background unless they visibly cover or cut across the face. The count concerns depictions, not unique identities, so repeated depictions receive separate boxes. Boxes are stored relative to the full page even while the interface displays only the current advertisement crop. Duplicate identities are resolved later within the current advertisement.

### C1. Outstanding individuals in a crowd

For an advertisement with `10_20` or `20_plus` faces, choose `yes` when one or more faces are visually prominent enough for detailed individual coding. Otherwise choose `no`.

Outstanding individuals are central, large, singled out, or otherwise analytically important. When present, draw their individual face boxes. Then draw group boxes for the remaining crowd.

### C2. Group boxes

Draw one box for each visually distinct remaining group. A group box should include all faces in that group. Separate groups only when faces form distinct clusters, panels, scenes, or portrait sets; do not split one crowd merely because it has rows.

## 5. Duplicate identity resolution

Duplicate resolution occurs within each advertisement after that advertisement's individual face boxes have been drawn. When a page contains multiple advertisements, finish the first advertisement's face drawing, duplicate resolution, individual coding, and group coding before moving to the second advertisement. Duplicate resolution applies only to individually boxed faces, not to unboxed members represented by a group box.

### D0. Are any boxed faces duplicates?

Choose `yes` when two or more boxes repeat the same face identity, for example through a mirror, collage repetition, repeated portrait, or repeated product shot. For nonhuman or schematic depictions, identity refers to the same represented character, object, or symbol. Do not merge merely similar-looking faces.

- `no`: every box in the current advertisement becomes its own canonical person and duplicate resolution ends for that advertisement.
- `yes`: continue to D1.

### D1. Number of unique face identities

Enter the number of unique face identities represented by the current advertisement's individual face boxes. Because D0 is `yes`, this number must be at least one and smaller than the number of boxes.

### D2-D3. Build identity groups

For each unique face identity:

1. Select the main face. Prefer the clearest and most analytically useful depiction.
2. Select every other box depicting that same identity.
3. Continue until every individual face box belongs to exactly one identity group.

An identity group may contain only its main face when the duplicated face belongs to another identity group. At least one completed identity group must contain a duplicate because D0 was answered `yes`.

Every box remains in the output. The main record stores `duplicate_person_ids`; each duplicate record stores `duplicate_of_person_id`. Detailed person coding is performed only for main records.

## 6. Individual person coding

The app iterates over canonical main faces only. The active face is visually highlighted. Age and gender presentation are coded first, followed immediately by facial-expression legibility.

### I0. Person depiction type

Ask this only when A2 is `multiple_types_present` for the current advertisement. Use the A2 definitions and choose the type of the selected main depiction. `multiple_types_present` is not available at person level.

### I1. Perceived age band

Values: `infant`, `child`, `adolescent`, `young_adult`, `middle_adult`, `older_adult`, `not_assessable`.

Code apparent age, not known chronological age. At a boundary, choose the more likely adjacent band. Use `not_assessable` only when a meaningful visual estimate cannot be made.

### I2. Perceived gender presentation

Values: `feminine`, `masculine`, `ambiguous_or_androgynous`, `not_assessable`.

Code visible presentation, not identity. `ambiguous_or_androgynous` is a substantive visible presentation, not an uncertainty response.

### I4. Facial-expression legibility

This four-point ordinal scale measures how well the visible face supports expression coding. It does not measure emotional intensity.

- `0_not_legible`: the face can be located, but no facial expression can be coded reliably
- `1_low_legibility`: only coarse expression information is available; important facial detail is difficult to distinguish
- `2_moderate_legibility`: the expression is generally readable, but some relevant detail is limited
- `3_high_legibility`: facial configuration and expression-relevant detail are clear and readily codeable

Consider face size, pose, occlusion, contrast, focus, reproduction quality, and visible facial detail. Choose the nearest level rather than avoiding a boundary decision.

The scale records the overall legibility of the expression, not why legibility is limited. The same low or moderate score can arise for different reasons: the depiction may contain few facial details; part of the face may be covered or otherwise occluded; profile or tilted orientation may hide relevant features; or the face may be too small, blurred, low-contrast, degraded, or poorly reproduced. Several limitations may also combine. Judge their combined effect on how confidently the visible facial expression can be coded rather than counting limitations or assigning a fixed penalty for any one cause. A face with an unusual orientation or partial covering can still receive high legibility when its expression-relevant features remain clear.

When the value is `0_not_legible`, skip gaze and smile-presence/intensity coding. Continue to face orientation and mouth-covering coding because these may remain observable even when expression is not legible.

### I3. Face orientation

Values: `beyond_profile`, `profile`, `three_quarter`, `frontal`, `tilted_down`, `tilted_up`, `not_assessable`.

The interface displays `beyond_profile` as **less than profile**. Use `tilted_down` or `tilted_up` when vertical head angle is the most distinctive orientation feature. Select the closest orientation when between categories.

### I5. Gaze

Values: `viewer_camera`, `another_person`, `advertised_product`, `other_object`, `off_frame_or_scene_direction`, `eyes_covered`, `closed_eyes`, `not_assessable`.

Code visible gaze direction, not inferred attention or narrative intention. Use `eyes_covered` when the eye region is visibly covered or blocked by an object, body part, text/graphic overlay, shadow, or another depiction element. Use `closed_eyes` only when the eyelids are visibly closed; use `not_assessable` when gaze cannot be judged for another reason. When `another_person` is selected, click that person's box. If the target person has no individual box, choose **Person without bounding box**; this selection immediately continues to the next question.

### I6. Mouth covering

Values: `no`, `yes`, `partly`, `not_assessable`.

Choose whether something visibly covers the mouth. Crop, face orientation, and poor image quality are not coverings; use `not_assessable` only when those conditions prevent a meaningful decision.

### I7. Mouth-covering cause

Asked only for `yes` or `partly`.

Values: `hand`, `beard`, `other_body_part`, `part_of_another_person`, `object`, `object_in_mouth`, `text_or_graphic_overlay`, `other`, `not_assessable`.

Record what visibly covers the mouth, not whether the covering is intentional. `other` requires a short concrete description.

### I8. Smile presence

Values: `yes`, `no`, `not_assessable`.

Code visible facial configuration only and do not infer felt emotion. When evidence is available but near a boundary, choose `yes` or `no`.

### I9. Smile intensity

Asked only when smile presence is `yes`.

- `1_slight`
- `2_clear`
- `3_broad`
- `4_laughter_like`

Choose the closest level. The scale describes visible configuration and does not claim emotional authenticity.

## 7. Group coding

For every group box, code:

- Group type: `interacting_group`, `posed_group`, `audience_or_crowd`, `background_population`, `separate_portraits_or_composite`
- Age composition: `young_only`, `middle_only`, `older_only`, `mostly_young`, `mostly_middle`, `mostly_older`, `mixed`, `not_assessable`
- Gender-presentation composition: `feminine_only`, `masculine_only`, `mostly_feminine`, `mostly_masculine`, `mixed`, `ambiguous_or_androgynous_present`, `not_assessable`
- Expression legibility distribution: `all_0_not_legible`, `mostly_0_not_legible`, `all_1_low_legibility`, `mostly_1_low_legibility`, `all_2_moderate_legibility`, `mostly_2_moderate_legibility`, `all_3_high_legibility`, `mostly_3_high_legibility`, `mixed_legibility`
- Dominant gaze: `toward_viewer_camera`, `toward_each_other`, `toward_object`, `off_frame_or_scene_direction`, `mixed`, `not_assessable`
- Smile prevalence: `none`, `minority`, `about_half`, `majority`, `all`, `not_assessable`
- Dominant smile intensity: `slight`, `clear`, `broad_or_laughter_like`, `mixed`

For group expression legibility, use the same four underlying levels as individual expression legibility, but code their distribution across the boxed group. Use an `all_*` value when all or nearly all grouped faces fall at the same level. Use a `mostly_*` value when one level clearly predominates but exceptions are visible. Use `mixed_legibility` when no single level clearly predominates.

As with individual legibility, score the overall expression-coding result rather than the cause. Low, moderate, or high group legibility may come from small faces, few facial details, covering or occlusion, profile or tilted orientation, blur, low contrast, or degraded reproduction quality.

Do not ask for group gaze, smile prevalence, or smile intensity when expression legibility is `all_0_not_legible`. Do not ask for group smile intensity when smile prevalence is `none` or `not_assessable`.

## 8. Urgent comments and completion

The urgent-comment action is available throughout annotation. Use it only for a genuinely difficult or important issue. The app stores the current page, advertisement, group or person, interrupted step, and timestamp with the comment.

A page is complete only after all required boxes, duplicate assignments, canonical-person fields, and group fields have been entered. Creation, update, completion, and per-screen timing information are retained in the structured JSON output.

## 9. Variables excluded from version 1.5

- Brand and product category
- Felt emotion
- Genuine or fake smile
- Duchenne smile
- Smile social function
- Intentional covering
- Exact head-pose degrees
- Race or ethnicity
- Actual gender identity
- Exact chronological age
- Per-duplicate detailed person coding

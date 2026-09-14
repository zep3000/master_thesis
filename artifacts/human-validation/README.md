# Human annotations and human--LLM validation results

This directory publishes the reviewed human-annotation record and numerical
evidence used for the main validation in Chapter 5. Three human coders and the
final LLM pipeline annotated the same 300 sampled archive pages. The tables
below retain their final denominators and notes.

The pseudonymized row-level exports are under `raw/`, with file-level checksums
and transformation counts in `raw/manifest.json`. They include the 900 records
from the three-coder validation, 201 brand/industry review records, and 1,599
face-box rows from the full-issue scans. Source images and visual error atlases
remain excluded because they reproduce licensed magazine imagery.

Direct assignee names are replaced by A, B, and C. Assignment/session codes,
comments, notes, and absolute machine paths are removed. Annotation-set IDs,
pseudonymous session IDs, timestamps, page identifiers, geometry, workflow
telemetry, and substantive judgments remain. The result is pseudonymized, not
anonymous: an external mapping could reconnect a session ID to a coder.

## Scope and definitions

- Advertisement and regular-individual matching use label-blind optimal
  one-to-one assignment at IoU 0.20 unless a table says otherwise.
- A human consensus individual requires spatial support from at least two of
  the three coders. A field majority requires at least two matching substantive
  labels among eligible coders.
- `H-H mean` is the unweighted mean over the three human pairings. Pooled
  human--LLM values sum comparison counts before computing the metric.
- Exact agreement requires the same category. Lenient agreement additionally
  accepts a one-step difference for the predeclared ordinal fields.
- End-to-end measures count spatially unmatched consensus entities and missing
  LLM values as failures. Conditional measures first restrict to accepted
  spatial matches.

## Annotation volume

| Coder | Advertisements | Regular individuals | Outstanding individuals | People areas |
|:---|---:|---:|---:|---:|
| A | 233 | 552 | 3 | 18 |
| B | 233 | 537 | 129 | 33 |
| C | 251 | 574 | 13 | 40 |
| LLM | 254 | 460 | 13 | 35 |

Outstanding individuals were selectively boxed on the 10+ people-area route
and are excluded from the regular-individual performance results.

## End-to-end regular-individual detection

| Pair/view | Ref. | Cand. | Matches | Prec. | Recall | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| A-B | 552 | 537 | 468 | 0.872 | 0.848 | 0.860 |
| A-C | 552 | 574 | 506 | 0.882 | 0.917 | 0.899 |
| B-C | 537 | 574 | 478 | 0.833 | 0.890 | 0.860 |
| Humans-LLM (pooled) | 1,663 | 1,380 | 1,163 | 0.843 | 0.699 | 0.764 |
| Human consensus-LLM | 540 | 460 | 396 | 0.861 | 0.733 | 0.792 |
| Any human-LLM | 667 | 460 | 412 | 0.896 | 0.618 | 0.731 |

These figures cover all 300 validation pages and include people-area
advertisements and advertisement-level matching failures. The any-human
reference is deduplicated across A, B, and C before assignment.

| IoU threshold | Regular individuals unmatched to every human | People areas unmatched to every human | Combined |
|:---|---:|---:|---:|
| 0.50 | 88 | 12 | 100 |
| 0.20 | 47 | 9 | 56 |
| 0.10 | 47 | 7 | 54 |

## Unmatched regular individuals

| Direction | Mutually exclusive reason | Instances | Share |
|:---|:---|---:|---:|
| False negative | Advertisement mismatch (person matches page-wide) | 6 | 2.4% |
|  | Overlapping LLM individual lost in one-to-one assignment | 0 | 0.0% |
|  | Covered by an LLM people area | 116 | 45.5% |
|  | No overlapping LLM individual or people area | 133 | 52.2% |
|  | Total | 255 | 100.0% |
| False positive | Advertisement mismatch (person matches page-wide) | 6 | 12.5% |
|  | Overlapping human individual; duplicate/assignment conflict | 5 | 10.4% |
|  | Covered by a human people area | 11 | 22.9% |
|  | No overlapping human individual or people area | 26 | 54.2% |
|  | Total | 48 | 100.0% |

People-area coverage is assigned only after advertisement mismatch and
one-to-one assignment conflicts have been accounted for.

## People-area evaluation

| People-area outcome | Human-human exact | Human-human lenient | Human-LLM exact | Human-LLM lenient |
|:---|---:|---:|---:|---:|
| Spatial detection (labels ignored) | 0.533 | n/a | 0.592 | n/a |
| Group type | 0.321 (0.601) | n/a | 0.398 (0.672) | n/a |
| Age composition | 0.211 (0.378) | 0.220 (0.395) | 0.306 (0.517) | 0.306 (0.517) |
| Gender composition | 0.414 (0.770) | 0.432 (0.804) | 0.490 (0.828) | 0.490 (0.828) |
| Expression legibility | 0.105 (0.178) | 0.382 (0.684) | 0.214 (0.362) | 0.500 (0.845) |
| Dominant gaze | 0.353 (0.609) | n/a | 0.358 (0.554) | n/a |
| Smile prevalence | 0.349 (0.594) | 0.443 (0.761) | 0.358 (0.554) | 0.520 (0.804) |
| Dominant smile intensity | 0.290 (0.396) | 0.622 (0.901) | 0.306 (0.455) | 0.571 (0.848) |

Values outside parentheses are end-to-end F1; parenthetical values are
matched-only micro-F1. Advertisement and people-area matching both use IoU
0.05 for this table. Human-human values are means over coder pairs;
human--LLM values pool counts against all three humans.

Lenient people-area matching ignores quantity qualifiers for age and gender
composition while requiring the same category family; it permits adjacent
ordered levels for legibility, smile prevalence, and dominant smile intensity.
Group type and dominant gaze are nominal and have no lenient variant.

## Conditional individual cohort

| Cohort step | Count |
|:---|---:|
| Matched advertisements before people-area exclusion | 231 |
| Excluded: LLM reported a people area | 28 |
| Excluded: at least one human reported a people area | 4 |
| Final people-area-free matched advertisements | 199 |
| Resulting people supported by at least 2 of 3 humans | 466 |

The exclusions are mutually exclusive. Within the final cohort, consensus--LLM
regular-individual detection has 466 reference entities, 415 candidates, 394
matches, 0.949 precision, 0.845 recall, and 0.894 F1.

## Human-majority availability

| Field | Eligible cases | Majority cases | Majority share | Unanimous (3/3) |
|:---|---:|---:|---:|---:|
| Depiction type | 466 | 457 | 98.1% | 352/406 (86.7%) |
| Perceived age | 457 | 410 | 89.7% | 138/400 (34.5%) |
| Gender presentation | 457 | 447 | 97.8% | 371/400 (92.8%) |
| Expression legibility | 457 | 352 | 77.0% | 37/400 (9.2%) |
| Face orientation | 457 | 430 | 94.1% | 234/400 (58.5%) |
| Gaze direction/target | 415 | 366 | 88.2% | 208/363 (57.3%) |
| Mouth covered | 457 | 442 | 96.7% | 366/400 (91.5%) |
| Mouth-covering cause | 17 | 14 | 82.4% | 3/6 (50.0%) |
| Smile presence | 415 | 395 | 95.2% | 264/363 (72.7%) |
| Smile intensity | 184 | 113 | 61.4% | 20/124 (16.1%) |

## Field agreement

| Field | Majority N | H-H mean | Majority-LLM | Any-human-LLM | Conditional e2e recall |
|:---|---:|---:|---:|---:|---:|
| Depiction type | 457 | 90.4% | 92.5% | 96.4% | 78.6% |
| Perceived age | 410 | 54.5% / 93.7% | 60.5% / 93.6% | 82.4% / 96.9% | 52.2% / 80.5% |
| Gender presentation | 447 | 93.6% | 90.6% | 90.6% | 78.1% |
| Expression legibility | 352 | 33.3% / 82.8% | 42.0% / 93.9% | 73.0% / 97.7% | 34.9% / 80.7% |
| Face orientation | 430 | 71.5% / 98.5% | 73.8% / 98.5% | 83.5% / 99.2% | 64.2% / 84.7% |
| Gaze target | 366 | 68.3% | 54.1% | 55.4% | 49.2% |
| Mouth covered | 442 | 93.7% | 90.0% | 93.1% | 77.4% |
| Mouth-covering cause | 14 | 68.9% | 21.4% | 23.5% | 21.4% |
| Smile presence | 395 | 80.8% | 81.6% | 86.7% | 73.2% |
| Smile intensity | 113 | 32.7% / 76.5% | 43.4% / 87.1% | 70.2% / 90.1% | 40.7% / 81.0% |

Slashes show exact/lenient agreement. Majority and any-human values require a
matched person; conditional end-to-end recall also treats unmatched consensus
people and missing LLM values as failures.

Leniency is limited to adjacent levels for perceived age, expression
legibility, face orientation, and smile intensity. All nominal and binary
fields require exact equality, and `not_assessable` always requires an exact
match.

## Coder category distributions

These distributions use the 400 people identified by all three human coders;
the LLM column uses the 355 members of that set that it also identified. Smile
intensity denominators differ because it is recorded only for faces each coder
classified as smiling.

| Perceived age | Human A | Human B | Human C | LLM |
|:---|---:|---:|---:|---:|
| Infant | 0.0% | 0.0% | 0.2% | 0.0% |
| Child | 4.0% | 3.8% | 3.5% | 3.9% |
| Adolescent | 1.5% | 0.2% | 2.0% | 0.3% |
| Young adult | 17.2% | 7.5% | 36.8% | 44.2% |
| Middle adult | 65.0% | 48.0% | 38.8% | 45.6% |
| Older adult | 9.0% | 34.2% | 14.8% | 5.6% |
| Not assessable | 3.2% | 6.2% | 4.0% | 0.3% |

| Expression legibility | Human A | Human B | Human C | LLM |
|:---|---:|---:|---:|---:|
| Not legible | 4.0% | 4.5% | 7.2% | 8.2% |
| Low legibility | 8.8% | 59.0% | 29.2% | 2.5% |
| Moderate legibility | 18.8% | 35.0% | 42.0% | 69.3% |
| High legibility | 68.5% | 1.5% | 21.5% | 20.0% |

| Smile intensity | Human A (n=198) | Human B (n=128) | Human C (n=182) | LLM (n=168) |
|:---|---:|---:|---:|---:|
| Slight | 33.8% | 64.1% | 33.0% | 12.5% |
| Clear | 19.2% | 33.6% | 39.6% | 63.7% |
| Broad | 28.8% | 1.6% | 23.1% | 22.0% |
| Laughter-like | 18.2% | 0.8% | 4.4% | 1.8% |

## Chance-corrected and baseline diagnostics

| Field | Coverage | Baseline | Cohen's kappa / QWK |
|:---|---:|---:|---:|
| Depiction type | 100.0% | 77.7% | 0.794 |
| Perceived age | 100.0% | 56.3% | 0.396 / 0.718 |
| Gender presentation | 100.0% | 66.4% | 0.808 |
| Expression legibility | 100.0% | 36.6% | 0.183 / 0.525 |
| Face orientation | 100.0% | 40.9% | 0.558 / 0.680 |
| Gaze direction/target | 91.9% | 36.1% | 0.436 |
| Mouth covered | 100.0% | 95.9% | 0.191 |
| Mouth-covering cause | 50.0% | 50.0% | 0.222 |
| Smile presence | 92.4% | 51.6% | 0.772 |
| Smile intensity | 85.8% | 46.0% | 0.292 / 0.652 |

Coverage is the share of matched cases with an LLM value. Baseline is the
accuracy from always predicting the most common human-majority label. QWK is
reported for ordered fields.

## Ordinal diagnostics

| Field | N | Spearman rho | Mean bias [95% CI] | MAE | Within one |
|:---|---:|---:|---:|---:|---:|
| Perceived age | 337 | 0.683 | -0.335 [-0.401, -0.271] | 0.389 | 99.1% |
| Expression legibility | 293 | 0.520 | 0.130 [0.003, 0.240] | 0.635 | 94.9% |
| Smile intensity | 91 | 0.669 | 0.099 [-0.053, 0.250] | 0.495 | 100.0% |

Bias is LLM minus human-majority category index. Confidence intervals resample
pages.

| Threshold | N | Human | LLM | Difference | Sens. | Spec. | Jaccard |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Legibility high | 293 | 61.8% | 85.7% | +23.9% | 96.7% | 32.1% | 0.681 |
| Intensity high | 91 | 31.9% | 26.4% | -5.5% | 58.6% | 88.7% | 0.472 |
| Young adult | 337 | 18.4% | 43.6% | +25.2% | 95.2% | 68.0% | 0.393 |
| Older adult | 337 | 17.2% | 6.5% | -10.7% | 37.9% | 100.0% | 0.379 |

`High` means moderate/high legibility or broad/laughter-like smile intensity.
Rows use the same matched people for human and LLM prevalence.

## Interpretation boundary

The tables support the thesis decisions to retain gender presentation,
depiction type, smile presence, and collapsed/ordinal uses with explicit
caveats; to treat perceived age and legibility as systematically shifted; and
to exclude gaze target and mouth-covering cause from downstream analysis.
People-area results are weak enough that advertisements containing people areas
are excluded from the Chapter 6 corpus. These are validation-sample findings,
not claims of population accuracy or fully adjudicated ground truth.

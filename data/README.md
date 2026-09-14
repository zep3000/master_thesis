# Data sources, provenance, and local layout

Data files are deliberately excluded from this repository. The `.gitignore`
rules retain this document while excluding everything else below `data/`.
This prevents licensed archive images, authentication material, human
annotations, and row-level research records from being published
accidentally.

## Face-detection dataset

The initial tabular input is Stefan Kluge's **The Economist Historical
Advertisements — Faces Dataset**:

- DOI and repository record: <https://doi.org/10.7801/384>
- Dataset description and download: <https://openbigdata.org/resource/the-economist-historical-advertisements-faces-dataset/>
- Related master dataset: <https://openbigdata.org/resource/the-economist-historical-advertisements-master-dataset/>

The publisher describes the Faces Dataset as 116,746 detected faces from
advertisements in 8,840 issues spanning 1843–2014. Each CSV row records an
advertisement/page filename, relative face bounding-box coordinates,
detection confidence, relative face size, predicted age, and a gender
probability. The dataset landing page identifies the release as CC BY 4.0.

The exact local source file used here is expected at:

```text
data/datasets/TheEconomistHistoricalArchives-Faces.csv
```

Its reproducibility fingerprint is:

```text
data rows: 116746
bytes:     20183553
SHA-256:  25aacec3586d2429f93dccce545972326037e1d46a956f588f623e2c07b0d42a
```

The dataset license does not grant redistribution rights for the underlying
magazine page images.

## Archive metadata and page images

Magazine scans and METS XML records come from **The Economist Historical
Archive**, published by Gale:

- Gale collection description: <https://www.gale.com/product-catalog/primary-sources/the-economist-historical-archive>
- German DBIS access record: <https://dbis.ur.de/ULBD/resources/10002>

The local collection was accessed through the DFG-funded German national
licence described by DBIS, whose available edition covers 1843–2007. Access
requires an eligible authenticated session. These archive records and images
are licensed source material and are therefore not distributed here.

`code/scripts/economist_mets_download.ipynb` discovers issue identifiers from
the archive's collection pages and stores validated issue-level METS XML under
`data/metadata/<year>/`. The METS records expose the page-image resources used
by `download_page_images_from_mets.ipynb` and
`download_full_pages_from_issues.ipynb`. Authentication cookies belong in
`code/auth/nationallizenzen_cookie.json`; that directory is ignored by Git.

## Derived-data lineage

The main transformations are:

| Output | Produced by | Principal inputs |
|---|---|---|
| `TheEconomistHistoricalArchives-Faces-deduplicated.csv` and `TheEconomistHistoricalArchives-Faces-deduplication-audit.csv` | `faces_csv_deduplication.ipynb` | Original Faces CSV |
| `TheEconomistHistoricalArchives-Faces-deduplicated_cleaned.csv` and dropped-row audit files | `faces_multi_page_entry_cleanup.ipynb` | Deduplicated Faces CSV |
| `full_pages_1940_2007_joined_manifest.csv` and image audit | `join_double_page_ads_for_sample.ipynb` | Cleaned detections and downloaded page images |
| Individual-only LLM JSONL and people-area-ad decision log | `clean_llm_people_area_ads.ipynb` | Complete-case assembled output from the final production pipeline |
| `face_detection_confidence_by_decade.csv` and its SVG | `face_detection_confidence_by_decade.ipynb` | Deduplicated Faces CSV |
| `faces_pages_by_decade.csv` and incidence SVGs | `faces_pages_by_decade.ipynb` | Deduplicated Faces CSV and METS download log |
| `faces_per_issue_decade_extremes_through_2007.csv` and summary SVG | `pages_with_most_least_faces_per_issue.ipynb` | Deduplicated Faces CSV |
| Evaluation Parquet tables and audits | `prepare_llm_evaluation_300.ipynb` | Human annotation JSONL and model-result JSONL |
| Advertisement IoU threshold figure | `generate_ad_iou_f1_curves.py` | Evaluation advertisement Parquet table |

All notebook paths above are relative to `code/scripts/`. Derived row-level
tables remain excluded; only specifically reviewed figures listed in
`code/output/figures/manifest.json` are versioned.

The preserved local working data currently has these key row counts (headers
excluded):

| File | Rows |
|---|---:|
| Original Faces CSV | 116,746 |
| Deduplicated Faces CSV | 78,353 |
| Deduplication audit | 49,781 |
| Cleaned Faces CSV | 63,832 |
| Entries dropped for spans over two pages | 640 |
| Canonical joined-image manifest | 33,047 |

## Human annotations and model results

Human annotation files were produced during this study with the included
sampling notebooks and local annotation tooling; they are not third-party
datasets and remain excluded. Model-result JSONL files were likewise generated
locally from the licensed page images. Reviewed model-only pipeline artifacts
may be distributed as GitHub release assets indexed by
`artifacts/pipeline/manifest.json`; they are not tracked as ordinary Git data.
The release archives exclude source imagery, human/gold records, credentials,
local logs, and machine-specific paths.

The publication-safe aggregate results of the main 300-page, three-coder
human--LLM evaluation are in `artifacts/human-validation/`. They report the
entity counts, spatial detection results, people-area results, cohort
attrition, human-majority availability, field agreement, chance-corrected
agreement, and selected ordinal diagnostics used in Chapter 5. These tables
support direct checking of the numerical claims without exposing row-level
coder records or licensed source imagery. They do not permit independent
recalculation from individual judgments.

Disclosure-safe aggregate results from the separate brand-name and industry
verification task are published under `artifacts/brand-industry-verification/`.
They can be regenerated with
`code/scripts/annotation/summarize_brand_industry_verification.py`. The raw
export remains excluded because it contains row-level human judgments, source
page identifiers, a persistent session code, exact timestamps, bounding boxes,
and free-text fields.

The evaluation preparation notebook expects its source JSONL files under:

```text
data/annotations/llm_evaluation_300/
```

Application autosaves belong under:

```text
annotation_app_bboxes_full_issues/data/
```

## Expected local tree

```text
data/
├── datasets/       # downloaded tabular source datasets
├── metadata/       # authenticated METS XML and download manifests
├── images/         # licensed source pages and joined spreads
├── annotations/    # local human/model annotation inputs
└── processed/      # derived CSV, JSON, Parquet, and audit outputs
```

Create these directories locally or link them to a separately backed-up data
store. Never force-add their contents to Git.

## Chapter 06: historical and relational smile analyses

`code/scripts/chapter06_analysis.py` consumes the cleaned page-level JSONL
`processed/llm-annotations/individual-only/llm-annotations-individual-ads-only.jsonl`
and its neighbouring `llm-annotations-people-area-ad-cleaning-log.csv`.
`clean_llm_people_area_ads.ipynb` derives both files from the complete-case
assembled production output while preserving the page universe and recording
every retained or removed advertisement.
The input excludes entire advertisements containing people areas; it preserves
page/spread records with no remaining advertisements. The analysis rejects
non-regular individual annotations and never uses face orientation or gaze.

Configure `CH06_INPUT`, `CH06_THESIS`, and `CH06_WORK`, or supply the corresponding
`--input`, `--thesis`, and `--work` command-line arguments. The local paired
repository layout is also recognised. The two notebook sources are
`chapter06_historical_analysis.ipynb` and `chapter06_relational_analysis.ipynb`.
`build_chapter06_notebooks.py --execute` builds them and creates executed private
review copies. Required Python packages are NumPy, pandas, Matplotlib, SciPy,
statsmodels, patsy, nbformat, nbclient, and ipykernel.

The retained chapter contains 7 native Quarto tables and 9 aggregate SVG
figures, starting with gender and age, then smile presence, intensity,
contextual comparisons, comparisons with prior studies, and within-ad
relationships. Corpus totals are merged into the coverage table. The detailed
gender-by-decade table is in the appendix alongside four compact
transcriptions of plotted values.
Tables, executed notebooks, row-level pickles/CSVs, model diagnostics, the
artifact index, and provenance stay in private/local output storage. Only the
aggregate figures are eligible for the public checksum-verified figure manifest.
Committed notebook outputs and attachments remain empty. Use `--integrate` to
append the generated structure after existing chapter notes and `--manifest` to
update the versioned figure interface after verifying the outputs.

Smile denominators use yes/(yes + no); intensity is conditional on smiling.
Descriptive intervals use 2,000 common issue-bootstrap resamples stratified by
year. Within-ad analyses distinguish any-smiling, mixed, and all-smiling
configurations and report their sample restrictions explicitly.

`chapter06_literature.py` adds three explicit comparison tables after the industry
results: Jofre and Cole (2024), the March 2024 financial-advertising working paper
cited as `unda2024GenderStereotypes`, and the expansive-smile expectation discussed
by Goffman and Döring/Pöschl. Published numbers are transcribed benchmarks, not
values read off curves. The Jofre ratio and around-1970 pattern pool news and ads;
our data contain ads only. The financial study overlaps the underlying archive;
the unique largest face per ad is a central-person proxy, checked against the
single-face subset. Tied maxima and uncertain-gender maxima are excluded before
calculating focal gender shares. Broad/laughter-like intensity is compared both
among all assessable faces and conditional on smiling, with fixed period, medium,
age, and area restrictions. Shared issue/year draws generate all new intervals.
No cross-study significance test or causal stereotype score is computed.

`chapter06_support.py` writes the compact numerical appendix, exports a
numerical index, and audits the bootstrap. Its audit checks within-year cluster
counts, explicit repeated-row equivalence, and the main interval endpoints
against an independent 5,000-resample run. Every retained curve or heatmap has
exact underlying values under `CH06_WORK/numerical-data`; the appendix
transcribes only the main decade summaries. The same location's README
documents units, denominators, suppressed cells, and interval types.
Row-level data, local kernel specifications and source archive material must
remain outside the tracked tree.

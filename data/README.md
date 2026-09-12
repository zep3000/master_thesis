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
| `face_detection_confidence_by_decade.csv` and its SVG | `face_detection_confidence_by_decade.ipynb` | Deduplicated Faces CSV |
| `faces_pages_by_decade.csv` and incidence SVGs | `faces_pages_by_decade.ipynb` | Deduplicated Faces CSV and METS download log |
| `faces_per_issue_decade_extremes_through_2007.csv` and summary SVG | `pages_with_most_least_faces_per_issue.ipynb` | Deduplicated Faces CSV |
| Evaluation Parquet tables and audits | `prepare_llm_evaluation_300.ipynb` | Human annotation JSONL and model-result JSONL |

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

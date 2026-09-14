# The History of Smiles — analysis and reproducibility materials

This repository is the public research record for a computational analysis of
facial expressions in historical advertising. It contains the analysis and
pipeline source, reviewed human-annotation records and aggregate evidence,
selected generated figures, and the local annotation interfaces used in the
study.

## Contents

- `code/scripts/`: analysis and data-preparation notebooks
- `code/output/figures/`: selected generated figures and a checksum manifest
- `annotation_app_bboxes_full_issues/`: local PDF bounding-box annotation app
- `annotation_comparison_app/`: local interface for comparing compatible
  human and model annotation sources
- `annotation_brand_verification_app/`: local advertisement-category and
  brand-name verification interface
- `code/pipeline/`: final pipeline source and the archived development
  iterations used to reach it
- `artifacts/pipeline/`: checksummed GitHub release assets containing the full
  reviewed model-only run artifacts
- `artifacts/human-validation/`: pseudonymized row-level human annotations,
  pipeline-development gold data, full-issue face boxes, and aggregate
  validation results
- `artifacts/brand-industry-verification/`: disclosure-safe aggregate audit
  results, error-transition inventories, and interpretation guidance
- `artifacts/full-issue-face-audit/`: complete blind visual-label output and
  the minimal detector-status join used for the Chapter 2 false-negative audit
- `data/README.md`: source provenance, data availability, processing lineage,
  and expected local layout

Notebook outputs are stripped so that embedded records, local paths, and
images are not accidentally distributed.

## Start here

- The data provenance, exclusions, and transformation chain are documented in
  `data/README.md`.
- The final annotation pipeline is described in `code/pipeline/final/README.md`.
- The downloadable run archives and their verified scope are described in
  `artifacts/pipeline/README.md`.
- The human-validation evidence and its limits are documented in
  `artifacts/human-validation/README.md`.
- The blind face-audit labels and disclosure boundary are documented in
  `artifacts/full-issue-face-audit/README.md`.

Raw archive scans and row-level analysis tables are not public. Reviewed human
annotation records are public after removal of direct assignee names,
assignment/session codes, comments, notes, and machine paths. Source-page
identifiers, pseudonymous session IDs, timestamps, and annotation geometry
remain to support transparent checking and reuse; source images remain
excluded under the archive access terms.

## Python environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Notebooks are intended to run from their own directory and refer to inputs
under `../../data/`.

## Generated figure interface

Files intended for downstream documents are declared in
`code/output/figures/manifest.json`. The manifest records their filenames,
sizes, and SHA-256 checksums so consumers can verify the artifacts before use.

## Licensing

No reuse license has been selected yet. Until a license is added, copyright is
retained by the author. Third-party datasets and archive images remain subject
to their own terms and are not included here.

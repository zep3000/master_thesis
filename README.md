# The History of Smiles — analysis and reproducibility materials

This repository contains analysis code, selected generated figures, and a
local bounding-box annotation tool for a computational analysis of facial
expressions in historical advertising.

## Contents

- `code/scripts/`: analysis and data-preparation notebooks
- `code/output/figures/`: selected generated figures and a checksum manifest
- `annotation_app_bboxes_full_issues/`: local PDF bounding-box annotation app
- `annotation_comparison_app/`: local interface for comparing compatible
  human and model annotation sources
- `annotation_brand_verification_app/`: local advertisement-category and
  brand-name verification interface
- `data/README.md`: source provenance, data availability, processing lineage,
  and expected local layout

Notebook outputs are stripped so that embedded records, local paths, and
images are not accidentally distributed.

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

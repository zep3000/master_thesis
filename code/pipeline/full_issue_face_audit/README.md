# Full-issue face audit

This experimental module measures whether faces retained by the historical
Faces dataset differ systematically from faces missed by that detector. It is
deliberately isolated from the frozen production pipeline in `../final/`.

The module does not run face detection. Human-drawn boxes from the full-issue
review app are forced through the frozen individual-person attribute schema.
Each request receives an enlarged face, local context, and the complete PDF
page with the target marked in red.

## Privacy and repository boundary

All source-derived or row-level material is written below `local/`, which is
ignored by Git. Do not commit PDFs, annotations, rendered pages, crops,
request ledgers, or row-level joined results. The tracked portion of this
folder contains code, tests, and documentation only.

## Inputs

- The 35 authoritative `*_annotations.csv` exports from the full-issue app.
- The fixed 35-issue sampling manifest.
- The final cleaned and deduplicated Faces dataset CSV.
- One whole-issue PDF for each sampled issue.
- Poppler `pdfinfo` and `pdftoppm` executables.
- The frozen pipeline in `../final/` and an OpenRouter key.

No machine-specific paths are built into the code. Supply them on the command
line.

## Matching definition

Two-page detector scans are assigned to the left or right physical page using
the detector-box horizontal center, after which x coordinates are transformed
to that physical page. Human and detector boxes are matched one-to-one within
pages using maximum-cardinality bipartite matching at IoU >= 0.17. This is the
relaxed audit threshold that reproduces the reviewed cleaned-data totals of
459 matched human faces and 1,140 missed human faces. The threshold is stored
in the manifest and can be changed for sensitivity analyses.

## Workflow

Prepare and audit all inputs, but select only a deterministic 40-face pilot:

```powershell
python prepare.py `
  --annotations-dir <annotation-dir> `
  --sample-manifest <sample.csv> `
  --detector-csv <cleaned-faces.csv> `
  --pdf-dir <pdf-dir> `
  --pdfinfo <path-to-pdfinfo>
```

Run the pilot with the frozen Qwen/Venice person schema:

```powershell
python run_attributes.py `
  --manifest local/manifest.json `
  --scope pilot `
  --key-file <openrouter-key.txt> `
  --pdftoppm <path-to-pdftoppm>
```

Build local review sheets and a joined pilot CSV:

```powershell
python build_review.py --manifest local/manifest.json
```

After the pilot has been reviewed, the same resumable runner can process all
1,599 faces with `--scope all`. Existing successful pilot records are reused.
For a deterministic tranche, add `--limit 500`; this selects the first 500
face IDs from the same sorted all-face order without creating another manifest.

## Tests

```powershell
python -m unittest discover -s tests -v
```

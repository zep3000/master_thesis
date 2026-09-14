# Brand-name and industry verification

This directory reports the disclosure-safe results of a human audit of the
pipeline's advertisement-level brand or advertiser names and industry
categories. The audit covers the first 200 item positions (0-199) in the brand
verification task. One additional completed position in the local export falls
outside that declared scope and is not analysed.

## Results

Of 200 reviewed items, 196 were accepted as suitable for scoring the two
fields. Four were marked as incorrect upstream identifications, so their field
verdicts were not requested.

| Outcome | Correct / N | Agreement |
|:---|---:|---:|
| Field-scorable identification | 196 / 200 | 98.0% |
| Industry category | 183 / 196 | 93.4% |
| Brand or advertiser name | 193 / 196 | 98.5% |
| Both fields | 182 / 196 | 92.9% |

The three field rows are conditional on an accepted identification. If the four
upstream failures are instead counted as failures, the end-to-end rates are
91.5% for industry, 96.5% for brand name, and 91.0% for both fields. Exact
values and both denominator choices are in `summary.csv`.

The verifier saw the model-assigned labels and either accepted or corrected
them. These are therefore single-verifier audit-agreement rates, not blinded
accuracy estimates and not inter-coder reliability.

## Error inventories

`industry-category-corrections.csv` lists all 13 grouped combinations of brand,
model category, and verified category. Five corrections concern watch brands
that should be `Clothing & accessories`, while the model placed them in
`Toiletries & cosmetics`, `Technology & electronics`, or `Business &
industrial`. Several other errors cross the broad `Business & industrial`
boundary. This clustering matters more than the high aggregate rate: analyses
of particular sectors can inherit concentrated, directionally non-random error.

`brand-name-corrections.csv` contains all three grouped name substitutions. Two
of the three also produce an industry correction because the model identified
the wrong advertised entity. Case differences alone were not counted as errors.
The files contain no source-record identifiers.

## Interpretation guide

1. Use the conditional rates when asking whether the two labels are dependable
   after the advertisement/face item has been accepted. Use the end-to-end rates
   when asking how often the pipeline yields a usable item and a correct label
   in one pass.
2. Read every percentage with its numerator and denominator. The 196 field
   judgments are nested inside 200 reviewed candidates; the denominators are not
   interchangeable.
3. Do not call these values population accuracy. The audited items come from a
   decade- and face-count-stratified evaluation set and are conditional on the
   pipeline producing an advertisement-level item.
4. For corpus-wide brand descriptions, 98.5% conditional agreement supports use
   with a modest error caveat. For industry comparisons, retain a stronger
   caveat and inspect sectors implicated by the correction inventory, especially
   `Clothing & accessories` and `Business & industrial`.
5. For analyses dominated by a small number of brands or sectors, rerun a
   targeted audit. A high pooled rate does not protect a narrow subgroup from a
   clustered error mechanism.

## Privacy and disclosure

The reviewed row-level export is published at
`../human-validation/raw/brand-industry/annotations.jsonl`. Note fields and
machine paths are removed; the pseudonymous session ID, exact timestamps,
source-page identifiers, advertisement and face boxes, and substantive field
judgments remain. It is pseudonymized, not anonymous. The aggregate CSVs in
this directory continue to omit item, image, page, session, time, note, and
bounding-box fields.

## Reproduce locally

Keep the completed export in an ignored local directory, then run from the
repository root:

```powershell
python code/scripts/annotation/summarize_brand_industry_verification.py `
  artifacts/human-validation/raw/brand-industry/annotations.jsonl `
  --output-dir artifacts/brand-industry-verification `
  --qmd-output path/to/private/thesis/tables/generated/brand-industry-verification.qmd
```

The script requires completed records at every item position from 0 through
199, rejects duplicate positions and incomplete field verdicts, and emits only
the allowlisted aggregate fields. `scope.json` records the applied boundary and
the number of excluded completed records.

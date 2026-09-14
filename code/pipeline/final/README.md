# Final pipeline implementation

This directory contains the final frozen implementation of the annotation
pipeline. The primary entry point is `run.py`; `prompts.py` contains the frozen
schemas and prompts, while `matching.py`, `verify.py`, and
`test_production_runner.py` provide supporting logic and checks.

The code is preserved for methodological inspection. Licensed source images,
human labels, credentials, and run outputs are not stored beside it. Reviewed
machine-readable outputs are indexed in `../../../artifacts/pipeline/`.
The pseudonymized 198-page difficult and 200-page stratified human-gold exports
used for pipeline development and evaluation are published separately under
`../../../artifacts/human-validation/raw/pipeline-development/`. They supported
prompt and pipeline selection plus held-out checks; no model weights were
trained on them.

Paths are local configuration rather than repository assumptions. Set
`PIPELINE_IMAGE_DIR` and either `OPENROUTER_API_KEY` or
`OPENROUTER_KEY_FILE` before attempting inference. Manifest regeneration also
accepts `PIPELINE_DIFFICULT_IMAGE_DIR` for the licensed difficult-cohort images;
its gold inputs resolve to the public pseudonymized exports above.

## Complete-case production baseline

The frozen 33,047-page production run contains three pages without a complete
result. `clean_incomplete_pages.py` removes records belonging to those reviewed
page IDs from the analysis-bearing JSONL files under `pages/`, `raw_values/`,
and `normalization/`. It deliberately leaves the frozen manifest, request
ledger, run status, and accounting history unchanged.

Preview and apply the cleanup with:

```powershell
python clean_incomplete_pages.py --run-root <production-run-root>
python clean_incomplete_pages.py --run-root <production-run-root> --apply
```

The command fails if the manifest size or detected count of incomplete pages
differs from the reviewed run. Applying it performs atomic file replacements,
verifies a 33,044-page unique and successful assembled baseline, and writes
`analysis_baseline.json` with exclusions, record counts, sizes, and SHA-256
hashes. The operation is idempotent; no model processing is invoked.

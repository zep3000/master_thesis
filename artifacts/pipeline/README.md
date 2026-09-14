# Pipeline machine artifacts

The large machine-readable artifacts are distributed as release assets instead
of ordinary Git objects. `manifest.json` records their filenames, checksums,
sizes, scope, and sanitization rules.

The archives preserve their `qwen_iteration/` relative paths so the accounting
notebook can inspect both archives after they are extracted into the same
directory. Set `PIPELINE_ARTIFACT_ROOT` to that extracted `qwen_iteration`
directory.

The release tagged `pipeline-documentation-2026-09-12` is published on GitHub.
The final archive includes the complete 33,047-image production manifest, the
structure and entity-stage outputs, raw and normalized values, assembled
results, request ledger, run status, and the final validation/ablation runs.
Its assembled production file contains 33,045 physical rows. Three reviewed
pages lack a complete result and one image identifier is duplicated, so the
analysis-bearing complete-case baseline contains 33,044 unique successful
pages. `code/pipeline/final/clean_incomplete_pages.py` documents and verifies
that cleanup without altering the frozen ledger or manifest.

Included material is limited to request ledgers, raw model responses,
intermediate machine results, normalized values, assembled outputs, and
non-sensitive run metadata. The archives exclude:

- source page images, crops, montages, and PDFs;
- human annotations and gold-relative record sets;
- generated narrative documents and their hardcoded narrative generators;
- console and local process logs;
- credentials, caches, and temporary workspaces;
- vendored third-party source trees;
- machine-specific absolute paths.

Where an otherwise eligible JSON, JSONL, or CSV file contained a local path,
the published copy replaces only the machine-specific root. The per-file
publication inventory inside each archive records original and published
SHA-256 hashes.

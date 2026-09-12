# Pipeline machine artifacts

The large machine-readable artifacts are distributed as release assets instead
of ordinary Git objects. `manifest.json` records their filenames, checksums,
sizes, scope, and sanitization rules.

The archives preserve their `qwen_iteration/` relative paths so the accounting
notebook can inspect both archives after they are extracted into the same
directory. Set `PIPELINE_ARTIFACT_ROOT` to that extracted `qwen_iteration`
directory.

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

# Repository guidance

This repository contains distributable analysis code and selected generated
artifacts. Keep the tracked tree suitable for public access.

## Privacy and repository boundaries

- Never commit raw datasets, human annotations, source PDFs or images,
  credentials, API keys, local logs, or row-level derived data.
- Reviewed model-only pipeline artifacts may be published as GitHub release
  assets when they are listed with sizes and SHA-256 hashes in
  `artifacts/pipeline/manifest.json`. Keep the large files themselves out of Git.
- Release assets must exclude human/gold records, source-derived imagery,
  generated narrative documents, credentials, local logs, absolute paths, and
  unreviewed third-party code.
- Local data belongs below `data/`; only `data/README.md` is tracked.
- Local archive credentials belong below `code/auth/`, which must remain
  ignored.
- Keep notebook outputs and attachments empty before committing.
- Avoid absolute paths and usernames in notebooks, documentation, and code.
- Code under `code/pipeline/iterations/` is archival. Preserve its scientific
  logic, but sanitize machine paths and omit dedicated narrative generators
  whose conclusions are hardcoded rather than computed from inputs.
- Keep `data/README.md` synchronized with source provenance, access terms,
  expected paths, and the notebooks that produce important derived files.

## Paths

- Notebooks in `code/scripts/` run with that directory as their working
  directory. Repository data paths therefore begin `../../data/` and figure
  outputs begin `../../code/output/figures/` (or `../output/figures/`).
- Notebooks in `code/scripts/annotation/` use `../../../data/`.
- Application inputs and autosaves belong below
  `annotation_app_bboxes_full_issues/data/` and remain ignored.

## Versioned figure interface

- Downstream consumers use `code/output/figures/manifest.json` as the complete
  allowlist of shared figures.
- Every manifest entry must use a basename only and must match a tracked file's
  exact byte length and lowercase SHA-256 digest.
- When a shared figure changes, update the figure and manifest in the same
  commit. Do not rename or remove a published figure without checking its
  consumers first.
- Do not add data files, tables, or incidental notebook outputs to the manifest.

## Verification before committing

- Parse every notebook as JSON and confirm all cell outputs and attachments are
  empty.
- Parse every archived Python file and verify that source and artifact
  inventories agree with the files being published.
- Verify pipeline release-asset hashes and scan the archive contents using the
  same exclusions that govern the tracked tree.
- Recompute every manifest size and SHA-256 value.
- Search tracked files for credentials, absolute local paths, and excluded
  material.
- Keep changes scoped and preserve existing user edits.

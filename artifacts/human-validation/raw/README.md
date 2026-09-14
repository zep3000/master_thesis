# Pseudonymized row-level human annotations

This directory contains the reviewed row-level human records used in the
thesis. `manifest.json` gives the record count, byte length, SHA-256 checksum,
and applied transformations for every published file.

## Contents

- `three-coder/coder-a.jsonl`, `coder-b.jsonl`, and `coder-c.jsonl`: 300 page
  records per coder for the common Chapter 5 validation set.
- `brand-industry/annotations.jsonl`: 201 completed review records. The first
  200 positions define the reported audit; position 200 is retained as part of
  the complete raw export but remains outside the declared analysis scope.
- `full-issue-scans/`: 35 issue-level CSV files containing 1,599 face boxes on
  557 pages. Coordinates are relative to page width and height.

## Publication processing

The three-coder exports replace every `assignee_name` with `A`, `B`, or `C`.
All datasets remove assignment/session codes, comment/note fields, and absolute
machine paths where present. Annotation-set IDs, pseudonymous session IDs, exact
timestamps, page identifiers, workflow telemetry, bounding boxes, and
substantive annotation values remain.

The full-issue CSVs required no identity redaction: their only columns are
`issue_page_identifier`, four relative bounding-box coordinates, and
`analyzable`. They were parsed, range-checked, and written with consistent UTF-8
and LF line endings.

These files are pseudonymized rather than anonymous. Session IDs and detailed
timestamps can support linkage if combined with information held elsewhere.
They contain no direct coder names, assignment/session codes, comments, email
addresses, credentials, or source images.

The export can be rebuilt with
`code/scripts/annotation/publish_human_annotations.py`. Source paths are passed
explicitly; the script has no machine-specific defaults.

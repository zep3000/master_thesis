# Pseudonymized row-level human annotations

This directory contains the reviewed row-level human records used in the
thesis. `manifest.json` gives the record count, byte length, SHA-256 checksum,
and applied transformations for every published file.

## Contents

- `three-coder/coder-a.jsonl`, `coder-b.jsonl`, and `coder-c.jsonl`: 300 page
  records per coder for the common Chapter 5 validation set.
- `pipeline-development/difficult-198.json` and `stratified-200.json`: the two
  full human-gold exports used while developing and evaluating the final LLM
  pipeline. The nominal 200-page difficult set contains 198 available pages;
  the stratified set contains all 200. The first 140 pages of each formed the
  development cohort, leaving 58 and 60 pages, respectively, as reserves.
  These records guided prompt and pipeline choices; they were not used to train
  model weights.
- `brand-industry/annotations.jsonl`: 201 completed review records. The first
  200 positions define the reported audit; position 200 is retained as part of
  the complete raw export but remains outside the declared analysis scope.
- `full-issue-scans/`: 35 issue-level CSV files containing 1,599 face boxes on
  557 pages. Coordinates are relative to page width and height.

## Publication processing

The three-coder exports replace every `assignee_name` with `A`, `B`, or `C`;
the named pipeline-development annotator is represented as `D`. The difficult
source contained no assignee-name field. All datasets remove assignment/session
codes, comment/note fields, and absolute machine paths where present.
Annotation-set IDs, pseudonymous session and source-assignment IDs, exact
timestamps, page identifiers, workflow telemetry, bounding boxes, and
substantive annotation values remain.

The difficult source export also contained one draft record belonging to a
different assignment. It is not part of the 198-page gold set and is excluded;
that selection is counted in `manifest.json`.

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

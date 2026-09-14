# Full-issue face-audit annotations

This artifact publishes the complete blind visual-label output used to compare
human-identified faces missed and found by the historical Faces Dataset
detector. The annotations were produced locally in a Codex session with the
GPT-5.5 Medium model. Each decision used only an extended face crop and an
opaque `blind_id`; human judgments, detector status, issue metadata, page
context, and downstream classifications were hidden during annotation.

## Files

- `raw/direct-visual-annotations.jsonl`: the complete 1,599-line annotation
  output in original blind-ID order. Every row contains `blind_id`,
  `analyzability`, `gender`, `age`, and `smile_presence`.
- `raw/blind-detection-status.csv`: the minimal post-annotation join index,
  containing only `blind_id` and whether the historical detector found or
  missed that human-identified face.
- `manifest.json`: record counts, schemas, label counts, byte lengths,
  checksums, and source-provenance hashes.

Join the two row-level files on `blind_id`. The publication notebook
`code/pipeline/full_issue_face_audit/direct_visual_annotation/full_issue_face_audit_attribute_profile.ipynb`
validates the one-to-one join and regenerates the aggregate Chapter 2 table.

## Disclosure boundary

The export contains no source images, crop files, bounding boxes, archive
paths, machine paths, credentials, free-text notes, or human analyzability
labels. Page and issue identifiers are also omitted from the join index because
they are unnecessary for reproducing the published table. The accompanying
human full-issue face boxes are published separately under
`artifacts/human-validation/raw/full-issue-scans/`.

Gender and age values are perceived visual labels from a single blind model
pass. They are analytical proxies, not verified identities or demographic
ground truth.

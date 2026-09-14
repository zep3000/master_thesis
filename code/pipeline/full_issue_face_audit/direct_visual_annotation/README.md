# Direct visual annotation

Local-only blind annotation workspace for the full-issue face audit.

This folder deliberately separates the visual annotation step from downstream
analysis. The visible annotation input is only an extended face crop plus an
opaque `blind_id`. Human labels, detector status, issue metadata, and audit
grouping fields are not shown in the review sheets.

## Prepare blind crops

```powershell
C:\master\.venv\Scripts\python.exe master_thesis\analysis\code\pipeline\full_issue_face_audit\direct_visual_annotation\prepare_blind_crops.py `
  --pdftoppm <path-to-pdftoppm>
```

Outputs are written below `local/direct_visual_annotation/`:

- `blind_manifest.json`: opaque id to crop path and private join key.
- `crops/`: one extended crop per annotated face.
- `sheets/`: contact sheets for Codex-session visual annotation.
- `raw_annotations.jsonl`: append-only annotation target.

Schema for each annotation line:

```json
{
  "blind_id": "blind_000001",
  "analyzability": "no expression can be inferred",
  "gender": "don't know",
  "age": "don't know",
  "smile_presence": "don't know",
  "notes": ""
}
```

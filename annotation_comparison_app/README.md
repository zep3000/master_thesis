# Annotation comparison app

Local interface for comparing compatible human and model annotation sources
page by page. It provides overlay controls, spatial entity matching,
field-level agreement matrices, provenance diagnostics, and optional local
manual match overrides.

No source images, human annotations, saved matches, or research-result JSON is
included. The bundled manifest is synthetic and exists only as a format
example.

## Run

Requires Node.js 18 or newer.

```powershell
npm start
```

Open `http://127.0.0.1:5177`. If that port is occupied, the server advances to
the next available port. The server binds to the loopback interface by default.

## Local configuration

Set any of these environment variables before starting the app:

| Variable | Purpose | Default |
|---|---|---|
| `ANNOTATION_COMPARISON_MANIFEST` | Default page manifest | `data/example_manifest.json` |
| `ANNOTATION_COMPARISON_MANIFEST_DIR` | Additional standalone manifests | `data/manifests` |
| `ANNOTATION_COMPARISON_LLM_DIR` | Compatible model-result JSONL files | `data/llm` |
| `ANNOTATION_COMPARISON_HUMAN_RESULTS_DIR` | Aggregate human exports | `data/human_exports` |
| `ANNOTATION_COMPARISON_HUMAN_SESSIONS_DIR` | Local annotation sessions | `data/human_sessions` |
| `ANNOTATION_COMPARISON_IMAGE_DIRS` | Image roots, separated by the platform path delimiter | `data/images` |
| `ANNOTATION_COMPARISON_MATCH_DIR` | Saved manual match overrides | `data/manual_matches` |
| `HOST` | Listening interface | `127.0.0.1` |
| `PORT` | Initial port | `5177` |

The released pipeline archives contain compatible machine-readable model
outputs. Extract an archive and point `ANNOTATION_COMPARISON_LLM_DIR` at a
directory containing the JSONL sources to inspect. Licensed images and human
records must be supplied locally.

Only configured manifests and image directories can be selected through the
API. The image endpoint serves only files resolved for the active manifest.

## Comparison behavior

- Stable source colors across pages
- Toggleable advertisement, face, and group overlays
- Spatial entity matching using bounding-box intersection over union
- Field-level matrices for page, advertisement, individual, and group labels
- Explicit agreement, disagreement, missing-value, and box-overlap indicators
- Provenance, validation, normalization, model, and local source diagnostics
- Optional local manual matches between entities from different sources

The source annotation files are read without modification. Manual matches are
separate local records under the configured match directory.

## Methodology documents

- `docs/annotation_playbook_v1.md` describes the annotation decisions.
- `docs/annotation_flow_v1.yaml` records the corresponding structured flow.

## Tests

```powershell
npm test
```

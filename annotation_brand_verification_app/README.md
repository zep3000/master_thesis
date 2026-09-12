# Brand verification app

Local, standalone interface for reviewing model-assigned advertisement
categories and brand or advertiser names. The annotator can flag an incorrect
advertisement identification, review category and name fields, enter
corrections, inspect model-provided face boxes, and add local notes.

No source images, completed human reviews, session identifiers, or real model
results are included. The bundled manifest is entirely synthetic.

## Run

Requires Node.js 20 or newer.

```powershell
npm start
```

Open `http://127.0.0.1:5182/`.

## Local configuration

| Variable | Purpose | Default |
|---|---|---|
| `BRAND_VERIFICATION_MANIFEST` | Verification manifest | `data/example_manifest.json` |
| `BRAND_VERIFICATION_IMAGE_DIR` | Root containing the manifest images | `data/images` |
| `BRAND_VERIFICATION_DATA_DIR` | Session and annotation output | `data/annotations` |
| `HOST` | Listening interface | `127.0.0.1` |
| `PORT` | Listening port | `5182` |

Image requests are resolved from the configured image root using the manifest
filename. Absolute paths stored in a manifest are not served.

## Build a local manifest

`scripts/build-manifest.js` combines a page manifest with compatible
machine-readable model results. Configure it with:

| Variable | Purpose |
|---|---|
| `BRAND_SOURCE_SET` | JSON containing `annotation_set.manifest.images` or `images` |
| `BRAND_MODEL_RESULTS` | Model-result JSONL |
| `BRAND_OUTPUT` | Optional output path; defaults to `data/generated_manifest.json` |
| `BRAND_TASK_ID` | Optional public task identifier |
| `BRAND_TASK_NAME` | Optional display name |

```powershell
npm run build:manifest
```

The builder writes filenames rather than absolute image paths and retains only
`year` and `decade` from source metadata. Generated manifests remain local
unless they undergo a separate publication review.

## Outputs

Each local session receives an eight-digit identifier. Item records and the
regenerated JSONL export are stored beneath the configured data directory.
These files contain human judgments and must not be committed.

## Tests

```powershell
npm test
```

# Pipeline code archive

This directory preserves the first-party source code used while developing and
running the visual-annotation pipeline. It is organized for inspection rather
than as an installable package.

- `iterations/` contains the successive experimental implementations. Original
  directory names are retained wherever they identify a distinct run family.
- `final/` contains the final standalone runner, prompts, normalization,
  matching, export, verification, and test code.
- `accounting/` contains an output-free notebook that reconstructs request,
  token, and provider-reported cost totals from the released machine artifacts.

Machine-specific paths were replaced with relative paths or environment-backed
configuration. The transformations and deliberately omitted source files are
listed in `source-inventory.json`.

The source archive does not contain page images, crops, human annotations,
credentials, console logs, or generated narrative documents. Machine-readable
run artifacts are distributed separately and indexed under
`../../artifacts/pipeline/`.

## Configuration

The final runner accepts these environment variables in addition to its command
line options:

- `OPENROUTER_API_KEY` for API authentication.
- `OPENROUTER_KEY_FILE` for an optional local key file.
- `PIPELINE_IMAGE_DIR` for local source images.
- `PIPELINE_PROJECT_ROOT` for legacy relative inputs used by auxiliary tools.

The historical iteration scripts are not guaranteed to run without their
excluded local inputs. They are retained to document prompt, schema,
normalization, routing, and evaluation development.

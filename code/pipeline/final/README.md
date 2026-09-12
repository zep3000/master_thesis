# Final pipeline implementation

This directory contains the final frozen implementation of the annotation
pipeline. The primary entry point is `run.py`; `prompts.py` contains the frozen
schemas and prompts, while `matching.py`, `verify.py`, and
`test_production_runner.py` provide supporting logic and checks.

The code is preserved for methodological inspection. Licensed source images,
human labels, credentials, and run outputs are not stored beside it. Reviewed
machine-readable outputs are indexed in `../../../artifacts/pipeline/`.

Paths are local configuration rather than repository assumptions. Set
`PIPELINE_IMAGE_DIR` and either `OPENROUTER_API_KEY` or
`OPENROUTER_KEY_FILE` before attempting inference.

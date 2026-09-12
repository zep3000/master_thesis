# Local comparison data

Only `example_manifest.json` is a tracked synthetic fixture. Supply research
inputs locally and keep them out of Git.

Default directories:

- `images/`: licensed page images;
- `llm/`: compatible model-result JSONL files;
- `human_exports/`: aggregate human annotation exports;
- `human_sessions/`: local annotation sessions;
- `manifests/`: additional page manifests;
- `manual_matches/`: match overrides written by the comparison app.

All locations can be changed with the environment variables documented in the
parent README.

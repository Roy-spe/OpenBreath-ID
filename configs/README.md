# Experiment settings

[`primary.json`](primary.json) contains the primary study's data, split,
training, duration and evaluation settings in a compact public schema.
It preserves the numerical recipe without including internal proposals,
screening decisions or audit-log references.

Use it with `scripts/prepare_reproduction.py` and the
[reproduction guide](../docs/REPRODUCIBILITY.md). It is not a replacement for
the unreleased execution artifacts required by some legacy research runners.

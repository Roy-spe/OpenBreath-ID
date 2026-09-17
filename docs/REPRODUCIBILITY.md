# Reproduction guide

Run commands from the repository root after completing the README installation.
Settings are in [`configs/primary.json`](../configs/primary.json); implementation
details are in [the method notes](SUPPLEMENT.md).

## 1. Prepare the data

Obtain the original data separately. Primary files must follow
`DATASET_ROOT/subj_NNN/Wake.mat`, with the two-channel variable `fieldValue`.
Retain the provider's subject labels.

```bash
python scripts/prepare_reproduction.py --dataset-root "PATH_TO_DATASET"
```

This checks file headers, the 97-person cohort and the five reconstructed
partitions. It verifies the split checksum, but not the historical signal
contents or signal quality. It writes no preparation files and runs no training.

## 2. Prepare training jobs

```bash
python scripts/prepare_reproduction.py --dataset-root "PATH_TO_DATASET" --output-dir private/run_01
```

Use a new directory under `private/`. It receives local split assignments,
a preflight summary and `VALIDATION_JOB_PREVIEW.json`. Keep these files local.

The preview contains explicit `argv` command arguments for the 30 primary
encoder runs: stacked CNN and BIE, five partitions, three seeds. It also contains
15 optional quality-head jobs, each depending on its BIE checkpoint/report.
Execute a chosen job's arguments from the repository root only when ready to
train; preparation does not execute them. Reports and checkpoints are directed
to the private run directory. Inspect supported arguments with:

```bash
python -m openbreath_id.neural_train --help
python -m openbreath_id.quality --help
```

Use training identities for fitting and validation identities for checkpoint
selection. Do not tune on the corresponding evaluation identities.

## 3. Evaluate enrollment/probe observations

Keep the same fixed anchors across durations. Average normalized 30-second
embeddings separately for enrollment and probe, normalize each resulting
template, and score with cosine similarity. Select operating thresholds on
validation identities and apply them unchanged to evaluation identities.
Average metrics across seeds within a partition, then across partitions.

Relevant utilities are `trial_manifest.py`, `duration_eval.py`,
`multifold_test.py`, `metrics.py` and `multifold_bootstrap.py` under
`src/openbreath_id/`. Some research runners still require unreleased trial
and execution artifacts; the compact public configuration is not a drop-in
replacement for those original artifacts. A complete fresh-checkout
training-to-EER runner is not yet available. Classical/fusion runners and
their feature extractor are not included.

## 4. Check and regenerate figures

```bash
python scripts/verify_release.py
python scripts/figure_duration.py
```

The second command prints Figure 2's LaTeX from the released aggregate CSV.
Both figures include editable `.tex` and prebuilt PDF/SVG/PNG files. With a
TeX installation providing `standalone`, PGF/TikZ and the required fonts:

```bash
pdflatex -output-directory=figures figures/figure_architecture_compact.tex
pdflatex -output-directory=figures figures/figure_duration_readable.tex
```

Synthetic tests and aggregate-figure checks validate implementation consistency,
not empirical biometric performance.

# Implementation notes

## Data and partitions

The primary study uses 97 primary Wake recordings, two stored channels at 6 Hz,
and five subject-disjoint partitions. Training/validation/evaluation sizes are
62/15/20, 62/15/20, 63/15/19, 63/15/19 and 63/15/19. Split seed: 2027;
training seeds: 2027, 2028, 2029. Every identity appears in one evaluation fold.
Physical left/right channel assignment is unverified. Results are within-recording
development evidence, not independent cross-visit validation.

## Preprocessing and models

For each 30-second window: replace non-finite samples with each channel's finite
median (or zero); flip Channel 2 when centered cross-channel correlation is
negative; subtract each channel's median; divide both by a shared robust scale;
clip to [-12,12]. The shared scale is the median of channel median absolute
deviations, with standard-deviation and unit-scale fallbacks. Construct sum and
difference after preprocessing. See `neural_data.py` for exact numerical guards.

| Model | Representation | Encoder parameters |
|---|---|---:|
| Stacked CNN | One width-72 temporal encoder over channels, sum/difference and masks | 2,156,160 |
| Shared two-tower | Shared width-72 channel encoder, concatenation and projection | 2,182,560 |
| BIE | Shared width-40 channel encoder, separate sum/difference branches and pairwise interactions | 2,187,760 |

All produce L2-normalized 256-D embeddings. The two-tower and BIE branches
produce 128-D outputs. Masks denote whole-channel availability, not feature-level
missingness. Stacked CNN and BIE are the primary comparison; shared two-tower is
a later architectural control. The backbone, projections and losses are defined
in `neural_models.py` and illustrated in Figure 1.

## Training

Use 40 epochs, 200 batches per epoch, 16 identities x 4 windows per batch, and
at most 600 training windows per recording. The objective is ArcFace + 0.5
supervised contrastive loss (scale 30, margin 0.3, temperature 0.07). Optimization
uses AdamW (learning rate 0.0003, weight decay 0.0001), cosine scheduling and
gradient-norm clipping at 5. No primary-training augmentation is used.

Select checkpoints by validation EER at 60 s + 0.5 x EER at 30 s, with at most
30 validation probes per identity. There is no duration-specific retraining or
enrollment-time classifier fitting. Full settings, including the optional
quality-head recipe, are in `configs/primary.json`.

## Trials and metrics

Enrollment starts at sample zero within a fixed 300-second anchor. A 30-minute
sample-index guard follows that anchor; the first probe anchor starts at sample
12,600. Probe anchors are nonoverlapping 300-second blocks. Retain all complete
anchors up to 120, otherwise select 120 evenly indexed anchors using integer
linspace. Shorter observations use prefixes of the same anchors. This guard is
array separation, not verified wall-clock time or a separate visit.

Pair every probe with each enrollment identity in its role. Matching identities
form genuine trials; different identities form impostor trials. Average segment
embeddings, renormalize each template, then use cosine similarity. The two primary
enrollment/probe durations are 300/60 s and 300/30 s; other short durations are
descriptive. Enrollment and probe durations are not channel durations.

Report EER, nminDCF (unit costs, target prior 0.01), and TAR/FAR at the unchanged
validation-selected threshold targeting 1% FAR. Macro-average seeds within each
partition, then partitions; do not pool raw scores across partitions. The paired
participant bootstrap uses genuine weight w_i and impostor weight w_i*w_j,
shared across systems/seeds, with 10,000 draws and seed 2027. These are conditional
fixed-model intervals. The original Holm family includes both BIE-versus-stacked
and quality-versus-mean comparisons at both primary durations; do not silently
redefine that family. Later experiments use different roles and pairing designs
and must not be ranked directly against these primary results.

# Real-MRI Controlled Pilot V0

Run date: 2026-09-14. Status: **V1 completed diagnostic; not eligible for an abstract claim or
release-gate evidence.**

V1 supersedes the initial artifact after adding per-subregion Dice loss, correcting the
unconditional-auxiliary evidence target, and enabling verified preprocessing cache reads.

## Purpose

This pilot asks whether the shared preprocessing, missing-contrast masks, 3D
segmentation-to-symbolic path, matched answer-only model, unconditional spatial auxiliary,
question-conditioned grounding, calibration, evidence, and bootstrap reporting execute on
real MRI. It is intentionally too small to estimate comparative performance.

## Frozen pilot configuration

- first-by-case-ID subjects within the locked split: 6 train and 4 validation;
- no test prediction;
- whole-volume per-contrast normalization and trilinear/nearest downsampling to `24³`;
- two epochs, width 4, seed `20260914`;
- 5,064-parameter two-level 3D U-Net pilot and 713-parameter controlled MRI-VLM;
- categorical presence, laterality, relative-volume, cross-region, and unanswerable tasks;
- identical initialization for the three MRI-VLM variants;
- identical deterministic modality-subset training schedule;
- five validation conditions: all contrasts, no T1-Gd, no FLAIR, FLAIR only, T1-Gd only;
- 500 subject-bootstrap resamples per reported interval;
- 10.3 seconds total wall time on CPU with the verified ten-case cache; model training took
  approximately one second per path.

```bash
mri-vlm-real-pilot /absolute/path/to/Task01_BrainTumour \
  --output artifacts/results/real_pilot_v0.json \
  --train-cases 6 --validation-cases 4 --spatial-size 24 \
  --epochs 2 --width 4
```

Result SHA-256:
`e548a4888df8d1ed4c7c16dfe9579ec8cabe19dfe751a0a7317ca015e24e82ff`.

## Validation diagnostic

| Available contrasts | Seg→symbolic accuracy | Seg whole-tumor Dice | Answer-only accuracy | Unconditional auxiliary accuracy / evidence Dice | Question-grounded accuracy / evidence Dice |
|---|---:|---:|---:|---:|---:|
| FLAIR+T1+T1-Gd+T2 | 0.30 | 0.026 | 0.30 | 0.20 / 0.268 | 0.20 / 0.216 |
| FLAIR+T1+T2 | 0.45 | 0.019 | 0.30 | 0.20 / 0.268 | 0.20 / 0.228 |
| T1+T1-Gd+T2 | 0.20 | 0.000 | 0.30 | 0.20 / 0.141 | 0.20 / 0.201 |
| FLAIR only | 0.10 | 0.021 | 0.20 | 0.20 / 0.473 | 0.20 / 0.316 |
| T1-Gd only | 0.10 | 0.007 | 0.30 | 0.20 / 0.093 | 0.20 / 0.071 |

Each VLM cell contains 20 validation questions from four subjects. Subject-bootstrap
intervals are stored in the JSON but are too unstable at `n=4` to interpret. Answer-only
unanswerable hallucination ranged from 0.25 to 0.50; both evidence-trained variants happened
to predict abstention for all four unanswerable examples. This is not an estimate of
abstention reliability.

## What this falsified

- Six training subjects, two epochs, and `24³` resolution are insufficient for either
  segmentation or language-conditioned prediction.
- The grounded variant did not outperform the unconditional auxiliary on answer accuracy;
  their evidence Dice values were also similar in most conditions. This pilot provides no
  evidence that question-conditioned grounding adds value.
- The modality-dropout 3D U-Net did not converge, so it is not the required strong modular
  MR baseline and cannot adjudicate the main study question.
- Mean validation subregion Dice was at most 0.010 in every condition after two epochs;
  the apparently nonzero whole-tumor Dice therefore does not indicate useful segmentation.
- Calibration numbers from an underfit model and four subjects are not scientifically
  meaningful even though the evaluation code executes.

## Engineering evidence obtained

The pilot establishes that real four-contrast NIfTI data can pass through one shared
deterministic preprocessing path; every method accepts the same modality masks; the
unconditional evidence head is invariant to question tokens; segmentation predictions are
converted to symbolic answers; and per-condition answer, evidence, hallucination,
calibration, timing, parameter-count, and subject-bootstrap outputs are serialized.

## Required next run

Before another comparative claim, cache preprocessing, prove one-case overfit separately
for segmentation and each VLM objective, raise spatial resolution, use a converged strong
segmentation model, and rebalance the QA distribution identified by the question-only
control. Only then expand to multiple seeds and all 15 conditions.

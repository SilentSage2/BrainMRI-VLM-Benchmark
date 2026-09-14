# Residual 3D Missing-Contrast Baseline Direction Runs

Run date: 2026-09-14. Status: **small development direction gate passed in V3; not eligible
for an abstract claim, Figure 4, or public release.** No test cases were read.

## Fixed scope

- Deterministically sampled 32 training and 8 validation subjects from the locked split.
- MRI-foreground crop, symmetric cubic padding, foreground-only z-score normalization, and
  `64³` resampling; cache identity includes the complete preprocessing specification.
- Three-level residual 3D U-Net with GroupNorm, 112,480 parameters, CPU only.
- Balanced cyclic schedule over all 15 non-empty subsets of FLAIR, T1, T1-Gd, and T2.
- Best epoch selected by mean full-contrast validation region Dice.
- Symbolic QA includes only laterality and edema-versus-core cases whose preprocessed
  reference answer agrees with the original physical-volume answer.

## Controlled sequence of runs

| Run | Change | Best epoch | Full-input result | Decision |
|---|---|---:|---|---|
| V1 | 10 epochs, LR 0.001, raw labels 1/2/3 | 2 | WT Dice 0.184; raw-region mean 0.162; symbolic balanced accuracy 0.667 | Underfit; reject |
| V2 | 30 epochs, LR 0.003, raw labels 1/2/3 | 26 | WT Dice 0.740; raw-region mean 0.289; symbolic balanced accuracy 0.875 | Raw label-2 endpoint is nonstandard; reject |
| V3 | Standard BraTS WT/TC/ET soft-Dice loss and endpoints | 30 | WT/TC/ET 0.749/0.542/0.424; symbolic balanced accuracy 0.917 | Direction gate passed |

V3 full-input enhancing-fraction MAE was 0.224, so continuous symbolic interpretation is
not ready. V3 used eight validation subjects and its best epoch was the final allowed epoch;
both facts preclude a convergence or generalization claim.

Single-contrast V3 WT Dice was 0.741 for FLAIR, 0.176 for T1, 0.386 for T1-Gd, and 0.556
for T2. T1-Gd-only ET Dice was 0.405. These values are useful for debugging the expected
contrast-specific direction but are too small-sample to support sensitivity claims.

## Reproduction and provenance

V3 command:

```bash
mri-vlm-baseline /absolute/path/to/Task01_BrainTumour \
  --output artifacts/results/baseline_direction_v3.json \
  --checkpoint artifacts/checkpoints/baseline_direction_v3.pt \
  --cache-dir data/processed/baseline-direction-v2 \
  --train-cases 32 --validation-cases 8 \
  --spatial-size 64 --epochs 30 --width 4 --learning-rate 0.003
```

- V1 result SHA-256: `8c5bb0065449fd6f0474ebdb7b0006982f97ce5340238b4c8fd6bfeca75a9f47`.
- V2 result SHA-256: `83f735b98da223e8a0a2c42df29ce4523250f2dc4a7dcd004d10ba7e33d2a627`.
- V3 result SHA-256: `23355769fe0caa5f9a41c394de8e0ca0efabe06f1a313b145e038e9e29c75939`.
- V3 checkpoint SHA-256:
  `91ff7b18bf56cec11a1522bca9e40500e7bd539e8851832bd1c5955f4c58dc4f`.
- V3 wall time: 472.3 seconds on the local ten-core ARM CPU; no accelerator was available.

## Next gate

Expand the same V3 protocol to a materially larger training/validation cohort and verify
that WT/TC/ET, symbolic balanced accuracy, and enhancing-fraction error stabilize. Then use
the same cohort, preprocessing, modality schedule, optimizer-step budget, and QA V1 targets
for answer-only, unconditional auxiliary, and question-grounded paths. Test remains unread.

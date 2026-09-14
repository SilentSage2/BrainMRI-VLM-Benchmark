# Modular MR V4 Multi-Seed Protocol

Freeze date: 2026-09-14. Status: **running; test split sealed**.

## Question

Does an MR-specialized residual 3D segmentation model followed by deterministic symbolic QA
support missing-contrast reasoning more reliably than the end-to-end MRI-VLM paths, and
does balanced modality dropout improve robustness over complete-input-only training?

## Frozen design

- Identical 64 training and 16 validation subjects used by matched V3; split seed 20260914.
- Training seeds 20260914, 20260915, and 20260916; `48³`; width 4; 20 epochs; AdamW at
  0.001.
- Two matched training regimes: cyclic balanced exposure to all 15 non-empty contrast
  subsets, and complete-input-only training.
- Standard BraTS whole-tumor, tumor-core, and enhancing-tumor Dice.
- QA V1 laterality, edema-versus-core, and enhancing-fraction-bin symbolic answers.
- All 15 contrast conditions, raw and balanced symbolic accuracy, enhancing-fraction MAE,
  subject scores, and subject bootstrap intervals.
- A voxel-softmax confidence proxy is reported separately from answer confidence. The
  predefined abstention threshold is 0.75; it must not be described as clinically
  calibrated.
- The primary ablation effect is dropout-minus-no-dropout subject accuracy averaged over
  the 14 incomplete-contrast conditions, with hierarchical seed/subject bootstrap CI.
- Every input result must record zero test cases read and identical validation IDs.

## Decision rule

The modular pathway is a credible main baseline only if it materially exceeds the QA V1
question-only balanced accuracy (0.417), retains interpretable WT/TC/ET performance, and
shows a stable missing-contrast pattern. A modality-dropout benefit requires a positive
hierarchical interval. If these conditions fail, the paper claim narrows to a benchmark of
failure modes rather than a reliable modular alternative.

No public or private clinical images are transmitted to external APIs. Any future external
foundation model must have locally runnable weights, a documented license and version, and
a disclosed overlap/leakage audit before comparison.

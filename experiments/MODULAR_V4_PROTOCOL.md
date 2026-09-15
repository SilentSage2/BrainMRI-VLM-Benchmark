# Modular MR V4 Multi-Seed Protocol

Freeze date: 2026-09-14. Status: **completed development experiment; test split sealed**.

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

## Completed result

| Training regime | Full balanced QA | Missing balanced QA | Full WT/TC/ET Dice |
|---|---:|---:|---:|
| Balanced modality dropout | 0.640 | 0.570 | 0.658 / 0.579 / 0.489 |
| Complete-input only | 0.787 | 0.578 | 0.799 / 0.769 / 0.709 |

The modular paths substantially exceeded the QA V1 question-only and fixed-slice controls
(both 0.417 balanced accuracy) and the matched grounded VLM (0.426 full-input; 0.424 across
missing conditions). This supports the narrow conclusion that MR-specialized perception
provides more reliable symbolic reasoning than the tested small end-to-end VLMs.

Balanced modality dropout improved raw subject answer accuracy averaged across the 14
incomplete-contrast conditions by `+0.069`; seed effects were `+0.019`, `+0.074`, and
`+0.115`, with hierarchical seed/subject bootstrap 95% CI `[+0.016, +0.124]`. However,
balanced accuracy was slightly lower than no-dropout (0.570 versus 0.578), and dropout
substantially reduced full-input segmentation and QA performance. The supported
interpretation is a raw-accuracy/worst-case robustness tradeoff, not uniform superiority.

The predefined 0.75 voxel-confidence abstention threshold had zero coverage for the dropout
model and 0.771 full-input coverage for no-dropout. This confirms that segmentation
softmax confidence is not a calibrated answer-confidence mechanism and should not be used
as such without a separate validation protocol.

Summary SHA-256:
`3c30b91131ee77015e858fc793d78301e662734aae5f62d4f2436bc42bd063db`.
Figure 5 PNG/PDF/CSV SHA-256:
`4a5ceebe1f0a07b78989a0d06ee426e8ca5cedf18999fe6a8c14b9d593c5faa2`,
`b89078534426d2dc5e320ae04ecd315ff8851df7c3172f353f3a5136f42e9a54`, and
`5b54b49b390923e2b84da5e7825ddc68d1433e03ef488758381d4d83dc6d3ece`.
The PNG/PDF hashes changed after a label-only paper-readability revision; the source CSV
and all numerical values are unchanged.

**Decision:** promote the modular pathway to the primary MR-specialized baseline and frame
the end-to-end grounding result as a controlled negative comparison. Keep the held-out test
closed until external/local-weight baseline selection and Figure 5 case-level review are
frozen.

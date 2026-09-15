# Frozen Narrow Claim Before Held-Out Evaluation

Freeze date: 2026-09-14. This document may not be revised in response to held-out outcomes.

## Claim eligible for testing

On mask-verifiable QA derived from registered multi-contrast brain MRI, an MR-specialized
3D segmentation-to-symbolic pipeline will support incomplete-contrast reasoning more
reliably than the **tested small controlled end-to-end VLMs**. Balanced modality dropout is
expected to improve raw subject-level accuracy under missing contrasts while sacrificing
some complete-input perception quality.

This is a workflow reliability and evaluation claim. It is not a claim that segmentation
is universally preferable to VLMs, that question-conditioned grounding never works, or
that the system is clinically useful.

## Locked primary comparison

- Systems: modular dropout, modular no-dropout, and question-grounded small 3D VLM.
- Population: 66 frozen held-out MSD Task01 subjects listed by identifier only in
  `configs/test_subject_manifest.json`.
- Conditions: all 14 incomplete subsets of FLAIR, T1, post-contrast T1, and T2; the complete
  subset is secondary.
- Endpoint: paired subject-averaged raw QA accuracy, modular dropout minus grounded VLM.
- Uncertainty: hierarchical bootstrap over the three training seeds and held-out subjects,
  10,000 draws with seed 20260914.
- Support criterion: positive mean effect with 95% interval excluding zero. Balanced
  accuracy, WT/TC/ET Dice, calibration proxy, and dropout/no-dropout tradeoffs must be
  reported regardless of direction.

## Locked interpretation branches

1. Positive modular interval: support the narrow claim, while reporting full-input cost and
   foundation-model limitation.
2. Interval crosses zero: conclude that development evidence did not generalize; present a
   benchmark/null result.
3. Negative interval: reject the modular reliability claim and center failure analysis.
4. Dropout benefit disappears: retain modular-versus-small-VLM comparison but reject the
   robustness-training claim.

No threshold, question family, contrast subset, seed, or subject may be removed after test
evaluation except for a documented execution failure defined before viewing outcomes.

## Explicit limitation

The comparison covers the repository's 14,962-parameter controlled VLM and 4,901-parameter
fixed-slice baseline. It does **not** compare against a large pretrained medical foundation
model. M3D-LaMed was audited but not executed because of compute, custom-code, data-overlap,
license-chain, and four-contrast interface constraints.

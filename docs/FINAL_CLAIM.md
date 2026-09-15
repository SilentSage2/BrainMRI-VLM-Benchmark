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

## Frozen figure and case schemas

Figure 4 retains three panels in this order: (A) full-input balanced QA accuracy for the
question-only, fixed-slice, answer-only, auxiliary, and grounded systems; (B) the 3-by-15
balanced-accuracy condition matrix for the matched 3D systems; and (C) grounded-minus-
comparator raw answer effects with hierarchical 95% intervals. The held-out version may
replace development values only; it may not add, remove, or reorder systems or conditions.

Development Figure 5 retains its modular four-panel audit, including the six profiles
selected before visual review. The final held-out Figure 5 replacement is frozen separately:
(A) full-input versus 14-incomplete-condition balanced QA for grounded VLM, modular
dropout, and modular no-dropout; (B) the same three systems in a 3-by-15 condition matrix;
(C) full-input WT/TC/ET Dice for both modular regimes; and (D) all three paired raw-accuracy
effects with hierarchical 95% intervals. Aggregate success/boundary/failure counts appear
in text/source data only. No test image or subject-specific identifier will be selected or
published after outcomes are viewed.

The machine-readable held-out summary schema is frozen as
`heldout-v1-complete-aggregate-20260914`. It must contain all 15 conditions for every
system, raw and balanced QA, WT/TC/ET Dice, enhancing-fraction error, calibration proxy,
coverage/selective accuracy (including undefined values at zero coverage), the three paired
comparisons with hierarchical intervals, and aggregate profile-bin counts. Ordered subject
identifiers are used only to verify pairing and are not published in the summary.

For held-out aggregate subject profiles, **success** means mean raw QA accuracy of at least
0.80 across the 14 incomplete conditions, **failure** means at most 0.40, and **boundary**
means strictly between 0.40 and 0.80. These are descriptive bins, not exclusions or primary
endpoints. Counts and denominators for all three bins must be reported, including zero
counts. Subjects may not be relabeled or omitted to improve the narrative.

An execution failure is limited to a missing frozen subject, unreadable required file,
checkpoint hash mismatch, non-finite tensor, or process interruption. A process interruption
must resume under the same one-shot lock. Model errors, poor performance, calibration
failure, or inconvenient confidence intervals are scientific outcomes, not execution
failures.

## Explicit limitation

The comparison covers the repository's 14,962-parameter controlled VLM and 4,901-parameter
fixed-slice baseline. It does **not** compare against a large pretrained medical foundation
model. M3D-LaMed was audited but not executed because of compute, custom-code, data-overlap,
license-chain, and four-contrast interface constraints.

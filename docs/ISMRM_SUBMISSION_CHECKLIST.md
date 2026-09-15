# ISMRM 2027 Submission Readiness

Checked: 2026-09-15. Official deadline: 28 October 2026. Internal content freeze:
22 October 2026.

## Complete before held-out access

| Item | Status | Evidence |
|---|---|---|
| Standard abstract structure | Pass | Motivation/Goals/Approach/Results synopsis; Impact; Introduction/Methods/Results/Discussion/Conclusion |
| Format budget | Pass | title 90/125 characters; synopsis 87/100 words; impact 35/40 words; body 589/750 words |
| Figure-caption budget | Pass | five captions, each 301 characters or fewer |
| Development evidence | Pass | 64/16 subjects, three seeds, all 15 contrast subsets, paired hierarchical intervals |
| Matched controls | Pass | question-only, fixed-slice 2D, answer-only, unconditional auxiliary, question-grounded |
| MR comparator | Pass | residual 3D segmentation-to-symbolic QA with dropout/no-dropout ablation |
| Failure analysis | Pass | seed variance, shortcut prior, confidence failure, full-input/dropout tradeoff |
| Main figures | Pass | five visually inspected high-resolution PNGs, vector PDFs, source CSV/JSON |
| Preview figure | Pass | no-caption 1200×1200 PNG; no result values; smartphone-readable text |
| Reproducibility | Pass | versioned commands, frozen configs/checkpoint hashes, clean-install tests |
| Claim discipline | Pass | compact VLM and single-dataset limits; no foundation-model or clinical-use claim |
| Held-out protocol | Pass and complete | One authorized run; 66/66 read; immutable lock complete; frozen schema verified |
| Final held-out figure | Pass | Frozen four-panel Figure 5 rendered to PNG/PDF/CSV and visually inspected |
| Abstract result fields | Pass | No unresolved held-out markers; locked positive branch with mandatory tradeoffs |

## Frozen post-unseal sequence

1. Run the preflight on a clean Git commit and record zero test reads.
2. After explicit authorization, invoke `mri-vlm-heldout-once` exactly once.
3. Verify all 66 subjects, nine checkpoint hashes, three seeds, and 15 conditions.
4. Use only `heldout-v1-complete-aggregate-20260914` fields to fill the two `[HELD-OUT]`
   blocks; preserve the interpretation branch selected by the primary interval.
5. Keep Figure 4 as the development mechanism/ablation result. Generate the frozen held-out
   Figure 5 replacement without changing panels, thresholds, exclusions, or publication
   policy.
6. Rerun abstract lint, post-unseal repository audit, clean-install quality gates, visual QA, and
   co-author review before submission.

## Current status

The single 66-subject held-out evaluation is complete. Its positive primary interval
supports the frozen narrow workflow claim, while lower full-input QA/Dice and negligible
confidence-rule coverage remain mandatory limitations. Automated gates and co-author review
remain before portal submission; no additional experiment is required to manufacture a
more favorable story.

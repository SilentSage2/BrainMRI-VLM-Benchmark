# Matched MRI-VLM V3 Multi-Seed Protocol

Freeze date: 2026-09-14. Status: **completed development experiment; test split sealed**.

V3 tests whether the V2 null/reversed grounding result survives a larger development
cohort and training variation. It does not change the primary comparison after seeing V2.

## Frozen configuration

- Split seed: `20260914`; identical 64 training and 16 validation subjects for every run.
- Training seeds: `20260914`, `20260915`, `20260916`.
- Input: four registered MRI contrasts, foreground crop, `48³`, preserved zero background.
- Training: 12 epochs, width 4, learning rate 0.003, cyclic 15-condition schedule.
- Paths: answer-only, unconditional whole-tumor auxiliary, and question-conditioned evidence.
- QA: the three frozen QA V1 families and frozen enhancing-fraction thresholds.
- Primary effect: full-input grounded-minus-unconditional-auxiliary subject answer accuracy.
- Secondary: grounded-minus-answer-only, all 15 missing-contrast conditions, calibration,
  grounded answer accuracy, and evidence Dice by question family.
- Uncertainty: hierarchical bootstrap that resamples training seeds and subjects within seed.
- Test access: forbidden; every seed artifact and the aggregate require `test_cases_read = 0`.

The run is recoverable at role and seed boundaries. Each role checkpoint includes its
fingerprint, best epoch, history, elapsed time, and completion flag. `progress.json` is
rewritten after each verified seed; a mismatched split seed or subject cohort aborts the
aggregate.

## Predefined interpretation

A positive mechanistic grounding claim requires a grounded-over-auxiliary mean effect of at
least `+0.05`, a hierarchical 95% interval excluding zero, and no material complete-input
calibration harm. This threshold is a development decision rule, not a clinical margin.

If the effect is smaller, crosses zero, or reverses, the project will reject the grounding
benefit claim. The ISMRM storyline will instead report whether generic segmentation
auxiliary supervision is a simpler reliability intervention than question-conditioned
voxel supervision under missing MRI contrasts.

## Baseline controls

- QA V1 question-only prior: 64/16 subjects, answer accuracy 0.521, balanced accuracy
  0.417, subject-bootstrap 95% CI for raw accuracy [0.417, 0.625].
- Fixed-policy 2D slice VLM: three axial slices at 25%, 50%, and 75% of preprocessed depth,
  identical QA/split/modality schedule. Its completed full-input result was 0.542 raw and
  0.417 balanced accuracy (best epoch 12; 4,901 parameters; 14.9 seconds). It failed to
  improve balanced accuracy over the question-only prior, suggesting shortcut-dominated
  behavior rather than useful visual reasoning.
- Residual 3D segmentation-to-symbolic baseline: retained from the earlier direction gate.

The question-only result confirms substantial fixed-template prior signal. Therefore raw
accuracy alone is not an acceptable VLM claim; balanced per-family accuracy and comparison
against this control remain mandatory.

Artifact provenance (artifacts are intentionally not committed): QA V1 control result
SHA-256 `9a1dd175df908b3424607fd869e9502909476541f2419dcdd36a59f6d9974826`;
slice VLM result SHA-256
`692891136e6ad74879147f5342932761477538398aa5fa6a0b849603ca3faee3`.

## Completed V3 result

| Full-input system | Mean raw accuracy | Mean balanced accuracy | Mean ECE |
|---|---:|---:|---:|
| Answer-only 3D | 0.535 | 0.417 | 0.086 |
| Unconditional auxiliary 3D | 0.472 | 0.422 | 0.093 |
| Question-grounded 3D | 0.479 | 0.426 | 0.124 |

The primary grounded-minus-auxiliary raw answer effect was `+0.007`; the three seed effects
were `-0.208`, `+0.229`, and `0.000`, and the hierarchical seed/subject bootstrap 95% CI was
`[-0.229, +0.243]`. Grounded-minus-answer-only was `-0.056`, 95% CI
`[-0.236, +0.063]`. Neither comparison meets the predefined `+0.05` effect and interval
criterion. Grounding also had worse calibration and lower mean grounded answer accuracy
than the auxiliary path (0.174 versus 0.222).

Across the 15 contrast conditions, mean balanced accuracy ranged 0.417–0.417 for
answer-only, 0.410–0.433 for auxiliary, and 0.394–0.433 for grounded. The small spread and
the constant answer-only result do not demonstrate contrast-sensitive visual reasoning.
Question-conditioned evidence was family-specific (Dice: laterality 0.530, relative volume
0.361, enhancing fraction 0.241), but this did not translate into a reliable answer benefit;
unconditional whole-tumor auxiliary Dice was 0.428.

Total per-seed wall time was 513.0, 399.2, and 314.7 seconds on four CPU threads. Result
summary SHA-256:
`530dc2bc9c6ce29338a8b979ea4766e095748f65a86708aef1c4f5a4827c314d`.

**Decision:** reject the positive grounding-benefit hypothesis for this development
configuration. The defensible storyline is a reliability/benchmark result: small MRI VLMs
can match strong language priors while question-conditioned voxel supervision shows high
seed variance and no advantage over ordinary spatial auxiliary supervision. The held-out
test and ISMRM claim gates remain closed pending a stronger external model and frozen
failure analysis.

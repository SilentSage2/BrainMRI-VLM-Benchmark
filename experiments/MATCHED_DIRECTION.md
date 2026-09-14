# Matched MRI-VLM Direction Runs

Run date: 2026-09-14. Status: **completed single-seed development diagnostic; suitable for
an explicitly labeled research-preview repository, not for an abstract performance claim.**

## Frozen design

- Same 32 training and 8 validation subjects as the residual 3D MR baseline.
- Test cases read: zero.
- Same `64³` foreground-crop preprocessing and cyclic 15-condition modality schedule.
- Same coordinate-aware hierarchical 3D visual encoder, GRU word encoder, initialization,
  optimizer steps, QA V1 targets, and answer class weights for all three paths.
- 15 epochs, learning rate 0.003, 14,962 parameters, CPU only.
- Best epoch selected by full-input validation balanced answer accuracy; ties select the
  later epoch identically for every role.

## V1 diagnostic and rejected interpretation

The first completed run accidentally used PyTorch weighted mean cross-entropy with batch
size one, which cancels the class weight. Grounded raw answer accuracy appeared 0.167 above
the comparators, but balanced accuracy showed no grounded-over-auxiliary advantage and
question-specific evidence was worse. This run is retained as a loss-contract failure.

## Corrected V2 results

| Full-input path | Answer accuracy | Balanced answer accuracy | Grounded answer accuracy | Mean evidence Dice |
|---|---:|---:|---:|---:|
| Answer only | 0.375 | 0.444 | — | — |
| Unconditional whole-tumor auxiliary | 0.542 | 0.472 | 0.417 | 0.638 |
| Question-conditioned grounding | 0.542 | 0.472 | 0.167 | 0.486 |

The paired grounded-minus-answer-only answer effect was +0.167 with subject-bootstrap 95%
CI [-0.042, 0.375]. Grounded-minus-unconditional-auxiliary answer effect was exactly 0.000
for every validation subject. Thus V2 provides **no evidence that question-conditioned
grounding improves answers beyond generic spatial supervision**. Its question-specific
evidence was also worse than unconditional whole-tumor evidence.

Across all 15 conditions, answer-only balanced accuracy was constant at 0.444, indicating
that it did not use contrast availability. Auxiliary balanced accuracy ranged 0.444–0.611;
grounded ranged 0.389–0.500. FLAIR-only evidence Dice was 0.704 auxiliary versus 0.574
grounded; T1-only was 0.121 versus 0.108. These are failure-analysis directions at `n=8`,
not MR sensitivity claims.

Training time was 198.9/201.6/211.4 seconds for answer-only/auxiliary/grounded; total wall
time was 741.4 seconds on a four-thread CPU execution. Accelerator memory is not applicable.

## Provenance

- Result SHA-256: `b2a4292b0a55f5152d16133043ece580ea2a3ea94203ad5a6c489cc35883efb6`.
- Run fingerprint: `8b3e1ad908b99f0b89549d627c852433b2f601459a2f7bbd60f8bf320023b5ac`.
- Checkpoint SHA-256: answer-only
  `70f66f462c1b72871d90fdec03d377992ef9e7f27556eab6f978acb5fad0c2ee`, auxiliary
  `056b9bc451404786080bc194fc6a3a97f9e010c85afdfff8e24ef08a93271cc5`, grounded
  `ae8061ddd8b908843f5d9ca29f7b8dceb14dab974e6ca2e9d3464832002a443c`.

## Decision

Do not claim a grounding benefit and do not open the ISMRM result gate. The next run must
use a larger cohort and multiple seeds. If the null/reversed effect persists, the defensible
story is that ordinary segmentation auxiliary supervision is a stronger and simpler source
of spatial reliability than a question-conditioned evidence head in this setting.

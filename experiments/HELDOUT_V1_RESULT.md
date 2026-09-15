# Held-Out V1 One-Shot Result

Run date: 2026-09-15. Status: **completed once after explicit authorization**.

## Audit identity

- Pre-unseal commit: `6ae522c898db807d7084da398189f3d29ab83ba5`.
- Protocol: `heldout-v1-frozen-20260914`.
- Result schema: `heldout-v1-complete-aggregate-20260914`.
- Frozen test manifest: 66 subjects; completed lock: 66 test cases read.
- Systems: three question-grounded VLM seeds, three modular-dropout seeds, and three
  modular-no-dropout seeds, each evaluated across all 15 non-empty contrast subsets.
- Inference: paired raw accuracy averaged over 14 incomplete conditions; hierarchical
  bootstrap over seeds and subjects, 10,000 draws with frozen seed 20260914.
- Execution: exit code 0; no interruption, resume, exclusion, threshold change, or rerun.
- Aggregate summary SHA-256: `ba4c332d0acbf70b53f49097a22b9e0acb526a308f013e276dcb142fea578ec4`;
  one-shot lock SHA-256: `d99b71ace8540cd4fc52a70df17257f51e15be6d5284f74a140cfed0b85c4431`.

MRI data, checkpoints, subject identifiers, per-subject predictions, the authorization
token, and generated result artifacts are intentionally not tracked. The table below is an
aggregate transcription of the locked local summary.

## Results

| System | Full raw QA | Full balanced QA | Incomplete raw QA | Incomplete balanced QA |
|---|---:|---:|---:|---:|
| Question-grounded small VLM | 0.451 | 0.420 | 0.449 | 0.418 |
| Modular dropout | 0.685 | 0.610 | 0.635 | 0.577 |
| Modular no-dropout | 0.749 | 0.701 | 0.581 | 0.561 |

| Frozen paired comparison on incomplete conditions | Mean raw effect | Hierarchical 95% CI |
|---|---:|---:|
| Modular dropout − grounded VLM (primary) | +0.187 | [+0.131, +0.257] |
| Modular no-dropout − grounded VLM | +0.132 | [+0.059, +0.224] |
| Modular dropout − no-dropout | +0.055 | [+0.025, +0.085] |

| Modular regime | Full-input WT Dice | TC Dice | ET Dice | Full ECE | Incomplete ECE |
|---|---:|---:|---:|---:|---:|
| Dropout | 0.718 | 0.607 | 0.486 | 0.083 | 0.081 |
| No-dropout | 0.818 | 0.717 | 0.653 | 0.052 | 0.150 |

At the frozen confidence threshold, modular-dropout coverage was 0.005 on complete input
and 0.003 averaged across incomplete conditions. Selective accuracy is consequently not
interpretable despite numerically high values on the tiny selected subset. Frozen
subject-profile counts were 10/66 success (mean raw incomplete-condition accuracy ≥0.80),
51/66 boundary (>0.40 and <0.80), and 5/66 failure (≤0.40).

## Locked interpretation

The primary interval excludes zero in the positive direction, selecting frozen branch 1
and supporting the narrow claim that the MR-specialized modular pipeline is more reliable
under missing contrasts than the tested small grounded VLM. The positive dropout-minus-
no-dropout interval also supports the prespecified missing-contrast robustness subclaim.

The benefit is not uniform: dropout reduces complete-input balanced QA by 0.091 and
WT/TC/ET Dice by 0.100/0.110/0.167 relative to no-dropout. This is a robustness–complete-
input-quality tradeoff, not evidence that dropout or modular reasoning is universally
better. The study did not execute a large pretrained medical foundation VLM and does not
establish clinical utility, external-dataset generalization, or calibrated abstention.

## Figure integrity

The final held-out Figure 5 was produced only by the frozen renderer and visually checked.

| Artifact | SHA-256 |
|---|---|
| PNG | `27db6347a59798097ac8d66e125a032d9c599082942c275900f213f5b25a3740` |
| PDF | `a0187f4e6d35297bc3805b5290fceeb4574094aec61e93ee67f285372f7b75dd` |
| Source CSV | `2eba920217c9b6b915c4548f0a324d955354666193aedb7676fdbbd506a26728` |

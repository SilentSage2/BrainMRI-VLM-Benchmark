# MRI-VLM Grounding Lab

**Robust 3D Vision-Language Reasoning over Multi-Sequence Brain MRI**

> **Status: public research preview; held-out evaluation remains sealed.** A three-seed
> 64/16-subject development experiment rejects a reliable grounding benefit in the tested
> small VLMs and finds the MR-specialized modular pathway substantially stronger. The
> external M3D-LaMed baseline has been audited but not executed. Claims, checkpoints,
> metrics, and the single-use 66-subject test protocol are frozen pending explicit
> authorization.

## Research question

When one or more complementary brain MRI contrasts are unavailable, does explicit
voxel-level evidence supervision improve the reliability of quantitative interpretation—
including grounded answers, calibration, and abstention—relative to answer-only modeling
and a strong segmentation-to-symbolic MR workflow under matched evaluation?

Inputs are FLAIR, T1-weighted, post-contrast T1-weighted, and T2-weighted brain MRI.
Questions test verifiable properties such as affected region, relative lesion volume,
enhancing-component presence, and changes after a controlled counterfactual. The model
must return both an answer and the voxel region supporting it.

`Grounding` in the repository name means that an answer is tied to an explicit spatial MRI
region rather than evaluated as text classification alone. The current reliability study
also tests whether that additional grounding machinery is justified against a simpler
segmentation-to-symbolic workflow; the name identifies the hypothesis under audit, not a
promise that grounding will win.

## Why this is a VLM project

The learned system contains a multi-sequence 3D visual encoder, a language-conditioned
fusion module, and an answer decoder with an evidence-grounding head. Segmentation masks
provide auditable supervision and question generation; segmentation alone is not the
headline task.

```text
FLAIR / T1 / T1-Gd / T2 volumes -> 3D visual tokens --+
                                                       +-> language-conditioned fusion
question text ------------------> text tokens ---------+          |
                                                                  +-> answer
                                                                  +-> voxel evidence
```

## Main hypothesis

Answer-plus-evidence training with modality dropout will outperform answer-only training
on **grounded answer accuracy** and **unanswerable hallucination rate** when one or more MRI
sequences are missing. Ordinary answer accuracy alone cannot establish the claim.

The intended audience is MR scientists working with retrospective or heterogeneous
multi-contrast datasets. The project does not argue that a VLM is inherently preferable:
the learned language-conditioned system must outperform or reveal information beyond a
strong modular segmentation-to-symbolic pipeline to justify its added complexity.

## Data direction

The initial candidate is Medical Segmentation Decathlon `Task01_BrainTumour`, which has
484 labeled training volumes across four MRI sequences and is published under CC BY-SA
4.0. Deterministic questions and evidence targets are derived from masks and image
geometry. They are synthetic research annotations, not radiologist reports.

See [the dataset decision](docs/decisions/0001-msd-brain-tumour.md).

The frozen protocol is documented in the [experiment specification](docs/EXPERIMENT_SPEC.md)
and [question taxonomy](docs/QUESTION_TAXONOMY.md). GitHub publication follows the
[release checklist](docs/RELEASE_CHECKLIST.md) after a VLM baseline exists.
The [V1 model decision](docs/MODEL_SELECTION.md) selects M3D-LaMed-Phi-3-4B as a
single-volume external baseline and defines `MRI-VLM-Small` for the matched multi-sequence
answer-only versus grounded comparison.
The [ISMRM 2027 abstract plan](docs/ISMRM_2027_ABSTRACT_PLAN.md) freezes the MR-specific
storyline, primary endpoint, figure plan, and submission decision gate.
The [figure contract](docs/FIGURE_PLAN.md) assigns each submission figure a claim, required
inputs, delivery date, and negative-result-safe fallback.
Working [standalone captions](docs/FIGURE_CAPTIONS.md) explicitly separate completed
descriptive/method figures from planned empirical results.
The working [ISMRM abstract package](docs/ISMRM_2027_ABSTRACT_DRAFT.md) and
[proceedings-format audit](docs/ISMRM_PROCEEDINGS_NOTES.md) keep text, figures, and frozen
experimental facts synchronized.
The [research-substance audit](docs/RESEARCH_QUALITY_AUDIT.md) records current failures,
mandatory controls, evidence gates, and the minimum defensible pivot.
The first [real-data control ledger](experiments/REAL_CONTROLS.md) reports a 65.9%
question-only validation accuracy and rejects the initial fixed-template distribution as
the final benchmark because of shortcut leakage.
The [first controlled real-MRI pilot](experiments/REAL_PILOT_V0.md) executes four model
paths but demonstrates underfitting and no grounded-over-auxiliary advantage; its results
are diagnostic and explicitly excluded from abstract claims.
The [preprocessing and one-case optimization gates](experiments/OPTIMIZATION_GATES.md)
verify cache identity and VLM learnability but expose unstable threshold-derived symbolic
answers despite high segmentation Dice; the gate remains failed pending QA V1.
The [development-only QA resolution audit](experiments/QA_RESOLUTION_STABILITY.md) quantifies
this failure across 418 train/validation subjects, identifies a duplicated comparison task,
and keeps all test cases unread while QA V1 is redesigned.
The [residual 3D missing-contrast baseline direction runs](experiments/STRONG_BASELINE_DIRECTION.md)
establish a standard BraTS WT/TC/ET training path and pass a small 32/8-subject development
gate, while explicitly withholding any generalization, Figure 4, or release claim.
The [balanced QA V1 protocol](experiments/QA_V1_PROTOCOL.md) freezes three nonredundant
question families with distinct evidence targets and a matched answer-only/auxiliary/
grounded comparison contract.
The [matched-run recovery ledger](experiments/MATCHED_RECOVERY.md) records the host-restart
interruption, rejects an unverifiable partial artifact, and defines fingerprint-checked
per-role reuse without touching the test split.
The [matched direction-run ledger](experiments/MATCHED_DIRECTION.md) reports the corrected
real-data result: question-conditioned grounding matched, but did not improve upon,
unconditional spatial supervision and produced worse evidence localization at `n=8`.

## Preliminary development diagnostic

These numbers are validation diagnostics from one seed and eight subjects. They are shown
to make the current negative evidence auditable, not to claim generalization.

| Path | Answer accuracy | Balanced answer accuracy | Grounded answer accuracy | Mean evidence Dice |
|---|---:|---:|---:|---:|
| Answer only | 0.375 | 0.444 | — | — |
| Unconditional whole-tumor auxiliary | 0.542 | 0.472 | 0.417 | 0.638 |
| Question-conditioned grounding | 0.542 | 0.472 | 0.167 | 0.486 |

Grounded-minus-auxiliary paired answer effect was 0.000 for every subject. Grounded-minus-
answer-only was +0.167, bootstrap 95% CI [-0.042, 0.375]. Test cases remain sealed.

The larger frozen V3 development run used 64/16 subjects and three seeds. Its
grounded-minus-auxiliary answer effect was +0.007 with hierarchical 95% CI
[-0.229, 0.243]; seed effects were -0.208, +0.229, and 0.000. This fails the predefined
grounding-benefit gate and exposes substantial training variance. The test split remains
sealed, so this is not a generalization claim.

The modular V4 experiment used the same subjects and seeds. Complete-input-only residual
3D segmentation followed by symbolic QA reached 0.787 full-input and 0.578 missing-contrast
balanced accuracy, versus 0.426 and 0.424 for the grounded VLM. Balanced modality dropout
improved missing-contrast raw subject accuracy by +0.069, hierarchical 95% CI
[+0.016, +0.124], while sacrificing complete-input performance. These remain development
results, not held-out claims.

## Planned systems

| System | Visual representation | Training target | Status |
|---|---|---|---|
| Question-only prior | none | answer | QA V1: 0.521 raw / 0.417 balanced validation accuracy |
| Slice-based VLM | fixed axial quartile slices | answer | 0.542 raw / 0.417 balanced; no gain over question-only |
| Residual 3D MR segmenter | four registered contrasts | WT/TC/ET masks → symbolic QA | Three-seed V4 complete; primary MR baseline |
| 3D MRI-VLM | coordinate-aware hierarchical 3D tokens | answer | Small real-data run completed |
| Auxiliary 3D MRI-VLM | same | answer + unconditional whole tumor | Small real-data run completed |
| Grounded 3D MRI-VLM | same + GRU question tokens | answer + question-specific evidence | Small real-data run completed; no benefit observed |

Primary metrics are answer accuracy, grounded answer accuracy, evidence Dice, numeric
tolerance accuracy, counterfactual consistency, calibration, and hallucination on
unanswerable questions. Every system is evaluated across the same missing-sequence matrix.

The frozen [V3 multi-seed protocol](experiments/MATCHED_V3_PROTOCOL.md) expands the matched
comparison to 64/16 subjects and three training seeds, adds hierarchical subject/seed
bootstrap uncertainty, and predefines a negative-result interpretation before aggregation.
The frozen [modular V4 protocol](experiments/MODULAR_V4_PROTOCOL.md) now evaluates whether
MR-specialized 3D segmentation followed by symbolic QA is more reliable, including a
matched modality-dropout ablation and all 15 missing-contrast conditions.
The [external foundation baseline audit](docs/EXTERNAL_BASELINE_AUDIT.md) records why
M3D-LaMed-Phi-3-4B is not responsibly runnable on the current CPU host, including model
size, custom-code, license-chain, data-overlap, and four-contrast compatibility constraints.
The [frozen narrow claim](docs/FINAL_CLAIM.md) and
[one-shot held-out runbook](docs/HELD_OUT_EXECUTION.md) lock the 66-subject test manifest,
checkpoint hashes, metrics, exclusions, bootstrap, interpretation branches, and anti-rerun
guard. Held-out execution still requires new explicit user authorization.

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/ruff check .
.venv/bin/mypy src tests
.venv/bin/pytest
.venv/bin/mri-vlm-v0-audit
```

V0 downloads no medical data. This is a retrospective research benchmark, not a medical
device or clinical decision-support system.

After obtaining MSD Task01 locally and installing `.[data]`, audit it without producing
derivatives:

```bash
mri-vlm-msd-audit /absolute/path/to/Task01_BrainTumour
```

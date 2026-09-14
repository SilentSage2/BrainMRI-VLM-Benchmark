# MRI-VLM Grounding Lab

**Robust 3D Vision-Language Reasoning over Multi-Sequence Brain MRI**

> **Status: audited research scaffold, not publish-ready.** `MRI-VLM-Small` is a controlled
> proof-of-pipeline tested on synthetic tensors; it is not a competitive trained VLM and no
> performance or novelty result is claimed. The public-release and ISMRM gates remain closed
> pending strong baselines, isolation ablations, real-data results, uncertainty, and failure
> analysis.

## Research question

Under matched data and compute, does explicit voxel-level evidence supervision improve a
3D MRI vision-language model's grounded question answering, counterfactual consistency,
and abstention under missing MRI sequences compared with answer-only fine-tuning?

Inputs are FLAIR, T1-weighted, post-contrast T1-weighted, and T2-weighted brain MRI.
Questions test verifiable properties such as affected region, relative lesion volume,
enhancing-component presence, and changes after a controlled counterfactual. The model
must return both an answer and the voxel region supporting it.

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

## Planned systems

| System | Visual representation | Training target | Status |
|---|---|---|---|
| Question-only prior | none | answer | Planned shortcut control |
| Slice-based VLM | sampled 2D slices | answer | Planned baseline |
| 3D MRI-VLM | pooled 3D tokens | answer | Planned baseline |
| Grounded 3D MRI-VLM | sequence-aware 3D tokens | answer + voxel evidence | Architecture smoke implemented |

Primary metrics are answer accuracy, grounded answer accuracy, evidence Dice, numeric
tolerance accuracy, counterfactual consistency, calibration, and hallucination on
unanswerable questions. Every system is evaluated across the same missing-sequence matrix.

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

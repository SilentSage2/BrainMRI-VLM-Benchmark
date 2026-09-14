# V0–V1 MRI-VLM Experiment Specification

## Falsifiable hypothesis

With the base 3D encoder, language decoder, subjects, QA examples, optimizer steps, and
parameter budget held fixed, adding voxel-evidence supervision and balanced modality
dropout will improve grounded answer accuracy by at least 0.05 absolute and reduce
unanswerable hallucination by at least 0.05 versus answer-only fine-tuning under missing
MRI sequences. Complete-input answer accuracy must remain within 0.01 absolute.

## Tasks

Mask- and geometry-derived questions initially cover presence, laterality, relative volume,
enhancing-component proportion, cross-region comparison, and deliberately unanswerable
requests. Numeric answers have fixed tolerances. Every answerable example includes one or
more voxel evidence sets; unanswerable examples include neither answer nor evidence.

Counterfactuals make one controlled change to a mask-derived property or modality input.
Originals and variants share a subject group and split.

## Comparisons

1. Strong missing-modality 3D segmentation followed by deterministic symbolic QA: the
   principal MR workflow baseline.
2. Answer-only 3D MRI-VLM.
3. The same 3D MRI-VLM with an evidence head and balanced modality dropout.
4. The same 3D model with unconditional segmentation auxiliary supervision, isolating
   generic spatial multi-task supervision from question-conditioned grounding.
5. Question-only majority/template control for language shortcut diagnosis.
6. Slice-based VLM using a fixed slice-selection policy for volumetric-information loss.

Ablations remove evidence loss, modality dropout, sequence identity embeddings, and
question conditioning of the spatial head. M3D-LaMed remains a non-matched contextual
baseline because it does not receive the same four-sequence input.

## Evaluation

- normalized categorical and numeric-tolerance answer accuracy;
- voxel evidence Dice;
- grounded answer accuracy, requiring answer and evidence correctness;
- unanswerable hallucination rate;
- pairwise counterfactual consistency;
- expected calibration error and selective accuracy;
- degradation across complete, single-missing, paired-input, and single-input conditions.
- question-family-by-dropped-contrast changes in accuracy, evidence Dice, and confidence,
  compared with preregistered contrast dependencies.

ISMRM reporting order is fixed: all 15 contrast conditions and the modular MR baseline;
calibration/abstention and spatial evidence; matched grounding effect and isolation
ablations; then shortcut and 2D diagnostics. Model novelty is not an endpoint.

Confidence intervals resample subjects, not questions or slices. Final stochastic models
use three seeds when the compute ceiling permits.

## Compute ceiling

V0 tests run on CPU in under one minute. V1 uses one accessible GPU, parameter-efficient
tuning, bounded 3D crops, and at most 24 GPU-hours per final seed. The final answer-only
versus grounded comparison targets at most 144 GPU-hours total.

## Acceptance gates

V0 requires executable schemas, subject isolation, duplicate audits, a frozen question
taxonomy, hand-verified answer/evidence/hallucination metrics, and an approved dataset
decision. V1 additionally requires a question-only control, matched VLM comparisons,
every missingness slice, three-seed or bootstrap uncertainty, qualitative evidence
overlays, compute disclosure, and failure analysis.

The ISMRM-facing narrative, figure plan, and October 2026 decision gates are frozen in
`docs/ISMRM_2027_ABSTRACT_PLAN.md`.

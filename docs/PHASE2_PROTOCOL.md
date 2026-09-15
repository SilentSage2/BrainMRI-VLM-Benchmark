# Phase II Credible-Baseline Protocol

Status: **design gate opened 2026-09-15; no Phase II model outcome inspected**.

## Why Phase II is necessary

Held-out V1 establishes only that a compact, task-specific segmentation-to-symbolic model
outperformed this repository's randomly initialized 14,962-parameter language-conditioned
3D network on segmentation-derived QA. It does not identify a general advantage over
pretrained medical VLMs. The modular model was also compact, trained on 64 subjects at
`48³`, and was not a mature nnU-Net baseline. Because the questions were deterministically
derived from tumor masks, the evaluation favored a pipeline whose intermediate output was
that same mask.

Phase II must improve both sides of the comparison and separate segmentation-derived QA
from genuinely language-dependent evaluation. The existing 66-subject set is retired from
all model selection and may not be called untouched held-out data for Phase II.

## Falsifiable questions

1. Does a standard nnU-Net v2 configuration materially improve WT/TC/ET segmentation over
   the compact residual 3D model on the unchanged 81-subject development partition?
2. Does a locally executed pretrained 3D medical vision encoder/VLM use MRI information
   beyond a question-only prior under matched missing-contrast inputs?
3. After controlling model capacity, pretraining, inputs, and optimization, does
   question-conditioned voxel grounding improve balanced answer accuracy or evidence
   localization over answer-only and unconditional-spatial-auxiliary controls?
4. Do any observed effects generalize to a new external cohort or blind evaluation set?

Question 4 is a release blocker. The Phase-I 66 subjects can be reported as historical
evidence only, never reused as the final Phase-II test set.

## Dataset boundaries

- MSD Task01 subjects assigned by seed `20260914`: 337 training, 81 development, and 66
  retired Phase-I test subjects.
- nnU-Net planning, training, checkpoint selection, and ablation decisions may use only
  the 337/81 partitions. The exporter must reject every identifier in the retired manifest.
- Segmentation-derived QA remains a diagnostic track and must be named as such.
- A primary VLM claim requires a second track whose target is not deterministically
  recoverable from the reference segmentation mask alone: report concepts, expert-written
  questions, or another independently sourced language target.
- A new external cohort or blind server must be selected, licensed, and frozen before final
  Phase-II inference.

## Segmentation track

The reference implementation is nnU-Net v2 with four channels and hierarchical BraTS
regions WT/TC/ET. The sequence is:

1. lossless channel export and affine/header verification;
2. nnU-Net dataset integrity check and experiment planning;
3. local 2D and 3D five-epoch benchmarks on a development-only fixture;
4. one GPU `3d_fullres` fold to measure wall time, VRAM, and validation Dice;
5. expand to the remaining folds only if the measured cost and first-fold quality justify
   it; and
6. compare against the compact residual model on identical subjects and endpoints.

The primary segmentation metrics are subject-level WT/TC/ET Dice with bootstrap intervals.
HD95, enhancing-lesion detection sensitivity, lesion-wise false positives, inference time,
and peak memory are mandatory secondary outcomes. A segmentation baseline is not called
strong merely because symbolic QA is high.

## Pretrained VLM track

Candidate families currently include M3D-LaMed/M3D-CLIP and Med3DVLM. A candidate is
eligible only after recording the exact code and weight revisions, licenses, training-data
overlap risk, input geometry, memory estimate, and whether four registered MRI contrasts
can be represented without giving it less information than the modular comparator.

The preferred compute-bounded design freezes a pretrained 3D vision encoder and trains a
small multi-contrast adapter plus task heads. A full 4B language model is an external
inference baseline, not the only Phase-II path. No candidate is described as executed until
its local checkpoint hash and run artifact exist.

Required controls remain question-only, image-only where meaningful, answer-only fusion,
unconditional spatial auxiliary, and question-conditioned grounding. Balanced accuracy is
primary for categorical QA; raw accuracy is secondary. A credible visual-dependence result
requires improvement over question-only with a paired interval excluding zero and
contrast-sensitive behavior that follows medically plausible missingness patterns.

## Compute plan

The current host is an Apple M4 MacBook Air with 24 GB unified memory. The installed
PyTorch build exposes neither CUDA nor MPS. It is suitable for conversion, integrity tests,
planning, CPU inference checks, and bounded benchmark runs, but not a standard five-fold
3D full-resolution training campaign.

- Local budget: at most a five-epoch benchmark or a small fixture; stop if projected
  full-fold time is impractical.
- GPU gate: one NVIDIA GPU with at least 10 GB VRAM is the nnU-Net minimum; target 16–24 GB
  for this four-channel task.
- Scale gate: measure one fold before authorizing additional folds. Record epoch time,
  maximum VRAM, total wall time, storage, and estimated cost.
- VLM gate: start with frozen-encoder inference/linear probing. Full fine-tuning of a 4B
  model requires a separate quantified GPU and storage budget.

## Phase II acceptance criteria

Before a new abstract claim or test run:

1. nnU-Net conversion and subject-exclusion tests pass from a clean environment.
2. Both model families use all authorized development data and a documented compute budget.
3. The VLM is genuinely pretrained and its visual dependence exceeds the question-only
   control under the frozen balanced endpoint.
4. Segmentation quality is independently assessed rather than inferred from symbolic QA.
5. The primary language target is not a deterministic restatement of the segmentation
   labels.
6. Model selection is complete before the new external/blind test set is opened.
7. Negative, null, and capacity-limited outcomes retain the same metrics and panels.

Until these conditions hold, Phase II is an engineering study, not an ISMRM-ready
VLM-versus-segmentation comparison.

# ISMRM 2027 Abstract Plan

## Target and decision rule

- Target: ISMRM 2027 Annual Meeting, Vancouver, 8–13 May 2027.
- Submission window currently advertised: 1–28 October 2026.
- Primary plan: submit a standard scientific abstract with completed results.
- Go/no-go date: 10 October 2026. Continue with the standard abstract only if the
  locked validation pipeline has produced the matched answer-only versus grounded
  comparison, missing-contrast results, uncertainty estimates, and auditable overlays.
- Fallback: use the registered-abstract route only if the 2027 call explicitly offers it
  and the study meets that route's requirements. Do not submit aspirational results in a
  standard abstract.

The current meeting dates and submission window come from the official 2027 meeting
pages. The section limits below use the official 2026 standard-abstract guidance as a
working constraint until the accessible 2027 instructions are published: 125-character
title, 100-word synopsis, 40-word impact statement, 750-word body, and up to five figures.

## Recommended title

**Voxel Grounding Improves Contrast-Aware Reasoning in 3D MRI Vision-Language Models**

Alternative, if missingness is the strongest result:

**Voxel-Grounded 3D Vision-Language Reasoning under Missing Brain MRI Contrasts**

Both titles intentionally name MRI, 3D, vision-language modeling, and voxel grounding.

## One-sentence story

Explicit voxel-evidence supervision makes a 3D MRI vision-language model less likely to
give unsupported answers when an informative contrast is missing and makes its failure
pattern more consistent with the contrast dependence of the question being asked.

## Why this is an MR abstract rather than a generic AI abstract

Multi-contrast MRI provides complementary information, but deployed or retrospective
exams can have missing or unusable sequences. A model can still produce a fluent answer
when the contrast needed to support that answer is absent. This study tests whether
voxel grounding turns that hidden failure into a measurable, contrast-aware behavior.

The headline experiment is therefore not simply “our VLM has higher accuracy.” It is a
predeclared matrix of question types by available MRI contrasts. Removing T1-Gd should
most strongly affect questions about enhancing tumor; removing FLAIR should most strongly
affect edema-related questions. A grounded model should either preserve an answer using
remaining evidence or abstain, rather than answer confidently from language priors.

## Falsifiable hypothesis

With architecture, data, optimization steps, and parameter budget matched, voxel-evidence
supervision plus balanced contrast dropout will:

1. improve grounded answer accuracy by at least 0.05 under missing-contrast conditions;
2. reduce unanswerable hallucination by at least 0.05;
3. keep complete-input answer accuracy within 0.01 of answer-only training; and
4. improve contrast-dependence alignment: the largest leave-one-contrast-out degradation
   should occur for the question families preregistered as dependent on that contrast.

Failure to meet these thresholds is a negative result, not grounds to revise the endpoint.

## Study design

### Data

Use the Medical Segmentation Decathlon Task01 BrainTumour training set: 484 labeled 3D
subjects with co-registered FLAIR, T1, T1-Gd, and T2 volumes. Questions and voxel evidence
are deterministically derived from the segmentation labels and audited image geometry.
The model never receives the reference segmentation as an input.

This is a mask-verifiable quantitative MRI reasoning benchmark, not a diagnostic study.
It does not claim radiologist-authored questions, report generation, clinical utility, or
generalization beyond this public dataset.

### Split and tasks

- Freeze a deterministic subject-level 70/15/15 train/validation/test split.
- Keep every question, paraphrase, crop, and counterfactual for a subject in one split.
- Evaluate presence, laterality, relative volume, enhancing fraction, cross-region
  comparison, and intentionally unanswerable questions.
- Cover all 15 non-empty combinations of the four contrasts.
- Define preregistered question-to-contrast dependencies before training.

### Matched comparisons

1. Question-only majority/template control.
2. Fixed-policy 2D slice VLM.
3. Answer-only 3D MRI-VLM.
4. Matched 3D MRI-VLM with voxel-evidence supervision and balanced contrast dropout.

A public general-purpose 3D medical VLM may be reported as a contextual baseline, but it
must not be presented as a matched comparison if it accepts only one volume rather than
the same four-contrast input.

### Endpoints and statistics

Primary endpoint: subject-aggregated grounded answer accuracy over missing-contrast
conditions. Key secondary endpoints are answer accuracy, evidence Dice, abstention and
hallucination rates, counterfactual consistency, calibration, and complete-input accuracy.

The MR-specific mechanistic endpoint is a **contrast-dependence alignment matrix**:

`question family × dropped contrast -> change in accuracy, evidence Dice, and confidence`

Report subject-level paired bootstrap 95% confidence intervals for matched differences.
Use three seeds where training variance is material. Freeze numerical tolerances,
evidence-Dice thresholds, decoding, and the test set before the final run.

## Planned figures

1. Method diagram: four MRI contrasts, availability mask, 3D encoder, language answer,
   voxel evidence, and abstention path.
2. Question-family by dropped-contrast heatmap showing paired performance changes.
3. Robustness curve from all four contrasts to single-contrast inputs, with confidence
   intervals for answer-only and grounded training.
4. Representative axial/sagittal/coronal evidence overlays including one correct answer,
   one appropriate abstention, and one failure.
5. Calibration or selective-accuracy plot, used only if it adds information beyond the
   main robustness result.

Do not spend a figure on a decorative architecture rendering if the result heatmap needs
the space.

## Abstract skeleton

### Synopsis draft (working, <=100 words)

Vision-language models can generate plausible answers even when an MRI contrast needed
to support the answer is missing. We evaluate matched 3D brain MRI models with and without
voxel-evidence supervision and balanced contrast dropout across every non-empty subset of
FLAIR, T1, T1-Gd, and T2. Mask-derived questions provide verifiable answers and spatial
evidence without exposing masks to the model. The primary endpoint is grounded answer
accuracy under missing contrasts; secondary analyses quantify hallucination, calibration,
counterfactual consistency, and question-specific contrast dependence. The study tests
whether grounding improves robustness while revealing when model behavior depends on the
MRI contrast relevant to each question.

### Impact draft (working, <=40 words)

Voxel-grounded evaluation can reveal whether a 3D MRI vision-language model uses the
contrast needed for its answer and can reduce unsupported responses when sequences are
missing, enabling more transparent assessment before clinical translation.

### Introduction

State the problem in three moves: multi-contrast MRI is complementary; missing contrasts
create a plausible-answer failure mode for VLMs; answer accuracy alone cannot show whether
the model used spatially relevant evidence.

### Methods

Describe the public dataset, subject split, deterministic mask-derived QA, matched model
pair, evidence loss, balanced contrast dropout, all 15 input combinations, preregistered
contrast dependencies, and subject-level paired bootstrap.

### Results

Lead with the primary matched difference and confidence interval, then the hallucination
difference and complete-input non-inferiority check. Next report the contrast-dependence
matrix. Finish with one failure-analysis result. Never use “significant” without a stated
test and uncertainty interval.

### Discussion and conclusion

Interpret improved grounding as controlled evidence of more robust, inspectable model
behavior—not clinical readiness. Discuss mask-derived task circularity, single-dataset
scope, synthetic language, inherited segmentation ontology, and lack of reader study.

## What would make the story publishable

The minimum credible result package is:

- a real-data end-to-end run, not synthetic-only metrics;
- a matched answer-only versus grounded comparison;
- all 15 contrast combinations or a prospectively justified subset;
- subject-level uncertainty estimates;
- a clear T1-Gd/enhancement or FLAIR/edema interaction in the contrast-dependence matrix;
- representative 3D evidence overlays plus honest failure cases; and
- no leakage, no clinical-performance claim, and no retrospective endpoint selection.

If grounding only improves evidence Dice but not unsupported-answer behavior, frame it as
a negative mechanistic result. If every question can be answered from morphology in any
single sequence, the intended contrast-awareness claim is not supported; revise the task
construction before training, not the conclusion after seeing test results.

## Sources to re-check at submission

- ISMRM 2027 meeting: https://www.ismrm.org/27m/
- ISMRM future meetings and submission dates:
  https://www.ismrm.org/meetings-workshops/future-ismrm-meetings/
- Most recent accessible standard-abstract instructions:
  https://www.ismrm.org/26m/call/standard/
- Most recent accessible registered-abstract instructions:
  https://www.ismrm.org/26m/call/registered/
- Medical Segmentation Decathlon: http://medicaldecathlon.com/
- Registry of Open Data on AWS: https://registry.opendata.aws/msd/

The 2027 call supersedes the working limits in this document as soon as it becomes
accessible.

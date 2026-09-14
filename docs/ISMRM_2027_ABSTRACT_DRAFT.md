# Title
Voxel Grounding Improves Contrast-Aware Reasoning in 3D MRI Vision-Language Models

## Synopsis

### Motivation
MRI vision-language models may give plausible answers when a contrast needed to support
the answer is missing.

### Goal(s)
Test whether voxel grounding makes 3D MRI question answering more robust and transparent
under missing contrasts.

### Approach
We compare matched answer-only and voxel-grounded models across every available subset of
FLAIR, T1, T1-Gd, and T2 using mask-verifiable questions.

### Results
[PLANNED — replace with the primary effect, 95% confidence interval, hallucination result,
and complete-input non-inferiority result from the locked table.]

## Impact
[PLANNED — write after results; state what MRI researchers can now evaluate or avoid, name
voxel grounding and missing contrasts, and make no clinical-use claim.]

## Main Body

### Introduction

Multi-contrast brain MRI provides complementary tissue information, but retrospective and
deployed examinations may lack one or more sequences. A vision-language model can still
produce a fluent answer when the contrast needed to support it is absent. Answer accuracy
alone cannot establish whether the response is tied to spatially relevant image evidence.
We test whether explicit voxel-evidence supervision makes 3D MRI question answering more
robust, inspectable, and appropriately uncertain under missing contrasts.

### Methods

The Medical Segmentation Decathlon Task01 BrainTumour dataset contains 484 labeled,
co-registered FLAIR, T1, T1-Gd, and T2 volumes. An integrity audit verified four-channel
shape, finite values, spacing and affine agreement, label range, subject uniqueness, and
fingerprint isolation. A deterministic hash split (seed 20260914) assigned 337/81/66
subjects to training/validation/test sets before generating any examples.

Questions and voxel evidence are deterministically derived from labels and geometry for
presence, laterality, relative volume, enhancing fraction, cross-region comparison, and
unanswerable tasks. Reference masks are never model inputs. The controlled MRI-VLM uses a
shared 3D convolutional encoder, learned sequence identities, availability masking,
masked-pooled question embeddings, question-conditioned fusion, categorical answer logits
including abstention, and voxel-evidence logits.

[PLANNED — train a question-only control, fixed-policy 2D baseline, matched answer-only 3D
model, unconditional segmentation-auxiliary control, question-conditioned grounded model,
and strong segmentation-to-symbolic-QA baseline using frozen preprocessing, examples,
optimizer steps, and compute. Evaluate every one of the 15 non-empty contrast subsets. Run
ablations without evidence loss, balanced contrast dropout, sequence identity, and question
conditioning of the spatial head.]

The primary endpoint is subject-aggregated grounded answer accuracy over missing-contrast
conditions. Secondary endpoints are answer accuracy, evidence Dice, hallucination on
unanswerable questions, counterfactual consistency, calibration, and complete-input
accuracy. Question-family-by-dropped-contrast effects test preregistered contrast
dependencies. Report paired subject bootstrap 95% confidence intervals and three seeds
where training variance is material. The held-out test set is evaluated once after model
and threshold selection.

### Results

[PLANNED — report audited example counts by split and question family; question-only and 2D
baseline results; matched answer-only versus grounded primary effect with 95% CI; all 15
contrast conditions; hallucination and calibration; complete-input non-inferiority;
ablation effects; three-seed dispersion; and the preregistered contrast-dependence test.
Insert only values exported from the locked results manifest.]

### Discussion

[PLANNED — interpret the observed effect direction and uncertainty. Discuss whether
T1-Gd/enhancement and FLAIR/edema sensitivities align with the preregistration. Report
negative and boundary results. Limitations must include one public tumor dataset,
mask-derived synthetic language, historical acquisition protocols, inherited segmentation
ontology, no reader study, and no evidence of clinical utility.]

### Conclusion

[PLANNED — answer the research question directly from the primary endpoint. If thresholds
are not met, state that voxel grounding did not establish improved missing-contrast
robustness and identify the supported negative conclusion.]

### References

1. Simpson AL, et al. A large annotated medical image dataset for the development and
   evaluation of segmentation algorithms. arXiv:1902.09063, 2019.
2. [PLANNED — add the final 3D medical VLM baseline citation after implementation review.]
3. [PLANNED — add directly relevant missing-sequence and grounded MRI references in citation
   order; verify bibliographic fields before submission.]

## Figure Captions

### Figure 1
Framework for voxel-grounded 3D MRI reasoning. A real, preselected validation case shows
co-registered FLAIR, T1, T1-Gd, T2 and the non-input evidence target. The implemented model
combines available contrasts and question tokens to predict answer and evidence logits.
Evaluation covers all 15 contrast subsets; the dashed box marks the pending training runner.

### Figure 2
Audited MSD Task01 cohort (n=484). Panels show the locked 337/81/66 subject split, physical
tumor-subregion volumes, and enhancing-to-whole-tumor fraction. Reference masks define
supervision and evaluation targets but are not model inputs. This is descriptive and makes
no performance or clinical claim.

### Figure 3
[PLANNED — contrast-dependence heatmap caption with split, n, baseline, model, metric,
effect direction, and 95% CI definition.]

### Figure 4
[PLANNED — all-subset robustness caption with split, n, seeds, subject-bootstrap intervals,
and complete-input non-inferiority result.]

### Figure 5
[PLANNED — evidence and failure-analysis caption with frozen success/boundary/failure
selection rules and aggregate context.]

## Preview Figure

[PLANNED — produce a simple no-caption preview derived from Figure 1 at 1200×1200 pixels.
It may contain only large labels for multi-contrast MRI, voxel-grounded VLM, and
answer/evidence/abstain. Verify readability at 360-pixel width and do not include results
until the result is frozen.]

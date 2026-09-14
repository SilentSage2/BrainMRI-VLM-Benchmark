# Title
Reliability of Voxel-Grounded 3D Vision-Language Models across Missing Brain MRI Contrasts

## Synopsis

### Motivation
Retrospective and heterogeneous brain MRI datasets often lack contrasts, while current
missing-modality methods do not verify whether language answers have spatial support.

### Goal(s)
Test whether voxel evidence improves interpretation reliability and abstention across
incomplete multi-contrast MRI.

### Approach
Compare answer-only, voxel-grounded, and segmentation-to-symbolic methods on mask-verifiable
questions across all 15 subsets of FLAIR, T1, T1-Gd, and T2.

### Results
[PLANNED — replace with the primary effect, 95% confidence interval, hallucination result,
and complete-input non-inferiority result from the locked table.]

## Impact
[PLANNED — state whether MR scientists can use voxel evidence and abstention to identify
unsupported interpretations in incomplete multi-contrast datasets, or whether the simpler
modular MR pipeline remains preferable. Make no clinical-use claim.]

## Main Body

### Introduction

Retrospective, multi-center, and protocol-heterogeneous brain MRI collections often lack
one or more complementary contrasts. Existing missing-modality methods primarily optimize
image synthesis or segmentation; they do not establish whether a language-level
interpretation is spatially supported or whether a system abstains when evidence is
insufficient. A model can therefore produce a plausible answer despite removal of the
contrast most relevant to the question. We test whether explicit voxel-evidence supervision
improves quantitative interpretation, calibration, and abstention across incomplete
multi-contrast MRI, and whether it adds value beyond a modular segmentation-to-symbolic MR
workflow.

### Methods

The Medical Segmentation Decathlon Task01 BrainTumour dataset contains 484 labeled,
co-registered FLAIR, T1, T1-Gd, and T2 volumes. An integrity audit verified four-channel
shape, finite values, spacing and affine agreement, label range, subject uniqueness, and
fingerprint isolation. A deterministic hash split (seed 20260914) assigned 337/81/66
subjects to training/validation/test sets before generating any examples.

Questions and voxel evidence are deterministically derived from labels and geometry.
A development-only resampling audit screens unstable boundary targets, removes an
algebraically duplicated comparison, and excludes enhancing presence from aggregate
accuracy because stability screening makes it single-class. Primary retained families
include laterality, enhancing fraction, and contrast-dependent support/abstention.
Reference masks are never model inputs. The controlled MRI-VLM uses a
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
dependencies. The principal MR comparison is against a 3D missing-modality segmentation
model followed by deterministic symbolic answers; language-only and fixed-slice controls
diagnose shortcuts and loss of volumetric information. Report paired subject bootstrap 95%
confidence intervals and three seeds
where training variance is material. The held-out test set is evaluated once after model
and threshold selection. Its aggregate label distribution was previously included in
dataset auditing and descriptive cohort analysis; no test predictions inform selection.

### Results

[DEVELOPMENT EVIDENCE — not yet eligible for the submitted abstract] On a frozen
64-training/16-validation-subject cohort over three seeds, question-conditioned grounding
did not improve full-input raw answer accuracy over matched unconditional spatial auxiliary
supervision (mean paired effect +0.007; hierarchical seed/subject bootstrap 95% CI
[-0.229, 0.243]). The seed effects (-0.208, +0.229, and 0.000) showed material training
variance. Mean balanced accuracy was 0.417 for answer-only, 0.422 for auxiliary, and 0.426
for grounded models; the QA V1 question-only prior and fixed-slice 2D VLM both achieved
0.417. Grounding had worse calibration (ECE 0.124 versus 0.093) and lower grounded answer
accuracy (0.174 versus 0.222) than the auxiliary model. Test cases remained unread.

[PLANNED — lead with all 15 contrast conditions and the segmentation-to-symbolic MR
baseline; then report the matched grounded effect with 95% CI, calibration and abstention,
evidence Dice, complete-input non-inferiority, question-only and 2D diagnostics, ablation
effects, three-seed dispersion, and the preregistered contrast-dependence test.
Insert only values exported from the locked results manifest.]

### Discussion

[PLANNED — interpret what the observed effect means for incomplete MRI analysis, including
whether the added VLM complexity improves on the modular MR baseline. Discuss whether
T1-Gd/enhancement and FLAIR/edema sensitivities align with the preregistration and whether
abstention is calibrated. Report negative and boundary results. Limitations must include one public tumor dataset,
mask-derived synthetic language, historical acquisition protocols, inherited segmentation
ontology, no reader study, and no evidence of clinical utility.]

### Conclusion

[PLANNED — answer whether voxel evidence makes incomplete multi-contrast MRI interpretation
more reliable than answer-only and modular MR alternatives. If thresholds are not met,
state that the added VLM complexity is not justified by this study.]

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
Evaluation covers all 15 contrast subsets; a small pilot runner is implemented while
full-scale training remains pending.

### Figure 2
Audited MSD Task01 cohort (n=484). Panels show the locked 337/81/66 subject split, physical
tumor-subregion volumes, and enhancing-to-whole-tumor fraction. Reference masks define
supervision and evaluation targets but are not model inputs. This is descriptive and makes
no performance or clinical claim.

### Figure 3
Resolution sensitivity of mask-derived QA targets in 418 training/validation subjects;
test cases were unread. Categorical stability increased from 85.6% at 16³ to 97.8% at 48³.
At 32³, train-derived margins made retained validation targets stable but enhancing
presence became single-class; duplicated comparison targets were removed. No model
performance is shown.

### Figure 4
[PLANNED — 15-condition robustness, calibration, abstention, and
question-by-removed-contrast sensitivity caption with support definition, split, n,
intervals, modular baseline, and complete-input non-inferiority result.]

### Figure 5
[PLANNED — paired VLM/modular evidence and failure-analysis caption with frozen
success/boundary/failure selection rules and aggregate context.]

## Preview Figure

[PLANNED — produce a simple no-caption preview derived from Figure 1 at 1200×1200 pixels.
It may contain only large labels for multi-contrast MRI, voxel-grounded VLM, and
answer/evidence/abstain. Verify readability at 360-pixel width and do not include results
until the result is frozen.]

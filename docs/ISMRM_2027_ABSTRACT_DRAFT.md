# Title
Reliability of Voxel-Grounded 3D Vision-Language Models across Missing Brain MRI Contrasts

## Synopsis

### Motivation
Missing MRI contrasts can destabilize automated tumor measurements while vision-language
answers remain plausible without spatial support.

### Goal(s)
Determine whether voxel-grounded 3D VLMs improve quantitative reasoning beyond a modular
MR pipeline under missing contrasts.

### Approach
We compare matched small VLMs with residual 3D segmentation-to-symbolic reasoning across
all 15 subsets of four co-registered brain MRI contrasts.

### Results
On development data, grounding had no reliable benefit; modular reasoning was stronger,
and modality dropout traded complete-input performance for raw missing-contrast accuracy.
On held-out subjects, modular dropout exceeded the grounded VLM by +0.187 raw
missing-contrast accuracy (95% CI +0.131 to +0.257), while reducing full-input performance.

## Impact
This study tests whether voxel grounding adds reliability beyond an established MR
segmentation workflow when contrasts are missing, or whether added VLM complexity is
unjustified, providing an auditable evaluation template rather than a clinical-use claim.

## Main Body

### Introduction

FLAIR, T1, post-contrast T1 (T1-Gd), and T2 provide complementary views of glioma
subregions, but retrospective and heterogeneous MRI collections may lack one or more
contrasts. Missing-sequence methods have primarily targeted segmentation, including
sequence-dropout training.¹ Meanwhile, 3D medical VLMs² and spatially grounded volumetric
MRI question answering³ are emerging, yet a plausible language answer need not be supported
by the available contrasts. We therefore asked whether question-conditioned voxel evidence
improves missing-contrast reasoning beyond matched answer models and a conventional
segmentation-to-symbolic MR workflow.

### Methods

The Medical Segmentation Decathlon Task01 BrainTumour dataset⁴ contains 484 co-registered
four-contrast volumes with tumor labels. A deterministic subject split (seed 20260914)
assigned 337/81/66 cases to training/validation/test before question generation. Labels
produced verifiable laterality, relative edema/core, enhancing-fraction, and
support/abstention targets; they were never model inputs. A development-only resampling
audit removed unstable, duplicated, and single-class targets.

We compared a question-only prior, fixed three-slice 2D VLM, and three parameter-matched
small 3D VLMs: answer-only, unconditional spatial auxiliary, and question-conditioned
voxel-grounded. A residual 3D segmentation model followed by deterministic QA provided the
MR-specialized comparator, trained either across balanced modality subsets or on complete
inputs only. Every system was evaluated on all 15 non-empty subsets of FLAIR, T1, T1-Gd,
and T2. Scores were aggregated by subject. Hierarchical bootstrap intervals resampled three
training seeds and subjects within seed. Primary held-out inference is the paired difference
in raw QA accuracy across 14 incomplete conditions between modular dropout and grounded VLM
(10,000 draws; frozen seed). Balanced QA accuracy, WT/TC/ET Dice, calibration proxy,
coverage, selective accuracy, and complete-input performance are mandatory secondary
outcomes.

### Results

Development experiments used 64 training and 16 validation subjects. Full-input balanced
accuracy was 0.417 for the question-only and 2D controls, 0.417/0.422/0.426 for
answer-only/auxiliary/grounded 3D VLMs, and 0.640/0.787 for modular dropout/no-dropout.
Grounded-minus-auxiliary raw accuracy was +0.007 (hierarchical 95% CI −0.229 to +0.243),
with seed effects −0.208, +0.229, and 0.000; thus question-conditioned grounding did not
meet the frozen benefit criterion. Across incomplete inputs, modular dropout and no-dropout
balanced accuracy was 0.570 and 0.578 versus 0.424 for the grounded VLM. Dropout improved
raw incomplete-condition accuracy by +0.069 (95% CI +0.016 to +0.124) but reduced full-input
WT/TC/ET Dice from 0.799/0.769/0.709 to 0.658/0.579/0.489. Its frozen 0.75 confidence rule
had zero coverage, invalidating the proposed abstention proxy.

On 66 held-out subjects, modular dropout, no-dropout, and grounded VLM achieved raw
incomplete-condition accuracy of 0.635, 0.581, and 0.449 and balanced accuracy of 0.577,
0.561, and 0.418. The primary dropout-minus-grounded effect was +0.187 (95% CI +0.131 to
+0.257); no-dropout-minus-grounded was +0.132 (+0.059 to +0.224), and dropout-minus-no-
dropout was +0.055 (+0.025 to +0.085). Full-input dropout/no-dropout balanced accuracy was
0.610/0.701 and WT/TC/ET Dice was 0.718/0.607/0.486 versus 0.818/0.717/0.653. Dropout
calibration-proxy ECE was 0.083 with full and 0.081 with incomplete input, but frozen-rule
coverage was only 0.005 and 0.003; selective accuracy was therefore not interpretable.
Profiles comprised 10 success, 51 boundary, and 5 failure subjects.

### Discussion

The development and held-out results show that spatial supervision alone does not
establish reliable vision-language reasoning: the controlled VLMs largely matched a strong
question prior, and grounding showed substantial training-seed variance. The modular pathway better
preserved mask-verifiable quantitative answers. Modality dropout improved missing-contrast
raw accuracy, but sacrificed complete-input QA and segmentation quality, and its confidence
rule had negligible coverage. The study therefore evaluates when added
VLM complexity is justified rather than assuming that it is. Limitations include one
historical public tumor dataset, compact non-foundation VLMs, synthetic mask-derived
language, label-derived evidence, no external cohort or reader study, and no clinical-use
evaluation. Aggregate label properties of the frozen test split were audited previously,
although no test predictions informed selection.

### Conclusion

On held-out data, modular MR perception outperformed the tested small grounded VLMs under
missing contrasts, and modality dropout improved raw robustness at a substantial
complete-input cost. This supports the narrow workflow claim, not superiority to medical
foundation VLMs or clinical utility.

### References

1. Feng X, Ghimire K, Kim DD, et al. Brain tumor segmentation for multi-modal MRI with
   missing information. J Digit Imaging. 2023;36:2075-2087.
2. Bai F, Du Y, Huang T, Meng MQH, Zhao B. M3D: advancing 3D medical image analysis with
   multi-modal large language models. arXiv:2404.00578, 2024.
3. Moukheiber L, Yeung CM, Xue H, et al. Beyond a single frame: multi-frame spatially
   grounded reasoning across volumetric MRI. arXiv:2604.15808, 2026.
4. Antonelli M, Reinke A, Bakas S, et al. The Medical Segmentation Decathlon. Nat Commun.
   2022;13:4128.

## Figure Captions

### Figure 1
Comparative framework. (A) Co-registered FLAIR, T1, T1-Gd, and T2 from a validation case
selected by median tumor burden; the label-derived overlay is not a model input. (B)
Matched end-to-end VLM and modular segmentation-to-symbolic pathways enter the same frozen
15-subset, subject-level evaluation.

### Figure 2
Audited MSD cohort (n=484). The deterministic 337/81/66 split, physical tumor-subregion
volumes, and enhancing-to-whole-tumor fraction define the study population and
mask-verifiable targets. This descriptive analysis included aggregate test-label
properties but no model predictions.

### Figure 3
Development-only target audit (n=418; test unread). Resolution affected categorical and
continuous mask-derived answers. Train-derived ambiguity margins stabilized retained
validation targets; single-class and algebraically duplicated targets were excluded before
the final protocol.

### Figure 4
Three-seed development comparison (n=16 validation subjects; test unread). Grounding did
not improve raw answer accuracy over unconditional auxiliary supervision (+0.007, 95% CI
−0.229 to +0.243), and all small VLMs remained near the 0.417 balanced-accuracy language
prior across 15 contrast subsets.

### Figure 5
Frozen held-out comparison (n=66; three seeds). Modular dropout exceeded the grounded VLM
in raw incomplete-condition accuracy by +0.187 (95% CI +0.131 to +0.257) and no-dropout by
+0.055 (+0.025 to +0.085), but reduced full-input balanced QA and WT/TC/ET Dice.

## Preview Figure

The completed no-caption 1200×1200 preview shows the four co-registered MRI contrasts,
“Grounded 3D VLM,” “Modular 3D MR,” and their shared “15-subset reliability” endpoint. It
contains no result values and remains legible when displayed at smartphone width.

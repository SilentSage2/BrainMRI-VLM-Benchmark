# Working Figure Captions

These captions describe only generated evidence. Figure 4 is development-only; Figure 5
is the frozen held-out aggregate and contains no subject identifiers or selected cases.

## Figure 1 — complete overall comparative framework

**Grounded VLM versus modular MR reasoning under missing contrasts.** (A) Co-registered
FLAIR, T1, T1-Gd, and T2 images and reference tumor regions for BRATS_415 at axial index
88. The case was selected before model evaluation as the validation subject closest to the
median whole-tumor volume (82.921 mL), rather than by visual appearance or model outcome.
The reference overlay is label-derived and is not an input. (B) The implemented matched
answer-only, auxiliary, and grounded 3D VLMs are compared with dropout and full-input-only
residual 3D segmentation-to-symbolic paths. Every path enters the same frozen 15-subset
evaluation with subject-level answer accuracy, WT/TC/ET Dice, hierarchical seed/subject
bootstrap, calibration, and failure analysis. This framework figure makes no performance
or clinical claim.

## Figure 2 — complete descriptive cohort

**Audited cohort and mask-derived quantitative targets.** (A) Deterministic subject-level
partition of all 484 labeled MSD Task01 BrainTumour cases using seed 20260914; all derived
examples from one subject retain the same partition. (B) Physical volumes of edema,
non-enhancing tumor, and enhancing tumor computed from reference labels and voxel spacing;
horizontal lines denote medians and “absent” reports zero-volume cases. The vertical axis
is logarithmic and excludes zero values from the violin density. (C) Distribution of the
enhancing-to-whole-tumor volume fraction; the dashed line is the cohort median. Reference
masks define supervision and evaluation targets but are not model inputs. This descriptive
figure does not report model performance or clinical validity.

## Figure 3 — complete development target-validity audit

**Resolution sensitivity of mask-derived QA targets.** Using 418 training/validation
subjects with test cases unread, (A) categorical answer stability and (B) enhancing-fraction
error are shown after nearest-neighbor resampling to 16³–48³. (C) At 32³, margins derived
only from unstable training cases retained 76/81, 81/81, and 73/81 validation cases with
100% stability; enhancing presence became single-class. The two comparison targets were
algebraically redundant. No model performance is shown.

## Figure 4 — complete development matched comparison

**Three-seed development comparison of MRI-VLM supervision under missing contrasts.**
(A) Full-input balanced answer accuracy for the QA V1 question-only prior, fixed three-slice
2D VLM, and matched 3D answer-only, unconditional spatial-auxiliary, and
question-conditioned grounded models on 16 validation subjects. Bars for trained 3D models
are means over three seeds. (B) Mean balanced accuracy across all 15 non-empty subsets of
FLAIR (F), T1, post-contrast T1 (G), and T2. (C) Full-input grounded-minus-comparator raw
answer effects with hierarchical 95% intervals resampling training seeds and subjects
within seed; the dotted line marks the predefined +0.05 development threshold. The
grounded-minus-auxiliary effect was +0.007 [-0.229, 0.243], failing the positive grounding
gate. All results are development-only; test cases were unread.

## Figure 5 — complete held-out modular robustness comparison

**Held-out reliability across missing brain MRI contrasts.** On 66 frozen subjects and
three seeds, (A) full-input and mean 14-incomplete-condition balanced QA compare the tested
grounded VLM with modular dropout/no-dropout; (B) shows all 15 FLAIR (F), T1, post-contrast
T1 (G), and T2 subsets; (C) shows full-input modular WT/TC/ET Dice; and (D) reports every
prespecified raw incomplete-condition paired effect with hierarchical seed/subject 95% CI.
Dropout exceeded grounded by +0.187 [+0.131, +0.257] and no-dropout by +0.055 [+0.025,
+0.085], but reduced full-input QA and Dice.

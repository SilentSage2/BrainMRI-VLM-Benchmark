# Working Figure Captions

These captions describe only generated evidence. Figures 3–5 remain planned and must not
be written as observed results before their locked source tables exist.

## Figure 1 — complete

**Audited cohort and mask-derived quantitative targets.** (A) Deterministic subject-level
partition of all 484 labeled MSD Task01 BrainTumour cases using seed 20260914; all derived
examples from one subject retain the same partition. (B) Physical volumes of edema,
non-enhancing tumor, and enhancing tumor computed from reference labels and voxel spacing;
horizontal lines denote medians and “absent” reports zero-volume cases. The vertical axis
is logarithmic and excludes zero values from the violin density. (C) Distribution of the
enhancing-to-whole-tumor volume fraction; the dashed line is the cohort median. Reference
masks define supervision and evaluation targets but are not model inputs. This descriptive
figure does not report model performance or clinical validity.

## Figure 2 — complete

**Controlled voxel-grounded 3D MRI vision-language model.** (A) Co-registered FLAIR, T1,
T1-Gd, and T2 images and reference tumor regions for BRATS_415 at axial index 88. The case
was selected before model evaluation as the validation subject closest to the median
whole-tumor volume (82.921 mL), rather than by visual appearance or model outcome. The
reference overlay shows the evidence target and is not an input. (B) Implemented forward
path: a shared 3D encoder processes available contrasts; availability masks and learned
sequence identities enter visual fusion with pooled question tokens; matched answer-only
and grounded variants output answer logits, including the abstention class, while the
grounded variant also predicts voxel evidence. This methods figure makes no performance
claim.

## Figures 3–5 — planned

Captions will be generated only from frozen real-data result tables. Each will state the
model versions, evaluation split, subject and question counts, missing-contrast conditions,
baseline or ablation, seed count, interval construction, and the result-supported claim.

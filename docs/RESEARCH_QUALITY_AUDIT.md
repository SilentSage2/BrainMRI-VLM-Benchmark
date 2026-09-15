# Research Substance Audit

Audit date: 2026-09-14. Verdict: **NOT YET PUBLISH-READY**.

Passing tests, complete documentation, and polished figures establish engineering hygiene;
they do not establish a research contribution. This audit remains a release blocker until
real locked experiments satisfy the evidence gates below.

## 1. Importance and falsifiability — conditional pass

Missing MRI contrasts are a substantive problem, but missing-modality brain-tumor
segmentation is already a mature research area. The question is valuable only in its
narrower form: does question-conditioned voxel supervision reduce unsupported answers and
change contrast-specific failure behavior in 3D MRI reasoning?

The main comparison is falsifiable through preregistered effect thresholds, a complete-
input non-inferiority margin, all 15 non-empty contrast subsets, and subject-level
uncertainty. A null or reversed effect is acceptable. The project must say **sensitivity
alignment**, not claim that a model “uses the clinically correct contrast,” because the
reference masks do not prove a unique contrast is necessary for each answer.

## 2. Novelty — at risk, not established

M3D-LaMed already evaluates general 3D medical VQA and segmentation. SGMRI-VQA (2026)
already studies spatially grounded volumetric MRI reasoning with 41,307 QA pairs and ten
VLM baselines. Missing-modality BraTS work already evaluates all modality combinations and
shows subregion-specific degradation.

The defensible contribution is therefore the intersection of:

- co-registered four-contrast 3D MRI rather than single-volume or multi-frame inputs;
- exhaustive 15-subset missing-contrast evaluation;
- question-conditioned voxel evidence and explicit abstention;
- matched models that isolate evidence supervision from architecture and compute; and
- a preregistered question-family-by-contrast sensitivity analysis.

If experiments do not demonstrate information beyond answer accuracy or existing
segmentation robustness, the novelty claim fails and the work should be reported as a
benchmark/negative study, not a new VLM method.

## 3. Data and evaluation validity — partial pass

Strengths: all 484 labeled cases passed image/label integrity checks; the subject split was
locked before derived examples; all sequence subsets are enumerated; metrics and subject
bootstrap contracts exist.

Open risks:

- one historical public tumor dataset cannot establish clinical or cross-site validity;
- synthetic mask-derived questions may contain template shortcuts;
- file hashing does not detect near-duplicate or related scans;
- evidence thresholds and numeric tolerances require validation sensitivity analyses;
- existing unanswerable templates may be solvable from language alone and do not by
  themselves establish missing-contrast uncertainty; and
- a label-derived answer does not prove which MRI contrast was required to obtain it.

Required remedies: question-only performance by template family, near-duplicate image
screening, threshold sensitivity, explicit support-availability definitions, and held-out
test evaluation only after all selection is frozen.

Protocol disclosure: the initial integrity audit and cohort figure inspected aggregate
label properties across all 484 cases, including the locked test split. No test prediction
or per-case model-outcome selection has occurred, but the split is not pristine unseen data.
All future selection must remain validation-only, and this limitation must appear in the
abstract if the same test split is reported.

## 4. Implementation depth — partial pass

The project now includes deterministic real-data preprocessing, QA/evidence materialization,
a memory-bounded cache, serialization recovery, a residual 3D MR segmenter, and matched
answer-only, unconditional-auxiliary, and question-grounded training paths. The corrected
V2 run records seed, split, parameter count, hardware, wall time, fingerprints, and artifact
hashes without reading test cases.

This is substantive implementation progress, but the hierarchical 3D VLM is still a compact
14,962-parameter controlled model with a GRU language encoder. The 32-train/8-validation run
is a direction diagnostic, not a materially sized converged experiment, and no external VLM
has yet been executed. It must not be described as the final flagship model.

## 5. Baselines and ablations — insufficient at present

Minimum credible comparison set:

1. question-only control to measure language/template leakage;
2. fixed-policy 2D slice baseline;
3. answer-only `MRI-VLM-Small` with matched compute;
4. the same model with **unconditional segmentation auxiliary supervision**, isolating
   generic extra spatial supervision from question-conditioned grounding;
5. question-conditioned voxel-grounded `MRI-VLM-Small`;
6. segmentation-to-symbolic-QA factorized baseline, using a strong 3D segmentation model;
7. M3D-LaMed as a clearly labeled, non-matched external 3D medical VLM baseline.

Required ablations remove evidence loss, balanced contrast dropout, sequence identities,
and question conditioning of the spatial head. Equalize examples, optimizer steps, input
subsets, and compute for matched claims. Parameter counts and wall time must be reported.

The unconditional spatial auxiliary is a hard requirement: without it, an improvement
cannot be attributed to grounded reasoning rather than ordinary multi-task segmentation.

## 6. Results needed to support a conclusion — preliminary negative evidence only

Small validation diagnostics now exist, including a 32-train/8-validation residual 3D
baseline and a corrected matched comparison of answer-only, unconditional spatial
auxiliary, and question-conditioned grounding paths. In the matched V2 run, grounded and
unconditional-auxiliary answer accuracy were identical for every validation subject, while
question-conditioned evidence Dice was lower. This is honest negative direction evidence,
not a generalization result: the cohort is eight subjects and one seed.

A subsequent frozen 64-train/16-validation, three-seed experiment strengthened the negative
finding: grounded-minus-unconditional-auxiliary answer accuracy was +0.007 with hierarchical
95% CI [-0.229, 0.243]. On the same cohort, residual 3D segmentation followed by QA reached
0.787 full-input and 0.578 missing-contrast balanced accuracy, compared with 0.426 and 0.424
for the grounded VLM. Balanced modality dropout improved raw incomplete-input subject
accuracy by +0.069 [0.016, 0.124] but harmed complete-input perception. These results
support a development benchmark/pivot, not a held-out performance claim.

A defensible ISMRM result package still requires:

- real validation and single-use held-out test results;
- primary paired effect size with subject-bootstrap 95% confidence interval;
- complete-input non-inferiority result;
- all 15 contrast conditions, not a favorable subset;
- question-only leakage results and strong factorized baseline;
- unconditional-auxiliary and question-conditioned-grounding isolation;
- three seeds when training variance is material;
- calibration and abstention under explicitly defined unsupported conditions;
- threshold/tolerance sensitivity analyses; and
- frozen-rule success, boundary, and failure examples.

No completion or improvement claim is allowed if these are absent. Null findings should
retain effect sizes, intervals, and failure analysis and can support a useful negative
conclusion.

## 7. Release and pivot gates

### Research release gate

Open a public GitHub repository only after a real-data smoke run, named strong baseline,
reproducible command, and honest preliminary table exist. A release must explain why the
work is worth examining using actual evidence, not repository completeness.

### Standard-abstract gate on 10 October

Proceed with a standard ISMRM abstract only if the primary matched comparison, auxiliary
control, strong baseline, all-subset evaluation, uncertainty, and failure analysis exist.
Otherwise use a registered-abstract route only if the 2027 call permits it, or do not
submit.

### Minimum reasonable pivot

If a competitive VLM cannot be trained by the gate, narrow the contribution to a rigorous
benchmark of contrast missingness, voxel-grounding faithfulness, and abstention across
existing 3D medical VLMs. If grounding is indistinguishable from unconditional auxiliary
segmentation, report that negative result and reject the mechanistic grounding claim.

## Evidence that would make the project worth public attention

At release, the summary must point to concrete artifacts showing:

- a previously missing four-contrast, exhaustive grounded-reasoning evaluation;
- fair matched isolation of question-conditioned grounding from extra supervision;
- a strong factorized segmentation baseline and an external 3D medical VLM comparison;
- reproducible real-data results with subject-level uncertainty and failures; and
- either a meaningful reduction in unsupported answers without complete-input harm, or a
  rigorous negative result that changes how MRI-VLM grounding should be evaluated.

Until then, the accurate description is **research-preview implementation with preliminary
negative validation evidence**, not publish-ready MRI-VLM research.

## Literature checked

- M3D-LaMed: https://arxiv.org/abs/2404.00578
- SGMRI-VQA: https://arxiv.org/abs/2604.15808
- Missing-modality brain-tumor segmentation: https://arxiv.org/abs/1904.07290
- Original BraTS benchmark and inter-rater context:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4833122/
- Missing-as-Masking, MICCAI 2024:
  https://papers.miccai.org/miccai-2024/520-Paper0067.html

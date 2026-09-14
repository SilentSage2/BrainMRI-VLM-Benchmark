# ISMRM Figure Contract and Delivery Plan

Target deadline: **28 October 2026**. Internal submission-ready deadline: **22 October**.
The final abstract will use four figures by default and a fifth only when it adds a distinct
result. Every plot must be regenerated from a versioned command and retain its source table.

| Figure | Scientific claim | Required input | Due | Failure-safe version |
|---|---|---|---|---|
| 1. Cohort and targets | The audited cohort supports varied physical-volume and enhancing-fraction questions without subject leakage. | 484 reference masks and locked split | 18 Sep | Descriptive cohort panel, explicitly not a model result |
| 2. Grounded MRI-VLM | Answers, voxel evidence, modality availability, and abstention are jointly testable. | Frozen architecture and one real case | 25 Sep | Methods schematic plus real input montage |
| 3. Contrast-dependence heatmap | Grounding changes question-specific sensitivity to FLAIR/T1/T1-Gd/T2 removal. | Matched models × question families × missing contrasts | 6 Oct | Report the predeclared matrix even if effects are null |
| 4. Robustness and uncertainty | Grounding improves—or fails to improve—performance as available contrasts decrease. | All 15 combinations, subject bootstrap, seeds | 10 Oct | Show effect sizes and confidence intervals without significance claims |
| 5. Evidence and failure analysis | Spatial evidence and abstention expose supported answers and failure modes. | Frozen representative-selection rule | 15 Oct | Use one success, one correct abstention, one shared failure |

Status on 14 September: Figure 1 is complete from all 484 audited masks in PNG and vector
PDF form, with a 484-row source CSV and JSON summary. Figure 2 is complete using a real
validation case selected by the frozen median-burden rule. The remaining panels require
model outputs and must not be populated with simulated performance.

## Milestones

- **14–18 Sep:** complete data audit, physical-volume summaries, Figure 1, and deterministic
  real-data QA/evidence manifest.
- **19–25 Sep:** finish preprocessing/dataloader, question-only control, one-case overfit,
  and Figure 2.
- **26 Sep–3 Oct:** run matched answer-only and grounded pilot; validate result tables and
  generate provisional Figures 3–4.
- **4–10 Oct:** run locked comparisons, all missing-contrast conditions, subject bootstrap,
  and the standard-abstract go/no-go review.
- **11–15 Oct:** repeat seeds where material, freeze qualitative selection, and Figure 5.
- **16–22 Oct:** freeze tables and figures, write the abstract, perform claim and leakage
  audits, and prepare the submission package.
- **23–26 Oct:** review and submission buffer. Do not plan first-time experiments here.

## Non-negotiable figure rules

- Never insert illustrative or simulated performance values into a result panel.
- Label descriptive, validation, and held-out test results unambiguously.
- Select qualitative cases by a frozen rule, not by visual appeal after inspecting test
  outcomes.
- Include units, sample sizes, uncertainty definitions, and readable text at final size.
- Retain CSV/JSON source data and the exact command/config used to produce every figure.
- Negative results keep their planned panel; the interpretation changes, not the endpoint.

## Reproduction

Figure 1 is generated from real reference masks and writes PNG, vector PDF, CSV source data,
and a JSON summary. Generated artifacts remain outside Git.

```bash
mri-vlm-cohort-figure /absolute/path/to/Task01_BrainTumour \
  --output-prefix artifacts/figures/figure1_cohort
```

Figure 2 writes PNG, vector PDF, and selection metadata. The reference overlay is shown as
the supervision target and is explicitly not a model input.

```bash
mri-vlm-method-figure /absolute/path/to/Task01_BrainTumour \
  --output-prefix artifacts/figures/figure2_method
```

# ISMRM Figure Contract and Delivery Plan

Target deadline: **28 October 2026**. Internal submission-ready deadline: **22 October**.
The final abstract will use four figures by default and a fifth only when it adds a distinct
result. Every plot must be regenerated from a versioned command and retain its source table.

| Figure | Scientific claim | Required input | Due | Failure-safe version |
|---|---|---|---|---|
| 1. Overall framework | The implemented pipeline connects multi-contrast MRI and questions to answer/evidence outputs and locked evaluation. | Implemented architecture, frozen protocol, and one real case | 25 Sep | Mark not-yet-implemented training elements explicitly |
| 2. Cohort and targets | The audited cohort supports varied physical-volume and enhancing-fraction questions without subject leakage. | 484 reference masks and locked split | 18 Sep | Move to supplement if four figures tell the story better |
| 3. Contrast-dependence heatmap | Grounding changes question-specific sensitivity to FLAIR/T1/T1-Gd/T2 removal. | Matched models × question families × missing contrasts | 6 Oct | Report the predeclared matrix even if effects are null |
| 4. Robustness and uncertainty | Grounding improves—or fails to improve—performance as available contrasts decrease. | All 15 combinations, subject bootstrap, seeds | 10 Oct | Show effect sizes and confidence intervals without significance claims |
| 5. Evidence and failure analysis | Spatial evidence and abstention expose supported answers and failure modes. | Frozen representative-selection rule | 15 Oct | Use one success, one correct abstention, one shared failure |

Status on 14 September: Figure 1 is a provisional overall framework using a real validation
case selected by the frozen median-burden rule; the implemented model and evaluation paths
are solid, while the pending training runner is dashed and labeled. Figure 2 is complete
from all 484 audited masks in PNG and vector PDF form, with a 484-row source CSV and JSON
summary. The remaining panels require model outputs and must not contain simulated
performance.

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

- Every figure starts with a contract specifying the scientific conclusion it can support,
  its null or disconfirming interpretation, required inputs, and prohibited claims.
- Never insert illustrative or simulated performance values into a result panel.
- Label descriptive, validation, and held-out test results unambiguously.
- Select qualitative cases by a frozen rule, not by visual appeal after inspecting test
  outcomes.
- Empirical result figures must include meaningful baselines, controlled ablations, sample
  sizes, and confidence intervals or error bars with their construction stated. Use three
  seeds when training variance is material and subject-level bootstrap for paired inference.
- Qualitative panels must show representative success, boundary, and failure cases, with
  the outcome-independent selection rule retained beside the source data.
- Method schematics must be checked against the implemented forward pass and configuration.
- Include units, sample sizes, uncertainty definitions, and readable text at the locked
  180-mm output width; use a color-vision-deficiency-safe palette and redundant labels.
- Retain CSV/JSON source data and the exact command/config used to produce every figure.
- Export high-resolution PNG and vector PDF; visually inspect both before release.
- Captions must be independently understandable and distinguish observation from inference.
- Negative results keep their planned panel; the interpretation changes, not the endpoint.
- Figures 3–5 remain explicitly `planned` until their locked real-data inputs exist.

## Figure contracts

### Figure 1 — overall framework, provisional

- Supports: the implemented model connects four MRI contrasts, availability masks, and
  question tokens to answer and voxel-evidence logits, followed by the implemented 15-way
  missing-contrast evaluation and subject-level metrics.
- Current limitation: the training protocol is frozen but its runner is not implemented;
  it is therefore drawn dashed and explicitly labeled pending.
- Cannot support: learned grounding, robustness, accuracy, or clinical utility.

### Figure 2 — descriptive, complete

- Supports: the audited cohort has a locked subject split and heterogeneous mask-derived
  physical-volume targets.
- Can refute: sufficient target diversity if distributions collapse or labels are absent.
- Cannot support: model performance, clinical validity, or population generalizability.

### Figure 3 — empirical, planned

- Supports only after locked evaluation: whether grounding changes question-specific
  leave-one-contrast-out sensitivity relative to the matched answer-only model.
- Disconfirming result: no aligned interaction, reversed dependence, or intervals spanning
  effects too broadly for the predeclared claim.

### Figure 4 — empirical, planned

- Supports only after locked evaluation: whether performance degrades differently as the
  number and identity of available contrasts change.
- Disconfirming result: no grounded-model advantage or violation of the complete-input
  non-inferiority margin.

### Figure 5 — qualitative plus empirical, planned

- Supports only after frozen case selection: how correct grounding, calibrated abstention,
  boundary behavior, and shared failure appear spatially.
- Cannot support prevalence or comparative performance without the corresponding aggregate
  result and uncertainty.

## Reproduction

Figure 2 is generated from real reference masks and writes PNG, vector PDF, CSV source data,
and a JSON summary. Generated artifacts remain outside Git.

```bash
mri-vlm-cohort-figure /absolute/path/to/Task01_BrainTumour \
  --output-prefix artifacts/figures/figure2_cohort
```

Figure 1 writes PNG, vector PDF, and selection metadata. The reference overlay is shown as
the supervision target and is explicitly not a model input.

```bash
mri-vlm-method-figure /absolute/path/to/Task01_BrainTumour \
  --output-prefix artifacts/figures/figure1_overview
```

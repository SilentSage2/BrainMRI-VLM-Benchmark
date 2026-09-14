# ISMRM Figure Contract and Delivery Plan

Target deadline: **28 October 2026**. Internal submission-ready deadline: **22 October**.
The final abstract will use four figures by default and a fifth only when it adds a distinct
result. Every plot must be regenerated from a versioned command and retain its source table.

| Figure | Scientific claim | Required input | Due | Failure-safe version |
|---|---|---|---|---|
| 1. Overall framework | The implemented pipeline connects multi-contrast MRI and questions to answer/evidence outputs and locked evaluation. | Implemented architecture, frozen protocol, and one real case | 25 Sep | Mark not-yet-implemented training elements explicitly |
| 2. Cohort and targets | The audited cohort supports varied physical-volume and enhancing-fraction questions without subject leakage. | 484 reference masks and locked split | 18 Sep | Move to supplement if four figures tell the story better |
| 3. QA target validity | Show how spatial resolution changes mask-derived answers and why ambiguity controls are required. | 418 development masks × four resolutions; test unread | 20 Sep | Retain as a real negative-methods result |
| 4. Fifteen-condition MR robustness and reliability | Compare grounded, answer-only, and modular segmentation-to-symbolic interpretation, calibration, and abstention across all contrast subsets. | Models × 15 contrast subsets, subject bootstrap, seeds | 10 Oct | Report the locked matrix even if the modular baseline wins |
| 5. Evidence and failure boundaries | Show where voxel evidence supports interpretation and where missing contrasts cause abstention or failure. | Frozen success/boundary/failure selection | 15 Oct | Include the same cases for VLM and modular MR baseline |

Status on 14 September: Figure 1 is a provisional overall framework using a real validation
case selected by the frozen median-burden rule; the implemented model and small pilot runner
are labeled without a performance claim. Figure 2 is complete
from all 484 audited masks in PNG and vector PDF form, with a 484-row source CSV and JSON
summary. Figure 3 uses 418 development masks and no test cases to document target
instability. Figures 4–5 require model outputs and must not contain simulated performance.

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

### Figure 3 — target-validity audit, in progress

- Supports: answer stability improves with spatial resolution, while boundary-derived
  categorical targets require development-derived ambiguity rules.
- Disconfirming result already observed: the presence screen becomes single-class and two
  comparison families are algebraically redundant, so both must be redesigned or removed.
- Cannot support: model performance, test generalization, or clinical utility.

### Figure 4 — empirical, planned

- Supports only after locked evaluation: how grounded, answer-only, and modular MR methods
  behave across all 15 contrast subsets, and whether calibration and abstention respond
  appropriately to the identity of a removed contrast.
- Disconfirming result: overconfident unsupported answers, no question-specific contrast
  sensitivity, or violation of the complete-input non-inferiority margin.

### Figure 5 — qualitative plus empirical, planned

- Supports only after frozen case selection: how correct grounding, calibrated abstention,
  boundary behavior, and shared failure compare with the modular MR pipeline spatially.
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

Figure 3 reads the development-only QA audit summary, verifies that no test cases were read,
and writes PNG, vector PDF, and source metadata.

```bash
mri-vlm-stability-figure artifacts/results/qa_resolution_stability_v1.json \
  --output-prefix artifacts/figures/figure3_qa_stability
```

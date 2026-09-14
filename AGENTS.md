# Repository instructions

## Purpose

MRI-VLM Grounding Lab studies evidence-grounded 3D vision-language reasoning over
multi-sequence brain MRI. It is not a medical device or clinical tool.

## Boundaries

- Keep V0 limited to the hypothesis, typed MRI/QA/evidence contracts, subject-level
  splits, audits, metrics, and deterministic fixtures.
- Do not train a large VLM before dataset, question-generation, evidence, missingness,
  compute, and acceptance protocols are reviewed.
- Never commit MRI data, weights, credentials, or run artifacts.
- Do not describe mask-derived text as radiology reports or claim clinical validity.

## Research quality

- Split by subject before generating questions, slices, patches, or counterfactuals.
- Keep every derived example from a subject in the same split.
- Include a question-only control and unanswerable questions to expose language shortcuts.
- Match training examples, optimizer steps, and compute across answer-only and grounded VLMs.
- Evaluate every predeclared missing-sequence condition and report negative results.
- Test label mapping, leakage, evidence metrics, numeric scoring, and serialization.

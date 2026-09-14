# ISMRM Format and Proceedings Notes

This is a writing-style audit, not an evidence source for model claims. The 2027 call will
supersede the working rules below.

## Official working constraints

The 2026 standard guidance specifies a 125-character title, structured 100-word Synopsis,
40-word Impact, 750-word body excluding references, up to five figures with captions of at
most 500 characters, and one no-caption preview figure readable on a smartphone. The
Synopsis headings are Motivation, Goal(s), Approach, and Results. Synopsis and Impact are
written for the broad meeting audience; the main body carries specialist detail.

- https://www.ismrm.org/26m/call/standard/
- https://www.ismrm.org/26m/call/submission-guide/impact-synopsis/

## Public proceedings sampled

- Missing brain-tumor MRI sequences, 2024:
  https://archive.ismrm.org/2024/3758.html
- Missing MRI data in multiple sclerosis, 2024:
  https://archive.ismrm.org/2024/4861.html
- Glioma vision-language modeling, 2025:
  https://archive.ismrm.org/2025/3389.html
- MRI vision-language motion correction, 2025:
  https://archive.ismrm.org/2025/3379.html

Across these examples, the Synopsis opens with a concrete MR limitation, narrows to one
technical goal, identifies the dataset/model in Approach, and reserves Results for the
core observed outcome. Impact translates the technical capability for a broader MRI
audience. Some public examples omit uncertainty or use broad claims; they are structural
references only and do not lower this project's requirement for numerical effects,
uncertainty, controlled comparisons, limitations, and reproducibility.

## Project writing rules

- One frozen results manifest supplies every number in Synopsis, body, captions, tables,
  README, and figures.
- `PLANNED` markers are mandatory until the corresponding real output exists.
- Results lead with effect size and interval, not adjectives such as “robust” or
  “significant.”
- Discussion contains both supported interpretation and disconfirming/negative findings.
- Conclusion answers the preregistered question and does not imply clinical readiness.
- Run `mri-vlm-abstract-lint docs/ISMRM_2027_ABSTRACT_DRAFT.md` after every writing change.

# ISMRM Format and Proceedings Notes

This is a writing-style audit, not an evidence source for model claims. Checked on
2026-09-14. The official 2027 meeting page confirms Vancouver, 8–13 May 2027, and an
abstract window of 1–28 October 2026. Detailed 2027 standard-abstract limits were not yet
posted, so the latest official 2026 standard guidance remains the working format and must
be rechecked when the 2027 call is published.

## Official working constraints

The 2026 standard guidance specifies a 125-character title, structured 100-word Synopsis,
40-word Impact, 750-word body excluding references, up to five figures with captions of at
most 500 characters, and one no-caption preview figure readable on a smartphone. The
Synopsis headings are Motivation, Goal(s), Approach, and Results. Synopsis and Impact are
written for the broad meeting audience; the main body carries specialist detail.

- https://www.ismrm.org/26m/call/standard/
- https://www.ismrm.org/26m/call/
- https://www.ismrm.org/27m/
- https://www.ismrm.org/meetings-workshops/future-ismrm-meetings/

The same guidance asks for substantive results, appropriate statistical analysis, and
high-quality images. It explicitly notes that impact need not be positive: a result showing
that a methodological pivot is needed can matter. That supports the present negative-result
story, but does not relax the requirement for the single frozen held-out evaluation.

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
- `[PLANNED]` or `[HELD-OUT]` markers are mandatory until the corresponding real output
  exists; either marker keeps the lint status at `draft`.
- Results lead with effect size and interval, not adjectives such as “robust” or
  “significant.”
- Discussion contains both supported interpretation and disconfirming/negative findings.
- Conclusion answers the preregistered question and does not imply clinical readiness.
- Run `mri-vlm-abstract-lint docs/ISMRM_2027_ABSTRACT_DRAFT.md` after every writing change.

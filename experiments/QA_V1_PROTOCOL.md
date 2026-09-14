# Balanced QA V1 Protocol

Status: **implemented target materialization; matched model training pending.** This protocol
replaces the rejected fixed-template V0.1 distribution for development experiments.

## Research role

QA V1 is designed to isolate whether question-conditioned voxel supervision adds value
beyond answer supervision and unconditional whole-tumor segmentation. It does not treat
mask-derived language as clinical ground truth.

## Retained families

| Family | Answer | Voxel evidence | Stability rule |
|---|---|---|---|
| Laterality | left/right/midline | whole tumor | Keep only when the preprocessed reference agrees with the original geometry-derived answer |
| Edema versus tumor core | edema/tumor core | the region supporting the answer: edema or tumor core | Same reference-agreement rule |
| Enhancing fraction | training-quartile bin Q1–Q4 | enhancing tumor | Quartile thresholds fitted on training subjects only |

Training-set enhancing-fraction cut points are 0.07244455, 0.16695966, and 0.27325105 for
Q1/median/Q3, computed from 337 locked training subjects. The thresholds must be serialized
with each run and cannot be refitted on validation or test data.

Each question is passed as deterministic word tokens rather than a question-type integer.
The three templates remain synthetic and fixed, so question-only controls and per-family
metrics are still mandatory. Class-balanced loss/metrics prevent majority-class aggregate
accuracy from appearing as image understanding.

## Removed or redefined targets

- Enhancing presence is excluded because its stability screen leaves a single validation
  class.
- Core-greater-than-half-whole is removed because it is algebraically redundant with
  edema-versus-core.
- The prior-scan question is removed because wording alone reveals that it is unsupported.
- Abstention will be evaluated by a validation-frozen selective-confidence policy over
  missing-contrast conditions, not by a recognizable unanswerable template.

## Matched comparison contract

Answer-only, unconditional auxiliary, and question-grounded models must share case IDs,
word vocabulary, initialization, optimizer steps, class weighting, preprocessing, and the
15-condition schedule. The unconditional auxiliary predicts whole tumor for every question;
the grounded head predicts the question-specific evidence above. This is the comparison
that can attribute any difference to question-conditioned spatial supervision.

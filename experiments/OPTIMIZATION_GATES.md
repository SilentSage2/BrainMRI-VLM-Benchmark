# Preprocessing and Optimization Gates

## Verified preprocessing cache

The versioned cache stores four normalized/resampled contrasts and the nearest-neighbor
label with SHA-256 fingerprints over dtype, shape, and tensor bytes. Cache identity includes
the preprocessing version/specification and source file size/mtime. Payloads with changed
identity, shape, or tensor digest are rejected.

For the six-train/four-validation pilot, ten cache files occupy 3.2 MB. A cached run and the
preceding uncached run produced identical configuration and metric objects after removing
wall-clock fields. Filtering QA generation to the same ten fixed case IDs reduced total
pilot wall time from 50.7 seconds to 7.2 seconds before the later loss revision. No test
case is eligible for pilot cache generation.

## One-case overfit gate

Scope: `BRATS_001`, full four-contrast input, seed `20260914`. This tests optimization and
target consistency only; it is not generalization evidence.

### Resolution audit

At `16³`, downsampling changed the reference answers for core-versus-whole and
edema-versus-core comparisons. At `24³`, `32³`, and `48³`, all five categorical reference
answers agreed with the original physical-volume rules. Consequently `16³` is rejected and
the gate now requires preprocessed-reference symbolic accuracy of 1.0.

### Corrected `24³` run

Command:

```bash
mri-vlm-overfit-check /absolute/path/to/Task01_BrainTumour \
  --output artifacts/results/one_case_overfit_v2.json \
  --cache-dir data/processed/overfit-cache \
  --spatial-size 24 --steps 200 --width 8
```

Result SHA-256:
`4168b1b97bcaee6eb12245fd40b2c0f8cabcfb9bb45bc91461b4d32cb7ad89aa`.
Wall time was 172.8 seconds on CPU.

| Gate | Result | Pass |
|---|---:|:---:|
| Preprocessed-reference symbolic accuracy | 1.000 | yes |
| Segmentation whole-tumor Dice | 0.966 | yes |
| Segmentation edema Dice | 0.936 | descriptive |
| Segmentation non-enhancing Dice | 1.000 | descriptive |
| Segmentation enhancing Dice | 0.990 | descriptive |
| Segmentation-derived symbolic accuracy | 0.400 | **no** |
| Answer-only training accuracy | 1.000 | yes |
| Unconditional auxiliary answer/evidence Dice | 1.000 / 0.974 | yes |
| Question-grounded answer/evidence Dice | 1.000 / 0.978 | yes |

The overall gate remains **failed**. A separate 200-step diagnosis showed that the predicted
segmentation preserved presence, laterality, and abstention but flipped both
core-versus-whole and edema-versus-core answers. For BRATS_001, core/whole is 0.525 and the
edema/core volume gap is only 5.0% of whole-tumor volume. High voxel Dice is therefore not
sufficient to stabilize threshold-derived reasoning near a decision boundary.

## Decision evidence

- The VLM optimization paths can memorize one real case, so the earlier small-pilot failure
  is not an elementary gradient or serialization failure.
- `24³` is the minimum audited pilot resolution for this case; this is not yet a cohort-wide
  guarantee.
- The segmentation-to-symbolic path exposes an important benchmark defect: questions near
  discrete decision boundaries must be excluded using a frozen ambiguity margin or scored
  continuously. Lowering the overfit gate would hide this defect and is prohibited.
- Before a larger run, perform cohort-wide resolution stability, freeze ambiguity margins,
  and regenerate balanced QA V1. The strong modular baseline remains pending.

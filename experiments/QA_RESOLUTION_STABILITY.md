# Development QA Resolution-Stability Audit

Run date: 2026-09-14. Status: **completed on train/validation labels only; QA V1 design
gate failed for the current presence and duplicated comparison families.**

## Scope and leakage boundary

The audit compares categorical answers and enhancing-fraction targets computed from the
original masks with nearest-neighbor labels resampled to `16³`, `24³`, `32³`, and `48³`.
It includes 337 training and 81 validation subjects. It reads zero test cases and therefore
does not use test behavior to select resolution or ambiguity margins.

```bash
mri-vlm-qa-stability /absolute/path/to/Task01_BrainTumour \
  --output-prefix artifacts/results/qa_resolution_stability_v1
```

Result SHA-256: JSON
`b044cdeb5ae447e99c3e56989d9fa7715ab95f9c2cf6ddd6b2627abae49f4f28`; CSV
`7fbf79408111dacc2dc8f1534d51c9ae5bd3878856c3de2668d0380bbf7de37f`.

## Development-set results

| Resolution | All four categorical targets stable | Presence | Laterality | Edema vs core | Core > half whole | Enhancing-fraction MAE / p95 |
|---:|---:|---:|---:|---:|---:|---:|
| `16³` | 85.6% | 91.4% | 99.8% | 94.3% | 94.3% | 0.038 / 0.110 |
| `24³` | 92.8% | 95.2% | 99.8% | 97.8% | 97.8% | 0.018 / 0.050 |
| `32³` | 96.4% | 98.6% | 100.0% | 97.8% | 97.8% | 0.011 / 0.036 |
| `48³` | 97.8% | 98.8% | 100.0% | 99.0% | 99.0% | 0.005 / 0.014 |

At `32³`, an exclusive ambiguity margin equal to the largest unstable *training* example
gave 100% stability among retained validation examples: 76/81 for presence, 81/81 for
laterality, and 73/81 for each comparison. This does not validate the current QA design:
all 76 retained presence answers were `yes`, and the two comparison questions have the
same decision boundary.

## Structural defect and decision

`edema >= tumor core` and `tumor core > 0.5 × whole tumor` are algebraic complements because
whole tumor is exactly edema plus tumor core. Treating them as separate families duplicates
one target and inflates the aggregate sample size. The fixed 0.1-mL enhancing-presence task
is also almost single-class after stability screening.

Therefore:

- remove the redundant cross-region comparison from primary evaluation;
- do not use enhancing presence as an aggregate-accuracy contributor;
- retain laterality and continuous enhancing fraction with explicit tolerances;
- redesign missing-contrast abstention so support depends on the available MR contrasts,
  not on recognizable question wording;
- use `32³` only for optimization smoke tests. The strong MR baseline must preserve more
  spatial detail through a higher-resolution crop/patch protocol before locked evaluation.

This negative result is eligible for a target-validity figure, but not as evidence of model
performance or clinical utility.

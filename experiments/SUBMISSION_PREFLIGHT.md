# Pre-Unseal Submission Audit

Audit date: 2026-09-14. Scope: repository state before any held-out prediction run.

## Reproducibility checks

- Ruff: pass.
- Strict mypy over `src` and `tests`: pass, 67 source files.
- Pytest: pass, 89 tests after freezing the complete held-out aggregate and figure schemas.
- Metadata-only V0 protocol audit: pass, with deterministic 40/12/8 fixture split and 15
  modality conditions.
- A clean Python 3.12 virtual environment with an editable `.[dev]` install passes the same
  four checks.
- The core dependency declaration now includes NumPy because checkpoint fingerprinting is
  part of the base package rather than a data-only feature.

## Test seal and repository hygiene

- Frozen held-out identifiers: 66, with no image or label values in Git.
- Development result manifests checked: 37; every recorded `test_cases_read` value is zero.
- `artifacts/results/heldout_v1` is absent.
- No MRI volume, derived volume, model checkpoint, run directory, credential, or detected
  secret pattern is tracked.
- Dataset license is documented as CC BY-SA 4.0; code is MIT. Public redistribution excludes
  the dataset, checkpoints, and generated result artifacts.
- The one-shot evaluator requires an exact authorization file and creates its lock before
  the first held-out image read. This evaluator was not invoked during preflight.

The machine-readable preflight report is generated locally at
`artifacts/results/submission_preflight.json` and remains ignored by Git. Its final hash and
tracked-file count are recorded below after the release candidate is staged.

## Figure QA

Figures 4 and 5 were rendered at 300 dpi, visually inspected at full resolution, and kept
outside Git with their vector PDFs and source tables.

| Artifact | Dimensions | SHA-256 |
|---|---:|---|
| Figure 1 PNG | 2198 x 1398 | `27056972c73f52570ac556aa9d4a7877a678cbf65d07f589417871d1549d3d5f` |
| Figure 1 PDF | vector | `a3c5b65b7542d283c3bfc2a3d713b85dc277a726cb1186a11d5bb8bd3ca95b46` |
| Figure 4 PNG | 4050 x 1140 | `b2419e63fecd2f8e984e97740eca489219d0aaa775602893e3cb8f8712a2323c` |
| Figure 4 PDF | vector | `74db27456073b03e1951a9b413d5493f6c7ac6913023fee3431187f4fbc95918` |
| Figure 4 source CSV | n/a | `23dba6206fff31850c1433f4514b7a8e635653e7c5da3b19f15bc3f617fcc547` |
| Figure 5 PNG | 4500 x 1140 | `4a5ceebe1f0a07b78989a0d06ee426e8ca5cedf18999fe6a8c14b9d593c5faa2` |
| Figure 5 PDF | vector | `b89078534426d2dc5e320ae04ecd315ff8851df7c3172f353f3a5136f42e9a54` |
| Figure 5 source CSV | n/a | `5b54b49b390923e2b84da5e7825ddc68d1433e03ef488758381d4d83dc6d3ece` |
| Preview PNG | 1200 x 1200 | `b97e8eb298fb46664834730ed5ff37a6bad1939b997bd35e94ad8e84bd9f8beb` |
| Preview PDF | vector | `2f1d8ea8c6ab5996cac95ba06942566da0558609a56e43e1c0db73d39d116cee` |

Figure 1's dual pathways and arrows, Figure 4's comparison labels, and Figure 5's condition
matrix, profile labels, and confidence interval are readable without overlap. Figures 4–5
are development-only and their captions state that the test set was unread. Figure 4/5
source-table hashes did not change during the label-only paper-readability revision.

The frozen held-out Figure 5 renderer was separately exercised with the completed
development aggregates only. Its three-system bars, 3-by-15 heatmap, modular Dice panel,
and three-effect forest plot were visually readable; no held-out file was opened or
created.

## Final machine audit

- Status: pass.
- Tracked file count: 109.
- Report SHA-256: `f61cc8e5e7203171bc77506c1f7bab50ee79b7776342af128a0f0e8fb3d20eed`.
- Remote CI: pass for release-candidate commit `bfb1dbb` (GitHub Actions run 9; editable
  install, Ruff, strict mypy, 84 tests, and V0 audit all succeeded).

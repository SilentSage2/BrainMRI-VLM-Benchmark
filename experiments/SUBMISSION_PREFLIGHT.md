# Pre-Unseal Submission Audit

Audit date: 2026-09-14. Scope: repository state before any held-out prediction run.

## Reproducibility checks

- Ruff: pass.
- Strict mypy over `src` and `tests`: pass, 64 source files.
- Pytest: pass, 84 tests.
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
| Figure 4 PNG | 4050 x 1140 | `a5b50e2aeeaed7e27d329b4ae935a29b74c0979c397be304e4ba1292abebf876` |
| Figure 4 PDF | vector | `19f21a69a8c6b800168f1a18deb99cadf8aa0b8d5b28bcf20acf1ebd1bff5199` |
| Figure 4 source CSV | n/a | `23dba6206fff31850c1433f4514b7a8e635653e7c5da3b19f15bc3f617fcc547` |
| Figure 5 PNG | 4500 x 1140 | `9a61c180735170db9969da76b627e196fa9201ae5e62c1f6682dce45bcfa3c26` |
| Figure 5 PDF | vector | `0439606ff474a4da58d9fa2ce4e50f349c19fcf77da045593b27e1fc51f2921e` |
| Figure 5 source CSV | n/a | `5b54b49b390923e2b84da5e7825ddc68d1433e03ef488758381d4d83dc6d3ece` |

Figure 4's long comparison labels and Figure 5's condition matrix, profile labels, and
confidence interval are readable without overlap. Both figures are development-only and
state that the test set was unread.

## Final machine audit

- Status: pass.
- Tracked file count: 105.
- Report SHA-256: `2c0a0f71d9fd5625a41905d5ed7c784df3dc4f90eb4008ed0d57e6d209e56559`.
- Remote CI: pass for release-candidate commit `bfb1dbb` (GitHub Actions run 9; editable
  install, Ruff, strict mypy, 84 tests, and V0 audit all succeeded).

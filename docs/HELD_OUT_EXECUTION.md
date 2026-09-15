# One-Shot Held-Out Execution Runbook

Status: **locked and not authorized**. Do not run the command below until the user explicitly
authorizes one-time held-out test access.

## Preconditions

1. `mri-vlm-submission-preflight` passes and records `test_reads_observed = 0`.
2. Git worktree is clean and the commit hash is recorded.
3. `configs/heldout_v1.toml` and `configs/test_subject_manifest.json` hashes match.
4. All nine checkpoint hashes match the frozen config.
5. Figure 4/5 schemas, metrics, exclusions, bootstrap seed, and interpretation branches are
   frozen in Git.
6. Result schema `heldout-v1-complete-aggregate-20260914` includes every condition and seed,
   the primary comparison, dropout ablation, all mandatory secondary metrics, and aggregate
   success/boundary/failure counts. Ordered subject identifiers must match across systems.
7. An authorization file outside Git contains exactly:
   `AUTHORIZE MRI-VLM HELDOUT V1 ONCE`.

## One-shot command

The eventual evaluator must use an output directory that has never existed. The evaluator
creates an irreversible local lock before opening the first held-out image. Any existing
lock or output aborts; a failed run is resumed from the same directory and may not be
restarted under a new name.

```bash
mri-vlm-heldout-once /absolute/path/to/Task01_BrainTumour \
  --config configs/heldout_v1.toml \
  --manifest configs/test_subject_manifest.json \
  --authorization-file /absolute/path/outside/repo/heldout_authorization.txt \
  --output-dir artifacts/results/heldout_v1
```

## Reporting rules

- Write all 15 conditions, all three seeds, and all frozen systems before reading any
  aggregate comparison.
- Reject aggregation if ordered subject identifiers differ across any system, seed, or
  condition.
- Preserve raw per-subject scores internally; publish only aggregate metrics and
  preselected examples consistent with dataset terms.
- Do not tune thresholds, prompts, preprocessing, checkpoint selection, or exclusions.
- Report crashes, missing subjects, and resumed execution in the immutable run ledger.
- Generate the final held-out Figure 5 only through `mri-vlm-heldout-figure`; keep Figure 4
  as the development mechanism/ablation result and do not hand-edit values.
- Update the abstract through the prewritten interpretation branch matching the result.

The test split is not pristine unseen data because aggregate label properties were included
in an earlier cohort audit. No test predictions, model outcomes, or per-case selection have
been used. This limitation must remain in the submission.

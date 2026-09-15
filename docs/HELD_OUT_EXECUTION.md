# One-Shot Held-Out Execution Runbook

Status: **completed exactly once on 15 September 2026 after explicit authorization**.
The local irreversible lock records 66 test cases read and protocol completion.

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

## Completion record

- Pre-unseal Git commit: `6ae522c898db807d7084da398189f3d29ab83ba5`.
- Preflight SHA-256: `f61cc8e5e7203171bc77506c1f7bab50ee79b7776342af128a0f0e8fb3d20eed`.
- Result schema: `heldout-v1-complete-aggregate-20260914`.
- Lock status: `complete`; test cases read: 66/66; no interruption or resume.
- Aggregate summary SHA-256: `ba4c332d0acbf70b53f49097a22b9e0acb526a308f013e276dcb142fea578ec4`.
- One-shot lock SHA-256: `d99b71ace8540cd4fc52a70df17257f51e15be6d5284f74a140cfed0b85c4431`.
- Primary effect: +0.187, hierarchical 95% CI [+0.131, +0.257].
- Final Figure 5 PNG SHA-256: `27db6347a59798097ac8d66e125a032d9c599082942c275900f213f5b25a3740`.
- Final Figure 5 PDF SHA-256: `a0187f4e6d35297bc3805b5290fceeb4574094aec61e93ee67f285372f7b75dd`.
- Final Figure 5 source CSV SHA-256: `2eba920217c9b6b915c4548f0a324d955354666193aedb7676fdbbd506a26728`.

The authorization token, subject-level outputs, MRI data, checkpoints, and generated result
artifacts remain outside Git. Only aggregate findings and reproducible code are public.

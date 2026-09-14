# Matched-Run Recovery Ledger

## Interrupted attempt — 2026-09-14

The first `matched_direction_v1` process was lost during a Codex host restart. No result
JSON was written. The cache contained the same 40 fingerprint-validated preprocessing
payloads used by the completed 32/8 baseline run, and no test case was read.

Only `answer_only.pt` existed (67 KB; SHA-256
`4f1ca74f8b035a9f41dfbc2d0208eb97091947182bb7d71b83b3320b5068ad17`). Inspection showed
only a best state dictionary and best epoch. It did not contain split/config fingerprint,
history, optimizer state, RNG state, or training duration. Consequently it cannot prove
that the exact frozen cases and schedule produced it and is not eligible for reuse. The
recovery runner quarantines it under an `unverified-*` filename rather than deleting it.

## Recovery behavior added

Completed per-role checkpoints now contain:

- a SHA-256 fingerprint over train/validation case IDs, preprocessing, QA thresholds,
  vocabulary, architecture version, seed, optimizer settings, epochs, and all 15 modality
  conditions;
- role identity and completion marker;
- best state, best epoch, complete validation history, and training duration.

A completed role is reused only when all identity fields match. Otherwise it is quarantined
and only that role is rerun. Mid-role optimizer/RNG checkpoints were absent in the interrupted
attempt, so safe within-role continuation was impossible; the affected role must restart
from the shared frozen initialization. This limitation is retained in the ledger rather
than inferred away.

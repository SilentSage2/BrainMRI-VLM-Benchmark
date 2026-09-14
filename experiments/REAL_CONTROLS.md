# Real-Data Control Ledger

## V0.1 fixed-template diagnostic — 2026-09-14

Status: **completed on validation; rejected as the final QA distribution because of severe
question-only shortcut performance.** No test prediction was made. Test masks had already
been included in the earlier integrity audit and aggregate cohort description, so the test
split is locked but not described as completely unseen.

### Frozen inputs

- MSD Task01 BrainTumour: 484 audited labeled subjects.
- Split seed: `20260914`; 337 train, 81 validation, 66 locked test subjects.
- Six deterministic examples per subject: enhancing presence, laterality, edema-versus-core
  volume, enhancing fraction, core-versus-whole comparison, and unavailable prior-scan
  change.
- Enhancing presence threshold: 0.1 mL.
- Midline relative-volume margin: 0.05.
- Question-only control fitted only on the training split using the per-type categorical
  majority or numeric median.
- Validation confidence interval: 2,000 subject-level bootstrap samples, seed `20260914`.

### Reproduction

```bash
mri-vlm-real-controls /absolute/path/to/Task01_BrainTumour \
  --output artifacts/results/real_controls_validation.json \
  --bootstrap-samples 2000
```

Generated files remain outside Git:

- result SHA-256:
  `3a88f117034a13ef3b1cebe06b5c4bf0063f8df88307fbeaee132d34f87eb12a`;
- 2,508-row train/validation-only manifest SHA-256:
  `d8ae0c03af9f06331e4147588c5a5309f8b443175102db45eee6e4401ae44b48`.

### Validation results

| Control | Metric | Estimate | Subject-bootstrap 95% CI |
|---|---|---:|---:|
| Question-only | Answer accuracy, 405 answerable examples | 0.659 | 0.610–0.706 |
| Question-only | Grounded answer accuracy | 0.000 | Not estimated |
| Question-only | Unanswerable hallucination, 81 examples | 0.000 | Not estimated |
| Reference-mask symbolic oracle | Answer accuracy | 1.000 | Tautological upper bound |
| Reference-mask symbolic oracle | Grounded answer accuracy | 1.000 | Tautological upper bound |

Question-only answer accuracy by family was 0.963 for enhancing presence, 0.815 for
edema-versus-core relative volume, 0.815 for core-versus-whole comparison, 0.420 for
laterality, and 0.284 for enhancing fraction.

### Interpretation

This diagnostic falsifies the assumption that the initial answer distribution adequately
tests image use. Fixed question wording makes unanswerable detection trivial, and class
imbalance lets a question-only model answer most presence and comparison questions. A 3D
model can exceed an apparently respectable aggregate accuracy without using MRI.

The result does **not** invalidate voxel-grounded evaluation: a question-only model has
zero grounded accuracy because it supplies no evidence. It does require a revised final
QA protocol with train-derived balancing or stratified sampling, multiple paraphrases,
reporting by family, and unsupported conditions that are not identifiable from wording
alone. V0.1 must not be mixed with later final results.

The reference-mask symbolic oracle only checks deterministic materialization. It is not the
required learned 3D segmentation-to-symbolic-QA baseline, which remains pending.

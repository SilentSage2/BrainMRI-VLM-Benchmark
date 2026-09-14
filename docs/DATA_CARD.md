# Data Card

Medical Segmentation Decathlon `Task01_BrainTumour` is the initial candidate. Its official
site describes multimodal multisite FLAIR, T1, T1-Gd, and T2 MRI, glioma subregion masks,
484 training volumes, 266 test volumes, and CC BY-SA 4.0 licensing.

VLM questions, typed answers, and voxel evidence are deterministically derived from the
labeled training subjects. These are synthetic research annotations—not clinical reports
or radiologist judgments. The unlabeled challenge test set is not used as validation.

The local adapter must verify archive checksum, pseudonymous subject uniqueness, modality
completeness, shapes, affines, orientation, spacing, finite values, and allowed mask labels.
All data remains outside Git.

## Local audit record

The official Task01 archive was audited on 2026-09-14 without adding data to Git.

- archive bytes: `7608266240`;
- archive SHA-256: `d423911308d2ae5396d9c6bf4fad2b68cfde2dd09044269da9c0d639c22753c4`;
- audited labeled cases: `484`;
- dataset-description fingerprint: `77f38db2fcf1a39a`;
- case-manifest SHA-256:
  `1edb42a5c45a9ce69b37a91c308da93a9fcbb5956ca362c972f98d34d4ecd5d3`;
- deterministic subject split (`seed=20260914`): train `337`, validation `81`, test `66`.

The audit verified finite arrays, four-channel image shape, image/label shape, spacing and
affine agreement, label values restricted to `0/1/2/3`, unique subject assignment, and
fingerprint isolation between splits. These counts are implementation results; no model
performance result has been produced yet.

The source stores all four sequences as channels of one 4D NIfTI volume. The adapter reads
the channel order from `dataset.json`, validates the 4D image against its 3D label, and
derives content-addressed logical sequence records. Run `mri-vlm-msd-audit` before any
question generation or preprocessing.

Limitations include curated tumor prevalence, historical acquisition protocols, synthetic
language, incomplete site metadata, and mask-based evidence that cannot represent every
reasoning cue a radiologist might use. Benchmark gains do not establish safety or utility.

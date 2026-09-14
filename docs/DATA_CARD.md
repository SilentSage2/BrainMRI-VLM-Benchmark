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

The source stores all four sequences as channels of one 4D NIfTI volume. The adapter reads
the channel order from `dataset.json`, validates the 4D image against its 3D label, and
derives content-addressed logical sequence records. Run `mri-vlm-msd-audit` before any
question generation or preprocessing.

Limitations include curated tumor prevalence, historical acquisition protocols, synthetic
language, incomplete site metadata, and mask-based evidence that cannot represent every
reasoning cue a radiologist might use. Benchmark gains do not establish safety or utility.

# Architecture

V0 contains typed MRI volumes, subject cases, grounded QA examples, canonical fingerprints,
subject-level splitting, integrity audits, answer/evidence metrics, and synthetic fixtures.

V1 adds a NIfTI adapter, mask-derived question generator, counterfactual transforms, a
slice-based VLM baseline, a sequence-aware 3D visual encoder, language-conditioned fusion,
answer decoder, voxel-evidence head, modality-mask sampler, and immutable run manifests.

The evidence head is part of the tested hypothesis; a standalone segmentation model is an
oracle/control component, not the flagship system.

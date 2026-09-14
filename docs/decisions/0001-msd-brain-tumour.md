# ADR 0001: Initial MRI dataset

- **Status:** Accepted for protocol work; download not yet performed
- **Decision date:** 2026-09-14

## Decision

Use Medical Segmentation Decathlon `Task01_BrainTumour` as the initial imaging and
voxel-evidence source for grounded MRI-VLM questions.

The official task describes four MRI sequences (FLAIR, T1w, T1-Gd, and T2w), glioma
subregion labels, 484 training volumes, 266 test volumes, and a CC BY-SA 4.0 license:

- https://medicaldecathlon.com/
- https://medicaldecathlon.com/dataaws/

## Rationale

The task supplies aligned multi-sequence volumes and segmentation labels under a license
that permits reproducible research and redistribution subject to attribution/share-alike.
Masks and geometry support deterministic, verifiable QA and voxel evidence. Generated
questions must be labeled as synthetic annotations, not radiology reports.

## Alternatives

NYU fastMRI is strong for reconstruction but requires individual application and a data
sharing agreement that prohibits redistribution. It also answers a different research
question. It remains a possible future independent reconstruction project, not a fallback
dataset for this VLM MVP.

## Consequences

No data enters Git. The adapter must validate local files and record the archive checksum.
Because public acquisition-site identifiers may be insufficient for a site-held-out split,
the MVP makes grounded reasoning under missing modalities—not cross-site generalization—
its primary claim.

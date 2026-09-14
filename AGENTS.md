# Repository instructions

## Purpose

SciVLM Lab is a research repository for leakage-aware scientific figure-text
retrieval and, only after the retrieval milestone is credible, grounded visual
question answering.

## Boundaries

- Keep the V0 slice limited to data contracts, provenance, group-aware splits,
  duplicate checks, retrieval metrics, and deterministic synthetic fixtures.
- Do not add a VLM, agent framework, web service, or UI before the V1 retrieval
  protocol and pretrained baseline are reproducible.
- Treat generated specifications as source data and rendered images as generated
  artifacts. Never commit external datasets, model weights, or run artifacts.
- Label planned results as planned. Do not imply that an unrun model improves a
  baseline.

## Quality

- Use typed Python interfaces and deterministic functions in the core package.
- Split by source group or generator family, never by image-caption pair.
- Test metrics, group isolation, duplicate leakage, canonical fingerprints, and
  serialization before model work.
- Record seeds, data fingerprints, code revision, environment, hardware, duration,
  and cost for every experiment.

# Architecture

The package deliberately begins with a small model-independent core:

- `schema`: immutable figure-caption records and split labels;
- `fingerprint`: canonical content hashing for declarative specifications;
- `split`: deterministic source-group assignment and isolation checks;
- `audit`: cross-split duplicate and provenance checks;
- `metrics`: bidirectional retrieval metrics from explicit ranked IDs;
- `synthetic`: a deterministic metadata fixture for protocol tests.

V1 will add narrow adapter interfaces for image and text encoders, contrastive objectives,
negative samplers, and immutable run manifests. Those interfaces are deferred until the
data and evaluation contracts pass review.

# Data Card

## V0 source

The V0 fixture consists of programmatically generated metadata for declarative scientific
charts. It contains no downloaded images, patient data, publication text, or third-party
assets. Every record is reproducible from a checked-in generator version and seed.

## Unit of grouping

`source_group` identifies all views, counterfactual variants, and captions derived from
one latent experimental setup. A source group belongs to exactly one split. Generator
families can be held out as a stronger domain-shift condition.

## Required provenance

Every figure-caption record must include stable IDs, generator family/version, chart and
relation types, visual style, canonical specification SHA-256, rendered image SHA-256,
and source group. External records will additionally require source document ID, dataset
version, license, and intended-use notes.

## Known limitations

Synthetic charts have cleaner captions, simpler layouts, and more explicit relations than
real publication figures. Renderer artifacts can become shortcuts. Generated data cannot
support claims about biomedical or scientific-literature performance; that requires a
separately audited external corpus.

## Storage policy

Declarative fixture metadata may be committed when small. Bulk rendered images, processed
datasets, checkpoints, and run artifacts remain outside Git and are addressed by content
fingerprints. No private or clinical data is permitted.

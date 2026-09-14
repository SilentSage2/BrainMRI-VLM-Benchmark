# V0–V1 Experiment Specification

- **Status:** V0 implementation started; V1 model work gated on protocol review
- **Primary artifact:** bidirectional scientific figure-caption retrieval benchmark
- **Initial data:** deterministic synthetic scientific-chart specifications

## Falsifiable hypothesis

Holding encoder, examples, optimizer budget, and evaluation candidates fixed,
relation-preserving hard negatives will improve relational-caption Recall@10 over random
in-batch negatives on the in-domain test split. The improvement counts as robust only if
it does not materially degrade the held-out-family and held-out-style slices.

The null outcome and a domain-shift regression are publishable negative results.

## Data protocol

Each example is derived from a declarative figure specification and records a figure ID,
caption ID, source group, generator family, chart type, relation type, visual style,
specification fingerprint, and rendered-image fingerprint.

Counterfactuals and alternate captions derived from one latent specification share a
source group. Splits operate on source groups. A generator family may be reserved wholly
for the shifted-domain test slice. Exact image or specification fingerprints may not
cross splits; perceptual duplicate checks are added with the renderer in V1.

The real-corpus milestone requires a separate decision record covering license,
redistribution, document identifiers, duplicate policy, and document-level splitting.

## Baselines and treatments

1. Frozen pretrained image-text encoder with zero-shot similarity.
2. Projection-only contrastive adapter with random in-batch negatives.
3. The same adapter with relation-preserving hard negatives.
4. Parameter-efficient encoder tuning only if projection-only training is insufficient.

No architecture may receive different evaluation candidates or split membership.

## Metrics and slices

Primary metrics, in both retrieval directions:

- Recall@1, Recall@5, and Recall@10;
- median rank;
- mean reciprocal rank as a diagnostic.

Report slices by chart type, relation type, generator family, visual style, and domain
status. Bootstrap paired query-level differences when query count supports it. For
stochastic training, use one development seed and at least three final seeds when the
compute ceiling permits; otherwise disclose the limitation.

## Hard-negative ablation

A hard negative preserves chart type and surface vocabulary while changing the relation
needed to match the figure (for example, increasing versus decreasing, or A greater than
B versus B greater than A). It must not be a valid caption for the query figure. Random
and hard-negative treatments use the same batch count, optimizer steps, model capacity,
and positive pairs.

## Compute ceiling

- V0 audit and tests: CPU, less than one minute on the reference development machine.
- V1 smoke run: CPU-capable tiny fixture.
- V1 meaningful comparison: one GPU, target of at most 12 GPU-hours per final seed.
- Full three-seed baseline/ablation matrix: target ceiling of 144 GPU-hours total.

Actual hardware, wall time, peak memory, energy/cost when available, and aborted runs are
recorded. The ceiling is a gate, not a promised spend.

## Acceptance criteria

V0 is complete when:

1. provenance and schema validation are executable;
2. group-aware splitting is deterministic and tested for group and duplicate leakage;
3. retrieval metrics agree with hand-computed fixtures;
4. a versioned synthetic fixture passes the audit from one command;
5. the real-corpus choice remains explicitly unresolved pending a license audit.

V1 is complete only when:

1. one command prepares a versioned dataset and another trains/evaluates;
2. the frozen baseline and trained models share the identical protocol;
3. random-versus-hard negatives are compared under matched compute;
4. at least one held-out-family or held-out-style slice is reported;
5. a results table, confidence/variance estimate, retrieval examples, near-duplicate
   analysis, compute disclosure, and failure taxonomy are based on actual runs.

## Stop/go gate for grounded QA

Grounded QA starts only after V1 produces a repeatable gain on at least one primary
retrieval metric without a severe domain-shift regression. If it does not, the next work
is representation failure analysis, not adding a generative model.

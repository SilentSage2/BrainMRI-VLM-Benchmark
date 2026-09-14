# SciVLM Lab

Research-grade evaluation of scientific figure-text retrieval under source leakage,
hard negatives, and domain shift.

> **Status: V0 protocol scaffold.** No model result is claimed yet. The checked-in
> code defines the data contract, deterministic fingerprints, group-aware splits,
> duplicate-leakage audit, and hand-verifiable retrieval metrics required before V1.

## Research question

When do structure-aware contrastive training and hard-negative selection improve
scientific figure-text retrieval over a frozen pretrained encoder, and do any gains
survive unseen generator families and visual styles?

The later grounded-QA milestone is conditional: it begins only if a retrieval change
produces a reproducible representation gain rather than a template-matching artifact.

## V0–V1 hypothesis

Under group-isolated evaluation, hard negatives that preserve chart type while changing
the underlying relation should improve Recall@10 on relational captions more than random
negatives. The effect may reverse on shifted domains if the model learns renderer or
caption-template shortcuts; held-out families and counterfactual pairs test that failure.

## Evaluation shape

```text
declarative figure specs -> render + caption -> provenance/duplicate audit
                                               |
                              group/family-aware split
                                               |
           frozen encoder ----+---- projection-only ---- contrastive tuning
                                               |
                  retrieval metrics + shift slices + failure review
```

The initial corpus is generated from versioned declarative scientific-chart specs. This
makes causal counterfactuals, exact provenance, and family-held-out splits possible. A
real scientific figure-caption corpus is an external-validity milestone and will not be
selected until license, redistribution, provenance, and document-level split constraints
are recorded.

## Quick start

Python 3.12 is the reference runtime.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/ruff check .
.venv/bin/mypy src tests
.venv/bin/pytest
.venv/bin/scivlm-v0-audit
```

The audit command uses only deterministic in-memory metadata. It does not download data
or create model artifacts.

## Planned comparison

| System | Training | Negative policy | Status |
|---|---|---|---|
| Frozen pretrained image-text encoder | none | none | Planned V1 baseline |
| Projection-only adapter | paired specs | random in-batch | Planned |
| Projection-only adapter | paired specs | relation-preserving hard | Planned ablation |
| Parameter-efficient encoder tuning | paired specs | best V1 policy | Conditional |

Primary metrics are text-to-image and image-to-text Recall@1/5/10 and median rank.
Secondary analysis covers chart type, relation type, template family, visual style,
near-duplicate density, and shifted-domain slices. Final comparisons use repeated seeds
or bootstrap confidence intervals as appropriate.

## Scope and non-goals

V0 does not train a model, render a large corpus, download an external dataset, implement
grounded QA, or build an application. See [the experiment specification](docs/EXPERIMENT_SPEC.md),
[data card](docs/DATA_CARD.md), and [evaluation contract](docs/EVALUATION.md).

## Repository layout

```text
src/scivlm/       typed data, split, audit, fingerprint, and metric core
configs/          versioned experiment configuration
tests/            correctness-critical unit tests
experiments/      compact manifests and result summaries only
docs/             protocol, data, architecture, and evaluation records
```

## Compute and limitations

V0 requires CPU only. V1 must fit on one accessible GPU, with a CPU fixture run in CI.
The synthetic-first design improves control but can exaggerate regularity and cannot
establish performance on real scientific literature. Results remain unclaimed until
the planned baselines and ablations run under one frozen protocol.

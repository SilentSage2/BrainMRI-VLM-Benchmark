# Evaluation Contract

Retrieval is evaluated in both directions against a frozen candidate set. A query may
have multiple relevant candidates, and the rank of its first relevant candidate defines
Recall@K, reciprocal rank, and median rank. Queries without any relevant candidate are a
protocol error rather than silently scored as zero.

Before scoring, the evaluator must verify:

1. unique query and candidate identifiers;
2. at least one relevant candidate for every query;
3. no source group occurs in multiple splits;
4. no exact specification or rendered-image fingerprint crosses splits;
5. candidate sets and relevance judgments are identical across compared systems.

Ties use a deterministic candidate-ID ordering fixed before model comparison. Both macro
query metrics and declared slices are reported; micro-averaging across captions is not a
substitute for source-group isolation.

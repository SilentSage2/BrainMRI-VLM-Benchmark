"""Hand-verifiable retrieval metrics over explicit rankings."""

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    recall_at: dict[int, float]
    mean_reciprocal_rank: float
    median_rank: float
    query_count: int


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    relevant: Mapping[str, frozenset[str]],
    *,
    ks: Sequence[int] = (1, 5, 10),
) -> RetrievalMetrics:
    """Score the first relevant result for every query."""
    if not rankings:
        raise ValueError("at least one query is required")
    if set(rankings) != set(relevant):
        raise ValueError("rankings and relevance must contain identical query IDs")
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")

    first_ranks: list[int] = []
    for query_id, ranked_ids in rankings.items():
        targets = relevant[query_id]
        if not targets:
            raise ValueError(f"query {query_id!r} has no relevant candidates")
        if len(ranked_ids) != len(set(ranked_ids)):
            raise ValueError(f"query {query_id!r} contains duplicate ranked candidates")
        try:
            first_rank = next(
                rank
                for rank, candidate_id in enumerate(ranked_ids, start=1)
                if candidate_id in targets
            )
        except StopIteration as error:
            raise ValueError(
                f"query {query_id!r} has no relevant candidate in its ranking"
            ) from error
        first_ranks.append(first_rank)

    recall_at = {
        k: sum(rank <= k for rank in first_ranks) / len(first_ranks)
        for k in sorted(set(ks))
    }
    return RetrievalMetrics(
        recall_at=recall_at,
        mean_reciprocal_rank=sum(1.0 / rank for rank in first_ranks) / len(first_ranks),
        median_rank=float(statistics.median(first_ranks)),
        query_count=len(first_ranks),
    )

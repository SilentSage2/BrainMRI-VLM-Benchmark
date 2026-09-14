import pytest

from scivlm.metrics import evaluate_rankings


def test_retrieval_metrics_match_hand_computation() -> None:
    metrics = evaluate_rankings(
        {
            "q1": ("a", "b", "c"),
            "q2": ("d", "e", "f"),
            "q3": ("g", "h", "i"),
        },
        {
            "q1": frozenset({"a"}),
            "q2": frozenset({"e"}),
            "q3": frozenset({"i"}),
        },
        ks=(1, 2, 3),
    )
    assert metrics.recall_at == {1: 1 / 3, 2: 2 / 3, 3: 1.0}
    assert metrics.mean_reciprocal_rank == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)
    assert metrics.median_rank == 2.0
    assert metrics.query_count == 3


def test_missing_relevant_candidate_is_a_protocol_error() -> None:
    with pytest.raises(ValueError, match="no relevant candidate"):
        evaluate_rankings({"q": ("a", "b")}, {"q": frozenset({"missing"})})


def test_duplicate_ranked_candidate_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_rankings({"q": ("a", "a")}, {"q": frozenset({"a"})})

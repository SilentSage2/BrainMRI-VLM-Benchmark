from pathlib import Path

import pytest

from mri_vlm.conditions import condition_id, modality_conditions
from mri_vlm.heldout_once_cli import (
    AUTHORIZATION_TEXT,
    aggregate_heldout,
    initialize_lock,
    validate_authorization,
)


def test_heldout_requires_exact_authorization(tmp_path: Path) -> None:
    path = tmp_path / "authorization.txt"
    path.write_text("no", encoding="utf-8")
    with pytest.raises(PermissionError, match="exact"):
        validate_authorization(path)
    path.write_text(AUTHORIZATION_TEXT, encoding="utf-8")
    validate_authorization(path)


def test_one_shot_lock_rejects_completed_run(tmp_path: Path) -> None:
    output = tmp_path / "heldout_v1"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    config = {"protocol": "test"}
    lock = initialize_lock(output, config, manifest)
    lock.write_text('{"status":"complete","protocol":"test"}', encoding="utf-8")
    with pytest.raises(FileExistsError, match="already complete"):
        initialize_lock(output, config, manifest)


def test_heldout_aggregate_freezes_secondary_metrics_and_profile_bins() -> None:
    subject_ids = ["case-a", "case-b", "case-c"]
    grounded_scores = [0.5, 0.5, 0.5]
    dropout_scores = [0.9, 0.6, 0.3]
    no_dropout_scores = [0.8, 0.7, 0.4]
    grounded = _grounded_evaluation(subject_ids, grounded_scores)
    dropout = _modular_evaluation(subject_ids, dropout_scores, coverage=0.0)
    no_dropout = _modular_evaluation(subject_ids, no_dropout_scores, coverage=0.5)
    evaluations = {
        "question_grounded": [grounded, grounded],
        "modular_dropout": [dropout, dropout],
        "modular_no_dropout": [no_dropout, no_dropout],
    }

    result = aggregate_heldout(evaluations, bootstrap_seed=7, bootstrap_samples=200)

    comparisons = result["primary_and_secondary_comparisons"]
    assert comparisons["modular_dropout_minus_question_grounded_missing_accuracy"][
        "mean"
    ] == pytest.approx(0.1)
    assert comparisons["modular_dropout_minus_no_dropout_missing_accuracy"][
        "mean"
    ] == pytest.approx(-1 / 30)
    assert result["modular_dropout_subject_profile_bins"]["counts"] == {
        "success_ge_0.80": 1,
        "boundary_gt_0.40_lt_0.80": 1,
        "failure_le_0.40": 1,
    }
    dropout_summary = result["systems"]["modular_dropout"]
    assert dropout_summary["full_input"]["selective_coverage"]["mean"] == 0.0
    assert dropout_summary["full_input"]["selective_answer_accuracy"]["mean"] is None


def test_heldout_aggregate_rejects_subject_order_mismatch() -> None:
    grounded = _grounded_evaluation(["case-b", "case-a"], [0.5, 0.5])
    modular = _modular_evaluation(["case-a", "case-b"], [0.8, 0.4], coverage=0.0)
    with pytest.raises(ValueError, match="subject order mismatch"):
        aggregate_heldout(
            {
                "question_grounded": [grounded],
                "modular_dropout": [modular],
                "modular_no_dropout": [modular],
            },
            bootstrap_seed=7,
            bootstrap_samples=20,
        )


def _grounded_evaluation(subject_ids: list[str], scores: list[float]) -> dict[str, object]:
    return {
        condition_id(condition): {
            "subject_ids": subject_ids,
            "subject_answer_scores": scores,
            "answer_accuracy": sum(scores) / len(scores),
            "balanced_answer_accuracy": 0.5,
            "ece_5bin": 0.1,
            "grounded_answer_accuracy": 0.4,
            "mean_evidence_dice_by_type": {"laterality": 0.3},
        }
        for condition in modality_conditions()
    }


def _modular_evaluation(
    subject_ids: list[str], scores: list[float], *, coverage: float
) -> dict[str, object]:
    return {
        condition_id(condition): {
            "subject_ids": subject_ids,
            "subject_symbolic_scores": scores,
            "symbolic_answer_accuracy": sum(scores) / len(scores),
            "symbolic_balanced_accuracy": 0.6,
            "enhancing_fraction_mae": 0.1,
            "segmentation_confidence_proxy_ece_5bin": 0.2,
            "selective_coverage": coverage,
            "selective_answer_accuracy": None if coverage == 0.0 else 0.7,
            "mean_region_dice": {
                "whole_tumor": 0.8,
                "tumor_core": 0.7,
                "enhancing_tumor": 0.6,
            },
        }
        for condition in modality_conditions()
    }

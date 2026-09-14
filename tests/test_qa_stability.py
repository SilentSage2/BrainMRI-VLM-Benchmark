import pytest
import torch

from mri_vlm.qa_stability_cli import LabelMeasurements, measure_label, summarize


def test_measure_label_uses_last_axis_for_left_right() -> None:
    label = torch.zeros(4, 4, 4, dtype=torch.long)
    label[0, 0, 0] = 1
    label[0, 0, 3] = 3
    measured = measure_label(label, voxel_ml=0.2)
    assert measured.whole_ml == pytest.approx(0.4)
    assert measured.edema_ml == pytest.approx(0.2)
    assert measured.core_ml == pytest.approx(0.2)
    assert measured.left_ml == pytest.approx(0.2)
    assert measured.right_ml == pytest.approx(0.2)


def test_categorical_answers_and_margins() -> None:
    measured = LabelMeasurements(
        whole_ml=10.0,
        edema_ml=4.0,
        core_ml=6.0,
        enhancing_ml=0.2,
        left_ml=7.0,
        right_ml=3.0,
    )
    assert measured.categorical_answers() == {
        "presence": "yes",
        "laterality": "left",
        "relative_volume": "tumor core",
        "cross_region_comparison": "yes",
    }
    assert measured.boundary_margins()["cross_region_margin_fraction"] == pytest.approx(0.1)


def test_summary_separates_train_and_validation() -> None:
    base = {
        "resolution": 24,
        "enhancing_fraction_absolute_error": 0.01,
        "presence_margin_ml": 0.2,
        "laterality_margin_fraction": 0.2,
        "relative_volume_margin_fraction": 0.2,
        "cross_region_margin_fraction": 0.1,
        "presence_reference": "yes",
        "laterality_reference": "left",
        "relative_volume_reference": "edema",
        "cross_region_comparison_reference": "no",
        "presence_stable": True,
        "laterality_stable": True,
        "relative_volume_stable": True,
        "cross_region_comparison_stable": True,
    }
    rows = [
        {"case_id": "a", "split": "train", **base},
        {
            "case_id": "b",
            "split": "validation",
            **base,
            "relative_volume_stable": False,
        },
    ]
    summary = summarize(rows, resolutions=(24,), seed=7)
    resolution = summary["by_resolution"]["24"]  # type: ignore[index]
    assert resolution["all_development"]["all_categorical_stable"] == 0.5
    assert resolution["by_split"]["train"]["all_categorical_stable"] == 1.0
    assert summary["test_cases_read"] == 0
    screen = summary["train_derived_ambiguity_screen"]["24"]  # type: ignore[index]
    assert screen["relative_volume"]["validation_retained_cases"] == 1

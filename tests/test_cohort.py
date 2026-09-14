import pytest

from mri_vlm.cohort import (
    TumorBurden,
    burden_from_counts,
    percentile,
    select_median_burden_case,
)
from mri_vlm.schema import Split


def test_burden_uses_physical_voxel_volume() -> None:
    row = burden_from_counts(
        case_id="case-1",
        split=Split.TRAIN,
        label_counts={0: 100, 1: 20, 2: 10, 3: 5},
        voxel_volume_mm3=2.0,
    )

    assert row.edema_ml == pytest.approx(0.04)
    assert row.non_enhancing_ml == pytest.approx(0.02)
    assert row.enhancing_ml == pytest.approx(0.01)
    assert row.whole_tumor_ml == pytest.approx(0.07)
    assert row.enhancing_fraction == pytest.approx(1.0 / 7.0)


def test_burden_rejects_empty_tumor() -> None:
    with pytest.raises(ValueError, match="whole-tumor"):
        burden_from_counts(
            case_id="case-1",
            split=Split.TEST,
            label_counts={0: 100},
            voxel_volume_mm3=1.0,
        )


def test_burden_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="unexpected"):
        burden_from_counts(
            case_id="case-1",
            split=Split.TEST,
            label_counts={0: 100, 4: 1},
            voxel_volume_mm3=1.0,
        )


def test_tumor_burden_rejects_negative_volume() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TumorBurden("case-1", Split.TRAIN, -1.0, 2.0, 3.0)


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [(0.0, 0.0), (0.25, 7.5), (0.5, 15.0), (0.75, 22.5), (1.0, 30.0)],
)
def test_percentile_interpolates(fraction: float, expected: float) -> None:
    assert percentile([0.0, 10.0, 20.0, 30.0], fraction) == pytest.approx(expected)


def test_representative_case_uses_validation_median_and_stable_tie_break() -> None:
    rows = (
        TumorBurden("train", Split.TRAIN, 10.0, 0.0, 0.0),
        TumorBurden("z-case", Split.VALIDATION, 20.0, 0.0, 0.0),
        TumorBurden("a-case", Split.VALIDATION, 10.0, 0.0, 0.0),
    )

    assert select_median_burden_case(rows).case_id == "a-case"

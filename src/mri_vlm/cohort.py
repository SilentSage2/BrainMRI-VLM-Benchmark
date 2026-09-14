"""Pure cohort-statistic contracts for mask-derived MRI tasks."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mri_vlm.schema import Split


@dataclass(frozen=True, slots=True)
class TumorBurden:
    case_id: str
    split: Split
    edema_ml: float
    non_enhancing_ml: float
    enhancing_ml: float

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must be non-empty")
        if min(self.edema_ml, self.non_enhancing_ml, self.enhancing_ml) < 0.0:
            raise ValueError("region volumes must be non-negative")
        if self.whole_tumor_ml <= 0.0:
            raise ValueError("whole-tumor volume must be positive")

    @property
    def whole_tumor_ml(self) -> float:
        return self.edema_ml + self.non_enhancing_ml + self.enhancing_ml

    @property
    def enhancing_fraction(self) -> float:
        return self.enhancing_ml / self.whole_tumor_ml


def burden_from_counts(
    *,
    case_id: str,
    split: Split,
    label_counts: Mapping[int, int],
    voxel_volume_mm3: float,
) -> TumorBurden:
    """Convert MSD labels 1/2/3 into physical subregion volumes."""
    if voxel_volume_mm3 <= 0.0:
        raise ValueError("voxel volume must be positive")
    if any(label not in {0, 1, 2, 3} for label in label_counts):
        raise ValueError("unexpected mask label")
    if any(count < 0 for count in label_counts.values()):
        raise ValueError("label counts must be non-negative")
    scale = voxel_volume_mm3 / 1000.0
    return TumorBurden(
        case_id=case_id,
        split=split,
        edema_ml=label_counts.get(1, 0) * scale,
        non_enhancing_ml=label_counts.get(2, 0) * scale,
        enhancing_ml=label_counts.get(3, 0) * scale,
    )


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linearly interpolated percentile without a numerical-library dependency."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def select_median_burden_case(
    rows: Sequence[TumorBurden], *, split: Split = Split.VALIDATION
) -> TumorBurden:
    """Select a representative case by a frozen, outcome-independent rule."""
    eligible = [row for row in rows if row.split is split]
    if not eligible:
        raise ValueError(f"no cases available in split {split.value}")
    target = percentile([row.whole_tumor_ml for row in eligible], 0.5)
    return min(eligible, key=lambda row: (abs(row.whole_tumor_ml - target), row.case_id))

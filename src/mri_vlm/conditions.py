"""Predeclared non-empty MRI modality conditions."""

from itertools import combinations

from mri_vlm.schema import Modality


def modality_conditions() -> tuple[frozenset[Modality], ...]:
    """Return all 15 non-empty subsets in stable size/name order."""
    ordered = tuple(Modality)
    return tuple(
        frozenset(condition)
        for size in range(1, len(ordered) + 1)
        for condition in combinations(ordered, size)
    )


def condition_id(condition: frozenset[Modality]) -> str:
    if not condition:
        raise ValueError("a modality condition must not be empty")
    return "+".join(sorted(modality.value for modality in condition))

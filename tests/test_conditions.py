import pytest

from mri_vlm.conditions import condition_id, modality_conditions
from mri_vlm.schema import ALL_MODALITIES, Modality


def test_all_non_empty_modality_conditions_are_enumerated_once() -> None:
    conditions = modality_conditions()
    assert len(conditions) == 15
    assert len(set(conditions)) == 15
    assert ALL_MODALITIES in conditions
    assert all(condition for condition in conditions)


def test_condition_identifier_is_stable() -> None:
    assert condition_id(frozenset({Modality.T2, Modality.FLAIR})) == "flair+t2"
    with pytest.raises(ValueError, match="must not be empty"):
        condition_id(frozenset())

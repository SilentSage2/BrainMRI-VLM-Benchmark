import pytest

from mri_vlm.models import M3D_LAMED_PHI3, MRI_VLM_SMALL, validate_comparison


def test_external_baseline_capabilities_are_explicit() -> None:
    assert M3D_LAMED_PHI3.supports_3d
    assert not M3D_LAMED_PHI3.supports_multiple_sequences
    assert M3D_LAMED_PHI3.requires_remote_code_review


def test_mismatched_sequence_inputs_cannot_be_called_matched() -> None:
    with pytest.raises(ValueError, match="equivalent MRI sequence"):
        validate_comparison(M3D_LAMED_PHI3, MRI_VLM_SMALL)


def test_controlled_model_supports_headline_task() -> None:
    assert MRI_VLM_SMALL.supports_multiple_sequences
    assert MRI_VLM_SMALL.supports_voxel_grounding

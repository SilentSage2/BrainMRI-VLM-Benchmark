import pytest
import torch

from mri_vlm.segmentation import ResidualUNet3D


def test_residual_unet_preserves_spatial_shape() -> None:
    model = ResidualUNet3D(width=4)
    output = model(torch.randn(2, 4, 16, 24, 32))
    assert output.shape == (2, 4, 16, 24, 32)


def test_residual_unet_rejects_nondivisible_shape() -> None:
    model = ResidualUNet3D(width=4)
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(1, 4, 15, 16, 16))


def test_residual_unet_rejects_invalid_width() -> None:
    with pytest.raises(ValueError, match="width"):
        ResidualUNet3D(width=6)

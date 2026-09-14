"""Residual 3D U-Net used by the missing-contrast modular MR baseline."""

from typing import cast

import torch
from torch import Tensor, nn


class ResidualUNet3D(nn.Module):
    """Three-level residual 3D U-Net with normalization suitable for batch size one."""

    def __init__(self, *, in_channels: int = 4, classes: int = 4, width: int = 8) -> None:
        super().__init__()
        if in_channels <= 0 or classes <= 1 or width < 4 or width % 4:
            raise ValueError("invalid channels/classes or width not divisible by four")
        self.encoder1 = ResidualBlock(in_channels, width)
        self.down1 = DownBlock(width, 2 * width)
        self.down2 = DownBlock(2 * width, 4 * width)
        self.down3 = DownBlock(4 * width, 8 * width)
        self.up3 = UpBlock(8 * width, 4 * width)
        self.up2 = UpBlock(4 * width, 2 * width)
        self.up1 = UpBlock(2 * width, width)
        self.output = nn.Conv3d(width, classes, kernel_size=1)

    def forward(self, volumes: Tensor) -> Tensor:
        if volumes.ndim != 5 or volumes.shape[1] != 4:
            raise ValueError("volumes must have shape [batch, 4, depth, height, width]")
        if any(dimension % 8 for dimension in volumes.shape[-3:]):
            raise ValueError("spatial dimensions must be divisible by eight")
        level1 = self.encoder1(volumes)
        level2 = self.down1(level1)
        level3 = self.down2(level2)
        bottleneck = self.down3(level3)
        decoded = self.up3(bottleneck, level3)
        decoded = self.up2(decoded, level2)
        decoded = self.up1(decoded, level1)
        return cast(Tensor, self.output(decoded))


class ResidualBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(inputs, outputs, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, outputs),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(outputs, outputs, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, outputs),
        )
        self.skip: nn.Module = (
            nn.Identity()
            if inputs == outputs
            else nn.Conv3d(inputs, outputs, kernel_size=1, bias=False)
        )
        self.activation = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.activation(self.body(inputs) + self.skip(inputs)))


class DownBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.down = nn.Conv3d(inputs, outputs, kernel_size=2, stride=2, bias=False)
        self.block = ResidualBlock(outputs, outputs)

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.block(self.down(inputs)))


class UpBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(inputs, outputs, kernel_size=2, stride=2)
        self.block = ResidualBlock(2 * outputs, outputs)

    def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
        upsampled = self.up(inputs)
        if upsampled.shape[-3:] != skip.shape[-3:]:
            raise ValueError("decoder and skip shapes do not agree")
        return cast(Tensor, self.block(torch.cat((upsampled, skip), dim=1)))

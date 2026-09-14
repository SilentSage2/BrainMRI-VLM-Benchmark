"""A compact sequence-aware 3D VLM for controlled CPU/GPU experiments."""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from mri_vlm.schema import Modality


@dataclass(frozen=True, slots=True)
class MRIModelOutput:
    answer_logits: Tensor
    evidence_logits: Tensor


class MRIVLMSmall(nn.Module):
    """Fuse available 3D MRI sequences with tokenized questions.

    This is the controlled architecture for answer-only versus grounded training. It is
    intentionally small; model size can change only through versioned configuration.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        answer_classes: int,
        width: int = 16,
        question_conditioned_evidence: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size <= 1 or answer_classes <= 1 or width <= 0:
            raise ValueError("vocab, answer classes, and width must be greater than one")
        self.modality_count = len(Modality)
        self.question_conditioned_evidence = question_conditioned_evidence
        self.visual_encoder = nn.Sequential(
            nn.Conv3d(1, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.modality_embeddings = nn.Parameter(torch.empty(self.modality_count, width))
        self.question_embeddings = nn.Embedding(vocab_size, width, padding_idx=0)
        self.fusion = nn.Linear(2 * width, width)
        self.answer_head = nn.Linear(width, answer_classes)
        self.evidence_conditioner = nn.Linear(width, 2 * width)
        self.evidence_head = nn.Conv3d(width, 1, kernel_size=1)
        nn.init.normal_(self.modality_embeddings, std=0.02)

    def forward(
        self,
        volumes: Tensor,
        modality_mask: Tensor,
        question_tokens: Tensor,
    ) -> MRIModelOutput:
        self._validate_inputs(volumes, modality_mask, question_tokens)
        batch, modalities, depth, height, width = volumes.shape
        encoded = self.visual_encoder(volumes.reshape(batch * modalities, 1, depth, height, width))
        channels = encoded.shape[1]
        encoded = encoded.reshape(batch, modalities, channels, depth, height, width)

        available = modality_mask.to(dtype=encoded.dtype)
        spatial_weights = available[:, :, None, None, None, None]
        available_count = available.sum(dim=1, keepdim=True)
        fused_spatial = (encoded * spatial_weights).sum(dim=1) / available_count[
            :, :, None, None, None
        ]

        pooled = encoded.mean(dim=(-3, -2, -1)) + self.modality_embeddings[None, :, :]
        fused_visual = (pooled * available[:, :, None]).sum(dim=1) / available_count

        token_mask = question_tokens.ne(0)
        question_encoded = self.question_embeddings(question_tokens)
        question_count = token_mask.sum(dim=1, keepdim=True).clamp_min(1)
        question_pooled = (question_encoded * token_mask[:, :, None]).sum(dim=1)
        question_pooled = question_pooled / question_count

        joint = torch.tanh(self.fusion(torch.cat((fused_visual, question_pooled), dim=1)))
        answer_logits = self.answer_head(joint)
        conditioned = fused_spatial
        if self.question_conditioned_evidence:
            scale, shift = self.evidence_conditioner(joint).chunk(2, dim=1)
            conditioned = conditioned * (1.0 + scale[:, :, None, None, None])
            conditioned = conditioned + shift[:, :, None, None, None]
        evidence_logits = self.evidence_head(conditioned).squeeze(1)
        return MRIModelOutput(answer_logits=answer_logits, evidence_logits=evidence_logits)

    def _validate_inputs(
        self, volumes: Tensor, modality_mask: Tensor, question_tokens: Tensor
    ) -> None:
        if volumes.ndim != 5 or volumes.shape[1] != self.modality_count:
            raise ValueError("volumes must have shape [batch, 4, depth, height, width]")
        if modality_mask.shape != volumes.shape[:2]:
            raise ValueError("modality_mask must have shape [batch, 4]")
        if question_tokens.ndim != 2 or question_tokens.shape[0] != volumes.shape[0]:
            raise ValueError("question_tokens must have shape [batch, tokens]")
        if not torch.all(modality_mask.sum(dim=1) > 0):
            raise ValueError("every example must contain at least one MRI sequence")
        if not torch.all(question_tokens.ne(0).any(dim=1)):
            raise ValueError("every example must contain a non-padding question token")


class MRIVLM3D(nn.Module):
    """Coordinate-aware hierarchical 3D VLM used for matched real-data runs."""

    def __init__(
        self,
        *,
        vocab_size: int,
        answer_classes: int,
        width: int = 8,
        question_conditioned_evidence: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size <= 1 or answer_classes <= 1 or width < 4 or width % 4:
            raise ValueError("invalid vocabulary/classes or width not divisible by four")
        features = 4 * width
        self.modality_count = len(Modality)
        self.question_conditioned_evidence = question_conditioned_evidence
        self.visual_encoder = nn.Sequential(
            _conv_norm_activation(4, width, stride=1),
            _conv_norm_activation(width, 2 * width, stride=2),
            _conv_norm_activation(2 * width, features, stride=2),
            _conv_norm_activation(features, features, stride=1),
        )
        self.modality_embeddings = nn.Parameter(torch.empty(self.modality_count, features))
        self.question_embeddings = nn.Embedding(vocab_size, features, padding_idx=0)
        self.question_encoder = nn.GRU(features, features, batch_first=True)
        self.fusion = nn.Linear(2 * features, features)
        self.answer_head = nn.Linear(features, answer_classes)
        self.evidence_conditioner = nn.Linear(features, 2 * features)
        self.evidence_head = nn.Conv3d(features, 1, kernel_size=1)
        nn.init.normal_(self.modality_embeddings, std=0.02)

    def forward(
        self, volumes: Tensor, modality_mask: Tensor, question_tokens: Tensor
    ) -> MRIModelOutput:
        _validate_common_inputs(volumes, modality_mask, question_tokens, self.modality_count)
        batch, modalities, depth, height, width = volumes.shape
        coordinates = _coordinates(
            batch * modalities,
            depth,
            height,
            width,
            device=volumes.device,
            dtype=volumes.dtype,
        )
        visual_input = torch.cat(
            (volumes.reshape(batch * modalities, 1, depth, height, width), coordinates), dim=1
        )
        encoded = self.visual_encoder(visual_input)
        channels, low_d, low_h, low_w = encoded.shape[1:]
        encoded = encoded.reshape(batch, modalities, channels, low_d, low_h, low_w)
        available = modality_mask.to(dtype=encoded.dtype)
        available_count = available.sum(dim=1, keepdim=True)
        fused_spatial = (encoded * available[:, :, None, None, None, None]).sum(dim=1)
        fused_spatial = fused_spatial / available_count[:, :, None, None, None]
        pooled = encoded.mean(dim=(-3, -2, -1)) + self.modality_embeddings[None]
        fused_visual = (pooled * available[:, :, None]).sum(dim=1) / available_count

        embedded = self.question_embeddings(question_tokens)
        encoded_question, _ = self.question_encoder(embedded)
        lengths = question_tokens.ne(0).sum(dim=1) - 1
        question = encoded_question[torch.arange(batch, device=question_tokens.device), lengths]
        joint = torch.tanh(self.fusion(torch.cat((fused_visual, question), dim=1)))
        answer_logits = self.answer_head(joint)
        conditioned = fused_spatial
        if self.question_conditioned_evidence:
            scale, shift = self.evidence_conditioner(joint).chunk(2, dim=1)
            conditioned = conditioned * (1.0 + scale[:, :, None, None, None])
            conditioned = conditioned + shift[:, :, None, None, None]
        evidence_logits = self.evidence_head(conditioned)
        evidence_logits = functional.interpolate(
            evidence_logits,
            size=(depth, height, width),
            mode="trilinear",
            align_corners=False,
        ).squeeze(1)
        return MRIModelOutput(answer_logits=answer_logits, evidence_logits=evidence_logits)


class SliceVLM2D(nn.Module):
    """Fixed-policy axial-slice VLM baseline for the same four-contrast QA task."""

    def __init__(self, *, vocab_size: int, answer_classes: int, width: int = 8) -> None:
        super().__init__()
        if vocab_size <= 1 or answer_classes <= 1 or width < 4:
            raise ValueError("invalid vocabulary, answer classes, or width")
        features = 4 * width
        self.modality_count = len(Modality)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(2 * self.modality_count, width, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(width, 2 * width, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * width, features, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.slice_embeddings = nn.Parameter(torch.empty(3, features))
        self.question_embeddings = nn.Embedding(vocab_size, features, padding_idx=0)
        self.question_encoder = nn.GRU(features, features, batch_first=True)
        self.fusion = nn.Linear(2 * features, features)
        self.answer_head = nn.Linear(features, answer_classes)
        nn.init.normal_(self.slice_embeddings, std=0.02)

    def forward(
        self, volumes: Tensor, modality_mask: Tensor, question_tokens: Tensor
    ) -> MRIModelOutput:
        _validate_common_inputs(volumes, modality_mask, question_tokens, self.modality_count)
        batch, _, depth, height, width = volumes.shape
        indices = torch.tensor(
            (depth // 4, depth // 2, min(depth - 1, 3 * depth // 4)),
            device=volumes.device,
        )
        slices = volumes.index_select(2, indices).permute(0, 2, 1, 3, 4)
        available = modality_mask.to(dtype=volumes.dtype)
        slices = slices * available[:, None, :, None, None]
        mask_planes = available[:, None, :, None, None].expand(-1, 3, -1, height, width)
        visual_input = torch.cat((slices, mask_planes), dim=2).reshape(
            batch * 3, 2 * self.modality_count, height, width
        )
        encoded = self.visual_encoder(visual_input).flatten(1).reshape(batch, 3, -1)
        visual = (encoded + self.slice_embeddings[None]).mean(dim=1)
        embedded = self.question_embeddings(question_tokens)
        encoded_question, _ = self.question_encoder(embedded)
        lengths = question_tokens.ne(0).sum(dim=1) - 1
        question = encoded_question[torch.arange(batch, device=volumes.device), lengths]
        joint = torch.tanh(self.fusion(torch.cat((visual, question), dim=1)))
        answer_logits = self.answer_head(joint)
        evidence_logits = torch.zeros(
            batch, depth, height, width, dtype=volumes.dtype, device=volumes.device
        )
        return MRIModelOutput(answer_logits=answer_logits, evidence_logits=evidence_logits)


def _conv_norm_activation(inputs: int, outputs: int, *, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(inputs, outputs, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(4, outputs),
        nn.GELU(),
    )


def _coordinates(
    batch: int,
    depth: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    axes = [
        torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=dtype)
        for size in (depth, height, width)
    ]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1, -1)


def _validate_common_inputs(
    volumes: Tensor, modality_mask: Tensor, question_tokens: Tensor, modality_count: int
) -> None:
    if volumes.ndim != 5 or volumes.shape[1] != modality_count:
        raise ValueError("volumes must have shape [batch, 4, depth, height, width]")
    if modality_mask.shape != volumes.shape[:2]:
        raise ValueError("modality_mask must have shape [batch, 4]")
    if question_tokens.ndim != 2 or question_tokens.shape[0] != volumes.shape[0]:
        raise ValueError("question_tokens must have shape [batch, tokens]")
    if not torch.all(modality_mask.sum(dim=1) > 0):
        raise ValueError("every example must contain at least one MRI sequence")
    if not torch.all(question_tokens.ne(0).any(dim=1)):
        raise ValueError("every example must contain a non-padding question token")

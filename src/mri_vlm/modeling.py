"""A compact sequence-aware 3D VLM for controlled CPU/GPU experiments."""

from dataclasses import dataclass

import torch
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

    def __init__(self, *, vocab_size: int, answer_classes: int, width: int = 16) -> None:
        super().__init__()
        if vocab_size <= 1 or answer_classes <= 1 or width <= 0:
            raise ValueError("vocab, answer classes, and width must be greater than one")
        self.modality_count = len(Modality)
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
        fused_spatial = (
            encoded * spatial_weights
        ).sum(dim=1) / available_count[:, :, None, None, None]

        pooled = encoded.mean(dim=(-3, -2, -1)) + self.modality_embeddings[None, :, :]
        fused_visual = (pooled * available[:, :, None]).sum(dim=1) / available_count

        token_mask = question_tokens.ne(0)
        question_encoded = self.question_embeddings(question_tokens)
        question_count = token_mask.sum(dim=1, keepdim=True).clamp_min(1)
        question_pooled = (question_encoded * token_mask[:, :, None]).sum(dim=1)
        question_pooled = question_pooled / question_count

        joint = torch.tanh(self.fusion(torch.cat((fused_visual, question_pooled), dim=1)))
        answer_logits = self.answer_head(joint)
        scale, shift = self.evidence_conditioner(joint).chunk(2, dim=1)
        conditioned = fused_spatial * (1.0 + scale[:, :, None, None, None])
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

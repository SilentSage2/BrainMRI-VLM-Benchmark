import io

import pytest
import torch

from mri_vlm.modeling import MRIVLMSmall


def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    volumes = torch.randn(2, 4, 8, 8, 8)
    modality_mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    question_tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    return volumes, modality_mask, question_tokens


def test_output_shapes_and_gradients() -> None:
    model = MRIVLMSmall(vocab_size=16, answer_classes=5, width=4)
    volumes, modality_mask, question_tokens = inputs()
    output = model(volumes, modality_mask, question_tokens)
    assert output.answer_logits.shape == (2, 5)
    assert output.evidence_logits.shape == (2, 8, 8, 8)
    loss = output.answer_logits.square().mean() + output.evidence_logits.square().mean()
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_masked_sequence_values_cannot_change_outputs() -> None:
    torch.manual_seed(11)
    model = MRIVLMSmall(vocab_size=16, answer_classes=5, width=4).eval()
    volumes, modality_mask, question_tokens = inputs()
    modified = volumes.clone()
    modified[0, 2:] = 10_000.0
    with torch.no_grad():
        original_output = model(volumes, modality_mask, question_tokens)
        modified_output = model(modified, modality_mask, question_tokens)
    assert torch.equal(original_output.answer_logits[0], modified_output.answer_logits[0])
    assert torch.equal(original_output.evidence_logits[0], modified_output.evidence_logits[0])


def test_checkpoint_round_trip() -> None:
    model = MRIVLMSmall(vocab_size=16, answer_classes=5, width=4).eval()
    volumes, modality_mask, question_tokens = inputs()
    expected = model(volumes, modality_mask, question_tokens).answer_logits
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = MRIVLMSmall(vocab_size=16, answer_classes=5, width=4).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    actual = restored(volumes, modality_mask, question_tokens).answer_logits
    assert torch.equal(expected, actual)


def test_all_sequences_missing_is_rejected() -> None:
    model = MRIVLMSmall(vocab_size=16, answer_classes=5, width=4)
    volumes, modality_mask, question_tokens = inputs()
    modality_mask[0] = False
    with pytest.raises(ValueError, match="at least one MRI sequence"):
        model(volumes, modality_mask, question_tokens)

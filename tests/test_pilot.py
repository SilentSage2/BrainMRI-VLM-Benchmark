import torch

from mri_vlm.pilot_cli import UNet3DSegmenter, _ece, _segmentation_loss, _symbolic_answer
from mri_vlm.schema import QuestionType


def test_unet_preserves_spatial_shape() -> None:
    model = UNet3DSegmenter(width=2)
    output = model(torch.randn(1, 4, 8, 8, 8))
    assert output.shape == (1, 4, 8, 8, 8)


def test_symbolic_answers_from_segmentation() -> None:
    label = torch.zeros(4, 4, 4, dtype=torch.long)
    label[:, :, :2] = 1
    label[0, 0, 0] = 3
    assert _symbolic_answer(label, QuestionType.PRESENCE) == "yes"
    assert _symbolic_answer(label, QuestionType.LATERALITY) == "left"
    assert _symbolic_answer(label, QuestionType.RELATIVE_VOLUME) == "edema"
    assert _symbolic_answer(label, QuestionType.UNANSWERABLE) == "abstain"


def test_ece_is_zero_for_matching_bin_accuracy() -> None:
    assert _ece([True, False], [0.5, 0.5], bins=2) == 0.0


def test_segmentation_loss_rewards_correct_subregions() -> None:
    target = torch.tensor([[[[0, 1], [2, 3]], [[0, 1], [2, 3]]]])
    correct = torch.nn.functional.one_hot(target, num_classes=4).permute(0, 4, 1, 2, 3)
    correct_logits = correct.float() * 10.0
    wrong_logits = torch.zeros_like(correct_logits)
    weights = torch.ones(4)
    assert _segmentation_loss(correct_logits, target, weights) < _segmentation_loss(
        wrong_logits, target, weights
    )

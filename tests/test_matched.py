import torch

from mri_vlm.matched_cli import weighted_answer_loss


def test_weighted_answer_loss_affects_single_example() -> None:
    logits = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([1])
    ordinary = weighted_answer_loss(logits, target, torch.ones(2))
    weighted = weighted_answer_loss(logits, target, torch.tensor([1.0, 3.0]))
    assert weighted == 3.0 * ordinary

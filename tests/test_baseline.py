from pathlib import Path

import torch

from mri_vlm.baseline_cli import (
    _balanced_accuracy,
    _brats_regions,
    _enhancing_fraction,
    brats_segmentation_loss,
    select_case_ids,
)


def test_balanced_accuracy_averages_class_recalls() -> None:
    assert _balanced_accuracy([("a", True), ("a", False), ("b", True)]) == 0.75


def test_enhancing_fraction_handles_empty_and_nonempty() -> None:
    empty = torch.zeros(2, 2, 2, dtype=torch.long)
    assert _enhancing_fraction(empty) == 0.0
    label = empty.clone()
    label[0, 0, 0] = 3
    label[0, 0, 1] = 1
    assert _enhancing_fraction(label) == 0.5


def test_brats_regions_use_standard_overlapping_targets() -> None:
    label = torch.tensor([[[0, 1, 2, 3]]])
    regions = _brats_regions(label, label)
    assert int(regions["whole_tumor"][0].sum()) == 3
    assert int(regions["tumor_core"][0].sum()) == 2
    assert int(regions["enhancing_tumor"][0].sum()) == 1


def test_brats_loss_rewards_correct_labels() -> None:
    target = torch.tensor([[[[0, 1], [2, 3]], [[0, 1], [2, 3]]]])
    correct = torch.nn.functional.one_hot(target, num_classes=4).permute(0, 4, 1, 2, 3)
    correct_logits = correct.float() * 10.0
    wrong_logits = torch.zeros_like(correct_logits)
    weights = torch.ones(4)
    assert brats_segmentation_loss(correct_logits, target, weights) < brats_segmentation_loss(
        wrong_logits, target, weights
    )


def test_case_selection_is_seeded(monkeypatch) -> None:
    class Case:
        def __init__(self, case_id: str) -> None:
            self.case_id = case_id

    monkeypatch.setattr(
        "mri_vlm.baseline_cli.discover_training_cases",
        lambda _: tuple(Case(f"case-{index}") for index in range(20)),
    )
    monkeypatch.setattr(
        "mri_vlm.baseline_cli.assign_subject",
        lambda case_id, seed: (
            __import__("mri_vlm.schema", fromlist=["Split"]).Split.TRAIN
            if int(case_id.split("-")[1]) < 15
            else __import__("mri_vlm.schema", fromlist=["Split"]).Split.VALIDATION
        ),
    )
    first = select_case_ids(Path("."), seed=7, train_count=3, validation_count=2)
    second = select_case_ids(Path("."), seed=7, train_count=3, validation_count=2)
    assert first == second

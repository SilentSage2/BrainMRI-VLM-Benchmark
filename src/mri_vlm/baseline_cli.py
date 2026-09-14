"""Train and evaluate a residual 3D missing-contrast segmentation baseline."""

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor

from mri_vlm.conditions import condition_id, modality_conditions
from mri_vlm.controls import bootstrap_mean_ci
from mri_vlm.data.msd import discover_training_cases
from mri_vlm.pilot_cli import _condition_mask, _ece, _symbolic_answer, _tensor_dice
from mri_vlm.preprocess import PreprocessedCase, PreprocessSpec, load_preprocessed_case
from mri_vlm.qa_v1 import (
    FROZEN_FRACTION_THRESHOLDS,
    QATargetV1,
    build_word_vocabulary,
    fraction_bin,
    materialize_targets,
)
from mri_vlm.real_qa import SplitQAExample, generate_real_examples
from mri_vlm.schema import Modality, QuestionType, Split
from mri_vlm.segmentation import ResidualUNet3D
from mri_vlm.split import assign_subject


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train residual 3D modular MR baseline")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed/baseline-v2"))
    parser.add_argument("--train-cases", type=int, default=32, help="zero uses the full split")
    parser.add_argument("--validation-cases", type=int, default=8, help="zero uses full split")
    parser.add_argument("--spatial-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--split-seed", type=int, default=20260914)
    parser.add_argument(
        "--no-modality-dropout", action="store_true", help="train on complete inputs only"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.train_cases < 0 or args.validation_cases < 0 or args.epochs <= 0:
        raise ValueError("case counts must be nonnegative and epochs positive")
    if args.spatial_size % 8:
        raise ValueError("spatial size must be divisible by eight")
    _set_seed(args.seed)
    started = time.monotonic()
    root = args.dataset_root.resolve()
    selected_ids = select_case_ids(
        root,
        seed=args.split_seed,
        train_count=args.train_cases,
        validation_count=args.validation_cases,
    )
    questions = generate_real_examples(
        root,
        seed=args.split_seed,
        include_splits=frozenset({Split.TRAIN, Split.VALIDATION}),
        include_case_ids=frozenset(selected_ids[Split.TRAIN] + selected_ids[Split.VALIDATION]),
    )
    questions_by_case: dict[str, list[SplitQAExample]] = defaultdict(list)
    for question in questions:
        if question.example.question_type in (
            QuestionType.LATERALITY,
            QuestionType.RELATIVE_VOLUME,
            QuestionType.ENHANCING_FRACTION,
        ):
            questions_by_case[question.example.case_id].append(question)

    spec = PreprocessSpec(
        spatial_size=args.spatial_size,
        version="brain-crop-zero-background-zscore-v2",
        preserve_background_zero=True,
        crop_to_foreground=True,
    )
    cache_root = args.cache_dir.resolve()
    datasets = {
        split: load_cases(root, cache_root, selected_ids[split], spec)
        for split in (Split.TRAIN, Split.VALIDATION)
    }
    vocabulary = build_word_vocabulary()
    qa_targets = {
        split: {
            case.case_id: materialize_targets(
                case,
                tuple(questions_by_case[case.case_id]),
                FROZEN_FRACTION_THRESHOLDS,
                vocabulary,
            )
            for case in datasets[split]
        }
        for split in (Split.TRAIN, Split.VALIDATION)
    }
    model = ResidualUNet3D(width=args.width)
    history, best_epoch, best_state = train(
        model,
        datasets[Split.TRAIN],
        datasets[Split.VALIDATION],
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        modality_dropout=not args.no_modality_dropout,
    )
    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "spec": asdict(spec),
            "width": args.width,
            "seed": args.seed,
            "best_epoch": best_epoch,
        },
        args.checkpoint,
    )
    conditions = modality_conditions()
    evaluation = evaluate(
        model,
        datasets[Split.VALIDATION],
        qa_targets[Split.VALIDATION],
        conditions=conditions,
        bootstrap_seed=args.seed,
    )
    full = cast(dict[str, object], evaluation[condition_id(frozenset(Modality))])
    region = cast(dict[str, float], full["mean_region_dice"])
    gates = {
        "full_whole_dice_at_least_0.60": cast(float, full["mean_whole_tumor_dice"]) >= 0.60,
        "full_mean_subregion_dice_at_least_0.40": sum(region.values()) / len(region) >= 0.40,
        "full_symbolic_balanced_accuracy_at_least_0.70": cast(
            float, full["symbolic_balanced_accuracy"]
        )
        >= 0.70,
    }
    output = {
        "status": "pass" if all(gates.values()) else "fail",
        "scope": "single-seed development direction run; no test cases read",
        "seed": args.seed,
        "environment": {
            "torch": torch.__version__,
            "device": "cpu",
            "threads": torch.get_num_threads(),
        },
        "config": {
            "train_cases": len(datasets[Split.TRAIN]),
            "validation_cases": len(datasets[Split.VALIDATION]),
            "epochs": args.epochs,
            "spatial_size": args.spatial_size,
            "width": args.width,
            "learning_rate": args.learning_rate,
            "split_seed": args.split_seed,
            "modality_dropout": not args.no_modality_dropout,
            "preprocessing": asdict(spec),
            "modality_schedule": [condition_id(item) for item in conditions],
        },
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_case_ids": selected_ids[Split.TRAIN],
        "validation_case_ids": selected_ids[Split.VALIDATION],
        "test_cases_read": 0,
        "history": history,
        "best_epoch": best_epoch,
        "evaluation": evaluation,
        "gates": gates,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "wall_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(args.output)}))


def select_case_ids(
    root: Path, *, seed: int, train_count: int, validation_count: int
) -> dict[Split, tuple[str, ...]]:
    grouped: dict[Split, list[str]] = defaultdict(list)
    for case in discover_training_cases(root):
        split = assign_subject(case.case_id, seed=seed)
        if split in (Split.TRAIN, Split.VALIDATION):
            grouped[split].append(case.case_id)
    randomizer = random.Random(seed)
    output: dict[Split, tuple[str, ...]] = {}
    for split, count in ((Split.TRAIN, train_count), (Split.VALIDATION, validation_count)):
        ids = sorted(grouped[split])
        randomizer.shuffle(ids)
        output[split] = tuple(ids if count == 0 else ids[:count])
    return output


def load_cases(
    root: Path,
    cache_root: Path,
    case_ids: tuple[str, ...],
    spec: PreprocessSpec,
) -> tuple[PreprocessedCase, ...]:
    return tuple(load_preprocessed_case(root, cache_root, case_id, spec) for case_id in case_ids)


def train(
    model: ResidualUNet3D,
    train_cases: tuple[PreprocessedCase, ...],
    validation_cases: tuple[PreprocessedCase, ...],
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
    modality_dropout: bool = True,
) -> tuple[list[dict[str, float]], int, dict[str, Tensor]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    weights = torch.tensor((0.05, 1.0, 2.0, 2.0))
    conditions = modality_conditions()
    history: list[dict[str, float]] = []
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, Tensor] = {}
    step = 0
    for epoch in range(1, epochs + 1):
        order = list(range(len(train_cases)))
        random.Random(seed + epoch).shuffle(order)
        losses: list[float] = []
        model.train()
        for index in order:
            case = train_cases[index]
            condition = (
                conditions[step % len(conditions)]
                if modality_dropout
                else frozenset(Modality)
            )
            mask = _condition_mask(condition)[0, :, None, None, None]
            optimizer.zero_grad()
            logits = model((case.volumes * mask).unsqueeze(0))
            loss = brats_segmentation_loss(logits, case.label.unsqueeze(0), weights)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            step += 1
        score = mean_region_dice(model, validation_cases, frozenset(Modality))
        history.append(
            {
                "epoch": float(epoch),
                "mean_train_loss": sum(losses) / len(losses),
                "validation_mean_region_dice": score,
            }
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
    return history, best_epoch, best_state


def mean_region_dice(
    model: ResidualUNet3D,
    cases: tuple[PreprocessedCase, ...],
    condition: frozenset[Modality],
) -> float:
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for case in cases:
            mask = _condition_mask(condition)[0, :, None, None, None]
            prediction = model((case.volumes * mask).unsqueeze(0)).argmax(dim=1).squeeze(0)
            values.extend(
                _tensor_dice(predicted, target)
                for predicted, target in _brats_regions(prediction, case.label).values()
            )
    return sum(values) / len(values)


def evaluate(
    model: ResidualUNet3D,
    cases: tuple[PreprocessedCase, ...],
    targets_by_case: dict[str, tuple[QATargetV1, ...]],
    *,
    conditions: tuple[frozenset[Modality], ...],
    bootstrap_seed: int,
) -> dict[str, object]:
    results: dict[str, object] = {}
    model.eval()
    for condition in conditions:
        whole_dice: list[float] = []
        region_dice: dict[str, list[float]] = {
            "whole_tumor": [],
            "tumor_core": [],
            "enhancing_tumor": [],
        }
        subject_symbolic: list[float] = []
        hits_by_type: dict[str, list[tuple[str, bool]]] = defaultdict(list)
        fraction_errors: list[float] = []
        retained_by_type: dict[str, int] = defaultdict(int)
        all_hits: list[bool] = []
        confidences: list[float] = []
        selective_hits: list[bool] = []
        subject_ids: list[str] = []
        with torch.no_grad():
            for case in cases:
                mask = _condition_mask(condition)[0, :, None, None, None]
                logits = model((case.volumes * mask).unsqueeze(0))
                probabilities = logits.softmax(dim=1)
                prediction = probabilities.argmax(dim=1).squeeze(0)
                foreground = prediction > 0
                voxel_confidence = probabilities.max(dim=1).values.squeeze(0)
                confidence = float(
                    voxel_confidence[foreground].mean()
                    if bool(foreground.any())
                    else voxel_confidence.mean()
                )
                whole_dice.append(_tensor_dice(prediction > 0, case.label > 0))
                for name, (predicted_region, target_region) in _brats_regions(
                    prediction, case.label
                ).items():
                    region_dice[name].append(_tensor_dice(predicted_region, target_region))
                case_hits: list[bool] = []
                for target in targets_by_case[case.case_id]:
                    question_type = target.question_type
                    expected = target.answer
                    predicted = (
                        fraction_bin(
                            _enhancing_fraction(prediction), FROZEN_FRACTION_THRESHOLDS
                        )
                        if question_type is QuestionType.ENHANCING_FRACTION
                        else _symbolic_answer(prediction, question_type)
                    )
                    hit = predicted == expected
                    case_hits.append(hit)
                    all_hits.append(hit)
                    confidences.append(confidence)
                    if confidence >= 0.75:
                        selective_hits.append(hit)
                    hits_by_type[question_type.value].append((expected, hit))
                    retained_by_type[question_type.value] += 1
                    if question_type is QuestionType.ENHANCING_FRACTION:
                        fraction_errors.append(
                            abs(_enhancing_fraction(prediction) - _enhancing_fraction(case.label))
                        )
                if case_hits:
                    subject_symbolic.append(sum(case_hits) / len(case_hits))
                    subject_ids.append(case.case_id)
        balanced_by_type = {
            question: _balanced_accuracy(items) for question, items in hits_by_type.items()
        }
        results[condition_id(condition)] = {
            "subjects": len(cases),
            "mean_whole_tumor_dice": sum(whole_dice) / len(whole_dice),
            "mean_region_dice": {
                name: sum(values) / len(values) for name, values in region_dice.items()
            },
            "symbolic_answer_accuracy": sum(subject_symbolic) / len(subject_symbolic),
            "symbolic_balanced_accuracy": sum(balanced_by_type.values()) / len(balanced_by_type),
            "balanced_accuracy_by_type": balanced_by_type,
            "stable_questions_retained": dict(retained_by_type),
            "subject_bootstrap_95ci": bootstrap_mean_ci(
                subject_symbolic, seed=bootstrap_seed, samples=1000
            ),
            "enhancing_fraction_mae": sum(fraction_errors) / len(fraction_errors),
            "segmentation_confidence_proxy_ece_5bin": _ece(all_hits, confidences, bins=5),
            "abstention_threshold": 0.75,
            "selective_coverage": len(selective_hits) / len(all_hits),
            "selective_answer_accuracy": (
                sum(selective_hits) / len(selective_hits) if selective_hits else None
            ),
            "subject_ids": subject_ids,
            "subject_symbolic_scores": subject_symbolic,
        }
    return results


def _balanced_accuracy(items: list[tuple[str, bool]]) -> float:
    by_answer: dict[str, list[bool]] = defaultdict(list)
    for answer, hit in items:
        by_answer[answer].append(hit)
    return sum(sum(values) / len(values) for values in by_answer.values()) / len(by_answer)


def brats_segmentation_loss(logits: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    """Cross-entropy plus soft Dice on standard BraTS WT/TC/ET regions."""
    cross_entropy = functional.cross_entropy(logits, target, weight=weights)
    probabilities = logits.softmax(dim=1)
    predicted_regions = torch.stack(
        (
            1.0 - probabilities[:, 0],
            probabilities[:, 2] + probabilities[:, 3],
            probabilities[:, 3],
        ),
        dim=1,
    )
    target_regions = torch.stack(
        (target > 0, (target == 2) | (target == 3), target == 3), dim=1
    ).float()
    intersection = (predicted_regions * target_regions).sum(dim=(0, 2, 3, 4))
    denominator = predicted_regions.sum(dim=(0, 2, 3, 4))
    denominator = denominator + target_regions.sum(dim=(0, 2, 3, 4))
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return cross_entropy + dice_loss


def _brats_regions(prediction: Tensor, target: Tensor) -> dict[str, tuple[Tensor, Tensor]]:
    return {
        "whole_tumor": (prediction > 0, target > 0),
        "tumor_core": (
            (prediction == 2) | (prediction == 3),
            (target == 2) | (target == 3),
        ),
        "enhancing_tumor": (prediction == 3, target == 3),
    }


def _enhancing_fraction(label: Tensor) -> float:
    whole = int((label > 0).sum())
    return int((label == 3).sum()) / whole if whole else 0.0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()

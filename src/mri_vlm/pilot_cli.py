"""Small real-MRI pilot for falsifying the model and evaluation pipeline."""

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from mri_vlm.conditions import condition_id, modality_conditions
from mri_vlm.controls import bootstrap_mean_ci
from mri_vlm.data.msd import discover_training_cases
from mri_vlm.metrics import answer_correct, dice_score
from mri_vlm.modeling import MRIVLMSmall
from mri_vlm.preprocess import PreprocessSpec, load_preprocessed_case
from mri_vlm.real_qa import SplitQAExample, generate_real_examples
from mri_vlm.schema import Modality, QuestionType, Split
from mri_vlm.split import assign_subject

ANSWER_VOCAB = ("yes", "no", "left", "right", "midline", "edema", "tumor core", "abstain")
ANSWER_TO_INDEX = {answer: index for index, answer in enumerate(ANSWER_VOCAB)}
PILOT_TYPES = (
    QuestionType.PRESENCE,
    QuestionType.LATERALITY,
    QuestionType.RELATIVE_VOLUME,
    QuestionType.CROSS_REGION_COMPARISON,
    QuestionType.UNANSWERABLE,
)


@dataclass(frozen=True, slots=True)
class PilotCase:
    case_id: str
    volumes: Tensor
    label: Tensor
    questions: tuple[SplitQAExample, ...]


class UNet3DSegmenter(nn.Module):
    """Compact two-level 3D U-Net, scalable by width for the final baseline."""

    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.encoder = _block(4, width)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = _block(width, 2 * width)
        self.up = nn.ConvTranspose3d(2 * width, width, kernel_size=2, stride=2)
        self.decoder = _block(2 * width, width)
        self.output = nn.Conv3d(width, 4, kernel_size=1)

    def forward(self, volumes: Tensor) -> Tensor:
        encoded = self.encoder(volumes)
        bottleneck = self.bottleneck(self.pool(encoded))
        upsampled = self.up(bottleneck)
        return cast(Tensor, self.output(self.decoder(torch.cat((encoded, upsampled), dim=1))))


def _block(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(inputs, outputs, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv3d(outputs, outputs, kernel_size=3, padding=1),
        nn.GELU(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small real-MRI controlled pilot")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-cases", type=int, default=6)
    parser.add_argument("--validation-cases", type=int, default=4)
    parser.add_argument("--spatial-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed/pilot-cache"))
    parser.add_argument("--seed", type=int, default=20260914)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(args.seed)
    started = time.monotonic()
    root = args.dataset_root.resolve()
    ids = _select_case_ids(
        root,
        seed=args.seed,
        train=args.train_cases,
        validation=args.validation_cases,
    )
    all_questions = generate_real_examples(
        root,
        seed=args.seed,
        include_splits=frozenset({Split.TRAIN, Split.VALIDATION}),
        include_case_ids=frozenset(ids[Split.TRAIN] + ids[Split.VALIDATION]),
    )
    questions_by_case: dict[str, list[SplitQAExample]] = defaultdict(list)
    selected = set(ids[Split.TRAIN]) | set(ids[Split.VALIDATION])
    for item in all_questions:
        if item.example.case_id in selected and item.example.question_type in PILOT_TYPES:
            questions_by_case[item.example.case_id].append(item)
    cases = {
        split: _load_cases(
            root,
            ids[split],
            questions_by_case,
            spatial_size=args.spatial_size,
            cache_root=args.cache_dir.resolve(),
        )
        for split in (Split.TRAIN, Split.VALIDATION)
    }

    segmenter = UNet3DSegmenter(width=args.width)
    segmentation_seconds = _train_segmenter(segmenter, cases[Split.TRAIN], epochs=args.epochs)

    torch.manual_seed(args.seed)
    base = MRIVLMSmall(
        vocab_size=len(QuestionType) + 1,
        answer_classes=len(ANSWER_VOCAB),
        width=args.width,
    )
    initial = base.state_dict()
    models: dict[str, MRIVLMSmall] = {}
    training_seconds: dict[str, float] = {}
    for role, evidence_mode in (
        ("answer_only", "none"),
        ("unconditional_auxiliary", "whole"),
        ("question_grounded", "question"),
    ):
        model = MRIVLMSmall(
            vocab_size=len(QuestionType) + 1,
            answer_classes=len(ANSWER_VOCAB),
            width=args.width,
            question_conditioned_evidence=evidence_mode == "question",
        )
        model.load_state_dict(initial)
        training_seconds[role] = _train_vlm(
            model,
            cases[Split.TRAIN],
            epochs=args.epochs,
            evidence_mode=evidence_mode,
        )
        models[role] = model

    conditions = _pilot_conditions()
    results: dict[str, object] = {
        "status": "small-real-data-pilot-not-for-claim",
        "seed": args.seed,
        "train_case_ids": ids[Split.TRAIN],
        "validation_case_ids": ids[Split.VALIDATION],
        "spatial_size": args.spatial_size,
        "epochs": args.epochs,
        "width": args.width,
        "conditions": [condition_id(condition) for condition in conditions],
        "model_parameter_counts": {
            "segmenter": sum(parameter.numel() for parameter in segmenter.parameters()),
            "mri_vlm_small": sum(parameter.numel() for parameter in base.parameters()),
        },
        "training_seconds": {"segmenter": segmentation_seconds, **training_seconds},
        "segmentation_symbolic": _evaluate_segmenter(
            segmenter, cases[Split.VALIDATION], conditions
        ),
        "vlm": {
            role: _evaluate_vlm(model, cases[Split.VALIDATION], conditions, role=role)
            for role, model in models.items()
        },
        "limitations": [
            "fixed first-by-ID cases, very small sample, low resolution, and two epochs",
            "categorical tasks only; numeric enhancing-fraction is excluded",
            "the compact U-Net pilot is not yet the final strong segmentation baseline",
            "confidence intervals quantify subjects but not training-seed variance",
            "validation only; no test prediction was generated",
        ],
        "wall_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": results["status"], "output": str(args.output)}))


def _select_case_ids(
    root: Path, *, seed: int, train: int, validation: int
) -> dict[Split, tuple[str, ...]]:
    grouped: dict[Split, list[str]] = defaultdict(list)
    for case in discover_training_cases(root):
        split = assign_subject(case.case_id, seed=seed)
        if split in (Split.TRAIN, Split.VALIDATION):
            grouped[split].append(case.case_id)
    return {
        Split.TRAIN: tuple(sorted(grouped[Split.TRAIN])[:train]),
        Split.VALIDATION: tuple(sorted(grouped[Split.VALIDATION])[:validation]),
    }


def _load_cases(
    root: Path,
    case_ids: tuple[str, ...],
    questions_by_case: dict[str, list[SplitQAExample]],
    *,
    spatial_size: int,
    cache_root: Path,
) -> tuple[PilotCase, ...]:
    loaded: list[PilotCase] = []
    for case_id in case_ids:
        preprocessed = load_preprocessed_case(
            root, cache_root, case_id, PreprocessSpec(spatial_size=spatial_size)
        )
        loaded.append(
            PilotCase(
                case_id=case_id,
                volumes=preprocessed.volumes,
                label=preprocessed.label,
                questions=tuple(
                    sorted(
                        questions_by_case[case_id],
                        key=lambda item: item.example.question_type.value,
                    )
                ),
            )
        )
    return tuple(loaded)


def _train_segmenter(model: nn.Module, cases: tuple[PilotCase, ...], *, epochs: int) -> float:
    started = time.monotonic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    weights = torch.tensor((0.05, 1.0, 2.0, 2.0))
    conditions = modality_conditions()
    model.train()
    step = 0
    for _ in range(epochs):
        for case in cases:
            optimizer.zero_grad()
            mask = _condition_mask(conditions[step % len(conditions)])[0, :, None, None, None]
            logits = cast(Tensor, model((case.volumes * mask).unsqueeze(0)))
            loss = _segmentation_loss(logits, case.label.unsqueeze(0), weights)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            step += 1
    return time.monotonic() - started


def _train_vlm(
    model: MRIVLMSmall,
    cases: tuple[PilotCase, ...],
    *,
    epochs: int,
    evidence_mode: str,
) -> float:
    started = time.monotonic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    conditions = modality_conditions()
    model.train()
    step = 0
    for _ in range(epochs):
        for case in cases:
            count = len(case.questions)
            volumes = case.volumes.unsqueeze(0).expand(count, -1, -1, -1, -1)
            mask = _condition_mask(conditions[step % len(conditions)]).expand(count, -1)
            tokens, targets, evidence = _case_targets(case)
            optimizer.zero_grad()
            output = model(volumes, mask, tokens)
            loss = functional.cross_entropy(output.answer_logits, targets)
            if evidence_mode != "none":
                evidence_target = (
                    (case.label > 0).float().unsqueeze(0).expand(count, -1, -1, -1)
                    if evidence_mode == "whole"
                    else evidence
                )
                loss = loss + _evidence_loss(output.evidence_logits, evidence_target)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            step += 1
    return time.monotonic() - started


def _case_targets(case: PilotCase) -> tuple[Tensor, Tensor, Tensor]:
    tokens = []
    targets = []
    evidence = []
    for item in case.questions:
        question_type = item.example.question_type
        tokens.append([list(QuestionType).index(question_type) + 1])
        answer = item.example.answer or "abstain"
        targets.append(ANSWER_TO_INDEX[answer])
        if question_type is QuestionType.PRESENCE:
            evidence.append(case.label == 3)
        elif question_type is QuestionType.UNANSWERABLE:
            evidence.append(torch.zeros_like(case.label, dtype=torch.bool))
        else:
            evidence.append(case.label > 0)
    return (
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.stack(evidence).float(),
    )


def _evidence_loss(logits: Tensor, target: Tensor) -> Tensor:
    binary = functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=torch.tensor(5.0)
    )
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return binary + dice_loss


def _segmentation_loss(logits: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    cross_entropy = functional.cross_entropy(logits, target, weight=weights)
    probabilities = logits.softmax(dim=1)
    one_hot = functional.one_hot(target, num_classes=4).permute(0, 4, 1, 2, 3).float()
    intersection = (probabilities[:, 1:] * one_hot[:, 1:]).sum(dim=(0, 2, 3, 4))
    denominator = probabilities[:, 1:].sum(dim=(0, 2, 3, 4))
    denominator = denominator + one_hot[:, 1:].sum(dim=(0, 2, 3, 4))
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return cross_entropy + dice_loss


def _pilot_conditions() -> tuple[frozenset[Modality], ...]:
    return (
        frozenset(Modality),
        frozenset({Modality.FLAIR, Modality.T1, Modality.T2}),
        frozenset({Modality.T1, Modality.T1GD, Modality.T2}),
        frozenset({Modality.FLAIR}),
        frozenset({Modality.T1GD}),
    )


def _condition_mask(condition: frozenset[Modality]) -> Tensor:
    return torch.tensor([[modality in condition for modality in Modality]], dtype=torch.bool)


def _evaluate_vlm(
    model: MRIVLMSmall,
    cases: tuple[PilotCase, ...],
    conditions: tuple[frozenset[Modality], ...],
    *,
    role: str,
) -> dict[str, object]:
    model.eval()
    results: dict[str, object] = {}
    for condition in conditions:
        subject_accuracy = []
        hits: list[bool] = []
        confidences: list[float] = []
        evidence_scores: list[float] = []
        hallucinations = 0
        unanswerable = 0
        with torch.no_grad():
            for case in cases:
                tokens, targets, evidence = _case_targets(case)
                count = len(case.questions)
                output = model(
                    case.volumes.unsqueeze(0).expand(count, -1, -1, -1, -1),
                    _condition_mask(condition).expand(count, -1),
                    tokens,
                )
                probabilities = output.answer_logits.softmax(dim=1)
                predicted = probabilities.argmax(dim=1)
                case_hits = predicted.eq(targets).tolist()
                subject_accuracy.append(sum(case_hits) / len(case_hits))
                hits.extend(case_hits)
                confidences.extend(probabilities.max(dim=1).values.tolist())
                for index, item in enumerate(case.questions):
                    if item.example.question_type is QuestionType.UNANSWERABLE:
                        unanswerable += 1
                        hallucinations += ANSWER_VOCAB[predicted[index]] != "abstain"
                    if role != "answer_only":
                        evidence_target = (
                            (case.label > 0).expand_as(evidence[index])
                            if role == "unconditional_auxiliary"
                            else evidence[index] > 0
                        )
                        predicted_voxels = frozenset(
                            torch.nonzero(
                                (output.evidence_logits[index] > 0).flatten(), as_tuple=False
                            )
                            .squeeze(1)
                            .tolist()
                        )
                        target_voxels = frozenset(
                            torch.nonzero(
                                evidence_target.flatten(), as_tuple=False
                            )
                            .squeeze(1)
                            .tolist()
                        )
                        evidence_scores.append(dice_score(predicted_voxels, target_voxels))
        results[condition_id(condition)] = {
            "answer_accuracy": sum(hits) / len(hits),
            "subject_bootstrap_95ci": bootstrap_mean_ci(
                subject_accuracy, seed=20260914, samples=500
            ),
            "ece_5bin": _ece(hits, confidences, bins=5),
            "unanswerable_hallucination_rate": hallucinations / unanswerable,
            "mean_evidence_dice": (
                sum(evidence_scores) / len(evidence_scores) if evidence_scores else None
            ),
            "subjects": len(cases),
            "questions": len(hits),
        }
    return results


def _evaluate_segmenter(
    model: nn.Module,
    cases: tuple[PilotCase, ...],
    conditions: tuple[frozenset[Modality], ...],
) -> dict[str, object]:
    model.eval()
    results: dict[str, object] = {}
    for condition in conditions:
        subject_accuracy = []
        whole_dice = []
        region_dice: dict[str, list[float]] = {
            "edema": [],
            "non_enhancing": [],
            "enhancing": [],
        }
        with torch.no_grad():
            for case in cases:
                volumes = case.volumes * _condition_mask(condition)[0, :, None, None, None]
                prediction = model(volumes.unsqueeze(0)).argmax(dim=1).squeeze(0)
                expected_answers = [item.example.answer or "abstain" for item in case.questions]
                predicted_answers = [
                    _symbolic_answer(prediction, item.example.question_type)
                    for item in case.questions
                ]
                subject_accuracy.append(
                    sum(
                        answer_correct(expected, predicted, item.example.answer_kind)
                        for expected, predicted, item in zip(
                            expected_answers, predicted_answers, case.questions, strict=True
                        )
                    )
                    / len(case.questions)
                )
                whole_dice.append(_tensor_dice(prediction > 0, case.label > 0))
                for name, label_value in (
                    ("edema", 1),
                    ("non_enhancing", 2),
                    ("enhancing", 3),
                ):
                    region_dice[name].append(
                        _tensor_dice(prediction == label_value, case.label == label_value)
                    )
        results[condition_id(condition)] = {
            "symbolic_answer_accuracy": sum(subject_accuracy) / len(subject_accuracy),
            "subject_bootstrap_95ci": bootstrap_mean_ci(
                subject_accuracy, seed=20260914, samples=500
            ),
            "mean_whole_tumor_dice": sum(whole_dice) / len(whole_dice),
            "mean_region_dice": {
                name: sum(values) / len(values) for name, values in region_dice.items()
            },
            "subjects": len(cases),
        }
    return results


def _symbolic_answer(label: Tensor, question_type: QuestionType) -> str:
    if question_type is QuestionType.UNANSWERABLE:
        return "abstain"
    if question_type is QuestionType.PRESENCE:
        return "yes" if torch.any(label == 3) else "no"
    whole = label > 0
    if question_type is QuestionType.LATERALITY:
        midpoint = label.shape[2] // 2
        left = int(whole[:, :, :midpoint].sum())
        right = int(whole[:, :, midpoint:].sum())
        if left + right == 0 or abs(left - right) / (left + right) <= 0.05:
            return "midline"
        return "left" if left > right else "right"
    edema = int((label == 1).sum())
    core = int(((label == 2) | (label == 3)).sum())
    if question_type is QuestionType.RELATIVE_VOLUME:
        return "edema" if edema >= core else "tumor core"
    if question_type is QuestionType.CROSS_REGION_COMPARISON:
        return "yes" if core > 0.5 * int(whole.sum()) else "no"
    raise ValueError(f"unsupported pilot question: {question_type}")


def _tensor_dice(predicted: Tensor, target: Tensor) -> float:
    denominator = int(predicted.sum() + target.sum())
    return 1.0 if denominator == 0 else 2.0 * int((predicted & target).sum()) / denominator


def _ece(hits: list[bool], confidences: list[float], *, bins: int) -> float:
    total = len(hits)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            i
            for i, confidence in enumerate(confidences)
            if lower <= confidence < upper or (index == bins - 1 and confidence == upper)
        ]
        if members:
            accuracy = sum(hits[i] for i in members) / len(members)
            confidence = sum(confidences[i] for i in members) / len(members)
            error += len(members) / total * abs(accuracy - confidence)
    return error


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


if __name__ == "__main__":
    main()

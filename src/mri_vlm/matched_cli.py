"""Matched answer-only, auxiliary, and voxel-grounded MRI-QA direction run."""

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor

from mri_vlm.baseline_cli import load_cases, select_case_ids
from mri_vlm.conditions import condition_id, modality_conditions
from mri_vlm.controls import bootstrap_mean_ci
from mri_vlm.metrics import dice_score
from mri_vlm.modeling import MRIVLM3D, SliceVLM2D
from mri_vlm.pilot_cli import _condition_mask, _ece, _evidence_loss
from mri_vlm.preprocess import PreprocessedCase, PreprocessSpec
from mri_vlm.qa_v1 import (
    ANSWER_VOCAB_V1,
    FROZEN_FRACTION_THRESHOLDS,
    QATargetV1,
    batch_targets,
    build_word_vocabulary,
    materialize_targets,
)
from mri_vlm.real_qa import SplitQAExample, generate_real_examples
from mri_vlm.schema import Modality, Split


@dataclass(frozen=True, slots=True)
class MatchedCase:
    case: PreprocessedCase
    targets: tuple[QATargetV1, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run matched MRI-QA models")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed/baseline-v2"))
    parser.add_argument("--train-cases", type=int, default=32)
    parser.add_argument("--validation-cases", type=int, default=8)
    parser.add_argument("--spatial-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=20260914,
        help="frozen cohort seed; independent of the training seed",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(args.seed)
    started = time.monotonic()
    root = args.dataset_root.resolve()
    ids = select_case_ids(
        root,
        seed=args.split_seed,
        train_count=args.train_cases,
        validation_count=args.validation_cases,
    )
    examples = generate_real_examples(
        root,
        seed=args.split_seed,
        include_splits=frozenset({Split.TRAIN, Split.VALIDATION}),
        include_case_ids=frozenset(ids[Split.TRAIN] + ids[Split.VALIDATION]),
    )
    grouped: dict[str, list[SplitQAExample]] = defaultdict(list)
    for item in examples:
        grouped[item.example.case_id].append(item)
    spec = PreprocessSpec(
        spatial_size=args.spatial_size,
        version="brain-crop-zero-background-zscore-v2",
        preserve_background_zero=True,
        crop_to_foreground=True,
    )
    vocabulary = build_word_vocabulary()
    prepared: dict[Split, tuple[MatchedCase, ...]] = {}
    for split in (Split.TRAIN, Split.VALIDATION):
        cases = load_cases(root, args.cache_dir.resolve(), ids[split], spec)
        prepared[split] = tuple(
            MatchedCase(
                case,
                materialize_targets(
                    case,
                    tuple(grouped[case.case_id]),
                    FROZEN_FRACTION_THRESHOLDS,
                    vocabulary,
                ),
            )
            for case in cases
        )
    if any(not item.targets for split in prepared.values() for item in split):
        raise ValueError("every selected case must retain at least one QA V1 target")
    class_weights = answer_class_weights(prepared[Split.TRAIN], len(ANSWER_VOCAB_V1))

    torch.manual_seed(args.seed)
    template = MRIVLM3D(
        vocab_size=len(vocabulary) + 1,
        answer_classes=len(ANSWER_VOCAB_V1),
        width=args.width,
    )
    initial = {name: value.detach().clone() for name, value in template.state_dict().items()}
    run_fingerprint = _fingerprint(
        {
            "model": "MRIVLM3D-v2-weighted-loss-late-tie",
            "seed": args.seed,
            "split_seed": args.split_seed,
            "train_case_ids": ids[Split.TRAIN],
            "validation_case_ids": ids[Split.VALIDATION],
            "epochs": args.epochs,
            "spatial_size": args.spatial_size,
            "width": args.width,
            "learning_rate": args.learning_rate,
            "preprocessing": asdict(spec),
            "fraction_thresholds": asdict(FROZEN_FRACTION_THRESHOLDS),
            "vocabulary": vocabulary,
            "modality_schedule": [condition_id(item) for item in modality_conditions()],
        }
    )
    roles = (
        ("answer_only", "none", True),
        ("unconditional_auxiliary", "whole", False),
        ("question_grounded", "question", True),
    )
    models: dict[str, MRIVLM3D] = {}
    histories: dict[str, object] = {}
    training_seconds: dict[str, float] = {}
    checkpoint_hashes: dict[str, str] = {}
    recovery_notes: list[str] = []
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for role, evidence_mode, conditioned in roles:
        model = MRIVLM3D(
            vocab_size=len(vocabulary) + 1,
            answer_classes=len(ANSWER_VOCAB_V1),
            width=args.width,
            question_conditioned_evidence=conditioned,
        )
        model.load_state_dict(initial)
        checkpoint = args.checkpoint_dir / f"{role}.pt"
        completed = _load_completed(checkpoint, fingerprint=run_fingerprint, role=role)
        if completed is not None:
            best_state = cast(dict[str, Tensor], completed["state_dict"])
            best_epoch = cast(int, completed["best_epoch"])
            history = cast(list[dict[str, float]], completed["history"])
            training_seconds[role] = cast(float, completed["training_seconds"])
            recovery_notes.append(f"reused verified completed checkpoint for {role}")
            model.load_state_dict(best_state)
            checkpoint_hashes[role] = _sha256(checkpoint)
            histories[role] = {"best_epoch": best_epoch, "epochs": history}
            models[role] = model
            continue
        if checkpoint.exists():
            legacy = checkpoint.with_name(f"{role}.unverified-{_sha256(checkpoint)[:8]}.pt")
            checkpoint.replace(legacy)
            recovery_notes.append(
                f"quarantined unverifiable {role} checkpoint as {legacy.name}; "
                "it lacked the run fingerprint/history/optimizer provenance"
            )
        role_started = time.monotonic()
        history, best_epoch, best_state = train_role(
            model,
            prepared[Split.TRAIN],
            prepared[Split.VALIDATION],
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            class_weights=class_weights,
            evidence_mode=evidence_mode,
            seed=args.seed,
        )
        training_seconds[role] = time.monotonic() - role_started
        model.load_state_dict(best_state)
        torch.save(
            {
                "complete": True,
                "run_fingerprint": run_fingerprint,
                "role": role,
                "state_dict": best_state,
                "best_epoch": best_epoch,
                "history": history,
                "training_seconds": training_seconds[role],
            },
            checkpoint,
        )
        checkpoint_hashes[role] = _sha256(checkpoint)
        histories[role] = {"best_epoch": best_epoch, "epochs": history}
        models[role] = model

    conditions = modality_conditions()
    evaluations: dict[str, object] = {
        role: evaluate_role(
            model,
            prepared[Split.VALIDATION],
            conditions=conditions,
            role=role,
            bootstrap_seed=args.seed,
        )
        for role, model in models.items()
    }
    full_id = condition_id(frozenset(Modality))
    effects = paired_effects(evaluations, full_id=full_id, seed=args.seed)
    output = {
        "status": "single-seed-development-direction-run",
        "scope": "matched QA V1 validation experiment; no test cases read",
        "seed": args.seed,
        "run_fingerprint": run_fingerprint,
        "recovery_notes": recovery_notes,
        "test_cases_read": 0,
        "config": {
            "train_cases": len(prepared[Split.TRAIN]),
            "validation_cases": len(prepared[Split.VALIDATION]),
            "split_seed": args.split_seed,
            "spatial_size": args.spatial_size,
            "epochs": args.epochs,
            "width": args.width,
            "learning_rate": args.learning_rate,
            "preprocessing": asdict(spec),
            "fraction_thresholds": {
                "q1": FROZEN_FRACTION_THRESHOLDS.q1,
                "median": FROZEN_FRACTION_THRESHOLDS.median,
                "q3": FROZEN_FRACTION_THRESHOLDS.q3,
            },
            "word_vocabulary": vocabulary,
            "answer_vocabulary": ANSWER_VOCAB_V1,
            "class_weights": class_weights.tolist(),
            "modality_schedule": [condition_id(item) for item in conditions],
            "question_rotation": "one QA family per case-step, deterministic cyclic order",
        },
        "model_parameters": sum(parameter.numel() for parameter in template.parameters()),
        "train_case_ids": ids[Split.TRAIN],
        "validation_case_ids": ids[Split.VALIDATION],
        "target_counts": {
            split.value: target_counts(prepared[split]) for split in (Split.TRAIN, Split.VALIDATION)
        },
        "training_seconds": training_seconds,
        "histories": histories,
        "checkpoint_sha256": checkpoint_hashes,
        "evaluation": evaluations,
        "full_input_paired_effects": effects,
        "environment": {
            "device": "cpu",
            "accelerator_memory": None,
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
        },
        "wall_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(args.output)}))


def answer_class_weights(cases: tuple[MatchedCase, ...], classes: int) -> Tensor:
    counts = Counter(target.answer_index for case in cases for target in case.targets)
    weights = torch.zeros(classes)
    for index, count in counts.items():
        weights[index] = 1.0 / count
    positive = weights > 0
    weights[positive] /= weights[positive].mean()
    return weights


def train_role(
    model: MRIVLM3D | SliceVLM2D,
    train_cases: tuple[MatchedCase, ...],
    validation_cases: tuple[MatchedCase, ...],
    *,
    epochs: int,
    learning_rate: float,
    class_weights: Tensor,
    evidence_mode: str,
    seed: int,
) -> tuple[list[dict[str, float]], int, dict[str, Tensor]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
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
            item = train_cases[index]
            target = item.targets[(epoch + index) % len(item.targets)]
            tokens, answers, evidence = batch_targets((target,))
            condition = conditions[step % len(conditions)]
            mask = _condition_mask(condition)
            optimizer.zero_grad()
            output = model(
                (item.case.volumes * mask[0, :, None, None, None]).unsqueeze(0), mask, tokens
            )
            loss = weighted_answer_loss(output.answer_logits, answers, class_weights)
            if evidence_mode != "none":
                evidence_target = (
                    (item.case.label > 0).float().unsqueeze(0)
                    if evidence_mode == "whole"
                    else evidence
                )
                loss = loss + _evidence_loss(output.evidence_logits, evidence_target)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            step += 1
        validation = evaluate_role(
            model,
            validation_cases,
            conditions=(frozenset(Modality),),
            role=(
                "answer_only"
                if evidence_mode == "none"
                else "question_grounded"
                if evidence_mode == "question"
                else "unconditional_auxiliary"
            ),
            bootstrap_seed=seed,
        )[condition_id(frozenset(Modality))]
        score = cast(float, cast(dict[str, object], validation)["balanced_answer_accuracy"])
        history.append(
            {
                "epoch": float(epoch),
                "mean_train_loss": sum(losses) / len(losses),
                "validation_balanced_answer_accuracy": score,
            }
        )
        if score >= best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    return history, best_epoch, best_state


def evaluate_role(
    model: MRIVLM3D | SliceVLM2D,
    cases: tuple[MatchedCase, ...],
    *,
    conditions: tuple[frozenset[Modality], ...],
    role: str,
    bootstrap_seed: int,
) -> dict[str, object]:
    model.eval()
    results: dict[str, object] = {}
    for condition in conditions:
        hits_by_type: dict[str, list[tuple[str, bool]]] = defaultdict(list)
        evidence_by_type: dict[str, list[float]] = defaultdict(list)
        subject_answer: list[float] = []
        subject_grounded: list[float] = []
        all_hits: list[bool] = []
        confidences: list[float] = []
        with torch.no_grad():
            for item in cases:
                tokens, answers, evidence = batch_targets(item.targets)
                count = len(item.targets)
                mask = _condition_mask(condition).expand(count, -1)
                volumes = item.case.volumes.unsqueeze(0).expand(count, -1, -1, -1, -1)
                volumes = volumes * mask[:, :, None, None, None]
                output = model(volumes, mask, tokens)
                probabilities = output.answer_logits.softmax(dim=1)
                predictions = probabilities.argmax(dim=1)
                hits = predictions.eq(answers).tolist()
                all_hits.extend(hits)
                confidences.extend(probabilities.max(dim=1).values.tolist())
                subject_answer.append(sum(hits) / len(hits))
                grounded_hits: list[bool] = []
                for index, target in enumerate(item.targets):
                    hits_by_type[target.question_type.value].append((target.answer, hits[index]))
                    if role == "answer_only":
                        continue
                    evidence_target = (
                        item.case.label > 0
                        if role == "unconditional_auxiliary"
                        else evidence[index] > 0
                    )
                    score = _tensor_evidence_dice(
                        output.evidence_logits[index] > 0, evidence_target
                    )
                    evidence_by_type[target.question_type.value].append(score)
                    grounded_hits.append(hits[index] and score >= 0.5)
                if grounded_hits:
                    subject_grounded.append(sum(grounded_hits) / len(grounded_hits))
        balanced_by_type = {
            name: _balanced_accuracy(values) for name, values in hits_by_type.items()
        }
        result: dict[str, object] = {
            "subjects": len(cases),
            "questions": len(all_hits),
            "subject_ids": [item.case.case_id for item in cases],
            "answer_accuracy": sum(all_hits) / len(all_hits),
            "balanced_answer_accuracy": sum(balanced_by_type.values()) / len(balanced_by_type),
            "balanced_accuracy_by_type": balanced_by_type,
            "answer_subject_bootstrap_95ci": bootstrap_mean_ci(
                subject_answer, seed=bootstrap_seed, samples=1000
            ),
            "ece_5bin": _ece(all_hits, confidences, bins=5),
            "mean_evidence_dice_by_type": {
                name: sum(values) / len(values) for name, values in evidence_by_type.items()
            },
            "subject_answer_scores": subject_answer,
            "subject_grounded_scores": subject_grounded,
        }
        if subject_grounded:
            result["grounded_answer_accuracy"] = sum(subject_grounded) / len(subject_grounded)
            result["grounded_subject_bootstrap_95ci"] = bootstrap_mean_ci(
                subject_grounded, seed=bootstrap_seed, samples=1000
            )
        else:
            result["grounded_answer_accuracy"] = None
            result["grounded_subject_bootstrap_95ci"] = None
        results[condition_id(condition)] = result
    return results


def paired_effects(evaluations: dict[str, object], *, full_id: str, seed: int) -> dict[str, object]:
    grounded = _condition_result(evaluations, "question_grounded", full_id)
    output: dict[str, object] = {}
    for comparator in ("answer_only", "unconditional_auxiliary"):
        reference = _condition_result(evaluations, comparator, full_id)
        grounded_scores = cast(list[float], grounded["subject_answer_scores"])
        reference_scores = cast(list[float], reference["subject_answer_scores"])
        differences = [a - b for a, b in zip(grounded_scores, reference_scores, strict=True)]
        output[f"grounded_minus_{comparator}_answer_accuracy"] = {
            "mean": sum(differences) / len(differences),
            "subject_bootstrap_95ci": bootstrap_mean_ci(differences, seed=seed, samples=2000),
        }
    return output


def target_counts(cases: tuple[MatchedCase, ...]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(target.question_type.value for case in cases for target in case.targets).items()
        )
    )


def _condition_result(
    evaluations: dict[str, object], role: str, condition: str
) -> dict[str, object]:
    return cast(dict[str, object], cast(dict[str, object], evaluations[role])[condition])


def _balanced_accuracy(items: list[tuple[str, bool]]) -> float:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for answer, hit in items:
        grouped[answer].append(hit)
    return sum(sum(values) / len(values) for values in grouped.values()) / len(grouped)


def weighted_answer_loss(logits: Tensor, targets: Tensor, class_weights: Tensor) -> Tensor:
    """Apply inverse-frequency weights without batch-size-one mean cancellation."""
    losses = functional.cross_entropy(logits, targets, reduction="none")
    return (losses * class_weights[targets]).mean()


def _tensor_evidence_dice(prediction: Tensor, target: Tensor) -> float:
    predicted = frozenset(torch.nonzero(prediction.flatten(), as_tuple=False).squeeze(1).tolist())
    expected = frozenset(torch.nonzero(target.flatten(), as_tuple=False).squeeze(1).tolist())
    return dice_score(predicted, expected)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_completed(path: Path, *, fingerprint: str, role: str) -> dict[str, object] | None:
    if not path.is_file():
        return None
    payload = cast(dict[str, object], torch.load(path, map_location="cpu", weights_only=True))
    if (
        payload.get("complete") is not True
        or payload.get("run_fingerprint") != fingerprint
        or payload.get("role") != role
    ):
        return None
    required = {"state_dict", "best_epoch", "history", "training_seconds"}
    return payload if required <= payload.keys() else None


if __name__ == "__main__":
    main()

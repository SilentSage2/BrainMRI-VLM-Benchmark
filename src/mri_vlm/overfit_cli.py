"""One-case learning checks for every real-MRI optimization path."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as functional

from mri_vlm.modeling import MRIVLMSmall
from mri_vlm.pilot_cli import (
    ANSWER_VOCAB,
    PILOT_TYPES,
    UNet3DSegmenter,
    _case_targets,
    _condition_mask,
    _evaluate_segmenter,
    _evaluate_vlm,
    _evidence_loss,
    _load_cases,
    _segmentation_loss,
    _select_case_ids,
    _set_seed,
    _symbolic_answer,
)
from mri_vlm.real_qa import SplitQAExample, generate_real_examples
from mri_vlm.schema import Modality, QuestionType, Split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overfit each objective to one real MRI")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed/overfit-cache"))
    parser.add_argument("--spatial-size", type=int, default=24)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(args.seed)
    root = args.dataset_root.resolve()
    case_id = _select_case_ids(root, seed=args.seed, train=1, validation=1)[Split.TRAIN][0]
    questions = generate_real_examples(
        root,
        seed=args.seed,
        include_splits=frozenset({Split.TRAIN}),
        include_case_ids=frozenset({case_id}),
    )
    grouped: dict[str, list[SplitQAExample]] = defaultdict(list)
    for item in questions:
        if item.example.question_type in PILOT_TYPES:
            grouped[case_id].append(item)
    case = _load_cases(
        root,
        (case_id,),
        grouped,
        spatial_size=args.spatial_size,
        cache_root=args.cache_dir.resolve(),
    )[0]
    condition: tuple[frozenset[Modality], ...] = (frozenset(Modality),)
    reference_consistency = sum(
        _symbolic_answer(case.label, item.example.question_type)
        == (item.example.answer or "abstain")
        for item in case.questions
    ) / len(case.questions)
    started = time.monotonic()

    segmenter = UNet3DSegmenter(width=args.width)
    _overfit_segmenter(segmenter, case, steps=args.steps)
    segmentation_result = _evaluate_segmenter(segmenter, (case,), condition)

    torch.manual_seed(args.seed)
    base = MRIVLMSmall(
        vocab_size=len(QuestionType) + 1,
        answer_classes=len(ANSWER_VOCAB),
        width=args.width,
    )
    initial = base.state_dict()
    vlm_results: dict[str, object] = {}
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
        _overfit_vlm(model, case, steps=args.steps, evidence_mode=evidence_mode)
        vlm_results[role] = _evaluate_vlm(model, (case,), condition, role=role)

    full_id = "flair+t1+t1gd+t2"
    segment_dice = cast(
        float,
        cast(dict[str, object], segmentation_result[full_id])["mean_whole_tumor_dice"],
    )
    thresholds = {
        "preprocessed_reference_symbolic_accuracy_1.0": reference_consistency == 1.0,
        "segmenter_whole_dice_at_least_0.80": segment_dice >= 0.80,
        "segmenter_symbolic_accuracy_1.0": _nested_metric(
            segmentation_result, full_id, "symbolic_answer_accuracy"
        )
        == 1.0,
        "answer_only_accuracy_1.0": _metric(vlm_results, "answer_only", full_id, "answer_accuracy")
        == 1.0,
        "unconditional_accuracy_1.0": _metric(
            vlm_results, "unconditional_auxiliary", full_id, "answer_accuracy"
        )
        == 1.0,
        "unconditional_evidence_dice_at_least_0.80": _metric(
            vlm_results, "unconditional_auxiliary", full_id, "mean_evidence_dice"
        )
        >= 0.80,
        "grounded_accuracy_1.0": _metric(
            vlm_results, "question_grounded", full_id, "answer_accuracy"
        )
        == 1.0,
        "grounded_evidence_dice_at_least_0.80": _metric(
            vlm_results, "question_grounded", full_id, "mean_evidence_dice"
        )
        >= 0.80,
    }
    output = {
        "status": "pass" if all(thresholds.values()) else "fail",
        "case_id": case_id,
        "seed": args.seed,
        "steps_per_path": args.steps,
        "spatial_size": args.spatial_size,
        "width": args.width,
        "preprocessed_reference_symbolic_accuracy": reference_consistency,
        "segmentation_symbolic": segmentation_result,
        "vlm": vlm_results,
        "thresholds": thresholds,
        "wall_seconds": time.monotonic() - started,
        "scope": "optimization-path test only; not generalization evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(args.output)}))


def _overfit_segmenter(model: UNet3DSegmenter, case: object, *, steps: int) -> None:
    from mri_vlm.pilot_cli import PilotCase

    if not isinstance(case, PilotCase):
        raise TypeError("expected PilotCase")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    weights = torch.tensor((0.05, 1.0, 2.0, 2.0))
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(case.volumes.unsqueeze(0))
        loss = _segmentation_loss(logits, case.label.unsqueeze(0), weights)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()


def _overfit_vlm(model: MRIVLMSmall, case: object, *, steps: int, evidence_mode: str) -> None:
    from mri_vlm.pilot_cli import PilotCase

    if not isinstance(case, PilotCase):
        raise TypeError("expected PilotCase")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    tokens, targets, evidence = _case_targets(case)
    count = len(case.questions)
    volumes = case.volumes.unsqueeze(0).expand(count, -1, -1, -1, -1)
    mask = _condition_mask(frozenset(Modality)).expand(count, -1)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        output = model(volumes, mask, tokens)
        loss = functional.cross_entropy(output.answer_logits, targets)
        if evidence_mode != "none":
            target = (
                (case.label > 0).float().unsqueeze(0).expand(count, -1, -1, -1)
                if evidence_mode == "whole"
                else evidence
            )
            loss = loss + _evidence_loss(output.evidence_logits, target)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()


def _metric(results: dict[str, object], role: str, condition: str, metric: str) -> float:
    role_results = cast(dict[str, object], results[role])
    condition_results = cast(dict[str, object], role_results[condition])
    value = condition_results[metric]
    if not isinstance(value, int | float):
        raise ValueError(f"metric {metric} is not numeric")
    return float(value)


def _nested_metric(results: dict[str, object], condition: str, metric: str) -> float:
    condition_results = cast(dict[str, object], results[condition])
    value = condition_results[metric]
    if not isinstance(value, int | float):
        raise ValueError(f"metric {metric} is not numeric")
    return float(value)


if __name__ == "__main__":
    main()

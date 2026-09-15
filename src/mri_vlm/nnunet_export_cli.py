"""Plan or materialize a leakage-safe nnU-Net v2 Phase II dataset."""

import argparse
import json
from pathlib import Path

from mri_vlm.nnunet_export import build_phase2_plan, materialize_phase2_export


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--retired-manifest",
        type=Path,
        default=Path("configs/test_subject_manifest.json"),
    )
    parser.add_argument("--dataset-name", default="Dataset501_MSDTask01Phase2Dev")
    parser.add_argument("--split-seed", type=int, default=20260914)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    plan = build_phase2_plan(
        args.dataset_root,
        args.retired_manifest,
        dataset_name=args.dataset_name,
        split_seed=args.split_seed,
    )
    payload = plan.summary()
    payload["mode"] = "plan-only"
    if args.write:
        if args.output_root is None:
            parser.error("--output-root is required with --write")
        payload["output"] = str(materialize_phase2_export(plan, args.output_root))
        payload["mode"] = "materialized"
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

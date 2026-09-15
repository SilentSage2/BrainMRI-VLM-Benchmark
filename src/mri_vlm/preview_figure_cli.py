"""Generate the no-caption, smartphone-readable ISMRM preview figure."""

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from mri_vlm.cohort import select_median_burden_case
from mri_vlm.figure_cli import load_burdens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the ISMRM preview figure")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser


def render_preview(root: Path, output_prefix: Path, *, seed: int) -> dict[str, object]:
    try:
        matplotlib: Any = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt: Any = importlib.import_module("matplotlib.pyplot")
        patches: Any = importlib.import_module("matplotlib.patches")
        nib: Any = importlib.import_module("nibabel")
        np: Any = importlib.import_module("numpy")
    except ImportError as error:
        raise RuntimeError("install the project with the 'figures' extra") from error

    selected = select_median_burden_case(load_burdens(root, seed=seed))
    image = np.asanyarray(
        nib.load(str(root / "imagesTr" / f"{selected.case_id}.nii.gz")).dataobj
    )
    label = np.asanyarray(
        nib.load(str(root / "labelsTr" / f"{selected.case_id}.nii.gz")).dataobj
    )
    axial_index = round(float(np.median(np.argwhere(label > 0)[:, 2])))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    figure = plt.figure(figsize=(4, 4), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 0.9))
    for channel, name in enumerate(("FLAIR", "T1", "T1-Gd", "T2")):
        axis = figure.add_subplot(grid[channel // 2, channel % 2])
        plane = np.rot90(image[:, :, axial_index, channel])
        foreground = plane[plane != 0]
        values = foreground if foreground.size else plane
        axis.imshow(
            plane,
            cmap="gray",
            vmin=float(np.quantile(values, 0.01)),
            vmax=float(np.quantile(values, 0.99)),
        )
        axis.set_title(name, fontweight="bold", fontsize=13, pad=2)
        axis.axis("off")

    flow = figure.add_subplot(grid[2, :])
    flow.set_xlim(0.0, 1.0)
    flow.set_ylim(0.0, 1.0)
    flow.axis("off")
    _box(flow, patches, 0.03, 0.53, 0.40, 0.33, "Grounded 3D VLM", "#FFF0C7")
    _box(flow, patches, 0.57, 0.53, 0.40, 0.33, "Modular 3D MR", "#D8EAD3")
    _box(flow, patches, 0.18, 0.02, 0.64, 0.29, "15-subset reliability", "#DDEAF3")
    _arrow(flow, patches, (0.23, 0.53), (0.40, 0.31))
    _arrow(flow, patches, (0.77, 0.53), (0.60, 0.31))
    figure.suptitle("Missing-contrast MRI reliability", fontsize=15, fontweight="bold")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".png"), dpi=300, pil_kwargs={"compress_level": 6})
    figure.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(figure)
    return {
        "case_id": selected.case_id,
        "selection_rule": "validation case closest to median whole-tumor volume",
        "axial_index": axial_index,
        "width_pixels": 1200,
        "height_pixels": 1200,
        "contains_result_values": False,
    }


def _box(
    axis: Any,
    patches: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    color: str,
) -> None:
    axis.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor="#333333",
            linewidth=1.5,
        )
    )
    axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=11)


def _arrow(axis: Any, patches: Any, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(
        patches.FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.5,
            color="#333333",
        )
    )


def main() -> None:
    args = build_parser().parse_args()
    prefix = args.output_prefix.resolve()
    metadata = render_preview(args.dataset_root.resolve(), prefix, seed=args.seed)
    metadata_path = prefix.with_name(f"{prefix.name}_metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "pass", **metadata}, sort_keys=True))


if __name__ == "__main__":
    main()

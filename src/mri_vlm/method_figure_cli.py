"""Generate a real-input MRI montage and the grounded VLM method schematic."""

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from mri_vlm.cohort import select_median_burden_case
from mri_vlm.figure_cli import load_burdens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the MRI-VLM method figure")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser


def render_method_figure(root: Path, output_prefix: Path, *, seed: int) -> dict[str, object]:
    try:
        matplotlib: Any = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt: Any = importlib.import_module("matplotlib.pyplot")
        patches: Any = importlib.import_module("matplotlib.patches")
        colors_module: Any = importlib.import_module("matplotlib.colors")
        nib: Any = importlib.import_module("nibabel")
        np: Any = importlib.import_module("numpy")
    except ImportError as error:
        raise RuntimeError("install the project with the 'figures' extra") from error

    selected = select_median_burden_case(load_burdens(root, seed=seed))
    image_path = root / "imagesTr" / f"{selected.case_id}.nii.gz"
    label_path = root / "labelsTr" / f"{selected.case_id}.nii.gz"
    image = np.asanyarray(nib.load(str(image_path)).dataobj)
    label = np.asanyarray(nib.load(str(label_path)).dataobj)
    tumor_coordinates = np.argwhere(label > 0)
    axial_index = round(float(np.median(tumor_coordinates[:, 2])))

    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8.5})
    figure = plt.figure(figsize=(7.2, 4.15), constrained_layout=True)
    grid = figure.add_gridspec(2, 5, height_ratios=(1.0, 0.78))
    modality_names = ("FLAIR", "T1", "T1-Gd", "T2")
    for channel, name in enumerate(modality_names):
        axis = figure.add_subplot(grid[0, channel])
        plane = np.rot90(image[:, :, axial_index, channel])
        axis.imshow(
            plane,
            cmap="gray",
            vmin=_quantile(np, plane, 0.01),
            vmax=_quantile(np, plane, 0.99),
        )
        axis.set_title(name, fontweight="bold")
        axis.axis("off")

    overlay_axis = figure.add_subplot(grid[0, 4])
    background = np.rot90(image[:, :, axial_index, 2])
    overlay = np.rot90(label[:, :, axial_index])
    overlay_axis.imshow(
        background,
        cmap="gray",
        vmin=_quantile(np, background, 0.01),
        vmax=_quantile(np, background, 0.99),
    )
    region_map = np.ma.masked_where(overlay == 0, overlay)
    cmap = colors_module.ListedColormap(("#009E73", "#0072B2", "#D55E00"))
    norm = colors_module.BoundaryNorm((0.5, 1.5, 2.5, 3.5), cmap.N)
    overlay_axis.imshow(region_map, cmap=cmap, norm=norm, alpha=0.62)
    overlay_axis.set_title("Reference evidence", fontweight="bold")
    overlay_axis.axis("off")

    legend_handles = (
        patches.Patch(color="#009E73", label="Edema"),
        patches.Patch(color="#0072B2", label="Non-enhancing"),
        patches.Patch(color="#D55E00", label="Enhancing"),
    )
    overlay_axis.legend(
        handles=legend_handles,
        loc="lower right",
        ncol=1,
        frameon=True,
        facecolor="white",
        framealpha=0.82,
        fontsize=6.5,
    )

    diagram_axis = figure.add_subplot(grid[1, :])
    diagram_axis.set_xlim(0.0, 1.0)
    diagram_axis.set_ylim(0.0, 1.0)
    diagram_axis.axis("off")
    _box(
        diagram_axis,
        patches,
        0.01,
        0.38,
        0.13,
        0.38,
        "4 MRI contrasts\n+ availability",
        "#DDEAF3",
    )
    _box(
        diagram_axis,
        patches,
        0.18,
        0.38,
        0.14,
        0.38,
        "Shared 3D encoder\n+ sequence identity",
        "#D8EAD3",
    )
    _box(
        diagram_axis,
        patches,
        0.18,
        0.05,
        0.14,
        0.2,
        "Question tokens\n+ masked pooling",
        "#DDEAF3",
    )
    _box(diagram_axis, patches, 0.37, 0.29, 0.15, 0.42, "Question-conditioned\nfusion", "#FFF0C7")
    _box(
        diagram_axis,
        patches,
        0.58,
        0.56,
        0.14,
        0.25,
        "Answer logits\n(including abstain)",
        "#E8D8EE",
    )
    _box(diagram_axis, patches, 0.58, 0.18, 0.14, 0.25, "Voxel evidence\nlogits", "#F5D0D0")
    _box(
        diagram_axis,
        patches,
        0.78,
        0.59,
        0.20,
        0.28,
        "Frozen training protocol\n"
        "answer loss + evidence loss\n"
        "balanced contrast dropout\n"
        "runner pending",
        "#F2F2F2",
        linestyle="--",
        fontsize=5.5,
    )
    _box(
        diagram_axis,
        patches,
        0.78,
        0.12,
        0.20,
        0.34,
        "Implemented evaluation\n"
        "15 non-empty contrast subsets\n"
        "answer, Dice, grounding,\n"
        "hallucination, calibration\n"
        "subject-level bootstrap",
        "#E5F5F9",
        fontsize=5.5,
    )
    _arrow(diagram_axis, patches, (0.14, 0.57), (0.18, 0.57))
    _arrow(diagram_axis, patches, (0.32, 0.57), (0.37, 0.53))
    _arrow(diagram_axis, patches, (0.32, 0.15), (0.39, 0.32))
    _arrow(diagram_axis, patches, (0.52, 0.5), (0.58, 0.68))
    _arrow(diagram_axis, patches, (0.52, 0.5), (0.58, 0.30))
    _arrow(diagram_axis, patches, (0.72, 0.68), (0.78, 0.34))
    _arrow(diagram_axis, patches, (0.72, 0.30), (0.78, 0.25))
    figure.suptitle(
        "Voxel-grounded 3D MRI vision-language reasoning under missing contrasts",
        fontsize=10,
        fontweight="bold",
        y=1.085,
    )
    figure.text(
        0.01,
        1.015,
        "A  Real co-registered inputs and reference regions: "
        f"{selected.case_id}, axial index {axial_index}",
        fontsize=8,
        fontweight="bold",
    )
    figure.text(
        0.01,
        0.435,
        "B  Implemented model and protocol-aware evaluation flow",
        fontsize=8,
        fontweight="bold",
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return {
        "case_id": selected.case_id,
        "selection_rule": "validation case closest to median whole-tumor volume",
        "whole_tumor_ml": selected.whole_tumor_ml,
        "axial_index": axial_index,
        "reference_mask_is_model_input": False,
    }


def _quantile(np: Any, plane: Any, fraction: float) -> float:
    foreground = plane[plane != 0]
    values = foreground if foreground.size else plane
    return float(np.quantile(values, fraction))


def _box(
    axis: Any,
    patches: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    color: str,
    *,
    linestyle: str = "-",
    fontsize: float = 7,
) -> None:
    box = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015",
        facecolor=color,
        edgecolor="#333333",
        linewidth=1.2,
        linestyle=linestyle,
    )
    axis.add_patch(box)
    axis.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def _arrow(axis: Any, patches: Any, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(
        patches.FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.3,
            color="#333333",
        )
    )


def main() -> None:
    args = build_parser().parse_args()
    prefix = args.output_prefix.resolve()
    metadata = render_method_figure(args.dataset_root.resolve(), prefix, seed=args.seed)
    metadata_path = prefix.with_name(f"{prefix.name}_metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "pass", **metadata}, sort_keys=True))


if __name__ == "__main__":
    main()

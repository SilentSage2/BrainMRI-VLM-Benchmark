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

    figure = plt.figure(figsize=(12, 6.7), constrained_layout=True)
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
    cmap = colors_module.ListedColormap(("#4DAF4A", "#377EB8", "#E41A1C"))
    norm = colors_module.BoundaryNorm((0.5, 1.5, 2.5, 3.5), cmap.N)
    overlay_axis.imshow(region_map, cmap=cmap, norm=norm, alpha=0.62)
    overlay_axis.set_title("Reference evidence", fontweight="bold")
    overlay_axis.axis("off")

    legend_handles = (
        patches.Patch(color="#4DAF4A", label="Edema"),
        patches.Patch(color="#377EB8", label="Non-enhancing"),
        patches.Patch(color="#E41A1C", label="Enhancing"),
    )
    overlay_axis.legend(
        handles=legend_handles,
        loc="lower right",
        ncol=1,
        frameon=True,
        facecolor="white",
        framealpha=0.82,
        fontsize=7,
    )

    diagram_axis = figure.add_subplot(grid[1, :])
    diagram_axis.set_xlim(0.0, 1.0)
    diagram_axis.set_ylim(0.0, 1.0)
    diagram_axis.axis("off")
    _box(
        diagram_axis,
        patches,
        0.02,
        0.25,
        0.16,
        0.5,
        "4 MRI contrasts\n+ availability mask",
        "#DDEAF3",
    )
    _box(
        diagram_axis,
        patches,
        0.23,
        0.25,
        0.16,
        0.5,
        "Shared 3D encoder\n+ sequence identity",
        "#D8EAD3",
    )
    _box(diagram_axis, patches, 0.44, 0.25, 0.16, 0.5, "Question-conditioned\nfusion", "#FFF0C7")
    _box(diagram_axis, patches, 0.69, 0.57, 0.14, 0.3, "Answer", "#E8D8EE")
    _box(diagram_axis, patches, 0.69, 0.15, 0.14, 0.3, "Voxel evidence", "#F5D0D0")
    _box(diagram_axis, patches, 0.86, 0.36, 0.12, 0.3, "Abstain when\nunsupported", "#E6E6E6")
    _arrow(diagram_axis, patches, (0.18, 0.5), (0.23, 0.5))
    _arrow(diagram_axis, patches, (0.39, 0.5), (0.44, 0.5))
    _arrow(diagram_axis, patches, (0.60, 0.5), (0.69, 0.72))
    _arrow(diagram_axis, patches, (0.60, 0.5), (0.69, 0.30))
    _arrow(diagram_axis, patches, (0.83, 0.72), (0.86, 0.56))
    diagram_axis.text(
        0.52,
        0.08,
        "Matched comparison: answer-only vs answer + voxel-evidence supervision "
        "and balanced contrast dropout",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
    figure.suptitle(
        "Voxel-grounded 3D MRI vision-language reasoning under missing contrasts",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.01,
        0.955,
        "A  Real co-registered inputs and reference regions: "
        f"{selected.case_id}, axial index {axial_index}",
        fontsize=11,
        fontweight="bold",
    )
    figure.text(0.01, 0.435, "B  Controlled model and outputs", fontsize=11, fontweight="bold")

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
) -> None:
    box = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015",
        facecolor=color,
        edgecolor="#333333",
        linewidth=1.2,
    )
    axis.add_patch(box)
    axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=10)


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

"""Model capability declarations used before heavyweight adapters are installed."""

from dataclasses import dataclass
from enum import StrEnum


class ModelRole(StrEnum):
    EXTERNAL_BASELINE = "external_baseline"
    CONTROLLED_MODEL = "controlled_model"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    role: ModelRole
    license_id: str
    parameter_billions: float
    supports_3d: bool
    supports_multiple_sequences: bool
    supports_text_generation: bool
    supports_voxel_grounding: bool
    requires_remote_code_review: bool

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.license_id.strip():
            raise ValueError("model ID and license must be non-empty")
        if self.parameter_billions <= 0.0:
            raise ValueError("parameter count must be positive")


M3D_LAMED_PHI3 = ModelSpec(
    model_id="GoodBaiBai88/M3D-LaMed-Phi-3-4B",
    role=ModelRole.EXTERNAL_BASELINE,
    license_id="Apache-2.0",
    parameter_billions=4.0,
    supports_3d=True,
    supports_multiple_sequences=False,
    supports_text_generation=True,
    supports_voxel_grounding=True,
    requires_remote_code_review=True,
)

MRI_VLM_SMALL = ModelSpec(
    model_id="mri-vlm-small",
    role=ModelRole.CONTROLLED_MODEL,
    license_id="MIT",
    parameter_billions=0.3,
    supports_3d=True,
    supports_multiple_sequences=True,
    supports_text_generation=True,
    supports_voxel_grounding=True,
    requires_remote_code_review=False,
)


def validate_comparison(left: ModelSpec, right: ModelSpec) -> None:
    """Reject capability mismatches that would invalidate a claimed matched comparison."""
    if left.supports_3d != right.supports_3d:
        raise ValueError("models disagree on 3D input support")
    if left.supports_multiple_sequences != right.supports_multiple_sequences:
        raise ValueError("models do not receive equivalent MRI sequence inputs")

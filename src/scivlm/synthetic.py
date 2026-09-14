"""Small deterministic metadata fixture for V0 protocol audits."""

import hashlib

from scivlm.fingerprint import sha256_json
from scivlm.schema import FigureTextRecord


def fixture_records(count: int = 60) -> tuple[FigureTextRecord, ...]:
    if count < 3:
        raise ValueError("fixture requires at least three records")

    records: list[FigureTextRecord] = []
    families = ("trend", "comparison", "correlation")
    relations = ("increasing", "greater-than", "positive-correlation")
    chart_types = ("line", "bar", "scatter")
    styles = ("light-grid", "dark-grid", "minimal")
    for index in range(count):
        family_index = index % len(families)
        spec = {
            "generator_version": "v0-fixture-1",
            "family": families[family_index],
            "group": f"group-{index:04d}",
            "relation": relations[family_index],
            "values": [index, index + family_index + 1, index + 2 * family_index + 2],
            "style": styles[(index // len(families)) % len(styles)],
        }
        spec_digest = sha256_json(spec)
        image_digest = hashlib.sha256(f"unrendered:{spec_digest}".encode()).hexdigest()
        records.append(
            FigureTextRecord(
                figure_id=f"figure-{index:04d}",
                caption_id=f"caption-{index:04d}",
                caption=f"Synthetic {families[family_index]} relation {relations[family_index]}.",
                source_group=f"group-{index:04d}",
                generator_family=families[family_index],
                generator_version="v0-fixture-1",
                chart_type=chart_types[family_index],
                relation_type=relations[family_index],
                visual_style=styles[(index // len(families)) % len(styles)],
                spec_sha256=spec_digest,
                image_sha256=image_digest,
            )
        )
    return tuple(records)

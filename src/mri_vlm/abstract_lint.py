"""Validate working ISMRM abstract limits without treating planned text as results."""

import argparse
import json
import re
from pathlib import Path
from typing import Any


def section(text: str, heading: str, *, level: int) -> str:
    marker = f"{'#' * level} {heading}\n"
    if marker not in text:
        raise ValueError(f"missing heading: {marker.strip()}")
    content = text.split(marker, 1)[1]
    if level == 1:
        return content.splitlines()[0].strip()
    next_heading = re.search(rf"^#{{1,{level}}} ", content, flags=re.MULTILINE)
    if next_heading:
        content = content[: next_heading.start()]
    return content.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def validate_abstract(text: str) -> dict[str, Any]:
    title = section(text, "Title", level=1)
    synopsis = section(text, "Synopsis", level=2)
    synopsis_parts = {
        name: section(synopsis, name, level=3)
        for name in ("Motivation", "Goal(s)", "Approach", "Results")
    }
    impact = section(text, "Impact", level=2)
    body = section(text, "Main Body", level=2)
    body_parts = {
        name: section(body, name, level=3)
        for name in ("Introduction", "Methods", "Results", "Discussion", "Conclusion")
    }
    captions = [
        match.group(1).strip()
        for match in re.finditer(
            r"^### Figure \d+\n(.*?)(?=^### Figure \d+|^## |\Z)",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
    ]
    title_characters = len(title)
    synopsis_words = sum(word_count(value) for value in synopsis_parts.values())
    impact_words = word_count(impact)
    body_words = sum(word_count(value) for value in body_parts.values())
    caption_characters = [len(caption) for caption in captions]
    counts: dict[str, Any] = {
        "title_characters": title_characters,
        "synopsis_words": synopsis_words,
        "impact_words": impact_words,
        "body_words": body_words,
        "caption_characters": caption_characters,
    }
    errors: list[str] = []
    if title_characters > 125:
        errors.append("title exceeds 125 characters")
    if synopsis_words > 100:
        errors.append("synopsis exceeds 100 words")
    if impact_words > 40:
        errors.append("impact exceeds 40 words")
    if body_words > 750:
        errors.append("body exceeds 750 words")
    if len(captions) > 5:
        errors.append("more than five figure captions")
    if any(length > 500 for length in caption_characters):
        errors.append("a figure caption exceeds 500 characters")
    planned = text.count("[PLANNED")
    heldout = text.count("[HELD-OUT")
    unresolved = planned + heldout
    return {
        "status": "invalid" if errors else ("draft" if unresolved else "submission-ready"),
        "planned_markers": planned,
        "heldout_markers": heldout,
        "unresolved_markers": unresolved,
        "counts": counts,
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate ISMRM abstract limits")
    parser.add_argument("draft", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = validate_abstract(args.draft.read_text(encoding="utf-8"))
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from mri_vlm.abstract_lint import validate_abstract, word_count


def _draft(*, title: str = "Short title", result: str = "[PLANNED result]") -> str:
    return f"""# Title
{title}

## Synopsis
### Motivation
Missing MRI contrasts are common.
### Goal(s)
Test grounded reasoning.
### Approach
Compare matched models.
### Results
{result}

## Impact
This may improve model evaluation.

## Main Body
### Introduction
Motivation.
### Methods
Methods.
### Results
{result}
### Discussion
Interpretation.
### Conclusion
Conclusion.
### References
None.

## Figure Captions
### Figure 1
An independently understandable caption.
"""


def test_word_count_handles_hyphenated_terms() -> None:
    assert word_count("subject-level MRI-VLM") == 2


def test_planned_abstract_is_valid_draft_not_submission_ready() -> None:
    result = validate_abstract(_draft())

    assert result["status"] == "draft"
    assert result["planned_markers"] == 2
    assert result["errors"] == []


def test_completed_abstract_without_markers_is_submission_ready() -> None:
    result = validate_abstract(_draft(result="Accuracy was 0.80."))

    assert result["status"] == "submission-ready"


def test_title_limit_is_enforced() -> None:
    result = validate_abstract(_draft(title="x" * 126))

    assert result["status"] == "invalid"
    assert "title exceeds 125 characters" in result["errors"]

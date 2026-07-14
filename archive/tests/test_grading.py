"""Unit tests for math answer extraction and grading."""

from __future__ import annotations

import pytest

from reasoning_branch_dataset.grading import (
    extract_math_answer,
    grade_math_answer,
    math_equal,
)

GRADER_TESTS = [
    (r"\left(3,\frac{\pi}{2}\right)", r"(3,\frac{\pi}{2})", True),
    (r"\boxed{(3,\frac{\pi}{2})}", r"\left(3,\frac{\pi}{2}\right)", True),
    (r"p-q", r"p - q", True),
    (r"\frac{1}{2}", r"0.5", True),
    (r"\frac{14}{3}", r"14/3", True),
    (r"9", r"9", True),
]


@pytest.mark.parametrize("ref,pred,expected", GRADER_TESTS)
def test_math_equal(ref: str, pred: str, expected: bool) -> None:
    assert math_equal(pred, ref) is expected


def test_extract_boxed() -> None:
    text = r"Reasoning...\n\boxed{(3, \frac{\pi}{2})}"
    assert extract_math_answer(text) == r"(3, \frac{\pi}{2})"


def test_grade_math_answer_ok() -> None:
    result = grade_math_answer(r"\boxed{(3, \frac{\pi}{2})}", r"\left( 3, \frac{\pi}{2} \right)")
    assert result["evaluation_status"] == "OK"
    assert result["is_correct"] == 1
    assert result["predicted_answer"] != ""


def test_grade_math_answer_empty_is_error() -> None:
    result = grade_math_answer("no answer here without boxed", "42")
    assert result["evaluation_status"] == "ERROR"
    assert result["is_correct"] is None


def test_grade_math_answer_wrong_is_ok_status() -> None:
    result = grade_math_answer(r"\boxed{7}", "9")
    assert result["evaluation_status"] == "OK"
    assert result["is_correct"] == 0

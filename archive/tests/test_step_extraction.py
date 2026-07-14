from reasoning_branch_dataset.action_study.step_extraction import (
    extract_handoff_step,
    strip_model_thinking,
)


def test_strip_model_thinking_prefers_visible_after_close():
    raw = "\x3cthink\x3einternal\x3c/think\x3e\n\nLet x be the count of composites."
    assert strip_model_thinking(raw) == "Let x be the count of composites."


def test_extract_handoff_step_from_thinking_only_block():
    raw = "\x3cthink\x3e" + "We should subtract primes from composites. " * 3 + "\x3c/think\x3e"
    step = extract_handoff_step(raw)
    assert step
    assert "subtract" in step.lower()


def test_extract_handoff_step_from_visible_paragraph():
    raw = "Step 1: The total composites below 1000 are 831.\n\nNext we exclude multiples of 2, 3, and 5."
    step = extract_handoff_step(raw)
    assert step.startswith("Step 1")

"""Math answer extraction and grading."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

try:
    from sympy import N, simplify
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    _SYMPY = True
    _TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)
except ImportError:  # pragma: no cover
    _SYMPY = False
    _TRANSFORMATIONS = ()


def _extract_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    idx = text.find(marker)
    if idx < 0:
        return None
    start = idx + len(marker)
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start : i - 1].strip()
    return None


def extract_math_answer(text: str, *, require_marker: bool = False) -> str:
    boxed = _extract_boxed(text)
    if boxed:
        return boxed
    m = re.search(r"####\s*([^\n]+)", text)
    if m:
        return m.group(1).strip()
    if require_marker:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def _strip_latex_wrappers(s: str) -> str:
    s = s.strip()
    boxed = _extract_boxed(s)
    if boxed:
        s = boxed
    s = re.sub(r"\\left\s*", "", s)
    s = re.sub(r"\\right\s*", "", s)
    s = re.sub(r"\\dfrac", r"\\frac", s)
    s = re.sub(r"\\tfrac", r"\\frac", s)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\$+", "", s)
    return s.strip()


def _normalize_text(s: str) -> str:
    s = _strip_latex_wrappers(s)
    s = re.sub(r"\s+", "", s)
    return s.lower()


def _split_top_level_tuple(s: str) -> list[str] | None:
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return None
    inner = s[1:-1].strip()
    if not inner:
        return []
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in inner:
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return parts


def _latex_frac_to_sympy(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = s.replace(r"\pi", "pi")
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\cdot", "*", s)
    s = re.sub(r"\\times", "*", s)
    s = re.sub(r"\\,", "", s)
    s = re.sub(r"\\;", "", s)
    s = re.sub(r"\\!", "", s)
    s = re.sub(r"\\left\s*", "", s)
    s = re.sub(r"\\right\s*", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def _try_float(s: str) -> float | None:
    s = _strip_latex_wrappers(s)
    s = _latex_frac_to_sympy(s)
    if not s:
        return None
    try:
        if _SYMPY:
            val = parse_expr(s, transformations=_TRANSFORMATIONS, evaluate=True)
            return float(N(val))
    except Exception:
        pass
    try:
        return float(Fraction(s))
    except Exception:
        pass
    try:
        return float(s)
    except Exception:
        return None


def _sympy_equal(a: str, b: str) -> bool:
    if not _SYMPY:
        return False
    try:
        ea = parse_expr(_latex_frac_to_sympy(a), transformations=_TRANSFORMATIONS, evaluate=True)
        eb = parse_expr(_latex_frac_to_sympy(b), transformations=_TRANSFORMATIONS, evaluate=True)
        if simplify(ea - eb) == 0:
            return True
        fa = float(N(ea))
        fb = float(N(eb))
        return abs(fa - fb) < 1e-6
    except Exception:
        return False


def math_equal(pred: str, ref: str) -> bool:
    pred = _strip_latex_wrappers(extract_math_answer(pred)).strip()
    ref = _strip_latex_wrappers(ref).strip()
    if not pred or not ref:
        return False
    if pred == ref:
        return True

    pred_n = _normalize_text(pred)
    ref_n = _normalize_text(ref)
    if pred_n == ref_n:
        return True

    pred_parts = _split_top_level_tuple(pred)
    ref_parts = _split_top_level_tuple(ref)
    if pred_parts is not None and ref_parts is not None and len(pred_parts) == len(ref_parts):
        return all(math_equal(p, r) for p, r in zip(pred_parts, ref_parts))

    fa = _try_float(pred)
    fb = _try_float(ref)
    if fa is not None and fb is not None and abs(fa - fb) < 1e-6:
        return True

    if _sympy_equal(pred, ref):
        return True

    if pred_n in ref_n or ref_n in pred_n:
        return True
    return False


def has_boxed_answer(text: str) -> bool:
    return _extract_boxed(text) is not None


def grade_math_answer(response_text: str, gold: str, *, require_marker: bool = True) -> dict[str, Any]:
    """Extract answer, score against gold, and surface evaluation errors."""
    return classify_generation_outcome(
        response_text,
        gold,
        finish_reason=None,
        require_marker=require_marker,
    )


def classify_generation_outcome(
    response_text: str,
    gold: str,
    *,
    finish_reason: str | None = None,
    require_marker: bool = True,
) -> dict[str, Any]:
    """Classify generation as OK / TRUNCATED / NO_FINAL_ANSWER / ERROR."""
    boxed = has_boxed_answer(response_text)
    try:
        if finish_reason == "length":
            extracted = extract_math_answer(response_text, require_marker=require_marker)
            return {
                "final_answer": extracted or "",
                "predicted_answer": extracted or "",
                "is_correct": None,
                "evaluation_status": "TRUNCATED",
                "evaluation_error": "max_tokens_reached",
                "has_boxed_answer": boxed,
                "finish_reason": finish_reason,
            }

        extracted = extract_math_answer(response_text, require_marker=require_marker)
        if not extracted or not extracted.strip():
            return {
                "final_answer": extracted or "",
                "predicted_answer": extracted or "",
                "is_correct": None,
                "evaluation_status": "NO_FINAL_ANSWER",
                "evaluation_error": "empty_extracted_answer",
                "has_boxed_answer": boxed,
                "finish_reason": finish_reason,
            }

        correct = math_equal(extracted, gold)
        return {
            "final_answer": extracted,
            "predicted_answer": extracted,
            "is_correct": int(correct),
            "evaluation_status": "OK",
            "evaluation_error": None,
            "has_boxed_answer": boxed,
            "finish_reason": finish_reason,
        }
    except Exception as exc:
        return {
            "final_answer": "",
            "predicted_answer": "",
            "is_correct": None,
            "evaluation_status": "ERROR",
            "evaluation_error": str(exc),
            "has_boxed_answer": boxed,
            "finish_reason": finish_reason,
        }

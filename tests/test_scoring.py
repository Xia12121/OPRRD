"""Answer-extraction / scoring regression tests (no dataset download needed).

These determine every reported accuracy, so they are pinned explicitly.
"""

from src.data.load_tasks import (
    extract_boxed,
    extract_final_number,
    extract_gsm8k_gold,
    normalize_math,
    numbers_equal,
    score,
)


def test_gsm8k_gold_and_pred():
    assert extract_gsm8k_gold("Janet ...\n#### 18") == "18"
    assert extract_final_number("... so the total is 1,234.0 dollars") == "1234.0"
    assert score("gsm8k", "Therefore the answer is 18.", {"gold": "18"}) is True
    assert score("gsm8k", "the answer is 19", {"gold": "18"}) is False


def test_numbers_equal_tolerance():
    assert numbers_equal("18", "18.0")
    assert numbers_equal("1,000", "1000")
    assert not numbers_equal("18", "19")
    assert not numbers_equal(None, "1")


def test_boxed_balanced_braces():
    assert extract_boxed("ans \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert extract_boxed("\\boxed{42}") == "42"
    assert extract_boxed("no box here") is None
    # last boxed wins
    assert extract_boxed("\\boxed{1} ... \\boxed{2}") == "2"


def test_math_normalisation_equivalences():
    assert normalize_math("\\frac{1}{2}") == normalize_math("\\dfrac{1}{2}")
    assert normalize_math("\\tfrac{1}{2}") == normalize_math("\\frac{1}{2}")
    assert normalize_math("x = 5") == "5"
    assert normalize_math("5\\%") == "5"
    assert normalize_math("\\left(3\\right)") == "(3)"


def test_math_scoring():
    gold = normalize_math("\\frac{1}{2}")
    assert score("math", "hence \\boxed{\\frac12}", {"gold": gold}) is True
    assert score("math", "\\boxed{3}", {"gold": normalize_math("3.0")}) is True
    assert score("math", "\\boxed{7}", {"gold": normalize_math("8")}) is False


def test_arc_and_boolq():
    arc_ex = {"gold": "C", "meta": {"labels": ["A", "B", "C", "D"]}}
    assert score("arc_challenge", "The answer is (C).", arc_ex) is True
    assert score("arc_challenge", "I think B is right", arc_ex) is False
    assert score("boolq", "Yes, the passage supports it.", {"gold": "yes"}) is True
    assert score("boolq", "No.", {"gold": "no"}) is True
    assert score("boolq", "It is true.", {"gold": "yes"}) is True


def test_arc_extraction_not_fooled_by_prose():
    """Regression: the old char-scan returned the first A-E letter of any word."""
    arc = {"gold": "D", "meta": {"labels": ["A", "B", "C", "D"]}}
    # 'Apples...' must NOT be read as 'A'; the real answer marker is 'D'.
    assert score("arc_challenge", "Apples grow on trees, so the choice is D", arc) is True
    arc_c = {"gold": "C", "meta": {"labels": ["A", "B", "C", "D"]}}
    # enumerated options then a conclusion -> take the LAST marked option.
    assert score("arc_challenge", "Option A is wrong; option C is correct.", arc_c) is True
    # no real answer letter present -> not a coin-flip match.
    assert score("arc_challenge", "The answer choice depends on Boron.", arc_c) is False


def test_arc_numeric_labels():
    arc = {"gold": "3", "meta": {"labels": ["1", "2", "3", "4"]}}
    assert score("arc_challenge", "Therefore the answer is 3.", arc) is True
    assert score("arc_challenge", "I pick 2", arc) is False


def test_math_text_wrapper_normalisation():
    """Regression: \\text{...} must drop its braces too (was -> '{5}')."""
    assert normalize_math("\\text{5}") == "5"
    assert normalize_math("\\mathrm{kg}") == "kg"
    assert normalize_math("\\boxed" ) == "\\boxed"  # unrelated token untouched
    gold = normalize_math("5")
    assert score("math", "The final answer is \\boxed{\\text{5}}.", {"gold": gold}) is True


def test_math_numeric_fallback_gated_on_numeric_gold():
    # non-numeric gold + no \boxed -> a bare number must NOT falsely match
    assert score("math", "the answer is 2", {"gold": normalize_math("\\frac{1}{2}")}) is False
    # numeric gold + no \boxed -> last-number fallback is allowed
    assert score("math", "so the total is 5", {"gold": normalize_math("5")}) is True

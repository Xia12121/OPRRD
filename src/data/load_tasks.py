"""Task loading, few-shot prompting, and answer scoring.

Tasks (IMPLEMENTATION_SPEC §0): GSM8K, MATH (Hendrycks competition_math),
ARC-Challenge, BoolQ. Each task exposes:
  * ``load_task(split, n)``        -> list of normalised examples.
  * ``build_prompt(example, k)``   -> few-shot question prompt string.
  * ``score(prediction, example)`` -> bool correct.

Scoring follows the community-standard extraction (GSM8K final number; MATH
\\boxed{} + Hendrycks normalisation; ARC option letter; BoolQ yes/no) so the
reported accuracies are comparable to published baselines.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Numeric / MATH answer normalisation                                          #
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def _to_number(s: str) -> Optional[float]:
    s = s.strip().replace("$", "").replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def extract_final_number(text: str) -> Optional[str]:
    """Last number in a generation (handles $, commas, decimals)."""
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    return matches[-1].replace("$", "").replace(",", "").rstrip(".")


def extract_gsm8k_gold(answer_field: str) -> Optional[str]:
    """Gold number after the ``####`` marker in a GSM8K answer."""
    m = re.search(r"####\s*(.+)", answer_field)
    if not m:
        return None
    return m.group(1).strip().replace("$", "").replace(",", "")


def numbers_equal(a: Optional[str], b: Optional[str], tol: float = 1e-4) -> bool:
    if a is None or b is None:
        return False
    fa, fb = _to_number(a), _to_number(b)
    if fa is not None and fb is not None:
        return abs(fa - fb) <= tol
    return a.strip() == b.strip()


def extract_boxed(text: str) -> Optional[str]:
    """Extract the content of the last ``\\boxed{...}`` (balanced braces)."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx + len("\\boxed")
    if i < len(text) and text[i] == " ":
        i += 1
    if i >= len(text) or text[i] != "{":
        # \boxed 1234 form
        m = re.match(r"\\boxed\s*(\S+)", text[idx:])
        return m.group(1) if m else None
    depth, start = 0, i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : j]
    return None


def normalize_math(s: Optional[str]) -> Optional[str]:
    """Hendrycks-MATH answer normalisation (compact port)."""
    if s is None:
        return None
    s = s.strip()
    # Remove common wrappers / formatting.
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "").replace(" ", "")
    # Remove \text{...}/\mathrm{...}/\mathbf{...} wrappers AND their braces,
    # keeping the inner content (a bare ``replace("\\text","")`` would orphan the
    # braces, so \boxed{\text{5}} -> {5} != 5). Loop a few times for nesting.
    for _ in range(3):
        s = re.sub(
            r"\\(?:text|mathrm|mathbf|mathit|mbox|textbf|textnormal)\s*\{([^{}]*)\}",
            r"\1",
            s,
        )
    # Strip any leftover brace-less wrapper tokens.
    for tok in ("\\text", "\\mathrm", "\\mathbf", "\\mathit", "\\mbox"):
        s = s.replace(tok, "")
    s = s.replace("dollar", "").replace("\\cdot", "")
    # Normalise \dfrac / \tfrac to \frac FIRST, so the \frac handling below
    # applies uniformly (otherwise \dfrac{1}{2} and \frac{1}{2} normalise apart).
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    # \frac a b -> \frac{a}{b}; then drop the leading backslash.
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = s.replace("\\frac", "frac")
    s = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", s)
    if s.endswith("."):
        s = s[:-1]
    # x = 5 -> 5
    if "=" in s:
        s = s.split("=")[-1]
    return s


# --------------------------------------------------------------------------- #
# Per-task loaders                                                             #
# --------------------------------------------------------------------------- #
def _hf_load(path: str, name: Optional[str], split: str):
    from datasets import load_dataset

    return load_dataset(path, name) if name else load_dataset(path)


def load_task(task: str, split: str = "test", n: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return a list of normalised examples for ``task``.

    Each example has: ``question`` (str), ``gold`` (scoring target),
    ``solution`` (gold reasoning for SFT, may be ""), and ``meta`` (raw fields).
    """
    from datasets import load_dataset

    task = task.lower()
    out: List[Dict[str, Any]] = []

    if task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        for ex in ds:
            out.append(
                {
                    "question": ex["question"],
                    "gold": extract_gsm8k_gold(ex["answer"]),
                    "solution": ex["answer"],
                    "meta": {},
                }
            )
    elif task == "math":
        # The original `hendrycks/competition_math` was removed from the Hub
        # (DMCA). `nlile/hendrycks-MATH-benchmark` is the same MATH data with
        # fields: problem, solution, answer, subject, level, unique_id
        # (train 12000 / test 500). Gold = the `answer` field, falling back to
        # the \boxed{} content of the solution.
        ds = load_dataset("nlile/hendrycks-MATH-benchmark", split=split)
        for ex in ds:
            ans = ex.get("answer") or extract_boxed(ex["solution"])
            out.append(
                {
                    "question": ex["problem"],
                    "gold": normalize_math(ans),
                    "solution": ex["solution"],
                    "meta": {"level": ex.get("level"), "type": ex.get("subject")},
                }
            )
    elif task in ("arc_challenge", "arc"):
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
        for ex in ds:
            labels = ex["choices"]["label"]
            texts = ex["choices"]["text"]
            choices = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
            gold = ex["answerKey"]
            gold_text = dict(zip(labels, texts)).get(gold, "")
            out.append(
                {
                    "question": f"{ex['question']}\n{choices}",
                    "gold": gold,
                    "solution": f"The answer is {gold}. {gold_text}",
                    "meta": {"labels": labels},
                }
            )
    elif task == "boolq":
        # BoolQ ships only train/validation (the test labels are held out), so
        # a requested 'test' split is served from 'validation'.
        if split == "test":
            split = "validation"
        ds = load_dataset("google/boolq", split=split)
        for ex in ds:
            ans = "yes" if ex["answer"] else "no"
            out.append(
                {
                    "question": f"{ex['passage']}\nQuestion: {ex['question']}?\nAnswer yes or no.",
                    "gold": ans,
                    "solution": ans,
                    "meta": {},
                }
            )
    else:
        raise ValueError(f"unknown task '{task}'")

    if n is not None:
        out = out[:n]
    return out


# --------------------------------------------------------------------------- #
# Scoring                                                                       #
# --------------------------------------------------------------------------- #
def _extract_choice(text: str, labels) -> Optional[str]:
    """Extract a multiple-choice answer (letter or digit) from a generation.

    Robust to prose: we do NOT scan for the first stray letter. We look for an
    option marker (answer/option/choice/correct) and take the standalone option
    token that follows it, preferring the LAST such marker (models often list
    options then conclude). Fall back to the last parenthesised/punctuated
    option token. The option alphabet is taken from the example's actual labels
    (handles both ``A-D`` and ``1-4`` label sets).
    """
    chars = "".join(re.escape(str(l)) for l in labels)
    if not chars:
        return None
    label_set = {str(l) for l in labels}
    tok = rf"\(?\b([{chars}])\b\)?"  # a standalone option token, optional parens

    # Primary: the token following an answer marker (last marker wins).
    found = []
    for m in re.finditer(r"(?i)\b(?:answer|option|choice|correct|select)\b", text):
        mm = re.search(tok, text[m.end() : m.end() + 12])
        if mm and mm.group(1) in label_set:
            found.append(mm.group(1))
    if found:
        return found[-1]

    # Fallback: last parenthesised "(C)" or punctuated "C)" / "C." / "C:" token.
    cands = []
    for m in re.finditer(rf"\(([{chars}])\)|\b([{chars}])[).:]", text):
        c = m.group(1) or m.group(2)
        if c in label_set:
            cands.append(c)
    return cands[-1] if cands else None


def score(task: str, prediction: str, example: Dict[str, Any]) -> bool:
    """Return whether ``prediction`` is correct for ``example`` under ``task``."""
    task = task.lower()
    if task == "gsm8k":
        return numbers_equal(extract_final_number(prediction), example["gold"])
    if task == "math":
        gold = example["gold"]
        if gold is None:
            return False
        boxed = extract_boxed(prediction)
        if boxed is not None:
            pred = normalize_math(boxed)
        elif _to_number(gold) is not None:
            # No \boxed{}: only trust the last-number fallback for NUMERIC golds;
            # for non-numeric golds (fractions, sets, expressions) a bare number
            # can never be correct, so do not falsely match.
            pred = normalize_math(extract_final_number(prediction))
        else:
            return False
        if pred is None:
            return False
        return pred == gold or numbers_equal(pred, gold)
    if task in ("arc_challenge", "arc"):
        labels = example["meta"].get("labels", ["A", "B", "C", "D", "E"])
        return _extract_choice(prediction, labels) == example["gold"]
    if task == "boolq":
        low = prediction.lower()
        yes = re.search(r"\byes\b|\btrue\b", low)
        no = re.search(r"\bno\b|\bfalse\b", low)
        if yes and not no:
            pred = "yes"
        elif no and not yes:
            pred = "no"
        elif yes and no:  # take whichever appears first
            pred = "yes" if yes.start() < no.start() else "no"
        else:
            pred = None
        return pred == example["gold"]
    raise ValueError(f"unknown task '{task}'")


# --------------------------------------------------------------------------- #
# Few-shot prompting                                                            #
# --------------------------------------------------------------------------- #
_GSM8K_FEWSHOT = [
    (
        "Natalia sold clips to 48 friends in April, and half as many in May. "
        "How many clips did she sell altogether?",
        "In May she sold 48 / 2 = 24 clips. Altogether 48 + 24 = 72 clips.\nThe answer is 72.",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday she babysat for 50 minutes. "
        "How much did she earn?",
        "Per minute she earns 12 / 60 = $0.2. For 50 minutes she earned 0.2 * 50 = $10.\n"
        "The answer is 10.",
    ),
    (
        "Betty is saving for a $100 wallet and has half of the money. Her parents give "
        "her $15, and her grandparents give twice as much as her parents. How much more "
        "money does she need?",
        "Betty has 100 / 2 = $50. Grandparents give 2 * 15 = $30. Now she has "
        "50 + 15 + 30 = $95. She still needs 100 - 95 = $5.\nThe answer is 5.",
    ),
    (
        "James writes a 3-page letter to 2 different friends twice a week. How many "
        "pages does he write a year?",
        "Each time he writes 3 * 2 = 6 pages. Twice a week that is 6 * 2 = 12 pages. "
        "In a year he writes 12 * 52 = 624 pages.\nThe answer is 624.",
    ),
]

# MATH exemplars demonstrate the \boxed{} final-answer convention so the boxed
# extraction path in score() is actually exercised (non-numeric golds otherwise
# can never match a last-number fallback).
_MATH_FEWSHOT = [
    (
        "What is the value of $\\sqrt{36 + 64}$?",
        "We have $36 + 64 = 100$, and $\\sqrt{100} = 10$. The final answer is $\\boxed{10}$.",
    ),
    (
        "If $2x + 3 = 11$, what is the value of $x$?",
        "Subtracting 3 gives $2x = 8$, so $x = 4$. The final answer is $\\boxed{4}$.",
    ),
    (
        "Simplify $\\frac{1}{2} + \\frac{1}{3}$.",
        "A common denominator gives $\\frac{3}{6} + \\frac{2}{6} = \\frac{5}{6}$. "
        "The final answer is $\\boxed{\\frac{5}{6}}$.",
    ),
    (
        "How many distinct primes are factors of $12$?",
        "We have $12 = 2^2 \\cdot 3$, whose prime factors are $2$ and $3$. "
        "The final answer is $\\boxed{2}$.",
    ),
]

_MATH_INSTRUCTION = (
    "Solve the problem step by step and put your final answer inside \\boxed{}.\n\n"
)


def build_fewshot_prefix(task: str, k: int = 4) -> str:
    """A fixed few-shot prefix used by the pre-flight gate and eval."""
    task = task.lower()
    shots = None
    if task == "gsm8k":
        shots = _GSM8K_FEWSHOT
    elif task == "math":
        shots = _MATH_FEWSHOT
    if shots is None:
        return ""  # ARC/BoolQ are scored zero-shot via instructions in the question
    return "".join(f"Question: {q}\nAnswer: {a}\n\n" for q, a in shots[:k])


def build_prompt(task: str, example: Dict[str, Any], k: int = 4) -> str:
    """Few-shot question prompt (the user turn handed to the chat template)."""
    task_l = task.lower()
    prefix = build_fewshot_prefix(task, k)
    if task_l == "math":
        return f"{_MATH_INSTRUCTION}{prefix}Question: {example['question']}\nAnswer:"
    if task_l == "gsm8k":
        return f"{prefix}Question: {example['question']}\nAnswer:"
    return example["question"]

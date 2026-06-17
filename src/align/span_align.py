"""Cross-tokenizer span alignment (Task A).

The cross-architecture difficulty is that teacher and student use *different*
tokenizers, so token index `t` is not the same text piece in both. We sidestep
this entirely by working in **character space**: a span is a half-open character
interval ``[a, b)`` over the shared text ``Z``, and each model mean-pools the
hidden states of whichever of *its own* tokens overlap that interval. The same
list of character spans is fed to both models -- that shared list is the bridge
(method-doc §3, IMPLEMENTATION_SPEC Task A).

Membership rule (non-negotiable, spec §2.2): token with char interval ``[c0,c1)``
counts in span ``[a,b)`` iff the intervals overlap, i.e. ``c0 < b and c1 > a``,
with degenerate tokens (``c1 <= c0``, e.g. special tokens whose offset is (0,0))
and empty spans (``b <= a``) excluded.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import torch

Span = Tuple[int, int]


def mean_pool_by_charspan(
    hidden: torch.Tensor,
    offsets: Sequence[Tuple[int, int]],
    spans: Sequence[Span],
    eps: float = 1e-6,  # kept for signature compatibility with the spec
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool ``hidden`` over the tokens overlapping each char span.

    Args:
        hidden: ``[T, d]`` hidden states for one layer of one sequence.
        offsets: length-``T`` char offsets ``[(c0, c1), ...]`` for the same
            tokenizer that produced ``hidden`` (HF ``return_offsets_mapping``).
        spans: length-``K`` char spans ``[(a, b), ...]`` over the shared text.
        eps: unused; present so the call matches the spec signature.

    Returns:
        ``(H, valid)`` with ``H: [K, d]`` (mean-pooled, 0 rows for empty spans)
        and ``valid: [K]`` bool marking spans that captured >= 1 token. Callers
        drop ``valid == False`` rows before building the Gram matrix.

    Independent of token *index*; depends only on character-interval overlap, so
    the very same ``spans`` aligns a Llama tokenisation and a Qwen tokenisation.
    """
    if hidden.dim() != 2:
        raise ValueError(f"hidden must be [T, d], got shape {tuple(hidden.shape)}")
    T, d = hidden.shape
    if len(offsets) != T:
        raise ValueError(f"offsets length {len(offsets)} != hidden T {T}")

    K = len(spans)
    H = hidden.new_zeros(K, d)
    valid = torch.zeros(K, dtype=torch.bool, device=hidden.device)
    if K == 0 or T == 0:
        return H, valid

    off = torch.as_tensor(
        [[int(c0), int(c1)] for c0, c1 in offsets], dtype=torch.long
    )  # [T, 2], on CPU -- membership is integer logic, kept off the autograd path
    sp = torch.as_tensor([[int(a), int(b)] for a, b in spans], dtype=torch.long)  # [K,2]

    c0 = off[:, 0].unsqueeze(1)  # [T, 1]
    c1 = off[:, 1].unsqueeze(1)  # [T, 1]
    a = sp[:, 0].unsqueeze(0)  # [1, K]
    b = sp[:, 1].unsqueeze(0)  # [1, K]

    # Overlap of [c0,c1) and [a,b), excluding degenerate tokens and empty spans.
    overlap = (c0 < b) & (c1 > a) & (c1 > c0) & (b > a)  # [T, K] bool
    overlap = overlap.to(hidden.device)

    counts = overlap.sum(dim=0)  # [K]
    valid = counts > 0

    weights = overlap.to(hidden.dtype)  # [T, K]; constant, carries no gradient
    summed = weights.t() @ hidden  # [K, d]; gradient flows from hidden
    denom = counts.clamp(min=1).to(hidden.dtype).unsqueeze(1)  # [K, 1]
    H = summed / denom
    return H, valid


# --------------------------------------------------------------------------- #
# Span selection strategies (response region only) -- IMPLEMENTATION_SPEC §3.   #
#   last_k | uniform | reasoning_step | all                                     #
# All operate on character offsets so the result is tokenizer-agnostic.         #
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"\S+")
# A reasoning "step" ends at a newline or a sentence terminator (ASCII + CJK).
_SENT_SPLIT_RE = re.compile(r"[^\n.!?。！？]*[.!?。！？]+|\S[^\n.!?。！？]*(?=\n|$)")


def _word_spans(text: str, start: int, end: int) -> List[Span]:
    """Whitespace-delimited word spans (char intervals) within ``[start, end)``."""
    return [(start + m.start(), start + m.end()) for m in _WORD_RE.finditer(text[start:end])]


def _step_spans(text: str, start: int, end: int) -> List[Span]:
    """Reasoning-step spans: split by lines, then by sentence terminators.

    Each returned span is tightened to its non-whitespace content so the pooled
    representation reflects the step itself, not surrounding blank space.
    """
    region = text[start:end]
    steps: List[Span] = []
    line_start = 0
    for line in region.splitlines(keepends=True):
        stripped_len = len(line.rstrip("\n"))
        line_region = line[:stripped_len]
        if line_region.strip():
            # Further split a line into sentences when terminators are present.
            for m in _SENT_SPLIT_RE.finditer(line_region):
                seg = m.group()
                lead = len(seg) - len(seg.lstrip())
                trail = len(seg) - len(seg.rstrip())
                a = start + line_start + m.start() + lead
                b = start + line_start + m.end() - trail
                if b > a:
                    steps.append((a, b))
        line_start += len(line)
    if not steps:  # no content matched -> fall back to one span over the region
        s = text[start:end]
        lead = len(s) - len(s.lstrip())
        trail = len(s) - len(s.rstrip())
        if end - trail > start + lead:
            steps.append((start + lead, end - trail))
    return steps


def _uniform_indices(n: int, k: int) -> List[int]:
    """``k`` indices spread evenly over ``range(n)`` (endpoints included)."""
    if n <= k:
        return list(range(n))
    seen, out = set(), []
    for i in range(k):
        j = round(i * (n - 1) / (k - 1))
        if j not in seen:
            seen.add(j)
            out.append(j)
    return out


def select_response_spans(
    text: str,
    resp_start: int,
    resp_end: int,
    strategy: str = "last_k",
    K: int = 16,
) -> List[Span]:
    """Select up to ``K`` char spans over the response region ``[resp_start, resp_end)``.

    Strategies (IMPLEMENTATION_SPEC §3 / method-doc §5):
        * ``last_k``         -- the last ``K`` word spans (default).
        * ``uniform``        -- ``K`` word spans spread evenly across the region.
        * ``reasoning_step`` -- up to ``K`` reasoning-step spans (the last ``K``).
        * ``all``            -- every word span (``K`` ignored).

    Returns char spans in left-to-right order. The caller is responsible for the
    K_min check (method-doc §4.6: skip the sample's relational term if too few
    valid spans remain).
    """
    if resp_end <= resp_start:
        return []

    if strategy == "reasoning_step":
        steps = _step_spans(text, resp_start, resp_end)
        return steps[-K:] if len(steps) > K else steps

    words = _word_spans(text, resp_start, resp_end)
    if not words:
        return []

    if strategy == "all":
        return words
    if strategy == "last_k":
        return words[-K:] if len(words) > K else words
    if strategy == "uniform":
        return [words[i] for i in _uniform_indices(len(words), K)]
    raise ValueError(
        f"unknown span strategy '{strategy}'; "
        "choose from {'last_k', 'uniform', 'reasoning_step', 'all'}"
    )

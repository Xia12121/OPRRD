"""Acceptance tests for char-span alignment (IMPLEMENTATION_SPEC Task A).

The cross-tokenizer test uses Qwen2.5 + GPT-2 as two genuinely different fast
tokenizers (Llama-3.1 is gated; the property under test -- same char spans pool
both tokenisations -- is identical). Tokenizer tests are skipped offline.
"""

import pytest
import torch

from src.align.span_align import (
    mean_pool_by_charspan,
    select_response_spans,
    _word_spans,
    _step_spans,
)


def _load(name):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name)
    except Exception as e:  # offline / no access
        pytest.skip(f"tokenizer {name} unavailable: {e}")


def test_pool_shape_and_validity_basic():
    # tokens:  "ab"[0,2) "cd"[2,4) "ef"[4,6) ; spans pick subsets by char overlap
    hidden = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=torch.float32
    )
    offsets = [(0, 2), (2, 4), (4, 6)]
    spans = [(0, 4), (4, 6), (10, 12)]  # last span beyond text -> invalid
    H, valid = mean_pool_by_charspan(hidden, offsets, spans)
    assert H.shape == (3, 2)
    assert valid.tolist() == [True, True, False]
    # span [0,4) overlaps tokens 0 and 1 -> mean of rows 0,1
    assert torch.allclose(H[0], torch.tensor([0.5, 0.5]))
    assert torch.allclose(H[1], torch.tensor([2.0, 2.0]))
    assert torch.allclose(H[2], torch.zeros(2))


def test_special_token_zero_offset_excluded():
    hidden = torch.tensor([[5.0], [1.0], [9.0]], dtype=torch.float32)
    offsets = [(0, 0), (0, 3), (0, 0)]  # BOS / EOS style zero-width offsets
    spans = [(0, 3)]
    H, valid = mean_pool_by_charspan(hidden, offsets, spans)
    assert valid.tolist() == [True]
    assert torch.allclose(H[0], torch.tensor([1.0]))  # only the real token


def test_empty_and_inverted_spans_invalid():
    hidden = torch.randn(4, 8)
    offsets = [(0, 2), (2, 4), (4, 6), (6, 8)]
    spans = [(3, 3), (5, 2)]  # empty, inverted
    H, valid = mean_pool_by_charspan(hidden, offsets, spans)
    assert valid.tolist() == [False, False]
    assert torch.allclose(H, torch.zeros(2, 8))


def test_gradient_flows_through_pooling():
    hidden = torch.randn(5, 4, requires_grad=True)
    offsets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    spans = [(0, 3), (3, 5)]
    H, valid = mean_pool_by_charspan(hidden, offsets, spans)
    H.sum().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_cross_tokenizer_same_spans_different_dims():
    """Same char spans pool both a Qwen and a GPT-2 tokenisation (d_s != d_t)."""
    student_tok = _load("Qwen/Qwen2.5-1.5B-Instruct")
    teacher_tok = _load("gpt2")
    text = "Natalia sold 48 clips in April and 24 in May. The total is 72 clips."

    # One shared set of character spans, chosen on the text (not on tokens).
    spans = select_response_spans(text, 0, len(text), strategy="last_k", K=6)
    assert len(spans) == 6

    d_s, d_t = 2048, 4096
    enc_s = student_tok(text, return_offsets_mapping=True, add_special_tokens=True)
    enc_t = teacher_tok(text, return_offsets_mapping=True, add_special_tokens=True)
    hs = torch.randn(len(enc_s["input_ids"]), d_s)
    ht = torch.randn(len(enc_t["input_ids"]), d_t)

    H_S, valid_s = mean_pool_by_charspan(hs, enc_s["offset_mapping"], spans)
    H_T, valid_t = mean_pool_by_charspan(ht, enc_t["offset_mapping"], spans)

    assert H_S.shape == (6, d_s)
    assert H_T.shape == (6, d_t)
    # Every chosen word span should capture >= 1 token in both tokenisations.
    assert valid_s.all() and valid_t.all()


def test_word_spans_cover_words():
    text = "  hello  world\n foo "
    spans = _word_spans(text, 0, len(text))
    words = [text[a:b] for a, b in spans]
    assert words == ["hello", "world", "foo"]


def test_step_spans_split_reasoning():
    text = "Step one is here.\nStep two follows.\nFinal answer: 42"
    spans = _step_spans(text, 0, len(text))
    chunks = [text[a:b] for a, b in spans]
    assert any("Step one" in c for c in chunks)
    assert any("Step two" in c for c in chunks)
    assert any("42" in c for c in chunks)
    # Each step span is trimmed of surrounding whitespace/newlines.
    for a, b in spans:
        assert text[a] != "\n" and text[b - 1] != "\n"


def test_select_strategies():
    text = "a bb ccc dddd eeeee ffffff ggggggg"
    n_words = len(_word_spans(text, 0, len(text)))
    assert n_words == 7

    last3 = select_response_spans(text, 0, len(text), "last_k", 3)
    assert [text[a:b] for a, b in last3] == ["eeeee", "ffffff", "ggggggg"]

    uni3 = select_response_spans(text, 0, len(text), "uniform", 3)
    assert len(uni3) == 3
    assert uni3[0][0] == 0  # includes the first word
    assert uni3[-1][1] == len(text)  # includes the last word

    allspans = select_response_spans(text, 0, len(text), "all", 99)
    assert len(allspans) == 7

    assert select_response_spans(text, 5, 5, "last_k", 3) == []  # empty region


def test_select_response_region_offset():
    """Spans must be absolute char offsets into the full text, not the slice."""
    text = "PROMPT TEXT|response word here"
    start = text.index("response")
    spans = select_response_spans(text, start, len(text), "all", 16)
    assert [text[a:b] for a, b in spans] == ["response", "word", "here"]

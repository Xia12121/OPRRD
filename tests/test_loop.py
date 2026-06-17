"""Acceptance tests for the training loop (IMPLEMENTATION_SPEC Task C).

Uses tiny randomly-initialised models (CPU) with two different tokenizers and
different hidden sizes, so the cross-architecture machinery is fully exercised.
"""

import pytest
import torch

from src.data.sft_data import collate_sft_batch
from src.train.loop import (
    KillCriteria,
    build_relational_texts,
    ce_branch,
    compute_layer_pairs,
    relational_branch,
    train,
)


@pytest.fixture(scope="module")
def toy():
    try:
        from src.models.build import build_toy_student_teacher

        return build_toy_student_teacher(
            d_student=64, d_teacher=96, n_student_layers=4, n_teacher_layers=6
        )
    except Exception as e:  # offline / no tokenizer access
        pytest.skip(f"toy models unavailable: {e}")


EXAMPLES = [
    {"question": "What is 2+2?", "answer": "We add two and two. Step one gives 4. The answer is 4."},
    {"question": "What is 3*3?", "answer": "Three times three. We multiply to get nine. The answer is 9."},
]


def _cfg(lambda_rel=0.05, on_policy=True, max_steps=5):
    return {
        "output_dir": "outputs/_test",
        "relational": {
            "operator": "cka",
            "layer_pairs_rel_depth": [0.5, 0.667],
            "K_spans": 8,
            "K_min": 2,
            "span_strategy": "last_k",
            "lambda_rel": lambda_rel,
            "on_policy": on_policy,
            "rollout_temperature": 0.7,
            "max_new_tokens": 12,
            "max_length": 96,
        },
        "training": {
            "learning_rate": 1e-3,
            "max_steps": max_steps,
            "gradient_accumulation_steps": 1,
            "warmup_ratio": 0.1,
            "scheduler": "cosine",
            "max_grad_norm": 1.0,
            "eval_every_n_steps": 1000,
        },
    }


def test_compute_layer_pairs_middle_layers():
    # student 28 layers, teacher 32 -> 0.5 -> (14,16), 0.667 -> (19,22)
    pairs = compute_layer_pairs(28, 32, [0.5, 0.667])
    assert pairs == [(14, 16), (19, 22)]
    # clamping into [1, L]
    assert compute_layer_pairs(4, 6, [0.0, 1.5]) == [(1, 2), (4, 6)]


def test_relational_branch_detach_and_student_grad(toy):
    """Off-policy (deterministic text): rel branch must give student grad, no teacher grad."""
    z_texts, bounds = build_relational_texts(
        toy.student, toy.student_tok, EXAMPLES, on_policy=False
    )
    pairs = compute_layer_pairs(4, 6, [0.5, 0.667])
    L_rel, stats = relational_branch(
        toy.student,
        toy.teacher,
        toy.student_tok,
        toy.teacher_tok,
        z_texts,
        bounds,
        pairs,
        operator_name="cka",
        K_spans=8,
        K_min=2,
        max_length=96,
    )
    assert torch.isfinite(L_rel)
    assert stats.n_terms > 0, "expected some relational terms on deterministic text"
    L_rel.backward()

    lora_grads = [
        p.grad for n, p in toy.student.named_parameters()
        if p.requires_grad and "lora" in n.lower()
    ]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in lora_grads)
    # The teacher is frozen: no parameter may carry a gradient.
    assert all(p.grad is None for p in toy.teacher.parameters())


def test_train_runs_on_policy_no_nan(toy):
    cfg = _cfg(lambda_rel=0.05, on_policy=True, max_steps=5)
    sft_batch = collate_sft_batch(
        toy.student_tok, [(e["question"], e["answer"]) for e in EXAMPLES], max_length=96
    )
    state = train(
        toy.student, toy.teacher, toy.student_tok, toy.teacher_tok,
        rel_example_stream=lambda s: EXAMPLES,
        sft_batch_stream=lambda s: sft_batch,
        cfg=cfg,
    )
    assert state.stop_reason is None
    assert len(state.history) == 5
    for rec in state.history:
        assert all(k in rec for k in ("ce_loss", "rel_loss", "mean_cka"))
        for k in ("ce_loss", "rel_loss", "total_loss"):
            assert torch.isfinite(torch.tensor(rec[k]))


def test_train_off_policy_runs(toy):
    cfg = _cfg(lambda_rel=0.05, on_policy=False, max_steps=3)
    sft_batch = collate_sft_batch(
        toy.student_tok, [(e["question"], e["answer"]) for e in EXAMPLES], max_length=96
    )
    state = train(
        toy.student, toy.teacher, toy.student_tok, toy.teacher_tok,
        rel_example_stream=lambda s: EXAMPLES,
        sft_batch_stream=lambda s: sft_batch,
        cfg=cfg,
    )
    assert state.stop_reason is None
    assert len(state.history) == 3
    # Off-policy on deterministic gold text -> relational terms must be computed.
    assert any(rec["n_rel_terms"] > 0 for rec in state.history)


def test_lambda_zero_is_pure_sft(toy):
    cfg = _cfg(lambda_rel=0.0, on_policy=True, max_steps=3)
    sft_batch = collate_sft_batch(
        toy.student_tok, [(e["question"], e["answer"]) for e in EXAMPLES], max_length=96
    )
    state = train(
        toy.student, toy.teacher, toy.student_tok, toy.teacher_tok,
        rel_example_stream=lambda s: EXAMPLES,
        sft_batch_stream=lambda s: sft_batch,
        cfg=cfg,
    )
    for rec in state.history:
        assert rec["rel_loss"] == 0.0
        assert rec["n_rel_terms"] == 0
        # total == ce when lambda_rel = 0
        assert abs(rec["total_loss"] - rec["ce_loss"]) < 1e-6


def test_kill_criteria_loss_nan():
    kc = KillCriteria(lambda_rel=0.05)
    assert kc.check(float("nan"), 0.1, 0.5, 1.0) == "loss_nan"
    assert kc.check(999.0, 0.1, 0.5, 1.0) == "loss_nan"
    assert kc.check(2.0, 0.1, 0.5, 1.0) is None


def test_kill_criteria_rel_not_learning():
    kc = KillCriteria(rel_window=5, lambda_rel=0.05)
    out = None
    # Flat CKA and flat rel loss for > window steps -> should trigger.
    for _ in range(8):
        out = kc.check(2.0, 0.5, 0.3, 1.0)
    assert out == "rel_not_learning"


def test_kill_criteria_runtime():
    kc = KillCriteria(max_runtime_s=10.0, lambda_rel=0.0)
    assert kc.check(2.0, 0.0, 0.0, 11.0) == "runtime"


def test_hidden_mse_operator_trains_projector(toy):
    """Claim-1 baseline: projector + student get grads, teacher stays frozen."""
    from src.losses.coordinate_mse import HiddenMSEOperator

    op = HiddenMSEOperator()
    op.build_projector(64, 96, device="cpu", dtype=torch.float32)
    z_texts, bounds = build_relational_texts(
        toy.student, toy.student_tok, EXAMPLES, on_policy=False
    )
    pairs = compute_layer_pairs(4, 6, [0.5, 0.667])
    L_rel, stats = relational_branch(
        toy.student, toy.teacher, toy.student_tok, toy.teacher_tok,
        z_texts, bounds, pairs, K_spans=8, K_min=2, max_length=96, operator=op,
    )
    assert torch.isfinite(L_rel) and stats.n_terms > 0
    L_rel.backward()
    proj_grads = [p.grad for p in op.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in proj_grads)
    assert all(p.grad is None for p in toy.teacher.parameters())


def test_mean_cka_only_for_cka_operator(toy):
    z_texts, bounds = build_relational_texts(
        toy.student, toy.student_tok, EXAMPLES, on_policy=False
    )
    pairs = compute_layer_pairs(4, 6, [0.5, 0.667])
    # genuine CKA -> float mean_cka in [0,1]
    _, s_cka = relational_branch(
        toy.student, toy.teacher, toy.student_tok, toy.teacher_tok,
        z_texts, bounds, pairs, operator_name="cka", K_spans=8, K_min=2, max_length=96,
    )
    assert s_cka.n_terms > 0 and s_cka.mean_cka is not None
    assert -1e-4 <= s_cka.mean_cka <= 1.0 + 1e-4
    # cosine-Frobenius ablation -> mean_cka is None (1 - loss is not a CKA)
    _, s_cf = relational_branch(
        toy.student, toy.teacher, toy.student_tok, toy.teacher_tok,
        z_texts, bounds, pairs, operator_name="cosine_frobenius",
        K_spans=8, K_min=2, max_length=96,
    )
    assert s_cf.n_terms > 0 and s_cf.mean_cka is None


def test_train_with_hidden_mse_baseline(toy):
    from src.losses.coordinate_mse import HiddenMSEOperator

    op = HiddenMSEOperator()
    op.build_projector(64, 96, device="cpu", dtype=torch.float32)
    cfg = _cfg(lambda_rel=0.05, on_policy=False, max_steps=3)
    sft_batch = collate_sft_batch(
        toy.student_tok, [(e["question"], e["answer"]) for e in EXAMPLES], max_length=96
    )
    state = train(
        toy.student, toy.teacher, toy.student_tok, toy.teacher_tok,
        rel_example_stream=lambda s: EXAMPLES,
        sft_batch_stream=lambda s: sft_batch,
        cfg=cfg,
        relational_operator=op,
        extra_params=list(op.parameters()),
    )
    assert state.stop_reason is None and len(state.history) == 3

"""Acceptance tests for the CKA relational loss (IMPLEMENTATION_SPEC Task B)."""

import torch

from src.losses.cka_loss import (
    cka_loss,
    cosine_frobenius_loss,
    get_relational_operator,
)


def _rand(K, d, seed, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(K, d, generator=g, dtype=dtype)


def _orthogonal(d, seed, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d, d, generator=g, dtype=dtype)
    q, r = torch.linalg.qr(a)
    # Fix signs so q is a proper, deterministic orthogonal matrix.
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q


def test_orthogonal_invariance_teacher_side():
    """Right-multiplying H_T by an orthogonal Q must not change the loss."""
    H_S = _rand(16, 2048, 1)
    H_T = _rand(16, 4096, 2)
    Q = _orthogonal(4096, 3)
    base = cka_loss(H_S, H_T)
    rot = cka_loss(H_S, H_T @ Q)
    assert torch.isfinite(base)
    assert (base - rot).abs().item() < 1e-5


def test_isotropic_scale_invariance():
    H_S = _rand(16, 2048, 4)
    H_T = _rand(16, 4096, 5)
    base = cka_loss(H_S, H_T)
    scaled = cka_loss(H_S, H_T * 7.0)
    assert (base - scaled).abs().item() < 1e-5
    # Scaling the student side is equally invariant.
    scaled_s = cka_loss(H_S * 0.13, H_T)
    assert (base - scaled_s).abs().item() < 1e-5


def test_self_consistency():
    """CKA of a matrix with itself is 1, so the loss is ~0."""
    H = _rand(16, 512, 6)
    loss = cka_loss(H, H.clone())
    assert loss.item() < 1e-4


def test_dimension_independence():
    H_S = _rand(20, 2048, 7)
    H_T = _rand(20, 4096, 8)
    loss = cka_loss(H_S, H_T)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert -1e-4 <= loss.item() <= 1.0 + 1e-4


def test_gradient_descent_decreases_loss_and_no_teacher_grad():
    H_S = _rand(16, 64, 9, dtype=torch.float32).requires_grad_(True)
    # Even if the teacher tensor *requests* grad, none must reach it (detach).
    H_T = _rand(16, 128, 10, dtype=torch.float32).requires_grad_(True)
    opt = torch.optim.SGD([H_S], lr=0.5)

    losses = []
    for _ in range(25):
        opt.zero_grad()
        loss = cka_loss(H_S, H_T)
        loss.backward()
        assert H_T.grad is None, "gradient leaked into the (frozen) teacher"
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
    # Monotone non-increasing within a small numerical tolerance.
    for a, b in zip(losses, losses[1:]):
        assert b <= a + 1e-5


def test_collapse_protection_teacher():
    """All-identical teacher rows -> degenerate Gram -> finite, no NaN."""
    H_S = _rand(16, 2048, 11)
    H_T = torch.ones(16, 4096, dtype=torch.float64)
    loss = cka_loss(H_S, H_T)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_collapse_protection_student_keeps_graph():
    H_S = torch.ones(16, 64, dtype=torch.float32, requires_grad=True)
    H_T = _rand(16, 128, 12, dtype=torch.float32)
    loss = cka_loss(H_S, H_T)
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss.backward()  # must not raise: the 0.0 path keeps a graph edge to H_S
    assert H_S.grad is not None


def test_small_K_returns_zero_with_graph():
    H_S = torch.randn(1, 64, requires_grad=True)
    H_T = torch.randn(1, 128)
    loss = cka_loss(H_S, H_T)
    assert loss.item() == 0.0
    loss.backward()
    assert H_S.grad is not None


def test_bf16_inputs_are_stable():
    """Low-precision activations are upcast internally; result stays finite."""
    H_S = _rand(16, 256, 13, dtype=torch.float32).bfloat16().requires_grad_(True)
    H_T = _rand(16, 512, 14, dtype=torch.float32).bfloat16()
    loss = cka_loss(H_S, H_T)
    assert torch.isfinite(loss)
    loss.backward()
    assert H_S.grad is not None and torch.isfinite(H_S.grad).all()


def test_cosine_frobenius_centering_changes_result():
    """The ablation operator must differ from CKA (centering matters)."""
    H_S = _rand(16, 256, 15)
    H_T = _rand(16, 512, 16)
    cka = cka_loss(H_S, H_T).item()
    cosf = cosine_frobenius_loss(H_S, H_T).item()
    assert abs(cka - cosf) > 1e-3
    # cosine-Frobenius is also self-consistent (0 on a matrix with itself).
    assert cosine_frobenius_loss(H_S, H_S.clone()).item() < 1e-8


def test_operator_registry():
    assert get_relational_operator("cka") is cka_loss
    assert get_relational_operator("cosine_frobenius") is cosine_frobenius_loss
    try:
        get_relational_operator("nope")
        assert False, "expected ValueError"
    except ValueError:
        pass

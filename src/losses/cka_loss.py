"""Relational distillation operators (Task B).

Implements the linear-CKA relational loss of OP-RRD §4.1 and, for the ablation
in Task G, the un-centered cosine-Frobenius (Similarity-Preserving) operator.

Why CKA (design rationale, method-doc §3 / §0.1):
  * Dimension-independent (D1): everything is a K x K Gram, so it works when the
    student and teacher have different hidden sizes d_s != d_t.
  * Invariant to per-side orthogonal transforms and isotropic scaling (D2): we do
    *not* want to penalise the two models for expressing the same relational
    geometry in different bases / scales. Linear CKA has exactly this invariance
    (Kornblith et al., 2019); centering is what buys the orthogonal invariance,
    which is precisely what the cosine-Frobenius ablation drops.

The teacher side is always detached: the teacher is frozen and must receive no
gradient (IMPLEMENTATION_SPEC Task B, non-negotiable constraint).
"""

from __future__ import annotations

import torch


def _work_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """Pick a numerically-safe dtype for the Gram/centering math.

    Low-precision activations (bf16/fp16, e.g. an 8-bit teacher or a bf16
    student) make the centered HSIC unstable, so we upcast to float32. We keep
    float64 if any input is already float64 (used by the invariance unit tests,
    which need < 1e-5 agreement).
    """
    if any(t.dtype == torch.float64 for t in tensors):
        return torch.float64
    return torch.float32


def cka_loss(
    H_S: torch.Tensor,
    H_T: torch.Tensor,
    eps: float = 1e-6,
    delta: float = 1e-6,
) -> torch.Tensor:
    """Linear-CKA relational loss ``1 - CKA(K_S, K_T)``.

    Args:
        H_S: ``[K, d_s]`` student span representations (carries gradient).
        H_T: ``[K, d_t]`` teacher span representations (detached internally).
        eps: stabiliser for the row L2-normalisation and the CKA denominator.
        delta: collapse threshold; if ``HSIC(K_S,K_S) < delta`` or
            ``HSIC(K_T,K_T) < delta`` the term contributes 0 (no gradient).

    Returns:
        Scalar in ``[0, 1]`` (numerically, up to ``eps``). Always carries a graph
        edge back to ``H_S`` so that ``backward`` is well-defined even on the
        degenerate / collapse paths.

    Shapes follow method-doc §4.1. ``H_S`` and ``H_T`` share the same K (rows are
    aligned spans) but may differ in feature dimension.
    """
    # Teacher is frozen: cut the graph on the teacher side. (Non-negotiable.)
    H_T = H_T.detach()

    K = H_S.shape[0]
    # K < 2 -> Gram has no off-diagonal structure; skip but keep the graph.
    if K < 2 or H_T.shape[0] < 2:
        return H_S.sum() * 0.0

    work = _work_dtype(H_S, H_T)
    H_S = H_S.to(work)
    H_T = H_T.to(work)

    # Step 1 -- row L2 normalisation (cosine kernel; removes per-span scale).
    H_S = H_S / (H_S.norm(dim=-1, keepdim=True) + eps)
    H_T = H_T / (H_T.norm(dim=-1, keepdim=True) + eps)

    # Step 2 -- Gram matrices (K x K).
    Ks = H_S @ H_S.transpose(-1, -2)
    Kt = H_T @ H_T.transpose(-1, -2)

    # Step 3 -- double centering:  C K C  with  C = I - (1/K) 1 1^T.
    # tr((C Ks C)(C Kt C)) == tr(Ks H Kt H), i.e. the standard HSIC numerator.
    Ks = Ks - Ks.mean(dim=0, keepdim=True) - Ks.mean(dim=1, keepdim=True) + Ks.mean()
    Kt = Kt - Kt.mean(dim=0, keepdim=True) - Kt.mean(dim=1, keepdim=True) + Kt.mean()

    # Step 4 -- HSIC (the common 1/(K-1)^2 factor cancels in the CKA ratio).
    # tr(A B) = sum(A * B) for symmetric A, B.
    hsic_st = (Ks * Kt).sum()
    hsic_ss = (Ks * Ks).sum()
    hsic_tt = (Kt * Kt).sum()

    # Collapse protection (method-doc §4.6): a degenerate Gram -> 0, no gradient.
    if hsic_ss < delta or hsic_tt < delta:
        return H_S.sum() * 0.0

    cka = hsic_st / (torch.sqrt(hsic_ss * hsic_tt) + eps)
    return 1.0 - cka


def cosine_frobenius_loss(
    H_S: torch.Tensor,
    H_T: torch.Tensor,
    eps: float = 1e-6,
    **_unused,
) -> torch.Tensor:
    """Un-centered Similarity-Preserving operator (Tung & Mori, 2019).

    Ablation for Task G: identical pipeline to :func:`cka_loss` *minus* the
    centering step, so it isolates the contribution of CKA's centering.
    ``L = || G_S - G_T ||_F^2 / K^2`` with ``G = Ĥ Ĥ^T`` the row-normalised
    (cosine) Gram. Bounded in ``[0, 4]`` (cosine entries in ``[-1, 1]``).
    """
    H_T = H_T.detach()
    K = H_S.shape[0]
    if K < 2 or H_T.shape[0] < 2:
        return H_S.sum() * 0.0

    work = _work_dtype(H_S, H_T)
    H_S = H_S.to(work)
    H_T = H_T.to(work)

    H_S = H_S / (H_S.norm(dim=-1, keepdim=True) + eps)
    H_T = H_T / (H_T.norm(dim=-1, keepdim=True) + eps)

    Gs = H_S @ H_S.transpose(-1, -2)
    Gt = H_T @ H_T.transpose(-1, -2)
    return ((Gs - Gt) ** 2).sum() / (K * K)


# Registry so the training loop / ablations can switch operator by config name.
RELATIONAL_OPERATORS = {
    "cka": cka_loss,
    "cosine_frobenius": cosine_frobenius_loss,
}


def get_relational_operator(name: str):
    """Return the relational-loss callable for ``name`` (config 'operator')."""
    try:
        return RELATIONAL_OPERATORS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown relational operator '{name}'; "
            f"choose from {sorted(RELATIONAL_OPERATORS)}"
        ) from exc

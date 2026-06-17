"""Coordinate hidden-MSE operator -- the Claim-1 counter-baseline.

This is the *intentionally* coordinate-aligned comparator (IMPLEMENTATION_SPEC
baseline 6, "Direct hidden MSE"). Because d_s != d_t a projection is unavoidable;
we give it the *best case* for coordinate alignment -- a learnable linear map
W: R^{d_s} -> R^{d_t} per (d_s, d_t) pair -- and minimise ``|| W h_S - h_T ||^2``
on the same rollout spans. Note we do NOT row-normalise and do NOT center: the
whole point is to penalise raw coordinate mismatch.

Claim 1 predicts this hurts (<= plain SFT) even with the learned projector,
because matching coordinates imposes gradients unrelated to relational geometry
across architectures (method-doc §4.7).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class HiddenMSEOperator(nn.Module):
    """Stateful relational operator with lazily-built per-shape linear projectors.

    Call signature matches the CKA operator -- ``op(H_S, H_T, eps)`` -> scalar --
    so it is a drop-in for the training loop. Its ``parameters()`` (the
    projectors) must be added to the optimiser by the caller.
    """

    def __init__(self):
        super().__init__()
        self.projectors = nn.ModuleDict()

    def _key(self, d_s: int, d_t: int) -> str:
        return f"{d_s}_{d_t}"

    def _get(self, d_s: int, d_t: int, device, dtype) -> nn.Linear:
        key = self._key(d_s, d_t)
        if key not in self.projectors:
            proj = nn.Linear(d_s, d_t, bias=False)
            proj.to(device=device, dtype=dtype)
            self.projectors[key] = proj
        return self.projectors[key]

    def build_projector(self, d_s: int, d_t: int, device=None, dtype=None) -> nn.Linear:
        """Eagerly create the projector so its params exist before the optimiser.

        Call once per (d_s, d_t) before training; otherwise the lazily-built
        projector would be created after the optimiser and never trained.
        """
        return self._get(d_s, d_t, device, dtype)

    def forward(self, H_S: torch.Tensor, H_T: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        H_T = H_T.detach()
        if H_S.shape[0] == 0:
            return H_S.sum() * 0.0
        d_s, d_t = H_S.shape[-1], H_T.shape[-1]
        proj = self._get(d_s, d_t, H_S.device, H_S.dtype)
        projected = proj(H_S)  # [K, d_t]
        return ((projected - H_T.to(projected.dtype)) ** 2).mean()

    # Allow the same ``op(H_S, H_T, eps)`` calling convention as functional ops.
    __call__ = forward

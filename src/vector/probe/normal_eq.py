"""
Streaming normal-equations accumulator for ridge regression.

Accumulates XᵀX and XᵀY in float64 for numerical stability,
then solves (XᵀX + λI)⁻¹ XᵀY.
"""

from __future__ import annotations

import torch
from torch import Tensor


class NormalEq:
    """
    Streaming (XᵀX, XᵀY) accumulator + ridge solve.

    Usage:
        neq = NormalEq(d_in=1056, d_out=1024)
        for X, Y in data_stream:
            neq.update(X, Y)
        W = neq.solve(lam_ratio=1e-4)   # (d_in, d_out) float32
    """

    def __init__(self, d_in: int, d_out: int) -> None:
        self.A = torch.zeros(d_in, d_in, dtype=torch.float64)
        self.B = torch.zeros(d_in, d_out, dtype=torch.float64)
        self.n = 0

    def update(self, X: Tensor, Y: Tensor) -> None:
        """
        Accumulate a mini-batch.
        X: (N, d_in),  Y: (N, d_out) — any dtype, promoted to float64.
        """
        X64, Y64 = X.double(), Y.double()
        self.A += X64.T @ X64
        self.B += X64.T @ Y64
        self.n += X.shape[0]

    def solve(self, lam_ratio: float = 1e-4) -> Tensor:
        """
        Solve (XᵀX + λI)⁻¹ XᵀY where λ = lam_ratio * n.
        Returns float32 tensor of shape (d_in, d_out).
        """
        lam = lam_ratio * self.n
        A_reg = self.A + lam * torch.eye(self.A.shape[0], dtype=torch.float64)
        return torch.linalg.solve(A_reg, self.B).float()

    @property
    def d_in(self) -> int:
        return self.A.shape[0]

    @property
    def d_out(self) -> int:
        return self.B.shape[1]

"""
Feature expansion for the linear chunk probe.

Feature vector: [o_end ‖ one_hot(offset_j)]
  o_end:      (hidden_dim,) — attention output at chunk end token
  one_hot(j): (chunk_size,) — position within chunk (0=start, chunk_size-1=end)

Combined: (hidden_dim + chunk_size,) per token position in chunk.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _make_one_hot(chunk_size: int) -> Tensor:
    return F.one_hot(
        torch.arange(chunk_size), chunk_size
    ).float()  # (chunk_size, chunk_size)


# Module-level cache keyed by chunk_size (typically only ever chunk_size=32)
_ONE_HOT_CACHE: dict[int, Tensor] = {}


def _one_hot(chunk_size: int) -> Tensor:
    if chunk_size not in _ONE_HOT_CACHE:
        _ONE_HOT_CACHE[chunk_size] = _make_one_hot(chunk_size)
    return _ONE_HOT_CACHE[chunk_size]


def expand(o_end: Tensor, chunk_size: int = 32) -> Tensor:
    """
    Expand a batch of o_end vectors into per-position feature rows.

    Args:
        o_end:      (N, hidden_dim) — one vector per chunk
        chunk_size: number of tokens per chunk

    Returns:
        X: (N * chunk_size, hidden_dim + chunk_size)
    """
    N, hidden_dim = o_end.shape
    one_hot = _one_hot(chunk_size).to(o_end.device)  # (chunk_size, chunk_size)
    o_r = (
        o_end.unsqueeze(1)
        .expand(-1, chunk_size, -1)
        .reshape(N * chunk_size, hidden_dim)
    )
    e_r = one_hot.unsqueeze(0).expand(N, -1, -1).reshape(N * chunk_size, chunk_size)
    return torch.cat([o_r, e_r], dim=-1)  # (N * chunk_size, hidden_dim + chunk_size)


def predict_chunk(W: Tensor, o_end: Tensor, chunk_size: int = 32) -> Tensor:
    """
    Predict K or V for a single chunk.

    Args:
        W:      (hidden_dim + chunk_size, kv_head_dim) weight matrix
        o_end:  (hidden_dim,) attention output at chunk end token
        chunk_size: number of tokens per chunk

    Returns:
        (chunk_size, kv_head_dim) float32
    """
    one_hot = _one_hot(chunk_size).to(o_end.device)  # (chunk_size, chunk_size)
    o_r = o_end.unsqueeze(0).expand(chunk_size, -1)  # (chunk_size, hidden_dim)
    X = torch.cat([o_r, one_hot], dim=-1)  # (chunk_size, hidden_dim + chunk_size)
    return X @ W  # (chunk_size, kv_head_dim)


def expand_k_only(K_chunk: Tensor) -> Tensor:
    """
    Flatten a batch of K_pre chunks into per-token feature rows (no o_end, no position).

    Pure K→V feature: just K_i^{pre-RoPE} per token.

    Args:
        K_chunk: (N, chunk_size, kv_dim) — pre-RoPE K per token per chunk

    Returns:
        X: (N * chunk_size, kv_dim)
    """
    N, chunk_size, kv_dim = K_chunk.shape
    return K_chunk.reshape(N * chunk_size, kv_dim).float()


def expand_v_only(V_chunk: Tensor) -> Tensor:
    """
    Flatten a batch of V chunks into per-token feature rows (no o_end, no position).

    Pure V→K_pre feature: just V_i per token.

    Args:
        V_chunk: (N, chunk_size, kv_dim) — V per token per chunk

    Returns:
        X: (N * chunk_size, kv_dim)
    """
    N, chunk_size, kv_dim = V_chunk.shape
    return V_chunk.reshape(N * chunk_size, kv_dim).float()


def predict_v_from_k(W_VK: Tensor, K_pre_chunk: Tensor) -> Tensor:
    """
    Predict V for a chunk using the pure K→V weight matrix.

    Args:
        W_VK:         (kv_dim, kv_dim) — square weight matrix from K_pre OLS
        K_pre_chunk:  (chunk_size, kv_dim) — pre-RoPE K for each token in chunk

    Returns:
        (chunk_size, kv_dim) float32
    """
    return K_pre_chunk.to(W_VK.dtype) @ W_VK  # (chunk_size, kv_dim)


def expand_with_k(o_end: Tensor, K_chunk: Tensor, chunk_size: int = 32) -> Tensor:
    """
    Expand a batch of (o_end, K_pre) pairs into per-position feature rows.

    Feature per token: [o_end ‖ one_hot(pos) ‖ K_i^{pre-RoPE}]

    Args:
        o_end:    (N, hidden_dim) — one vector per chunk
        K_chunk:  (N, chunk_size, kv_head_dim) — pre-RoPE K per token per chunk
        chunk_size: number of tokens per chunk

    Returns:
        X: (N * chunk_size, hidden_dim + chunk_size + kv_head_dim)
    """
    X_base = expand(o_end, chunk_size)  # (N*chunk_size, hidden_dim + chunk_size)
    N = o_end.shape[0]
    kv_head_dim = K_chunk.shape[-1]
    K_flat = K_chunk.reshape(N * chunk_size, kv_head_dim).to(X_base.dtype)
    return torch.cat([X_base, K_flat], dim=-1)


def predict_chunk_with_k(
    W: Tensor, o_end: Tensor, K_pre: Tensor, chunk_size: int = 32
) -> Tensor:
    """
    Predict V for a single chunk using the K-augmented weight matrix.

    Feature per token: [o_end ‖ one_hot(pos) ‖ K_i^{pre-RoPE}]

    Args:
        W:        (hidden_dim + chunk_size + kv_head_dim, kv_head_dim) augmented weight
        o_end:    (hidden_dim,)
        K_pre:    (chunk_size, kv_head_dim) — pre-RoPE K for each token in chunk
        chunk_size: number of tokens per chunk

    Returns:
        (chunk_size, kv_head_dim) float32
    """
    one_hot = _one_hot(chunk_size).to(o_end.device)  # (chunk_size, chunk_size)
    o_r = o_end.unsqueeze(0).expand(chunk_size, -1)  # (chunk_size, hidden_dim)
    K_f = K_pre.to(o_end.dtype)  # (chunk_size, kv_head_dim)
    X = torch.cat(
        [o_r, one_hot, K_f], dim=-1
    )  # (chunk_size, hidden_dim+chunk_size+kv_head_dim)
    return X @ W  # (chunk_size, kv_head_dim)

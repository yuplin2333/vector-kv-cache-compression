"""
Abstract ModelAdapter: architecture-specific knowledge about where KV and RoPE live.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class ModelAdapter(ABC):
    """Architecture-specific hooks and RoPE application."""

    @abstractmethod
    def get_attn_module(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """Return the self_attn submodule for layer L."""

    @abstractmethod
    def make_capture_hooks(
        self, model: nn.Module, layers: list[int]
    ) -> tuple[dict[str, dict[int, Any]], list[Any]]:
        """
        Register forward hooks to capture:
          captured["attn_out"][L]  : (seq, hidden_dim) float32 CPU — attention output
          captured["k_pre"][L]     : (seq, num_kv_heads, head_dim) float32 — pre-RoPE K
          captured["v"][L]         : (seq, num_kv_heads, head_dim) float32 — post v_proj V

        Returns (captured_dict, handle_list).
        """

    @abstractmethod
    def apply_rope(
        self,
        K_pre: Tensor,
        chunk_start: int,
        model: nn.Module,
        layer_idx: int,
        device: torch.device,
        num_heads: int | None = None,
        positions: Tensor | None = None,
    ) -> Tensor:
        """
        Apply RoPE to pre-RoPE K (or Q) starting at absolute position chunk_start.

        K_pre:     (n_tokens, n_heads * head_dim) float32
        chunk_start: used to build positions when positions=None
        num_heads: number of heads; default = num_kv_heads (backward compat)
        positions: (n_tokens,) absolute positions; default = arange(chunk_start, ...)

        Returns: (n_tokens, n_heads * head_dim) float32 on `device`
        """

    def remove_rope(
        self,
        K_post: Tensor,
        chunk_start: int,
        model: nn.Module,
        layer_idx: int,
        device: torch.device,
        num_heads: int | None = None,
        positions: Tensor | None = None,
    ) -> Tensor:
        """
        Invert RoPE from post-RoPE K (or Q) back to pre-RoPE space.

        K_post:    (n_tokens, n_heads * head_dim) float32
        chunk_start: used to build positions when positions=None
        num_heads: number of heads; default = num_kv_heads (backward compat)
        positions: (n_tokens,) absolute positions; default = arange(chunk_start, ...)

        Returns: (n_tokens, n_heads * head_dim) float32 on `device`
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement remove_rope."
        )

    def k_pre_head_first(self) -> bool:
        """Return True if the captured k_pre tensor has shape (B, kv_heads, seq, head_dim).
        Return False (default) for the standard (B, seq, kv_heads, head_dim) layout."""
        return False

    def get_q_pre_rope_module(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """
        Return the submodule whose output is pre-RoPE Q (post QK-Norm if any).

        Used by baseline scorers (H2O, SnapKV) to capture Q during prefill.
        Raises NotImplementedError by default; override in concrete adapters that
        support baselines.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_q_pre_rope_module. "
            "Required for H2O / SnapKV baselines."
        )

    @abstractmethod
    def chat_kwargs(self) -> dict[str, Any]:
        """Extra kwargs for tokenizer.apply_chat_template()."""

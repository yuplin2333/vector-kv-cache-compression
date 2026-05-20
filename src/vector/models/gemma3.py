"""
Gemma-3 ModelAdapter.

Key Gemma-3-specific details:
- RoPE: model.model.rotary_emb at model level (global attention layers).
  Gemma-3 interleaves global and sliding-window (local) attention layers.
  apply_rope / remove_rope use the global rotary_emb; this is correct for
  calibration (K→V OLS) and global-attention layers in threeway decode.
  Sliding-window layers may use different frequencies — extend if needed.
- k_norm exists: model.model.layers[L].self_attn.k_norm (applied after k_proj,
  before RoPE). Output shape: (batch, seq, num_kv_heads, head_dim).
- Capture pre-RoPE K by hooking k_norm output (4D, same as Qwen3).
- rotate_half convention: same as Qwen3 / Llama3.
- chat_kwargs: {} (Gemma-3 needs no special template kwargs).
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor

from vector.models.base import ModelAdapter


def _rotate_half(x: Tensor) -> Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class Gemma3Adapter(ModelAdapter):
    @staticmethod
    def _get_model_core(model: nn.Module) -> Any:
        core = getattr(model, "model", None)
        assert core is not None
        return core

    @classmethod
    def _get_layers(cls, model: nn.Module) -> list[Any]:
        core = cls._get_model_core(model)
        layers = getattr(core, "layers", None)
        assert layers is not None
        return list(layers)

    @classmethod
    def _get_attn(cls, model: nn.Module, layer_idx: int) -> nn.Module:
        layers = cls._get_layers(model)
        assert 0 <= layer_idx < len(layers)
        attn = getattr(layers[layer_idx], "self_attn", None)
        assert isinstance(attn, nn.Module)
        return attn

    def get_attn_module(self, model: nn.Module, layer_idx: int) -> nn.Module:
        return self._get_attn(model, layer_idx)

    def get_q_pre_rope_module(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """Return q_norm, whose output is pre-RoPE Q (post QK-Norm)."""
        q_norm = getattr(self._get_attn(model, layer_idx), "q_norm", None)
        assert isinstance(q_norm, nn.Module), (
            f"Gemma3 layer {layer_idx} self_attn has no q_norm"
        )
        return q_norm

    def make_capture_hooks(
        self, model: nn.Module, layers: list[int]
    ) -> tuple[dict[str, dict[int, Any]], list[Any]]:
        captured: dict[str, dict[int, Any]] = {
            "attn_out": {L: None for L in layers},
            "k_pre": {L: None for L in layers},
            "v": {L: None for L in layers},
        }
        handles: list[Any] = []

        for L in layers:
            # Hook attention output (post W_O projection)
            def _make_attn_hook(idx: int):
                def _hook(module: nn.Module, inp: Any, out: Any) -> None:
                    # out is tuple; out[0] is (batch, seq, hidden_dim)
                    captured["attn_out"][idx] = out[0].float().detach().cpu()

                return _hook

            handles.append(
                self._get_attn(model, L).register_forward_hook(_make_attn_hook(L))
            )

            # Hook k_norm output → pre-RoPE K
            # k_norm output shape: (batch, num_kv_heads, seq, head_dim)
            # (head-first layout; see k_pre_head_first())
            def _make_knorm_hook(idx: int):
                def _hook(module: nn.Module, inp: Any, out: Any) -> None:
                    captured["k_pre"][idx] = out.float().detach().cpu()

                return _hook

            handles.append(
                getattr(self._get_attn(model, L), "k_norm").register_forward_hook(
                    _make_knorm_hook(L)
                )
            )

            # Hook v_proj output → V
            # v_proj output shape: (batch, seq, num_kv_heads * head_dim)
            def _make_v_hook(idx: int):
                def _hook(module: nn.Module, inp: Any, out: Any) -> None:
                    captured["v"][idx] = out.float().detach().cpu()

                return _hook

            handles.append(
                getattr(self._get_attn(model, L), "v_proj").register_forward_hook(
                    _make_v_hook(L)
                )
            )

        return captured, handles

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
        Apply RoPE using model.model.rotary_emb (global attention RoPE).

        K_pre: (n_tokens, n_heads * head_dim) float32
        Returns: same shape, post-RoPE, on `device`
        """
        n_tokens = K_pre.shape[0]
        model_core = self._get_model_core(model)
        rotary_emb = getattr(model_core, "rotary_emb", None)
        assert callable(rotary_emb)

        config = cast(Any, model).config
        num_kv_heads = getattr(config, "num_key_value_heads")
        assert isinstance(num_kv_heads, int)
        effective_num_heads = num_heads if num_heads is not None else num_kv_heads

        head_dim_cfg = getattr(config, "head_dim", None)
        if head_dim_cfg is None:
            hidden_size = getattr(config, "hidden_size")
            num_attn_heads = getattr(config, "num_attention_heads")
            assert isinstance(hidden_size, int) and isinstance(num_attn_heads, int)
            head_dim: int = hidden_size // num_attn_heads
        else:
            assert isinstance(head_dim_cfg, int)
            head_dim = head_dim_cfg

        K = K_pre.view(n_tokens, effective_num_heads, head_dim).to(device)
        dummy = K.permute(1, 0, 2).unsqueeze(0)  # (1, eff_heads, n_tokens, head_dim)

        if positions is None:
            pos_tensor = torch.arange(
                chunk_start, chunk_start + n_tokens, dtype=torch.long, device=device
            ).unsqueeze(0)  # (1, n_tokens)
        else:
            pos_tensor = positions.to(device).unsqueeze(0)

        rope_out = rotary_emb(dummy, pos_tensor)
        assert isinstance(rope_out, tuple) and len(rope_out) == 2
        cos, sin = rope_out
        assert isinstance(cos, Tensor) and isinstance(sin, Tensor)
        cos = cos.unsqueeze(1)  # broadcast over heads
        sin = sin.unsqueeze(1)

        K_rot = dummy * cos + _rotate_half(dummy) * sin
        result_dim = effective_num_heads * head_dim
        return (
            K_rot.squeeze(0)
            .permute(1, 0, 2)
            .reshape(n_tokens, result_dim)
            .float()
            .to(device)
        )

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
        Invert RoPE: x = x_rot * cos - rotate_half(x_rot) * sin
        """
        n_tokens = K_post.shape[0]
        model_core = self._get_model_core(model)
        rotary_emb = getattr(model_core, "rotary_emb", None)
        assert callable(rotary_emb)

        config = cast(Any, model).config
        num_kv_heads = getattr(config, "num_key_value_heads")
        assert isinstance(num_kv_heads, int)
        effective_num_heads = num_heads if num_heads is not None else num_kv_heads

        head_dim_cfg = getattr(config, "head_dim", None)
        if head_dim_cfg is None:
            hidden_size = getattr(config, "hidden_size")
            num_attn_heads = getattr(config, "num_attention_heads")
            assert isinstance(hidden_size, int) and isinstance(num_attn_heads, int)
            head_dim: int = hidden_size // num_attn_heads
        else:
            assert isinstance(head_dim_cfg, int)
            head_dim = head_dim_cfg

        K = K_post.view(n_tokens, effective_num_heads, head_dim).to(device)
        dummy = K.permute(1, 0, 2).unsqueeze(0)  # (1, eff_heads, n_tokens, head_dim)

        if positions is None:
            pos_tensor = torch.arange(
                chunk_start, chunk_start + n_tokens, dtype=torch.long, device=device
            ).unsqueeze(0)
        else:
            pos_tensor = positions.to(device).unsqueeze(0)

        rope_out = rotary_emb(dummy, pos_tensor)
        assert isinstance(rope_out, tuple) and len(rope_out) == 2
        cos, sin = rope_out
        assert isinstance(cos, Tensor) and isinstance(sin, Tensor)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        K_pre = dummy * cos - _rotate_half(dummy) * sin
        result_dim = effective_num_heads * head_dim
        return (
            K_pre.squeeze(0)
            .permute(1, 0, 2)
            .reshape(n_tokens, result_dim)
            .float()
            .to(device)
        )

    def k_pre_head_first(self) -> bool:
        return True  # k_norm output: (B, kv_heads, seq, head_dim)

    def chat_kwargs(self) -> dict[str, Any]:
        return {}

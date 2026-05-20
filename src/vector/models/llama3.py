"""
Llama-3.1 ModelAdapter.

Key Llama-3.1 differences vs Qwen3:
- No k_norm layer; capture after k_proj directly
- RoPE at model level: model.model.rotary_emb
- Same rotate_half convention as Qwen3
- No special chat_kwargs needed
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


class Llama3Adapter(ModelAdapter):
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
        """Return q_proj, whose output is pre-RoPE Q (Llama3 has no q_norm)."""
        q_proj = getattr(self._get_attn(model, layer_idx), "q_proj", None)
        assert isinstance(q_proj, nn.Module), (
            f"Llama3 layer {layer_idx} self_attn has no q_proj"
        )
        return q_proj

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

            def _make_attn_hook(idx: int):
                def _hook(module: nn.Module, inp: Any, out: Any) -> None:
                    captured["attn_out"][idx] = out[0].float().detach().cpu()

                return _hook

            handles.append(
                self._get_attn(model, L).register_forward_hook(_make_attn_hook(L))
            )

            # Llama3 has no k_norm; hook k_proj directly
            def _make_k_hook(idx: int):
                def _hook(module: nn.Module, inp: Any, out: Any) -> None:
                    # out: (batch, seq, num_kv_heads * head_dim)
                    captured["k_pre"][idx] = out.float().detach().cpu()

                return _hook

            handles.append(
                getattr(self._get_attn(model, L), "k_proj").register_forward_hook(
                    _make_k_hook(L)
                )
            )

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
        K_pre: (n_tokens, n_heads * head_dim) float32
        num_heads: number of heads; default = num_kv_heads (for backward compat)
        positions: (n_tokens,) absolute positions; default = arange(chunk_start, ...)
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
            pos_tensor = positions.to(device).unsqueeze(0)  # (1, n_tokens)

        rope_out = rotary_emb(dummy, pos_tensor)
        assert isinstance(rope_out, tuple) and len(rope_out) == 2
        cos, sin = rope_out
        assert isinstance(cos, Tensor) and isinstance(sin, Tensor)
        cos = cos.unsqueeze(1)
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

    def chat_kwargs(self) -> dict[str, Any]:
        return {}

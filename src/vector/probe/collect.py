"""
Forward hook setup and multi-layer data extraction for calibration.

Registers hooks on self_attn (attn output), k_norm/k_proj (pre-RoPE K),
and v_proj (V) for each requested layer.  Returns per-chunk (o_end, V, K)
triples for accumulation into NormalEq.
"""

from __future__ import annotations


import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from vector.models.base import ModelAdapter


def extract_multilayer(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    adapter: ModelAdapter,
    device: torch.device,
    layers: list[int],
    texts: list[str],
    seq_len: int = 512,
    chunk_size: int = 32,
) -> dict[int, tuple[list[Tensor], list[Tensor], list[Tensor]]]:
    """
    Run one forward pass and extract (o_end, V_chunk, K_chunk) per chunk per layer.

    Returns:
        dict mapping layer_idx → (os_list, Vs_list, Ks_list)
        where each entry in os_list is (hidden_dim,), Vs/Ks are (chunk_size, kv_head_dim).
    """
    captured, handles = adapter.make_capture_hooks(model, layers)

    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
        padding=True,
        return_attention_mask=True,
    )
    ids_raw = enc["input_ids"]
    mask_raw = enc["attention_mask"]
    assert isinstance(ids_raw, Tensor)
    assert isinstance(mask_raw, Tensor)
    ids = ids_raw.to(device)
    mask = mask_raw.to(device)

    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask)

    for h in handles:
        h.remove()

    B = len(texts)
    result: dict[int, tuple[list[Tensor], list[Tensor], list[Tensor]]] = {}

    for L in layers:
        o_L = captured["attn_out"][L]  # (B, seq, hidden_dim) float32 cpu
        k_L = captured["k_pre"][
            L
        ]  # (B, seq, kv_heads, head_dim) or (B, seq, kv_head_dim)
        v_L = captured["v"][L]  # (B, seq, kv_head_dim) or (B, seq, kv_heads, head_dim)

        if o_L is None or k_L is None or v_L is None:
            result[L] = ([], [], [])
            continue

        # Normalize shapes: flatten kv_heads*head_dim if needed.
        # Some adapters (e.g. Gemma3) capture k_pre as (B, kv_heads, seq, head_dim)
        # instead of the standard (B, seq, kv_heads, head_dim). Transpose first.
        if k_L.dim() == 4:
            if adapter.k_pre_head_first():
                k_L = k_L.transpose(
                    1, 2
                )  # (B, kv_heads, seq, head_dim) → (B, seq, kv_heads, head_dim)
            k_L = k_L.reshape(k_L.shape[0], k_L.shape[1], -1)  # (B, seq, kv_head_dim)
        if v_L.dim() == 4:
            v_L = v_L.reshape(v_L.shape[0], v_L.shape[1], -1)

        os_, Vs, Ks = [], [], []
        for b in range(B):
            slen = int(mask[b].sum())
            for s in range(0, slen - chunk_size + 1, chunk_size):
                T = s + chunk_size - 1
                os_.append(o_L[b, T])  # (hidden_dim,)
                Vs.append(v_L[b, s : T + 1])  # (chunk_size, kv_head_dim)
                Ks.append(k_L[b, s : T + 1])  # (chunk_size, kv_head_dim)

        result[L] = (os_, Vs, Ks)

    return result

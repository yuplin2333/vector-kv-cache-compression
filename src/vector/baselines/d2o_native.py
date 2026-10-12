# SPDX-License-Identifier: Apache-2.0
# Adapted from D2O (https://github.com/AIoT-MLSys-Lab/D2O, Apache License 2.0;
# see patches/D2O/LICENSE). Modified by the VECTOR authors: port to transformers 4.57, Qwen3 support.
"""Faithful reimplementation of D2O (Wan et al., ICLR'25) for transformers 4.57.x,
architecture-general (Llama-3.1, Qwen3), via an attention-forward monkeypatch.

Why this exists
---------------
The official D2O repo (AIoT-MLSys-Lab/D2O) ships Llama-only modeling for transformers
4.37.2 and has no Qwen support, while our paper reports every competitor on Llama-3.1,
Qwen3-14B and Qwen3-0.6B. A kvpress prefill-press version of D2O
failed its Llama validation gate by 9-31 points, because D2O's real behavior is fundamentally incompatible with kvpress's cache layout:

  * D2O de-GQAs the cache. On GQA models it stores the ``repeat_kv``'d (query-head) K/V and
    selects/merges **per query-head** (Llama 32 heads, Qwen3-14B 40, Qwen3-0.6B 16). kvpress
    keeps the unrepeated ``num_key_value_heads`` cache, so a press can only select once per
    kv-head -- far coarser, and it collapses at high compression.
  * D2O compresses at the **first decode step** (folding the first generated token's
    attention into the heavy-hitter score), then rolls the budget forward on every later
    decode step. A prefill-time press cannot do either.

So D2O-on-Qwen genuinely requires porting D2O's modeling. This module ports it line-for-line
from ``D2O/.../modeling_llama3_7b_13b_d2o.py`` (attention + ``D2O_LayerKVCache``)
and ``modeling_llama3_new.py`` (global cross-layer allocation), adapted to the 4.57.x
attention interface and made architecture-general (Qwen3 QK-norm handled). It is validated to
reproduce official D2O on Llama-3.1 before any Qwen3 number is trusted.

Faithful to official behavior -- INCLUDING choices that are arguably wasteful. We replicate,
we do NOT "fix":
  * per-query-head storage => D2O does not preserve GQA KV sharing; at nominal p_c it uses
    ~(num_attention_heads / num_key_value_heads)x the token budget of a GQA-respecting method
    (Llama 4x, Qwen3-14B 5x, Qwen3-0.6B 2x). Reported as a footnote, not altered here.
  * compression at the first decode step + rolling maintenance thereafter.
  * global cross-layer budget: weights = softmax(-entropy_per_layer); per-layer total budget
    = clamp(weights * num_layers * rho, 0.01, 1.0), rho = hh_ratio + recent_ratio = 1 - p_c.
  * layers 0, 1 and the last layer are never compressed.
  * heavy-hitter score = column-sum of the FULL prefill attention (O(L^2) -> the same <=16k
    cap as the official D2O Llama row applies).

Design in 4.57.x
----------------
Official D2O (4.37.2) let each attention own a legacy tuple cache. 4.57.x uses ``Cache``
objects and ``past_key_values.update``. We keep the standard ``DynamicCache`` growing
(uncompressed, for correct RoPE positions / ``cache_position`` bookkeeping) and, separately,
maintain D2O's per-query-head compressed cache on each attention module, computing attention
manually (eager) against it. Retained keys carry their original RoPE; the decode query gets
its true absolute position from the standard cache length -- so relative positions are exact.
"""

from __future__ import annotations

import types

import torch
from torch import nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # cos/sin: (B, seq, head_dim); unsqueeze the head dim (matches Llama/Qwen3 unsqueeze_dim=1).
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    b, kvh, s, d = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(b, kvh, n_rep, s, d).reshape(b, kvh * n_rep, s, d)


def _fresh_state() -> dict:
    return {
        "k": None,  # (1, num_heads, cur_len, d) -- per-query-head compressed cache
        "v": None,
        "hh_score": None,  # (num_heads, cur_len) accumulated heavy-hitter score
        "prefill_score": None,  # (num_heads, prefill_len)
        "entropy": torch.tensor(0.0),  # scalar multimodal_entropy
        "prefill_size": 0,
        "layer_hh_ratio": 0.0,
        "layer_recent_ratio": 0.0,
        "hh_size": 0,
        "recent_size": 0,
    }


def _d2o_compress(
    keys: torch.Tensor,
    values: torch.Tensor,
    hh_score: torch.Tensor,
    hh_size: int,
    recent_size: int,
    n_sink: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Port of ``D2O_LayerKVCache.__call__`` select+merge. Operates on a per-query-head cache.

    keys/values: (1, num_heads, seq_len, head_dim); hh_score: (num_heads, seq_len).
    Keeps ``hh_size`` heavy hitters (sink-protected first ``n_sink``) + a ``recent_size``
    window, then cosine-merges each evicted key into its most-similar kept key when the
    head-averaged max similarity >= the mean-similarity threshold. Returns compressed
    (keys, values, hh_score).
    """
    _, num_heads, seq_len, head_dim = keys.shape
    cache_size = hh_size + recent_size
    if seq_len <= cache_size or hh_size <= 0:
        return keys, values, hh_score

    # hh-selection from the non-recent region; force-keep the first n_sink (sink) tokens.
    select_hh_scores = hh_score[:, : seq_len - recent_size].clone()
    select_hh_scores[:, :n_sink] = float("inf")
    sort_indices = torch.argsort(select_hh_scores, dim=-1, descending=True)
    keep_topk = sort_indices[:, :hh_size].sort(dim=-1).values  # (num_heads, hh_size)
    keep_recent = torch.arange(
        seq_len - recent_size, seq_len, device=keys.device
    ).repeat(num_heads, 1)
    keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)  # (num_heads, cache_size)

    mask = torch.zeros(num_heads, seq_len, dtype=torch.bool, device=keys.device)
    mask = mask.scatter(-1, keep_idx, True)

    k = keys.squeeze(0)  # (num_heads, seq_len, d)
    v = values.squeeze(0)
    k_keep = k[mask].view(num_heads, cache_size, head_dim)
    v_keep = v[mask].view(num_heads, cache_size, head_dim)
    hh_keep = hh_score[mask].view(num_heads, cache_size)

    # cosine-merge evicted -> nearest kept above the mean-similarity threshold.
    n_evict = seq_len - cache_size
    if n_evict > 0:
        k_pruned = k[~mask].view(num_heads, n_evict, head_dim)
        v_pruned = v[~mask].view(num_heads, n_evict, head_dim)
        kn_pruned = k_pruned / k_pruned.norm(dim=-1, keepdim=True)
        kn_keep = k_keep / k_keep.norm(dim=-1, keepdim=True)
        sim = kn_pruned @ kn_keep.transpose(-1, -2)  # (num_heads, n_evict, cache_size)
        max_values, max_indices = sim.max(dim=-1)  # (num_heads, n_evict)
        threshold = max_values.mean()
        filt = max_values.mean(0) >= threshold  # (n_evict,) shared across heads
        if filt.any():
            merged_indices = max_indices[:, filt].unsqueeze(-1).expand(-1, -1, head_dim)
            merge_w = max_values[:, filt].unsqueeze(-1)  # (num_heads, m, 1)
            k_src = k_pruned[:, filt, :] * merge_w
            v_src = v_pruned[:, filt, :] * merge_w
            k_keep = k_keep.scatter_reduce(
                1, merged_indices, k_src, reduce="mean", include_self=True
            )
            v_keep = v_keep.scatter_reduce(
                1, merged_indices, v_src, reduce="mean", include_self=True
            )

    return k_keep.unsqueeze(0), v_keep.unsqueeze(0), hh_keep


def _d2o_allocate(model: nn.Module) -> None:
    """Global cross-layer budget (modeling_llama3_new.py:730-747). Sets each layer's
    layer_hh_ratio / layer_recent_ratio from all layers' prefill entropies. Called once,
    lazily, at the first decode step (every layer's prefill entropy is known by then)."""
    attns = [layer.self_attn for layer in model.model.layers]  # type: ignore[union-attr]
    entropies = torch.stack([a._d2o["entropy"].to(torch.float32) for a in attns])
    num_layers = len(attns)
    rho = attns[0]._d2o_uniform_hh + attns[0]._d2o_uniform_recent
    weights = torch.softmax(-entropies, dim=0)
    dynamic = torch.clamp(weights * num_layers * rho, min=0.01, max=1.0)
    for i, a in enumerate(attns):
        hh_split = a._d2o_uniform_hh / rho
        recent_split = a._d2o_uniform_recent / rho
        a._d2o["layer_hh_ratio"] = float(dynamic[i] * hh_split)
        a._d2o["layer_recent_ratio"] = float(dynamic[i] * recent_split)


def _d2o_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,  # absorbs attention_mask / position_ids / use_cache (unused: D2O builds its own)
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    q = self.q_proj(hidden_states).view(hidden_shape)
    k = self.k_proj(hidden_states).view(hidden_shape)
    v = self.v_proj(hidden_states).view(hidden_shape)
    # QK-norm on the head dim before transpose (Qwen3); Llama-3.1 has no q_norm/k_norm.
    if getattr(self, "q_norm", None) is not None:
        q = self.q_norm(q)
        k = self.k_norm(k)
    q = q.transpose(1, 2)  # (B, num_heads, q_len, d)
    k = k.transpose(1, 2)  # (B, num_kv_heads, q_len, d)
    v = v.transpose(1, 2)

    cos, sin = position_embeddings
    q, k = _apply_rope(q, k, cos, sin)

    # Keep the standard cache growing (uncompressed) so RoPE positions / cache_position stay
    # correct across decode steps; D2O attention itself runs on the per-query-head cache below.
    if past_key_values is not None:
        past_key_values.update(
            k,
            v,
            self.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )

    groups = self.num_key_value_groups
    k_rep = _repeat_kv(k, groups)  # (B, num_heads, q_len, d)
    v_rep = _repeat_kv(v, groups)
    st = self._d2o
    q_len = hidden_states.shape[1]
    num_layers = self.config.num_hidden_layers
    compressible = self.layer_idx not in (0, 1, num_layers - 1)

    if q_len > 1:
        # ---- PREFILL: heavy-hitter score + multimodal entropy ----
        st["k"] = k_rep
        st["v"] = v_rep
        st["prefill_size"] = k_rep.shape[2]

        # Output via memory-efficient SDPA -- avoids materializing the full L x L matrix that
        # D2O's eager scoring otherwise needs (that is the 16k O(L^2) memory wall).
        out = nn.functional.scaled_dot_product_attention(
            q, k_rep, v_rep, is_causal=True, scale=self.scaling
        )

        # D2O's heavy-hitter score is the column-sum of the full causal softmax attention
        # (attn summed over queries); the multimodal entropy is the per-head variance of that
        # same column-sum. Both derive ONLY from the column-sum, so we accumulate it in query
        # BLOCKS -> peak O(block x L) instead of O(L^2). Each query's softmax is still over all
        # its (causal) keys, so the column-sum is numerically identical to the full-matrix form.
        num_heads = k_rep.shape[1]
        colsum = torch.zeros(num_heads, q_len, dtype=torch.float32, device=q.device)
        key_pos = torch.arange(q_len, device=q.device).unsqueeze(0)  # (1, L)
        block = 1024
        for qs in range(0, q_len, block):
            qb = q[:, :, qs : qs + block]  # (1, num_heads, b, d)
            a = (
                torch.matmul(qb, k_rep.transpose(2, 3)) * self.scaling
            )  # (1, num_heads, b, L)
            qpos = torch.arange(qs, qs + qb.shape[2], device=q.device).unsqueeze(
                1
            )  # (b, 1)
            a = a.masked_fill(key_pos > qpos, float("-inf"))  # causal for this block
            a = nn.functional.softmax(a, dim=-1, dtype=torch.float32)
            colsum += a.sum(dim=2).squeeze(
                0
            )  # sum over the b queries -> (num_heads, L)

        st["prefill_score"] = colsum
        st["hh_score"] = None
        variance = colsum.var(dim=-1)  # (num_heads,)
        vmin, vmax = variance.min(), variance.max()
        st["entropy"] = ((variance - vmin) / (vmax - vmin + 1e-6)).mean()
    else:
        # ---- DECODE ----
        model = self._d2o_model
        if not getattr(model, "_d2o_allocated", False):
            _d2o_allocate(model)
            model._d2o_allocated = True

        st["k"] = torch.cat([st["k"], k_rep], dim=2)
        st["v"] = torch.cat([st["v"], v_rep], dim=2)
        attn = (
            torch.matmul(q, st["k"].transpose(2, 3)) * self.scaling
        )  # (B, nH, 1, cur_len)
        attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32)
        out = torch.matmul(
            attn.to(q.dtype), st["v"]
        )  # output uses the pre-compression cache

        if compressible and st["layer_hh_ratio"] != 0:
            new_score = attn.detach().sum(0).sum(1)  # (num_heads, cur_len)
            if st["hh_score"] is None:
                # first decode step: fold the first generated token's attention into the score
                ps = st["prefill_size"]
                acc = st["prefill_score"] + new_score[:, :ps]
                acc = torch.cat(
                    [acc, new_score[:, ps:]], dim=1
                )  # (num_heads, prefill+1)
                # budget fixed here (from the prefill+1 length), reused on later steps.
                score_len = acc.shape[-1]
                st["hh_size"] = int(score_len * st["layer_hh_ratio"])
                st["recent_size"] = int(score_len * st["layer_recent_ratio"])
            else:
                acc = new_score.clone()
                acc[:, :-1] = acc[:, :-1] + st["hh_score"]
            st["hh_score"] = acc
            st["k"], st["v"], st["hh_score"] = _d2o_compress(
                st["k"],
                st["v"],
                st["hh_score"],
                st["hh_size"],
                st["recent_size"],
                self.n_sink,
            )

    out = out.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    out = self.o_proj(out)
    return out, None


def reset_d2o(model: nn.Module) -> None:
    """Reset all per-sample D2O state. MUST be called before each ``model.generate`` call."""
    for layer in model.model.layers:  # type: ignore[union-attr]
        layer.self_attn._d2o = _fresh_state()
    model._d2o_allocated = False  # type: ignore[assignment]


def patch_model_for_d2o(
    model: nn.Module, compression_ratio: float, n_sink: int = 4
) -> None:
    """Install D2O onto every attention module of ``model`` (Llama-3.1 or Qwen3).

    ``compression_ratio`` = p_c (fraction evicted); D2O keeps ~(1 - p_c), split evenly into
    heavy-hitter and recent halves: hh_ratio = recent_ratio = (1 - p_c) / 2 (the official
    ``experiments/d2o_longbench.sh`` mapping). Load the model with attn_implementation="eager".
    """
    assert 0.0 <= compression_ratio < 1.0, "compression_ratio (p_c) must be in [0, 1)"
    half = (1.0 - compression_ratio) / 2.0
    for layer in model.model.layers:  # type: ignore[union-attr]
        attn = layer.self_attn
        attn._d2o_model = model
        attn._d2o_uniform_hh = half
        attn._d2o_uniform_recent = half
        attn.n_sink = n_sink
        attn._d2o = _fresh_state()
        attn.forward = types.MethodType(_d2o_attention_forward, attn)
    model._d2o_allocated = False  # type: ignore[assignment]

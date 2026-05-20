from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
from kvpress.presses.base_press import BasePress
from torch import Tensor, nn
from transformers.models.llama.modeling_llama import (
    rotate_half as _rotate_half_transformers,
)


def _safe_r2(true_tensor: Tensor, pred_tensor: Tensor) -> float:
    """R² of pred vs true; returns 1.0 if SST==SSE==0, 0.0 if SST==0 and SSE>0."""
    true_flat = true_tensor.float().reshape(-1)
    pred_flat = pred_tensor.float().reshape(-1)
    sse = ((true_flat - pred_flat) ** 2).sum()
    sst = ((true_flat - true_flat.mean()) ** 2).sum()
    if sst.item() <= 0.0:
        return 1.0 if sse.item() <= 0.0 else 0.0
    r2 = 1.0 - (sse / sst).item()
    assert torch.isfinite(torch.tensor(r2)), f"r2 must be finite, got {r2}"
    return float(r2)


def _export_vector_stats(
    vector_stats: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Aggregate per-layer VECTOR stats into a payload for export.

    Returns None if no stats have been recorded yet.
    The returned dict has two keys:
      "layers"    — the raw per-layer records
      "aggregate" — mean values across all recorded layers
    """
    if not vector_stats:
        return None
    r2_global_vals = [float(x["r2_global"]) for x in vector_stats]
    r2_approx_vals = [
        float(x["r2_approx"]) for x in vector_stats if x["r2_approx"] is not None
    ]
    n_exact_vals = [int(x["n_exact_total"]) for x in vector_stats]
    n_approx_vals = [int(x["n_approx_total"]) for x in vector_stats]
    n_evict_vals = [int(x["n_evict_total"]) for x in vector_stats]
    return {
        "layers": vector_stats,
        "aggregate": {
            "n_layers": len(vector_stats),
            "r2_global_mean": float(sum(r2_global_vals) / len(r2_global_vals)),
            "r2_approx_mean": (
                None
                if not r2_approx_vals
                else float(sum(r2_approx_vals) / len(r2_approx_vals))
            ),
            "n_exact_total_mean": float(sum(n_exact_vals) / len(n_exact_vals)),
            "n_approx_total_mean": float(sum(n_approx_vals) / len(n_approx_vals)),
            "n_evict_total_mean": float(sum(n_evict_vals) / len(n_evict_vals)),
        },
    }


def _load_wkv(weights_path: str) -> dict[int, Tensor]:
    """Load W_KV weights (K_pre→V) from a .pt file; requires meta['v_k_only']=True."""
    d = torch.load(weights_path, weights_only=False, map_location="cpu")
    meta = d.get("meta", {})
    assert meta.get("v_k_only", False), (
        f"weights file {weights_path!r} must have meta['v_k_only']=True; got meta={meta}"
    )
    W_KV: dict[int, Tensor] = {}
    for key, val in d.items():
        m = re.match(r"L(\d+)_linear_V_Wout", key)
        if m:
            layer_idx = int(m.group(1))
            assert val.shape[0] == val.shape[1], (
                f"W_KV layer {layer_idx} must be square, got {val.shape}"
            )
            W_KV[layer_idx] = val
    assert W_KV, f"No W_KV weights found in {weights_path!r}"
    return W_KV


def _load_wvk(weights_path: str) -> dict[int, Tensor]:
    """Load W_VK weights (V→K_pre) from a .pt file; requires meta['k_v_only']=True."""
    d = torch.load(weights_path, weights_only=False, map_location="cpu")
    meta = d.get("meta", {})
    assert meta.get("k_v_only", False), (
        f"weights file {weights_path!r} must have meta['k_v_only']=True; got meta={meta}"
    )
    W_VK: dict[int, Tensor] = {}
    for key, val in d.items():
        m = re.match(r"L(\d+)_linear_K_Wout", key)
        if m:
            layer_idx = int(m.group(1))
            assert val.shape[0] == val.shape[1], (
                f"W_VK layer {layer_idx} must be square, got {val.shape}"
            )
            W_VK[layer_idx] = val
    assert W_VK, f"No W_VK weights found in {weights_path!r}"
    return W_VK


@dataclass
class VECTORPress(BasePress):
    """Threeway + K-only-for-V KV cache compression as a KVPress press.

    Three-way funnel algorithm:
      Step 1 (importance filter): sort tokens by rank(k_cos_post_rope)+rank(v_norm) ascending.
              Top (n_exact + n_approx) are candidates; rest → EVICT.
      Step 2 (difficulty split): among candidates, sort by v_err descending.
              Top n_exact → EXACT (K+V kept); rest → APPROX (K kept, V reconstructed JIT).

    V reconstruction for APPROX tokens: V_pred = K_pre @ W_KV where
      - K_pre = inverse-RoPE(K_post)
      - W_KV is a pre-calibrated 1024×1024 OLS matrix per layer (K_pre→V, R²=0.698)

    Memory formula (K-only-for-V):
      n_exact×2C + n_approx×C = (1-cr-ar)×2C + ar×C  per token

    Parameters
    ----------
    compression_ratio : float
        Fraction of tokens to evict entirely. Controls total VRAM savings.
    approx_ratio : float | None
        Fraction of tokens to approximate (K only). Defaults to compression_ratio/2.
        Constraint: approx_ratio <= 1 - compression_ratio (n_exact >= 0).
    weights_path : str
        Path to .pt file containing W_KV weights (key format: L{L}_linear_V_Wout).
        Must have meta["v_k_only"] == True.
    budget_allocation : str
        One of "global_uniform", "per_head_uniform", "per_head_dynamic".
        - global_uniform:    averaged scores across heads; physical eviction (truncated cache)
        - per_head_uniform:  each head selects independently, same count per head; masked_key_indices
        - per_head_dynamic:  global top-k across all heads; variable count per head; masked_key_indices
    """

    compression_ratio: float = 0.0
    approx_ratio: float | None = None
    weights_path: str = ""
    budget_allocation: str = "per_head_uniform"
    n_sink: int = 4
    n_local: int = 4

    def __post_init__(self) -> None:
        assert self.budget_allocation in (
            "global_uniform",
            "per_head_uniform",
            "per_head_dynamic",
        ), f"Unknown budget_allocation: {self.budget_allocation!r}"
        assert 0 <= self.compression_ratio < 1, "compression_ratio must be in [0, 1)"
        # W_KV is loaded lazily on first use via _ensure_weights().
        # weights_path may be set after construction (e.g. by the evaluation CLI).

        # approx_ratio=None means "derive from compression_ratio at compress() time".
        # Do NOT resolve it here — compression_ratio may be overridden later by the
        # KVPress evaluation CLI (which does press.compression_ratio = value after init).
        self._vector_stats: list[dict[str, Any]] = []
        self._active_approx_ratio: float | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_weights(self) -> None:
        """Load W_KV from weights_path on first use; no-op thereafter."""
        if hasattr(self, "W_KV"):
            return
        assert self.weights_path, (
            f"{self.__class__.__name__}: weights_path must be set before use"
        )
        self.W_KV: dict[int, Tensor] = _load_wkv(self.weights_path)

    def reset_vector_stats(self) -> None:
        self._vector_stats = []

    def export_vector_stats(self) -> dict[str, Any] | None:
        return _export_vector_stats(self._vector_stats)

    @staticmethod
    def _safe_r2(true_tensor: Tensor, pred_tensor: Tensor) -> float:
        return _safe_r2(true_tensor, pred_tensor)

    def _record_layer_stats(
        self,
        layer_idx: int,
        values_true: Tensor,
        values_pred: Tensor,
        eligible_start: int,
        eligible_end: int,
        approx_mask: Tensor,
        n_exact_total: int,
        n_approx_total: int,
        n_evict_total: int,
    ) -> None:
        assert values_true.shape == values_pred.shape, (
            f"values_true/values_pred shape mismatch: {values_true.shape} vs {values_pred.shape}"
        )
        assert values_true.ndim == 4 and values_true.shape[0] == 1, (
            f"expected values tensor shape (1,h,n,d), got {values_true.shape}"
        )
        _, h, n, _ = values_true.shape
        assert approx_mask.shape == (h, n), (
            f"approx_mask shape mismatch: expected {(h, n)}, got {tuple(approx_mask.shape)}"
        )
        true_elig = values_true[0, :, eligible_start:eligible_end, :]
        pred_elig = values_pred[0, :, eligible_start:eligible_end, :]
        r2_global = self._safe_r2(true_elig, pred_elig)

        approx_elig_mask = approx_mask[:, eligible_start:eligible_end]
        if int(approx_elig_mask.sum().item()) > 0:
            approx_elig_mask_expand = approx_elig_mask.unsqueeze(-1).expand_as(
                true_elig
            )
            true_approx = true_elig[approx_elig_mask_expand]
            pred_approx = pred_elig[approx_elig_mask_expand]
            r2_approx = self._safe_r2(true_approx, pred_approx)
        else:
            r2_approx = None

        self._vector_stats.append(
            {
                "layer_idx": int(layer_idx),
                "budget_allocation": self.budget_allocation,
                "compression_ratio": float(self.compression_ratio),
                "approx_ratio": (
                    None
                    if self._active_approx_ratio is None
                    else float(self._active_approx_ratio)
                ),
                "n_heads": int(h),
                "n_eligible_per_head": int(eligible_end - eligible_start),
                "n_exact_total": int(n_exact_total),
                "n_approx_total": int(n_approx_total),
                "n_evict_total": int(n_evict_total),
                "r2_global": float(r2_global),
                "r2_approx": r2_approx,
            }
        )

    def _inverse_rope(self, keys: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """K_pre = K_post * cos - rotate_half(K_post) * sin.

        keys : (b, h, n, d)
        cos  : (1, n, d)  from kwargs["position_embeddings"]
        sin  : (1, n, d)
        """
        cos_ = cos.unsqueeze(1).to(keys.dtype)  # (1, 1, n, d)
        sin_ = sin.unsqueeze(1).to(keys.dtype)
        return keys * cos_ - _rotate_half_transformers(keys) * sin_

    def _compute_scores(
        self,
        keys: Tensor,
        values: Tensor,
        cos: Tensor,
        sin: Tensor,
        layer_idx: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute importance and difficulty scores, plus V_pred for write-back.

        Returns
        -------
        importance : (b, h, n)  — rank(k_cos_post_rope asc) + rank(v_norm desc);
                                   lower = more important; sort ascending to keep
        v_err      : (b, h, n)  — MSE of K_pre→V prediction; high = hard to approximate
        V_pred     : (b, h, n, d) — K_pre @ W_KV, reused for approx token write-back
        """
        b, h, n, d = keys.shape

        # k_cos_post_rope: cosine similarity to per-head mean K (low = distinctive)
        mean_k = keys.mean(dim=2, keepdim=True)  # (b, h, 1, d)
        k_norm = keys.norm(dim=-1).clamp_min(1e-12)  # (b, h, n)
        mean_k_norm = mean_k.norm(dim=-1).clamp_min(1e-12)  # (b, h, 1)
        k_cos = (keys * mean_k).sum(dim=-1) / (k_norm * mean_k_norm)  # (b, h, n)

        # v_norm: L2 norm of value vectors (high = semantically important)
        v_norm = values.norm(dim=-1)  # (b, h, n)

        # importance = rank_sum(k_cos asc, v_norm desc): lower rank = more important
        rank_kcos = torch.argsort(torch.argsort(k_cos, dim=-1), dim=-1).float()
        rank_vnorm = torch.argsort(
            torch.argsort(v_norm, dim=-1, descending=True), dim=-1
        ).float()
        importance = rank_kcos + rank_vnorm  # (b, h, n)

        # V_pred = K_pre @ W_KV
        W_KV = self.W_KV[layer_idx].to(
            device=keys.device, dtype=keys.dtype
        )  # (kv_dim, kv_dim)
        kv_dim = h * d
        K_pre = self._inverse_rope(keys.float(), cos, sin).to(
            keys.dtype
        )  # (b, h, n, d)
        K_pre_flat = K_pre.permute(0, 2, 1, 3).reshape(b, n, kv_dim)  # (b, n, kv_dim)
        V_pred_flat = K_pre_flat @ W_KV  # (b, n, kv_dim)
        V_pred = V_pred_flat.reshape(b, n, h, d).permute(0, 2, 1, 3)  # (b, h, n, d)

        # v_err: per-token per-head MSE
        v_err = (
            ((values.float() - V_pred.float()) ** 2).mean(dim=-1).to(keys.dtype)
        )  # (b, h, n)

        return importance, v_err, V_pred

    @staticmethod
    def _threeway_select(
        importance: Tensor,
        difficulty: Tensor,
        n_exact: int,
        n_approx: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Three-way selection on 1-D score tensors of length n.

        Step 1: sort importance ascending → candidates = top (n_exact+n_approx); rest = evict.
        Step 2: among candidates, sort difficulty descending → top n_exact = exact; rest = approx.

        Returns (exact_idx, approx_idx, evict_idx) — 1-D index tensors into [0, n).
        """
        n_keep = n_exact + n_approx
        order_imp = torch.argsort(
            importance
        )  # ascending; idx 0 = lowest rank_sum = most important
        candidate_idx = order_imp[:n_keep]
        evict_idx = order_imp[n_keep:]

        diff_cands = difficulty[candidate_idx]
        order_diff = torch.argsort(diff_cands, descending=True)
        exact_idx = candidate_idx[order_diff[:n_exact]]
        approx_idx = candidate_idx[order_diff[n_exact:]]

        return exact_idx, approx_idx, evict_idx

    # ------------------------------------------------------------------
    # compress() entry point
    # ------------------------------------------------------------------

    def compress(
        self,
        module: nn.Module,
        hidden_states: Tensor,
        keys: Tensor,
        values: Tensor,
        attentions: Tensor,
        kwargs: dict,
    ) -> tuple[Tensor, Tensor]:
        if self.compression_ratio == 0:
            return keys, values

        self._ensure_weights()

        b, h, n, d = keys.shape
        assert b == 1, f"VECTORPress only supports batch_size=1, got {b}"

        layer_idx = int(module.layer_idx)  # type: ignore[arg-type]
        assert layer_idx in self.W_KV, f"No W_KV weights for layer {layer_idx}"

        if self.budget_allocation != "global_uniform":
            assert module.config._attn_implementation != "eager", (  # type: ignore[attr-defined]
                f"budget_allocation={self.budget_allocation!r} uses masked_key_indices, "
                "which is not supported with attn_implementation='eager'"
            )

        # Resolve approx_ratio lazily so that post-init changes to compression_ratio are respected.
        # Default: min(cr/2, (1-cr)/2)
        #   cr/2      — keeps ar <= cr/2 <= cr, so n_evict = (cr-ar)*n >= 0
        #   (1-cr)/2  — keeps ar <= (1-cr)/2 <= (1-cr), so n_exact = (1-cr-ar)*n >= 0
        approx_ratio = (
            self.approx_ratio
            if self.approx_ratio is not None
            else min(self.compression_ratio / 2, (1 - self.compression_ratio) / 2)
        )
        self._active_approx_ratio = float(approx_ratio)
        cr = self.compression_ratio
        assert approx_ratio <= (1 - cr) + 1e-9, (
            f"approx_ratio {approx_ratio} too large: n_exact would be < 0 "
            f"(must be <= 1 - compression_ratio = {1 - cr:.4f})"
        )
        assert approx_ratio <= cr + 1e-9, (
            f"approx_ratio {approx_ratio} too large: n_evict would be < 0 "
            f"(must be <= compression_ratio = {cr:.4f})"
        )

        # Eligible token range: exclude sink (leading) and local window (trailing).
        # These tokens are always kept exact; only eligible tokens are subject to
        # eviction / approximation.
        eligible_start = self.n_sink
        eligible_end = n - self.n_local
        assert eligible_end > eligible_start, (
            f"Sequence too short for sink/local protection: n={n}, "
            f"n_sink={self.n_sink}, n_local={self.n_local}"
        )
        n_eligible = eligible_end - eligible_start

        cos, sin = kwargs["position_embeddings"]
        importance, v_err, V_pred = self._compute_scores(
            keys, values, cos, sin, layer_idx
        )

        # n_exact / n_approx are counted over eligible tokens only.
        # n_approx = 2A (doubled) to compensate K retention cost; see K-only-for-V formula:
        #   exact(B-A)×2C + approx(2A)×C = 2BC  where B=1-cr, A=approx_ratio
        n_exact = int(n_eligible * (1 - self.compression_ratio - approx_ratio))
        n_approx = int(n_eligible * 2 * approx_ratio)

        if self.budget_allocation == "global_uniform":
            return self._compress_global_uniform(
                layer_idx,
                keys,
                values,
                importance,
                v_err,
                V_pred,
                n_exact,
                n_approx,
                eligible_start,
                eligible_end,
            )
        elif self.budget_allocation == "per_head_uniform":
            return self._compress_per_head_uniform(
                module,
                keys,
                values,
                importance,
                v_err,
                V_pred,
                h,
                n_exact,
                n_approx,
                eligible_start,
                eligible_end,
            )
        else:  # per_head_dynamic
            return self._compress_per_head_dynamic(
                module,
                keys,
                values,
                importance,
                v_err,
                V_pred,
                h,
                n_eligible,
                n_exact,
                n_approx,
                eligible_start,
                eligible_end,
            )

    # ------------------------------------------------------------------
    # Budget allocation paths
    # ------------------------------------------------------------------

    def _compress_global_uniform(
        self,
        layer_idx: int,
        keys: Tensor,
        values: Tensor,
        importance: Tensor,
        v_err: Tensor,
        V_pred: Tensor,
        n_exact: int,
        n_approx: int,
        eligible_start: int,
        eligible_end: int,
    ) -> tuple[Tensor, Tensor]:
        """Physical eviction: all heads share the same token positions (averaged scores).

        Sink tokens [0, eligible_start) and local window [eligible_end, n) are always kept exact.
        """
        n = keys.shape[2]
        h = keys.shape[1]
        imp = importance[0].mean(dim=0)  # (n,)
        diff = v_err[0].mean(dim=0)  # (n,)

        # Select only within eligible range; returned indices are local to that slice.
        exact_local, approx_local, _ = self._threeway_select(
            imp[eligible_start:eligible_end],
            diff[eligible_start:eligible_end],
            n_exact,
            n_approx,
        )
        exact_idx = exact_local + eligible_start
        approx_idx = approx_local + eligible_start

        approx_mask = torch.zeros((h, n), dtype=torch.bool, device=keys.device)
        if approx_idx.numel() > 0:
            approx_mask[:, approx_idx] = True
        n_exact_total = h * int(n_exact)
        n_approx_total = h * int(approx_idx.numel())
        n_evict_total = h * int(
            (eligible_end - eligible_start) - n_exact - int(approx_idx.numel())
        )
        self._record_layer_stats(
            layer_idx=layer_idx,
            values_true=values,
            values_pred=V_pred,
            eligible_start=eligible_start,
            eligible_end=eligible_end,
            approx_mask=approx_mask,
            n_exact_total=n_exact_total,
            n_approx_total=n_approx_total,
            n_evict_total=n_evict_total,
        )

        values = values.clone()
        values[0, :, approx_idx, :] = V_pred[0, :, approx_idx, :]

        sink_idx = torch.arange(eligible_start, device=keys.device)
        local_idx = torch.arange(eligible_end, n, device=keys.device)
        keep_idx = torch.cat([sink_idx, exact_idx, approx_idx, local_idx]).sort().values
        return keys[:, :, keep_idx, :].contiguous(), values[
            :, :, keep_idx, :
        ].contiguous()

    def _compress_per_head_uniform(
        self,
        module: nn.Module,
        keys: Tensor,
        values: Tensor,
        importance: Tensor,
        v_err: Tensor,
        V_pred: Tensor,
        h: int,
        n_exact: int,
        n_approx: int,
        eligible_start: int,
        eligible_end: int,
    ) -> tuple[Tensor, Tensor]:
        """Per-head independent selection with same count per head; masked_key_indices for evict.

        Only tokens in [eligible_start, eligible_end) are eligible for eviction/approximation.
        Sink tokens [0, eligible_start) and local window [eligible_end, n) are never evicted.
        """
        values_true = values
        values = values.clone()
        batch_list, head_list, seq_list = [], [], []
        n = keys.shape[2]
        approx_mask = torch.zeros((h, n), dtype=torch.bool, device=keys.device)

        for head_h in range(h):
            # Score only within eligible range; indices are local to the slice.
            imp_elig = importance[0, head_h, eligible_start:eligible_end]
            diff_elig = v_err[0, head_h, eligible_start:eligible_end]
            exact_local, approx_local, evict_local = self._threeway_select(
                imp_elig, diff_elig, n_exact, n_approx
            )

            approx_idx = approx_local + eligible_start
            evict_idx = evict_local + eligible_start

            values[0, head_h, approx_idx, :] = V_pred[0, head_h, approx_idx, :]
            if approx_idx.numel() > 0:
                approx_mask[head_h, approx_idx] = True

            n_evict = evict_idx.shape[0]
            batch_list.append(evict_idx.new_zeros(n_evict))
            head_list.append(evict_idx.new_full((n_evict,), head_h))
            seq_list.append(evict_idx)

        module.masked_key_indices = (  # type: ignore[attr-defined]
            torch.cat(batch_list),
            torch.cat(head_list),
            torch.cat(seq_list),
        )
        n_exact_total = h * int(n_exact)
        n_approx_total = int(approx_mask.sum().item())
        n_evict_total = int(torch.cat(seq_list).numel())
        self._record_layer_stats(
            layer_idx=int(module.layer_idx),  # type: ignore[attr-defined]
            values_true=values_true,
            values_pred=V_pred,
            eligible_start=eligible_start,
            eligible_end=eligible_end,
            approx_mask=approx_mask,
            n_exact_total=n_exact_total,
            n_approx_total=n_approx_total,
            n_evict_total=n_evict_total,
        )
        return keys, values

    def _compress_per_head_dynamic(
        self,
        module: nn.Module,
        keys: Tensor,
        values: Tensor,
        importance: Tensor,
        v_err: Tensor,
        V_pred: Tensor,
        h: int,
        n_eligible: int,
        n_exact: int,
        n_approx: int,
        eligible_start: int,
        eligible_end: int,
    ) -> tuple[Tensor, Tensor]:
        """Global top-k across all heads; variable budget per head; masked_key_indices for evict.

        Only tokens in [eligible_start, eligible_end) are eligible for eviction/approximation.
        Flat indices are computed over the eligible slice (h × n_eligible), then offset back
        to global sequence positions before being stored in masked_key_indices.
        """
        n_exact_total = h * n_exact
        n_approx_total = h * n_approx
        n_keep_total = n_exact_total + n_approx_total

        # Operate on eligible slice only: shape (h, n_eligible)
        imp_elig = importance[0, :, eligible_start:eligible_end]
        diff_elig = v_err[0, :, eligible_start:eligible_end]

        imp_flat = imp_elig.reshape(-1)  # (h * n_eligible,)
        diff_flat = diff_elig.reshape(-1)

        # Step 1: global importance filter over eligible tokens
        order_imp = torch.argsort(imp_flat)
        candidate_flat = order_imp[:n_keep_total]
        evict_flat = order_imp[n_keep_total:]

        # Step 2: difficulty split within candidates
        diff_cands = diff_flat[candidate_flat]
        order_diff = torch.argsort(diff_cands, descending=True)
        approx_flat_idx = candidate_flat[order_diff[n_exact_total:]]
        evict_flat_idx = evict_flat

        # Convert flat (h * n_eligible) indices → (head, seq) in eligible space,
        # then offset seq to global positions.
        approx_head = approx_flat_idx // n_eligible
        approx_seq = approx_flat_idx % n_eligible + eligible_start
        evict_head = evict_flat_idx // n_eligible
        evict_seq = evict_flat_idx % n_eligible + eligible_start

        values_true = values
        values = values.clone()
        values[0, approx_head, approx_seq, :] = V_pred[0, approx_head, approx_seq, :]
        approx_mask = torch.zeros(
            (h, keys.shape[2]), dtype=torch.bool, device=keys.device
        )
        if approx_seq.numel() > 0:
            approx_mask[approx_head, approx_seq] = True
        n_approx_total_actual = int(approx_seq.numel())
        n_evict_total = int(evict_seq.numel())
        n_exact_total_actual = h * n_eligible - n_approx_total_actual - n_evict_total
        self._record_layer_stats(
            layer_idx=int(module.layer_idx),  # type: ignore[attr-defined]
            values_true=values_true,
            values_pred=V_pred,
            eligible_start=eligible_start,
            eligible_end=eligible_end,
            approx_mask=approx_mask,
            n_exact_total=n_exact_total_actual,
            n_approx_total=n_approx_total_actual,
            n_evict_total=n_evict_total,
        )

        module.masked_key_indices = (  # type: ignore[attr-defined]
            evict_flat_idx.new_zeros(evict_flat_idx.shape[0]),
            evict_head,
            evict_seq,
        )
        return keys, values

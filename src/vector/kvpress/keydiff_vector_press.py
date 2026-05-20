from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from transformers.models.llama.modeling_llama import (
    rotate_half as _rotate_half_transformers,
)

from kvpress.presses.keydiff_press import KeyDiffPress

from vector.kvpress.vector_press import _export_vector_stats, _load_wkv, _safe_r2


@dataclass
class KeyDiffVECTORPress(KeyDiffPress):
    """KeyDiff importance scoring + VECTOR threeway V-approximation.

    Extends KeyDiff (key-cosine-distinctiveness eviction) with VECTOR's K→V
    approximation tier. Tokens are split three ways:
      - Evict  : lowest KeyDiff score (most similar to mean key) → removed
      - Approx : easiest to approximate (low v_err) among kept → V_pred write-back
      - Exact  : rest → unchanged

    Memory formula (K-only-for-V):
      exact(B-A)×2C + approx(2A)×C = 2BC  (B=1-compression_ratio, A=approx_ratio)

    Parameters
    ----------
    weights_path : str
        Path to .pt file with W_KV weights (key format: L{L}_linear_V_Wout).
        Must have meta["v_k_only"] == True.
    approx_ratio : float | None
        Fraction of ALL tokens to approximate (V_pred write-back).
        Default: min(cr/2, (1-cr)/2).
    n_sink : int
        Number of leading tokens (attention sinks) to protect from eviction/approx.
    n_local : int
        Number of trailing tokens (local window) to protect from eviction/approx.
    """

    weights_path: str = ""
    approx_ratio: float | None = None
    n_sink: int = 4
    n_local: int = 4

    def __post_init__(self) -> None:
        assert self.n_sink >= 0, f"n_sink must be >= 0, got {self.n_sink}"
        assert self.n_local > 0, f"n_local must be > 0, got {self.n_local}"
        # W_KV is loaded lazily on first use via _ensure_weights().
        self._vector_stats: list[dict[str, Any]] = []
        super().__post_init__()  # ScorerPress: validates compression_ratio

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

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio == 0:
            return keys, values

        b, h, n, d = keys.shape
        assert b == 1, f"KeyDiffVECTORPress only supports batch_size=1, got {b}"

        # Eligible token range — protect sinks and local window.
        eligible_start = self.n_sink
        eligible_end = n - self.n_local
        assert eligible_end > eligible_start, (
            f"ctx_len={n} too small for n_sink={self.n_sink} + n_local={self.n_local}"
        )
        n_eligible = eligible_end - eligible_start

        # Lazy approx_ratio; constraints: ar < 1-cr (n_exact > 0), ar <= cr (n_evict >= 0).
        cr = self.compression_ratio
        approx_ratio = (
            self.approx_ratio
            if self.approx_ratio is not None
            else min(cr / 2, (1 - cr) / 2)
        )
        assert approx_ratio < (1 - cr) + 1e-9, (
            f"approx_ratio {approx_ratio} too large: n_exact would be <= 0 "
            f"(must be < 1 - compression_ratio = {1 - cr:.4f})"
        )
        assert approx_ratio <= cr + 1e-9, (
            f"approx_ratio {approx_ratio} too large: n_evict would be < 0 "
            f"(must be <= compression_ratio = {cr:.4f})"
        )

        # Importance scores from KeyDiff: high = distinctive = keep.
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        # scores: (b, h, n)
        assert torch.isfinite(scores).all(), "KeyDiff scores must be finite"

        # Inline V_pred computation (inverse-RoPE + K_pre @ W_KV).
        self._ensure_weights()
        layer_idx = int(module.layer_idx)  # type: ignore[attr-defined]
        assert layer_idx in self.W_KV, f"No W_KV weights for layer {layer_idx}"
        W_KV = self.W_KV[layer_idx].to(device=keys.device, dtype=keys.dtype)

        cos, sin = kwargs["position_embeddings"]
        cos_ = cos.unsqueeze(1).to(keys.dtype)  # (1, 1, n, d)
        sin_ = sin.unsqueeze(1).to(keys.dtype)
        K_pre = keys * cos_ - _rotate_half_transformers(keys) * sin_  # inverse-RoPE

        kv_dim = h * d
        K_pre_flat = K_pre.permute(0, 2, 1, 3).reshape(b, n, kv_dim)  # (b, n, kv_dim)
        V_pred = (
            (K_pre_flat @ W_KV).reshape(b, n, h, d).permute(0, 2, 1, 3)
        )  # (b, h, n, d)
        v_err = (
            ((values.float() - V_pred.float()) ** 2).mean(dim=-1).to(keys.dtype)
        )  # (b, h, n)
        assert torch.isfinite(v_err).all(), "v_err must be finite"

        # Budget — all counts based on eligible interval (not full n).
        cr_evict = cr - approx_ratio
        n_evict_per_head = int(n_eligible * cr_evict)
        n_keep_elig = n_eligible - n_evict_per_head
        _approx_frac = 2 * approx_ratio / (1 - cr + approx_ratio)  # 2A/(B+A)
        n_approx = int(n_keep_elig * _approx_frac)

        assert n_keep_elig >= 0, (
            f"n_keep_elig={n_keep_elig} < 0 — compression_ratio or approx_ratio too large"
        )
        assert 0 <= n_approx <= n_keep_elig, (
            f"n_approx={n_approx} out of [0, n_keep_elig={n_keep_elig}]"
        )

        # Per-head threeway split on eligible range.
        values_true = values  # save ref before clone for R² computation
        values = values.clone()
        approx_mask = torch.zeros((h, n), dtype=torch.bool, device=keys.device)
        keep_elig_sorted_per_head: list[torch.Tensor] = []

        for hd in range(h):
            scores_elig_hd = scores[0, hd, eligible_start:eligible_end]
            v_err_elig_hd = v_err[0, hd, eligible_start:eligible_end]

            order_imp = torch.argsort(scores_elig_hd, descending=True)
            cand = order_imp[:n_keep_elig]  # eligible-local indices, high score = keep
            assert len(cand) == n_keep_elig, (
                f"head {hd}: len(cand)={len(cand)} != n_keep_elig={n_keep_elig}"
            )

            if n_approx > 0:
                diff_cands = v_err_elig_hd[cand]
                order_diff = torch.argsort(diff_cands, descending=False)  # easy first
                approx_local = cand[order_diff[:n_approx]]  # eligible-local coords
            else:
                approx_local = torch.zeros(0, dtype=torch.long, device=keys.device)

            # Write V_pred for approx tokens.
            approx_seq = approx_local + eligible_start
            if approx_seq.numel() > 0:
                approx_mask[hd, approx_seq] = True
                values[0, hd, approx_seq, :] = V_pred[0, hd, approx_seq, :].to(
                    values.dtype
                )

            keep_elig_sorted_per_head.append(cand.sort().values)

        # Record per-layer stats (before physical eviction via gather).
        true_elig = values_true[0, :, eligible_start:eligible_end, :]
        pred_elig = V_pred[0, :, eligible_start:eligible_end, :]
        r2_global = _safe_r2(true_elig, pred_elig)
        approx_elig_mask = approx_mask[:, eligible_start:eligible_end]
        if int(approx_elig_mask.sum().item()) > 0:
            expand = approx_elig_mask.unsqueeze(-1).expand_as(true_elig)
            r2_approx: float | None = _safe_r2(true_elig[expand], pred_elig[expand])
        else:
            r2_approx = None
        self._vector_stats.append(
            {
                "layer_idx": layer_idx,
                "compression_ratio": float(self.compression_ratio),
                "approx_ratio": float(approx_ratio),
                "n_heads": int(h),
                "n_eligible_per_head": int(n_eligible),
                "n_exact_total": int(h * (n_keep_elig - n_approx)),
                "n_approx_total": int(approx_mask.sum().item()),
                "n_evict_total": int(h * n_evict_per_head),
                "r2_global": r2_global,
                "r2_approx": r2_approx,
            }
        )

        # Build gather indices: sink + eligible keep + local (same count for all heads).
        n_kept_total = self.n_sink + n_keep_elig + self.n_local
        sink_idx = torch.arange(self.n_sink, device=keys.device)
        local_idx = torch.arange(n - self.n_local, n, device=keys.device)

        indices = torch.zeros(b, h, n_kept_total, dtype=torch.long, device=keys.device)
        for hd in range(h):
            keep_elig_seq = keep_elig_sorted_per_head[hd] + eligible_start
            indices[0, hd] = torch.cat([sink_idx, keep_elig_seq, local_idx])

        idx_exp = indices.unsqueeze(-1).expand(-1, -1, -1, d)
        return keys.gather(2, idx_exp).contiguous(), values.gather(
            2, idx_exp
        ).contiguous()

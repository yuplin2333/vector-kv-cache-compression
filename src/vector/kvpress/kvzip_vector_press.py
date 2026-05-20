from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import (
    rotate_half as _rotate_half_transformers,
)

from kvpress.presses.kvzip_press import KVzipPress
from kvpress.utils import extract_keys_and_values

from vector.kvpress.vector_press import _export_vector_stats, _load_wkv, _safe_r2


@dataclass
class KVzipVECTORPress(KVzipPress):
    """KVzip importance scoring + VECTOR threeway V-approximation.

    Demonstrates that VECTOR's K→V approximation is orthogonal to the choice of
    importance scoring: any eviction baseline can be upgraded by replacing V for
    easy-to-approximate tokens with V_pred = K_pre @ W_KV (offline OLS, R²=0.698).

    Algorithm:
      1. Initial prefill  — compute V_pred + v_err per layer via a forward hook.
      2. Context reconstruction — KVzip scores per-token importance (inherited).
      3. compress_post() — threeway split using KVzip scores + v_err:
           - Evict  : bottom compression_ratio by KVzip score → masked_key_indices
           - Approx : easiest approx_ratio among kept (low v_err) → V_pred write-back
           - Exact  : rest → unchanged

    Inherits all KVzip parameters (compression_ratio, layerwise, n_sink,
    kvzip_plus_normalization) and overhead (2–3× prefill cost for reconstruction).

    Parameters
    ----------
    weights_path : str
        Path to .pt file with W_KV weights (key format: L{L}_linear_V_Wout).
        Must have meta["v_k_only"] == True.
    approx_ratio : float | None
        Fraction of ALL tokens to approximate (V_pred write-back).
        Default: min(cr/2, (1-cr)/2).
        SNR constraint: approx_ratio <= (1 - compression_ratio) / 2.
    """

    weights_path: str = ""
    approx_ratio: float | None = None
    n_local: int = 4

    def __post_init__(self) -> None:
        # W_KV is loaded lazily on first use via _ensure_weights().
        # Runtime stores — initialised in __call__ before each forward pass.
        self._vpred_store: dict[int, Tensor] = {}
        self._verr_store: dict[int, Tensor] = {}
        self._vector_stats: list[dict[str, Any]] = []
        super().__post_init__()  # KVzip: validates compression_ratio, resets state, logs warning

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

    # ------------------------------------------------------------------
    # V_pred computation hook (initial prefill only)
    # ------------------------------------------------------------------

    def _vpred_hook(
        self,
        module: nn.Module,
        input: list,
        kwargs: dict,
        output: list,
    ) -> list:
        """Forward hook that computes V_pred and v_err during the initial prefill.

        Guards:
        - cache_position[0] == 0  →  we are at the start of the initial prefill
        - layer_idx not in _vpred_store  →  don't recompute (handles multi-chunk prefills)
        """
        if kwargs["cache_position"][0] != 0:
            return output

        layer_idx = int(module.layer_idx)  # type: ignore[arg-type]
        if layer_idx in self._vpred_store:
            return output

        self._ensure_weights()
        assert layer_idx in self.W_KV, f"No W_KV weights for layer {layer_idx}"

        keys, values = extract_keys_and_values(kwargs["past_key_values"], layer_idx)
        b, h, n, d = keys.shape
        kv_dim = h * d

        W_KV = self.W_KV[layer_idx].to(device=keys.device, dtype=keys.dtype)

        cos, sin = kwargs["position_embeddings"]
        cos_ = cos.unsqueeze(1).to(keys.dtype)  # (1, 1, n, d)
        sin_ = sin.unsqueeze(1).to(keys.dtype)
        K_pre = keys * cos_ - _rotate_half_transformers(keys) * sin_  # inverse-RoPE

        K_pre_flat = K_pre.permute(0, 2, 1, 3).reshape(b, n, kv_dim)  # (b, n, kv_dim)
        V_pred = (
            (K_pre_flat @ W_KV).reshape(b, n, h, d).permute(0, 2, 1, 3)
        )  # (b, h, n, d)
        v_err = (
            ((values.float() - V_pred.float()) ** 2).mean(dim=-1).to(keys.dtype)
        )  # (b, h, n)

        self._vpred_store[layer_idx] = V_pred
        self._verr_store[layer_idx] = v_err
        return output

    # ------------------------------------------------------------------
    # __call__ — add V_pred hooks around KVzip's context manager
    # ------------------------------------------------------------------

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:  # type: ignore[override]
        self._vpred_store = {}
        self._verr_store = {}

        # Register V_pred computation hooks BEFORE KVzip's context manager so they
        # fire during the initial prefill (which happens inside `with super().__call__`).
        vpred_hooks = [
            layer.self_attn.register_forward_hook(self._vpred_hook, with_kwargs=True)
            for layer in model.model.layers
        ]
        try:
            with super().__call__(model):
                yield
            # After yield: KVzip has already run reconstruction + our compress_post().
        finally:
            for h in vpred_hooks:
                h.remove()

    # ------------------------------------------------------------------
    # compress_post — threeway split + V_pred write-back
    # ------------------------------------------------------------------

    def compress_post(self, model: PreTrainedModel) -> None:  # type: ignore[override]
        """Apply threeway selection on KVzip scores + v_err, then write V_pred."""
        if self.compression_ratio == 0:
            return

        assert self._vpred_store, (
            "_vpred_store is empty — _vpred_hook did not fire. "
            "Ensure the initial prefill starts at cache_position[0]==0."
        )
        assert self.score_val is not None, (
            "score_val is None before compress_post; KVzip reconstruction did not run"
        )
        score_val = self.score_val

        # Lazy approx_ratio: same formula as VECTORPress
        # Constraints: ar < 1-cr (n_exact > 0) and ar <= cr (n_evict >= 0)
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

        n_layer, bsz, n_kv_heads, ctx_len = score_val.shape
        assert bsz == 1, f"KVzipVECTORPress only supports batch_size=1, got {bsz}"

        # Eligible token range (uniform across layers; ctx_len is the same for all).
        eligible_start = self.n_sink
        eligible_end = ctx_len - self.n_local
        assert eligible_end > eligible_start, (
            f"ctx_len={ctx_len} too small for n_sink={self.n_sink} + n_local={self.n_local}"
        )
        n_eligible_per_head = eligible_end - eligible_start
        n_eligible_total_per_layer = n_kv_heads * n_eligible_per_head

        # Per-layer eviction counts.
        # KVzip must evict only (cr - approx_ratio) of tokens, not cr.
        # Reason: the full budget formula is:
        #   exact(B-A)×2C + approx(2A)×C = 2BC  (B=1-cr, A=approx_ratio)
        # So n_survivors = n_exact + n_approx = (B+A)×n = (1-cr+ar)×n.
        # KVzip's job is to evict the bottom (cr-ar) fraction; VECTOR then splits
        # survivors into exact/approx.
        #
        # Eviction is performed directly on the eligible slice — sink and local tokens
        # are never candidates regardless of their scores.
        cr_evict = cr - approx_ratio  # fraction truly evicted (K and V both gone)
        assert cr_evict >= 0, (
            f"cr_evict = cr - approx_ratio = {cr:.4f} - {approx_ratio:.4f} < 0; "
            "this should have been caught by the approx_ratio <= cr assertion above"
        )
        score_val_elig = score_val[
            :, :, :, eligible_start:eligible_end
        ]  # (L, 1, h, n_elig)
        if self.layerwise:
            n_evict_scalar = int(n_eligible_total_per_layer * cr_evict)
            n_evict_layers = n_evict_scalar * torch.ones(
                n_layer, device=score_val.device, dtype=torch.int
            )
        else:
            n_evict_total = int(score_val_elig.numel() * cr_evict)
            pruned_flat = torch.topk(-score_val_elig.reshape(-1), n_evict_total).indices
            n_tokens_per_layer_elig = bsz * n_kv_heads * n_eligible_per_head
            n_evict_layers = torch.bincount(
                pruned_flat // n_tokens_per_layer_elig, minlength=n_layer
            ).int()

        # n_approx (doubled) is computed per-layer from that layer's n_keep.
        # Ratio: n_approx / n_keep = 2A / (B+A) where B=1-cr, A=approx_ratio.
        # In layerwise mode n_keep is uniform so this equals int(tokens * 2*A).
        # In non-layerwise mode n_keep varies per layer (KVzip global distribution),
        # so we must compute n_approx per-layer to avoid n_approx > n_keep.
        _approx_frac = 2 * approx_ratio / (1 - cr + approx_ratio)  # 2A/(B+A)

        for layer in model.model.layers:
            module = layer.self_attn
            layer_idx = int(module.layer_idx)  # type: ignore[arg-type]

            assert module.config._attn_implementation != "eager", (  # type: ignore[attr-defined]
                "KVzipVECTORPress requires attn_implementation != 'eager' "
                "(uses masked_key_indices)"
            )

            scores = score_val[layer_idx, 0]  # (n_kv_heads, ctx_len), HIGH=important
            v_err = self._verr_store[layer_idx][0]  # (n_kv_heads, ctx_len)
            V_pred = self._vpred_store[layer_idx]  # (1, n_kv_heads, ctx_len, head_dim)

            n_evict = int(n_evict_layers[layer_idx].item())
            # Sanity guard: n_evict_layers is derived from n_eligible_total_per_layer,
            # so this should never fire unless there is a bug in the computation above.
            assert n_evict <= n_eligible_total_per_layer, (
                f"layer {layer_idx}: n_evict={n_evict} > n_eligible_total_per_layer="
                f"{n_eligible_total_per_layer} (bug in n_evict_layers computation)"
            )
            # n_keep_elig: eligible survivors; sink+local are always kept on top of this.
            n_keep_elig = n_eligible_total_per_layer - n_evict
            # Per-layer approx count: n_approx / n_keep_elig = 2A/(B+A).
            n_approx_layer = int(n_keep_elig * _approx_frac)
            assert n_approx_layer <= n_keep_elig, (
                f"layer {layer_idx}: n_approx_layer={n_approx_layer} > n_keep_elig={n_keep_elig}. "
                f"_approx_frac={_approx_frac:.6f} cr={cr:.4f} approx_ratio={approx_ratio:.4f}"
            )

            # Step 1: importance filter — operate only on eligible slice.
            # Build a flat view over the eligible positions: (n_kv_heads, n_eligible_per_head).
            scores_elig = scores[:, eligible_start:eligible_end].reshape(-1)
            order_imp = torch.argsort(scores_elig, descending=True)
            candidate_flat_elig = order_imp[:n_keep_elig]
            evict_flat_elig = order_imp[n_keep_elig:]

            # Step 2: difficulty split — easiest n_approx_layer among candidates
            # (low v_err = easy to approximate with K_pre @ W_KV).
            if n_approx_layer > 0:
                v_err_elig = v_err[:, eligible_start:eligible_end].reshape(-1)
                diff_cands = v_err_elig[candidate_flat_elig]
                order_diff = torch.argsort(
                    diff_cands, descending=False
                )  # ascending: easy first
                approx_flat_elig = candidate_flat_elig[order_diff[:n_approx_layer]]
            else:
                approx_flat_elig = torch.zeros(
                    0, dtype=torch.long, device=scores_elig.device
                )

            # Convert flat (n_kv_heads * n_eligible_per_head) indices → global (head, seq).
            approx_head = approx_flat_elig // n_eligible_per_head
            approx_seq = approx_flat_elig % n_eligible_per_head + eligible_start
            evict_head = evict_flat_elig // n_eligible_per_head
            evict_seq = evict_flat_elig % n_eligible_per_head + eligible_start

            # Record per-layer stats before write-back (so values_true is unmodified).
            cache_layer = self._cache.layers[layer_idx]  # type: ignore[attr-defined]
            true_elig = cache_layer.values[0, :, eligible_start:eligible_end, :]
            pred_elig = V_pred[0, :, eligible_start:eligible_end, :]
            r2_global = _safe_r2(true_elig, pred_elig)
            if approx_flat_elig.numel() > 0:
                approx_mask = torch.zeros(
                    (n_kv_heads, ctx_len), dtype=torch.bool, device=V_pred.device
                )
                approx_mask[approx_head, approx_seq] = True
                approx_elig_mask = approx_mask[:, eligible_start:eligible_end]
                expand = approx_elig_mask.unsqueeze(-1).expand_as(true_elig)
                r2_approx: float | None = _safe_r2(true_elig[expand], pred_elig[expand])
            else:
                r2_approx = None
            self._vector_stats.append(
                {
                    "layer_idx": layer_idx,
                    "compression_ratio": float(self.compression_ratio),
                    "approx_ratio": float(approx_ratio),
                    "n_heads": int(n_kv_heads),
                    "n_eligible_per_head": int(n_eligible_per_head),
                    "n_exact_total": int(n_keep_elig - n_approx_layer),
                    "n_approx_total": int(approx_flat_elig.numel()),
                    "n_evict_total": int(evict_flat_elig.numel()),
                    "r2_global": r2_global,
                    "r2_approx": r2_approx,
                }
            )

            # Write V_pred for approx tokens in-place on the live cache.
            if approx_flat_elig.numel() > 0:
                cache_layer.values[0, approx_head, approx_seq, :] = V_pred[
                    0, approx_head, approx_seq, :
                ].to(cache_layer.values.dtype)

            # Set masked_key_indices for evict tokens (KVzip fake-key masking).
            module.masked_key_indices = (  # type: ignore[attr-defined]
                evict_flat_elig.new_zeros(evict_flat_elig.shape[0]),
                evict_head,
                evict_seq,
            )

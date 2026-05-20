from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers.models.llama.modeling_llama import (
    rotate_half as _rotate_half_transformers,
)

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.snapkv_press import SnapKVPress

from vector.kvpress.vector_press import _load_wkv


@dataclass
class VECTORWrapperPress(BasePress):
    """Generic wrapper that adds VECTOR's K→V approximation tier to any ScorerPress.

    Wraps any KVPress ScorerPress (KeyDiff, SnapKV, PyramidKV, Knorm, …) and
    applies VECTOR's threeway split on top of its importance scores:

      - Evict  : lowest-scoring tokens → removed entirely
      - Approx : easiest to approximate (low v_err) among kept → V_pred write-back
      - Exact  : rest → unchanged K+V

    Two budget modes depending on whether the wrapped press has a per-layer budget:

    Standard ScorerPress (e.g. KeyDiff, SnapKV):
      Uses the same "B+A" budget as KeyDiffVECTORPress / SnapKVVECTORPress.
      Evicts only (cr - ar) fraction, keeps (1 - cr + ar) fraction total.
      Memory: exact(B-A)×2C + approx(2A)×C = 2BC  (B=1-cr, A=approx_ratio)

    Per-layer budget press (e.g. PyramidKV via get_layer_budget()):
      Respects the press's own per-layer n_kept.  approx_ratio is re-derived
      per layer from cr_eff = 1 - n_kept/n_eligible so the constraints
      ar < 1-cr_eff and ar ≤ cr_eff always hold.

    Window-size alignment for SnapKV subclasses (including PyramidKV):
      SnapKVPress.score() pads the last window_size tokens with max score.
      __post_init__ automatically sets press.window_size = n_local so that
      padding covers exactly the excluded [n-n_local, n) region.
      For PyramidKVPress this also aligns get_layer_budget()'s window_size
      with VECTOR's protected local region.

    Parameters
    ----------
    press : ScorerPress
        Any ScorerPress whose score() method provides per-token importance.
    weights_path : str
        Path to .pt file with W_KV weights (key format: L{L}_linear_V_Wout).
        Must have meta["v_k_only"] == True.
    approx_ratio : float | None
        Fraction of ALL tokens to approximate (V_pred write-back).
        For standard presses: default min(cr/2, (1-cr)/2).
        For per-layer budget presses: acts as an upper cap per layer.
    n_sink : int
        Number of leading tokens (attention sinks) to protect from eviction/approx.
    n_local : int
        Number of trailing tokens (local window) to protect from eviction/approx.
        Also controls window_size for SnapKV-family presses (see above).
    """

    press: ScorerPress = None  # type: ignore[assignment,misc]
    weights_path: str = ""
    approx_ratio: float | None = None
    n_sink: int = 4
    n_local: int = 4

    def __post_init__(self) -> None:
        assert isinstance(self.press, ScorerPress), (
            f"VECTORWrapperPress requires a ScorerPress, got {type(self.press).__name__}"
        )
        assert self.n_sink >= 0, f"n_sink must be >= 0, got {self.n_sink}"
        assert self.n_local > 0, f"n_local must be > 0, got {self.n_local}"
        # For SnapKV-family presses (including PyramidKV): align window_size with
        # n_local so that score padding covers exactly the excluded trailing region.
        if isinstance(self.press, SnapKVPress):
            self.press.window_size = self.n_local
        # W_KV is loaded lazily on first use via _ensure_weights().

    def _ensure_weights(self) -> None:
        """Load W_KV from weights_path on first use; no-op thereafter."""
        if hasattr(self, "W_KV"):
            return
        assert self.weights_path, (
            f"{self.__class__.__name__}: weights_path must be set before use"
        )
        self.W_KV: dict[int, Tensor] = _load_wkv(self.weights_path)

    @property
    def compression_ratio(self) -> float:
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value: float) -> None:
        self.press.compression_ratio = value

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
        assert b == 1, f"VECTORWrapperPress only supports batch_size=1, got {b}"

        # Eligible token range — protect sinks and local window.
        eligible_start = self.n_sink
        eligible_end = n - self.n_local
        assert eligible_end > eligible_start, (
            f"ctx_len={n} too small for n_sink={self.n_sink} + n_local={self.n_local}"
        )
        n_eligible = eligible_end - eligible_start

        # Importance scores from the wrapped press: high score = keep.
        scores = self.press.score(
            module, hidden_states, keys, values, attentions, kwargs
        )
        # scores: (b, h, n)
        assert torch.isfinite(scores).all(), (
            f"{type(self.press).__name__} scores must be finite"
        )

        # Budget and approx_ratio — two modes:
        #
        # (A) Standard ScorerPress: mirrors KeyDiffVECTORPress / SnapKVVECTORPress.
        #     Evict only (cr - ar) fraction; keep (1-cr+ar) = B+A fraction total.
        #     Memory: exact(B-A)×2C + approx(2A)×C = 2BC  (B=1-cr, A=approx_ratio)
        #
        # (B) Per-layer budget press (e.g. PyramidKV via get_layer_budget):
        #     Respect the press's own n_kept; derive ar per layer from cr_eff.
        if hasattr(self.press, "get_layer_budget"):
            # (B) Per-layer path.
            n_keep_elig = self.press.get_layer_budget(module, n_eligible)  # type: ignore[attr-defined]
            assert 0 <= n_keep_elig <= n_eligible, (
                f"get_layer_budget returned {n_keep_elig} outside [0, {n_eligible}]"
            )
            cr_eff = 1.0 - n_keep_elig / n_eligible if n_eligible > 0 else 0.0
            approx_ratio = (
                self.approx_ratio
                if self.approx_ratio is not None
                else min(cr_eff / 2, (1 - cr_eff) / 2)
            )
        else:
            # (A) Standard path.
            cr = self.press.compression_ratio
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
            cr_eff = cr
            cr_evict = cr - approx_ratio
            n_evict = int(n_eligible * cr_evict)
            n_keep_elig = n_eligible - n_evict

        assert 0 <= n_keep_elig <= n_eligible, (
            f"n_keep_elig={n_keep_elig} out of [0, {n_eligible}]"
        )

        # n_approx: fraction of kept tokens to approximate.
        # Derivation (both modes): n_approx / n_keep = 2A/(B+A).
        denom = 1 - cr_eff + approx_ratio
        _approx_frac = (2 * approx_ratio / denom) if denom > 0 else 0.0
        n_approx = int(n_keep_elig * _approx_frac)
        assert 0 <= n_approx <= n_keep_elig, (
            f"n_approx={n_approx} out of [0, n_keep_elig={n_keep_elig}]"
        )

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

        # Per-head threeway split on eligible range.
        values = values.clone()
        keep_elig_sorted_per_head: list[torch.Tensor] = []

        for hd in range(h):
            scores_elig_hd = scores[0, hd, eligible_start:eligible_end]
            v_err_elig_hd = v_err[0, hd, eligible_start:eligible_end]

            order_imp = torch.argsort(scores_elig_hd, descending=True)
            cand = order_imp[:n_keep_elig]  # eligible-local indices, high score = keep

            if n_approx > 0:
                diff_cands = v_err_elig_hd[cand]
                order_diff = torch.argsort(diff_cands, descending=False)  # easy first
                approx_local = cand[order_diff[:n_approx]]  # eligible-local coords
            else:
                approx_local = torch.zeros(0, dtype=torch.long, device=keys.device)

            # Write V_pred for approx tokens.
            approx_seq = approx_local + eligible_start
            if approx_seq.numel() > 0:
                values[0, hd, approx_seq, :] = V_pred[0, hd, approx_seq, :].to(
                    values.dtype
                )

            keep_elig_sorted_per_head.append(cand.sort().values)

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

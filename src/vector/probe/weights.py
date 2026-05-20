"""
Save and load probe weight files (.pt).

File format is backwards-compatible with analysis/closed_form_all_layers.pt
produced by the original exp_closed_form_probe.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import Tensor


def save_weights(
    W_V: dict[int, Tensor],
    W_K: dict[int, Tensor],
    path: Path,
    meta: dict[str, Any] | None = None,
) -> None:
    """
    Save probe weights to a .pt file.

    Keys follow the convention: L{layer}_linear_{V|K}_Wout
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {}
    for L, W in W_V.items():
        payload[f"L{L}_linear_V_Wout"] = W
    for L, W in W_K.items():
        payload[f"L{L}_linear_K_Wout"] = W
    if meta is not None:
        payload["meta"] = meta

    torch.save(payload, path)
    logger.info(f"Saved probe weights → {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def load_weights(
    path: Path,
    layers: list[int] | None = None,
) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
    """
    Load probe weights from a .pt file.

    Args:
        path:   path to .pt file
        layers: subset of layers to load (None = all available)

    Returns:
        (W_V, W_K) — each dict {layer_idx: Tensor(input_dim, kv_head_dim)}
    """
    path = Path(path)
    d = torch.load(path, weights_only=False, map_location="cpu")
    assert isinstance(d, dict), f"weights file must contain a dict, got {type(d)}"

    def _load(prefix: str) -> dict[int, Tensor]:
        result = {}
        for key, val in d.items():
            if not key.endswith(f"{prefix}_Wout"):
                continue
            # Key format: L{layer}_linear_{V|K}_Wout
            parts = key.split("_")
            assert len(parts) == 4, f"invalid weight key format: {key}"
            assert parts[0].startswith("L"), f"invalid weight key prefix: {key}"
            assert parts[1] == "linear", f"invalid weight key format: {key}"
            assert parts[2] == prefix, f"unexpected weight key prefix: {key}"
            layer_idx = int(parts[0][1:])  # strip leading 'L'
            if layers is None or layer_idx in layers:
                assert isinstance(val, Tensor), (
                    f"weight value for key={key} must be Tensor, got {type(val)}"
                )
                result[layer_idx] = val.float()
        return result

    W_V = _load("V")
    W_K = _load("K")
    logger.info(
        f"Loaded V weights for {len(W_V)} layers, K weights for {len(W_K)} layers."
    )
    return W_V, W_K

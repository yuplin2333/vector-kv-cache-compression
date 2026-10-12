"""
Offline calibration loop: stream C4, accumulate NormalEq, solve per layer.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from vector.config import ModelConfig, ProbeConfig
from vector.data.calibration import CALIB_STREAMS
from vector.models.base import ModelAdapter
from vector.probe.collect import extract_multilayer
from vector.probe.features import expand_k_only, expand_v_only
from vector.probe.normal_eq import NormalEq


# ---------------------------------------------------------------------------
# R² helper
# ---------------------------------------------------------------------------


def _r2_k_only(W_VK: Tensor, val_K: Tensor, val_Y: Tensor) -> float:
    """Quick R² for pure K→V model: V_pred = K_pre @ W_VK."""
    with torch.no_grad():
        X = expand_k_only(val_K.float())  # (n*chunk_size, kv_dim)
        pred = (X @ W_VK).reshape(-1, val_K.shape[1], W_VK.shape[1])
        true = val_Y.float()
        ss_res = ((true - pred) ** 2).sum().item()
        ss_tot = ((true - true.mean()) ** 2).sum().item()
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _r2_v_only(W_VK: Tensor, val_V: Tensor, val_Y: Tensor) -> float:
    """Quick R² for pure V→K_pre model: K_pred = V @ W_VK."""
    with torch.no_grad():
        X = expand_v_only(val_V.float())  # (n*chunk_size, kv_dim)
        pred = (X @ W_VK).reshape(-1, val_V.shape[1], W_VK.shape[1])
        true = val_Y.float()
        ss_res = ((true - pred) ** 2).sum().item()
        ss_tot = ((true - true.mean()) ** 2).sum().item()
    return 1.0 - ss_res / (ss_tot + 1e-12)


# ---------------------------------------------------------------------------


def calibrate(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    adapter: ModelAdapter,
    model_cfg: ModelConfig,
    probe_cfg: ProbeConfig,
    device: torch.device,
    k_only_for_v: bool = False,
    v_only_for_k: bool = False,
) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
    """
    Run offline calibration on C4 data.

    k_only_for_v: pure K_pre→V OLS; W_V is (kv_dim, kv_dim); W_K returned empty.
    v_only_for_k: pure V→K_pre OLS; W_K is (kv_dim, kv_dim); W_V returned empty.

    Returns:
        W_V: dict {layer_idx: Tensor(kv_dim, kv_dim)}  — non-empty iff k_only_for_v
        W_K: dict {layer_idx: Tensor(kv_dim, kv_dim)}  — non-empty iff v_only_for_k
    """
    assert k_only_for_v or v_only_for_k, (
        "Exactly one of k_only_for_v or v_only_for_k must be True."
    )
    assert not (k_only_for_v and v_only_for_k), (
        "k_only_for_v and v_only_for_k are mutually exclusive."
    )

    layers = probe_cfg.target_layers
    if not layers:
        layers = list(range(model_cfg.num_layers))

    kv_dim = model_cfg.kv_head_dim
    chunk_size = model_cfg.chunk_size
    lam = probe_cfg.lambda_ratio

    # Validation set
    logger.info(f"Collecting val ({probe_cfg.val_seqs} seqs, {len(layers)} layers) ...")
    val_V: dict[int, list[Tensor]] = {L: [] for L in layers}
    val_K: dict[int, list[Tensor]] = {L: [] for L in layers}

    with tqdm(total=probe_cfg.val_seqs, desc="Calib val", unit="seq") as pbar:
        for batch in CALIB_STREAMS[probe_cfg.calib_data](
            "validation", probe_cfg.batch_size, probe_cfg.val_seqs
        ):
            res = extract_multilayer(
                model,
                tokenizer,
                adapter,
                device,
                layers,
                batch,
                probe_cfg.seq_len,
                chunk_size,
            )
            for L in layers:
                _, vs, ks = res[L]
                val_V[L].extend(vs)
                val_K[L].extend(ks)

            inc = min(len(batch), max(probe_cfg.val_seqs - pbar.n, 0))
            pbar.update(inc)
            pbar.set_postfix_str(f"{len(val_K[layers[0]])} chunks/layer")

    MAX_VAL_CHUNKS = 200
    val_V_s: dict[int, Tensor] = {}  # stacked: (n, chunk_size, kv_dim)
    val_K_s: dict[int, Tensor] = {}
    for L in layers:
        n = min(len(val_K[L]), MAX_VAL_CHUNKS)
        if n == 0:
            logger.warning(f"Layer {L}: no val chunks collected")
            continue
        val_V_s[L] = torch.stack(val_V[L][:n])
        val_K_s[L] = torch.stack(val_K[L][:n])
    logger.info(f"  Val chunks: {len(val_K[layers[0]])} (capped at {MAX_VAL_CHUNKS})")

    # Accumulator — square map (kv_dim, kv_dim) for whichever direction is requested
    if k_only_for_v:
        acc_V = {L: NormalEq(kv_dim, kv_dim) for L in layers}
    else:  # v_only_for_k
        acc_K = {L: NormalEq(kv_dim, kv_dim) for L in layers}

    # Stream training data
    logger.info(f"Streaming {probe_cfg.train_seqs} train seqs ...")
    done = 0
    n_chunks = 0
    t0 = time.perf_counter()

    with tqdm(total=probe_cfg.train_seqs, desc="Calib train", unit="seq") as pbar:
        for batch in CALIB_STREAMS[probe_cfg.calib_data](
            "train", probe_cfg.batch_size, probe_cfg.train_seqs
        ):
            res = extract_multilayer(
                model,
                tokenizer,
                adapter,
                device,
                layers,
                batch,
                probe_cfg.seq_len,
                chunk_size,
            )
            for L in layers:
                _, vs, ks = res[L]
                if not ks:
                    continue
                if k_only_for_v:
                    Y_V = torch.stack(vs).reshape(-1, kv_dim)  # (n*chunk_size, kv_dim)
                    X_V = expand_k_only(torch.stack(ks))  # (n*chunk_size, kv_dim)
                    acc_V[L].update(X_V, Y_V)
                else:  # v_only_for_k
                    Y_K = torch.stack(ks).reshape(-1, kv_dim)  # (n*chunk_size, kv_dim)
                    X_K = expand_v_only(torch.stack(vs))  # (n*chunk_size, kv_dim)
                    acc_K[L].update(X_K, Y_K)

            done += len(batch)
            n_chunks += len(res[layers[0]][2]) if res[layers[0]][2] else 0

            inc = min(len(batch), max(probe_cfg.train_seqs - pbar.n, 0))
            pbar.update(inc)
            pbar.set_postfix_str(f"{n_chunks} chunks/layer")

    t_collect = time.perf_counter() - t0
    logger.info(f"  Collection done: {n_chunks} chunks/layer in {t_collect:.1f}s")

    # Solve
    logger.info("Solving linear systems ...")
    r2_values: list[float] = []

    if k_only_for_v:
        W_V: dict[int, Tensor] = {}
        for L in layers:
            W_V[L] = acc_V[L].solve(lam)
            r2 = (
                _r2_k_only(W_V[L], val_K_s[L], val_V_s[L])
                if L in val_K_s
                else float("nan")
            )
            if math.isfinite(r2):
                r2_values.append(r2)
            logger.info(f"  Layer {L:2d}: V R²={r2:.4f}")
        assert len(r2_values) > 0, "No finite V R² values collected during calibration"
        logger.info(f"  Mean over layers: V R²={sum(r2_values) / len(r2_values):.4f}")
        return W_V, {}
    else:  # v_only_for_k
        W_K: dict[int, Tensor] = {}
        for L in layers:
            W_K[L] = acc_K[L].solve(lam)
            r2 = (
                _r2_v_only(W_K[L], val_V_s[L], val_K_s[L])
                if L in val_V_s
                else float("nan")
            )
            if math.isfinite(r2):
                r2_values.append(r2)
            logger.info(f"  Layer {L:2d}: K R²={r2:.4f}")
        assert len(r2_values) > 0, "No finite K R² values collected during calibration"
        logger.info(f"  Mean over layers: K R²={sum(r2_values) / len(r2_values):.4f}")
        return {}, W_K

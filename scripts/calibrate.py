#!/usr/bin/env python3
"""
Offline probe calibration.

Streams C4 data, solves ridge regression for all layers, saves W_K and W_V.

Supported models:
    Dense:  Qwen/Qwen3-0.6B, Qwen/Qwen3-14B
            meta-llama/Llama-3.1-8B-Instruct
            google/gemma-3-4b-it
    MoE:    Qwen/Qwen3-30B-A3B

Usage:
    python scripts/calibrate.py --model Qwen/Qwen3-0.6B --train-seqs 10000 \\
        --weights weights/qwen3_0.6b_probe.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from vector.utils import ensure_padding_config, set_random_seed

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="VECTOR offline calibration")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model name")
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument(
        "--attn-implementation",
        choices=["flash_attention_3", "flash_attention_2", "sdpa"],
        default="flash_attention_3",
    )
    ap.add_argument("--train-seqs", type=int, default=10000)
    ap.add_argument("--val-seqs", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lam", type=float, default=1e-4, help="Ridge λ ratio")
    ap.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers (default: all)",
    )
    ap.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/probe.pt"),
        help="Output path for probe weights",
    )
    ap.add_argument(
        "--k-only-for-v",
        action="store_true",
        default=False,
        help=(
            "Pure K→V OLS: feature is K_i^{pre-RoPE} only (no o_end, no position). "
            "W_V becomes a square (kv_dim, kv_dim) matrix."
        ),
    )
    ap.add_argument(
        "--v-only-for-k",
        action="store_true",
        default=False,
        help=(
            "Pure V→K_pre OLS: feature is V_i only (no o_end, no position). "
            "W_K becomes a square (kv_dim, kv_dim) matrix. Ablation counterpart of "
            "--k-only-for-v."
        ),
    )
    ap.add_argument(
        "--calib-data",
        choices=["c4", "wikitext103"],
        default="c4",
        help="Calibration corpus (paper default: c4; wikitext103 = domain ablation)",
    )
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("=" * 80)
    for key, val in sorted(vars(args).items()):
        logger.info(f"  {key}: {val}")
    logger.info("=" * 80)
    set_random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  |  model: {args.model}")
    logger.info(f"Attention implementation: {args.attn_implementation}")

    from vector.config import ModelConfig, ProbeConfig
    from vector.models.registry import get_adapter
    from vector.probe.fit import calibrate
    from vector.probe.weights import save_weights

    model_cfg = ModelConfig.from_pretrained(args.model, chunk_size=args.chunk_size)
    if args.layers:
        model_cfg.adapter = "auto"

    adapter = get_adapter(args.model)

    probe_cfg = ProbeConfig(
        target_layers=args.layers or list(range(model_cfg.num_layers)),
        train_seqs=args.train_seqs,
        val_seqs=args.val_seqs,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lambda_ratio=args.lam,
        weights_path=args.weights,
        calib_data=args.calib_data,
    )

    logger.info(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Gemma3 is a multimodal model; AutoModelForCausalLM would load the full
    # vision-language class instead of the text backbone. Force Gemma3ForCausalLM.
    if "gemma-3-" in args.model.lower():
        from transformers import Gemma3ForCausalLM

        model = Gemma3ForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=args.attn_implementation,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=args.attn_implementation,
        )
    ensure_padding_config(tokenizer, model)
    model.eval()
    logger.info("Model loaded.")

    assert args.k_only_for_v or args.v_only_for_k, (
        "Either --k-only-for-v or --v-only-for-k is required."
    )
    assert not (args.k_only_for_v and args.v_only_for_k), (
        "--k-only-for-v and --v-only-for-k are mutually exclusive."
    )

    if args.k_only_for_v:
        logger.info("Pure K→V mode (k_only_for_v=True): W_V will be (kv_dim, kv_dim)")
        W_V, W_K = calibrate(
            model,
            tokenizer,
            adapter,
            model_cfg,
            probe_cfg,
            device,
            k_only_for_v=True,
        )
        assert not W_K, (
            f"calibrate() returned non-empty W_K in k_only_for_v mode: {list(W_K.keys())}"
        )
        meta = {
            "model": args.model,
            "chunk_size": args.chunk_size,
            "train_seqs": args.train_seqs,
            "seq_len": args.seq_len,
            "calib_data": args.calib_data,
            "lam": args.lam,
            "layers": list(W_V.keys()),
            "v_k_only": True,
        }
        save_weights(W_V, {}, args.weights, meta=meta)
    else:  # v_only_for_k
        logger.info("Pure V→K mode (v_only_for_k=True): W_K will be (kv_dim, kv_dim)")
        W_V, W_K = calibrate(
            model,
            tokenizer,
            adapter,
            model_cfg,
            probe_cfg,
            device,
            v_only_for_k=True,
        )
        assert not W_V, (
            f"calibrate() returned non-empty W_V in v_only_for_k mode: {list(W_V.keys())}"
        )
        meta = {
            "model": args.model,
            "chunk_size": args.chunk_size,
            "train_seqs": args.train_seqs,
            "seq_len": args.seq_len,
            "calib_data": args.calib_data,
            "lam": args.lam,
            "layers": list(W_K.keys()),
            "k_v_only": True,
        }
        save_weights({}, W_K, args.weights, meta=meta)

    logger.info("Done.")


if __name__ == "__main__":
    main()

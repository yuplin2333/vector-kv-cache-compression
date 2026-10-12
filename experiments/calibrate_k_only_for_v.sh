#!/usr/bin/env bash
# Offline K_pre -> V OLS calibration (C4), one W_OLS per model.
# Settings match the weights behind the paper's results:
#   Llama-3.1-8B-Instruct  5,000 sequences x 2,048 tokens
#   Qwen3-14B / Qwen3-0.6B 10,000 sequences x 2,048 tokens
# Usage: MODEL=Qwen/Qwen3-14B bash experiments/calibrate_k_only_for_v.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
case "${MODEL}" in
    meta-llama/Llama-3.1-8B-Instruct) TAG=llama31_8b;  TRAIN_SEQS=5000 ;;
    Qwen/Qwen3-14B)                   TAG=qwen3_14b;   TRAIN_SEQS=10000 ;;
    Qwen/Qwen3-0.6B)                  TAG=qwen3_0.6b;  TRAIN_SEQS=10000 ;;
    *) echo "unknown model ${MODEL}"; exit 1 ;;
esac
WEIGHTS="${REPO_ROOT}/weights/${TAG}_probe_konly.pt"
SEQ_LEN=2048
SEED=42

echo "  model:      ${MODEL}"
echo "  weights:    ${WEIGHTS}"
echo "  train_seqs: ${TRAIN_SEQS}"
echo "  seq_len:    ${SEQ_LEN}"

uv run python "${REPO_ROOT}/scripts/calibrate.py" \
    --model "${MODEL}" \
    --k-only-for-v \
    --train-seqs "${TRAIN_SEQS}" \
    --seq-len "${SEQ_LEN}" \
    --weights "${WEIGHTS}" \
    --seed "${SEED}"

echo ""
echo "=== Calibration done. Weights at ${WEIGHTS} ==="

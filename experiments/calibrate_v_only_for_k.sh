#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
WEIGHTS="${REPO_ROOT}/weights/llama31_8b_probe_vonly.pt"
TRAIN_SEQS=10000
SEQ_LEN=4096
SEED=42

echo "  model:      ${MODEL}"
echo "  weights:    ${WEIGHTS}"
echo "  train_seqs: ${TRAIN_SEQS}"
echo "  seq_len:    ${SEQ_LEN}"

uv run python "${REPO_ROOT}/scripts/calibrate.py" \
    --model "${MODEL}" \
    --v-only-for-k \
    --train-seqs "${TRAIN_SEQS}" \
    --seq-len "${SEQ_LEN}" \
    --weights "${WEIGHTS}" \
    --seed "${SEED}"

echo ""
echo "=== Calibration done. Weights at ${WEIGHTS} ==="

#!/usr/bin/env bash
# W_OLS variants for the calibration-data ablation (experiments/ols_calib_ablation.sh),
# Llama-3.1-8B-Instruct. Everything except the corpus / sample count matches
# experiments/calibrate_k_only_for_v.sh (C4, 5,000 x 2,048):
#   wikitext103 = WikiText-103 articles, 5,000 x 2,048   (domain shift)
#   c4_n500     = C4, 500 x 2,048                        (10x less calibration data)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="meta-llama/Llama-3.1-8B-Instruct"

run () {  # $1=calib-data $2=train-seqs $3=weights
    uv run python "${REPO_ROOT}/scripts/calibrate.py" \
        --model "${MODEL}" \
        --k-only-for-v \
        --calib-data "$1" \
        --train-seqs "$2" \
        --seq-len 2048 \
        --weights "$3" \
        --seed 42
}

run wikitext103 5000 "${REPO_ROOT}/weights/llama31_8b_probe_konly_wikitext103.pt"
run c4 500 "${REPO_ROOT}/weights/llama31_8b_probe_konly_c4_n500.pt"
echo "=== Ablation calibrations done ==="

#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_DIR="${REPO_ROOT}/kvpress/evaluation"
OUTPUT_DIR="${REPO_ROOT}/results/kvpress_no_press"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET="needle_in_haystack"
ESSAYS_DIR="${REPO_ROOT}/data/NIAH/PaulGrahamEssays"

CONTEXT_LENGTHS=(3000 6000 9000 12000 15000 18000 21000 24000 27000 30000)
NEEDLE_DEPTHS=(0 11 22 33 44 56 67 78 89 100)

echo "  output:  ${OUTPUT_DIR}"
echo "  essays:  ${ESSAYS_DIR} (n_haystacks=5)"

for CTX_LEN in "${CONTEXT_LENGTHS[@]}"; do
    for DEPTH in "${NEEDLE_DEPTHS[@]}"; do
        echo ""
        echo "--- no_press cr=0.00 max_ctx=${CTX_LEN} depth=${DEPTH} ---"
        uv run python "${EVAL_DIR}/evaluate.py" \
            --dataset "${DATASET}" \
            --model "${MODEL}" \
            --press_name no_press \
            --compression_ratio 0.00 \
            --max_context_length "${CTX_LEN}" \
            --needle_depth "${DEPTH}" \
            --essays_dir "${ESSAYS_DIR}" \
            --n_haystacks 5 \
            --output_dir "${OUTPUT_DIR}" \
            --seed 42
    done
done

echo ""
echo "=== All done. Results in ${OUTPUT_DIR} ==="

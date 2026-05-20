#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_DIR="${REPO_ROOT}/kvpress/evaluation"
OUTPUT_DIR="${REPO_ROOT}/results/kvpress_no_press"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET="longbench"
TASKS=(
    narrativeqa
    qasper
    multifieldqa_en
    hotpotqa
    2wikimqa
    musique
    gov_report
    qmsum
    multi_news
    trec
    triviaqa
    samsum
    passage_count
    passage_retrieval_en
    lcc
    repobench-p
)

echo "  output:  ${OUTPUT_DIR}"
for TASK in "${TASKS[@]}"; do
    echo ""
    echo "--- no_press task=${TASK} cr=0.00 ---"
    uv run python "${EVAL_DIR}/evaluate.py" \
        --dataset "${DATASET}" \
        --data_dir "${TASK}" \
        --model "${MODEL}" \
        --press_name no_press \
        --compression_ratio 0.00 \
        --output_dir "${OUTPUT_DIR}" \
        --seed 42
done

echo ""
echo "=== All done. Results in ${OUTPUT_DIR} ==="

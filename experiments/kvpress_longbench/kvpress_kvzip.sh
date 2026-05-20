#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_DIR="${REPO_ROOT}/kvpress/evaluation"
OUTPUT_DIR="${REPO_ROOT}/results/kvpress_kvzip"
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

METHODS=(
    kvzip
)

RATIOS=(0.25 0.50 0.75 0.90)

echo "  output:  ${OUTPUT_DIR}"

for METHOD in "${METHODS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        for RATIO in "${RATIOS[@]}"; do
            echo ""
            echo "--- ${METHOD} task=${TASK} cr=${RATIO} ---"
            uv run python "${EVAL_DIR}/evaluate.py" \
                --dataset "${DATASET}" \
                --data_dir "${TASK}" \
                --model "${MODEL}" \
                --press_name "${METHOD}" \
                --compression_ratio "${RATIO}" \
                --output_dir "${OUTPUT_DIR}" \
                --seed 42
        done
    done
done

echo ""
echo "=== All done. Results in ${OUTPUT_DIR} ==="

#!/usr/bin/env bash
# Reconstructability-signal ablation (Llama-3.1-8B, LongBench QA tasks).
#
# Question: does the OLS reconstruction-error signal (v_err) do real work, or would
# splitting the SAME candidate pool by importance rank alone do just as well?
#
# Three arms, identical candidate pool + memory budget, only the within-candidate
# approx/exact split rule differs:
#   keydiff_vector          -> approx = lowest v_err        (VECTOR, the claim)
#   keydiff_vector_imprank  -> approx = least important     (naive rank cutoff, no OLS)
#   keydiff_vector_random   -> approx = random subset       (control)
#
# Compare the arms with scripts/ablation_reconstructability_compare.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="${REPO_ROOT}/kvpress/evaluation"
WEIGHTS="${REPO_ROOT}/weights/llama31_8b_probe_konly.pt"
OUTPUT_DIR="${REPO_ROOT}/results/ablation_reconstructability"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SLUG="${MODEL//\//--}"  # evaluate.py replaces '/' with '--' in the run-dir name
DATASET="longbench"

GPU="${GPU:-2}"

# TASKS can be overridden via env (space-separated) to shard the sweep across GPUs.
if [[ -n "${TASKS:-}" ]]; then
    read -r -a TASKS <<< "${TASKS}"
else
    TASKS=(
        hotpotqa
        2wikimqa
        qasper
        narrativeqa
        musique
        multifieldqa_en
    )
fi

METHODS=(
    keydiff_vector
    keydiff_vector_imprank
    keydiff_vector_random
)

RATIOS=(0.75 0.90)

echo "GPU=${GPU}  weights=${WEIGHTS}  output=${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Loop order task -> ratio -> method so each (task, ratio) cell gets all three arms
# computed close together, giving comparable triples early.
for TASK in "${TASKS[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            echo ""
            echo "=== ${METHOD} | ${TASK} | cr=${RATIO} ==="
            # bash-level skip: avoid reloading the model just to skip a done run
            # (each reload is a host-RAM spike -> reduces OOM risk on a shared box).
            RUN_DIR="${OUTPUT_DIR}/${DATASET}__${TASK}__${MODEL_SLUG}__${METHOD}__${RATIO}"
            if [[ -f "${RUN_DIR}/metrics.json" ]]; then
                echo "skip (already done): ${RUN_DIR}"
                continue
            fi
            CUDA_VISIBLE_DEVICES="${GPU}" uv run python "${EVAL_DIR}/evaluate.py" \
                --dataset "${DATASET}" \
                --data_dir "${TASK}" \
                --model "${MODEL}" \
                --press_name "${METHOD}" \
                --compression_ratio "${RATIO}" \
                --vector_weights "${WEIGHTS}" \
                --output_dir "${OUTPUT_DIR}" \
                --seed 42
        done
    done
done

echo ""
echo "=== Reconstructability ablation done. Results in ${OUTPUT_DIR} ==="

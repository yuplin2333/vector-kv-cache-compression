#!/usr/bin/env bash
# Ablation: keydiff_vector (K_pre→V) vs keydiff_v2k (V→K_pre)
#
# Runbook:
#   Calibrate v2k weights first (one-time):
#     CUDA_VISIBLE_DEVICES=0 python scripts/calibrate.py \
#         --model meta-llama/Llama-3.1-8B-Instruct \
#         --v-only-for-k \
#         --train-seqs 10000 --seq-len 4096 \
#         --weights weights/llama31_8b_probe_vonly.pt --seed 42
#
#   Smoke check (n=5, 1 task, 1 ratio):
#     bash experiments/ablation_keydiff_v2k_longbench.sh smoke
#
#   Full ablation:
#     bash experiments/ablation_keydiff_v2k_longbench.sh full

set -euo pipefail

MODE="${1:-full}"
if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
    echo "Usage: $0 [smoke|full]"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="${REPO_ROOT}/kvpress/evaluation"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET="longbench"

WEIGHTS_KV="${REPO_ROOT}/weights/llama31_8b_probe_konly.pt"   # K_pre→V (existing)
WEIGHTS_VK="${REPO_ROOT}/weights/llama31_8b_probe_vonly.pt"   # V→K_pre (calibrate first)

TASKS=(hotpotqa qasper)
RATIOS=(0.50 0.75 0.90)

if [[ "${MODE}" == "smoke" ]]; then
    TASKS=(hotpotqa)
    RATIOS=(0.75)
fi

echo "mode:       ${MODE}"
echo "model:      ${MODEL}"
echo "tasks:      ${TASKS[*]}"
echo "ratios:     ${RATIOS[*]}"
echo ""

assert_weights_exist() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: weights file not found: ${path}"
        echo "Run calibration first (see runbook at top of this script)."
        exit 1
    fi
}

assert_weights_exist "${WEIGHTS_KV}"
assert_weights_exist "${WEIGHTS_VK}"

# Run one method × task × ratio combination.
run_eval() {
    local press_name="$1"
    local weights="$2"
    local task="$3"
    local cr="$4"
    local output_dir="$5"

    echo "--- ${press_name}  task=${task}  cr=${cr} ---"
    uv run python "${EVAL_DIR}/evaluate.py" \
        --dataset "${DATASET}" \
        --data_dir "${task}" \
        --model "${MODEL}" \
        --press_name "${press_name}" \
        --compression_ratio "${cr}" \
        --vector_weights "${weights}" \
        --dump_vector_stats True \
        --output_dir "${output_dir}" \
        --seed 42
}

for TASK in "${TASKS[@]}"; do
    for CR in "${RATIOS[@]}"; do
        # keydiff_vector (K_pre→V) results already collected in main experiments;
        # uncomment to re-run for direct comparison in the same output dir.
        # run_eval "keydiff_vector" "${WEIGHTS_KV}" "${TASK}" "${CR}" \
        #     "${REPO_ROOT}/results/ablation_keydiff_vector_lb"
        run_eval "keydiff_v2k" "${WEIGHTS_VK}" "${TASK}" "${CR}" \
            "${REPO_ROOT}/results/ablation_keydiff_v2k_lb"
    done
done

echo ""
echo "=== All done. ==="
echo "keydiff_vector results: ${REPO_ROOT}/results/ablation_keydiff_vector_lb"
echo "keydiff_v2k    results: ${REPO_ROOT}/results/ablation_keydiff_v2k_lb"

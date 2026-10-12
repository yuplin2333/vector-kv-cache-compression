#!/usr/bin/env bash
# SCBench multi-turn (shared-context KV reuse), Llama-3.1-8B.
#
# Tasks: one per SCBench category, picked a priori by FULL-KV headroom only (5-session no_press
# pilot; never by any base-vs-VECTOR comparison):
#   string retrieval = scbench_vt, semantic retrieval = scbench_choice_eng,
#   global information = scbench_many_shot, multi-tasking = scbench_summary_with_needles.
# Methods: base scorer vs +VECTOR for SnapKV / KeyDiff / PyramidKV at p_c in {0.75, 0.90}, plus no_press.
# Driver: scripts/scbench_multiturn.py (context compressed once, reused across all turns).
# Scoring: scripts/scbench_score.py in RocketKV/.venv-rocketkv-457 (SCBench compute_scores).
#
# Usage: GPU=2 bash experiments/scbench_multiturn.sh            (REVERSE=1 on a second GPU)
#        METHODS="snapkv snapkv_vector" PCS="0.90" TASKS="scbench_vt" GPU=3 bash ...
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-${REPO_ROOT}/results/scbench}"
GPU="${GPU:-0}"
WEIGHTS="${REPO_ROOT}/weights/llama31_8b_probe_konly.pt"
TASKS=(${TASKS:-scbench_many_shot scbench_choice_eng scbench_vt scbench_summary_with_needles})
METHODS=(${METHODS:-snapkv snapkv_vector keydiff keydiff_vector pyramidkv pyramidkv_vector})
PCS=(${PCS:-0.75 0.90})

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CELLS=()
for T in "${TASKS[@]}"; do
    CELLS+=("${T} no_press 0.00")
    for PC in "${PCS[@]}"; do
        for M in "${METHODS[@]}"; do CELLS+=("${T} ${M} ${PC}"); done
    done
done
if [[ "${REVERSE:-0}" == "1" ]]; then
    REV=(); for ((i=${#CELLS[@]}-1; i>=0; i--)); do REV+=("${CELLS[i]}"); done; CELLS=("${REV[@]}")
fi

cd "${REPO_ROOT}"
for CELL in "${CELLS[@]}"; do
    read -r T M PC <<< "${CELL}"
    # SnapKV / PyramidKV (+VECTOR) are query-aware: question 1 is compressed together with the shared
    # context (later questions stay unseen).
    QA_ARGS=(); QA_SUFFIX=""
    if [[ "${M}" == snapkv* || "${M}" == pyramidkv* ]]; then QA_ARGS=(--query_aware); QA_SUFFIX="__query_aware"; fi
    if [[ -f "${OUT}/${T}__${M}__${PC}${QA_SUFFIX}/DONE" ]]; then echo "skip (done): ${CELL}${QA_SUFFIX}"; continue; fi
    # Non-blocking per-cell lock: a second driver (REVERSE=1) skips a cell that is in progress
    # instead of appending duplicate sessions to the same predictions.jsonl. Released on exit.
    mkdir -p "${OUT}/.locks"
    exec 9>"${OUT}/.locks/${T}__${M}__${PC}${QA_SUFFIX}.lock"
    if ! flock -n 9; then echo "skip (running elsewhere): ${CELL}"; exec 9>&-; continue; fi
    echo "=== ${CELL} ==="
    echo "RUN_CELL ${CELL}"
    .venv/bin/python scripts/scbench_multiturn.py --task "${T}" --press_name "${M}" \
        --compression_ratio "${PC}" --vector_weights "${WEIGHTS}" --output_dir "${OUT}" "${QA_ARGS[@]}" \
        || echo "CELL_FAILED ${CELL}"
    exec 9>&-
done
echo "=== SCBench sweep finished (GPU ${GPU}) ==="

#!/usr/bin/env bash
# OLS calibration-data ablation, Llama-3.1-8B, LongBench (16 tasks, full split).
#
# Three W_OLS calibrations, everything else identical (k-only-for-v, seq_len 2048, seed 42):
#   c4       = weights/llama31_8b_probe_konly.pt            (C4, 5000 seqs — the weights behind the paper)
#   wiki     = weights/llama31_8b_probe_konly_wikitext103.pt (WikiText-103 articles, 5000 seqs — domain shift)
#   c4_n500  = weights/llama31_8b_probe_konly_c4_n500.pt     (C4, 500 seqs — 10x less calibration data)
# Weights: experiments/calibrate_k_only_for_v.sh (c4) and experiments/calibrate_ols_ablation.sh
# (wiki, c4_n500). The base scorers are run too (tag "base"), which also yields per-sample
# base-vs-VECTOR pairs for bootstrap CIs (scripts/bootstrap_ci_longbench.py).
#
# Usage: GPU=2 bash experiments/ols_calib_ablation.sh      (REVERSE=1 on a second GPU)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="${REPO_ROOT}/kvpress/evaluation"
OUT="${OUT:-${REPO_ROOT}/results/ols_ablation}"
GPU="${GPU:-0}"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
TASKS=(narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum
       multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p)
PCS=(${PCS:-0.90 0.75})
declare -A WEIGHTS=(
    [c4]="${REPO_ROOT}/weights/llama31_8b_probe_konly.pt"
    [wiki]="${REPO_ROOT}/weights/llama31_8b_probe_konly_wikitext103.pt"
    [c4_n500]="${REPO_ROOT}/weights/llama31_8b_probe_konly_c4_n500.pt"
)
TAGS=(${TAGS:-c4 wiki c4_n500 base})

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CELLS=()
for PC in "${PCS[@]}"; do
    for TAG in "${TAGS[@]}"; do
        if [[ "${TAG}" == "base" ]]; then METHODS=(keydiff snapkv); else METHODS=(keydiff_vector snapkv_vector); fi
        for M in "${METHODS[@]}"; do
            for T in "${TASKS[@]}"; do CELLS+=("${TAG} ${M} ${PC} ${T}"); done
        done
    done
done
if [[ "${REVERSE:-0}" == "1" ]]; then
    REV=(); for ((i=${#CELLS[@]}-1; i>=0; i--)); do REV+=("${CELLS[i]}"); done; CELLS=("${REV[@]}")
fi

mkdir -p "${OUT}/.locks"
for CELL in "${CELLS[@]}"; do
    read -r TAG M PC T <<< "${CELL}"
    # SnapKV is query-aware: run it (and SnapKV+VECTOR) with the question in the compressed context,
    # as in the paper's main table.
    QA_ARGS=(); QA_SUFFIX=""
    if [[ "${M}" == snapkv* ]]; then QA_ARGS=(--query_aware); QA_SUFFIX="__query_aware"; fi
    DIR="${OUT}/${TAG}/longbench__${T}__meta-llama--Llama-3.1-8B-Instruct__${M}__${PC}${QA_SUFFIX}"
    if [[ -f "${DIR}/metrics.json" ]]; then echo "skip (done): ${CELL}"; continue; fi
    exec 9>"${OUT}/.locks/${TAG}__${M}__${PC}__${T}${QA_SUFFIX}.lock"
    if ! flock -n 9; then echo "skip (running elsewhere): ${CELL}"; exec 9>&-; continue; fi
    # A dir without metrics.json is a killed run; evaluate.py would write into DIR/1 (invisible to skip-done), so archive it (mv, never rm).
    if [[ -d "${DIR}" ]]; then mkdir -p "${REPO_ROOT}/results/_incomplete_archive"; mv "${DIR}" "${REPO_ROOT}/results/_incomplete_archive/ols_${TAG}_$(basename "${DIR}").$(date +%s)"; fi
    echo "=== ${CELL} ==="
    echo "RUN_CELL ${CELL}"
    WARGS=()
    if [[ "${TAG}" != "base" ]]; then
        [[ -f "${WEIGHTS[${TAG}]}" ]] || { echo "missing weights ${WEIGHTS[${TAG}]}"; exit 1; }
        WARGS=(--vector_weights "${WEIGHTS[${TAG}]}")
    fi
    "${REPO_ROOT}/.venv/bin/python" "${EVAL_DIR}/evaluate.py" --dataset longbench --data_dir "${T}" \
        --model "${MODEL}" --press_name "${M}" --compression_ratio "${PC}" \
        --output_dir "${OUT}/${TAG}" --seed 42 "${WARGS[@]}" "${QA_ARGS[@]}" || echo "CELL_FAILED ${CELL}"
    exec 9>&-
done
echo "=== OLS ablation sweep finished (GPU ${GPU}) ==="

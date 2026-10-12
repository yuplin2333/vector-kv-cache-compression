#!/usr/bin/env bash
# VECTOR long-context eval on RULER (Llama-3.1-8B), 16k-128k.
# Same kvpress evaluate.py harness as the LongBench results; loads locally-generated
# RULER (data/RULER/<length>/, built by scripts/ruler_build_local.py) via RULER_LOCAL_DIR.
#
# Methods: no_press (No Comp) + 4 base scorers + their 4 +VECTOR variants. p_c ∈ {0.75, 0.90}
# (VECTOR's target regime). VECTOR is FlashAttention/sdpa-compatible (a deliberate contrast to
# D2O's O(L^2) 16k wall), so it runs to 128k on one GPU.
#
# Env knobs (so different lengths can run in parallel on different GPUs with different sample caps):
#   LENGTHS="16384 32768"   GPU=0   FRACTION=0.5   RATIOS="0.75 0.90"   bash experiments/ruler_longcontext.sh
# FRACTION subsamples the 100/task generated pool (method comparison, not a leaderboard; note the cap).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="${REPO_ROOT}/kvpress/evaluation"
export RULER_LOCAL_DIR="${RULER_LOCAL_DIR:-${REPO_ROOT}/data/RULER}"
export HF_TOKEN="${HF_TOKEN:-$(cat "${HOME}/.cache/huggingface/token" 2>/dev/null || true)}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
WEIGHTS="${WEIGHTS:-${REPO_ROOT}/weights/llama31_8b_probe_konly.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/ruler_longcontext}"
GPU="${GPU:-0}"
FRACTION="${FRACTION:-1.0}"
LENGTHS=(${LENGTHS:-16384 32768 65536 131072})
RATIOS=(${RATIOS:-0.75 0.90})
BASE_METHODS=(keydiff snapkv kvzip pyramidkv)
VECTOR_METHODS=(keydiff_vector snapkv_vector kvzip_vector pyramidkv_vector)
# METHODS overrides the method list/order (e.g. reverse order on a second GPU so two
# skip-done drivers converge from opposite ends instead of colliding on the first slow cell).
METHODS=(${METHODS:-${BASE_METHODS[@]} ${VECTOR_METHODS[@]}})
# SKIP_NOPRESS=1 skips the No-Comp cell (already done on a length being back-filled).
SKIP_NOPRESS="${SKIP_NOPRESS:-0}"

mkdir -p "${OUTPUT_DIR}"
echo "GPU=${GPU} model=${MODEL} lengths=${LENGTHS[*]} ratios=${RATIOS[*]} fraction=${FRACTION} out=${OUTPUT_DIR}"

MODEL_TAG="${MODEL//\//--}"
run_cell () {  # $1=length $2=method $3=ratio
    local LEN="$1" METHOD="$2" RATIO="$3"
    # skip-done: evaluate.py names the run dir ruler__<len>__<model>__<method>__<ratio>__fraction<f>.
    local RSTR FSTR DIR
    RSTR="$(printf '%.2f' "${RATIO}")"
    FSTR="$(printf '%.3f' "${FRACTION}")"
    # SnapKV / PyramidKV (and their +VECTOR variants) are query-aware methods: the question is appended
    # to the context before compression, as in the paper's main table.
    local QA_ARGS=() QA_SUFFIX=""
    if [[ "${METHOD}" == snapkv* || "${METHOD}" == pyramidkv* ]]; then QA_ARGS=(--query_aware); QA_SUFFIX="__query_aware"; fi
    DIR="${OUTPUT_DIR}/ruler__${LEN}__${MODEL_TAG}__${METHOD}__${RSTR}__fraction${FSTR}${QA_SUFFIX}"
    if [[ -f "${DIR}/metrics.json" ]]; then
        echo "skip (done): ${METHOD} ${LEN} cr=${RSTR}${QA_SUFFIX}"
        return 0
    fi
    # Non-blocking per-cell lock so several drivers can share the queue without duplicating a cell.
    mkdir -p "${OUTPUT_DIR}/.locks"
    exec 9>"${OUTPUT_DIR}/.locks/${LEN}__${METHOD}__${RSTR}${QA_SUFFIX}.lock"
    if ! flock -n 9; then echo "skip (running elsewhere): ${METHOD} ${LEN} cr=${RSTR}"; exec 9>&-; return 0; fi
    # A dir without metrics.json is a killed run; evaluate.py would write into DIR/1 (invisible to skip-done), so archive it (mv, never rm).
    if [[ -d "${DIR}" ]]; then mkdir -p "${REPO_ROOT}/results/_incomplete_archive"; mv "${DIR}" "${REPO_ROOT}/results/_incomplete_archive/$(basename "${DIR}").$(date +%s)"; fi
    echo "RUN_CELL ${METHOD} ${LEN} ${RSTR}${QA_SUFFIX}"
    CUDA_VISIBLE_DEVICES="${GPU}" uv run python "${EVAL_DIR}/evaluate.py" \
        --dataset ruler --data_dir "${LEN}" \
        --model "${MODEL}" \
        --press_name "${METHOD}" \
        --compression_ratio "${RATIO}" \
        --vector_weights "${WEIGHTS}" \
        --fraction "${FRACTION}" \
        --output_dir "${OUTPUT_DIR}" \
        --seed 42 "${QA_ARGS[@]}"
    local rc=$?
    exec 9>&-
    return ${rc}
}

for LEN in "${LENGTHS[@]}"; do
    echo ""; echo "########## length=${LEN} ##########"
    # No Comp once per length (ratio ignored -> 0).
    if [[ "${SKIP_NOPRESS}" != "1" ]]; then
        echo "=== no_press | ${LEN} ==="; run_cell "${LEN}" no_press 0.0 || echo "FAILED no_press ${LEN}"
    fi
    for METHOD in "${METHODS[@]}"; do
        for RATIO in "${RATIOS[@]}"; do
            echo "=== ${METHOD} | ${LEN} | cr=${RATIO} ==="
            run_cell "${LEN}" "${METHOD}" "${RATIO}" || echo "FAILED ${METHOD} ${LEN} ${RATIO}"
        done
    done
done
echo ""; echo "=== RULER long-context sweep done. Results in ${OUTPUT_DIR} ==="

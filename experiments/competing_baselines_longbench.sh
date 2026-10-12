#!/usr/bin/env bash
# Competing-baseline comparison on LongBench (16 English tasks, full split).
#
# Runs competing KV-compression baselines that are built into kvpress (same eval
# harness, same memory budget => fair comparison) for the LongBench comparison table.
#
# Methods (all built into kvpress, no external integration, no vector weights needed):
#   chunkkv             ChunkKV — SnapKV over chunks
#   expected_attention  KVPress's own strong baseline
#   adakv_snapkv        AdaKV (head-adaptive budget) over SnapKV
#   tova                TOVA
# Compared against the VECTOR-augmented presses (experiments/kvpress_longbench/).
# D2O and RocketKV need their official repos: experiments/d2o_longbench.sh,
# scripts/d2o_native_longbench.py and experiments/rocketkv_longbench.sh.
#
# Usage: MODEL=Qwen/Qwen3-14B GPU=0 bash experiments/competing_baselines_longbench.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="${REPO_ROOT}/kvpress/evaluation"
OUTPUT_DIR="${REPO_ROOT}/results/competing_baselines"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DATASET="longbench"
GPU="${GPU:-0}"

TASKS=(
    narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique
    gov_report qmsum multi_news trec triviaqa samsum
    passage_count passage_retrieval_en lcc repobench-p
)

# Override via env, e.g. METHODS="chunkkv" for a quick first pass.
if [[ -n "${METHODS:-}" ]]; then
    read -r -a METHODS <<< "${METHODS}"
else
    METHODS=(chunkkv expected_attention adakv_snapkv tova)
fi

RATIOS=(0.25 0.50 0.75 0.90)
if [[ "${REVERSE:-0}" == "1" ]]; then  # second driver on the same queue starts from the other end
    REV=(); for ((i=${#TASKS[@]}-1; i>=0; i--)); do REV+=("${TASKS[i]}"); done; TASKS=("${REV[@]}")
fi

# evaluate.py names run dirs by the model id with "/" replaced by "--".
MODEL_TAG="${MODEL//\//--}"
# Query-aware methods run query-aware (question appended to the context before compression), per the
# paper's protocol "follow each method's original setting": ChunkKV (SnapKV over chunks), AdaKV-SnapKV,
# TOVA (scores by the current query's attention). ExpectedAttention is query-agnostic by design.
QA_METHODS=" chunkkv adakv_snapkv tova "

echo "GPU=${GPU}  model=${MODEL}  methods=${METHODS[*]}  output=${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

for METHOD in "${METHODS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        for RATIO in "${RATIOS[@]}"; do
            QA_ARGS=(); QA_SUFFIX=""
            if [[ "${QA_METHODS}" == *" ${METHOD} "* ]]; then QA_ARGS=(--query_aware); QA_SUFFIX="__query_aware"; fi
            RUN_DIR="${OUTPUT_DIR}/${DATASET}__${TASK}__${MODEL_TAG}__${METHOD}__${RATIO}${QA_SUFFIX}"
            if [[ -f "${RUN_DIR}/metrics.json" ]]; then
                echo "skip (done): ${METHOD} ${TASK} ${RATIO}"
                continue
            fi
            # Non-blocking per-cell lock so several GPUs can share one queue without duplicating a cell.
            mkdir -p "${OUTPUT_DIR}/.locks"
            exec 9>"${OUTPUT_DIR}/.locks/$(basename "${RUN_DIR}").lock"
            if ! flock -n 9; then echo "skip (running elsewhere): ${METHOD} ${TASK} ${RATIO}"; exec 9>&-; continue; fi
            # A dir without metrics.json is a killed run; evaluate.py would write into DIR/1 (invisible to skip-done), so archive it (mv, never rm).
            if [[ -d "${RUN_DIR}" ]]; then mkdir -p "${REPO_ROOT}/results/_incomplete_archive"; mv "${RUN_DIR}" "${REPO_ROOT}/results/_incomplete_archive/$(basename "${RUN_DIR}").$(date +%s)"; fi
            echo ""
            echo "=== ${METHOD} | ${TASK} | cr=${RATIO} ==="
            echo "RUN_CELL ${METHOD} ${TASK} ${RATIO}"
            CUDA_VISIBLE_DEVICES="${GPU}" uv run python "${EVAL_DIR}/evaluate.py" \
                --dataset "${DATASET}" \
                --data_dir "${TASK}" \
                --model "${MODEL}" \
                --press_name "${METHOD}" \
                --compression_ratio "${RATIO}" \
                --output_dir "${OUTPUT_DIR}" \
                --seed 42 "${QA_ARGS[@]}" || echo "CELL_FAILED ${METHOD} ${TASK} ${RATIO}"
            exec 9>&-
        done
    done
done

echo ""
echo "=== Competing baselines done. Results in ${OUTPUT_DIR} ==="

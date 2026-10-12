#!/usr/bin/env bash
# RocketKV (NVlabs/RocketKV, ICML'25) competitor eval on LongBench, mapped to our p_c grid.
#
# RocketKV is a two-stage decode-time compressor (SnapKV coarse eviction + hybrid sparse
# attention). Its native knob is an ABSOLUTE --token_budget; patches/RocketKV adds
# --compression_ratio (p_c): per example, token_budget = round((1-p_c) * prompt_len), and the
# Stage-1 eviction target (prompt_budget) is bound to (1-p_c) * prompt_len as well, so the
# permanently retained KV matches the kvpress-style p_c of the main tables (by default Stage-1
# keeps a cr^r-widened prompt_budget, ~40% of the prompt at nominal p_c=0.90). Stage-2 knobs and
# RocketKV's algorithm are unchanged.
#
# Runs in RocketKV's own venv (Llama: transformers 4.45.2 / torch 2.4.1 / flash-attn 2.6.3;
# Qwen3: PY=RocketKV/.venv-rocketkv-457/bin/python with transformers 4.57.6, MODEL_TAG=qwen3-14b
# or qwen3-0.6b). Score with scripts/rocketkv_aggregate.py (kvpress's LongBench scorer). Full
# context (model_max_len 127500, truncation=middle) — no 16k cap, unlike D2O.
#
# Usage: GPU=0 PCS="0.25 0.50 0.75 0.90" bash experiments/rocketkv_longbench.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RK_DIR="${REPO_ROOT}/RocketKV"
PY="${PY:-${RK_DIR}/.venv-rocketkv/bin/python}"
# resolve in case the caller passes a relative path (run_cell cds into RK_DIR)
case "${PY}" in /*) ;; *) PY="${REPO_ROOT}/${PY}";; esac
OUT="${OUT:-${REPO_ROOT}/results/rocketkv_longbench}"
GPU="${GPU:-0}"
METHOD="${METHOD:-rocket}"
MODEL_TAG="${MODEL_TAG:-llama3.1-8b-instruct}"
PCS=(${PCS:-0.25 0.50 0.75 0.90})
DATASETS=(${DATASETS:-narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum \
          multi_news trec triviaqa samsum passage_retrieval_en passage_count lcc repobench-p})
# REVERSE=1 iterates tasks in reverse — run a second GPU this way so two skip-done drivers on the
# same p_c converge from opposite ends instead of colliding on the first uncompleted task.
if [[ "${REVERSE:-0}" == "1" ]]; then
    _rev=(); for ((i=${#DATASETS[@]}-1;i>=0;i--)); do _rev+=("${DATASETS[i]}"); done
    DATASETS=("${_rev[@]}")
fi

# Models are loaded from the local HF cache (offline); download them first. The pipeline config
# is patched to meta-llama/Llama-3.1-8B-Instruct, the model used in the main tables.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Byte-exact kvpress protocol — feed the SAME rows kvpress uses (per-task JSONL dumps of
# Xnhyacinth/LongBench, see README; prompts built by pipeline/kvpress_prompt.py). Verified
# token-identical to kvpress on all 16 tasks by scripts/rocketkv_protocol_check.py.
export LB_KVPROTO_DIR="${LB_KVPROTO_DIR:-${RK_DIR}/dataset/longbench_kvproto}"

mkdir -p "${OUT}"
echo "GPU=${GPU} method=${METHOD} pcs=${PCS[*]} out=${OUT}"

run_cell () {  # $1=pc $2=dataset
    local PC="$1" DS="$2"
    local ofolder="${OUT}/${PC}/${DS}/"
    # evaluate.py-style skip-done: eval_longbech.py writes pred/<method>/<ds>_llama3_pc_<pc>.jsonl
    local PCTAG; PCTAG="$(printf 'pc_%.2f' "${PC}")"
    local jsonl="${ofolder}pred/${METHOD}/${DS}_llama3_${PCTAG}.jsonl"
    if [[ -f "${jsonl}" ]]; then echo "skip (done): ${METHOD} ${DS} pc=${PC}"; return 0; fi
    ( cd "${RK_DIR}" && CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" pipeline/inf_stream_llm/main.py \
        --method "${METHOD}" \
        --compression_ratio "${PC}" \
        --exp_desc "longbench_${DS}_${MODEL_TAG}_${METHOD}_pc${PC}" \
        --pipeline_config_dir "config/pipeline_config/longbench/${MODEL_TAG}.json" \
        --eval_config_dir "config/eval_config/longbench/${DS}.json" \
        --output_folder_dir "${ofolder}" )
}

for PC in "${PCS[@]}"; do
    echo ""; echo "########## p_c=${PC} ##########"
    for DS in "${DATASETS[@]}"; do
        echo "=== ${METHOD} | ${DS} | p_c=${PC} ==="
        run_cell "${PC}" "${DS}" || echo "FAILED ${METHOD} ${DS} ${PC}"
    done
done
echo ""; echo "=== RocketKV LongBench sweep done. Results in ${OUT} ==="

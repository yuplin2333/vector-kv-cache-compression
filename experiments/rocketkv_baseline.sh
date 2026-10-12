#!/usr/bin/env bash
# RocketKV Full-KV (No-Comp) baseline on LongBench — the harness reference point for the
# RocketKV rows (checks that RocketKV's harness reproduces kvpress's No-Comp score).
# Same 16 English tasks, RocketKV venv, full context.
# Usage: GPU=1 bash experiments/rocketkv_baseline.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RK_DIR="${REPO_ROOT}/RocketKV"
PY="${PY:-${RK_DIR}/.venv-rocketkv/bin/python}"
case "${PY}" in /*) ;; *) PY="${REPO_ROOT}/${PY}";; esac
OUT="${OUT:-${REPO_ROOT}/results/rocketkv_longbench/baseline}"
GPU="${GPU:-1}"
MODEL_TAG="${MODEL_TAG:-llama3.1-8b-instruct}"
DATASETS=(narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum \
          multi_news trec triviaqa samsum passage_retrieval_en passage_count lcc repobench-p)
if [[ "${REVERSE:-0}" == "1" ]]; then
    for ((i=${#DATASETS[@]}-1; i>=0; i--)); do REV+=("${DATASETS[i]}"); done
    DATASETS=("${REV[@]}")
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Byte-exact kvpress protocol (see experiments/rocketkv_longbench.sh).
export LB_KVPROTO_DIR="${LB_KVPROTO_DIR:-${RK_DIR}/dataset/longbench_kvproto}"
mkdir -p "${OUT}"
echo "GPU=${GPU} baseline out=${OUT}"
for DS in "${DATASETS[@]}"; do
    o="${OUT}/${DS}/"
    # result.json is written by the LongBench scorer only after all predictions are done.
    if [[ -f "${o}pred/baseline/result.json" ]]; then echo "skip (done): ${DS}"; continue; fi
    echo "=== baseline | ${DS} ==="
    ( cd "${RK_DIR}" && CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" pipeline/baseline/main.py \
        --method baseline --exp_desc "lb_${DS}_baseline" \
        --pipeline_config_dir "config/pipeline_config/longbench/${MODEL_TAG}.json" \
        --eval_config_dir "config/eval_config/longbench/${DS}.json" \
        --output_folder_dir "${o}" ) || echo "FAILED ${DS}"
done
echo "=== RocketKV baseline done. ${OUT} ==="

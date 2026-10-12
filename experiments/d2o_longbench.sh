#!/usr/bin/env bash
# D2O competitor comparison on LongBench, aligned to the kvpress protocol of the main tables.
#
# Runs the OFFICIAL D2O repo (AIoT-MLSys-Lab/D2O, ICLR'25) — cloned to ./D2O with
# patches/D2O applied, method logic unchanged — on Llama-3.1-8B-Instruct over
# the 16 English LongBench-v1 tasks (full split), matching the main-table setup
# (same prompts, seed 42). Score with scripts/d2o_rescore_kvpress.py.
# The Qwen3 rows use the faithful port in scripts/d2o_native_longbench.py.
#
# Budget mapping: D2O keeps ~ hh_ratio + recent_ratio of tokens; we set
#   hh = recent = (1 - p_c)/2  so total kept = 1 - p_c, matching our p_c grid.
#
# Context cap (documented caveat): D2O uses
# eager (materialized L×L) attention (its H2O-style scoring needs the full attention
# matrix, so no FlashAttention) -> memory is O(L^2) per layer. Empirically a single
# ~16k-token sample already peaks ~124 GiB on an EMPTY 140 GiB H200 (it makes several
# full (bsz,heads,L,L) copies per layer: fp32 softmax + bf16 result + score reductions),
# so the originally-planned 24k cap is INFEASIBLE even on the largest single GPU. We
# therefore (a) removed D2O's redundant .clone() calls before its score .sum()s (a pure
# memory optimization -- bit-identical output, see modeling_llama3_7b_13b_d2o.py:632/643)
# and (b) cap max_length=16k (config/model2maxlen.json). At 16k the per-layer prefill
# peak is ~84 GiB (fits an empty card with headroom) and only ~4.5% of LongBench samples
# are truncated further (median 5569; narrativeqa is the main long-context task hit).
# This eager-attention memory wall is itself a limitation of materialized-attention
# methods (H2O/D2O family) that VECTOR -- query-agnostic + FlashAttention-compatible --
# does not share.
#
# Requires a (near-)EMPTY GPU. Pass GPU=<idx>. Uses the dedicated D2O venv (D2O/.venv-d2o,
# see README) + HF_TOKEN for the gated model.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D2O_DIR="${REPO_ROOT}/D2O/LLM_merge_new"
D2O_PY="${REPO_ROOT}/D2O/.venv-d2o/bin/python"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_TAG="Llama-3.1-8B-Instruct"   # run_pred derives this via basename
GPU="${GPU:-0}"


# p_c -> per-side ratio (hh = recent = (1-p_c)/2)
declare -A HALF=( [0.25]=0.375 [0.50]=0.25 [0.75]=0.125 [0.90]=0.05 )

cd "${D2O_DIR}"
echo "GPU=${GPU}  model=${MODEL}  D2O dir=${D2O_DIR}"

for PC in 0.25 0.50 0.75 0.90; do
    R="${HALF[$PC]}"
    # action tag "d2o_kvproto": D2O now builds the model input with kvpress's EXACT recipe
    # (context_prefix/question_template/answer_prefix + separator trick; verified token-id
    # identical to kvpress pipeline.preprocess -- see run_pred_long_bench_sample.py get_pred).
    # This is apples-to-apples with the VECTOR/kvpress main table.
    ACTION="d2o_kvproto_pc${PC}"
    RESULT="pred/${MODEL_TAG}_${ACTION}/result1.json"
    if [[ -f "${RESULT}" ]]; then
        echo "skip (done): p_c=${PC} -> ${RESULT}"
        continue
    fi
    echo ""
    echo "=== D2O p_c=${PC}  (hh=recent=${R}) ==="
    CUDA_VISIBLE_DEVICES="${GPU}" "${D2O_PY}" run_pred_long_bench_sample.py \
        --model_name_or_path "${MODEL}" \
        --cache_dir "${HF_HOME}" \
        --use_d2o True --model_type llama3 \
        --hh_ratio "${R}" --recent_ratio "${R}" \
        --action_name "${ACTION}" --e False
    echo "--- scoring p_c=${PC} ---"
    "${D2O_PY}" eval_long_bench.py --model "${MODEL_TAG}_${ACTION}"
    echo "result: ${RESULT}"
done

echo ""
echo "=== D2O LongBench done. Per-p_c scores in pred/${MODEL_TAG}_d2o_pc*/result1.json ==="

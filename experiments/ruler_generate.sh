#!/usr/bin/env bash
# Generate RULER synthetic data at long context lengths for the VECTOR long-context eval,
# using the OFFICIAL NVIDIA/RULER generator (the community-standard, tokenizer-specific path).
#
# Why we generate ourselves: RULER is a synthetic generator, not a fixed dataset. kvpress's
# ready-made simonjegou/ruler mirror only ships 4k/8k/16k; there is no universal pre-made 128k
# drop-in. We generate {16k,32k,64k,128k} with the Llama-3.1-8B tokenizer (same tokenizer the
# simonjegou mirror standardizes on), seed=42, then convert to kvpress's HF format with
# scripts/ruler_build_local.py (-> data/RULER/<length>/, loaded by evaluate.py via RULER_LOCAL_DIR).
#
# Prereqs (one-time, see RULER/README):
#   - RULER cloned to ./RULER; gen deps in .venv (wonderwords nltk bs4 tenacity
#     tiktoken html2text + nltk punkt); source data downloaded under
#     RULER/scripts/data/synthetic/json/ (PaulGrahamEssays.json, squad.json, hotpotqa.json).
#   - HF_TOKEN for the gated Llama-3.1 tokenizer.
#
# Output: RULER/scripts/data/data/<length>/<task>/validation.jsonl (gitignored).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export HF_TOKEN="${HF_TOKEN:-$(cat "${HOME}/.cache/huggingface/token" 2>/dev/null || true)}"

RULER_DATA="${REPO_ROOT}/RULER/scripts/data"
TOKENIZER="meta-llama/Meta-Llama-3.1-8B"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
SEQ_LENGTHS=(${SEQ_LENGTHS:-16384 32768 65536 131072})
TASKS=(
    niah_single_1 niah_single_2 niah_single_3
    niah_multikey_1 niah_multikey_2 niah_multikey_3
    niah_multivalue niah_multiquery
    vt cwe fwe qa_1 qa_2
)

cd "${RULER_DATA}"
for LEN in "${SEQ_LENGTHS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        OUT="${RULER_DATA}/data/${LEN}/${TASK}/validation.jsonl"
        if [[ -f "${OUT}" ]] && [[ "$(wc -l < "${OUT}")" -ge "${NUM_SAMPLES}" ]]; then
            echo "skip (done): ${LEN}/${TASK} ($(wc -l < "${OUT}") lines)"
            continue
        fi
        echo "=== generate ${LEN} / ${TASK} (n=${NUM_SAMPLES}) ==="
        python prepare.py \
            --save_dir "${RULER_DATA}/data/${LEN}" \
            --benchmark synthetic --task "${TASK}" \
            --tokenizer_path "${TOKENIZER}" --tokenizer_type hf \
            --max_seq_length "${LEN}" --model_template_type base \
            --num_samples "${NUM_SAMPLES}" --random_seed 42
    done
done
echo "=== RULER generation done. Raw data under ${RULER_DATA}/data/<length>/ ==="

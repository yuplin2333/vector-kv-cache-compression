# MIT License

# VECTOR evaluation add-on: kvpress-exact LongBench prompts for RocketKV.
# Minimal, dependency-light prompt builder that reproduces kvpress's LongBench input
# construction BYTE-EXACTLY. Imported by pipeline/inf_stream_llm/infllm_utils.py (RocketKV
# runs) and by scripts/rocketkv_protocol_check.py (token-level verification vs kvpress's
# real pipeline.preprocess). No relative imports: passes only (tokenizer, row) so it is
# venv-independent (used in both the main .venv and .venv-rocketkv for drift checking).
#
# Byte-exactness contract (verified by scripts/rocketkv_protocol_check.py against the real
# kvpress pipeline.preprocess on all 16 LongBench tasks):
#  1. context/question/answer_prefix come from the SAME dataset kvpress uses
#     (Xnhyacinth/LongBench, built by kvpress/evaluation/benchmarks/longbench/
#     create_huggingface_dataset.py) — no re-formatting from raw official jsonl.
#  2. The separator trick: whole context inside one user turn; apply_chat_template
#     (add_generation_prompt=True); split on the separator; question + assistant-header
#     suffix + answer_prefix encoded together (add_special_tokens=False).
#  3. Only the context is truncated (to max_context_length); the question is kept intact.
#  4. enable_thinking=False (matches kvpress pipeline default; required for Qwen3).
import os

import torch
from datasets import load_dataset

DATASET_NAME = "Xnhyacinth/LongBench"  # the exact dataset kvpress's evaluate.py loads


def load_longbench_rows(dataset: str, test_fraction: float = 1.0, seed: int = 42):
    """Load the same row set the kvpress harness uses (same sampling semantics).

    Mirrors evaluate.py exactly: pandas df.sample(frac, random_state=seed) when
    fraction < 1, else the untouched dataset order.
    """
    data = load_dataset(DATASET_NAME, data_dir=dataset, split="test")
    df = data.to_pandas()
    if test_fraction < 1.0:
        df = df.sample(frac=test_fraction, random_state=seed)
    return list(df.itertuples(index=True)) if test_fraction < 1.0 else list(data)


def build_input_ids(tokenizer, context: str, question: str, answer_prefix: str, max_context_length: int):
    """Token ids of EXACTLY what kvpress feeds its model for one (context, question) pair."""
    sep = "#" * (len(context) + 10)
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": context + sep}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    ctx_str, q_suffix = templated.split(sep)
    ctx_ids = tokenizer.encode(ctx_str, add_special_tokens=False)
    q_ids = tokenizer.encode(question + q_suffix + answer_prefix, add_special_tokens=False)
    if len(ctx_ids) > max_context_length:
        ctx_ids = ctx_ids[:max_context_length]
    return ctx_ids + q_ids


def build_row_input_ids(tokenizer, row, max_context_length: int, device: str = "cpu"):
    ids = build_input_ids(tokenizer, row["context"], row["question"], row["answer_prefix"], max_context_length)
    return torch.tensor([ids], device=device)

#!/usr/bin/env python3
"""Run our faithful D2O reimplementation (src/vector/baselines/d2o_native.py) on LongBench,
using kvpress's EXACT prompt protocol and scorer, so results are directly comparable to the
VECTOR / kvpress main table. Architecture-general: Llama-3.1 (validation) and Qwen3.

Protocol (identical to D2O/.../run_pred_long_bench_sample.py, which was verified
token-ID-identical to kvpress's pipeline.preprocess):
  * prompt = context_prefix + question_template in the user turn (chat template,
    add_generation_prompt=True), answer_prefix AFTER the assistant gen-prompt; context
    truncated to --max-context tokens (question kept intact).
  * greedy generation (num_beams=1, do_sample=False), max_new_tokens = kvpress's
    DATA_NAME_TO_MAX_NEW_TOKENS[task].
  * scored with kvpress's own longbench scorer (official LongBench metric).

Budget: D2O keeps ~(1 - p_c), split evenly hh = recent = (1 - p_c)/2 (the official
experiments/d2o_longbench.sh mapping).

Output: per (task, ratio) run dir named like the competing baselines so scripts/competing_baselines_aggregate.py
picks it up unchanged:
  <output-dir>/longbench__<task>__<model_tag>__d2o__<ratio>__max_context<ctx>/metrics.json
plus predictions.jsonl in the same dir for inspection / resume.

Usage:
  # Validation on Llama-3.1 (compare to official D2O: 2wikimqa 52.78/51.34, hotpotqa 58.67/58.80)
  CUDA_VISIBLE_DEVICES=0 python scripts/d2o_native_longbench.py \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --tasks 2wikimqa hotpotqa --ratios 0.25 0.90 \
      --output-dir results/d2o_native_val
  # Production on Qwen3
  CUDA_VISIBLE_DEVICES=0 python scripts/d2o_native_longbench.py \
      --model Qwen/Qwen3-14B --ratios 0.25 0.50 0.75 0.90 \
      --output-dir results/competing_baselines
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from vector.baselines.d2o_native import patch_model_for_d2o, reset_d2o  # noqa: E402

# kvpress LongBench scorer + prompt tables (single source of truth; same files the VECTOR
# main table uses). kvpress is a gitignored runtime checkout, like evaluate.py.
KV_LB_DIR = REPO_ROOT / "kvpress/evaluation/benchmarks/longbench"
sys.path.insert(0, str(REPO_ROOT / "kvpress/evaluation"))
from benchmarks.longbench.calculate_metrics import scorer  # noqa: E402  # type: ignore[import]

ALL_TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]  # fmt: skip


def _load_kv_prompt_tables() -> tuple[dict, dict, dict, dict]:
    """context_prefix / question_template / answer_prefix / DATA_NAME_TO_MAX_NEW_TOKENS,
    loaded by exec-ing the kvpress source up to the '# Longbench' marker (exactly as the D2O
    harness does), so the prompt is byte-identical to the VECTOR pipeline."""
    src = (KV_LB_DIR / "create_huggingface_dataset.py").read_text()
    ns: dict = {}
    exec(src[: src.index("# Longbench")], ns)
    return (
        ns["context_prefix"],
        ns["question_template"],
        ns["answer_prefix"],
        ns["DATA_NAME_TO_MAX_NEW_TOKENS"],
    )


def _build_input_ids(
    tokenizer, json_obj, task, ctx_prefix, q_template, ans_prefix, max_ctx
):
    """kvpress-exact input construction (context+question in the user turn, answer_prefix
    after the assistant gen-prompt; only the context is truncated to max_ctx tokens)."""
    context = ctx_prefix[task].format(**json_obj)
    inp = (
        json_obj["input"].removesuffix("Type:") if task == "trec" else json_obj["input"]
    )
    question = q_template[task].format(input=inp)
    sep = "#" * (len(context) + 10)
    # enable_thinking=False MUST match kvpress's pipeline (pipeline.py:149) -- the VECTOR main
    # table runs Qwen3 with thinking DISABLED. Omitting it makes Qwen3 default to thinking mode
    # (emits <think>...</think>, burns the max_new_tokens budget) -> unfair + tanks scores.
    # Ignored by non-thinking tokenizers (Llama-3.1), so the Llama validation is unaffected.
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": context + sep}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    ctx_str, q_suffix = templated.split(sep)
    ctx_ids = tokenizer.encode(ctx_str, add_special_tokens=False)
    q_ids = tokenizer.encode(
        question + q_suffix + ans_prefix[task], add_special_tokens=False
    )
    if len(ctx_ids) > max_ctx:
        ctx_ids = ctx_ids[:max_ctx]
    return ctx_ids + q_ids


def run_task_ratio(model, tokenizer, task, ratio, tables, max_ctx, n_samples, run_dir):
    ctx_prefix, q_template, ans_prefix, max_new = tables
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        logger.info(
            f"skip (done): {task} p_c={ratio} -> {json.loads(metrics_file.read_text())}"
        )
        return json.loads(metrics_file.read_text())

    data = load_dataset("THUDM/LongBench", task, split="test")
    if n_samples:
        data = data.select(range(min(n_samples, len(data))))
    max_gen = max_new[task]
    device = model.device

    patch_model_for_d2o(model, compression_ratio=ratio)
    preds, answers_all, all_classes = [], [], data[0]["all_classes"]
    for json_obj in data:
        input_ids = torch.tensor(
            [
                _build_input_ids(
                    tokenizer,
                    json_obj,
                    task,
                    ctx_prefix,
                    q_template,
                    ans_prefix,
                    max_ctx,
                )
            ],
            device=device,
        )
        context_length = input_ids.shape[-1]
        reset_d2o(model)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )[0]
        pred = tokenizer.decode(out[context_length:], skip_special_tokens=True)
        preds.append(pred)
        answers_all.append(json_obj["answers"])  # type: ignore[index]

    score = scorer(task, preds, answers_all, all_classes)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(score))
    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for p, a in zip(preds, answers_all):
            f.write(json.dumps({"pred": p, "answers": a}, ensure_ascii=False) + "\n")
    logger.success(f"{task} p_c={ratio}: score={score}  (n={len(preds)})")
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.25, 0.50, 0.75, 0.90])
    ap.add_argument("--tasks", nargs="+", default=ALL_TASKS)
    ap.add_argument("--max-context", type=int, default=16000)
    ap.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="cap per task (smoke); full split if unset",
    )
    ap.add_argument("--output-dir", default="results/competing_baselines")
    args = ap.parse_args()

    tables = _load_kv_prompt_tables()
    model_tag = args.model.replace("/", "--")
    out_root = (
        REPO_ROOT / args.output_dir
        if not os.path.isabs(args.output_dir)
        else Path(args.output_dir)
    )

    logger.info(
        f"loading {args.model} (eager attention for D2O full-attention scoring)"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda",
    )
    model.eval()

    summary: dict = {}
    for ratio in args.ratios:
        rstr = f"{ratio:.2f}"  # 0.25 / 0.50 / 0.75 / 0.90 -- matches competing_baselines_aggregate.py
        for task in args.tasks:
            run_dir = (
                out_root
                / f"longbench__{task}__{model_tag}__d2o__{rstr}__max_context{args.max_context}"
            )
            score = run_task_ratio(
                model,
                tokenizer,
                task,
                ratio,
                tables,
                args.max_context,
                args.n_samples,
                run_dir,
            )
            summary[f"{task}@{rstr}"] = score
            logger.info(f"running summary: {json.dumps(summary)}")

    logger.success(f"DONE. summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()

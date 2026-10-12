"""SCBench multi-turn evaluation of kvpress presses (incl. the VECTOR presses) on a shared context.

Protocol:
  * Prompts: SCBench's own multi-turn construction (`create_multiturn_prompt`, golden-context
    mode, Llama chat template) from the SCBench code vendored in RocketKV/eval/scbench_utils.
  * Turn 1: the shared context is prefilled ONCE under the press (compression happens here, as in
    kvpress's LongBench evaluation: the press sees the context, not the question). The first question is then fed on
    top of the compressed cache.
  * Turns 2..T: the follow-up prompt (golden previous answer + next question) is appended to the
    SAME compressed cache — nothing is recompressed (kvpress hooks skip non-prefill forwards).
  * Each turn's generated tokens are removed from the cache before the next turn (SCBench golden-
    context mode: the golden answer, not the model's, is what stays in the history), mirroring
    kvpress's `_remove_answer_from_cache`.
  * Positions: the compressed cache is physically shorter than the logical sequence, so every forward
    after compression passes explicit `position_ids` continuing from the logical length (as kvpress's
    `generate_answer` does).
  * Truncation: SCBench's budget `max_input_length = max_seq_len - max_new_tokens * n_turns` (for mixed
    tasks, sum(max_new_tokens) * n_turns // 2), applied as middle truncation of the context tokens.

Predictions are written in SCBench's jsonl format and scored separately with SCBench's
`compute_scores` (scripts/scbench_score.py, run in RocketKV/.venv-rocketkv-457).

Usage:
    python scripts/scbench_multiturn.py --task scbench_kv --press_name snapkv_vector \
        --compression_ratio 0.9 --vector_weights weights/llama31_8b_probe_konly.pt \
        --output_dir results/scbench
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import torch
import transformers
from datasets import load_dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "kvpress/evaluation"))
sys.path.insert(0, str(REPO_ROOT / "RocketKV"))
# SCBench's eval_utils imports transformers.SinkCache (removed in transformers>=4.57); it is only
# referenced by cache types we do not use, so a placeholder is enough for the import to succeed
# (transformers' lazy module still lists the name, so `from transformers import SinkCache` fails).
transformers.SinkCache = None  # type: ignore[attr-defined]
from eval.scbench_utils.eval_utils import (  # noqa: E402  # type: ignore[import]
    DATA_NAME_TO_MAX_NEW_TOKENS,
    create_multiturn_prompt,
    get_ground_truth,
)
from evaluate_registry import PRESS_REGISTRY  # noqa: E402  # type: ignore[import]

CTX_SENTINEL = "@@SCBENCH_SHARED_CONTEXT@@"


def context_key(eg: dict) -> str:
    # scbench_vt stores its shared context under "input"; scbench_mf's context is a list of ints
    # that SCBench's template str()-formats into the prompt.
    keys = [
        k for k in ("context", "input") if k in eg and isinstance(eg[k], (str, list))
    ]
    assert len(keys) >= 1, f"no context field in example keys {list(eg)}"
    return keys[0]


def build_turns(
    eg: dict, task: str, tok, max_input_length: int
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return (context_ids, [q1_ids, followup_2_ids, ...]) with the context middle-truncated."""
    ck = context_key(eg)
    probe = dict(eg)
    probe[ck] = CTX_SENTINEL
    enc = create_multiturn_prompt(
        probe, data_name=task, tok=tok, use_chat_template=True, use_vllm=True
    )
    first = enc["prompts"][0]
    assert isinstance(first, str) and first.count(CTX_SENTINEL) == 1, (
        "context sentinel not found exactly once"
    )
    head, tail = first.split(CTX_SENTINEL)
    for p in enc["prompts"][1:]:
        assert CTX_SENTINEL not in p, "context leaked into a follow-up prompt"

    # Take the context exactly as SCBench's template inserts it (some templates strip() it, and
    # scbench_mf str()-formats a list), so head + context + tail == SCBench's real first prompt.
    real = create_multiturn_prompt(
        eg, data_name=task, tok=tok, use_chat_template=True, use_vllm=True
    )["prompts"][0]
    assert real.startswith(head) and real.endswith(tail), (
        "SCBench prompt does not wrap the context as expected"
    )
    ctx_text = real[len(head) : len(real) - len(tail)]
    raw = eg[ck] if isinstance(eg[ck], str) else str(eg[ck])
    assert ctx_text.strip() == raw.strip(), (
        "context inserted by SCBench differs beyond whitespace"
    )
    head_ids = tok.encode(head, add_special_tokens=False)
    ctx_ids = tok.encode(ctx_text, add_special_tokens=False)
    tail_ids = tok.encode(tail, add_special_tokens=False)
    budget = max_input_length - len(head_ids) - len(tail_ids)
    assert budget > 0, f"no room for context: max_input_length={max_input_length}"
    if len(ctx_ids) > budget:  # SCBench truncate_input(manner="middle")
        half = budget // 2
        ctx_ids = ctx_ids[:half] + ctx_ids[-half:]

    context_ids = torch.tensor([head_ids + ctx_ids])
    turn_ids = [torch.tensor([tail_ids])] + [
        torch.tensor([tok.encode(p, add_special_tokens=False)])
        for p in enc["prompts"][1:]
    ]
    return context_ids, turn_ids


def remove_from_cache(cache: DynamicCache, lengths: list[int]) -> None:
    for layer_idx, n in enumerate(lengths):
        layer = cache.layers[layer_idx]
        assert layer.keys is not None and layer.values is not None, (
            f"layer {layer_idx} cache is empty"
        )
        layer.keys = layer.keys[:, :, :n]
        layer.values = layer.values[:, :, :n]


@torch.inference_mode()
def run_session(
    model,
    tok,
    press,
    context_ids,
    turn_ids,
    max_new: list[int],
    stop_ids: set[int],
    query_aware: bool,
) -> list[str]:
    """query_aware=False: compress the shared context only; question 1 is fed afterwards.
    query_aware=True (SnapKV/PyramidKV family, as kvpress's --query_aware): question 1 is appended to
    the context and compressed with it; later questions are still unseen at compression time."""
    device = model.device
    cache = DynamicCache()
    first = torch.cat([context_ids, turn_ids[0]], dim=1) if query_aware else context_ids
    with press(model) if press is not None else contextlib.nullcontext():
        first_out = model(
            input_ids=first.to(device), past_key_values=cache, logits_to_keep=1
        )
    logical_len = first.shape[1]  # positions continue from the uncompressed length

    answers = []
    for turn, (ids, n_new) in enumerate(zip(turn_ids, max_new)):
        if turn == 0 and query_aware:
            out = (
                first_out  # question 1 was already prefilled together with the context
            )
        else:
            ids = ids.to(device)
            pos = torch.arange(
                logical_len, logical_len + ids.shape[1], device=device
            ).unsqueeze(0)
            out = model(
                input_ids=ids, past_key_values=cache, position_ids=pos, logits_to_keep=1
            )
            logical_len += ids.shape[1]
        # The turn prompt stays in the history; only the generated answer is removed afterwards.
        kept = [cache.get_seq_length(i) for i in range(len(cache))]

        generated: list[int] = []
        nxt = int(out.logits[0, -1].argmax())
        step_pos = logical_len
        for _ in range(n_new):
            if nxt in stop_ids:
                break
            generated.append(nxt)
            out = model(
                input_ids=torch.tensor([[nxt]], device=device),
                past_key_values=cache,
                position_ids=torch.tensor([[step_pos]], device=device),
            )
            step_pos += 1
            nxt = int(out.logits[0, -1].argmax())
        remove_from_cache(cache, kept)
        answers.append(tok.decode(generated))
    return answers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument(
        "--press_name", required=True, help="kvpress registry name, or no_press"
    )
    ap.add_argument("--compression_ratio", type=float, required=True)
    ap.add_argument("--vector_weights", default=None)
    ap.add_argument("--max_seq_len", type=int, default=131072)
    ap.add_argument(
        "--query_aware",
        action="store_true",
        help="compress question 1 together with the context (query-aware presses: SnapKV/PyramidKV)",
    )
    ap.add_argument(
        "--n_sessions",
        type=int,
        default=None,
        help="evaluate only the first N sessions",
    )
    ap.add_argument("--output_dir", type=Path, required=True)
    args = ap.parse_args()

    press = None
    if args.press_name != "no_press":
        assert 0.0 < args.compression_ratio < 1.0, (
            f"bad compression_ratio {args.compression_ratio}"
        )
        press = PRESS_REGISTRY[args.press_name]
        if hasattr(press, "weights_path"):
            assert args.vector_weights is not None, (
                f"{args.press_name} requires --vector_weights"
            )
            press.weights_path = args.vector_weights
        assert hasattr(press, "compression_ratio"), (
            f"{type(press).__name__} has no compression_ratio"
        )
        press.compression_ratio = args.compression_ratio
    else:
        assert args.compression_ratio == 0.0, "no_press must use --compression_ratio 0"

    tag = f"{args.task}__{args.press_name}__{args.compression_ratio:.2f}"
    if args.query_aware:
        tag += "__query_aware"
    out_dir = args.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "predictions.jsonl"

    data = load_dataset("microsoft/SCBench", args.task, split="test")
    if args.n_sessions is not None:
        data = data.select(range(min(args.n_sessions, len(data))))
    n_turns = len(data[0]["multi_turns"])
    mnt = DATA_NAME_TO_MAX_NEW_TOKENS[args.task]
    if isinstance(mnt, dict):  # mixed-task sessions (SCBench budget rule for these)
        max_input_length = args.max_seq_len - sum(mnt.values()) * n_turns // 2
    else:
        max_input_length = args.max_seq_len - mnt * n_turns

    done = set()
    if pred_file.exists():  # resume at session granularity
        rows = [
            json.loads(line)
            for line in pred_file.read_text().splitlines()
            if line.strip()
        ]
        done = {r["id"] for r in rows}
        logger.info(f"resuming: {len(done)} sessions already in {pred_file}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        # sdpa = the attention backend used for the kvpress LongBench / RULER runs
        args.model,
        dtype="auto",
        attn_implementation="sdpa",
        device_map="cuda",
    ).eval()
    assert model.generation_config is not None, (
        "model has no generation_config (needed for stop tokens)"
    )
    eos = model.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos]) | {tok.eos_token_id}

    for i, eg in enumerate(data):
        if i in done:
            continue
        eg = dict(eg)
        context_ids, turn_ids = build_turns(eg, args.task, tok, max_input_length)
        assert len(turn_ids) == len(eg["multi_turns"]), "turn count mismatch"
        if isinstance(mnt, dict):
            max_new = [mnt[t["task"]] for t in eg["multi_turns"]]
        else:
            max_new = [mnt] * len(turn_ids)
        answers = run_session(
            model,
            tok,
            press,
            context_ids,
            turn_ids,
            max_new,
            stop_ids,
            args.query_aware,
        )

        gts = get_ground_truth(eg, args.task)
        with pred_file.open("a") as f:
            for turn_idx, (ans, gt, turn) in enumerate(
                zip(answers, gts, eg["multi_turns"])
            ):
                case = {
                    "id": i,
                    "turn_idx": turn_idx,
                    "prediction": ans,
                    "ground_truth": gt,
                }
                if "task" in turn:
                    case["task"] = turn["task"]
                if "lang" in eg:
                    case["lang"], case["repo"] = eg["lang"], eg["repo"]
                    if (
                        turn.get("task", "scbench_repoqa") == "scbench_repoqa"
                        and "name" in turn
                    ):
                        case["func_name"] = turn["name"]
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
        logger.info(
            f"[{tag}] session {i + 1}/{len(data)} ctx={context_ids.shape[1]} turn1={answers[0][:60]!r}"
        )

    (out_dir / "DONE").write_text(f"{len(data)} sessions\n")
    logger.info(f"[{tag}] done -> {pred_file}")


if __name__ == "__main__":
    main()

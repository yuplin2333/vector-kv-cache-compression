# pyright: reportMissingImports=false
"""Token-level protocol check: RocketKV's replicated prompt builder vs kvpress's REAL
pipeline.preprocess, across ALL 16 LongBench tasks.
(NOTE: imports resolve into the ./kvpress and ./RocketKV checkouts; pyright suppression intended.)

Runs in any venv. Two modes:
  python scripts/rocketkv_protocol_check.py --dump <out.json>
      computes, for 3 rows of every task: (a) ids from kvpress's real
      KVPressTextGenerationPipeline.preprocess (method invoked unbound on a tokenizer
      stub — the model is NOT needed), (b) ids from RocketKV's
      pipeline/kvpress_prompt.build_input_ids; asserts exact equality and records sha256
      of the combined input ids for cross-venv drift checks.
  python scripts/rocketkv_protocol_check.py --check <out.json>
      recomputes (b) (sha only) in the CURRENT venv and compares against the dump ->
      proves tokenizer drift does not change the model-visible input across venvs
      (main .venv vs .venv-rocketkv).
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RK_DIR = REPO_ROOT / "RocketKV"
KV_EVAL_DIR = REPO_ROOT / "kvpress" / "evaluation"
TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]
N_ROWS = 3
MODEL_NAME = (
    "meta-llama/Llama-3.1-8B-Instruct"  # tokenizer-arch tag only (no model load)
)
MAX_CONTEXT_LENGTH = (
    127500  # RocketKV pipeline_config model_max_len (never truncates LongBench)
)
KVPROTO_DIR = os.environ.get("LB_KVPROTO_DIR")  # optional: pre-dumped per-task JSONLs


def _load_rows(task):
    if KVPROTO_DIR:
        with open(os.path.join(KVPROTO_DIR, f"{task}.jsonl"), encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    from datasets import load_dataset

    return list(load_dataset("Xnhyacinth/LongBench", data_dir=task, split="test"))


def get_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tok


def kvpress_real_ids(tok, row):
    """Invoke the REAL kvpress preprocess (unbound) on a stub pipeline object."""
    sys.path.insert(0, str(KV_EVAL_DIR))
    from kvpress.pipeline import KVPressTextGenerationPipeline as P

    class _Stub:  # preprocess() only touches self.tokenizer
        tokenizer = tok

    out = P.preprocess(
        _Stub(),  # type: ignore[arg-type]
        row["context"],
        [row["question"]],
        row["answer_prefix"],
        MAX_CONTEXT_LENGTH,
    )
    ids = out["context_ids"][0].tolist() + out["questions_ids"][0][0].tolist()
    return ids


def rocket_builder_ids(tok, row):
    sys.path.insert(0, str(RK_DIR))
    from pipeline.kvpress_prompt import build_input_ids

    return build_input_ids(
        tok, row["context"], row["question"], row["answer_prefix"], MAX_CONTEXT_LENGTH
    )


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--dump", type=str)
    group.add_argument("--check", type=str)
    args = ap.parse_args()

    tok = get_tokenizer()
    results = {}
    all_ok = True
    for task in TASKS:
        rows = _load_rows(task)[:N_ROWS]
        for i, row in enumerate(rows):
            combined = rocket_builder_ids(tok, row)
            sha = hashlib.sha256(json.dumps(combined).encode()).hexdigest()
            entry = {
                "len": len(combined),
                "sha": sha,
                "n_questions": row["answer_prefix"] != "",
            }
            if args.dump:
                real = kvpress_real_ids(tok, row)
                entry["exact_match"] = real == combined
                if real != combined:
                    all_ok = False
            results[f"{task}__{i}"] = entry

    if args.check:
        dump = json.load(open(args.check))
        mismatch = [k for k, v in results.items() if dump[k]["sha"] != v["sha"]]
        if mismatch:
            print(
                f"CHECK FAILED: {len(mismatch)} sha mismatches across venvs: {mismatch[:5]}"
            )
            sys.exit(1)
        print(f"CHECK PASS: {len(results)} rows, token ids identical across venvs.")
        return

    if not all_ok:
        bad = [k for k, v in results.items() if not v.get("exact_match")]
        print(f"MISMATCH vs real kvpress preprocess: {bad}")
        sys.exit(1)
    print(
        f"EXACT MATCH on all {len(results)} (task,row) pairs (builder == real kvpress preprocess)."
    )
    json.dump(results, open(args.dump, "w"), indent=1)
    print(f"dumped to {args.dump}")


if __name__ == "__main__":
    main()

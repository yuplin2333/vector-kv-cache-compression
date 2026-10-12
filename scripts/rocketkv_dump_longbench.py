"""Dump the exact LongBench rows kvpress evaluates (Xnhyacinth/LongBench) to per-task JSONL files.

RocketKV's venv (datasets 2.x) cannot read the hub-cache schema of Xnhyacinth/LongBench, so the
rows are written once from the main venv and read by the patched RocketKV loader through
LB_KVPROTO_DIR (patches/RocketKV/eval/longbench_utils/eval_long_bench.py). Same rows, same order.

Usage:
    python scripts/rocketkv_dump_longbench.py RocketKV/dataset/longbench_kvproto
"""

import json
import sys
from pathlib import Path

from datasets import load_dataset

TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]  # fmt: skip


def main() -> None:
    assert len(sys.argv) == 2, __doc__
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        rows = load_dataset("Xnhyacinth/LongBench", data_dir=task, split="test")
        with open(out / f"{task}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{task}: {len(rows)} rows")


if __name__ == "__main__":
    main()

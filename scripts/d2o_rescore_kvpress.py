"""Rescore the official-repo Llama-3.1 D2O LongBench predictions with kvpress's scorer.

The official D2O scorer (D2O/LLM_merge_new/eval_long_bench.py, i.e. THUDM/LongBench)
takes the first line of trec/triviaqa/samsum via prediction.lstrip('\\n').split('\\n')[0], which
zeroes Llama outputs that start with " \\n". kvpress's scorer (behind every other LongBench row, and the
Qwen3 D2O rows via scripts/d2o_native_longbench.py) uses lstrip(). Inputs were already
byte-exact to kvpress; this makes the scorer identical too.

Usage:
    python scripts/d2o_rescore_kvpress.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "kvpress/evaluation/benchmarks/longbench"))
from calculate_metrics import scorer  # noqa: E402  # type: ignore[import]

TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]  # fmt: skip
PRED_ROOT = REPO_ROOT / "D2O/LLM_merge_new/pred"


def main() -> None:
    print(
        "| p_c | official-scorer Avg16 | kvpress-scorer Avg16 | changed tasks (official → kvpress) |"
    )
    print("|---|---|---|---|")
    for pc in ["0.25", "0.50", "0.75", "0.90"]:
        d = PRED_ROOT / f"Llama-3.1-8B-Instruct_d2o_kvproto_pc{pc}"
        official = json.loads((d / "result1.json").read_text())
        new = {}
        for t in TASKS:
            rows = [
                json.loads(line)
                for line in (d / f"{t}.jsonl").read_text().splitlines()
                if line.strip()
            ]
            assert len(rows) > 0, f"no predictions in {d / t}.jsonl"
            new[t] = scorer(
                t,
                [r["pred"] for r in rows],
                [r["answers"] for r in rows],
                rows[0]["all_classes"],
            )
        changed = ", ".join(
            f"{t} {official[t]}→{new[t]}"
            for t in TASKS
            if abs(official[t] - new[t]) > 1e-9
        )
        avg_off = sum(official[t] for t in TASKS) / len(TASKS)
        avg_new = sum(new.values()) / len(TASKS)
        print(f"| {pc} | {avg_off:.2f} | {avg_new:.2f} | {changed} |")


if __name__ == "__main__":
    main()

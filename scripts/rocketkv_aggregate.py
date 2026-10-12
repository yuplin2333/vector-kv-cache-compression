"""Aggregate RocketKV LongBench results (byte-exact kvpress protocol) into a per-task table.

Layout produced by experiments/rocketkv_{longbench,baseline}.sh:
    <root>/<p_c>/<task>/pred/rocket/result.json      (compressed cells)
    <baseline_root>/<task>/pred/baseline/result.json (full-KV reference)

Scores are recomputed from the stored predictions with kvpress's LongBench scorer (the one behind
every other row of the LongBench comparison table), NOT read from RocketKV's result.json: the official LongBench scorer that
RocketKV ships takes the first line of trec/triviaqa/samsum via lstrip('\n'), which zeroes Llama
outputs starting with " \n"; kvpress uses lstrip(). Byte-exact inputs + one scorer = one protocol.

Avg16 is reported only when all 16 tasks are present (a partial mean is never printed as Avg16).

Usage:
    python scripts/rocketkv_aggregate.py --root results/rocketkv_longbench \
        --baseline-root results/rocketkv_longbench/baseline
"""

import argparse
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
PCS = ["0.25", "0.50", "0.75", "0.90"]


def read_score(pred_dir: Path, task: str) -> float | None:
    if not (
        pred_dir / "result.json"
    ).exists():  # written only after all predictions are done
        return None
    preds = sorted(pred_dir.glob(f"{task}_*.jsonl"))
    assert len(preds) == 1, (
        f"expected exactly one prediction file in {pred_dir}, got {preds}"
    )
    rows = [
        json.loads(line) for line in preds[0].read_text().splitlines() if line.strip()
    ]
    return scorer(
        task,
        [r["pred"] for r in rows],
        [r["answers"] for r in rows],
        rows[0]["all_classes"],
    )


def row(scores: dict[str, float | None]) -> tuple[list[str], str]:
    cells = [f"{s:.2f}" if s is not None else "—" for s in (scores[t] for t in TASKS)]
    done = [s for s in scores.values() if s is not None]
    avg = (
        f"{sum(done) / len(TASKS):.2f}"
        if len(done) == len(TASKS)
        else f"({len(done)}/16)"
    )
    return cells, avg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--baseline-root", type=Path, required=True)
    args = ap.parse_args()

    rows = {
        "Full-KV": {
            t: read_score(args.baseline_root / t / "pred/baseline", t) for t in TASKS
        }
    }
    for pc in PCS:
        rows[f"p_c={pc}"] = {
            t: read_score(args.root / pc / t / "pred/rocket", t) for t in TASKS
        }

    print("| RocketKV | " + " | ".join(TASKS) + " | Avg16 |")
    print("|---" * (len(TASKS) + 2) + "|")
    for name, scores in rows.items():
        cells, avg = row(scores)
        print(f"| {name} | " + " | ".join(cells) + f" | {avg} |")


if __name__ == "__main__":
    main()

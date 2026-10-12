#!/usr/bin/env python3
"""Aggregate competing-baseline LongBench scores into a per-(method, ratio) table.

Each run writes a bare float to
  results/competing_baselines/longbench__<task>__<model>__<method>__<ratio>[/<n>]/metrics.json

We reduce these to the paper's `Avg` metric: the arithmetic mean over the 16
LongBench tasks, per (method, compression_ratio). Duplicate runs of the same
(task, method, ratio) — e.g. from an interrupted parallel launch that produced
numbered subdirs — are averaged. A (method, ratio) cell is only reported as a
final Avg when all 16 tasks are present; otherwise its partial count is shown.

Usage:
    python scripts/competing_baselines_aggregate.py [--dir results/competing_baselines]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# The 16 LongBench-v1 English tasks used in the paper (Chinese tasks excluded).
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
RATIOS = ["0.25", "0.50", "0.75", "0.90"]

# longbench__<task>__<model>__<method>__<ratio>[__<suffix><num>]*   (model may contain "--")
# Trailing suffixes like __fraction1.000 / __max_context16000 (e.g. the D2O press runs,
# which cap context at 16k) are tolerated and ignored for grouping; a final __query_aware
# (evaluate.py --query_aware) marks the method as "<method>[qa]".
DIR_RE = re.compile(
    r"^longbench__(?P<task>.+?)__(?P<model>.+)__(?P<method>[^_]+(?:_[^_]+)*)__(?P<ratio>[0-9.]+)"
    r"(?:__[a-zA-Z_]+[0-9.]+)*(?P<qa>__query_aware)?$"
)


def parse_run_dir(name: str) -> tuple[str, str, str, str] | None:
    """Return (model, task, method, ratio) from a run-dir name, or None if no match."""
    m = DIR_RE.match(name)
    if not m:
        return None
    # Query-aware runs are reported as a separate method row, e.g. "chunkkv[qa]".
    method = m.group("method") + ("[qa]" if m.group("qa") else "")
    return m.group("model"), m.group("task"), method, m.group("ratio")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/competing_baselines")
    ap.add_argument(
        "--model",
        default=None,
        help="filter to a model by its dir tag substring (e.g. Qwen3-14B); "
        "default reports every model found",
    )
    ap.add_argument(
        "--per-task",
        action="store_true",
        help="also print the full per-task table (16 tasks + Avg) per model",
    )
    args = ap.parse_args()

    root = Path(args.dir)
    assert root.is_dir(), f"missing results dir: {root}"

    # (model, method, ratio, task) -> list of scores (list handles duplicate runs)
    scores: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for mfile in root.rglob("metrics.json"):
        # The run dir is the first ancestor whose name matches DIR_RE (metrics.json
        # may sit under an extra numbered subdir like .../0.50/1/metrics.json).
        parsed = None
        for anc in mfile.parents:
            parsed = parse_run_dir(anc.name)
            if parsed is not None:
                break
        if parsed is None:
            continue
        model, task, method, ratio = parsed
        if args.model is not None and args.model not in model:
            continue
        val = json.loads(mfile.read_text())
        assert isinstance(val, (int, float)), f"non-scalar metric in {mfile}: {val!r}"
        scores[(model, method, ratio, task)].append(float(val))

    # Per-task mean (average any duplicate runs).
    task_mean: dict[tuple[str, str, str, str], float] = {
        k: sum(v) / len(v) for k, v in scores.items()
    }

    models = sorted({k[0] for k in scores})
    for model in models:
        methods = sorted({k[1] for k in scores if k[0] == model})
        print(f"\n########## model: {model} ##########")
        print("\n=== Avg16 matrix (complete cells only; N/16 if partial) ===")
        print(f"{'method':>20} | " + " | ".join(f"{r:>6}" for r in RATIOS))
        for method in methods:
            cells = []
            for ratio in RATIOS:
                present = [t for t in TASKS if (model, method, ratio, t) in task_mean]
                if len(present) == 16:
                    avg = (
                        sum(task_mean[(model, method, ratio, t)] for t in present) / 16
                    )
                    cells.append(f"{avg:6.2f}")
                else:
                    cells.append(f"{len(present):>2}/16 ")
            print(f"{method:>20} | " + " | ".join(cells))

        if args.per_task:
            # Full per-task table: one row per (method, ratio), 16 task cols + Avg.
            short = [t[:8] for t in TASKS]  # abbreviated headers to keep width sane
            print("\n=== per-task (rows: method @ p_c; cols: 16 tasks + Avg) ===")
            print(
                f"{'method@p_c':>22} | "
                + " | ".join(f"{s:>8}" for s in short)
                + " | Avg"
            )
            for method in methods:
                for ratio in RATIOS:
                    vals = [task_mean.get((model, method, ratio, t)) for t in TASKS]
                    cells = [
                        f"{v:8.2f}" if v is not None else f"{'--':>8}" for v in vals
                    ]
                    got = [v for v in vals if v is not None]
                    avg = (
                        f"{sum(got) / len(got):.2f}"
                        if len(got) == 16
                        else f"{len(got)}/16"
                    )
                    print(
                        f"{method + '@' + ratio:>22} | "
                        + " | ".join(cells)
                        + f" | {avg}"
                    )


if __name__ == "__main__":
    main()

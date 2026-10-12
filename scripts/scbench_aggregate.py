"""Aggregate SCBench multi-turn scores (score.json written by scripts/scbench_score.py).

SCBench's multi-turn scorer returns one score per turn index (mixed-task summary_with_needles: a
(summary, passkey) pair per turn). A task's score is the mean over turns (as in the SCBench paper);
summary_with_needles is reported as two columns. A second table splits turn 1 from turns >= 2,
i.e. the first query vs. later queries answered from the SAME compressed cache.

Usage:
    python scripts/scbench_aggregate.py results/scbench
"""

import json
import sys
from pathlib import Path

TASKS = [
    "scbench_vt",
    "scbench_choice_eng",
    "scbench_many_shot",
    "scbench_summary_with_needles",
]
COLS = ["vt", "choice_eng", "many_shot", "sum(needles)", "passkey(needles)"]
BASES = ["snapkv", "keydiff", "pyramidkv"]
PCS = ["0.75", "0.90"]
# Query-aware methods are read from their __query_aware runs (question 1 compressed with the
# context, as in the paper protocol).
QUERY_AWARE = ("snapkv", "pyramidkv")
QA_SUFFIX = "__query_aware"


def per_turn(run_dir: Path) -> dict[str, list[float]]:
    """Column name -> per-turn scores for one run dir."""
    task = run_dir.name.split("__")[0]
    vals = [
        float(x) for x in json.loads((run_dir / "score.json").read_text()).split(",")
    ]
    if task == "scbench_summary_with_needles":
        assert len(vals) % 2 == 0, f"expected (summary, passkey) pairs in {run_dir}"
        return {"sum(needles)": vals[0::2], "passkey(needles)": vals[1::2]}
    return {COLS[TASKS.index(task)]: vals}


def load(root: Path, method: str, pc: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for t in TASKS:
        d = (
            root
            / f"{t}__{method}__{pc}{QA_SUFFIX if method.startswith(QUERY_AWARE) else ''}"
        )
        assert (d / "score.json").exists(), f"missing score for {d}"
        out.update(per_turn(d))
    return out


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def table(root: Path, reducer, title: str) -> None:
    print(f"\n### {title}\n")
    print("| p_c | method | " + " | ".join(COLS) + " | Avg |")
    print("|---" * (len(COLS) + 3) + "|")
    rows = [("—", "Full-KV", load(root, "no_press", "0.00"))]
    for pc in PCS:
        for b in BASES:
            rows.append((pc, b, load(root, b, pc)))
            rows.append((pc, f"{b}+VECTOR", load(root, f"{b}_vector", pc)))
    for pc, name, scores in rows:
        cells = [reducer(scores[c]) for c in COLS]
        print(
            f"| {pc} | {name} | "
            + " | ".join(f"{c:.2f}" for c in cells)
            + f" | {mean(cells):.2f} |"
        )


def main() -> None:
    assert len(sys.argv) == 2, __doc__
    root = Path(sys.argv[1])
    table(root, mean, "Mean over all turns")
    table(root, lambda xs: xs[0], "Turn 1 only")
    table(
        root,
        lambda xs: mean(xs[1:]),
        "Turns >= 2 (later queries on the same compressed cache)",
    )


if __name__ == "__main__":
    main()

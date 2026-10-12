"""Paired bootstrap CIs for SCBench multi-turn: base scorer vs. base + VECTOR.

Run with RocketKV/.venv-rocketkv-457/bin/python (SCBench's scorer needs evaluate/rouge).
Every prediction row is scored with SCBench's own get_score_one (the function behind its
compute_scores), checked to reproduce score.json's per-turn numbers. We resample SESSIONS with
replacement (turns of one session share a compressed cache, so they are not independent), apply the
same resample to both runs, and recompute the official aggregate: per-turn mean over sessions, then
mean over turns; Avg = mean of the 5 columns. Reports 95% CIs of Delta = (+VECTOR) - base.

Usage:
    RocketKV/.venv-rocketkv-457/bin/python scripts/bootstrap_ci_scbench.py results/scbench
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.remove(
    str(Path(__file__).resolve().parent)
)  # scripts/evaluate.py shadows HF `evaluate`
sys.path.insert(0, str(REPO_ROOT / "RocketKV"))
from eval.scbench_utils.compute_scores import (  # noqa: E402  # type: ignore[import]
    Multiturnbench_to_Infinitebench,
    get_score_one,
)

MODEL_NAME = "Llama-3.1-8B-Instruct"
TASKS = [
    "scbench_vt",
    "scbench_choice_eng",
    "scbench_many_shot",
    "scbench_summary_with_needles",
]
COLS = ["vt", "choice_eng", "many_shot", "sum(needles)", "passkey(needles)"]
COL_TASK = dict(zip(COLS, TASKS + ["scbench_summary_with_needles"]))
BASES = ["snapkv", "keydiff", "pyramidkv"]
PCS = ["0.75", "0.90"]
# Query-aware methods are read from their __query_aware runs (question 1 compressed with the
# context, as in the paper protocol).
QUERY_AWARE = ("snapkv", "pyramidkv")
QA_SUFFIX = "__query_aware"
N_BOOT = 10000


def column_of(task: str, row: dict) -> str:
    if task == "scbench_summary_with_needles":
        return {
            "scbench_summary": "sum(needles)",
            "scbench_passkey": "passkey(needles)",
        }[row["task"]]
    return COLS[TASKS.index(task)]


def load_cell(root: Path, method: str, pc: str) -> dict[str, np.ndarray]:
    """Column -> score matrix (n_sessions, n_turn_slots) in [0, 100], NaN where a session has no
    row for that (column, turn). Sessions are ordered by id so two runs align row for row."""
    out: dict[str, np.ndarray] = {}
    for task in TASKS:
        d = (
            root
            / f"{task}__{method}__{pc}{QA_SUFFIX if method.startswith(QUERY_AWARE) else ''}"
        )
        rows = [
            json.loads(line)
            for line in (d / "predictions.jsonl").read_text().splitlines()
            if line.strip()
        ]
        ids = sorted({r["id"] for r in rows})
        n_turns = max(r["turn_idx"] for r in rows) + 1
        mats = {
            c: np.full((len(ids), n_turns), np.nan)
            for c in {column_of(task, r) for r in rows}
        }
        pos = {sid: k for k, sid in enumerate(ids)}
        for r in rows:
            sub = r.get("task", task)
            name = Multiturnbench_to_Infinitebench.get(sub, sub)
            mats[column_of(task, r)][pos[r["id"]], r["turn_idx"]] = 100 * get_score_one(
                r["prediction"], r["ground_truth"], name, MODEL_NAME
            )
        # Hard check against the official per-turn numbers in score.json.
        official = [
            float(x) for x in json.loads((d / "score.json").read_text()).split(",")
        ]
        ours = (
            [
                v
                for t in range(n_turns)
                for c in ("sum(needles)", "passkey(needles)")
                for v in [np.nanmean(mats[c][:, t])]
            ]
            if task == "scbench_summary_with_needles"
            else list(np.nanmean(mats[COLS[TASKS.index(task)]], axis=0))
        )
        assert np.allclose(ours, official, atol=0.0011), (
            f"{d.name}: per-turn scores differ from score.json"
        )
        out.update(mats)
    return out


def aggregate(
    mats: dict[str, np.ndarray], rows: dict[str, np.ndarray]
) -> dict[str, float]:
    res = {c: float(np.nanmean(np.nanmean(mats[c][rows[c]], axis=0))) for c in COLS}
    res["Avg"] = float(np.mean([res[c] for c in COLS]))
    return res


def main() -> None:
    assert len(sys.argv) == 2, __doc__
    root = Path(sys.argv[1])
    rng = np.random.default_rng(0)
    print("| p_c | scorer | column | base | +VECTOR | Δ | 95% CI of Δ | P(Δ≤0) |")
    print("|---|---|---|---|---|---|---|---|")
    for pc in PCS:
        for b in BASES:
            mb, mv = load_cell(root, b, pc), load_cell(root, f"{b}_vector", pc)
            full = {c: np.arange(mb[c].shape[0]) for c in COLS}
            pb, pv = aggregate(mb, full), aggregate(mv, full)
            boots = {c: np.empty(N_BOOT) for c in COLS + ["Avg"]}
            for k in range(N_BOOT):
                # one session resample per task, shared by that task's columns and by both runs
                n_sessions = {COL_TASK[c]: mb[c].shape[0] for c in COLS}
                per_task = {t: rng.integers(0, n, n) for t, n in n_sessions.items()}
                rows = {c: per_task[COL_TASK[c]] for c in COLS}
                ab, av = aggregate(mb, rows), aggregate(mv, rows)
                for c in boots:
                    boots[c][k] = av[c] - ab[c]
            for c in COLS + ["Avg"]:
                lo, hi = np.percentile(boots[c], [2.5, 97.5])
                print(
                    f"| {pc} | {b} | {c} | {pb[c]:.2f} | {pv[c]:.2f} | {pv[c] - pb[c]:+.2f} | "
                    f"[{lo:+.2f}, {hi:+.2f}] | {(boots[c] <= 0).mean():.3f} |"
                )


if __name__ == "__main__":
    main()

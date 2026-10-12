"""Paired bootstrap CIs on LongBench (Llama-3.1-8B, 16 tasks) from the OLS-ablation re-runs.

Comparisons (all on identical samples, so the bootstrap is paired):
  * base scorer vs. base + VECTOR (paper W_OLS = C4 x 5000) — significance of the main-table gain;
  * VECTOR with WikiText-103 / C4 x 500 calibration vs. the paper calibration — calibration ablation.
KeyDiff runs query-agnostic, SnapKV query-aware (paper protocol; dirs suffixed __query_aware).
Per-sample scores use kvpress's LongBench metric (the scorer behind every LongBench number) and are checked
to reproduce each run's metrics.json. Resampling is stratified by task; Avg = mean of the 16 task
means. Reports the 95% CI of Delta = A - B over 10k resamples.

Usage:
    python scripts/bootstrap_ci_longbench.py results/ols_ablation
"""

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "kvpress/evaluation/benchmarks/longbench"))
from calculate_metrics import dataset2metric  # noqa: E402  # type: ignore[import]

TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]  # fmt: skip
MODEL = "meta-llama--Llama-3.1-8B-Instruct"
N_BOOT = 10000
REF_ITEM = re.compile(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"")
# (label A, (tag, method), label B, (tag, method)) — Delta = A - B
COMPARISONS = [
    ("KeyDiff+VECTOR", ("c4", "keydiff_vector"), "KeyDiff", ("base", "keydiff")),
    ("SnapKV+VECTOR", ("c4", "snapkv_vector"), "SnapKV", ("base", "snapkv")),
    (
        "KeyDiff+V (WikiText)",
        ("wiki", "keydiff_vector"),
        "KeyDiff+V (C4)",
        ("c4", "keydiff_vector"),
    ),
    (
        "KeyDiff+V (C4x500)",
        ("c4_n500", "keydiff_vector"),
        "KeyDiff+V (C4)",
        ("c4", "keydiff_vector"),
    ),
    (
        "SnapKV+V (WikiText)",
        ("wiki", "snapkv_vector"),
        "SnapKV+V (C4)",
        ("c4", "snapkv_vector"),
    ),
    (
        "SnapKV+V (C4x500)",
        ("c4_n500", "snapkv_vector"),
        "SnapKV+V (C4)",
        ("c4", "snapkv_vector"),
    ),
]


def parse_list(cell: str) -> list[str]:
    """Lists are stored as numpy-array reprs ("['a' 'b']", no commas)."""
    # literal_eval decodes escapes (\\n, \\', ...) exactly as Python wrote them
    items = [ast.literal_eval(m.group(0)) for m in REF_ITEM.finditer(cell)]
    assert items, f"no items parsed from {cell!r}"
    return items


def sample_scores(d: Path, task: str) -> np.ndarray:
    """Per-sample score in [0, 100], mirroring kvpress calculate_metrics.scorer exactly."""
    df = pd.read_csv(d / "predictions.csv", keep_default_na=False)
    out = np.empty(len(df))
    metric = dataset2metric[task]
    for i, (pred, answers, classes) in enumerate(
        zip(df["predicted_answer"], df["answers"], df["all_classes"])
    ):
        pred = str(pred)
        if task in ["trec", "triviaqa", "samsum", "lsht"]:
            pred = pred.lstrip().split("\n")[0]
        all_classes = parse_list(classes) if classes else []
        out[i] = 100 * max(
            metric(pred.lstrip(), gt, all_classes=all_classes)
            for gt in parse_list(answers)
        )
    official = float(json.loads((d / "metrics.json").read_text()))
    assert abs(round(out.mean(), 2) - official) < 0.011, (
        f"{d.name}: {out.mean():.2f} vs metrics.json {official}"
    )
    return out


def load(root: Path, tag: str, method: str, pc: str) -> list[np.ndarray]:
    suffix = "__query_aware" if method.startswith("snapkv") else ""
    return [
        sample_scores(
            root / tag / f"longbench__{t}__{MODEL}__{method}__{pc}{suffix}", t
        )
        for t in TASKS
    ]


def main() -> None:
    assert len(sys.argv) == 2, __doc__
    root = Path(sys.argv[1])
    rng = np.random.default_rng(0)
    cache: dict[tuple, list[np.ndarray]] = {}
    print("| p_c | A | B | A | B | Δ | 95% CI of Δ | P(Δ≤0) |")
    print("|---|---|---|---|---|---|---|---|")
    for pc in ["0.75", "0.90"]:
        for la, ka, lb, kb in COMPARISONS:
            for k in (ka, kb):
                if (k, pc) not in cache:
                    cache[(k, pc)] = load(root, k[0], k[1], pc)
            a, b = cache[(ka, pc)], cache[(kb, pc)]
            assert all(len(x) == len(y) for x, y in zip(a, b)), "sample counts differ"
            pa = np.mean([x.mean() for x in a])
            pb = np.mean([y.mean() for y in b])
            deltas = np.empty(N_BOOT)
            for r in range(N_BOOT):
                ia = ib = 0.0
                for x, y in zip(a, b):
                    idx = rng.integers(0, len(x), len(x))
                    ia += x[idx].mean()
                    ib += y[idx].mean()
                deltas[r] = (ia - ib) / len(TASKS)
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            print(
                f"| {pc} | {la} | {lb} | {pa:.2f} | {pb:.2f} | {pa - pb:+.2f} | "
                f"[{lo:+.2f}, {hi:+.2f}] | {(deltas <= 0).mean():.3f} |"
            )


if __name__ == "__main__":
    main()

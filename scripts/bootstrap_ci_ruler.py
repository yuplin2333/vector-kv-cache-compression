"""Paired bootstrap confidence intervals for RULER: base scorer vs. base + VECTOR.

For each (length, p_c, scorer) cell, both runs scored the SAME samples (same fraction/seed; checked
row by row). We resample samples with replacement *within each RULER subtask* (stratified), apply
the identical resample to both runs, and recompute the official RULER score (mean over the 13
subtask string-match scores, as in kvpress's calculate_metrics). The 2.5/97.5 percentiles of the
paired difference give a 95% CI for Delta = (+VECTOR) - base. Greedy decoding is deterministic, so
sample resampling (not seeds) is the relevant source of uncertainty.

Usage:
    python scripts/bootstrap_ci_ruler.py results/ruler_longcontext [--n-boot 10000]
"""

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

LENGTHS = ["16384", "32768", "65536", "131072"]
PCS = ["0.75", "0.90"]
SCORERS = ["keydiff", "snapkv", "kvzip", "pyramidkv"]
CTRL = re.compile(r"[\x00-\x1f]")


REF_ITEM = re.compile(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"")


def parse_refs(cell: str) -> list[str]:
    """Answers are stored as a numpy-array repr, e.g. "['a' 'b']" (no commas; numpy switches to
    double quotes for items containing an apostrophe), so ast/eval would concatenate them."""
    # literal_eval decodes escapes (\\n, \\', ...) exactly as Python wrote them
    items = [ast.literal_eval(m.group(0)) for m in REF_ITEM.finditer(cell)]
    assert items, f"no answers parsed from {cell!r}"
    return items


def per_sample_scores(df: pd.DataFrame) -> np.ndarray:
    """Per-sample string-match score in [0, 100], matching kvpress's RULER calculate_metrics."""
    out = np.empty(len(df))
    for i, (pred, refs, task) in enumerate(
        zip(df["predicted_answer"], df["answer"], df["task"])
    ):
        pred = CTRL.sub("", str(pred).strip()).strip().lower()
        refs = parse_refs(refs)
        hits = [1.0 if r.lower() in pred else 0.0 for r in refs]
        out[i] = 100 * (
            max(hits) if task.split("_")[0] == "qa" else sum(hits) / len(hits)
        )
    return out


QUERY_AWARE = (
    "snapkv",
    "pyramidkv",
)  # query-aware methods use their __query_aware runs (paper protocol)


def run_dir(root: Path, length: str, method: str, pc: str) -> Path:
    qa = method.startswith(QUERY_AWARE)
    pattern = (
        f"ruler__{length}__meta-llama--Llama-3.1-8B-Instruct__{method}__{pc}__fraction*"
    )
    hits = sorted(
        d for d in root.glob(pattern) if d.name.endswith("__query_aware") == qa
    )
    assert len(hits) == 1, (
        f"expected one run dir for {length}/{method}/{pc}, got {hits}"
    )
    return hits[0]


def load(d: Path) -> pd.DataFrame:
    df = pd.read_csv(d / "predictions.csv", keep_default_na=False)
    df["score"] = per_sample_scores(df)
    # Hard check: per-sample scores must reproduce kvpress's official per-subtask metrics exactly.
    official = json.loads((d / "metrics.json").read_text())
    for task, g in df.groupby("task"):
        ours = round(float(g["score"].mean()), 2)
        assert abs(ours - official[task]["string_match"]) < 0.011, (
            f"{d.name} {task}: {ours} vs {official[task]}"
        )
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("| length | p_c | scorer | base | +VECTOR | Δ | 95% CI of Δ | P(Δ≤0) |")
    print("|---|---|---|---|---|---|---|---|")
    for length in LENGTHS:
        for pc in PCS:
            for sc in SCORERS:
                b = load(run_dir(args.root, length, sc, pc))
                v = load(run_dir(args.root, length, f"{sc}_vector", pc))
                assert (
                    len(b) == len(v)
                    and (b["task"].to_numpy() == v["task"].to_numpy()).all()
                ), "rows differ"
                assert (b["question"].to_numpy() == v["question"].to_numpy()).all(), (
                    "sample order differs"
                )
                tasks = sorted(b["task"].unique())
                idx_by_task = [np.flatnonzero(b["task"].to_numpy() == t) for t in tasks]
                bs, vs = b["score"].to_numpy(), v["score"].to_numpy()
                point_b = np.mean([bs[ix].mean() for ix in idx_by_task])
                point_v = np.mean([vs[ix].mean() for ix in idx_by_task])
                deltas = np.empty(args.n_boot)
                for k in range(args.n_boot):
                    tb, tv = [], []
                    for ix in idx_by_task:
                        r = ix[rng.integers(0, len(ix), len(ix))]
                        tb.append(bs[r].mean())
                        tv.append(vs[r].mean())
                    deltas[k] = np.mean(tv) - np.mean(tb)
                lo, hi = np.percentile(deltas, [2.5, 97.5])
                p_le0 = (deltas <= 0).mean()
                print(
                    f"| {int(length) // 1024}k | {pc} | {sc} | {point_b:.2f} | {point_v:.2f} | "
                    f"{point_v - point_b:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {p_le0:.3f} |"
                )


if __name__ == "__main__":
    main()

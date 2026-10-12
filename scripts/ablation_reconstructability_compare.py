#!/usr/bin/env python
"""Reconstructability-signal ablation — compare the three approx-selection arms.

Reads results/ablation_reconstructability/*/metrics.json and prints, per (task, compression ratio),
the score for each arm plus the key deltas:
    v_err (VECTOR)  vs  imprank (naive importance cutoff)  vs  random (control)

Usage:
    python scripts/ablation_reconstructability_compare.py [--results-dir results/ablation_reconstructability]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# run-dir press_method  ->  friendly arm label
ARMS = {
    "keydiff_vector": "v_err",
    "keydiff_vector_imprank": "imprank",
    "keydiff_vector_random": "random",
}
ARM_ORDER = ["v_err", "imprank", "random"]


def _score(metrics_path: Path) -> float:
    obj = json.loads(metrics_path.read_text())
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, dict):
        # LongBench scorer may return {task: score} or {"score": x}; take the mean
        # of numeric leaves as a robust fallback.
        vals = [float(v) for v in obj.values() if isinstance(v, (int, float))]
        assert vals, f"no numeric score in {metrics_path}: {obj}"
        return sum(vals) / len(vals)
    raise ValueError(f"unexpected metrics format in {metrics_path}: {type(obj)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/ablation_reconstructability")
    args = ap.parse_args()

    root = Path(args.results_dir)
    assert root.is_dir(), f"results dir not found: {root}"

    # cell[(task, ratio)][arm] = score
    cell: dict[tuple[str, str], dict[str, float]] = {}
    for metrics_path in sorted(root.glob("*/metrics.json")):
        parts = metrics_path.parent.name.split("__")
        # longbench__<task>__<model>__<press_method>__<ratio>[__suffix]
        task, method, ratio = parts[1], parts[3], parts[4]
        arm = ARMS.get(method)
        if arm is None:
            continue  # ignore unrelated presses
        cell.setdefault((task, ratio), {})[arm] = _score(metrics_path)

    if not cell:
        print(f"No ablation results found under {root} yet.")
        return

    # Per-cell table.
    hdr = f"{'task':<16}{'cr':>5}  " + "".join(f"{a:>10}" for a in ARM_ORDER)
    hdr += f"{'v_err-imp':>12}{'v_err-rnd':>12}"
    print(hdr)
    print("-" * len(hdr))

    d_imp: list[float] = []
    d_rnd: list[float] = []
    per_arm_all: dict[str, list[float]] = {a: [] for a in ARM_ORDER}
    for task, ratio in sorted(cell):
        row = cell[(task, ratio)]
        line = f"{task:<16}{ratio:>5}  "
        for a in ARM_ORDER:
            line += f"{row[a]:>10.2f}" if a in row else f"{'--':>10}"
            if a in row:
                per_arm_all[a].append(row[a])
        if "v_err" in row and "imprank" in row:
            di = row["v_err"] - row["imprank"]
            d_imp.append(di)
            line += f"{di:>+12.2f}"
        else:
            line += f"{'--':>12}"
        if "v_err" in row and "random" in row:
            dr = row["v_err"] - row["random"]
            d_rnd.append(dr)
            line += f"{dr:>+12.2f}"
        else:
            line += f"{'--':>12}"
        print(line)

    print("-" * len(hdr))
    avg_line = f"{'MEAN':<16}{'':>5}  "
    for a in ARM_ORDER:
        vals = per_arm_all[a]
        avg_line += f"{(sum(vals) / len(vals)):>10.2f}" if vals else f"{'--':>10}"
    avg_line += f"{(sum(d_imp) / len(d_imp)):>+12.2f}" if d_imp else f"{'--':>12}"
    avg_line += f"{(sum(d_rnd) / len(d_rnd)):>+12.2f}" if d_rnd else f"{'--':>12}"
    print(avg_line)

    # Bottom line.
    if d_imp:
        n_pos = sum(1 for x in d_imp if x > 0)
        print(
            f"\nv_err beats imprank in {n_pos}/{len(d_imp)} cells; "
            f"mean margin {sum(d_imp) / len(d_imp):+.2f}."
        )
    if d_rnd:
        n_pos = sum(1 for x in d_rnd if x > 0)
        print(
            f"v_err beats random  in {n_pos}/{len(d_rnd)} cells; "
            f"mean margin {sum(d_rnd) / len(d_rnd):+.2f}."
        )


if __name__ == "__main__":
    main()

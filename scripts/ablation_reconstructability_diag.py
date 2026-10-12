#!/usr/bin/env python
"""Aggregate the reconstructability diagnostic: why is importance-rank (imprank) worse than random?

Reads results/ablation_reconstructability_diag/*/vector_stats.jsonl (produced by press `keydiff_vector_diag`
with --dump_vector_stats). Each per-layer record carries, over the SAME candidate pool:
  verr_approx_vecterr  mean v_err of the tokens the v_err rule would approximate
  verr_approx_imprank  mean v_err of the tokens imprank would approximate (least important)
  verr_approx_random   mean v_err of a random subset (== candidate-pool mean)
  corr_imp_verr        Pearson corr(importance, v_err) over the candidate pool

If verr_approx_imprank > verr_approx_random > verr_approx_vecterr, that directly explains
the task ranking (imprank reconstructs the hardest tokens). corr_imp_verr < 0 is the cause
(least-important <-> higher v_err).

Usage: python scripts/ablation_reconstructability_diag.py [--results-dir results/ablation_reconstructability_diag]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/ablation_reconstructability_diag")
    args = ap.parse_args()
    root = Path(args.results_dir)
    assert root.is_dir(), f"results dir not found: {root}"

    # per (task, ratio) -> lists of per-layer values
    keys = [
        "verr_approx_vecterr",
        "verr_approx_imprank",
        "verr_approx_random",
        "corr_imp_verr",
    ]
    by_cell: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {k: [] for k in keys}
    )

    n_files = 0
    for stats_path in sorted(root.glob("*/vector_stats.jsonl")):
        n_files += 1
        parts = stats_path.parent.name.split("__")
        task, ratio = parts[1], parts[4]
        for line in stats_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # evaluate.py wraps the press export under "stats" -> {"layers", "aggregate"}.
            stats = row.get("stats", row)
            for layer in stats.get("layers", []):
                for k in keys:
                    v = layer.get(k)
                    if v is not None:
                        by_cell[(task, ratio)][k].append(float(v))

    if not by_cell:
        print(f"No diagnostic stats found under {root}.")
        return

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    hdr = (
        f"{'task':<16}{'cr':>5}"
        f"{'verr@v_err':>12}{'verr@random':>13}{'verr@imprank':>14}"
        f"{'corr(imp,verr)':>16}"
    )
    print(f"(read {n_files} stats files)\n")
    print(hdr)
    print("-" * len(hdr))
    all_v = {k: [] for k in keys}
    for task, ratio in sorted(by_cell):
        c = by_cell[(task, ratio)]
        for k in keys:
            all_v[k].extend(c[k])
        print(
            f"{task:<16}{ratio:>5}"
            f"{mean(c['verr_approx_vecterr']):>12.4f}"
            f"{mean(c['verr_approx_random']):>13.4f}"
            f"{mean(c['verr_approx_imprank']):>14.4f}"
            f"{mean(c['corr_imp_verr']):>16.3f}"
        )
    print("-" * len(hdr))
    print(
        f"{'MEAN':<16}{'':>5}"
        f"{mean(all_v['verr_approx_vecterr']):>12.4f}"
        f"{mean(all_v['verr_approx_random']):>13.4f}"
        f"{mean(all_v['verr_approx_imprank']):>14.4f}"
        f"{mean(all_v['corr_imp_verr']):>16.3f}"
    )

    ve, rnd, imp = (
        mean(all_v["verr_approx_vecterr"]),
        mean(all_v["verr_approx_random"]),
        mean(all_v["corr_imp_verr"]) if False else mean(all_v["verr_approx_imprank"]),
    )
    corr = mean(all_v["corr_imp_verr"])
    print("\nMechanism check:")
    print(
        f"  mean v_err of approximated tokens:  v_err={ve:.4f} < random={rnd:.4f} < imprank={imp:.4f}"
        if ve < rnd < imp
        else f"  v_err={ve:.4f}  random={rnd:.4f}  imprank={imp:.4f}  (ordering not as hypothesized)"
    )
    print(
        f"  corr(importance, v_err) = {corr:+.3f}  "
        + (
            "(< 0: least-important <-> higher v_err -> imprank reconstructs the hardest)"
            if corr < 0
            else "(>= 0: hypothesis not supported by correlation)"
        )
    )


if __name__ == "__main__":
    main()

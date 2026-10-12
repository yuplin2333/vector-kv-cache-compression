#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the RULER long-context sweep (experiments/ruler_longcontext.sh) into a
length x method table per compression ratio.

Each run writes a per-task dict {task: {"string_match": score}} to
  results/ruler_longcontext/ruler__<length>__<model>__<method>__<ratio>[/<n>]/metrics.json

The RULER score for a cell is the mean of the 13 per-task string_match scores. We report
it per (method, ratio) across context lengths, so the base-vs-+VECTOR trend at long context
is visible. A cell needs all expected tasks present to be reported as final.

Usage: python scripts/ruler_aggregate.py [--dir results/ruler_longcontext]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# ruler__<length>__<model>__<method>__<ratio>[__<suffix><num>]*   (model may contain "--";
# method may contain "_"). Trailing suffixes like __fraction0.500 are tolerated and ignored.
DIR_RE = re.compile(
    r"^ruler__(?P<length>\d+)__(?P<model>.+)__(?P<method>[^_]+(?:_[^_]+)*)__(?P<ratio>[0-9.]+)"
    r"(?:__[a-zA-Z_]+[0-9.]+)*$"
)
N_TASKS = 13
BASE = ["keydiff", "snapkv", "kvzip", "pyramidkv"]
ORDER = (
    ["no_press"]
    + BASE
    + [f"{b}_vector" for b in BASE]
    + [f"{m}[agnostic]" for b in ("snapkv", "pyramidkv") for m in (b, f"{b}_vector")]
)


QA_SUFFIX = "__query_aware"
# Query-aware methods: SnapKV / PyramidKV (+VECTOR) are reported from their __query_aware runs (the
# paper's protocol); their earlier query-agnostic runs are listed separately as "<method>[agnostic]".
QA_FAMILY = ("snapkv", "pyramidkv")


def parse(name: str):
    qa = name.endswith(QA_SUFFIX)
    m = DIR_RE.match(name.removesuffix(QA_SUFFIX))
    if not m:
        return None
    method = m.group("method")
    if method.startswith(QA_FAMILY) and not qa:
        method += "[agnostic]"
    elif qa and not method.startswith(QA_FAMILY):
        method += "[query_aware]"
    return (m.group("length"), m.group("model"), method, m.group("ratio"))


def ruler_score(metrics: dict) -> float | None:
    vals = [
        v["string_match"]
        for v in metrics.values()
        if isinstance(v, dict) and "string_match" in v
    ]
    return sum(vals) / len(vals) if len(vals) == N_TASKS else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/ruler_longcontext")
    args = ap.parse_args()
    root = Path(args.dir)
    assert root.is_dir(), f"missing results dir: {root}"

    # (model, method, ratio, length) -> ruler score (averaged over any duplicate run dirs)
    cells: dict[tuple, list[float]] = defaultdict(list)
    partial: dict[tuple, int] = {}
    for mfile in root.rglob("metrics.json"):
        parsed = None
        for anc in mfile.parents:
            parsed = parse(anc.name)
            if parsed:
                break
        if not parsed:
            continue
        length, model, method, ratio = parsed
        metrics = json.loads(mfile.read_text())
        s = ruler_score(metrics)
        if s is None:
            partial[(model, method, ratio, length)] = len(metrics)
        else:
            cells[(model, method, ratio, length)].append(s)

    lengths = sorted({k[3] for k in cells} | {k[3] for k in partial}, key=int)
    models = sorted({k[0] for k in cells})
    for model in models:
        print(
            f"\n########## model: {model}  (RULER score = mean over {N_TASKS} tasks) ##########"
        )
        ratios = sorted({k[2] for k in cells if k[0] == model})
        for ratio in ratios:
            print(f"\n=== p_c={ratio}  (cols: context lengths) ===")
            print(f"{'method':>26} | " + " | ".join(f"{L:>7}" for L in lengths))
            methods = [
                m
                for m in ORDER
                if any(k[1] == m and k[2] in (ratio, "0.00") for k in cells)
            ]
            for method in methods:
                r = "0.00" if method == "no_press" else ratio
                row = []
                for L in lengths:
                    key = (model, method, r, L)
                    if key in cells:
                        row.append(f"{sum(cells[key]) / len(cells[key]):7.2f}")
                    elif key in partial:
                        row.append(f"{partial[key]:>2}/{N_TASKS}  ")
                    else:
                        row.append(f"{'--':>7}")
                print(f"{method:>26} | " + " | ".join(row))


if __name__ == "__main__":
    main()

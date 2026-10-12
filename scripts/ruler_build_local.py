#!/usr/bin/env python3
"""Convert RULER raw data (NVIDIA/RULER prepare.py output) into a local kvpress-format HF
dataset, saved to disk (no HF hub push), byte-identical in schema to the simonjegou/ruler
mirror that kvpress's evaluate.py expects.

Schema note: the simonjegou/ruler mirror was built by kvpress's create_huggingface_dataset.py
from an OLDER RULER whose `input` field CONCATENATED context+question+answer_prefix; its parser
then regex-splits that into (context, question, answer_prefix). The current NVIDIA/RULER instead
emits `input` (context+question) and a SEPARATE `answer_prefix` column. So we first reconstruct
the old concatenated form (`input + answer_prefix`) and then apply kvpress's exact split — this
reproduces simonjegou's columns, spacing and all, so evaluate.py treats our local data identically.

Verified against a live simonjegou/ruler(16384) row: e.g. niah context ends at the haystack,
question = "What is the special magic number for ...? ", answer_prefix = "The special magic ...".

Output: <out>/<length>/ (a save_to_disk HF dataset). evaluate.py loads it via RULER_LOCAL_DIR.

Usage:
  python scripts/ruler_build_local.py --raw RULER/scripts/data/data --out data/RULER \
      [--lengths 16384 32768 65536 131072]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "kvpress/evaluation/benchmarks/ruler"))
# kvpress's patterns + per-category max_new_tokens (single source of truth for the schema).
from create_huggingface_dataset import (  # noqa: E402  # type: ignore[import]
    ANSWER_PATTERNS,
    MAX_NEW_TOKENS,
    QUESTION_PATTERNS,
)


def _split(text: str, task_cat: str) -> tuple[str, str, str]:
    """kvpress's split_context_question, applied to the reconstructed concatenated input."""
    q_pat, a_pat = QUESTION_PATTERNS[task_cat], ANSWER_PATTERNS[task_cat]
    q_idx = list(q_pat.finditer(text))[-1].start()
    context, qa = text[:q_idx], text[q_idx:]
    a_idx = a_pat.search(qa).start()  # type: ignore[union-attr]
    question, answer_prefix = qa[:a_idx], qa[a_idx:]
    return context, question, answer_prefix


def build_length(ldir: Path) -> pd.DataFrame:
    rows = []
    for task_path in sorted(ldir.glob("**/*.jsonl")):
        task = task_path.parent.stem
        cat = task.split("_")[0]
        df = pd.read_json(task_path, lines=True)
        # Reconstruct the concatenated input the kvpress parser expects.
        full = df["input"].astype(str) + df["answer_prefix"].astype(str)
        ctx, q, ap = zip(*(_split(t, cat) for t in full))
        out = pd.DataFrame(
            {
                "context": ctx,
                "question": q,
                "answer_prefix": ap,
                "answer": df["outputs"],
                "task": task,
                "max_new_tokens": MAX_NEW_TOKENS[cat],
            }
        )
        rows.append(out)
    return pd.concat(rows).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="RULER/scripts/data/data")
    ap.add_argument("--out", default="data/RULER")
    ap.add_argument("--lengths", nargs="+", type=int, default=None)
    args = ap.parse_args()

    raw_root = (
        (REPO_ROOT / args.raw) if not Path(args.raw).is_absolute() else Path(args.raw)
    )
    out_root = (
        (REPO_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    )
    assert raw_root.is_dir(), f"missing raw RULER dir: {raw_root}"

    ldirs = (
        [raw_root / str(x) for x in args.lengths]
        if args.lengths
        else sorted(p for p in raw_root.glob("*") if p.is_dir() and p.name.isdigit())
    )
    for ldir in ldirs:
        assert ldir.is_dir(), f"missing length dir: {ldir}"
        df = build_length(ldir)
        dest = out_root / ldir.name
        dest.mkdir(parents=True, exist_ok=True)
        Dataset.from_pandas(df, preserve_index=False).save_to_disk(str(dest))
        print(f"[{ldir.name}] {len(df)} rows, {df['task'].nunique()} tasks -> {dest}")


if __name__ == "__main__":
    main()

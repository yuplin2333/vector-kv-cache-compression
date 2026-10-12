"""Score SCBench multi-turn predictions with SCBench's own `compute_scores`.

Run with RocketKV/.venv-rocketkv-457/bin/python (it has SCBench's scoring deps:
evaluate/rouge/jieba), on run dirs written by scripts/scbench_multiturn.py. Writes score.json next to
predictions.jsonl in every finished run dir (one with a DONE marker) that has none yet.

Usage:
    RocketKV/.venv-rocketkv-457/bin/python scripts/scbench_score.py results/scbench
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# scripts/evaluate.py would shadow the HF `evaluate` package that SCBench's scorer imports.
sys.path.remove(str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "RocketKV"))
from eval.scbench_utils.compute_scores import compute_scores  # noqa: E402  # type: ignore[import]

MODEL_NAME = "Llama-3.1-8B-Instruct"
MAX_SEQ_LEN = 131072


def main() -> None:
    assert len(sys.argv) == 2, __doc__
    root = Path(sys.argv[1])
    run_dirs = sorted(d for d in root.iterdir() if (d / "DONE").exists())
    assert run_dirs, f"no finished run dirs under {root}"
    for d in run_dirs:
        out = d / "score.json"
        if out.exists():
            continue
        task = d.name.split("__")[0]
        score = compute_scores(
            d / "predictions.jsonl", task, MODEL_NAME, max_seq_length=MAX_SEQ_LEN
        )
        out.write_text(json.dumps(score, indent=2))
        print(d.name, json.dumps(score))


if __name__ == "__main__":
    main()

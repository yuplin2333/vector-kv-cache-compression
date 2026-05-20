from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

LONGBENCH_TASKS = [
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


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_run_dir_name(run_dir_name: str) -> tuple[str, str, str, str, float]:
    parts = run_dir_name.split("__")
    assert len(parts) == 5 or len(parts) == 6, (
        "Run directory name must have 5 or 6 parts separated by '__', "
        f"got {len(parts)} parts: {run_dir_name}"
    )
    benchmark, subtask, model, press_method, ratio_str = parts[:5]

    suffix_parts = parts[5:]
    if len(suffix_parts) > 0:
        assert suffix_parts == ["query_aware"], (
            f"Unsupported extra suffix in run directory name: {run_dir_name}. "
            "Only '__query_aware' is allowed as an extra suffix."
        )

    try:
        compress_ratio = float(ratio_str)
    except ValueError as e:
        raise ValueError(
            f"Invalid compression ratio in run directory name: {run_dir_name}. "
        ) from e

    return benchmark, subtask, model, press_method, compress_ratio


def _extract_score(metrics_obj: Any, metrics_path: Path) -> float:
    if isinstance(metrics_obj, (int, float)):
        return float(metrics_obj)

    raise ValueError(
        f"Unsupported metrics.json type in {metrics_path}: {type(metrics_obj)}"
    )


def _collect_rows(
    results_dir: Path, *, strict: bool
) -> tuple[list[dict[str, Any]], int, int]:
    run_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir())
    assert len(run_dirs) > 0, (
        f"No first-level subdirectories found under: {results_dir}"
    )

    grouped_rows: dict[tuple[str, str, float], dict[str, Any]] = {}
    skipped_missing_metrics = 0
    skipped_non_longbench = 0

    for run_dir in run_dirs:
        benchmark, subtask, model, press_method, compress_ratio = _parse_run_dir_name(
            run_dir.name
        )
        if benchmark != "longbench":
            skipped_non_longbench += 1
            continue
        assert subtask in LONGBENCH_TASKS, (
            f"Unknown LongBench task '{subtask}' in {run_dir.name}"
        )

        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            if strict:
                raise FileNotFoundError(f"Missing metrics.json: {metrics_path}")
            print(f"WARNING: Missing metrics.json, skipped run dir: {run_dir}")
            skipped_missing_metrics += 1
            continue
        assert metrics_path.is_file(), f"metrics.json is not a file: {metrics_path}"

        score = _extract_score(_load_json(metrics_path), metrics_path)
        group_key = (model, press_method, compress_ratio)
        if group_key not in grouped_rows:
            grouped_rows[group_key] = {
                "model": model,
                "press_method": press_method,
                "compress_ratio": compress_ratio,
            }
        row = grouped_rows[group_key]
        assert subtask not in row, (
            f"Duplicate subtask '{subtask}' for "
            f"(model={model}, press_method={press_method}, compress_ratio={compress_ratio})"
        )
        row[subtask] = score

    rows = list(grouped_rows.values())
    rows.sort(
        key=lambda row: (
            row["model"],
            row["press_method"],
            row["compress_ratio"],
        )
    )
    assert len(rows) > 0, f"No valid longbench runs found under: {results_dir}"
    return rows, skipped_missing_metrics, skipped_non_longbench


def _write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "press_method", "compress_ratio", *LONGBENCH_TASKS]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read first-level run folders under a results directory, keep only longbench "
            "runs, and export a wide CSV: model + press_method + compress_ratio + 16 tasks."
        )
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing first-level run folders (each with metrics.json).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Default: <results_dir>/summary.csv "
            "(for example results/kvpress_kvzip/summary.csv)"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any longbench run folder is missing metrics.json.",
    )
    args = parser.parse_args()

    assert args.results_dir.exists(), f"Directory does not exist: {args.results_dir}"
    assert args.results_dir.is_dir(), f"Path must be a directory: {args.results_dir}"

    output_csv = (
        args.output_csv
        if args.output_csv is not None
        else args.results_dir / "summary.csv"
    )
    rows, skipped_missing_metrics, skipped_non_longbench = _collect_rows(
        args.results_dir, strict=args.strict
    )
    _write_csv(rows, output_csv)
    print(f"Wrote {len(rows)} rows to: {output_csv}")
    if skipped_missing_metrics > 0:
        print(
            f"Skipped {skipped_missing_metrics} run folders missing metrics.json "
            "(use --strict to fail instead)"
        )
    if skipped_non_longbench > 0:
        print(
            f"Skipped {skipped_non_longbench} non-longbench run folders "
            "(only benchmark='longbench' is exported)"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import ImageGrid


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        obj = yaml.safe_load(f)
    assert isinstance(obj, dict), f"{path} must contain a YAML mapping"
    return obj


def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _extract_score(metrics_obj: Any, metrics_path: Path, metric_key: str) -> float:
    if isinstance(metrics_obj, list):
        assert len(metrics_obj) > 0, f"{metrics_path} is empty"
        major_key, minor_key = metric_key.split(".")
        values: list[float] = []
        for idx, row in enumerate(metrics_obj):
            assert isinstance(row, dict), f"{metrics_path}[{idx}] must be an object"
            assert major_key in row, f"{metrics_path}[{idx}] missing '{major_key}'"
            metric_obj = row[major_key]
            assert isinstance(metric_obj, dict), (
                f"{metrics_path}[{idx}]['{major_key}'] must be an object"
            )
            assert minor_key in metric_obj, (
                f"{metrics_path}[{idx}]['{major_key}'] missing '{minor_key}'"
            )
            value = metric_obj[minor_key]
            assert isinstance(value, (int, float)), (
                f"{metrics_path}[{idx}]['{major_key}']['{minor_key}'] must be numeric"
            )
            values.append(float(value))
        return float(sum(values) / len(values))

    if isinstance(metrics_obj, dict):
        if "score" in metrics_obj:
            score = metrics_obj["score"]
            assert isinstance(score, (int, float)), (
                f"{metrics_path} key 'score' must be numeric"
            )
            return float(score)
        raise AssertionError(
            f"{metrics_path} is a dict but has no 'score'. "
            "For NIAH, metrics.json is expected to be a list."
        )

    raise AssertionError(
        f"Unsupported metrics.json type in {metrics_path}: {type(metrics_obj)}"
    )


def _format_ratio_tag(ratio: float) -> str:
    return f"{ratio:.2f}"


def _create_red_yellow_green_cmap() -> LinearSegmentedColormap:
    # Red -> yellow -> green: low score to high score
    return LinearSegmentedColormap.from_list(
        "niah_score_cmap", ["#F0496E", "#EBB839", "#0CD79F"]
    )


def _format_context_tick(context_length: int) -> str:
    if context_length >= 1000:
        return f"{context_length // 1000}k"
    return str(context_length)


def _normalize_model_tag(model_name: str) -> str:
    return model_name.replace("/", "--")


def _parse_run_dir_name(run_name: str, expected_prefix: str) -> dict[str, Any]:
    assert run_name.startswith(expected_prefix), (
        f"Run directory name must start with '{expected_prefix}', got: {run_name}"
    )
    payload = run_name[len(expected_prefix) :]
    match = re.search(
        # r"__([0-9]+(?:\.[0-9]+)?)__max_context([0-9]+)__needle_depth([0-9]+(?:\.[0-9]+)?)$",
        (
            r"__([0-9]+(?:\.[0-9]+)?)__max_context([0-9]+)"
            r"(?:__query_aware)?"
            r"__needle_depth([0-9]+(?:\.[0-9]+)?)$"
        ),
        payload,
    )
    assert match is not None, (
        "Unable to parse run directory name. Expected format: "
        f"{expected_prefix}<model_tag>__<press_name>__<ratio>__max_context<ctx_len>__needle_depth<depth>. "
        "Also supports an optional '__query_aware' marker before needle depth. "
        f"Got: {run_name}"
    )
    ratio = float(match.group(1))
    max_context = int(match.group(2))
    needle_depth = float(match.group(3))

    left = payload[: match.start()]
    parts = left.split("__")
    assert len(parts) >= 2, (
        "Unable to parse model/press from run directory. Expected at least "
        f"<model_tag>__<press_name> before ratio fields. Got: {run_name}"
    )
    press_name = parts[-1]
    model_tag = "__".join(parts[:-1])
    return {
        "model_tag": model_tag,
        "press_name": press_name,
        "compression_ratio": ratio,
        "max_context_length": max_context,
        "needle_depth": needle_depth,
    }


def _parse_layout(layout_text: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*([0-9]+)\s*[xX\*]\s*([0-9]+)\s*", layout_text)
    assert match is not None, (
        "--layout must be in '<rows>x<cols>' or '<rows>*<cols>' format, "
        f"got: {layout_text}"
    )
    rows = int(match.group(1))
    cols = int(match.group(2))
    assert rows > 0 and cols > 0, (
        f"--layout rows/cols must be > 0, got rows={rows}, cols={cols}"
    )
    return rows, cols


def _build_heatmap_data(
    results_dir: Path,
    run_prefix: str,
    metric_key: str,
    model_filter: str | None,
    press_name_filter: str | None,
    compression_ratio_filter: float | None,
) -> dict[str, Any]:
    assert results_dir.exists(), f"results_dir does not exist: {results_dir}"
    assert results_dir.is_dir(), f"results_dir must be a directory: {results_dir}"
    assert "." in metric_key, (
        f"--metric-key must be in '<major>.<minor>' format, got: {metric_key}"
    )

    run_dirs = sorted(
        [
            p
            for p in results_dir.iterdir()
            if p.is_dir() and p.name.startswith(run_prefix)
        ]
    )
    assert len(run_dirs) > 0, (
        f"No run directories with prefix '{run_prefix}' found under {results_dir} "
    )

    records: list[tuple[int, float, float, float, str, str, str]] = []
    for run_dir in run_dirs:
        config_path = run_dir / "config.yaml"
        metrics_path = run_dir / "metrics.json"
        if not (config_path.exists() and metrics_path.exists()):
            continue

        parsed_name = _parse_run_dir_name(run_dir.name, run_prefix)
        cfg = _load_yaml(config_path)
        dataset = cfg.get("dataset")
        if dataset != "needle_in_haystack":
            continue

        model = cfg.get("model")
        assert isinstance(model, str) and len(model.strip()) > 0, (
            f"config.model must be a non-empty string in {config_path}"
        )
        press_name = cfg.get("press_name")
        assert isinstance(press_name, str) and len(press_name.strip()) > 0, (
            f"config.press_name must be a non-empty string in {config_path}"
        )

        cr = cfg.get("compression_ratio")
        if cr is None:
            cr = parsed_name["compression_ratio"]
        assert isinstance(cr, (int, float)), (
            f"compression_ratio must be numeric in {config_path}"
        )
        cr = float(cr)

        if model_filter is not None and model != model_filter:
            continue
        if press_name_filter is not None and press_name != press_name_filter:
            continue
        if (
            compression_ratio_filter is not None
            and abs(cr - compression_ratio_filter) > 1e-8
        ):
            continue

        max_context = cfg.get("max_context_length")
        needle_depth = cfg.get("needle_depth")
        if max_context is None:
            max_context = parsed_name["max_context_length"]
        if needle_depth is None:
            needle_depth = parsed_name["needle_depth"]
        assert isinstance(max_context, int), (
            f"config.max_context_length must be int in {config_path}, got {type(max_context)}"
        )
        assert isinstance(needle_depth, (int, float)), (
            f"config.needle_depth must be numeric in {config_path}, got {type(needle_depth)}"
        )

        model_tag_from_name = _normalize_model_tag(model)
        assert model_tag_from_name == parsed_name["model_tag"], (
            f"Run directory model tag mismatch in {run_dir.name}. "
            f"Expected model tag '{model_tag_from_name}' from config.model, "
            f"got '{parsed_name['model_tag']}'"
        )
        assert press_name == parsed_name["press_name"], (
            f"Run directory press_name mismatch in {run_dir.name}. "
            f"Expected '{press_name}' from config, got '{parsed_name['press_name']}'"
        )

        metrics_obj = _load_json(metrics_path)
        score = _extract_score(metrics_obj, metrics_path, metric_key)
        records.append(
            (
                max_context,
                float(needle_depth),
                score,
                cr,
                run_dir.name,
                model,
                press_name,
            )
        )

    assert len(records) > 0, (
        f"No needle_in_haystack runs found under {results_dir} "
        f"(compression_ratio filter={compression_ratio_filter})"
    )

    unique_groups = sorted({(r[5], r[6], r[3]) for r in records})
    assert len(unique_groups) == 1, (
        "Filtered records still include multiple (model, press_name, compression_ratio) groups. "
        "Please refine filters with --model/--press-name/--compression-ratio. "
        f"Found groups: {unique_groups}"
    )
    selected_model, selected_press, selected_ratio = unique_groups[0]

    unique_ratios = sorted({r[3] for r in records})
    if compression_ratio_filter is None:
        assert len(unique_ratios) == 1, (
            "Found multiple compression ratios in input directory. "
            "Please set --compression-ratio explicitly. "
            f"Found: {[round(x, 4) for x in unique_ratios]}"
        )
    selected_ratio = unique_ratios[0]

    context_lengths = sorted({r[0] for r in records})
    needle_depths = sorted({r[1] for r in records})
    heatmap = np.full((len(needle_depths), len(context_lengths)), np.nan, dtype=float)
    depth_to_idx = {depth: i for i, depth in enumerate(needle_depths)}
    context_to_idx = {ctx: i for i, ctx in enumerate(context_lengths)}

    seen_cells: set[tuple[int, int]] = set()
    for max_context, needle_depth, score, _, run_name, _, _ in records:
        i = depth_to_idx[needle_depth]
        j = context_to_idx[max_context]
        cell = (i, j)
        assert cell not in seen_cells, (
            "Duplicate run for same (needle_depth, max_context_length) cell. "
            f"Offending run: {run_name}, cell=({needle_depth}, {max_context})"
        )
        seen_cells.add(cell)
        heatmap[i, j] = score

    missing_positions = np.argwhere(np.isnan(heatmap))
    missing_cells = [
        (needle_depths[int(i)], context_lengths[int(j)]) for i, j in missing_positions
    ]
    assert not np.isnan(heatmap).any(), (
        "Heatmap has missing cells. Ensure all depth-length combinations exist. "
        f"Depth count={len(needle_depths)}, length count={len(context_lengths)}, "
        f"filled={len(seen_cells)}, missing_count={len(missing_cells)}, "
        f"missing_cells(needle_depth, context_length)={missing_cells}"
    )

    return {
        "results_dir": results_dir,
        "selected_model": selected_model,
        "selected_press": selected_press,
        "selected_ratio": selected_ratio,
        "context_lengths": context_lengths,
        "needle_depths": needle_depths,
        "heatmap": heatmap,
    }


def _should_show_xlabel(policy: str, row: int, rows: int) -> bool:
    if policy == "all":
        return True
    if policy == "bottom":
        return row == rows - 1
    if policy == "none":
        return False
    raise AssertionError(f"Unsupported xlabel policy: {policy}")


def _should_show_ylabel(policy: str, col: int) -> bool:
    if policy == "all":
        return True
    if policy == "left":
        return col == 0
    if policy == "none":
        return False
    raise AssertionError(f"Unsupported ylabel policy: {policy}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot NIAH heatmap(s) from KVPress evaluation results. "
            "Supports single-panel and multi-panel grid layouts."
        )
    )
    parser.add_argument(
        "results_dirs",
        nargs="+",
        type=Path,
        help=(
            "One or more directories containing NIAH run folders. "
            "For multi-panel mode, pass directories in row-major order "
            "(left-to-right, top-to-bottom)."
        ),
    )
    parser.add_argument(
        "--layout",
        type=str,
        default=None,
        help=(
            "Grid layout for multi-panel mode, e.g. 2x4 or 2*4. "
            "If omitted, defaults to 1xN."
        ),
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="needle_in_haystack__",
        help="Run directory name prefix to scan. Default: needle_in_haystack__.",
    )
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=None,
        help=(
            "Filter runs by compression ratio. "
            "Required when results_dir contains multiple ratios."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Filter by model name from config.yaml (e.g. meta-llama/Llama-3.1-8B-Instruct). "
            "Compared by exact string match."
        ),
    )
    parser.add_argument(
        "--press-name",
        type=str,
        default=None,
        help="Filter by press_name from config.yaml (exact match).",
    )
    parser.add_argument(
        "--metric-key",
        type=str,
        default="rouge-l.f",
        help="Metric path for list-style metrics.json. Default: rouge-l.f.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help=(
            "Figure title. In single-panel mode this is the subplot title; "
            "in multi-panel mode this is a global suptitle."
        ),
    )
    parser.add_argument(
        "--panel-titles",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Per-panel titles in row-major order. "
            "Length must match number of input directories."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Default: <first_results_dir>/niah_heatmap[_<rows>x<cols>].pdf.",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=0.46,
        help="Heatmap cell size scale (inches per cell). Default: 0.46.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=12.0,
        help=(
            "Base font size for subplot title/axis labels/ticks/colorbar labels. "
            "Default: 12.0."
        ),
    )
    parser.add_argument(
        "--gap-scale",
        type=float,
        default=1.0,
        help=(
            "Global gap scale for subplot spacing and colorbar spacing. "
            "Larger means wider gaps. Default: 1.0."
        ),
    )
    parser.set_defaults(show_colorbar=True, show_ylabel=True, show_xlabel=True)
    parser.add_argument(
        "--show-colorbar",
        dest="show_colorbar",
        action="store_true",
        help="Show colorbar. Default behavior.",
    )
    parser.add_argument(
        "--hide-colorbar",
        dest="show_colorbar",
        action="store_false",
        help=(
            "Hide colorbar text/ticks, but keep reserved colorbar space so the "
            "heatmap panel stays aligned with figures where colorbar is shown."
        ),
    )
    parser.add_argument(
        "--colorbar-label",
        type=str,
        default="ROUGE-L",
        help="Colorbar label text. Default: ROUGE-L.",
    )
    parser.add_argument(
        "--show-ylabel",
        dest="show_ylabel",
        action="store_true",
        help="Show y-axis label ('Needle Depth (%%)'). Default behavior.",
    )
    parser.add_argument(
        "--hide-ylabel",
        dest="show_ylabel",
        action="store_false",
        help="Hide y-axis label while keeping y-axis ticks.",
    )
    parser.add_argument(
        "--show-xlabel",
        dest="show_xlabel",
        action="store_true",
        help="Show x-axis label ('Context Length'). Default behavior.",
    )
    parser.add_argument(
        "--hide-xlabel",
        dest="show_xlabel",
        action="store_false",
        help="Hide x-axis label while keeping x-axis ticks.",
    )
    parser.add_argument(
        "--multi-xlabel-policy",
        type=str,
        choices=["all", "bottom", "none"],
        default="bottom",
        help=(
            "X-label policy in multi-panel mode: all, bottom, or none. Default: bottom."
        ),
    )
    parser.add_argument(
        "--multi-ylabel-policy",
        type=str,
        choices=["all", "left", "none"],
        default="left",
        help=("Y-label policy in multi-panel mode: all, left, or none. Default: left."),
    )
    parser.add_argument(
        "--multi-colorbar-mode",
        type=str,
        choices=["single", "edge"],
        default="single",
        help=(
            "Colorbar layout in multi-panel mode when colorbar is enabled: "
            "'single' for one shared colorbar on the right, "
            "'edge' for one colorbar per row on the rightmost column."
        ),
    )
    args = parser.parse_args()

    assert args.cell_size > 0.0, f"--cell-size must be >0, got {args.cell_size}"
    assert args.font_size > 0.0, f"--font-size must be >0, got {args.font_size}"
    assert args.gap_scale > 0.0, f"--gap-scale must be >0, got {args.gap_scale}"
    results_dirs = args.results_dirs
    for results_dir in results_dirs:
        assert results_dir.exists(), f"results_dir does not exist: {results_dir}"
        assert results_dir.is_dir(), f"results_dir must be a directory: {results_dir}"

    if args.layout is None:
        rows = 1
        cols = len(results_dirs)
    else:
        rows, cols = _parse_layout(args.layout)
        assert rows * cols == len(results_dirs), (
            f"--layout expects exactly rows*cols directories, got rows={rows}, cols={cols}, "
            f"rows*cols={rows * cols}, provided={len(results_dirs)}"
        )

    panel_data = [
        _build_heatmap_data(
            results_dir=results_dir,
            run_prefix=args.run_prefix,
            metric_key=args.metric_key,
            model_filter=args.model,
            press_name_filter=args.press_name,
            compression_ratio_filter=args.compression_ratio,
        )
        for results_dir in results_dirs
    ]
    assert len(panel_data) == len(results_dirs)
    if args.panel_titles is not None:
        assert len(args.panel_titles) == len(panel_data), (
            "--panel-titles count must equal number of input directories. "
            f"Got titles={len(args.panel_titles)}, dirs={len(panel_data)}"
        )

    context_lengths_ref = panel_data[0]["context_lengths"]
    needle_depths_ref = panel_data[0]["needle_depths"]
    for idx, one_panel in enumerate(panel_data):
        assert one_panel["context_lengths"] == context_lengths_ref, (
            "All panels must have identical context_lengths for aligned multi-panel plotting. "
            f"Mismatch at panel index {idx}: {one_panel['results_dir']}"
        )
        assert one_panel["needle_depths"] == needle_depths_ref, (
            "All panels must have identical needle_depths for aligned multi-panel plotting. "
            f"Mismatch at panel index {idx}: {one_panel['results_dir']}"
        )

    context_lengths = context_lengths_ref
    needle_depths = needle_depths_ref
    panel_w = max(2.6, args.cell_size * len(context_lengths) + 0.8)
    panel_h = max(2.4, args.cell_size * len(needle_depths) + 0.8)
    fig_w = max(5.5, panel_w * cols + 1.8)
    fig_h = max(4.5, panel_h * rows + 1.2)

    is_multi = len(results_dirs) > 1
    colorbar_mode = "single" if not is_multi else args.multi_colorbar_mode

    fig = plt.figure(figsize=(fig_w, fig_h))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(rows, cols),
        axes_pad=0.32 * args.gap_scale,
        cbar_location="right",
        cbar_mode=colorbar_mode,
        cbar_size="3.5%",
        cbar_pad=0.08 * args.gap_scale,
    )

    cmap = _create_red_yellow_green_cmap()
    images = []
    for idx, one_panel in enumerate(panel_data):
        ax = grid[idx]
        row = idx // cols
        col = idx % cols
        heatmap = one_panel["heatmap"]
        im = ax.imshow(
            heatmap,
            aspect="equal",
            origin="upper",
            cmap=cmap,
            interpolation="nearest",
            # vmin=0.0,
            # vmax=1.0,
        )
        images.append(im)

        ax.set_xticks(np.arange(len(context_lengths)))
        ax.set_xticklabels(
            [_format_context_tick(x) for x in context_lengths],
            rotation=45,
            ha="right",
            fontsize=args.font_size,
        )
        ax.set_yticks(np.arange(len(needle_depths)))
        ax.set_yticklabels([f"{d:g}" for d in needle_depths], fontsize=args.font_size)

        if is_multi:
            show_xlabel = _should_show_xlabel(
                args.multi_xlabel_policy, row=row, rows=rows
            )
            show_ylabel = _should_show_ylabel(args.multi_ylabel_policy, col=col)
        else:
            show_xlabel = args.show_xlabel
            show_ylabel = args.show_ylabel

        ax.set_xlabel("Context Length" if show_xlabel else "", fontsize=args.font_size)
        ax.set_ylabel(
            "Needle Depth (%)" if show_ylabel else "", fontsize=args.font_size
        )

        auto_title = (
            f"{one_panel['selected_press']}, "
            f"cr={_format_ratio_tag(one_panel['selected_ratio'])}"
        )
        panel_title = (
            args.panel_titles[idx] if args.panel_titles is not None else auto_title
        )
        if is_multi:
            ax.set_title(panel_title, fontsize=args.font_size)
        else:
            if args.panel_titles is not None:
                ax.set_title(panel_title, fontsize=args.font_size)
            else:
                ax.set_title(
                    args.title if args.title is not None else auto_title,
                    fontsize=args.font_size,
                )

    if colorbar_mode == "single":
        cbar = grid[0].cax.colorbar(images[0])
        cbar.set_label(args.colorbar_label, fontsize=args.font_size)
        cbar.ax.tick_params(labelsize=args.font_size)
        if not args.show_colorbar:
            grid[0].cax.set_visible(False)
    elif colorbar_mode == "edge":
        right_col_indices = [r * cols + (cols - 1) for r in range(rows)]
        for idx in right_col_indices:
            cbar = grid[idx].cax.colorbar(images[idx])
            cbar.set_label(args.colorbar_label, fontsize=args.font_size)
            cbar.ax.tick_params(labelsize=args.font_size)
            if not args.show_colorbar:
                grid[idx].cax.set_visible(False)
    else:
        raise AssertionError(f"Unsupported colorbar mode: {colorbar_mode}")

    if is_multi and args.title is not None:
        fig.suptitle(args.title, fontsize=args.font_size)

    if args.output is None:
        if is_multi:
            output_path = results_dirs[0] / f"niah_heatmap_{rows}x{cols}.pdf"
        else:
            output_path = results_dirs[0] / "niah_heatmap.pdf"
    else:
        output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved heatmap to: {output_path}")
    for idx, one_panel in enumerate(panel_data):
        print(
            f"panel[{idx}] dir={one_panel['results_dir']} "
            f"model={one_panel['selected_model']} "
            f"press_name={one_panel['selected_press']} "
            f"compression_ratio={_format_ratio_tag(one_panel['selected_ratio'])}"
        )
    print(f"context_lengths: {context_lengths}")
    print(f"needle_depths: {needle_depths}")


if __name__ == "__main__":
    main()

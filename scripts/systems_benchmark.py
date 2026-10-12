#!/usr/bin/env python
"""Systems microbenchmark: latency and memory of VECTOR vs base eviction and full KV.

Measures the runtime/memory cost of VECTOR's value reconstruction vs the base
eviction baseline and full KV, at the PyTorch level (no custom kernel;
an upper bound on the overhead). Methodology adapted from kvpress/notebooks/speed_and_memory.ipynb
(warmup, torch.cuda.synchronize, reset_peak_memory_stats).

Configs compared, at p_c in {0.75, 0.90}, batch=1, fixed context lengths:
  no_press        full KV cache (baseline)
  keydiff         base eviction (KeyDiff)
  keydiff_vector  VECTOR (KeyDiff importance + K->V value reconstruction)

Reported per config: prefill time (TTFT proxy), decode throughput (tok/s),
peak GPU memory, and measured KV-cache size. NOTE: keydiff_vector MATERIALIZES the
reconstructed V (academic shortcut), so its measured cache size is LARGER than the
2*(1-p_c) deployment target — the memory saving is argued analytically in the paper,
this script quantifies the COMPUTE overhead of reconstruction.

The model is loaded ONCE and reused across configs (shared-box OOM safety); peak
memory is reset between configs so each number reflects that config's run.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/systems_benchmark.py \
      --weights weights/llama31_8b_probe_konly.pt \
      --context-lengths 4096 8192 16384 --max-new-tokens 64 \
      --out results/systems_benchmark.json
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from kvpress import KeyDiffPress, PyramidKVPress, SnapKVPress
from vector.kvpress import (
    VECTORWrapperPress,
    KeyDiffVECTORPress,
    SnapKVVECTORPress,
)

CKPT = "meta-llama/Llama-3.1-8B-Instruct"

# Kwarg to keep only the last logit at prefill (name changed across transformers
# versions). Resolved at runtime in main(); avoids materializing seq_len x vocab logits.
_LOGITS_KW: dict = {}


def _resolve_logits_kw(model) -> dict:
    import inspect

    params = inspect.signature(type(model).forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return {name: 1}
    return {}


def cache_size_gb(cache: DynamicCache) -> float:
    total = 0
    for layer in cache.layers:
        for t in (layer.keys, layer.values):
            if t is not None:
                total += t.element_size() * t.nelement()
    return total / 1024**3


def _sync() -> None:
    torch.cuda.synchronize()


def measure(model, press, n_tokens: int, max_new_tokens: int, device: str) -> dict:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.max_memory_allocated()  # weights already resident

    inputs = torch.arange(n_tokens).reshape(1, n_tokens).to(device)

    # ---- prefill (TTFT proxy) + measured compressed cache size ----
    # Run the FULL-length prefill twice: the first (untimed) warms GPU clocks AND lets
    # cuBLAS autotune the exact V_pred GEMM shape, so the second (timed) isn't polluted
    # by one-time first-call overhead (which otherwise inflated the first VECTOR press).
    def _prefill():
        cache = DynamicCache()
        ctx = press(model) if press is not None else _nullctx()
        with torch.no_grad(), ctx:
            _sync()
            t0 = time.time()
            model(inputs, past_key_values=cache, **_LOGITS_KW)
            _sync()
            dt = time.time() - t0
            sz = cache_size_gb(cache)
        del cache
        torch.cuda.empty_cache()
        return dt, sz

    _prefill()  # warm (untimed)
    torch.cuda.reset_peak_memory_stats()
    prefill_time, csize = _prefill()  # timed

    # ---- decode throughput via TWO-POINT timing ----
    # generate with a short and a long max_new_tokens; both include the SAME prefill,
    # so (t_long - t_short) cancels prefill exactly and isolates per-token decode cost.
    # This is robust when prefill >> decode (a plain total-minus-prefill subtraction is
    # noise-dominated there).
    model.generation_config.eos_token_id = None
    model.generation_config.stop_strings = None
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    def _gen_time(n_new: int):
        ctx = press(model) if press is not None else _nullctx()
        with torch.no_grad(), ctx:
            _sync()
            t0 = time.time()
            out = model.generate(
                inputs,
                max_new_tokens=n_new,
                generation_config=model.generation_config,
                pad_token_id=-1,
            )
            _sync()
            return time.time() - t0, out

    n_short = 8
    n_long = max_new_tokens + n_short
    t_short, _ = _gen_time(n_short)
    t_long, out = _gen_time(n_long)
    peak = torch.cuda.max_memory_allocated()
    assert out.shape == (1, n_tokens + n_long), out.shape

    decode_time = max(t_long - t_short, 1e-6)
    return {
        "prefill_time_s": round(prefill_time, 4),
        "decode_tok_per_s": round((n_long - n_short) / decode_time, 2),
        "decode_ms_per_tok": round(1000 * decode_time / (n_long - n_short), 3),
        "peak_mem_gb": round(peak / 1024**3, 3),
        "kv_cache_gb": round(csize, 4),
        "weights_baseline_gb": round(base_mem / 1024**3, 3),
    }


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# Base scorer -> (base-eviction factory, +VECTOR hybrid factory). All reuse the same
# per-model W_OLS weights; this shows VECTOR's added overhead across base scorers.
def _scorer_factories(weights: str):
    return {
        "keydiff": (
            lambda p: KeyDiffPress(compression_ratio=p),
            lambda p: KeyDiffVECTORPress(compression_ratio=p, weights_path=weights),
        ),
        "snapkv": (
            lambda p: SnapKVPress(compression_ratio=p),
            lambda p: SnapKVVECTORPress(compression_ratio=p, weights_path=weights),
        ),
        "pyramidkv": (
            lambda p: PyramidKVPress(compression_ratio=p),
            lambda p: VECTORWrapperPress(
                press=PyramidKVPress(compression_ratio=p), weights_path=weights
            ),
        ),
    }


# Metric fields produced by measure() that we aggregate across repeats.
_METRIC_FIELDS = (
    "prefill_time_s",
    "decode_tok_per_s",
    "decode_ms_per_tok",
    "peak_mem_gb",
    "kv_cache_gb",
    "weights_baseline_gb",
)


def _aggregate(runs: list[dict]) -> dict:
    """Mean/std over repeated measure() dicts. std is 0.0 for a single run.

    Timing (prefill/decode) is noisy, so we report mean +/- sample std over
    `repeats` runs; deterministic fields (kv_cache_gb) just carry std ~= 0.
    """
    out: dict = {"repeats": len(runs)}
    for f in _METRIC_FIELDS:
        vals = [r[f] for r in runs if f in r]
        if not vals:
            continue
        out[f] = round(statistics.mean(vals), 4)
        out[f"{f}_std"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/llama31_8b_probe_konly.pt")
    ap.add_argument(
        "--context-lengths", type=int, nargs="+", default=[4096, 8192, 16384]
    )
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.75, 0.90])
    ap.add_argument("--scorers", type=str, nargs="+", default=["keydiff"])
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", default="results/systems_benchmark.json")
    ap.add_argument("--model", default=CKPT, help="HF model id to benchmark")
    ap.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="measurements per config; report mean +/- std (timing is noisy)",
    )
    args = ap.parse_args()

    device = "cuda:0"
    # attn impl: prefer flash_attention_2, fall back to sdpa (FA2 may be unavailable).
    attn = "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype="auto", attn_implementation=attn
        ).to(device)  # type: ignore[arg-type]
    except (ImportError, ValueError, RuntimeError):
        attn = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype="auto", attn_implementation=attn
        ).to(device)  # type: ignore[arg-type]
    model.eval()
    global _LOGITS_KW
    _LOGITS_KW = _resolve_logits_kw(model)
    print(
        f"model={args.model} loaded (attn_implementation={attn}, logits_kw={_LOGITS_KW})"
    )

    # GPU-clock warmup so the first recorded row isn't a cold-clock artifact.
    _ = measure(model, None, 2048, 8, device)

    factories = _scorer_factories(args.weights)
    for s in args.scorers:
        assert s in factories, f"unknown scorer {s!r}; known: {list(factories)}"

    # Build (label, press) configs: full-KV once, then base vs +VECTOR per scorer/ratio.
    def _configs():
        yield ("no_press", None)
        for s in args.scorers:
            base_fac, vec_fac = factories[s]
            for p_c in args.ratios:
                yield (f"{s}@{p_c:.2f}", base_fac(p_c))
                yield (f"{s}+VECTOR@{p_c:.2f}", vec_fac(p_c))

    results = []
    for n_tokens in args.context_lengths:
        for label, press in _configs():
            # Preload W_OLS weights OUTSIDE the timed region so the one-time 675MB
            # torch.load (cold disk) never contaminates prefill timing.
            if press is not None and hasattr(press, "_ensure_weights"):
                press._ensure_weights()
            runs: list[dict] = []
            err: str | None = None
            for _ in range(args.repeats):
                try:
                    runs.append(
                        measure(model, press, n_tokens, args.max_new_tokens, device)
                    )
                except torch.cuda.OutOfMemoryError as e:
                    err = f"OOM: {e}"
                    torch.cuda.empty_cache()
                    break
            agg = {"error": err} if err else _aggregate(runs)
            row = {
                "ctx_len": n_tokens,
                "config": label,
                "attn": attn,
                **agg,
                "runs": runs,  # raw per-repeat metrics, for re-analysis
            }
            results.append(row)
            print(
                f"[ctx={n_tokens:>6}] {label:<22} "
                + (
                    str(err)
                    if err
                    else f"prefill={agg['prefill_time_s']:.3f}+/-{agg['prefill_time_s_std']:.3f}s  "
                    f"decode={agg['decode_tok_per_s']:.1f}+/-{agg['decode_tok_per_s_std']:.1f}tok/s  "
                    f"peak={agg['peak_mem_gb']:.2f}GB  kv={agg['kv_cache_gb']:.3f}GB"
                )
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nsaved {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()

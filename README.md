# VECTOR

VECTOR is a plug-in augmentation for eviction-based KV cache compression.
It adds a three-way token routing strategy (retain, approximate, evict) with
offline-calibrated value reconstruction from keys.

## Repository Layout

- `src/vector/`: core implementation (KVPress integration, calibration, model adapters)
- `src/vector/baselines/`: faithful D2O port used for the Qwen3 competitor rows
- `patches/`: drop-in replacement files for third-party repositories
  (`kvpress/`, `RocketKV/`, `D2O/`), mirroring each repository's own layout
- `experiments/`: calibration and experiment scripts
- `scripts/`: calibration, evaluation drivers and result post-processing utilities

## Quick Start

The commands below are organized as a full reproducible flow. Each step includes
what it does and why it is needed.

### 1) Prepare Python environment

This project targets Python 3.11 and uses `uv` for dependency management.

```bash
# Enter repository root
cd /path/to/vector

# Verify Python version (must be 3.11.x)
python --version

# Install dependencies from pyproject.toml / uv.lock
uv sync

# Activate the created virtual environment (optional but convenient)
source .venv/bin/activate
```

### 2) Clone KVPress baseline code

VECTOR extends KVPress evaluation. Clone KVPress into `./kvpress` so all
experiment scripts can find `kvpress/evaluation`.

```bash
# Clone KVPress to ./kvpress (commit used for all results)
git clone https://github.com/NVIDIA/kvpress.git ./kvpress
git -C kvpress checkout 91773cb8b5713fe6ad54e3e7a97dd343fb40d37b
```

### 3) Apply VECTOR patches to KVPress evaluation

These files register VECTOR presses and add evaluation options used by the
scripts in this repository.

```bash
# Copy patched evaluator and registry
cp patches/kvpress/evaluation/evaluate.py \
   kvpress/evaluation/evaluate.py
cp patches/kvpress/evaluation/evaluate_registry.py \
   kvpress/evaluation/evaluate_registry.py

# Copy NIAH utility patch
cp patches/kvpress/evaluation/benchmarks/needle_in_haystack/utils.py \
   kvpress/evaluation/benchmarks/needle_in_haystack/utils.py
```

### 4) Run offline calibration (choose one mode)

Calibration generates OLS weights used by VECTOR approximation during inference.

```bash
# K -> V calibration (main path used by VECTOR); MODEL=Qwen/Qwen3-14B or Qwen/Qwen3-0.6B
# for the other backbones
bash experiments/calibrate_k_only_for_v.sh

# Alternative ablation: V -> K calibration
bash experiments/calibrate_v_only_for_k.sh
```

### 5) Run evaluation experiments

Run one LongBench script and one NIAH script as sanity checks before launching
full experiment sweeps.

```bash
# LongBench example
bash experiments/kvpress_longbench/kvpress_keydiff_vector.sh

# NIAH example
bash experiments/kvpress_niah/kvpress_keydiff_vector.sh
```

### 6) Additional experiments

| Experiment | Run | Post-process |
|---|---|---|
| Competing baselines in kvpress (ChunkKV, ExpectedAttention, AdaKV-SnapKV, TOVA) | `experiments/competing_baselines_longbench.sh` (`MODEL=...`) | `scripts/competing_baselines_aggregate.py` |
| D2O, Llama-3.1 (official repo) | `experiments/d2o_longbench.sh` | `scripts/d2o_rescore_kvpress.py` |
| D2O, Qwen3 (faithful port) | `scripts/d2o_native_longbench.py` | `scripts/competing_baselines_aggregate.py` |
| RocketKV (official repo) | `experiments/rocketkv_longbench.sh`, `experiments/rocketkv_baseline.sh` | `scripts/rocketkv_aggregate.py` |
| RULER, 16k-128k | `experiments/ruler_generate.sh`, `scripts/ruler_build_local.py`, `experiments/ruler_longcontext.sh` | `scripts/ruler_aggregate.py`, `scripts/bootstrap_ci_ruler.py` |
| SCBench multi-turn | `experiments/scbench_multiturn.sh` | `scripts/scbench_score.py`, `scripts/scbench_aggregate.py`, `scripts/bootstrap_ci_scbench.py` |
| Reconstructability-signal ablation | `experiments/ablation_reconstructability_longbench.sh` | `scripts/ablation_reconstructability_compare.py` |
| Calibration-data ablation | `experiments/calibrate_ols_ablation.sh`, `experiments/ols_calib_ablation.sh` | `scripts/bootstrap_ci_longbench.py results/ols_ablation` |
| Latency / memory | `scripts/systems_benchmark.py` | — |

SnapKV / PyramidKV (and their VECTOR variants) and the SnapKV-family competitors (ChunkKV,
AdaKV-SnapKV, TOVA) are query-aware methods and run with kvpress's `--query_aware`; the runners
set this automatically and suffix the run directories with `__query_aware`.

The reconstructability diagnostic uses the `keydiff_vector_diag` press with
`--dump_vector_stats`; aggregate it with `scripts/ablation_reconstructability_diag.py`.

#### Third-party repositories

D2O, RocketKV and RULER are cloned to the repository root at the commits below. Copy the
files under `patches/<repo>/` over the checkout (same relative paths), as for kvpress.

```bash
git clone https://github.com/AIoT-MLSys-Lab/D2O.git ./D2O
git -C D2O checkout 69db7d7e59393a161d3e2f9099a4ab72088e05a2
cp -r patches/D2O/. D2O/

git clone https://github.com/NVlabs/RocketKV.git ./RocketKV
git -C RocketKV checkout 63637c8eb4b06b5eb83cd687723b0eeb19f48f65
cp -r patches/RocketKV/. RocketKV/

git clone https://github.com/NVIDIA/RULER.git ./RULER
git -C RULER checkout c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
```

The patches leave each method's algorithm unchanged. They align the inputs with kvpress
(the same LongBench rows and chat-template prompts, verified token-identical by
`scripts/rocketkv_protocol_check.py`), add Llama-3.1 / Qwen3 support, and bind RocketKV's
permanently retained KV to the compression ratio `p_c`.

Each repository runs in its own virtual environment:

- `D2O/.venv-d2o`: Python 3.10, torch 2.1.2, transformers 4.37.2 (D2O's `requirements.txt`
  without flash-attn / deepspeed).
- `RocketKV/.venv-rocketkv`: Python 3.11, torch 2.4.1, transformers 4.45.2, flash-attn 2.6.3
  (RocketKV's `requirements.txt`); used for Llama-3.1.
- `RocketKV/.venv-rocketkv-457`: the same with transformers 4.57.6 and tree-sitter 0.21.3; used
  for Qwen3 (`PY=RocketKV/.venv-rocketkv-457/bin/python MODEL_TAG=qwen3-14b`) and for SCBench
  scoring.
- RULER data generation runs in the main `.venv` with RULER's generation dependencies
  (`wonderwords nltk bs4 tenacity tiktoken html2text`); download its source data as described in
  RULER's README.

RocketKV reads the kvpress LongBench rows from local JSONL files:

```bash
python scripts/rocketkv_dump_longbench.py RocketKV/dataset/longbench_kvproto
```

### Notes

- The experiment scripts call `uv run ...` internally, so `uv sync` is required.
- Download any benchmark data required by the selected script before large runs.
- For full sweeps, run the other scripts under `experiments/kvpress_longbench/`,
  and `experiments/kvpress_niah/`.

## License

The code in this repository is released under the MIT License (see `LICENSE`), except for
files that carry their own license header, which remain under that license.

Files under `patches/` are modified copies of third-party code and keep their original
licenses and copyright notices:

- `patches/kvpress/`: [kvpress](https://github.com/NVIDIA/kvpress), Apache License 2.0
  (`patches/kvpress/LICENSE`).
- `patches/D2O/`: [D2O](https://github.com/AIoT-MLSys-Lab/D2O), Apache License 2.0
  (`patches/D2O/LICENSE`). `src/vector/baselines/d2o_native.py` is adapted from D2O and is
  also under the Apache License 2.0.
- `patches/RocketKV/`: [RocketKV](https://github.com/NVlabs/RocketKV), NVIDIA License
  (`patches/RocketKV/LICENSE`), which permits use for non-commercial (research or evaluation)
  purposes only; it includes code from longctx_bench under the MIT License
  (`patches/RocketKV/third-party-licenses/`). These files are distributed under the same terms.

Modified third-party files are marked with a "Modified by the VECTOR authors" notice.

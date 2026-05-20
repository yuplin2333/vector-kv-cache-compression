# VECTOR

VECTOR is a plug-in augmentation for eviction-based KV cache compression.
It adds a three-way token routing strategy (retain, approximate, evict) with
offline-calibrated value reconstruction from keys.

## Repository Layout

- `src/vector/`: core implementation (KVPress integration, calibration, model adapters)
- `patches/kvpress/`: KVPress evaluation patches and registry extensions
- `experiments/`: calibration and KVPress experiment scripts
- `scripts/`: calibration and result post-processing utilities

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
# Clone KVPress to ./kvpress
git clone https://github.com/NVIDIA/kvpress.git ./kvpress
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
# K -> V calibration (main path used by VECTOR)
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

### Notes

- The experiment scripts call `uv run ...` internally, so `uv sync` is required.
- Download any benchmark data required by the selected script before large runs.
- For full sweeps, run the other scripts under `experiments/kvpress_longbench/`,
  and `experiments/kvpress_niah/`.

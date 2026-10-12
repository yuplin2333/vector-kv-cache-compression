"""
VECTOR: Value Estimation via Collinearity and Three-way Orthogonal Routing.

Offline calibration produces per-layer projection matrices W_K, W_V that map
the attention output at the end of each 32-token chunk (o_end) to reconstructed
K and V tensors.  At inference time, chunks with low reconstruction error are
compressed by storing only o_end and recomputing K/V on-the-fly.
"""

from vector.config import CompressConfig, EvalConfig, ModelConfig, ProbeConfig

__all__ = [
    "ModelConfig",
    "ProbeConfig",
    "CompressConfig",
    "EvalConfig",
]


def main() -> None:
    print("VECTOR: use scripts/calibrate.py and experiments/kvpress_*/ scripts")

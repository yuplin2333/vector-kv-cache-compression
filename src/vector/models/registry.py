"""
Auto-detect ModelAdapter from HF model name prefix.
"""

from __future__ import annotations

from vector.models.base import ModelAdapter

# Ordered list: first matching prefix wins
_REGISTRY: list[tuple[str, type[ModelAdapter]]] = []


def _register() -> None:
    from vector.models.gemma3 import Gemma3Adapter
    from vector.models.llama3 import Llama3Adapter
    from vector.models.qwen3 import Qwen3Adapter

    _REGISTRY.extend(
        [
            ("Qwen/Qwen3-", Qwen3Adapter),
            ("meta-llama/Llama-3.1-", Llama3Adapter),
            ("meta-llama/Meta-Llama-3-", Llama3Adapter),
            ("google/gemma-3-", Gemma3Adapter),
        ]
    )


def get_adapter(model_name: str) -> ModelAdapter:
    """Return the appropriate adapter instance for a given HF model name."""
    if not _REGISTRY:
        _register()
    for prefix, cls in _REGISTRY:
        if model_name.startswith(prefix):
            return cls()
    raise ValueError(
        f"No adapter found for model {model_name!r}. "
        f"Supported prefixes: {[p for p, _ in _REGISTRY]}"
    )

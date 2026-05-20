"""
Tokenizer/model tokenization helpers.
"""

from __future__ import annotations

from transformers import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def ensure_padding_config(
    tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel
) -> None:
    """
    Ensure tokenizer/model padding fields are always valid.
    """
    if tokenizer.pad_token is None:
        assert tokenizer.eos_token is not None, (
            "tokenizer.pad_token is None and tokenizer.eos_token is None; "
            "cannot set a valid padding token"
        )
        tokenizer.pad_token = tokenizer.eos_token

    pad_id = tokenizer.pad_token_id
    assert isinstance(pad_id, int), "tokenizer.pad_token_id must be int"

    if model.config.pad_token_id is None:
        model.config.pad_token_id = pad_id
    assert model.config.pad_token_id is not None, (
        "model.config.pad_token_id must not be None"
    )

    generation_cfg = model.generation_config
    assert generation_cfg is not None, "model.generation_config must not be None"
    if generation_cfg.pad_token_id is None:
        generation_cfg.pad_token_id = model.config.pad_token_id
    assert generation_cfg.pad_token_id is not None, (
        "model.generation_config.pad_token_id must not be None"
    )

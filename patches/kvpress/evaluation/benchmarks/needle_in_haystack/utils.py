# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by the VECTOR authors: VECTOR presses and ablation presses registered, local RULER loading.

import logging
import random
from pathlib import Path
from typing import Optional

import pandas as pd
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Canonical Kamradt NIAH needle / question (matches alessiodevoto/paul_graham_essays)
NIAH_NEEDLE = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
NIAH_QUESTION = "What is the best thing to do in San Francisco?"
NIAH_ANSWER_PREFIX = "The best thing to do in San Francisco is"
NIAH_MAX_NEW_TOKENS = 40


def load_multi_haystack_df(
    essays_dir: str | Path,
    n_haystacks: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a DataFrame with n_haystacks rows, each a different shuffled concatenation
    of all Paul Graham essays in essays_dir. Needle/question are the canonical Kamradt
    NIAH values. The contexts are very long and will be truncated by
    insert_needle_in_haystack to the target max_context_length.

    Parameters
    ----------
    essays_dir : str | Path
        Directory containing Paul Graham essay .txt files.
    n_haystacks : int
        Number of independent haystacks (rows) to create.
    seed : int
        Base random seed for reproducible shuffles.
    """
    essays_dir = Path(essays_dir)
    texts = [
        p.read_text(encoding="utf-8", errors="replace")
        for p in sorted(essays_dir.glob("*.txt"))
    ]
    assert len(texts) > 0, f"No .txt files found in {essays_dir}"
    logger.info(f"Loaded {len(texts)} essays from {essays_dir}; building {n_haystacks} haystacks.")

    rng = random.Random(seed)
    rows = []
    for _ in range(n_haystacks):
        shuffled = texts[:]
        rng.shuffle(shuffled)
        rows.append(
            {
                "context": "\n\n".join(shuffled),  # truncated later by insert_needle_in_haystack
                "needle": NIAH_NEEDLE,
                "question": NIAH_QUESTION,
                "answer_prefix": NIAH_ANSWER_PREFIX,
                "max_new_tokens": NIAH_MAX_NEW_TOKENS,
            }
        )
    return pd.DataFrame(rows)


def insert_needle_in_haystack(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    max_context_length: int,
    needle_depth: int | list[int],
    context_wrapper: str = "This is a very long story book: <book> {context} </book>.",
    needle_text: Optional[str] = None,
    answer_prefix: Optional[str] = None,
    question_text: Optional[str] = None,
) -> pd.DataFrame:
    """
    Inserts the "needle" string into each haystack row in the DataFrame at specified depths.
    A new row is created for each (haystack, depth) combination.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns "context", "needle", "question", "answer_prefix",
        "max_new_tokens". Each row is treated as an independent haystack. needle_text,
        question_text, and answer_prefix are taken from the first row if not provided.
    tokenizer : PreTrainedTokenizer
        Tokenizer used to encode/decode contexts.
    max_context_length : int
        Maximum token length for the context (including needle).
    needle_depth : int | list[int]
        Depth percentage(s) in [0, 100] at which to insert the needle.
    needle_text : Optional[str]
        Override needle text. If None, uses df["needle"].iloc[0].
    answer_prefix : Optional[str]
        Override answer prefix. If None, uses df["answer_prefix"].iloc[0].
    question_text : Optional[str]
        Override question text. If None, uses df["question"].iloc[0].

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per (haystack, depth) combination.
        Backward-compatible: a single-row input df produces the same output as before.
    """
    needle_text = needle_text or df["needle"].iloc[0]
    question_text = question_text or df["question"].iloc[0]
    answer_prefix = answer_prefix or df["answer_prefix"].iloc[0]
    max_new_tokens = df["max_new_tokens"].iloc[0]

    logger.info(f"Preparing dataset for inference. Needle: {needle_text}")
    tokenized_needle = tokenizer.encode(needle_text, add_special_tokens=False)
    # Account for system prompts and other overhead
    context_length_limit = max_context_length - len(tokenized_needle) - 150

    needle_depth = [needle_depth] if isinstance(needle_depth, int) else needle_depth
    new_rows = []

    for _, row in df.iterrows():
        tokenized_context = tokenizer.encode(
            str(row["context"]), add_special_tokens=False
        )[:context_length_limit]
        for depth in needle_depth:
            needle_index = int(len(tokenized_context) * depth / 100)
            new_tokenized_context = (
                tokenized_context[:needle_index]
                + tokenized_needle
                + tokenized_context[needle_index:]
            )
            decoded_context = tokenizer.decode(new_tokenized_context, skip_special_tokens=True)
            final_context = context_wrapper.format(context=decoded_context)
            new_rows.append(
                {
                    "context": final_context,
                    "needle": needle_text,
                    "needle_depth": depth,
                    "question": question_text,
                    "answer_prefix": answer_prefix,
                    "max_new_tokens": max_new_tokens,
                }
            )

    return pd.DataFrame(new_rows)

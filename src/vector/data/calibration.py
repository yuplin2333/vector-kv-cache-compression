"""
C4 streaming data loader for offline calibration.
"""

from __future__ import annotations

from typing import Iterator

from datasets import load_dataset


def stream_c4(
    split: str = "train",
    batch_size: int = 8,
    max_seqs: int | None = None,
) -> Iterator[list[str]]:
    """
    Yield batches of text strings from the C4 English dataset.

    Args:
        split: "train" or "validation"
        batch_size: number of texts per batch
        max_seqs: stop after yielding this many total texts (None = unlimited)
    """
    dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)
    buf: list[str] = []
    done = 0
    for sample in dataset:
        buf.append(sample["text"])
        if len(buf) < batch_size:
            continue
        yield buf[:batch_size]
        buf = buf[batch_size:]
        done += batch_size
        if max_seqs is not None and done >= max_seqs:
            return

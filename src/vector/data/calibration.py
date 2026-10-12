"""
Streaming text loaders for offline calibration (C4 = paper default; WikiText-103 = domain ablation).
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


def _wikitext_articles(split: str) -> Iterator[str]:
    """Rebuild whole WikiText-103 articles from its line-level rows.

    An article title is a row " = Title = " (single "=", not a "= =" section header) with an empty
    row on BOTH sides. The two-sided blank check matters: table-legend rows such as
    " = Position ; GP = " have the same shape but sit between non-empty rows, and must stay
    inside the current article.
    """
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    rows: list[str] = dataset["text"]
    lines: list[str] = []
    for idx, text in enumerate(rows):
        stripped = text.strip()
        is_title = (
            stripped.startswith("= ")
            and stripped.endswith(" =")
            and not stripped.startswith("= =")
            and idx > 0
            and rows[idx - 1] == ""
            and idx + 1 < len(rows)
            and rows[idx + 1] == ""
        )
        if is_title and "".join(lines).strip():
            yield "".join(lines)
            lines = []
        lines.append(text)
    assert "".join(lines).strip(), "trailing WikiText-103 article is empty"
    yield "".join(lines)


def stream_wikitext103(
    split: str = "train",
    batch_size: int = 8,
    max_seqs: int | None = None,
) -> Iterator[list[str]]:
    """Yield batches of whole WikiText-103 articles (same contract as stream_c4)."""
    buf: list[str] = []
    done = 0
    for article in _wikitext_articles(split):
        buf.append(article)
        if len(buf) < batch_size:
            continue
        yield buf[:batch_size]
        buf = buf[batch_size:]
        done += batch_size
        if max_seqs is not None and done >= max_seqs:
            return


CALIB_STREAMS = {"c4": stream_c4, "wikitext103": stream_wikitext103}

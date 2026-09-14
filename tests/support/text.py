"""A whole text family — a tokenizer and a model — written to disk, so a test needs no hub.

``from_pretrained`` reads a directory as readily as a hub id, and a family small enough to build
costs nothing to write: measured, 2736 parameters in 40 ms and read back in 20 ms. That is what lets
the tests of a text run exercise the real library rather than a stand-in, and still run inside the
gate, which is the suite minus everything that reaches for a hub.
"""

from __future__ import annotations

import json
from pathlib import Path

WORDS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "brown", "dog", "cat", "on", "the", "porch")
"""Enough of a vocabulary to write sentences of different lengths with, which is what padding is read from."""

WIDTH = 16
"""What this family's encoder publishes, and therefore what a head built over it is sized by."""


def text_family(home: Path, *, width: int = WIDTH) -> Path:
    """A directory holding a tokenizer and a tiny BERT, ready for ``model_name`` to name it."""
    from transformers import BertConfig, BertModel

    home.mkdir(parents=True, exist_ok=True)
    (home / "vocab.txt").write_text("\n".join(WORDS) + "\n", encoding="utf-8")
    (home / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "BertTokenizer"}), encoding="utf-8")
    config = BertConfig(
        vocab_size=len(WORDS),
        hidden_size=width,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=width,
        max_position_embeddings=32,
    )
    BertModel(config).save_pretrained(home)
    return home

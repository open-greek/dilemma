"""Batched tokenization with subword-to-word mapping.

Replicates gr-nlp-toolkit's preprocessing exactly:
- NFD normalization + strip combining marks (category Mn) + lowercase
- HuggingFace BERT tokenizer for nlpaueb/bert-base-greek-uncased-v1
"""

import unicodedata
from typing import NamedTuple

import torch
from transformers import AutoTokenizer

from ._revisions import GREEK_BERT_REV


_TOKENIZER_NAME = "nlpaueb/bert-base-greek-uncased-v1"


def strip_accents_and_lowercase(s: str) -> str:
    """Exact replica of gr-nlp-toolkit's preprocessing."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    ).lower()


class BatchEncoding(NamedTuple):
    """Result of batch tokenization."""
    input_ids: torch.Tensor       # (batch, padded_seq_len)
    attention_mask: torch.Tensor  # (batch, padded_seq_len)
    word_masks: list[list[bool]]  # per-sentence first-subword masks
    subword2word: list[dict]      # per-sentence {subword_idx -> word_idx}
    word_forms: list[list[str]]   # per-sentence normalized word forms (stripped+lowered)
    raw_forms: list[list[str]]    # per-sentence original word forms (polytonic)


def _get_tokenizer():
    """Lazy-load and cache the tokenizer."""
    if not hasattr(_get_tokenizer, "_tok"):
        _get_tokenizer._tok = AutoTokenizer.from_pretrained(
            _TOKENIZER_NAME, revision=GREEK_BERT_REV
        )
    return _get_tokenizer._tok


def batch_tokenize(
    sentences: list[str],
    max_length: int = 512,
) -> BatchEncoding:
    """Tokenize a batch of sentences with padding and word mapping.

    Args:
        sentences: List of raw text sentences.
        max_length: Maximum subword sequence length (BERT limit).

    Returns:
        BatchEncoding with input_ids, attention_mask, word_masks,
        subword2word mappings, and original word forms.
    """
    tokenizer = _get_tokenizer()

    # Pre-split into whitespace words and normalize each word, then tokenize
    # with is_split_into_words=True. This keeps word boundaries identical to
    # sentence.split(): even when a word contains punctuation or an elision
    # mark (which BERT's basic tokenizer would otherwise split into extra
    # tokens), every resulting subword keeps its source-word index via
    # word_ids(), so word_forms / raw_forms / masks stay aligned. The old
    # approach inferred word boundaries from "##" prefixes and zipped
    # raw_forms by position, which drifted on any punctuation/elision split.
    batch_words = [s.split() for s in sentences]
    batch_norm = [
        [strip_accents_and_lowercase(w) for w in words] for words in batch_words
    ]

    enc = tokenizer(
        batch_norm,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    )

    all_word_masks = []
    all_s2w = []
    all_forms = []
    all_raw_forms = []

    for i in range(len(sentences)):
        word_ids = enc.word_ids(i)  # per-subword source-word index (None=special)

        mask = []
        s2w = {}
        present = []  # source-word indices, in order, that survived truncation
        seq = 0       # 1-based sequential word index (matches decode's head refs)
        prev = None
        for j, wid in enumerate(word_ids):
            if wid is None:
                mask.append(False)
                s2w[j] = 0  # special tokens -> root
            else:
                if wid != prev:
                    seq += 1
                    mask.append(True)
                    present.append(wid)
                else:
                    mask.append(False)
                s2w[j] = seq
                prev = wid

        all_word_masks.append(mask)
        all_s2w.append(s2w)
        all_forms.append([batch_norm[i][w] for w in present])
        all_raw_forms.append([batch_words[i][w] for w in present])

    return BatchEncoding(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        word_masks=all_word_masks,
        subword2word=all_s2w,
        word_forms=all_forms,
        raw_forms=all_raw_forms,
    )

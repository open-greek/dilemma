#!/usr/bin/env python3
"""Polytonic Modern Greek (Katharevousa-era) sentence loader for the
next-word LM pipeline.

Source
------
glossAPI/Wikisource_Greek_texts on HuggingFace Hub (5,394 texts, ~130M
chars). The parquet is expected to already be in the local HF cache
(``~/.cache/huggingface/hub/datasets--glossAPI--Wikisource_Greek_texts``).
If it is not, the upstream ``build/build_polytonic_freq.py`` script is
the one that seeded that cache; run that first.

Register
--------
Wikisource Greek bundles three disjoint registers under one roof:

    1. Ancient Greek originals (Plato, Demosthenes, Homer, ...). These
       overlap with GLAUx and Diorisis and do NOT add new signal to a
       Katharevousa slice, so we drop them by filtering out any doc
       whose ``author_year`` mentions ``π.Χ`` (B.C.).
    2. Byzantine / late-antique Greek (Plutarch, Lucian, Gregory of
       Nazianzus, Agathias, ...). Stylistically close to classical. Also
       dropped here; they are not what "Katharevousa-era" means and they
       don't match the keyboard's primary typing surface.
    3. Post-1800 authors writing in polytonic script. This is the slice
       we want: Papadiamantis, Roidis, Souris, Papantoniou, Mavilis,
       Karyotakis, and 19th-c. Katharevousa translations of European
       fiction. Some of these poets are stylistic Demotic; they are still
       polytonic in Unicode and still a decent proxy for the kind of
       text a Tonos user might compose.

Filter
------
A doc is kept iff:

  * ``author_year`` does not mention ``π.Χ``.
  * ``author_year`` contains at least one 18xx/19xx year (so we drop
    uncategorized ancient authors whose years are only expressed as
    ``περ. 175 - περ. 235``).
  * The doc has at least 20 Greek words and >= 40% of them carry at
    least one polytonic code point in U+1F00-U+1FFF. This removes the
    handful of modern Demotic monotonic documents that snuck into
    Wikisource.

Sentence segmentation
---------------------
Wikisource text is paragraph-level prose. We split on ``.`` ``;`` ``!``
``·`` ``?`` plus newlines with double-break. Greek `;` is a question
mark; keep it as a sentence end. The output matches the GLAUx loader's
contract: ``BOS`` prepended, ``EOS`` appended, polytonic NFC preserved,
non-Greek tokens dropped.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple

BOS_TOK = "<s>"
EOS_TOK = "</s>"

# Polytonic Greek code point ranges
GREEK_LETTER_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]+")
POLYTONIC_RE = re.compile(r"[\u1F00-\u1FFF]")

# Sentence-splitter: keep delimiters so we can emit one sentence per chunk.
# ; is Greek question mark, · is Greek ano teleia (semicolon).
SENT_SPLIT_RE = re.compile(r"([.!?;·]+)")

# A "word" token for the LM: runs of Greek letters, optionally with a
# medial apostrophe (elision) or hyphen. We tokenize greedily and then
# re-filter by is_polytonic-Greek.
WORD_RE = re.compile(
    r"[\u0370-\u03FF\u1F00-\u1FFF]+"
    r"(?:[\u2019'’-][\u0370-\u03FF\u1F00-\u1FFF]+)*"
)

DEFAULT_PARQUET = Path.home() / (
    ".cache/huggingface/hub/datasets--glossAPI--Wikisource_Greek_texts/"
    "snapshots/5590ddc5476c6d272eda114639d1e9952e20a1d8/"
    "wikisource_greek_deduped.parquet"
)

_BC_MARK = "π.Χ"  # Greek B.C. abbreviation as used by Wikisource
_MODERN_YEAR_RE = re.compile(r"\b(?:18\d\d|19\d\d|20\d\d)\b")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _is_greek_token(s: str) -> bool:
    return bool(GREEK_LETTER_RE.search(s))


def _polytonic_ratio(text: str) -> tuple[int, int]:
    """Return (polytonic_word_count, total_greek_word_count)."""
    t = nfc(text)
    total = 0
    poly = 0
    for m in GREEK_LETTER_RE.finditer(t):
        total += 1
        if POLYTONIC_RE.search(m.group()):
            poly += 1
    return poly, total


def _passes_register_filter(
    author_year, text, min_words: int = 20,
    min_poly_ratio: float = 0.4,
) -> bool:
    if not isinstance(author_year, str):
        return False
    if not isinstance(text, str) or not text:
        return False
    if _BC_MARK in author_year:
        return False
    if not _MODERN_YEAR_RE.search(author_year):
        return False
    poly, total = _polytonic_ratio(text)
    if total < min_words:
        return False
    return (poly / total) >= min_poly_ratio


def _split_sentences(text: str) -> Iterator[str]:
    """Chop a paragraph-structured text into sentence candidates.

    We split on sentence-ending punctuation and on blank lines. Each
    yielded sentence is a raw substring; token extraction happens in the
    caller.
    """
    for sent, _closer in split_sentences_with_closers(text):
        yield sent


def split_sentences_with_closers(text: str) -> Iterator[tuple[str, str]]:
    """:func:`_split_sentences`, with the punctuation that closed each
    sentence when it follows the last word directly (``τόμ.``), else ''.

    A word right before a period may be an abbreviation (τόμ., σελ.),
    which a word list must not take for a word.
    """
    if not text:
        return
    # Normalize to NFC once up front so composed forms match the LM vocab.
    text = nfc(text)
    # Collapse paragraph separators to a single newline so the split
    # regex below treats them as sentence-ending.
    # A blank line is also a sentence end.
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        # Split on punctuation, keeping the delimiters attached to the
        # preceding chunk (for symmetry with GLAUx, which discards them).
        parts = SENT_SPLIT_RE.split(para)
        # parts is [text, punct, text, punct, ..., text]
        buf: list[str] = []
        for i, p in enumerate(parts):
            if not p:
                continue
            if i % 2 == 0:
                # text chunk
                buf.append(p)
            else:
                # punctuation chunk; close off the sentence here
                sent = "".join(buf).strip()
                if sent:
                    closer = p if not buf[-1][-1:].isspace() else ""
                    yield sent, closer
                buf = []
        tail = "".join(buf).strip()
        if tail:
            yield tail, ""


def _tokenize(sentence: str) -> list[str]:
    """Extract Greek-letter tokens from a sentence, NFC-normalized.

    Non-Greek tokens (Latin, numerals, purely-punctuation) are dropped,
    matching the behaviour of ``train_lm.iter_glaux_sentences``.
    """
    toks: list[str] = []
    for m in WORD_RE.finditer(sentence):
        tok = nfc(m.group()).lower()
        if _is_greek_token(tok):
            toks.append(tok)
    return toks


class PolytonicMGDocument(NamedTuple):
    """One Wikisource document that passes the register filter."""

    key: str            # url, else ``author::title``; stable across runs
    author: str
    title: str
    author_year: str
    text: str


def sentence_id(doc_key: str, index: int) -> str:
    """The sentence id the LM's train/dev split hashes.

    ``index`` is the sentence's ordinal among everything
    :func:`_split_sentences` yields for the document, counting the short
    sentences the LM skips, so ids are stable whatever filter a caller
    applies afterwards.
    """
    return f"polymg:{doc_key}:{index}"


def iter_polytonic_mg_documents(
    parquet_path: Path | None = None,
    min_poly_ratio: float = 0.4,
    max_docs: int | None = None,
) -> Iterator[PolytonicMGDocument]:
    """Yield every Wikisource document that passes the register filter.

    This is the one place that decides which documents form the slice, so
    the language model (:func:`iter_polytonic_mg_sentences`) and the
    polytonic Modern Greek word list (``export_mg_polytonic.py``) read the
    same documents under the same keys.
    """
    import pandas as pd

    path = Path(parquet_path or DEFAULT_PARQUET)
    if not path.exists():
        raise SystemExit(
            f"Polytonic MG parquet not found at {path}.\n"
            "Run `python build/build_polytonic_freq.py --stats` once to "
            "seed the HuggingFace cache, or pass --mg-parquet explicitly."
        )
    df = pd.read_parquet(path)

    if max_docs is not None:
        df = df.head(max_docs)

    def field(row, name: str) -> str:
        value = row.get(name, "")
        return value if isinstance(value, str) else ""

    for _, row in df.iterrows():
        text = row.get("text", "")
        author_year = row.get("author_year", "")
        if not isinstance(text, str) or not text:
            continue
        if not _passes_register_filter(
            author_year, text, min_poly_ratio=min_poly_ratio
        ):
            continue
        # Stable doc key: url if present, else title+author.
        url, title, author = field(row, "url"), field(row, "title"), field(row, "author")
        yield PolytonicMGDocument(
            key=url or f"{author}::{title}",
            author=author,
            title=title,
            author_year=author_year,
            text=text,
        )


def iter_polytonic_mg_sentences(
    parquet_path: Path | None = None,
    min_poly_ratio: float = 0.4,
    min_sentence_tokens: int = 3,
    max_docs: int | None = None,
) -> Iterable[tuple[str, list[str]]]:
    """Yield ``(sentence_id, tokens)`` for every kept Wikisource sentence.

    ``sentence_id`` is globally unique and stable across runs, which
    lets the train/dev splitter in ``train_lm.py`` bucket it
    deterministically. We key it on ``polymg:{url_or_title}:{i}`` where
    ``i`` is the sentence ordinal within the document.
    """
    for doc in iter_polytonic_mg_documents(
        parquet_path, min_poly_ratio=min_poly_ratio, max_docs=max_docs
    ):
        for i, sent in enumerate(_split_sentences(doc.text)):
            toks = _tokenize(sent)
            if len(toks) < min_sentence_tokens:
                continue
            yield sentence_id(doc.key, i), [BOS_TOK] + toks + [EOS_TOK]


def summarize(parquet_path: Path | None = None) -> None:
    """CLI helper: print corpus stats for sanity."""
    import pandas as pd

    path = Path(parquet_path or DEFAULT_PARQUET)
    df = pd.read_parquet(path)
    n_docs_kept = 0
    n_sent = 0
    n_tok = 0
    samples: list[list[str]] = []
    for i, (_sid, toks) in enumerate(
        iter_polytonic_mg_sentences(parquet_path=path)
    ):
        n_sent += 1
        n_tok += len(toks)
        if len(samples) < 5:
            samples.append(toks)
    # re-count docs by a second pass (cheap)
    for _, row in df.iterrows():
        if _passes_register_filter(row.get("author_year", ""), row.get("text", "")):
            n_docs_kept += 1
    print(f"source parquet     : {path}")
    print(f"docs total         : {len(df):,}")
    print(f"docs kept (Kathar.): {n_docs_kept:,}")
    print(f"sentences yielded  : {n_sent:,}")
    print(f"tokens (incl BOS/EOS): {n_tok:,}")
    print()
    print("sample sentences:")
    for toks in samples:
        print("  " + " ".join(toks))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Override the Wikisource parquet path.",
    )
    ap.add_argument(
        "--summarize",
        action="store_true",
        help="Print stats and a few sample sentences, then exit.",
    )
    args = ap.parse_args()
    if args.summarize:
        summarize(args.parquet)
    else:
        # Streaming dump useful for spot checks via `head`.
        for sid, toks in iter_polytonic_mg_sentences(
            parquet_path=args.parquet
        ):
            print(sid + "\t" + " ".join(toks))

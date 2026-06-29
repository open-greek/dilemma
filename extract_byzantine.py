"""Byzantine vernacular corpus loader for the next-word LM pipeline.

Factored out of ``train_lm.py`` to stay merge-friendly alongside other
corpus ingestors (Diorisis, polytonic MG) landing on sibling branches.

The corpus lives at ``~/Documents/corpus-of-open-greek/sources/byzantine/``
and consists of medieval / early-modern Greek vernacular literature
(Digenes Akritas, Ptochoprodromika, Poulologos, Apokopos, etc.). Most
files are polytonic; two of the largest (Erotokritos and Chronicle of
the Moreas) are *monotonic*. The polytonic LM artifact should not
learn monotonic forms as parallel vocab entries, so those two files
are filtered out by default. The threshold is expressed as a
polytonic-character ratio on the file contents, not a hard-coded
blacklist, so adding new texts to the corpus doesn't need code
changes.

Interface mirrors ``iter_glaux_sentences`` in ``train_lm.py``: yields
``(sent_id, [token, ...])`` tuples where ``tokens[0] == BOS_TOK`` and
``tokens[-1] == EOS_TOK``. Punctuation is collapsed to ``</s>`` for
sentence-enders and dropped otherwise, matching the GLAUx pass.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator

# Match the reserved tokens in train_lm.py. Duplicated here so this
# module stays importable without a circular dependency.
BOS_TOK = "<s>"
EOS_TOK = "</s>"

SENT_END = {".", ";", "·", "!", "?"}

DEFAULT_CORPUS_DIR = (
    Path.home() / "Documents" / "corpus-of-open-greek" / "sources" / "byzantine"
)

# Files that are substantially monotonic. Computed by measuring the
# ratio of polytonic-only codepoints to all Greek codepoints per file,
# thresholded at ~2%. Erotokritos is 0.0% polytonic, Chronikon tou
# Moreos is 0.6% (stray polytonic quotations inside a monotonic
# edition). Everything else is 5-17% polytonic.
#
# We compute this at load time rather than hard-coding so the filter
# keeps working if the corpus directory grows.
MONOTONIC_THRESHOLD = 0.02  # <2% polytonic codepoints => monotonic file

_POLYTONIC_BLOCK_LO = "\u1F00"
_POLYTONIC_BLOCK_HI = "\u1FFF"
_GREEK_BLOCK_LO = "\u0370"
_GREEK_BLOCK_HI = "\u03FF"

# Combining diacritics that indicate polytonic orthography. Tonos
# (U+0301) and dialytika (U+0308) appear in both polytonic and
# monotonic Greek, so they don't count.
_POLYTONIC_COMBINING = {
    "\u0313",  # combining comma above (spiritus lenis)
    "\u0314",  # combining reversed comma above (spiritus asper)
    "\u0342",  # combining Greek perispomeni (circumflex)
    "\u0345",  # combining ypogegrammeni (iota subscript)
    "\u0300",  # combining grave (varia)
}


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _polytonic_share(text: str) -> float:
    """Fraction of Greek codepoints that are polytonic-only.

    Counts precomposed polytonic glyphs in the U+1F00-U+1FFF block
    plus any Greek characters bearing a polytonic combining diacritic
    in NFD form, divided by the total Greek-letter count.
    """
    if not text:
        return 0.0
    greek = 0
    poly_precomposed = 0
    for c in text:
        if _GREEK_BLOCK_LO <= c <= _GREEK_BLOCK_HI:
            greek += 1
        elif _POLYTONIC_BLOCK_LO <= c <= _POLYTONIC_BLOCK_HI:
            greek += 1
            poly_precomposed += 1
    if greek == 0:
        return 0.0

    # decomposed combining marks
    comb_poly = 0
    for c in unicodedata.normalize("NFD", text):
        if c in _POLYTONIC_COMBINING:
            comb_poly += 1
    return (poly_precomposed + comb_poly) / greek


def classify_files(
    corpus_dir: Path,
) -> tuple[list[Path], list[Path]]:
    """Split corpus files into polytonic and monotonic buckets.

    ``corpus.txt`` is the concatenation of the per-work files, so
    including it would double-count. Any file that isn't a ``.txt``
    running-text file is skipped.
    """
    polytonic: list[Path] = []
    monotonic: list[Path] = []
    for p in sorted(corpus_dir.glob("*.txt")):
        if p.name == "corpus.txt":
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        share = _polytonic_share(text)
        if share >= MONOTONIC_THRESHOLD:
            polytonic.append(p)
        else:
            monotonic.append(p)
    return polytonic, monotonic


# Token regex: grab contiguous runs of Greek letters (both blocks),
# allowing medial apostrophe / modifier letter apostrophe for elided
# forms like κ᾿, ν᾿, σ᾿ which appear frequently in vernacular texts.
# A single leading Latin capital that visually matches a Greek capital
# (M, T, A, ...) is tolerated because source files occasionally
# contain those as transcription artifacts. We fix them up to the
# Greek letter after matching.
_APOSTROPHES = "'\u02BC\u1FBD\u1FBF\u2019\u1FBE"  # ' ʼ ᾽ ᾿ ’ ι

_LATIN_HOMOGLYPHS = {
    "A": "Α", "B": "Β", "E": "Ε", "Z": "Ζ", "H": "Η", "I": "Ι",
    "K": "Κ", "M": "Μ", "N": "Ν", "O": "Ο", "P": "Ρ", "T": "Τ",
    "X": "Χ", "Y": "Υ",
}

_TOKEN_RE = re.compile(
    rf"[{''.join(_LATIN_HOMOGLYPHS)}]?"
    rf"[\u0370-\u03FF\u1F00-\u1FFF]+"
    rf"(?:[{_APOSTROPHES}][\u0370-\u03FF\u1F00-\u1FFF]*)*"
    rf"|[{_APOSTROPHES}][\u0370-\u03FF\u1F00-\u1FFF]+"
)

_SENT_SPLIT_RE = re.compile(r"([.;·!?])")


def _line_is_metadata(line: str) -> bool:
    """Heuristic: lines with more Latin letters than Greek are editor
    metadata (page headers, citation blocks, "Επιμέλεια Wilhelm
    Wagner, ..." notes, verse numbers). Strip them before splitting
    into sentences so they don't glue onto the first real sentence.
    """
    greek = 0
    latin = 0
    for c in line:
        if ("A" <= c <= "Z") or ("a" <= c <= "z"):
            latin += 1
        elif (_GREEK_BLOCK_LO <= c <= _GREEK_BLOCK_HI
              or _POLYTONIC_BLOCK_LO <= c <= _POLYTONIC_BLOCK_HI):
            greek += 1
    return latin > greek and latin > 3


def _fix_latin_homoglyphs(token: str) -> str:
    """Replace a leading Latin homoglyph with its Greek twin."""
    if token and token[0] in _LATIN_HOMOGLYPHS:
        return _LATIN_HOMOGLYPHS[token[0]] + token[1:]
    return token


def _is_greek_token(t: str) -> bool:
    for c in t:
        if (_GREEK_BLOCK_LO <= c <= _GREEK_BLOCK_HI
                or _POLYTONIC_BLOCK_LO <= c <= _POLYTONIC_BLOCK_HI):
            return True
    return False


def _sentence_looks_polytonic(tokens: list[str]) -> bool:
    """Reject purely-monotonic sentences inside a polytonic-classified
    file. Editor metadata paragraphs at the top of some files are
    monotonic Greek prose ("Σημείωση: μετά τους στίχους..."). A real
    sentence from a polytonic source contains at least one token with
    a polytonic diacritic.
    """
    for t in tokens:
        for c in t:
            if _POLYTONIC_BLOCK_LO <= c <= _POLYTONIC_BLOCK_HI:
                return True
        # also check combining diacritics
        for c in unicodedata.normalize("NFD", t):
            if c in _POLYTONIC_COMBINING:
                return True
    return False


def _iter_raw_sentences(text: str) -> Iterator[list[str]]:
    """Split text into sentences by punctuation.

    Editor metadata lines (mostly Latin script) are dropped before the
    punctuation split so they don't glue onto the first real
    sentence.
    """
    kept = []
    for line in text.split("\n"):
        if _line_is_metadata(line):
            continue
        kept.append(line)
    flat = " ".join(kept)

    parts = _SENT_SPLIT_RE.split(flat)
    buf: list[str] = []
    for i, p in enumerate(parts):
        if i % 2 == 0:
            raw = _TOKEN_RE.findall(p)
            for t in raw:
                buf.append(_fix_latin_homoglyphs(t))
        else:
            if buf:
                yield buf
                buf = []
    if buf:
        yield buf


def iter_byzantine_sentences(
    corpus_dir: Path | None = None,
    include_monotonic: bool = False,
) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(sent_id, [token, ...])`` from the Byzantine corpus.

    Each sentence is framed with ``BOS_TOK`` / ``EOS_TOK`` so the
    output matches ``iter_glaux_sentences`` in ``train_lm.py``.

    Parameters
    ----------
    corpus_dir: location of the ``.txt`` files. Defaults to
        ``~/Documents/corpus-of-open-greek/sources/byzantine``.
    include_monotonic: if True, include files classified as monotonic
        (Erotokritos, Chronicle of the Moreas). Defaults to False
        because mixing monotonic surface forms (``της``) with
        polytonic (``τῆς``) would inflate the vocab with parallel
        entries for the same word. A polytonic keyboard LM should not
        learn monotonic spellings.
    """
    if corpus_dir is None:
        corpus_dir = DEFAULT_CORPUS_DIR
    corpus_dir = Path(corpus_dir)

    polytonic, monotonic = classify_files(corpus_dir)
    files = list(polytonic)
    if include_monotonic:
        files.extend(monotonic)

    for path in sorted(files):
        doc_id = path.stem
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        text = nfc(text)

        # Files we already classified as polytonic still contain the
        # occasional monotonic editor-note sentence; drop those so
        # they don't seed parallel monotonic vocab entries.
        is_polytonic_file = path in polytonic

        sent_counter = 0
        for raw_toks in _iter_raw_sentences(text):
            clean = [t for t in raw_toks if _is_greek_token(t)]
            if not clean:
                continue
            if (is_polytonic_file
                    and not _sentence_looks_polytonic(clean)):
                continue
            sent_counter += 1
            sent_id = f"byz:{doc_id}:{sent_counter}"
            tokens = [BOS_TOK] + clean + [EOS_TOK]
            if len(tokens) >= 3:
                yield sent_id, tokens


def corpus_stats(corpus_dir: Path | None = None) -> dict:
    """Return a summary of what the loader will emit."""
    if corpus_dir is None:
        corpus_dir = DEFAULT_CORPUS_DIR
    corpus_dir = Path(corpus_dir)
    polytonic, monotonic = classify_files(corpus_dir)

    def _count(files: Iterable[Path], require_polytonic: bool) -> tuple[int, int]:
        n_sents = 0
        n_toks = 0
        for p in files:
            text = nfc(p.read_text(encoding="utf-8"))
            for sent in _iter_raw_sentences(text):
                clean = [t for t in sent if _is_greek_token(t)]
                if not clean:
                    continue
                if require_polytonic and not _sentence_looks_polytonic(clean):
                    continue
                n_sents += 1
                n_toks += len(clean) + 2  # <s> + toks + </s>
        return n_sents, n_toks

    p_sents, p_toks = _count(polytonic, require_polytonic=True)
    m_sents, m_toks = _count(monotonic, require_polytonic=False)

    return {
        "corpus_dir": str(corpus_dir),
        "polytonic_files": [p.name for p in polytonic],
        "monotonic_files_excluded": [p.name for p in monotonic],
        "polytonic_sentences": p_sents,
        "polytonic_tokens": p_toks,
        "monotonic_sentences": m_sents,
        "monotonic_tokens": m_toks,
    }


if __name__ == "__main__":
    import json

    stats = corpus_stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print()
    print("Sample sentences:")
    for i, (sid, toks) in enumerate(iter_byzantine_sentences()):
        if i >= 5:
            break
        print(f"  {sid}: {' '.join(toks)}")

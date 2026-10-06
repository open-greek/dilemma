#!/usr/bin/env python3
"""Compile the intermediate n-gram counts from ``train_lm.py`` into the
mmap-friendly binary artifact ``build/lm/grc_ngram.bin`` consumed by
an iOS keyboard extension.

This is the on-disk contract with the Swift reader. Once a keyboard ships a
reader against this, bumping the file format requires a version bump.

BINARY FORMAT (v2)
==================

All multi-byte integers are little-endian. Offsets are byte offsets
from the start of the file, so the entire file can be mmap'd and
random-accessed without copying.

Overall layout:

    [ HEADER                      ] 128 bytes, fixed
    [ VOCAB OFFSETS               ] 4 * (V+1) bytes  -- u32 into pool
    [ VOCAB STRING POOL           ] UTF-8 bytes, sorted alphabetically
    [ VOCAB COUNTS                ] 4 * V bytes     -- u32 unigram count
    [ UNIGRAM TOP-K               ] Kuni * (4+2) bytes  -- id + logprob_q16
    [ BIGRAM CONTEXT INDEX        ] sorted by w1; see below
    [ BIGRAM SUGGESTION TABLE     ] packed top-Kbi per context
    [ TRIGRAM CONTEXT INDEX       ] sorted by (w1, w2); see below
    [ TRIGRAM SUGGESTION TABLE    ] packed top-Ktri per context
    [ PADDING                     ] optional: zero bytes to a multiple of 8
    [ OUT-OF-VOCABULARY UNIGRAMS  ] optional: u64 hash * M + u8 count * M
    [ FOOTER                      ] optional: 16 bytes, ends in "GNUO"

The last three sections are the out-of-vocabulary unigram table
described below. A file without them ends at the trigram suggestion
table, which is all a reader that does not know the table reads.

Integers:
    u16 = uint16          u32 = uint32           u64 = uint64
    i16 = int16
    q16 = int16 fixed-point: value / 1024.0 gives a natural-log prob
          in the range roughly (-32, 0]. Good to ~0.001 nat resolution.

Version 2 changes from v1
-------------------------

* Split ``top_k`` into three independent values ``top_k_uni`` /
  ``top_k_bi`` / ``top_k_tri``. The keyboard asks for up to 6
  suggestions but also filters by diacritic-blind prefix during
  mid-word completion, so bigram contexts need deeper lists than
  trigram contexts. Defaults: 10 / 30 / 15.
* Adds a ``VOCAB COUNTS`` section: one u32 unigram count per vocab
  entry, in vocab id order. Used by the Swift reader to rank global
  prefix completions by corpus frequency rather than alphabetically
  when the current bigram/trigram top-K doesn't cover the stem. Cheap
  to produce (data is already in ``unigrams.json``), cheap to carry
  (~320 KB for a 80 K vocab), and lets the mid-word completion path
  always return something useful even outside the context's top-K.

Homograph spellings kept past the cut
------------------------------------

A context keeps its ``top_k_bi`` / ``top_k_tri`` most frequent
continuations, and also, past that cut, every continuation that is one
spelling of a homograph set and was seen at least
``KEEP_HOMOGRAPH_MIN_COUNT`` (3) times in that context. A homograph
spelling is a vocabulary token whose bare letters (no accents,
breathings, iota subscript or diaeresis, final sigma as σ, lowercase,
elision mark dropped) are shared with another vocabulary token that
differs by more than an acute written as a grave, by case or by the
elision glyph: ἢ and ἡ, ἐκείνῃ and ἐκείνη, αὐτοῦ and αὑτοῦ, but not
καί and καὶ.

The keyboard scores a spelling it is choosing between by looking it up
in these rows, so a spelling cut from a row is scored by backoff while
its twin keeps its trigram or bigram probability. After τῇ the bigram
row held ἐκείνη but not ἐκείνῃ, and the dative lost to the nominative
in the one context that decides between them. Kept entries follow the
top-K entries in the same descending-count order, so the first
``top_k`` entries of every row, and with them every next-word
suggestion list, are what they were without the rule. Exporting with
``--keep-homograph-min-count 0`` turns the rule off.

Rows completed after a homograph spelling
-----------------------------------------

The rule above keeps a homograph spelling as a continuation. The other
side of the same choice is the row a homograph spelling heads: to choose
between ἢ and ἡ, a keyboard scores the word typed next after each of
them, and the cut leaves almost every next word out of both bigram rows,
so both readings fall back to that word's unigram probability and the
choice is made without evidence. The bigram row of every homograph
spelling (as a context, the same set of vocabulary tokens) therefore
also keeps, past its cut, every continuation seen at least
``HOMOGRAPH_CONTEXT_MIN_COUNT`` (3) times after it. Trigram rows are cut
as before. The completed rows stay in descending-count order, so their
first ``top_k`` entries and the next-word lists are unchanged.
Exporting with ``--homograph-context-min-count 0`` turns this off, and
with both minimums at 0 the exporter writes the plain top-K rows.

Header (128 bytes, zero-padded)
-------------------------------

    off  size  field
      0    4   magic = "GNLM"
      4    4   format_version (u32)       = 2
      8    4   order (u32)                = 3
     12    4   top_k_uni (u32)            = 10
     16    4   vocab_size (u32)            V
     20    4   id_pad  (u32)              sorted-vocab index of <PAD>
     24    4   id_unk  (u32)              sorted-vocab index of <UNK>
     28    4   id_bos  (u32)              sorted-vocab index of <s>
     32    4   id_eos  (u32)              sorted-vocab index of </s>
     36    4   top_k_bi (u32)             (v2) the bigram top-K cut
     40    8   total_tokens (u64)         training corpus size
     48    8   vocab_offsets_off (u64)
     56    8   string_pool_off (u64)
     64    8   string_pool_size (u64)
     72    8   unigram_topk_off (u64)
     80    8   bigram_index_off (u64)
     88    4   bigram_index_len (u32)      number of (w1, range) rows
     92    4   bigram_suggestions_len (u32)
     96    8   bigram_suggestions_off (u64)
    104    8   trigram_index_off (u64)
    112    4   trigram_index_len (u32)     number of (w1,w2,range) rows
    116    4   trigram_suggestions_len (u32)
    120    8   trigram_suggestions_off (u64)

v2 addendum at offsets beyond 127:  the header runs 128 bytes but
only up through 127 was ever addressed in v1. The vocab-counts
section is recorded via its own pair of u64/u32 offsets which we
squeeze into the previously-reserved slots at 36 (for top_k_bi) and
via appending counts right after the string pool (so the reader
can compute its offset from ``string_pool_off + string_pool_size``
without another header field). ``top_k_bi`` records where rows are
cut, not how long they are: a row can run past it with kept homograph
spellings, and the row of a homograph spelling with its completion. ``top_k_tri`` is not in the header at all (it is in the
``.version`` sidecar). A reader needs neither, only each row's
``suggestion_count``.

Vocab
-----

Tokens are sorted by UTF-8 byte order (so ``<`` reserved tokens sort
before real Greek). id 0..3 are reserved (PAD, UNK, BOS, EOS). The
vocab table lets the reader do binary search on a UTF-8 NFC input to
find its id.

    vocab_offsets : u32 * (V+1)     byte offsets into the string pool
    string_pool   : UTF-8 bytes

``token_i`` = ``pool[offsets[i] : offsets[i+1]]``.

Unigram top-K (global fallback)
-------------------------------

A single block of ``top_k`` entries, ordered by descending probability.
Used when the trigram and bigram contexts both miss. Each entry:

    u32 word_id
    i16 logprob_q16

(6 bytes per entry.)

Bigram / trigram layout
-----------------------

For both bigram and trigram sections:

    *_index       : context keys, sorted ascending, binary-searchable
    *_suggestions : flat array of (u32 word_id, i16 logprob_q16),
                    each context's top_k entries, kept homograph
                    spellings and, in the row of a homograph
                    spelling, its completion, in descending
                    probability

Bigram index row (per context, w1):

    u32 w1
    u32 suggestion_offset   -- index into *_suggestions (NOT byte offset)
    u16 suggestion_count    -- top_k, plus any entries kept past it
    u16 reserved

(12 bytes per row.)

Trigram index row (per context, w1, w2):

    u32 w1
    u32 w2
    u32 suggestion_offset
    u16 suggestion_count
    u16 reserved

(16 bytes per row.)

Suggestion entry (shared layout for bigram and trigram):

    u32 word_id
    i16 logprob_q16

(6 bytes per entry.)

Reserved-token handling
~~~~~~~~~~~~~~~~~~~~~~~

The four reserved tokens (<PAD>, <UNK>, <s>, </s>) are present in the
vocab so the reader can binary-search incoming context words that hit
sentence-start / unknown forms. They are **not** proposed as
suggestions: every suggestion entry in the bigram and trigram tables
points to a real Greek word. The Swift reader does not need to filter
reserved ids out of suggestion lists.

Lookup algorithm (pseudocode, matches what the keyboard needs)
-------------------------------------------------------

    def id_of(token):
        return binary_search(vocab, token)  # or UNK

    def predict(w_prev, w_curr):
        i = id_of(w_prev); j = id_of(w_curr)
        # trigram
        row = binsearch(trigram_index, (i, j))
        if row:
            return load(trigram_suggestions, row.offset, row.count)
        # bigram
        row = binsearch(bigram_index, j)
        if row:
            return load(bigram_suggestions, row.offset, row.count)
        # unigram fallback (no context)
        return load(unigram_topk, 0, TOP_K)

If the user has typed zero tokens of context, start at the bigram
lookup with ``j = BOS``.

Out-of-vocabulary unigram table (optional, after the trigram tables)
--------------------------------------------------------------------

The vocabulary holds the 80K most frequent training types, so most
spellings a keyboard proposes from its dictionary have no id, and every
spelling of a set of letters that is out of the vocabulary scores the
same (nothing). This table gives such spellings their exact training
count, which a reader uses at the unigram tier,
``α^2 * c(w) / total_tokens``, exactly as it scores a vocabulary word
that no context row holds.

It holds the training types a reader cannot resolve to a vocabulary
id, seen at least ``OOV_UNIGRAM_MIN_COUNT`` times, that a keyboard
weighs against another spelling of the same letters: types that, up to
case and the elision glyph, are a spelling of the Hunspell dictionary
(``--oov-dictionary``) whose bare letters another dictionary spelling
shares with more than a grave, case or elision-glyph difference
(:func:`spelling_key`). A type the dictionary cannot propose is never
looked up, and one it has no twin for rarely is, so both stay out to
keep the table small. A type whose lowercase or U+2019 spelling is in
the vocabulary stays out too, since a reader resolves it to that id
and never reaches the table (Κατ’ to κατ’, ὥστ᾽ to ὥστ’). Every
elision glyph is stored as U+2019, the one a reader tries, with the
counts of a word's glyph variants summed (μεθ᾿ and μεθ’ are one
entry). Every out-of-vocabulary type was seen no more often than the
rarest vocabulary word (20 times), so a key's count, at most the sum
of one word's glyph variants, stays under the u8 cap of 255.

The table's selection follows the dictionary it was built against, so
the LM must be re-exported after every rebuild of the dictionary. The
``.version`` sidecar records the dictionary's sha256
(``oov_dictionary_sha256``); ``export_lm.py --check-dictionary`` and
``export_hunspell.py`` both report when it no longer matches.

Keys are hashes rather than strings, which keeps the table at 9 bytes
an entry:

    pad       : zero bytes to the next multiple of 8
    keys      : u64 * M   FNV-1a 64-bit hash of the type's UTF-8 bytes
                          (NFC, elision glyph as U+2019), ascending
                          (binary-searchable), little-endian
    counts    : u8  * M   exact training count, capped at 255
    footer    : 16 bytes, the last in the file
                u64 table_off   byte offset of ``keys``, little-endian
                u32 M           number of entries, little-endian
                4s  magic       "GNUO"

A reader finds the table by the footer magic. The header does not
point at it, and every header offset is unchanged, so a reader that
does not know the table reads the file as before. A reader looks a
spelling up in the order it resolves vocabulary ids: the literal
spelling, then lowercased, then with its elision glyph written as
U+2019, then both. A 64-bit hash collides with one of M entries with probability
about M / 2^64 per lookup, which the table accepts in exchange for not
storing strings; the exporter checks that no two entries collide.

Probabilities
-------------

Probabilities are **Stupid Backoff** natural logs, not Kneser-Ney.
Stupid Backoff is cheap to store, known to work well for keyboard
prediction, and doesn't need backoff weights on disk. The scoring at
build time is:

    S(w | w1, w2) =    c(w1, w2, w) / c(w1, w2)            if > 0
                  = α * c(w2, w)    / c(w2)                 else-if > 0
                  = α^2 * c(w)      / N                     otherwise

with α = 0.4. The logprob stored in the artifact is simply
``log(c / denom)`` for the matching level (no alpha). The Swift reader
doesn't need alpha either; because it falls back through levels
explicitly, relative ordering within a level is what matters for top-K
suggestions.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import struct
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_DIR = SCRIPT_DIR / "build" / "lm"
VERSION_FILE = SCRIPT_DIR / "VERSION"

MAGIC = b"GNLM"
FORMAT_VERSION = 2
ORDER = 3
TOP_K_UNI = 10        # global fallback top-K
TOP_K_BI = 30         # per-bigram top-K (mid-word prefix filtering wants depth)
TOP_K_TRI = 15        # per-trigram top-K
LOGP_SCALE = 1024.0   # q16 fixed-point scale
# A homograph spelling seen at least this often in a context stays in
# that context's row past the top-K cut. 0 turns the rule off. Measured
# on held-out sentences typed as bare letters, 3 kept almost all of the
# gain of 1 (+3.24 against +3.04 corrected words per 1,000 on sentences
# with little overlap with training) for a fifth of the added size.
KEEP_HOMOGRAPH_MIN_COUNT = 3
# The bigram row of a homograph spelling keeps every continuation seen at
# least this often after it. 0 turns the rule off. Measured on the
# held-out sentences typed as bare letters, sentences with little overlap
# with training: 104 words fixed and 2 broken in 137,525 (+0.74 per
# 1,000), for 1.6 MB.
HOMOGRAPH_CONTEXT_MIN_COUNT = 3
# Glyphs that close an elided word in the corpus: U+2019 (the corpus
# spelling), U+02BC, U+1FBD (koronis), ASCII apostrophe, U+1FBF (psili).
ELISION_MARKS = "\u2019\u02bc\u1fbd'\u1fbf"
RESERVED_TOKENS = ("<PAD>", "<UNK>", "<s>", "</s>")
# Out-of-vocabulary unigram table: minimum training count of a type it
# carries; 0 leaves the table out. On the same held-out sentences the
# table corrected a further 4.5 words per 1,000 over the homograph rows
# alone, the whole gain an unrestricted in-memory count table gave.
OOV_UNIGRAM_MIN_COUNT = 1
OOV_UNIGRAM_MAGIC = b"GNUO"
OOV_COUNT_CAP = 255
DEFAULT_OOV_DICTIONARY = SCRIPT_DIR / "build" / "hunspell" / "grc_polytonic.dic"
FNV64_OFFSET = 0xCBF29CE484222325
FNV64_PRIME = 0x100000001B3


def quantize_logprob(p: float) -> int:
    """Clamp logprob to i16 fixed-point. p is a natural-log probability."""
    q = int(round(p * LOGP_SCALE))
    if q > 32767:
        q = 32767
    if q < -32768:
        q = -32768
    return q


def get_dilemma_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR,
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return out[:12]
    except Exception:
        return "unknown"


def get_semver() -> str:
    try:
        return VERSION_FILE.read_text().strip()
    except Exception:
        return "0.0.0"


def load_vocab(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_unigrams(path: Path) -> dict[int, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def load_bigrams(path: Path):
    """Yield (w1, w2, count) from the packed tsv."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            a, b, c = line.rstrip("\n").split("\t")
            yield int(a), int(b), int(c)


def load_trigrams(path: Path):
    """Yield (w1, w2, w3, count) from the packed tsv."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            a, b, c, k = line.rstrip("\n").split("\t")
            yield int(a), int(b), int(c), int(k)


def bare_letters(token: str) -> str:
    """The letters of ``token`` as a user types them on a bare keyboard.

    Accents, breathings, iota subscript and diaeresis are dropped, final
    sigma becomes σ, case is folded and a trailing elision mark is
    removed, so ἢ, ἡ and Ἡ all give η.
    """
    nfd = unicodedata.normalize("NFD", token)
    letters = "".join(c for c in nfd if not unicodedata.combining(c))
    return letters.replace("ς", "σ").lower().rstrip(ELISION_MARKS)


def spelling_key(token: str) -> str:
    """``token`` with the differences that do not make another word
    folded away: a grave is the acute written before another word, case
    is position, and the elision glyph is typography. καί and καὶ, Καί
    and δ᾽ and δ’ each give one key; ἢ and ἡ give two.
    """
    nfd = unicodedata.normalize("NFD", token).replace("\u0300", "\u0301")
    nfd = "".join("\u2019" if c in ELISION_MARKS else c for c in nfd)
    return unicodedata.normalize("NFC", nfd).lower()


def homograph_spelling_ids(vocab: list[str]) -> set[int]:
    """Ids of the vocabulary tokens that are one spelling of a homograph
    set: tokens whose bare letters another token shares with a different
    :func:`spelling_key`. Reserved tokens are never included.
    """
    spellings: dict[str, set[str]] = defaultdict(set)
    for tok in vocab:
        if tok not in RESERVED_TOKENS:
            spellings[bare_letters(tok)].add(spelling_key(tok))
    return {
        i for i, tok in enumerate(vocab)
        if tok not in RESERVED_TOKENS and len(spellings[bare_letters(tok)]) > 1
    }


def keep_past_cut(
    entries: list[tuple[int, int]], top_k: int,
    keep_ids: set[int] | frozenset = frozenset(), keep_min_count: int = 0,
    keep_all_min_count: int = 0,
) -> list[tuple[int, int]]:
    """The ``top_k`` first of ``entries``, a context's (word id, count)
    continuations sorted by descending count, followed by every later
    entry in ``keep_ids`` whose count is at least ``keep_min_count``, and
    by every later entry at all whose count is at least
    ``keep_all_min_count`` (0: none).

    The kept entries stay in count order, so the row is still sorted and
    its first ``top_k`` entries are the plain cut.
    """
    top = entries[:top_k]
    floors = [m for m, on in ((keep_min_count, bool(keep_ids)),
                              (keep_all_min_count, True)) if on and m > 0]
    if not floors:
        return top
    floor = min(floors)
    for wid, c in entries[top_k:]:
        if c < floor:
            break
        if ((keep_all_min_count > 0 and c >= keep_all_min_count)
                or (keep_min_count > 0 and c >= keep_min_count
                    and wid in keep_ids)):
            top.append((wid, c))
    return top


def fnv1a64(token: str) -> int:
    """FNV-1a 64-bit hash of the UTF-8 bytes of ``token``: the key of the
    out-of-vocabulary unigram table."""
    h = FNV64_OFFSET
    for b in token.encode("utf-8"):
        h = ((h ^ b) * FNV64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def table_spelling(token: str) -> str:
    """The spelling the out-of-vocabulary table stores ``token`` under:
    NFC, a trailing elision glyph written as U+2019, case kept."""
    token = unicodedata.normalize("NFC", token)
    if token and token[-1] in ELISION_MARKS:
        token = token[:-1] + "\u2019"
    return token


def lookup_key(token: str) -> str:
    """``token`` as a reader looks it up at worst: NFC, lowercased, a
    trailing elision glyph written as U+2019."""
    return table_spelling(token).lower()


def resolves_in_vocab(token: str, vocab: set[str]) -> bool:
    """True when a reader finds a vocabulary id for ``token``: it tries
    the literal spelling, then lowercased, then the U+2019 spelling and
    its lowercase, and a type it resolves never reaches the table."""
    spelling = table_spelling(token)
    return any(v in vocab for v in (
        token, token.lower(), spelling, spelling.lower()))


def read_hunspell_words(dic_path: Path) -> set[str]:
    """Every word a Hunspell .dic spells, its flagged stems expanded with
    the SFX rules of the .aff beside it. Only zero-strip rules (the
    only kind the Dilemma export writes) are applied."""
    suffixes: dict[str, list[str]] = defaultdict(list)
    aff_path = dic_path.with_suffix(".aff")
    if aff_path.exists():
        for line in aff_path.read_text(encoding="utf-8").splitlines():
            f = line.split()
            if len(f) >= 5 and f[0] == "SFX" and f[2] == "0":
                suffixes[f[1]].append("" if f[3] == "0" else f[3])
    words: set[str] = set()
    with open(dic_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            entry = line.split("\t", 1)[0].split(" ", 1)[0].strip()
            if not entry or (i == 0 and entry.isdigit()):
                continue
            stem, _, flags = entry.partition("/")
            stem = unicodedata.normalize("NFC", stem)
            words.add(stem)
            for flag in filter(None, flags.split(",")):
                words.update(stem + suf for suf in suffixes.get(flag, ()))
    return words


def contested_lookup_keys(words: set[str]) -> set[str]:
    """:func:`lookup_key` of every dictionary spelling whose bare letters
    another dictionary spelling shares with a different
    :func:`spelling_key`: the spellings a keyboard chooses between when
    those letters are typed."""
    keys_by_letters: dict[str, set[str]] = defaultdict(set)
    for w in words:
        keys_by_letters[bare_letters(w)].add(spelling_key(w))
    return {lookup_key(w) for w in words
            if len(keys_by_letters[bare_letters(w)]) > 1}


def load_type_counts(path: Path):
    """Yield (token, count) from ``train_lm.py``'s type_counts.tsv.gz."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            tok, c = line.rstrip("\n").split("\t")
            yield tok, int(c)


def select_oov_unigrams(
    type_counts, vocab: set[str], min_count: int,
    lookup_keys: set[str] | None,
) -> list[tuple[int, int]]:
    """(hash, count) entries of the out-of-vocabulary unigram table,
    sorted by hash. Training types a reader resolves to a vocabulary id
    (:func:`resolves_in_vocab`) are left out; the rest are summed under
    their :func:`table_spelling`, and a spelling is kept when its count
    is at least ``min_count`` and its :func:`lookup_key` is in
    ``lookup_keys`` (every spelling when it is None). Counts are capped
    at 255. Two spellings with one hash raise ValueError."""
    if min_count <= 0:
        return []
    summed: dict[str, int] = defaultdict(int)
    for tok, c in type_counts:
        if tok in RESERVED_TOKENS or resolves_in_vocab(tok, vocab):
            continue
        summed[table_spelling(tok)] += c
    table: dict[int, tuple[str, int]] = {}
    for spelling, c in summed.items():
        if c < min_count:
            continue
        if lookup_keys is not None and spelling.lower() not in lookup_keys:
            continue
        h = fnv1a64(spelling)
        if h in table:
            raise ValueError(
                f"FNV-1a collision: {table[h][0]!r} and {spelling!r}")
        table[h] = (spelling, min(c, OOV_COUNT_CAP))
    return sorted((h, c) for h, (_, c) in table.items())


def oov_unigram_table(table_off: int, entries: list[tuple[int, int]]) -> bytes:
    """The table and its footer, for a file whose table starts (after
    padding) at ``table_off``, a multiple of 8."""
    assert table_off % 8 == 0
    keys = struct.pack(f"<{len(entries)}Q", *(h for h, _ in entries))
    counts = bytes(c for _, c in entries)
    footer = struct.pack("<QI4s", table_off, len(entries), OOV_UNIGRAM_MAGIC)
    return keys + counts + footer


def read_oov_unigram_table(data: bytes) -> dict[int, int]:
    """{hash: count} of the out-of-vocabulary unigram table at the end of
    an exported file, or {} when the file has none."""
    if len(data) < 16 or data[-4:] != OOV_UNIGRAM_MAGIC:
        return {}
    off, n, _ = struct.unpack_from("<QI4s", data, len(data) - 16)
    keys = struct.unpack_from(f"<{n}Q", data, off)
    return dict(zip(keys, data[off + 8 * n: off + 9 * n]))


def dictionary_drift(
    version_path: Path = BUILD_DIR / "grc_ngram.version",
    dic_path: Path = DEFAULT_OOV_DICTIONARY,
) -> str | None:
    """A warning when the LM described by ``version_path`` carries an
    out-of-vocabulary table selected against another ``dic_path`` than
    the one on disk now, else None (also when either file is missing or
    the LM has no table)."""
    version_path, dic_path = Path(version_path), Path(dic_path)
    if not version_path.exists() or not dic_path.exists():
        return None
    recorded = json.loads(version_path.read_text(encoding="utf-8")).get(
        "oov_dictionary_sha256")
    if not recorded:
        return None
    current = hashlib.sha256(dic_path.read_bytes()).hexdigest()
    if current == recorded:
        return None
    return (f"{version_path} was exported against a {dic_path.name} with "
            f"sha256 {recorded[:12]}, and the one on disk now has "
            f"{current[:12]}. Re-export the LM (python export_lm.py) so its "
            f"out-of-vocabulary table follows the dictionary.")


def build_bigram_contexts(
    bigrams, w1_total: dict[int, int], top_k: int,
    excluded_continuation_ids: set[int],
    keep_ids: set[int] | frozenset = frozenset(), keep_min_count: int = 0,
    complete_ids: set[int] | frozenset = frozenset(),
    complete_min_count: int = 0,
    stats: dict | None = None,
):
    """Group bigrams by w1, sort continuations by prob, keep top_k.

    Continuations in ``excluded_continuation_ids`` (reserved tokens like
    <UNK>, <PAD>, <s>) are dropped from suggestions so the keyboard
    never proposes them to the user. Continuations in ``keep_ids`` seen
    at least ``keep_min_count`` times stay past the cut
    (:func:`keep_past_cut`), and so does every continuation seen at least
    ``complete_min_count`` times after a context in ``complete_ids``.
    ``stats``, when given, receives the number of completed rows
    (``context_rows``) and of the entries they gained over the first two
    rules (``context_entries``).
    """
    by_w1: dict[int, list[tuple[int, int]]] = {}
    for w1, w2, c in bigrams:
        if w2 in excluded_continuation_ids:
            continue
        by_w1.setdefault(w1, []).append((w2, c))

    rows = added = 0
    contexts: list[tuple[int, list[tuple[int, float]]]] = []
    for w1 in sorted(by_w1):
        denom = w1_total.get(w1, 0)
        if denom <= 0:
            continue
        entries = by_w1[w1]
        entries.sort(key=lambda t: (-t[1], t[0]))
        top = keep_past_cut(entries, top_k, keep_ids, keep_min_count)
        if complete_min_count > 0 and w1 in complete_ids:
            completed = keep_past_cut(entries, top_k, keep_ids,
                                      keep_min_count, complete_min_count)
            if len(completed) > len(top):
                rows += 1
                added += len(completed) - len(top)
            top = completed
        scored = [(w2, math.log(c / denom)) for w2, c in top]
        contexts.append((w1, scored))
    if stats is not None:
        stats["context_rows"] = rows
        stats["context_entries"] = added
    return contexts


def build_trigram_contexts(
    trigrams, bigram_counts: dict[tuple[int, int], int], top_k: int,
    excluded_continuation_ids: set[int],
    keep_ids: set[int] | frozenset = frozenset(), keep_min_count: int = 0,
):
    """Group trigrams by (w1, w2), rank continuations, keep top_k.

    Continuations in ``excluded_continuation_ids`` (reserved tokens like
    <UNK>, <PAD>, <s>) are dropped from suggestions so the keyboard
    never proposes them to the user. Continuations in ``keep_ids`` seen
    at least ``keep_min_count`` times stay past the cut
    (:func:`keep_past_cut`).
    """
    by_ctx: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for w1, w2, w3, c in trigrams:
        if w3 in excluded_continuation_ids:
            continue
        by_ctx.setdefault((w1, w2), []).append((w3, c))

    contexts: list[tuple[tuple[int, int], list[tuple[int, float]]]] = []
    for ctx in sorted(by_ctx):
        denom = bigram_counts.get(ctx, 0)
        if denom <= 0:
            continue
        entries = by_ctx[ctx]
        entries.sort(key=lambda t: (-t[1], t[0]))
        top = keep_past_cut(entries, top_k, keep_ids, keep_min_count)
        scored = [(w3, math.log(c / denom)) for w3, c in top]
        contexts.append((ctx, scored))
    return contexts


def write_binary(
    out_path: Path,
    id2tok: list[str],
    vocab_counts: list[int],
    unigram_topk: list[tuple[int, float]],
    bigram_ctx: list[tuple[int, list[tuple[int, float]]]],
    trigram_ctx: list[tuple[tuple[int, int], list[tuple[int, float]]]],
    total_tokens: int,
    reserved_ids: dict[str, int],
    oov_unigrams: list[tuple[int, int]] | None = None,
):
    """Write the artifact. ``oov_unigrams``, (hash, count) entries from
    :func:`select_oov_unigrams`, appends the out-of-vocabulary unigram
    table and its footer; without it the file ends at the trigram
    suggestions."""
    V = len(id2tok)
    assert len(vocab_counts) == V, "vocab_counts must be one-per-vocab-id"
    # suggestion_count is a u16.
    for _, sug in list(bigram_ctx) + list(trigram_ctx):
        assert len(sug) <= 0xFFFF, "a context row holds more than 65,535 entries"

    # serialize vocab string pool + offsets
    pool_parts = []
    offsets = [0]
    for tok in id2tok:
        b = tok.encode("utf-8")
        pool_parts.append(b)
        offsets.append(offsets[-1] + len(b))
    pool = b"".join(pool_parts)

    # --- Plan layout ---
    HEADER_SIZE = 128
    vocab_offsets_off = HEADER_SIZE
    vocab_offsets_bytes = 4 * (V + 1)
    string_pool_off = vocab_offsets_off + vocab_offsets_bytes
    string_pool_size = len(pool)

    # v2: per-vocab counts column right after the string pool.
    # The reader locates it at string_pool_off + string_pool_size,
    # so no new header field is needed.
    vocab_counts_off = string_pool_off + string_pool_size
    vocab_counts_bytes = 4 * V

    unigram_topk_off = vocab_counts_off + vocab_counts_bytes

    bigram_index_off = unigram_topk_off + TOP_K_UNI * 6  # (u32 + i16) per entry
    bigram_index_len = len(bigram_ctx)
    bigram_index_bytes = bigram_index_len * 12
    bigram_suggestions_off = bigram_index_off + bigram_index_bytes
    bigram_suggestions_len = sum(len(sug) for _, sug in bigram_ctx)
    bigram_suggestions_bytes = bigram_suggestions_len * 6

    trigram_index_off = bigram_suggestions_off + bigram_suggestions_bytes
    trigram_index_len = len(trigram_ctx)
    trigram_index_bytes = trigram_index_len * 16
    trigram_suggestions_off = trigram_index_off + trigram_index_bytes
    trigram_suggestions_len = sum(len(sug) for _, sug in trigram_ctx)
    trigram_suggestions_bytes = trigram_suggestions_len * 6

    total_size = trigram_suggestions_off + trigram_suggestions_bytes
    oov_part = b""
    if oov_unigrams:
        oov_table_off = (total_size + 7) // 8 * 8
        oov_part = bytes(oov_table_off - total_size) + oov_unigram_table(
            oov_table_off, oov_unigrams)
        total_size += len(oov_part)

    # --- Header ---
    header = bytearray(HEADER_SIZE)
    header[0:4] = MAGIC
    struct.pack_into("<I", header, 4, FORMAT_VERSION)
    struct.pack_into("<I", header, 8, ORDER)
    # top_k_uni held at offset 12 for compatibility with v1 field layout.
    struct.pack_into("<I", header, 12, TOP_K_UNI)
    struct.pack_into("<I", header, 16, V)
    struct.pack_into("<I", header, 20, reserved_ids["<PAD>"])
    struct.pack_into("<I", header, 24, reserved_ids["<UNK>"])
    struct.pack_into("<I", header, 28, reserved_ids["<s>"])
    struct.pack_into("<I", header, 32, reserved_ids["</s>"])
    # v2: formerly-reserved slot now carries top_k_bi so the reader
    # knows the maximum per-bigram count to expect.
    struct.pack_into("<I", header, 36, TOP_K_BI)
    struct.pack_into("<Q", header, 40, total_tokens)
    struct.pack_into("<Q", header, 48, vocab_offsets_off)
    struct.pack_into("<Q", header, 56, string_pool_off)
    struct.pack_into("<Q", header, 64, string_pool_size)
    struct.pack_into("<Q", header, 72, unigram_topk_off)
    struct.pack_into("<Q", header, 80, bigram_index_off)
    struct.pack_into("<I", header, 88, bigram_index_len)
    struct.pack_into("<I", header, 92, bigram_suggestions_len)
    struct.pack_into("<Q", header, 96, bigram_suggestions_off)
    struct.pack_into("<Q", header, 104, trigram_index_off)
    struct.pack_into("<I", header, 112, trigram_index_len)
    struct.pack_into("<I", header, 116, trigram_suggestions_len)
    struct.pack_into("<Q", header, 120, trigram_suggestions_off)

    # --- Body ---
    parts = [bytes(header)]

    # vocab offsets
    parts.append(b"".join(struct.pack("<I", o) for o in offsets))
    # string pool
    parts.append(pool)

    # v2: per-vocab counts (u32 * V). Clamped to u32 max if ever bigger.
    counts_bytes = bytearray(vocab_counts_bytes)
    for i, c in enumerate(vocab_counts):
        if c < 0:
            c = 0
        if c > 0xFFFFFFFF:
            c = 0xFFFFFFFF
        struct.pack_into("<I", counts_bytes, i * 4, c)
    parts.append(bytes(counts_bytes))

    # unigram top-K
    uni_bytes = bytearray(TOP_K_UNI * 6)
    for i in range(TOP_K_UNI):
        if i < len(unigram_topk):
            wid, logp = unigram_topk[i]
        else:
            wid, logp = 1, -32.0  # pad with UNK at minimum logprob
        struct.pack_into("<Ih", uni_bytes, i * 6, wid, quantize_logprob(logp))
    parts.append(bytes(uni_bytes))

    # bigram index + suggestions
    bi_index = bytearray(bigram_index_bytes)
    bi_sug = bytearray(bigram_suggestions_bytes)
    sug_cursor = 0
    for i, (w1, sug_list) in enumerate(bigram_ctx):
        struct.pack_into("<I", bi_index, i * 12, w1)
        struct.pack_into("<I", bi_index, i * 12 + 4, sug_cursor)
        struct.pack_into("<H", bi_index, i * 12 + 8, len(sug_list))
        struct.pack_into("<H", bi_index, i * 12 + 10, 0)  # reserved
        for j, (w2, logp) in enumerate(sug_list):
            struct.pack_into(
                "<Ih", bi_sug, (sug_cursor + j) * 6,
                w2, quantize_logprob(logp)
            )
        sug_cursor += len(sug_list)
    parts.append(bytes(bi_index))
    parts.append(bytes(bi_sug))

    # trigram index + suggestions
    tri_index = bytearray(trigram_index_bytes)
    tri_sug = bytearray(trigram_suggestions_bytes)
    sug_cursor = 0
    for i, ((w1, w2), sug_list) in enumerate(trigram_ctx):
        struct.pack_into("<I", tri_index, i * 16, w1)
        struct.pack_into("<I", tri_index, i * 16 + 4, w2)
        struct.pack_into("<I", tri_index, i * 16 + 8, sug_cursor)
        struct.pack_into("<H", tri_index, i * 16 + 12, len(sug_list))
        struct.pack_into("<H", tri_index, i * 16 + 14, 0)  # reserved
        for j, (w3, logp) in enumerate(sug_list):
            struct.pack_into(
                "<Ih", tri_sug, (sug_cursor + j) * 6,
                w3, quantize_logprob(logp)
            )
        sug_cursor += len(sug_list)
    parts.append(bytes(tri_index))
    parts.append(bytes(tri_sug))
    parts.append(oov_part)

    out_path.write_bytes(b"".join(parts))
    assert out_path.stat().st_size == total_size, \
        f"Size mismatch: planned {total_size}, wrote {out_path.stat().st_size}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="in_dir", default=str(BUILD_DIR),
                    help="Intermediate directory from train_lm.py")
    ap.add_argument("--out", default=str(BUILD_DIR / "grc_ngram.bin"))
    ap.add_argument("--version-out",
                    default=str(BUILD_DIR / "grc_ngram.version"))
    ap.add_argument("--keep-homograph-min-count", type=int,
                    default=KEEP_HOMOGRAPH_MIN_COUNT,
                    help="Keep a homograph spelling past a context's "
                         "top-K cut when it was seen at least this "
                         "often in that context (default "
                         f"{KEEP_HOMOGRAPH_MIN_COUNT}); 0 exports the "
                         "plain top-K rows.")
    ap.add_argument("--homograph-context-min-count", type=int,
                    default=HOMOGRAPH_CONTEXT_MIN_COUNT,
                    help="Keep, past the cut of the bigram row of a "
                         "homograph spelling, every continuation seen at "
                         "least this often after it (default "
                         f"{HOMOGRAPH_CONTEXT_MIN_COUNT}); 0 cuts those "
                         "rows like any other.")
    ap.add_argument("--oov-min-count", type=int,
                    default=OOV_UNIGRAM_MIN_COUNT,
                    help="Carry the exact training count of each "
                         "out-of-vocabulary spelling the dictionary "
                         "proposes against another spelling of its "
                         "letters, when seen at least this often "
                         f"(default {OOV_UNIGRAM_MIN_COUNT}); 0 leaves "
                         "the table out.")
    ap.add_argument("--oov-dictionary", default=str(DEFAULT_OOV_DICTIONARY),
                    help="Hunspell .dic (with its .aff beside it) whose "
                         "spellings the out-of-vocabulary table serves "
                         "(default: the grc export of export_hunspell.py).")
    ap.add_argument("--check-dictionary", action="store_true",
                    help="Export nothing; exit 1 when the LM at "
                         "--version-out was exported against another "
                         "--oov-dictionary than the one on disk.")
    args = ap.parse_args()

    if args.check_dictionary:
        drift = dictionary_drift(Path(args.version_out),
                                 Path(args.oov_dictionary))
        print(drift or "The LM's out-of-vocabulary table matches "
              f"{args.oov_dictionary}.")
        sys.exit(1 if drift else 0)

    in_dir = Path(args.in_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    oov_min = max(0, args.oov_min_count)
    oov_dictionary = Path(args.oov_dictionary)
    type_counts_path = in_dir / "type_counts.tsv.gz"
    if oov_min:
        for need, how in ((type_counts_path, "rerun train_lm.py, which "
                           "writes it"),
                          (oov_dictionary, "run export_hunspell.py "
                           "--variant grc first, or pass --oov-dictionary")):
            if not need.exists():
                raise SystemExit(
                    f"No {need}: the out-of-vocabulary unigram table needs "
                    f"it ({how}). --oov-min-count 0 exports without the "
                    f"table.")

    t0 = time.time()
    id2tok = load_vocab(in_dir / "vocab.json")

    # The on-disk vocab must be sorted alphabetically for binary
    # search, but token ids were assigned by frequency in train_lm.py.
    # Re-sort and remap all n-gram ids so the ids in the binary match
    # the sorted vocab.
    sort_perm = sorted(range(len(id2tok)), key=lambda i: id2tok[i])
    remap = [0] * len(id2tok)
    for new_id, old_id in enumerate(sort_perm):
        remap[old_id] = new_id
    sorted_vocab = [id2tok[i] for i in sort_perm]

    # Look up reserved-token ids in the sorted vocab so the header
    # records the on-disk positions (not the train-time positions).
    reserved_ids = {
        tok: sorted_vocab.index(tok)
        for tok in ("<PAD>", "<UNK>", "<s>", "</s>")
    }

    uni_by_old = load_unigrams(in_dir / "unigrams.json")
    total_tokens = sum(uni_by_old.values())
    uni_by_new: dict[int, int] = {}
    for old_id, c in uni_by_old.items():
        uni_by_new[remap[old_id]] = c

    # unigram top-K (excluding the reserved tokens so they don't
    # dominate the fallback list)
    reserved_new_ids = set(remap[i] for i in (0, 1, 2, 3))
    unigram_ranked = sorted(
        ((wid, c) for wid, c in uni_by_new.items()
         if wid not in reserved_new_ids),
        key=lambda t: (-t[1], t[0])
    )
    unigram_topk = [
        (wid, math.log(c / total_tokens))
        for wid, c in unigram_ranked[:TOP_K_UNI]
    ]

    # v2: dense per-vocab counts column, indexed by sorted-vocab id.
    # Reserved tokens get 0 so the reader can trivially rank them last.
    vocab_counts = [0] * len(sorted_vocab)
    for wid, c in uni_by_new.items():
        if wid in reserved_new_ids:
            continue
        vocab_counts[wid] = c

    # bigrams
    bigram_counts: dict[tuple[int, int], int] = {}
    # we also need w1 totals (sum of c over all w2 given w1) to compute
    # P(w2 | w1). Use unigram counts of w1, which approximates it and
    # uses less memory than recomputing.
    w1_totals = uni_by_new

    remapped_bigrams = []
    for old_a, old_b, c in load_bigrams(in_dir / "bigrams.tsv.gz"):
        a = remap[old_a]
        b = remap[old_b]
        bigram_counts[(a, b)] = c
        remapped_bigrams.append((a, b, c))
    print(f"  loaded {len(remapped_bigrams):,} bigrams", flush=True)

    # The keyboard never needs to propose the reserved tokens. Drop
    # them from suggestion lists. </s> stays out too: the period key
    # on the keyboard already represents sentence-end.
    excluded = {
        reserved_ids["<PAD>"],
        reserved_ids["<UNK>"],
        reserved_ids["<s>"],
        reserved_ids["</s>"],
    }

    keep_min = max(0, args.keep_homograph_min_count)
    context_min = max(0, args.homograph_context_min_count)
    homograph_ids = (homograph_spelling_ids(sorted_vocab)
                     if keep_min or context_min else set())
    keep_ids = homograph_ids if keep_min else set()
    if keep_min:
        print(f"  keeping {len(keep_ids):,} homograph spellings past the "
              f"top-K cut when seen >= {keep_min} times in a context",
              flush=True)

    context_stats: dict[str, int] = {}
    bigram_ctx = build_bigram_contexts(
        remapped_bigrams, w1_totals, TOP_K_BI, excluded,
        keep_ids, keep_min,
        homograph_ids if context_min else set(), context_min,
        context_stats,
    )
    if context_min:
        print(f"  completed {context_stats['context_rows']:,} bigram rows "
              f"of homograph spellings with "
              f"{context_stats['context_entries']:,} continuations seen "
              f">= {context_min} times", flush=True)
    print(f"  built {len(bigram_ctx):,} bigram contexts "
          f"(top_k={TOP_K_BI})", flush=True)

    # trigrams (stream rather than hold everything)
    remapped_trigrams = []
    for old_a, old_b, old_c, k in load_trigrams(
        in_dir / "trigrams.tsv.gz"
    ):
        a = remap[old_a]
        b = remap[old_b]
        c = remap[old_c]
        remapped_trigrams.append((a, b, c, k))
    print(f"  loaded {len(remapped_trigrams):,} trigrams", flush=True)

    trigram_ctx = build_trigram_contexts(
        remapped_trigrams, bigram_counts, TOP_K_TRI, excluded,
        keep_ids, keep_min,
    )
    print(f"  built {len(trigram_ctx):,} trigram contexts "
          f"(top_k={TOP_K_TRI})", flush=True)

    oov_unigrams: list[tuple[int, int]] = []
    oov_dictionary_sha256 = None
    if oov_min:
        dictionary_words = read_hunspell_words(oov_dictionary)
        oov_unigrams = select_oov_unigrams(
            load_type_counts(type_counts_path), set(sorted_vocab), oov_min,
            contested_lookup_keys(dictionary_words),
        )
        oov_dictionary_sha256 = hashlib.sha256(
            oov_dictionary.read_bytes()).hexdigest()
        print(f"  {len(oov_unigrams):,} out-of-vocabulary spellings in the "
              f"unigram table ({len(dictionary_words):,} dictionary words)",
              flush=True)

    write_binary(
        out, sorted_vocab, vocab_counts, unigram_topk,
        bigram_ctx, trigram_ctx,
        total_tokens, reserved_ids, oov_unigrams,
    )

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"\nWrote {out} ({size_mb:.2f} MB) in {time.time() - t0:.1f}s",
          flush=True)

    # sidecar .version
    commit = get_dilemma_commit()
    semver = get_semver()
    version_info = {
        "semver": semver,
        "dilemma_commit": commit,
        "format_version": FORMAT_VERSION,
        "n_gram_order": ORDER,
        "top_k_uni": TOP_K_UNI,
        "top_k_bi": TOP_K_BI,
        "top_k_tri": TOP_K_TRI,
        "keep_homograph_min_count": keep_min,
        "homograph_spellings": len(keep_ids),
        "homograph_context_min_count": context_min,
        "homograph_context_rows": context_stats.get("context_rows", 0),
        "homograph_context_entries": context_stats.get("context_entries", 0),
        "oov_unigram_min_count": oov_min,
        "oov_unigrams": len(oov_unigrams),
        "oov_dictionary_sha256": oov_dictionary_sha256,
        "vocab_size": len(sorted_vocab),
        "bigram_contexts": len(bigram_ctx),
        "trigram_contexts": len(trigram_ctx),
        "total_tokens": total_tokens,
        "artifact_size_bytes": out.stat().st_size,
    }
    Path(args.version_out).write_text(
        json.dumps(version_info, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Wrote {args.version_out}")
    print(json.dumps(version_info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

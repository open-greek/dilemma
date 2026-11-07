"""A Python reader of the GNLM v2 next-word model (``export_lm.py``) that
answers queries the way the keyboard's Swift reader (``LMReader``) does.

``eval_lm.py`` has the reference reader the Ancient Greek evaluation uses;
this one adds what the monotonic evaluation and the tests need:

* the same validation as ``LMReader.init``: magic, format 2, order 3, every
  section inside the file, and the optional out-of-vocabulary table found by
  its ``GNUO`` footer and ignored when the footer does not check out;
* :meth:`GNLMReader.problems`, structural checks a model must pass: sorted
  vocabulary with the four reserved tokens where the header says, row
  indexes sorted, suggestion offsets contiguous, rows in descending
  probability, no reserved id in any row, ascending table keys;
* the keyboard's queries, step for step: :meth:`suggest` (trigram row,
  bigram row, global unigram list, deduplicated, reserved ids skipped),
  :meth:`score` (stupid backoff with alpha 0.4, the vocabulary's count
  column at the unigram tier, the table for a spelling out of the
  vocabulary) and :meth:`suggest_completions` (row entries whose
  diacritic-blind form starts with the stem, then a scan of the whole
  vocabulary, ranked by the typed marks kept, score, count and spelling).

Two lowercase functions, as in the keyboard. Vocabulary and table lookups
use the Greek-aware one (``GreekText.lowercased``, :func:`greek_lower`):
final ς at the end of a word, σ before an apostrophe of elision. Prefix
matching uses Swift's plain ``lowercased()``, which has no final-sigma rule
(ΟΔΟΣ gives οδοσ), so :func:`swift_lower` lowercases character by character
as it does; the ς is folded to σ there anyway.
"""

from __future__ import annotations

import bisect
import math
import struct
import unicodedata
from pathlib import Path

import numpy as np

MAGIC = b"GNLM"
FORMAT_VERSION = 2
HEADER_SIZE = 128
OOV_MAGIC = b"GNUO"
OOV_FOOTER_SIZE = 16
LOG_ALPHA = math.log(0.4)
OOV_SCORE = float("-inf")
SENTENCE_START = "<s>"
RESERVED = ("<PAD>", "<UNK>", "<s>", "</s>")
# The apostrophe of elision the vocabulary is keyed on, and the two glyphs
# LMReader.foldingApostrophe writes as it (text from a hardware keyboard
# or a paste).
CORPUS_ELISION = "’"
FOLDED_ELISIONS = "'ʼ"
FNV64_OFFSET = 0xCBF29CE484222325
FNV64_PRIME = 0x100000001B3


class FormatError(ValueError):
    """The file is not a model ``LMReader`` would load."""


def greek_lower(s: str) -> str:
    """``GreekText.lowercased``: lowercase with final ς at the end of a
    word (Python's ``str.lower`` applies the same Final_Sigma rule as the
    Greek locale), but σ before an apostrophe, where the word is elided."""
    low = s.lower()
    if "ς" not in low:
        return low
    return low.replace("ς’", "σ’").replace("ς'", "σ'")


def swift_lower(s: str) -> str:
    """``String.lowercased()`` as Swift does it: no final-sigma context."""
    return "".join(c.lower() for c in s)


def normalize(s: str) -> str:
    """``LMReader.normalize``: lowercase, diacritics stripped, ς as σ."""
    if not s:
        return s
    nfd = unicodedata.normalize("NFD", swift_lower(s))
    stripped = unicodedata.normalize(
        "NFC", "".join(c for c in nfd if not unicodedata.combining(c)))
    return stripped.replace("ς", "σ")


def fnv1a64(token: str) -> int:
    h = FNV64_OFFSET
    for b in token.encode("utf-8"):
        h = ((h ^ b) * FNV64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def typed_marks(stem: str) -> list[tuple[int, set[str]]]:
    """Positions of the stem's letters that carry marks, with the marks."""
    out = []
    for k, ch in enumerate(stem):
        marks = {c for c in unicodedata.normalize("NFD", ch)
                 if unicodedata.combining(c)}
        if marks:
            out.append((k, marks))
    return out


def keeps_typed_marks(token: str, required) -> bool:
    for k, marks in required:
        if k >= len(token):
            return False
        if not marks <= set(unicodedata.normalize("NFD", token[k])):
            return False
    return True


class GNLMReader:
    """Read a GNLM v2 file and answer the keyboard's queries."""

    def __init__(self, source: Path | bytes):
        data = source if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
        self.data = bytes(data)
        self._parse()

    # ---- loading ----------------------------------------------------------

    def _u32(self, off: int) -> int:
        return struct.unpack_from("<I", self.data, off)[0]

    def _u64(self, off: int) -> int:
        return struct.unpack_from("<Q", self.data, off)[0]

    def _parse(self) -> None:
        d = self.data
        if len(d) < HEADER_SIZE or d[:4] != MAGIC:
            raise FormatError("bad magic")
        if self._u32(4) != FORMAT_VERSION:
            raise FormatError(f"format {self._u32(4)}, not {FORMAT_VERSION}")
        if self._u32(8) != 3:
            raise FormatError(f"order {self._u32(8)}, not 3")
        self.top_k_uni = self._u32(12)
        self.V = V = self._u32(16)
        self.id_pad, self.id_unk, self.id_bos, self.id_eos = (
            self._u32(20), self._u32(24), self._u32(28), self._u32(32))
        self.top_k_bi = self._u32(36) or self.top_k_uni
        self.total_tokens = self._u64(40)
        pool_off, pool_size = self._u64(56), self._u64(64)
        sections = [
            (self._u64(48), V + 1, 4),
            (pool_off, pool_size, 1),
            (pool_off + pool_size, V, 4),
            (self._u64(72), self.top_k_uni, 6),
            (self._u64(80), self._u32(88), 12),
            (self._u64(96), self._u32(92), 6),
            (self._u64(104), self._u32(112), 16),
            (self._u64(120), self._u32(116), 6),
        ]
        end = HEADER_SIZE
        for off, count, stride in sections:
            stop = off + count * stride
            if stop > len(d):
                raise FormatError("a section runs past the end of the file")
            end = max(end, stop)
        self.sections_end = end
        (offs_off, _, _), _, (counts_off, _, _), (uni_off, _, _), \
            (bi_off, n_bi, _), (bis_off, n_bis, _), \
            (tri_off, n_tri, _), (tris_off, n_tris, _) = sections
        self.offsets = np.frombuffer(d, dtype="<u4", count=V + 1, offset=offs_off)
        pool = d[pool_off:pool_off + pool_size]
        self.vocab_bytes = [pool[self.offsets[i]:self.offsets[i + 1]] for i in range(V)]
        self.vocab = [b.decode("utf-8") for b in self.vocab_bytes]
        self.index = {t: i for i, t in enumerate(self.vocab)}
        self.counts = np.frombuffer(d, dtype="<u4", count=V, offset=counts_off)
        self.unigram_topk = [struct.unpack_from("<Ih", d, uni_off + 6 * i)
                             for i in range(self.top_k_uni)]
        bi = np.frombuffer(d, dtype=np.dtype([("w1", "<u4"), ("off", "<u4"),
                                              ("cnt", "<u2"), ("res", "<u2")]),
                           count=n_bi, offset=bi_off)
        tri = np.frombuffer(d, dtype=np.dtype([("w1", "<u4"), ("w2", "<u4"),
                                               ("off", "<u4"), ("cnt", "<u2"),
                                               ("res", "<u2")]),
                            count=n_tri, offset=tri_off)
        sug = np.dtype([("id", "<u4"), ("q", "<i2")])
        self.bi_rows, self.tri_rows = bi, tri
        self.bi_sug = np.frombuffer(d, dtype=sug, count=n_bis, offset=bis_off)
        self.tri_sug = np.frombuffer(d, dtype=sug, count=n_tris, offset=tris_off)
        self.bi_index = {int(w): (int(o), int(c)) for w, o, c in
                         zip(bi["w1"], bi["off"], bi["cnt"])}
        self.tri_index = {(int(a), int(b)): (int(o), int(c)) for a, b, o, c in
                          zip(tri["w1"], tri["w2"], tri["off"], tri["cnt"])}
        self.bi_ids = self.bi_sug["id"].astype(np.int64)
        self.bi_q = self.bi_sug["q"].astype(np.int64)
        self.tri_ids = self.tri_sug["id"].astype(np.int64)
        self.tri_q = self.tri_sug["q"].astype(np.int64)
        self.reserved_ids = {self.id_pad, self.id_unk, self.id_bos, self.id_eos}
        # Ids suggest_ids and suggest_completions never return; see
        # hide_double_accented.
        self.hidden_ids: set[int] = set()
        self.oov_off, self.oov_n = self._oov_table()
        if self.oov_n:
            self.oov_keys = np.frombuffer(d, dtype="<u8", count=self.oov_n,
                                          offset=self.oov_off)
            self.oov_counts = np.frombuffer(d, dtype=np.uint8, count=self.oov_n,
                                            offset=self.oov_off + 8 * self.oov_n)
        else:
            self.oov_keys = np.zeros(0, dtype="<u8")
            self.oov_counts = np.zeros(0, dtype=np.uint8)
        self._norm = None

    def _oov_table(self) -> tuple[int, int]:
        """``LMReader.oovTable``: (offset, count), or (0, 0) when the footer
        is missing or does not check out."""
        d = self.data
        end = len(d)
        if end < self.sections_end + OOV_FOOTER_SIZE or d[-4:] != OOV_MAGIC:
            return 0, 0
        off = self._u64(end - OOV_FOOTER_SIZE)
        n = self._u32(end - 8)
        table_end = end - OOV_FOOTER_SIZE
        if not (self.sections_end <= off <= table_end and off % 8 == 0
                and n <= (table_end - off) // 9):
            return 0, 0
        return off, n

    # ---- structural checks ------------------------------------------------

    def problems(self) -> list[str]:
        """Every structural rule the model breaks (empty when none)."""
        out = []
        if any(self.vocab_bytes[i] >= self.vocab_bytes[i + 1] for i in range(self.V - 1)):
            out.append("vocabulary not strictly sorted by UTF-8 bytes")
        for name, i in zip(RESERVED, (self.id_pad, self.id_unk, self.id_bos, self.id_eos)):
            if i >= self.V or self.vocab[i] != name:
                out.append(f"header id of {name} is {i}, not its vocabulary position")
        if any(unicodedata.normalize("NFC", t) != t for t in self.vocab):
            out.append("vocabulary has a token that is not NFC")
        if self.total_tokens <= 0:
            out.append("total_tokens is 0")
        if int(self.counts[list(self.reserved_ids)].sum()):
            out.append("a reserved token has a count")
        for name, rows, sug, keyf in (
                ("bigram", self.bi_rows, self.bi_sug, lambda r: (r["w1"],)),
                ("trigram", self.tri_rows, self.tri_sug,
                 lambda r: (r["w1"], r["w2"]))):
            if len(rows):
                keys = (rows["w1"].astype(np.uint64) << np.uint64(32)) | (
                    rows["w2"].astype(np.uint64) if name == "trigram"
                    else np.uint64(0))
                if np.any(keys[1:] <= keys[:-1]):
                    out.append(f"{name} index not strictly sorted")
                expected = np.concatenate(([0], np.cumsum(rows["cnt"].astype(np.int64))[:-1]))
                if not np.array_equal(rows["off"].astype(np.int64), expected):
                    out.append(f"{name} suggestion offsets not contiguous")
                if int(rows["cnt"].astype(np.int64).sum()) != len(sug):
                    out.append(f"{name} rows do not cover the suggestion table")
                if np.any(rows["cnt"] == 0):
                    out.append(f"{name} index has an empty row")
            if len(sug):
                if np.isin(sug["id"], list(self.reserved_ids)).any():
                    out.append(f"{name} suggestions hold a reserved id")
                if np.any(sug["id"] >= self.V):
                    out.append(f"{name} suggestions hold an id out of range")
                q = sug["q"].astype(np.int64)
                starts = rows["off"].astype(np.int64)
                same_row = np.ones(len(q) - 1, dtype=bool) if len(q) > 1 else np.zeros(0, bool)
                if len(q) > 1:
                    boundary = starts[1:] - 1
                    same_row[boundary[(boundary >= 0) & (boundary < len(q) - 1)]] = False
                    if np.any((q[1:] > q[:-1]) & same_row):
                        out.append(f"{name} row not in descending probability")
                if np.any(q > 0):
                    out.append(f"{name} suggestion with a log probability above 0")
        if self.oov_n:
            if np.any(self.oov_keys[1:] <= self.oov_keys[:-1]):
                out.append("out-of-vocabulary keys not strictly ascending")
        return out

    # ---- vocabulary lookups ----------------------------------------------

    def id_exact(self, token: str) -> int | None:
        return self.index.get(token)

    @staticmethod
    def _fold_elision(token: str) -> str | None:
        """``LMReader.foldingApostrophe``: U+0027 and U+02BC written as
        U+2019, or None when the token holds neither."""
        if not any(c in FOLDED_ELISIONS for c in token):
            return None
        return "".join(CORPUS_ELISION if c in FOLDED_ELISIONS else c
                       for c in token)

    def id_case_insensitive(self, token: str) -> int | None:
        i = self.index.get(token)
        if i is not None:
            return i
        lower = greek_lower(token)
        if lower != token and (i := self.index.get(lower)) is not None:
            return i
        folded = self._fold_elision(token)
        if folded is None:
            return None
        if (i := self.index.get(folded)) is not None:
            return i
        fl = greek_lower(folded)
        return self.index.get(fl) if fl != folded else None

    def context_id(self, token: str | None) -> int | None:
        """``LMReader.contextID``: monotonic text has one apostrophe, so a
        context word resolves as :meth:`id_case_insensitive` does."""
        if token is None:
            return None
        return self.id_case_insensitive(token)

    def oov_count(self, token: str) -> int:
        if not self.oov_n:
            return 0
        for t in self._oov_spellings(token):
            h = np.uint64(fnv1a64(t))
            k = int(np.searchsorted(self.oov_keys, h))
            if k < self.oov_n and self.oov_keys[k] == h:
                return int(self.oov_counts[k])
        return 0

    def _oov_spellings(self, token: str):
        yield token
        lower = greek_lower(token)
        if lower != token:
            yield lower
        folded = self._fold_elision(token)
        if folded is not None:
            yield folded
            if greek_lower(folded) != folded:
                yield greek_lower(folded)

    # ---- rows -------------------------------------------------------------

    def trigram_row(self, a: int, b: int):
        r = self.tri_index.get((a, b))
        if r is None:
            return None
        off, cnt = r
        return self.tri_ids[off:off + cnt], self.tri_q[off:off + cnt]

    def bigram_row(self, a: int):
        r = self.bi_index.get(a)
        if r is None:
            return None
        off, cnt = r
        return self.bi_ids[off:off + cnt], self.bi_q[off:off + cnt]

    # ---- next word --------------------------------------------------------

    def suggest_ids(self, prev_id: int | None, curr_id: int | None,
                    limit: int) -> list[int]:
        """``LMReader.suggest`` on resolved context ids: the trigram row,
        then the bigram row, then the global unigram list, each in order,
        reserved ids and repeats skipped, up to ``limit``."""
        out: list[int] = []
        seen: set[int] = set()
        rows = []
        if prev_id is not None and curr_id is not None:
            rows.append(self.trigram_row(prev_id, curr_id))
        if curr_id is not None:
            rows.append(self.bigram_row(curr_id))
        rows.append((np.array([w for w, _ in self.unigram_topk], dtype=np.int64), None))
        reserved = self.reserved_ids | self.hidden_ids
        for row in rows:
            if row is None:
                continue
            ids = row[0]
            for start in range(0, len(ids), 32):
                for w in ids[start:start + 32].tolist():
                    if w in reserved or w in seen:
                        continue
                    seen.add(w)
                    out.append(w)
                    if len(out) >= limit:
                        return out
        return out

    def suggest(self, prev: str | None, curr: str, limit: int = 6) -> list[str]:
        return [self.vocab[w] for w in self.suggest_ids(
            self.context_id(prev), self.context_id(curr), limit)]

    # ---- scoring ----------------------------------------------------------

    def unigram_score(self, count: int) -> float:
        if count == 0 or self.total_tokens == 0:
            return OOV_SCORE
        return 2 * LOG_ALPHA + math.log(count) - math.log(self.total_tokens)

    def score_id(self, word_id: int, prev_id: int | None, curr_id: int | None) -> float:
        if word_id in self.reserved_ids:
            return OOV_SCORE
        if prev_id is not None and curr_id is not None:
            row = self.trigram_row(prev_id, curr_id)
            if row is not None:
                hit = np.flatnonzero(row[0] == word_id)
                if len(hit):
                    return row[1][hit[0]] / 1024.0
        if curr_id is not None:
            row = self.bigram_row(curr_id)
            if row is not None:
                hit = np.flatnonzero(row[0] == word_id)
                if len(hit):
                    return LOG_ALPHA + row[1][hit[0]] / 1024.0
        return self.unigram_score(int(self.counts[word_id]))

    def hide_double_accented(self) -> set[int]:
        """Leave the double-accented enclitic forms (όνομά) out of
        suggestions, as the keyboard does (it writes them only before an
        enclitic), and return their ids."""
        self.hidden_ids = {i for i, t in enumerate(self.vocab)
                           if unicodedata.normalize("NFD", t).count("\u0301") >= 2}
        return self.hidden_ids

    def knows(self, word: str) -> bool:
        """``LMEngine.knows``: the word has a vocabulary id that is no
        reserved token, or a count in the out-of-vocabulary table."""
        word_id = self.id_case_insensitive(word)
        if word_id is not None:
            return word_id not in self.reserved_ids
        return self.oov_count(word) > 0

    def score(self, word: str, prev: str | None, curr: str | None) -> float:
        word_id = self.id_case_insensitive(word)
        if word_id is None:
            return self.unigram_score(self.oov_count(word))
        return self.score_id(word_id, self.context_id(prev), self.context_id(curr))

    # ---- mid-word completion ---------------------------------------------

    def _normalized_index(self):
        if self._norm is None:
            norm = [normalize(t) for t in self.vocab]
            order = sorted(range(self.V), key=lambda i: (norm[i], i))
            self._norm = (norm, [norm[i] for i in order], order)
        return self._norm

    def prefix_ids(self, normalized_stem: str) -> list[int]:
        """Vocabulary ids whose normalized form starts with the stem."""
        norm, sorted_norm, order = self._normalized_index()
        lo = bisect.bisect_left(sorted_norm, normalized_stem)
        hi = bisect.bisect_left(sorted_norm, normalized_stem + "\U0010ffff")
        return order[lo:hi]

    def suggest_completions(self, prev: str | None, curr: str | None,
                            stem: str, limit: int = 6) -> list[str]:
        """``LMReader.suggestCompletions``."""
        if limit <= 0:
            return []
        nstem = normalize(stem)
        if not nstem:
            return self.suggest(prev, curr or "", limit)
        norm, _, _ = self._normalized_index()
        required = typed_marks(stem)
        prev_id, curr_id = self.context_id(prev), self.context_id(curr)
        pool: list[tuple[bool, float, int, str]] = []
        seen: set[int] = set()

        def collect(row, backoff):
            if row is None:
                return
            for w, q in zip(row[0].tolist(), row[1].tolist()):
                if w in self.reserved_ids or w in seen or w in self.hidden_ids:
                    continue
                if not norm[w].startswith(nstem):
                    continue
                seen.add(w)
                tok = self.vocab[w]
                pool.append((keeps_typed_marks(tok, required), backoff + q / 1024.0,
                             int(self.counts[w]), tok))

        if prev_id is not None and curr_id is not None:
            collect(self.trigram_row(prev_id, curr_id), 0.0)
        if curr_id is not None:
            collect(self.bigram_row(curr_id), LOG_ALPHA)
        for w in self.prefix_ids(nstem):
            if w in seen or w in self.reserved_ids or w in self.hidden_ids:
                continue
            seen.add(w)
            tok = self.vocab[w]
            c = int(self.counts[w])
            pool.append((keeps_typed_marks(tok, required), self.unigram_score(c), c, tok))
        pool.sort(key=lambda x: (not x[0], -x[1], -x[2], x[3]))
        title = bool(stem) and stem[0].isupper()
        out: list[str] = []
        got: set[str] = set()
        for _, _, _, tok in pool:
            if title and tok and not tok[0].isupper():
                tok = tok[0].upper() + tok[1:]
            if tok not in got:
                got.add(tok)
                out.append(tok)
                if len(out) >= limit:
                    break
        return out

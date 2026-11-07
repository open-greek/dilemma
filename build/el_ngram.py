"""Streaming n-gram counts for the monotonic Modern Greek next-word model.

``train_lm.py --lang el`` calls :func:`main`. The Ancient Greek path of
``train_lm.py`` keeps every sentence in a Python list and counts with
``Counter``; at the 200M tokens of the monotonic corpus that would take
tens of gigabytes of Python objects. This module keeps the corpus as int32
arrays and counts with numpy, then writes the same intermediate files the
Ancient Greek path writes, so ``export_lm.py`` compiles both.

Steps
-----
1. **Encode.** Every token of the extracted corpus (``build/el_corpus``,
   from ``extract_el_monotonic_lm.py``) is resolved to a spelling of the
   monotonic word list (``build/hunspell/el_monotonic.dic``) or to
   ``<UNK>``, and stored as an int32 code: the spelling's index in the
   sorted word list (plus 4 reserved ids) times 16, plus a 4-bit outcome
   that says how it was resolved. One array per shard, with the start of
   every document and its cap and sample buckets, cached under
   ``build/lm_el_encoded`` and reused while the shard, the word list and
   the resolver code are unchanged.
2. **Select.** Training documents are kept by the token cap the corpus
   manifest records for their source (``el_corpus.cap_bucket``) and,
   optionally, by an independent document-level sample, for the 20M and
   50M-token subsamples the pruning is tuned on. A source's weight in the
   corpus manifest repeats its arrays that many times, which is the same as
   multiplying its counts.
3. **Vocabulary.** The most frequent word-list spellings, plus every
   spelling the out-of-vocabulary table would otherwise store with a
   saturated count (below).
4. **Count.** Bigram and trigram keys are packed into uint64 (21 bits per
   id) and counted with ``np.unique`` over sentence-aligned chunks; the
   sorted runs are merged by key.
5. **Prune and write** ``vocab.json``, ``unigrams.json``,
   ``bigrams.tsv.gz``, ``trigrams.tsv.gz``, ``type_counts.tsv.gz``, the dev
   sentences and ``stats.json``.

Resolution
----------
The keyboard looks words up as NFC strings, literal first and then
lowercased (``LMReader.idCaseInsensitive``), and adds the enclitic's second
accent itself. Counting follows the same spellings:

* ``<NUM>``, ``<FOREIGN>`` and abbreviations with their period (``κ.``,
  ``π.χ.``) are ``<UNK>``: the word list holds none of them.
* A double-accented enclitic host (``όνομά`` in ``το όνομά μου``,
  ``άκουσέ`` in ``άκουσέ με``) keeps only its first tonos, so it is counted
  as ``όνομα``. The word list holds no double accents; the keyboard adds
  the second accent by rule when the enclitic follows.
* A capitalized first word of a sentence is lowercased when its lowercase
  is a list spelling (``Θα`` to ``θα``), and kept when only the capitalized
  spelling is one (``Αθήνα``).
* Inside a sentence a token is kept when it is a list spelling (``Νίκη``,
  the name, beside ``νίκη``), else lowercased when that is one (``Βουλή``
  to ``βουλή``), which is what the reader's lowercase fallback does.
* A token in capitals is ``<UNK>``. In running text these are acronyms
  and names (ΠΑΣΟΚ, ΗΠΑ, a speaker's name in the Parliament minutes), and
  resolving them by their letters, as the keyboard's all-caps pass resolves
  what a person types, maps acronyms onto words: ΕΣΥ (the health service)
  onto εσύ, ΝΑΤΟ onto νάτο. The capital articles Ο and Η and the
  disjunctive Ή are lowercased; any other one-letter capital is a label or
  an initial (τμήμα Α), not the interjection α.
* Everything else is ``<UNK>``. So the vocabulary holds only list
  spellings and the reserved tokens, and the model never proposes a word
  the list rejects.

Rows the keyboard cannot reach
------------------------------
The reader resolves a context word to an id only when the word is in the
vocabulary, so a row headed by ``<UNK>`` (bigram ``<UNK>``, trigram
``(<UNK>, w)`` or ``(w, <UNK>)``) is never read. They are left out
(``drop_unknown_contexts``), which leaves their room to trigram rows that
are read. Rows headed by ``<s>`` stay: the reader takes the literal ``<s>``
as the sentence start.

The out-of-vocabulary count table
---------------------------------
``export_lm.py`` appends exact training counts for the list spellings that
are out of the vocabulary and share their letters with another spelling
(πότε and ποτέ), one ``u8`` each, which ``LMReader`` reads as is. With
200M tokens many such spellings occur more than 255 times, and a capped
count would rank two frequent twins as equals. Every such spelling seen
more than 255 times therefore joins the vocabulary, where counts are u32,
so no count in the table is capped (``oov_counts_saturated`` is 0).

Pre-pruned intermediates
------------------------
The bigram and trigram files hold only what ``export_lm.py`` can keep: in
every row the entries down to the count of the row's last top-K entry (ties
included), the homograph spellings seen at least ``keep_homograph_min_count``
times, the completion of a homograph spelling's bigram row, and the bigram
of every kept trigram context (its denominator). Export cuts a row by count
and then by id, so any superset of the entries it keeps gives the same
file; the test suite checks that export from these files and from the full
tables is byte-identical. ``stats.json`` records the cuts the files were
pruned for, and ``export_lm.py`` refuses settings that would need more.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
import unicodedata
from array import array
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dilemma_build import el_corpus, el_lexicon, el_text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "build" / "el_corpus"
DEFAULT_DICTIONARY = ROOT / "build" / "hunspell" / "el_monotonic.dic"
DEFAULT_OUT = ROOT / "build" / "lm_el"
SANITY_OUT = ROOT / "build" / "lm_el_sanity"
CORPUS_MANIFEST = ROOT / "data" / "el_lm_corpora.json"

RESERVED = ("<PAD>", "<UNK>", "<s>", "</s>")
PAD, UNK, BOS, EOS = 0, 1, 2, 3
N_RESERVED = len(RESERVED)
ID_BITS = 21
ID_MASK = (1 << ID_BITS) - 1
MAX_VOCAB = 1 << ID_BITS

# Outcome of a token's resolution, the low 3 bits of its code; bit 3 marks
# an enclitic host whose second tonos was removed.
OUTCOME_BITS = 4
EXACT, LOWERED_INITIAL, LOWERED_INSIDE = 0, 1, 2
UNK_PLACEHOLDER, UNK_ABBREVIATION, UNK_CAPS, UNK_NOT_LISTED = 4, 5, 6, 7
ENCLITIC_FOLDED = 8
OUTCOME_NAMES = {
    EXACT: "exact", LOWERED_INITIAL: "lowered_sentence_initial",
    LOWERED_INSIDE: "lowered_inside_sentence",
    UNK_PLACEHOLDER: "unk_placeholder", UNK_ABBREVIATION: "unk_abbreviation",
    UNK_CAPS: "unk_capitals", UNK_NOT_LISTED: "unk_not_in_word_list",
}

# The one-letter capitals that are words wherever they stand: the articles
# (after a colon or a quotation mark too) and the disjunctive Ή. Other
# one-letter capitals are labels and initials (τμήμα Α, Γ Παπανδρέου), not
# the interjection α.
ONE_LETTER_CAPITAL_WORDS = frozenset({"Ο", "Η", "Ή"})

# Same values as export_lm.py, for the pre-pruned files.
TOP_K_BI = 30
TOP_K_TRI = 15
KEEP_HOMOGRAPH_MIN_COUNT = 3
HOMOGRAPH_CONTEXT_MIN_COUNT = 3
OOV_COUNT_CAP = 255

SAMPLE_SEED = 2626
# The shipped settings (docs/el-monotonic-lm.md, "Pruning" and "Vocabulary
# size"): measured on the full capped corpus, these give a file of about
# 60 MB with the best next-word top-1 and top-3 of the settings near that
# size.
DEFAULT_VOCAB_SIZE = 150_000
MIN_COUNT_BI = 2
MIN_COUNT_TRI = 4
MIN_BIGRAM_FOR_TRI = 5
DEFAULT_CHUNK_TOKENS = 64_000_000


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

def fold_enclitic_accent(token: str) -> str:
    """``token`` with every tonos after the first removed: the spelling of
    a word without the second accent an enclitic gives it (όνομά to
    όνομα, άκουσέ to άκουσε). A token with at most one tonos is returned
    unchanged."""
    nfd = unicodedata.normalize("NFD", token)
    if nfd.count("́") < 2:
        return token
    first = nfd.index("́")
    rest = nfd[first + 1:].replace("́", "")
    return unicodedata.normalize("NFC", nfd[:first + 1] + rest)


def load_dictionary(path: Path) -> list[str]:
    """The spellings of a word-list ``.dic`` (first field of every line
    after the count), sorted by code point, which is UTF-8 byte order."""
    words = set()
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            word = line.split("\t", 1)[0].split(" ", 1)[0].strip()
            if not word or (i == 0 and word.isdigit()) or word.startswith("#"):
                continue
            words.add(unicodedata.normalize("NFC", word.split("/", 1)[0]))
    return sorted(words)


class Resolver:
    """Resolve corpus tokens to word-list spellings (see the module
    docstring). ``code(token, initial)`` returns ``full_id * 16 + outcome``,
    where ``full_id`` is the spelling's index in ``entries`` plus
    ``N_RESERVED``, or ``UNK``."""

    def __init__(self, entries: list[str]):
        self.entries = entries
        self.index = {e: i + N_RESERVED for i, e in enumerate(entries)}
        self._initial: dict[str, int] = {}
        self._inside: dict[str, int] = {}

    def resolve(self, token: str, initial: bool) -> tuple[int, int]:
        """``(full_id, outcome)`` for ``token``; ``initial`` when it opens
        a sentence."""
        if token in el_text.PLACEHOLDERS:
            return UNK, UNK_PLACEHOLDER
        if token.endswith("."):
            return UNK, UNK_ABBREVIATION
        word = fold_enclitic_accent(token)
        folded = ENCLITIC_FOLDED if word != token else 0
        if el_lexicon.case_shape(word) == "upper":
            n_letters = sum(c in el_lexicon.LETTERS for c in word)
            if n_letters > 1 or (word not in ONE_LETTER_CAPITAL_WORDS
                                 and el_lexicon.APOSTROPHE not in word):
                return UNK, UNK_CAPS | folded
        low = el_lexicon.lower_el(word)
        if initial and low != word and low in self.index:
            return self.index[low], LOWERED_INITIAL | folded
        if word in self.index:
            return self.index[word], EXACT | folded
        if low != word and low in self.index:
            return self.index[low], LOWERED_INSIDE | folded
        return UNK, UNK_NOT_LISTED | folded

    def code(self, token: str, initial: bool) -> int:
        cache = self._initial if initial else self._inside
        c = cache.get(token)
        if c is None:
            full_id, outcome = self.resolve(token, initial)
            c = cache[token] = (full_id << OUTCOME_BITS) | outcome
        return c


def resolver_digest() -> str:
    """sha256 of the code that decides an encoded array's content, so a
    change to it invalidates the cache."""
    src = "".join(inspect.getsource(f) for f in (
        fold_enclitic_accent, load_dictionary, Resolver, encode_shard,
        el_lexicon.case_shape, el_lexicon.lower_el))
    src += repr((RESERVED, OUTCOME_BITS, SAMPLE_SEED, el_corpus.CAP_SEED,
                 sorted(ONE_LETTER_CAPITAL_WORDS),
                 el_corpus.BUCKETS, sorted(el_text.PLACEHOLDERS)))
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def sample_bucket(doc_id: str, seed: int = SAMPLE_SEED) -> int:
    """A document's bucket for subsamples, independent of the cap bucket
    and of the split."""
    return el_corpus._bucket(f"sample:{seed}:{doc_id}")


# --------------------------------------------------------------------------
# Encoding (worker processes)
# --------------------------------------------------------------------------

_RESOLVER: Resolver | None = None


def _init_worker(entries: list[str]) -> None:
    global _RESOLVER
    _RESOLVER = Resolver(entries)


def encode_shard(path: str) -> dict:
    """Encode one shard: ``codes`` (int32: ``<s>``, the codes of a
    sentence's tokens and ``</s>``, for every sentence in order),
    ``doc_starts`` (int64, the index of each document's first code), and
    each document's ``cap_bucket`` and ``sample_bucket`` (int16)."""
    resolver = _RESOLVER
    code = resolver.code
    initial = resolver._initial
    inside = resolver._inside
    bos = BOS << OUTCOME_BITS
    eos = EOS << OUTCOME_BITS
    codes = array("i")
    doc_starts = array("q")
    cap_b = array("h")
    sample_b = array("h")
    for doc in el_corpus.iter_shard_documents(Path(path)):
        doc_starts.append(len(codes))
        cap_b.append(el_corpus.cap_bucket(doc["id"]))
        sample_b.append(sample_bucket(doc["id"]))
        for sent in doc["s"]:
            toks = sent.split(" ")
            codes.append(bos)
            first = toks[0]
            c = initial.get(first)
            codes.append(c if c is not None else code(first, True))
            codes.extend([c if (c := inside.get(t)) is not None
                          else code(t, False) for t in toks[1:]])
            codes.append(eos)
    return {
        "codes": np.frombuffer(codes, dtype=np.int32).copy(),
        "doc_starts": np.frombuffer(doc_starts, dtype=np.int64).copy(),
        "cap_bucket": np.frombuffer(cap_b, dtype=np.int16).copy(),
        "sample_bucket": np.frombuffer(sample_b, dtype=np.int16).copy(),
    }


def _encode_to(task: tuple[str, str]) -> str:
    shard, out = task
    arrays = encode_shard(shard)
    tmp = out + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, out)
    return out


# --------------------------------------------------------------------------
# Encoded corpus
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class EncodedShard:
    """One encoded shard of one source and split."""

    def __init__(self, source: str, split: str, path: Path, weight: int,
                 cap_threshold: int):
        self.source = source
        self.split = split
        self.path = path
        self.weight = weight
        self.cap_threshold = cap_threshold
        self._arrays = None

    def arrays(self) -> dict:
        if self._arrays is None:
            with np.load(self.path) as z:
                self._arrays = {k: z[k] for k in z.files}
        return self._arrays

    def release(self) -> None:
        self._arrays = None

    def doc_mask(self, caps: bool, sample_fraction: float | None) -> np.ndarray:
        a = self.arrays()
        mask = np.ones(len(a["doc_starts"]), dtype=bool)
        if caps and self.split == "train" and self.cap_threshold < el_corpus.BUCKETS:
            mask &= a["cap_bucket"] < self.cap_threshold
        if sample_fraction is not None:
            mask &= a["sample_bucket"] < round(sample_fraction * el_corpus.BUCKETS)
        return mask

    def codes(self, caps: bool = True,
              sample_fraction: float | None = None) -> np.ndarray:
        """The codes of the selected documents, in order."""
        a = self.arrays()
        mask = self.doc_mask(caps, sample_fraction)
        if mask.all():
            return a["codes"]
        lengths = np.diff(np.append(a["doc_starts"], len(a["codes"])))
        return a["codes"][np.repeat(mask, lengths)]


def encode_corpus(corpus_dir: Path, dictionary: Path, cache_dir: Path,
                  entries: list[str], *, workers: int = 0,
                  max_shards: int | None = None,
                  sources: list[str] | None = None) -> list[EncodedShard]:
    """Encode (or reuse) every train and dev shard of the corpus."""
    corpus_dir = Path(corpus_dir)
    meta = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    dic_sha = sha256_file(dictionary)
    digest = resolver_digest()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp_path = cache_dir / "stamp.json"
    stamp = {"dictionary_sha256": dic_sha, "resolver_sha256": digest}
    old = (json.loads(stamp_path.read_text(encoding="utf-8"))
           if stamp_path.exists() else None)
    if old is not None and {k: old.get(k) for k in stamp} != stamp:
        for p in cache_dir.glob("*/*.npz"):
            p.unlink()
        old = None
    shard_digests = old.get("shards", {}) if old else {}

    shards: list[EncodedShard] = []
    todo: list[tuple[str, str]] = []
    for source, entry in meta["sources"].items():
        if sources and source not in sources:
            continue
        weight = int(entry.get("weight", 1))
        threshold = int(entry["cap"]["bucket_threshold"])
        for split in ("train", "dev"):
            listed = entry["splits"][split]["shards"]
            if max_shards is not None:
                listed = listed[:max_shards]
            for shard in listed:
                src = corpus_dir / source / shard["file"]
                out = cache_dir / source / (shard["file"].replace(".jsonl.gz", ".npz"))
                out.parent.mkdir(parents=True, exist_ok=True)
                key = f"{source}/{shard['file']}"
                if not (out.exists() and shard_digests.get(key) == shard["sha256"]):
                    todo.append((str(src), str(out)))
                    shard_digests[key] = shard["sha256"]
                shards.append(EncodedShard(source, split, out, weight, threshold))
    if todo:
        t0 = time.time()
        n = workers or min(len(todo), os.cpu_count() or 4)
        print(f"  encoding {len(todo)} shard(s) with {n} worker(s)", flush=True)
        if n <= 1:
            _init_worker(entries)
            for task in todo:
                _encode_to(task)
        else:
            with ProcessPoolExecutor(max_workers=n, initializer=_init_worker,
                                     initargs=(entries,)) as pool:
                for done in pool.map(_encode_to, todo):
                    pass
        print(f"  encoded in {time.time() - t0:.1f}s", flush=True)
    stamp["shards"] = dict(sorted(shard_digests.items()))
    stamp_path.write_text(json.dumps(stamp, indent=1), encoding="utf-8")
    return shards


def selection(shards: list[EncodedShard], *, caps: bool = True,
              sample_fraction: float | None = None,
              split: str = "train") -> list[tuple[EncodedShard, np.ndarray]]:
    """``(shard, codes)`` of the selected documents of every shard of a
    split. Dev is never capped or sampled."""
    out = []
    for sh in shards:
        if sh.split != split:
            continue
        if split == "train":
            out.append((sh, sh.codes(caps, sample_fraction)))
        else:
            out.append((sh, sh.codes(False, None)))
    return out


# --------------------------------------------------------------------------
# Counting
# --------------------------------------------------------------------------

def outcome_counts(selected) -> dict:
    """Token counts per resolution outcome (unweighted), BOS/EOS left out."""
    totals = np.zeros(16, dtype=np.int64)
    for sh, codes in selected:
        ids = codes >> OUTCOME_BITS
        word = (ids != BOS) & (ids != EOS)
        totals += np.bincount(codes[word] & 15, minlength=16)
    out = {name: int(totals[k] + totals[k | ENCLITIC_FOLDED])
           for k, name in OUTCOME_NAMES.items()}
    out["enclitic_folded"] = int(totals[ENCLITIC_FOLDED:].sum())
    return out


def full_counts(selected, n_full: int) -> np.ndarray:
    """Weighted count of every full id (reserved ids included)."""
    counts = np.zeros(n_full, dtype=np.int64)
    for sh, codes in selected:
        counts += sh.weight * np.bincount(codes >> OUTCOME_BITS, minlength=n_full)
    return counts


def _export_lm():
    """``export_lm`` (repository root), whose spelling keys and homograph
    rule this module must share."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import export_lm
    return export_lm


def contested_mask(entries: list[str]) -> np.ndarray:
    """True for every full id whose spelling shares its letters with another
    list spelling that differs by more than case (``export_lm`` keys)."""
    ex = _export_lm()
    bare_letters, spelling_key = ex.bare_letters, ex.spelling_key
    keys: dict[str, set[str]] = defaultdict(set)
    for e in entries:
        keys[bare_letters(e)].add(spelling_key(e))
    mask = np.zeros(len(entries) + N_RESERVED, dtype=bool)
    for i, e in enumerate(entries):
        if len(keys[bare_letters(e)]) > 1:
            mask[i + N_RESERVED] = True
    return mask


def choose_vocab(counts: np.ndarray, entries: list[str], vocab_size: int,
                 contested: np.ndarray | None,
                 oov_cap: int = OOV_COUNT_CAP) -> tuple[list[int], int]:
    """Full ids of the vocabulary, reserved ids first, then by descending
    count (ties by spelling): the ``vocab_size - 4`` most frequent seen
    spellings, plus every contested spelling seen more than ``oov_cap``
    times (so the out-of-vocabulary table never caps a count). Returns the
    ids and the number of such additions."""
    seen = np.flatnonzero(counts[N_RESERVED:] > 0) + N_RESERVED
    # Sort by (-count, spelling); spellings sort like their indices.
    order = seen[np.lexsort((seen, -counts[seen]))]
    budget = max(0, vocab_size - N_RESERVED)
    top = order[:budget]
    forced = np.array([], dtype=np.int64)
    if contested is not None:
        rest = order[budget:]
        forced = rest[contested[rest] & (counts[rest] > oov_cap)]
    chosen = np.concatenate([top, forced])
    chosen = chosen[np.lexsort((chosen, -counts[chosen]))]
    if len(chosen) + N_RESERVED > MAX_VOCAB:
        raise ValueError(f"vocabulary of {len(chosen) + N_RESERVED:,} exceeds "
                         f"{MAX_VOCAB:,} ids")
    return list(range(N_RESERVED)) + chosen.tolist(), len(forced)


def map_to_vocab(vocab_full_ids: list[int], n_full: int) -> np.ndarray:
    """Lookup array: full id -> vocabulary id, ``UNK`` for the rest."""
    m = np.full(n_full, UNK, dtype=np.int32)
    m[np.asarray(vocab_full_ids, dtype=np.int64)] = np.arange(
        len(vocab_full_ids), dtype=np.int32)
    return m


def _merge_runs(runs: list[tuple[np.ndarray, np.ndarray]]):
    """Merge sorted (keys, counts) runs into one sorted run."""
    runs = [r for r in runs if len(r[0])]
    if not runs:
        return np.zeros(0, np.uint64), np.zeros(0, np.int64)
    if len(runs) == 1:
        return runs[0]
    keys = np.concatenate([k for k, _ in runs])
    counts = np.concatenate([c for _, c in runs])
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    counts = counts[order]
    starts = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
    return keys[starts], np.add.reduceat(counts, starts)


def ngram_keys(ids: np.ndarray, order: int) -> np.ndarray:
    """Packed keys of the bigrams (order 2) or trigrams (order 3) inside the
    sentences of ``ids`` (a stream of ``<s> ... </s>`` sentences)."""
    a = ids.astype(np.uint64)
    if order == 2:
        valid = ids[:-1] != EOS
        return ((a[:-1] << ID_BITS) | a[1:])[valid]
    valid = (ids[:-2] != EOS) & (ids[1:-1] != EOS)
    return ((a[:-2] << (2 * ID_BITS)) | (a[1:-1] << ID_BITS) | a[2:])[valid]


def count_ngrams(selected, full_to_vocab: np.ndarray, *,
                 chunk_tokens: int = DEFAULT_CHUNK_TOKENS):
    """Weighted unigram counts (by vocabulary id) and the sorted packed
    bigram and trigram ``(keys, counts)`` of the selected documents.

    Each shard's codes are mapped to vocabulary ids and cut into chunks at
    sentence ends; each chunk's keys are counted with ``np.unique`` and the
    resulting sorted runs are merged whenever they hold more than
    ``4 * chunk_tokens`` keys, and once at the end."""
    V = int(full_to_vocab.max()) + 1
    uni = np.zeros(V, dtype=np.int64)
    runs = {2: [], 3: []}
    pending = {2: 0, 3: 0}
    for sh, codes in selected:
        if not len(codes):
            continue
        ids = full_to_vocab[codes >> OUTCOME_BITS]
        uni += sh.weight * np.bincount(ids, minlength=V)
        # Every sentence ends with </s>, so the last end is len(ids).
        ends = np.flatnonzero(ids == EOS) + 1
        start = 0
        while start < len(ids):
            j = int(np.searchsorted(ends, start + chunk_tokens, side="right")) - 1
            if j < 0 or ends[j] <= start:
                j = int(np.searchsorted(ends, start, side="right"))
            stop = int(ends[j])
            chunk = ids[start:stop]
            for n in (2, 3):
                k, c = np.unique(ngram_keys(chunk, n), return_counts=True)
                runs[n].append((k, c.astype(np.int64) * sh.weight))
                pending[n] += len(k)
                if pending[n] > 4 * chunk_tokens:
                    runs[n] = [_merge_runs(runs[n])]
                    pending[n] = len(runs[n][0][0])
            start = stop
    bi = _merge_runs(runs[2])
    tri = _merge_runs(runs[3])
    return uni, bi, tri


# --------------------------------------------------------------------------
# Pruning
# --------------------------------------------------------------------------

def homograph_vocab_mask(vocab: list[str]) -> np.ndarray:
    """True for the vocabulary ids ``export_lm.homograph_spelling_ids``
    returns (that function decides which rows keep spellings past the
    cut)."""
    mask = np.zeros(len(vocab), dtype=bool)
    ids = list(_export_lm().homograph_spelling_ids(vocab))
    if ids:
        mask[np.asarray(ids)] = True
    return mask


def _row_keep(groups: np.ndarray, counts: np.ndarray, top_k: int) -> np.ndarray:
    """For entries grouped by ``groups`` (sorted ascending), True for every
    entry whose count reaches that of its group's ``top_k``-th largest
    count (every entry of a group with fewer than ``top_k``)."""
    n = len(groups)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort((-counts, groups))
    g = groups[order]
    c = counts[order]
    starts = np.flatnonzero(np.concatenate(([True], g[1:] != g[:-1])))
    sizes = np.diff(np.append(starts, n))
    kth_pos = starts + np.minimum(sizes, top_k) - 1
    threshold = np.where(sizes >= top_k, c[kth_pos], 0)
    thr_sorted = np.repeat(threshold, sizes)
    keep_sorted = c >= thr_sorted
    keep = np.zeros(n, dtype=bool)
    keep[order] = keep_sorted
    return keep


def prune(bi, tri, homograph: np.ndarray, *, min_count_bi: int = 1,
          min_count_tri: int = 1, min_bigram_for_tri: int = 1,
          drop_unknown_contexts: bool = True, preprune: bool = True,
          top_k_bi: int = TOP_K_BI, top_k_tri: int = TOP_K_TRI,
          keep_min: int = KEEP_HOMOGRAPH_MIN_COUNT,
          context_min: int = HOMOGRAPH_CONTEXT_MIN_COUNT) -> tuple:
    """The bigrams and trigrams to write. First the Ancient Greek trainer's
    pruning: bigrams seen fewer than ``min_count_bi`` times, trigrams seen
    fewer than ``min_count_tri`` times and trigrams whose context bigram
    (after the bigram cut) was seen fewer than ``min_bigram_for_tri`` times
    go. Then, with ``drop_unknown_contexts``, every row headed by ``<UNK>``.
    With ``preprune``, only the superset of entries ``export_lm.py`` can
    keep is written (see the module docstring)."""
    bk, bc = bi
    tk, tc = tri
    m = bc >= min_count_bi
    bk, bc = bk[m], bc[m]
    ctx = tk >> np.uint64(ID_BITS)       # the packed bigram key of (w1, w2)
    if len(bk):
        pos = np.minimum(np.searchsorted(bk, ctx), len(bk) - 1)
        ctx_count = np.where(bk[pos] == ctx, bc[pos], 0)
    else:
        ctx_count = np.zeros(len(tk), dtype=np.int64)
    m = (tc >= min_count_tri) & (ctx_count >= min_bigram_for_tri)
    if drop_unknown_contexts:
        w1 = (tk >> np.uint64(2 * ID_BITS)).astype(np.int64)
        w2 = ((tk >> np.uint64(ID_BITS)) & np.uint64(ID_MASK)).astype(np.int64)
        m &= (w1 != UNK) & (w2 != UNK)
    tk, tc = tk[m], tc[m]
    if drop_unknown_contexts:
        b1 = (bk >> np.uint64(ID_BITS)).astype(np.int64)
        keep_b = b1 != UNK
        bk, bc = bk[keep_b], bc[keep_b]
    if not preprune:
        return (bk, bc), (tk, tc)

    # Trigram rows: the top-K superset plus kept homograph spellings.
    t3 = (tk & np.uint64(ID_MASK)).astype(np.int64)
    t_ctx = (tk >> np.uint64(ID_BITS))
    cont = t3 >= N_RESERVED
    keep_t = np.zeros(len(tk), dtype=bool)
    idx = np.flatnonzero(cont)
    keep_t[idx] = _row_keep(t_ctx[idx], tc[idx], top_k_tri)
    if keep_min > 0:
        keep_t |= cont & homograph[t3] & (tc >= keep_min)
    tk, tc = tk[keep_t], tc[keep_t]

    # Bigram rows: the same, plus the completion of homograph rows, plus
    # every bigram that is a kept trigram context (its denominator).
    b1 = (bk >> np.uint64(ID_BITS)).astype(np.int64)
    b2 = (bk & np.uint64(ID_MASK)).astype(np.int64)
    cont = b2 >= N_RESERVED
    keep_b = np.zeros(len(bk), dtype=bool)
    idx = np.flatnonzero(cont)
    keep_b[idx] = _row_keep(b1[idx], bc[idx], top_k_bi)
    if keep_min > 0:
        keep_b |= cont & homograph[b2] & (bc >= keep_min)
    if context_min > 0:
        keep_b |= cont & homograph[b1] & (bc >= context_min)
    needed = np.unique(tk >> np.uint64(ID_BITS))
    keep_b |= np.isin(bk, needed, assume_unique=False)
    return (bk[keep_b], bc[keep_b]), (tk, tc)


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def _gz_lines(path: Path, lines) -> None:
    with open(path, "wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=3) as gz:
        buf = []
        for line in lines:
            buf.append(line)
            if len(buf) >= 200_000:
                gz.write(("\n".join(buf) + "\n").encode("utf-8"))
                buf = []
        if buf:
            gz.write(("\n".join(buf) + "\n").encode("utf-8"))


def write_ngrams(out: Path, bi, tri) -> None:
    bk, bc = bi
    tk, tc = tri
    b1 = (bk >> np.uint64(ID_BITS)).tolist()
    b2 = (bk & np.uint64(ID_MASK)).tolist()
    _gz_lines(out / "bigrams.tsv.gz",
              (f"{a}\t{b}\t{c}" for a, b, c in zip(b1, b2, bc.tolist())))
    t1 = (tk >> np.uint64(2 * ID_BITS)).tolist()
    t2 = ((tk >> np.uint64(ID_BITS)) & np.uint64(ID_MASK)).tolist()
    t3 = (tk & np.uint64(ID_MASK)).tolist()
    _gz_lines(out / "trigrams.tsv.gz",
              (f"{a}\t{b}\t{c}\t{k}" for a, b, c, k in
               zip(t1, t2, t3, tc.tolist())))


def write_type_counts(out: Path, counts: np.ndarray, entries: list[str]) -> int:
    """``spelling<TAB>count`` of every seen word-list spelling, most frequent
    first (ties by spelling), weighted like the n-grams: what
    ``export_lm.py`` chooses the out-of-vocabulary table from."""
    seen = np.flatnonzero(counts[N_RESERVED:] > 0) + N_RESERVED
    order = seen[np.lexsort((seen, -counts[seen]))]
    _gz_lines(out / "type_counts.tsv.gz",
              (f"{entries[i - N_RESERVED]}\t{counts[i]}" for i in order.tolist()))
    return len(order)


def write_dev(out: Path, dev_selected, entries: list[str]) -> dict:
    """Dev sentences as resolved spellings (``<UNK>`` for tokens with no
    list spelling), one file per source and one for all, with ``<s>`` and
    ``</s>``."""
    names = np.array(list(RESERVED) + entries, dtype=object)
    per_source: dict[str, list[str]] = defaultdict(list)
    counts = {}
    for sh, codes in dev_selected:
        ids = codes >> OUTCOME_BITS
        toks = names[ids]
        ends = np.flatnonzero(ids == EOS) + 1
        start = 0
        lines = per_source[sh.source]
        for e in ends.tolist():
            lines.append(" ".join(toks[start:e].tolist()))
            start = e
    with open(out / "dev_sentences.txt", "w", encoding="utf-8") as allf:
        for source, lines in per_source.items():
            text = "\n".join(lines) + "\n" if lines else ""
            (out / f"dev_sentences_{source}.txt").write_text(text, encoding="utf-8")
            allf.write(text)
            counts[source] = len(lines)
    return counts


def git_commit() -> str:
    """HEAD, marked ``-dirty`` when tracked files differ from it."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode()
    except Exception:
        return "unknown"
    return f"{commit}-dirty" if status.strip() else commit


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def sample_fraction_for(target_tokens: int, train_selected) -> float:
    """The document-sample fraction that keeps about ``target_tokens``
    weighted training tokens of the capped mixture."""
    total = sum(sh.weight * len(codes) for sh, codes in train_selected)
    return min(1.0, target_tokens / max(1, total))


class Counted:
    """Everything :func:`count_corpus` measured, for one or more writes."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def count_corpus(corpus_dir: Path, dictionary: Path, *, vocab_size: int,
                 cache_dir: Path | None = None, caps: bool = True,
                 sample_fraction: float | None = None,
                 sample_tokens: int | None = None, sanity: bool = False,
                 workers: int = 0, chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
                 oov_safe_vocab: bool = True, entries: list[str] | None = None,
                 shards: list[EncodedShard] | None = None) -> Counted:
    """Encode (or reuse) the corpus, select the training documents, choose
    the vocabulary and count unigrams, bigrams and trigrams."""
    corpus_dir, dictionary = Path(corpus_dir), Path(dictionary)
    if entries is None:
        entries = load_dictionary(dictionary)
        print(f"Word list: {len(entries):,} spellings from {dictionary}", flush=True)
    n_full = len(entries) + N_RESERVED
    if shards is None:
        cache = Path(cache_dir) if cache_dir else ROOT / "build" / "lm_el_encoded"
        shards = encode_corpus(corpus_dir, dictionary, cache, entries,
                               workers=workers, max_shards=1 if sanity else None)
    fraction = sample_fraction
    if sample_tokens:
        fraction = sample_fraction_for(sample_tokens, selection(shards, caps=caps))
    train_sel = selection(shards, caps=caps, sample_fraction=fraction)

    t0 = time.time()
    counts = full_counts(train_sel, n_full)
    contested = contested_mask(entries) if oov_safe_vocab else None
    vocab_full, forced = choose_vocab(counts, entries, vocab_size, contested)
    names = list(RESERVED) + entries
    vocab = [names[i] for i in vocab_full]
    full_to_vocab = map_to_vocab(vocab_full, n_full)
    print(f"Vocabulary: {len(vocab):,} ids ({forced:,} added so no "
          f"out-of-vocabulary count exceeds {OOV_COUNT_CAP}) of "
          f"{int((counts[N_RESERVED:] > 0).sum()):,} seen spellings "
          f"in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    uni, bi, tri = count_ngrams(train_sel, full_to_vocab, chunk_tokens=chunk_tokens)
    print(f"Counted {len(bi[0]):,} bigrams and {len(tri[0]):,} trigrams "
          f"over {int(uni.sum()):,} weighted tokens in "
          f"{time.time() - t0:.1f}s", flush=True)

    per_source: dict[str, dict] = {}
    for sh, codes in train_sel:
        ids = codes >> OUTCOME_BITS
        n_words = int(((ids != BOS) & (ids != EOS)).sum())
        d = per_source.setdefault(sh.source, {"weight": sh.weight,
                                              "train_tokens": 0,
                                              "weighted_tokens": 0})
        d["train_tokens"] += n_words
        d["weighted_tokens"] += n_words * sh.weight
    return Counted(
        corpus_dir=corpus_dir, dictionary=dictionary, entries=entries,
        shards=shards, caps=caps, sample_fraction=fraction, sanity=sanity,
        vocab_size_requested=vocab_size, vocab=vocab, vocab_full=vocab_full,
        forced=forced, counts=counts, uni=uni, bi=bi, tri=tri,
        homograph=homograph_vocab_mask(vocab),
        outcomes=outcome_counts(train_sel), per_source=per_source)


def write_intermediates(c: Counted, out: Path, *,
                        min_count_bi: int = MIN_COUNT_BI,
                        min_count_tri: int = MIN_COUNT_TRI,
                        min_bigram_for_tri: int = MIN_BIGRAM_FOR_TRI,
                        preprune: bool = True, dev: bool = True) -> dict:
    """Prune ``c``'s counts and write the files ``export_lm.py`` reads, the
    dev sentences (unless ``dev`` is False) and ``stats.json``."""
    t0 = time.time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bi_w, tri_w = prune(
        c.bi, c.tri, c.homograph, min_count_bi=min_count_bi,
        min_count_tri=min_count_tri, min_bigram_for_tri=min_bigram_for_tri,
        drop_unknown_contexts=True, preprune=preprune)
    write_ngrams(out, bi_w, tri_w)
    (out / "vocab.json").write_text(json.dumps(c.vocab, ensure_ascii=False),
                                    encoding="utf-8")
    (out / "unigrams.json").write_text(
        json.dumps({str(i): int(n) for i, n in enumerate(c.uni.tolist()) if n}),
        encoding="utf-8")
    n_types = write_type_counts(out, c.counts, c.entries)
    per_source = {k: dict(v) for k, v in c.per_source.items()}
    if dev:
        dev_counts = write_dev(out, selection(c.shards, split="dev"), c.entries)
        for source, n in dev_counts.items():
            per_source.setdefault(source, {})["dev_sentences"] = n
    print(f"Wrote {len(bi_w[0]):,} bigrams and {len(tri_w[0]):,} trigrams "
          f"in {time.time() - t0:.1f}s", flush=True)

    counts = c.counts
    in_vocab = int(counts[np.asarray(c.vocab_full[N_RESERVED:], dtype=np.int64)].sum())
    words_weighted = int(counts[N_RESERVED:].sum() + counts[UNK])
    corpus_manifest = c.corpus_dir / "manifest.json"
    corpus_meta = json.loads(corpus_manifest.read_text(encoding="utf-8"))
    stats = {
        "lang": "el",
        "created_by": "train_lm.py --lang el",
        "commit": git_commit(),
        "corpus_dir": str(c.corpus_dir),
        "corpus_manifest_sha256": sha256_file(corpus_manifest),
        "el_lm_corpora_sha256": corpus_meta.get("corpus_manifest_sha256"),
        "corpus_commit": corpus_meta.get("commit"),
        "dictionary": str(c.dictionary),
        "dictionary_sha256": sha256_file(c.dictionary),
        "dictionary_entries": len(c.entries),
        "resolver_sha256": resolver_digest(),
        "sanity": bool(c.sanity),
        "caps": c.caps,
        "sample_fraction": c.sample_fraction,
        "sample_seed": SAMPLE_SEED,
        "per_source": per_source,
        "resolution": c.outcomes,
        "n_train_tokens": int(c.uni.sum()),
        "n_train_words_weighted": words_weighted,
        "n_seen_spellings": int((counts[N_RESERVED:] > 0).sum()),
        "type_counts": n_types,
        "vocab_size_requested": c.vocab_size_requested,
        "vocab_size": len(c.vocab),
        "vocab_added_for_oov_cap": c.forced,
        "oov_count_cap": OOV_COUNT_CAP,
        "in_vocab_word_share": in_vocab / max(1, words_weighted),
        "unk_word_share": int(counts[UNK]) / max(1, words_weighted),
        "n_bigrams_counted": len(c.bi[0]),
        "n_trigrams_counted": len(c.tri[0]),
        "n_bigrams_written": len(bi_w[0]),
        "n_trigrams_written": len(tri_w[0]),
        "min_count_bi": min_count_bi,
        "min_count_tri": min_count_tri,
        "min_bigram_for_tri": min_bigram_for_tri,
        "drop_unknown_contexts": True,
        "prepruned_for": None if not preprune else {
            "top_k_bi": TOP_K_BI, "top_k_tri": TOP_K_TRI,
            "keep_homograph_min_count": KEEP_HOMOGRAPH_MIN_COUNT,
            "homograph_context_min_count": HOMOGRAPH_CONTEXT_MIN_COUNT,
        },
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    return stats


def train(args) -> dict:
    t0 = time.time()
    c = count_corpus(
        args.corpus_dir, args.dictionary, vocab_size=args.vocab_size,
        cache_dir=args.cache_dir, caps=not args.no_caps,
        sample_fraction=args.sample_fraction, sample_tokens=args.sample_tokens,
        sanity=args.sanity, workers=args.workers,
        chunk_tokens=args.chunk_tokens, oov_safe_vocab=args.oov_safe_vocab)
    stats = write_intermediates(
        c, Path(args.out), min_count_bi=args.min_count_bi,
        min_count_tri=args.min_count_tri,
        min_bigram_for_tri=args.min_bigram_for_tri,
        preprune=not args.no_preprune)
    print(f"\nWrote intermediates to {args.out}/ in {time.time() - t0:.1f}s",
          flush=True)
    return stats


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="train_lm.py --lang el",
        description="Count the monotonic Modern Greek next-word model.")
    ap.add_argument("--lang", choices=["el"], default="el")
    ap.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS),
                    help="Extracted corpus (extract_el_monotonic_lm.py).")
    ap.add_argument("--dictionary", default=str(DEFAULT_DICTIONARY),
                    help="Word list whose spellings make the vocabulary "
                         "(export_el_monotonic.py).")
    ap.add_argument("--out", default=None,
                    help=f"Intermediate directory (default {DEFAULT_OUT}; "
                         f"{SANITY_OUT} with --sanity).")
    ap.add_argument("--cache-dir", default=None,
                    help="Encoded shards (default build/lm_el_encoded).")
    ap.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    ap.add_argument("--no-oov-safe-vocab", dest="oov_safe_vocab",
                    action="store_false",
                    help="Do not add the contested spellings seen more than "
                         f"{OOV_COUNT_CAP} times to the vocabulary.")
    ap.add_argument("--min-count-bi", type=int, default=MIN_COUNT_BI,
                    help="Drop bigram entries with a lower weighted count "
                         f"(default {MIN_COUNT_BI}).")
    ap.add_argument("--min-count-tri", type=int, default=MIN_COUNT_TRI,
                    help="Drop trigram entries with a lower weighted count "
                         f"(default {MIN_COUNT_TRI}).")
    ap.add_argument("--min-bigram-for-tri", type=int, default=MIN_BIGRAM_FOR_TRI,
                    help="Drop trigram rows whose two-word context has a "
                         f"lower weighted count (default {MIN_BIGRAM_FOR_TRI}).")
    ap.add_argument("--no-caps", action="store_true",
                    help="Read every training document, ignoring the token "
                         "caps of the corpus manifest.")
    ap.add_argument("--sample-tokens", type=int, default=None,
                    help="Keep a document-level sample of about this many "
                         "weighted training tokens.")
    ap.add_argument("--sample-fraction", type=float, default=None,
                    help="Keep this share of the training documents.")
    ap.add_argument("--sanity", action="store_true",
                    help="First shard of every source, a 5%% document "
                         "sample; under a minute end to end.")
    ap.add_argument("--no-preprune", action="store_true",
                    help="Write every n-gram that survives the count cuts, "
                         "not only those export_lm.py can keep.")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    return ap


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    if args.sanity and args.sample_fraction is None and not args.sample_tokens:
        args.sample_fraction = 0.05
    partial = args.sanity or args.sample_tokens or args.sample_fraction is not None
    if args.out is None:
        args.out = str(SANITY_OUT if args.sanity else DEFAULT_OUT)
    if partial and Path(args.out).resolve() == DEFAULT_OUT.resolve():
        raise SystemExit(f"A sampled run cannot write {DEFAULT_OUT}; pass --out.")
    stats = train(args)
    print(json.dumps({k: v for k, v in stats.items() if k != "per_source"},
                     ensure_ascii=False, indent=2))
    return stats


if __name__ == "__main__":
    main()

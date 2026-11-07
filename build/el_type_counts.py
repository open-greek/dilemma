"""Per-source type counts of the extracted monotonic Modern Greek corpus.

The monotonic word list (``export_el_monotonic.py``) and its evaluation
read how often, and where, each token of the corpus that
``extract_el_monotonic_lm.py`` wrote occurs. Counting 400M tokens takes a
few minutes, so the counts are written once per source and split and
reused until the corpus changes.

For every token, exactly as the shards spell it (case kept, U+2019 kept,
abbreviations with their periods, placeholders included), a table holds:

* ``count``: its tokens;
* ``initial``: its tokens that open a sentence, where a capital says
  nothing about the word (Είναι, Κίνα);
* ``docs``: the documents it occurs in, its spread.

A token with U+2019 (σ’, απ’, ’ναι) also gets its neighbors, so the word
list can tell an elision from a numbering mark or a quotation mark:

* ``next_word``, ``next_vowel`` and ``next_clitic``: its tokens followed
  by a word (a token that starts with a Greek letter), by a word that
  starts with a vowel (απ’ όλα), and by an article or a weak pronoun
  (``CLITICS``: πάρ’ το, δώσ’ μου, απ’ τη);
* ``prev_word`` and ``prev_vowel``: its tokens after a word, and after a
  word that ends with a vowel or with U+2019 (να ’χει, σ’ ’κείνο).

The tables cover every document of the split, whatever the token cap of
the source: the cap sizes the next-word model's mixture, while the word
list wants all the evidence the training text has. The dev split is never
read for the list, so dev text stays held out from it.

Each table is ``<out_dir>/<split>/<source>.tsv.gz``: a first line ``#``
plus JSON (the source, the split, the sha256 of every shard read, token,
word, sentence and document totals), then ``token<TAB>count<TAB>initial
<TAB>docs`` rows sorted by token, with ``<TAB>next_word<TAB>next_vowel
<TAB>next_clitic<TAB>prev_word<TAB>prev_vowel`` after them on the rows of
tokens with U+2019. gzip headers carry no name and no time,
so a recount of the same shards gives the same bytes. A table whose shard
digests differ from the corpus manifest's is stale and is recounted.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from dilemma_build import el_corpus, el_lexicon, el_text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "build" / "el_monotonic" / "type_counts"
SCHEMA_VERSION = 3
# Articles and weak pronouns, before which colloquial speech elides a
# vowel that standard spelling keeps (πάρ’ το, δώσ’ μου, απ’ τη, μέσ’ στο).
CLITICS = frozenset(
    "το τα τον την τη τους τις του της των μου μας σου σας με σε "
    "στο στα στον στην στη στους στις στου στης στων".split())
APOSTROPHE = el_lexicon.APOSTROPHE
_LETTERS = el_lexicon.LETTERS
_VOWEL_LETTERS = frozenset(c for c in el_lexicon.LETTERS
                           if el_lexicon.bare_key(c) in el_lexicon.VOWELS)


@dataclass
class TypeCounts:
    """The counts of one source and split."""

    source: str
    split: str
    count: Counter = field(default_factory=Counter)
    initial: Counter = field(default_factory=Counter)
    docs: Counter = field(default_factory=Counter)
    # token with U+2019 -> [next_word, next_vowel, next_clitic, prev_word,
    # prev_vowel]
    context: dict = field(default_factory=dict)
    tokens: int = 0
    words: int = 0
    sentences: int = 0
    documents: int = 0
    shards: dict = field(default_factory=dict)

    def merge(self, other: "TypeCounts") -> None:
        self.count.update(other.count)
        self.initial.update(other.initial)
        self.docs.update(other.docs)
        for token, values in other.context.items():
            mine = self.context.setdefault(token, [0, 0, 0, 0, 0])
            for k, v in enumerate(values):
                mine[k] += v
        self.tokens += other.tokens
        self.words += other.words
        self.sentences += other.sentences
        self.documents += other.documents
        self.shards.update(other.shards)

    def header(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, "source": self.source,
                "split": self.split, "tokens": self.tokens,
                "words": self.words, "sentences": self.sentences,
                "documents": self.documents, "types": len(self.count),
                "shards": dict(sorted(self.shards.items()))}


def is_word_token(token: str) -> bool:
    """A token the keyboard could look up: not a placeholder, not an
    abbreviation with its period."""
    return token not in el_text.PLACEHOLDERS and not token.endswith(".")


def _count_context(tokens: list[str], context: dict) -> None:
    """Add the neighbors of the tokens with U+2019 in one sentence."""
    last = len(tokens) - 1
    for k, token in enumerate(tokens):
        if APOSTROPHE not in token:
            continue
        values = context.setdefault(token, [0, 0, 0, 0, 0])
        if k < last:
            nxt = tokens[k + 1]
            if nxt[:1] in _LETTERS:
                values[0] += 1
                if nxt[0] in _VOWEL_LETTERS:
                    values[1] += 1
                elif nxt.lower() in CLITICS:
                    values[2] += 1
        if k > 0:
            prev = tokens[k - 1]
            if prev[:1] in _LETTERS or prev[:1] == APOSTROPHE:
                values[3] += 1
                if prev[-1] in _VOWEL_LETTERS or prev[-1] == APOSTROPHE:
                    values[4] += 1


def count_documents(source: str, split: str, documents) -> TypeCounts:
    """Count ``documents`` (dicts with an ``s`` list of sentences)."""
    out = TypeCounts(source, split)
    count, initial, docs = out.count, out.initial, out.docs
    context = out.context
    placeholders = el_text.PLACEHOLDERS
    for doc in documents:
        seen: set[str] = set()
        out.documents += 1
        for sentence in doc["s"]:
            tokens = sentence.split(" ")
            count.update(tokens)
            initial[tokens[0]] += 1
            seen.update(tokens)
            if APOSTROPHE in sentence:
                _count_context(tokens, context)
            out.sentences += 1
            out.tokens += len(tokens)
            out.words += sum(1 for t in tokens
                             if t not in placeholders and not t.endswith("."))
        docs.update(seen)
    return out


def _count_shard(args) -> TypeCounts:
    source, split, path, digest = args
    out = count_documents(source, split, el_corpus.iter_shard_documents(path))
    out.shards[Path(path).name] = digest
    return out


def corpus_shards(corpus_dir: Path, source: str, split: str) -> dict[str, str]:
    """``{shard file: sha256}`` of one source and split, from the corpus
    manifest."""
    meta = json.loads((Path(corpus_dir) / "manifest.json").read_text(encoding="utf-8"))
    entry = meta["sources"][source]
    return {s["file"]: s["sha256"] for s in entry["splits"][split]["shards"]}


def corpus_sources(corpus_dir: Path) -> list[str]:
    meta = json.loads((Path(corpus_dir) / "manifest.json").read_text(encoding="utf-8"))
    return list(meta["sources"])


def count_source(corpus_dir: Path, source: str, split: str,
                 workers: int = 1) -> TypeCounts:
    """Count every shard of one source and split."""
    corpus_dir = Path(corpus_dir)
    shards = corpus_shards(corpus_dir, source, split)
    jobs = [(source, split, corpus_dir / source / name, digest)
            for name, digest in sorted(shards.items())]
    total = TypeCounts(source, split)
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            for part in pool.map(_count_shard, jobs):
                total.merge(part)
    else:
        for job in jobs:
            total.merge(_count_shard(job))
    return total


def table_path(out_dir: Path, source: str, split: str) -> Path:
    return Path(out_dir) / split / f"{source}.tsv.gz"


def write_table(counts: TypeCounts, path: Path) -> None:
    """Write ``counts`` byte-stably (see the module docstring)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                           compresslevel=6) as fh:
            fh.write(("#" + json.dumps(counts.header(), ensure_ascii=False,
                                       sort_keys=True) + "\n").encode("utf-8"))
            rows = []
            context = counts.context
            for token in sorted(counts.count):
                extra = ""
                if token in context:
                    extra = "".join(f"\t{v}" for v in context[token])
                rows.append(f"{token}\t{counts.count[token]}\t"
                            f"{counts.initial.get(token, 0)}\t"
                            f"{counts.docs.get(token, 0)}{extra}\n")
                if len(rows) >= 100_000:
                    fh.write("".join(rows).encode("utf-8"))
                    rows.clear()
            fh.write("".join(rows).encode("utf-8"))
    tmp.replace(path)


def read_header(path: Path) -> dict | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            first = fh.readline()
    except (OSError, EOFError):
        return None
    if not first.startswith("#"):
        return None
    try:
        return json.loads(first[1:])
    except json.JSONDecodeError:
        return None


def read_table(path: Path) -> TypeCounts:
    header = read_header(path)
    if header is None or header.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: not a type-count table of schema {SCHEMA_VERSION}")
    out = TypeCounts(header["source"], header["split"], tokens=header["tokens"],
                     words=header["words"], sentences=header["sentences"],
                     documents=header["documents"], shards=dict(header["shards"]))
    count, initial, docs = out.count, out.initial, out.docs
    context = out.context
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            token, n, first, d, *extra = line.rstrip("\n").split("\t")
            if extra:
                context[token] = [int(v) for v in extra]
            count[token] = int(n)
            if first != "0":
                initial[token] = int(first)
            docs[token] = int(d)
    return out


def ensure_tables(corpus_dir: Path, out_dir: Path = DEFAULT_OUT_DIR, *,
                  split: str = "train", sources: list[str] | None = None,
                  workers: int = 1, log=print) -> dict[str, Path]:
    """Count every source whose table is missing or stale; return
    ``{source: table path}``."""
    corpus_dir = Path(corpus_dir)
    out: dict[str, Path] = {}
    for source in sources or corpus_sources(corpus_dir):
        path = table_path(out_dir, source, split)
        wanted = corpus_shards(corpus_dir, source, split)
        header = read_header(path) if path.exists() else None
        if (header is None or header.get("schema_version") != SCHEMA_VERSION
                or header.get("shards") != wanted):
            counts = count_source(corpus_dir, source, split, workers)
            write_table(counts, path)
            if log:
                log(f"  counted {source} {split}: {counts.words:,} words, "
                    f"{len(counts.count):,} types")
        out[source] = path
    return out

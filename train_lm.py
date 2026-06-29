#!/usr/bin/env python3
"""Extract sentence-level n-gram counts from Ancient Greek corpora for
the polytonic next-word prediction LM.

Produces intermediate JSON/NPZ files under ``build/lm/`` which the
companion ``export_lm.py`` compiles into the mmap-friendly binary
artifact ``grc_ngram.bin`` consumed by the Tonos iOS keyboard extension.

Corpora
-------

Each corpus has its own loader that yields ``(sentence_id, [tokens])``
in the same shape (``BOS`` + polytonic NFC tokens + ``EOS``). ``main``
chains them through a single ingest loop. Adding a new corpus means
dropping in a new loader module and appending one entry to
``build_corpus_sources``; the counting, vocab-building, split, and eval
logic stay unchanged. This keeps parallel branches that add new corpora
mergeable without rework.

Currently registered:
    - GLAUx             (``iter_glaux_sentences``, inline below)
    - Diorisis          (``extract_diorisis_lm.iter_diorisis_sentences``)
    - Polytonic MG      (``extract_polytonic_mg.iter_polytonic_mg_sentences``)
    - Byzantine         (``extract_byzantine.iter_byzantine_sentences``)

Pipeline
--------

1. Walk each corpus's XML files, collect one token list per sentence in
   document order, preserving NFC polytonic forms exactly (no accent
   stripping, no case folding).
2. Deterministic train/dev split on sentence hash (seed=4242): 98%
   train / 2% dev. Sentence ids from different corpora share the same
   hash space but use per-corpus prefixes so the GLAUx dev bucket stays
   identical to the GLAUx-only baseline run. Same split across runs so
   eval numbers are reproducible.
3. Tokens: all surface forms with at least one Greek letter.
   Sentence-ending punctuation in GLAUx (``.`` ``;`` ``·`` etc.) is
   collapsed to ``</s>`` and a ``<s>`` token is prepended to every
   sentence. Diorisis has explicit ``<sentence>`` structure and no
   internal punctuation tokens, so ``</s>`` is appended unconditionally.
4. Truncate vocabulary to the top --vocab-size most frequent types.
   Everything else -> ``<UNK>``.
5. Count unigrams, bigrams, trigrams on the training split. Discard
   n-grams whose count falls below --min-count-{bi,tri}, and discard
   trigrams whose underlying (w1, w2) bigram was seen fewer than
   --min-bigram-for-tri times. The last filter is what keeps the
   artifact small: rare contexts fall back to bigram lookup at
   inference.
6. Write:
       build/lm/vocab.json          list[str] of tokens in id order
       build/lm/unigrams.json       {id: count}
       build/lm/bigrams.tsv.gz      w1_id \t w2_id \t count
       build/lm/trigrams.tsv.gz     w1_id \t w2_id \t w3_id \t count
       build/lm/dev_sentences.txt   one sentence per line, held-out eval
       build/lm/stats.json          corpus stats, for the README/report

Usage
-----

    python train_lm.py --sanity               # few files per corpus, < 1 min
    python train_lm.py                        # full: GLAUx + Diorisis + MG
    python train_lm.py --no-diorisis          # drop Diorisis from the mix
    python train_lm.py --no-polytonic-mg      # drop the Katharevousa slice
    python train_lm.py --glaux /path/to/xml   # custom GLAUx location
    python train_lm.py --diorisis /path       # custom Diorisis location

The sanity pass is deliberately small so the rest of the pipeline
(``export_lm.py``, ``eval_lm.py``) can be exercised end-to-end in under
a minute before committing to a long run.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_DIR = SCRIPT_DIR / "build" / "lm"
DEFAULT_GLAUX = Path.home() / "Documents" / "glaux" / "xml"
# Diorisis lives under dilemma/data/diorisis/xml in the canonical
# checkout. Fall back to that checkout if this script runs from a
# git worktree that doesn't have the XML unpacked locally.
DEFAULT_DIORISIS = SCRIPT_DIR / "data" / "diorisis" / "xml"
if not DEFAULT_DIORISIS.exists():
    _alt = (Path.home() / "Documents" / "dilemma"
            / "data" / "diorisis" / "xml")
    if _alt.exists():
        DEFAULT_DIORISIS = _alt

# Sentence-ending punctuation treated as </s>
SENT_END = {".", ";", "·", "!", "?"}

# Reserved vocabulary ids (stable across runs)
PAD_TOK = "<PAD>"       # id 0, never emitted; reserved for Swift reader
UNK_TOK = "<UNK>"       # id 1
BOS_TOK = "<s>"         # id 2
EOS_TOK = "</s>"        # id 3
RESERVED = [PAD_TOK, UNK_TOK, BOS_TOK, EOS_TOK]

SPLIT_SEED = 4242       # deterministic dev split
DEV_FRACTION = 0.02     # 2% held out


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def is_greek_token(s: str) -> bool:
    """True iff the token has at least one Greek letter.

    Polytonic Greek lives in U+0370..U+03FF and U+1F00..U+1FFF.
    """
    return any("\u0370" <= c <= "\u03FF" or "\u1F00" <= c <= "\u1FFF"
               for c in s)


def sentence_goes_to_dev(sent_id: str) -> bool:
    """Deterministic 2% dev split keyed on sentence id + seed."""
    h = hashlib.blake2b(
        f"{SPLIT_SEED}:{sent_id}".encode("utf-8"), digest_size=8
    ).digest()
    # interpret first 4 bytes as uint32, modulo 10000
    bucket = int.from_bytes(h[:4], "little") % 10_000
    return bucket < int(DEV_FRACTION * 10_000)


def iter_glaux_sentences(glaux_dir: Path, max_files: int | None = None):
    """Yield (sent_id, [token, ...]) for every sentence in GLAUx.

    Punctuation is converted: sentence-enders -> ``</s>``, everything
    else is dropped. ``<s>`` is prepended; ``</s>`` appended if the
    sentence did not already end with a sentence-ender.
    """
    xml_files = sorted(glaux_dir.glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No GLAUx XML files found at {glaux_dir}")
    if max_files is not None:
        xml_files = xml_files[:max_files]

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            continue

        doc_id = xml_file.stem
        for sent in tree.findall(".//sentence"):
            sid = sent.get("id") or sent.get("struct_id") or "?"
            global_sid = f"{doc_id}:{sid}"
            tokens: list[str] = [BOS_TOK]
            ends_with_eos = False
            for w in sent.findall("word"):
                form = w.get("form") or ""
                postag = w.get("postag") or ""
                if not form:
                    continue

                if postag and postag[0] == "u":
                    # punctuation; keep only sentence-enders as </s>
                    raw = form.strip()
                    if raw in SENT_END:
                        tokens.append(EOS_TOK)
                        ends_with_eos = True
                    continue

                form = nfc(form)
                if not is_greek_token(form):
                    continue
                tokens.append(form)

            if not ends_with_eos:
                tokens.append(EOS_TOK)

            # sanity: need at least one real token between <s> and </s>
            if len(tokens) >= 3:
                yield global_sid, tokens


def build_vocab(
    train_sents: list[list[str]], vocab_size: int
) -> tuple[list[str], dict[str, int], Counter]:
    """Build vocab from training sentences.

    Returns (id2tok, tok2id, raw_counts). id2tok[0..3] are reserved.
    """
    raw = Counter()
    for toks in train_sents:
        for t in toks:
            if t in (BOS_TOK, EOS_TOK):
                continue
            raw[t] += 1

    # most frequent fills the remaining vocab budget
    budget = vocab_size - len(RESERVED)
    top = [w for w, _ in raw.most_common(budget)]

    id2tok = list(RESERVED) + top
    tok2id = {t: i for i, t in enumerate(id2tok)}
    return id2tok, tok2id, raw


def count_ngrams(
    train_sents: list[list[str]],
    tok2id: dict[str, int],
    min_count_bi: int,
    min_count_tri: int,
    min_bigram_for_tri: int,
) -> tuple[Counter, Counter, Counter, int]:
    uni = Counter()
    bi = Counter()
    tri = Counter()
    unk_id = tok2id[UNK_TOK]
    total_tokens = 0

    for toks in train_sents:
        ids = [tok2id.get(t, unk_id) for t in toks]
        total_tokens += len(ids)
        n = len(ids)
        for i in range(n):
            uni[ids[i]] += 1
            if i >= 1:
                bi[(ids[i - 1], ids[i])] += 1
            if i >= 2:
                tri[(ids[i - 2], ids[i - 1], ids[i])] += 1

    # prune low-count n-grams to keep the artifact small
    if min_count_bi > 1:
        bi = Counter({k: v for k, v in bi.items() if v >= min_count_bi})
    if min_count_tri > 1:
        tri = Counter({k: v for k, v in tri.items() if v >= min_count_tri})
    if min_bigram_for_tri > 1:
        # only keep trigrams whose (w1, w2) bigram was seen >= threshold
        # in training. Rare contexts fall back to bigram lookup, which
        # is what the keyboard would do anyway.
        tri = Counter({
            (a, b, c): k for (a, b, c), k in tri.items()
            if bi.get((a, b), 0) >= min_bigram_for_tri
        })

    return uni, bi, tri, total_tokens


def build_corpus_sources(args, max_files):
    """Return list of (corpus_name, iterator) for the ingest loop.

    Each iterator yields ``(sentence_id, [tokens])`` with the same
    shape. Parallel branches that add new corpora should import their
    loader and append one line here; they should not modify the ingest
    loop in ``main`` itself.

    ``sentence_id`` must be stable across runs and unique across the
    whole registry so the hash-based dev split is deterministic. The
    GLAUx loader yields bare ``doc_id:sid`` strings (unprefixed, so the
    GLAUx bucket assignments match the GLAUx-only baseline run); every
    other loader namespaces its ids (e.g. ``diorisis:...``) to keep
    hash spaces independent.
    """
    sources: list[tuple[str, object]] = []

    glaux = Path(args.glaux)
    sources.append(("glaux", iter_glaux_sentences(glaux, max_files)))

    if not args.no_diorisis:
        # Imported lazily so GLAUx-only runs don't require the betacode
        # dependency.
        from extract_diorisis_lm import iter_diorisis_sentences
        diorisis = Path(args.diorisis)
        sources.append(
            ("diorisis", iter_diorisis_sentences(diorisis, max_files))
        )

    if not args.no_polytonic_mg:
        # Lazy import: pandas / HF parquet cache are only needed when
        # this slice is active. See extract_polytonic_mg.py for the
        # Wikisource 19th-c. Katharevousa register filter.
        from extract_polytonic_mg import iter_polytonic_mg_sentences

        mg_parquet = (
            Path(args.polytonic_mg_parquet)
            if args.polytonic_mg_parquet
            else None
        )
        # ``max_files`` on the XML corpora roughly bounds sentence count
        # for the sanity pass; for the parquet-backed MG slice we
        # translate it to a doc cap of the same order of magnitude.
        max_mg_docs = 50 if max_files is not None else None
        sources.append((
            "polytonic_mg",
            iter_polytonic_mg_sentences(
                parquet_path=mg_parquet, max_docs=max_mg_docs
            ),
        ))

    if not args.no_byzantine:
        # Lazy import; the Byzantine loader just does filesystem reads,
        # but keeping the same shape as the other entries makes the
        # registry pattern uniform.
        from extract_byzantine import iter_byzantine_sentences

        byz_dir = (
            Path(args.byzantine_dir) if args.byzantine_dir else None
        )
        sources.append((
            "byzantine",
            iter_byzantine_sentences(corpus_dir=byz_dir),
        ))

    return sources


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--glaux", type=str, default=str(DEFAULT_GLAUX))
    ap.add_argument("--diorisis", type=str, default=str(DEFAULT_DIORISIS))
    ap.add_argument("--no-diorisis", action="store_true",
                    help="Skip the Diorisis corpus; GLAUx only "
                         "(reproduces the original baseline).")
    ap.add_argument("--no-polytonic-mg", action="store_true",
                    help="Skip the Wikisource 19th-c. Katharevousa "
                         "slice (extract_polytonic_mg.py).")
    ap.add_argument("--polytonic-mg-parquet", type=str, default=None,
                    help="Override path to the Wikisource parquet "
                         "(defaults to the HF cache location).")
    ap.add_argument("--no-byzantine", action="store_true",
                    help="Skip the Byzantine vernacular corpus "
                         "(extract_byzantine.py).")
    ap.add_argument("--byzantine-dir", type=str, default=None,
                    help="Override the Byzantine corpus directory "
                         "(defaults to "
                         "~/Documents/corpus-of-open-greek/sources/byzantine).")
    ap.add_argument("--out", type=str, default=str(BUILD_DIR))
    ap.add_argument("--vocab-size", type=int, default=80_000)
    ap.add_argument("--min-count-bi", type=int, default=1)
    ap.add_argument("--min-count-tri", type=int, default=1)
    ap.add_argument("--min-bigram-for-tri", type=int, default=3,
                    help="Only keep trigrams whose (w1,w2) bigram "
                         "appeared at least this many times in "
                         "training. Keeps the trigram table focused "
                         "on contexts the user is actually likely to "
                         "type.")
    ap.add_argument("--sanity", action="store_true",
                    help="Only read ~40 XML files per corpus. "
                         "Runs the full pipeline end-to-end in <1 min.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    max_files = 40 if args.sanity else None

    t0 = time.time()
    train_sents: list[list[str]] = []
    dev_sents: list[list[str]] = []
    # Parallel list: corpus-of-origin for every entry in dev_sents, so
    # we can write per-corpus dev files for apples-to-apples evaluation
    # against the GLAUx-only baseline.
    dev_sent_corpora: list[str] = []
    n_raw_tokens = 0
    per_corpus_stats: dict[str, dict[str, int]] = {}

    for corpus_name, iterator in build_corpus_sources(args, max_files):
        c0 = time.time()
        c_train = c_dev = c_tokens = 0
        print(f"Scanning {corpus_name} ...", flush=True)
        for i, (sid, toks) in enumerate(iterator):
            n_raw_tokens += len(toks)
            c_tokens += len(toks)
            if sentence_goes_to_dev(sid):
                dev_sents.append(toks)
                dev_sent_corpora.append(corpus_name)
                c_dev += 1
            else:
                train_sents.append(toks)
                c_train += 1
            if (i + 1) % 50_000 == 0:
                print(f"  ...{corpus_name}: {i+1:,} sentences "
                      f"({c_tokens:,} tokens)", flush=True)
        per_corpus_stats[corpus_name] = {
            "train_sentences": c_train,
            "dev_sentences": c_dev,
            "tokens": c_tokens,
        }
        print(f"  {corpus_name}: {c_train:,} train / {c_dev:,} dev "
              f"sentences ({c_tokens:,} tokens) "
              f"in {time.time() - c0:.1f}s", flush=True)

    print(f"Collected {len(train_sents):,} train / {len(dev_sents):,} dev "
          f"sentences ({n_raw_tokens:,} tokens) "
          f"in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    id2tok, tok2id, raw_counts = build_vocab(train_sents, args.vocab_size)
    in_vocab = sum(1 for w in raw_counts if w in tok2id)
    print(f"Vocab: {len(id2tok):,} types (top-{args.vocab_size} of "
          f"{len(raw_counts):,} distinct; {in_vocab:,} kept) "
          f"in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    uni, bi, tri, total_tokens = count_ngrams(
        train_sents, tok2id,
        args.min_count_bi, args.min_count_tri,
        args.min_bigram_for_tri,
    )
    print(f"Unigrams: {len(uni):,}   Bigrams: {len(bi):,}   "
          f"Trigrams: {len(tri):,}   ({total_tokens:,} tokens) "
          f"in {time.time() - t0:.1f}s", flush=True)

    # ---- persist ----
    (out / "vocab.json").write_text(
        json.dumps(id2tok, ensure_ascii=False), encoding="utf-8"
    )

    with open(out / "unigrams.json", "w", encoding="utf-8") as f:
        # small; plain dict is fine
        json.dump({str(k): v for k, v in uni.items()}, f)

    with gzip.open(out / "bigrams.tsv.gz", "wt", encoding="utf-8") as f:
        for (a, b), c in bi.items():
            f.write(f"{a}\t{b}\t{c}\n")

    with gzip.open(out / "trigrams.tsv.gz", "wt", encoding="utf-8") as f:
        for (a, b, c), k in tri.items():
            f.write(f"{a}\t{b}\t{c}\t{k}\n")

    with open(out / "dev_sentences.txt", "w", encoding="utf-8") as f:
        for toks in dev_sents:
            f.write(" ".join(toks) + "\n")

    # Per-corpus dev files let downstream eval compare against
    # single-corpus baselines on the identical held-out sentences.
    per_corpus_dev: dict[str, list[list[str]]] = {}
    for toks, name in zip(dev_sents, dev_sent_corpora):
        per_corpus_dev.setdefault(name, []).append(toks)
    for name, toks_list in per_corpus_dev.items():
        path = out / f"dev_sentences_{name}.txt"
        with open(path, "w", encoding="utf-8") as f:
            for toks in toks_list:
                f.write(" ".join(toks) + "\n")

    oov_share = 1.0 - (
        sum(raw_counts[w] for w in raw_counts if w in tok2id)
        / max(1, sum(raw_counts.values()))
    )

    stats = {
        "glaux_dir": str(Path(args.glaux)),
        "diorisis_dir": (
            None if args.no_diorisis else str(Path(args.diorisis))
        ),
        "polytonic_mg_parquet": (
            None if args.no_polytonic_mg
            else (args.polytonic_mg_parquet or "<hf_cache_default>")
        ),
        "byzantine_dir": (
            None if args.no_byzantine
            else (args.byzantine_dir or "<default>")
        ),
        "per_corpus": per_corpus_stats,
        "sanity": args.sanity,
        "n_train_sentences": len(train_sents),
        "n_dev_sentences": len(dev_sents),
        "n_train_tokens": total_tokens,
        "n_distinct_types": len(raw_counts),
        "vocab_size": len(id2tok),
        "oov_token_share_train": oov_share,
        "n_unigrams": len(uni),
        "n_bigrams": len(bi),
        "n_trigrams": len(tri),
        "min_count_bi": args.min_count_bi,
        "min_count_tri": args.min_count_tri,
        "min_bigram_for_tri": args.min_bigram_for_tri,
        "split_seed": SPLIT_SEED,
        "dev_fraction": DEV_FRACTION,
    }
    (out / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote intermediates to {out}/", flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

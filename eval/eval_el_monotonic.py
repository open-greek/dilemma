#!/usr/bin/env python3
"""Coverage and false accepts of the monotonic Modern Greek word list.

Measures ``build/hunspell/el_monotonic.dic`` (``export_el_monotonic.py``)
against text it has never read and against two other word lists:

* **Held-out coverage.** The share of word tokens of the corpus's dev split
  (``build/el_corpus``, split by document group, so no dev document
  supplied any training evidence; the exporter reads the training split
  only) that the list accepts, per source and overall:

  - ``exact``: as spelled, the way a spelling checker accepts a word: a
    lowercase token if it is an entry; a capitalized token if it or its
    lowercase is an entry; an all-capital token if it is in the acronym
    list or its letters are an entry's (Greek capitals drop the tonos);
    A token with the second accent an enclitic gives it (όνομά μου) counts
    as accepted when the entry without that accent is one, as the
    keyboard adds that accent by rule and the list never holds it;
  - ``accent-insensitive``: its letters without tonos and dialytika
    (``el_lexicon.bare_key``) are an entry's, so the keyboard could
    restore the accents from its stripped index.

  Abbreviation tokens (with their period) are measured against the
  abbreviation list. Placeholders (<NUM>, <FOREIGN>) are not words.

* **False-accept spot checks.** A fixed set of misspellings (unaccented
  polysyllables, accented monosyllables, double accents, Latin
  look-alikes, old spellings, ASCII apostrophes) that must not be
  entries, and, for every common entry (C or M), its spellings with the
  tonos moved to another vowel or dropped: how many of those the list
  accepts. A moved accent is often another word (πότε, ποτέ), so that
  share is an upper bound.

* **elspell cross-check.** The LibreOffice Greek dictionary (el_GR 0.9 by
  Steve Stavropoulos and others, MPL 1.1/GPL 2.0/LGPL 2.1), pinned by
  commit and sha256, is downloaded with ``--download`` into
  ``data/raw/elspell/`` (gitignored). Its licenses do not allow shipping
  it in a closed app, so it is used here only, never to admit words. The
  check reports its own dev coverage as a baseline, the dev tokens it
  accepts that the list rejects (candidate false rejects), the dev tokens
  the list accepts that it rejects (candidate false accepts, or its gaps),
  and the list's entries it lacks, by rating.

* **Triantafyllidis headwords** (``data/triantafyllidis_headwords.json``,
  CC BY-NC-ND 4.0): the share of the single-word, monotonic headwords the
  list holds. Evaluation only; the exporter never reads them.

Usage:
    python eval/eval_el_monotonic.py --download      # first run: fetch elspell
    python eval/eval_el_monotonic.py                 # writes build/el_monotonic/eval.json
    python eval/eval_el_monotonic.py --no-elspell    # skip the cross-check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dilemma_build import el_lexicon as L  # noqa: E402
from dilemma_build import el_text  # noqa: E402
from dilemma_build import el_type_counts as TC  # noqa: E402

DEFAULT_DIC = ROOT / "build" / "hunspell" / "el_monotonic.dic"
DEFAULT_OUT = ROOT / "build" / "el_monotonic" / "eval.json"
DEFAULT_CORPUS = ROOT / "build" / "el_corpus"
TRIANTAFYLLIDIS = ROOT / "data" / "triantafyllidis_headwords.json"
ELSPELL_DIR = ROOT / "data" / "raw" / "elspell"
ELSPELL_COMMIT = "74d8e826359dbf38224fac1b9c02254b7fe0186a"
ELSPELL_URL = ("https://raw.githubusercontent.com/LibreOffice/dictionaries/"
               f"{ELSPELL_COMMIT}/el_GR/{{name}}")
ELSPELL_FILES = {
    "el_GR.dic": "e5b9b9c2cf05bbc59e03fe302b462dae85968f822f4fc219a8ed2879d6943720",
    "el_GR.aff": "868dfe13ad7d34251915a745e44d38542831463d56e60732ba3a5bc1dcca582a",
    "README_el_GR.txt": "8b942846bba64ac1b28c38aee294f1667d0063e2162d59030d0da09332e8c342",
}
ELSPELL_LICENSE = "MPL-1.1 OR GPL-2.0 OR LGPL-2.1"
SAMPLES = 40

# Misspellings a monotonic list must not hold, with what is wrong.
SPOT_CHECKS = {
    "ειναι": "no tonos", "καλημερα": "no tonos", "ανθρωπος": "no tonos",
    "ναί": "accented monosyllable", "δέν": "accented monosyllable",
    "θά": "accented monosyllable", "τό": "accented monosyllable",
    "μιά": "accented monosyllable by synizesis", "ποιός": "accented monosyllable by synizesis",
    "γειά": "accented monosyllable by synizesis",
    "όνομά": "double accent", "άνθρωπός": "double accent", "σπίτιά": "double accent",
    "τo": "Latin o", "κaι": "Latin a", "εivαι": "Latin i and v",
    "έιναι": "accent on the wrong letter", "πιστέυω": "accent on the wrong letter",
    "οτί": "accent on the wrong syllable", "εώς": "accent on the wrong syllable",
    "πρωϊνό": "unneeded dialytika", "τσάϊ": "unneeded dialytika",
    "προυπόθεση": "missing dialytika", "μαιμού": "missing dialytika",
    "νοιώθω": "old spelling", "κάθησε": "old spelling", "βαθειά": "old spelling",
    "διάλλειμα": "misspelling", "εισητήριο": "misspelling", "μύνημα": "misspelling",
    "γιαυτό": "two words written as one", "σ'": "ASCII apostrophe",
    "απ'": "ASCII apostrophe", "καλος": "no tonos",
}


# -- inputs --------------------------------------------------------------------

def read_dic(path: Path) -> dict[str, str]:
    """``{form: bucket}`` of a word list in the exporter's format."""
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    out = {}
    for line in lines[1:]:
        if line:
            form, fields = line.split("\t", 1)
            out[form] = fields.split()[0][3:]
    if int(lines[0]) != len(out):
        raise SystemExit(f"{path}: the count line says {lines[0]}, the file has {len(out)}")
    return out


def read_lines(path: Path) -> set[str]:
    if not Path(path).exists():
        return set()
    return {line for line in Path(path).read_text(encoding="utf-8").split("\n") if line}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_elspell(directory: Path, download: bool) -> Path | None:
    """The pinned elspell files, downloaded when asked; None when absent."""
    directory = Path(directory)
    for name, pin in ELSPELL_FILES.items():
        path = directory / name
        if not path.exists():
            if not download:
                return None
            directory.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(ELSPELL_URL.format(name=name), timeout=120) as r:
                tmp.write_bytes(r.read())
            tmp.replace(path)
        if sha256(path) != pin:
            raise SystemExit(f"{path}: sha256 {sha256(path)[:12]} is not the pinned {pin[:12]}")
    return directory


def read_elspell(directory: Path) -> set[str]:
    """elspell's words, NFC, ASCII apostrophes as U+2019. Its .dic is fully
    expanded (no word carries affix flags); one that did would need the
    .aff rules applied, which this reader refuses rather than guesses."""
    raw = (Path(directory) / "el_GR.dic").read_bytes().decode("iso-8859-7")
    lines = raw.split("\n")
    words = set()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        if "/" in line:
            raise SystemExit("el_GR.dic has affix flags; expand them before comparing")
        words.add(unicodedata.normalize("NFC", line.replace("'", L.APOSTROPHE)))
    return words


# -- coverage ------------------------------------------------------------------------

class Lists:
    def __init__(self, entries, acronyms=(), abbreviations=()):
        self.entries = set(entries)
        self.acronyms = set(acronyms)
        self.abbreviations = set(abbreviations)
        self.bare = {L.bare_key(f) for f in self.entries} | {
            L.bare_key(a) for a in self.acronyms}

    def accepts(self, token: str) -> bool:
        if token in self.entries:
            return True
        shape = L.case_shape(token.replace(L.APOSTROPHE, ""))
        if shape == "title":
            return L.lower_el(token) in self.entries or self.enclitic(token)
        if shape == "upper":
            return token in self.acronyms or L.bare_key(token) in self.bare
        return shape == "lower" and self.enclitic(token)

    def enclitic(self, token: str) -> bool:
        """A second accent before an enclitic (όνομά μου, Πρόεδρό μας),
        which the keyboard adds by rule to the entry without it."""
        if L.tonos_count(token) != 2:
            return False
        k = max(i for i, c in enumerate(token) if c in _UNACCENT)
        base = token[:k] + _UNACCENT[token[k]] + token[k + 1:]
        return (base in self.entries or L.lower_el(base) in self.entries) and (
            L.accent_position(base) in (2, 3))

    def accepts_bare(self, token: str) -> bool:
        return L.bare_key(token) in self.bare


def is_word(token: str) -> bool:
    if L.lower_el(token) == L.OTI:
        return True
    return (any(c in L.LETTERS for c in token)
            and all(c in L.LETTERS or c == L.APOSTROPHE for c in token))


def coverage(tables: list, lists: Lists, *, missing_limit: int = SAMPLES) -> dict:
    per_source = {}
    total = Counter()
    missing = Counter()
    for table in tables:
        c = Counter()
        for token, n in table.count.items():
            if token in el_text.PLACEHOLDERS:
                continue
            if token.endswith("."):
                c["abbreviation_tokens"] += n
                if token in lists.abbreviations:
                    c["abbreviations_listed"] += n
                continue
            if not is_word(token):
                c["other_tokens"] += n
                continue
            c["tokens"] += n
            shape = L.case_shape(token.replace(L.APOSTROPHE, ""))
            c[f"{shape}_tokens"] += n
            if lists.accepts(token):
                c["exact"] += n
                c[f"{shape}_exact"] += n
            else:
                missing[token] += n
            if lists.accepts_bare(token):
                c["bare"] += n
        total.update(c)
        per_source[table.source] = summarize(c, table.documents)
    out = {"overall": summarize(total, sum(t.documents for t in tables)),
           "per_source": per_source,
           "macro_exact": round(sum(s["exact_share"] for s in per_source.values())
                                / max(1, len(per_source)), 4),
           "missing_top": [[t, n] for t, n in missing.most_common(missing_limit)]}
    return out


def summarize(c: Counter, documents: int) -> dict:
    tokens = c["tokens"]
    share = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    out = {"documents": documents, "word_tokens": tokens,
           "exact_share": share(c["exact"], tokens),
           "accent_insensitive_share": share(c["bare"], tokens),
           "abbreviation_tokens": c["abbreviation_tokens"],
           "abbreviation_share": share(c["abbreviations_listed"], c["abbreviation_tokens"])}
    for shape in ("lower", "title", "upper", "mixed"):
        out[f"{shape}_tokens"] = c[f"{shape}_tokens"]
        out[f"{shape}_exact_share"] = share(c[f"{shape}_exact"], c[f"{shape}_tokens"])
    return out


# -- false accepts -------------------------------------------------------------------

_ACCENT = {"α": "ά", "ε": "έ", "η": "ή", "ι": "ί", "ο": "ό", "υ": "ύ", "ω": "ώ",
           "ϊ": "ΐ", "ϋ": "ΰ"}
_UNACCENT = {v: k for k, v in _ACCENT.items()}


def accent_variants(word: str) -> list[str]:
    """``word`` with its tonos on each other vowel letter that could carry
    it (a single vowel, or the second letter of a digraph)."""
    plain = "".join(_UNACCENT.get(c, c) for c in word)
    if plain == word:
        return []
    out = []
    for k, c in enumerate(plain):
        if c not in _ACCENT:
            continue
        if k + 1 < len(plain) and plain[k:k + 2] in L.DIGRAPHS:
            continue        # the tonos goes on the second letter of αι, ου, ...
        variant = plain[:k] + _ACCENT[c] + plain[k + 1:]
        if variant != word:
            out.append(variant)
    return out


def false_accepts(dic: dict[str, str]) -> dict:
    spot = {form: why for form, why in SPOT_CHECKS.items() if form in dic}
    moved = moved_in = dropped = dropped_in = 0
    moved_examples = []
    for form, bucket in sorted(dic.items()):
        if bucket not in ("C", "M") or L.case_shape(form) != "lower":
            continue
        if L.APOSTROPHE in form or L.syllable_count(form) < 2:
            continue
        plain = "".join(_UNACCENT.get(c, c) for c in form)
        dropped += 1
        if plain in dic:
            dropped_in += 1
        for variant in accent_variants(form):
            moved += 1
            if variant in dic:
                moved_in += 1
                if len(moved_examples) < SAMPLES and dic[variant] == "X":
                    moved_examples.append([form, variant, dic[variant]])
    return {
        "spot_checks": len(SPOT_CHECKS),
        "spot_checks_accepted": spot,
        "common_entries_checked": dropped,
        "tonos_dropped_accepted": dropped_in,
        "tonos_moved_variants": moved,
        "tonos_moved_accepted": moved_in,
        "tonos_moved_accepted_share": round(moved_in / moved, 4) if moved else None,
        "tonos_moved_accepted_unattested_examples": moved_examples,
    }


# -- comparisons ---------------------------------------------------------------------

def compare_elspell(dic: dict[str, str], elspell: set[str], tables: list,
                    lists: Lists) -> dict:
    els = Lists(elspell)
    ours_only = Counter(b for f, b in dic.items() if f not in elspell
                        and f.lower() not in elspell)
    examples = {b: [] for b in "CMRX"}
    for form, bucket in sorted(dic.items(), key=lambda kv: (kv[1], kv[0])):
        if form not in elspell and form.lower() not in elspell:
            if len(examples[bucket]) < SAMPLES and L.APOSTROPHE not in form:
                examples[bucket].append(form)
    we_reject = Counter()
    they_reject = Counter()
    for table in tables:
        for token, n in table.count.items():
            if token in el_text.PLACEHOLDERS or token.endswith(".") or not is_word(token):
                continue
            ours, theirs = lists.accepts(token), els.accepts(token)
            if theirs and not ours:
                we_reject[token] += n
            elif ours and not theirs:
                they_reject[token] += n
    return {
        "license": ELSPELL_LICENSE,
        "commit": ELSPELL_COMMIT,
        "words": len(elspell),
        "dev_coverage": coverage(tables, els, missing_limit=0)["overall"],
        "dev_tokens_elspell_accepts_list_rejects": sum(we_reject.values()),
        "dev_types_elspell_accepts_list_rejects": len(we_reject),
        "dev_top_elspell_accepts_list_rejects": [[t, n] for t, n in
                                                  we_reject.most_common(SAMPLES)],
        "dev_tokens_list_accepts_elspell_rejects": sum(they_reject.values()),
        "dev_types_list_accepts_elspell_rejects": len(they_reject),
        "dev_top_list_accepts_elspell_rejects": [[t, n] for t, n in
                                                  they_reject.most_common(SAMPLES)],
        "entries_elspell_lacks_by_bucket": dict(sorted(ours_only.items())),
        "entries_elspell_lacks_examples": examples,
    }


def compare_headwords(dic: dict[str, str], path: Path) -> dict | None:
    if not Path(path).exists():
        return None
    heads = json.loads(Path(path).read_text(encoding="utf-8"))
    words = sorted({unicodedata.normalize("NFC", h) for h in heads
                    if L.orthography_reason(unicodedata.normalize("NFC", h)) is None})
    bare = {L.bare_key(f) for f in dic}
    exact = [w for w in words if w in dic or (L.case_shape(w) == "title"
                                              and L.lower_el(w) in dic)]
    lacking = [w for w in words if w not in set(exact)]
    return {"license": "CC BY-NC-ND 4.0 (evaluation only)",
            "headwords": len(heads), "single_word_monotonic": len(words),
            "exact_share": round(len(exact) / len(words), 4) if words else None,
            "accent_insensitive_share": round(
                sum(1 for w in words if L.bare_key(w) in bare) / len(words), 4)
            if words else None,
            "lacking_examples": lacking[:SAMPLES]}


# -- main ----------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dic", type=Path, default=DEFAULT_DIC)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--type-counts", type=Path, default=TC.DEFAULT_OUT_DIR)
    ap.add_argument("--elspell-dir", type=Path, default=ELSPELL_DIR)
    ap.add_argument("--download", action="store_true",
                    help="download the pinned elspell files if they are missing")
    ap.add_argument("--no-elspell", action="store_true")
    ap.add_argument("--headwords", type=Path, default=TRIANTAFYLLIDIS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    dic = read_dic(args.dic)
    stem = args.dic.with_suffix("")
    lists = Lists(dic, read_lines(f"{stem}_acronyms.txt"),
                  read_lines(f"{stem}_abbreviations.txt"))
    paths = TC.ensure_tables(args.corpus, args.type_counts, split="dev",
                             workers=args.workers)
    tables = [TC.read_table(paths[s]) for s in sorted(paths)]
    result = {"dic": str(args.dic.relative_to(ROOT)) if args.dic.is_relative_to(ROOT)
              else str(args.dic),
              "dic_sha256": sha256(args.dic), "entries": len(dic),
              "buckets": dict(sorted(Counter(dic.values()).items())),
              "split": "dev",
              "coverage": coverage(tables, lists),
              "false_accepts": false_accepts(dic)}
    if not args.no_elspell:
        directory = fetch_elspell(args.elspell_dir, args.download)
        if directory is None:
            print(f"elspell is not in {args.elspell_dir}; pass --download to fetch it, "
                  "or --no-elspell", file=sys.stderr)
            return 1
        result["elspell"] = compare_elspell(dic, read_elspell(directory), tables, lists)
    result["triantafyllidis"] = compare_headwords(dic, args.headwords)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    cov = result["coverage"]["overall"]
    print(f"{result['entries']:,} entries; dev word tokens {cov['word_tokens']:,}: "
          f"exact {cov['exact_share']:.2%}, accent-insensitive "
          f"{cov['accent_insensitive_share']:.2%}, lowercase exact "
          f"{cov['lower_exact_share']:.2%}; abbreviations {cov['abbreviation_share']}")
    for source, s in result["coverage"]["per_source"].items():
        print(f"  {source:22} {s['word_tokens']:>10,}  exact {s['exact_share']:.2%}  "
              f"accent-insensitive {s['accent_insensitive_share']:.2%}")
    fa = result["false_accepts"]
    print(f"spot checks accepted: {len(fa['spot_checks_accepted'])} of {fa['spot_checks']} "
          f"{sorted(fa['spot_checks_accepted'])}; tonos moved on C/M entries: "
          f"{fa['tonos_moved_accepted']:,} of {fa['tonos_moved_variants']:,} accepted; "
          f"tonos dropped: {fa['tonos_dropped_accepted']}")
    if "elspell" in result:
        e = result["elspell"]
        print(f"elspell ({e['words']:,} words): dev exact "
              f"{e['dev_coverage']['exact_share']:.2%}; dev tokens it accepts and the "
              f"list rejects {e['dev_tokens_elspell_accepts_list_rejects']:,}; the list "
              f"accepts and it rejects {e['dev_tokens_list_accepts_elspell_rejects']:,}; "
              f"entries it lacks {e['entries_elspell_lacks_by_bucket']}")
    if result["triantafyllidis"]:
        t = result["triantafyllidis"]
        print(f"Triantafyllidis headwords: exact {t['exact_share']:.2%}, "
              f"accent-insensitive {t['accent_insensitive_share']:.2%} "
              f"of {t['single_word_monotonic']:,}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build a lemma-keyed corpus attestation profile from GLAUx + Diorisis.

For every Ancient Greek lemma this records its token frequency stratified by
century, dialect, and genre, derived from the two corpora that carry clean
per-text date / dialect / genre metadata. The form-keyed sibling builders
(build_glaux_freq.py, build_diorisis_freq.py) aggregate by accent-stripped
surface FORM and collapse everything to genre; this builder is the missing
lemma-keyed, dimensionally-stratified view.

Only GLAUx and Diorisis are read here. The other corpus_freq sources
(PG, First1KGreek, PTA, canonical-greekLit) lack reliable date/dialect/genre
metadata and would dilute the dimensions, so they are deliberately excluded.

Output: data/lemma_attestation.json  (see SCHEMA_VERSION / _meta for the
contract). The file is a pure, deterministic function of its inputs:
sorted keys, ensure_ascii=False, no wall-clock in the body, and SHA-256
content hashes of every input under _meta.source_sha.

Usage:
    python build/build_lemma_attestation.py                  # full build
    python build/build_lemma_attestation.py --stats          # report, no write
    python build/build_lemma_attestation.py --limit 30       # smoke test
    python build/build_lemma_attestation.py --glaux ~/Documents/glaux/xml \\
        --metadata ~/Documents/glaux/metadata.txt \\
        --diorisis data/diorisis/xml --output data/lemma_attestation.json
"""

import argparse
import csv
import hashlib
import json
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_GLAUX_DIR = Path.home() / "Documents" / "glaux" / "xml"
DEFAULT_METADATA = Path.home() / "Documents" / "glaux" / "metadata.txt"
DEFAULT_DIORISIS_DIR = DATA_DIR / "diorisis" / "xml"
OUTPUT_PATH = DATA_DIR / "lemma_attestation.json"

SCHEMA_VERSION = 1

# --- genre: collapse each corpus's labels into the same 10 bins used by
# build_glaux_freq.py / build_diorisis_freq.py, for cross-source consistency.
GENRE_ORDER = [
    "philosophy", "poetry", "history", "oratory", "science",
    "narrative", "epistles", "religion", "commentary", "other",
]

SOURCE_ORDER = ["glaux", "diorisis", "oga"]

GLAUX_GENRE_MAP = {
    "Philosophy": "philosophy",
    "Philosophic Dialogue": "philosophy",
    "Epic poetry": "poetry",
    "Lyric poetry": "poetry",
    "Tragedy": "poetry",
    "Comedy": "poetry",
    "Scientific Poetry": "poetry",
    "Religious Poetry": "poetry",
    "History": "history",
    "Religious History": "history",
    "Biography": "history",
    "Oratory": "oratory",
    "Rhetoric": "oratory",
    "Medicine": "science",
    "Mathematics": "science",
    "Engineering": "science",
    "Physics": "science",
    "Biology": "science",
    "Astronomy/Astrology": "science",
    "Alchemy": "science",
    "Geography": "science",
    "Narrative": "narrative",
    "Religious Narrative": "narrative",
    "Mythography": "narrative",
    "Paradoxography": "narrative",
    "Epistolography": "epistles",
    "Religious Epistle": "epistles",
    "Theology": "religion",
    "Religious Wisdom": "religion",
    "Religious Prophecy": "religion",
    "Oracle": "religion",
    "Commentary": "commentary",
    "Language": "commentary",
    "Dialogue": "philosophy",
    "Polyhistory": "other",
    "Art": "other",
    "Military": "other",
    "Music": "other",
    "Oneirocritic": "other",
}

DIORISIS_GENRE_MAP = {
    "Philosophy": "philosophy",
    "Poetry": "poetry",
    "Tragedy": "poetry",
    "Comedy": "poetry",
    "Oratory": "oratory",
    "Narrative": "history",
    "Technical": "science",
    "Essays": "philosophy",
    "Letters": "epistles",
    "Religion": "religion",
}

# --- part of speech: unify GLAUx's positional postag and Diorisis's word
# labels into one coarse vocabulary so dominant_pos is comparable across
# sources. GLAUx tags proper nouns as plain nouns; Diorisis's "proper" is
# folded to noun to match.
GLAUX_POS_MAP = {
    "n": "noun", "v": "verb", "a": "adjective", "d": "adverb",
    "l": "article", "g": "particle", "c": "conjunction",
    "r": "preposition", "p": "pronoun", "m": "numeral",
    "i": "interjection", "e": "interjection",
    "b": "particle",  # GLAUx code reserved for the enclitic τε
    # 'x' (irregular) and 'z' (non-inflecting / foreign junk) -> other
}

DIORISIS_POS_MAP = {
    "noun": "noun", "verb": "verb", "adjective": "adjective",
    "adverb": "adverb", "article": "article", "particle": "particle",
    "conjunction": "conjunction", "preposition": "preposition",
    "pronoun": "pronoun", "numeral": "numeral",
    "interjection": "interjection", "exclamation": "interjection",
    "proper": "noun",
}

# Deterministic tie-break order when two POS have equal token counts.
POS_ORDER = [
    "noun", "verb", "adjective", "adverb", "pronoun", "article",
    "preposition", "conjunction", "particle", "numeral",
    "interjection", "other",
]
POS_INDEX = {p: i for i, p in enumerate(POS_ORDER)}


def strip_accents(s):
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if unicodedata.category(c) != "Mn"))


def is_lexical_greek(s):
    """True iff s is a Greek lexical string: only Greek-block letters and
    combining marks, starting with a base letter, and not an all-caps label.

    Rejects the residue the corpora occasionally annotate as a lemma:
    numerals in keraia notation (αʹ, ιβʹ), math/metrical/musical notation
    (∠, ⏑, 𝈶), Latin transliterations, punctuation, stray spacing breathing
    marks, and Beta Code control characters. Callers NFC-normalize first, so
    the keraia (U+0374 -> U+02B9) and lower-numeral sign fall outside the
    Greek letter range and are dropped. A leading combining mark (a Beta Code
    accent that never attached to a vowel) is rejected. All-uppercase strings
    of >=2 letters are geometry diagram labels (the segment ΑΒ, the triangle
    ΑΒΓ), not lexemes, and are dropped; proper nouns (Άρβηλον) keep their
    lower-case tail and survive.
    """
    n_letters = n_upper = 0
    for i, c in enumerate(s):
        cat = unicodedata.category(c)
        if cat[0] == "M":
            if i == 0:
                return False  # combining mark with no base = residue
            continue
        if ('Ͱ' <= c <= 'Ͽ' or 'ἀ' <= c <= '῿') and cat[0] == "L":
            n_letters += 1
            if cat == "Lu":
                n_upper += 1
            continue
        return False
    if n_letters == 0:
        return False
    if n_letters >= 2 and n_upper == n_letters:
        return False
    return True


def year_to_century(y):
    """Map a signed year to a signed century integer (no century 0).

    -8 = 8th c. BC, 1 = 1st c. AD, 2 = 2nd c. AD.
    Year -800..-701 -> -8; 1..100 -> 1; 101..200 -> 2.
    """
    if y > 0:
        return (y - 1) // 100 + 1
    if y < 0:
        return -(((-y) - 1) // 100 + 1)
    return 1  # no year 0 in the scheme; does not occur in the data


def dominant_pos(pos_counter):
    """Most frequent unified POS, tie-broken by POS_ORDER (lower index wins)."""
    if not pos_counter:
        return "other"
    return max(pos_counter.items(),
               key=lambda kv: (kv[1], -POS_INDEX.get(kv[0], len(POS_ORDER))))[0]


class LemmaProfile:
    """Mutable per-lemma accumulator.

    Two views are kept. The DEDUPED frequency stream (total + by_genre /
    by_century / by_dialect) is fed only by the preferred source for each work
    (GLAUx wins shared works), so each text is counted once. source_counts holds
    every source's INDEPENDENT token count for the lemma -- overlapping, never
    summed -- which preserves both lemmatizers' evidence (agreement = confidence,
    a source-only reading = recall). POS is pooled across sources for
    dominant_pos so even a total=0 (single-source) lemma gets a real POS.
    """
    __slots__ = ("total", "by_genre", "by_century", "by_dialect",
                 "source_counts", "pos")

    def __init__(self):
        self.total = 0
        self.by_genre = Counter()
        self.by_century = Counter()
        self.by_dialect = Counter()
        self.source_counts = Counter()
        self.pos = Counter()

    def observe(self, source, pos, n=1):
        """Record one source's independent lemmatization of n tokens."""
        self.source_counts[source] += n
        self.pos[pos] += n

    def add_deduped(self, genre, century, dialect, n=1):
        """Add n preferred-source tokens to the deduped frequency stream."""
        self.total += n
        self.by_genre[genre] += n
        if century is not None:
            self.by_century[str(century)] += n
        if dialect:
            self.by_dialect[dialect] += n


def load_glaux_metadata(path, agg_hash):
    """stem -> (century | None, genre_bin, dialect | '').

    Century is the floored midpoint of (STARTDATE, ENDDATE). agg_hash is
    updated with the file bytes for the source content hash.
    """
    raw = path.read_bytes()
    agg_hash["glaux_metadata"] = hashlib.sha256(raw).hexdigest()
    meta = {}
    text = raw.decode("utf-8")
    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    for row in reader:
        stem = (row.get("TLG") or "").strip()
        if not stem:
            continue
        genre = GLAUX_GENRE_MAP.get((row.get("GENRE_STANDARD") or "").strip(),
                                    "other")
        dialect = (row.get("DIALECT") or "").strip()
        century = None
        try:
            start = int((row.get("STARTDATE") or "").strip())
            end = int((row.get("ENDDATE") or "").strip())
            century = year_to_century((start + end) // 2)
        except (ValueError, TypeError):
            pass
        meta[stem] = (century, genre, dialect)
    return meta


def fold_file_hash(agg, stem, data):
    """Fold one file's content into the order-independent-per-corpus aggregate.

    Files are visited in sorted order, so updating with stem + per-file digest
    yields a deterministic content hash of the whole corpus.
    """
    agg.update(stem.encode("utf-8"))
    agg.update(b"\0")
    agg.update(hashlib.sha256(data).digest())


def diorisis_work_id(root, filename):
    """TLG 'AUTHOR-WORK' id (e.g. '0527-001') for a Diorisis text, formatted to
    match GLAUx file stems, or None if it can't be determined.

    Used to defer to GLAUx on shared works (the two corpora annotate largely the
    same texts). Reads the header tlgAuthor/tlgId, stripping any letter suffix
    on the work number (Diorisis splits a few Plutarch Lives as 051a/051b that
    GLAUx keeps under one numeric id); falls back to the '(NNNN) - ... (NNN)'
    filename convention.
    """
    a = root.find(".//tlgAuthor")
    t = root.find(".//tlgId")
    author = (a.text or "").strip() if a is not None else ""
    work = re.sub(r"\D", "", (t.text or "").strip() if t is not None else "")
    if author.isdigit() and work:
        return f"{int(author):04d}-{int(work):03d}"
    m = re.search(r"\((\d{3,4})\)[^()]*\((\d{1,3})[a-z]?\)\.xml$", filename)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):03d}"
    return None


def process_glaux(glaux_dir, meta, profiles, limit, stats, nc_stems=frozenset(),
                  gorman_stems=frozenset()):
    glaux_hash = hashlib.sha256()
    work_ids = set()  # GLAUx file stems = TLG work ids, for the Diorisis dedup
    files = sorted(glaux_dir.glob("*.xml"))
    if nc_stems:
        before = len(files)
        files = [f for f in files if f.stem not in nc_stems]
        print(f"  Excluded {before - len(files)} "
              f"GLAUx text(s) (NonCommercial or PROIEL-derived)")
    if limit:
        files = files[:limit]
    print(f"GLAUx: {len(files)} files")
    for i, xf in enumerate(files):
        stem = xf.stem
        data = xf.read_bytes()
        fold_file_hash(glaux_hash, stem, data)
        century, genre, dialect = meta.get(stem, (None, "other", ""))
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            stats["parse_errors"] += 1
            continue
        work_ids.add(stem)
        # Gorman-derived works: the manual sentences ARE Gorman's trees
        # (held-out gold, never ingested); the auto sentences pass.
        skip_manual = stem in gorman_stems
        for sent in root.iter("sentence"):
            if skip_manual and sent.get("analysis") == "manual":
                continue
            for w in sent.findall(".//word"):
                postag = w.get("postag", "")
                if postag and postag[0] == "u":
                    continue  # punctuation
                lemma = w.get("lemma", "")
                if not lemma:
                    stats["glaux_unlemmatized"] += 1
                    continue
                lemma = unicodedata.normalize("NFC", lemma)
                if not is_lexical_greek(lemma):
                    stats["glaux_nonlexical_lemma"] += 1
                    continue
                pos = GLAUX_POS_MAP.get(postag[0] if postag else "", "other")
                p = profiles[lemma]
                p.observe("glaux", pos)
                p.add_deduped(genre, century, dialect)  # GLAUx preferred everywhere
                stats["glaux_tokens"] += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)} files, "
                  f"{stats['glaux_tokens']:,} tokens, "
                  f"{len(profiles):,} lemmas", flush=True)
    return glaux_hash.hexdigest(), work_ids


def process_diorisis(diorisis_dir, profiles, limit, stats, glaux_work_ids):
    dio_hash = hashlib.sha256()
    files = sorted(diorisis_dir.glob("*.xml"))
    kept_wids = set()
    if limit:
        files = files[:limit]
    print(f"Diorisis: {len(files)} files")
    for i, xf in enumerate(files):
        data = xf.read_bytes()
        fold_file_hash(dio_hash, xf.name, data)
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            stats["parse_errors"] += 1
            continue
        # Dedup: if GLAUx already annotates this TLG work, Diorisis's tokens for
        # it are kept only as independent lemmatization evidence (source_counts),
        # NOT added to the deduped frequency stream -- GLAUx is preferred there.
        wid = diorisis_work_id(root, xf.name)
        deferred = wid is not None and wid in glaux_work_ids
        if not deferred and wid is not None:
            kept_wids.add(wid)
        if deferred:
            stats["diorisis_deferred_works"] += 1
            genre = "other"
            century = None
        else:
            stats["diorisis_kept_works"] += 1
            genre_el = root.find(".//genre")
            raw_genre = (genre_el.text or "").strip() if genre_el is not None else ""
            genre = DIORISIS_GENRE_MAP.get(raw_genre, "other")
            # Date lives at <creation><date>; the header also carries edition and
            # processing dates, so this exact path matters.
            date_el = root.find(".//creation/date")
            century = None
            if date_el is not None and date_el.text:
                try:
                    century = year_to_century(int(date_el.text.strip()))
                except ValueError:
                    stats["diorisis_bad_date"] += 1
        for w in root.iter("word"):
            lem = w.find("lemma")
            if lem is None:
                continue
            entry = lem.get("entry", "")
            if not entry:
                stats["diorisis_unlemmatized"] += 1
                continue
            entry = unicodedata.normalize("NFC", entry)
            if not is_lexical_greek(entry):
                stats["diorisis_nonlexical_lemma"] += 1
                continue
            pos = DIORISIS_POS_MAP.get((lem.get("POS") or "").lower(), "other")
            p = profiles[entry]
            p.observe("diorisis", pos)
            if deferred:
                stats["diorisis_evidence_tokens"] += 1
            else:
                p.add_deduped(genre, century, None)  # Diorisis-only work
                stats["diorisis_tokens"] += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)} files, "
                  f"{stats['diorisis_tokens']:,} tokens, "
                  f"{len(profiles):,} lemmas", flush=True)
    return dio_hash.hexdigest(), kept_wids


def process_oga(profiles, stats, claimed_wids, glaux_meta, limit=0):
    """Third attestation source: cog's standardized OGA export (all-auto
    Trankit/GreTa annotation - acceptable-but-dispreferred evidence).

    Work-level dedup priority glaux > diorisis > oga: an OGA work already
    claimed by either contributes only source_counts (independent evidence),
    not the deduped frequency stream. Gorman-annotated works are skipped
    entirely (OGA's models trained on Gorman; see build/cog_annotations.py).
    Century/genre: inherited from GLAUx metadata for TLG works it knows,
    else century from cog's OGA dating artifact with genre "other".
    Homograph digits (λέγω3) are stripped from the attestation key.
    """
    import cog_annotations as C
    manifest = C.load_manifest()
    if manifest is None:
        print("OGA: cog export not found, skipping "
              f"({C.DEFAULT_EXPORT}); set DILEMMA_COG_OGA")
        return None, ""
    gorman = C.gorman_work_ids()
    dating = C.load_oga_dating()
    oga_hash = hashlib.sha256()
    works = manifest["works"]
    if limit:
        works = works[:limit]
    print(f"OGA ({manifest['export']['release_id']}): {len(works)} works")
    for i, w in enumerate(works):
        stem = C.work_tlg_stem(w["work_id"])
        if stem and stem in gorman:
            stats["oga_gorman_skipped_works"] += 1
            continue
        fold_file_hash(oga_hash, w["work_id"], w["sha256"].encode())
        deferred = stem is not None and stem in claimed_wids
        if deferred:
            stats["oga_deferred_works"] += 1
            genre = "other"
            century = None
        else:
            stats["oga_kept_works"] += 1
            if stem and stem in glaux_meta:
                century, genre, _dialect = glaux_meta[stem]
            else:
                genre = "other"
                dot_key = ".".join(w["work_id"].split(".")[:2])
                century = dating.get(dot_key)
        for rec in C.iter_work_tokens(C.DEFAULT_EXPORT, w):
            if (rec.get("pos") or "") == "u":
                continue
            lemma = rec.get("lemma") or ""
            if not lemma:
                stats["oga_unlemmatized"] += 1
                continue
            lemma = C.strip_homograph_digits(
                unicodedata.normalize("NFC", lemma))
            if not is_lexical_greek(lemma):
                stats["oga_nonlexical_lemma"] += 1
                continue
            pos = GLAUX_POS_MAP.get(rec.get("pos") or "", "other")
            p = profiles[lemma]
            p.observe("oga", pos)
            if deferred:
                stats["oga_evidence_tokens"] += 1
            else:
                p.add_deduped(genre, century, None)
                stats["oga_tokens"] += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(works)} works, "
                  f"{stats['oga_tokens']:,} tokens, "
                  f"{len(profiles):,} lemmas", flush=True)
    return oga_hash.hexdigest(), C.pin_line(manifest)


def build_output(profiles, observed_dialects, total_tokens, source_sha):
    genre_idx = {g: i for i, g in enumerate(GENRE_ORDER)}
    source_idx = {s: i for i, s in enumerate(SOURCE_ORDER)}

    def ordered(counter, key):
        """Counter -> dict with keys in a canonical (not lexical) order."""
        return {k: counter[k] for k in sorted(counter, key=key)}

    # Top-level lemmas sorted by key; inner dicts get a human-meaningful order
    # (century chronological, genre by GENRE_ORDER, source by SOURCE_ORDER,
    # dialect alphabetical). Output is dumped with sort_keys=False so these
    # orders survive -- sort_keys=True would re-sort them lexically, putting
    # by_century into string order (-1, -3, ... -8, 1) instead of -8 .. 6.
    lemmas = {}
    for lemma in sorted(profiles):
        p = profiles[lemma]
        entry = {
            "total": p.total,
            "source_counts": ordered(p.source_counts,
                                     lambda s: source_idx.get(s, 99)),
            "by_genre": ordered(p.by_genre, lambda g: genre_idx.get(g, 99)),
            "by_century": ordered(p.by_century, int),
        }
        if p.by_dialect:
            entry["by_dialect"] = ordered(p.by_dialect, lambda d: d)
        entry["dominant_pos"] = dominant_pos(p.pos)
        lemmas[lemma] = entry

    meta = {
        "schema_version": SCHEMA_VERSION,
        "sources": list(SOURCE_ORDER),
        "dedup": ("each TLG work counted once; GLAUx preferred, Diorisis "
                  "contributes only works GLAUx lacks (joined on author-work id)"),
        "genres": GENRE_ORDER,
        "dialects": sorted(observed_dialects),
        "century_scheme": ("signed century integer, -8 = 8th c. BC, "
                           "1 = 1st c. AD, 2 = 2nd c. AD; no century 0"),
        "pos_labels": POS_ORDER,
        "total_tokens": total_tokens,
        "n_lemmas": len(lemmas),
        "source_sha": source_sha,
        "notes": [
            "Keys are the corpus's own lemma annotation (GLAUx @lemma, "
            "Diorisis @entry), NFC-normalized, not accent-stripped. Neither "
            "corpus uses homograph digit suffixes, so the two merge directly. "
            "The consumer is expected to accent-fold for its own join. Keys are "
            "restricted to lexical Greek (Greek-block letters plus combining "
            "marks, starting with a base letter); the corpora's occasional "
            "numeral (keraia), symbol, Latin, Beta Code residue, and all-caps "
            "geometry-label lemmas are dropped (~0.2% of tokens).",
            "Corpus lemmas do not encode LSJ's (A)/(B) homograph distinctions, "
            "so there is one profile per lemma string. dominant_pos (the most "
            "frequent coarse POS) is provided to help pick a noun-vs-verb "
            "homograph, but cannot separate same-POS homographs.",
            "by_century for GLAUx uses the floored midpoint of the text's "
            "(STARTDATE, ENDDATE) range; Diorisis uses its single "
            "<creation><date>.",
            "by_dialect is GLAUx-only (Diorisis carries no dialect) and is "
            "present only for lemmas with GLAUx tokens of known dialect. "
            "Compound labels (e.g. 'Ionic/Epic', 'Attic/Koine') are kept "
            "verbatim; blank dialects are omitted.",
            "total_tokens counts only lemmatized Greek tokens; punctuation, "
            "non-Greek, and unlemmatized tokens are excluded.",
            "GLAUx and Diorisis independently annotate largely the same texts. "
            "To avoid double-counting, total and the by_* breakdowns are a "
            "DEDUPED frequency: each TLG work is counted once, using GLAUx's "
            "copy for any work it contains (it has dialect and richer metadata) "
            "and Diorisis only for works GLAUx lacks, joined on TLG author-work "
            "id. So total/by_* are a union of works, not a sum.",
            "source_counts is separate: each source's INDEPENDENT token count "
            "for the lemma (overlapping; do NOT sum, and it does not equal "
            "total). Two sources agreeing is a confidence signal. A lemma "
            "produced only by a non-preferred source's reading of a shared work "
            "has total 0 but a non-empty source_counts -- a real but "
            "single-source, lower-confidence attestation; filter on total > 0 "
            "for frequency-backed lemmas only.",
            "Ordering: lemma keys sorted by code point; within each lemma, "
            "by_century is chronological, by_genre follows the genres list, "
            "source_counts follows the sources list, by_dialect is alphabetical.",
        ],
    }
    return {"_meta": meta, "lemmas": lemmas}


def report(stats, profiles, total_tokens):
    print(f"\nLemmas: {len(profiles):,}")
    print(f"Deduped tokens (total; union of works, counted once): "
          f"{total_tokens:,}")
    print(f"  glaux:    {stats['glaux_tokens']:,}")
    print(f"  diorisis: {stats['diorisis_tokens']:,} "
          f"(from {stats['diorisis_kept_works']:,} GLAUx-absent works)")
    print(f"  oga:      {stats['oga_tokens']:,} "
          f"(from {stats['oga_kept_works']:,} previously uncovered works; "
          f"{stats['oga_gorman_skipped_works']:,} Gorman works skipped)")
    print(f"Independent evidence in source_counts (not summed into total):")
    print(f"  diorisis on {stats['diorisis_deferred_works']:,} shared works: "
          f"{stats['diorisis_evidence_tokens']:,} tokens")
    print(f"  oga on {stats['oga_deferred_works']:,} shared works: "
          f"{stats['oga_evidence_tokens']:,} tokens")
    print("Skipped / dropped:")
    print(f"  glaux unlemmatized:        {stats['glaux_unlemmatized']:,}")
    print(f"  glaux non-lexical lemma:   {stats['glaux_nonlexical_lemma']:,}")
    print(f"  diorisis unlemmatized:     {stats['diorisis_unlemmatized']:,}")
    print(f"  diorisis non-lexical lemma:{stats['diorisis_nonlexical_lemma']:,}")
    print(f"  diorisis bad date:        {stats['diorisis_bad_date']:,}")
    print(f"  oga unlemmatized:         {stats['oga_unlemmatized']:,}")
    print(f"  oga non-lexical lemma:    {stats['oga_nonlexical_lemma']:,}")
    print(f"  parse errors:             {stats['parse_errors']:,}")

    # Coverage: fraction of tokens assigned each dimension.
    cent = dial = genred = 0
    for p in profiles.values():
        cent += sum(p.by_century.values())
        dial += sum(p.by_dialect.values())
        genred += p.total - p.by_genre.get("other", 0)
    if total_tokens:
        print("\nCoverage (fraction of lemmatized tokens):")
        print(f"  with a century:        {100*cent/total_tokens:5.1f}%")
        print(f"  with a dialect:        {100*dial/total_tokens:5.1f}% "
              f"(GLAUx-only; {100*dial/max(1,stats['glaux_tokens']):.1f}% of GLAUx)")
        print(f"  with a non-other genre:{100*genred/total_tokens:5.1f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--glaux", type=Path, default=DEFAULT_GLAUX_DIR)
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--diorisis", type=Path, default=DEFAULT_DIORISIS_DIR)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--stats", action="store_true",
                   help="Report only, do not write the JSON.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N files of each corpus "
                        "(smoke test; gives partial counts).")
    args = p.parse_args()

    # Openly licensed by default: always drop the NonCommercial GLAUx texts
    # (this pass uses only GLAUx + Diorisis, both otherwise permissive).
    from nc_filter import excluded_glaux_stems, gorman_glaux_stems
    nc_stems = excluded_glaux_stems(args.metadata)
    gorman_stems = gorman_glaux_stems(args.metadata)

    t0 = time.time()
    stats = Counter()
    source_sha = {}
    profiles = defaultdict(LemmaProfile)

    print("Loading GLAUx metadata...", flush=True)
    glaux_meta = load_glaux_metadata(args.metadata, source_sha)
    observed_dialects = {d for (_, _, d) in glaux_meta.values() if d}
    print(f"  {len(glaux_meta)} texts, dialects: {sorted(observed_dialects)}")

    source_sha["glaux_xml"], glaux_work_ids = process_glaux(
        args.glaux, glaux_meta, profiles, args.limit, stats, nc_stems,
        gorman_stems)
    source_sha["diorisis_xml"], diorisis_wids = process_diorisis(
        args.diorisis, profiles, args.limit, stats, glaux_work_ids)
    oga_sha, oga_pin = process_oga(
        profiles, stats, glaux_work_ids | diorisis_wids, glaux_meta,
        args.limit)
    if oga_sha:
        source_sha["oga_export"] = oga_sha
        source_sha["oga_pin"] = oga_pin

    total_tokens = (stats["glaux_tokens"] + stats["diorisis_tokens"]
                    + stats["oga_tokens"])
    report(stats, profiles, total_tokens)

    if args.stats:
        print(f"\n(stats only, {time.time()-t0:.1f}s)")
        return 0

    print(f"\nBuilding output for {len(profiles):,} lemmas...", flush=True)
    output = build_output(profiles, observed_dialects, total_tokens, source_sha)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        # NOT sort_keys=True: build_output already orders every dict (lemmas by
        # key, inner dicts by their natural comparator). Lexical re-sorting here
        # would scramble by_century back into string order.
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(args.output)
    size_mb = args.output.stat().st_size / 1e6
    print(f"Wrote {args.output} ({size_mb:.1f} MB, {time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

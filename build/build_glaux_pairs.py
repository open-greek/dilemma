#!/usr/bin/env python3
"""Extract form->lemma pairs with morphological tags from GLAUx corpus.

GLAUx (Keersmaekers, 2021) is a 17M-token automatically annotated corpus
of Ancient Greek (8th century BCE–4th century CE). Each token has AGDT-style
9-position morphological tags.

Extracts training pairs for Dilemma's multi-task learning heads:
- POS tag (position 1)
- Nominal group: Gender + Number + Case (positions 7, 3, 8)
- Verbal group: Tense + Mood + Voice (positions 4, 5, 6)

Output: data/glaux_pairs.json in Dilemma's training pair format. Each
(form, lemma) pair is written once, with the analysis of its first token as
``pos`` / ``tags``, its token count as ``count``, and, when its tokens carry
more than one analysis, every analysis with its own count as ``analyses``
(``[pos, tags, count]``, commonest first). GLAUx is tagged token by token, so
the counts say which reading of an ambiguous spelling the texts mean (φέρε is
an imperative 956 times and an unaugmented imperfect 45).

The command-line build also recovers malformed foreign-style name lemmas
only when the same source group supplies a unique singular nominative that
is independently defined as a name. Merged lexical evidence and source name
pairs are required. Original/corrected rows and source hashes are written to
changes/glaux_name_lemmas.json beside the output for exact reversal.

Usage:
    python build_glaux_pairs.py                    # extract all
    python build_glaux_pairs.py --stats            # show stats only
    python build_glaux_pairs.py --glaux ~/path     # custom GLAUx path
"""

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_GLAUX = Path.home() / "Documents" / "glaux" / "xml"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dilemma.form_sanitize import has_editorial_sigla  # noqa: E402

# What GLAUx puts in the lemma field of an editorial gap, where the manuscript
# is damaged or unreadable. The surviving letters stay in the form, so the
# token still reads as Greek.
GAP_LEMMA = "G"

# Lemma values that are annotation placeholders rather than lemmas: the gap
# marker above, the CoNLL-U empty field, an ellipsis, and the empty string.
PLACEHOLDER_LEMMAS = frozenset({GAP_LEMMA, "_", "...", ""})

# AGDT position 1 -> Wiktionary POS
AGDT_POS = {
    "n": "noun", "v": "verb", "a": "adj", "p": "pron",
    "d": "adv", "c": "conj", "r": "prep", "l": "article",
    "g": "intj", "m": "num",
}

# AGDT single-char codes -> Dilemma tag labels
AGDT_NUMBER = {"s": "singular", "p": "plural", "d": "dual"}
AGDT_GENDER = {"m": "masculine", "f": "feminine", "n": "neuter"}
AGDT_CASE = {"n": "nominative", "g": "genitive", "d": "dative",
             "a": "accusative", "v": "vocative"}
AGDT_TENSE = {"p": "present", "i": "imperfect", "a": "aorist",
              "f": "future", "r": "perfect", "l": "pluperfect",
              "t": "future-perfect"}
AGDT_MOOD = {"i": "indicative", "s": "subjunctive", "o": "optative",
             "m": "imperative", "n": "infinitive", "p": "participle"}
AGDT_VOICE = {"a": "active", "m": "middle", "p": "passive"}
AGDT_PERSON = {"1": "first-person", "2": "second-person", "3": "third-person"}


def nfc(s):
    return unicodedata.normalize("NFC", s)


def is_greek(s):
    return any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF' for c in s)


def parse_postag(postag):
    """Parse 9-position AGDT morphological tag into Dilemma tag list.

    Returns (pos, tags) where pos is a Wiktionary POS string and
    tags is a list of morphological feature labels.
    """
    if not postag or len(postag) < 9:
        return None, []

    pos_code = postag[0]
    pos = AGDT_POS.get(pos_code)
    if not pos:
        return None, []

    tags = []

    # Person (position 2)
    person = AGDT_PERSON.get(postag[1])
    if person:
        tags.append(person)

    # Number (position 3)
    number = AGDT_NUMBER.get(postag[2])
    if number:
        tags.append(number)

    # Tense (position 4)
    tense = AGDT_TENSE.get(postag[3])
    if tense:
        tags.append(tense)

    # Mood (position 5)
    mood = AGDT_MOOD.get(postag[4])
    if mood:
        tags.append(mood)

    # Voice (position 6)
    voice = AGDT_VOICE.get(postag[5])
    if voice:
        tags.append(voice)

    # Gender (position 7)
    gender = AGDT_GENDER.get(postag[6])
    if gender:
        tags.append(gender)

    # Case (position 8)
    case = AGDT_CASE.get(postag[7])
    if case:
        tags.append(case)

    return pos, tags


# GLAUx's DIALECT column against the dialect tag the paradigm builders read.
# Attic, Attic/Koine and Koine stay in the default (Attic) slice: Koine is the
# continuation of Attic and textbook paradigms include its forms. The marked
# dialects are the ones whose forms must not fill an Attic paradigm slot
# (Doric δωσῶ, Ionic τιθεῖς, Epic λῦσε).
GLAUX_DIALECT_TAGS = {
    "Ionic/Epic": "Epic",
    "Ionic": "Ionic",
    "Doric": "Doric",
    "Aeolic": "Aeolic",
    "Attic": "Attic",
    "Attic/Koine": "Attic",
    "Koine": "Attic",
}


def glaux_dialects(metadata_path) -> dict:
    """Map GLAUx file stem to a dialect tag, '' for the Attic default."""
    path = Path(metadata_path)
    if not path.exists():
        return {}
    out = {}
    text = path.read_text(encoding="utf-8")
    for row in csv.DictReader(text.splitlines(), delimiter="\t"):
        stem = (row.get("TLG") or "").strip()
        if stem:
            out[stem] = GLAUX_DIALECT_TAGS.get(
                (row.get("DIALECT") or "").strip(), ""
            )
    return out


def _name_key(text):
    return "".join(c for c in unicodedata.normalize("NFD", text.casefold())
                   if unicodedata.category(c) != "Mn")


def _malformed_name_key(text, known_heads):
    """Limit recovery to foreign-style corruption, not Greek dialect forms."""
    if not text or _name_key(text) in known_heads:
        return False
    letters = [c for c in text if c.isalpha()]
    foreign = any("GREEK" not in unicodedata.name(c, "") for c in letters)
    unmarked_lower = (letters and text == text.lower()
                      and not any(unicodedata.category(c) == "Mn"
                                  for c in unicodedata.normalize("NFD", text)))
    return bool(foreign or unmarked_lower)


def _capital_greek(text):
    return (bool(text) and text[0].isupper()
            and not any(c.isupper() for c in text[1:])
            and all(unicodedata.combining(c) or "GREEK" in unicodedata.name(c, "")
                    for c in unicodedata.normalize("NFD", text)))


def _pure_noun(row):
    return row.get("pos") == "noun" and all(
        analysis[0] == "noun" for analysis in row.get("analyses", []))


def _fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def load_name_evidence(lexical_path, name_pairs_path):
    """Require merged lexical support AND an exact independent name reading."""
    lexical_path, name_pairs_path = Path(lexical_path), Path(name_pairs_path)
    payload = json.loads(lexical_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or not isinstance(payload.get("heads"), dict):
        raise ValueError("invalid merged lexical-head evidence")
    heads = {head: sources for head, sources in payload["heads"].items() if sources}
    names = {}
    for row in json.loads(name_pairs_path.read_text(encoding="utf-8")):
        head = row.get("lemma", "")
        if (row.get("pos") == "name" and row.get("form") == head
                and _capital_greek(head) and head in heads):
            names[head] = sorted(heads[head])
    return {"known_heads": {_name_key(head) for head in heads}, "names": names,
            "inputs": {"grc_lexical_heads": _fingerprint(lexical_path),
                       "ag_pairs": _fingerprint(name_pairs_path)}}


def repair_name_lemmas(pairs, evidence, *, source_commit, sources_by_key=None,
                       source_paths=None):
    """Recover uniquely anchored malformed names without discarding readings.

    The independent nominative must occur in the same raw lemma group. Only
    capitalized noun rows move; mixed analyses, competing anchors, and rows
    that would collide with an existing reading remain unchanged.
    """
    if re.fullmatch(r"[0-9a-f]{40}", source_commit or "") is None:
        raise ValueError("name-correction audit requires a full source commit hash")
    candidates = defaultdict(set)
    anchors = defaultdict(list)
    for row in pairs:
        form, lemma = row["form"], row["lemma"]
        if (not _pure_noun(row) or form not in evidence["names"]
                or not _malformed_name_key(lemma, evidence["known_heads"])):
            continue
        analyses = [row.get("tags", [])] + [a[1] for a in row.get("analyses", [])]
        if any({"nominative", "singular"} <= set(tags) for tags in analyses):
            candidates[lemma].add(form)
            anchors[lemma].append({"original": row, "source_xml": sorted(
                (sources_by_key or {}).get((form, lemma), ()))})
    owners = {lemma: next(iter(heads)) for lemma, heads in candidates.items()
              if len(heads) == 1}
    proposals = {}
    for index, row in enumerate(pairs):
        target = owners.get(row["lemma"])
        if target and _pure_noun(row) and _capital_greek(row["form"]):
            proposals[index] = target
    existing = {(row["form"], row["lemma"]) for row in pairs}
    proposed_keys = Counter((pairs[i]["form"], target) for i, target in proposals.items())
    result, corrections = [], []
    for index, row in enumerate(pairs):
        target = proposals.get(index)
        if (target is None or (row["form"], target) in existing
                or proposed_keys[row["form"], target] != 1):
            result.append(row)
            continue
        corrected = {**row, "lemma": target}
        result.append(corrected)
        sources = sorted((sources_by_key or {}).get((row["form"], row["lemma"]), ()))
        corrections.append({"original": row, "corrected": corrected,
                            "evidence": {"nominative": target,
                                         "name_pos_source": "ag_pairs",
                                         "lexical_sources": evidence["names"][target],
                                         "anchor_rows": anchors[row["lemma"]]},
                            "source_xml": sources})
    # Every retained XML file establishes the absence of a competing anchor,
    # including files with no corrected rows. Pin the complete input inventory.
    source_paths = source_paths or {}
    audit = {"schema_version": 2, "source_commit": source_commit,
             "builder_sha256": _fingerprint(Path(__file__))["sha256"],
             "inputs": evidence["inputs"],
             "source_xml": {name: _fingerprint(path) for name, path in sorted(source_paths.items())},
             "corpus_xml_inventory": sorted(source_paths),
             "inferred_groups": {old: sorted(heads) for old, heads in sorted(candidates.items())},
             "corrections": corrections}
    return result, audit


def unapply_name_corrections(pairs, audit):
    """Restore the exact original pair rows, rejecting an unrelated artifact."""
    pending = {json.dumps(r["corrected"], ensure_ascii=False, sort_keys=True): r["original"]
               for r in audit["corrections"]}
    result = []
    for row in pairs:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        result.append(pending.pop(key, row))
    if pending:
        raise ValueError("corrected rows do not match the name-correction audit")
    return result


def _reading_tags(tags):
    """Work-level dialect labels are not part of the token's AGDT analysis."""
    return frozenset(tags) - frozenset(GLAUX_DIALECT_TAGS.values())


def load_name_corrections(audit_path, glaux_dir, *, lexical_path=None,
                          name_pairs_path=None, metadata_path=None):
    """Validate a name audit and index its exact original source readings.

    Raw-corpus consumers must use this same audit rather than independently
    infer a replacement. Missing or stale evidence fails before extraction.
    """
    audit_path, glaux_dir = Path(audit_path), Path(glaux_dir)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit.get("schema_version") != 2
            or re.fullmatch(r"[0-9a-f]{40}", audit.get("source_commit", "")) is None
            or audit.get("builder_sha256") != _fingerprint(Path(__file__))["sha256"]):
        raise ValueError("invalid or stale GLAUx name-correction audit; rebuild the pairs and audit")
    paths = {
        "grc_lexical_heads": Path(lexical_path or DATA_DIR / "grc_lexical_heads.json"),
        "ag_pairs": Path(name_pairs_path or DATA_DIR / "ag_pairs.json"),
    }
    if "glaux_metadata" in audit.get("inputs", {}):
        paths["glaux_metadata"] = Path(metadata_path or glaux_dir.parent / "metadata.txt")
    for source, path in paths.items():
        if audit.get("inputs", {}).get(source) != _fingerprint(path):
            raise ValueError(f"stale GLAUx name-correction evidence: {source}")
    verified = load_name_evidence(paths["grc_lexical_heads"], paths["ag_pairs"])
    try:
        from .nc_filter import excluded_glaux_stems, gorman_glaux_stems
    except ImportError:
        from nc_filter import excluded_glaux_stems, gorman_glaux_stems
    metadata = Path(metadata_path or glaux_dir.parent / "metadata.txt")
    excluded = excluded_glaux_stems(metadata)
    gorman = gorman_glaux_stems(metadata)
    inventory = sorted(path.name for path in glaux_dir.glob("*.xml")
                       if path.stem not in excluded)
    if (audit.get("corpus_xml_inventory") != inventory
            or set(audit.get("source_xml", {})) != set(inventory)):
        raise ValueError("stale GLAUx name-correction XML inventory")
    for source, expected in audit.get("source_xml", {}).items():
        if Path(source).name != source or not source.endswith(".xml"):
            raise ValueError("invalid source XML name in name-correction audit")
        if _fingerprint(glaux_dir / source) != expected:
            raise ValueError(f"stale GLAUx name-correction source: {source}")
    by_source = defaultdict(dict)
    anchor_cache = {}
    for change in audit.get("corrections", []):
        original, corrected = change["original"], change["corrected"]
        target = corrected["lemma"]
        evidence = change.get("evidence", {})
        if (corrected != {**original, "lemma": target}
                or not _pure_noun(original) or not _capital_greek(original["form"])
                or not change.get("source_xml")
                or target not in verified["names"]
                or evidence.get("nominative") != target
                or evidence.get("name_pos_source") != "ag_pairs"
                or evidence.get("lexical_sources") != verified["names"][target]
                or audit.get("inferred_groups", {}).get(original["lemma"]) != [target]
                or not _malformed_name_key(original["lemma"], verified["known_heads"])):
            raise ValueError("invalid original reading in GLAUx name-correction audit")
        anchors = evidence.get("anchor_rows", [])
        if not anchors:
            raise ValueError("missing nominative anchor in GLAUx name-correction audit")
        for anchor in anchors:
            row = anchor["original"]
            if (row["form"] != target or row["lemma"] != original["lemma"]
                    or not _pure_noun(row) or not anchor.get("source_xml")):
                raise ValueError("invalid nominative anchor in GLAUx name-correction audit")
            for source in anchor["source_xml"]:
                if source not in audit["source_xml"]:
                    raise ValueError("unverified nominative anchor source")
                if source not in anchor_cache:
                    root = ET.parse(glaux_dir / source).getroot()
                    observed = set()
                    for sentence in root.iter("sentence"):
                        if Path(source).stem in gorman and sentence.get("analysis") == "manual":
                            continue
                        for word in sentence.findall(".//word"):
                            pos, tags = parse_postag(word.get("postag", ""))
                            if pos == "noun" and {"nominative", "singular"} <= set(tags):
                                observed.add((nfc(word.get("form", "")), nfc(word.get("lemma", ""))))
                    anchor_cache[source] = observed
                if (target, original["lemma"]) not in anchor_cache[source]:
                    raise ValueError("nominative anchor does not match its GLAUx source")
        readings = [(original.get("pos"), _reading_tags(original.get("tags", [])))]
        readings.extend((pos, _reading_tags(tags))
                        for pos, tags, _count in original.get("analyses", []))
        for source in change["source_xml"]:
            if source not in audit.get("source_xml", {}):
                raise ValueError("unverified source XML in name-correction audit")
            for pos, tags in readings:
                key = (nfc(original["form"]), nfc(original["lemma"]), pos, tags)
                previous = by_source[source].setdefault(key, target)
                if previous != target:
                    raise ValueError("conflicting targets in GLAUx name-correction audit")
    return {"by_source": dict(by_source), "sha256": _fingerprint(audit_path)["sha256"],
            "source_commit": audit["source_commit"]}


def apply_name_correction(corrections, source_file, form, lemma, postag):
    """Apply only an audited original form/lemma/POS/analysis in its source."""
    entries = corrections["by_source"].get(Path(source_file).name)
    if not entries:
        return lemma
    pos, tags = parse_postag(postag)
    return entries.get((nfc(form), nfc(lemma), pos, _reading_tags(tags)), lemma)


def extract_glaux(glaux_dir, stats_only=False, metadata_path=None, *,
                  name_evidence=None, correction_audit=None, source_commit=None):
    """Extract form->lemma pairs from GLAUx XML files.

    NonCommercial GLAUx source texts are always dropped (the build is openly
    licensed by default); see build/nc_filter.py.
    """
    xml_files = sorted(Path(glaux_dir).glob("*.xml"))
    if not xml_files:
        print(f"No XML files found in {glaux_dir}")
        return []

    from nc_filter import excluded_glaux_stems, gorman_glaux_stems
    meta = metadata_path or (Path(glaux_dir).parent / "metadata.txt")
    nc = excluded_glaux_stems(meta)
    gorman = gorman_glaux_stems(meta)
    dialects = glaux_dialects(meta)
    before = len(xml_files)
    xml_files = [x for x in xml_files if x.stem not in nc]
    print(f"Excluded {before - len(xml_files)} "
          f"GLAUx text(s) (NonCommercial or PROIEL-derived)")
    print(f"Skipping manual (Gorman-derived) sentences in "
          f"{sum(1 for x in xml_files if x.stem in gorman)} text(s)")

    print(f"Processing {len(xml_files)} GLAUx files...")

    pairs = []
    seen = set()  # (form, lemma) dedup
    entry_by_key = {}
    analysis_counts = defaultdict(Counter)  # (form, lemma) -> (pos, tags) -> tokens
    dialect_votes = defaultdict(set)
    sources_by_key = defaultdict(set)
    total_tokens = 0
    skipped_punct = 0
    skipped_no_lemma = 0
    skipped_gap = 0
    skipped_editorial = 0
    skipped_non_greek = 0
    skipped_dup = 0

    pos_counts = Counter()
    has_nominal = 0
    has_verbal = 0
    tense_counts = Counter()

    for i, xml_file in enumerate(xml_files):
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            continue

        # Gorman-derived works: the manual sentences ARE Gorman's trees
        # (held-out gold, never ingested); only the auto sentences pass.
        skip_manual = xml_file.stem in gorman
        file_dialect = dialects.get(xml_file.stem, "")
        for sentence in tree.findall(".//sentence"):
            if skip_manual and sentence.get("analysis") == "manual":
                continue
            words = sentence.findall(".//word")
            if not words:
                continue
            for word in words:
                form = word.get("form", "")
                lemma = word.get("lemma", "")
                postag = word.get("postag", "")

                total_tokens += 1

                # Skip punctuation
                if postag and postag[0] == "u":
                    skipped_punct += 1
                    continue

                # Skip the gaps the treebank marks itself. Where a
                # manuscript is damaged or unreadable the editor records a
                # lacuna, and the annotation says so three ways at once:
                # relation="GAP", postag "z" (unanalyzable), and the lemma
                # field set to the gap marker. The surviving letters are
                # still carried in `form`, so "Gdou" below is what is left
                # of an occurrence of the genitive of Hades after its first
                # two letters were lost.
                #
                # The `is_greek` test further down does not catch these,
                # because it asks whether ANY character is Greek and the
                # surviving letters are. Reading the annotation is both
                # exact and cheaper than guessing from the characters.
                # The lemma field is checked as well as the relation and the
                # postag, because an annotator sometimes assigns a part of
                # speech to a token whose text is mostly gap: `τGῆς` is tagged
                # a feminine genitive numeral and `G?ἔπλει` a verb. Both still
                # carry the gap marker as their lemma, and all 113 such rows
                # in GLAUx do, so the lemma is the reliable signal. Real Greek
                # proper nouns transliterated from Latin keep Latin homoglyphs
                # in their own lemmas (`Σιλουίαν` under `σιλvια`, `Οὐολουμνίαν`
                # under `Vολυμνια`), so a blanket Latin-letter test would throw
                # those away; this one does not touch them.
                if (postag[:1] == "z" or word.get("relation") == "GAP"
                        or lemma in PLACEHOLDER_LEMMAS):
                    skipped_gap += 1
                    continue

                # "_" is the CoNLL-U marker for a field left empty. It is a
                # plain string, so a bare truthiness test lets it through:
                # GLAUx carries one such row, for διατείνας, and Diorisis has
                # the same form under its real lemma διατείνω.
                if not form or lemma in PLACEHOLDER_LEMMAS:
                    skipped_no_lemma += 1
                    continue

                # NFC normalize (GLAUx uses NFD)
                form = nfc(form)
                lemma = nfc(lemma)

                # Leiden brackets and parentheses describe an editor's
                # reconstruction; they are not part of the Greek spelling.
                # Reject both fields here so every consumer of the shared
                # GLAUx pair artifact inherits the same hygiene policy.
                if has_editorial_sigla(form) or has_editorial_sigla(lemma):
                    skipped_editorial += 1
                    continue

                if not is_greek(form):
                    skipped_non_greek += 1
                    continue

                # Parse morphological tag
                pos, tags = parse_postag(postag)

                if (name_evidence is not None and pos == "noun" and _capital_greek(form)
                        and _malformed_name_key(lemma, name_evidence["known_heads"])):
                    sources_by_key[form, lemma].add(xml_file.name)

                # Dedup. A (form, lemma) pair is written once, but every
                # work that attests it votes on its dialect below, and every
                # token counts toward its analysis.
                key = (form, lemma)
                dialect_votes[key].add(file_dialect)
                analysis_counts[key][(pos, tuple(tags))] += 1
                if key in seen:
                    skipped_dup += 1
                    continue
                seen.add(key)

                entry = {"form": form, "lemma": lemma}
                if pos:
                    entry["pos"] = pos
                    pos_counts[pos] += 1
                if tags:
                    entry["tags"] = tags

                    # Count morphological coverage
                    tag_set = set(tags)
                    if tag_set & {"masculine", "feminine", "neuter"}:
                        has_nominal += 1
                    if tag_set & {"present", "imperfect", "aorist", "future",
                                  "perfect", "pluperfect"}:
                        has_verbal += 1
                        for t in tag_set & {"present", "imperfect", "aorist",
                                            "future", "perfect", "pluperfect"}:
                            tense_counts[t] += 1

                pairs.append(entry)
                entry_by_key[key] = entry

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(xml_files)} files, {len(pairs):,} pairs", flush=True)

    # Token counts. Ties keep first-seen order (a Counter keeps insertion
    # order and the sort is stable), so the output is deterministic.
    ambiguous = 0
    for key, entry in entry_by_key.items():
        counts = analysis_counts[key]
        entry["count"] = sum(counts.values())
        if len(counts) > 1:
            ambiguous += 1
            entry["analyses"] = [
                [pos, list(tags), n]
                for (pos, tags), n in sorted(counts.items(), key=lambda kv: -kv[1])
            ]
    print(f"\nPairs whose tokens carry more than one analysis: {ambiguous:,}")

    # A form keeps the Attic default if any work attesting it is explicitly
    # Attic, Attic/Koine or Koine, or if no work attesting it records a
    # dialect at all. A form attested only in dialect works carries that
    # dialect, so the paradigm builders route it to that dialect's slice.
    dialect_counts = Counter()
    for key, votes in dialect_votes.items():
        entry = entry_by_key.get(key)
        if entry is None or "Attic" in votes or votes == {""}:
            continue
        marked = [d for d in ("Epic", "Ionic", "Doric", "Aeolic")
                  if d in votes]
        if not marked:
            continue
        entry.setdefault("tags", []).append(marked[0])
        dialect_counts[marked[0]] += 1
    if dialect_counts:
        print("\nDialect-tagged pairs (attested only in that dialect): "
              + ", ".join(f"{d} {n:,}" for d, n in dialect_counts.most_common()))

    print(f"\nTotal tokens: {total_tokens:,}")
    print(f"Skipped: {skipped_punct:,} punct, {skipped_gap:,} editorial gaps, "
          f"{skipped_editorial:,} editorial sigla, "
          f"{skipped_no_lemma:,} no lemma, "
          f"{skipped_non_greek:,} non-Greek, {skipped_dup:,} duplicates")
    print(f"Unique pairs: {len(pairs):,}")

    print(f"\nPOS distribution:")
    for pos, count in pos_counts.most_common():
        print(f"  {pos:10s}: {count:,}")

    print(f"\nMorphological coverage:")
    print(f"  With nominal features (G+N+C): {has_nominal:,}")
    print(f"  With verbal features (T+M+V):  {has_verbal:,}")

    print(f"\nTense distribution (verbs):")
    total_tense = sum(tense_counts.values())
    for t, c in tense_counts.most_common():
        print(f"  {t:15s}: {c:,} ({100*c/total_tense:.1f}%)")

    if name_evidence is not None:
        pairs, audit = repair_name_lemmas(
            pairs, name_evidence, source_commit=source_commit,
            sources_by_key=sources_by_key,
            source_paths={path.name: path for path in xml_files})
        if Path(meta).is_file():
            audit["inputs"]["glaux_metadata"] = _fingerprint(meta)
        if correction_audit is not None:
            correction_audit.update(audit)
        print(f"Recovered malformed name rows: {len(audit['corrections']):,}")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Extract GLAUx training pairs")
    parser.add_argument("--glaux", type=str, default=str(DEFAULT_GLAUX),
                        help="Path to GLAUx xml/ directory")
    parser.add_argument("--stats", action="store_true",
                        help="Show stats only, don't save")
    parser.add_argument("--output", type=str,
                        default=str(DATA_DIR / "glaux_pairs.json"),
                        help="Output path (default: glaux_pairs.json)")
    parser.add_argument("--lexical-heads", type=Path,
                        default=DATA_DIR / "grc_lexical_heads.json",
                        help="Required merged lexical evidence for name recovery")
    parser.add_argument("--name-pairs", type=Path, default=DATA_DIR / "ag_pairs.json",
                        help="Required independent source name readings")
    parser.add_argument("--changes-output", type=Path,
                        help="Reversible audit (default: OUTPUT_DIR/changes/glaux_name_lemmas.json)")
    parser.add_argument("--source-commit", help="Full source commit for the audit (default: current HEAD)")
    args = parser.parse_args()

    evidence = load_name_evidence(args.lexical_heads, args.name_pairs)
    commit = args.source_commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR, text=True).strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        parser.error("--source-commit must be a full Git commit hash")
    audit = {}

    # Always drop the NonCommercial GLAUx source texts: glaux_pairs.json is a
    # openly licensed artifact (see build/nc_filter.py and NOTICE).
    pairs = extract_glaux(args.glaux, stats_only=args.stats,
                          name_evidence=evidence, correction_audit=audit,
                          source_commit=commit)

    if not args.stats and pairs:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False)
        changes = args.changes_output or out.parent / "changes" / "glaux_name_lemmas.json"
        changes.parent.mkdir(parents=True, exist_ok=True)
        changes.write_text(json.dumps(audit, ensure_ascii=False, sort_keys=True,
                                      indent=1) + "\n", encoding="utf-8")
        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"\nSaved {len(pairs):,} pairs to {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

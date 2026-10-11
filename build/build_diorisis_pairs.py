#!/usr/bin/env python3
"""Extract form->lemma pairs from Diorisis Ancient Greek corpus.

Diorisis (Vatri & McGillivray, 2018) is a 10M-token lemmatized corpus
of Ancient Greek (Homer through the 5th century CE). Lemma accuracy is ~91.4%
(vs GLAUx's 98.8%), so pairs from this corpus should be treated as
lower-confidence and only used when they don't conflict with existing
entries from Wiktionary, LSJ, or GLAUx.

Each token has:
- form: Beta Code surface form
- lemma entry: Unicode Greek lemma (dictionary headword)
- POS: noun, verb, adjective, pronoun, etc.
- morph: morphological analysis string(s)

Output: data/diorisis_pairs.json - list of {form, lemma, pos} dicts,
matching the format of data/glaux_pairs.json.

Usage:
    python build/build_diorisis_pairs.py
    python build/build_diorisis_pairs.py --diorisis ~/path/to/xml
    python build/build_diorisis_pairs.py --stats  # show stats only
"""

import argparse
import hashlib
import json
import subprocess
import sys
import unicodedata
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from beta_code import beta_code_to_greek

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from dilemma.form_sanitize import canonicalize_final_elision  # noqa: E402
from build.corpus_freq_key import beta_trailing_paren_is_elision  # noqa: E402
from build.diorisis_source import DiorisisSourceContext  # noqa: E402

DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_DIORISIS_DIR = DATA_DIR / "diorisis" / "xml"
OUTPUT_PATH = DATA_DIR / "diorisis_pairs.json"


def source_revision():
    """Retain code hashes when an exported source archive has no Git metadata."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# Map Diorisis POS labels to Wiktionary-style POS (matching GLAUx pairs format)
DIORISIS_POS_MAP = {
    "noun": "noun",
    "verb": "verb",
    "adjective": "adj",
    "adverb": "adv",
    "pronoun": "pron",
    "article": "article",
    "particle": "particle",
    "preposition": "prep",
    "conjunction": "conj",
    "interjection": "intj",
    "proper": "noun",       # proper nouns -> noun
    "numeral": "num",
}


def nfc(s):
    return unicodedata.normalize("NFC", s)


def is_greek(s):
    return any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF' for c in s)


def is_pure_greek(s):
    """Check that every character is Greek letter or combining mark."""
    for c in s:
        cat = unicodedata.category(c)
        if cat.startswith("L"):
            if not ('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF'):
                return False
        elif cat.startswith("M"):
            pass  # combining marks ok
        else:
            return False
    return len(s) > 0


def strip_non_greek(s):
    """Remove non-Greek, non-combining-mark characters (quotes, apostrophes)."""
    return "".join(c for c in s
                   if unicodedata.category(c).startswith("M")
                   or ('\u0370' <= c <= '\u03FF')
                   or ('\u1F00' <= c <= '\u1FFF'))


def betacode_to_unicode(bc):
    """Convert Beta Code form to NFC-normalized Unicode Greek.

    Preserve a final apostrophe or Diorisis's elision ``)`` as Greek koronis.
    Remove that marker before conversion so it cannot become a final-vowel
    breathing, and keep medial σ before elision. Other non-Greek residue is
    stripped; a real final breathing on a short vowel word remains unchanged.
    """
    try:
        elided = bc.endswith("'") or beta_trailing_paren_is_elision(bc)
        if elided:
            bc = bc[:-1]
        uni = beta_code_to_greek(bc)
        uni = nfc(uni)
        cleaned = strip_non_greek(uni)
        if cleaned and elided:
            if cleaned.endswith("ς"):
                cleaned = cleaned[:-1] + "σ"
            cleaned = canonicalize_final_elision(cleaned + "’")
        return cleaned if cleaned else ""
    except Exception:
        return ""


def parse_morph(morph_str, pos):
    """Parse Diorisis morph string into Dilemma-style tag list.

    Diorisis morph strings look like:
        'masc acc pl'
        'pres subj act 1st sg (attic epic doric)'
        'indeclform (particle)'
        'fem gen sg (attic doric aeolic)'

    We extract the main features, ignoring dialect markers in parentheses.
    """
    if not morph_str:
        return []

    # Strip dialect info in parentheses
    clean = morph_str
    if "(" in clean:
        clean = clean[:clean.index("(")].strip()

    parts = clean.lower().split()
    tags = []

    # Gender
    for p in parts:
        if p in ("masc", "masc/neut"):
            tags.append("masculine")
            break
        elif p == "fem":
            tags.append("feminine")
            break
        elif p == "neut":
            tags.append("neuter")
            break

    # Number
    for p in parts:
        if p == "sg":
            tags.append("singular")
            break
        elif p == "pl":
            tags.append("plural")
            break
        elif p == "dual":
            tags.append("dual")
            break

    # Case
    case_map = {
        "nom": "nominative", "gen": "genitive", "dat": "dative",
        "acc": "accusative", "voc": "vocative",
        "nom/voc/acc": "nominative",  # take first
        "nom/voc": "nominative",
        "nom/acc": "nominative",
        "gen/dat": "genitive",
    }
    for p in parts:
        if p in case_map:
            tags.append(case_map[p])
            break

    # Verbal features: tense
    tense_map = {
        "pres": "present", "imperf": "imperfect", "aor": "aorist",
        "fut": "future", "perf": "perfect", "plup": "pluperfect",
    }
    for p in parts:
        if p in tense_map:
            tags.append(tense_map[p])
            break

    # Mood
    mood_map = {
        "ind": "indicative", "subj": "subjunctive", "opt": "optative",
        "imperat": "imperative", "inf": "infinitive", "part": "participle",
    }
    for p in parts:
        if p in mood_map:
            tags.append(mood_map[p])
            break

    # Voice
    voice_map = {
        "act": "active", "mid": "middle", "pass": "passive",
        "mp": "middle",  # medio-passive
    }
    for p in parts:
        if p in voice_map:
            tags.append(voice_map[p])
            break

    # Person
    person_map = {
        "1st": "first-person", "2nd": "second-person", "3rd": "third-person",
    }
    for p in parts:
        if p in person_map:
            tags.append(person_map[p])
            break

    return tags


def extract_diorisis(diorisis_dir, stats_only=False, *, token_audit=None):
    """Extract pairs, retaining distinct readings of quote-normalized cells."""
    xml_files = sorted(Path(diorisis_dir).glob("*.xml"))
    if not xml_files:
        print(f"No XML files found in {diorisis_dir}")
        print("Download from: https://figshare.com/articles/dataset/"
              "The_Diorisis_Ancient_Greek_Corpus/6187256")
        return []

    print(f"Processing {len(xml_files)} Diorisis files...")

    t0 = time.time()
    pairs = []
    readings = _PairReadings(pairs)
    total_tokens = 0
    skipped_no_lemma = 0
    skipped_non_greek = 0
    skipped_convert = 0
    skipped_dup = 0

    for i, xml_file in enumerate(xml_files):
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            continue

        source_context = DiorisisSourceContext(tree.getroot())
        if token_audit is not None and source_context.audit:
            token_audit.append({"source": xml_file.name,
                                "source_sha256": hashlib.sha256(xml_file.read_bytes()).hexdigest(),
                                "records": source_context.audit})
        normalized_keys = set()
        for record in source_context.audit:
            if record["normalized_beta"] == record["original_beta"]:
                continue
            form = betacode_to_unicode(record["normalized_beta"])
            for reading in record["readings"]:
                normalized_keys.add((form, nfc(reading["lemma"])))
        readings.touch(normalized_keys)
        for word in tree.findall(".//word"):
            total_tokens += 1

            bc_form = source_context.beta(word)
            if not bc_form:
                skipped_no_lemma += 1
                continue

            # Convert Beta Code form to Unicode
            form = betacode_to_unicode(bc_form)
            if not form:
                skipped_convert += 1
                continue

            if not is_greek(form):
                skipped_non_greek += 1
                continue
            variants, primary = _word_pair_readings(word, form)
            added = readings.observe(variants, primary)
            if primary is None:
                skipped_no_lemma += 1
            elif not added:
                skipped_dup += 1

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(xml_files)} files, "
                  f"{len(pairs):,} pairs", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal tokens: {total_tokens:,} ({elapsed:.1f}s)")
    print(f"Skipped: {skipped_no_lemma:,} no lemma, "
          f"{skipped_convert:,} conversion failures, "
          f"{skipped_non_greek:,} non-Greek, "
          f"{skipped_dup:,} duplicates")
    print(f"Distinct exported readings: {len(pairs):,}")

    pos_counts = Counter(pair.get("pos") for pair in pairs if pair.get("pos"))
    has_nominal = sum(bool(set(pair.get("tags", ())) &
                           {"masculine", "feminine", "neuter"}) for pair in pairs)
    has_verbal = sum(bool(set(pair.get("tags", ())) &
                          {"present", "imperfect", "aorist", "future",
                           "perfect", "pluperfect"}) for pair in pairs)
    print(f"\nPOS distribution:")
    for pos, count in pos_counts.most_common():
        print(f"  {pos:10s}: {count:,}")

    print(f"\nMorphological coverage:")
    print(f"  With nominal features (G+N+C): {has_nominal:,}")
    print(f"  With verbal features (T+M+V):  {has_verbal:,}")

    return pairs


def _word_pair_readings(word, form):
    """Keep every source analysis; identify the legacy first reading separately."""
    variants = []
    primary = None
    for lemma_index, lemma_el in enumerate(word.findall("lemma")):
        lemma = nfc(lemma_el.get("entry", ""))
        if not lemma or not is_greek(lemma):
            continue
        pos = DIORISIS_POS_MAP.get(lemma_el.get("POS", ""))
        analyses = lemma_el.findall("analysis")
        for analysis_index, analysis in enumerate(analyses or [None]):
            tags = parse_morph(analysis.get("morph", ""), pos) if analysis is not None else []
            pair = {"form": form, "lemma": lemma}
            if pos:
                pair["pos"] = pos
            if tags:
                pair["tags"] = tags
            variants.append(pair)
            if lemma_index == 0 and analysis_index == 0:
                primary = pair
    return variants, primary


class _PairReadings:
    """Preserve all analyses only for keys touched by an audited normalization.

    Earlier alternatives are buffered so discovering a normalized cell in a
    later document cannot erase an earlier ordinary reading. Untouched keys
    keep their existing first-lemma/first-analysis selection and ordering.
    """

    def __init__(self, pairs):
        self.pairs = pairs
        self.variants = {}
        self.emitted = {}
        self.normalized = set()

    @staticmethod
    def signature(pair):
        return pair.get("pos"), tuple(pair.get("tags", ()))

    def emit(self, key, signature, pair):
        emitted = self.emitted.setdefault(key, set())
        if signature in emitted:
            return 0
        emitted.add(signature)
        self.pairs.append(pair)
        return 1

    def touch(self, keys):
        # Dictionary insertion order is source order; never order source
        # readings by the arbitrary iteration order of the incoming key set.
        newly_normalized = keys - self.normalized
        self.normalized.update(keys)
        if not newly_normalized:
            return
        for key, variants in self.variants.items():
            if key in newly_normalized:
                for signature, pair in variants.items():
                    self.emit(key, signature, pair)

    def observe(self, variants, primary):
        added = 0
        for pair in variants:
            key = pair["form"], pair["lemma"]
            signature = self.signature(pair)
            self.variants.setdefault(key, {}).setdefault(signature, pair)
            if key in self.normalized:
                added += self.emit(key, signature, pair)
        if primary is not None:
            key = primary["form"], primary["lemma"]
            if key not in self.normalized and key not in self.emitted:
                added += self.emit(key, self.signature(primary), primary)
        return added


def main():
    parser = argparse.ArgumentParser(
        description="Extract Diorisis form-lemma pairs")
    parser.add_argument("--diorisis", type=str,
                        default=str(DEFAULT_DIORISIS_DIR),
                        help="Path to Diorisis xml/ directory")
    parser.add_argument("--stats", action="store_true",
                        help="Show stats only, don't save")
    parser.add_argument("--output", type=str,
                        default=str(OUTPUT_PATH),
                        help="Output path")
    parser.add_argument("--token-audit", type=Path,
                        help="Reversible source-token audit (default: changes beside the output)")
    args = parser.parse_args()

    token_audit = []
    pairs = extract_diorisis(args.diorisis, stats_only=args.stats, token_audit=token_audit)

    if not args.stats and pairs:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False)
        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"\nSaved {len(pairs):,} pairs to {out} ({size_mb:.1f} MB)")
        audit_path = args.token_audit or out.parent / "changes" / "diorisis_token_normalizations.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        source_files = ["build/build_diorisis_pairs.py", "build/diorisis_source.py"]
        payload = {"schema_version": 1,
                   "source_commit": source_revision(),
                   "source_sha256": {name: hashlib.sha256((SCRIPT_DIR / name).read_bytes()).hexdigest()
                                     for name in source_files},
                   "sources": token_audit}
        audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(f"Saved source-token audit to {audit_path}")


if __name__ == "__main__":
    main()

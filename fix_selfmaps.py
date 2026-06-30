#!/usr/bin/env python3
"""Fix self-map entries in lookup.db using form_of evidence from kaikki dumps.

Scans AG kaikki dumps for form_of/alt_of references (including senses-level),
then patches self-map entries in lookup.db where form_of evidence says the form
should map to a different lemma.

This is a targeted fix for cases like ἐρίσαντε (which should map to ἐρίζω but
self-maps because its Wiktionary page wasn't detected as a form-of page during
the original build).

Usage:
    python fix_selfmaps.py [--kaikki DIR] [--dry-run]
"""

import argparse
import json
import sqlite3
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DB_PATH = DATA_DIR / "lookup.db"
CORPUS_FREQ_PATH = DATA_DIR / "corpus_freq.json"

# A monotonic form whose accent-stripped surface is at least this frequent is
# treated as a Modern-Greek function word (articles, particles) and never
# rewritten to a polytonic AG lemma. Ordinary AG/Byzantine content words sit
# well below this (e.g. ανάγκη ~20.6K), so their polytonic restoration is kept.
MG_FUNCTION_FREQ = 100_000

# Manual corrections for ambiguous forms where Wiktionary lists multiple
# form_of targets and first-write-wins picks the wrong one. These are
# cases where both mappings are technically valid but one is clearly
# more common/expected.
# Format: {form: correct_lemma}
AMBIGUOUS_CORRECTIONS = {
    # δοῖεν is 3pl aor opt of δίδωμι ("to give"), not pres opt of δέω
    # ("to bind"). Both verbs list it, but δίδωμι is the standard reading.
    "δοῖεν": "δίδωμι",
}

# Import helpers from build_data
from build_data import (
    strip_length_marks, to_monotonic, strip_accents, _is_greek,
    DUMPS, resolve_dump, DEFAULT_DUMP_DIR
)


def _is_polytonic(s: str) -> bool:
    """True if s carries a breathing mark or circumflex (Ancient-Greek
    polytonic), i.e. it is not a monotonic Modern-Greek-style form."""
    nfd = unicodedata.normalize("NFD", s)
    return any(ord(c) in (0x0313, 0x0314, 0x0342) for c in nfd)


def collect_form_of_targets(dump_dir: Path) -> dict:
    """Scan AG kaikki dumps for strict form_of relationships.

    Only collects entries that have explicit senses[].form_of data with
    "form-of" in the sense tags. This excludes alt_of (variant spellings)
    and gloss-only form-of patterns, which are more likely to be legitimate
    independent headwords that should self-map.

    Returns {form_variant: target_lemma} for strict form_of references.
    """
    targets = {}

    for wikt_lang, filename in DUMPS["grc"].items():
        path = resolve_dump(filename, dump_dir)
        if not path.exists():
            print(f"  Skipping {filename} (not found)")
            continue

        print(f"  Scanning {path.name}...")
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                word = entry.get("word", "")
                pos = entry.get("pos", "")
                if not word or pos == "name":
                    continue

                lemma = strip_length_marks(word)
                if not _is_greek(lemma.replace(" ", "")) or " " in lemma:
                    continue

                # Only use senses[].form_of with "form-of" tag.
                # This is the most reliable signal that a page is
                # a morphological form, not a variant headword.
                # Skip alt_of (variant spellings like Βορέας/Βορέης)
                # and top-level form_of (less reliable).
                for sense in entry.get("senses", []):
                    sense_form_of = sense.get("form_of", [])
                    sense_tags = sense.get("tags", [])
                    if not sense_form_of or "form-of" not in sense_tags:
                        continue

                    for ref in sense_form_of:
                        ref_word = strip_length_marks(ref.get("word", ""))
                        if not ref_word or not _is_greek(ref_word.replace(" ", "")):
                            continue
                        if " " in ref_word or lemma == ref_word:
                            continue

                        for key in (lemma, lemma.lower(),
                                    to_monotonic(lemma), to_monotonic(lemma).lower(),
                                    strip_accents(lemma.lower())):
                            if key and key not in targets:
                                targets[key] = ref_word
                        count += 1

        print(f"    Found {count:,} strict form_of references")

    return targets


def resolve_chain(form, targets, lemma_set, max_depth=5):
    """Follow form_of chain to find the ultimate lemma.

    Uses both form_of targets and the existing lemma set.
    """
    seen = {form}
    current = targets.get(form)
    if not current:
        return None

    depth = 0
    while depth < max_depth:
        if current in lemma_set:
            return current
        next_target = targets.get(current)
        if not next_target or next_target in seen:
            break
        seen.add(next_target)
        current = next_target
        depth += 1

    # Try accent-stripped/lowercase versions
    if current not in lemma_set:
        for variant in (current.lower(), to_monotonic(current),
                        to_monotonic(current).lower(),
                        strip_accents(current.lower())):
            if variant in lemma_set:
                return variant

    return current if current != form else None


def main():
    parser = argparse.ArgumentParser(description="Fix self-map entries in lookup.db")
    parser.add_argument("--kaikki", type=str, default=None,
                        help="Path to kaikki dump directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be fixed without modifying DB")
    args = parser.parse_args()

    dump_dir = Path(args.kaikki) if args.kaikki else DEFAULT_DUMP_DIR

    print("Collecting form_of relationships from kaikki dumps...")
    targets = collect_form_of_targets(dump_dir)
    print(f"Total form_of targets: {len(targets):,}")

    print(f"\nReading lookup.db...")
    conn = sqlite3.connect(str(DB_PATH))

    # Build set of known lemmas (forms that are valid lemma targets)
    lemma_set = set()
    lemma_by_stripped = {}
    for row in conn.execute("SELECT text FROM lemmas"):
        lemma_set.add(row[0])
        lemma_by_stripped[strip_accents(row[0].lower())] = row[0]

    # Find self-map entries
    self_maps = conn.execute("""
        SELECT l.rowid, l.form, le.text, l.lang
        FROM lookup l JOIN lemmas le ON l.lemma_id = le.id
        WHERE l.form = le.text AND l.src = 'grc'
    """).fetchall()
    print(f"AG self-map entries: {len(self_maps):,}")

    # Build set of MG headwords to protect from AG overrides.
    # MG forms that self-map are legitimate Modern Greek lemmas and should
    # not be resolved to AG forms (e.g. ποιότητα should not become ποιότης).
    mg_self_maps = set()
    for row in conn.execute("""
        SELECT l.form FROM lookup l JOIN lemmas le ON l.lemma_id = le.id
        WHERE l.form = le.text AND l.src = 'el'
    """):
        mg_self_maps.add(row[0])
    # Also add stripped variants for matching
    mg_stripped = set()
    for f in mg_self_maps:
        mg_stripped.add(strip_accents(f.lower()))
    print(f"MG self-map forms to protect: {len(mg_self_maps):,}")

    # Build set of productive lemmas: forms that morphologically distinct
    # entries map TO. If a form is the target of other lookups where the
    # source form is a genuinely different word (not just a case/accent
    # variant), it's a real lemma and should not be resolved.
    # E.g., ἔργον has forms like ἔργου, ἔργῳ mapping to it.
    # But ἐρίσαντε having ερίσαντε (monotonic) map to it doesn't count.
    productive_lemmas = set()
    for row in conn.execute("""
        SELECT l.form, le.text FROM lookup l
        JOIN lemmas le ON l.lemma_id = le.id
        WHERE l.form != le.text AND l.src = 'grc'
    """):
        form_stripped = strip_accents(row[0].lower())
        lemma_stripped = strip_accents(row[1].lower())
        if form_stripped != lemma_stripped:
            productive_lemmas.add(row[1])
    print(f"Productive AG lemmas: {len(productive_lemmas):,}")

    # Surface frequency, for telling MG function words from AG content words.
    corpus_freq = {}
    if CORPUS_FREQ_PATH.exists():
        corpus_freq = json.load(open(CORPUS_FREQ_PATH, encoding="utf-8")).get(
            "forms", {})

    def freq(form):
        v = corpus_freq.get(strip_accents(form.lower()))
        return v[0] if v else 0

    # Find fixable self-maps
    fixes = []
    mg_protected = 0
    lemma_protected = 0
    for rowid, form, lemma, lang in self_maps:
        if form not in targets:
            continue
        # Skip if form is an MG headword (legitimate MG lemma)
        if form in mg_self_maps or strip_accents(form.lower()) in mg_stripped:
            mg_protected += 1
            continue
        # Skip if form is a productive AG lemma (other forms map to it)
        if form in productive_lemmas:
            lemma_protected += 1
            continue
        resolved = resolve_chain(form, targets, lemma_set)
        if resolved and resolved != form:
            # A monotonic (Modern-Greek-style) form resolving to a polytonic
            # Ancient-Greek lemma is only a leak when it crosses to a DIFFERENT
            # word (e.g. ή -> ὅ, ήδη -> οἶδα, είτε -> εἰμί) or when the form is
            # an ultra-frequent MG function word (articles). Same-word polytonic
            # accent/breathing restoration of ordinary content words
            # (ανάγκη -> ἀνάγκη, αἰτία ...) is legitimate and helps
            # Ancient/Byzantine text, so it is kept.
            if not _is_polytonic(form) and _is_polytonic(resolved):
                same_word = (strip_accents(form.lower())
                             == strip_accents(resolved.lower()))
                if not same_word or freq(form) >= MG_FUNCTION_FREQ:
                    mg_protected += 1
                    continue
            fixes.append((rowid, form, resolved, lang))
    if mg_protected:
        print(f"MG-protected (skipped): {mg_protected:,}")
    if lemma_protected:
        print(f"Lemma-protected (skipped): {lemma_protected:,}")

    print(f"Fixable self-maps: {len(fixes):,}")

    if args.dry_run:
        for _, form, resolved, lang in fixes[:20]:
            print(f"  {form} -> {resolved} (lang={lang})")
        if len(fixes) > 20:
            print(f"  ... and {len(fixes) - 20} more")

    if not fixes and not AMBIGUOUS_CORRECTIONS:
        print("Nothing to fix!")
        return

    # Apply self-map fixes
    if fixes and not args.dry_run:
        print(f"\nApplying {len(fixes):,} self-map fixes to lookup.db...")

    # Ensure all target lemmas exist in the lemmas table
    existing_lemmas = {row[0]: row[1] for row in
                       conn.execute("SELECT text, id FROM lemmas")}
    max_id = conn.execute("SELECT MAX(id) FROM lemmas").fetchone()[0] or 0

    if fixes and not args.dry_run:
        new_lemmas = set()
        for _, _, resolved, _ in fixes:
            if resolved not in existing_lemmas:
                new_lemmas.add(resolved)

        for lemma in sorted(new_lemmas):
            max_id += 1
            conn.execute("INSERT INTO lemmas (id, text) VALUES (?, ?)",
                          (max_id, lemma))
            existing_lemmas[lemma] = max_id

        # Update lookup entries
        for rowid, form, resolved, lang in fixes:
            lemma_id = existing_lemmas[resolved]
            conn.execute("UPDATE lookup SET lemma_id = ? WHERE rowid = ?",
                          (lemma_id, rowid))

        conn.commit()

    # --- Pass 2: apply manual corrections for ambiguous form_of entries ---
    if AMBIGUOUS_CORRECTIONS and not args.dry_run:
        correction_count = 0
        for form, correct_lemma in AMBIGUOUS_CORRECTIONS.items():
            if correct_lemma not in existing_lemmas:
                max_id += 1
                conn.execute("INSERT INTO lemmas (id, text) VALUES (?, ?)",
                              (max_id, correct_lemma))
                existing_lemmas[correct_lemma] = max_id

            lemma_id = existing_lemmas[correct_lemma]
            # Update all entries for this form (all lang variants)
            updated = conn.execute(
                "UPDATE lookup SET lemma_id = ? WHERE form = ?",
                (lemma_id, form)).rowcount
            if updated:
                correction_count += 1
                print(f"  Corrected: {form} -> {correct_lemma} "
                      f"({updated} entries)")

        if correction_count:
            conn.commit()
            print(f"Ambiguous corrections applied: {correction_count}")

    # Verify
    for test_form, expected in [("ἐρίσαντε", "ἐρίζω"),
                                ("δοῖεν", "δίδωμι")]:
        result = conn.execute("""
            SELECT le.text FROM lookup l JOIN lemmas le ON l.lemma_id = le.id
            WHERE l.form = ? AND l.lang = 'all'
        """, (test_form,)).fetchone()
        if result:
            status = "ok" if result[0] == expected else "WRONG"
            print(f"  Verify: {test_form} -> {result[0]} ({status})")

    conn.execute("ANALYZE")
    conn.close()
    print("Done!")


if __name__ == "__main__":
    main()

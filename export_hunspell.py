#!/usr/bin/env python3
"""Export Dilemma lookup tables as Hunspell .dic + .aff pairs.

This produces a compact Hunspell-format artifact for the Tonos iOS
keyboard extension. Two variants are emitted under build/hunspell/:

    el_GR_monotonic.dic / .aff   Modern Greek, monotonic
    grc_polytonic.dic  / .aff    Ancient + Medieval, polytonic

Both are accompanied by a sidecar .version file carrying a semver and
the dilemma commit hash the artifact was built from. Tonos reads this
to detect updates.

Approach
--------

Greek inflection is highly accent-sensitive, so exact SFX flag-sharing
across lemmas produces very long tails (~325K distinct signatures for
~426K MG lemmas). Rather than force every form through a shared affix
class, we take a pragmatic middle ground:

1. Group forms by lemma (from lookup.db).
2. For each lemma, compute the longest common prefix (stem) across
   its inflected forms and the multiset of suffixes.
3. If two or more lemmas share the exact same suffix-signature we emit
   one SFX rule-set and point both lemma stems at it.
4. Lemmas with a unique signature still get stem-compression:
   one .dic line per lemma, with a per-lemma SFX flag. We cap the
   suffix-rule expansion cost and fall back to plain wordlist entries
   for pathological cases.
5. Singleton lemmas (exactly one form) go in the .dic as plain words,
   no flag.

Frequency scoring
-----------------

Each .dic line carries a morphological field ``fr:<bucket>`` where
bucket is one of C, M, R, X (Common, Medium, Rare, eXtremely rare /
unseen in corpus). Buckets are computed from corpus counts found in
data/corpus_freq.json (GLAUx + Diorisis, 27M AG tokens) for AG forms
and data/mg_form_freq.json (Wiktionary + HNC-derived) for MG. Lookup
is done on the stripped-accent form so polytonic forms can inherit the
monotonic corpus count. Bucket edges:

    C: count >= 1000
    M: count >= 100
    R: count >= 1
    X: count == 0  (lemma-generated forms never observed)

Tonos can parse ``fr:`` to rank spelling candidates (e.g. prefer
suggesting 'και' over some obscure homograph).

Limitations
-----------

- Hunspell does not natively handle Greek iota subscript / polytonic
  breathings as separate graphemes, but since we store and match
  against NFC-normalized strings this is fine.
- Some Wiktionary-generated forms may be spurious for real usage.
  They are kept but tagged X so the consumer can filter.
- Capitalization is represented via the KEEPCASE flag. Proper names
  are tagged with a capital-only flag so 'σεραφειμ' is not falsely
  accepted for 'Σεραφείμ'. We err on the side of acceptance.

Usage
-----

    python3 export_hunspell.py                 # full export
    python3 export_hunspell.py --sanity 10000  # 10K-lemma sanity pass
    python3 export_hunspell.py --variant el    # only MG
    python3 export_hunspell.py --variant grc   # only AG
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from dilemma.form_sanitize import has_editorial_sigla, sanitize_form

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "build" / "hunspell"

# AG function-word forms that dilemma.py resolves via hardcoded rules
# rather than via the lookup table. These need to be added to the AG
# polytonic artifact explicitly or the keyboard will reject extremely
# common words like τὸ, τὴν, μοι. Mapping: form -> lemma.
AG_FUNCTION_WORDS = {
    # Definite article (ὁ)
    "ὁ": "ὁ", "ἡ": "ὁ", "τό": "ὁ", "τοῦ": "ὁ", "τῆς": "ὁ",
    "τῶν": "ὁ", "τόν": "ὁ", "τήν": "ὁ", "τά": "ὁ", "τοῖς": "ὁ",
    "ταῖς": "ὁ", "τῷ": "ὁ", "τῇ": "ὁ", "τούς": "ὁ", "τάς": "ὁ",
    "τοῖν": "ὁ", "ταῖν": "ὁ", "οἱ": "ὁ", "αἱ": "ὁ", "τώ": "ὁ",
    # Grave variants
    "τὸ": "ὁ", "τοὺς": "ὁ", "τὰ": "ὁ", "τὸν": "ὁ", "τὴν": "ὁ",
    "τὰς": "ὁ", "αἵ": "ὁ", "οἵ": "ὁ",
    # 1st/2nd person pronouns
    "μοι": "ἐγώ", "μοί": "ἐγώ", "μου": "ἐγώ", "με": "ἐγώ",
    "ἐμοί": "ἐγώ", "ἐμοῦ": "ἐγώ", "ἐμέ": "ἐγώ", "ἐγώ": "ἐγώ",
    "ἡμεῖς": "ἐγώ", "ἡμῶν": "ἐγώ", "ἡμῖν": "ἐγώ", "ἡμᾶς": "ἐγώ",
    "σοι": "σύ", "σοί": "σύ", "σου": "σύ", "σε": "σύ",
    "σοῦ": "σύ", "σύ": "σύ",
    "ὑμεῖς": "σύ", "ὑμῶν": "σύ", "ὑμῖν": "σύ", "ὑμᾶς": "σύ",
}

LOOKUP_DB = DATA / "lookup.db"
CORPUS_FREQ = DATA / "corpus_freq.json"
MG_FORM_FREQ = DATA / "mg_form_freq.json"
# Curated iconic AG polytonic surface forms and lemmas that are always
# promoted to bucket C, regardless of raw corpus token count. See the
# file's own _comment field for rationale.
CANONICAL_AG_FORMS = DATA / "canonical_ag_forms.json"

# Greek combining marks considered polytonic (absent in monotonic text)
POLYTONIC_MARKS = {0x0313, 0x0314, 0x0342, 0x0345}
# Grave accent (U+0300) is technically monotonic-absent too, but we
# treat it as polytonic for classification purposes.
POLYTONIC_MARKS_EXT = POLYTONIC_MARKS | {0x0300}

# Spacing (non-combining) characters that Greek orthography uses as a
# genuine mark on a word. Unicode files them as Sk (modifier symbol),
# not Mn (nonspacing mark), so a combining-category test on its own
# reports a fully marked form such as 'δ᾽' as bare. Members:
#   U+1FBD GREEK KORONIS - the elision and crasis apostrophe. Now that
#     build_lookup_db.py runs sanitize_form at ingestion, a trailing
#     combining psili used as an apostrophe is rewritten to this
#     character before the exporter ever sees it, so elided forms that
#     lose their accent along with the elided syllable (δ᾽, κατ᾽, μετ᾽,
#     μηδ᾽) reach this gate carrying no combining mark at all.
#   U+1FBF GREEK PSILI and U+1FFE GREEK DASIA - the spacing breathings.
#     sanitize_form reattaches a leading one to its base letter but
#     leaves a trailing one alone, so a source that writes elision with
#     a spacing psili (κατ᾿, μετ᾿) reaches the exporter carrying it.
#     Dasia does not currently occur in that position in lookup.db; it
#     is listed so the pair stays symmetric if a source emits one.
# Excluded on purpose, because accepting them would defeat this gate
# rather than repair it:
#   U+00B4 ACUTE ACCENT and U+0384 GREEK TONOS - no Greek word is
#     spelled with a free-standing acute. Where one turns up it is
#     either an accent that failed to combine or the numeral sign
#     keraia, as in the Milesian numerals 'τ΄' and 'ϟ΄'; accepting it
#     would ship both as correct polytonic spellings.
#   U+0375 GREEK LOWER NUMERAL SIGN - a numeral marker, not a mark on
#     a word.
#   U+2019 RIGHT SINGLE QUOTATION MARK and U+02BC MODIFIER LETTER
#     APOSTROPHE - ordinary punctuation and a modifier letter. Some
#     sources do write elision with them, but they are also what a
#     stray quotation mark in noisy text looks like, and neither
#     carries Greek-specific meaning the way the koronis does. A form
#     that needs one should have it normalized to U+1FBD at ingestion.
SPACING_DIACRITICS = {0x1FBD, 0x1FBF, 0x1FFE}

# Greek vowel letters, both cases. Final sigma is a consonant, so the
# aphaeresized '᾽ς' is not affected by the rule below.
GREEK_VOWELS = frozenset("αεηιουωΑΕΗΙΟΥΩ")

# Leaving the three characters above out of SPACING_DIACRITICS does not
# by itself keep numerals out of the artifact, because a source that
# prints the keraia as a koronis writes the Milesian numerals with a
# character that IS in the set: lookup.db holds 'α᾽' (1) under lemma
# α, 'ε᾽' (5) under πέντε and 'ο᾽' (70) under lemma ο. It also holds
# vowel-initial spellings that simply lost their breathing on the way
# in, such as 'ημειβετ᾿' beside the correct 'ἠμείβετ᾿'.
#
# What separates those from a real elision is the first letter, not the
# length - 'δ᾽', 'μ᾽', 'σ᾽', 'κ᾽' and 'ν᾽' are one-letter elisions
# that have to keep passing. Polytonic Greek writes a breathing over
# every word-initial vowel, and neither elision, which drops the end of
# a word, nor aphaeresis, whose koronis stands in front of the letters
# that survive, can take that breathing away. sanitize_form has already
# reattached any leading spacing breathing to its vowel before a form
# reaches this gate, so a form whose only mark is a spacing one and
# whose first letter is a bare vowel is not an elided word at all.
#
# A numeral whose letters start with a consonant still passes, and must:
# 'δ᾽' is the numeral 4 as well as the elided δέ, 'κ᾽' is 20 as well as
# the elided κε, and no rule on the spelling can tell those apart. That
# costs nothing, because the keyboard has to accept the string either
# way.


def _first_letter(s: str) -> str:
    """Return the first letter of s, decomposed so a precomposed vowel
    is reported by its base letter, or '' if s has no letter at all."""
    for c in unicodedata.normalize("NFD", s):
        if unicodedata.category(c)[0] == "L":
            return c
    return ""


def _is_greek_letter(c: str) -> bool:
    return "Ͱ" <= c <= "Ͽ" or "ἀ" <= c <= "῿"


def marked_by_spacing_diacritic(s: str) -> bool:
    """True if s is spelled the way an elided or aphaeresized word is:
    its only mark is a spacing koronis or breathing, and it opens on a
    Greek consonant rather than on a bare vowel. See the comment above
    SPACING_DIACRITICS for why the opening letter is the test."""
    if not any(ord(c) in SPACING_DIACRITICS for c in s):
        return False
    first = _first_letter(s)
    return bool(first) and _is_greek_letter(first) and first not in GREEK_VOWELS


def has_polytonic(s: str) -> bool:
    """True if the string carries any mark exclusive to polytonic script
    (breathings, circumflex, grave, iota subscript)."""
    nfd = unicodedata.normalize("NFD", s)
    return any(ord(c) in POLYTONIC_MARKS_EXT for c in nfd)


def has_any_diacritic(s: str) -> bool:
    """True if the string carries a mark that makes it AG orthography.
    Used to decide whether a form is 'accented enough' to ship in the
    AG polytonic variant. We keep acute-only forms (e.g. 'ζωή') because
    they are the canonical AG lexicon entries, and elided or
    aphaeresized forms whose only mark is a spacing koronis or
    breathing (e.g. 'δ᾽', 'κατ᾿', '᾽ς'), but we drop fully-stripped
    fallback keys the DB carries for case-insensitive lookup. A
    combining mark settles it on its own; a spacing mark counts only on
    a form spelled the way an elided word is, which is what
    marked_by_spacing_diacritic decides."""
    nfd = unicodedata.normalize("NFD", s)
    if any(unicodedata.category(c) == "Mn" for c in nfd):
        return True
    return marked_by_spacing_diacritic(s)


def strip_accents(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize(
        "NFC", "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    ).lower()


def freq_lookup(form: str, freq_map: dict[str, int]) -> int:
    """Look up a form's corpus count under several normalization keys.

    The MG frequency map stores keys with monotonic accents preserved
    and lowercase (e.g. 'παπαδοπούλου'), while the AG corpus_freq map
    uses fully-stripped lowercase keys. We try: the form as-is,
    lowercased, and accent-stripped + lowercased, returning the first
    non-zero hit (or 0).
    """
    if form in freq_map:
        return freq_map[form]
    lower = form.lower()
    if lower in freq_map:
        return freq_map[lower]
    stripped = strip_accents(form)
    if stripped in freq_map:
        return freq_map[stripped]
    return 0


def freq_bucket(count: int) -> str:
    if count >= 1000:
        return "C"
    if count >= 100:
        return "M"
    if count >= 1:
        return "R"
    return "X"


# Lemma-aggregate thresholds. A lemma whose combined corpus count
# across all its inflected forms exceeds the C threshold is considered
# culturally common enough that every one of its polytonic forms should
# land in bucket C, even if individual inflections are rare. This
# corrects for the corpus-mass dilution effect where highly-inflected
# classical lemmas (e.g. Ζεύς, λόγος, ἀείδω) have their frequency mass
# spread across 100+ surface forms. Numbers were chosen so the C promo
# triggers for lemmas that are unambiguously famous in the classical
# canon; they are higher than per-form thresholds to keep the bucket
# precise (roughly: Zeus-famous, not hapax-famous).
LEMMA_AGG_C_MIN = 20_000
LEMMA_AGG_M_MIN = 5_000


def load_canonical_ag_sets() -> tuple[set[str], set[str]]:
    """Load (canonical_forms, canonical_lemmas) from CANONICAL_AG_FORMS.

    Both sets contain NFC-normalized polytonic strings. The forms set
    is matched by exact surface equality (so monotonic variants do not
    get promoted along with their polytonic siblings). The lemmas set
    is matched against each (form, lemma) pair's lemma text; any form
    of a canonical lemma gets promoted to at least C.
    """
    if not CANONICAL_AG_FORMS.exists():
        return set(), set()
    with open(CANONICAL_AG_FORMS, encoding="utf-8") as f:
        raw = json.load(f)
    forms = {unicodedata.normalize("NFC", s)
             for s in raw.get("forms_c", [])}
    lemmas = {unicodedata.normalize("NFC", s)
              for s in raw.get("lemmas_c", [])}
    return forms, lemmas


def compute_lemma_totals(
    form_lemma: list[tuple[str, str]],
    freq_map: dict[str, int],
) -> dict[str, int]:
    """Return lemma -> sum of corpus counts across all its surface forms.

    Each form's count is looked up via freq_lookup (which tries NFC,
    lowercase, and accent-stripped lowercase keys). A form shared by
    multiple lemmas is counted once per lemma, which is a mild double-
    count, but acceptable because we only use the aggregate to cross a
    fairly conservative promotion threshold, not as a precise measure.
    """
    by_lemma: dict[str, set[str]] = defaultdict(set)
    for form, lemma in form_lemma:
        by_lemma[lemma].add(form)
    totals: dict[str, int] = {}
    for lemma, forms in by_lemma.items():
        t = 0
        for f in forms:
            t += freq_lookup(f, freq_map)
        totals[lemma] = t
    return totals


def bucket_for(
    form: str,
    lemma: str,
    form_count: int,
    lemma_total: int,
    canonical_forms: set[str],
    canonical_lemmas: set[str],
) -> str:
    """Choose a frequency bucket taking canonical seed and lemma total
    into account. Rules, from strongest to weakest:

      1. If the exact polytonic form is in the canonical seed -> C.
      2. If the lemma is in the canonical seed AND this particular form
         carries any polytonic mark (breathing, circumflex, grave, iota
         subscript) -> C. Acute-only or fully-stripped forms of a
         canonical lemma are NOT promoted, so that a polytonic form
         always outranks a monotonic sibling at tiebreak.
      3. Lemma-aggregate: count across all forms of this lemma >= 20K -> C;
         >= 5K -> at least M. Same polytonic-only gating applies.
      4. Otherwise use the per-form corpus count with the default edges.
    """
    if form in canonical_forms:
        return "C"
    form_is_polytonic = has_polytonic(form)
    if lemma in canonical_lemmas and form_is_polytonic:
        return "C"
    if lemma_total >= LEMMA_AGG_C_MIN and form_is_polytonic:
        return "C"
    base = freq_bucket(form_count)
    if (lemma_total >= LEMMA_AGG_M_MIN and form_is_polytonic
            and base == "R"):
        return "M"
    return base


def load_freq_maps() -> tuple[dict[str, int], dict[str, int]]:
    """Return (mg_freq, ag_freq). Both map stripped-lowercase form to count."""
    mg_freq: dict[str, int] = {}
    ag_freq: dict[str, int] = {}

    if MG_FORM_FREQ.exists():
        with open(MG_FORM_FREQ, encoding="utf-8") as f:
            raw = json.load(f)
        for form, count in raw.items():
            mg_freq[form] = count

    if CORPUS_FREQ.exists():
        with open(CORPUS_FREQ, encoding="utf-8") as f:
            raw = json.load(f)
        for form, counts in raw.get("forms", {}).items():
            ag_freq[form] = counts[0] if counts else 0

    return mg_freq, ag_freq


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        return out
    except Exception:
        return "unknown"


def read_version_file() -> str:
    vp = ROOT / "VERSION"
    if vp.exists():
        return vp.read_text().strip()
    return "0.0.0"


# Hunspell flag format: we use FLAG num mode so each rule gets an
# integer flag 1..65535. Hunspell limits combined flags on a word via
# comma separation (e.g. word/123,456). This gives us >64K flags,
# plenty for Greek's highly varied inflection signatures.
def gen_flag_names():
    # Reserve 1..9 as system flags (we use none currently but leave
    # headroom for future). Start at 1000 so our rule flags are easy
    # to distinguish from any reserved values.
    i = 1000
    while i < 65500:
        yield str(i)
        i += 1


def select_forms(
    conn: sqlite3.Connection,
    variant: str,
    keep_lemmas: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Return [(form, lemma_text)] for the given variant.

    variant='el'  -> Modern Greek monotonic forms. This is the union of:
                     (a) every src='el' row (native MG overrides), and
                     (b) every monotonic form from src='grc' + lang='all'.
                     Rationale: a lot of shared vocabulary (articles,
                     common verbs, proper names) is stored on the 'grc'
                     side with src='grc' even though it is perfectly
                     valid Modern Greek. Excluding those would miss
                     words like 'δεν', 'καί', 'σκύλος'.
    variant='grc' -> all src='grc' rows containing polytonic marks.
                     This is the Ancient + Medieval vocabulary with
                     diacritics. We exclude the stripped monotonic
                     duplicate keys the DB carries as fallback.
                     If keep_lemmas is provided, any lemma whose text
                     is in that set is kept even if none of its forms
                     carry a breathing/circumflex/grave/iota-subscript
                     mark. This rescues canonical AG lemmas whose
                     spellings are entirely acute-only (e.g. Πλάτων,
                     Μένανδρος, πόλεμος) which would otherwise be
                     excluded as 'pure-monotonic, not AG'.
    """
    cur = conn.cursor()

    if variant == "el":
        # MG-specific overrides from lookup.db
        rows = cur.execute(
            """
            SELECT k.form, l.text
            FROM lookup k JOIN lemmas l ON k.lemma_id = l.id
            WHERE k.src = 'el'
            """
        )
        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str]] = []
        for form, lemma in rows:
            if has_polytonic(form):
                continue
            key = (form, lemma)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    if variant == "grc":
        # AG vocabulary. The DB stores, per src='grc' lemma, three
        # parallel copies of each form: truly-polytonic, acute-only,
        # and fully-stripped. For the AG keyboard artifact we want:
        #
        #  - every truly-polytonic form (breathing, circumflex, grave,
        #    iota subscript), and
        #  - every acute-only form whose lemma has a polytonic sibling
        #    in its paradigm (so σοφίας rides along with σοφίᾳ).
        #
        # We skip fully-stripped forms (they are just lookup fallback
        # keys and would bloat the artifact without adding correct
        # AG orthography).
        rows = cur.execute(
            """
            SELECT k.form, l.text
            FROM lookup k JOIN lemmas l ON k.lemma_id = l.id
            WHERE k.src = 'grc'
            """
        )
        by_lemma: dict[str, list[str]] = defaultdict(list)
        for form, lemma in rows:
            if not has_any_diacritic(form):
                continue
            by_lemma[lemma].append(form)

        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str]] = []
        keep = keep_lemmas or set()
        for lemma, forms in by_lemma.items():
            is_canonical_keep = lemma in keep
            if not is_canonical_keep:
                if not any(has_polytonic(f) for f in forms):
                    continue  # pure-monotonic lemma, not AG
                # also require lemma text itself is diacritic-bearing
                if not has_any_diacritic(lemma):
                    continue
            for f in forms:
                key = (f, lemma)
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
        return out

    raise ValueError(f"Unknown variant {variant!r}")


def longest_common_prefix(strs: list[str]) -> str:
    if not strs:
        return ""
    cp = strs[0]
    for s in strs[1:]:
        i = 0
        while i < len(cp) and i < len(s) and cp[i] == s[i]:
            i += 1
        cp = cp[:i]
        if not cp:
            break
    return cp


def filter_by_lemma_freq(
    form_lemma: list[tuple[str, str]],
    freq_map: dict[str, int],
    min_lemma_count: int = 1,
    strict_acute_min: int | None = None,
) -> list[tuple[str, str]]:
    """Drop lemmas whose most-frequent form is below min_lemma_count.

    This removes Wiktionary-generated lemmas (and their inflected forms)
    that have never been observed in any corpus, which are typically
    spurious artifacts of Lua paradigm expansion. Lemmas with any
    attested form are kept whole (all their generated forms survive).

    If strict_acute_min is set, then within a surviving lemma, acute-only
    forms (no breathings/circumflex/iota-sub/grave) must individually
    meet that threshold to be kept. Polytonic-bearing forms are always
    kept when their lemma is kept. This is mainly used for the AG
    polytonic variant, where the DB carries many acute-only inflections
    that are not canonical AG spellings (e.g. post-Byzantine usage).
    """
    if min_lemma_count <= 0 and strict_acute_min is None:
        return form_lemma

    by_lemma: dict[str, list[str]] = defaultdict(list)
    for form, lemma in form_lemma:
        by_lemma[lemma].append(form)

    keep_lemmas: set[str] = set()
    for lemma, forms in by_lemma.items():
        candidates = set(forms) | {lemma}
        for f in candidates:
            if freq_lookup(f, freq_map) >= min_lemma_count:
                keep_lemmas.add(lemma)
                break

    out: list[tuple[str, str]] = []
    for form, lemma in form_lemma:
        if lemma not in keep_lemmas:
            continue
        if strict_acute_min is not None and not has_polytonic(form):
            # acute-only or undecorated form: require per-form count
            if freq_lookup(form, freq_map) < strict_acute_min:
                continue
        out.append((form, lemma))
    return out


def sanitize_export_pairs(
    form_lemma: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], int, int]:
    """Sanitize and deduplicate pairs before affix compression.

    Returns ``(pairs, changed_forms, editorial_dropped)``. Editorial notation
    is never meaningful Hunspell input and can make regex-based consumers
    reject the complete affix file.
    """
    sanitized: list[tuple[str, str]] = []
    changed_forms = 0
    editorial_dropped = 0
    dedup: set[tuple[str, str]] = set()
    for form, lemma in form_lemma:
        clean_form = sanitize_form(form)
        clean_lemma = sanitize_form(lemma)
        if not clean_form:
            continue
        if (has_editorial_sigla(clean_form)
                or has_editorial_sigla(clean_lemma)):
            editorial_dropped += 1
            continue
        pair = (clean_form, clean_lemma)
        if pair in dedup:
            continue
        dedup.add(pair)
        if clean_form != form:
            changed_forms += 1
        sanitized.append(pair)
    return sanitized, changed_forms, editorial_dropped


def build_sfx_rules(
    form_lemma: list[tuple[str, str]],
    freq_map: dict[str, int],
    max_singletons_to_inline: int = 1,
    canonical_forms: set[str] | None = None,
    canonical_lemmas: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Given (form, lemma) pairs, produce (aff_lines, dic_lines).

    Strategy:
      - Group by lemma.
      - For each lemma compute stem = LCP(forms).
      - Build suffix-signature = tuple(sorted(set(form[len(stem):]))).
      - Lemmas sharing a signature share a flag.
      - Singleton lemmas (1 form) just go in .dic as plain words.

    The .aff returned contains only the SFX rule blocks, not the header.

    Frequency bucketing uses bucket_for(), which combines per-form
    corpus count, per-lemma aggregate count, and the canonical seed
    lists passed in. When canonical_forms/canonical_lemmas are None,
    behaviour reduces to the original per-form-count bucketing.
    """
    if canonical_forms is None:
        canonical_forms = set()
    if canonical_lemmas is None:
        canonical_lemmas = set()
    lemma_totals = compute_lemma_totals(form_lemma, freq_map)
    # Group by lemma
    by_lemma: dict[str, set[str]] = defaultdict(set)
    for form, lemma in form_lemma:
        by_lemma[lemma].add(form)

    # Singletons: emit as plain words.
    singletons: list[tuple[str, str]] = []  # (form, lemma)
    # Multi-form lemmas bucketed by signature
    sig_to_lemmas: dict[tuple, list[tuple[str, str, list[str]]]] = defaultdict(list)
    # values: (lemma, stem, [suffixes preserving duplicates])

    for lemma, forms in by_lemma.items():
        flist = sorted(forms)
        if len(flist) == 1:
            singletons.append((flist[0], lemma))
            continue
        stem = longest_common_prefix(flist)
        # If stem is empty, the lemma's forms share no prefix; this
        # happens with suppletive verbs (e.g. λέγω/εἶπον). Skip
        # compression and emit as individual words.
        if not stem:
            for f in flist:
                singletons.append((f, lemma))
            continue
        suffixes = [f[len(stem):] for f in flist]
        sig = tuple(sorted(set(suffixes)))
        sig_to_lemmas[sig].append((lemma, stem, suffixes))

    # Only signatures with 2+ lemmas get a flag (real compression win).
    # Unique/singleton signatures are inlined as plain wordlist entries.
    # This keeps the .aff under control and keeps flag IDs below
    # Hunspell's FLAG num cap (65535).
    sorted_sigs = sorted(
        sig_to_lemmas.items(), key=lambda x: (-len(x[1]), x[0])
    )

    flag_iter = gen_flag_names()
    sig_to_flag: dict[tuple, str] = {}
    aff_blocks: list[str] = []
    inlined_lemmas: list[tuple[str, list[str]]] = []  # (lemma, forms)
    MAX_FLAGS = 60_000  # headroom below 65535

    flags_assigned = 0
    for sig, lemma_info in sorted_sigs:
        if len(lemma_info) < 2 or flags_assigned >= MAX_FLAGS:
            # Inline: every form of every lemma becomes a plain entry.
            for lemma, stem, suffixes in lemma_info:
                forms = [stem + s for s in suffixes]
                inlined_lemmas.append((lemma, forms))
            continue
        flag = next(flag_iter)
        flags_assigned += 1
        sig_to_flag[sig] = flag
        unique_suffixes = sig  # already a sorted unique tuple
        n = len(unique_suffixes)
        header = f"SFX {flag} Y {n}"
        body = [header]
        for suf in unique_suffixes:
            add = suf if suf else "0"
            body.append(f"SFX {flag} 0 {add} .")
        aff_blocks.append("\n".join(body))

    # Build .dic lines
    dic_lines: list[str] = []

    def fmt_entry(word: str, flag: str | None, bucket: str) -> str:
        morph = f"fr:{bucket}"
        if flag:
            return f"{word}/{flag}\t{morph}"
        return f"{word}\t{morph}"

    def pick_bucket(form: str, lemma: str, form_count: int) -> str:
        return bucket_for(
            form=form,
            lemma=lemma,
            form_count=form_count,
            lemma_total=lemma_totals.get(lemma, 0),
            canonical_forms=canonical_forms,
            canonical_lemmas=canonical_lemmas,
        )

    # Emit stems with flags (shared-signature lemmas). A stem line
    # represents all forms of possibly-several lemmas sharing the
    # same suffix signature; we pick the strongest (highest) bucket
    # across the covered forms so the stem does not under-rank any
    # iconic form it expands to. Canonical seed membership is checked
    # against each individual expanded form.
    emitted_stems: set[tuple[str, str]] = set()
    bucket_rank = {"C": 3, "M": 2, "R": 1, "X": 0}
    for sig, lemma_info in sorted_sigs:
        if sig not in sig_to_flag:
            continue
        flag = sig_to_flag[sig]
        for lemma, stem, suffixes in lemma_info:
            best_bucket = "X"
            for suf in set(suffixes):
                full = stem + suf
                c = freq_lookup(full, freq_map)
                b = pick_bucket(full, lemma, c)
                if bucket_rank[b] > bucket_rank[best_bucket]:
                    best_bucket = b
            key = (stem, flag)
            if key in emitted_stems:
                continue
            emitted_stems.add(key)
            dic_lines.append(fmt_entry(stem, flag, best_bucket))

    # Emit inlined lemmas (unique-signature, expanded to individual forms)
    emitted_plain: set[str] = set()
    for lemma, forms in inlined_lemmas:
        for form in forms:
            if form in emitted_plain:
                continue
            emitted_plain.add(form)
            c = freq_lookup(form, freq_map)
            dic_lines.append(fmt_entry(form, None, pick_bucket(form, lemma, c)))

    # Emit singletons as plain words
    for form, lemma in singletons:
        if form in emitted_plain:
            continue
        emitted_plain.add(form)
        c = freq_lookup(form, freq_map)
        dic_lines.append(fmt_entry(form, None, pick_bucket(form, lemma, c)))

    return aff_blocks, dic_lines


def write_variant(
    variant: str,
    form_lemma: list[tuple[str, str]],
    freq_map: dict[str, int],
    out_dir: Path,
    dic_name: str,
    lang_tag: str,
    version: str,
    commit: str,
    canonical_forms: set[str] | None = None,
    canonical_lemmas: set[str] | None = None,
) -> dict:
    """Emit <dic_name>.dic, <dic_name>.aff, <dic_name>.version. Return stats."""
    out_dir.mkdir(parents=True, exist_ok=True)

    aff_blocks, dic_lines = build_sfx_rules(
        form_lemma, freq_map,
        canonical_forms=canonical_forms,
        canonical_lemmas=canonical_lemmas,
    )

    # .aff header
    aff_header = [
        f"# Dilemma Hunspell export for {lang_tag}",
        f"# Version {version} (built from commit {commit})",
        f"# Generated by export_hunspell.py",
        "SET UTF-8",
        f"LANG {lang_tag}",
        "FLAG num",
        # TRY order: common Greek letters first, for Hunspell's own
        # suggestion mechanism. Roughly ordered by Modern Greek letter
        # frequency.
        "TRY αεοιτνσρπλκημυδγωβχφξζθψηάέήίόύώϊϋΐΰ"
        "αβγδεζηθικλμνξοπρστυφχψω",
        # Flag used to mark morphological info field
        "WORDCHARS αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
        "άέήίόύώϊϋΐΰᾰᾱῐῑῠῡἀἁἂἃἄἅἆἇἐἑἒἓἔἕἠἡἢἣἤἥἦἧ"
        "ἰἱἲἳἴἵἶἷὀὁὂὃὄὅὐὑὒὓὔὕὖὗὠὡὢὣὤὥὦὧᾳῃῳ'ʼ’",
        "",
    ]
    aff_text = "\n".join(aff_header) + "\n".join(aff_blocks) + "\n"

    aff_path = out_dir / f"{dic_name}.aff"
    aff_path.write_text(aff_text, encoding="utf-8")

    # .dic: first line is count, then entries
    dic_header = f"{len(dic_lines)}\n"
    dic_path = out_dir / f"{dic_name}.dic"
    with open(dic_path, "w", encoding="utf-8") as f:
        f.write(dic_header)
        for line in dic_lines:
            f.write(line)
            f.write("\n")

    # Sidecar version file
    ver_path = out_dir / f"{dic_name}.version"
    ver_path.write_text(
        f"version: {version}\n"
        f"commit: {commit}\n"
        f"variant: {lang_tag}\n"
        f"entries: {len(dic_lines)}\n"
        f"aff_rules: {sum(len(b.splitlines()) - 1 for b in aff_blocks)}\n",
        encoding="utf-8",
    )

    return {
        "aff_bytes": aff_path.stat().st_size,
        "dic_bytes": dic_path.stat().st_size,
        "entries": len(dic_lines),
        "aff_rules": sum(len(b.splitlines()) - 1 for b in aff_blocks),
        "aff_path": str(aff_path),
        "dic_path": str(dic_path),
    }


DEFAULT_MIN_LEMMA_FREQ = {"el": 1, "grc": 3}


def run_export(sanity: int | None, variants: list[str],
               min_lemma_freq: int | None = None) -> None:
    if not LOOKUP_DB.exists():
        print(f"ERROR: {LOOKUP_DB} not found. Download with "
              f"`huggingface-cli download ciscoriordan/dilemma --local-dir . "
              f"--include 'data/*'`",
              file=sys.stderr)
        sys.exit(1)

    commit = get_git_commit()
    version = read_version_file()

    print(f"Dilemma Hunspell export")
    print(f"  version: {version}")
    print(f"  commit:  {commit}")
    print(f"  sanity:  {sanity if sanity else 'off (full export)'}")
    print(f"  output:  {OUT}")
    print()

    print("Loading frequency maps...")
    mg_freq, ag_freq = load_freq_maps()
    print(f"  MG freq: {len(mg_freq):,} entries")
    print(f"  AG freq: {len(ag_freq):,} entries")
    canonical_forms, canonical_lemmas = load_canonical_ag_sets()
    print(f"  AG canonical forms (pin to C): {len(canonical_forms):,}")
    print(f"  AG canonical lemmas (pin to C): {len(canonical_lemmas):,}")
    print()

    conn = sqlite3.connect(str(LOOKUP_DB))
    conn.execute("PRAGMA mmap_size=268435456")

    for variant in variants:
        print(f"=== Variant: {variant} ===")
        keep = canonical_lemmas if variant == "grc" else None
        form_lemma = select_forms(conn, variant, keep_lemmas=keep)
        print(f"  {len(form_lemma):,} raw (form, lemma) pairs from lookup.db")

        if variant == "grc":
            # Augment with hardcoded AG function words (article, pronouns)
            # that dilemma.py resolves via rules, not lookup.db
            existing_forms = {f for f, _ in form_lemma}
            added = 0
            for form, lemma in AG_FUNCTION_WORDS.items():
                if form not in existing_forms:
                    form_lemma.append((form, lemma))
                    added += 1
            if added:
                print(f"  +{added} AG function-word forms")

        if variant == "el":
            # Augment MG from mg_form_freq.json: any form with count >= 1
            # that isn't already present. Resolve its lemma by querying
            # lookup.db for the first match on the form (src='grc' or
            # src='el'). This pulls in widely-attested MG vocabulary
            # (δεν, καί, σκύλος) that is stored only on the 'grc' side
            # of the lookup table.
            existing_forms = {f for f, _ in form_lemma}
            added = 0
            for form, count in mg_freq.items():
                if count < 1:
                    continue
                if form in existing_forms:
                    continue
                if has_polytonic(form):
                    continue
                # Find lemma
                row = conn.execute(
                    "SELECT l.text FROM lookup k "
                    "JOIN lemmas l ON k.lemma_id = l.id "
                    "WHERE k.form = ? LIMIT 1",
                    (form,)
                ).fetchone()
                lemma = row[0] if row else form
                form_lemma.append((form, lemma))
                added += 1
            print(f"  +{added:,} forms augmented from mg_form_freq.json")

        threshold = (min_lemma_freq if min_lemma_freq is not None
                     else DEFAULT_MIN_LEMMA_FREQ.get(variant, 0))
        if threshold > 0:
            fm = mg_freq if variant == "el" else ag_freq
            before = len(form_lemma)
            # For AG, additionally require acute-only forms to themselves
            # be corpus-attested (>= 1) - this trims post-Byzantine
            # spellings that lookup.db carries for fallback only.
            strict = 1 if variant == "grc" else None
            form_lemma = filter_by_lemma_freq(
                form_lemma, fm,
                min_lemma_count=threshold,
                strict_acute_min=strict,
            )
            print(f"  After freq filter (>= {threshold}"
                  f"{', strict_acute>=1' if strict else ''}): "
                  f"{len(form_lemma):,} pairs (dropped {before-len(form_lemma):,})")
        if sanity:
            # Cap to the first N lemmas for a sanity pass
            seen_lemmas: set[str] = set()
            capped: list[tuple[str, str]] = []
            for f, l in form_lemma:
                if l not in seen_lemmas:
                    if len(seen_lemmas) >= sanity:
                        continue
                    seen_lemmas.add(l)
                capped.append((f, l))
            form_lemma = [(f, l) for f, l in form_lemma if l in seen_lemmas]
            print(f"  Sanity pass: {len(seen_lemmas):,} lemmas, "
                  f"{len(form_lemma):,} forms")

        # Belt-and-braces guard: sanitize every form so a misplaced combining
        # breathing (leading U+0313/U+0314 or trailing U+0313/U+0314 used as
        # an apostrophe) never reaches the .dic. See form_sanitize.sanitize_form
        # for the full rules. Lemmas are sanitized too so that two lemma
        # spellings that differ only by this bug collapse onto one paradigm.
        sanitized, changed_forms, editorial_dropped = sanitize_export_pairs(
            form_lemma
        )
        if changed_forms:
            print(f"  Guard: sanitized {changed_forms:,} forms "
                  f"(misplaced/orphan breathing marks)")
        if editorial_dropped:
            print(f"  Guard: dropped {editorial_dropped:,} "
                  "editorial-notation pairs")
        form_lemma = sanitized

        if variant == "el":
            stats = write_variant(
                variant=variant,
                form_lemma=form_lemma,
                freq_map=mg_freq,
                out_dir=OUT,
                dic_name="el_GR_monotonic",
                lang_tag="el_GR",
                version=version,
                commit=commit,
            )
        else:
            stats = write_variant(
                variant=variant,
                form_lemma=form_lemma,
                freq_map=ag_freq,
                out_dir=OUT,
                dic_name="grc_polytonic",
                lang_tag="grc",
                version=version,
                commit=commit,
                canonical_forms=canonical_forms,
                canonical_lemmas=canonical_lemmas,
            )

        print(f"  entries: {stats['entries']:,}")
        print(f"  aff rules: {stats['aff_rules']:,}")
        print(f"  .dic: {stats['dic_bytes']/1024/1024:.2f} MB "
              f"({stats['dic_path']})")
        print(f"  .aff: {stats['aff_bytes']/1024/1024:.2f} MB "
              f"({stats['aff_path']})")
        total_mb = (stats['dic_bytes'] + stats['aff_bytes']) / 1024 / 1024
        print(f"  total: {total_mb:.2f} MB")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", type=int, default=None,
                    help="Cap to N lemmas for a quick sanity run")
    ap.add_argument("--variant", choices=["el", "grc", "both"],
                    default="grc",
                    help="Which variant to emit. Defaults to grc "
                         "(Ancient + Medieval polytonic); the el "
                         "(Modern Greek monotonic) variant is retained "
                         "for other downstream consumers but is not "
                         "shipped in the Tonos keyboard.")
    ap.add_argument("--min-lemma-freq", type=int, default=None,
                    help="Override per-variant default: drop lemmas "
                         "whose most-attested form has corpus count < "
                         "this value. Defaults are el=1, grc=3.")
    args = ap.parse_args()

    variants = ["el", "grc"] if args.variant == "both" else [args.variant]
    run_export(args.sanity, variants, min_lemma_freq=args.min_lemma_freq)


if __name__ == "__main__":
    main()

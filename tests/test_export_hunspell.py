"""Tests for the Hunspell exporter's 'is this form marked?' gate.

`has_any_diacritic` decides which Ancient Greek surface forms are
orthographically marked enough to ship in the polytonic keyboard
artifact. It has two jobs that pull against each other, so both
directions are tested here:

  - Keep every genuinely marked spelling, including an elided form
    whose accent disappeared with the elided syllable and whose only
    remaining mark is a SPACING character (U+1FBD GREEK KORONIS and
    the spacing breathings). Unicode categorizes those as Sk rather
    than Mn, so a combining-mark test alone silently drops δ᾽, κατ᾽,
    μετ᾽ and μηδ᾽ - which lookup.db does contain, and which the
    keyboard has to accept.

  - Reject the fully-stripped fallback keys lookup.db carries for
    case-insensitive lookup. A bare μη or και must never reach a
    polytonic dictionary.

  - Reject a fallback key even when a spacing mark survives the
    stripping. This is the direction a widened gate gets wrong:
    strip_accents removes combining marks only, so the key of an
    ACCENTED elided form keeps its trailing koronis (ταῦτ᾽ -> ταυτ᾽).
    A Milesian numeral written with a koronis in place of the keraia
    has the same shape (ε᾽, the numeral five, which the corpora
    lemmatize to πέντε). Neither is Greek orthography and neither may
    reach the keyboard.

Those last cases are checked through select_forms rather than through
has_any_diacritic, because what has to be true is that the form is not
admitted; which of the two functions carries the rule is a choice the
exporter is free to make.

Measured against data/lookup.db on 2026-09-20: 136 src='grc' forms
carry a spacing mark and no combining mark, and 80 of them reach
select_forms' output. Most are the elided spellings the widening was
meant to rescue; the rest are the class tested below.
"""

from __future__ import annotations

import sqlite3
import sys
import unicodedata

import pytest

from dilemma.form_sanitize import (
    has_editorial_sigla,
    resolve_editorial_form,
    sanitize_form,
)
from export_hunspell import (
    SPACING_DIACRITICS,
    has_any_diacritic,
    sanitize_export_pairs,
    select_forms,
)

KORONIS = "᾽"
SPACING_PSILI = "᾿"
SPACING_DASIA = "῾"
COMBINING_PSILI = "̓"


def test_editorial_forms_resolve_only_optional_final_nu():
    assert resolve_editorial_form("καθᾶσι(ν") == ("καθᾶσι", "καθᾶσιν")
    assert resolve_editorial_form("καθᾶσι(ν)") == ("καθᾶσι", "καθᾶσιν")
    assert resolve_editorial_form("ἀ[ζηχὲς") == ()
    assert resolve_editorial_form("λη(νοῦ)") == ()
    assert has_editorial_sigla("[δηρ]ιάζομαι") is True
    assert has_editorial_sigla("δηριάζομαι") is False


def test_export_guard_drops_editorial_forms_and_lemmas():
    pairs, changed, dropped = sanitize_export_pairs([
        ("κατεβρεχθῶσι(ν", "βρέχω"),
        ("ἀ[ζηχὲς", "ἀζηχής"),
        ("καθαρός", "[καθ]αρός"),
        ("γραφῆσ", "γραφή"),
    ])

    assert pairs == [("γραφῆς", "γραφή")]
    assert changed == 1
    assert dropped == 3


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# Elided and aphaeresized forms whose ONLY mark is a spacing koronis.
# Every one of these is in lookup.db, and the Tonos dictionary shipped
# in April 2026 accepts all of them.
KORONIS_ONLY_FORMS = [
    "δ᾽",      # δέ
    "κατ᾽",    # κατά
    "μετ᾽",    # μετά
    "μηδ᾽",    # μηδέ
    "καθ᾽",    # κατά before a rough breathing
    "παρ᾽",    # παρά
    "δι᾽",     # διά
    "ποτ᾽",    # ποτέ
    "τιν᾽",    # τις
    "πολλ᾽",   # πολύς
    "μ᾽",      # ἐγώ
    "σ᾽",      # σύ
    "κ᾽",      # ἄν
    "᾽ς",      # εἰς, aphaeresis: the koronis leads instead of trails
    "᾽ν",      # ἐν
]

# Same word class, written by sources that use the spacing psili as the
# elision apostrophe. sanitize_form reattaches a LEADING spacing psili
# to its base letter but leaves a trailing one alone, so these reach
# the exporter still carrying it.
SPACING_BREATHING_FORMS = [
    "κατ᾿",
    "μετ᾿",
    "δι᾿",
    "καθ᾿",
]

# Unaccented spellings: the fallback keys the gate exists to reject.
# Every entry here is bare of any mark, spacing or combining, so it tests
# only the half of the rejection the old combining-mark test already got
# right. The fallback keys that kept a spacing koronis cannot be listed
# here, because a form on its own does not say whether its koronis is an
# elision mark: 'κατ᾽' and 'ταυτ᾽' have the same shape and only the first
# is a spelling. Those are in KORONIS_BEARING_REJECTS below, where the
# lemma's other forms supply the context.
UNACCENTED_FORMS = [
    "μη",
    "και",
    "ανθρωπος",
    "λογος",
    "δε",
    "τε",
    "ΑΝΘΡΩΠΟΣ",
    "",
]

# Marked spellings that were never at risk, kept as controls.
ACCENTED_FORMS = [
    "ζωή",
    "λόγος",
    "ἄνθρωπος",
    "ἀλλ᾽",
    "σοφίᾳ",
    "τὸ",
]


@pytest.mark.parametrize("form", KORONIS_ONLY_FORMS)
def test_koronis_only_form_counts_as_marked(form):
    assert has_any_diacritic(nfc(form)) is True


@pytest.mark.parametrize("form", SPACING_BREATHING_FORMS)
def test_spacing_breathing_form_counts_as_marked(form):
    assert has_any_diacritic(nfc(form)) is True


def test_spacing_dasia_counts_as_marked():
    """The dasia does not occur word-finally in today's lookup.db, but
    it is the exact counterpart of the spacing psili and sanitize_form
    already treats the two as a pair, so the gate must not split them."""
    assert has_any_diacritic(SPACING_DASIA + "ρ") is True


@pytest.mark.parametrize("form", ACCENTED_FORMS)
def test_accented_form_still_counts_as_marked(form):
    assert has_any_diacritic(nfc(form)) is True


@pytest.mark.parametrize("form", UNACCENTED_FORMS)
def test_unaccented_form_is_still_rejected(form):
    assert has_any_diacritic(nfc(form)) is False


@pytest.mark.parametrize("form", [
    "τ΄",        # U+0384 GREEK TONOS as the numeral sign keraia
    "ϟ΄",        # likewise, Milesian 90
    "ρ͵",        # U+0375 GREEK LOWER NUMERAL SIGN
    "λογο´ς",    # U+00B4 ACUTE ACCENT, an accent that failed to combine
    "ʼκτων",     # U+02BC MODIFIER LETTER APOSTROPHE
    "’στι",      # U+2019 RIGHT SINGLE QUOTATION MARK
    "δ’",
])
def test_excluded_spacing_characters_do_not_count_as_marked(form):
    """Numeral signs, a stranded acute and quotation-mark apostrophes
    are not evidence that a spelling is accented. Accepting them would
    let the stripped fallback keys back in through a side door."""
    assert has_any_diacritic(nfc(form)) is False


# Rows data/lookup.db holds today whose only mark is a spacing one and
# whose first letter is a bare vowel. Not one is a word: three are
# Milesian numerals written with a koronis where the keraia belongs, two
# are stripped fallback keys that lost the breathing off a vowel-initial
# word, and one is transmission noise that came with a coined lemma.
# Polytonic Greek writes a breathing over every word-initial vowel, and
# neither elision nor aphaeresis can take it away, so a bare vowel at
# the front rules the form out however the rest of it is spelled.
VOWEL_INITIAL_REJECTS = [
    "ε᾽",          # the numeral five, lemmatized to πέντε
    "α᾽",          # the numeral one, under a lemma spelled α
    "ο᾽",          # the numeral seventy, under a lemma spelled ο
    "ημειβετ᾿",    # stripped key of ἠμείβετ᾿, lemma ἀμείβω
    "εντ᾽",        # tail of πευκήεντ᾽, lemma πευκήεις
    "ηωξδ᾽",       # residue, lemma ηωξδε
]


@pytest.mark.parametrize("form", VOWEL_INITIAL_REJECTS)
def test_vowel_initial_spacing_mark_form_is_rejected(form):
    assert has_any_diacritic(nfc(form)) is False


def test_spacing_mark_on_a_non_greek_stem_is_rejected():
    """lookup.db also carries OCR residue in which Latin letters stand
    in for Greek ones ('G?δ᾽'). A spacing mark on a run that does not
    even open on a Greek letter is no evidence of Greek orthography."""
    assert has_any_diacritic(nfc("G?δ᾽")) is False


def test_gate_does_not_accept_every_spacing_modifier():
    """Guard against the lazy repair of accepting Unicode category Sk
    wholesale.

    Swept over every code point in the standard, the gate differs from
    a plain combining-mark test on exactly the three characters named
    in SPACING_DIACRITICS and on nothing else. Note that a handful of
    other Sk characters (U+0385 GREEK DIALYTIKA TONOS, U+1FCD GREEK
    PSILI AND VARIA and the rest of that series) pass either version,
    because their CANONICAL decomposition contains a combining mark;
    that is long-standing behavior and not part of this rule.

    The sweep runs on a consonant stem, because a spacing mark counts
    only on a form that opens on a consonant, the way an elided or
    aphaeresized word does. The vowel stem is swept right after, where
    the gate has to agree with the combining-mark test everywhere.
    """
    def combining_marks_only(s: str) -> bool:
        return any(unicodedata.category(c) == "Mn"
                   for c in unicodedata.normalize("NFD", s))

    delta = {
        cp for cp in range(sys.maxunicode + 1)
        if has_any_diacritic("δ" + chr(cp))
        != combining_marks_only("δ" + chr(cp))
    }
    assert delta == SPACING_DIACRITICS

    on_a_vowel = {
        cp for cp in range(sys.maxunicode + 1)
        if has_any_diacritic("α" + chr(cp))
        != combining_marks_only("α" + chr(cp))
    }
    assert on_a_vowel == set()


def test_sanitized_trailing_breathing_survives_the_gate():
    """The path that made this a defect: build_lookup_db.py runs
    sanitize_form at ingestion, so a trailing combining psili used as
    an apostrophe is already a koronis by the time the exporter runs.
    The gate has to accept what that rewrite produces."""
    raw = "κατ" + COMBINING_PSILI
    assert has_any_diacritic(raw) is True, "raw form carries a combining mark"
    sanitized = sanitize_form(raw)
    assert sanitized == "κατ" + KORONIS
    assert has_any_diacritic(sanitized) is True


def _lookup_db(rows):
    """Build an in-memory lookup.db with the real schema and given rows.

    rows: iterable of (form, lemma, src).
    """
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE lemmas (id INTEGER PRIMARY KEY, text TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE lookup (form TEXT NOT NULL, lemma_id INTEGER NOT NULL, "
        "src TEXT NOT NULL, lang TEXT NOT NULL DEFAULT 'all')"
    )
    lemma_ids: dict[str, int] = {}
    for form, lemma, src in rows:
        if lemma not in lemma_ids:
            lemma_ids[lemma] = len(lemma_ids) + 1
            conn.execute("INSERT INTO lemmas (id, text) VALUES (?, ?)",
                         (lemma_ids[lemma], lemma))
        conn.execute(
            "INSERT INTO lookup (form, lemma_id, src) VALUES (?, ?, ?)",
            (form, lemma_ids[lemma], src),
        )
    conn.commit()
    return conn


def test_select_forms_keeps_elided_and_drops_unaccented():
    """End-to-end over select_forms, which is where the gate is applied.

    The κατά rows below are the ones lookup.db actually stores: a
    polytonic form, an acute-only form, the stripped fallback key, the
    two elided spellings, and the accented and capitalized variants of
    the elided spelling that the corpora also attest. Only the stripped
    fallback key should be dropped.

    κατά cannot show the opposite error, because its elided form carries
    no accent to strip: strip_accents('κατ᾽') is 'κατ᾽' itself, so the
    fallback key and the spelling are one row. What the variants below
    do guard is the mirror-image repair - 'κατ᾽' is also what you get by
    stripping its siblings 'κάτ᾽', 'κὰτ᾽' and 'Κατ᾽', so a rule that
    drops every form equal to a stripped sibling would throw the real
    spelling away. KORONIS_BEARING_REJECTS below covers the too-wide
    direction.
    """
    conn = _lookup_db([
        ("κατὰ", "κατά", "grc"),      # grave: unambiguously polytonic
        ("κατά", "κατά", "grc"),      # acute-only
        ("κατα", "κατά", "grc"),      # stripped fallback key
        ("κατ᾽", "κατά", "grc"),      # elided, spacing koronis
        ("κατ᾿", "κατά", "grc"),      # elided, spacing psili
        ("κάτ᾽", "κατά", "grc"),      # accented variant of the elision
        ("κὰτ᾽", "κατά", "grc"),      # Homeric accented variant
        ("Κατ᾽", "κατά", "grc"),      # sentence-initial capital
    ])
    forms = {f for f, _ in select_forms(conn, "grc")}
    assert forms == {"κατὰ", "κατά", "κατ᾽", "κατ᾿",
                     "κάτ᾽", "κὰτ᾽", "Κατ᾽"}


def test_select_forms_drops_a_wholly_unaccented_lemma():
    """A lemma with no marked form anywhere is not Ancient Greek
    orthography and must not reach the polytonic artifact, elision
    apostrophes or not."""
    conn = _lookup_db([
        ("και", "και", "grc"),
        ("μη", "μη", "grc"),
    ])
    assert select_forms(conn, "grc") == []


# Forms that carry a spacing koronis or spacing psili and are still not
# spellings. Each case gives the rows lookup.db stores for one lemma, the
# form that must not be admitted, and forms of the same lemma that must
# survive alongside it, so a repair cannot pass by dropping the whole
# paradigm. Every row below is in data/lookup.db today, and every
# `rejected` form reaches select_forms' output under a gate that reads
# any spacing koronis as a mark.
#
# There are two ways a form ends up in this shape:
#
#   - A stripped fallback key. build_lookup_db stores one per form for
#     accent-insensitive lookup, and strip_accents removes combining
#     marks only, so the key of an accented elided form keeps its
#     koronis: ταῦτ᾽ -> ταυτ᾽, μάλ᾽ -> μαλ᾽, πεῖθ᾿ -> πειθ᾿.
#   - A Milesian numeral whose keraia was written as a koronis. 'ε᾽' is
#     the numeral five; the corpora lemmatize it to πέντε, which is how
#     it ends up inside a real paradigm.
# Why the three cases marked below are expected failures: the exporter
# cannot tell a stripped key from a real spelling. The key and the
# correct elision of a proclitic have the same shape, and which of the
# two is the spelling is a property of the lemma, not of the string. The
# counts below are from data/form_profile.db, summed over the five
# spellings of the mark (U+2019, U+1FBD, U+1FBF, U+02BC and the ASCII
# apostrophe): κατ’ 28,397 against κάτ’ 4, and ταῦτ’ 4,920 against
# ταυτ’ 4. The wrong member of each pair is rare, not absent, so
# frequency alone does not separate them either.
#
# ποτ᾽ is the case that settles it, because both spellings sit under one
# lemma: lookup.db gives ποτ᾽ and πότ᾽ both to ποτέ, and the corpora
# write ποτ’ 2,719 times against πότ’ 138 - so here the UNACCENTED
# spelling is the correct one, the reverse of ταῦτ᾽. (τοτ᾽ is not a
# second example: it belongs to the enclitic lemma τοτέ while τότ᾽
# belongs to τότε, so those two are different words, not two spellings
# of one.) Dropping a form because its lemma has an
# accented sibling of the same skeleton takes καθ᾽, μετ᾽, μηδ᾽ and ποτ᾽
# with it, which is what the widening exists to keep. The repair belongs
# upstream in build_data.py's _add_lookup, whose accent-stripped key
# keeps the koronis (ταῦτ᾽ -> ταυτ᾽) instead of dropping the elision
# mark along with the accent it stood beside.
NOT_SEPARABLE_AT_THE_GATE = pytest.mark.xfail(
    strict=True,
    reason="a stripped fallback key that kept its elision mark has the "
           "same shape as a real elided proclitic; see the comment above",
)

KORONIS_BEARING_REJECTS = [
    pytest.param(
        "οὗτος",
        ["οὗτος", "ουτος", "ταῦτα", "ταύτῃ", "ταυτη",
         "ταῦτ᾽", "ταῦθ᾽", "τοῦτ᾽", "ταυτ᾽"],
        "ταυτ᾽",
        ["ταῦτ᾽", "ταῦθ᾽", "τοῦτ᾽", "ταῦτα"],
        id="stripped-key-of-an-accented-elision",
        marks=NOT_SEPARABLE_AT_THE_GATE,
    ),
    pytest.param(
        "μάλα",
        ["μάλα", "μαλα", "μὰλα", "μάλ᾽", "μάλισθ᾽", "μαλ᾽"],
        "μαλ᾽",
        ["μάλ᾽", "μάλισθ᾽", "μάλα"],
        id="stripped-key-of-another-accented-elision",
        marks=NOT_SEPARABLE_AT_THE_GATE,
    ),
    pytest.param(
        # The spacing psili half of the same rule: a source that writes
        # elision with U+1FBF leaves it on the stripped key too.
        "πείθω",
        ["πείθω", "πειθω", "πεῖθ᾿", "πείθ᾿", "πειθ᾿"],
        "πειθ᾿",
        ["πεῖθ᾿", "πείθ᾿", "πείθω"],
        id="stripped-key-that-kept-a-spacing-psili",
        marks=NOT_SEPARABLE_AT_THE_GATE,
    ),
    pytest.param(
        # 'πέντἐ' is the only polytonic row this lemma has - epsilon with
        # a combining psili, a text-transmission artifact - and it is the
        # reason the whole πέντε paradigm, numeral included, is in the
        # polytonic artifact at all.
        "πέντε",
        ["πέντε", "πεντε", "πέντἐ", "πέντ᾽", "πένθ᾽", "ε᾽"],
        "ε᾽",
        ["πέντε", "πέντ᾽", "πένθ᾽"],
        id="milesian-numeral-written-with-a-koronis",
    ),
]


@pytest.mark.parametrize("lemma,rows,rejected,kept", KORONIS_BEARING_REJECTS)
def test_koronis_bearing_non_spelling_is_not_admitted(
        lemma, rows, rejected, kept):
    """A koronis is not by itself evidence that a form is a spelling.

    These are checked through select_forms because being kept out of the
    artifact is what matters; whether the decision is made in
    has_any_diacritic or when the lemma's forms are gathered is up to the
    exporter.
    """
    conn = _lookup_db([(form, lemma, "grc") for form in rows])
    admitted = {f for f, _ in select_forms(conn, "grc")}
    assert rejected not in admitted
    assert set(kept) <= admitted

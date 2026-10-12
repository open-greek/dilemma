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
    canonicalize_final_elision,
    has_editorial_sigla,
    resolve_editorial_form,
    sanitize_form,
)
from export_hunspell import (
    AG_FUNCTION_WORDS,
    AG_EXPORT_OVERRIDES,
    BARE_ELISION_STEMS,
    FormProfileEvidence,
    GRC_CLOSED_LIST_FORMS,
    SPACING_DIACRITICS,
    add_contextual_acute_twins,
    add_grc_reviewed_forms,
    build_sfx_rules,
    exact_form_key,
    filter_by_lemma_freq,
    filter_dominated_spelling_variants,
    filter_grc_orthography,
    filter_new_tonos_structural_forms,
    filter_unattested_new_forms,
    finalize_grc_pairs,
    grc_orthography_reason,
    grc_pair_orthography_reason,
    has_any_diacritic,
    has_required_initial_breathing,
    is_bare_elision_fallback,
    is_productive_second_accent,
    is_proparoxytone_or_properispomenon,
    load_lm_head_required_forms,
    load_unsupported_bare_elisions,
    mark_skeleton,
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


def test_final_elision_canonicalization_does_not_fold_numeral_prime():
    assert canonicalize_final_elision("μηδ’") == "μηδ᾽"
    assert canonicalize_final_elision("μεθʼ") == "μεθ᾽"
    assert canonicalize_final_elision("αʹ") == "αʹ"


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


def test_select_forms_includes_language_shared_headwords_owned_by_el():
    conn = _lookup_db([
        ("λέγω", "λέγω", "el"),
        ("λέγει", "λέγω", "grc"),
        ("γῆ", "γῆ", "el"),
        ("γῆς", "γῆ", "grc"),
    ])

    admitted = set(select_forms(
        conn,
        "grc",
        attestation_freq={
            exact_form_key("λέγω"): 100,
            exact_form_key("γῆ"): 100,
        },
    ))

    assert ("λέγω", "λέγω") in admitted
    assert ("γῆ", "γῆ") in admitted


def test_select_forms_rejects_el_only_language_shared_lemma():
    conn = _lookup_db([
        ("επικοινωνεί", "επικοινωνώ", "el"),
        ("ηπειρώτικου", "ηπειρώτικος", "el"),
    ])

    admitted = select_forms(
        conn,
        "grc",
        attestation_freq={
            exact_form_key("επικοινωνεί"): 100,
            exact_form_key("ηπειρώτικου"): 100,
        },
    )

    assert admitted == []


@pytest.mark.skipif(
    not __import__("export_hunspell").LOOKUP_DB.exists(),
    reason="lookup.db not downloaded",
)
def test_lookup_db_files_the_relative_graves_under_the_relative():
    from export_hunspell import LOOKUP_DB
    conn = sqlite3.connect(f"file:{LOOKUP_DB}?mode=ro", uri=True)
    try:
        rows = dict(conn.execute(
            "SELECT k.form, l.text FROM lookup k JOIN lemmas l "
            "ON l.id = k.lemma_id WHERE k.lang = 'all' AND k.form IN "
            "('ὃ', 'ἣ', 'οἳ', 'αἳ', 'τὼ', 'τὸ', 'τὴν')"
        ))
    finally:
        conn.close()
    # AGDT's demonstrative ὁ no longer shuts out GLAUx's relative ὅς; the
    # article's own spellings, the dual τὼ among them, stay out.
    assert rows == {"ὃ": "ὅς", "ἣ": "ὅς", "οἳ": "ὅς", "αἳ": "ὅς"}


def test_select_forms_keeps_corpus_attested_acute_only_lemma():
    conn = _lookup_db([
        ("χάρις", "χάρις", "grc"),
        ("χάριτος", "χάρις", "grc"),
        ("χάριτι", "χάρις", "grc"),
        ("χάριν", "χάρις", "grc"),
    ])
    freq = {
        exact_form_key("χάρις"): 10,
        exact_form_key("χάριτος"): 8,
        exact_form_key("χάριτι"): 6,
        exact_form_key("χάριν"): 7,
    }

    selected = select_forms(conn, "grc", attestation_freq=freq)
    admitted = {
        form for form, _lemma in filter_by_lemma_freq(
            selected,
            {"χαρις": 10, "χαριτος": 8, "χαριτι": 6, "χαριν": 7},
            min_lemma_count=3,
            strict_acute_min=1,
            strict_form_freq_map=freq,
        )
    }

    assert admitted == {"χάρις", "χάριτος", "χάριτι", "χάριν"}


def test_select_forms_keeps_sparse_acute_only_grc_paradigm_without_lm_hit():
    conn = _lookup_db([
        ("λύω", "λύω", "grc"),
        ("λύομεν", "λύω", "grc"),
        ("λέλυκα", "λύω", "grc"),
    ])

    profile_freq = {
        exact_form_key("λύω"): 24,
        exact_form_key("λύομεν"): 35,
        exact_form_key("λέλυκα"): 1,
    }
    assert set(select_forms(conn, "grc", attestation_freq=profile_freq)) == {
        ("λύω", "λύω"),
        ("λύομεν", "λύω"),
        ("λέλυκα", "λύω"),
    }


def test_pinned_textbook_cells_do_not_require_per_form_attestation():
    pairs = [("λύω", "λύω"), ("λύοιμι", "λύω"), ("λελύσθαι", "λύω")]
    admitted = filter_by_lemma_freq(
        pairs,
        {"λυω": 100},
        min_lemma_count=3,
        strict_acute_min=1,
        strict_form_freq_map={exact_form_key("λύω"): 24},
        keep_forms={"λύοιμι"},
    )
    # λελύσθαι is neither attested nor pinned.
    assert admitted == [("λύω", "λύω"), ("λύοιμι", "λύω")]


def test_contextual_grave_always_adds_acute_twin():
    pairs, added = add_contextual_acute_twins([
        ("περιτομὴ", "περιτομή"),
        ("Πειραιεὺς", "Πειραιεύς"),
        ("λόγος", "λόγος"),
    ])

    assert added == 2
    assert ("περιτομή", "περιτομή") in pairs
    assert ("Πειραιεύς", "Πειραιεύς") in pairs


def test_reviewed_grave_twin_ignores_the_lemma_dependent_consonant_check():
    # The baseline grave Ὃκ keeps its acute twin although lookup.db files it
    # under a lemma it does not spell; an unreviewed Ὃκ does not.
    assert add_contextual_acute_twins([("Ὃκ", "ὅς")], {"Ὃκ"})[1] == 1
    assert add_contextual_acute_twins([("Ὃκ", "ὅς")])[1] == 0


def test_grc_orthography_rejects_editorial_greek_signs():
    assert grc_orthography_reason("ϲφόδρα") == "nonword_character"
    assert grc_orthography_reason("σφόδρα") is None


def test_contextual_grave_does_not_recreate_an_explicit_reject():
    pairs, added = add_contextual_acute_twins([("στὸ", "εἰς")])

    assert added == 0
    assert ("στό", "εἰς") not in pairs


@pytest.mark.parametrize("form,reason", [
    ("Ἰακώβου,", "nonword_character"),
    ("ὑπάγεις;", "nonword_character"),
    ("ηὕρισκον·", "nonword_character"),
    ("παρέξοὖσαν", "internal_breathing"),
    ("ἀντἐγερθῆτον", "internal_breathing"),
    ("ἄνιοιμι", "accent_before_antepenult"),
    ("αὐτου", "explicit_reject"),
    ("τού", "explicit_reject"),
    ("στὸ", None),
    ("τῷν", "explicit_reject"),
    ("πᾶντα", "explicit_reject"),
    ("Θεόσ", "final_nonfinal_sigma"),
    ("Αἰάντεσσ", "final_nonfinal_sigma"),
    ("Αἰολίδ", None),
    ("κάί", "misplaced_second_accent"),
    ("μὴτ", "bare_elision"),
])
def test_grc_orthography_rejects_reported_export_junk(form, reason):
    assert grc_orthography_reason(form) == reason


def test_pair_filter_rejects_truncated_final_consonant_but_keeps_headword():
    assert grc_pair_orthography_reason("Αἰολίδ", "Αἰολίς") == (
        "impossible_final_consonant"
    )
    assert grc_pair_orthography_reason("Ἰακώβ", "Ἰακώβ") is None
    assert grc_pair_orthography_reason("κὰτ", "κατά") is None


@pytest.mark.parametrize("form", [
    "κἀγώ", "τοὔνομα", "αὐτός", "λέλυκα", "θάλασσάν", "γέγονέν",
    "βέλεσσιν", "περιτομή", "περιτομὴ", "δ᾽",
])
def test_grc_orthography_keeps_valid_compounds_and_sparse_forms(form):
    assert grc_orthography_reason(form) is None


@pytest.mark.parametrize("form,host", [
    ("θάλασσάν", "θάλασσαν"),
    ("λέγουσίν", "λέγουσιν"),
    ("γέγονέν", "γέγονεν"),
])
def test_productive_enclitic_second_accent_is_derived_from_attested_host(
    form, host
):
    assert is_productive_second_accent(
        form, {exact_form_key(host): 100}
    )


@pytest.mark.parametrize("form,host", [
    ("λύκοί", "λύκοι"),      # paroxytone host
    ("ὕπνῴ", "ὕπνῳ"),        # paroxytone host
    ("καλός", "καλος"),      # one accent only
])
def test_second_accent_needs_proparoxytone_or_properispomenon_host(
    form, host
):
    assert not is_productive_second_accent(
        form, {exact_form_key(host): 100}
    )


def test_properispomenon_host_takes_a_second_accent():
    assert is_productive_second_accent(
        "δῶρόν", {exact_form_key("δῶρον"): 100}
    )
    assert is_proparoxytone_or_properispomenon("ἄνθρωπος")
    assert is_proparoxytone_or_properispomenon("σῶμα")
    assert not is_proparoxytone_or_properispomenon("λόγος")
    assert not is_proparoxytone_or_properispomenon("καλός")


def test_mark_skeleton_folds_every_mark_but_keeps_elision():
    assert mark_skeleton("ἑγώ") == mark_skeleton("ἐγὼ") == "εγω"
    assert mark_skeleton("ταΐς") == mark_skeleton("ταῖς") == "ταις"
    assert mark_skeleton("τή") == mark_skeleton("τῇ") == "τη"
    assert mark_skeleton("Ἀπόλλων᾽") != mark_skeleton("Ἀπόλλων")


def test_dominated_new_respelling_is_rejected_but_baseline_is_kept():
    pairs = [
        ("τών", "ὁ"),
        ("ἑγώ", "ἐγώ"),
        ("ταΐς", "ὁ"),
        ("τᾷ", "ὁ"),
        ("θάλασσάν", "θάλασσα"),
        ("κάλως", "καλῶς"),
        ("Τίμων", "Τίμων"),
    ]
    exact = {
        exact_form_key("τῶν"): 580_000,
        exact_form_key("τών"): 200,
        exact_form_key("ἐγώ"): 32_000,
        exact_form_key("ἑγώ"): 54,
        exact_form_key("ταῖς"): 56_000,
        exact_form_key("ταΐς"): 10,
        exact_form_key("τά"): 356_000,
        exact_form_key("τᾷ"): 910,
        exact_form_key("θάλασσαν"): 3_000,
        exact_form_key("θάλασσάν"): 1,
        exact_form_key("καλῶς"): 12_000,
        exact_form_key("κάλως"): 1,
        exact_form_key("τιμῶν"): 854,
        exact_form_key("Τίμων"): 158,
    }
    dominant = {
        "των": 580_000, "εγω": 32_000, "ταις": 56_000, "τα": 356_000,
        "θαλασσαν": 3_000, "καλως": 12_000, "τιμων": 854,
    }
    treebank = {
        exact_form_key("τών"): 2,
        exact_form_key("ἑγώ"): 1,
        exact_form_key("τᾷ"): 731,
        exact_form_key("Τίμων"): 103,
    }
    kept, rejected = filter_dominated_spelling_variants(
        pairs, exact, dominant, treebank, {"κάλως"}
    )
    assert {form for form, _lemma in rejected} == {"τών", "ἑγώ", "ταΐς"}
    # Doric τᾷ has treebank support; θάλασσάν is a productive second accent;
    # κάλως is reviewed shipped surface; Τίμων holds a large share.
    assert {form for form, _lemma in kept} == {
        "τᾷ", "θάλασσάν", "κάλως", "Τίμων",
    }


def test_unattested_new_generator_forms_need_an_explicit_evidence_class():
    pairs = [
        ("λύοιμι", "λύω"),
        ("κεκοινωεἴης", "κοινόω"),
        ("θάλασσάν", "θάλασσα"),
        ("κάλως", "καλῶς"),
    ]
    exact = {
        exact_form_key("θάλασσαν"): 100,
    }
    kept, rejected = filter_unattested_new_forms(
        pairs,
        exact,
        compatibility_forms={"κάλως"},
        protected_forms={"λύοιμι"},
    )

    assert ("κεκοινωεἴης", "κοινόω") in rejected
    assert {form for form, _lemma in kept} == {
        "λύοιμι", "θάλασσάν", "κάλως",
    }


def test_unaccented_ag_exceptions_are_a_closed_function_word_list():
    expected = {
        "τε", "γε", "τις", "τι", "τινος", "τινι", "τινα", "τινε",
        "τινοιν", "τινες", "τινων", "τισι", "τισιν", "τινας",
        "φημι", "φησι", "φησιν", "φαμεν", "φατε", "φασι", "φασιν",
        "μιν", "νιν", "σφε", "σφι", "σφιν", "σφων", "σφας",
        "σφισι", "σφισιν", "πω", "πη", "ποθι", "ποθεν", "νυν",
        "νυ", "θην", "κε", "ποτε", "που", "πως", "περ", "τοι",
        "κεν",
    }
    assert expected <= set(AG_FUNCTION_WORDS)
    assert not any(has_any_diacritic(form) for form in expected)
    assert {"εἰμι", "ἐστι", "ἐστιν", "εἰσι", "εἰσιν"} <= set(
        AG_FUNCTION_WORDS
    )
    assert {"οὐκ", "οὐχ"} <= set(AG_FUNCTION_WORDS)
    assert AG_FUNCTION_WORDS["του"] == "τις"
    assert AG_FUNCTION_WORDS["τῳ"] == "τις"


def test_export_overrides_are_narrow_runtime_resolved_forms():
    assert {
        "τ᾽": "τε",
        "μεθ᾽": "μετά",
        "ὐπό": "ὑπό",
        "εἶνε": "εἵνω",
        "μαῦρον": "μαυρός",
        "μαῦροι": "μαυρός",
        "ἦτο": "εἰμί",
        "ἑκατέρως": "ἑκάτερος",
    }.items() <= AG_EXPORT_OVERRIDES.items()
    assert AG_EXPORT_OVERRIDES["ἀγαποῦσα"] == "ἀγαπάω"
    assert AG_EXPORT_OVERRIDES["τοιονδί"] == "τοιόσδε"
    assert AG_EXPORT_OVERRIDES["λυχνίας"] == "λυχνία"
    assert AG_EXPORT_OVERRIDES["πληρέστατα"] == "πλήρης"
    assert len(AG_EXPORT_OVERRIDES) == 25


def test_exact_form_key_preserves_iota_subscript_and_folds_elision():
    assert exact_form_key("Τῼ") == "τῳ"
    assert exact_form_key("τῳ") != exact_form_key("τωι")
    assert exact_form_key("μηδ’") == exact_form_key("μηδ᾽")


@pytest.mark.parametrize("form", [
    "ἀνήρ", "ἄνθρωπος", "αὐτός", "εἰμί", "οὐ", "ηὐλόγει",
    "ῥήτωρ", "Ῥώμη", "λόγος", "᾽στι", "τε",
])
def test_grc_initial_breathing_accepts_well_formed_words(form):
    assert has_required_initial_breathing(form)


@pytest.mark.parametrize("form", [
    "ανήρ", "άνθρωπος", "αυτός", "είναι", "ότι", "εγώ", "ήταν",
    "αλλά", "Ρώμη", "ρητωρ", "ηπειρώτικου", "Οκτωβρίου",
])
def test_grc_initial_breathing_rejects_unmarked_vowels_and_rho(form):
    assert not has_required_initial_breathing(form)


@pytest.mark.parametrize("form", ["στὸ", "στὰ", "ῥήτωρ", "τοὔνομα", "κἀγώ"])
def test_grc_orthography_keeps_contextual_graves_and_initial_crasis(form):
    assert grc_orthography_reason(form) is None


@pytest.mark.parametrize("form", [
    "θάλασσάν", "δῶρόν", "Σωτῆρί", "σήμερόν",       # before an enclitic
    "Αἴγυπτόνδε", "τοῖόνδε", "οἷοίπερ", "ὁκοῖόντι",  # fused enclitic
])
def test_grc_orthography_keeps_enclitic_second_accents(form):
    assert grc_orthography_reason(form) is None


@pytest.mark.parametrize("form", [
    "λύκοί", "κάί", "Ἀριστοτέλούς",      # paroxytone host
    "τήντῶν", "πρότοῦ", "ἀνάμέσον",      # words run together
    "τῇδὲ", "ἀκόῦσαί",                   # a grave, three accents
])
def test_grc_orthography_rejects_other_second_accents(form):
    assert grc_orthography_reason(form) == "misplaced_second_accent"


@pytest.mark.parametrize("form", [
    # accentless proclitics, dialect enclitics, proclitic crasis
    "ἁ", "αἰ", "εἰν", "ἐντι", "κἀν", "κοὐκ", "χὠ", "τἀν",
    # a fused enclitic keeps the host's accent
    "οὗτινος", "ᾧτινι", "ὧντινων", "οἷστισιν", "τοῖσιδε", "τοῖσδεσσι",
    # crasis with a later coronis or with vocative ὦ
    "ἐγᾦμαι", "μέντἄν", "καλοκἀγαθίας", "ταὧς", "ὦνθρωπε", "ὦγαθέ",
    # iota adscript after a long vowel
    "ζῶια", "τῆιδε", "ῥάιδιον", "ζώιωι",
])
def test_grc_orthography_keeps_dialect_crasis_and_adscript_spellings(form):
    assert grc_orthography_reason(form) is None


@pytest.mark.parametrize("form", [
    # Latin v written ου between vowels is a consonant, not a syllable
    "Ὀκτάουιος", "Ὀκτάουιον", "Ὀκτάουϊος", "Ἀριόουιστος", "Ἀριόουιστον",
    "Ῥάουεννα", "Ῥάουενναν",
    # fused -περ, and the -δε of the epic and Ionic datives of ὅδε
    "οἷονπερ", "οἷοσπερ", "οἷσιπερ", "τῇσιδε", "τοιῇσιδε",
    # crasis with ὦ, and inside ταὧς
    "ὦλεθρε", "ὦρνιθες", "ταὧν", "ταὧνι", "ταὧσι",
    # the Ionic enclitic dative of τις
    "τεῳ",
])
def test_grc_orthography_keeps_attested_forms_the_rules_misread(form):
    # All eighteen are attested, and all were dropped from 1.3.6 by these
    # rules; the keyboard's gate lists them as words it must accept.
    assert grc_orthography_reason(form) is None


@pytest.mark.parametrize("form,reason", [
    # ου that is not between two vowels is still a syllable
    ("ἄκουουσι", "accent_before_antepenult"),
    # a circumflex too far back is still wrong without the crasis ὦ
    ("σῶματος", "circumflex_before_penult"),
    # only the ὅδε datives keep one accent before -δε
    ("ἄνθρωπουδε", "accent_before_antepenult"),
])
def test_grc_orthography_allowances_stay_narrow(form, reason):
    assert grc_orthography_reason(form) == reason


def test_locative_de_needs_the_second_accent():
    # The locative takes the enclitic's second accent; only the listed
    # demonstratives of ὅδε keep one accent before a fused -δε.
    assert grc_orthography_reason("πόλεμόνδε") is None
    assert grc_orthography_reason("πόλεμονδε") == "accent_before_antepenult"


@pytest.mark.parametrize("form", ["ὑπέκ", "διέκ", "χερουβίμ", "ἐφούδ"])
def test_consonant_final_closed_list_survives_the_fragment_rule(form):
    kept, rejected = filter_new_tonos_structural_forms(
        [(form, form)], compatibility_forms=set(),
        protected_forms=set(GRC_CLOSED_LIST_FORMS),
    )
    assert rejected == []
    assert grc_pair_orthography_reason(form, form) is None


@pytest.mark.parametrize("form", ["γὰρ᾽", "ἂλλ᾽", "μὴ᾽"])
def test_grc_orthography_rejects_grave_on_elided_word(form):
    assert grc_orthography_reason(form) == "grave_on_elided_word"


@pytest.mark.parametrize("form,lemma", [
    ("ἔκ", "ἐκ"), ("παρέκ", "παρέκ"), ("παρὲκ", "παρέκ"),
])
def test_consonant_final_complete_words_are_kept(form, lemma):
    assert grc_pair_orthography_reason(form, lemma) is None
    kept, rejected = filter_new_tonos_structural_forms(
        [(form, lemma)], compatibility_forms=set(),
        protected_forms=set(GRC_CLOSED_LIST_FORMS),
    )
    assert kept == [(form, lemma)] and rejected == []


def test_reviewed_lm_head_is_pinned():
    forms = load_lm_head_required_forms()
    assert len(forms) == 975
    assert {"του", "τώρα"} <= forms
    # The polytonic Modern article and the anastrophe accent of ἐκ are
    # required. The Diorisis macron artifact δῑ is gone from the LM since
    # its retraining, and the Modern Greek στὴν, which the LM writes only
    # with the grave, is a reviewed nonword, so στήν is not pinned.
    assert {"τή", "ἔκ"} <= forms
    assert "δῑ" not in forms and "στήν" not in forms
    # A grave whose acute twin is a reviewed nonword pins nothing.
    assert "γιά" not in forms and "θά" not in forms


@pytest.mark.parametrize("form", ["καὶτοὺς", "ὓστερον", "ἀποθνῂσκει"])
def test_grc_orthography_rejects_grave_before_ultima(form):
    assert grc_orthography_reason(form) == "grave_before_ultima"


@pytest.mark.parametrize("form", ["άὐτοῦ", "ἀντἐγερθῆτον", "δόκανα᾽"])
def test_grc_orthography_rejects_joined_or_overlong_accent_forms(form):
    assert grc_orthography_reason(form) is not None


def test_grc_orthography_filter_reports_rejected_pairs():
    kept, rejected = filter_grc_orthography([
        ("αὐτός", "αὐτός"),
        ("αυτός", "αὐτός"),
        ("ῥήτωρ", "ῥήτωρ"),
        ("ρητωρ", "ῥήτωρ"),
        ("λόγος", "λόγος"),
        ("τῳ", "τις"),
    ])

    assert kept == [("αὐτός", "αὐτός"), ("ῥήτωρ", "ῥήτωρ"),
                    ("λόγος", "λόγος"), ("τῳ", "τις")]
    assert rejected == [
        ("αυτός", "αὐτός"), ("ρητωρ", "ῥήτωρ")
    ]


def test_new_tonos_structural_filter_preserves_reviewed_forms_only():
    kept, rejected = filter_new_tonos_structural_forms(
        [("βηθλεέμ", "βηθλεέμ"), ("ὐπέρ", "ὐπέρ"),
         ("κὰτ", "κατά"), ("λόγος", "λόγος")],
        compatibility_forms={"βηθλεέμ"},
        protected_forms=set(),
    )
    assert kept == [("βηθλεέμ", "βηθλεέμ"), ("κὰτ", "κατά"),
                    ("λόγος", "λόγος")]
    assert rejected == [("ὐπέρ", "ὐπέρ")]


def _evidence(exact, dominant=None, treebank=None):
    return FormProfileEvidence(
        {exact_form_key(form): count for form, count in exact.items()},
        dominant or {},
        {exact_form_key(form): count for form, count in (treebank or {}).items()},
        {},
    )


def test_finalize_grc_pairs_applies_every_rule_in_release_order():
    evidence = _evidence(
        exact={
            "λέλυκα": 1, "ἐγώ": 32_000, "ἑγώ": 54, "τᾷ": 910,
            "περιτομὴ": 3, "γ᾽": 5_000,
        },
        dominant={"εγω": 32_000, "τα": 356_000},
        treebank={"ἑγώ": 1, "τᾷ": 731},
    )
    pairs = [
        ("λέλυκα", "λύω"),            # sparse but attested
        ("κεκοινωεἴης", "κοινόω"),    # generator join
        ("λύοιμι", "λύω"),            # unattested, pinned textbook cell
        ("λυθείημεν", "λύω"),         # unattested, not pinned
        ("ἐγώ", "ἐγώ"), ("ἑγώ", "ἐγώ"),  # weak respelling
        ("τᾷ", "ὁ"),                  # dialect spelling with treebank support
        ("περιτομὴ", "περιτομή"),     # grave: gains its acute twin
        ("γ᾽", "γε"),                 # new elided form
        ("χἠ", "καί"),                # closed list, unaccented crasis
        ("ἁδελφὸς", "ἁδελφὸς"),       # reviewed baseline grave
        ("καὶτοὺς", "καί"),           # joined words
    ]
    compatibility = {"ἁδελφὸς"}
    kept, report = finalize_grc_pairs(
        pairs,
        evidence=evidence,
        compatibility_forms=compatibility,
        textbook_forms={"λύοιμι"},
        export_overrides=GRC_CLOSED_LIST_FORMS,
        protected_forms={"λύοιμι"} | set(GRC_CLOSED_LIST_FORMS),
    )
    forms = {form for form, _lemma in kept}
    assert {"λέλυκα", "λύοιμι", "ἐγώ", "τᾷ", "περιτομὴ", "περιτομή",
            "γ᾽", "χἠ", "ἁδελφὸς", "ἁδελφός"} == forms
    assert report["invalid"] == {
        "grave_before_ultima": 1, "internal_breathing": 1,
    }
    assert report["unattested"] == 1
    assert report["dominated"] == 1
    # The twin of a reviewed baseline grave inherits the review.
    assert report["acute_twins"] == 2


def test_a_reviewed_reject_goes_with_its_twin_whatever_protected_it():
    from export_hunspell import filter_reviewed_rejects
    pairs = [("στρατηγού", "στρατηγός"), ("στρατηγοὺ", "στρατηγός"),
             ("στρατηγοῦ", "στρατηγός"), ("ἁδελφὸς", "ἁδελφὸς")]
    kept, rejected = filter_reviewed_rejects(pairs, frozenset({"στρατηγού"}))
    assert [form for form, _ in kept] == ["στρατηγοῦ", "ἁδελφὸς"]
    assert len(rejected) == 2
    # Through the finalizer, a reviewed reject goes even from the baseline.
    evidence = _evidence(exact={"ἁδελφὸς": 3})
    kept, report = finalize_grc_pairs(
        [("ἁδελφὸς", "ἁδελφὸς")],
        evidence=evidence,
        compatibility_forms={"ἁδελφὸς"},
        textbook_forms=set(),
        export_overrides={},
        protected_forms=set(),
        reviewed_rejects=frozenset({"ἁδελφός"}),
    )
    assert kept == []
    assert report["reviewed"] == 2


def test_polytonic_modern_greek_particles_are_accepted():
    # θὰ and γιὰ have no lemma source; pre-1982 polytonic writes them so.
    for form in ("θὰ", "γιὰ"):
        assert form in GRC_CLOSED_LIST_FORMS
        assert grc_orthography_reason(form) is None
    # Their acute twins are the monotonic misspellings and stay rejected.
    from export_hunspell import add_contextual_acute_twins
    pairs, added = add_contextual_acute_twins([("θὰ", "θα"), ("γιὰ", "γιά")])
    assert added == 0 and len(pairs) == 2
    for form in ("θά", "γιά"):
        assert grc_orthography_reason(form) == "explicit_reject"


def test_relative_pronoun_graves_come_in_with_their_acute_twins():
    # The closed list carries these whatever lookup.db holds.
    relatives = {"ὃ", "ἣ", "οἳ", "αἳ"}
    assert {AG_FUNCTION_WORDS[form] for form in relatives} == {"ὅς"}
    assert AG_FUNCTION_WORDS["τὼ"] == "ὁ"
    # An accented ὅ, ἥ, οἵ or αἵ is the relative, not the proclitic article.
    assert not any(
        AG_FUNCTION_WORDS.get(form) == "ὁ"
        for form in relatives | {"ὅ", "ἥ", "οἵ", "αἵ"}
    )
    pairs, _added = add_grc_reviewed_forms(
        [], GRC_CLOSED_LIST_FORMS, {}, set()
    )
    kept, _report = finalize_grc_pairs(
        pairs,
        evidence=_evidence(exact={}),
        compatibility_forms=set(),
        textbook_forms=set(),
        export_overrides=GRC_CLOSED_LIST_FORMS,
        protected_forms=set(GRC_CLOSED_LIST_FORMS),
    )
    forms = {form for form, _lemma in kept}
    assert relatives | {"ὅ", "ἥ", "οἵ", "αἵ", "τὼ", "τώ"} <= forms


def test_iota_subscript_crasis_spellings_come_in_through_the_closed_list():
    from export_hunspell import GRC_IOTA_SUBSCRIPT_CRASIS
    forms = set(GRC_IOTA_SUBSCRIPT_CRASIS) | {"κᾀκ", "κᾀξ"}
    assert {"κᾂν", "κᾀπὶ", "κᾀπειδὰν", "κᾄπειτ᾽", "κᾀκ", "κᾀξ"} <= forms
    for form in forms:
        assert form == unicodedata.normalize("NFC", form)
        # κ, then a vowel carrying the crasis breathing and an iota subscript.
        nfd = unicodedata.normalize("NFD", form)
        assert nfd[0] == "κ" and "\u0313" in nfd and "\u0345" in nfd
        assert form in GRC_CLOSED_LIST_FORMS
    # The unaccented proclitic crasis stays with κἀκ and κἀξ.
    assert AG_FUNCTION_WORDS["κᾀκ"] == AG_FUNCTION_WORDS["κἀκ"] == "καί"
    pairs, _added = add_grc_reviewed_forms([], GRC_CLOSED_LIST_FORMS, {}, set())
    kept, _report = finalize_grc_pairs(
        pairs,
        evidence=_evidence(exact={}),
        compatibility_forms=set(),
        textbook_forms=set(),
        export_overrides=GRC_CLOSED_LIST_FORMS,
        protected_forms=set(GRC_CLOSED_LIST_FORMS),
    )
    out = {form for form, _lemma in kept}
    assert forms | {"κᾀπί"} <= out


def test_the_spelling_review_is_checked_when_loaded(tmp_path):
    import json
    from export_hunspell import load_grc_spelling_review
    assert load_grc_spelling_review(tmp_path / "absent.json") == frozenset()
    def review(rows):
        path = tmp_path / "review.json"
        path.write_text(json.dumps({"schema_version": 1, "reviews": rows},
                                   ensure_ascii=False), encoding="utf-8")
        return path
    ok = review([{"form": "στρατηγού", "decision": "reject"}])
    assert load_grc_spelling_review(ok) == frozenset({"στρατηγού"})
    for rows in (
        [{"form": "βίω", "decision": "reject"}] * 2,
        [{"form": unicodedata.normalize("NFD", "βίω"), "decision": "reject"}],
        [{"form": "βίω", "decision": "keep"}],
    ):
        with pytest.raises(ValueError):
            load_grc_spelling_review(review(rows))


def test_add_grc_reviewed_forms_adds_baseline_forms_only_when_absent():
    pairs, added = add_grc_reviewed_forms(
        [("λόγος", "λόγος")],
        {"τε": "τε"},
        {"λύω": ["λύω", "λύοιμι"]},
        {"λόγος", "ἄνθρωπος"},
    )
    assert pairs == [
        ("λόγος", "λόγος"), ("τε", "τε"), ("λύω", "λύω"),
        ("λύοιμι", "λύω"), ("ἄνθρωπος", "ἄνθρωπος"),
    ]
    assert added == {"overrides": 1, "textbook": 2, "compatibility": 1}


def test_new_tonos_structural_filter_keeps_new_elided_forms():
    pairs = [("δι᾽", "διά"), ("γ᾽", "γε"), ("θ᾽", "τε"), ("παρ᾽", "παρά")]
    kept, rejected = filter_new_tonos_structural_forms(
        pairs, compatibility_forms=set(), protected_forms=set()
    )
    assert kept == pairs
    assert rejected == []


def test_affix_compression_does_not_accept_bare_elision_stems():
    pairs = [
        ("παρά", "παρά"),
        ("παρὰ", "παρά"),
        ("παρ᾽", "παρά"),
        ("παρ", "παρά"),
        ("κατά", "κατά"),
        ("κατὰ", "κατά"),
        ("κατ᾽", "κατά"),
        ("κατ", "κατά"),
    ]

    aff_blocks, dic_lines = build_sfx_rules(pairs, {})
    emitted_words = {line.split("/", 1)[0].split("\t", 1)[0]
                     for line in dic_lines}

    assert not (emitted_words & BARE_ELISION_STEMS)
    assert not any(" 0 0 ." in block for block in aff_blocks)
    assert {"παρά", "παρὰ", "παρ᾽", "κατά", "κατὰ", "κατ᾽"} <= emitted_words


def test_data_driven_bare_elision_filter_keeps_independent_headword():
    elided_pairs = {
        (exact_form_key("ἀφ"), exact_form_key("ἀπό")),
        (exact_form_key("ταῦτ"), exact_form_key("οὗτος")),
        # ἄν᾽ belongs to ἀνά, not the particle ἄν.
        (exact_form_key("ἄν"), exact_form_key("ἀνά")),
    }

    assert is_bare_elision_fallback("ἀφ", "ἀπό", elided_pairs, set())
    assert is_bare_elision_fallback(
        "ταῦτ", "οὗτος", elided_pairs, set()
    )
    assert not is_bare_elision_fallback(
        "ἄν", "ἄν", elided_pairs, {exact_form_key("ἄν")}
    )


def test_bare_elision_guard_checks_compatibility_after_injection():
    conn = _lookup_db([
        ("Γλαῦκ᾽", "Γλαῦκος", "grc"),
        ("Τληπόλεμ᾽", "Τληπόλεμος", "grc"),
        ("Τληπόλεμ", "Τληπόλεμος", "grc"),
        # These complete homographs have independent lexical owners.
        ("Βηθλέεμ᾽", "Βηθλέεμα", "grc"),
        ("Βηθλέεμ", "Βηθλέεμ", "grc"),
        ("Χάροπ᾽", "Χάροψ", "grc"),
        ("Χάροπ", "Χάροπ", "el"),
        ("Ναΐδ᾽", "Ναίς", "grc"),
        ("Ναῒδ", "Ναΐδ", "grc"),
        ("κάτ᾽", "κατά", "grc"),
        ("κάτ", "κατά", "grc"),
        ("ἄν᾽", "ἀνά", "grc"),
        ("ἄν", "ἄν", "grc"),
        ("Ἰακώβ᾽", "Ἰακώβ", "grc"),
    ])
    keys = load_unsupported_bare_elisions(
        conn, {exact_form_key("Ἰακώβ")})
    assert keys == {"Γλαῦκ", "Τληπόλεμ"}
    # Compatibility injects self-maps even when lookup has no bare row.
    bare = [(form, form) for form in
            ("Γλαῦκ", "Τληπόλεμ", "Βηθλέεμ", "Χάροπ", "Ναΐδ", "κάτ", "ἄν", "Ἰακώβ")]
    marked = [("Γλαῦκ᾽", "Γλαῦκος"), ("Τληπόλεμ᾽", "Τληπόλεμος")]
    kept, report = finalize_grc_pairs(
        bare + marked, evidence=_evidence(exact={"Γλαῦκ": 1}),
        compatibility_forms={form for form, _ in bare + marked},
        textbook_forms=set(), export_overrides={}, protected_forms=set(),
        bare_elision_keys=keys,
    )
    assert {form for form, _ in kept} == {
        "Βηθλέεμ", "Χάροπ", "Ναΐδ", "κάτ", "ἄν", "Ἰακώβ", "Γλαῦκ᾽", "Τληπόλεμ᾽"}
    assert report["bare_elision_rejected"] == [
        ("Γλαῦκ", "Γλαῦκ"), ("Τληπόλεμ", "Τληπόλεμ")]
    _aff, lines = build_sfx_rules(kept, {})
    emitted = {line.split("\t", 1)[0].split("/", 1)[0] for line in lines}
    assert not (emitted & {"Γλαῦκ", "Τληπόλεμ"})


def test_bare_elision_guard_preserves_case_and_contextual_accents():
    conn = _lookup_db([
        ("Θύνοντ᾽", "θύω", "grc"),
        ("Λαμβάνοιτ’", "λαμβάνω", "grc"),
        ("Τοὺτ᾽", "οὗτος", "grc"),
    ])
    keys = load_unsupported_bare_elisions(conn, set())
    assert keys == {"Θύνοντ", "Λαμβάνοιτ", "Τούτ"}
    assert "θύνοντ" not in keys


def test_affix_compression_uses_only_real_forms_as_flagged_bases():
    pairs = [
        ("τι", "τις"), ("τινα", "τις"), ("τινος", "τις"),
        ("μι", "μις"), ("μινα", "μις"), ("μινος", "μις"),
        ("λόγος", "λόγος"), ("λόγου", "λόγος"),
        ("ἄνθρωπος", "ἄνθρωπος"), ("ἀνθρώπου", "ἄνθρωπος"),
    ]

    aff_blocks, dic_lines = build_sfx_rules(pairs, {})
    real_forms = {form for form, _lemma in pairs}
    flagged_bases = {
        line.split("/", 1)[0] for line in dic_lines if "/" in line.split("\t", 1)[0]
    }

    assert aff_blocks
    assert flagged_bases == {"τι", "μι"}
    assert flagged_bases <= real_forms
    assert {"λόγος", "λόγου", "ἄνθρωπος", "ἀνθρώπου"} <= {
        line.split("\t", 1)[0] for line in dic_lines
    }


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


def test_both_treebanks_on_one_spelling_confirm_a_respelling():
    # πρώτω, the dual, has 40 corpus tokens against 4,006 for the dative
    # πρώτῳ, so the raw-count floor asks for 25 treebank tokens and it has
    # 9. GLAUx and Diorisis both annotate it, which the floor cannot see.
    pairs = [
        ("πρώτω", "πρῶτος"),
        ("ἑγώ", "ἐγώ"),
    ]
    exact = {
        exact_form_key("πρώτῳ"): 4_006,
        exact_form_key("πρώτω"): 40,
        exact_form_key("ἐγώ"): 32_000,
        exact_form_key("ἑγώ"): 54,
    }
    dominant = {"πρωτω": 4_006, "εγω": 32_000}
    treebank = {exact_form_key("πρώτω"): 9, exact_form_key("ἑγώ"): 1}
    confirmed = frozenset({exact_form_key("πρώτω")})
    kept, rejected = filter_dominated_spelling_variants(
        pairs, exact, dominant, treebank, set(), None, confirmed,
    )
    assert {form for form, _lemma in kept} == {"πρώτω"}
    assert {form for form, _lemma in rejected} == {"ἑγώ"}


def test_confirmation_must_be_worth_something_against_the_common_word():
    # Both treebanks carry a stray tagging of ὅτι as ὄτι, but 3 tokens
    # against 241,498 is the mis-tagging rate of a very common word, not
    # evidence of a second spelling.
    pairs = [("ὄτι", "ὅτι"), ("πρώτω", "πρῶτος")]
    exact = {
        exact_form_key("ὅτι"): 241_498, exact_form_key("ὄτι"): 617,
        exact_form_key("πρώτῳ"): 4_006, exact_form_key("πρώτω"): 40,
    }
    dominant = {"οτι": 241_498, "πρωτω": 4_006}
    treebank = {exact_form_key("ὄτι"): 3, exact_form_key("πρώτω"): 9}
    confirmed = frozenset({exact_form_key("ὄτι"), exact_form_key("πρώτω")})
    kept, rejected = filter_dominated_spelling_variants(
        pairs, exact, dominant, treebank, set(), None, confirmed,
    )
    assert {form for form, _lemma in kept} == {"πρώτω"}
    assert {form for form, _lemma in rejected} == {"ὄτι"}


def test_confirmation_is_absent_by_default():
    # Without the set, the floor decides on its own, as it did before.
    pairs = [("πρώτω", "πρῶτος")]
    exact = {exact_form_key("πρώτῳ"): 4_006, exact_form_key("πρώτω"): 40}
    kept, rejected = filter_dominated_spelling_variants(
        pairs, exact, {"πρωτω": 4_006}, {exact_form_key("πρώτω"): 9}, set(),
    )
    assert not kept and len(rejected) == 1


# --- treebank spellings no lemma source proposes ----------------------------


def test_treebank_spelling_reasons():
    from export_hunspell import treebank_spelling_reason
    for form in ("σπανίως", "κτήσεις", "ὁρμόν", "Δηριάδη", "ῥόδον",
                 "Ῥώμη", "συνελήμφθη"):
        assert treebank_spelling_reason(form) is None, form
    assert treebank_spelling_reason("ὄντα᾽") == "elided"
    assert treebank_spelling_reason("ΚαὶΤὸ") == "inner_capital"
    assert treebank_spelling_reason("ὑλικῶι") == "iota_adscript"
    assert treebank_spelling_reason("λῠ́ω") == "length_mark"
    assert treebank_spelling_reason("δὔ") == "breathing_after_consonant"
    assert treebank_spelling_reason("κᾂν") == "breathing_after_consonant"
    assert treebank_spelling_reason("τάγαθόν") == "second_accent"
    assert treebank_spelling_reason("τοὺ") == "explicit_reject"
    assert treebank_spelling_reason("ἀλλ") == "bare_elision"


def test_treebank_spellings_are_admitted_on_their_own_count():
    from export_hunspell import add_treebank_spellings, treebank_spelling_key
    counts = {
        treebank_spelling_key("σπανίως"): 271,
        treebank_spelling_key("ὁρμόν"): 13,
        treebank_spelling_key("Δηριάδη"): 5,
        treebank_spelling_key("θυίσκην"): 4,
        treebank_spelling_key("Λόγος"): 40,
        treebank_spelling_key("τάγαθόν"): 6,
    }
    spellings = {
        treebank_spelling_key("σπανίως"): {"σπανίως"},
        treebank_spelling_key("ὁρμόν"): {"ὁρμόν", "ὁρμὸν"},
        treebank_spelling_key("Δηριάδη"): {"Δηριάδη"},
        treebank_spelling_key("θυίσκην"): {"θυίσκην"},
        treebank_spelling_key("Λόγος"): {"Λόγος"},
        treebank_spelling_key("τάγαθόν"): {"τάγαθόν"},
    }
    # A grave counts with its acute, and case is kept.
    assert treebank_spelling_key("ὁρμὸν") == "ὁρμόν"
    assert treebank_spelling_key("Λόγος") != treebank_spelling_key("λόγος")
    existing = [("λόγος", "λόγος"), ("σπανίως", "σπανίως")]
    out, added = add_treebank_spellings(existing, counts, spellings, 5)
    # σπανίως is there already; θυίσκην has 4 tokens; Λόγος is a capitalized
    # λόγος; τάγαθόν carries a second accent.
    assert set(out) - set(existing) == {
        ("ὁρμόν", "ὁρμόν"), ("ὁρμὸν", "ὁρμὸν"), ("Δηριάδη", "Δηριάδη")}
    assert added == 3
    assert add_treebank_spellings(existing, counts, spellings, 0) == (
        existing, 0)


def test_treebank_counts_leave_out_the_held_out_sentences(tmp_path):
    import json
    from collections import Counter
    from export_hunspell import (
        load_heldout_free_treebank_counts, treebank_spelling_key)
    db = tmp_path / "form_profile.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE forms (form_id INTEGER PRIMARY KEY, "
                 "form TEXT NOT NULL, form_norm TEXT NOT NULL)")
    conn.execute("CREATE TABLE form_profile (form_id INTEGER PRIMARY KEY, "
                 "total_count INTEGER, source_counts_json TEXT)")
    rows = [
        ("σπανίως", {"glaux": 9, "diorisis": 4, "pg": 50}),
        ("ὁρμὸν", {"glaux": 2}), ("ὁρμόν", {"diorisis": 4}),
        ("Δηριάδη", {"glaux": 6}),
        ("λόγῳ", {"pg": 30}),
    ]
    for i, (form, sources) in enumerate(rows, 1):
        conn.execute("INSERT INTO forms VALUES (?, ?, ?)", (i, form, form))
        conn.execute("INSERT INTO form_profile VALUES (?, ?, ?)",
                     (i, sum(sources.values()), json.dumps(sources)))
    conn.commit()
    conn.close()
    counts, spellings = load_heldout_free_treebank_counts(
        Counter({"σπανίως": 3, "Δηριάδη": 6}), db)
    # Each treebank loses every held-out occurrence; the larger one counts.
    assert counts[treebank_spelling_key("σπανίως")] == 6
    # GLAUx's grave and Diorisis's acute: the larger treebank, not the sum.
    assert counts[treebank_spelling_key("ὁρμόν")] == 4
    assert spellings[treebank_spelling_key("ὁρμόν")] == {"ὁρμόν", "ὁρμὸν"}
    # All of Δηριάδη's tokens are held out, and the OCR'd Patrologia is no
    # treebank.
    assert treebank_spelling_key("Δηριάδη") not in counts
    assert treebank_spelling_key("λόγῳ") not in counts


def test_select_forms_takes_modern_greek_rows_the_treebanks_attest():
    from export_hunspell import treebank_spelling_key
    conn = _lookup_db([
        ("σπανίως", "σπανίως", "el"),
        ("κτήσεις", "κτήσεις", "el"),
        ("τώρα", "τώρα", "el"),
        ("Βία", "Βία", "el"),
        ("ἑκατέρως", "ἑκατέρως", "el"),
        ("λόγος", "λόγος", "grc"),
    ])
    counts = {treebank_spelling_key(form): n for form, n in (
        ("σπανίως", 271), ("κτήσεις", 129), ("τώρα", 2), ("Βία", 5),
        ("ἑκατέρως", 40))}
    freq = {exact_form_key("λόγος"): 100}
    assert set(select_forms(conn, "grc", attestation_freq=freq)) == {
        ("λόγος", "λόγος")}
    taken = set(select_forms(conn, "grc", attestation_freq=freq,
                             treebank_counts=counts, treebank_min_count=5))
    # τώρα has 2 tokens; the capitalized Βία is left to the treebank-spelling
    # rule; ἑκατέρως keeps the lemma the closed list gives it.
    assert taken == {("λόγος", "λόγος"), ("σπανίως", "σπανίως"),
                     ("κτήσεις", "κτήσεις")}


def test_real_enclitics_and_postclassical_conjunction_are_closed_forms():
    assert {"του": "τις", "τῳ": "τις", "ἐνῷ": "ἐνῷ"}.items() <= GRC_CLOSED_LIST_FORMS.items()
    assert grc_orthography_reason("τῳ") is None


def test_exact_rating_does_not_borrow_from_letters_or_lemma_or_affix():
    pairs = [("χρυσός", "χρυσός"), ("χρῦσος", "χρυσός"),
             ("λόγος", "λόγος"), ("λόγοςα", "λόγος"),
             ("τύπος", "τύπος"), ("τύποςα", "τύπος")]
    aff, lines = build_sfx_rules(
        pairs, {"χρυσος": 200_000, "χρυσός": 2_000, "λόγος": 2000},
        canonical_lemmas={"χρυσός"}, exact_frequency=True)
    assert not aff
    assert "χρῦσος\tfr:X" in lines
    assert "χρυσός\tfr:C" in lines
    assert "λόγοςα\tfr:X" in lines


def test_case_filter_uses_real_ownership_and_exact_case_evidence():
    from export_hunspell import cased_form_key, filter_case_artifacts
    evidence = _evidence({"Σωκράτης": 9000, "ψυχῇ": 2000})
    evidence = evidence._replace(cased={cased_form_key("Σωκράτης"): 9000,
                                        cased_form_key("ψυχῇ"): 2000})
    pairs = [("Πόλῃ", "Πόλῃ"), ("πόλῃ", "πόλις"),
             ("Σωκράτης", "Σωκράτης"), ("σωκράτης", "σωκράτης"),
             ("ψυχῇ", "Ψυχή")]
    kept, rejected = filter_case_artifacts(
        pairs, evidence, {exact_form_key("Πόλῃ"): 2,
                          exact_form_key("Σωκράτης"): 1,
                          exact_form_key("ψυχῇ"): 1})
    assert rejected == [pairs[0], pairs[3]]
    assert kept == [pairs[1], pairs[2], pairs[4]]


def test_source_split_word_junk_cannot_return_through_compatibility():
    pairs = [("γί", "γία"), ("ἐνέρ", "ἐνέρ"), ("μεγί", "μέγας"), ("γία", "γία")]
    kept, _ = finalize_grc_pairs(
        pairs, evidence=_evidence({f: 5000 for f, _ in pairs}),
        compatibility_forms={f for f, _ in pairs}, textbook_forms=set(),
        export_overrides={}, protected_forms=set())
    assert kept == [("γία", "γία")]


def test_builtin_spelling_rejects_override_compatibility_and_grave_twins():
    bad = ["τής", "τὴς", "τού", "τοὺ", "ταίς", "ταὶς",
           "στό", "στά", "θά", "γιά"]
    good = ["τῆς", "τοῦ", "ταῖς", "στὸ", "στὰ", "θὰ", "γιὰ"]
    pairs = [(form, form) for form in bad + good]
    kept, report = finalize_grc_pairs(
        pairs, evidence=_evidence({form: 5000 for form in bad + good}),
        compatibility_forms=set(bad + good), textbook_forms=set(),
        export_overrides={}, protected_forms=set(bad + good))
    assert {form for form, _ in kept} == set(good)
    assert report["reviewed"] >= len(bad)


def test_postclassical_evidence_removes_heldout_and_requires_repetition():
    from export_hunspell import postclassical_admission_reason
    def reason(*args, **kwargs):
        return postclassical_admission_reason(*args, pos="noun", **kwargs)
    assert reason(8, {"pg": 8, "pta": 5}, 3) == "repeated_multiple_sources"
    assert reason(8, {"pg": 8, "first1k": 5}, 3) is None
    assert reason(5, {"pg": 5, "first1k": 1, "oga": 1}, 4) is None
    assert reason(8, {"first1k": 8, "oga": 8}, 3) is None
    assert reason(30, {"pg": 30}, 4) is None
    assert reason(25, {"pta": 25}, 3, heldout_count=5) is None
    assert reason(25, {"pta": 25}, 8, heldout_count=5) == "repeated_multiple_works"
    assert reason(25, {"pta": 25, "pg": 4}, 6) is None
    assert reason(25, {"pta": 25, "pg": 4}, 7) == "repeated_multiple_works"
    assert reason(1, {"pg": 1}, 1, lexical=True) == "attested_lexical_form"
    assert reason(1, {"pg": 1}, 1, lexical=True, heldout_count=1) is None
    assert reason(8, {"pg": 8, "pta": 5}, 3, heldout_count=5) is None


@pytest.mark.parametrize("form,count,sources,works", [
    ("ΚΠερί", 20, {"pg": 20}, 9),
    ("Κτόν", 45, {"pg": 45}, 23),
    ("Στίχ", 327, {"first1k": 327, "oga": 327, "pg": 1}, 12),
    ("κκλησίαν", 5, {"first1k": 5, "pg": 4}, 6),
    ("τεῖται", 20, {"first1k": 11, "glaux": 1, "oga": 12, "pg": 10}, 15),
    ("δικαιοσύ", 13, {"diorisis": 1, "first1k": 4, "oga": 5, "pg": 9, "pta": 1}, 16),
    ("θρώπων", 91, {"diorisis": 3, "first1k": 57, "oga": 60, "pg": 71, "pta": 1}, 75),
    ("Λὐξάνεσθε", 6, {"diorisis": 1, "first1k": 1, "oga": 1, "pg": 6}, 3),
])
def test_shared_transcription_errors_do_not_establish_unknown_words(form, count, sources, works):
    from export_hunspell import postclassical_admission_reason
    assert postclassical_admission_reason(count, sources, works, pos="noun") is None


def test_reviewed_paradigms_derive_cells_instead_of_listing_words():
    from export_hunspell import derive_reviewed_paradigm as derive
    forms = derive({"method": "lookup_paradigm", "prefix": "παν",
                    "remove_initial_breathing": True}, ["ἀμώμητον", "ἀμωμήτῳ"])
    assert forms == {"παναμώμητον", "παναμωμήτῳ"}
    forms = derive({"method": "verb_paradigm", "prefix": "ἀνθ",
                    "remove_initial_breathing": True}, ["ὁπλίσαντες"])
    assert forms == {"ἀνθοπλίσαντες"}
    assert "κατετέθης" in derive({"method": "passive_aorist_indicative", "stem": "κατετέθ"}, [])
    assert "λαμπρυνθῶμεν" in derive({"method": "passive_aorist_subjunctive", "stem": "λαμπρυνθ"}, [])
    forms = derive({"method": "eta_noun", "stem": "πινακοθήκ"}, [])
    assert {"πινακοθήκῃ", "πινακοθηκῶν", "πινακοθῆκαι"} <= forms


def test_exact_rating_compresses_only_equal_bucket_cells():
    pairs = [("βα", "lemma1"), ("βαν", "lemma1"), ("βας", "lemma1"),
             ("γα", "lemma2"), ("γαν", "lemma2"), ("γας", "lemma2")]
    aff, lines = build_sfx_rules(pairs, {"βα": 2000, "βαν": 2000,
                                       "γα": 2000, "γαν": 2000}, exact_frequency=True)
    assert aff
    assert any(line.startswith("βα/") and line.endswith("fr:C") for line in lines)
    assert "βας\tfr:X" in lines
    assert "γας\tfr:X" in lines


def test_exact_rating_keeps_capital_and_lowercase_counts_separate():
    _aff, lines = build_sfx_rules([("Κοίνος", "Κοίνος"), ("κοίνος", "κοίνος")],
                                  {"κοίνος": 2000}, exact_frequency=True)
    assert "Κοίνος\tfr:X" in lines
    assert "κοίνος\tfr:C" in lines


def test_postclassical_loader_uses_lexical_and_independent_corpus_evidence(tmp_path):
    import json
    from collections import Counter
    from export_hunspell import add_postclassical_corpus_forms, lexical_head_key
    profile = tmp_path / "profile.db"
    conn = sqlite3.connect(profile)
    conn.executescript("CREATE TABLE forms(form_id INTEGER, form TEXT);"
                       "CREATE TABLE form_profile(form_id INTEGER,total_count INTEGER,"
                       "source_counts_json TEXT,n_works INTEGER,dominant_pos TEXT DEFAULT 'noun');")
    rows = [("πανάγιον", 8, {"pg": 8, "pta": 5}, 4),
            ("σπανίως", 1, {"pg": 1}, 1),
            ("καθόλου", 1, {"pg": 1}, 1),
            ("σπίτι", 1, {"byzantine_vernacular": 1}, 1),
            ("Παύλοα", 1, {"byzantine_vernacular": 1}, 1),
            ("Σωκράτους", 1, {"byzantine_vernacular": 1}, 1),
            ("λόγους", 7, {"oga": 7, "first1k": 7}, 1),
            ("Ἰωάννς", 1, {"first1k": 1, "oga": 1}, 1),
            ("Ἰσσάκ", 2, {"first1k": 1, "oga": 2, "pta": 1}, 2),
            ("Νατθαῖος", 1, {"glaux": 1}, 1)]
    unclassified = ["διάτοι", "διατοῦτο", "τρῶν", "Ἁλληλουιά", "δὔ"]
    rows.extend((f, 100, {"pta": 100, "oga": 100}, 50) for f in unclassified)
    for i, (form, count, sources, works) in enumerate(rows):
        conn.execute("INSERT INTO forms VALUES(?,?)", (i, form))
        conn.execute("INSERT INTO form_profile(form_id,total_count,source_counts_json,n_works) VALUES(?,?,?,?)",
                     (i, count, json.dumps(sources), works))
    for form in unclassified:
        conn.execute("UPDATE form_profile SET dominant_pos='other' WHERE form_id IN "
                     "(SELECT form_id FROM forms WHERE form=?)", (form,))
    conn.commit(); conn.close()
    lookup = _lookup_db([("σπίτι", "σπίτι", "el"),
                         ("Παύλοα", "Παύλοα", "el"),
                         ("Σωκράτους", "Σωκράτης", "grc"),
                         ("σπανίως", "σπάνιος", "grc"),
                         ("καθόλου", "καθόλου", "grc"),
                         *[(f, f, "grc") for f in ("Ἰωάννς", "Ἰσσάκ", "Νατθαῖος")]])
    out, audit = add_postclassical_corpus_forms([], lookup, Counter({"καθόλου":1}),
                                                profile_path=profile, lexical_heads=frozenset(map(lexical_head_key,
                                                    ["Σωκράτης", "σπάνιος", "καθόλου"])),
                                                vernacular_heads=frozenset({lexical_head_key("σπίτι")}))
    assert {form for form, _ in out} == {"πανάγιον", "σπανίως", "σπίτι", "Σωκράτους"}
    assert {row["reason"] for row in audit} == {"repeated_multiple_sources", "attested_lexical_form"}


def test_new_corpus_capital_without_lexical_owner_cannot_duplicate_lower_word():
    from export_hunspell import cased_form_key, filter_case_artifacts
    evidence = _evidence({"Νέον": 100})
    evidence = evidence._replace(cased={cased_form_key("Νέον"): 100})
    pairs = [("νέον", "νέος"), ("Νέον", "Νέον")]
    kept, rejected = filter_case_artifacts(pairs, evidence, {})
    assert kept == [pairs[0]]
    assert rejected == [pairs[1]]


def test_exact_rating_canonical_admission_does_not_claim_frequency():
    _aff, lines = build_sfx_rules([("ἄγνωστον", "ἄγνωστος")], {},
                                  canonical_forms={"ἄγνωστον"},
                                  canonical_lemmas={"ἄγνωστος"}, exact_frequency=True)
    assert lines == ["ἄγνωστον\tfr:X"]


def test_exact_rating_loader_never_uses_legacy_lowercased_counts(tmp_path, monkeypatch):
    import gzip
    import json
    import export_hunspell as exporter
    monkeypatch.setattr(exporter, "DATA", tmp_path)
    with gzip.open(tmp_path / "hunspell_grc_form_freq.json.gz", "wt") as stream:
        json.dump({"forms": {"λόγος": 1000},
                   "cased_forms": {"Λόγος": 1000, "λόγος": 2}}, stream)
    evidence = _evidence({"λόγος": 5000})._replace(cased={"λόγος": 3})
    assert exporter.load_exact_rating_counts(evidence) == {"Λόγος": 1000, "λόγος": 3}


def test_source_review_admits_only_the_documented_forms():
    from export_hunspell import derive_reviewed_paradigm
    assert derive_reviewed_paradigm(
        {"method": "source_forms", "forms": ["κάρτευς", "ἐπεστονάχιζε"]}, []
    ) == {"κάρτευς", "ἐπεστονάχιζε"}


def test_source_review_normalizes_contextual_grave_to_acute():
    from export_hunspell import derive_reviewed_paradigm
    assert derive_reviewed_paradigm(
        {"method": "source_forms", "forms": ["κεντροβαρὲς", "ὑπερανεστηκὸς"]}, []
    ) == {"κεντροβαρές", "ὑπερανεστηκός"}


def test_reviewed_archimedes_grave_and_acute_survive_export_filters():
    import json
    from export_hunspell import DATA, derive_reviewed_paradigm, finalize_grc_pairs
    specs = json.loads((DATA / "hunspell_grc_postclassical_review.json").read_text())["paradigms"]
    spec = next(row for row in specs if row["lemma"] == "ἀπόχρη")
    assert spec["forms"] == ["ἀποχρὴ"]
    acute = derive_reviewed_paradigm(spec, [])
    assert acute == {"ἀποχρή"}
    pairs, report = finalize_grc_pairs(
        [("ἀποχρὴ", "ἀπόχρη")], evidence=_evidence({"ἀποχρή": 1}),
        compatibility_forms=set(), textbook_forms=acute,
        export_overrides={}, protected_forms=acute)
    assert set(pairs) == {("ἀποχρὴ", "ἀπόχρη"), ("ἀποχρή", "ἀπόχρη")}
    assert report["acute_twins"] == 1


@pytest.mark.parametrize("capital,lower,lemma", [
    ("Σάρκα", "σάρκα", "σάρξ"), ("Σαφής", "σαφής", "σαφής"),
    ("Τύπος", "τύπος", "τύπος"), ("Φράζε", "φράζε", "φράζω"),
    ("Πάρα", "πάρα", "παρά"), ("Τίσι", "τίσι", "τίσις"),
])
def test_shared_modern_self_map_cannot_establish_ancient_name(capital, lower, lemma):
    from export_hunspell import load_case_ownership, filter_case_artifacts
    conn = _lookup_db([(capital, capital, "el"), (lower, lemma, "grc"),
                       ("Σωκράτης", "Σωκράτης", "grc"),
                       ("Ψυχή", "Ψυχή", "grc"), ("ψυχῇ", "ψυχή", "grc")])
    pairs = [(capital, capital), (lower, lemma), ("Σωκράτης", "Σωκράτης"),
             ("ψυχῇ", "ψυχή")]
    kept, rejected = filter_case_artifacts(pairs, _evidence({}), load_case_ownership(conn))
    assert rejected == [(capital, capital)]
    assert kept == pairs[1:]


@pytest.mark.parametrize("form,reason", [
    ("σμγ᾽", "unreviewed_elision_or_numeral"),
    ("εἰπεῖν᾽", "unreviewed_elision_or_numeral"),
    ("ἧμάς", "unreviewed_contextual_accent"),
    ("ἧμίν", "unreviewed_contextual_accent"),
    ("δΰ", "unexplained_diaeresis"),
    ("ἐυρραθάμιγγος", "diphthong_breathing_on_first_vowel"),
    ("εὐρραθάμιγγος", None), ("ἄϋπνος", None),
])
def test_unknown_corpus_context_marks_need_lexical_review(form, reason):
    from export_hunspell import postclassical_spelling_reason
    assert postclassical_spelling_reason(form) == reason


def test_exact_inflection_proof_uses_pinned_source_case_and_marks():
    from export_hunspell import exact_inflection_lemma, lexical_head_key
    heads = frozenset(map(lexical_head_key, ["λόγος", "Σωκράτης", "Ἰωάννς", "Παῦλοα", "λογός"]))
    conn = _lookup_db([("Σωκράτους", "Σωκράτης", "grc"),
                       ("λόγους", "λόγος", "grc"),
                       ("Ἰωάννς", "Ἰωάννς", "grc"),
                       ("ψυχάς", "ψυχή", "el"),
                       ("λογου", "λόγος", "grc"),
                       ("Παύλοα", "Παῦλοα", "grc"),
                       ("λόγος", "λογός", "grc")])
    assert exact_inflection_lemma(conn, "λόγους", "λόγους", lexical_heads=heads) == "λόγος"
    assert exact_inflection_lemma(conn, "Σωκράτους", "Σωκράτους", lexical_heads=heads) == "Σωκράτης"
    for form in ("σωκράτους", "λογους", "λογου", "λόγούς", "Ἰωάννς", "ψυχάς", "Παύλοα", "λόγος"):
        assert exact_inflection_lemma(conn, form, form, lexical_heads=heads) is None


_CORRUPT_CORPUS_INFLECTIONS = [
    ("Μαρκελλλίνου", "Μαρκελλλῖνος"),
    ("συννντρίμματι", "σύννντριμμα"),
    ("συγγράμμματα", "συγγράμμμα"),
    ("αἱμορρραγίην", "αἱμορρραγία"),
    ("ἐλάσσσων", "ἔλασσσον"),
    ("μελισσσάων", "μελίσσσος"),
    ("ἀπορρρέοντα", "ἀπορρρέω"),
    ("ΛΓτῇ", "ΛΓτός"),
    ("ΧΙΙπροθυμίας", "ΧΙΙπροθυμία"),
    ("ζϠκαπρός", "ζϠκαφρός"),
    ("ἀπόΑΕ", "ἀποάω"),
    ("ἐνΘήβαις", "ἐνΘήβης"),
]


@pytest.mark.parametrize("form,lemma", _CORRUPT_CORPUS_INFLECTIONS + [
    ("συγγράμματα", "συγγράμμμα"),
    ("συγγράμμματα", "σύγγραμμα"),
    ("λόγους", "λό γος"),
])
def test_corrupt_lookup_row_is_not_lexical_inflection_evidence(form, lemma):
    from export_hunspell import exact_inflection_lemma, lexical_head_key
    heads = frozenset({lexical_head_key(lemma)})
    conn = _lookup_db([(form, lemma, "grc")])
    assert exact_inflection_lemma(conn, form, form, lexical_heads=heads) is None


def test_corpus_admission_rejects_notation_even_with_repeated_evidence(tmp_path):
    import json
    from collections import Counter
    from export_hunspell import add_postclassical_corpus_forms, lexical_head_key
    good = [("Μαρκελλίνου", "Μαρκελλῖνος"),
            ("συντρίμματι", "σύντριμμα"),
            ("συγγράμματα", "σύγγραμμα"),
            ("αἱμορραγίην", "αἱμορραγία"),
            ("ἐλάσσων", "ἐλάσσων"),
            ("μελισσάων", "μέλισσα"),
            ("ἀπορρέοντα", "ἀπορρέω"),
            ("Βρισηίδος", "Βρισηίς"),
            ("Σωκράτους", "Σωκράτης"),
            ("γία", "γία")]
    rows = _CORRUPT_CORPUS_INFLECTIONS + good
    profile = tmp_path / "profile.db"
    with sqlite3.connect(profile) as conn:
        conn.executescript("CREATE TABLE forms(form_id INTEGER, form TEXT);"
                           "CREATE TABLE form_profile(form_id INTEGER,total_count INTEGER,"
                           "source_counts_json TEXT,n_works INTEGER,dominant_pos TEXT);")
        for i, (form, _lemma) in enumerate(rows):
            conn.execute("INSERT INTO forms VALUES(?,?)", (i, form))
            conn.execute("INSERT INTO form_profile VALUES(?,?,?,?,?)",
                         (i, 100, json.dumps({"glaux": 100, "pta": 100}), 50, "noun"))
    lookup = _lookup_db([(form, lemma, "grc") for form, lemma in rows])
    out, audit = add_postclassical_corpus_forms([], lookup, Counter(), profile_path=profile,
        lexical_heads=frozenset(lexical_head_key(l) for _f, l in rows),
        vernacular_heads=frozenset())
    assert {form for form, _lemma in out} == {form for form, _lemma in good}
    assert len(audit) == len(good)


def test_sparse_inflection_requires_an_independently_defined_head():
    from export_hunspell import exact_inflection_lemma, lexical_head_key
    conn = _lookup_db([("βουλὴντοιάνδε", "βουλήντοιας", "grc"),
                       ("ἀντειπεῖνκαὶ", "ἀντειπεῖνκῃ", "grc"),
                       ("λόγους", "λόγος", "grc"),
                       ("ἀνανήψεις", "ἀνανηψις", "grc")])
    heads = frozenset(map(lexical_head_key, ["λόγος", "ἀνάνηψις"]))
    assert exact_inflection_lemma(conn, "βουλήντοιάνδε", "βουλὴντοιάνδε", lexical_heads=heads) is None
    assert exact_inflection_lemma(conn, "ἀντειπεῖνκαί", "ἀντειπεῖνκαὶ", lexical_heads=heads) is None
    assert exact_inflection_lemma(conn, "λόγους", "λόγους", lexical_heads=heads) == "λόγος"
    assert exact_inflection_lemma(conn, "ἀνανήψεις", "ἀνανήψεις", lexical_heads=heads) == "ἀνανηψις"
    assert lexical_head_key("Λόγος") not in heads
    assert lexical_head_key("λογος") in heads
    assert lexical_head_key("ἀνανήψις") == lexical_head_key("ἀνάνηψις")
    assert lexical_head_key("ἀνάνηψις") != lexical_head_key("ἁνάνηψις")


def test_sparse_inflection_can_follow_an_explicit_contracted_citation():
    from export_hunspell import exact_inflection_lemma, lexical_head_key
    conn = _lookup_db([("καταμετρηθείσης", "καταμετρῶ", "grc")])
    heads = frozenset(map(lexical_head_key, ["καταμετρέω", "καταμετράω"]))
    aliases = {lexical_head_key("καταμετρῶ"): ("καταμετρέω",)}
    assert exact_inflection_lemma(conn, "καταμετρηθείσης", "καταμετρηθείσης",
        lexical_heads=heads, lemma_aliases=aliases) == "καταμετρέω"
    assert exact_inflection_lemma(conn, "καταμετρηθείσης", "καταμετρηθείσης",
        lexical_heads=frozenset(), lemma_aliases=aliases) is None
    aliases[lexical_head_key("καταμετρῶ")] += ("καταμετράω",)
    assert exact_inflection_lemma(conn, "καταμετρηθείσης", "καταμετρηθείσης",
        lexical_heads=heads, lemma_aliases=aliases) is None


def test_sparse_inflection_preserves_a_defined_exact_pos_alternative():
    from export_hunspell import exact_inflection_lemma, lexical_head_key
    conn = _lookup_db([("νευροειδές", "νευροειδές", "grc")])
    heads = frozenset({lexical_head_key("νευροειδής")})
    owners = {"νευροειδές": ("νευροειδής",)}
    assert exact_inflection_lemma(conn, "νευροειδές", "νευροειδές",
        lexical_heads=heads, exact_pos_owners=owners) == "νευροειδής"
    assert exact_inflection_lemma(conn, "νευροειδές", "νευροειδές",
        lexical_heads=frozenset(), exact_pos_owners=owners) is None
    assert exact_inflection_lemma(conn, "νευροειδές", "νευροειδές",
        lexical_heads=heads) is None
    assert exact_inflection_lemma(conn, "Νευροειδές", "Νευροειδές",
        lexical_heads=heads, exact_pos_owners=owners) is None


def test_sparse_evidence_merges_normalized_alias_and_form_conflicts(tmp_path):
    import json
    from export_hunspell import load_grc_inflection_evidence, lexical_head_key
    lexical = tmp_path / "lexical.json"
    lexical.write_text(json.dumps({"contracted_verb_aliases": {
        "τιμῶ": [{"lemma": "τιμάω", "sources": ["one"]}],
        "τιμώ": [{"lemma": "τιμέω", "sources": ["two"]}]}}))
    pos = tmp_path / "pos.json"
    pos.write_text(json.dumps({"τιμά": {"NOUN": "τιμή"},
                               "τιμὰ": {"VERB": "τιμάω"}}))
    heads = frozenset(map(lexical_head_key, ["τιμάω", "τιμέω", "τιμή"]))
    aliases, owners = load_grc_inflection_evidence(heads, lexical_path=lexical, pos_path=pos)
    assert aliases[lexical_head_key("τιμῶ")] == ("τιμάω", "τιμέω")
    assert set(owners["τιμά"]) == {"τιμάω", "τιμή"}


def test_modern_head_evidence_is_scoped_to_edited_vernacular(tmp_path):
    import json
    from collections import Counter
    from export_hunspell import add_postclassical_corpus_forms, lexical_head_key
    rows = [("παλάτια", "παλάτι", {"byzantine_vernacular": 1}),
            ("παλατιών", "παλάτι", {"pg": 1}),
            ("Παλάτια", "παλάτι", {"byzantine_vernacular": 1}),
            ("παλάττια", "ἀδήλωτο", {"byzantine_vernacular": 1}),
            ("συγγράμμματα", "σύγγραμμα", {"byzantine_vernacular": 1}),
            ("σπίτι", "σπίτι", {"byzantine_vernacular": 1})]
    profile = tmp_path / "profile.db"
    with sqlite3.connect(profile) as conn:
        conn.executescript("CREATE TABLE forms(form_id INTEGER, form TEXT);"
                           "CREATE TABLE form_profile(form_id INTEGER,total_count INTEGER,"
                           "source_counts_json TEXT,n_works INTEGER,dominant_pos TEXT);")
        for i, (form, _lemma, sources) in enumerate(rows):
            conn.execute("INSERT INTO forms VALUES(?,?)", (i, form))
            conn.execute("INSERT INTO form_profile VALUES(?,?,?,?,?)",
                         (i, 1, json.dumps(sources), 1, "noun"))
    lookup = _lookup_db([(form, lemma, "grc") for form, lemma, _sources in rows])
    heads = frozenset(map(lexical_head_key, ["παλάτι", "σύγγραμμα", "σπίτι"]))
    out, audit = add_postclassical_corpus_forms(
        [], lookup, Counter({"σπίτι": 1}), profile_path=profile,
        lexical_heads=frozenset(), vernacular_heads=heads)
    assert out == [("παλάτια", "παλάτι")]
    assert audit[0]["reason"] == "attested_lexical_form"


def test_reviewed_source_forms_preserve_valid_surfaces_with_corrupt_annotations():
    import json
    from export_hunspell import DATA, derive_reviewed_paradigm
    specs = json.loads((DATA / "hunspell_grc_postclassical_review.json").read_text())["paradigms"]
    forms = {form for spec in specs if spec["method"] == "source_forms"
             for form in derive_reviewed_paradigm(spec, [])}
    assert {"πανάγιον", "προκεκαθαρμένη", "συνεγκλειομένων", "ἀναθαρρυνόμενοι"} <= forms
    assert "ἀναθαρρρυνόμενοι" not in forms


def test_reviewed_sparse_textual_forms_keep_their_exact_spelling():
    import json
    from export_hunspell import DATA, derive_reviewed_paradigm
    specs = json.loads((DATA / "hunspell_grc_postclassical_review.json").read_text())["paradigms"]
    forms = {form: spec["lemma"] for spec in specs if spec["method"] == "source_forms"
             for form in derive_reviewed_paradigm(spec, [])}
    expected = {"κλώστρας": "κλώστρα", "μαγαρισμένε": "μαγαρίζω",
                "μαγαρισμένη": "μαγαρίζω", "μαζωμένα": "μαζώνω",
                "μαραμένη": "μαραίνω", "μαυλίστρα": "μαυλίστρα",
                "μεταλαμπαδευομένην": "μεταλαμπαδεύω", "μυρισμένα": "μυρίζω",
                "πανίτσιν": "πανίτσιν", "σπήλιο": "σπήλιο", "σπήτι": "σπίτι",
                "τιμημένος": "τιμώ", "τορνεμένα": "τορνεύω", "φηκάρια": "φηκάρι",
                "ἔποικα": "ποιέω", "ἔποικεν": "ποιέω", "ἔχρητο": "χράομαι",
                "τάχεϊ": "τάχος", "συμπεριόντα": "συμπερίειμι",
                "συνανιόντε": "συνάνειμι", "προσυποσχεῖν": "προσυπέχω",
                "ἀπῆσθ᾽": "ἄπειμι"}
    assert {form: forms.get(form) for form in expected} == expected
    assert "ὤοντο" not in forms


def test_verified_name_ownership_preserves_source_names_and_lower_homonyms():
    from export_hunspell import include_reviewed_name_ownership, filter_case_artifacts
    ownership = {exact_form_key("Σάξονας"): 2}
    include_reviewed_name_ownership(ownership, [
        {"form": "Σάξονας", "lemma": "Σάξων"},
        {"form": "λόγους", "lemma": "λόγος"}])
    pairs = [("Σάξονας", "Σάξων"), ("σάξονας", "σάξων")]
    kept, rejected = filter_case_artifacts(pairs, _evidence({}), ownership)
    assert kept == pairs
    assert rejected == []
    assert exact_form_key("λόγους") not in ownership




def test_reviewed_name_cells_survive_flat_common_word_ownership():
    from export_hunspell import (
        add_reviewed_postclassical_paradigms, filter_case_artifacts,
        include_reviewed_name_ownership, load_case_ownership,
    )
    expected = {
        "Παρίους": "Πάριος", "Πῶλε": "Πῶλος", "Τύχικον": "Τύχικος",
        "Κορέ": "Κορέ", "Ἰωαδά": "Ἰωαδά", "Ὀνησίμῳ": "Ὀνήσιμος",
    }
    conn = _lookup_db([(form, lemma.lower(), "grc")
                       for form, lemma in expected.items()]
                      + [("Πόλῃ", "πόλις", "grc"),
                         ("Στρατιώτῃ", "στρατιώτης", "grc")])
    pairs, protected, source_audit = add_reviewed_postclassical_paradigms([], conn)
    reviewed = {form: lemma for form, lemma in pairs if form in expected}
    assert reviewed == expected
    assert expected.keys() <= protected
    ownership = load_case_ownership(conn)
    include_reviewed_name_ownership(ownership, source_audit)
    name_pairs = list(expected.items())
    negatives = [("Πόλῃ", "πόλις"), ("Στρατιώτῃ", "στρατιώτης")]
    kept, rejected = filter_case_artifacts(name_pairs + negatives,
                                           _evidence({}), ownership)
    assert kept == name_pairs
    assert rejected == negatives


def test_unclassified_corpus_tokens_require_lexical_review():
    from export_hunspell import postclassical_admission_reason
    assert postclassical_admission_reason(100, {"pta": 100, "oga": 100}, 50,
                                         pos="other") is None


def test_classified_unknown_still_needs_independent_source_evidence():
    from export_hunspell import postclassical_admission_reason
    assert postclassical_admission_reason(100, {"pg": 100}, 50, pos="verb") is None
    assert postclassical_admission_reason(8, {"pta": 8, "oga": 6}, 5,
                                         pos="verb") == "repeated_multiple_sources"


def test_shared_name_only_owner_preserves_name_but_not_unattested_lowercase():
    from export_hunspell import load_case_ownership, filter_case_artifacts
    conn = _lookup_db([("Γερμανία", "Γερμανία", "el"),
                       ("γερμανία", "Γερμανία", "el")])
    pairs = [("Γερμανία", "Γερμανία"), ("γερμανία", "γερμανία")]
    kept, rejected = filter_case_artifacts(pairs, _evidence({})._replace(cased={}), load_case_ownership(conn))
    assert kept == [pairs[0]]
    assert rejected == [pairs[1]]


def test_ancient_name_and_common_homograph_preserve_both_owners():
    from export_hunspell import load_case_ownership, filter_case_artifacts
    conn = _lookup_db([("Ψυχή", "Ψυχή", "grc"),
                       ("ψυχή", "ψυχή", "grc")])
    pairs = [("Ψυχή", "Ψυχή"), ("ψυχή", "ψυχή")]
    kept, rejected = filter_case_artifacts(pairs, _evidence({}), load_case_ownership(conn))
    assert kept == pairs
    assert rejected == []


def test_joined_editorial_phrase_needs_review_without_rejecting_compatibility():
    from export_hunspell import postclassical_spelling_reason
    assert postclassical_spelling_reason("ἐπιτοπουλύ") == "joined_editorial_phrase"
    assert grc_orthography_reason("ἐπιτοπουλύ") is None


def test_heldout_trusted_token_cannot_restore_untrusted_works():
    from export_hunspell import postclassical_admission_reason
    assert postclassical_admission_reason(
        21, {"pta": 21, "pg": 3, "oga": 3, "first1k": 3}, 10,
        heldout_count=1, pos="noun") is None


@pytest.mark.parametrize("bad,correct", [
    ("τέσσσαρας", "τέσσαρας"),
    ("ἀναθαρρρυνόμενοι", "ἀναθαρρυνόμενοι"),
    ("συμμμαχίας", "συμμαχίας"),
    ("μέλλλει", "μέλλει"),
])
def test_reviewed_historical_ocr_errors_do_not_survive_compatibility(bad, correct):
    from export_hunspell import load_grc_spelling_review
    rejects = load_grc_spelling_review()
    assert bad in rejects
    assert correct not in rejects
    control = "Βρισηίδος"  # A damaged historical lemma does not invalidate its form.
    kept, report = finalize_grc_pairs(
        [(bad, bad), (correct, correct), (control, control)],
        evidence=_evidence(exact={bad: 1, correct: 1, control: 1}),
        compatibility_forms={bad, correct, control},
        textbook_forms=set(),
        export_overrides={},
        protected_forms=set(),
        reviewed_rejects=rejects,
    )
    assert {form for form, _lemma in kept} == {correct, control}
    assert report["reviewed"] == 1


def test_reviewed_irregular_compound_errors_do_not_reject_independent_homographs():
    from export_hunspell import load_grc_spelling_review
    bad = {"δεἰσί", "ἄνεσει", "εἴσιών", "ἄνιόντι", "ἄνιών", "ἀντάνιόν",
           "συμπάροντες", "συμπάροντι", "συμπάροντων", "συμπάρων", "συμπάρησαν",
           "ἄνεστί", "ἄνεστίν"}
    good = {"ἄνιόν", "ἐπείσιόν", "ὕπεστί", "ὕπεστίν", "εἴσω", "ὑποκάτω",
            "παρέξω", "ἄνων", "ἄνοντος", "προσάνων", "ἄνιοι",
            "ἀνέσει", "εἰσιών", "ἀνιόντι", "ἀνιών", "συμπαρόντες",
            "συμπαρόντι", "συμπαρόντων", "συμπαρών", "συμπαρῆσαν"}
    rejects = load_grc_spelling_review()
    assert bad <= rejects
    assert not good & rejects
    forms = sorted(bad | good)
    kept, report = finalize_grc_pairs(
        [(form, form) for form in forms],
        evidence=_evidence(exact={form: 1 for form in forms}),
        compatibility_forms=set(forms),
        textbook_forms=set(),
        export_overrides={},
        protected_forms=set(),
        reviewed_rejects=rejects,
    )
    assert {form for form, _lemma in kept} == good
    assert report["reviewed"] == len(bad)


def test_diorisis_quote_index_keeps_case_and_distinguishes_existing_elision(tmp_path):
    import json
    from export_hunspell import cased_form_key, load_diorisis_quote_keys
    resolved = 'duplicated_closing_quote_with_same_source_cell'
    unresolved = 'unresolved_terminal_quote_or_elision'
    records = [
        {'original_beta': "*pu=r'", 'normalized_beta': '*pu=r', 'reason': resolved},
        {'original_beta': "u(/dwr'", 'normalized_beta': "u(/dwr'", 'reason': unresolved},
        {'original_beta': "a)ll''", 'normalized_beta': "a)ll'", 'reason': resolved},
    ]
    path = tmp_path/'audit.json'
    path.write_text(json.dumps({'schema_version': 1, 'sources': [{'records': records}]}))
    keys, meta = load_diorisis_quote_keys(path)
    assert keys == {cased_form_key('Πῦρ᾽'), cased_form_key('ὕδωρ᾽')}
    assert cased_form_key('πῦρ᾽') not in keys
    assert meta['keys'] == 2 and len(meta['sha256']) == 64


def test_quote_counts_and_generated_ownership_cannot_prove_elision():
    from export_hunspell import cased_form_key
    form = 'ὕδωρ᾽'
    kept, report = finalize_grc_pairs(
        [(form, 'ὕδωρ')], evidence=_evidence(exact={form: 1000}),
        compatibility_forms=set(), textbook_forms=set(), export_overrides={},
        protected_forms={form}, quote_keys=frozenset({cased_form_key(form)}))
    assert kept == []
    assert report['quote_rejected'] == [(form, 'ὕδωρ')]


def test_independently_reviewed_elision_and_shipped_elision_survive_quote_gate():
    from export_hunspell import cased_form_key
    forms = {'εἶ᾽', 'ἀνίησ᾽'}
    kept, report = finalize_grc_pairs(
        [(form, form) for form in sorted(forms)],
        evidence=_evidence(exact={form: 5 for form in forms}),
        compatibility_forms={'εἶ᾽'}, textbook_forms=set(), export_overrides={},
        protected_forms=forms, quote_keys=frozenset(map(cased_form_key, forms)),
        quote_approved_forms={'ἀνίησ᾽'})
    assert {form for form, _ in kept} == forms
    assert not report['quote_rejected']


def test_quote_exceptions_preserve_case_and_do_not_override_spelling_review():
    from export_hunspell import cased_form_key
    form = 'Πῦρ᾽'
    kept, report = finalize_grc_pairs(
        [(form, 'πῦρ')], evidence=_evidence(exact={form: 10}),
        compatibility_forms=set(), textbook_forms=set(), export_overrides={},
        protected_forms={form}, quote_keys=frozenset({cased_form_key(form)}),
        quote_approved_forms={'πῦρ᾽'})
    assert kept == [] and report['quote_evidence'] == 1
    kept, report = finalize_grc_pairs(
        [(form, 'πῦρ')], evidence=_evidence(exact={form: 10}),
        compatibility_forms={form}, textbook_forms=set(), export_overrides={},
        protected_forms={form}, quote_keys=frozenset({cased_form_key(form)}),
        quote_approved_forms={form}, reviewed_rejects=frozenset({form}))
    assert kept == [] and report['reviewed'] == 1


def test_exact_elision_heads_do_not_infer_inflections_or_fold_case(tmp_path):
    import json
    from export_hunspell import cased_form_key, load_exact_elision_heads
    (tmp_path/'dge_headwords.json').write_text(json.dumps(['εἶ’', 'ἀνίημι']))
    (tmp_path/'cunliffe_headwords.json').write_text(json.dumps(['Ἀπόλλων’']))
    (tmp_path/'grc_lexical_heads.json').write_text(json.dumps({
        'schema_version': 2, 'heads': {'ἀλλ᾽': ['source'], 'ἀνίησ᾽': []}}))
    heads, inputs = load_exact_elision_heads(tmp_path)
    assert heads == set(map(cased_form_key, ('εἶ᾽', 'Ἀπόλλων᾽', 'ἀλλ᾽')))
    assert cased_form_key('ἀνίησ᾽') not in heads
    assert cased_form_key('ἀπόλλων᾽') not in heads
    assert len(inputs) == 3


def test_cited_elision_homographs_survive_without_compatibility_or_corpus_proof():
    import json
    from export_hunspell import DATA, cased_form_key, derive_reviewed_paradigm
    expected = {"Πλάτων᾽": "Πλάτων", "δέοι᾽": "δέω", "εἷσ᾽": "ἵζω",
                "θεοῖσ᾽": "θεός", "νίκησ᾽": "νικάω", "πανταχοῦ᾽": "πανταχοῦ",
                "παῖ᾽": "παίω", "ποῦ᾽": "ποῦ", "χείρ᾽": "χείρ",
                "ἀνδρῶν᾽": "ἀνδρών", "ἄν᾽": "ἀνά", "Ἀγάθων᾽": "Ἀγάθων",
                "Ἀπόλλων᾽": "Ἀπόλλων", "ἔνι᾽": "ἔνιοι", "ὥρασ᾽": "ὥρασι",
                "εἶ᾽": "εἶα", "ἀνίησ᾽": "ἀνίημι"}
    specs = json.loads((DATA / "hunspell_grc_postclassical_review.json").read_text())["paradigms"]
    forms = {form: spec["lemma"] for spec in specs if spec["method"] == "source_forms"
             for form in derive_reviewed_paradigm(spec, [])}
    assert {form: forms.get(form) for form in expected} == expected
    # These are exact cited exceptions; raw quotation counts or an arbitrary
    # historical dictionary entry are not their admission evidence.
    kept, report = finalize_grc_pairs(
        sorted(expected.items()), evidence=_evidence(exact={}),
        compatibility_forms=set(), textbook_forms=set(), export_overrides={},
        protected_forms=set(expected), quote_keys=frozenset(map(cased_form_key, expected)),
        quote_approved_forms=set(expected))
    assert set(kept) == set(expected.items())
    assert not report["quote_rejected"]


def test_confirmed_closing_quote_rejection_overrides_compatibility_and_elision_approval():
    from export_hunspell import cased_form_key, load_grc_spelling_review
    forms = {"ἄφεσ᾽", "παῖ᾽", "Ἀπόλλων᾽", "ἀνίησ᾽"}
    kept, report = finalize_grc_pairs(
        [(form, form) for form in sorted(forms)], evidence=_evidence(exact={}),
        compatibility_forms=forms, textbook_forms=set(), export_overrides={},
        protected_forms=forms, quote_keys=frozenset(map(cased_form_key, forms)),
        quote_approved_forms=forms, reviewed_rejects=load_grc_spelling_review())
    assert {form for form, _ in kept} == forms - {"ἄφεσ᾽"}
    assert report["reviewed"] == 1


def test_reviewed_split_word_fragment_does_not_override_complete_homographs():
    from export_hunspell import load_grc_spelling_review
    rejects = load_grc_spelling_review()
    bad = {"ἐνεί", "ἐνεὶ"}
    good = {"ἔνει", "ἐνείραντα", "ἐνείκασθαι", "ἐκεῖ", "ἐκείνην", "ἐκείνοις"}
    assert "ἐνεί" in rejects
    assert not good & rejects
    forms = bad | good
    kept, report = finalize_grc_pairs(
        [(form, form) for form in sorted(forms)], evidence=_evidence(exact={}),
        compatibility_forms=forms, textbook_forms=set(), export_overrides={},
        protected_forms=forms, reviewed_rejects=rejects)
    assert {form for form, _ in kept} == good
    assert report["reviewed"] >= len(bad)


def test_reviewed_circumflex_transcription_preserves_real_verb_and_connectives():
    from export_hunspell import load_grc_spelling_review
    rejects = load_grc_spelling_review()
    bad = "ἐπεῖ"
    good = {"ἔπει", "ἐπεί", "ἐπεὶ", "ἐκεῖ", "ἐπεῖδεν"}
    assert bad in rejects
    assert not good & rejects
    forms = good | {bad}
    kept, report = finalize_grc_pairs(
        [(form, form) for form in sorted(forms)], evidence=_evidence(exact={}),
        compatibility_forms=forms, textbook_forms=set(), export_overrides={},
        protected_forms=forms, reviewed_rejects=rejects)
    assert {form for form, _ in kept} == good
    assert report["reviewed"] == 1

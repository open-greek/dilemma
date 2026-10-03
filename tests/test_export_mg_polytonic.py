#!/usr/bin/env python3
"""Tests for the polytonic Modern Greek word list (export_mg_polytonic.py)
and the Modern Greek elisions it derives for export_morphology.py.

The unit tests build their corpus counts by hand, so they run without the
Wikisource parquet. The artifact tests read build/hunspell/ and skip when
the list has not been built.

Run with:
    python -m pytest tests/test_export_mg_polytonic.py -v
"""

import json
import sys
from collections import Counter
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import export_mg_polytonic as mg  # noqa: E402
from export_mg_polytonic import (  # noqa: E402
    Candidate,
    CorpusCounts,
    DocumentInfo,
    Token,
    gather_candidates,
    mg_orthography_reason,
    modern_greek_elisions,
    monotonic_signal,
    monotonic_spelling,
    oxytone_twins,
    select_forms,
    tokenize,
)

K = mg.KORONIS


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def forms(sentence: str) -> list[str]:
    return [t.form for t in tokenize(sentence)]


# --------------------------------------------------------------------------
# Tokens and monotonic signals
# --------------------------------------------------------------------------

def test_tokens_keep_elision_and_aphaeresis_marks_as_the_koronis():
    assert forms("κι' ἐγώ ’ναι ἀπ’ τὸ σπίτι") == [
        "κι" + K, "ἐγώ", K + "ναι", "ἀπ" + K, "τὸ", "σπίτι"]
    # The koronis itself, and the spacing psili, are marks, not letters.
    assert forms("γι᾽ αὐτό") == ["γι" + K, "αὐτό"]
    assert forms("στ᾿ ἀλώνι") == ["στ" + K, "ἀλώνι"]


def test_the_spacing_dasia_is_an_aphaeresis_mark_or_a_breathing():
    # Before a consonant it marks aphaeresis (νά ῾βρω for νὰ εὕρω) ...
    assert forms("νά ῾βρω ἐδῶθε") == ["νά", K + "βρω", "ἐδῶθε"]
    assert forms("῾κάστηκε") == [K + "κάστηκε"]
    # ... and before a vowel it is the vowel's breathing, set before a
    # capital the old way, as is the spacing psili.
    assert forms("῾Η Πηγὴ ᾿Αφ᾿ οὗ") == ["Ἡ", "Πηγὴ", "Ἀφ" + K, "οὗ"]


def test_a_mark_after_a_final_sigma_closes_a_quotation():
    assert forms("'ὁ λόγος' εἶπε") == ["ὁ", "λόγος", "εἶπε"]


def test_a_word_run_into_a_hyphen_numeral_or_digit_is_not_plain():
    plain = {t.form: t.plain for t in tokenize("ἀπο-φασίζω αʹ 1ον καλός")}
    assert plain == {"ἀπο": False, "φασίζω": False, "α": False,
                     "ον": False, "καλός": True}


def test_letters_printed_as_compatibility_symbols_are_read_as_letters():
    # Micro sign for mu, as OCR leaves it, and the curled beta.
    assert forms("\u00b5ακαρίσωμεν τὸν \u03d0ίο") == [
        "μακαρίσωμεν", "τὸν", "βίο"]


def test_a_word_opening_a_line_is_initial():
    tokens = tokenize("Ἦρθε τὸ βράδυ\nΚι ἔπεσε")
    assert [t.initial for t in tokens] == [True, False, False, True, False]


def test_monotonic_signals():
    def signal(word, **kw):
        return monotonic_signal(Token(nfc(word), False, False, False, True)
                                ._replace(**kw))
    assert signal("είναι") == "missing_breathing"
    assert signal("ο") == "missing_breathing"
    assert signal("και") == "unaccented_monosyllable"
    assert signal("να") == "unaccented_monosyllable"
    # Polytonic spellings, and the breathing on a diphthong's second vowel.
    assert signal("εἶναι") is None
    assert signal("αὐτός") is None
    assert signal("καὶ") is None
    # Not judged: capitals, elided or aphaeresized words, numerals.
    assert signal("Είναι") is None
    assert signal("ναι", lead=True) is None
    assert signal("ον", plain=False) is None
    # A weak pronoun is unaccented in polytonic writing too.
    assert signal("το") is None


def test_an_abbreviation_is_a_consonant_final_word_before_a_period():
    def abbr(word):
        return mg.is_abbreviation(Token(word, False, False, False, True))
    assert abbr("τόμ") and abbr("σελ") and abbr("Θεόδ")
    assert not abbr("λόγος") and not abbr("τώρα") and not abbr("ἀήρ")


def test_a_capital_that_carries_the_iota_is_a_capital():
    # ᾍ and ᾘ are titlecase letters, which str.isupper() does not count.
    tokens = tokenize("εἶπε τοῦ ᾍδη καὶ ᾘσθανόμην")
    assert [mg._position(t) for t in tokens] == [
        "lower", "lower", "cap", "lower", "cap"]
    assert mg.is_capital("ᾍ") and not mg.is_capital("ᾅ")


def test_a_capital_after_an_ellipsis_a_bracket_or_a_colon_is_initial():
    assert [t.initial for t in tokenize(
        "ἴσως… Χάχ, Χάχ (γελᾷ) Χάχ [γελᾷ] Χάχ")] == [
        True, True, False, True, True, True, True]
    assert [t.initial for t in tokenize("εἶπε: Τώρα, Τώρα")] == [
        True, True, False]


def test_a_word_broken_by_a_soft_hyphen_is_not_plain():
    plain = [t.plain for t in tokenize("ἀπο" + chr(0xAD) + "φασίζω καλός")]
    assert plain == [False, False, True]


@pytest.mark.parametrize("word", [
    "και", "να", "θα", "δεν", "δε", "για", "μια", "πια",
    "στο", "στα", "στη", "στην", "στον", "στις", "στους"])
def test_an_unaccented_particle_or_article_is_a_monotonic_signal(word):
    token = Token(word, False, False, False, True)
    assert monotonic_signal(token) == "unaccented_monosyllable"


def test_a_word_ending_in_a_final_consonant_is_no_abbreviation():
    def abbr(word):
        return mg.is_abbreviation(Token(word, False, False, False, True))
    assert not any(abbr(w) for w in ("λόγον", "αἴξ", "Πέλοψ", "πάρ", "λόγος"))


def test_split_sentences_with_closers_matches_the_language_model_split():
    from extract_polytonic_mg import _split_sentences, split_sentences_with_closers
    text = "Ἐν τῷ τόμ. Β΄ σελ. 5 ἀρχίζει!\n\nΝέα παράγραφος ; τέλος"
    pairs = list(split_sentences_with_closers(text))
    assert [s for s, _ in pairs] == list(_split_sentences(text))
    assert pairs[0] == ("Ἐν τῷ τόμ", ".")
    assert pairs[-1][1] == ""
    # A space before the period: no word sits right before it.
    assert pairs[2][1] == "!"
    assert pairs[3] == ("Νέα παράγραφος", "")


# --------------------------------------------------------------------------
# Orthography
# --------------------------------------------------------------------------

@pytest.mark.parametrize("word", [
    "του", "της", "μας", "σας", "τους", "των", "κι",
])
def test_modern_greek_writes_the_weak_pronouns_and_ki_without_an_accent(word):
    assert mg_orthography_reason(word) is None


@pytest.mark.parametrize("word, reason", [
    ("το", "missing_tonal_accent"),      # the article is τὸ
    ("και", "missing_tonal_accent"),
    ("είναι", "missing_initial_breathing"),
    ("θά", "explicit_reject"),           # the particle is θὰ
    ("τού", "explicit_reject"),
    ("ἀλλ", "bare_elision"),
    ("ὐπὸ", "smooth_initial_upsilon"),
    ("ὃταν", "grave_before_ultima"),
    ("ἄλλό", "misplaced_second_accent"),
])
def test_the_grc_rules_still_reject_what_polytonic_greek_never_writes(
        word, reason):
    assert mg_orthography_reason(nfc(word)) == reason


@pytest.mark.parametrize("word", [
    "τέλειωσε", "πλάγιασε", "μετάνοιωσα", "κουβέντιαζε", "σκοτείνιασε",
    "γάϊδαρος", "βόιδια", "κλάϋματα", "ἀγκάλιασέ", "γάϊδαρός",
])
def test_synizesis_keeps_the_accent_within_three_syllables(word):
    assert mg_orthography_reason(nfc(word)) is None


def test_an_accent_further_back_than_synizesis_explains_is_rejected():
    assert mg_orthography_reason(nfc("ἄγαπητος")) == "accent_before_antepenult"
    assert not mg.mg_accent_window_ok(nfc("ἄγαπητος"))


def test_a_consonant_final_word_is_kept_unless_it_lost_a_vowel():
    lexicon = {"τόμο", "τώρα"}
    assert mg_orthography_reason("ἄχ", lexicon) is None
    assert mg_orthography_reason("κονιάκ", lexicon) is None
    assert mg_orthography_reason("τόμ", lexicon) == "truncated_word"
    # Without a lexicon the grc rule stands.
    assert mg_orthography_reason("ἄχ") == "truncated_fragment"


def test_elided_and_aphaeresized_spellings():
    for word in ("κι" + K, "γι" + K, "τώρ" + K, "στ" + K, K + "ναι", K + "ς"):
        assert mg_orthography_reason(word) is None, word
    assert mg_orthography_reason("ζῶ" + K) == "vowel_before_elision"
    assert mg_orthography_reason("γί" + K) == "vowel_before_elision"
    assert mg_orthography_reason(K + "ἐγώ") == "vowel_after_aphaeresis"


def test_the_oxytone_twins():
    assert oxytone_twins("στὴν") == ["στήν"]
    assert oxytone_twins("τοπικό") == ["τοπικὸ"]
    assert oxytone_twins("δάσκαλός") == []      # an enclitic's accent
    assert oxytone_twins("τώρα") == []
    assert oxytone_twins("μπορεῖ") == []
    assert oxytone_twins("γι" + K) == []


def test_monotonic_spelling():
    assert monotonic_spelling("ἔμενα") == "έμενα"
    assert monotonic_spelling("εἶχε") == "είχε"
    assert monotonic_spelling("σόϊ") == "σόι"
    assert monotonic_spelling("ψυχῇ") == "ψυχή"
    assert monotonic_spelling("κάθ" + K) == "κάθ"


def test_a_rare_spelling_is_a_word_only_if_its_monotonic_spelling_differs():
    known = {"χρονιά", "είχε", "εινε", "μητε"}.__contains__
    full = mg.FullForms(["κάθε", "κατὰ", "εἶνε", "μήτε"])
    assert mg.distinct_word("χρονιά", "χρόνια", known)
    assert not mg.distinct_word("εἴχε", "εἶχε", known)      # a misspelling
    assert not mg.distinct_word("θάνατου", "θανάτου", known)  # not a word
    # An elided spelling is judged by the attested words it stands for, not
    # by lookup.db, whose accent-stripped keys match any stem plus a vowel.
    assert mg.distinct_word("κάθ" + K, "καθ" + K, known, full)   # κάθε
    assert not mg.distinct_word("εἰν" + K, "εἶν" + K, known, full)
    assert not mg.distinct_word("κάθ" + K, "καθ" + K, known)     # no words


def test_the_words_an_elided_spelling_stands_for():
    full = mg.FullForms(["τώρα", "γιὰ", "γιατὶ", "ἐδῶ", "μήτε", "κατὰ",
                         "κάθε", "ποὺ", "εἶναι", K + "στὰ"])
    assert full.of("τώρ" + K) == ["τώρα"]          # accent kept in place
    assert full.of("γι" + K) == ["γιὰ"]            # an oxytone loses it
    assert full.of("γιατ" + K) == ["γιατὶ"]
    assert full.of("ἐδ" + K) == ["ἐδῶ"]
    assert full.of("π" + K) == ["ποὺ"]             # the digraph ου
    assert full.of("εἶν" + K) == ["εἶναι"]
    assert full.of("καθ" + K) == ["κατὰ"]          # before a rough breathing
    assert full.of("καθ" + K, aspirated=False) == []
    assert full.of("κάθ" + K) == ["κάθε"]
    assert full.of(K + "στ" + K) == [K + "στὰ"]
    # A paroxytone keeps its accent when it is elided.
    assert full.of("μητ" + K) == [] and full.of("μήτ" + K) == ["μήτε"]


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def cand(form, tokens=10, authors=3):
    return Candidate(form, tokens, authors, authors)


def test_selection_thresholds_grc_overlap_and_twins():
    candidates = {c.form: c for c in [
        cand("τώρα", 1500), cand("σπίτι", 120), cand("κι", 4000),
        cand("στὴν", 2400), cand("στήν", 9),
        cand("ἀκόμη", 2, 2),       # too few tokens
        cand("φτερούγα", 30, 1),   # one author
        cand("λόγος", 900),        # grc has it
    ]}
    sel = select_forms(candidates, grc_words={"λόγος", "στήν"})
    # στήν is grc's; στὴν carries the pair's combined count.
    assert sel.entries == {"τώρα": 1500, "σπίτι": 120, "κι": 4000,
                           "στὴν": 2409}
    assert sel.report["too_few_tokens"] == 1
    assert sel.report["single_author"] == 1
    # λόγος, στήν, and στήν again as στὴν's twin.
    assert sel.report["in_grc"] == 3


def test_the_twin_of_a_listed_oxytone_is_added():
    sel = select_forms({"τοπικό": cand("τοπικό", 40)}, grc_words=set())
    assert sel.entries == {"τοπικό": 40, "τοπικὸ": 40}


def test_a_weak_respelling_goes_unless_it_is_another_word():
    candidates = {c.form: c for c in [
        cand("εἶχε", 2400), cand("εἴχε", 7),
        cand("χρόνια", 384), cand("χρονιά", 19),
    ]}
    known = {"χρονιά", "είχε"}.__contains__
    sel = select_forms(candidates, grc_words=set(), known_word=known)
    assert "εἴχε" not in sel.entries
    assert "χρονιά" in sel.entries
    # Without the lexicon both go.
    sel = select_forms(candidates, grc_words=set())
    assert "χρονιά" not in sel.entries


def test_a_reviewed_reject_and_its_twin_are_not_listed():
    candidates = {"πράξιν": cand("πράξιν", 30)}
    sel = select_forms(candidates, set(), frozenset({"πράξιν"}))
    assert sel.entries == {}
    candidates = {"τουκαὶ": cand("τουκαὶ", 4)}
    sel = select_forms(candidates, set(), frozenset({"τουκαί"}))
    assert sel.entries == {}


def test_a_breathing_misspelling_goes_whatever_the_counts():
    candidates = {c.form: c for c in [
        # Five times commoner in the slice, with no floor on the count.
        cand("ἕτοιμος", 65), cand("ἔτοιμος", 4),
        cand("ἅγιο", 97), cand("ἄγιο", 3),
        # At least as common and in grc.
        cand("ἐμᾶς", 104), cand("ἑμᾶς", 25),
        # grc's own breathing variants decide nothing when they are rarer:
        # ἦμαι stays beside grc's ἧμαι.
        cand("ἦμαι", 13), cand("ἧμαι", 2),
        # Words that differ only in the breathing.
        cand("ἢ", 4300), cand("ἣ", 30),
        cand("ὄντας", 31), cand("ὅντας", 12),
    ]}
    sel = select_forms(candidates, grc_words={"ἐμᾶς", "ἧμαι", "ἢ", "ὄντας"},
                       known_word=lambda word: True)
    assert {"ἕτοιμος", "ἅγιο", "ἦμαι", "ἣ", "ὅντας"} <= set(sel.entries)
    assert not {"ἔτοιμος", "ἄγιο", "ἑμᾶς"} & set(sel.entries)
    assert sel.report["breathing_respelling"] == 3


def test_a_rare_spelling_that_is_a_word_must_hold_its_own_beside_its_twin():
    # ὄποιος is not ὁποῖος (another word by its monotonic spelling), but it
    # is ὅποιος with the wrong breathing.
    candidates = {c.form: c for c in [
        cand("ὁποῖος", 900), cand("ὅποιος", 152), cand("ὄποιος", 3)]}
    known = {"όποιος", "οποίος"}.__contains__
    sel = select_forms(candidates, set(), known_word=known)
    assert "ὅποιος" in sel.entries and "ὄποιος" not in sel.entries


def test_an_unaccented_elision_must_stand_for_a_word_accented_on_its_end():
    candidates = {c.form: c for c in [
        cand("μήτε", 280), cand("μήτ" + K, 53), cand("μητ" + K, 3),
        cand("γιὰ", 3500), cand("γι" + K, 290),
        cand("κι" + K, 3900), cand("κι", 4000),
    ]}
    sel = select_forms(candidates, set())
    assert {"μήτ" + K, "γι" + K, "κι" + K} <= set(sel.entries)
    assert "μητ" + K not in sel.entries
    assert sel.report["unaccented_elision"] == 1


def test_the_rare_spelling_rule_starts_at_a_hundred_tokens_and_a_twentieth():
    def kept(top, rare):
        candidates = {c.form: c for c in [cand("εἶχε", top), cand("εἴχε", rare)]}
        return "εἴχε" in select_forms(candidates, grc_words=set()).entries
    assert not kept(100, 4)
    assert kept(99, 4)       # beside a rarer word the rule does not apply
    assert kept(100, 5)      # a twentieth is not under a twentieth


def test_the_breathing_rule_needs_five_times_the_tokens_or_grc():
    def kept(twin, rare, grc=frozenset()):
        candidates = {c.form: c for c in [
            cand("ἕτοιμος", twin), cand("ἔτοιμος", rare)]}
        return "ἔτοιμος" in select_forms(candidates, grc_words=set(grc)).entries
    assert not kept(20, 4)
    assert kept(19, 4)
    # As common as the other spelling, which grc has.
    assert not kept(4, 4, {"ἕτοιμος"})
    assert kept(3, 4, {"ἕτοιμος"})


@pytest.mark.parametrize("word", [
    "αἳ", "ἣ", "οἳ", "ὅντας", "ἄρματα", "ἄρματά", "οὗλα"])
def test_a_reviewed_breathing_homograph_stands_beside_a_commoner_twin(word):
    twin = mg.breathing_twin(word)
    candidates = {c.form: c for c in [cand(twin, 2000), cand(word, 20)]}
    sel = select_forms(candidates, grc_words=set())
    assert word in sel.entries


def test_a_name_does_not_compete_with_a_word_of_its_letters():
    candidates = {c.form: c for c in [cand("διά", 5000), cand("Δία", 50)]}
    sel = select_forms(candidates, grc_words={"διά", "διὰ"})
    assert set(sel.entries) == {"Δία"}


def test_a_word_is_a_truncation_only_of_a_word_the_slice_attests():
    # ἄχα has too few tokens to be a word ἄχ could have lost its vowel from.
    candidates = {c.form: c for c in [cand("ἄχ", 10), cand("ἄχα", 2)]}
    assert "ἄχ" in select_forms(candidates, grc_words=set()).entries
    candidates["ἄχα"] = cand("ἄχα", 3)
    assert "ἄχ" not in select_forms(candidates, grc_words=set()).entries


def test_a_rare_spelling_that_is_a_word_is_still_a_misspelling_of_its_own():
    # δώρα is a word (its monotonic spelling differs from δωρά's), but it
    # is δῶρα with the wrong accent: the same monotonic spelling, and that
    # rare beside it.
    candidates = {c.form: c for c in [
        cand("δωρά", 1000), cand("δῶρα", 300), cand("δώρα", 10)]}
    known = {"δώρα", "δωρά"}.__contains__
    sel = select_forms(candidates, set(), known_word=known)
    assert "δῶρα" in sel.entries and "δώρα" not in sel.entries
    candidates["δῶρα"] = cand("δῶρα", 199)
    sel = select_forms(candidates, set(), known_word=known)
    assert "δώρα" in sel.entries


def test_a_capital_final_sigma_is_the_lowercase_final_sigma():
    assert mg._lowercase(K + "Σ") == K + "ς"
    assert mg._lowercase("ΣΑΣ") == "σας"
    candidates = {c.form: c for c in [cand(K + "ς", 50), cand(K + "Σ", 10)]}
    sel = select_forms(candidates, grc_words=set())
    assert set(sel.entries) == {K + "ς"}


def test_the_reviewed_modern_greek_spellings_are_well_formed_and_correct():
    from export_hunspell import load_grc_spelling_review
    rejects = load_grc_spelling_review(mg.MG_SPELLING_REVIEW)
    assert {"τουκαὶ", "τουναντίον", "ἤταν", "ἐδῷ", "ξῦλον"} <= rejects
    # The variants the review accepted stay out of it.
    assert not {"θέσῃ", "πᾷς", "τρομερᾶς", "ἀκριβῆς", "κορῶνα",
                "ποιητῆ"} & rejects
    payload = json.loads(mg.MG_SPELLING_REVIEW.read_text("utf-8"))
    for row in payload["reviews"]:
        assert row["correct"] not in rejects, row


def make_counts(rows, authors=("Α", "Β", "Γ")):
    """rows: [(form, position, {document index: tokens})]."""
    counts = CorpusCounts()
    counts.documents = [DocumentInfo(f"d{i}", a, f"t{i}", words=1000)
                        for i, a in enumerate(authors)]
    for form, position, per_doc in rows:
        counts.forms[(nfc(form), position)].update(per_doc)
    return counts


def test_case_follows_the_texts():
    counts = make_counts([
        ("τώρα", "lower", {0: 10, 1: 5}),
        ("Τώρα", "initial", {0: 7}),
        ("Μαρούλα", "cap", {0: 6, 1: 2}),
        ("Μαρούλα", "initial", {1: 1}),
        ("Φεῦγα", "initial", {0: 2, 1: 2}),   # only ever opening a line
    ])
    candidates = gather_candidates(counts, {0, 1, 2})
    assert candidates["τώρα"].tokens == 22
    assert candidates["τώρα"].authors == 2
    assert "Τώρα" not in candidates
    assert candidates["Μαρούλα"] == Candidate("Μαρούλα", 9, 2, 2)
    assert "μαρούλα" not in candidates
    assert candidates["φεῦγα"].tokens == 4
    assert "Φεῦγα" not in candidates


def test_each_case_stands_on_its_own_tokens_and_authors():
    counts = make_counts([
        # A word and a name: the lowercase word by six authors, the name
        # capitalized by one.
        ("διάολος", "lower", {0: 4, 1: 3}),
        ("Διάολος", "cap", {2: 12}),
        # One author's word beside another author's name: neither borrows
        # the other's author, and the capitals opening sentences go with
        # the case more authors write inside a sentence.
        ("λάζος", "lower", {0: 2}),
        ("Λάζος", "cap", {1: 4, 2: 2}),
        ("Λάζος", "initial", {0: 3}),
    ])
    candidates = gather_candidates(counts, {0, 1, 2})
    assert candidates["διάολος"] == Candidate("διάολος", 7, 2, 2)
    assert candidates["Διάολος"] == Candidate("Διάολος", 12, 1, 1)
    assert candidates["λάζος"] == Candidate("λάζος", 2, 1, 1)
    assert candidates["Λάζος"] == Candidate("Λάζος", 9, 3, 3)
    sel = select_forms(candidates, grc_words=set())
    assert set(sel.entries) == {"διάολος", "Λάζος"}


def test_a_word_and_its_contextual_twin_share_their_case():
    counts = make_counts([
        ("ἀετὲ", "lower", {0: 1}),
        ("Ἀετέ", "cap", {1: 1}),
        # Lines of verse opening with the word, acute and grave.
        ("Ἀετέ", "initial", {2: 2}),
        ("Ἀετὲ", "initial", {2: 1}),
    ])
    candidates = gather_candidates(counts, {0, 1, 2})
    # One author writes the word lowercase and one capitalized inside a
    # sentence, so the line-initial capitals go with the lowercase word.
    assert candidates["ἀετέ"] == Candidate("ἀετέ", 2, 1, 1)
    assert candidates["ἀετὲ"] == Candidate("ἀετὲ", 2, 2, 2)
    assert candidates["Ἀετέ"] == Candidate("Ἀετέ", 1, 1, 1)


def test_a_capital_hunspell_accepts_through_a_lowercase_entry_is_not_listed():
    candidates = {c.form: c for c in [
        cand("Βουλήν", 20), cand("τοπικός", 20), cand("Τοπικός", 20)]}
    sel = select_forms(candidates, grc_words={"βουλήν", "βουλὴν"})
    # grc accepts Βουλήν through βουλήν; the list's τοπικός covers its own
    # capital.
    assert set(sel.entries) == {"τοπικός", "τοπικὸς"}
    assert mg.grc_accepts("Βουλήν", {"βουλήν"})
    assert not mg.grc_accepts("βουλήν", {"Βουλήν"})


def test_the_grc_rejects_and_the_unaccented_words_hold_in_either_case():
    assert mg_orthography_reason("Θά") == "explicit_reject"
    assert mg_orthography_reason("Γιά") == "explicit_reject"
    assert mg_orthography_reason("Του") is None
    assert mg_orthography_reason("Κι") is None


def test_a_capital_after_a_quotation_dash_or_colon_is_initial():
    tokens = tokenize("εἶπε: «Τώρα ἔλα» — Ναί, ἔλα")
    assert [t.initial for t in tokens] == [True, True, False, True, False]


def test_held_out_documents_and_authors_are_not_read():
    counts = make_counts([], authors=("Α", "Β", "Γ"))
    counts.documents[1].dev_sentences = 2
    counts.documents[2].signals = 20      # 2% of its words
    assert mg.source_documents(counts) == {0, 1}
    assert mg.source_documents(counts, holdout_dev_documents=True) == {0}
    fold = mg.author_fold("Α", 3)
    assert 0 not in mg.source_documents(counts, holdout_author_fold=(fold, 3))


# The held-out text must stay out of the list: the language model's dev
# sentences, the documents and authors an evaluation holds out, and the
# sentences that read as monotonic. These tests feed count_corpus a
# document of known sentences.

LETTERS = "αβγδεζηθικ"


def marker(i: int) -> str:
    """A word that occurs in sentence ``i`` only."""
    return "μ" + "".join(LETTERS[int(d)] for d in f"{i:03d}") + "ος"


def fake_document(monkeypatch, sentences, key="wikisource/test", author="Α",
                  more=()):
    """Serve the slice as one document of ``sentences`` (and the documents
    ``more``, as (key, author, sentences))."""
    import extract_polytonic_mg as E
    docs = [E.PolytonicMGDocument(k, a, "τίτλος", "1900",
                                  ". ".join(sents) + ".")
            for k, a, sents in [(key, author, sentences), *more]]
    monkeypatch.setattr(E, "iter_polytonic_mg_documents",
                        lambda *a, **k: iter(docs))


def sentences_300():
    """300 sentences; every third is too short to count but still takes a
    sentence number, as the language model numbers them."""
    out = []
    for i in range(300):
        if i % 3 == 1:
            out.append("Ναί")
        elif i == 0:
            out.append(f"{marker(i)} και μέρα")       # a monotonic signal
        elif i == 3:
            out.append(f"{marker(i)} τοῦ τόμ")        # τόμ. Β΄
        else:
            out.append(f"{marker(i)} καλὴ μέρα")
    return out


def test_the_dev_sentences_are_the_language_models_and_are_not_counted(
        monkeypatch):
    import extract_polytonic_mg as E
    import train_lm
    fake_document(monkeypatch, sentences_300())
    counts = mg.count_corpus()
    markers = {marker(i) for i in range(300)}
    dev_markers = {t.form for sents in counts.dev.values() for toks in sents
                   for t in toks} & markers
    lm_dev = {sid: toks for sid, toks in E.iter_polytonic_mg_sentences()
              if train_lm.sentence_goes_to_dev(sid)}
    # The same sentences, by the same numbering: the short ones count.
    assert {int(sid.rsplit(":", 1)[1]) for sid in lm_dev} == {92, 180, 273}
    assert dev_markers == {marker(i) for i in (92, 180, 273)}
    counted = {form for (form, _position) in counts.forms}
    assert not dev_markers & counted
    trained = {marker(i) for i in range(300)
               if i % 3 != 1 and i not in (0, 92, 180, 273)}
    assert trained <= counted


def test_dev_sentences_do_not_decide_whether_a_document_is_read(monkeypatch):
    # Monotonic dev sentences: 9 signal words, over 1% of the document's
    # words, but none in its training sentences.
    sentences = sentences_300()
    for i in (92, 180, 273):
        sentences[i] = f"{marker(i)} και να δεν"
    fake_document(monkeypatch, sentences)
    counts = mg.count_corpus()
    info = counts.documents[0]
    assert info.signals == 1            # sentence 0's και, a training one
    assert not info.monotonic
    assert mg.source_documents(counts) == {0}


def test_a_training_copy_of_a_dev_sentence_adds_nothing(monkeypatch):
    # Another document repeats dev sentence 92 word for word, beside a
    # sentence of its own.
    copy = [f"{marker(92)} καλὴ μέρα", "ἄλλη λέξη ἐδῶ", "Ναί"]
    fake_document(monkeypatch, sentences_300(),
                  more=[("wikisource/copy", "Β", copy)])
    counts = mg.count_corpus()
    counted = {form for (form, _position) in counts.forms}
    assert marker(92) not in counted
    assert counts.dev_repeats == 1
    assert {"ἄλλη", "λέξη", "ἐδῶ"} <= counted
    # Its lowercase words still count towards its document's share.
    assert counts.documents[1].words == 6


def test_a_two_word_sentence_is_too_short_to_count(monkeypatch):
    sentences = sentences_300()
    sentences[2] = f"{marker(2)} μέρα"
    fake_document(monkeypatch, sentences)
    counted = {form for (form, _position) in mg.count_corpus().forms}
    assert marker(2) not in counted and marker(5) in counted


def test_the_sentence_ids_and_the_split_are_stable():
    import train_lm
    from extract_polytonic_mg import sentence_id
    assert sentence_id("wikisource/test", 92) == "polymg:wikisource/test:92"
    assert [i for i in range(300) if train_lm.sentence_goes_to_dev(
        sentence_id("wikisource/test", i))] == [85, 92, 180, 273, 289]


def test_a_sentence_with_a_monotonic_signal_and_an_abbreviation_add_nothing(
        monkeypatch):
    fake_document(monkeypatch, sentences_300())
    counts = mg.count_corpus()
    counted = {form for (form, _position) in counts.forms}
    assert marker(0) not in counted and "και" not in counted
    assert counts.skipped_sentences == 1
    assert marker(3) in counted and "τόμ" not in counted


def test_held_out_documents_give_no_tokens_and_no_authors():
    counts = make_counts([
        ("τώρα", "lower", {0: 5, 1: 4, 2: 3}),
        ("Μαρούλα", "cap", {1: 6, 2: 2}),
    ])
    candidates = gather_candidates(counts, {0, 2})
    assert candidates["τώρα"] == Candidate("τώρα", 8, 2, 2)
    assert candidates["Μαρούλα"] == Candidate("Μαρούλα", 2, 1, 1)
    candidates = gather_candidates(counts, {0})
    assert candidates["τώρα"] == Candidate("τώρα", 5, 1, 1)
    assert "Μαρούλα" not in candidates


def test_author_folds_do_not_depend_on_the_process():
    # Python's own hash() of a string changes from run to run.
    assert [mg.author_fold(a, 5) for a in (
        "Αλέξανδρος Παπαδιαμάντης", "Κωστής Παλαμάς", "Γεώργιος Βιζυηνός",
        "Ανδρέας Καρκαβίτσας", "Διονύσιος Σολωμός")] == [1, 3, 3, 1, 4]


def test_a_short_document_with_a_slip_or_two_is_not_monotonic():
    info = DocumentInfo("d", "Α", "t", words=40, signals=2)
    assert not info.monotonic
    info.signals = 3
    assert info.monotonic
    # Exactly 1% of the words is not over it.
    assert not DocumentInfo("d", "Α", "t", words=300, signals=3).monotonic
    assert DocumentInfo("d", "Α", "t", words=299, signals=3).monotonic


def test_author_folds_are_deterministic():
    assert mg.author_fold("Αλέξανδρος Παπαδιαμάντης", 5) == mg.author_fold(
        "Αλέξανδρος Παπαδιαμάντης", 5)
    assert {mg.author_fold(str(i), 5) for i in range(50)} == set(range(5))


def test_the_list_is_written_in_the_grc_format(tmp_path):
    stats = mg.write_list({"τώρα": 1500, "σπίτι": 120, "κι": 5}, tmp_path,
                          variant="grc-mg", source="test",
                          version="9.9.9", commit="abc")
    lines = (tmp_path / "grc_mg_polytonic.dic").read_text("utf-8").splitlines()
    assert lines == ["3", "κι\tfr:R", "σπίτι\tfr:M", "τώρα\tfr:C"]
    version = (tmp_path / "grc_mg_polytonic.version").read_text("utf-8")
    assert "version: 9.9.9\n" in version and "entries: 3\n" in version
    assert "SET UTF-8" in (tmp_path / "grc_mg_polytonic.aff").read_text("utf-8")
    assert stats["entries"] == 3


def test_an_evaluation_list_never_replaces_the_shipping_list(tmp_path):
    with pytest.raises(ValueError, match="shipping list"):
        mg.write_list({"τώρα": 5}, mg.OUT, variant="grc-mg (evaluation)",
                      source="test")
    for argv in (["--holdout-author-fold", "0/5"],
                 ["--holdout-dev-documents", "--out-dir", str(mg.OUT)]):
        with pytest.raises(SystemExit) as exit_info:
            mg.main(argv)
        assert exit_info.value.code == 2
    mg.write_list({"τώρα": 5}, tmp_path, variant="grc-mg (evaluation)",
                  source="test", grc_sha256="abc123")
    version = (tmp_path / "grc_mg_polytonic.version").read_text("utf-8")
    assert "variant: grc-mg (evaluation)\n" in version
    assert "grc_dictionary_sha256: abc123\n" in version


def test_the_shipping_directory_is_known_by_any_spelling_of_its_path(
        tmp_path, monkeypatch):
    out = tmp_path / "build" / "hunspell"
    out.mkdir(parents=True)
    monkeypatch.setattr(mg, "OUT", out)
    (tmp_path / "link").symlink_to(out)
    for spelling in (out, tmp_path / "build" / "x" / ".." / "hunspell",
                     tmp_path / "link", tmp_path / "build" / "HUNSPELL"):
        assert mg.is_shipping_dir(spelling), spelling
        with pytest.raises(ValueError, match="shipping list"):
            mg.write_list({"τώρα": 5}, spelling,
                          variant="grc-mg (evaluation)", source="test")
    assert not mg.is_shipping_dir(tmp_path / "build" / "other")
    # macOS also reaches the data volume through /System/Volumes/Data,
    # which no resolve() undoes.
    firmlink = Path("/System/Volumes/Data" + str(out.resolve()))
    if firmlink.is_dir():
        assert mg.is_shipping_dir(firmlink)
    # Before build/hunspell exists.
    monkeypatch.setattr(mg, "OUT", tmp_path / "new" / "hunspell")
    assert mg.is_shipping_dir(tmp_path / "new" / "x" / ".." / "Hunspell")
    assert not list((tmp_path / "build" / "hunspell").iterdir())


def test_the_version_marks_a_build_from_uncommitted_code(monkeypatch):
    import subprocess
    monkeypatch.setattr(mg, "get_git_commit", lambda: "abc123")
    monkeypatch.setattr(subprocess, "check_output",
                        lambda cmd, **kw: b" M export_mg_polytonic.py\n")
    assert mg.build_commit() == "abc123-dirty"
    monkeypatch.setattr(subprocess, "check_output", lambda cmd, **kw: b"")
    assert mg.build_commit() == "abc123"


# --------------------------------------------------------------------------
# Spellings Modern Greek avoids (mg:avoid)
# --------------------------------------------------------------------------

def avoid_counts():
    return make_counts([
        # με beside μὲ: 6% of the letters' lowercase tokens.
        ("μὲ", "lower", {0: 500, 1: 400}),
        ("με", "lower", {0: 40, 1: 20}),
        ("Με", "cap", {0: 300}),           # capitals say nothing
        ("Μὲ", "initial", {0: 5000}),
        # που beside ποὺ and ποῦ.
        ("ποὺ", "lower", {0: 50}),
        ("ποῦ", "lower", {1: 40}),
        ("που", "lower", {0: 3}),
        # ἐκείνῃ beside ἐκείνη, the same word.
        ("ἐκείνη", "lower", {0: 300, 1: 300}),
        ("ἐκείνῃ", "lower", {0: 20, 1: 5}),
        # ὅ beside ὁ: another word (ὅ,τι).
        ("ὁ", "lower", {0: 900, 1: 900}),
        ("ὅ", "lower", {0: 60, 1: 30}),
        # The enclitic's second acute: γυναῖκά με.
        ("γυναῖκα", "lower", {0: 300, 1: 300}),
        ("γυναῖκά", "lower", {0: 20, 1: 10}),
        # σας is a weak pronoun, unaccented by rule, however rare.
        ("σᾶς", "lower", {0: 600}),
        ("σας", "lower", {1: 3}),
        # Elided spellings are not judged: ἄλλ᾽ is ἄλλο, not a slip.
        ("ἀλλ" + K, "lower", {0: 1000, 1: 900}),
        ("ἄλλ" + K, "lower", {0: 15, 1: 15}),
    ])


AVOID_GRC = {"με", "μὲ", "μέ", "που", "ποὺ", "ποῦ", "ἐκείνη", "ἐκείνῃ", "ὁ",
             "ὅ", "γυναῖκα", "γυναῖκά", "σᾶς", "σας", "ἀλλ" + K, "ἄλλ" + K}
KNOWN = {"με", "που", "ό", "ο", "γυναίκα"}.__contains__


def test_a_spelling_under_a_tenth_of_its_letters_is_avoided():
    avoided = mg.modern_greek_avoids(avoid_counts(), {0, 1, 2}, AVOID_GRC,
                                     known_word=KNOWN)
    assert set(avoided) == {"με", "που", "ἐκείνῃ"}
    assert avoided["με"] == mg.Avoided("με", 60, 960, "μέ")
    assert avoided["που"].preferred == "πού"
    # A held-out document gives the rule nothing.
    held_out = mg.modern_greek_avoids(avoid_counts(), {1, 2}, AVOID_GRC,
                                      known_word=KNOWN)
    assert held_out["με"] == mg.Avoided("με", 20, 420, "μέ")


def test_another_word_is_not_avoided_but_the_reviewed_weak_forms_are():
    report = Counter()
    avoided = mg.modern_greek_avoids(avoid_counts(), {0, 1, 2}, AVOID_GRC,
                                     known_word=KNOWN, report=report)
    # ὅ (ὅ,τι) is another word than ὁ; με and που are words too, but
    # reviewed weak forms Modern Greek writes accented.
    assert "ὅ" not in avoided and {"με", "που"} <= set(avoided)
    assert report["avoid_kept:different_word"] == 1
    assert report["avoid_kept:enclitic_accent"] == 1
    # Without the lexicon no spelling is shown to be another word.
    avoided = mg.modern_greek_avoids(avoid_counts(), {0, 1, 2}, AVOID_GRC)
    assert "ὅ" in avoided


def test_the_share_must_be_under_a_tenth_with_confidence():
    def avoided(n, total):
        counts = make_counts([("δύο", "lower", {0: total - n, 1: 1}),
                              ("δυό", "lower", {0: n})])
        return "δυό" in mg.modern_greek_avoids(counts, {0, 1}, {"δύο"})
    # 900 of 10,000 is under a tenth with 95% confidence, 950 is not.
    assert avoided(900, 10_000)
    assert not avoided(950, 10_000)
    # 9 of 100 is under a tenth, but not with confidence.
    assert not avoided(9, 100)
    assert mg.wilson_upper(900, 10_000) < 0.1 < mg.wilson_upper(950, 10_000)


def test_the_preferred_spelling_must_be_a_word_written_as_widely():
    def avoided(rows, grc, entries=None):
        return set(mg.modern_greek_avoids(make_counts(rows), {0, 1, 2}, grc,
                                          entries))
    rows = [("παππούς", "lower", {0: 300}), ("παπποῦς", "lower", {1: 2, 2: 2})]
    # Neither grc nor the list accepts παππούς.
    assert avoided(rows, {"παπποῦς"}) == set()
    # One author writes παππούς, two write παπποῦς.
    assert avoided(rows, {"παπποῦς", "παππούς"}) == set()
    rows[0] = ("παππούς", "lower", {0: 300, 1: 1})
    assert avoided(rows, {"παπποῦς", "παππούς"}) == {"παπποῦς"}
    # Written as widely, but neither grc nor the list accepts it.
    assert avoided(rows, {"παπποῦς"}) == set()
    # The list's own word is a word as well as grc's.
    assert avoided(rows, {"παπποῦς"}, {"παππούς": 300}) == {"παπποῦς"}


def test_a_spelling_the_list_carries_is_not_avoided():
    rows = [("ἐδῶ", "lower", {0: 1300, 1: 100}), ("ἐδώ", "lower", {1: 100})]
    counts = make_counts(rows)
    grc = {"ἐδῶ", "ἐδώ"}
    assert set(mg.modern_greek_avoids(counts, {0, 1}, grc)) == {"ἐδώ"}
    # The list carries ἐδὼ, the grave twin of grc's ἐδώ.
    assert mg.modern_greek_avoids(counts, {0, 1}, grc, {"ἐδὼ": 100}) == {}


def test_the_avoid_lines_are_read_from_the_lists_own_sources():
    counts = make_counts([
        ("μὲ", "lower", {0: 500, 1: 400}),
        ("με", "lower", {0: 40, 1: 20}),
        # A held-out author and a monotonic document, each of which would
        # lift με over a tenth.
        ("με", "lower", {2: 900}),
        ("με", "lower", {3: 900}),
    ], authors=("Α", "Β", "Ε", "Δ"))
    counts.documents[3].signals = 900
    fold = mg.author_fold("Ε", 5)
    sources = mg.source_documents(counts, holdout_author_fold=(fold, 5))
    assert sources == {0, 1}
    lst = mg.select_list(counts, sources, {"με", "μὲ"})
    assert lst.avoid == ["με"]


def test_the_list_names_the_grc_spellings_modern_greek_avoids():
    grc = AVOID_GRC | {"ἐκείνῃ"}
    lst = mg.select_list(avoid_counts(), {0, 1, 2}, grc, known_word=KNOWN)
    # grc's spellings; an elided or enclitic-accent spelling never.
    assert lst.avoid == ["με", "που", "ἐκείνῃ"]
    # A spelling grc lacks gets no line.
    lst = mg.select_list(avoid_counts(), {0, 1, 2}, AVOID_GRC - {"που"},
                         known_word=KNOWN)
    assert lst.avoid == ["με", "ἐκείνῃ"]
    # The contextual grave twin of an avoided acute is marked as well.
    counts = make_counts([("δύο", "lower", {0: 2000, 1: 1000}),
                          ("δυό", "lower", {0: 50, 1: 20})])
    lst = mg.select_list(counts, {0, 1}, {"δύο", "δυό", "δυὸ"})
    assert lst.avoid == ["δυό", "δυὸ"]


def test_avoid_lines_are_written_with_the_mg_avoid_field(tmp_path):
    mg.write_list({"τώρα": 1500}, tmp_path, variant="grc-mg", source="t",
                  avoid=["που", "με"])
    lines = (tmp_path / "grc_mg_polytonic.dic").read_text("utf-8").splitlines()
    assert lines == ["3", "με\tmg:avoid", "που\tmg:avoid", "τώρα\tfr:C"]
    version = (tmp_path / "grc_mg_polytonic.version").read_text("utf-8")
    assert "entries: 1\n" in version and "mg_avoid: 2\n" in version


# --------------------------------------------------------------------------
# Spellings that are also Ancient Greek words
# --------------------------------------------------------------------------

def form_profile(tmp_path, rows):
    """A form_profile.db with ``rows``: {form: source counts}."""
    import sqlite3
    db = tmp_path / "form_profile.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE forms (form_id INTEGER PRIMARY KEY, form TEXT, "
        "form_norm TEXT);"
        "CREATE TABLE form_profile (form_id INTEGER PRIMARY KEY, "
        "total_count INTEGER, n_works INTEGER, source_counts_json TEXT);"
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"
        "INSERT INTO meta VALUES ('content_hash', 'abc');")
    for i, (form, sources) in enumerate(rows.items(), 1):
        conn.execute("INSERT INTO forms VALUES (?, ?, ?)", (i, form, form))
        conn.execute("INSERT INTO form_profile VALUES (?, ?, 1, ?)",
                     (i, sum(sources.values()), json.dumps(sources)))
    conn.commit()
    conn.close()
    return mg.load_ancient_corpora(db)


def test_the_ancient_counts_are_the_treebanks_only(tmp_path):
    corpora = form_profile(tmp_path, {"ἐσὺ": {
        "byzantine_vernacular": 23, "pg": 40, "first1k": 30, "oga": 30,
        "glaux": 1, "diorisis": 2}})
    assert corpora.identity == "form_profile.db abc"
    assert corpora.tokens("ἐσὺ") == 3
    assert corpora.tokens("ἐσύ") == 0
    assert mg.load_ancient_corpora(tmp_path / "missing.db") is None


def test_an_ancient_greek_word_is_known_by_the_treebanks_or_its_grc_twin(
        tmp_path):
    corpora = form_profile(tmp_path, {
        "του": {"glaux": 542, "diorisis": 362, "pg": 2454},
        "τὴ": {"glaux": 15, "diorisis": 11, "pg": 1067},
        # A grc twin the treebanks attest: τόν inside a sentence.
        "τὸν": {"pg": 3}, "τόν": {"glaux": 15, "diorisis": 5},
        # An entry and a grc twin that only OCR'd editions attest.
        "στὴν": {"first1k": 40, "pg": 29, "oga": 38},
        "στήν": {"first1k": 2, "pg": 15, "oga": 3, "glaux": 1},
        "θὲ": {"pg": 120, "first1k": 35},
        "κυρὰ": {"byzantine_vernacular": 11, "pg": 20},
        "κυρά": {"glaux": 8, "byzantine_vernacular": 4, "pg": 2},
    })
    grc = {"τόν", "στήν", "κυρά"}
    assert mg.ancient_word("του", grc, corpora)
    assert mg.ancient_word("τὴ", grc, corpora)
    assert mg.ancient_word("τὸν", grc, corpora)
    assert not mg.ancient_word("τὸν", set(), corpora)
    # A twin attested only by OCR sources does not cap, nor do OCR or
    # vernacular tokens of the entry itself.
    assert not mg.ancient_word("στὴν", grc, corpora)
    assert not mg.ancient_word("θὲ", grc, corpora)
    assert not mg.ancient_word("κυρὰ", grc, corpora)


class FakeCorpora:
    identity = "form_profile.db test"

    def __init__(self, tokens):
        self._tokens = tokens       # form -> treebank tokens

    def tokens(self, form):
        return self._tokens.get(form, 0)


def test_an_ancient_word_is_written_with_fr_r(tmp_path):
    counts = make_counts([
        ("του", "lower", {0: 3000, 1: 2000}),
        ("τώρα", "lower", {0: 1000, 1: 500}),
        ("σπίτι", "lower", {0: 10, 1: 5}),
    ])
    corpora = FakeCorpora({"του": 904, "σπίτι": 50})
    lst = mg.select_list(counts, {0, 1}, set(), ancient_corpora=corpora)
    # σπίτι is fr:R anyway.
    assert lst.ancient == {"του"}
    mg.write_selection(lst, tmp_path, variant="grc-mg", source="t",
                       ancient_corpora=corpora)
    lines = (tmp_path / "grc_mg_polytonic.dic").read_text("utf-8").splitlines()
    assert lines == ["3", "σπίτι\tfr:R", "του\tfr:R", "τώρα\tfr:C"]
    version = (tmp_path / "grc_mg_polytonic.version").read_text("utf-8")
    assert "ancient_corpora: form_profile.db test\n" in version
    assert "ancient_capped: 1\n" in version
    assert "buckets: C=1 M=0 R=2\n" in version


# --------------------------------------------------------------------------
# Modern Greek elisions
# --------------------------------------------------------------------------

def test_modern_greek_elisions_pair_each_elided_spelling_with_its_word():
    counts = make_counts([
        ("τώρ" + K, "lower", {0: 8, 1: 4}),
        ("τώρα", "lower", {0: 50, 1: 30}),
        ("στ" + K, "lower", {0: 8, 1: 4}),
        ("στὸ", "lower", {0: 60, 1: 40}),
        ("στὰ", "lower", {0: 30, 1: 10}),
        # The aspirated preposition beside its plain spelling, and a rare
        # word that would otherwise be read as its full form.
        ("καθ" + K, "lower", {0: 15, 1: 15}),
        ("κατ" + K, "lower", {0: 15, 1: 15}),
        ("καθὸ", "lower", {0: 3, 1: 2}),
        # Too rare to enter.
        ("ὅλ" + K, "lower", {0: 5, 1: 4}),
        ("ὅλα", "lower", {0: 50, 1: 30}),
    ])
    counts.before_vowel[nfc("τώρ" + K)].update({0: 8, 1: 4})
    counts.before_vowel["τώρα"].update({0: 30, 1: 6})
    # στ᾽ stands for στὸ and στὰ alike: its share is of both.
    counts.before_vowel[nfc("στ" + K)].update({0: 6})
    counts.before_vowel["στὸ"].update({0: 2})
    counts.before_vowel["στὰ"].update({1: 4})
    pairs, found = modern_greek_elisions(counts, {0, 1, 2})
    assert pairs == {"τώρα": "τώρ" + K, "στὸ": "στ" + K, "κι": "κι" + K}
    rows = {f.full: f for f in found}
    assert rows["τώρα"].elided_tokens == 12
    assert rows["τώρα"].elided_before_vowel == 12
    assert mg.modern_greek_elision_shares(found) == {
        "τώρα": 0.25, "στὸ": 0.5, "κι": 0.0}


def test_an_aspirated_preposition_is_left_to_its_plain_elision():
    counts = make_counts([
        ("ἀφ" + K, "lower", {0: 10, 1: 10}),
        ("ἀπ" + K, "lower", {0: 10, 1: 10}),
        # A word ἀφ᾽ could stand for, commoner than it.
        ("ἀφοῦ", "lower", {0: 60, 1: 40}),
    ])
    pairs, _ = modern_greek_elisions(counts, {0, 1})
    assert "ἀφοῦ" not in pairs


def test_the_share_counts_only_the_words_the_elision_stands_for():
    counts = make_counts([
        ("τ" + K, "lower", {0: 20, 1: 10}),
        ("τὸ", "lower", {0: 300, 1: 200}),
        ("τὰ", "lower", {0: 100, 1: 100}),
        # The article's genitive and datives: τ᾽ is none of them.
        ("τοῦ", "lower", {0: 200, 1: 200}),
        ("τῇ", "lower", {0: 40, 1: 40}),
        ("τῷ", "lower", {0: 40, 1: 40}),
        ("τᾶ", "lower", {0: 40, 1: 40}),
    ])
    counts.before_vowel[nfc("τ" + K)].update({0: 10})
    counts.before_vowel["τὸ"].update({0: 12})
    counts.before_vowel["τὰ"].update({1: 8})
    counts.before_vowel["τοῦ"].update({0: 100})
    counts.before_vowel["τῇ"].update({0: 50})
    counts.before_vowel["τῷ"].update({1: 50})
    counts.before_vowel["τᾶ"].update({1: 50})
    pairs, found = modern_greek_elisions(counts, {0, 1})
    assert pairs["τὸ"] == "τ" + K
    # 10 of the 30 times τὸ or τὰ stands before a vowel, to three places.
    assert mg.modern_greek_elision_shares(found)["τὸ"] == 0.333
    words = mg.FullForms(["τὸ", "τὰ", "τοῦ", "τῇ", "τῷ", "τᾶ"])
    assert words.of("τ" + K) == ["τοῦ", "τὰ", "τὸ", "τᾶ"]


def test_an_elided_spelling_needs_ten_tokens_to_enter():
    def paired(n):
        counts = make_counts([
            ("ὅλ" + K, "lower", {0: n - 1, 1: 1}),
            ("ὅλα", "lower", {0: 50, 1: 30}),
        ])
        return "ὅλα" in modern_greek_elisions(counts, {0, 1})[0]
    assert paired(10) and not paired(9)


def test_the_full_form_must_be_at_least_as_common_as_the_elided_one():
    counts = make_counts([
        ("ἀμ" + K, "lower", {0: 10, 1: 10}),
        ("ἀμὴ", "lower", {0: 3, 1: 3}),
    ])
    pairs, _ = modern_greek_elisions(counts, {0, 1})
    assert "ἀμὴ" not in pairs
    # As common is common enough.
    counts.forms[("ἀμὴ", "lower")].update({0: 7, 1: 7})
    pairs, _ = modern_greek_elisions(counts, {0, 1})
    assert pairs["ἀμὴ"] == "ἀμ" + K


def test_an_elision_and_its_word_agree_on_the_lost_accent():
    # An unaccented elided spelling stands for a word accented on the vowel
    # it lost, not for a misspelling without the accent (εἰναι, ᾽να).
    full = mg.FullForms(["γιο", "γιὰ", "εἰναι", K + "να"])
    assert full.of("γι" + K) == ["γιὰ"]
    assert full.of("εἰν" + K) == [] and full.of(K + "ν" + K) == []


def test_an_elision_needs_a_word_written_by_two_authors():
    candidates = {c.form: c for c in [cand("γι" + K, 10), cand("γιὰ", 5, 1)]}
    sel = select_forms(candidates, set())
    assert sel.entries == {} and sel.report["unaccented_elision"] == 1
    candidates["γιὰ"] = cand("γιὰ", 5, 2)
    assert "γι" + K in select_forms(candidates, set()).entries


def test_a_circumflex_stays_within_the_last_two_syllables():
    assert mg_orthography_reason("πῶλησε") is not None
    assert mg_orthography_reason("πώλησε") is None


def test_a_diaeresis_keeps_its_vowel_a_syllable():
    # κά-λα-ϊ-α: the ι with a diaeresis is no glide.
    assert not mg.mg_accent_window_ok("κάλαϊα")
    assert mg.mg_accent_window_ok("κάλαια")


def test_letters_spread_over_many_spellings_avoid_none():
    # Eleven spellings of the same letters, none taking a tenth.
    spellings = ["α", "ἀ", "ἁ", "ἄ", "ἅ", "ἆ", "ἇ", "ά", "ᾶ", "ᾳ", "ᾷ"]
    counts = make_counts([(sp, "lower", {0: 1000, 1: 1}) for sp in spellings])
    assert mg.modern_greek_avoids(counts, {0, 1}, set(spellings)) == {}


def test_an_unaccented_monosyllable_is_looked_up_without_its_accent(tmp_path):
    import sqlite3
    db = tmp_path / "lookup.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE lookup (form TEXT, lemma_id INTEGER, "
                 "src TEXT, lang TEXT)")
    conn.executemany("INSERT INTO lookup VALUES (?, 1, 'x', 'el')",
                     [("μια",), ("χρονιά",)])
    conn.commit()
    conn.close()
    known = mg.load_known_words(db)
    # Monotonic writing leaves the accent off a monosyllable.
    assert known("μιά") and known("χρονιά")
    assert not known("χρόνια") and not known("βιά")


def test_a_final_sigma_written_medial_competes_with_the_word():
    counts = make_counts([("πῶς", "lower", {0: 900, 1: 100}),
                          ("πῶσ", "lower", {0: 10})])
    assert mg.letters_key("πῶς") == mg.letters_key("πῶσ")
    assert set(mg.modern_greek_avoids(counts, {0, 1}, {"πῶς"})) == {"πῶσ"}


def test_an_enclitic_accent_form_without_its_plain_spelling_is_judged():
    # γυναῖκά stands beside the monotonic-accented γυναίκα, not beside
    # γυναῖκα, so the enclitic's accent does not explain it.
    counts = make_counts([("γυναίκα", "lower", {0: 900, 1: 100}),
                          ("γυναῖκά", "lower", {0: 20, 1: 10})])
    avoided = mg.modern_greek_avoids(counts, {0, 1}, {"γυναίκα"})
    assert set(avoided) == {"γυναῖκά"}


def test_the_tokens_before_a_vowel_are_counted_for_the_elisions(monkeypatch):
    sentences = sentences_300()
    sentences[2] = f"{marker(2)} τώρα ἔλα τώρα ᾽ναι τώρα καλή"
    fake_document(monkeypatch, sentences)
    counts = mg.count_corpus()
    # Only the τώρα before ἔλα: ᾽ναι starts with its mark, καλή with a
    # consonant.
    assert sum(counts.before_vowel["τώρα"].values()) == 1


def test_a_twin_that_is_not_well_formed_is_not_added():
    sel = select_forms({"γιὰ": cand("γιὰ", 3500)}, grc_words=set())
    assert sel.entries == {"γιὰ": 3500}
    assert sel.report["twin_rejected"] == 1


# --------------------------------------------------------------------------
# Built artifacts
# --------------------------------------------------------------------------

HUNSPELL = ROOT / "build" / "hunspell"
MG_DIC = HUNSPELL / "grc_mg_polytonic.dic"
GRC_DIC = HUNSPELL / "grc_polytonic.dic"


def read_list(path: Path) -> tuple[dict[str, str], list[str]]:
    """The list's words with their fr: field, and its mg:avoid spellings."""
    lines = path.read_text("utf-8").splitlines()
    assert int(lines[0]) == len(lines) - 1
    assert lines[1:] == sorted(lines[1:])
    out: dict[str, str] = {}
    avoid: list[str] = []
    for line in lines[1:]:
        form, _, field = line.partition("\t")
        assert form == nfc(form) and form not in out and form not in avoid
        if field == "mg:avoid":
            avoid.append(form)
            continue
        assert field in ("fr:C", "fr:M", "fr:R"), line
        out[form] = field
    return out, avoid


@pytest.mark.skipif(not MG_DIC.exists() or not GRC_DIC.exists(),
                    reason="grc_mg_polytonic.dic not built")
def test_the_built_list_holds_its_invariants():
    from export_lm import read_hunspell_words
    entries, avoid = read_list(MG_DIC)
    grc = read_hunspell_words(GRC_DIC)
    assert not [f for f in entries if mg.grc_accepts(f, grc)]
    # The mg:avoid lines name grc spellings: με and που, which Modern Greek
    # writes μὲ and ποὺ or ποῦ, but not μὲ or ποὺ themselves, nor another
    # word (ὅ, as in ὅ,τι; the numeral ἕν), nor an enclitic-accent form.
    assert set(avoid) <= grc
    assert {"με", "που", "σε"} <= set(avoid)
    assert not {"μὲ", "ποὺ", "ποῦ", "του", "κι", "ὅ", "ἕν", "ἓν",
                "γυναῖκά"} & set(avoid)
    # Nor the contextual twin of a listed spelling.
    from export_hunspell import contextual_acute
    listed = {contextual_acute(f) for f in entries}
    assert not [f for f in avoid if contextual_acute(f) in listed]
    lexicon = set(entries)
    invalid = [f for f in entries if mg_orthography_reason(f, lexicon)]
    assert not invalid, invalid[:20]
    # The core function words of polytonic Modern Greek are accepted:
    # by the list, or already by grc.
    core = ["του", "της", "των", "τους", "μας", "σας", "μου", "σου", "κι",
            "κι" + K, "θὰ", "νὰ", "γιὰ", "στὸ", "στὴ", "στὴν", "στὰ",
            "στὸν", "στοὺς", "δὲν", "ἀπ" + K, "γι" + K, "τώρα", "ἀκόμη",
            "σπίτι", "ὄχι"]
    missing = [w for w in core if w not in entries and w not in grc]
    assert not missing, missing
    # Unaccented article and particle spellings stay misspellings.
    assert not {"το", "τα", "την", "και", "να", "θα", "δεν"} & set(entries)
    version = MG_DIC.with_suffix(".version").read_text("utf-8")
    assert f"entries: {len(entries)}\n" in version
    assert f"mg_avoid: {len(avoid)}\n" in version
    # The shipping list, not an evaluation variant, and selected against
    # the grc dictionary beside it.
    assert "variant: grc-mg\n" in version
    import hashlib
    grc_sha = hashlib.sha256(GRC_DIC.read_bytes()).hexdigest()
    assert f"grc_dictionary_sha256: {grc_sha}\n" in version
    # του is the Ancient enclitic genitive too: no sign of Modern Greek.
    assert entries["του"] == "fr:R" and entries["τώρα"] == "fr:C"
    assert "ancient_corpora: form_profile.db " in version
    # No spelling a recorded review rejected, nor its contextual twin.
    rejects = {contextual_acute(f) for f in mg.load_mg_spelling_review()}
    assert not [f for f in entries if contextual_acute(f) in rejects]

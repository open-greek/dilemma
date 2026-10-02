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


def fake_document(monkeypatch, sentences, key="wikisource/test", author="Α"):
    import extract_polytonic_mg as E
    doc = E.PolytonicMGDocument(key, author, "τίτλος", "1900",
                                ". ".join(sentences) + ".")
    monkeypatch.setattr(E, "iter_polytonic_mg_documents",
                        lambda *a, **k: iter([doc]))


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
    pairs, found = modern_greek_elisions(counts, {0, 1, 2})
    assert pairs == {"τώρα": "τώρ" + K, "στὸ": "στ" + K, "κι": "κι" + K}
    rows = {f.full: f for f in found}
    assert rows["τώρα"].elided_tokens == 12
    assert rows["τώρα"].elided_before_vowel == 12


def test_an_aspirated_preposition_is_left_to_its_plain_elision():
    counts = make_counts([
        ("ἀφ" + K, "lower", {0: 10, 1: 10}),
        ("ἀπ" + K, "lower", {0: 10, 1: 10}),
        # A word ἀφ᾽ could stand for, commoner than it.
        ("ἀφοῦ", "lower", {0: 60, 1: 40}),
    ])
    pairs, _ = modern_greek_elisions(counts, {0, 1})
    assert "ἀφοῦ" not in pairs


def test_the_full_form_must_be_at_least_as_common_as_the_elided_one():
    counts = make_counts([
        ("ἀμ" + K, "lower", {0: 10, 1: 10}),
        ("ἀμὴ", "lower", {0: 3, 1: 3}),
    ])
    pairs, _ = modern_greek_elisions(counts, {0, 1})
    assert "ἀμὴ" not in pairs


# --------------------------------------------------------------------------
# Built artifacts
# --------------------------------------------------------------------------

HUNSPELL = ROOT / "build" / "hunspell"
MG_DIC = HUNSPELL / "grc_mg_polytonic.dic"
GRC_DIC = HUNSPELL / "grc_polytonic.dic"


def read_list(path: Path) -> dict[str, str]:
    lines = path.read_text("utf-8").splitlines()
    assert int(lines[0]) == len(lines) - 1
    out = {}
    for line in lines[1:]:
        form, _, field = line.partition("\t")
        assert field in ("fr:C", "fr:M", "fr:R"), line
        assert form == nfc(form) and form not in out, form
        out[form] = field
    assert list(out) == sorted(out)
    return out


@pytest.mark.skipif(not MG_DIC.exists() or not GRC_DIC.exists(),
                    reason="grc_mg_polytonic.dic not built")
def test_the_built_list_holds_its_invariants():
    from export_lm import read_hunspell_words
    entries = read_list(MG_DIC)
    grc = read_hunspell_words(GRC_DIC)
    assert not set(entries) & grc
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
    # The shipping list, not an evaluation variant, and selected against
    # the grc dictionary beside it.
    assert "variant: grc-mg\n" in version
    import hashlib
    grc_sha = hashlib.sha256(GRC_DIC.read_bytes()).hexdigest()
    assert f"grc_dictionary_sha256: {grc_sha}\n" in version
    # No spelling a recorded review rejected, nor its contextual twin.
    from export_hunspell import contextual_acute
    rejects = {contextual_acute(f) for f in mg.load_mg_spelling_review()}
    assert not [f for f in entries if contextual_acute(f) in rejects]

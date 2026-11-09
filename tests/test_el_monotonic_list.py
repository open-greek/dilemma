"""Spelling rules, syllable counting, rating edges, admission routes and the
output format of the monotonic Modern Greek word list
(build/el_lexicon.py, build/el_type_counts.py, export_el_monotonic.py).

Every input is a small file written by the test: a few Wiktionary rows, a
lookup.db with three rows, a FrequencyWords file and type-count tables
with chosen counts. Nothing is downloaded and no corpus is read.

Run with:
    python -m pytest tests/test_el_monotonic_list.py -x -v
"""

import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import export_el_monotonic as ex  # noqa: E402
from dilemma_build import el_lexicon as L  # noqa: E402
from dilemma_build import el_type_counts as TC  # noqa: E402

A = L.APOSTROPHE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -- spelling ----------------------------------------------------------------

@pytest.mark.parametrize("form", [
    "είναι", "ή", "πού", "πώς", "ποιος", "μια", "γεια", "σπίτια", "κορόιδο",
    "προϊόν", "ταΐζω", "Ευρωπαϊκή", "τσάι", "ρολόι", "Αθήνα", "άνθρωπος",
    "και", "να", "σ" + A, "απ" + A, "είν" + A, A + "ναι", A + "θελες",
])
def test_spelling_accepts(form):
    assert L.orthography_reason(form) is None


@pytest.mark.parametrize("form,reason", [
    ("ναί", "accented_monosyllable"),
    ("δέν", "accented_monosyllable"),
    ("θά", "accented_monosyllable"),
    ("ποιός", "accented_synizesis"),
    ("μιά", "accented_synizesis"),
    ("όνομά", "double_accent"),
    ("ειναι", "unaccented_polysyllable"),
    ("vα", "latin"),
    ("ϊσος", "dialytika_without_vowel_before"),
    ("τσάϊ", "unneeded_dialytika"),
    ("πρωϊνό", "unneeded_dialytika"),
    ("ΗΠΑ", "all_caps"),
    ("καλόσ", "final_sigma"),
    ("καλςο", "medial_final_sigma"),
    ("ἄνθρωπος", "not_monotonic_greek"),
    ("έυκολο", "accent_too_early"),
    ("ναιιι", "repeated_letters"),
    ("μεταξ" + A, "unaccented_elision"),
    ("απ" + A + "το", "apostrophe_inside"),
    ("ΑθήΝα", "mixed_case"),
])
def test_spelling_rejects(form, reason):
    assert L.orthography_reason(form) == reason


def test_one_letter_elisions_need_the_mark():
    assert L.orthography_reason("σ" + A) is None
    assert L.orthography_reason("σ") == "final_sigma"
    assert L.orthography_reason("τ") == "no_vowel"


def test_lexicon_exceptions_for_synizesis_and_loanwords():
    wiktionary = {"ποιος", "μια"}
    assert ex._spelling_ok("ιός", wiktionary) is None          # no ιος twin
    assert ex._spelling_ok("ποιός", wiktionary) == "accented_synizesis"
    assert ex._spelling_ok("κάμεραμαν", wiktionary) is None
    assert L.orthography_reason("κάμεραμαν") == "accent_too_early"


# -- syllables ---------------------------------------------------------------

@pytest.mark.parametrize("word,syllables", [
    ("κορόιδο", 3), ("χάιδεψε", 3), ("ρολόι", 3), ("τσάι", 2), ("Μάιος", 2),
    ("σπίτια", 2), ("βοήθεια", 3), ("μια", 1), ("μία", 2), ("ποιος", 1),
    ("γεια", 1), ("πιο", 1), ("δυο", 1), ("προϊόν", 3), ("άνθρωπος", 3),
    ("υιός", 2), ("ουρανός", 3), ("αυτοκίνητο", 5),
])
def test_syllable_count_with_synizesis(word, syllables):
    assert L.syllable_count(word) == syllables


def test_accent_position():
    assert L.accent_position("άνθρωπος") == 3
    assert L.is_proparoxytone("άνθρωπος")
    assert L.accent_position("σπίτια") == 2
    assert L.accent_position("ουρανός") == 1


@pytest.mark.parametrize("a,b,same", [
    ("κάθισε", "κάθησε", True), ("γλυκιά", "γλυκειά", True),
    ("νιώθεις", "νοιώθεις", True), ("σέξι", "σέξυ", True),
    ("Ελλάδα", "Ελάδα", True), ("τσάι", "τσε", False),
    ("προϊόν", "προιόν", False),
])
def test_sound_key(a, b, same):
    assert (L.sound_key(a) == L.sound_key(b)) is same


def test_slip_keys_cover_one_typing_slip():
    keys = ex.slip_keys("ειναι")
    assert {"εινα", "ειαναι", "ιεναι", "ειναιη", "εικαι"} <= keys
    assert "ειναι" not in keys and "ενα" not in keys


def test_bare_key_is_the_stripped_key():
    assert L.bare_key("Προϊόντα") == "προιοντα"
    assert L.bare_key("καλός") == "καλοσ"


# -- ratings -----------------------------------------------------------------

@pytest.mark.parametrize("rate,bucket", [
    (1000.0, "C"), (37.0, "C"), (36.999, "M"), (3.7, "M"), (3.699, "R"),
    (0.0001, "R"), (0.0, "X"),
])
def test_bucket_edges(rate, bucket):
    assert L.bucket_for_rate(rate) == bucket


def test_rate_per_million():
    assert L.rate_per_million(37, 1_000_000) == 37.0
    assert L.bucket_for_rate(L.rate_per_million(37, 1_000_000)) == "C"
    assert L.bucket_for_rate(L.rate_per_million(36, 1_000_000)) == "M"


@pytest.mark.parametrize("form,twin,lemma,twin_lemma,shared,distinct", [
    ("πινάκες", "πίνακες", "πινάκα", "πίνακας", "πινάκων", None),
    ("κάνονες", "κανόνες", "κάνονας", "κανόνας", "κανόνων", None),
    ("σχεδία", "σχέδια", "σχεδία", "σχέδιο", None, "σχεδίες"),
    ("οροί", "όροι", "ορός", "όρος", None, "ορός"),
    ("αγωνιά", "αγωνία", "αγωνιώ", "αγωνία", None, "αγωνιώ"),
    ("πολίτικων", "πολιτικών", "πολίτικος", "πολιτικός", None, "πολίτικη"),
])
def test_accent_alternatives_keep_valid_rare_words(form, twin, lemma, twin_lemma,
                                                  shared, distinct):
    groups = {lemma: {form: "wiktionary"}, twin_lemma: {twin: "wiktionary"}}
    rows = {form: (30, 0, 30), twin: (5000, 0, 100)}
    sel = ex.Selection()
    sel.entries[form] = ex.Entry(form, "paradigm", "R", "test")
    sel.entries[twin] = ex.Entry(twin, "paradigm", "C", "test")
    if shared:
        groups[lemma][shared] = groups[twin_lemma][shared] = "wiktionary"
        sel.entries[shared] = ex.Entry(shared, "paradigm", "C", "test")
    if distinct:
        groups[lemma][distinct] = "wiktionary"
        sel.entries[distinct] = ex.Entry(distinct, "paradigm", "M", "test")
    lexicon = ex.Lexicon(groups, set(sel.entries))
    evidence = ex.Evidence({form: 1, twin: 5000}, [table("open", rows)])
    marked = ex.accent_alternatives_to_mark(sel, evidence, lexicon)
    assert (marked.get(form) == (twin,)) is (distinct is None)
    assert form in sel.entries     # metadata must never delete a valid spelling
    if form in marked:
        sel.entries[form].accent_alternatives = marked[form]
        assert f"{form}\tfr:R ac:{twin}" in ex.dic_lines(sel.entries)


def test_accent_alternatives_keep_every_common_reading():
    sel = ex.Selection(entries={w: ex.Entry(w, "paradigm", bucket, "test")
                               for w, bucket in [("αδικά", "R"), ("άδικα", "C"),
                                                  ("αδίκα", "C")]})
    evidence = ex.Evidence({"αδικά": 1, "άδικα": 1000, "αδίκα": 1000},
                          [table("open", {w: (n, 0, 30) for w, n in
                                           [("αδικά", 1), ("άδικα", 1000),
                                            ("αδίκα", 1000)]})])
    assert ex.accent_alternatives_to_mark(sel, evidence, ex.Lexicon({}, set())) == {
        "αδικά": ("άδικα", "αδίκα")}


def test_reviewed_variants_keep_frequency_and_acceptance_with_preferred_spellings():
    forms = {"τσαι": "R", "τσάι": "C", "γινεί": "R", "γίνει": "C",
             "διευκρίνηση": "M", "διευκρίνιση": "M"}
    selection = ex.Selection(entries={form: ex.Entry(form, "paradigm", bucket, "test")
                                      for form, bucket in forms.items()})
    reviewed = ex.load_review(ex.REVIEW)["preferred_spellings"]
    audit = ex.apply_preferred_spellings(selection, reviewed)
    assert {form: entry.bucket for form, entry in selection.entries.items()} == forms
    lines = ex.dic_lines(selection.entries)
    assert "τσαι\tfr:R ac:τσάι us:dialectal" in lines
    assert "γινεί\tfr:R ac:γίνει us:rare" in lines
    assert "διευκρίνηση\tfr:M sp:διευκρίνιση us:archaic_spelling" in lines
    assert len(audit) == 3 and all(row["sources"] for row in audit)
    del selection.entries["γίνει"]
    row = next(row for row in ex.apply_preferred_spellings(selection, reviewed)
               if row["form"] == "γινεί")
    assert row["not_applied"] == "preferred spelling lacks C/M evidence"


def test_reviewed_dual_reading_acronyms_need_two_independent_sources():
    rows = {"ΕΚ": (100, 0, 50), "εκ": (1000, 0, 100),
            "ΠΑ": (30, 0, 15), "πα": (1000, 0, 100),
            "ΤΟ": (100, 0, 50), "το": (10000, 0, 100)}
    first, second = table("parl", rows), table("wiki", rows)
    accepts = ex.load_review(ex.REVIEW)["acronym_accepts"]
    entries = {"εκ", "πα", "το"}
    kept, audit = ex.select_acronyms(ex.Evidence({}, [first, second]), entries, {},
                                     accepts=accepts)
    assert kept == ["ΕΚ", "ΠΑ"]
    assert all(row["attested_sources"] == ["parl", "wiki"] for row in audit)
    for weaker in ([first], [first, first]):
        kept, _ = ex.select_acronyms(ex.Evidence({}, weaker), entries, {}, accepts=accepts)
        assert kept == []
    kept, _ = ex.select_acronyms(ex.Evidence({}, [first, second]), entries,
                                {"ΕΚ": "fixture reject"}, accepts=accepts)
    assert kept == ["ΠΑ"]


def test_uppercase_frequency_does_not_hide_attested_lowercase_abbreviations():
    rows = {"Ε.Ε.": (100, 0, 50), "ε.ε.": (20, 0, 10),
            "Α.Ε.": (20, 0, 10), "α.ε.": (2, 0, 2),
            "Δ.Σ.": (100, 0, 50), "δ.σ.": (15, 0, 5),
            "Α.": (100, 0, 50), "α.": (20, 0, 10),
            "Γ.": (100, 0, 50), "Ξ.Ω.": (9, 0, 9)}
    evidence = ex.Evidence({}, [table("parl", rows)])
    accepts = ex.load_review(ex.REVIEW)["abbreviation_accepts"]
    kept, audit = ex.select_abbreviations(evidence, Counter(), set(), {}, accepts=accepts)
    assert kept == ["δ.σ.", "ε.ε."]
    assert not any("ς" in abbreviation for abbreviation in kept)
    row = next(row for row in audit if row["abbreviation"] == "ε.ε.")
    assert row["spelling_tokens"] == 20 and row["spelling_docs"] == 10
    kept, _ = ex.select_abbreviations(evidence, Counter(), set(),
                                      {"ε.ε.": "fixture reject"}, accepts=accepts)
    assert kept == ["δ.σ."]
    weak = ex.Evidence({}, [table("parl", {"Ε.Ε.": (100, 0, 50), "ε.ε.": (9, 0, 9)})])
    assert ex.select_abbreviations(weak, Counter(), set(), {}, accepts=accepts)[0] == []


def test_subtitles_do_not_veto_acronyms_confirmed_by_two_sources():
    # Subtitle counts fold case, so they cannot separate an acronym from a
    # spoken word. Two independently cased sources supply that evidence.
    rows = {"ΝΑΤΟ": (200, 0, 100), "νάτο": (1, 0, 1)}
    tables = [table("wiki", rows), table("parl", rows)]
    evidence = ex.Evidence({"νάτο": 900_000}, tables)
    kept, audit = ex.select_acronyms(evidence, {"νάτο"}, {})
    assert kept == ["ΝΑΤΟ"]
    assert audit[0]["capitalized_sources"] == ["parl", "wiki"]
    # One source (even repeated) is not independent corroboration.
    for weaker in ([tables[0]], [tables[0], tables[0]]):
        kept, _ = ex.select_acronyms(ex.Evidence({"νάτο": 900_000}, weaker), {"νάτο"}, {})
        assert kept == []


def test_open_corpus_words_need_independent_spread_and_spelling_filters():
    words = ("φυτοπροστατευτικών", "καταταγέντων", "έχειν", "πρέπεινα")
    rows = {w: (50, 0, 12) for w in words}
    rows["πρέπει"] = (1000, 0, 100)
    rows["μονοπωλιακών"] = (100, 0, 30)
    second = {w: v for w, v in rows.items() if w != "μονοπωλιακών"}
    evidence = ex.Evidence({}, [table("wiki", rows), table("parl", second)])
    lexicon = ex.Lexicon({"πρέπει": {"πρέπει": "wiktionary"}}, {"πρέπει"})
    sel = ex.select(lexicon, evidence, function_words={"να": "fixture"}, review={"rejects": {}},
                    reviewed_loanwords=[])
    assert {"φυτοπροστατευτικών", "καταταγέντων"} <= set(sel.entries)
    assert not {"έχειν", "πρέπεινα", "μονοπωλιακών"} & set(sel.entries)
    assert sel.rejected["open_corpus_katharevousa"] == 1
    assert sel.rejected["run_together_words"] == 1
    assert ex.open_corpus_candidates(ex.Evidence({}, [table("wiki", rows)])) == set()
    assert ex.open_corpus_candidates(ex.Evidence({}, [table("wiki", rows),
                                                      table("wiki", rows)])) == set()
    # A formal word can be concentrated in one register, but it still needs
    # independent occurrences in at least five documents of another source.
    primary = table("opengov", {"καταταγέντων": (423, 0, 298)})
    secondary = table("parliament", {"καταταγέντων": (6, 0, 6)})
    assert "καταταγέντων" in ex.open_corpus_candidates(ex.Evidence({}, [primary, secondary]))
    weak = table("parliament", {"καταταγέντων": (4, 0, 4)})
    assert "καταταγέντων" not in ex.open_corpus_candidates(ex.Evidence({}, [primary, weak]))


def test_acronym_confidence_is_pooled_after_independent_source_confirmation():
    # ΝΕΤ has a 10:1 majority in two sources, and strong pooled confidence.
    # A separate Wilson test on each source would incorrectly reject it.
    evidence = ex.Evidence({"νετ": 129}, [
        table("wiki", {"ΝΕΤ": (145, 0, 111), "Νετ": (6, 0, 4), "νετ": (4, 0, 4)}),
        table("parl", {"ΝΕΤ": (225, 0, 176)})])
    kept, audit = ex.select_acronyms(evidence, {"νετ"}, {})
    assert kept == ["ΝΕΤ"]
    assert audit[0]["capitalized_sources"] == ["parl", "wiki"]


def test_open_only_route_excludes_obsolete_inflections_but_keeps_modern_words():
    known = {L.bare_key(w): [w] for w in ("κατάσταση", "μάχη", "αρμοδιότητας",
                                         "μίσθωσης", "πατρίδας", "γλώσσας",
                                         "ρυπαίνω", "απώλεσα", "απάρτιζα",
                                         "ορίσθηκα", "επικαιροποιήσου")}
    for form in ("κατάστασιν", "μάχην", "αρμοδιότητος", "μισθώσεως",
                 "πατρίδος", "γλώσσης", "νομοθετείν", "συνεταιρίζεσθαι",
                 "αποφασίζομεν", "αντιλαμβανόμεθα"):
        assert ex.open_corpus_archaic_form(form, known), form
    for form in ("καταταγέντων", "φυτοπροστατευτικών", "παρόν", "προϊόν",
                 "εντούτοις", "γνώσει", "νομικώς", "ρυπαίνων", "απώλεσαν",
                 "απάρτιζαν", "ορίσθηκαν", "επικαιροποιήσουν"):
        assert not ex.open_corpus_archaic_form(form, known), form


def test_formal_corpus_fragments_do_not_become_words():
    rows = {form: (100, 0, 50) for form in ("θέσιο", "βάθμιας", "βάθμια", "παράγρ")}
    evidence = ex.Evidence({}, [table("opengov", rows), table("parl", rows)])
    selection = ex.select(ex.Lexicon({}, set()), evidence, function_words={},
                          review=ex.load_review(ex.REVIEW), reviewed_loanwords=[])
    assert not set(rows) & set(selection.entries)
    assert selection.rejected["reviewed_reject"] == 4


def test_wilson_lower_bound():
    assert L.wilson_lower(0, 0) == 0.0
    assert 0.80 < L.wilson_lower(24, 25) < 0.81
    assert L.wilson_lower(10, 10) < 1.0


# -- reviewed data files -----------------------------------------------------

def test_every_function_word_passes_the_spelling_check():
    words = ex.load_function_words(ex.FUNCTION_WORDS)
    assert {"και", "ο", "ή", "να", "θα", "πού", "πώς"} <= set(words)
    for word in words:
        assert L.orthography_reason(word) is None, word


def test_review_file_loads_and_explains_every_entry():
    data = json.loads(ex.REVIEW.read_text(encoding="utf-8"))
    review = ex.load_review(ex.REVIEW)
    assert set(review) == {"rejects", "corpus_accepts", "elision_accepts",
                           "abbreviation_rejects", "acronym_rejects",
                           "abbreviation_accepts", "acronym_accepts", "preferred_spellings"}
    for key in review:
        for entry in data.get(key, []):
            assert entry["reason"].strip(), entry


# -- elisions ------------------------------------------------------------------

def neighbors(next_word=0, next_vowel=0, next_clitic=0, prev_word=0, prev_vowel=0):
    return [next_word, next_vowel, next_clitic, prev_word, prev_vowel]


@pytest.fixture
def elision_tables():
    attested = {"από", "είναι", "έχει", "εκεί", "κατά", "άδηλος", "εδώ",
                "καλό", "μεταξύ", "εξ", "τους", "έγραψα", "μετά"}
    return ex.elision_index(attested)


def test_elision_needs_its_full_word_and_a_vowel_after(elision_tables):
    attested, after = elision_tables
    admitted = attested
    good = neighbors(next_word=100, next_vowel=95)
    assert ex.elision_reason("απ" + A, attested, after, admitted, set(), good) is None
    assert ex.elision_reason("κατ" + A, attested, after, admitted, set(), good) is None
    assert ex.elision_reason("είν" + A, attested, after, admitted, set(), good) is None
    # μεταξ’: unaccented, from a word accented on the dropped vowel
    assert ex.elision_reason("μεταξ" + A, attested, after, admitted, set(), good) is None
    # an aspirated consonant before a word that had a rough breathing
    assert ex.elision_reason("μεθ" + A, attested, after, admitted, set(), good) is None
    assert ex.elision_reason("αφ" + A, attested, after, admitted, set(), good) is None
    assert (ex.elision_reason("εχ" + A, attested, after, admitted, set(), good)
            == "elision_without_full_word")
    # colloquial elision before a weak pronoun counts as an elision context
    clitic = neighbors(next_word=100, next_vowel=40, next_clitic=55)
    assert ex.elision_reason("απ" + A, attested, after, admitted, set(), clitic) is None
    # a numbering mark: stands before anything
    numbering = neighbors(next_word=40, next_vowel=8, next_clitic=2)
    assert (ex.elision_reason("καλ" + A, attested, after, admitted, set(), numbering)
            == "elision_not_before_a_vowel")
    # η and ω are not elided: εδ’ is no elision of εδώ
    assert (ex.elision_reason("εδ" + A, attested, after, admitted, set(), good)
            == "elision_without_full_word")
    # a word that does not elide, unless reviewed
    assert (ex.elision_reason("εξ" + A, attested, after, admitted, set(), good)
            == "elision_of_a_word")
    # a reviewed acceptance waives only that check; the full word is still needed
    assert (ex.elision_reason("εξ" + A, attested, after, admitted, {"εξ" + A}, good)
            == "elision_without_full_word")
    # a closing quotation mark after a word in ς
    assert (ex.elision_reason("τουσ" + A, attested, after, admitted, set(), good)
            == "quotation_mark_after_a_word")


def test_aphaeresis_needs_its_full_word_and_a_vowel_before(elision_tables):
    attested, after = elision_tables
    after_vowel = neighbors(prev_word=40, prev_vowel=40)
    assert ex.elision_reason(A + "ναι", attested, after, attested, set(), after_vowel) is None
    assert ex.elision_reason(A + "χει", attested, after, attested, set(), after_vowel) is None
    assert ex.elision_reason(A + "κει", attested, after, attested, set(), after_vowel) is None
    assert ex.elision_reason(A + "γραψα", attested, after, attested, set(),
                             after_vowel) is None
    # an opening quotation mark follows anything
    quote = neighbors(prev_word=100, prev_vowel=65)
    assert (ex.elision_reason(A + "χει", attested, after, attested, set(), quote)
            == "aphaeresis_not_after_a_vowel")
    # the accent must match what is left of the full word: άδηλος -> δηλος
    assert (ex.elision_reason(A + "δήλος", attested, after, attested, set(), after_vowel)
            == "elision_without_full_word")


# -- abbreviation shapes ---------------------------------------------------------

@pytest.mark.parametrize("spelling,rejected", [
    ("π.χ.", False), ("κ.λπ.", False), ("σελ.", False), ("μ.Χ.", False),
    ("κ.", False), ("ά.", True), ("ς.", True), ("του.Β.", True),
])
def test_abbreviation_shapes(spelling, rejected):
    words = {"του", "λόγο", "σελίδα"}
    assert (ex.abbreviation_shape_reason(spelling, words) is not None) is rejected


def test_two_words_run_together_in_capitals_are_found():
    keys = {"κειμενο", "σελιδα", "διαταξη", "τησ", "ευρω", "τεχνικη"}
    assert ex.run_together_split("κειμενοσελιδα", keys) == ("κειμενο", "σελιδα")
    assert ex.run_together_split("διαταξητησ", keys) == ("διαταξη", "τησ")
    # the first word is too short to tell a run-together from a name
    assert ex.run_together_split("ευρωτεχνικη", keys) is None
    assert ex.run_together_split("σελιδατη", keys) is None     # too short
    assert ex.run_together_split("κειμενοσελ", keys) is None


def test_learned_monosyllables_of_fixed_phrases_are_function_words():
    words = ex.load_function_words(ex.FUNCTION_WORDS)
    assert {"εις", "τοις", "τω"} <= set(words)


# -- type counts -----------------------------------------------------------------

def test_type_counts_record_elision_neighbors(tmp_path):
    docs = [{"s": ["απ’ όλα τα καλά", "πάρ’ το τώρα", "να ’χει δίκιο"]},
            {"s": ["απ’ όλους"]}]
    counts = TC.count_documents("x", "train", docs)
    assert counts.count["απ’"] == 2 and counts.docs["απ’"] == 2
    assert counts.context["απ’"] == [2, 2, 0, 0, 0]
    assert counts.context["πάρ’"] == [1, 0, 1, 0, 0]
    assert counts.context["’χει"] == [1, 0, 0, 1, 1]
    assert counts.initial["απ’"] == 2
    path = tmp_path / "t.tsv.gz"
    TC.write_table(counts, path)
    back = TC.read_table(path)
    assert back.count == counts.count and back.context == counts.context
    first = path.read_bytes()
    TC.write_table(counts, path)
    assert path.read_bytes() == first


# -- end to end ----------------------------------------------------------------------

SHARD = {"train-00000.jsonl.gz": "ab" * 32}
WORDS = 1_000_000


def table(source, rows, context=None, words=WORDS):
    """``rows``: token -> (count, initial, docs)."""
    tc = TC.TypeCounts(source, "train", words=words, tokens=words, sentences=1000,
                       documents=100, shards=dict(SHARD))
    for token, (n, ini, docs) in rows.items():
        tc.count[token] = n
        if ini:
            tc.initial[token] = ini
        tc.docs[token] = docs
    tc.context.update(context or {})
    return tc


def write_fixture(tmp_path):
    pairs = []

    def paradigm(lemma, pos, forms):
        for form, tags in forms:
            pairs.append({"form": form, "lemma": lemma, "pos": pos, "tags": tags})

    paradigm("καλός", "adj", [
        ("καλός", ["masculine", "nominative", "singular"]),
        ("καλή", ["feminine", "nominative", "singular"]),
        ("καλό", ["neuter", "nominative", "singular"]),
        ("καλοί", ["masculine", "nominative", "plural"]),
    ])
    paradigm("σπίτι", "noun", [
        ("σπίτι", ["nominative", "singular"]), ("σπιτιού", ["genitive", "singular"]),
        ("σπίτια", ["nominative", "plural"]),
    ])
    paradigm("είμαι", "verb", [("είναι", ["third-person", "present", "singular"])])
    paradigm("βουλή", "noun", [("βουλή", ["nominative", "singular"]),
                               ("βουλής", ["genitive", "singular"])])
    paradigm("Αθήνα", "name", [("Αθήνα", ["nominative", "singular"]),
                               ("Αθήνας", ["genitive", "singular"])])
    # a name page whose letters are also a word's: the paradigm reader
    # drops it, and the name route admits it on its capitalized share
    paradigm("Ρώμη", "name", [("Ρώμη", ["nominative", "singular"])])
    paradigm("ρώμη", "noun", [("ρώμη", ["nominative", "singular"])])
    paradigm("Σοφία", "name", [("Σοφία", ["nominative", "singular"])])
    paradigm("σοφία", "noun", [("σοφία", ["nominative", "singular"])])
    # a capitalized sense Wiktionary has (no name page) that text writes
    # in lowercase
    paradigm("Μέσα", "adv", [("Μέσα", [])])
    paradigm("μέσα", "adv", [("μέσα", [])])
    paradigm("πλάι", "adv", [("πλάι", [])])
    paradigm("ζαβολιάρης", "noun", [("ζαβολιάρης", ["nominative", "singular"]),
                                    ("ζαβολιάρη", ["genitive", "singular"])])
    paradigm("γράφω", "verb", [
        ("γράφω", ["first-person", "present", "singular"]),
        ("γράφει", ["third-person", "present", "singular"]),
        ("έγραψα", ["first-person", "past", "singular"]),
        ("γράφοντάς", ["gerund"]),
    ])
    # a rare spelling with a misplaced accent, its own lemma (καλήμερα), and
    # a colloquial accent of the commoner spelling's lemma (ακούσαν)
    paradigm("καλημέρα", "noun", [("καλημέρα", ["nominative", "singular"])])
    paradigm("καλήμερα", "intj", [("καλήμερα", [])])
    paradigm("ακούω", "verb", [("άκουσαν", ["third-person", "past", "plural"]),
                               ("ακούσαν", ["third-person", "past", "plural"])])
    paradigm("Σαββάτο", "name", [("Σαββάτο", ["nominative", "singular"])])
    # a rare short word one key from και
    paradigm("ξάι", "noun", [("ξάι", ["nominative", "singular"])])
    # common words that one source writes with a capital
    paradigm("ένωση", "noun", [("ένωση", ["nominative", "singular"])])
    paradigm("αίθουσα", "noun", [("αίθουσα", ["nominative", "singular"])])
    # full words of elisions
    paradigm("όλος", "adj", [("όλα", ["neuter", "nominative", "plural"])])
    paradigm("άλλος", "adj", [("άλλα", ["neuter", "nominative", "plural"])])
    paradigm("ολέ", "intj", [("ολέ", [])])
    # a word whose letters are an acronym's
    paradigm("σύριζα", "adv", [("σύριζα", [])])
    mg_pairs = tmp_path / "mg_pairs.json"
    mg_pairs.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")

    db = tmp_path / "lookup.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE lemmas (id INTEGER PRIMARY KEY, text TEXT)")
    con.execute("CREATE TABLE lookup (form TEXT, lemma_id INTEGER, src TEXT)")
    con.execute("INSERT INTO lemmas VALUES (1, 'γράφω'), (2, 'καλός'), (3, 'πλάι')")
    con.executemany("INSERT INTO lookup VALUES (?, ?, ?)", [
        ("γράφουμε", 1, "el"), ("γραφω", 1, "el"), ("Καλός", 2, "el"),
        ("πλαι", 3, "el")])
    con.commit()
    con.close()

    freq = tmp_path / "mg_freq.txt"
    freq.write_text("\n".join([
        "zzzz 999000", "καλός 500", "σπίτι 5000", "ρωτάω 500", "σπιτί 60",
        "και 400", "γράφουμε 20", "απ 10", "επικροτεί 5", "θεσπιστεί 5",
        "είναι 20000", "είνα 50", "είαι 50", "είναικαι 60", "καλημέρα 300",
        "άκουσαν 300", "σπίτια 100",
    ]) + "\n", encoding="utf-8")

    function_words = tmp_path / "function_words.json"
    function_words.write_text(json.dumps({
        "schema_version": 1, "description": "fixture",
        "categories": {"conjunction": ["και", "ή", "αλλά"], "preposition": ["από"],
                       "particle": ["να"], "article": ["του"]}}, ensure_ascii=False), encoding="utf-8")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({
        "schema_version": 1, "rejects": [],
        "corpus_accepts": [{"form": "είαι", "reason": "fixture"}],
        "elision_accepts": [], "abbreviation_rejects": [],
        "acronym_rejects": []}), encoding="utf-8")

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shards = [{"file": f, "sha256": d} for f, d in SHARD.items()]
    splits = {"splits": {"train": {"shards": shards}, "dev": {"shards": shards}}}
    (corpus / "manifest.json").write_text(json.dumps({"sources": {
        "wiki": splits, "parl": splits}}), encoding="utf-8")
    rows = {
        "καλός": (100, 0, 50), "καλή": (10, 0, 5), "καλό": (1, 0, 1),
        "σπίτι": (500, 0, 80), "σπίτια": (5, 0, 3),
        "βουλή": (10, 0, 5), "Βουλή": (40, 0, 20),
        "Αθήνα": (40, 0, 12), "Αθήνας": (3, 0, 2),
        "γράφω": (40, 0, 10), "έγραψα": (4, 0, 2), "γράφοντάς": (10, 0, 3),
        "ρωτάω": (20, 0, 4), "σπιτί": (6, 0, 2),
        "είναι": (2000, 0, 99), "είνα": (5, 0, 2), "είαι": (5, 0, 2),
        "επικροτεί": (25, 0, 12), "θεσπιστεί": (10, 0, 4),
        "Κηφισιά": (40, 0, 4),
        "Ρώμη": (40, 0, 12), "ρώμη": (2, 0, 2),
        "Σοφία": (30, 0, 12), "σοφία": (60, 0, 30),
        "μέσα": (500, 0, 80), "Μέσα": (60, 30, 25),
        "πλάι": (20, 0, 5), "πλαι": (4, 0, 3),
        "και": (900, 50, 99), "Και": (60, 60, 40), "από": (300, 0, 90),
        "απ’": (50, 0, 10), "καλ’": (20, 0, 5), "’γραψα": (15, 0, 6),
        "π.χ.": (30, 0, 5), "σελ.": (20, 0, 4), "του.Β.": (15, 0, 3),
        "π.Χ.": (12, 0, 3), "Π.χ.": (11, 11, 3), "Σελ.": (4, 4, 2),
        "ΟΗΕ": (40, 0, 10), "ΚΑΙ": (30, 0, 6), "ΚΑΛΟΣΣΠΙΤΙ": (40, 0, 6),
        # είναι and και run together
        "είναικαι": (6, 0, 3),
        "καλημέρα": (200, 0, 40), "καλήμερα": (3, 0, 2),
        "άκουσαν": (200, 0, 30), "ακούσαν": (1, 0, 1),
        "Σάββατο": (150, 10, 40), "Σαββάτο": (3, 0, 2),
        "ξάι": (3, 0, 2),
        # names spelled with the accent elsewhere or with a dialytika that
        # a far commoner spelling of the same letters lacks
        "Πάτρα": (4100, 0, 80), "Πατρά": (40, 0, 6),
        "Γουινέα": (200, 0, 20), "Γουϊνέα": (40, 0, 6),
        "Ένωση": (100, 0, 20), "ένωση": (5, 0, 3),
        "αίθουσα": (25, 0, 10),
        "όλα": (300, 0, 50), "άλλα": (200, 0, 40), "αλλά": (400, 0, 90), "ολέ": (5, 0, 3),
        "όλ’": (40, 0, 8), "ολ’": (12, 0, 4), "αλλ’": (50, 0, 10), "άλλ’": (15, 0, 5),
        "σύριζα": (3, 0, 2), "ΣΥΡΙΖΑ": (200, 0, 30), "ΣΠΙΤΙΑ": (150, 0, 20),
    }
    context = {"απ’": [50, 48, 1, 45, 40], "καλ’": [20, 3, 1, 20, 15],
               "’γραψα": [15, 2, 3, 15, 15], "όλ’": [40, 38, 1, 40, 20],
               "ολ’": [12, 11, 1, 12, 6], "αλλ’": [50, 48, 1, 50, 25],
               "άλλ’": [15, 14, 1, 15, 8]}
    # Parliament minutes write Ένωση and Αίθουσα with a capital; the other
    # source writes αίθουσα in lowercase
    parl = {"Ένωση": (100, 0, 20), "ένωση": (3, 0, 2),
            "Αίθουσα": (600, 0, 30), "αίθουσα": (2, 0, 2)}
    out_dir = tmp_path / "type_counts"
    TC.write_table(table("wiki", rows, context), TC.table_path(out_dir, "wiki", "train"))
    TC.write_table(table("parl", parl), TC.table_path(out_dir, "parl", "train"))
    return {"mg_pairs": mg_pairs, "lookup_db": db, "mg_freq": freq,
            "function_words": function_words, "review": review, "corpus": corpus,
            "type_counts": out_dir}


def run_export(paths, out_dir, *extra):
    argv = ["--out-dir", str(out_dir), "--corpus", str(paths["corpus"]),
            "--type-counts", str(paths["type_counts"]),
            "--mg-pairs", str(paths["mg_pairs"]), "--lookup-db", str(paths["lookup_db"]),
            "--mg-freq", str(paths["mg_freq"]),
            "--function-words", str(paths["function_words"]),
            "--review", str(paths["review"]),
            "--reviewed-loanwords", str(paths["review"].with_name("none.json")),
            "--allow-unpinned", "--workers", "1",
            "--version", "9.9.9", "--commit", "c" * 40,
            # each fixture pool has 1M words, so no larger source would
            # confirm a C rating (test_small_pools_rate_c_only_when_confirmed)
            "--c-confirm-min-words", "0", *extra]
    assert ex.main(argv) == 0


def read_dic(path):
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    assert lines[-1] == ""
    body = lines[1:-1]
    assert int(lines[0]) == len(body)
    return body


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("el_monotonic")
    paths = write_fixture(tmp)
    out = tmp / "out"
    run_export(paths, out)
    return paths, out


def test_dic_format(exported):
    _, out = exported
    body = read_dic(out / "el_monotonic.dic")
    assert body == sorted(body, key=lambda s: s.encode("utf-8"))
    pattern = re.compile(r"^[^\t ]+\tfr:([CMRX])( mg:generated)?$")
    for line in body:
        m = pattern.match(line)
        assert m, line
        if m.group(2):
            assert m.group(1) == "X"
        assert L.orthography_reason(line.split("\t")[0]) is None


def test_routes_and_ratings(exported):
    _, out = exported
    ratings = dict(line.split("\t") for line in read_dic(out / "el_monotonic.dic"))
    # paradigm of an attested lemma, rated per million in its best source
    assert ratings["καλός"] == "fr:C"
    assert ratings["καλή"] == "fr:M"
    assert ratings["καλό"] == "fr:R"
    assert ratings["καλοί"] == "fr:X mg:generated"
    # a lookup.db form mg_pairs lacks joins its lemma; aliases do not
    assert ratings["γράφουμε"] == "fr:M"          # 20 FrequencyWords tokens
    assert "γραφω" not in ratings and "Καλός" not in ratings
    # nor does a stripped alias that the stripping makes a monosyllable
    # (πλαι, read as one syllable, for πλάι)
    assert "πλάι" in ratings and "πλαι" not in ratings
    # no double-accented enclitic forms
    assert "γράφοντάς" not in ratings
    # an unattested lemma stays out
    assert "ζαβολιάρης" not in ratings
    # a name with its own evidence keeps its capital
    assert ratings["Αθήνα"] == "fr:C" and "Αθήνας" in ratings
    # an open-corpus rate counts for C only in 10 documents and for M only
    # in 3: Κηφισιά (40 per million in 4 documents) is held at M, έγραψα
    # (4 per million in 2 documents) at R
    assert ratings["Κηφισιά"] == "fr:M"
    assert ratings["έγραψα"] == "fr:R"
    # function words
    assert "και" in ratings and "ή" in ratings and "από" in ratings
    assert "Και" not in ratings
    # corpus word attested in both sources; a formal word rare in subtitles
    # needs wider spread in the open corpus
    assert "ρωτάω" in ratings
    assert "επικροτεί" in ratings and "θεσπιστεί" not in ratings
    # a misplaced accent of a far commoner word is no corpus word
    assert "σπιτί" not in ratings
    # nor is a typing slip of one (a letter left out of είναι), unless the
    # review file accepts it
    assert ratings["είναι"] == "fr:C"
    assert "είνα" not in ratings and "είαι" in ratings
    # a name page whose letters are a word's is a name when text writes it
    # with its capital (Ρώμη), not when it writes the word (Σοφία, a third
    # of the time); a capitalized twin of a word that is no name and that
    # text writes in lowercase is left out (Μέσα)
    assert ratings["Ρώμη"] == "fr:C" and "Σοφία" not in ratings
    assert "Μέσα" not in ratings and "μέσα" in ratings
    # names from the corpus; institutional capitals of common words are not
    assert "Κηφισιά" in ratings and "Βουλή" not in ratings
    # elisions with U+2019: a real one, an aphaeresis, not a numbering mark
    assert "απ’" in ratings and "’γραψα" in ratings and "καλ’" not in ratings
    # an elided word written without its mark is not listed
    assert "απ" not in ratings


def rejected_as(out, reason):
    audit = json.loads((out / "el_monotonic.audit.json").read_text(encoding="utf-8"))
    return {form for form, _ in audit["rejected_top"].get(reason, [])}


def test_misspellings_of_commoner_words_are_left_out(exported):
    _, out = exported
    ratings = dict(line.split("\t") for line in read_dic(out / "el_monotonic.dic"))
    # two words run together after one of the frequent heads
    assert "είναικαι" not in ratings
    assert "είναικαι" in rejected_as(out, "run_together_words")
    # a rare entry beside a far commoner spelling with another accent;
    # a form of the commoner spelling's own lemma stays (ακούσαν)
    assert "καλήμερα" not in ratings and "καλημέρα" in ratings
    assert "Σαββάτο" not in ratings and "Σάββατο" in ratings
    assert {"καλήμερα", "Σαββάτο"} <= rejected_as(
        out, "rare_accent_twin_of_a_commoner_spelling")
    assert "ακούσαν" in ratings and "άκουσαν" in ratings
    # a rare short word one key from a very common one
    assert "ξάι" not in ratings
    assert "ξάι" in rejected_as(out, "rare_short_slip_of_a_common_word")
    # names with the accent elsewhere, or with a dialytika, than a far
    # commoner name of the same letters
    assert "Πάτρα" in ratings and "Πατρά" not in ratings
    assert "Γουινέα" in ratings and "Γουϊνέα" not in ratings
    assert "Πατρά" in rejected_as(out, "accent_variant_of_a_commoner_name")
    assert "Γουϊνέα" in rejected_as(out, "dialytika_variant_of_a_commoner_name")
    # an unaccented elision beside a commoner accented one (ολ’ beside όλ’),
    # but not when the unaccented one is the commoner (αλλ’ beside άλλ’)
    assert "όλ’" in ratings and "ολ’" not in ratings
    assert "ολ’" in rejected_as(out, "elision_without_the_tonos_of_a_commoner_twin")
    assert "αλλ’" in ratings and "άλλ’" in ratings


def test_a_capitalized_common_word_needs_its_capital_in_two_sources(exported):
    _, out = exported
    ratings = dict(line.split("\t") for line in read_dic(out / "el_monotonic.dic"))
    # Both sources write Ένωση with its capital; only the Parliament
    # minutes write Αίθουσα with one, though they make up most of its uses.
    assert "Ένωση" in ratings and "ένωση" in ratings
    assert "Αίθουσα" not in ratings and "αίθουσα" in ratings
    assert "Αίθουσα" in rejected_as(out, "capitalized_common_word")


def test_small_pools_rate_c_only_when_confirmed():
    def pool(source, words, rows):
        return table(source, {t: (n, 0, 50) for t, n in rows.items()}, words=words)

    small = pool("tatoeba", 1_000_000, {"Κηφισιά": 40, "λόγος": 40})
    large = pool("wikipedia", 20_000_000, {"λόγος": 800, "Κηφισιά": 20})
    subtitles = {"λόγος": 50, "zzzz": 999_950}
    evidence = ex.Evidence(subtitles, [small, large])
    # 40 per million in the small pool alone: held under the C edge
    rates = evidence.rates("Κηφισιά")
    assert rates["tatoeba"] < L.C_PER_MILLION
    assert L.bucket_for_rate(max(rates.values())) == "M"
    # confirmed by the large pool (40 per million there too)
    assert evidence.rating("λόγος")[0] == "C"
    assert evidence.rates("λόγος")["tatoeba"] == 40.0
    # the rule is off when no pool counts as small
    assert ex.Evidence(subtitles, [small, large],
                       c_confirm_min_words=0).rating("Κηφισιά")[0] == "C"


def test_names_need_large_case_preserving_corroboration_for_c():
    subtitles = {"τομ": 500, "μαρία": 500, "λόγος": 500, "filler": 998_500}
    small = table("tatoeba", {"Τομ": (100, 0, 50), "Μαρία": (100, 0, 50)},
                  words=1_000_000)
    large = table("wikipedia", {"Τομ": (400, 0, 200), "Μαρία": (900, 0, 400)},
                  words=20_000_000)
    evidence = ex.Evidence(subtitles, [small, large])
    assert evidence.rating("Τομ")[0] == "M"
    assert evidence.rating("Μαρία")[0] == "C"
    assert evidence.rating("λόγος")[0] == "C"


def test_a_test_rating_rule_is_refused_in_the_shipping_directory(exported):
    paths, _ = exported
    with pytest.raises(ex.ExportError, match="test rating rule"):
        ex.run(ex_args(paths, ex.OUT, c_confirm_min_words=0))


def test_companion_lists(exported):
    _, out = exported
    abbreviations = (out / "el_monotonic_abbreviations.txt").read_text(
        encoding="utf-8").split()
    assert "π.χ." in abbreviations and "σελ." in abbreviations
    assert "του.Β." not in abbreviations
    # spellings that differ in case are counted together; another spelling
    # with its own evidence is listed too (π.Χ., before Christ), but not a
    # listed one with a sentence-initial capital (Π.χ.) or one with too
    # few tokens (Σελ.)
    assert "π.Χ." in abbreviations
    assert "Π.χ." not in abbreviations and "Σελ." not in abbreviations
    acronyms = (out / "el_monotonic_acronyms.txt").read_text(encoding="utf-8").split()
    # ΚΑΙ is a word in capitals; ΚΑΛΟΣΣΠΙΤΙ is two words run together;
    # ΣΠΙΤΙΑ is σπίτια in capitals, while ΣΥΡΙΖΑ is far commoner than the
    # word σύριζα in both sources
    assert acronyms == ["ΟΗΕ", "ΣΥΡΙΖΑ"]
    audit = json.loads((out / "el_monotonic.audit.json").read_text(encoding="utf-8"))
    rows = {r["acronym"]: r.get("rejected") for r in audit["acronym_review"]}
    assert rows["ΚΑΛΟΣΣΠΙΤΙ"] == "two words run together: καλοσ + σπιτι"
    assert rows["ΣΠΙΤΙΑ"] == "its letters are an entry of the word list"
    assert rows["ΣΥΡΙΖΑ"] is None


def test_aff_and_version(exported):
    paths, out = exported
    aff = (out / "el_monotonic.aff").read_text(encoding="utf-8").splitlines()
    rules = [line for line in aff if line and not line.startswith("#")]
    assert rules[:3] == ["SET UTF-8", "LANG el", "FLAG num"]
    assert rules[3].startswith("TRY ")
    letters = rules[3][4:]
    assert set(letters) == set(L.LOWER_LETTERS) and len(letters) == len(L.LOWER_LETTERS)
    assert len(rules) == 4
    version = dict(line.split(": ", 1) for line in
                   (out / "el_monotonic.version").read_text(encoding="utf-8").splitlines())
    assert version["version"] == "9.9.9"
    assert version["commit"] == "c" * 40
    assert version["variant"] == "el"
    body = read_dic(out / "el_monotonic.dic")
    assert int(version["entries"]) == len(body)
    buckets = Counter(line.split("\tfr:")[1][0] for line in body)
    assert version["buckets"] == " ".join(f"{b}={buckets[b]}" for b in "CMRX")
    assert version["dic_sha256"] == sha(out / "el_monotonic.dic")
    assert version["mg_pairs_sha256"] == sha(paths["mg_pairs"])
    assert version["mg_freq_sha256"] == sha(paths["mg_freq"])
    assert version["lookup_db_sha256"] == sha(paths["lookup_db"])
    assert version["review_sha256"] == sha(paths["review"])
    assert version["function_words_sha256"] == sha(paths["function_words"])
    assert "type_counts_sha256" in version and "corpus_manifest_sha256" in version


def test_rerun_is_byte_identical_and_inputs_change_the_version(exported, tmp_path):
    paths, out = exported
    again = tmp_path / "again"
    run_export(paths, again)
    for name in ("el_monotonic.dic", "el_monotonic.aff", "el_monotonic.version",
                 "el_monotonic_abbreviations.txt", "el_monotonic_acronyms.txt",
                 "el_monotonic.audit.json", "el_monotonic.evidence.tsv.gz"):
        assert (again / name).read_bytes() == (out / name).read_bytes(), name
    freq = paths["mg_freq"].read_text(encoding="utf-8")
    changed = dict(paths)
    changed["mg_freq"] = tmp_path / "mg_freq.txt"
    changed["mg_freq"].write_text(freq + "καλή 1\n", encoding="utf-8")
    third = tmp_path / "third"
    run_export(changed, third)
    assert ((third / "el_monotonic.version").read_bytes()
            != (out / "el_monotonic.version").read_bytes())


def test_partial_run_is_refused_in_the_shipping_directory(exported):
    paths, _ = exported
    before = sorted(ex.OUT.glob("el_monotonic*")) if ex.OUT.exists() else []
    stamps = {p: p.stat().st_mtime_ns for p in before}
    with pytest.raises(ex.ExportError, match="max-lemmas"):
        ex.run(ex_args(paths, ex.OUT, max_lemmas=2))
    after = sorted(ex.OUT.glob("el_monotonic*")) if ex.OUT.exists() else []
    assert after == before
    assert {p: p.stat().st_mtime_ns for p in after} == stamps


def ex_args(paths, out_dir, **overrides):
    import argparse
    args = argparse.Namespace(
        out_dir=out_dir, corpus=paths["corpus"], type_counts=paths["type_counts"],
        mg_pairs=paths["mg_pairs"], lookup_db=paths["lookup_db"], no_lookup=False,
        mg_freq=paths["mg_freq"], function_words=paths["function_words"],
        review=paths["review"], reviewed_loanwords=paths["review"].with_name("none.json"),
        max_lemmas=None, allow_unpinned=True, workers=1, version="9.9.9",
        commit="c" * 40)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_unpinned_inputs_are_refused(exported, tmp_path, monkeypatch):
    paths, _ = exported
    manifest = tmp_path / "hf_manifest.json"
    manifest.write_text(json.dumps({"files": {"data/mg_pairs.json": {
        "sha256": "0" * 64}}}), encoding="utf-8")
    digest = sha(paths["mg_pairs"])
    mismatched = ex.check_hf_pins({"mg_pairs": paths["mg_pairs"]},
                                  {"mg_pairs": digest}, manifest)
    assert len(mismatched) == 1 and "data/mg_pairs.json" in mismatched[0]
    assert ex.check_hf_pins({"mg_pairs": paths["mg_pairs"]},
                            {"mg_pairs": "0" * 64}, manifest) == []
    monkeypatch.setattr(ex, "check_hf_pins", lambda *args, **kwargs: mismatched)
    with pytest.raises(ex.ExportError, match="hf_manifest"):
        ex.run(ex_args(paths, tmp_path / "pinned", allow_unpinned=False))


def test_an_export_to_the_shipping_directory_flags_a_stale_model(
        exported, tmp_path, monkeypatch, capsys):
    paths, _ = exported
    shipping = tmp_path / "hunspell"
    version = tmp_path / "lm" / "el_ngram.version"
    version.parent.mkdir()
    monkeypatch.setattr(ex, "OUT", shipping)
    monkeypatch.setattr(ex, "LM_VERSION", version)
    version.write_text(json.dumps({"oov_dictionary_sha256": "0" * 64}),
                       encoding="utf-8")
    ex.run(ex_args(paths, shipping))
    note = capsys.readouterr().out
    assert "NOTE:" in note and "train_lm.py --lang el" in note
    dic = shipping / "el_monotonic.dic"
    assert "train_lm.py --lang el" in ex.stale_model_note(dic)
    # A model built from this list, or no model at all, needs nothing.
    version.write_text(json.dumps({"oov_dictionary_sha256": sha(dic)}),
                       encoding="utf-8")
    assert ex.stale_model_note(dic) is None
    ex.run(ex_args(paths, shipping))
    assert "NOTE:" not in capsys.readouterr().out
    version.unlink()
    assert ex.stale_model_note(dic) is None
    # Another directory is not the shipping list and is not compared.
    version.write_text(json.dumps({"oov_dictionary_sha256": "0" * 64}),
                       encoding="utf-8")
    ex.run(ex_args(paths, tmp_path / "other"))
    assert "NOTE:" not in capsys.readouterr().out

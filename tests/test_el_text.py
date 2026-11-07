"""Normalization, tokenization and filters of the monotonic Modern Greek
corpus (build/el_text.py). Fast, no files, no network."""

from collections import Counter

import pytest

from dilemma_build import el_text as T

AP = "\u2019"


def sents(text, **kw):
    return T.process_document(text, **kw).sentences


def one(text, **kw):
    out = sents(text, **kw)
    assert len(out) == 1, out
    return out[0]


# -- normalization ---------------------------------------------------------

def test_nfc_and_oxia_become_tonos_letters():
    oxia = "καλ\u1f71"            # καλά with U+1F71 (oxia)
    decomposed = "καλα\u0301"  # α + combining acute
    assert one(oxia + " " + decomposed) == ["καλά", "καλά"]
    stats = T.process_document(oxia).stats
    assert stats["norm_oxia_chars"] == 1


def test_greek_question_mark_and_ano_teleia_are_sentence_ends():
    # U+037E and U+0387 normalize to ; and U+00B7
    assert sents("Πού είσαι\u037e Εδώ\u0387 καλά") == [
        ["Πού", "είσαι"], ["Εδώ"], ["καλά"]]


def test_spacing_tonos_before_a_vowel_is_combined():
    assert one("\u0384Οταν ήρθε") == ["Όταν", "ήρθε"]


def test_invisible_characters_are_removed():
    assert one("κα\u00adλά\u200b παιδιά") == ["καλά", "παιδιά"]
    # a no-break space separates words like a space
    assert one("παι\u00a0διά") == ["παι", "διά"]


# -- apostrophes -----------------------------------------------------------

@pytest.mark.parametrize("mark", ["'", "\u2019", "\u02bc", "\u0384", "\u00b4", "\u1fbd"])
def test_elision_marks_fold_to_u2019(mark):
    assert one(f"σ{mark} αγαπώ") == ["σ" + AP, "αγαπώ"]


def test_elision_written_without_space_stays_one_token():
    assert one("σ\u0384αγαπώ πολύ") == ["σ" + AP + "αγαπώ", "πολύ"]


def test_aphaeresis_keeps_a_leading_mark_and_no_tonos_is_required():
    assert one("θα 'ρθω και τα 'θελες") == ["θα", AP + "ρθω", "και", "τα", AP + "θελες"]


def test_quotation_marks_are_dropped():
    assert one("είπε 'Ναι' χθες") == ["είπε", "Ναι", "χθες"]
    assert one("το σπίτι' μας") == ["το", "σπίτι", "μας"]


def test_elision_after_vowel_words_gia_and_kai():
    assert one("γι' αυτό κι' εγώ") == ["γι" + AP, "αυτό", "κι" + AP, "εγώ"]


def test_sigma_before_the_elision_mark_is_not_final():
    assert one("Δώς' μου") == ["Δώσ" + AP, "μου"]
    assert T.lower_el("ΔΩΣ" + AP) == "δωσ" + AP


# -- numerals, placeholders, homoglyphs ------------------------------------

def test_greek_numerals_become_num():
    out = one("της Β\u0384 τάξης, η ΙΑ\u0384 και ΣΤ\u0384 και το α\u0384 εδάφιο και η Α\u0374 θέση")
    assert out == ["της", "<NUM>", "τάξης", "η", "<NUM>", "και", "<NUM>", "και",
                   "το", "<NUM>", "εδάφιο", "και", "η", "<NUM>", "θέση"]


def test_elided_prepositions_are_not_numerals():
    assert one("υπ\u2019 αριθμόν και ΥΠ\u2019 ΑΡΙΘΜ. και απ\u2019 όλους") == [
        "υπ" + AP, "αριθμόν", "και", "ΥΠ" + AP, "ΑΡΙΘΜ.", "και", "απ" + AP, "όλους"]


def test_numbers_latin_urls_and_mail_are_placeholders():
    assert one("Το 2020 και 3,5 στις 31/08/24 ή 18ου, το iPhone στο www.gov.gr "
               "και info@example.com σήμερα") == [
        "Το", "<NUM>", "και", "<NUM>", "στις", "<NUM>", "ή", "<NUM>", "το",
        "<FOREIGN>", "στο", "<FOREIGN>", "και", "<FOREIGN>", "σήμερα"]


def test_latin_homoglyphs_in_greek_words_are_repaired_and_counted():
    pd = T.process_document("Δεν τo ξέρω vα σου πω")
    assert pd.sentences == [["Δεν", "το", "ξέρω", "να", "σου", "πω"]]
    assert pd.homoglyph_repairs == Counter({("τo", "το"): 1, ("vα", "να"): 1})
    assert pd.stats["norm_homoglyph_repairs"] == 2


def test_latin_words_are_not_repaired():
    assert one("το OK και το TAXI") == ["το", "<FOREIGN>", "και", "το", "<FOREIGN>"]


# -- words, hyphens, final sigma -------------------------------------------

def test_hyphenated_compounds_are_split_like_the_keyboard():
    assert one("ο γερο-Ανέστης και οι ελληνο-τουρκικές σχέσεις") == [
        "ο", "γερο", "Ανέστης", "και", "οι", "ελληνο", "τουρκικές", "σχέσεις"]
    for tok in one("Κοινωνικο-οικονομικά θέματα"):
        assert "-" not in tok


def test_oti_with_its_comma_is_one_token():
    # ό,τι (whatever) is one word; ό alone is no word.
    assert one("Κάνε ό,τι θέλεις") == ["Κάνε", T.OTI, "θέλεις"]
    assert one("Ό,τι κι αν πεις.") == ["Ό,τι", "κι", "αν", "πεις"]
    assert one("ό,τι.")[0] == T.OTI
    assert T.process_document("ό,τι").stats["tok_oti"] == 1
    # Only the comma with no space, and only before the end of the word.
    assert one("ό, τι") == ["ό", "τι"]
    assert one("ό,τιδήποτε") == ["ό", "τιδήποτε"]
    assert one("καλό,τι") == ["καλό", "τι"]


def test_word_final_sigma_is_normalized_except_before_a_hyphen():
    pd = T.process_document("ο καλόσ δρόμος")
    assert pd.sentences == [["ο", "καλός", "δρόμος"]]
    assert pd.stats["norm_final_sigma"] == 1


def test_case_is_kept_and_lower_el_applies_final_sigma():
    assert one("Η ΕΕ και ο Νίκος") == ["Η", "ΕΕ", "και", "ο", "Νίκος"]
    assert T.lower_el("ΟΔΟΣ") == "οδος"
    assert T.lower_el("Σ" + AP + "ΑΓΑΠΩ") == "σ" + AP + "αγαπω"


# -- sentences and abbreviations -------------------------------------------

def test_keyboard_sentence_breakers_split():
    assert sents("Πού είσαι; Εντάξει! Γεια σου... τι κάνεις· καλά\nνέα γραμμή") == [
        ["Πού", "είσαι"], ["Εντάξει"], ["Γεια", "σου"], ["τι", "κάνεις"],
        ["καλά"], ["νέα", "γραμμή"]]


def test_abbreviations_do_not_end_sentences():
    out = sents("Ο κ. Παπαδόπουλος είπε π.χ. ότι η Ε.Ε. και τα Τ.Ε.Ι. "
                "θα πάρουν 3 δισ. ευρώ κ.λπ. Η επόμενη πρόταση.")
    assert out == [
        ["Ο", "κ.", "Παπαδόπουλος", "είπε", "π.χ.", "ότι", "η", "Ε.Ε.", "και",
         "τα", "Τ.Ε.Ι.", "θα", "πάρουν", "<NUM>", "δισ.", "ευρώ", "κ.λπ."],
        ["Η", "επόμενη", "πρόταση"]]


def test_unknown_word_before_period_and_lowercase_is_a_candidate():
    pd = T.process_document("Κόστισε 5 δρχμ. ανά μήνα. Τέλος.")
    assert pd.abbreviation_candidates["δρχμ"] == 1


def test_numbered_list_marker_is_a_number():
    assert one("ΙΑ. λαμβάνοντας υπόψη την πρόταση") == [
        "<NUM>", "λαμβάνοντας", "υπόψη", "την", "πρόταση"]


def test_require_terminal_punctuation_and_stop_lines():
    text = "Ιστορία\nΗ πόλη ιδρύθηκε νωρίς.\nΛίστα χωρίς τελεία\nΠαραπομπές\nΚάτι άλλο."
    assert sents(text, require_terminal_punctuation=True,
                 stop_at_lines={"Παραπομπές"}) == [["Η", "πόλη", "ιδρύθηκε", "νωρίς"]]


def test_wrap_mode_joins_single_line_breaks():
    text = "Η πρόταση συνεχίζεται\nστην επόμενη γραμμή.\n\nΝέα παράγραφος"
    assert sents(text, line_breaks="wrap") == [
        ["Η", "πρόταση", "συνεχίζεται", "στην", "επόμενη", "γραμμή"],
        ["Νέα", "παράγραφος"]]


# -- filters ----------------------------------------------------------------

def test_polytonic_sentence_is_dropped_and_document_too():
    # One quoted polytonic word in a long monotonic document drops only its
    # sentence; a document with many polytonic words is dropped whole.
    body = "Καλή μέρα σε όλους τους φίλους μας εδώ στην πόλη σήμερα. " * 10
    pd = T.process_document(body + "Το ρητό λέει ἦν και τελειώνει.")
    assert len(pd.sentences) == 10
    assert pd.drop_reason is None
    assert pd.stats["sents_dropped_polytonic"] == 1
    poly = T.process_document("Ἐν ἀρχῇ ἦν ὁ λόγος καὶ ὁ λόγος ἦν πρὸς τὸν θεόν.")
    assert poly.drop_reason == "polytonic"


def test_katharevousa_document_is_dropped_but_modern_verbs_are_not():
    kath = T.process_document("Εν τω μεταξύ ουδέν ήτο γνωστόν τοις πάσι.")
    assert kath.drop_reason == "katharevousa"
    # κλαις and φταις end in -αις but are ordinary verbs
    assert sents("Γιατί κλαις; Δε φταις εσύ") == [["Γιατί", "κλαις"], ["Δε", "φταις", "εσύ"]]
    assert not T.is_katharevousa_marker("τοις", "εκατό")
    assert not T.is_katharevousa_marker("εντούτοις")


def test_typing_without_accents_is_dropped():
    pd = T.process_document("Δηλαδη απο την μια για να μπορει κανεις.")
    assert pd.drop_reason == "unaccented"
    mixed = T.process_document("Ήρθε χθες το βράδυ στο σπίτι. Δεν ηξερα τι να πω σε αυτο.")
    assert mixed.sentences == [["Ήρθε", "χθες", "το", "βράδυ", "στο", "σπίτι"]]


@pytest.mark.parametrize("word", ["μια", "ποιος", "ποιοι", "γεια", "δυο", "πιο", "και", "τους"])
def test_synizesis_words_need_no_tonos(word):
    assert not T.is_unaccented_polysyllable(word)


@pytest.mark.parametrize("word", ["απο", "ειναι", "μπορει", "αυτοι", "ζωη"])
def test_unaccented_polysyllables_are_flagged(word):
    assert T.is_unaccented_polysyllable(word)


def test_all_capital_sentences_are_dropped_acronyms_are_kept():
    pd = T.process_document("ΑΡΘΡΟ 2 ΠΑΡΑΓΡΑΦΟΣ 3. Η ΕΕ και το ΝΑΤΟ συμφώνησαν.")
    assert pd.sentences == [["Η", "ΕΕ", "και", "το", "ΝΑΤΟ", "συμφώνησαν"]]


def test_mostly_foreign_sentence_is_dropped():
    body = "Μια ελληνική πρόταση με αρκετές λέξεις για το κείμενο. " * 4
    pd = T.process_document(body + "This is an English sentence with ένα word.")
    assert len(pd.sentences) == 4
    assert pd.stats["sents_dropped_foreign"] == 1
    latin = T.process_document("This is an English text with ένα word only.")
    assert latin.drop_reason == "foreign"


# -- keys and determinism ---------------------------------------------------

def test_sentence_key_folds_case_only():
    assert T.sentence_key(["Καλή", "μέρα"]) == T.sentence_key(["καλή", "μέρα"])
    assert T.sentence_key(["καλή", "μέρα"]) != T.sentence_key(["καλή", "νύχτα"])


def test_processing_is_deterministic():
    text = "Ο κ. Παπαδόπουλος είπε σ' αυτό: «Ναι, το 2020 ήταν καλά». Γεια!"
    a = T.process_document(text)
    b = T.process_document(text)
    assert a.sentences == b.sentences and a.stats == b.stats


def test_no_token_contains_space_hyphen_or_greek_extended():
    text = ("Σ' αγαπώ, γερο-Ανέστη! Το 1990 η Ε.Ε. και ο κ. Ν. Παππάς "
            "είπαν: «εντάξει». Καλά\u1f71 να 'ρθεις.")
    for s in sents(text):
        for tok in s:
            assert " " not in tok and "-" not in tok
            assert not any("ἀ" <= c <= "\u1fff" for c in tok)

"""Ancient -σις nouns and s-stem neuters gain their regular cells (#114, #115).

The rule tests need no data. The shipped-answer tests read data/lookup.db,
whose generated rows arrived with the 1.3.25 lookup rebuild.
"""
import sqlite3

import pytest

from dilemma import core
from dilemma.grc_noun_gaps import (
    SOURCE, acute_last_vowel, complete_noun_gaps, is_plain_lemma,
    select_noun_gap_lemmas, sis_cells, sstem_cells,
)


def _forms(cells):
    return [form for form, _ in cells]


# Rules ---------------------------------------------------------------------

def test_sstem_cells_are_the_attic_contracted_cells():
    assert _forms(sstem_cells("στέφος")) == [
        "στέφους", "στέφει", "στέφη", "στεφῶν", "στέφεσι", "στέφεσιν", "στεφοῖν"]
    assert sstem_cells("τεῖχος")[0][0] == "τείχους"
    assert sstem_cells("μέγεθος")[0][0] == "μεγέθους"
    assert sstem_cells("τρῦχος")[0][0] == "τρύχους"
    assert sstem_cells("σύναρσις") == []


def test_sis_cells_are_the_attic_cells_without_the_vocative():
    assert _forms(sis_cells("σύναρσις")) == [
        "συνάρσεως", "συνάρσει", "συνάρσεις", "συνάρσεων", "συνάρσεσι",
        "συνάρσεσιν", "συναρσέοιν", "σύναρσιν"]
    assert sis_cells("ῥήτρευσις")[0][0] == "ῥητρεύσεως"
    assert sis_cells("αἴσθησις")[0][0] == "αἰσθήσεως"
    assert "σύναρσι" not in _forms(sis_cells("σύναρσις"))
    assert sis_cells("στέφος") == []


def test_acute_goes_on_the_last_vowel_of_the_stem():
    assert acute_last_vowel("στεφ") == "στέφ"
    assert acute_last_vowel("τειχ") == "τείχ"      # second letter of a diphthong
    assert acute_last_vowel("αἰσθησ") == "αἰσθήσ"
    assert acute_last_vowel("σκ") is None


def test_plain_lemma_is_lowercase_single_word_and_singly_accented():
    assert is_plain_lemma("σύναρσις")
    assert not is_plain_lemma("Διάθεσις")       # capitalised
    assert not is_plain_lemma("θάλπ-ος")        # hyphenated dictionary string
    assert not is_plain_lemma("συναρσις")       # unaccented
    assert not is_plain_lemma("διερμήνεύσις")   # doubly accented artefact


def test_selection_needs_a_self_map_or_the_sstem_evidence():
    table = {
        "σύναρσις": "σύναρσις",                 # headword only: selected
        "Διάθεσις": "Διάθεσις",                 # capitalised: skipped
        "αἴσθησις": "αἴσθησις", "αἰσθήσεως": "αἴσθησις",
        "ὄψις": "ὄψις",                          # not -σις
        "κίνησις": "κινέω",                     # not a self-map
        # s-stem neuter: -εος and a neuter plural under the lemma; its own
        # key reads as another lemma.
        "γλάγος": "γάλα", "γλάγεος": "γλάγος", "γλάγεα": "γλάγος",
        # -εος but no plural evidence (a masculine, or an epic genitive).
        "Πάτροκλος": "Πάτροκλος", "Πατρόκλεος": "Πάτροκλος",
        "κῦδος": "κῦδος", "κύδεος": "κῦδος",
        # plural evidence must be on the lemma's own stem
        "σκέπος": "σκέπος", "σκέπεος": "σκέπος", "σκεπέων": "σκέπος",
        "θάλπ-ος": "θάλπ-ος", "θάλπεος": "θάλπ-ος", "θάλπεα": "θάλπ-ος",
    }
    sis, sstem = select_noun_gap_lemmas(table.items())
    assert sis == ["αἴσθησις", "σύναρσις"]
    assert sstem == ["γλάγος", "σκέπος"]


def test_selection_wants_ancient_neuter_evidence():
    rows = [
        # A Modern Greek adjective's table is no evidence (#114 review).
        ("υπερβαρέος", "υπέρβαρος", "wiktionary_el"),
        ("υπερβαρέα", "υπέρβαρος", "wiktionary_el"),
        # The same rows from Ancient sources, but the lemma lacks the
        # breathing its vowel needs.
        ("υβρέος", "υβρος", "grc_merged"), ("υβρέα", "υβρος", "grc_merged"),
        # A masculine and an adjective that an Ancient table also declines
        # in the second declension.
        ("λοχέος", "λόχος", "grc_merged"), ("λοχέων", "λόχος", "grc_merged"),
        ("λόχου", "λόχος", "wiktionary_grc"), ("λόχον", "λόχος", "wiktionary_grc"),
        ("πράεος", "πρᾶος", "grc_merged"), ("πράεα", "πρᾶος", "grc_merged"),
        ("πρᾴου", "πρᾶος", "grc_merged"), ("πρᾷοι", "πρᾶος", "grc_merged"),
        # A corpus lemmatizer's second-declension slip does not count.
        ("γλήνεος", "γλῆνος", "wiktionary_grc"), ("γλήνεα", "γλῆνος", "wiktionary_grc"),
        ("γλῆνον", "γλῆνος", "glaux"), ("γλήνου", "γλῆνος", "glaux"),
        # Reviewed: Homer's φῖλος.
        ("φίλεος", "φῖλος", "wiktionary_grc"), ("φίλεα", "φῖλος", "wiktionary_grc"),
        # Breathing and rho are fine when written.
        ("ἔσθεος", "ἔσθος", "grc_lexicon_expansion"),
        ("ἔσθεα", "ἔσθος", "grc_lexicon_expansion"),
        ("ῥάκεος", "ῥάκος", "wiktionary_grc"), ("ῥάκεα", "ῥάκος", "wiktionary_grc"),
        # A -σις head without its breathing is no -σις noun of the table.
        ("όδευσις", "όδευσις", "lbg_headwords"),
        ("ὅδευσις", "ὅδευσις", "lbg_headwords"),
    ]
    sis, sstem = select_noun_gap_lemmas(rows)
    assert sstem == ["γλῆνος", "ἔσθος", "ῥάκος"]
    assert sis == ["ὅδευσις"]


def test_completion_fills_absent_keys_only():
    lookup = {"σύναρσις": "σύναρσις", "κινήσει": "κινέω", "κίνησις": "κίνησις",
              "στέφος": "στέφος", "στέφει": "στέφω"}
    added = complete_noun_gaps(lookup, ["κίνησις", "σύναρσις"], ["στέφος"])
    # accented, lowercase and accentless keys
    assert lookup["συνάρσεως"] == lookup["συναρσεως"] == "σύναρσις"
    assert lookup["σύναρσιν"] == lookup["συναρσιν"] == "σύναρσις"
    assert lookup["στέφους"] == lookup["στεφους"] == lookup["στεφῶν"] == "στέφος"
    # existing readings are kept, and a cell whose own spelling already
    # reads as another lemma is not added under any key
    assert lookup["κινήσει"] == "κινέω" and "κινησει" not in lookup
    assert lookup["στέφει"] == "στέφω" and "στεφει" not in lookup
    assert "κινήσει" not in {row["form"] for row in added}
    assert {row["kind"] for row in added if row["lemma"] == "στέφος"} == {"sstem"}
    assert all(row["cell"] for row in added)


def test_completion_does_not_hide_a_reading_through_the_cells_spellings():
    # λαθῶν's monotonic spelling λαθών is the participle of λανθάνω: the
    # cell would hide it (#114 review). αἰτίωσιν's accent tells it from the
    # verb's αἰτιῶσιν, which shares only the accentless key, so it is added.
    lookup = {"λάθος": "λάθος", "λαθών": "λανθάνω", "λαθων": "λανθάνω",
              "αἰτίωσις": "αἰτίωσις", "αιτιωσιν": "αἰτιάω"}
    complete_noun_gaps(lookup, ["αἰτίωσις"], ["λάθος"])
    assert "λαθῶν" not in lookup and lookup["λαθων"] == "λανθάνω"
    assert lookup["λάθους"] == "λάθος"
    assert lookup["αἰτίωσιν"] == "αἰτίωσις" and lookup["αιτιωσιν"] == "αἰτιάω"


def test_completion_needs_a_self_mapped_sis_head():
    lookup = {"κίνησις": "κινέω"}
    assert complete_noun_gaps(lookup, ["κίνησις"], []) == []
    assert lookup == {"κίνησις": "κινέω"}


def test_completion_skips_keys_left_out_on_purpose_and_ambiguous_keys():
    lookup = {"σχετλίασις": "σχετλίασις", "σχετλιάσις": "σχετλιάσις",
              "σύναρσις": "σύναρσις"}
    added = complete_noun_gaps(lookup, ["σχετλίασις", "σχετλιάσις", "σύναρσις"], [],
                               taken={"συνάρσεων"})
    # Both accent doublets give σχετλιάσεως: neither takes it.
    assert "σχετλιάσεως" not in lookup and "σχετλιασεως" not in lookup
    # The accusative keeps each lemma's own accent, so each has its own key.
    assert lookup["σχετλίασιν"] == "σχετλίασις"
    assert lookup["σχετλιάσιν"] == "σχετλιάσις"
    assert "σχετλιασιν" not in lookup
    assert "συνάρσεων" not in lookup and lookup["συναρσεων"] == "σύναρσις"
    assert len({row["form"] for row in added}) == len(added)


def test_completion_does_not_depend_on_input_order():
    def run(sis):
        lookup = {lemma: lemma for lemma in sis}
        return complete_noun_gaps(lookup, sis, [])
    lemmas = ["σύναρσις", "ῥήτρευσις", "αἰτίωσις"]
    assert run(lemmas) == run(list(reversed(lemmas)))


# Shipped answers (need the 1.3.25 lookup rebuild) -------------------------

_MODELS = {}


def _model(lang):
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    if lang not in _MODELS:
        _MODELS.clear()
        _MODELS[lang] = core.Dilemma(lang=lang, resolve_articles=True)
    return _MODELS[lang]


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_shipped_contracted_sstem_cells(lang):
    forms = ["στέφους", "τέκους", "τρύχους", "κέρνους", "σκέπους", "γλάγους",
             "θύους", "κύους"]
    expected = ["στέφος", "τέκος", "τρῦχος", "κέρνος", "σκέπος", "γλάγος",
                "θύος", "κύος"]
    assert _model(lang).lemmatize_batch_pos(
        forms, ["NOUN"] * len(forms), guess=False) == expected


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_shipped_sis_cells(lang):
    forms = ["συνάρσει", "συνάρσεως", "συνάρσεις", "σύναρσιν", "ῥητρεύσει",
             "αἰτίωσιν"]
    expected = ["σύναρσις"] * 4 + ["ῥήτρευσις", "αἰτίωσις"]
    assert _model(lang).lemmatize_batch_pos(
        forms, ["NOUN"] * len(forms), guess=False) == expected


def test_shipped_generated_rows_name_their_source():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    with sqlite3.connect(f"file:{core.LOOKUP_DB_PATH}?mode=ro", uri=True) as conn:
        rows = dict(conn.execute(
            "SELECT l.form, s.code FROM lookup l JOIN sources s ON s.id = l.source_id "
            "WHERE l.lang = 'all' AND l.form IN ('στέφους', 'συνάρσεως', 'συναρσεως')"))
        count = conn.execute(
            "SELECT count(*) FROM lookup l JOIN sources s ON s.id = l.source_id "
            "WHERE s.code = ?", (SOURCE,)).fetchone()[0]
    assert rows == dict.fromkeys(("στέφους", "συνάρσεως", "συναρσεως"), SOURCE)
    assert 39_000 < count < 42_000


def test_shipped_cells_do_not_hide_other_readings():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    # #114 review: cells of masculines, adjectives and a Modern Greek lemma,
    # and cells whose monotonic spelling reads as another lemma, are gone.
    with sqlite3.connect(f"file:{core.LOOKUP_DB_PATH}?mode=ro", uri=True) as conn:
        lemmas = {lemma for (lemma,) in conn.execute(
            "SELECT DISTINCT m.text FROM lookup l JOIN lemmas m ON m.id = l.lemma_id "
            "JOIN sources s ON s.id = l.source_id WHERE s.code = ?", (SOURCE,))}
    assert not lemmas & {"φῖλος", "υπέρβαρος", "λόχος", "πρᾶος", "λάϊνος",
                         "ἀδράνεος", "ὄχθος", "ὦρος", "οὖρος"}
    model = _model("grc")
    assert model.lemmatize_batch(["λαθῶν", "φιλοῖν"], guess=False) == [
        "λανθάνω", "φίλος"]


"""LSJ suffix fields must become complete, independently testable inputs."""
import importlib.util
import json
from pathlib import Path
import sys
import unicodedata

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))
import expand_lsj as m
import expand_sophocles


@pytest.mark.parametrize("word,suffix,expected", [
    ("ἀγκών", "ῶνος", "ἀγκῶνος"),
    ("κραγγών", "όνος", "κραγγόνος"),
    ("ῥῆγμα", "ατος", "ῥῆγματος"),
    ("τένων", "οντος", "τένοντος"),
    ("ἐλπίς", "ίδος", "ἐλπίδος"),
    ("σωτήρ", "ῆρος", "σωτῆρος"),
    ("ποιμήν", "ένος", "ποιμένος"),
    ("βασιλεύς", "έως", "βασιλέως"),
    ("ἀγαθοδαίμων", "όνος", "ἀγαθοδαιμόνος"),
    ("αἰγίπους", "ποδος", "αἰγίποδος"),
    ("ἄθριξ", "τριχος", "ἄτριχος"),
    ("ἀΐς", "ιδος", "ἀϊδος"),
    ("τριόδους", "όδοντος", "τριόδοντος"),
])
def test_closed_abbreviations_preserve_the_lexical_prefix(word, suffix, expected):
    assert m.build_genitive_from_itype(word, suffix) == expected
    assert m.build_genitive_from_itype(unicodedata.normalize("NFD", word),
                                     unicodedata.normalize("NFD", suffix)) == expected


@pytest.mark.parametrize("word,suffix", [("τένων", "ου"), ("ξένος", "ιδος"),
                                        ("ὀρυζίτης", "πλακοῦς")])
def test_unsupported_pair_does_not_use_overlap_fallback(word, suffix):
    assert not m.build_genitive_from_itype(word, suffix)
    assert m.expand_noun(None, word, "m", suffix)[1] == "unsupported-genitive-abbreviation"


def test_source_suffix_breathing_is_withheld_instead_of_made_medial():
    assert m.expand_noun(None, "ἀγασυλλίς", "f", "ἴδος") == (set(), "unsupported-genitive-abbreviation")


@pytest.mark.parametrize("word,gender,source", [("λοχηγέω", "m", "λοχηγός"),
    ("σχινδάλαμος", "m", "σχινδ-αλμός"), ("ὕσκλ-ος", "m", "ύσκλος")])
def test_shared_prefix_alone_does_not_prove_a_full_genitive(word, gender, source):
    assert m.resolve_noun_genitive(word, gender, source)[1] == "unsupported-genitive-abbreviation"


@pytest.mark.parametrize("word,genitive", [("ἀνήρ", "ἀνδρός"), ("πατήρ", "πατρός"),
                                          ("μήτηρ", "μητρός"), ("Ζεύς", "Διός")])
def test_full_irregular_source_genitives_survive(word, genitive):
    evidence = {word: (genitive, "ἄλλου")}
    assert m.resolve_noun_genitive(word, "m", genitive, evidence) == (genitive, "")


def test_full_source_is_not_replaced_by_a_conflicting_variant():
    assert m.resolve_noun_genitive("πατήρ", "m", "πατρός", {"πατήρ": ("πατέρος",)}) == ("πατρός", "")


def test_unique_compatible_variant_completes_a_suffix():
    assert m.resolve_noun_genitive("ῥῆγμα", "n", "ατος", {"ῥῆγμα": ("ῥήγματος",)}) == ("ῥήγματος", "")


def test_conflicting_suffix_evidence_is_not_arbitrarily_selected():
    for variants in [("δαιμόνος", "δαίμονος"), ("δαίμονος", "δαιμόνος")]:
        assert m.resolve_noun_genitive("δαίμων", "m", "ονος", {"δαίμων": variants}) == ("", "conflicting-wiktionary-genitive")
    assert m.resolve_noun_genitive("πατήρ", "m", "", {"πατήρ": ("πατρός", "πατέρος")}) == ("", "ambiguous-wiktionary-genitive")
    assert m.resolve_noun_genitive("ἀγκών", "m", "ῶνος", {"ἀγκών": ("ἀγκώνος",)}) == ("", "conflicting-wiktionary-genitive")


def test_loader_merges_raw_sources_without_overwriting_variants(tmp_path, monkeypatch):
    en, el = tmp_path / "en.jsonl", tmp_path / "el.jsonl"
    def row(forms):
        return {"word": "ναῦς", "pos": "noun", "forms": forms}
    en.write_text(json.dumps(row([
        {"form": "νηός", "tags": ["genitive", "singular"]},
        {"form": "νηῶν", "tags": ["genitive", "plural"]},
    ]), ensure_ascii=False) + "\n")
    el.write_text(json.dumps(row([
        {"form": "νεώς", "tags": ["genitive"]},
        {"form": "νηός", "tags": ["genitive"]},
        {"form": "νεοῖν", "tags": ["genitive", "dual"]},
    ]), ensure_ascii=False) + "\n")
    monkeypatch.setattr(m, "KAIKKI_AG", en)
    monkeypatch.setattr(m, "KAIKKI_AG_EL", el)
    assert m.load_wiktionary_genitives() == {"ναῦς": ("νεώς", "νηός")}


def test_sophocles_preserves_the_exact_abbreviated_field(tmp_path, monkeypatch):
    (tmp_path / "chunk.xml").write_text(
        '<root><entry><orth>τένων</orth>, οντος, ὁ. A sinew.</entry>'
        '<entry><orth>ῥῆγμα</orth>, ατος, τό. A breach.</entry></root>')
    monkeypatch.setattr(expand_sophocles, "SOPH_DIR", tmp_path)
    entries = expand_sophocles.parse_sophocles_entries()
    assert entries["τένων"]["genitive"] == "οντος"
    assert entries["ῥῆγμα"]["genitive"] == "ατος"


@pytest.mark.skipif(not m.WTP_DB.exists() or importlib.util.find_spec("wikitextprocessor") is None,
                    reason="requires the optional pinned Wiktionary Lua cache and runtime")
@pytest.mark.parametrize("word,gender,suffix,full,expected,bad", [
    ("ἀγκών", "m", "ῶνος", "ἀγκῶνος", "ἀγκῶνος", "ωνός"),
    ("ῥῆγμα", "n", "ατος", "ῥήγματος", "ῥήγματος", "ᾶτος"),
    ("τένων", "m", "οντος", "τένοντος", "τένοντος", "τένωοντος"),
    ("ἐλπίς", "f", "ίδος", "ἐλπίδος", "ἐλπίδος", "*-δς"),
])
def test_real_suffix_and_full_genitive_give_identical_paradigms(word, gender, suffix, full, expected, bad):
    wtp = m.get_wtp()
    abbreviated, err = m.expand_noun(wtp, word, gender, suffix)
    assert not err
    explicit, err = m.expand_noun(wtp, word, gender, full)
    assert not err and abbreviated == explicit
    assert expected in abbreviated and bad not in abbreviated
    assert not any("*" in form for form in abbreviated)


def test_noun_notes_are_removed_at_the_table_boundary(monkeypatch):
    class Wtp:
        def start_page(self, word): pass
        def expand(self, template):
            return ('| class="form" | κἀγώ\n! class="notes-header" | Notes:\n'
                    '| class="notes" colspan="13" | * cluster *-δς\n|}\n')
    forms, error = m.expand_noun(Wtp(), "λόγος", "m", "λόγου")
    assert not error and forms == {"κἀγώ"}

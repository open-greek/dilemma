"""A head lsj9_forms.tsv prints with a hyphen is keyed by LSJ's joined headword."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
import expand_lsj as L  # noqa: E402

LISTED = frozenset({"Βακχεία", "ἀασιφρονία", "καθηδύνω", "ἀδαμαντοπέδιλος",
                    "καλοκἀγαθία", "λοῖσθος", "βραγχάω"})


@pytest.mark.parametrize("printed,head,rule", [
    # the export's division is dropped; accent and breathing stay as printed
    ("Βακχ-εία", "Βακχεία", "joined"),
    ("ἀασι-φρονία", "ἀασιφρονία", "joined"),
    ("καθ-ηδύνω", "καθηδύνω", "joined"),
    ("ἀδαμαντο-πέδι-λος", "ἀδαμαντοπέδιλος", "joined"),
    # LSJ's own coronis inside the word is not touched
    ("καλοκἀγαθ-ία", "καλοκἀγαθία", "joined"),
    # no hyphen: the head stands, listed or not
    ("Βακχεία", "Βακχεία", ""),
    ("ἄλλος", "ἄλλος", ""),
    # LSJ does not list the joined word: never a hyphenated key
    ("Σινα-ῖτις", "", "inner-hyphen"),
    ("-", "", "inner-hyphen"),
    # the headword list's soft hyphens and dashes are no part of a word either
    ("βραγ\u00adχάω", "βραγχάω", "joined"),
    ("βου\u00adμα\u00adστόω", "", "inner-hyphen"),
    ("ἐπιθρύξω—ἐω", "", "inner-hyphen"),
    ("βιβλ–", "", "inner-hyphen"),
])
def test_settle_lsj_head_reads_a_divided_head_under_the_joined_headword(
        printed, head, rule):
    assert L.settle_lsj_head(printed, LISTED) == (head, rule)
    if rule:
        assert L.settle_lsj_head(printed, frozenset()) == ("", "inner-hyphen")


def write_lsj9(directory, forms, headwords):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "lsj9_forms.tsv").write_text(
        "headword\tgrammar\tgenitive\tetymology\n"
        + "".join("\t".join(row) + "\n" for row in forms), encoding="utf-8")
    (directory / "lsj9_headwords.json").write_text(
        json.dumps(headwords, ensure_ascii=False), encoding="utf-8")


FORMS = [
    ("Βακχᾶς", "ὁ", "", ""),
    ("Βακχ-εία", "ἡ", "", ""),
    ("Βᾰκχ-εύς", "ὁ", "έως", ""),
    # an adjective printed with a hyphen before the noun printed without
    ("λοῖσθ-ος", "ον", "", ""),
    ("λοῖσθος", "ὁ", "", ""),
    ("καθ-ηδύνω", "", "", ""),
    # LSJ's headword list lacks the joined word
    ("Σινα-ῖτις", "ἡ", "ιδος", ""),
]
HEADWORDS = [
    {"id": 1, "headword": "Βακχᾶς", "grammar": "ὁ"},
    {"id": 2, "headword": "Βακχεία", "hyphenated": "Βακχ-εία", "grammar": "ἡ"},
    {"id": 3, "headword": "Βᾰκχεύς", "hyphenated": "Βᾰκχ-εύς", "grammar": "ὁ",
     "genitive": "έως"},
    {"id": 4, "headword": "λοῖσθος", "hyphenated": "λοῖσθ-ος", "grammar": "ον"},
    {"id": 5, "headword": "λοῖσθος", "grammar": "ὁ"},
    {"id": 6, "headword": "καθηδύνω", "hyphenated": "καθ-ηδύνω"},
    {"id": 7, "headword": "Βακχεῖον", "hyphenated": "Βακχ-εῖον", "grammar": "τό"},
    # the list's own artefacts: a soft hyphen inside a listed word, a soft
    # hyphen inside a word LSJ does not have, and a dash between two words
    {"id": 8, "headword": "καθ\u00adηδύνω"},
    {"id": 9, "headword": "βου\u00adμα\u00adστόω"},
    {"id": 10, "headword": "ἐπιθρύξω—ἐω"},
]
LIST_JOINED = {"καθ\u00adηδύνω": "καθηδύνω"}
LIST_MALFORMED = {"βου\u00adμα\u00adστόω": "inner-hyphen",
                  "ἐπιθρύξω—ἐω": "inner-hyphen"}


@pytest.fixture
def lsj9(tmp_path, monkeypatch):
    for name in ("LSJ9_DIR", "LSJ9_FORMS", "LSJ9_HEADWORDS", "LSJ9_GLOSSES",
                 "LSJ9_GLOSSES_SUPPLEMENT"):
        monkeypatch.setattr(L, name, getattr(L, name))
    write_lsj9(tmp_path / "lsj9", FORMS, HEADWORDS)
    L.configure_inputs(None, tmp_path / "lsj9")
    return tmp_path


def test_forms_export_is_keyed_by_the_joined_headword(lsj9):
    rules = L.new_lsj_head_rules()
    entries = L.parse_lsj9_entries(head_rules=rules)
    assert list(entries) == ["Βακχᾶς", "λοῖσθος", "Βακχεία", "Βακχεύς",
                             "καθηδύνω"]
    assert not any("-" in head or "-" in entry["headword"]
                   for head, entry in entries.items())
    # the joined entry keeps the grammar of its row and LSJ's quantity marks
    assert entries["Βακχεύς"] == {
        "headword": "Βακχεύς", "orth_orig": "Βᾰκχεύς", "article": "ὁ",
        "gender": "m", "itype": "έως", "genitive": "έως",
        "printed": "Βακχ-εύς"}
    # the row printed without a hyphen keeps its key and its grammar
    assert entries["λοῖσθος"]["gender"] == "m"
    assert "printed" not in entries["λοῖσθος"]
    assert rules == {
        "joined_heads": {"Βακχ-εία": "Βακχεία", "Βακχ-εύς": "Βακχεύς",
                         "λοῖσθ-ος": "λοῖσθος", "καθ-ηδύνω": "καθηδύνω"},
        "malformed_heads": {"Σινα-ῖτις": "inner-hyphen"}}


def test_merged_entries_have_one_entry_per_joined_headword(lsj9):
    rules = L.new_lsj_head_rules()
    entries = L.parse_lsj_entries(rules)
    # rows printed without a hyphen first, then the headword list's order
    assert list(entries) == ["Βακχᾶς", "λοῖσθος", "Βακχεία", "Βακχεύς",
                             "καθηδύνω", "Βακχεῖον"]
    assert entries["Βακχεία"]["gender"] == "f"
    assert entries["Βακχεύς"]["genitive"] == "έως"
    # only the headword list has this entry
    assert entries["Βακχεῖον"]["gender"] == "n"
    assert rules["joined_heads"]["Βακχ-εία"] == "Βακχεία"
    assert rules["joined_heads"]["καθ\u00adηδύνω"] == "καθηδύνω"
    assert L.head_rules_record(rules)["malformed_heads"] == {
        "count": 3, "heads": {"Σινα-ῖτις": "inner-hyphen", **LIST_MALFORMED}}
    assert not any(mark in head for head in entries for mark in "-\u00ad—–")


def test_noun_step_generates_under_the_joined_head_and_records_the_rule(
        lsj9, monkeypatch):
    table = lsj9 / "ag_lookup.json"
    table.write_text(json.dumps({"κοινόν": "base"}, ensure_ascii=False),
                     encoding="utf-8")
    monkeypatch.setattr(L, "AG_LOOKUP", table)
    monkeypatch.setattr(L, "noun_generation_input_hashes", lambda *a, **kw: {})
    monkeypatch.setattr(L, "load_wiktionary_forms", lambda: {})
    monkeypatch.setattr(L, "load_wiktionary_genitives", lambda: {})
    monkeypatch.setattr(L, "get_wtp", lambda: None)
    calls = []

    def expand_noun(wtp, head, gender, genitive="", wikt_genitives=None):
        calls.append(head)
        return {head, head + "ς"}, ""

    monkeypatch.setattr(L, "expand_noun", expand_noun)
    L.expand_all()
    lookup = json.loads(table.read_text(encoding="utf-8"))
    assert "Βακχεία" in calls and "Βακχεύς" in calls
    assert not any("-" in head for head in calls)
    assert lookup["Βακχείας"] == "Βακχεία"
    assert not any("-" in form or "-" in lemma for form, lemma in lookup.items())
    audit = json.loads((lsj9 / "ag_noun_generation_audit.json")
                       .read_text(encoding="utf-8"))
    assert audit["head_rules"] == {
        "joined_heads": {"count": 5, "heads": {
            "Βακχ-εία": "Βακχεία", "Βακχ-εύς": "Βακχεύς",
            "καθ-ηδύνω": "καθηδύνω", "λοῖσθ-ος": "λοῖσθος", **LIST_JOINED}},
        "malformed_heads": {"count": 3, "heads": {
            "Σινα-ῖτις": "inner-hyphen", **LIST_MALFORMED}}}
    # superseded lemmas and resolutions name the joined head
    assert not any("-" in lemma for lemma in audit["superseded_lemmas"])
    assert not any("-" in lemma for lemma in audit["resolutions"])


def test_verb_step_never_expands_a_hyphenated_head(lsj9, monkeypatch):
    table = lsj9 / "ag_lookup.json"
    table.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(L, "AG_LOOKUP", table)
    monkeypatch.setattr(L, "load_wiktionary_forms", lambda: {})
    monkeypatch.setattr(L, "load_lsj_head_texts", lambda: {})
    monkeypatch.setattr(L, "load_lsj_supplement_head_texts", lambda: {})
    monkeypatch.setattr(L, "verify_lsj_glosses",
                        lambda: {"sha256": "0" * 64, "size": 0})
    monkeypatch.setattr(L, "get_wtp", lambda: None)
    calls = []

    def expand_verb(wtp, head, head_text="", supplement_text="",
                    principal_part_forms=None):
        calls.append(head)
        return {head, head[:-1] + "εις"}, ""

    monkeypatch.setattr(L, "expand_verb", expand_verb)
    # the corpus checks read the attestation and treebank tables of data/
    L.expand_verbs(corpus_checks=False)
    # not καθ-ηδύνω, and none of the list's broken spellings
    assert calls == ["καθηδύνω"]
    lookup = json.loads(table.read_text(encoding="utf-8"))
    assert lookup["καθηδύνεις"] == "καθηδύνω"
    assert not any(mark in form or mark in lemma
                   for form, lemma in lookup.items() for mark in "-\u00ad—–")

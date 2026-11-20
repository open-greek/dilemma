"""Lookup readings of #147: forms of a displaced head, AGDT lemma validation
and reviewed readings.

The builder runs on a small fixture in a temporary data directory; no
production data is read or written.
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_lookup_db as B  # noqa: E402
import test_lookup_build_corrections as T  # noqa: E402


# Pure helpers --------------------------------------------------------------

@pytest.mark.parametrize("form,head,lemma,reason", [
    ("ἡμέρας", "ἡμέρα", "ἡμέρας", "form_with_an_entry_of_its_own"),
    ("πλείονα", "πολύς", "πλείων", "form_with_an_entry_of_its_own"),
    ("ἴριδος", "Ἶρις", "ἶρις", "name_and_word"),
    ("ῥᾳθυμίαν", "ῥαθυμία", "ῥᾳθυμία", "spelling_of_the_citation_form"),
    ("γλωττίδος", "γλωττίς", "γλωσσίς", "spelling_of_the_citation_form"),
    ("εἰσιέναι", "εἰσέρχομαι", "εἴσειμι", "compound_of_eimi"),
    ("τιτύσκετο", "τιτύσκομαι", "τιτύσκω", "middle_and_active"),
    ("πλευρᾶς", "πλευρά", "πλευρόν", None),
    ("εἴσεται", "οἶδα", "εἶμι", None),
    ("ὀποῦ", "ὀπός", "ὄψ", None),
])
def test_reading_convention(form, head, lemma, reason):
    base = {"πλείων": "πολύς"}
    assert B.reading_convention(form, head, lemma, base) == reason


def _restore(ag, base, restored, heads, pairs, key_evidence=None):
    candidates = B.paradigm_form_candidates(
        ag, restored=restored, base=base, heads=heads,
        corpus_forms={pair["form"] for pair in pairs})
    case = {form for form, (_, _, kind) in candidates.items() if kind == "case"}
    evidence = B.corpus_form_evidence(pairs, candidates.keys(), grave_folded=case)
    return candidates, B.restore_displaced_paradigm_forms(
        ag, candidates=candidates, evidence=evidence, heads=heads,
        key_evidence=key_evidence or {})


def test_forms_follow_a_restored_head():
    # κλαίω took its key back from κλείω (restore_displaced_lexical_heads);
    # its forms that the base files under κλαίω follow, with or without
    # corpus tokens, unless the corpus prefers κλείω for the form.
    ag = {"κλαίω": "κλαίω", "κλαίειν": "κλείω", "κλαίοιτε": "κλείω",
          "κλαιειν": "κλείω", "κλαίουσι": "κλείω", "κλείουσι": "κλείω"}
    base = {"κλαίω": "κλαίω", "κλαίειν": "κλαίω", "κλαίοιτε": "κλαίω",
            "κλαίουσι": "κλαίω", "κλείουσι": "κλείω"}
    restored = [{"form": "κλαίω", "lemma": "κλείω", "head": "κλαίω",
                 "head_count": 40, "lemma_count": 0}]
    pairs = [{"form": "κλαίειν", "lemma": "κλαίω", "count": 75},
             {"form": "κλαίουσι", "lemma": "κλείω", "count": 3},
             {"form": "κλαίουσι", "lemma": "κλαίω", "count": 1}]
    _, records = _restore(ag, base, restored, {"κλαίω", "κλείω"}, pairs)
    assert ag["κλαίειν"] == ag["κλαίοιτε"] == "κλαίω"
    assert ag["κλαιειν"] == "κλαίω"            # the looser key follows
    assert ag["κλαίουσι"] == "κλείω"           # the corpus prefers κλείω
    assert ag["κλείουσι"] == "κλείω"           # the base reads it so
    assert {r["form"]: r["kind"] for r in records} == {
        "κλαίειν": "restored", "κλαίοιτε": "restored", "κλαιειν": "alias"}


def test_lowercase_forms_follow_a_head_restored_from_its_name():
    # ἔρως took its key back from Ἔρως; Wiktionary files ἔρωτα under the
    # god as well. The capitalised forms stay the name's, and a lowercase
    # form needs two tokens and more than the name has.
    ag = {"ἔρως": "ἔρως", "ἔρωτα": "Ἔρως", "Ἔρωτα": "Ἔρως",
          "ἐρώτων": "Ἔρως", "ἔρωτι": "Ἔρως"}
    base = {"ἔρως": "Ἔρως", "ἔρωτα": "Ἔρως", "Ἔρωτα": "Ἔρως",
            "ἔρωτι": "Ἔρως"}
    restored = [{"form": "ἔρως", "lemma": "Ἔρως", "head": "ἔρως",
                 "head_count": 423, "lemma_count": 0}]
    pairs = [{"form": "ἔρωτα", "lemma": "ἔρως", "count": 581},
             {"form": "Ἔρωτα", "lemma": "Ἔρως", "count": 111},
             {"form": "ἔρωτι", "lemma": "ἔρως", "count": 1}]
    _restore(ag, base, restored, {"ἔρως", "Ἔρως"}, pairs)
    assert ag["ἔρωτα"] == "ἔρως"
    assert ag["Ἔρωτα"] == "Ἔρως"
    assert ag["ἔρωτι"] == ag["ἐρώτων"] == "Ἔρως"


def test_forms_of_a_head_that_reads_itself_need_a_clear_corpus_majority():
    ag = {"πλευρά": "πλευρά", "πλευρᾶς": "πλευρόν", "πλευρῶν": "πλευρόν",
          "αἰνέω": "αἰνέω", "αἰνέσει": "αἴνεσις",
          "ἡμέρα": "ἡμέρα", "ἡμέρας": "ἡμέρας",
          "πλευρόν": "πλευρόν", "ἀφελής": "ἀφελής", "ἀφελεῖ": "ἀφαιρέω",
          "πολύς": "πολύς", "πλείονα": "πλείων", "πλείων": "πλείων"}
    base = {"πλευρᾶς": "πλευρά", "πλευρῶν": "πλευρά", "αἰνέσει": "αἰνέω",
            "ἡμέρας": "ἡμέρα", "ἀφελεῖ": "ἀφελής", "πλείονα": "πολύς",
            "πλείων": "πολύς"}
    heads = {"πλευρά", "πλευρόν", "αἰνέω", "αἴνεσις", "ἡμέρα", "ἡμέρας",
             "ἀφελής", "ἀφαιρέω", "πολύς", "πλείων"}
    pairs = [{"form": "πλευρᾶς", "lemma": "πλευρά", "count": 300},
             {"form": "πλευρῶν", "lemma": "πλευρά", "count": 1},
             {"form": "αἰνέσει", "lemma": "αἰνέω", "count": 14},
             {"form": "αἰνέσει", "lemma": "αἴνεσις", "count": 6},
             {"form": "ἡμέρας", "lemma": "ἡμέρα", "count": 5509},
             {"form": "ἀφελεῖ", "lemma": "ἀφελής", "count": 34},
             {"form": "ἀφελεῖ", "lemma": "ἀφαιρέω", "count": 21},
             {"form": "πλείονα", "lemma": "πολύς", "count": 962}]
    _restore(ag, base, [], heads, pairs)
    assert ag["πλευρᾶς"] == "πλευρά"
    assert ag["πλευρῶν"] == "πλευρόν"          # one token is no evidence
    assert ag["αἰνέσει"] == "αἴνεσις"          # both readings are common
    assert ag["ἀφελεῖ"] == "ἀφαιρέω"
    assert ag["ἡμέρας"] == "ἡμέρας"            # lemma conventions stay
    assert ag["πλείονα"] == "πλείων"


# Builder on a fixture ------------------------------------------------------

def test_agdt_lemmas_are_validated_against_the_headword_lists(tmp_path):
    perseus = [
        {"form": "Κῦρόν", "lemma": "ξψρυς", "pos": "n"},            # Beta Code
        {"form": "Σκιπίωνα", "lemma": "Σκιπιων", "pos": "n"},       # no accent
        {"form": "ψηφισάμενοι", "lemma": "ψηφίζομαι", "pos": "v"},  # unlisted
        {"form": "λόγου", "lemma": "λόγος", "pos": "n"},
        {"form": "Λόγῳ", "lemma": "Λόγος", "pos": "n"},             # case fold
        {"form": "περιῄει", "lemma": "περί-εἶμι", "pos": "v"},      # joined
    ]
    glaux = [{"form": "Κῦρόν", "lemma": "Κῦρος", "pos": "noun", "count": 3},
             {"form": "ψηφισάμενοι", "lemma": "ψηφίζω", "pos": "verb",
              "count": 43}]
    data, env = T._fixture(tmp_path, {
        "grc_lexical_heads.json": {
            "heads": ["σύνοχος", "γε", "λόγος", "μοναχικός", "λαβή", "κλεῖδα",
                      "περίειμι"],
            "vernacular_heads": []},
        "ag_headwords.json": ["λόγος", "Κῦρος", "ψηφίζω", "Σκιπίων"],
        "perseus_pairs.json": perseus,
        "glaux_pairs.json": glaux,
    })
    result = T._build(data, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 gap-fills under a lemma no headword list has" in result.stdout
    rows = T._rows(data)
    assert rows["Κῦρόν"] == "Κῦρος"               # GLAUx fills the key
    assert rows["ψηφισάμενοι"] == "ψηφίζω"
    assert "Σκιπίωνα" not in rows
    assert rows["λόγου"] == "λόγος" and rows["Λόγῳ"] == "λόγος"
    assert rows["περιῄει"] == "περίειμι"
    rejected = T._rejections(data)
    for form, lemma in (("Κῦρόν", "ξψρυς"), ("Σκιπίωνα", "Σκιπιων"),
                        ("ψηφισάμενοι", "ψηφίζομαι")):
        assert ("AG", B.PERSEUS_PAIRS_PATH.name, "corpus_lemma_not_a_headword",
                form, lemma) in rejected


def test_reviewed_lookup_readings():
    readings = B.REVIEWED_LOOKUP_READINGS
    assert readings["Δι᾽"] == "Ζεύς"
    assert readings["ἀδάματ᾽"] == readings["ἀδάματος"] == "ἀδάματος"
    assert readings["ξυνέρξεθ᾽"] == readings["συνέργω"] == "συνέργω"
    assert readings["Φαεινίδα"] == "Φαεινίς"
    # Two names of one word stay as the full forms read them (#147).
    for key in ("εἶπ᾽", "ἴθ᾽", "ἰδόντ᾽", "πλεῖστ᾽"):
        assert key not in readings


def test_a_listed_inflected_form_follows_but_a_wiktionary_entry_does_not():
    # The dictionary head lists hold inflected forms too (ἕσπετο); a form
    # Wiktionary defines in an entry of its own is the head rule's to judge.
    ag = {"ἕπομαι": "ἕπομαι", "ἕσπετο": "ἕπω", "ἑπόμενος": "ἕπω",
          "πλευρά": "πλευρόν", "πλευρᾶς": "πλευρόν"}
    base = {"ἕπομαι": "ἕπομαι", "ἕσπετο": "ἕπομαι", "ἑπόμενος": "ἕπομαι",
            "πλευρά": "πλευρά", "πλευρᾶς": "πλευρά"}
    restored = [{"form": "ἕπομαι", "lemma": "ἕπω", "head": "ἕπομαι",
                 "head_count": 900, "lemma_count": 0}]
    pairs = [{"form": "ἕσπετο", "lemma": "ἕπομαι", "count": 32},
             {"form": "ἑπόμενος", "lemma": "ἕπομαι", "count": 152},
             {"form": "πλευρᾶς", "lemma": "πλευρά", "count": 300}]
    heads = {"ἕπομαι", "ἕπω", "ἕσπετο", "ἑπόμενος", "πλευρά", "πλευρόν"}
    candidates = B.paradigm_form_candidates(
        ag, restored=restored, base=base, heads=heads,
        wiktionary_heads={"ἑπόμενος"},
        corpus_forms={pair["form"] for pair in pairs})
    B.restore_displaced_paradigm_forms(
        ag, candidates=candidates, heads=heads,
        evidence=B.corpus_form_evidence(pairs, candidates.keys()))
    assert ag["ἕσπετο"] == "ἕπομαι"
    assert ag["ἑπόμενος"] == "ἕπω"
    # πλευρά reads πλευρόν in the table but itself in the base.
    assert ag["πλευρᾶς"] == "πλευρά"


def test_names_the_agdt_check_leaves_to_a_lowercase_glaux_head():
    readings = B.REVIEWED_LOOKUP_READINGS
    assert readings["Κῦρόν"] == "Κῦρος" and readings["Πλαγγόνα"] == "Πλαγγών"
    assert readings["Γλαῦκός"] == "Γλαῦκος"

"""Recover malformed source name keys from independent exact name evidence."""

import copy
import json
from pathlib import Path

import pytest

from build import build_glaux_pairs as b


COMMIT = "a" * 40


@pytest.fixture
def evidence(tmp_path):
    lexical = tmp_path / "heads.json"
    lexical.write_text(json.dumps({"schema_version": 2, "heads": {
        "Κέλσος": ["lgpn_names", "wiktionary_el_grc"],
        "Ἰουλία": ["lgpn_names"], "Αὐρηλία": ["wiktionary_el_grc"],
        "Λόγος": ["wiktionary_el_grc"], "γνωστός": ["lsj9_headwords"],
    }}, ensure_ascii=False))
    pairs = tmp_path / "ag_pairs.json"
    pairs.write_text(json.dumps([
        {"form": head, "lemma": head, "pos": "name"}
        for head in ("Κέλσος", "Ἰουλία", "Αὐρηλία")
    ] + [{"form": "Λόγος", "lemma": "Λόγος", "pos": "noun"}], ensure_ascii=False))
    return b.load_name_evidence(lexical, pairs)


def row(form, lemma="ξελσυς", case="nominative", **extra):
    return {"form": form, "lemma": lemma, "pos": "noun",
            "tags": ["singular", "masculine", case], "count": 3, **extra}


def test_unique_independent_nominative_repairs_only_capitalized_noun_readings(evidence):
    original = [row("Κέλσος"), row("Κέλσου", case="genitive"),
                row("Κέλσῳ", case="dative"), row("κέλσου", case="genitive"),
                row("Κέλσον", pos="verb"),
                row("Κέλσε", analyses=[["noun", ["vocative"], 2],
                                        ["verb", ["imperative"], 1]])]
    before = copy.deepcopy(original)
    corrected, audit = b.repair_name_lemmas(original, evidence, source_commit=COMMIT)
    assert [r["lemma"] for r in corrected] == ["Κέλσος"] * 3 + ["ξελσυς"] * 3
    assert original == before
    assert corrected[1]["tags"] == original[1]["tags"]
    assert corrected[1]["count"] == original[1]["count"]
    assert audit["source_commit"] == COMMIT
    assert audit["corrections"][1]["original"] == original[1]
    assert audit["corrections"][1]["corrected"] == corrected[1]
    assert audit["corrections"][1]["evidence"]["lexical_sources"] == [
        "lgpn_names", "wiktionary_el_grc"]
    assert b.unapply_name_corrections(corrected, audit) == original
    assert b.repair_name_lemmas(original, evidence, source_commit=COMMIT) == (corrected, audit)


@pytest.mark.parametrize("reverse", [False, True])
def test_competing_nominatives_keep_every_original_reading(evidence, reverse):
    rows = [row("Ἰουλία", "ιυλια"), row("Αὐρηλία", "ιυλια"),
            row("Ἰουλίας", "ιυλια", "genitive")]
    if reverse:
        rows.reverse()
    corrected, audit = b.repair_name_lemmas(rows, evidence, source_commit=COMMIT)
    assert corrected == rows
    assert audit["corrections"] == []
    assert audit["inferred_groups"] == {"ιυλια": ["Αὐρηλία", "Ἰουλία"]}


@pytest.mark.parametrize("rows", [
    [row("Κέλσου", case="genitive")],
    [row("Κέλσος", case="accusative")],
    [row("Κέλσος", tags=["plural", "nominative"])],
    [row("Λόγος")],  # Capitalized lexical noun is not a source name reading.
    [row("Κέλσος", "γνωστος")],  # Recognized head, even without its accent.
    [row("Κέλσος", "Ξελσυς")],  # Do not rewrite Greek citation/dialect forms.
    [row("Κέλσος", "ξέλσυς")],
])
def test_missing_anchor_or_nonmalformed_key_is_not_rewritten(evidence, rows):
    corrected, audit = b.repair_name_lemmas(rows, evidence, source_commit=COMMIT)
    assert corrected == rows
    assert not audit["corrections"]


def test_existing_target_reading_does_not_get_silently_merged(evidence):
    rows = [row("Κέλσος"), row("Κέλσου", case="genitive"),
            row("Κέλσου", "Κέλσος", "genitive", count=7)]
    corrected, audit = b.repair_name_lemmas(rows, evidence, source_commit=COMMIT)
    assert [r["lemma"] for r in corrected] == ["Κέλσος", "ξελσυς", "Κέλσος"]
    assert [r["count"] for r in corrected] == [3, 3, 7]
    assert b.unapply_name_corrections(corrected, audit) == rows


def test_reversal_rejects_changed_artifact(evidence):
    corrected, audit = b.repair_name_lemmas([row("Κέλσος")], evidence, source_commit=COMMIT)
    corrected[0] = {**corrected[0], "count": 100}
    with pytest.raises(ValueError, match="do not match"):
        b.unapply_name_corrections(corrected, audit)


def test_required_evidence_and_commit_fail_clearly(tmp_path, evidence):
    with pytest.raises(FileNotFoundError):
        b.load_name_evidence(tmp_path / "missing.json", tmp_path / "also_missing.json")
    with pytest.raises(ValueError, match="full source commit"):
        b.repair_name_lemmas([row("Κέλσος")], evidence, source_commit=None)


def test_extractor_audit_pins_contributing_xml_and_round_trips(evidence, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(b.__file__).parent))
    xml = tmp_path / "xml"
    xml.mkdir()
    source = xml / "9999-001.xml"
    source.write_text('<treebank><sentence id="1">'
                      '<word id="1" form="Κέλσος" lemma="ξελσυς" postag="n-s---mn-"/>'
                      '<word id="2" form="Κέλσου" lemma="ξελσυς" postag="n-s---mg-"/>'
                      '</sentence></treebank>')
    audit = {}
    corrected = b.extract_glaux(xml, name_evidence=evidence,
                                correction_audit=audit, source_commit=COMMIT)
    assert [r["lemma"] for r in corrected] == ["Κέλσος", "Κέλσος"]
    assert audit["source_xml"] == {source.name: b._fingerprint(source)}
    assert all(r["source_xml"] == [source.name] for r in audit["corrections"])
    assert str(tmp_path) not in json.dumps(audit)
    assert [r["lemma"] for r in b.unapply_name_corrections(corrected, audit)] == ["ξελσυς"] * 2

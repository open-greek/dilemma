"""Complete source-owned names survive the sparse POS-table selection."""
from __future__ import annotations

import hashlib
import json

import pytest

from build import build_treebank_pos_lookup as build
from export_hunspell import (
    cased_form_key,
    exact_form_key,
    include_attested_name_ownership,
    load_capital_name_owners,
    load_diorisis_quote_keys,
    FormProfileEvidence,
    lexical_head_key,
)


def test_pos_audit_keeps_unambiguous_names_and_all_source_readings(tmp_path, monkeypatch):
    heads = tmp_path / "grc_lexical_heads.json"
    heads.write_text(json.dumps({"schema_version": 2, "heads": {
        "Ἱπποκράτης": ["source-a"], "Ἱπποκράτη": ["source-b"],
        "Ἰταλικός": ["source-a"], "σαφής": ["source-a"],
    }}))
    monkeypatch.setattr(build, "DATA_DIR", tmp_path)
    monkeypatch.setattr(build, "TREEBANK_DIRS", [])
    monkeypatch.setattr(build, "AGDT_DIR", tmp_path / "no-agdt")
    monkeypatch.setattr(build, "LOOKUP_DB_PATH", tmp_path / "no-lookup")
    monkeypatch.setattr(build, "OUTPUT_PATH", tmp_path / "pos.json")
    monkeypatch.setattr(build, "AUDIT_PATH", None)
    sources = {
        "GLAUX": [{"form": "Ἱπποκράτην", "lemma": "Ἱπποκράτης", "pos": "noun"},
                  {"form": "Ἰταλικῶν", "lemma": "Ἰταλικός", "pos": "adj"}],
        "DIORISIS": [{"form": "Ἱπποκράτην", "lemma": "Ἱπποκράτη", "pos": "noun"},
                     {"form": "Ἱπποκράτην", "lemma": "Ἱπποκράτη", "pos": "noun",
                      "tags": ["accusative"]},
                     {"form": "Ἱπποκράτην", "lemma": "Ἱπποκράτης", "pos": "noun"},
                     {"form": "Ἱπποκράτην", "lemma": "Ἱπποκράτιος", "pos": "noun"},
                     {"form": "Σαφής", "lemma": "σαφής", "pos": "adj"},
                     {"form": "Ἱπποκράτην", "lemma": "Ἱπποκράτης", "pos": "verb"}],
    }
    for source in ("GLAUX", "DIORISIS", "NT", "PTNK", "PEDALION"):
        path = tmp_path / (source + ".json")
        monkeypatch.setattr(build, source + "_PAIRS_PATH", path)
        if source in sources:
            path.write_text(json.dumps(sources[source]))
    # Neither of these names needs POS disambiguation, so the runtime table
    # can stay empty while the independent ownership evidence remains full.
    assert build.build_lookup() == {}
    audit_path = (tmp_path / "pos.audit.json")
    audit = json.loads(audit_path.read_text())
    assert audit["schema_version"] == 2
    assert set(audit["capital_name_owners"]) == {"Ἱπποκράτην", "Ἰταλικῶν"}
    assert audit["capital_name_owners"]["Ἱπποκράτην"] == [
        {"upos": "NOUN", "lemma": "Ἱπποκράτη", "source_votes": {"diorisis": 1}},
        {"upos": "NOUN", "lemma": "Ἱπποκράτης",
         "source_votes": {"diorisis": 1, "glaux": 2}},
    ]
    readings = load_capital_name_owners(audit_path, lexical_path=heads)
    ownership = {exact_form_key("Ἱπποκράτην"): 2}
    evidence = FormProfileEvidence({}, {}, {}, {}, cased={"Ἱπποκράτην": 271})
    retained = include_attested_name_ownership(
        ownership, evidence, readings=readings,
        lexical_heads=frozenset(map(lexical_head_key, ("Ἱπποκράτης", "Ἱπποκράτη"))),
    )
    assert retained[0]["lemmas"] == ["Ἱπποκράτη", "Ἱπποκράτης"]


def test_name_audit_requires_complete_schema_and_current_heads(tmp_path):
    heads = tmp_path / "heads.json"
    heads.write_text("{}")
    path = tmp_path / "audit.json"
    with pytest.raises(FileNotFoundError):
        load_capital_name_owners(path, lexical_path=heads)
    path.write_text(json.dumps({"schema_version": 1, "language": "grc"}))
    with pytest.raises(ValueError, match="complete capital name owners"):
        load_capital_name_owners(path, lexical_path=heads)
    payload = {"schema_version": 2, "language": "grc", "capital_name_owners": {},
               "inputs": {"grc_lexical_heads": {"size": 0, "sha256": "stale"}}}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="stale lexical-head"):
        load_capital_name_owners(path, lexical_path=heads)
    payload["inputs"]["grc_lexical_heads"] = {
        "size": heads.stat().st_size, "sha256": hashlib.sha256(heads.read_bytes()).hexdigest()}
    payload["capital_name_owners"] = {"Ἱπποκράτην": [
        {"upos": "NOUN", "lemma": "Ἱπποκράτης", "source_votes": {"glaux": 0}}]}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid capital name source votes"):
        load_capital_name_owners(path, lexical_path=heads)


def test_resolved_plain_terminal_quote_remains_excluded_from_elision(tmp_path):
    path = tmp_path / "quotes.json"
    reason = "terminal_quote_with_same_source_cell"
    path.write_text(json.dumps({"schema_version": 1, "sources": [{"records": [
        {"original_beta": "e)pa/ion'", "normalized_beta": "e)pa/ion", "reason": reason},
    ]}]}))
    keys, meta = load_diorisis_quote_keys(path)
    assert keys == {cased_form_key("ἐπάιον᾽")}
    assert meta["tokens_by_reason"] == {reason: 1}

"""Reviewed lexical input stays sourced and separate from corpus attestations."""
import json

import pytest

import export_mg_polytonic as mg


def source(tmp_path):
    payload = json.loads(mg.REVIEWED_LOANWORDS.read_text())
    path = tmp_path / "reviewed.json"
    path.write_text(json.dumps(payload, ensure_ascii=False))
    return path, payload


def test_reviewed_word_requires_independent_frequency_evidence(tmp_path):
    path, _ = source(tmp_path)
    assert mg.reviewed_loanword_forms({"μάρκετ": 49}, set(), set(), frozenset(), path) == {}
    assert mg.reviewed_loanword_forms({"μάρκετ": 1722}, set(), set(), frozenset(), path) == {"μάρκετ": 1722}


@pytest.mark.parametrize("covered,rejected", [(True, False), (False, True)])
def test_reviewed_word_cannot_override_existing_or_rejected_forms(tmp_path, covered, rejected):
    path, _ = source(tmp_path)
    assert mg.reviewed_loanword_forms({"μάρκετ": 1722}, set(),
        {"μάρκετ"} if covered else set(),
        frozenset({"μάρκετ"}) if rejected else frozenset(), path) == {}
    assert mg.reviewed_loanword_forms({"μάρκετ": 1722}, {"μάρκετ"}, set(), frozenset(), path) == {}


@pytest.mark.parametrize("change", ["one_source", "unsupported_form", "wrong_language", "duplicate"])
def test_reviewed_source_validation(tmp_path, change):
    path, payload = source(tmp_path)
    entry = payload["entries"][0]
    if change == "one_source":
        entry["evidence"].pop()
    elif change == "unsupported_form":
        entry["form"] = "λάθος"
    elif change == "wrong_language":
        entry["language"] = "grc"
    else:
        payload["entries"].append(entry.copy())
    path.write_text(json.dumps(payload, ensure_ascii=False))
    with pytest.raises(ValueError):
        mg.reviewed_loanword_forms({"μάρκετ": 1722}, set(), set(), frozenset(), path)


def test_output_records_reviewed_provenance_without_claiming_attestation(tmp_path, monkeypatch):
    monkeypatch.setattr(mg, "bucket_source", lambda *_: "fixture frequencies")
    selection = mg.ModernGreekList({}, [], {}, {}, generated=frozenset({"μάρκετ"}),
        buckets={"μάρκετ": "M"}, reviewed_loanwords=frozenset({"μάρκετ"}))
    stats = mg.write_selection(selection, tmp_path, variant=mg.SHIPPING_VARIANT, source="fixture")
    assert stats["attested"] == 0
    assert stats["generated"] == stats["reviewed_loanwords"] == 1
    assert (tmp_path / "grc_mg_polytonic.dic").read_text().splitlines() == [
        "1", "μάρκετ\tfr:M mg:generated mg:reviewed"]
    assert "reviewed_loanwords: 1\n" in (tmp_path / "grc_mg_polytonic.version").read_text()

"""Raw GLAUx consumers apply only the original readings in the shared audit."""

from collections import Counter, defaultdict
import json
from pathlib import Path

import pytest

from build import build_glaux_pairs as pairs
from build import build_lemma_attestation as attestation
from build import build_tagger_data as tagger


COMMIT = "a" * 40


@pytest.fixture
def audited(tmp_path, monkeypatch):
    xml = tmp_path / "xml"
    xml.mkdir()
    metadata = tmp_path / "metadata.txt"
    metadata.write_text("TLG\tDIALECT\tSOURCE_LICENSE\tTREEBANK_ANNOTATIONS\n"
                        "9999-001\tAttic\tCC BY-SA\tGorman\n")
    source = xml / "9999-001.xml"
    source.write_text('<treebank><sentence id="1" analysis="auto">'
                      '<word id="1" form="Κέλσος" lemma="ξελσυς" postag="n-s---mn-"/>'
                      '<word id="2" form="Κέλσου" lemma="ξελσυς" postag="n-s---mg-"/>'
                      '<word id="3" form="Κέλσον" lemma="ξελσυς" postag="a-s---ma-"/>'
                      '</sentence><sentence id="2" analysis="manual">'
                      '<word id="4" form="Κέλσος" lemma="ξελσυς" postag="n-s---mn-"/>'
                      '</sentence></treebank>')
    lexical = tmp_path / "heads.json"
    lexical.write_text(json.dumps({"schema_version": 2,
        "heads": {"Κέλσος": ["lgpn_names", "wiktionary_el_grc"]}}))
    names = tmp_path / "ag_pairs.json"
    names.write_text(json.dumps([{"form": "Κέλσος", "lemma": "Κέλσος", "pos": "name"}]))
    evidence = pairs.load_name_evidence(lexical, names)
    audit = {}
    corrected = pairs.extract_glaux(xml, metadata_path=metadata,
                                    name_evidence=evidence, correction_audit=audit,
                                    source_commit=COMMIT)
    audit_path = tmp_path / "name_changes.json"
    audit_path.write_text(json.dumps(audit))

    def load():
        return pairs.load_name_corrections(audit_path, xml, lexical_path=lexical,
                                           name_pairs_path=names, metadata_path=metadata)
    return {"xml": xml, "source": source, "metadata": metadata,
            "lexical": lexical, "names": names, "audit_path": audit_path,
            "audit": audit, "corrected": corrected, "load": load}


@pytest.mark.parametrize("field", ["form", "lemma", "pos", "case", "source"])
def test_shared_apply_requires_full_original_reading(audited, field):
    values = {"source": "9999-001.xml", "form": "Κέλσου", "lemma": "ξελσυς",
              "postag": "n-s---mg-"}
    corrections = audited["load"]()
    assert pairs.apply_name_correction(corrections, values["source"], values["form"],
                                      values["lemma"], values["postag"]) == "Κέλσος"
    if field == "pos":
        values["postag"] = "a-s---mg-"
    elif field == "case":
        values["postag"] = "n-s---md-"
    else:
        values[field] = {"form": "κέλσου", "lemma": "ἄλλος",
                         "source": "9999-002.xml"}[field]
    assert pairs.apply_name_correction(corrections, values["source"], values["form"],
                                      values["lemma"], values["postag"]) == values["lemma"]


@pytest.mark.parametrize("field", ["source", "lexical", "names", "metadata", "builder"])
def test_stale_evidence_fails_before_use(audited, field):
    if field == "builder":
        payload = json.loads(audited["audit_path"].read_text())
        payload["builder_sha256"] = "0" * 64
        audited["audit_path"].write_text(json.dumps(payload))
    else:
        path = audited[field]
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="stale"):
        audited["load"]()


def test_attestation_uses_corrections_without_counting_excluded_readings(audited):
    corrections = audited["load"]()
    profiles = defaultdict(attestation.LemmaProfile)
    stats = Counter()
    attestation.process_glaux(audited["xml"], {}, profiles, 0, stats,
                               gorman_stems={"9999-001"}, name_corrections=corrections)
    assert profiles["Κέλσος"].total == 2
    # The different adjective reading remains under its original source key.
    assert profiles["ξελσυς"].total == 1
    assert stats["glaux_tokens"] == 3


def test_tagger_uses_same_corrections_and_keeps_other_source_readings(audited, monkeypatch):
    monkeypatch.setattr(tagger, "GLAUX_XML", audited["xml"])
    monkeypatch.setattr(tagger, "GLAUX_META", audited["metadata"])
    corrections = audited["load"]()
    # An unlisted source cannot inherit another source's correction.
    (audited["xml"] / "9999-002.xml").write_text(
        '<treebank><sentence><word id="1" form="Κέλσος" '
        'lemma="ξελσυς" postag="n-s---mn-"/></sentence></treebank>')
    sentences, excluded = tagger.extract(name_corrections=corrections)
    assert excluded == 0
    assert [token["lemma"] for token in sentences[0]["tokens"]] == [
        "Κέλσος", "Κέλσος", "ξελσυς"]
    assert sentences[1]["tokens"][0]["lemma"] == "ξελσυς"
    assert len(sentences) == 2  # The manual held-out sentence was not read.


def test_loading_audit_does_not_mutate_reversal_information(audited):
    before = audited["audit_path"].read_bytes()
    audited["load"]()
    assert audited["audit_path"].read_bytes() == before
    original = pairs.unapply_name_corrections(audited["corrected"], audited["audit"])
    assert {row["lemma"] for row in original} == {"ξελσυς"}


def test_edited_target_without_independent_name_evidence_is_rejected(audited):
    audit = json.loads(audited["audit_path"].read_text())
    audit["corrections"][0]["corrected"]["lemma"] = "Πατάτα"
    audited["audit_path"].write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="invalid original reading"):
        audited["load"]()


def test_new_source_file_invalidates_absence_of_competing_anchors(audited):
    (audited["xml"] / "9999-002.xml").write_text("<treebank/>")
    with pytest.raises(ValueError, match="XML inventory"):
        audited["load"]()


def test_colliding_nominative_still_pins_its_source(tmp_path, monkeypatch):
    xml = tmp_path / "xml"
    xml.mkdir()
    metadata = tmp_path / "metadata.txt"
    metadata.write_text("TLG\tDIALECT\tSOURCE_LICENSE\tTREEBANK_ANNOTATIONS\n"
                        "9999-001\tAttic\tCC BY-SA\t\n"
                        "9999-002\tAttic\tCC BY-SA\t\n")
    anchor = xml / "9999-001.xml"
    anchor.write_text('<treebank><sentence>'
                      '<word form="Κέλσος" lemma="ξελσυς" postag="n-s---mn-"/>'
                      '<word form="Κέλσος" lemma="Κέλσος" postag="n-s---mn-"/>'
                      '</sentence></treebank>')
    oblique = xml / "9999-002.xml"
    oblique.write_text('<treebank><sentence>'
                       '<word form="Κέλσου" lemma="ξελσυς" postag="n-s---mg-"/>'
                       '</sentence></treebank>')
    lexical = tmp_path / "heads.json"
    lexical.write_text(json.dumps({"schema_version": 2,
                                  "heads": {"Κέλσος": ["lgpn_names"]}}))
    names = tmp_path / "names.json"
    names.write_text(json.dumps([{"form": "Κέλσος", "lemma": "Κέλσος", "pos": "name"}]))
    audit = {}
    pairs.extract_glaux(xml, name_evidence=pairs.load_name_evidence(lexical, names),
                        correction_audit=audit, source_commit=COMMIT)
    assert len(audit["corrections"]) == 1
    assert audit["corrections"][0]["source_xml"] == [oblique.name]
    assert set(audit["source_xml"]) == {anchor.name, oblique.name}
    assert audit["corrections"][0]["evidence"]["anchor_rows"][0]["source_xml"] == [anchor.name]
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))
    pairs.load_name_corrections(path, xml, lexical_path=lexical, name_pairs_path=names)
    anchor.unlink()
    with pytest.raises(ValueError, match="XML inventory"):
        pairs.load_name_corrections(path, xml, lexical_path=lexical, name_pairs_path=names)

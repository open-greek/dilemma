"""Production GLAUx readers require the metadata that defines exclusions."""

import copy
import json

import pytest

from build import build_glaux_pairs as pairs


COMMIT = "a" * 40
HEADER = "TLG\tDIALECT\tSOURCE_LICENSE\tTREEBANK_ANNOTATIONS\n"


def word(form, lemma="ξελσυς", tag="n-s---mn-"):
    return f'<word form="{form}" lemma="{lemma}" postag="{tag}"/>'


def write_source(path, automatic, manual=""):
    path.write_text('<treebank><sentence analysis="auto">' + automatic
                    + '</sentence><sentence analysis="manual">' + manual
                    + '</sentence></treebank>')


@pytest.fixture
def corpus(tmp_path):
    xml = tmp_path / "xml"
    xml.mkdir()
    metadata = tmp_path / "metadata.txt"
    metadata.write_text(HEADER
                        + "9999-001\tAttic\tCC BY-NC-SA\t\n"
                        + "9999-002\tAttic\tCC BY-SA\tPROIEL\n"
                        + "9999-003\tAttic\tCC BY-SA\tGorman\n"
                        + "9999-004\tAttic\tCC BY-SA\t\n")
    write_source(xml / "9999-001.xml", word("Νικητής"))
    write_source(xml / "9999-002.xml", word("Νικητής"))
    write_source(xml / "9999-003.xml", word("Κέλσος"), word("Νικητής"))
    write_source(xml / "9999-004.xml", word("Κέλσου", tag="n-s---mg-"))
    lexical = tmp_path / "heads.json"
    lexical.write_text(json.dumps({"schema_version": 2,
                                  "heads": {"Κέλσος": ["lgpn_names", "wiktionary_el_grc"]}}))
    names = tmp_path / "names.json"
    names.write_text(json.dumps([{"form": "Κέλσος", "lemma": "Κέλσος", "pos": "name"}]))
    evidence = pairs.load_name_evidence(lexical, names)
    return {"xml": xml, "metadata": metadata, "lexical": lexical,
            "names": names, "evidence": evidence, "audit_path": tmp_path / "audit.json"}


def build(corpus):
    audit = {}
    result = pairs.extract_glaux(corpus["xml"], name_evidence=corpus["evidence"],
                                 correction_audit=audit, source_commit=COMMIT)
    corpus["audit_path"].write_text(json.dumps(audit))
    return result, audit


def load(corpus, **kwargs):
    return pairs.load_name_corrections(corpus["audit_path"], corpus["xml"],
                                      lexical_path=corpus["lexical"],
                                      name_pairs_path=corpus["names"], **kwargs)


@pytest.mark.parametrize("stats_only", [False, True])
@pytest.mark.parametrize("path_mode", ["default", "explicit", "directory"])
def test_extraction_fails_before_reading_without_metadata(corpus, monkeypatch, stats_only, path_mode):
    corpus["metadata"].unlink()
    kwargs = {}
    if path_mode == "explicit":
        kwargs["metadata_path"] = corpus["metadata"]
    elif path_mode == "directory":
        kwargs["metadata_path"] = corpus["xml"]
    # A missing metadata file must not fall through to the helpers' empty sets.
    monkeypatch.setattr(pairs, "glaux_dialects", lambda _path: pytest.fail("read started"))
    with pytest.raises(FileNotFoundError, match="metadata is required for source exclusions"):
        pairs.extract_glaux(corpus["xml"], stats_only=stats_only, **kwargs)


def test_extraction_requires_metadata_even_for_empty_xml_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="metadata is required"):
        pairs.extract_glaux(tmp_path / "empty")


def test_real_exclusions_are_applied_and_metadata_is_bound(corpus):
    original_evidence = copy.deepcopy(corpus["evidence"])
    result, audit = build(corpus)
    assert [(r["form"], r["lemma"], r["count"]) for r in result] == [
        ("Κέλσος", "Κέλσος", 1), ("Κέλσου", "Κέλσος", 1)]
    assert audit["corpus_xml_inventory"] == ["9999-003.xml", "9999-004.xml"]
    assert audit["inputs"]["glaux_metadata"] == pairs._fingerprint(corpus["metadata"])
    assert corpus["evidence"] == original_evidence
    loaded = load(corpus)
    assert set(loaded["by_source"]) == {"9999-003.xml", "9999-004.xml"}
    assert [r["lemma"] for r in pairs.unapply_name_corrections(result, audit)] == ["ξελσυς"] * 2


@pytest.mark.parametrize("path_mode", ["default", "explicit", "directory"])
def test_loader_requires_current_metadata(corpus, path_mode):
    build(corpus)
    corpus["metadata"].unlink()
    kwargs = {}
    if path_mode == "explicit":
        kwargs["metadata_path"] = corpus["metadata"]
    elif path_mode == "directory":
        kwargs["metadata_path"] = corpus["xml"]
    with pytest.raises(FileNotFoundError, match="metadata is required for source exclusions"):
        load(corpus, **kwargs)


def test_loader_rejects_audit_without_metadata_fingerprint(corpus):
    _, audit = build(corpus)
    del audit["inputs"]["glaux_metadata"]
    corpus["audit_path"].write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="missing its required metadata fingerprint"):
        load(corpus)


def test_loader_rejects_changed_metadata(corpus):
    build(corpus)
    corpus["metadata"].write_text(corpus["metadata"].read_text().replace("Gorman", "PROIEL"))
    with pytest.raises(ValueError, match="stale.*glaux_metadata"):
        load(corpus)


def test_metadata_fingerprint_is_required_even_without_corrections(corpus):
    corpus["evidence"]["names"] = {}
    _, audit = build(corpus)
    assert audit["corrections"] == []
    assert audit["inputs"]["glaux_metadata"] == pairs._fingerprint(corpus["metadata"])
    # Load against the original independent name files, but no corrections remain to apply.
    assert load(corpus)["by_source"] == {}


def test_metadata_cannot_change_mid_extraction(corpus, monkeypatch):
    dialects = pairs.glaux_dialects

    def alter_metadata(path):
        result = dialects(path)
        path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(pairs, "glaux_dialects", alter_metadata)
    audit = {}
    with pytest.raises(ValueError, match="metadata changed during extraction"):
        pairs.extract_glaux(corpus["xml"], name_evidence=corpus["evidence"],
                            correction_audit=audit, source_commit=COMMIT)
    assert audit == {}

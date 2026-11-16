"""The Sophocles OCR lost initial breathings (ο and ρ chunks); restore them
from evidence or withhold the head (#107, #125)."""
import json
from pathlib import Path
import sys

import pytest

from dilemma_build import sophocles_breathing as B

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("word,kind,expected", [
    ("οφφίκιον", "smooth", "ὀφφίκιον"), ("ρῆσις", "rough", "ῥῆσις"),
    ("όλκιμος", "rough", "ὅλκιμος"), ("ουδωνάριον", "smooth", "οὐδωνάριον"),
    ("Ουίκα", "smooth", "Οὐίκα"), ("Αιβυρνίς", "smooth", "Αἰβυρνίς"),
    ("όις", "smooth", "ὄις"), ("υιός", "rough", "υἱός"), ("Ρόδος", "rough", "Ῥόδος"),
])
def test_breathing_sits_on_the_initial_vowel_diphthong_or_rho(word, kind, expected):
    assert B.needs_breathing(word)
    assert B.add_breathing(word, kind) == expected
    assert B.breathing_of(expected) == kind
    assert not B.needs_breathing(expected)


def test_words_that_need_no_breathing():
    for word in ("λόγος", "ἀγαθός", "ῥῆσις", "Ὅμηρος", "οὐδείς"):
        assert not B.needs_breathing(word)


OTHER = {"ὀφφικιάλιος", "ὀφφικίζω", "ἁγνεύω", "ὄρος", "ὅρος", "ὀλόθερμος",
         "ὁλόθερμος", "ὁλοθελῶς", "ὀλοθείω", "ἀφθαρσία"}


def review(*heads):
    return B.review_heads({h: {"sophocles/0800.xml"} for h in heads}, OTHER)


def test_rules_and_their_evidence():
    r = review("οφφίκιον", "αγνεύω", "ρῆσις", "ρθοποδίζω", "όρος", "ολόθεος",
               "οξυγόνος", "άφθαρσία", "ϊκανᾶτος", "λόγος")
    c, w = r["corrected"], r["withheld"]
    assert c["οφφίκιον"]["head"] == "ὀφφίκιον" and c["οφφίκιον"]["rule"] == "word_family"
    assert c["οφφίκιον"]["evidence"] == ["ὀφφικίζω", "ὀφφικιάλιος"]
    assert c["αγνεύω"] == {"head": "ἁγνεύω", "rule": "same_head",
                           "evidence": ["ἁγνεύω"], "sources": ["sophocles/0800.xml"]}
    assert c["ρῆσις"]["head"] == "ῥῆσις" and c["ρῆσις"]["rule"] == "initial_rho"
    assert w["ρθοποδίζω"]["reason"] == "rho_before_consonant"
    assert w["όρος"]["reason"] == "both_breathings_attested"
    assert w["όρος"]["candidates"] == ["ὄρος", "ὅρος"]
    assert w["ολόθεος"]["reason"] == "family_breathings_differ"
    assert w["οξυγόνος"]["reason"] == "no_evidence"
    assert w["άφθαρσία"]["reason"] == "two_accents"
    assert w["ϊκανᾶτος"]["reason"] == "diaeresis_on_initial"
    assert "λόγος" not in c and "λόγος" not in w
    assert r["counts"] == {"corrected": len(c), "withheld": len(w)}
    assert B.correct("οφφίκιον", r) == "ὀφφίκιον"
    assert B.correct("όρος", r) is None
    assert B.correct("λόγος", r) == "λόγος"
    assert B.correct("όρος", None) == "όρος"


def test_review_is_deterministic():
    assert review("οφφίκιον", "όρος") == review("όρος", "οφφίκιον")


def test_lexical_heads_store_breathed_heads_and_the_review(tmp_path):
    from dilemma_build.build_grc_lexical_heads import (
        EDITED_INPUTS, KAIKKI_INPUTS, VERNACULAR_INPUTS, build_heads)
    data, kaikki = tmp_path / "data", tmp_path / "kaikki"
    data.mkdir(); kaikki.mkdir()
    for name in EDITED_INPUTS:
        (data / name).write_text(json.dumps(
            ["ὀφφικιάλιος", "ὄρος", "ὅρος"] if name == "dge_headwords.json" else []))
    for _, filename, _ in KAIKKI_INPUTS + VERNACULAR_INPUTS:
        (kaikki / filename).write_text("")
    lsj = tmp_path / "lsj.json"
    lsj.write_text("[]")
    chunks = data / "sophocles" / "chunks"; chunks.mkdir(parents=True)
    (chunks / "0800.xml").write_text(
        "<root><entry><orth>οφφίκιον</orth>, ου, το</entry>"
        "<entry><orth>όρος</orth></entry></root>")
    (chunks / "0950.xml").write_text("<root><entry><orth>ρῆσις</orth></entry></root>")
    result = build_heads(kaikki, data, lsj)
    heads = result["heads"]
    assert heads["ὀφφίκιον"] == ["sophocles/0800.xml"]
    assert heads["ῥῆσις"] == ["sophocles/0950.xml"]
    assert not {"οφφίκιον", "ρῆσις", "όρος"} & set(heads)
    assert heads["ὄρος"] == heads["ὅρος"] == ["dge_headwords"]
    assert result["sophocles_breathing"]["withheld"]["όρος"]["reason"] == \
        "both_breathings_attested"
    assert "build/sophocles_breathing.py" in result["source_sha256"]


def test_sophocles_expansion_reads_the_review(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "build"))
    import expand_sophocles as S
    (tmp_path / "0800.xml").write_text(
        "<root><entry><orth>οφφίκιον</orth>, ου, τό, office</entry>"
        "<entry><orth>όρος</orth>, ου, ὁ, boundary</entry></root>")
    monkeypatch.setattr(S, "SOPH_DIR", tmp_path)
    rev = review("οφφίκιον", "όρος")
    entries = S.parse_sophocles_entries(rev)
    assert set(entries) == {"ὀφφίκιον"}
    assert set(S.parse_sophocles_entries(None)) == {"οφφίκιον", "όρος"}
    heads = tmp_path / "heads.json"
    heads.write_text(json.dumps({"heads": {}}))
    with pytest.raises(SystemExit, match="sophocles_breathing"):
        S.load_breathing_review(heads, required=True)
    heads.write_text(json.dumps({"heads": {}, "sophocles_breathing": rev}))
    assert S.load_breathing_review(heads, required=True) == rev


def test_lexical_heads_settle_the_ocr_letters_before_the_review(tmp_path):
    """The micro sign is mu and a reviewed correction replaces the head, in
    the lexical heads as in the expansion (build/sophocles_ocr.py)."""
    from dilemma_build.build_grc_lexical_heads import (
        EDITED_INPUTS, KAIKKI_INPUTS, VERNACULAR_INPUTS, build_heads)
    micro = "µ"
    data, kaikki = tmp_path / "data", tmp_path / "kaikki"
    data.mkdir(); kaikki.mkdir()
    for name in EDITED_INPUTS:
        (data / name).write_text(json.dumps(
            ["ὁμοούσιος", "ὀρνίθειος"] if name == "dge_headwords.json" else []))
    for _, filename, _ in KAIKKI_INPUTS + VERNACULAR_INPUTS:
        (kaikki / filename).write_text("")
    lsj = tmp_path / "lsj.json"
    lsj.write_text("[]")
    chunks = data / "sophocles" / "chunks"; chunks.mkdir(parents=True)
    (chunks / "0800.xml").write_text(
        f"<root><entry><orth>ο{micro}οούσιος</orth>, ον</entry>"
        "<entry><orth>ορνιθείον</orth>, ου, το, hen-coop</entry>"
        f"<entry><orth>{micro}έρος</orth>, ους, τὸ</entry>"
        "<entry><orth>᾽Αβραμιαῖος</orth>, α, ον</entry></root>")
    plain = build_heads(kaikki, data, lsj)
    assert "sophocles_head_corrections" not in plain["inputs"]
    assert "ὀρνιθεῖον" not in plain["heads"]
    (data / "sophocles_head_corrections.json").write_text(json.dumps({
        "schema_version": 1, "corrections": {"ορνιθείον": {
            "head": "ὀρνιθεῖον", "source": "sophocles/0800.xml",
            "reason": "accent"}}}, ensure_ascii=False), encoding="utf-8")
    result = build_heads(kaikki, data, lsj)
    heads = result["heads"]
    assert heads["ὁμοούσιος"] == ["dge_headwords", "sophocles/0800.xml"]
    assert heads["μέρος"] == heads["ὀρνιθεῖον"] == ["sophocles/0800.xml"]
    assert heads["Ἀβραμιαῖος"] == ["sophocles/0800.xml"]
    assert not any(micro in head for head in heads)
    review = result["sophocles_breathing"]
    assert review["corrected"]["ομοούσιος"]["head"] == "ὁμοούσιος"
    assert "ορνιθείον" not in review["corrected"]
    assert "ορνιθείον" not in review["withheld"]
    assert len(result["inputs"]["sophocles_head_corrections"]["sha256"]) == 64
    assert "build/sophocles_ocr.py" in result["source_sha256"]


def test_sophocles_expansion_refuses_heads_built_with_other_corrections(tmp_path):
    sys.path.insert(0, str(ROOT / "build"))
    import expand_sophocles as S
    import hashlib
    rev = review("οφφίκιον")
    heads = tmp_path / "heads.json"
    current = hashlib.sha256(S.HEAD_CORRECTIONS.read_bytes()).hexdigest()
    heads.write_text(json.dumps({"heads": {}, "sophocles_breathing": rev, "inputs": {
        "sophocles_head_corrections": {"sha256": current, "size": 1}}}))
    assert S.load_breathing_review(heads, required=True) == rev
    for inputs in ({}, {"sophocles_head_corrections": {"sha256": "0" * 64, "size": 1}}):
        heads.write_text(json.dumps(
            {"heads": {}, "sophocles_breathing": rev, "inputs": inputs}))
        with pytest.raises(SystemExit, match="head corrections"):
            S.load_breathing_review(heads, required=True)

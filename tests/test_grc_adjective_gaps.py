"""Ancient -ικός adjectives declined as masculine nouns regain their feminines."""
import json
from pathlib import Path
import sys

import pytest

from dilemma.grc_adjective_gaps import complete_ikos_adjectives, ikos_cells

ROOT = Path(__file__).resolve().parent.parent


def test_ikos_cells_decline_the_feminine_and_neuter_plural():
    forms = [form for form, _ in ikos_cells("μοναχικός")]
    assert forms == ["μοναχική", "μοναχικῆς", "μοναχικῇ", "μοναχικήν", "μοναχικαί",
                     "μοναχικῶν", "μοναχικαῖς", "μοναχικάς", "μοναχικά", "μοναχικαῖν"]
    with pytest.raises(ValueError):
        ikos_cells("οἶκος")


def test_completion_fills_absent_keys_only_and_needs_a_head():
    lookup = {"μοναχικός": "μοναχικός", "μοναχικῶν": "μοναχικός",
              "μοναχικά": "μοναχικά",          # an existing reading stays
              "ἀγγελτικοῦ": "ἀγγελτικός"}      # no self-mapped head
    added = complete_ikos_adjectives(lookup, ["μοναχικός", "ἀγγελτικός"])
    assert lookup["μοναχικήν"] == lookup["μοναχικην"] == "μοναχικός"
    assert lookup["μοναχικά"] == "μοναχικά"
    assert lookup["μοναχικα"] == "μοναχικός"
    assert "ἀγγελτική" not in lookup
    assert {row["lemma"] for row in added} == {"μοναχικός"}
    assert "μοναχικῶν" not in {row["form"] for row in added}


def test_capitalized_adjective_is_keyed_like_the_capitalized_heads():
    lookup = {"Πλατωνικός": "Πλατωνικός"}
    complete_ikos_adjectives(lookup, ["Πλατωνικός"])
    assert (lookup["Πλατωνικήν"] == lookup["πλατωνικήν"] == lookup["πλατωνικην"]
            == "Πλατωνικός")
    assert "Πλατωνικην" not in lookup


def test_sophocles_parser_reads_ocr_adjective_endings():
    sys.path.insert(0, str(ROOT / "build"))
    import expand_sophocles as S
    assert S.adjective_endings(", ή, όν, announcing") == "three"
    assert S.adjective_endings(", ἦ, ov, (ἀδικέω)") == "three"
    assert S.adjective_endings(", ή, 6v, (ἄβυσσος)") == "three"
    assert S.adjective_endings(", ov, unjust") == "two"
    assert S.adjective_endings(", ἡ, the deep") is None
    assert S.adjective_endings(", οῦ, ὁ, a monk") is None


def test_finder_lists_supported_ikos_adjectives(tmp_path):
    sys.path.insert(0, str(ROOT / "build"))
    import find_grc_ikos_adjectives as F
    data = tmp_path
    (data / "ag_lookup.json").write_text(json.dumps({
        "μοναχικός": "μοναχικός", "μοναχικοῦ": "μοναχικός",       # Sophocles: three endings
        "ἀγγελτικός": "ἀγγελτικός",                               # corpus: adjective
        "βολιστικός": "βολιστικός", "βολιστική": "βολιστικός",    # already has a feminine
        "δαρεικός": "δαρεικός",                                   # a noun: no evidence
        "*σχολαστικός": "*σχολαστικός",                           # OCR debris
        "ἄδικος": "ἄδικος",                                       # not -ικός oxytone
        "ἀνικός": "ἀνικός",                                       # Sophocles: two endings
        "φρουρικός": "φρουρικός",                                 # review: adjective
        "Πλατωνικός": "Πλατωνικός",                               # review: adjective
        "Γρανικός": "Γρανικός",                                   # capitalized, unreviewed
        "παθικός": "παθικός",                                     # review: exclude
    }, ensure_ascii=False))
    (data / "lsj9_pos_lookup.json").write_text("{}")
    (data / "lemma_attestation.json").write_text(json.dumps({"lemmas": {
        "ἀγγελτικός": {"dominant_pos": "adj"}, "ἀνικός": {"dominant_pos": "adj"},
        "παθικός": {"dominant_pos": "adj"}, "Γρανικός": {"dominant_pos": "adj"},
        "δαρεικός": {"dominant_pos": "noun"}}}, ensure_ascii=False))
    chunks = data / "sophocles" / "chunks"
    chunks.mkdir(parents=True)
    (chunks / "0001.xml").write_text(
        "<root><entry><orth>μοναχικός</orth>, ή, όν, monastic.</entry>"
        "<entry><orth>ἀνικός</orth>, όν, x.</entry></root>", encoding="utf-8")
    (data / "grc_ikos_review.json").write_text(json.dumps({"entries": [
        {"lemma": "φρουρικός", "decision": "adjective", "reason": "x"},
        {"lemma": "Πλατωνικός", "decision": "adjective", "reason": "x"},
        {"lemma": "παθικός", "decision": "exclude", "reason": "x"},
    ]}, ensure_ascii=False))
    record = F.find(data)
    assert sorted(record["lemmas"], key=lambda r: r["lemma"]) == sorted([
        {"lemma": "μοναχικός", "evidence": ["sophocles"]},
        {"lemma": "ἀγγελτικός", "evidence": ["corpus"]},
        {"lemma": "φρουρικός", "evidence": ["review"]},
        {"lemma": "Πλατωνικός", "evidence": ["review"]},
    ], key=lambda r: r["lemma"])
    assert len(record["inputs"]) == 5


def test_every_review_entry_has_a_decision_and_a_reason():
    review = json.loads((ROOT / "data" / "grc_ikos_review.json").read_text(encoding="utf-8"))
    lemmas = [entry["lemma"] for entry in review["entries"]]
    assert len(lemmas) == len(set(lemmas))
    for entry in review["entries"]:
        assert entry["decision"] in ("adjective", "exclude") and entry["reason"]
    listed = {row["lemma"]: row["evidence"] for row in json.loads(
        (ROOT / "data" / "grc_ikos_adjectives.json").read_text(encoding="utf-8"))["lemmas"]}
    for entry in review["entries"]:
        assert (entry["lemma"] in listed) == (entry["decision"] == "adjective")

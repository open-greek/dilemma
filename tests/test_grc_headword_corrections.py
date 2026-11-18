"""Reviewed declensions and self-maps for heads the exports leave formless."""
import json
from pathlib import Path
import sys
import unicodedata

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build"))
import build_lookup_db as B  # noqa: E402
import expand_lsj as m  # noqa: E402

RECORD = json.loads((ROOT / "data" / "grc_headword_corrections.json")
                    .read_text(encoding="utf-8"))


def entry(head, genitive):
    return {"headword": head, "orth_orig": head, "article": "ὁ", "gender": "m",
            "itype": genitive, "genitive": genitive}


def test_record_is_normalized_and_every_entry_states_its_evidence():
    for record in RECORD["noun_declensions"] + RECORD["headword_self_maps"]:
        assert record["evidence"]
        assert unicodedata.is_normalized("NFC", record["headword"])


@pytest.mark.parametrize("head,genitive", [("ὅμηλυς", "ὁμήλῠδος"),
                                           ("κάτηλυς", "κατήλῠδος")])
def test_corrected_heads_resolve_their_abbreviated_genitive(head, genitive):
    record = next(r for r in RECORD["noun_declensions"]
                  if r["headword"] == head)
    resolved, error = m.resolve_noun_genitive(head, record["gender"],
                                              record["genitive"])
    assert not error
    # The prefix keeps the nominative accent; the module moves it.
    assert m.strip_diacritics(resolved) == m.strip_diacritics(genitive)
    assert resolved.endswith("ῠδος")


def test_misread_field_is_withheld_without_the_correction():
    assert m.resolve_noun_genitive("κάτηλυς", "m", "ὕδος")[1] == (
        "unsupported-genitive-abbreviation")


def test_missing_head_is_added_after_every_other_noun():
    entries = {"κατήλυσις": entry("κατήλυσις", "εως")}
    assert m.apply_noun_corrections(entries, RECORD["noun_declensions"]) == 1
    assert list(entries) == ["κατήλυσις", "ὅμηλυς"]
    assert entries["ὅμηλυς"]["gender"] == "m"
    assert entries["ὅμηλυς"]["genitive"] == "ῠδος"


def test_misread_genitive_is_replaced_and_moved_last():
    entries = {"κάτηλυς": entry("κάτηλυς", "ὕδος"),
               "κατήλυσις": entry("κατήλυσις", "εως")}
    m.apply_noun_corrections(entries, RECORD["noun_declensions"])
    assert list(entries) == ["κατήλυσις", "ὅμηλυς", "κάτηλυς"]
    assert entries["κάτηλυς"]["genitive"] == "ῠδος"


@pytest.mark.parametrize("head,genitive", [("ὅμηλυς", "εος"),
                                           ("κάτηλυς", "ῠδος")])
def test_a_head_the_export_already_gives_stays_as_it_is(head, genitive):
    entries = {head: entry(head, genitive)}
    before = dict(entries[head])
    m.apply_noun_corrections(entries, [r for r in RECORD["noun_declensions"]
                                       if r["headword"] == head])
    assert entries[head] == before


def test_self_maps_must_stand_in_the_named_list(tmp_path):
    path = tmp_path / "grc_headword_corrections.json"
    path.write_text(json.dumps({"headword_self_maps": [
        {"headword": "γοσταί", "list": "dge_headwords.json", "evidence": "x"}]},
        ensure_ascii=False), encoding="utf-8")
    listing = tmp_path / "dge_headwords.json"
    listing.write_text(json.dumps(["γοσταί"], ensure_ascii=False), encoding="utf-8")
    assert B.reviewed_headword_self_maps(path) == ["γοσταί"]
    listing.write_text(json.dumps(["ἀγοστή"], ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="not in dge_headwords.json"):
        B.reviewed_headword_self_maps(path)


def test_shipped_self_map_is_a_dge_head_and_has_a_registered_source():
    assert B.reviewed_headword_self_maps(
        ROOT / "data" / "grc_headword_corrections.json") == ["γοσταί"]
    _, by_input = B.load_lookup_sources(ROOT / "data" / "lookup_sources.json")
    assert by_input["grc_headword_corrections.json"] == "grc_headword_corrections"

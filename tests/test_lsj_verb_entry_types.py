"""A dictionary cross-reference or prefix does not license a verb paradigm."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
import expand_lsj


@pytest.mark.parametrize("word,text,reason", [
    ("ἡμι", "insep. Prefix, used in compos., half-.", "prefix-headword"),
    ("ἤτω", "late form for ἔστω, 3 sg. imper. of εἰμί (sum), IG3.3509.",
     "inflected-imperative-headword"),
    ("ἴστω", "3sg. imper. of οἶδα; cf. ἴττω.", "inflected-imperative-headword"),
    ("ἴτω", "[ῐ], 3sg. sg. imper. from εἶμι (ibo); in Trag. almost an exclam.",
     "inflected-imperative-headword"),
    ("κεκρίσθω", ", 3 sg. imperat. pf. Pass. of κρύπτω.",
     "inflected-imperative-headword"),
    ("ἠμί", "I say, the 1st pers. of pres. being used in Att. dialogue in "
     "emphatic repetitions; otherwise only in 3 sg. ἠσί Hermipp.1.",
     "person-limited-verb-entry"),
    ("ἡμί", "*I say*, the 1st pers. being used in Att. dialogue in emphatic "
     "repetitions; *otherwise only in 3 sg.* ἠσί Hermipp.1. II. impf. ἦν, "
     "3 sg. ἦ (*the only part used by Hom.*).",
     "person-limited-verb-entry"),
])
def test_nonparadigmatic_entries_never_call_lua(word, text, reason):
    assert expand_lsj.expand_verb(None, word, head_text=text) == (set(), reason)


@pytest.mark.parametrize("text", [
    "3sg. ἄησι Hes.Op.516; imper. 3sg. ἀήτω A.R.4.768; inf. ἀῆναι.",
    "3 sg. imper. στερείτω Pl.Lg.958e; otherwise pres. occurs only in form στερίσκω.",
    "late form of ἀποκτείνω, Plb.1.69.11.",
    "3 sg. χοῖ, 3 pl. χοῦσι; inf. χοῦν; part. χῶν.",
    "I say, 1st pers. pres.; 3 sg. ἠσί; a complete paradigm is supplied.",
    "describe or represent in the form of man, Ph.1.282.",
    "", 
])
def test_actual_verbs_and_unknown_descriptions_are_not_rejected(text):
    assert not expand_lsj.verb_entry_rejection(text)


def test_driver_uses_original_quantity_marked_headword(monkeypatch, tmp_path):
    entries = {"ἡμι": {"orth_orig": "ἡμῐ", "gender": ""},
               "λύω": {"orth_orig": "λύω", "gender": ""}}
    monkeypatch.setattr(expand_lsj, "parse_lsj_entries", lambda: entries)
    monkeypatch.setattr(expand_lsj, "load_wiktionary_forms", lambda: set())
    monkeypatch.setattr(expand_lsj, "load_lsj_head_texts", lambda: {
        "ἡμῐ": "insep. Prefix, used in compos., half-.", "λύω": "loose."})
    monkeypatch.setattr(expand_lsj, "get_wtp", lambda: object())
    lookup = tmp_path / "ag_lookup.json"
    lookup.write_text(json.dumps({"ἡμι": "independent_source"}))
    monkeypatch.setattr(expand_lsj, "AG_LOOKUP", lookup)
    calls = []

    def expand(_wtp, word, head_text=""):
        calls.append(word)
        return {"λύω", "λύεις"}, ""

    monkeypatch.setattr(expand_lsj, "expand_verb", expand)
    expand_lsj.expand_verbs()
    result = json.loads(lookup.read_text())
    assert calls == ["λύω"]
    assert result["ἡμι"] == "independent_source"
    assert result["λύεις"] == "λύω"
    assert "ῆς" not in result

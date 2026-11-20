"""The LSJ9-derived runtime tables carry no stem-divided head (#149).

LSJ9 prints a head divided at its stem (ψῡχ-ή, ἀασι-φρονία) and with
vowel-length marks. data/lsj9_frequency.json and data/lsj9_pos_lookup.json
used to copy those keys, which match no lemma or token.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from dilemma.citation import HEAD_HYPHENS, hyphenated_lemma

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = _load("build_lsj9_frequency", "build/build_lsj9_frequency.py")
P = _load("build_lsj9_pos_lookup", "build/build_lsj9_pos_lookup.py")


# --- lsj9_frequency.json -------------------------------------------------------

def test_frequency_keys_take_the_citation_spelling():
    assert F.citation_key("ψῡχ-ή") == "ψυχή"
    assert F.citation_key("πᾰρά") == "παρά"
    assert F.citation_key("κατα-λύω") == "καταλύω"
    assert F.citation_key("ἀ-") == "ἀ-"          # a prefix entry keeps its hyphen
    assert F.citation_key("λόγος") == "λόγος"


def test_frequency_entries_that_join_are_added():
    table, changes = F.build({"τροχός": 25, "τροχ-ός": 1, "ψῡχ-ή": 107,
                              "λόγος": 245})
    assert table == {"τροχός": 26, "ψυχή": 107, "λόγος": 245}
    assert changes == {"τροχός": [["τροχός", 25], ["τροχ-ός", 1]],
                       "ψυχή": [["ψῡχ-ή", 107]]}


def test_frequency_script_writes_checks_and_records(tmp_path, capsys):
    export_dir = tmp_path / "lsj9"
    export_dir.mkdir()
    (export_dir / "lsj9_frequency.json").write_text(json.dumps(
        {"λόγος": 3, "ψῡχ-ή": 2, "ψυχή": 1}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    out, record = tmp_path / "freq.json", tmp_path / "record.json"
    args = ["--lsj9-dir", str(export_dir), "--output", str(out),
            "--record", str(record)]
    assert F.main([*args, "--check"]) == 1
    assert F.main(args) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {"λόγος": 3, "ψυχή": 3}
    entry = json.loads(record.read_text(encoding="utf-8"))
    assert entry["changed_keys"] == 1 and entry["merged_keys"] == 1
    assert entry["merges"] == {"ψυχή": [["ψῡχ-ή", 2], ["ψυχή", 1]]}
    assert F.main([*args, "--check"]) == 0
    capsys.readouterr()


def test_tracked_frequency_table_has_citation_keys():
    path = ROOT / "data" / "lsj9_frequency.json"
    record = ROOT / "data" / "changes" / "lsj9_frequency_keys.json"
    table = json.loads(path.read_text(encoding="utf-8"))
    assert not [key for key in table if hyphenated_lemma(key)]
    assert not [key for key in table if key != F.citation_key(key)]
    entry = json.loads(record.read_text(encoding="utf-8"))
    assert entry["result"]["entries"] == len(table) == 55_466
    assert entry["source"]["entries"] == 55_495
    assert table["ψυχή"] == 107 and table["παρά"] == 216


# --- lsj9_pos_lookup.json ------------------------------------------------------

def test_divided_heads_are_matched_joined():
    joined = P.join_headword_pos({"ἀασι-φρονία": "NOUN", "λόγος": "NOUN",
                                  "καλ-ός": "ADJ", "καλός": "NOUN"})
    # the undivided key's POS stands where both spellings are keyed
    assert joined == {"λόγος": "NOUN", "καλός": "NOUN", "ἀασιφρονία": "NOUN"}


def test_hyphenated_rows_are_not_read():
    table = P.build(
        {"ἀασιφρονίαι": "ἀασιφρονία",          # joined lemma: kept
         "ἀασι-φρονίαι": "ἀασι-φρονία",        # divided key and lemma: dropped
         "λιβε": "βορρό-λιψ",                  # divided lemma: dropped
         "λόγου": "λόγος"},
        {"ἀασι-φρονία": "NOUN", "βορρό-λιψ": "NOUN", "λόγος": "NOUN"})
    assert table == {
        "ἀασιφρονίαι": {"NOUN": "ἀασιφρονία"},
        "αασιφρονιαι": {"NOUN": "ἀασιφρονία"},
        "λόγου": {"NOUN": "λόγος"},
        "λογου": {"NOUN": "λόγος"},
    }


def test_tracked_pos_table_has_no_hyphenated_cell():
    """Holds from the central rebuild on (build/build_lsj9_pos_lookup.py)."""
    path = ROOT / "data" / "lsj9_pos_lookup.json"
    if not path.exists():
        pytest.skip("lsj9_pos_lookup.json not present")
    table = json.loads(path.read_text(encoding="utf-8"))
    assert not [key for key in table if any(c in HEAD_HYPHENS for c in key)]
    assert not [lemma for cells in table.values() for lemma in cells.values()
                if hyphenated_lemma(lemma)]

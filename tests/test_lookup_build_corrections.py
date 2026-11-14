"""Lookup build corrections: expansion artefacts (#103), reviewed non-lemmas
and capitalised corpus twins (#108), and reviewed lookup overrides (#110).

The builder runs on a small fixture in a temporary data directory; no
production data is read or written.
"""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_lookup_db as B  # noqa: E402


# Pure helper ---------------------------------------------------------------

HEADS = frozenset({"τε", "σύνοχος", "μός", "κλεῖδα", "μόνος", "μονός"})
BY_LETTERS = {}
for _head in HEADS:
    BY_LETTERS.setdefault(B._bare_letters(_head), set()).add(_head)


@pytest.mark.parametrize("lemma,verdict", [
    ("λαβλυμι", ("unaccented_expansion_lemma_unlisted", None)),
    ("αελιυς", ("unaccented_expansion_lemma_unlisted", None)),
    ("σερvιλια", ("unaccented_expansion_lemma_unlisted", None)),
    ("προ-ανακηρυσσω", ("unaccented_expansion_lemma_unlisted", None)),
    ("λαβ’μι", ("unaccented_expansion_lemma_unlisted", None)),
    ("συνοχος", ("unaccented_expansion_lemma_repaired", "σύνοχος")),
    ("μος", ("unaccented_expansion_lemma_repaired", "μός")),
    # Two heads with these letters: no repair.
    ("μονος", ("unaccented_expansion_lemma_unlisted", None)),
    ("τε", None),
    ("λαβή", None),
    ("Ῥώμη", None),
])
def test_unaccented_expansion_lemma(lemma, verdict):
    assert B.unaccented_expansion_lemma(lemma, HEADS, BY_LETTERS) == verdict


def test_expansion_heads_index_keeps_case_and_ancient_heads_only(tmp_path):
    path = tmp_path / "heads.json"
    path.write_text(json.dumps({"heads": ["Βάβυρσα", "σύνοχος"],
                                "vernacular_heads": ["μπαλκόνι"]},
                               ensure_ascii=False), encoding="utf-8")
    heads, by_letters = B.load_expansion_lemma_heads(path)
    assert heads == {"Βάβυρσα", "σύνοχος", "μπαλκόνι"}
    assert by_letters == {"Βαβυρσα": {"Βάβυρσα"}, "συνοχος": {"σύνοχος"}}


# Builder on a fixture ------------------------------------------------------

SCRIPT = """
from functools import partial
from pathlib import Path
import sys
import build_lookup_db as builder

data = Path(sys.argv[1])
original_data = builder.DATA_DIR
def relocate(value):
    if isinstance(value, Path) and value.parent == original_data:
        return data / value.name
    if isinstance(value, (tuple, list)):
        return type(value)(relocate(item) for item in value)
    return value
for name, value in list(vars(builder).items()):
    setattr(builder, name, relocate(value))
builder.DATA_DIR = data
builder._write_citation_rejection_report = partial(
    builder._write_citation_rejection_report,
    path=builder.CITATION_REJECTIONS_PATH,
)
builder.build()
"""

# Readings of the dictionary expansion: keys of the reference that the
# Wiktionary base lacks.
EXPANSION = {
    "λαβῃ": "λαβλυμι",          # OCR garble, in no head list: dropped
    "Αἴλιοι": "αελιυς",          # Latin read as Beta Code: dropped
    "σύνοχα": "συνοχος",         # unaccented doublet: repaired
    "γε": "γε",                  # a listed unaccented head: kept
    "μοναχικοῦ": "μοναχικός",    # accented: kept
}
WIKTIONARY = {"λόγος": "λόγος"}


def _fixture(tmp_path, extra=None):
    data = tmp_path / "data"
    data.mkdir()
    ag = {**WIKTIONARY, **EXPANSION, **(extra or {}).pop("ag", {})}
    sources = {
        "ag_lookup.json": ag,
        "mg_lookup.json": {"λόγος": "λόγος"},
        "med_lookup.json": {},
        "grc_lexical_heads.json": {"heads": ["σύνοχος", "γε", "λόγος",
                                             "μοναχικός", "λαβή"],
                                   "vernacular_heads": []},
        **(extra or {}),
    }
    for name, value in sources.items():
        (data / name).write_text(json.dumps(value, ensure_ascii=False),
                                 encoding="utf-8")
    (data / "lookup_sources.json").write_bytes(
        (ROOT / "data" / "lookup_sources.json").read_bytes())
    base = data / "ag_lookup_wikt.json"
    base.write_text(json.dumps(WIKTIONARY, ensure_ascii=False), encoding="utf-8")
    reference = tmp_path / "expansion_reference.json"
    reference.write_text(json.dumps({**WIKTIONARY, **EXPANSION},
                                    ensure_ascii=False), encoding="utf-8")
    pin = lambda path: {"size": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (data / "ag_expansion_reference.json").write_text(json.dumps({
        "schema_version": 1, "reference": pin(reference),
        "reference_base": pin(base)}))
    (data / "changes").mkdir()
    for name in ("el_present_completion.json", "el_imperfect_completion.json"):
        (data / "changes" / name).write_text(json.dumps({"lookup_additions": []}))
    env = {**os.environ, "DILEMMA_AG_EXPANSION_REFERENCE": str(reference),
           "PYTHONHASHSEED": "0"}
    return data, env


def _build(data, env):
    result = subprocess.run([sys.executable, "-c", SCRIPT, str(data)], cwd=ROOT,
                            env=env, capture_output=True, text=True, timeout=60)
    return result


def _rows(data, lang="all"):
    with sqlite3.connect(data / "lookup.db") as conn:
        return dict(conn.execute(
            "SELECT form, lemmas.text FROM lookup JOIN lemmas "
            "ON lookup.lemma_id = lemmas.id WHERE lang = ?", (lang,)))


def _rejections(data):
    lines = (data / "citation_hygiene_rejections.tsv").read_text(
        encoding="utf-8").splitlines()[1:]
    return {tuple(line.split("\t")) for line in lines}


def test_unaccented_expansion_lemmas_are_dropped_or_repaired(tmp_path):
    data, env = _fixture(tmp_path)
    result = _build(data, env)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = _rows(data)
    assert "λαβῃ" not in rows and "Αἴλιοι" not in rows
    assert rows["σύνοχα"] == "σύνοχος"
    assert rows["γε"] == "γε" and rows["μοναχικοῦ"] == "μοναχικός"
    assert "λαβλυμι" not in set(rows.values()) | set(_rows(data, "grc").values())
    rejected = _rejections(data)
    layer = B.AG_EXPANSION_LAYER
    assert ("AG", layer, "unaccented_expansion_lemma_unlisted", "λαβῃ",
            "λαβλυμι") in rejected
    assert ("AG", layer, "unaccented_expansion_lemma_repaired_to_σύνοχος",
            "σύνοχα", "συνοχος") in rejected


def test_missing_lexical_heads_stop_a_build_with_expansion_readings(tmp_path):
    data, env = _fixture(tmp_path)
    (data / "grc_lexical_heads.json").unlink()
    result = _build(data, env)
    assert result.returncode != 0
    assert "grc_lexical_heads.json is missing" in result.stderr

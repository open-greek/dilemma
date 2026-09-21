"""Headword-only import guards for the Triantafyllidis MDC dataset."""

from __future__ import annotations

import importlib.util
import json
import tarfile
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

import dilemma.core as core


ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "build_triantafyllidis_headwords",
    ROOT / "build" / "build_triantafyllidis_headwords.py",
)
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


def test_runtime_unions_dictionary_and_wiktionary_headwords(
        monkeypatch, tmp_path):
    dictionary = tmp_path / "dictionary.json"
    wiktionary = tmp_path / "wiktionary.json"
    dictionary.write_text('["λεξικό"]\n', encoding="utf-8")
    wiktionary.write_text('["βικιλεξικό"]\n', encoding="utf-8")
    monkeypatch.setitem(
        core._CONVENTION_HEADWORDS,
        "triantafyllidis",
        (dictionary, wiktionary),
    )

    assert core._load_convention_headwords("triantafyllidis") == {
        "λεξικό", "βικιλεξικό",
    }


def test_dictionary_headword_does_not_make_dialectal_ke_canonical():
    dilemma = core.Dilemma.__new__(core.Dilemma)
    remap = dilemma._build_convention_map("triantafyllidis")
    assert {remap[word] for word in ("ἄν", "ἐάν", "κε", "κέν")} == {"αν"}


@pytest.mark.parametrize(("raw", "expected", "reason"), [
    (" άλφα ", "άλφα", None),
    ("λέξη 2", "λέξη", None),
    ("λέξιος -α -ο", "λέξιος", None),
    ("λέξιος 2 -α / -ο", "λέξιος", None),
    ("α\N{COMBINING ACUTE ACCENT}", "ά", None),
    ("αντι-ήρωας", "αντι-ήρωας", None),
    ("α- 1", None, "bound_form"),
    ("δύο λέξεις", None, "ambiguous_multiword"),
    ("λέξιος /", None, "ambiguous_multiword"),
    ("word", None, "non_greek_residue"),
    ("λέξη!", None, "non_greek_residue"),
    (None, None, "missing_or_nonstring"),
])
def test_clean_headword_is_conservative(raw, expected, reason):
    drops = Counter()
    repairs = Counter()
    got = builder.clean_headword(raw, drops, repairs, set())
    assert got == expected
    if got is not None:
        assert unicodedata.is_normalized("NFC", got)
        assert not drops
    else:
        assert drops[reason] == 1


def test_latin_homoglyph_repair_requires_independent_target():
    drops = Counter()
    repairs = Counter()
    assert builder.clean_headword(
        "Aθήνα", drops, repairs, {"Αθήνα"}
    ) == "Αθήνα"
    assert repairs == {"validated_latin_homoglyph": 1}

    drops = Counter()
    repairs = Counter()
    assert builder.clean_headword("Aθήνα", drops, repairs, set()) is None
    assert drops == {"non_greek_residue": 1}


def test_build_deduplicates_and_reports_every_rejection(monkeypatch, tmp_path):
    source = tmp_path / "source.parquet"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(
        builder,
        "load_lemmas",
        lambda _source: ["λέξη", "λε\N{COMBINING ACUTE ACCENT}ξη",
                         "λέξη 2", "λέξιος -α -ο", "α- 1", "δύο λέξεις"],
    )

    headwords, stats, rejections = builder.build(
        source, homoglyph_targets=set()
    )

    assert headwords == ["λέξη", "λέξιος"]
    assert stats["source_rows"] == 6
    assert stats["clean_headwords"] == 2
    assert stats["duplicate_clean_headwords"] == 2
    assert stats["repaired_by_reason"] == {
        "display_index_removed": 2,
        "inflection_tail_removed": 1,
    }
    assert stats["dropped_by_reason"] == {
        "ambiguous_multiword": 1,
        "bound_form": 1,
    }
    assert rejections == [
        ("α- 1", "bound_form"),
        ("δύο λέξεις", "ambiguous_multiword"),
    ]


def test_parquet_projection_never_copies_entry_content(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    sentinel = "PROTECTED_DICTIONARY_ENTRY_SENTINEL"
    parquet = tmp_path / "dictionary.parquet"
    pq.write_table(
        pa.table({
            "lemma": ["λέξη", "λέξη 2", "α- 1"],
            "entry_text": [sentinel, sentinel, sentinel],
            "pronunciation": [sentinel, sentinel, sentinel],
            "page_no": [1, 2, 3],
            "source_url": [sentinel, sentinel, sentinel],
        }),
        parquet,
    )
    archive = tmp_path / "dictionary.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(parquet, arcname="nested/dictionary.parquet")

    headwords, stats, _ = builder.build(
        archive, homoglyph_targets=set()
    )
    output = tmp_path / "headwords.json"
    validation = tmp_path / "validation.json"
    validation.write_text("[]\n", encoding="utf-8")
    builder.write_artifacts(
        archive, output, headwords, stats, validation_path=validation
    )

    assert headwords == ["λέξη"]
    assert sentinel not in output.read_text(encoding="utf-8")
    assert sentinel not in builder.meta_path_for(output).read_text(
        encoding="utf-8"
    )
    meta = json.loads(
        builder.meta_path_for(output).read_text(encoding="utf-8")
    )
    assert meta["fields_read"] == ["lemma"]


def test_archive_requires_exactly_one_parquet_member(tmp_path):
    archive = tmp_path / "ambiguous.tar.gz"
    one = tmp_path / "one.parquet"
    two = tmp_path / "two.parquet"
    one.write_bytes(b"one")
    two.write_bytes(b"two")
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(one, arcname="one.parquet")
        tf.add(two, arcname="two.parquet")

    with pytest.raises(ValueError, match="exactly one Parquet"):
        builder.load_lemmas(archive)

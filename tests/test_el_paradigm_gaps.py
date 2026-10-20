import json
import sqlite3
from pathlib import Path

import pytest

from dilemma.el_paradigm_gaps import complete_lookup, paradigm_cells
from scripts.complete_el_paradigms import refresh, sha256, unapply_refresh


def test_reviewed_paradigms_fill_gaps_and_keep_existing_readings():
    lookup = {head: head for head in ("αιτών", "εμπίπτω", "απορρέω", "συνιστώ")}
    lookup["αιτούντα"] = "separate existing reading"
    changes = complete_lookup(lookup)
    for form, head in [("αιτούντος", "αιτών"), ("αιτούντες", "αιτών"),
                       ("εμπίπτουν", "εμπίπτω"), ("συνιστούν", "συνιστώ"),
                       ("απορρέουν", "απορρέω")]:
        assert lookup[form] == head
    assert lookup["αιτούντα"] == "separate existing reading"
    assert changes and all(c["evidence"] for c in changes)
    assert complete_lookup(lookup) == []


def test_completion_requires_each_existing_modern_head():
    lookup = {"αιτών": "αιτώ", "άγνωστος": "άγνωστος"}
    assert complete_lookup(lookup) == []
    assert "αιτούντος" not in lookup
    assert all("past" not in c["tags"] and "passive" not in c["tags"] for c in paradigm_cells())


def base_snapshot(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    heads = {head: head for head in ("αιτών", "εμπίπτω", "απορρέω", "συνιστώ")}
    (base / "mg_lookup.json").write_text(json.dumps(heads))
    with sqlite3.connect(base / "lookup.db") as db:
        db.executescript("CREATE TABLE lemmas(id INTEGER PRIMARY KEY,text TEXT);"
                         "CREATE TABLE lookup(form TEXT,lemma_id INTEGER,src TEXT,lang TEXT);")
        for index, head in enumerate(heads):
            db.execute("INSERT INTO lemmas VALUES (?,?)", (index, head))
            db.execute("INSERT INTO lookup VALUES (?,?,?,?)", (head, index, "el", "all"))
        db.execute("INSERT INTO lookup VALUES ('εμπίπτουν',0,'grc','all')")
    with sqlite3.connect(base / "spell_index.db") as db:
        db.execute("CREATE TABLE spell(stripped TEXT PRIMARY KEY,forms TEXT)")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": {f"data/{p.name}": {"sha256": sha256(p)}
                                               for p in base.iterdir()}}))
    return base, manifest


def test_refresh_preserves_ancient_rows_and_records_every_addition(tmp_path):
    base, manifest = base_snapshot(tmp_path)
    report = refresh(base, tmp_path / "output", manifest)
    assert report["changes"]
    with sqlite3.connect(tmp_path / "output/lookup.db") as db:
        rows = db.execute("SELECT src,lang FROM lookup WHERE form='εμπίπτουν'").fetchall()
        assert rows == [("grc", "all"), ("el", "el")]
        for change in report["changes"]:
            assert db.execute("SELECT count(*) FROM lookup WHERE form=? AND lemma_id=? AND src=? AND lang=?",
                              change["lookup_row"]).fetchone()[0] == 1
    assert sha256(base / "lookup.db") == report["input_sha256"]["lookup.db"]


def test_refresh_rejects_unpinned_base(tmp_path):
    base, manifest = base_snapshot(tmp_path)
    (base / "mg_lookup.json").write_text("{}")
    with pytest.raises(ValueError, match="pinned manifest"):
        refresh(base, tmp_path / "output", manifest)


def test_refresh_is_logically_reversible_and_refuses_unrelated_changes(tmp_path):
    base, manifest = base_snapshot(tmp_path)
    output = tmp_path / "output"
    audit = refresh(base, output, manifest)
    unapply_refresh(output, audit)
    for name, table in [("lookup.db", "lookup"), ("spell_index.db", "spell")]:
        with sqlite3.connect(base / name) as before, sqlite3.connect(output / name) as after:
            assert sorted(before.execute(f"SELECT * FROM {table}")) == sorted(after.execute(f"SELECT * FROM {table}"))
    with pytest.raises(ValueError, match="unrelated work"):
        unapply_refresh(output, audit)


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "data/lookup.db").exists(), reason="built lookup unavailable")
def test_published_data_has_structured_modern_readings():
    from dilemma import Dilemma
    engine = Dilemma(lang="el", convention="triantafyllidis", skip_pos=True)
    for form, lemma in [("αιτούντος", "αιτών"), ("αιτούντες", "αιτών"),
                        ("εμπίπτουν", "εμπίπτω"), ("συνιστούν", "συνιστώ"),
                        ("απορρέουν", "απορρέω")]:
        candidates = engine.lemmatize_verbose(form, guess=False)
        assert candidates and candidates[0].lemma == lemma
        assert candidates[0].source == "lookup"

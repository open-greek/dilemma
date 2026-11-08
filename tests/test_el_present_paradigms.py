import copy
import json
import sqlite3

import pytest

from dilemma.el_present_paradigms import (
    complete_present_pairs, present_forms, unapply_present_pairs,
)


@pytest.mark.parametrize("head,expected", [
    ("συλλαμβάνω", ("συλλαμβάνω", "συλλαμβάνεις", "συλλαμβάνει", "συλλαμβάνουμε",
                     "συλλαμβάνετε", "συλλαμβάνουν")),
    ("διαβιβάζω", ("διαβιβάζω", "διαβιβάζεις", "διαβιβάζει", "διαβιβάζουμε",
                   "διαβιβάζετε", "διαβιβάζουν")),
    ("μεθάω", ("μεθάω", "μεθάς", "μεθάει", "μεθάμε", "μεθάτε", "μεθάνε")),
    ("ακούω", ("ακούω", "ακούς", "ακούει", "ακούμε", "ακούτε", "ακούν")),
    ("σιγοκαίω", ("σιγοκαίω", "σιγοκαίς", "σιγοκαίει", "σιγοκαίμε", "σιγοκαίτε", "σιγοκαίνε")),
    ("κλαίω", ("κλαίω", "κλαις", "κλαίει", "κλαίμε", "κλαίτε", "κλαίνε")),
    ("λέω", ("λέω", "λες", "λέει", "λέμε", "λέτε", "λένε")),
    ("πρωτολέω", ("πρωτολέω", "πρωτολές", "πρωτολέει", "πρωτολέμε", "πρωτολέτε", "πρωτολένε")),
    ("τρώω", ("τρώω", "τρως", "τρώει", "τρώμε", "τρώτε", "τρώνε")),
    ("πλέω", ("πλέω", "πλέεις", "πλέει", "πλέουμε", "πλέετε", "πλέουν")),
    ("σπάω", ("σπάω", "σπας", "σπάει", "σπάμε", "σπάτε", "σπάνε")),
    ("σκάω", ("σκάω", "σκας", "σκάει", "σκάμε", "σκάτε", "σκάνε")),
    ("δρω", ("δρω", "δρας", "δρα", "δρούμε", "δράτε", "δρουν")),
    ("ζω", ("ζω", "ζεις", "ζει", "ζούμε", "ζείτε", "ζουν")),
])
def test_reviewed_present_classes(head, expected):
    assert present_forms(head)[1] == expected


@pytest.mark.parametrize("head", ["αρόω", "άγνωστος", "γράφουν", "Γράφω", "γραφw", "γράφω λέω"])
def test_unsupported_or_malformed_heads_are_not_completed(head):
    assert present_forms(head) is None


def test_completion_requires_a_real_self_mapped_head_and_preserves_existing_readings():
    pairs = [{"form": "συλλαμβάνω", "lemma": "συλλαμβάνω", "pos": "verb", "tags": []}]
    lookup = {"συλλαμβάνω": ("συλλαμβάνω", 5), "συλλαμβάνεις": ("existing reading", 5)}
    heads = {"συλλαμβάνω": [{"source": "el", "headword": "συλλαμβάνω"}],
             "διαβιβάζω": [{"source": "el", "headword": "διαβιβάζω"}]}
    before_pairs, before_lookup = copy.deepcopy(pairs), copy.deepcopy(lookup)
    audit = complete_present_pairs(pairs, lookup, heads)
    assert lookup["συλλαμβάνει"] == ("συλλαμβάνω", 1)
    assert lookup["συλλαμβάνεις"] == ("existing reading", 5)
    assert not any(r["lemma"] == "διαβιβάζω" for r in pairs)
    assert all("present" in c["corrected"]["tags"] for c in audit["changes"])
    assert all(c["original"] is None and c["rule_evidence"] for c in audit["changes"])
    assert complete_present_pairs(pairs, lookup, heads)["changes"] == []
    unapply_present_pairs(pairs, lookup, audit)
    assert pairs == before_pairs and lookup == before_lookup


def test_undo_refuses_unrelated_edits_before_removing_anything():
    pairs, lookup = [], {"γράφω": ("γράφω", 5)}
    audit = complete_present_pairs(pairs, lookup, {"γράφω": []})
    lookup["γράφει"] = ("later reading", 5)
    before = copy.deepcopy((pairs, lookup))
    with pytest.raises(ValueError, match="changed after generation"):
        unapply_present_pairs(pairs, lookup, audit)
    assert (pairs, lookup) == before


def test_existing_cell_is_not_expanded_with_an_inferred_variant():
    pairs = [{"form": "εισακούεις", "lemma": "εισακούω", "pos": "verb",
              "tags": ["present", "active", "indicative", "second-person", "singular"]}]
    lookup = {"εισακούω": ("εισακούω", 5), "εισακούεις": ("εισακούω", 5)}
    report = complete_present_pairs(pairs, lookup, {"εισακούω": []})
    assert not any(c["corrected"]["form"] == "εισακούς" for c in report["changes"])
    assert "εισακούς" not in lookup


def test_a_defective_head_does_not_license_a_full_personal_paradigm():
    pairs, lookup = [], {"δέω": ("δέω", 5)}
    heads = {"δέω": [{"source": "el", "senses": [{"tags": ["impersonal"]}]}]}
    report = complete_present_pairs(pairs, lookup, heads)
    assert report["changes"] == [] and report["unsupported_heads"] == ["δέω"]


def test_personal_and_impersonal_senses_are_kept_separately():
    pairs, lookup = [], {"βρέχω": ("βρέχω", 5)}
    heads = {"βρέχω": [{"source": "el", "senses": [
        {"tags": ["impersonal"], "glosses": ["πέφτει βροχή"]},
        {"tags": [], "glosses": ["κάνω κάτι υγρό"]}]}]}
    report = complete_present_pairs(pairs, lookup, heads)
    assert "βρέχεις" in lookup and report["unsupported_heads"] == []


def test_extractor_collects_defined_heads_without_form_or_participle_pages(tmp_path):
    import build_data as B

    entries = [
        {"word": "συλλαμβάνω", "pos": "verb", "senses": [{"glosses": ["πιάνω"]}]},
        {"word": "συλλάβω", "pos": "verb", "senses": [{"form_of": [{"word": "συλλαμβάνω"}],
          "glosses": ["τύπος του συλλαμβάνω"]}]},
        {"word": "συλληφθείς", "pos": "verb", "senses": [{"tags": ["participle"],
          "form_of": [{"word": "συλλαμβάνω"}], "glosses": ["μετοχή του συλλαμβάνω"]}]},
    ]
    path = tmp_path / "source.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in entries))
    records = {}
    B.extract_pairs(path, "el-el", verb_heads=set(), present_head_records=records)
    assert set(records) == {"συλλαμβάνω"}
    assert records["συλλαμβάνω"][0]["source"] == "el"


def test_snapshot_completion_preserves_all_old_rows_and_can_be_unapplied(tmp_path):
    from scripts.complete_el_present_snapshot import refresh, sha, unapply

    base = tmp_path / "base"
    base.mkdir()
    pairs = [{"form": "γράφω", "lemma": "γράφω", "pos": "verb", "tags": []}]
    lookup = {"γράφω": "γράφω"}
    (base / "mg_pairs.json").write_text(json.dumps(pairs, ensure_ascii=False))
    (base / "mg_lookup.json").write_text(json.dumps(lookup, ensure_ascii=False))
    with sqlite3.connect(base / "lookup.db") as db:
        db.executescript("CREATE TABLE lemmas(id INTEGER PRIMARY KEY,text TEXT);"
                         "CREATE TABLE lookup(form TEXT,lemma_id INTEGER,src TEXT,lang TEXT);")
        db.execute("INSERT INTO lemmas VALUES (1,'γράφω')")
        db.execute("INSERT INTO lookup VALUES ('γράφω',1,'el','all')")
        db.execute("INSERT INTO lookup VALUES ('γράφει',1,'grc','all')")
    with sqlite3.connect(base / "spell_index.db") as db:
        db.execute("CREATE TABLE spell(stripped TEXT PRIMARY KEY,forms TEXT)")
        db.execute("INSERT INTO spell VALUES ('γραφει','γράφει\tgrc')")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": {"data/" + p.name: {"sha256": sha(p)}
                                               for p in base.iterdir()}}))
    output = tmp_path / "output"
    report = refresh(base, output, manifest, {"γράφω": [{"source": "el"}]}, {"el": "a" * 64})
    with sqlite3.connect(output / "lookup.db") as db:
        assert db.execute("SELECT src,lang FROM lookup WHERE form='γράφει'").fetchall() == [
            ("grc", "all"), ("el", "el")]
    assert report["changes"] and report["lookup_rows"] and report["spell_rows"]
    unapply(output, report)
    assert json.loads((output / "mg_pairs.json").read_text()) == pairs
    assert json.loads((output / "mg_lookup.json").read_text()) == lookup
    for name, table in (("lookup.db", "lookup"), ("spell_index.db", "spell")):
        with sqlite3.connect(base / name) as before, sqlite3.connect(output / name) as after:
            assert sorted(before.execute(f"SELECT * FROM {table}")) == sorted(after.execute(f"SELECT * FROM {table}"))
    with pytest.raises(ValueError, match="changed after completion"):
        unapply(output, report)

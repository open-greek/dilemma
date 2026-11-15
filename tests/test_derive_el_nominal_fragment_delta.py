"""The Modern Greek fragment delta removes only the fragments' own readings (#122)."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import derive_el_nominal_fragment_delta as D  # noqa: E402
import refresh_el_generator_delta as R  # noqa: E402


def _entry(word, pos, *cells):
    return {"word": word, "pos": pos, "lang_code": "el",
            "senses": [{"glosses": ["x"]}],
            "forms": [{"form": form, "tags": tags, "source": "declension"}
                      for form, tags in cells]}


ENTRIES = [
    _entry("αρχαΐζων", "adj",
           ("αρχαΐζοντος", ["genitive", "masculine", "singular"]),
           ("ούσης", ["feminine", "genitive", "singular"]),
           ("όντων", ["genitive", "masculine", "plural"])),
    _entry("αισώπειος", "adj", ("ος", ["masculine", "nominative", "singular"])),
    {"word": "ἀρχή", "pos": "noun", "lang_code": "grc",
     "forms": [{"form": "ῆς", "tags": ["genitive"], "source": "declension"}]},
]
LOOKUP = {"αρχαΐζων": "αρχαΐζω", "αρχαΐζοντος": "αρχαΐζω",
          "ούσης": "αρχαΐζω", "ουσης": "αρχαΐζω", "όντων": "ων",
          "οντων": "ων", "ος": "αισώπειος"}
POS = {"ούσης": {"ADJ": "αρχαΐζων"}, "ουσης": {"ADJ": "αρχαΐζων"},
       "όντων": {"NOUN": "ων", "ADJ": "αρχαΐζων"},
       "ος": {"ADJ": "αισώπειος", "PROPN": "Ος"},
       "αρχαΐζοντος": {"ADJ": "αρχαΐζων", "VERB": "αρχαΐζω"}}


def test_cells_follow_the_generator_rule():
    cells = list(D.fragment_cells(("dump", e) for e in ENTRIES))
    assert [(c["lemma"], c["form"]) for c in cells] == [
        ("αρχαΐζων", "ούσης"), ("αρχαΐζων", "όντων"), ("αισώπειος", "ος")]


def test_delta_removes_only_readings_of_the_fragment_headword():
    cells = list(D.fragment_cells(("dump", e) for e in ENTRIES))
    files = D.derive(cells, LOOKUP, POS)
    assert files["mg_lookup.json"]["set"] == {
        "ος": ["αισώπειος", None], "ουσης": ["αρχαΐζω", None],
        "ούσης": ["αρχαΐζω", None]}
    assert files["mg_pos_lookup.json"]["set"] == {
        "ος": [{"ADJ": "αισώπειος", "PROPN": "Ος"}, {"PROPN": "Ος"}],
        "ουσης": [{"ADJ": "αρχαΐζων"}, None],
        "ούσης": [{"ADJ": "αρχαΐζων"}, None],
        "όντων": [{"NOUN": "ων", "ADJ": "αρχαΐζων"}, {"NOUN": "ων"}]}


def test_script_writes_a_delta_the_refresh_tool_applies_and_undoes(tmp_path):
    dumps, base, out = tmp_path / "dumps", tmp_path / "base", tmp_path / "out"
    dumps.mkdir(); base.mkdir()
    (dumps / "kaikki.org-en-dictionary-Greek.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in ENTRIES),
        encoding="utf-8")
    for name, value in (("mg_lookup.json", LOOKUP), ("mg_pos_lookup.json", POS)):
        R.dump(value, base / name)
    delta = tmp_path / "delta.json"
    assert D.main(["--dumps", str(dumps), "--base", str(base),
                   "--out", str(delta)]) == 0
    record = json.loads(delta.read_text(encoding="utf-8"))
    assert record["rule"] == D.RULE
    assert set(record["dumps"]) == {"kaikki.org-en-dictionary-Greek.jsonl"}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": {}}), encoding="utf-8")
    R.main(["apply", "--base", str(base), "--manifest", str(manifest),
            "--delta", str(delta), "--output", str(out)])
    lookup = json.loads((out / "mg_lookup.json").read_text(encoding="utf-8"))
    pos = json.loads((out / "mg_pos_lookup.json").read_text(encoding="utf-8"))
    assert "ούσης" not in lookup and lookup["όντων"] == "ων"
    assert pos["όντων"] == {"NOUN": "ων"} and "ούσης" not in pos
    assert pos["αρχαΐζοντος"] == POS["αρχαΐζοντος"]
    R.main(["undo", "--data", str(out), "--audit",
            str(out / "el_generator_refresh_audit.json")])
    assert json.loads((out / "mg_pos_lookup.json").read_text(encoding="utf-8")) == POS
    assert json.loads((out / "mg_lookup.json").read_text(encoding="utf-8")) == LOOKUP


RECORD = ROOT / "data" / "changes" / "el_nominal_fragments.json"
EL_DUMPS = {"kaikki.org-en-dictionary-Greek.jsonl",
            "kaikki.org-el-dictionary-Greek.jsonl",
            "kaikki.org-el-dictionary-GreekDialects.jsonl"}


def _record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_committed_record_is_a_generator_delta_over_the_pinned_dumps():
    record = _record()
    assert record["rule"] == R.RULE and record["language"] == "el"
    # 1.3.24's build_data.py against the release branch's.
    assert record["generator"]["old"] == "c4ced976d92cc8744bb0f51c7f319e81349783f0"
    assert len(record["generator"]["new"]) == 40
    pins = json.loads((ROOT / "data" / "kaikki_inputs.json").read_text(encoding="utf-8"))
    assert set(record["dumps"]) == EL_DUMPS
    for name, digest in record["dumps"].items():
        assert pins["files"][name]["sha256"] == digest, name
    assert set(record["files"]) == {"mg_lookup.json", "mg_pos_lookup.json",
                                    "mg_pairs.json"}


def test_committed_record_changes_only_fragment_readings():
    files = _record()["files"]
    lookup = files["mg_lookup.json"]["set"]
    pos = files["mg_pos_lookup.json"]["set"]
    assert (len(lookup), len(pos)) == (26, 39)
    assert lookup["ούσης"] == ["αρχαΐζω", None]
    assert lookup["ΧΧΧ"] == ["Καραβίτης", None]
    # A freed key goes to its next source, as a full build gives it.
    assert lookup["ος"] == ["αισώπειος", "Ος"]
    assert lookup["ες"] == ["μουσικομανής", "εις"]
    assert sum(after is not None for _, after in lookup.values()) == 7
    assert pos["ούσης"] == [{"ADJ": "αρχαΐζων"}, None]
    # One reading left: the table keeps only ambiguous forms.
    assert pos["όντων"] == [{"ADJ": "αρχαΐζων", "NOUN": "ων"}, None]
    # A removal only, never a new or changed POS reading.
    for before, after in pos.values():
        if after is not None:
            assert set(after.items()) < set(before.items())
            lost = set(before) - set(after)
            assert lost <= {"ADJ", "NOUN"} and len(after) >= 2


def test_committed_record_removes_fragment_training_pairs():
    pairs = _record()["files"]["mg_pairs.json"]
    removed = {(r["lemma"], r["form"]) for r in pairs["remove"]}
    assert len(pairs["remove"]) == 127
    assert {("αρχαΐζων", "ούσης"), ("αισώπειος", "ος"),
            ("κρεατομηχανή", "μηχανή"), ("βρόντημα", "γ")} <= removed
    # The only additions are the two γένη pairs the generator resolves
    # through the key ος, which now names Ος instead of αισώπειος.
    assert [(r["form"], r["lemma"]) for r in pairs["add"]] == [("γένη", "Ος")] * 2

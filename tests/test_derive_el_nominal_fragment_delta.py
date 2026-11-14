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


def test_committed_record_matches_its_generator():
    record = json.loads((ROOT / "data" / "changes" / "el_nominal_fragments.json")
                        .read_text(encoding="utf-8"))
    assert record["rule"] == D.RULE
    assert record["generator_sha256"] == D.sha256(D.GENERATOR)
    pos = record["files"]["mg_pos_lookup.json"]["set"]
    assert pos["ούσης"] == [{"ADJ": "αρχαΐζων"}, None]
    assert pos["όντων"][1] == {"NOUN": "ων"}
    # A removal only, never a new or changed reading.
    for before, after in pos.values():
        assert after is None or (set(after.items()) < set(before.items()))
    for before, after in record["files"]["mg_lookup.json"]["set"].items():
        assert after[1] is None

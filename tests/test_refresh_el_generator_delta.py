"""The Modern Greek generator delta reaches a snapshot only where it agrees."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import refresh_el_generator_delta as R  # noqa: E402


def _write(directory: Path, files: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in files.items():
        (directory / name).write_text(json.dumps(value, ensure_ascii=False),
                                      encoding="utf-8")


def _heads(*lemmas):
    return {"schema_version": 1, "language": "el",
            "heads": [{"lemma": lemma, "evidence": []} for lemma in lemmas]}


def _tables(lookup, pos, pairs, heads):
    files = {name: ({"schema_version": 1, "language": "el", "cells": []}
                    if name == "el_participles.json" else _heads())
             for name in R.FILES if name.startswith("el_")}
    files.update({"mg_lookup.json": lookup, "mg_pos_lookup.json": pos,
                  "mg_pairs.json": pairs, "el_noun_heads.json": heads})
    return files


def test_delta_applies_where_the_snapshot_agrees_and_undoes(tmp_path):
    pair = lambda form, lemma: {"form": form, "lemma": lemma, "pos": "noun",
                                "tags": ["nominative", "singular"]}
    old = _tables({"ατσίδα": "ατσίδι", "γάτες": "γάτα", "χιονόνερα": "χιονόνερο"},
                  {"ατσίδας": {"NOUN": "ατσίδι"}},
                  [pair("γάτος", "γάτα")], _heads("γάτα"))
    new = _tables({"ατσίδα": "ατσίδα", "ατσίδας": "ατσίδας", "γάτες": "γάτα",
                   "χιονόνερα": "χιονόνερο", "χιονόνερο": "χιονόνερο"},
                  {"ατσίδας": {"NOUN": "ατσίδας"}},
                  [pair("γάτος", "γάτος")], _heads("γάτα", "γάτος"))
    _write(tmp_path / "old", old)
    _write(tmp_path / "new", new)
    (tmp_path / "dumps").mkdir()
    (tmp_path / "dumps" / "d.jsonl").write_text("{}\n")
    delta = tmp_path / "delta.json"
    assert R.main(["diff", str(tmp_path / "old"), str(tmp_path / "new"),
                   "--dumps", str(tmp_path / "dumps"), "--old-commit", "a",
                   "--new-commit", "b", "--out", str(delta)]) == 0
    record = json.loads(delta.read_text())
    assert record["files"]["mg_lookup.json"]["set"] == {
        "ατσίδα": ["ατσίδι", "ατσίδα"], "ατσίδας": [None, "ατσίδας"],
        "χιονόνερο": [None, "χιονόνερο"]}

    # The published snapshot differs: it has its own reading of ατσίδα.
    base = _tables({"ατσίδα": "ατσίδα2", "γάτες": "γάτα", "χιονόνερα": "χιονόνερο"},
                   {"ατσίδας": {"NOUN": "ατσίδι"}, "ζ": {"NOUN": "ζ"}},
                   [pair("γάτος", "γάτα"), pair("ζ", "ζ")], _heads("γάτα"))
    _write(tmp_path / "base", base)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": {
        f"data/{name}": {"sha256": hashlib.sha256(
            (tmp_path / "base" / name).read_bytes()).hexdigest()}
        for name in R.FILES}}))
    out = tmp_path / "out"
    assert R.main(["apply", "--base", str(tmp_path / "base"), "--manifest",
                   str(manifest), "--delta", str(delta), "--output", str(out)]) == 0
    lookup = json.loads((out / "mg_lookup.json").read_text())
    assert lookup["χιονόνερο"] == "χιονόνερο" and lookup["ατσίδας"] == "ατσίδας"
    assert lookup["ατσίδα"] == "ατσίδα2"          # a conflict stays as published
    assert json.loads((out / "mg_pos_lookup.json").read_text())["ατσίδας"] == {
        "NOUN": "ατσίδας"}
    pairs = json.loads((out / "mg_pairs.json").read_text())
    assert pair("γάτος", "γάτος") in pairs and pair("γάτος", "γάτα") not in pairs
    assert pair("ζ", "ζ") in pairs
    assert [h["lemma"] for h in json.loads(
        (out / "el_noun_heads.json").read_text())["heads"]] == ["γάτα", "γάτος"]
    audit = json.loads((out / "el_generator_refresh_audit.json").read_text())
    assert audit["counts"]["mg_lookup.json: conflict"] == 1

    assert R.main(["undo", "--data", str(out), "--audit",
                   str(out / "el_generator_refresh_audit.json")]) == 0
    for name in ("mg_lookup.json", "mg_pos_lookup.json", "el_noun_heads.json"):
        assert json.loads((out / name).read_text()) == base[name]
    assert sorted(map(json.dumps, json.loads((out / "mg_pairs.json").read_text()))) \
        == sorted(map(json.dumps, base["mg_pairs.json"]))

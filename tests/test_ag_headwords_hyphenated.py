"""data/ag_headwords.json carries no hyphenated string, and the record of
the ones removed reproduces the list they were removed from."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from dilemma.citation import hyphenated_lemma

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "data" / "ag_headwords.json"
RECORD = ROOT / "data" / "changes" / "ag_headwords_hyphenated.json"

spec = importlib.util.spec_from_file_location(
    "prune_ag_headwords", ROOT / "scripts" / "prune_ag_headwords.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def test_prune_drops_exactly_the_hyphenated_strings():
    headwords = ["-σον", "ὁτ-", "παρά-ἐπιδημέω", "παρεπιδημέω", "προ-δουλεύω",
                 "σι—ληνός", "συν-αγανακτέω", "συναγανακτέω", "–⏑τατος"]
    kept, removed = P.prune(headwords)
    # a hyphen at an edge (an affix) or beside a metrical sign is not
    # between two letters
    assert kept == ["-σον", "ὁτ-", "παρεπιδημέω", "συναγανακτέω", "–⏑τατος"]
    assert removed == ["παρά-ἐπιδημέω", "προ-δουλεύω", "σι—ληνός",
                       "συν-αγανακτέω"]
    entry = P.record(P.serialize(headwords), headwords, kept, removed)
    assert entry["removed_count"] == 4
    # only συν-αγανακτέω has its spelling without the hyphen in the list
    assert entry["joined_spelling_listed"] == 1
    assert entry["source"]["entries"] == 9 and entry["result"]["entries"] == 5


def test_script_prunes_records_and_is_idempotent(tmp_path, capsys):
    headwords = ["λόγος", "προ-δουλεύω", "συν-αγανακτέω", "συναγανακτέω"]
    listed, record = tmp_path / "ag_headwords.json", tmp_path / "record.json"
    listed.write_bytes(P.serialize(headwords))
    source = listed.read_bytes()
    args = ["--list", str(listed), "--record", str(record)]
    assert P.main([*args, "--check"]) == 1
    assert listed.read_bytes() == source and not record.exists()
    assert P.main(args) == 0
    assert json.loads(listed.read_bytes()) == ["λόγος", "συναγανακτέω"]
    entry = json.loads(record.read_text(encoding="utf-8"))
    assert entry["removed"] == ["προ-δουλεύω", "συν-αγανακτέω"]
    assert entry["source"]["sha256"] == hashlib.sha256(source).hexdigest()
    assert entry["result"]["sha256"] == hashlib.sha256(
        listed.read_bytes()).hexdigest()
    pruned, recorded = listed.read_bytes(), record.read_bytes()
    assert P.main(args) == 0 and P.main([*args, "--check"]) == 0
    assert listed.read_bytes() == pruned and record.read_bytes() == recorded
    # a pruned list that is not the recorded one is reported
    listed.write_bytes(P.serialize(["λόγος"]))
    assert P.main([*args, "--check"]) == 1
    # a list in another serialization is not rewritten
    listed.write_text(json.dumps(headwords, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    assert P.main(args) == 2
    capsys.readouterr()


@pytest.fixture(scope="module")
def tracked():
    if not LIST.exists() or not RECORD.exists():
        pytest.skip("headword list or its change record not present")
    return (json.loads(LIST.read_text(encoding="utf-8")),
            json.loads(RECORD.read_text(encoding="utf-8")))


def test_tracked_list_has_no_hyphenated_string(tracked):
    headwords, _ = tracked
    assert [head for head in headwords if hyphenated_lemma(head)] == []


def test_record_lists_every_removed_string_with_its_reason(tracked):
    headwords, entry = tracked
    removed = entry["removed"]
    assert entry["rule"] == P.RULE and entry["reason"] == P.REASON
    assert removed == sorted(set(removed))
    assert len(removed) == entry["removed_count"] == 6560
    assert all(hyphenated_lemma(head) for head in removed)
    assert not set(removed) & set(headwords)
    for head in ("προ-δουλεύω", "ἐπί-εἰσκομίζω", "πήν-τος",
                 "συντονολυ—διστί"):
        assert head in removed


def test_record_and_list_reproduce_the_list_that_was_pruned(tracked):
    # The published 1.3.26 list is the pruned list plus the recorded
    # strings, byte for byte, so the edit can be replayed and undone.
    headwords, entry = tracked
    assert entry["result"]["sha256"] == hashlib.sha256(
        LIST.read_bytes()).hexdigest()
    source = P.serialize(sorted(set(headwords) | set(entry["removed"])))
    assert hashlib.sha256(source).hexdigest() == entry["source"]["sha256"]
    assert len(source) == entry["source"]["size"]
    kept, removed = P.prune(json.loads(source))
    assert kept == headwords and sorted(removed) == entry["removed"]

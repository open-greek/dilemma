"""Passage regressions distinguish missing grammar from citation conventions."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from dilemma import Dilemma
from dilemma.core import LemmaCandidate, LOOKUP_DB_PATH
from eval.audit_verb_lemmas import DEFAULT_CASES, audit_case, audit_cases


CASE = {
    "id": "reviewed-example",
    "tokens": ["ἐγώ", "εἶδον", "αὐτόν"],
    "target_index": 1,
    "source": {"work": "test"},
    "accepted_lemmas": ["εἶδον", "ὁράω"],
    "required_readings": [{"lemma": "ὁράω", "analyses": [{
        "tense": "aorist", "mood": "indicative", "voice": "active",
        "person": "1", "number": "sg"}]}],
}


class ExampleEngine:
    def __init__(self):
        self.candidate = LemmaCandidate("εἶδον", grammar={
            **deepcopy(CASE["required_readings"][0]), "base_lemma": "ὁράω"})

    def lemmatize(self, word, *, guess):
        assert word == "εἶδον" and guess is False
        return "εἶδον"

    def lemmatize_pos(self, word, tag, *, guess):
        assert word == "εἶδον" and tag == "VERB" and guess is False
        return "ὁράω"

    def lemmatize_verbose(self, word, *, guess):
        assert word == "εἶδον" and guess is False
        return [self.candidate]

    def lemmatize_batch(self, tokens, *, guess):
        assert tokens == CASE["tokens"] and guess is False
        return ["ἐγώ", "εἶδον", "αὐτός"]

    def lemmatize_batch_pos(self, tokens, tags, *, guess):
        assert tokens == CASE["tokens"] and tags == ["X", "VERB", "X"]
        assert guess is False
        return ["ἐγώ", "ὁράω", "αὐτός"]


def test_audit_accepts_valid_citation_differences_and_uses_passage_target():
    report = audit_cases(ExampleEngine(), [CASE])
    assert report["summary"]["passed"] == 1
    assert report["summary"]["failed"] == 0


def test_audit_reports_a_missing_owner_even_with_a_valid_single_answer():
    case = deepcopy(CASE)
    case["required_readings"].append({"lemma": "another verb", "analyses": []})
    assert audit_case(ExampleEngine(), case)["issues"] == [{
        "api": "verbose", "reason": "missing_grammatical_reading", "lemma": "another verb"}]


def test_audit_detects_an_unsupported_person_and_number():
    engine = ExampleEngine()
    engine.candidate.grammar["analyses"][0]["number"] = "pl"
    reasons = {issue["reason"] for issue in audit_case(engine, CASE)["issues"]}
    assert reasons == {"missing_analysis", "unsupported_analysis"}


def test_audit_detects_inconsistent_batch_results():
    engine = ExampleEngine()
    engine.lemmatize_batch = lambda tokens, **kwargs: ["ἐγώ", "ὁράω", "αὐτός"]
    assert audit_case(engine, CASE)["issues"] == [{
        "api": "batch", "reason": "differs_from_scalar"}]


def test_passage_fixture_has_unique_ids_and_preserves_source_readings():
    fixture = json.loads(DEFAULT_CASES.read_text(encoding="utf-8"))
    cases = fixture["cases"]
    assert len({case["id"] for case in cases}) == len(cases)
    files = {source["file"] for source in fixture["sources"]["diorisis"]["inputs"]}
    for case in cases:
        assert 0 <= case["target_index"] < len(case["tokens"])
        assert len(case["tokens"]) == len(case["original_beta"])
        assert case["source"]["file"] in files
        assert case["source_readings"] and case["accepted_lemmas"]
        assert not Path(case["source"]["file"]).is_absolute()
        if case.get("required_readings"):
            assert case["review_evidence"]


@pytest.mark.skipif(not LOOKUP_DB_PATH.exists(), reason="Requires downloaded lookup data")
@pytest.mark.parametrize("lang", ["grc", "all"])
def test_reviewed_passages_with_downloaded_lookup_and_available_pos_tables(lang):
    fixture = json.loads(DEFAULT_CASES.read_text(encoding="utf-8"))
    engine = Dilemma(lang=lang, convention=None, normalize=False,
                     resolve_articles=False, skip_pos=False)
    report = audit_cases(engine, fixture["cases"])
    failures = [(case["id"], case["issues"]) for case in report["cases"] if case["issues"]]
    assert not failures, failures
    assert engine._model is None

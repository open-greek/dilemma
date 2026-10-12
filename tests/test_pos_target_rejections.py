"""Explicit Ancient Greek POS calls cannot revive rejected corpus targets."""
import hashlib
import json

import pytest

from dilemma import core


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    heads = tmp_path / "heads.json"
    heads.write_text(json.dumps({"schema_version": 2, "heads": {
        "Ἀκαμαντίς": ["lexicon"], "Νομᾶς": ["lexicon"],
        "ὥρα": ["lexicon"], "Ὥρα": ["lexicon"], "λόγος": ["lexicon"],
    }}), encoding="utf-8")
    audit = tmp_path / "pos.audit.json"
    payload = {"schema_version": 2, "language": "grc", "inputs": {
        "grc_lexical_heads": {"size": heads.stat().st_size,
                               "sha256": hashlib.sha256(heads.read_bytes()).hexdigest()}},
        "rejected_corpus_targets": {"αξαμαντις": {"glaux": 1}, "νυμα": {"glaux": 1}}}
    audit.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(core, "GRC_LEXICAL_HEADS_PATH", heads)
    monkeypatch.setattr(core, "TREEBANK_POS_AUDIT_PATH", audit)
    return heads, audit, payload


def model(lemmas, pos=None, *, lang="grc", grammatical=None):
    result = core.Dilemma.__new__(core.Dilemma)
    result.lang = lang
    result._convention_name = None
    candidates = [core.LemmaCandidate(lemma=lemma, source="lookup") for lemma in lemmas]
    result._lemmatize_verbose_impl = lambda word, **kwargs: list(candidates)
    result._pos_table_lookup = lambda word, upos: pos
    result._grammatical_readings = lambda word: grammatical or {}
    result._apply_convention = lambda lemma: lemma
    result._article_pos_lemma = lambda word, upos: None
    result.lemmatize = lambda word, **kwargs: lemmas[0] if lemmas else None
    result.lemmatize_batch = lambda words, **kwargs: [lemmas[0] if lemmas else None] * len(words)
    return result


@pytest.mark.parametrize("source", ["table", "fallback", "empty_candidates"])
def test_undefined_target_is_vetoed_from_every_explicit_pos_path(evidence, source):
    instance = model([] if source == "empty_candidates" else ["αξαμαντις"],
                     "αξαμαντις" if source != "fallback" else None)
    assert instance.lemmatize_pos("Ἀκαμαντίδος", "NOUN", guess=False) is None
    assert instance.lemmatize_batch_pos(["Ἀκαμαντίδος"], ["NOUN"], guess=False) == [None]


def test_rejected_lookup_cannot_block_independent_valid_pos_reading(evidence):
    instance = model(["νυμα"], "Νομᾶς")
    assert instance.lemmatize_pos("Νομᾷ", "NOUN", guess=False) == "Νομᾶς"
    assert instance.lemmatize_batch_pos(["Νομᾷ"], ["NOUN"], guess=False) == ["Νομᾶς"]
    assert instance.lemmatize("Νομᾷ", guess=False) == "νυμα"


def test_rejected_batch_baseline_does_not_hide_retained_structured_candidate(evidence):
    instance = model(["αξαμαντις", "Ἀκαμαντίς"])
    assert instance.lemmatize_pos("Ἀκαμαντίδος", "NOUN", guess=False) == "Ἀκαμαντίς"
    assert instance.lemmatize_batch_pos(["Ἀκαμαντίδος"], ["NOUN"], guess=False) == ["Ἀκαμαντίς"]


@pytest.mark.parametrize("attested", [True, False])
def test_rejected_baseline_is_distinct_from_attestation_failure(evidence, attested):
    instance = model(["νυμα"], "Νομᾶς")
    instance._resolve_attested_form = lambda word: word if attested else None
    expected = "Νομᾶς" if attested else None
    assert instance.lemmatize_pos("Νομᾷ", "NOUN", attested_only=True) == expected
    assert instance.lemmatize_batch_pos(["Νομᾷ"], ["NOUN"], attested_only=True) == [expected]


@pytest.mark.parametrize("targets,expected", [({"Νομᾶς"}, "Νομᾶς"),
                                             ({"Νομᾶς", "λόγος"}, None)])
def test_rejected_priority_entry_uses_only_unique_retained_grammatical_reading(
    evidence, targets, expected,
):
    instance = model(["νυμα"], "νυμα", grammatical={"NOUN": targets | {"νυμα"}})
    assert instance.lemmatize_pos("Νομᾷ", "NOUN", guess=False) == expected
    assert instance.lemmatize_batch_pos(["Νομᾷ"], ["NOUN"], guess=False) == [expected]


@pytest.mark.parametrize("upos,expected", [("NOUN", "ὥρα"), ("PROPN", "Ὥρα")])
def test_exact_pos_citation_wins_over_stripped_breathing_collision(evidence, upos, expected):
    instance = model(["ὤρα"], expected)
    assert instance.lemmatize_pos("Ὦρα", upos, guess=False) == expected
    assert instance.lemmatize_batch_pos(["Ὦρα"], [upos], guess=False) == [expected]


@pytest.mark.parametrize("kind", ["missing", "stale_heads", "wrong_schema", "bad_count", "missing_pin"])
def test_unbound_rejection_evidence_fails_clearly(evidence, kind):
    heads, audit, payload = evidence
    if kind == "missing":
        audit.unlink()
    elif kind == "stale_heads":
        heads.write_bytes(heads.read_bytes() + b"\n")
    else:
        if kind == "wrong_schema":
            payload["schema_version"] = 1
        elif kind == "bad_count":
            payload["rejected_corpus_targets"]["νυμα"]["glaux"] = True
        else:
            payload["inputs"].pop("grc_lexical_heads")
        audit.write_text(json.dumps(payload), encoding="utf-8")
    for method in ("scalar", "batch"):
        instance = model(["λόγος"])
        with pytest.raises(ValueError, match="Ancient Greek POS data mismatch"):
            if method == "scalar":
                instance.lemmatize_pos("λόγος", "NOUN", guess=False)
            else:
                instance.lemmatize_batch_pos(["λόγος"], ["NOUN"], guess=False)


@pytest.mark.parametrize("lang,word", [("el", "λογος"), ("all", "λόγος")])
def test_modern_pos_behavior_does_not_require_ancient_audit(evidence, lang, word):
    evidence[1].unlink()
    instance = model(["νυμα"], lang=lang)
    assert instance.lemmatize_pos(word, "NOUN", guess=False) == "νυμα"
    assert instance.lemmatize_batch_pos([word], ["NOUN"], guess=False) == ["νυμα"]


@pytest.mark.parametrize("lang", ["el", "all"])
def test_modern_accent_collision_keeps_prior_candidate_order(evidence, lang):
    evidence[1].unlink()
    instance = model(["νόμος"], "νομός", lang=lang)
    assert instance.lemmatize_pos("νομών", "NOUN", guess=False) == "νόμος"
    assert instance.lemmatize_batch_pos(["νομών"], ["NOUN"], guess=False) == ["νόμος"]


def test_nongreek_pos_passthrough_does_not_require_ancient_audit(evidence):
    evidence[1].unlink()
    instance = model(["123"])
    assert instance.lemmatize_pos("123", "NUM", guess=False) == "123"


def test_rejection_evidence_is_cached_per_model(evidence, monkeypatch):
    instance = model(["λόγος"])
    assert instance.lemmatize_pos("λόγος", "NOUN", guess=False) == "λόγος"
    monkeypatch.setattr(core, "_bound_pos_rejections", lambda *args: pytest.fail("revalidated per word"))
    assert instance.lemmatize_batch_pos(["λόγος", "λόγος"], ["NOUN", "NOUN"], guess=False) == ["λόγος", "λόγος"]


def test_shipped_data_blocks_undefined_target_and_preserves_known_pos_readings():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("shipped Dilemma lookup is unavailable")
    instance = core.Dilemma(lang="grc")
    assert instance.lemmatize_pos("Ἀκαμαντίδος", "NOUN", guess=False) != "αξαμαντις"
    assert instance.lemmatize_batch_pos(["Ἀκαμαντίδος"], ["NOUN"], guess=False) != ["αξαμαντις"]
    assert instance.lemmatize_pos("Νομᾷ", "NOUN", guess=False) == "Νομᾶς"
    assert instance.lemmatize_pos("Ὦρα", "NOUN", guess=False) == "ὥρα"
    assert instance.lemmatize_pos("Ὦρα", "PROPN", guess=False) == "Ὥρα"
    assert instance.lemmatize_pos("ψυχῇ", "NOUN", guess=False) == "ψυχή"

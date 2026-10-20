from collections import Counter
from types import SimpleNamespace
import pytest

from eval.eval_glossapi_frequency import (ContextCounter, accent_restorations, candidate_records, validate_runtime_inputs,
    load_ud, predict, score)
from scripts.audit_glossapi_coverage import RawSample


def test_context_cleaning_preserves_exact_raw_windows_and_prose():
    text = "α. Η αίτηση εξετάζεται.\nΧειροκίνητη ενέργεια:\nΤο ίδιο μεγάλο κείμενο επαναλαμβάνεται εδώ πολλές φορές σήμερα.\n" * 3
    cleaner = ContextCounter({})
    cleaner.observe(text, "document", 30)
    raw = RawSample(token_cap=30)
    raw.add_document(text, identity="document")
    assert cleaner.raw == raw.tokens
    assert cleaner.removed["enumeration_label"] > 0
    assert cleaner.removed["field_label"] > 0
    assert cleaner.removed["duplicate_line"] > 0
    assert cleaner.clean["αίτηση"] > 0
    assert "α" not in cleaner.clean


def test_stress_restoration_is_unambiguous_and_uppercase_only():
    choices = accent_restorations({"νόμος": "νόμος", "νομός": "νομός",
                                   "υπουργείο": "υπουργείο"}, {"νομος", "υπουργειο"})
    assert choices == {"υπουργειο": "υπουργείο"}
    cleaner = ContextCounter(choices)
    cleaner.observe("ΥΠΟΥΡΓΕΙΟ υπουργειο ΝΟΜΟΣ", "id", 100)
    assert cleaner.clean == Counter({"υπουργείο": 1, "υπουργειο": 1, "νομος": 1})


def test_ud_loader_excludes_multiword_headers_and_keeps_sentence_context(tmp_path):
    fixture = tmp_path / "test.conllu"
    fixture.write_text("1-2\tστο\t_\t_\n1\tσε\tσε\tADP\n2\tτο\tο\tDET\n\n1\tΛέξη\tλέξη\tNOUN\n")
    rows = load_ud(fixture)
    assert [r["previous"] for r in rows] == ["", "σε", ""]


def test_ranking_evaluation_counts_paired_gains_and_losses():
    rows = [{"gold": "β", "runtime": "α", "pool": ["α", "β"]},
            {"gold": "α", "runtime": "α", "pool": ["α", "β"]}]
    prior = {"α": .2, "β": .8}
    assert predict(rows[0], prior) == "β"
    report = score(rows, prior, ["α", "α"])
    assert report["gains"] == report["losses"] == 1
    assert report["paired_exact_p"] == 1


def test_candidate_pool_uses_validated_source_alternatives_without_gold_pos():
    class Engine:
        def lemmatize_verbose(self, word, **kwargs):
            return [SimpleNamespace(lemma="μέτρο", source="lookup", score=1, proper=False)]

        def lemmatize_pos(self, word, pos, **kwargs):
            assert kwargs == {"guess": False}
            return "μετράω" if pos == "VERB" else None

    rows = [{"form": "μέτρα", "gold": "unseen gold", "upos": "NOUN", "previous": ""},
            {"form": "μέτρα", "gold": "unseen gold", "upos": "VERB", "previous": "τα"}]
    result = candidate_records(rows, Engine(), {"μέτρα": {"VERB": "μετράω", "NOUN": "μέτρο"}})
    assert result[0]["pool"] == ["μέτρο", "μετράω"]
    assert result[1]["pool"] == []


def test_incomplete_runtime_is_rejected_before_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.eval_glossapi_frequency.DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="missing or differs"):
        validate_runtime_inputs()

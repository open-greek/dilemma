"""Regression coverage for the pinned Ancient Greek expansion layer."""

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from overlay_lsj import (
    overlay_expansion,
    resolve_target_editorial_entries,
    verify_file,
    verify_sentinels,
    load_noun_generation_audit,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "ag_expansion_reference.json"
LOOKUP_DB = ROOT / "data" / "lookup.db"


def test_overlay_adds_only_expansion_entries_and_preserves_current_values():
    reference_base = {"base": "old-base", "base-conflict": "old"}
    reference = {
        **reference_base,
        "ἀποκλείεται": "historical-lemma",
        "ῥαψάμενοι": "new-historical-lemma",
        "δακτύλι": "historical-lemma",
        "κατεβρεχθῶσι(ν": "βρέχω",
        "unmarked": "generated-fallback",
    }
    target = {
        "base": "new-base",
        "δακτύλι": "current-lemma",
        "new": "new-lemma",
    }

    stats = overlay_expansion(
        reference, reference_base, target, excluded_forms={"ἀποκλείεται"}
    )

    assert target == {
        "base": "new-base",
        "δακτύλι": "current-lemma",
        "new": "new-lemma",
        "ῥαψάμενοι": "new-historical-lemma",
    }
    assert stats["expansion_entries"] == 5
    assert stats["unmarked_entries"] == 1
    assert stats["editorial_entries"] == 1
    assert stats["excluded_entries"] == 1
    assert stats["added"] == 1
    assert stats["conflicts_preserved"] == 1


def test_target_editorial_cleanup_resolves_only_movable_nu():
    target = {
        "κατεβρεχθῶσι(ν": "βρέχω",
        "κατεβρεχθῶσι": "current-lemma",
        "ἀ[ζηχὲς": "ἀζηχής",
        "plain": "plain",
    }

    stats = resolve_target_editorial_entries(target)

    assert target == {
        "κατεβρεχθῶσι": "current-lemma",
        "plain": "plain",
        "κατεβρεχθῶσιν": "βρέχω",
    }
    assert stats == {
        "target_editorial_removed": 2,
        "target_movable_nu_entries": 1,
        "target_movable_nu_added": 1,
        "target_movable_nu_present": 1,
        "target_movable_nu_conflicts": 1,
    }


def test_overlay_does_not_restore_superseded_verb_generation():
    reference_base = {"ἀβακίζομαι": "ἀβακίζομαι"}
    reference = {
        **reference_base,
        "ἀβακιζόομαι": "ἀβακίζομαι",
        "ἀβακιζόμεθα": "ἀβακίζομαι",
        "σύμεφην": "σύμφημι",
        "φῆμι": "φημί",
        "ἀμφιένυμι": "ἀμφιέννυμι",
        "ἀμφιέννυμι": "ἀμφιέννυμι",
        "λύεις": "λύω",
    }
    target = {
        # Current independent evidence and corrected generation always win.
        "ἀβακίζομαι": "ἀβακίζομαι",
        "ἀβακιζόμεθα": "ἀβακίζομαι",
        "ἀμφιέννυμι": "ἀμφιέννυμι",
        "σύμεφην": "independent-current-analysis",
    }
    before = dict(target)
    stats = overlay_expansion(reference, reference_base, target)
    assert target == {**before, "λύεις": "λύω"}
    assert stats["superseded_verb_entries"] == 3
    assert stats["already_present"] == 3


def test_pin_verification_rejects_changed_bytes(tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b"known bytes")
    pin = {
        "size": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    verify_file(artifact, pin, "fixture")

    artifact.write_bytes(b"changed bytes")
    with pytest.raises(SystemExit, match="mismatch"):
        verify_file(artifact, pin, "fixture")


def test_manifest_pins_reference_base_and_traced_forms():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert len(manifest["reference"]["revision"]) == 40
    assert len(manifest["reference"]["sha256"]) == 64
    assert len(manifest["reference_base"]["sha256"]) == 64
    assert len(manifest["behavior_baseline"]["sha256"]) == 64
    assert len(manifest["exclusions"]["sha256"]) == 64
    assert manifest["expected_counts"]["expansion_entries"] == 7_581_466
    assert manifest["expected_counts"]["unmarked_entries"] == 3_839_827
    assert manifest["expected_counts"]["editorial_entries"] == 247
    assert manifest["expected_counts"]["excluded_entries"] == 36_839
    assert manifest["artifact_floor"]["grc_source_rows"] == 10_000_000
    assert manifest["sentinels"]["ῥαψάμενοι"] == "ῥάπτω"
    verify_sentinels(manifest["sentinels"], manifest["sentinels"])


def test_superseded_noun_family_keeps_independent_and_current_spellings():
    reference = {"ωνός": "ἀγκών", "ἀγκῶνος": "ἀγκών", "παλαιός": "παλαιός"}
    base = {"ἀγκῶνος": "ἀγκών"}
    target = {"ωνός": "independent-current"}
    stats = overlay_expansion(reference, base, target,
                              superseded_noun_lemmas={"ἀγκών"})
    assert target == {"ωνός": "independent-current", "παλαιός": "παλαιός"}
    assert stats["superseded_noun_entries"] == 0
    target = {}
    stats = overlay_expansion(reference, base, target,
                              superseded_noun_lemmas={"ἀγκών"})
    assert "ωνός" not in target and "ἀγκῶνος" not in target
    assert stats["superseded_noun_entries"] == 1


def test_noun_audit_rejects_different_inputs_even_with_identical_source(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "build"))
    import expand_lsj
    monkeypatch.setattr(expand_lsj, "noun_generation_input_hashes", lambda: {"fixture": "current"})
    path = tmp_path / "audit.json"
    audit = {"schema_version": 1, "generator_sha256": hashlib.sha256(
        (ROOT / "build/expand_lsj.py").read_bytes()).hexdigest(),
        "input_sha256": {"fixture": "old"}, "superseded_lemmas": ["ἀγκών"]}
    path.write_text(json.dumps(audit))
    with pytest.raises(SystemExit, match="noun inputs changed"):
        load_noun_generation_audit(path)
    audit["input_sha256"] = {"fixture": "current"}
    path.write_text(json.dumps(audit))
    assert load_noun_generation_audit(path) == frozenset({"ἀγκών"})


@pytest.mark.skipif(not LOOKUP_DB.exists(), reason="lookup.db not downloaded")
def test_shipped_lookup_contains_recovered_expansion_sentinels():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    connection = sqlite3.connect(f"file:{LOOKUP_DB}?mode=ro", uri=True)
    try:
        grc_rows = connection.execute(
            "SELECT COUNT(*) FROM lookup WHERE src = 'grc'"
        ).fetchone()[0]
        assert grc_rows >= manifest["artifact_floor"]["grc_source_rows"]
        for form, lemma in manifest["sentinels"].items():
            rows = connection.execute(
                "SELECT l.text FROM lookup k "
                "JOIN lemmas l ON l.id = k.lemma_id "
                "WHERE k.form = ? AND k.src = 'grc'",
                (form,),
            ).fetchall()
            assert (lemma,) in rows, f"missing {form} -> {lemma}"
    finally:
        connection.close()


@pytest.mark.skipif(not LOOKUP_DB.exists(), reason="lookup.db not downloaded")
def test_shipped_lookup_contains_no_editorial_sigla():
    connection = sqlite3.connect(f"file:{LOOKUP_DB}?mode=ro", uri=True)
    try:
        count = connection.execute(
            "SELECT COUNT(*) FROM lookup k "
            "JOIN lemmas l ON l.id = k.lemma_id "
            "WHERE instr(k.form, '[') OR instr(k.form, ']') "
            "OR instr(k.form, '(') OR instr(k.form, ')') "
            "OR instr(l.text, '[') OR instr(l.text, ']') "
            "OR instr(l.text, '(') OR instr(l.text, ')')"
        ).fetchone()[0]
        assert count == 0
    finally:
        connection.close()


def test_historical_nonlexical_lemmas_do_not_regenerate_source_notation():
    reference = {
        "έφʼωμαι": "ἐφʼ ᾧ",
        "λεχέτω": "λέγω =⟩ λέχω",
        "οὐδεπέτω": "οὐδέ πω",
        "παρέσται": "παρά,εἰσ-εἰμί",
        "ἀναθαρρρέω": "ἀναθαρρρέω",
        "μέλλλω": "μέλλλω",
        "ἀπορρρέω": "ἀπορρρέω",
        "ὑποδ̣έω": "ὑποδ̣έω",
        "γῶνται": "γῶ",
        "ὕεται": "ὕω",
        "τἀμά": "ἐμός",
    }
    target = {}
    stats = overlay_expansion(reference, {}, target)
    assert target == {"γῶνται": "γῶ", "ὕεται": "ὕω", "τἀμά": "ἐμός"}
    assert stats["nonlexical_lemma_entries"] == 8


def test_historical_nonlexical_guard_preserves_base_and_current_evidence():
    reference = {
        "έφʼωμαι": "ἐφʼ ᾧ",
        "μέλλλω": "μέλλλω",
        "παρέσται": "παρά,εἰσ-εἰμί",
    }
    base = {"μέλλλω": "independent-base-analysis"}
    target = {"έφʼωμαι": "independent-current-analysis"}
    stats = overlay_expansion(reference, base, target)
    assert target == {"έφʼωμαι": "independent-current-analysis"}
    assert stats["already_present"] == 1
    assert stats["nonlexical_lemma_entries"] == 1


def test_overlay_withholds_irregular_compounds_but_preserves_independent_rows():
    reference_base = {"ἐπανῄει": "ἐπάνειμι"}
    reference = {
        **reference_base,
        "ἐπάνηει": "ἐπάνειμι",
        "ἄνεστί": "ἄνειμι",
        "πάρεισι": "πάρειμι",
        "σύνηδει": "σύνοιδα",
        "ᾔει": "εἶμι",
    }
    target = {"πάρεισι": "independent-source", "ἄνειμι": "ἄνειμι"}
    before = dict(target)
    stats = overlay_expansion(reference, reference_base, target)
    assert target == {**before, "ᾔει": "εἶμι"}
    assert stats["superseded_verb_entries"] == 3
    assert stats["already_present"] == 1

#!/usr/bin/env python3
"""Tests for the per-work morphology reconciliation pass.

Run with:
    python -m pytest tests/test_morph_reconcile.py -x -v
"""

import ast
import pathlib

import pytest

import dilemma.morph_reconcile as mr
from dilemma.morph_reconcile import (
    reconcile_token,
    reconcile_work,
    vote_readings,
    Reading,
    Provenance,
    ReconciledToken,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def verb(source, lemma, **feats):
    """A verbal Reading with the given lemma and feats."""
    return {"source": source, "lemma": lemma, "upos": "VERB", "feats": feats}


def methods(tok):
    return [p.method for p in tok.provenance]


# ---------------------------------------------------------------------------
# Purity: no heavy imports
# ---------------------------------------------------------------------------


def test_module_is_pure_no_heavy_imports():
    """The module must import nothing beyond a small stdlib set -- no torch /
    onnxruntime / mlx / transformers / numpy. Checked against the actual
    import statements (via ast) so docstring mentions don't trip it."""
    src = pathlib.Path(mr.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported.update(n.name.split(".")[0] for n in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "re", "unicodedata", "dataclasses"}, \
        f"morph_reconcile imports beyond the pure stdlib set: {imported}"


# ---------------------------------------------------------------------------
# Voting + variant storage
# ---------------------------------------------------------------------------


def test_vote_majority_and_keeps_all_variants():
    readings = [
        verb("a", "λύω", Person="3", Number="Sing", Voice="Act"),
        verb("b", "λύω", Person="3", Number="Sing", Voice="Mid"),
        verb("c", "λύω", Person="3", Number="Sing", Voice="Mid"),
    ]
    tok = reconcile_token("λύεται", readings)
    # majority (2x Mid) wins the chosen value...
    assert tok.chosen["feats"]["Voice"] == "Mid"
    # ...but the disagreement is surfaced and EVERY reading is retained.
    assert "Voice" in tok.contested_fields
    assert tok.contested is True
    assert len(tok.readings) == 3
    assert {r.source for r in tok.readings} == {"a", "b", "c"}
    assert "vote" in methods(tok)


def test_vote_tie_keeps_primary_but_flags():
    readings = [
        verb("a", "λύω", Number="Sing"),
        verb("b", "λύω", Number="Plur"),
    ]
    chosen, disagree = vote_readings(readings)
    # tie -> primary's value is kept, but the field is reported.
    assert chosen["feats"]["Number"] == "Sing"
    assert "Number" in disagree


def test_single_source_no_disagreement():
    tok = reconcile_token(
        "ἄειδε", [verb("a", "ἀείδω", Person="2", Number="Sing",
                       Tense="Past", Mood="Imp", Voice="Act")])
    assert tok.contested is False
    assert tok.contested_fields == []


# ---------------------------------------------------------------------------
# High-precision rule 1: middle ending vs Active
# ---------------------------------------------------------------------------


def test_mid_ending_forces_voice_and_resolves():
    # ὑπέσχετο tagged active: the -ετο ending is mediopassive -> Mid (forced).
    tok = reconcile_token(
        "ὑπέσχετο",
        [verb("a", "ὑπισχνέομαι", Person="3", Number="Sing", Voice="Act",
              Tense="Past", Aspect="Perf", Mood="Ind")])
    assert tok.chosen["feats"]["Voice"] == "Mid"
    assert tok.contested is False
    assert "rule:mid-ending" in methods(tok)


def test_mid_ending_not_fired_when_already_mid():
    tok = reconcile_token(
        "ὑπέσχετο",
        [verb("a", "ὑπισχνέομαι", Person="3", Number="Sing", Voice="Mid",
              Tense="Past", Aspect="Perf", Mood="Ind")])
    assert tok.chosen["feats"]["Voice"] == "Mid"
    assert "rule:mid-ending" not in methods(tok)


def test_mid_ending_resolves_a_voted_disagreement():
    # Sources split Act/Mid; the forced ending decides it and clears contested.
    readings = [
        verb("a", "ὑπισχνέομαι", Person="3", Number="Sing", Voice="Act"),
        verb("b", "ὑπισχνέομαι", Person="3", Number="Sing", Voice="Pass"),
    ]
    tok = reconcile_token("ὑπέσχετο", readings)
    assert tok.chosen["feats"]["Voice"] == "Mid"
    assert "Voice" not in tok.contested_fields


def test_mid_ending_skips_active_infinitive():
    # The aorist ACTIVE infinitive ends in -σαι (λῦσαι, ποιῆσαι), colliding
    # with the mediopassive 2sg -σαι. It must NOT be flipped to middle.
    for form, lemma in [("λῦσαι", "λύω"), ("ποιῆσαι", "ποιέω")]:
        tok = reconcile_token(
            form, [verb("a", lemma, Voice="Act", Tense="Past", Aspect="Perf",
                        Mood="Inf")])
        assert tok.chosen["feats"]["Voice"] == "Act", form
        assert "rule:mid-ending" not in methods(tok), form


def test_mid_ending_skips_sai_infinitive_by_verbform():
    # Real infinitives carry VerbForm=Inf with NO Mood field; the -σαι active
    # infinitive (ἐκπέρσαι) must not be flipped to middle.
    tok = reconcile_token(
        "ἐκπέρσαι",
        [verb("a", "ἐκπέρθω", Voice="Act", Tense="Past", Aspect="Perf",
              VerbForm="Inf")])
    assert tok.chosen["feats"]["Voice"] == "Act"
    assert "rule:mid-ending" not in methods(tok)


def test_mid_ending_skips_active_participle():
    # The fem-pl active participle ends in -ουσαι (φέρουσαι), colliding with the
    # mediopassive 2sg -σαι; it must not be flipped to middle.
    tok = reconcile_token(
        "φέρουσαι",
        [verb("a", "φέρω", Voice="Act", VerbForm="Part", Gender="Fem",
              Number="Plur", Case="Nom")])
    assert tok.chosen["feats"]["Voice"] == "Act"
    assert "rule:mid-ending" not in methods(tok)


def test_mid_ending_skips_active_optative_3sg():
    # The aorist active optative 3sg also ends in -σαι (γηθήσαι): active, not mid.
    tok = reconcile_token(
        "γηθήσαι",
        [verb("a", "γηθέω", Voice="Act", Tense="Past", Aspect="Perf",
              Mood="Opt", VerbForm="Fin", Person="3", Number="Sing")])
    assert tok.chosen["feats"]["Voice"] == "Act"
    assert "rule:mid-ending" not in methods(tok)


# ---------------------------------------------------------------------------
# Flagger: elided present (demoted from auto-apply -- ambiguous in Homer)
# ---------------------------------------------------------------------------


def test_elision_present_3sg_flagged_not_forced():
    # πέτετʼ ( = past πέτετο OR present πέτεται ) tagged present 3sg. Homer
    # elides the present mediopassive -ται too, so the tense is NOT forced --
    # only flagged contested with a Past suggestion for the adjudicator.
    tok = reconcile_token(
        "πέτετ̓",
        [verb("a", "πέτομαι", Person="3", Number="Sing", Tense="Pres",
              Voice="Mid", Mood="Ind")])
    assert tok.chosen["feats"]["Tense"] == "Pres"          # NOT forced
    assert "rule:elision-tense" not in methods(tok)
    assert "Tense" in tok.contested_fields
    assert "flag:elided-present" in methods(tok)


def test_elision_present_recovered_by_adjudicator_agreement():
    # the demoted case stays recoverable: the flagger suggests Past, and an
    # adjudicator that agrees promotes it through the AND-gate.
    tok = reconcile_token(
        "πέτετ̓",
        [verb("a", "πέτομαι", Person="3", Number="Sing", Tense="Pres",
              Voice="Mid", Mood="Ind")],
        adjudicator=lambda t: {"changes": {"Tense": "Past"},
                               "confidence": 0.9})
    assert tok.chosen["feats"]["Tense"] == "Past"
    assert "Tense" not in tok.contested_fields
    assert any(m.startswith("adjudicator+") for m in methods(tok))


def test_elision_tense_caveat_does_not_force_nt_mediopassive():
    # ἐπαυρίσκοντʼ: -ντʼ can be present mediopassive -νται, so the tense is
    # NOT forced -- it is only flagged contested.
    tok = reconcile_token(
        "ἐπαυρίσκοντ̓",
        [verb("a", "ἐπαυρίσκω", Person="3", Number="Plur", Tense="Pres",
              Voice="Mid,Pass", Mood="Ind")])
    assert tok.chosen["feats"]["Tense"] == "Pres"        # unchanged
    assert "rule:elision-tense" not in methods(tok)
    assert "Tense" in tok.contested_fields               # but surfaced
    assert "flag:elided-present" in methods(tok)


def test_elision_tense_does_not_force_2pl():
    # An elided -τʼ on a 2pl is the active -τε (present), so not forced.
    tok = reconcile_token(
        "φέρετ̓",
        [verb("a", "φέρω", Person="2", Number="Plur", Tense="Pres",
              Voice="Act", Mood="Ind")])
    assert tok.chosen["feats"]["Tense"] == "Pres"
    assert "rule:elision-tense" not in methods(tok)
    assert "Tense" in tok.contested_fields


# ---------------------------------------------------------------------------
# High-precision rule 3: perfect / strong-aorist stem
# ---------------------------------------------------------------------------


def test_perf_stem_forces_aspect():
    # ὀρώρηται carries the ὄρνυμι Attic-reduplicated perfect stem -> Perf.
    tok = reconcile_token(
        "ὀρώρηται",
        [verb("a", "ὄρνυμι", Person="3", Number="Sing", Tense="Pres",
              Voice="Mid,Pass", Mood="Ind")])
    assert tok.chosen["feats"]["Aspect"] == "Perf"
    assert "rule:perf-stem" in methods(tok)


def test_perf_stem_strong_aorist():
    tok = reconcile_token(
        "λάθοντο",
        [verb("a", "λανθάνω", Person="3", Number="Plur", Tense="Past",
              Voice="Mid", Mood="Ind")])
    assert tok.chosen["feats"]["Aspect"] == "Perf"


def test_perf_stem_skips_pluperfect():
    # The pluperfect (Tense=Pqp) already encodes the perfect system and is
    # conventionally Aspect-less in the treebanks; the rule must not add Perf.
    tok = reconcile_token(
        "ὀρώρει",
        [verb("a", "ὄρνυμι", Person="3", Number="Sing", Tense="Pqp",
              Voice="Act", Mood="Ind", VerbForm="Fin")])
    assert tok.chosen["feats"].get("Aspect") != "Perf"
    assert "rule:perf-stem" not in methods(tok)


def test_perf_stem_gated_on_lemma():
    # Same surface substring but a different lemma -> rule must not fire.
    tok = reconcile_token(
        "λάθοντο",
        [verb("a", "ἄλλος", Person="3", Number="Plur", Tense="Past",
              Voice="Mid")])
    assert tok.chosen["feats"].get("Aspect") != "Perf"


def test_perf_stem_custom_table():
    tok = reconcile_token(
        "βέβηκε",
        [verb("a", "βαίνω", Person="3", Number="Sing", Tense="Pres",
              Voice="Act", Mood="Ind")],
        perf_stems={"βαίνω": ("βεβηκ",)})
    assert tok.chosen["feats"]["Aspect"] == "Perf"


# ---------------------------------------------------------------------------
# High-precision rule 4: aorist Mid,Pass -> Mid (the project voice convention)
# ---------------------------------------------------------------------------


def test_aorist_midpass_becomes_mid():
    # The aorist keeps the middle distinct from the -θη- passive, so an aorist
    # middle ending tagged the syncretic Mid,Pass must be Mid.
    for form, lemma in [("ἔρυτο", "ῥύομαι"), ("λωβήσασθε", "λωβάομαι"),
                        ("ἐπᾶλτο", "ἐφάλλομαι")]:
        tok = reconcile_token(
            form, [verb("a", lemma, Tense="Past", Aspect="Perf",
                        Voice="Mid,Pass", Mood="Ind")])
        assert tok.chosen["feats"]["Voice"] == "Mid", form
        assert "rule:aorist-mid-voice" in methods(tok), form


def test_aorist_mid_voice_not_on_present_or_imperfect():
    # present (syncretic) and imperfect (Past+Imp, syncretic) keep Mid,Pass.
    pres = reconcile_token(
        "λύεται", [verb("a", "λύω", Tense="Pres", Voice="Mid,Pass",
                        Person="3", Number="Sing", Mood="Ind")])
    assert pres.chosen["feats"]["Voice"] == "Mid,Pass"
    impf = reconcile_token(
        "ἐλύετο", [verb("a", "λύω", Tense="Past", Aspect="Imp",
                        Voice="Mid,Pass", Person="3", Number="Sing", Mood="Ind")])
    assert impf.chosen["feats"]["Voice"] == "Mid,Pass"
    assert "rule:aorist-mid-voice" not in methods(impf)


def test_aorist_mid_voice_not_on_theta_passive():
    # a -θη- aorist passive is not a middle; leave the voice alone.
    tok = reconcile_token(
        "ἐλύθη", [verb("a", "λύω", Tense="Past", Aspect="Perf",
                       Voice="Mid,Pass", Person="3", Number="Sing", Mood="Ind")])
    assert tok.chosen["feats"]["Voice"] == "Mid,Pass"
    assert "rule:aorist-mid-voice" not in methods(tok)


# ---------------------------------------------------------------------------
# Flaggers stay contested (never auto-apply)
# ---------------------------------------------------------------------------


def test_lemmatizer_disagreement_flags_not_applies():
    tok = reconcile_token(
        "ἤχθετο",
        [verb("a", "ἔχθω", Person="3", Number="Sing", Tense="Past",
              Voice="Mid", Mood="Ind")],
        lexicon_hints={"lemmatizer_lemma": "ἄχθομαι"})
    # the chosen lemma is NOT overwritten...
    assert tok.chosen["lemma"] == "ἔχθω"
    # ...but lemma is contested with the lemmatizer's suggestion recorded.
    assert "lemma" in tok.contested_fields
    assert "flag:lemmatizer-lemma" in methods(tok)


def test_lexicon_transitive_flags_voice():
    tok = reconcile_token(
        "ἐρωήσαιτ̓",
        [verb("a", "ἐρωέω", Person="3", Number="Sing", Voice="Mid",
              Mood="Opt")],
        lexicon_hints={"transitive_sense": True})
    assert tok.chosen["feats"]["Voice"] == "Mid"     # unchanged
    assert "Voice" in tok.contested_fields
    assert "flag:lexicon-transitive" in methods(tok)


def test_syntax_subject_number_flags():
    tok = reconcile_token(
        "ἐρωήσαιτ̓",
        [verb("a", "ἐρωέω", Person="3", Number="Sing", Voice="Act",
              Mood="Opt")],
        syntax_hints={"subject_number": "Dual"})
    assert tok.chosen["feats"]["Number"] == "Sing"   # unchanged
    assert "Number" in tok.contested_fields
    assert "flag:syntax-subj-number" in methods(tok)


# ---------------------------------------------------------------------------
# Adjudicator promotion gate (AND-gate)
# ---------------------------------------------------------------------------


def test_adjudicator_not_called_when_uncontested():
    calls = []

    def adj(tok):
        calls.append(tok)
        return {"changes": {"Voice": "Act"}, "confidence": 1.0}

    tok = reconcile_token(
        "ὑπέσχετο",
        [verb("a", "ὑπισχνέομαι", Person="3", Number="Sing", Voice="Mid",
              Tense="Past", Aspect="Perf", Mood="Ind")],
        adjudicator=adj)
    assert calls == []                # uncontested -> adjudicator untouched
    assert tok.contested is False


def test_adjudicator_promotes_only_on_agreement_with_vote():
    readings = [
        verb("a", "λύω", Person="3", Number="Sing", Voice="Mid"),
        verb("b", "λύω", Person="3", Number="Sing", Voice="Mid"),
        verb("c", "λύω", Person="3", Number="Plur", Voice="Mid"),
    ]
    # vote majority on Number is Sing; adjudicator agrees -> promote.
    tok = reconcile_token(
        "λύεται", readings,
        adjudicator=lambda t: {"changes": {"Number": "Sing"},
                               "confidence": 0.9})
    assert tok.chosen["feats"]["Number"] == "Sing"
    assert "Number" not in tok.contested_fields
    assert any(m.startswith("adjudicator+") for m in methods(tok))


def test_adjudicator_blocked_without_agreeing_signal():
    readings = [
        verb("a", "λύω", Person="3", Number="Sing", Voice="Mid"),
        verb("b", "λύω", Person="3", Number="Plur", Voice="Mid"),
    ]
    # Number ties (1 Sing / 1 Plur) so there is NO deterministic signal;
    # the adjudicator's Dual is not corroborated -> stays contested.
    tok = reconcile_token(
        "λύεται", readings,
        adjudicator=lambda t: {"changes": {"Number": "Dual"},
                               "confidence": 0.99})
    assert tok.chosen["feats"]["Number"] != "Dual"
    assert "Number" in tok.contested_fields
    assert "adjudicator:suggest" in methods(tok)


def test_adjudicator_promotes_on_agreement_with_lexicon_flag():
    # transitive-sense flag suggests Voice=Act; adjudicator agrees -> promote.
    tok = reconcile_token(
        "ἐρωήσαιτ̓",
        [verb("a", "ἐρωέω", Person="3", Number="Sing", Voice="Mid",
              Mood="Opt")],
        lexicon_hints={"transitive_sense": True},
        adjudicator=lambda t: {"changes": {"Voice": "Act"},
                               "confidence": 0.95})
    assert tok.chosen["feats"]["Voice"] == "Act"
    assert "Voice" not in tok.contested_fields


def test_adjudicator_error_is_swallowed():
    def boom(tok):
        raise RuntimeError("model down")

    # Number disagreement (Sing/Plur tie) keeps the token contested -- no
    # rule resolves it -- so the adjudicator is actually invoked.
    tok = reconcile_token(
        "λύεται",
        [verb("a", "λύω", Person="3", Number="Sing", Voice="Mid"),
         verb("b", "λύω", Person="3", Number="Plur", Voice="Mid")],
        adjudicator=boom)
    # the contested token survives; the failure is recorded as provenance.
    assert isinstance(tok, ReconciledToken)
    assert tok.contested is True
    assert "adjudicator:error" in methods(tok)


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------


def test_to_dict_shape_and_meta_passthrough():
    tok = reconcile_token(
        "λύεται",
        [verb("a", "λύω", Person="3", Number="Sing", Voice="Act"),
         verb("b", "λύω", Person="3", Number="Sing", Voice="Mid")],
        meta={"line": 57, "col": "ag", "idx": 6})
    d = tok.to_dict()
    assert d["form"] == "λύεται"
    assert set(d["chosen"]) == {"lemma", "upos", "feats"}
    assert isinstance(d["readings"], list) and len(d["readings"]) == 2
    assert d["readings"][0]["source"] == "a"
    assert isinstance(d["contested"], bool)
    assert isinstance(d["contested_fields"], list)
    assert all({"method", "evidence", "confidence"} <= set(p)
               for p in d["provenance"])
    # meta merged at top level (matches the data-contract example)
    assert d["line"] == 57 and d["col"] == "ag" and d["idx"] == 6


def test_reconcile_work_batches():
    toks = reconcile_work([
        {"form": "ὑπέσχετο",
         "readings": [verb("a", "ὑπισχνέομαι", Person="3", Number="Sing",
                           Voice="Act", Tense="Past", Mood="Ind")]},
        {"form": "λύει",
         "readings": [verb("a", "λύω", Person="3", Number="Sing",
                           Tense="Pres", Voice="Act", Mood="Ind")]},
    ])
    assert len(toks) == 2
    assert toks[0].chosen["feats"]["Voice"] == "Mid"   # rule fired
    assert toks[1].contested is False


def test_reading_objects_accepted_directly():
    tok = reconcile_token(
        "λύεται",
        [Reading("a", "λύω", "VERB", {"Voice": "Act"})])
    assert tok.chosen["feats"]["Voice"] == "Mid"       # mid-ending rule
    assert isinstance(tok.readings[0], Reading)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

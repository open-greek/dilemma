"""Tests for the paradigm orchestrator + minimal template fallback.

Run with:
    python -m pytest tests/test_paradigm.py -x -v

The tests in `TestJsonSourceResolution` and `TestFillCanonicalDict`
require paradigm JSON sources reachable via `$DILEMMA_PARADIGM_DATA`.
When the env var is unset, those tests skip; the slot-key, slot-grid,
and template-fallback tests still run on a clean checkout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dilemma.paradigm import (
    ParadigmForm,
    ParadigmSlot,
    ParadigmSource,
    fill_canonical_dict,
    generate,
    generate_paradigm,
    iter_slots,
    reset_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_jtauber_data() -> bool:
    """True iff jtauber_ag_paradigms.json is reachable through the
    `$DILEMMA_PARADIGM_DATA` env var.

    JSON-source assertions are gated behind this; the slot grid and
    template fallback tests don't need the file.
    """
    custom = os.environ.get("DILEMMA_PARADIGM_DATA")
    if custom and (Path(custom) / "jtauber_ag_paradigms.json").exists():
        return True
    return False


needs_jtauber = pytest.mark.skipif(
    not _has_jtauber_data(),
    reason="$DILEMMA_PARADIGM_DATA not configured with jtauber JSON; "
           "JSON-source tests skipped",
)


# ---------------------------------------------------------------------------
# Slot key shapes
# ---------------------------------------------------------------------------


class TestSlotKey:
    """Every slot shape emits the canonical inflection key shape."""

    def test_verb_finite_indicative(self):
        s = ParadigmSlot.verb_finite(
            voice="active", tense="aorist", mood="indicative",
            person="1", number="sg",
        )
        assert s.key == "active_aorist_indicative_1sg"

    def test_verb_finite_subjunctive(self):
        s = ParadigmSlot.verb_finite(
            voice="middle", tense="aorist", mood="subjunctive",
            person="3", number="pl",
        )
        assert s.key == "middle_aorist_subjunctive_3pl"

    def test_verb_infinitive(self):
        s = ParadigmSlot.verb_infinitive(voice="active", tense="aorist")
        assert s.key == "active_aorist_infinitive"

    def test_verb_participle(self):
        s = ParadigmSlot.verb_participle(
            voice="active", tense="aorist",
            case="nom", gender="m", number="sg",
        )
        assert s.key == "active_aorist_participle_nom_m_sg"

    def test_noun(self):
        assert ParadigmSlot.noun(case="genitive", number="pl").key == "genitive_pl"

    def test_adj(self):
        s = ParadigmSlot.adj(case="dative", gender="f", number="sg")
        assert s.key == "dative_f_sg"


# ---------------------------------------------------------------------------
# Slot grid coverage
# ---------------------------------------------------------------------------


class TestIterSlots:
    """The slot iterator should hit every canonical leaf shape exactly once."""

    def test_no_first_person_dual(self):
        # Greek has no 1du. The iterator must filter it out.
        for slot in iter_slots("verb"):
            if slot.person == "1" and slot.number == "du":
                pytest.fail(f"unexpected 1du slot: {slot.key}")

    def test_imperative_has_no_first_person(self):
        for slot in iter_slots("verb"):
            if slot.mood == "imperative":
                assert slot.person in ("2", "3"), (
                    f"imperative slot has non-2/3 person: {slot.key}"
                )

    def test_keys_unique(self):
        keys = [s.key for s in iter_slots("verb")]
        assert len(keys) == len(set(keys)), "verb slot keys are not unique"

    def test_noun_grid_size(self):
        # 5 cases × 3 numbers = 15
        slots = list(iter_slots("noun"))
        assert len(slots) == 15

    def test_adj_grid_size(self):
        # 5 cases × 3 genders × 3 numbers = 45
        slots = list(iter_slots("adj"))
        assert len(slots) == 45

    def test_unknown_pos_yields_empty(self):
        assert list(iter_slots("particle")) == []


# ---------------------------------------------------------------------------
# JSON-source resolution (jtauber > Morpheus > dilemma_corpus > template)
# ---------------------------------------------------------------------------


@needs_jtauber
class TestJsonSourceResolution:
    """When the JSONs are present, the precedence chain should fire."""

    def setup_method(self):
        # Each method gets a fresh source cache so $DILEMMA_PARADIGM_DATA
        # changes during the test class don't leak.
        reset_cache()

    def test_grafw_aorist_1sg_is_egrapsa(self):
        slot = ParadigmSlot.verb_finite(
            voice="active", tense="aorist", mood="indicative",
            person="1", number="sg",
        )
        f = generate("γράφω", slot)
        assert f is not None
        assert f.form == "ἔγραψα"
        assert f.source == ParadigmSource.JTAUBER.value

    def test_aiteo_present_active_1sg(self):
        # Contract verb: jtauber should still produce the contracted form
        # αἰτῶ from its template-built tables.
        slot = ParadigmSlot.verb_finite(
            voice="active", tense="present", mood="indicative",
            person="1", number="sg",
        )
        f = generate("αἰτέω", slot)
        assert f is not None
        assert f.form == "αἰτῶ"
        assert f.source == ParadigmSource.JTAUBER.value

    def test_ferw_aorist_1sg_is_enenka(self):
        # Suppletive aorist: must come from a JSON source, not a template.
        slot = ParadigmSlot.verb_finite(
            voice="active", tense="aorist", mood="indicative",
            person="1", number="sg",
        )
        f = generate("φέρω", slot)
        assert f is not None
        assert f.form == "ἤνεγκα"
        assert f.source == ParadigmSource.JTAUBER.value

    def test_str_returns_bare_form(self):
        slot = ParadigmSlot.verb_finite(
            voice="active", tense="aorist", mood="indicative",
            person="1", number="sg",
        )
        f = generate("γράφω", slot)
        assert str(f) == "ἔγραψα"

    def test_generate_paradigm_grafw_has_canonical_keys(self):
        p = generate_paradigm("γράφω", "verb")
        # Spot-check a few canonical keys jtauber definitely covers.
        assert "active_aorist_indicative_1sg" in p
        assert "active_present_indicative_1sg" in p
        assert "active_aorist_infinitive" in p
        assert "active_aorist_participle_nom_m_sg" in p
        assert all(isinstance(v, ParadigmForm) for v in p.values())

    def test_chora_noun_paradigm_via_dilemma(self):
        # χώρα isn't in Morpheus's ag_noun_paradigms.json (verified
        # empirically); dilemma_ag_noun_paradigms.json has it.
        p = generate_paradigm("χώρα", "noun")
        assert "nominative_sg" in p
        assert p["nominative_sg"].form == "χώρα"
        assert p["nominative_sg"].source == ParadigmSource.DILEMMA_CORPUS.value


# ---------------------------------------------------------------------------
# Template fallback
# ---------------------------------------------------------------------------


class TestTemplateFallback:
    """Templates fire when no JSON source has the cell.

    Tests pin the orchestrator at a fake data dir so JSON sources are
    empty; only the template path can produce forms.
    """

    def setup_method(self, tmp_path_factory=None):
        import os

        # Use an empty tmp dir as data root so all JSON loads return {}.
        # We re-set it per method via a context manager in each test.
        reset_cache()

    def _pin_empty_sources(self, tmp_path, monkeypatch):
        # Pin the orchestrator at an empty tmp dir so JSON sources
        # always miss; only the template path can produce forms.
        monkeypatch.setattr(
            "dilemma.paradigm._candidate_data_dirs",
            lambda: [tmp_path],
        )
        reset_cache()

    def test_thematic_omega_present_indicative_active(self, tmp_path, monkeypatch):
        self._pin_empty_sources(tmp_path, monkeypatch)
        slot = ParadigmSlot.verb_finite(
            voice="active", tense="present", mood="indicative",
            person="1", number="sg",
        )
        f = generate("γράφω", slot)
        assert f is not None
        assert f.form == "γράφω"
        assert f.source == ParadigmSource.TEMPLATE.value

    def test_thematic_omega_present_indicative_2pl(self, tmp_path, monkeypatch):
        self._pin_empty_sources(tmp_path, monkeypatch)
        slot = ParadigmSlot.verb_finite(
            voice="active", tense="present", mood="indicative",
            person="2", number="pl",
        )
        f = generate("γράφω", slot)
        assert f is not None
        assert f.form == "γράφετε"
        assert f.source == ParadigmSource.TEMPLATE.value

    def test_thematic_omega_present_infinitive(self, tmp_path, monkeypatch):
        self._pin_empty_sources(tmp_path, monkeypatch)
        slot = ParadigmSlot.verb_infinitive(voice="active", tense="present")
        f = generate("γράφω", slot)
        assert f is not None
        assert f.form == "γράφειν"

    def test_contract_verb_returns_none(self, tmp_path, monkeypatch):
        # αἰτέω is a contract verb (-έω); its present 1sg is αἰτῶ, not
        # αἰτέω. Our template doesn't handle the contraction, so it
        # must return None rather than emit a wrong form.
        self._pin_empty_sources(tmp_path, monkeypatch)
        slot = ParadigmSlot.verb_finite(
            voice="active", tense="present", mood="indicative",
            person="1", number="sg",
        )
        assert generate("αἰτέω", slot) is None

    def test_athematic_mi_verb_returns_none(self, tmp_path, monkeypatch):
        self._pin_empty_sources(tmp_path, monkeypatch)
        # δίδωμι is athematic; templates don't synthesise μι-verbs.
        slot = ParadigmSlot.verb_finite(
            voice="active", tense="present", mood="indicative",
            person="1", number="sg",
        )
        assert generate("δίδωμι", slot) is None

    def test_aorist_returns_none_for_template(self, tmp_path, monkeypatch):
        # Aorist stems are unpredictable from a present-tense lemma.
        # Template must decline to synthesise.
        self._pin_empty_sources(tmp_path, monkeypatch)
        slot = ParadigmSlot.verb_finite(
            voice="active", tense="aorist", mood="indicative",
            person="1", number="sg",
        )
        assert generate("γράφω", slot) is None

    def test_future_perfect_returns_none(self, tmp_path, monkeypatch):
        # Even for a regular verb, future_perfect is outside template scope.
        self._pin_empty_sources(tmp_path, monkeypatch)
        slot = ParadigmSlot.verb_finite(
            voice="active", tense="future_perfect", mood="indicative",
            person="1", number="sg",
        )
        assert generate("γράφω", slot) is None

    def test_first_decl_long_a_noun(self, tmp_path, monkeypatch):
        self._pin_empty_sources(tmp_path, monkeypatch)
        nom_sg = generate("χώρα", ParadigmSlot.noun(case="nominative", number="sg"))
        gen_sg = generate("χώρα", ParadigmSlot.noun(case="genitive", number="sg"))
        dat_sg = generate("χώρα", ParadigmSlot.noun(case="dative", number="sg"))
        acc_sg = generate("χώρα", ParadigmSlot.noun(case="accusative", number="sg"))
        assert nom_sg is not None and nom_sg.form == "χώρα"
        assert gen_sg is not None and gen_sg.form == "χώρας"
        assert dat_sg is not None and dat_sg.form == "χώρᾳ"
        assert acc_sg is not None and acc_sg.form == "χώραν"
        for f in (nom_sg, gen_sg, dat_sg, acc_sg):
            assert f.source == ParadigmSource.TEMPLATE.value

    def test_second_decl_os_adj(self, tmp_path, monkeypatch):
        self._pin_empty_sources(tmp_path, monkeypatch)
        nom_m = generate("καλός", ParadigmSlot.adj(
            case="nominative", gender="m", number="sg",
        ))
        nom_f = generate("καλός", ParadigmSlot.adj(
            case="nominative", gender="f", number="sg",
        ))
        nom_n = generate("καλός", ParadigmSlot.adj(
            case="nominative", gender="n", number="sg",
        ))
        # Citation form is the lemma verbatim (preserves accent).
        assert nom_m is not None and nom_m.form == "καλός"
        # Derived feminine / neuter: the template doesn't compute
        # accent shifts, so we only check the diacritic-stripped
        # base. The actual classical forms are καλή / καλόν; our
        # template emits the right consonant-and-vowel skeleton but
        # may not place the accent on the same syllable as the
        # source-derived form.
        import unicodedata
        def strip_acc(s):
            nfd = unicodedata.normalize("NFD", s)
            return "".join(c for c in nfd if not unicodedata.combining(c))
        assert nom_f is not None and strip_acc(nom_f.form) == "καλη"
        assert nom_n is not None and strip_acc(nom_n.form) == "καλον"

    def test_third_decl_noun_returns_none(self, tmp_path, monkeypatch):
        # γέρων (3rd-decl consonant stem) shouldn't get a template.
        self._pin_empty_sources(tmp_path, monkeypatch)
        slot = ParadigmSlot.noun(case="genitive", number="sg")
        assert generate("γέρων", slot) is None

    def test_allow_template_false_skips_template(self, tmp_path, monkeypatch):
        # Even when the template would fire, allow_template=False
        # opts out and the orchestrator returns None.
        self._pin_empty_sources(tmp_path, monkeypatch)
        slot = ParadigmSlot.verb_finite(
            voice="active", tense="present", mood="indicative",
            person="1", number="sg",
        )
        assert generate("γράφω", slot, allow_template=False) is None


# ---------------------------------------------------------------------------
# fill_canonical_dict
# ---------------------------------------------------------------------------


@needs_jtauber
class TestFillCanonicalDict:
    """The build-pipeline-facing helper should fill missing cells idempotently."""

    def setup_method(self):
        reset_cache()

    def test_fill_adds_missing_cells(self):
        # Start from an empty inflections map for γράφω; the helper
        # should populate it with everything the JSON sources have.
        canonical = {
            "γ/γράφω.yml": {
                "lemma": "γράφω",
                "pos": "verb",
                "inflections": {"attic": {}},
            },
        }
        filled, stats = fill_canonical_dict(
            canonical, allow_template=False, dialect="attic",
        )
        attic = filled["γ/γράφω.yml"]["inflections"]["attic"]
        sources = filled["γ/γράφω.yml"]["inflections_source"]["attic"]
        assert "active_aorist_indicative_1sg" in attic
        assert attic["active_aorist_indicative_1sg"] == "ἔγραψα"
        assert sources["active_aorist_indicative_1sg"] == "jtauber"
        assert stats["lemmas_touched"] == 1
        assert stats["jtauber"] > 0

    def test_fill_preserves_existing_cells(self):
        # An existing cell with a custom value (e.g. from kaikki) must
        # NOT be overwritten by jtauber. The fill should only add
        # missing keys.
        canonical = {
            "γ/γράφω.yml": {
                "lemma": "γράφω",
                "pos": "verb",
                "inflections": {"attic": {
                    "active_aorist_indicative_1sg": "ZZZ_KEEP_ME",
                }},
            },
        }
        filled, _ = fill_canonical_dict(canonical, allow_template=False)
        attic = filled["γ/γράφω.yml"]["inflections"]["attic"]
        assert attic["active_aorist_indicative_1sg"] == "ZZZ_KEEP_ME"
        # And the sidecar source map mustn't claim this cell came
        # from a generator: only added cells get recorded.
        sources = filled["γ/γράφω.yml"].get("inflections_source", {}).get("attic", {})
        assert "active_aorist_indicative_1sg" not in sources

    def test_fill_is_idempotent(self):
        # Running fill twice on the same dict should produce a
        # byte-identical JSON serialisation. The build relies on this
        # to skip rewriting unchanged YAMLs.
        canonical = {
            "γ/γράφω.yml": {
                "lemma": "γράφω",
                "pos": "verb",
                "inflections": {"attic": {}},
            },
        }
        once, _ = fill_canonical_dict(
            json.loads(json.dumps(canonical)), allow_template=False,
        )
        twice_input = json.loads(json.dumps(once))
        twice, _ = fill_canonical_dict(twice_input, allow_template=False)
        assert json.dumps(once, sort_keys=True, ensure_ascii=False) == \
            json.dumps(twice, sort_keys=True, ensure_ascii=False)

    def test_fill_skips_non_inflectable_pos(self):
        canonical = {
            "x.yml": {
                "lemma": "καί",
                "pos": "conj",
                "inflections": {"attic": {}},
            },
        }
        filled, stats = fill_canonical_dict(canonical, allow_template=False)
        # Non-inflectable POS gets passed through untouched.
        assert stats["lemmas_touched"] == 0
        assert "inflections_source" not in filled["x.yml"]
        assert filled["x.yml"]["inflections"] == {"attic": {}}


# ---------------------------------------------------------------------------
# End-to-end determinism on a canonical-shaped input
# ---------------------------------------------------------------------------


class TestFillDeterminismTemplateOnly:
    """Determinism guarantees that hold on a clean checkout (no JSON sources).

    These tests run with the JSON sources empty so only the template
    fallback path is exercised. They guard against regressions where
    someone adds a `set()` or `defaultdict` whose iteration order
    leaks into the output even when no paradigm JSONs are present.
    """

    def setup_method(self):
        reset_cache()

    def _pin_empty_sources(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "dilemma.paradigm._candidate_data_dirs",
            lambda: [tmp_path],
        )
        reset_cache()

    @staticmethod
    def _template_only_canonical():
        # All lemmas the template fallback can fill from scratch:
        # plain thematic -ω verb (γράφω -> present-system active),
        # 1st-decl long-α noun (χώρα), 1st-decl η noun (τέχνη),
        # 2nd-decl -ος noun (λόγος), 2-1-2 -ος adj (καλός).
        return {
            "g/γράφω.yml": {"lemma": "γράφω", "pos": "verb"},
            "x/χώρα.yml": {"lemma": "χώρα", "pos": "noun"},
            "t/τέχνη.yml": {"lemma": "τέχνη", "pos": "noun"},
            "l/λόγος.yml": {"lemma": "λόγος", "pos": "noun"},
            "k/καλός.yml": {"lemma": "καλός", "pos": "adj"},
        }

    def test_template_fill_byte_identical(self, tmp_path, monkeypatch):
        self._pin_empty_sources(tmp_path, monkeypatch)
        canonical = self._template_only_canonical()
        first = json.dumps(
            fill_canonical_dict(
                json.loads(json.dumps(canonical, ensure_ascii=False)),
                allow_template=True, dialect="attic",
            )[0],
            ensure_ascii=False,
        )
        # Second invocation: separate input copy, fresh empty source cache.
        reset_cache()
        second = json.dumps(
            fill_canonical_dict(
                json.loads(json.dumps(canonical, ensure_ascii=False)),
                allow_template=True, dialect="attic",
            )[0],
            ensure_ascii=False,
        )
        assert first == second


@needs_jtauber
class TestFillDeterminism:
    """Two fills of the same canonical input must produce byte-identical JSON.

    The downstream build pipeline relies on this: it writes the
    `pending_canonical` hash to a tempfile, runs `python -m dilemma
    paradigm fill`, parses the output, then later compares each
    candidate YAML against any existing on-disk file and skips
    unchanged ones. If the fill output drifts between runs (e.g. dict
    iteration order, set iteration leaking into output, hash-randomised
    set membership), every YAML gets rewritten and downstream caches
    invalidate. The test below builds a representative multi-POS
    canonical-shape input -- mixing verbs, nouns, adjectives, pre-
    populated and empty inflection maps -- then asserts byte-for-byte
    equality of `json.dumps` (NOT `sort_keys=True`) across two
    independent fill invocations.
    """

    def setup_method(self):
        reset_cache()

    @staticmethod
    def _representative_canonical():
        # Mixed corpus of lemmas hitting each precedence-chain branch
        # we ship: jtauber (γράφω), Morpheus (ag_noun_paradigms covers
        # ζηλωτός as adj), dilemma_corpus (χώρα is in
        # dilemma_ag_noun_paradigms), and a non-inflectable POS
        # entry (καί) that must pass through untouched. Each entry is
        # given a distinct shape so the full fill_canonical_dict
        # branches (no inflections, empty attic, partial attic, sources
        # block, additional_pos passthrough) all execute.
        return {
            "g/γράφω.yml": {
                "lemma": "γράφω",
                "pos": "verb",
                # Empty inflections on the verb -> fill populates from jtauber.
                "inflections": {"attic": {}},
            },
            "f/φέρω.yml": {
                "lemma": "φέρω",
                "pos": "verb",
                # Pre-populated cells: fill must not overwrite the user
                # value AND must add missing cells. Tests the "skip
                # existing" branch.
                "inflections": {
                    "attic": {
                        "active_present_indicative_1sg": "ZZZ_KEEP_ME",
                    },
                },
            },
            "x/χώρα.yml": {
                "lemma": "χώρα",
                "pos": "noun",
                # No inflections key at all -> fill must create the
                # nested {attic: {}} and populate from dilemma noun
                # source.
            },
            "z/ζηλωτός.yml": {
                "lemma": "ζηλωτός",
                "pos": "adj",
                # adj path through generate(): exercises the noun-fallback
                # order (Morpheus first, dilemma_corpus second).
                "inflections": {"attic": {}},
            },
            "k/καί.yml": {
                "lemma": "καί",
                "pos": "conj",
                # Non-inflectable: fill must leave it byte-identical.
                "inflections": {"attic": {}},
            },
            "g/γέρων.yml": {
                "lemma": "γέρων",
                "pos": "noun",
                # 3rd-decl consonant stem: dilemma_corpus may have
                # partial coverage; ensures we exercise lemmas where
                # only some cells are filled. No template fallback.
                "inflections": {"attic": {}},
            },
        }

    def _run_fill_via_api(self, canonical_in):
        # Use json roundtrip to mimic the on-disk read the CLI does.
        # The bug (if any) lives downstream of json.loads, so the
        # roundtrip ensures we measure the same code path.
        data = json.loads(json.dumps(canonical_in, ensure_ascii=False))
        filled, _ = fill_canonical_dict(data, allow_template=False, dialect="attic")
        return json.dumps(filled, ensure_ascii=False)

    def test_two_consecutive_fills_byte_identical(self):
        # Core regression: same input, two fill invocations, identical
        # bytes. Without sort_keys -- so any drift in dict iteration
        # order surfaces immediately.
        canonical = self._representative_canonical()
        first = self._run_fill_via_api(canonical)
        second = self._run_fill_via_api(canonical)
        assert first == second, (
            "fill_canonical_dict produced non-byte-identical output "
            "across two consecutive invocations on the same input. "
            "This breaks the downstream compare-and-skip write phase and "
            "indicates a non-deterministic ordering somewhere in the "
            "fill chain (set iteration, dict-from-set, randomised hash, ...)."
        )

    def test_fill_then_refill_byte_identical(self):
        # Idempotence: filling once then filling the result again must
        # yield byte-identical output. Distinct from the test above
        # because some bugs only manifest when a second pass over the
        # already-filled cells is requested.
        canonical = self._representative_canonical()
        once = self._run_fill_via_api(canonical)
        twice_input = json.loads(once)
        twice = json.dumps(twice_input, ensure_ascii=False)
        # First sanity: parsing+re-dumping shouldn't move bytes.
        assert once == twice
        # Now run fill on the already-filled output: no new cells
        # should appear, byte-identical guaranteed.
        twice_data = json.loads(once)
        filled, stats = fill_canonical_dict(
            twice_data, allow_template=False, dialect="attic",
        )
        third = json.dumps(filled, ensure_ascii=False)
        assert once == third
        # And the stats should reflect zero new fills (every cell the
        # sources covered was filled in pass one).
        assert stats["lemmas_touched"] == 0

    def test_input_dict_order_does_not_change_per_lemma_cells(self):
        # When the *input* dict is reordered (different insertion order
        # for the outer {filepath: entry} keys), the resulting cells
        # PER LEMMA must be the same set with the same values. Output
        # bytes will differ because the outer JSON encodes input key
        # order, but per-entry semantics must not.
        canonical = self._representative_canonical()
        keys = list(canonical.keys())
        reversed_canonical = {k: canonical[k] for k in reversed(keys)}
        a, _ = fill_canonical_dict(
            json.loads(json.dumps(canonical, ensure_ascii=False)),
            allow_template=False, dialect="attic",
        )
        b, _ = fill_canonical_dict(
            json.loads(json.dumps(reversed_canonical, ensure_ascii=False)),
            allow_template=False, dialect="attic",
        )
        assert set(a.keys()) == set(b.keys())
        for k in a:
            a_attic = a[k].get("inflections", {}).get("attic", {})
            b_attic = b[k].get("inflections", {}).get("attic", {})
            # Same cell set, same forms.
            assert a_attic == b_attic, (
                f"per-lemma cells differ for {k!r} when input order "
                f"changes: only-in-A={set(a_attic) - set(b_attic)}, "
                f"only-in-B={set(b_attic) - set(a_attic)}"
            )
            # And the source attribution must match too.
            a_src = a[k].get("inflections_source", {}).get("attic", {})
            b_src = b[k].get("inflections_source", {}).get("attic", {})
            assert a_src == b_src

    def test_fill_via_cli_subprocess_byte_identical(self):
        # End-to-end: run the actual `python -m dilemma paradigm fill`
        # CLI in two separate processes (so any fork-time
        # non-determinism, e.g. PYTHONHASHSEED, surfaces) and assert
        # byte-identical output files. The downstream build calls dilemma
        # exactly this way; this test catches CLI-only regressions
        # the in-process tests above might miss.
        import subprocess
        import tempfile
        canonical = self._representative_canonical()
        with tempfile.TemporaryDirectory() as td:
            in_path = Path(td) / "in.json"
            out_a = Path(td) / "out_a.json"
            out_b = Path(td) / "out_b.json"
            in_path.write_text(json.dumps(canonical, ensure_ascii=False))
            for out_path in (out_a, out_b):
                env = dict(os.environ)
                # Honour caller's $DILEMMA_PARADIGM_DATA; otherwise
                # the CLI loads no sources and the test is meaningless.
                # @needs_jtauber decorator already gates on this var.
                proc = subprocess.run(
                    [
                        "python3", "-m", "dilemma", "paradigm", "fill",
                        "--in", str(in_path),
                        "--out", str(out_path),
                        "--dialect", "attic",
                    ],
                    env=env, capture_output=True, text=True, check=False,
                )
                assert proc.returncode == 0, (
                    f"CLI failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
                )
            assert out_a.read_bytes() == out_b.read_bytes(), (
                "two consecutive `python -m dilemma paradigm fill` "
                "subprocess invocations on the same input produced "
                "different bytes; this is the exact regression the "
                "downstream canonical builder consumes."
            )

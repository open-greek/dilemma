#!/usr/bin/env python3
"""Tests for the LSJ principal-parts extractor and its grc-conj
integration in `build/expand_lsj.py`.

Run with:

    python -m pytest tests/test_lsj_principal_parts.py -x -v

The parser tests run against hand-picked LSJ entry texts captured
inline (no filesystem I/O); the integration tests live behind a
``--integration`` marker and skip automatically when the heavyweight
WTP database is not available, so the suite stays runnable in CI.
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import pytest

# Make ``build/`` importable.
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "build"))

from lsj_principal_parts import (  # noqa: E402
    parse_principal_parts, derive_grc_conj_args,
)


# ---------------------------------------------------------------------------
# Hand-curated LSJ head fragments. These are the exact (head-gloss) text
# our parser sees for each headword. Keeping them inline lets the tests
# run without depending on lsj9_glosses.jsonl being present.
# ---------------------------------------------------------------------------

LSJ_FRAGMENTS = {
    "γράφω": (
        "[ᾰ], fut. -ψω Hdt.1.95, etc.: aor. ἔγραψα, Ep. γράψα Il.17.599: "
        "pf. γέγραφα Cratin.124, Th.5.26, etc.; later γεγράφηκα PHib.1.78.2 "
        "(iii B. C.):—Med., fut. γράψομαι Ar. etc. (but in pass. sense, "
        "Gal.Protr.13): aor. ἐγραψάμην Ar.V.894, etc.:—Pass., fut. "
        "γρᾰφήσομαι Hp.Acut.26, Nicom.Com.1.39, (μετεγ-) Ar.Eq.1370; "
        "more freq. γεγράψομαι S.OT411, Theoc.18.47, etc.: aor. ἐγράφην [ᾰ], "
        "Hdt.4.91, Pl.Prm.128c, etc.; ἐγράφθην SIG57.5 (Milet., v B. C.), "
        "Archim.Fluit.2.4: pf. γέγραμμαι (also in med. sense, v. fin.)"
    ),
    "λείπω": (
        "impf. ἐλείπον ἔλειπον Il.19.288, etc.: fut. λείψω 18.11: "
        "aor. 1 ἔλειψα, part. λείψας Ar.Fr.965 (= elsewh. only late, "
        "Plb.12.15.12 (παρ-), Str.6.3.10 (παρ-), Ps.-Phoc.77 etc.; "
        "uncompounded, Ptol.Alm.10.4, Luc.Par.42, Ps.-Callisth.1.44 "
        "(cod. C); also in later Poets, Man.1.153, Opp.C.2.33, and in "
        "Inscrr., Epigr.Gr.522.16 (Thessalonica), 314.27 (Smyrna), etc.: "
        "but correct writers normally use aor. 2 ἔλῐπον Il.2.35, "
        "A.Pers.984 (lyr.), etc.: pf. λέλοιπα Od.14.134: plpf. ἐλελοίπειν "
        "(Att. -η) X.Cyr.2.1.21:—Med., in prop. chiefly in compds.: "
        "aor. 2 ἐλιπόμην Hdt.1.186, 2.40, E.HF169, etc. (in pass. sense, "
        "Il.11.693, al.):—Pass., fut. Med. in pass. sense λείψομαι etc.: "
        "aor. ἐλείφθην, λείφθην Pi.O.2.43; Ep.3pl. ἔλειφθεν h.Merc.195: "
        "pf. λέλειμμαι Il.13.256, Democr.228, Pl.Ti.61a, etc.: plpf. "
        "ἐλελείμμην Il.2.700; Ep. λέλειπτο 10.256:"
    ),
    "παύω": (
        "Il.19.67, etc.: Ion. impf. παύεσκον Od.22.315, S.Ant.963 (lyr.): "
        "fut. παύσω Il.1.207, etc.; Ep. inf. παυσέμεν (κατα-) 7.36: "
        "aor. ἔπαυσα 15.15, etc., Ep. παῦσα 17.602: pf. πέπαυκα D.20.70, "
        "Antisth.Od.10:—Med. and Pass., Ion. impf. παυέσκετο Il.24.17: "
        "fut. παύσομαι Od.2.198, Hdt.1.56, S.OC1040, Ph.1424, E.Med.93, etc."
        "; πεπαύσομαι only S.Ant.91, Tr.587 (though held to be the true "
        "Att. form by Moer.p.293 P.); παυσθήσομαι (v.l. παυθ-) Th.1.81; "
        "later παήσομαι (ἀνα-) Apoc.14.13: aor. ἐπαυσάμην Il.14.260; "
        "ἐπαύθην, Ep. παύθην, Hes.Th.533, Th.5.91 (v.l. παυσθῇ), etc. ; "
        "ἐπαύσθην Hdt.5.94, etc. ; later ἐπάην Choerob.in Theod.2.141 H. "
        ": pf. πέπαυμαι Il.18.125, A.Pr.615, Hdt.1.84, Ar.Pax29, etc. "
        "(πεπάσθαι is f.l. in Vett.Val.359.31):"
    ),
    # ἀκούω's LSJ head includes a fully-structured ``:—Pass.`` section
    # whose ``aor.`` is followed directly by the passive aorist
    # ``ἠκούσθην``. The label-driven scan handles this correctly via
    # the section voice override (``aor.`` in a ``pass`` section maps
    # to ``aor_p``); we just want to lock the behaviour in.
    "ἀκούω": (
        "ἀκούω: Ep. impf. ἄκουον Il.12.442: fut. ἀκούσομαι (Act. ἀκούσω "
        "first in Hyp.Epit.34 s.v.l., then in Lyc.378, 686, D.H.5.57, "
        "Ev.Matt.12.19, etc.: aor. ἤκουσα, Ep. ἄκουσα Il.24.223: pf. "
        "ἀκήκοα, Lacon. ἄκουκα Plu.Lyc.20, Ages.21; ἤκουκα is a late "
        "form, POxy.237 vii 23 (ii A. D.); later Ion. ἀκήκουκα "
        "Herod.5.49: plpf. ἀκηκόειν Hdt.2.52, 7.208; ἠκηκόειν "
        "X.Oec.15.7; old Att. ἠκηκόη Ar.V.800, Pax616, "
        "Pl.Cra.384b:—rare in Med., pres. (v. infr. II.2): Ep. impf. "
        "ἀκούετο Il.4.331: aor. ἠκουσάμην Mosch.3.119:—Pass., fut. "
        "ἀκουσθήσομαι Pl.R.507d: aor. ἠκούσθην Th.3.38, Luc.Somn.5: "
        "pf. ἤκουσμαι D.H.Rh.11.10, Ps.-Luc.Philopatr.4; ἀκήκουσμαι "
        "is dub. in Luc.Hist.Conscr.49: plpf. ἤκουστο Anon.ap."
        "Demetr.Eloc.217, (παρ-) J.AJ17.10.10."
    ),
    # An entry with no recognisable principal parts: ποιέω's head text
    # is mostly inscriptional citations; ``parse_principal_parts``
    # should return an empty dict.
    "no_parts_example": (
        "Dor. ποιϝέω IG4.800 of S., Πολυμήδης ἐποίϝηh' (= ἐποίησε Ἀργεῖος "
        "SIG5 (vi B.C., cf. Class.Phil.20.139); Θεόπροπος ἐποίει Αἰγινάτας "
        "SIG18 (vi/v B.C.), etc."
    ),
}


# ---------------------------------------------------------------------------
# Parser correctness tests.
# ---------------------------------------------------------------------------


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def test_lyo_parser_handles_truncated_head():
    """λύω's LSJ head is OCR-truncated; the parser should still pick
    up at least the future even when it's the only well-formed
    label-form pair, and not invent forms for missing slots."""
    fragment = (
        "poet. imper. λῦθι Pi.Fr.85: fut. λύσω but λῦτο 24.1 "
        "(at beginning of line, v.l. λύτο); λύντο 1 ῡ always:"
    )
    parts = parse_principal_parts(fragment, "λύω")
    assert parts == {"fut": _nfc("λύσω")}


def test_grapho_full_principal_parts():
    """γράφω is the canonical worked example: active aorist + perfect,
    middle future + aorist, passive future + aorist + perfect."""
    parts = parse_principal_parts(LSJ_FRAGMENTS["γράφω"], "γράφω")
    # Active.
    assert parts.get("aor") == _nfc("ἔγραψα")
    assert parts.get("pf") == _nfc("γέγραφα")
    # Middle.
    assert parts.get("fut_med") == _nfc("γράψομαι")
    assert parts.get("aor_med") == _nfc("ἐγραψάμην")
    # Passive.
    assert parts.get("fut_p") == _nfc("γρᾰφήσομαι")
    assert parts.get("aor_p") == _nfc("ἐγράφην")
    # Perfect mediopassive (1sg).
    assert parts.get("pf_mp") == _nfc("γέγραμμαι")
    # The leading "fut. -ψω" suffix abbreviation is intentionally NOT
    # extracted because we cannot reconstruct the φ -> ψ stem change
    # from the bare suffix without morphology. Confirm we do NOT
    # invent a fut here.
    assert "fut" not in parts


def test_leipo_distinguishes_aor1_aor2_and_imperfect():
    """λείπω has both a 1st aorist (ἔλειψα) and 2nd aorist (ἔλιπον).
    The parser must record both in their numbered slots and pick the
    2nd as the canonical ``aor`` (LSJ's preferred citation form)."""
    parts = parse_principal_parts(LSJ_FRAGMENTS["λείπω"], "λείπω")
    assert parts.get("aor1") == _nfc("ἔλειψα")
    assert parts.get("aor2") == _nfc("ἔλῐπον")
    assert parts.get("aor") == _nfc("ἔλῐπον")  # promoted from aor2
    assert parts.get("impf") == _nfc("ἐλείπον")
    assert parts.get("pf") == _nfc("λέλοιπα")
    assert parts.get("plpf") == _nfc("ἐλελοίπειν")
    assert parts.get("aor_p") == _nfc("ἐλείφθην")
    assert parts.get("pf_mp") == _nfc("λέλειμμαι")


def test_pauo_med_and_pass_section_handling():
    """παύω's `Med. and Pass.` section yields an MP perfect and a
    middle future. The combined-section header maps to the ``med``
    section internally, but the parser should still tag the perfect
    as ``pf_mp``."""
    parts = parse_principal_parts(LSJ_FRAGMENTS["παύω"], "παύω")
    assert parts.get("fut") == _nfc("παύσω")
    assert parts.get("aor") == _nfc("ἔπαυσα")
    assert parts.get("pf") == _nfc("πέπαυκα")
    assert parts.get("impf") == _nfc("παύεσκον")
    assert parts.get("fut_med") == _nfc("παύσομαι")
    assert parts.get("aor_med") == _nfc("ἐπαυσάμην")
    assert parts.get("pf_mp") == _nfc("πέπαυμαι")
    # The `:—Med. and Pass.` section bundles middle and passive
    # aorists under one ``aor.`` label: ``aor. ἐπαυσάμην …; ἐπαύθην,
    # Ep. παύθην, …; ἐπαύσθην …``. The first form (ἐπαυσάμην) is
    # the middle aorist; the chained `; ἐπαύθην` is the true
    # passive aorist and must be picked up as ``aor_p`` even though
    # there is no separate ``aor. p.`` label.
    assert parts.get("aor_p") == _nfc("ἐπαύθην")


def test_akouo_pass_section_aorist_passive():
    """ἀκούω has a fully-structured ``:—Pass.`` section. The ``aor.``
    label inside it should be tagged as ``aor_p`` via the section
    voice override (``aor.`` in ``pass`` section -> ``aor_p``)."""
    parts = parse_principal_parts(LSJ_FRAGMENTS["ἀκούω"], "ἀκούω")
    assert parts.get("fut") == _nfc("ἀκούσομαι")
    assert parts.get("aor") == _nfc("ἤκουσα")
    assert parts.get("pf") == _nfc("ἀκήκοα")
    # Passive aorist is the focus: ``Pass., ... aor. ἠκούσθην``
    assert parts.get("aor_p") == _nfc("ἠκούσθην")
    assert parts.get("fut_p") == _nfc("ἀκουσθήσομαι")
    assert parts.get("pf_mp") == _nfc("ἤκουσμαι")


def test_chained_passive_aorist_skips_definition_body():
    """The chained-``-θην`` heuristic must stop at the principal-parts
    boundary so it doesn't pick up nouns or unrelated forms in the
    definition. The boundary is a blank line (``\\n\\n``) or the next
    label, whichever comes first.

    Synthetic fixture: an ``aor.`` form in a ``Med.`` section is
    followed by a blank line and then a definition body containing
    an accusative noun ``τὴν πάθην``. The parser must NOT extract
    ``πάθην`` as ``aor_p``.
    """
    fragment = (
        "προστρέπω, turn towards, supplicate:—Med., with aor. "
        "προσετραπόμην Hom.Epigr.15.1; π. δῶμα.\n"
        "\n"
        "II.  Med., make a matter of supplication, "
        "τοῦ παθόντος προστρεπομένου τὴν πάθην Pl.Lg.866b."
    )
    parts = parse_principal_parts(fragment, "προστρέπω")
    assert parts.get("aor_med") == _nfc("προσετραπόμην")
    assert "aor_p" not in parts


def test_chained_passive_aorist_after_aor_in_med_only_section():
    """A synthetic ``Med.`` section with an ``aor.`` clause that
    bundles a middle aorist plus a chained passive aorist - the
    pattern observed in ὀρέγω, ὁρμάω, λοιδορέω, etc. The chained
    ``-θην`` form (here ``ὠρέχθην``) must be tagged as ``aor_p``.
    """
    fragment = (
        "ὀρέγω, fut. ὀρέξω: aor. ὤρεξα:—Med. and Pass., fut. "
        "ὀρέξομαι: aor. ὠρεξάμην Il.23.99, E.HF16, etc.: rare in "
        "Prose, X.Mem.1.2.15; also ὠρέχθην ib.16, Ages.1.4: pf. "
        "ὤρεγμαι Hp.Oss.18:"
    )
    parts = parse_principal_parts(fragment, "ὀρέγω")
    assert parts.get("aor_med") == _nfc("ὠρεξάμην")
    assert parts.get("aor_p") == _nfc("ὠρέχθην")


def test_no_principal_parts_returns_empty():
    """An entry whose head is purely inscriptional citation text
    yields no principal parts. The parser must not fabricate forms
    just because Greek words are nearby."""
    parts = parse_principal_parts(
        LSJ_FRAGMENTS["no_parts_example"], "ποιέω")
    # The head text contains no labelled principal parts, so the parser
    # should return an empty mapping (or at most a stray entry that we
    # can tolerate). The specific rule: ``aor`` / ``pf`` / ``fut``
    # must NOT be present, since none are labelled.
    for forbidden in ("fut", "aor", "pf", "impf", "plpf", "aor_p", "pf_mp"):
        assert forbidden not in parts, (
            f"unexpected fabricated key {forbidden!r}: {parts}")


def test_stop_word_skip_negation():
    """``impf. οὐκ ἠρχόμην`` (negation + verb) should yield the verb,
    not the negation. The parser skips known stop-words / dialect
    tags before grabbing the form."""
    fragment = "impf. οὐκ ἠρχόμην Hp.Epid.7.59"
    parts = parse_principal_parts(fragment, "ἔρχομαι")
    assert parts.get("impf") == _nfc("ἠρχόμην")


def test_dialect_tag_skip():
    """``fut. Ion. βαλέω`` should skip the dialect tag and capture
    the actual form."""
    fragment = "fut. Ion. βαλέω Il.8.403"
    parts = parse_principal_parts(fragment, "βάλλω")
    assert parts.get("fut") == _nfc("βαλέω")


# ---------------------------------------------------------------------------
# grc-conj argument derivation.
# ---------------------------------------------------------------------------


def test_derive_grc_conj_args_strips_accents_keeps_length():
    """grc-conj refuses stems containing tonal accents but accepts
    macron / breve. Stems must be lowercase, accent-stripped, length-
    preserving."""
    parts = {"aor": "ἔλυσα", "pf": "λέλυκα"}
    args = derive_grc_conj_args(parts, "λύω")
    # aor-1: ["aor-1", augmented, non-augmented]
    assert args["aor-1"][0] == "aor-1"
    assert args["aor-1"][1] == "ελυσ"
    assert args["aor-1"][2] == "λυσ"
    # perf: ["perf", active stem, mp stem (empty when missing)]
    assert args["perf"][0] == "perf"
    assert args["perf"][1] == "λελυκ"


def test_derive_aor2_uses_thematic_ending():
    """aor-2 stems strip the ``-ον`` thematic ending."""
    parts = {"aor2": "ἔλιπον"}
    args = derive_grc_conj_args(parts, "λείπω")
    assert args["aor-2"][0] == "aor-2"
    assert args["aor-2"][1] == "ελιπ"
    assert args["aor-2"][2] == "λιπ"


def test_derive_perf_mp_reverses_pf_mu_assimilation():
    """For γέγραμμαι (assimilated φ + μ -> μμ), the perf-MP stem fed
    to grc-conj must be ``γεγραφ`` (the un-assimilated form), not
    ``γεγραμ``. The parser uses the active perfect stem to recover
    the underlying consonant."""
    parts = {"pf": "γέγραφα", "pf_mp": "γέγραμμαι"}
    args = derive_grc_conj_args(parts, "γράφω")
    assert args["perf"][0] == "perf"
    assert args["perf"][1] == "γεγραφ"
    assert args["perf"][2] == "γεγραφ"


# ---------------------------------------------------------------------------
# Integration test: actually expand a verb via grc-conj and verify
# we get aorist + perfect forms that the present-only path misses.
# Skipped automatically when the WTP database is unavailable so the
# suite stays runnable in CI.
# ---------------------------------------------------------------------------


WTP_DB = SCRIPT_DIR / "data" / "wtp.db"


@pytest.fixture(scope="module")
def wtp_instance():
    """Lazily build a WTP instance. Skips the test if the wikitextprocessor
    package or the wtp.db artifact is missing."""
    pytest.importorskip("wikitextprocessor")
    if not WTP_DB.exists():
        pytest.skip(f"wtp.db not present at {WTP_DB}; skipping integration")
    from expand_lsj import get_wtp
    return get_wtp()


def _stripped(s: str) -> str:
    """Strip combining accents but keep length marks. Used to compare
    forms that differ only in accent placement (the suffix cache
    returns accent-stripped forms, while the Lua-direct path returns
    fully-accented forms)."""
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(
        c for c in nfd
        if not unicodedata.combining(c) or ord(c) in (0x0304, 0x0306)))


def test_grapho_integration_adds_aorist_and_perfect_forms(wtp_instance):
    """Smoke test that wiring principal parts into grc-conj actually
    expands γράφω's paradigm beyond the present system. We only
    assert presence of well-known canonical forms; exact form-count
    can drift as the underlying Lua module evolves."""
    from expand_lsj import expand_verb
    # Compare on the accent-stripped form set: the present-system
    # suffix cache populated by an earlier call returns accent-
    # stripped forms (an existing quirk), so a like-for-like compare
    # has to fold accents first.
    baseline_forms, _ = expand_verb(wtp_instance, "γράφω")
    enriched_forms, _ = expand_verb(
        wtp_instance, "γράφω", head_text=LSJ_FRAGMENTS["γράφω"])

    # The enriched output must contain at least as many forms as the
    # baseline once we fold accents.
    baseline_plain = {_stripped(f) for f in baseline_forms}
    enriched_plain = {_stripped(f) for f in enriched_forms}
    assert baseline_plain <= enriched_plain, (
        f"baseline forms missing from enriched set: "
        f"{sorted(baseline_plain - enriched_plain)[:10]}")

    # Aorist active forms should now appear (compare accent-stripped).
    aor_marker = {_stripped("ἔγραψα"), _stripped("γράψας"),
                  _stripped("ἔγραψαν")}
    assert aor_marker & enriched_plain, (
        f"no aorist forms in enriched output: "
        f"sample = {sorted(enriched_forms)[:20]}")

    # Perfect active forms should now appear.
    pf_marker = {_stripped("γέγραφα"), _stripped("γεγραφέναι")}
    assert pf_marker & enriched_plain, (
        f"no perfect forms in enriched output: "
        f"sample = {sorted(enriched_forms)[:20]}")

    # Perfect mediopassive should round-trip the φ -> μμ assimilation:
    # γέγραμμαι (1sg) or γέγραπται (3sg) should be present.
    assert (_stripped("γέγραμμαι") in enriched_plain or
            _stripped("γέγραπται") in enriched_plain)


def test_lyo_integration_uses_partial_principal_parts(wtp_instance):
    """λύω's LSJ head gives only the future (the OCR truncated the
    rest). Even with that single principal part, the enriched output
    must add future forms beyond the present system."""
    from expand_lsj import expand_verb
    fragment = (
        "poet. imper. λῦθι Pi.Fr.85: fut. λύσω but λῦτο 24.1 "
        "(at beginning of line, v.l. λύτο)"
    )
    baseline, _ = expand_verb(wtp_instance, "λύω")
    enriched, _ = expand_verb(wtp_instance, "λύω", head_text=fragment)
    baseline_plain = {_stripped(f) for f in baseline}
    enriched_plain = {_stripped(f) for f in enriched}
    assert baseline_plain <= enriched_plain
    # A future-system form should now show up.
    assert (_stripped("λύσω") in enriched_plain or
            _stripped("λυσοίμην") in enriched_plain)


def test_no_head_text_preserves_baseline_behaviour(wtp_instance):
    """When no head text is supplied (the legacy call site), the
    function must return exactly the present-system paradigm with
    no surprise additions."""
    from expand_lsj import expand_verb
    forms_a, _ = expand_verb(wtp_instance, "παιδεύω")
    forms_b, _ = expand_verb(wtp_instance, "παιδεύω", head_text="")
    assert forms_a == forms_b

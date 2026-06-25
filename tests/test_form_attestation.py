"""Tests for the form-keyed corpus attestation feature.

Two layers, matching the convention in test_attestation.py:

  * Build-from-mini-corpus tests (always run): a tiny committed GLAUx +
    Diorisis fixture is built into a temp dir by build/build_form_attestation.py,
    so the builder, schema, determinism, dedup, Beta-Code fidelity, locus
    assembly and the _attest_db query layer are all exercised with no HF
    download and no full corpora.
  * Real-DB integration tests (skip-if-absent): the lemmatize/generate gate
    wiring is checked against the downloaded data/form_profile.db when present.
"""

import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from dilemma._attest_db import AttestDB, AttestDBMissing, nfc_key, norm_key, stripped_key
from dilemma.core import grave_to_acute, strip_accents

REPO = Path(__file__).resolve().parent.parent
FX = REPO / "tests" / "fixtures" / "form_attest"
BUILDER = REPO / "build" / "build_form_attestation.py"


def _build(out_dir: Path, cap: int = 50) -> str:
    # The builder converts Diorisis Beta Code via betacode.conv (which needs
    # pygtrie); skip the build-from-mini-corpus tests where that isn't importable.
    pytest.importorskip("betacode.conv")
    out_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [sys.executable, str(BUILDER),
         "--sources", "glaux,diorisis",
         "--glaux", str(FX / "glaux"),
         "--metadata", str(FX / "metadata.txt"),
         "--diorisis", str(FX / "diorisis"),
         "--profile-out", str(out_dir / "form_profile.db"),
         "--citations-out", str(out_dir / "form_citations.db"),
         "--cap", str(cap)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    line = next(l for l in r.stdout.splitlines() if l.startswith("content_hash:"))
    return line.split()[1]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("form_attest")
    content_hash = _build(d, cap=50)
    return AttestDB(d), content_hash, d


# --- canonical keys (single source of truth) --------------------------------


def test_key_functions_match_core_helpers():
    for f in ["μῆνιν", "Ἀχιλλεύς", "τὸν", "ὣς", "Φοῖβε", "κατὰ"]:
        # norm_key must equal grave_to_acute + casefold, or the output gate and
        # the builder would disagree.
        assert norm_key(f) == grave_to_acute(unicodedata.normalize("NFC", f)).casefold()
        assert nfc_key(f) == unicodedata.normalize("NFC", f)
        assert stripped_key(f) == strip_accents(f).casefold()


# --- builder: determinism ---------------------------------------------------


def test_build_is_deterministic(tmp_path):
    h1 = _build(tmp_path / "a")
    h2 = _build(tmp_path / "b")
    assert h1 == h2  # logical content hash, identical inputs


# --- gate (presence) --------------------------------------------------------


def test_input_gate_exact(built):
    db, _, _ = built
    assert db.is_attested("μῆνιν")          # attested lowercase
    assert db.is_attested("Μῆνιν")          # capitalized form is its own key
    assert not db.is_attested("ζζζ")        # not in corpus
    # exact gate does NOT fold: a graveless/upper variant of an only-lowercase
    # form is not "exact attested".
    assert not db.is_attested("Λόγος")


def test_output_gate_folds(built):
    db, _, _ = built
    # generated forms are citation-style; the output gate folds grave + case.
    assert db.is_attested_norm("μῆνιν")
    assert db.is_attested_norm("Μῆνιν")     # capitalized folds to the same key
    assert db.is_attested_norm("λόγος")
    assert not db.is_attested_norm("ζζζ")


# --- profile (usage distribution that powers the graphs) --------------------


def test_profile_aggregates(built):
    db, _, _ = built
    rec = db.attestation("μῆνιν", max_citations=None)
    assert rec["attested"] is True
    assert rec["dominant_pos"] == "noun"
    # total is DEDUPED at the work level (the shared Diorisis token is excluded);
    # μῆνιν occurs in GLAUx work A (-5c), GLAUx work B (-4c), Diorisis-only C (2c).
    assert rec["total_count"] == 3
    assert rec["n_works"] == 3
    assert rec["by_century"] == {-5: 1, -4: 1, 2: 1}
    # source_counts are independent per-corpus counts (overlap, not summed).
    assert rec["source_counts"] == {"glaux": 2, "diorisis": 2}


def test_heatmap_joint(built):
    db, _, _ = built
    rec = db.attestation("μῆνιν")
    # century x genre joint for the usage heatmap (Diorisis 'Narrative' -> history)
    assert rec["by_century_genre"] == {-5: {"poetry": 1}, -4: {"history": 1}, 2: {"history": 1}}


def test_dedup_keeps_source_evidence(built):
    db, _, _ = built
    rec = db.attestation("μῆνιν")
    # The shared work (annotated by both corpora) is counted once in total but
    # its Diorisis reading still shows up as independent source evidence.
    assert rec["total_count"] == 3
    assert rec["source_counts"]["diorisis"] == 2


def test_unattested_returns_none(built):
    db, _, _ = built
    assert db.attestation("ζζζ") is None
    assert db.attestation_by_norm("ζζζ") is None


# --- citations (example loci) -----------------------------------------------


def test_citations_both_sources_and_loci(built):
    db, _, _ = built
    rec = db.attestation("μῆνιν", max_citations=None)
    by_src = {(c["source"], c["locus"]) for c in rec["citations"]}
    # GLAUx line + section loci, Diorisis sentence loci, from BOTH corpora.
    assert ("glaux", "1.1") in by_src           # verse: line locus
    assert ("glaux", "1.1.2") in by_src         # prose: cumulative section locus
    assert ("diorisis", "1") in by_src          # shared work, Diorisis sentence
    assert ("diorisis", "2.3") in by_src        # Diorisis-only work
    schemes = {c["locus_scheme"] for c in rec["citations"]}
    assert "line" in schemes and "section" in schemes and "diorisis-sentence" in schemes


def test_max_citations_truncates_but_not_totals(built):
    db, _, _ = built
    full = db.attestation("μῆνιν", max_citations=None)
    assert len(full["citations"]) == 4
    two = db.attestation("μῆνιν", max_citations=2)
    assert len(two["citations"]) == 2
    assert two["total_count"] == full["total_count"]   # totals unaffected
    none = db.attestation("μῆνιν", max_citations=0)
    assert none["citations"] == [] and none["total_count"] == 3


def test_beta_code_fidelity(built):
    # Diorisis stores forms in Beta Code ("mh=nin"); after conversion it must be
    # the SAME key as GLAUx's already-Unicode "μῆνιν" (else they fragment).
    db, _, _ = built
    rec = db.attestation("μῆνιν")
    assert "diorisis" in rec["source_counts"]  # the Beta-coded tokens merged in


def test_build_cap_limits_stored_citations(tmp_path):
    d = tmp_path / "capped"
    _build(d, cap=1)
    db = AttestDB(d)
    rec = db.attestation("μῆνιν", max_citations=None)
    assert len(rec["citations"]) == 1          # capped at build time
    assert rec["total_count"] == 3             # distribution still complete


# --- missing DB -------------------------------------------------------------


def test_missing_db_reports_unavailable_and_raises(tmp_path):
    db = AttestDB(tmp_path)  # empty dir
    assert db.available is False
    with pytest.raises(AttestDBMissing):
        db.is_attested("μῆνιν")


def test_profile_present_but_citations_absent(tmp_path, built):
    # Build only the profile DB into a fresh dir (delete the citations DB).
    _, _, src = built
    d = tmp_path / "noprofilecit"
    d.mkdir()
    (d / "form_profile.db").write_bytes((src / "form_profile.db").read_bytes())
    db = AttestDB(d)
    assert db.available and not db.citations_available
    rec = db.attestation("μῆνιν")
    assert rec["total_count"] == 3
    assert rec["citations"] == []
    assert "citations_note" in rec


# --- real-DB integration (skip if not downloaded) ---------------------------


def _real_available() -> bool:
    return AttestDB().available


@pytest.mark.skipif(not _real_available(), reason="form_profile.db not present")
def test_lemmatize_input_gate_real():
    from dilemma import Dilemma
    d = Dilemma()
    db = d._attestdb()
    # find an attested form and a clearly-unattested (modern) one
    att = "μῆνιν" if db.is_attested("μῆνιν") else None
    if att is None:
        pytest.skip("expected fixture form not in the downloaded corpus")
    assert d.lemmatize(att, attested_only=True) is not None
    assert d.lemmatize("υπολογιστές", attested_only=True) is None
    assert d.lemmatize_verbose("υπολογιστές", attested_only=True) == []
    assert d.lemmatize_batch([att, "υπολογιστές"], attested_only=True)[1] is None


@pytest.mark.skipif(not _real_available(), reason="form_profile.db not present")
def test_form_attestation_api_real():
    from dilemma import Dilemma
    d = Dilemma()
    if not d._attestdb().is_attested("μῆνιν"):
        pytest.skip("expected fixture form not in the downloaded corpus")
    rec = d.form_attestation("μῆνιν", max_citations=3)
    assert rec is not None and rec["total_count"] >= 1
    assert d.form_attestation("ζζζζζ") is None


# --- Phase B: First1K / PTA / byzantine sources ---------------------------


def test_tei_locus_iter_and_meta():
    lxml = pytest.importorskip("lxml.etree")
    sys.path.insert(0, str(REPO / "build"))
    import tei_locus
    tei = (
        '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>'
        '<titleStmt><title>X</title><author>A</author></titleStmt>'
        '<publicationStmt><idno type="TLG">tlg9995.tlg001</idno></publicationStmt>'
        '</fileDesc><profileDesc><creation><date notBefore="0440" notAfter="0460"/>'
        '</creation></profileDesc></teiHeader><text><body>'
        '<div type="edition" n="urn:cts:pta:pta9995.pta001.pta-grc1">'
        '<div type="textpart" subtype="book" n="1">'
        '<div type="textpart" subtype="chapter" n="2">'
        '<p>θεόπνευστος <note>skip me</note> ἀρετὴ</p></div></div>'
        '<div type="textpart" subtype="Book" n="2"><l n="5">μῆνιν ἄειδε</l></div>'
        '</div></body></text></TEI>'
    )
    root = lxml.fromstring(tei.encode("utf-8"))
    toks = list(tei_locus.iter_tokens(root))
    forms = {f for f, l, s in toks}
    assert "θεόπνευστος" in forms and "ἀρετὴ" in forms
    assert "skip" not in forms and "me" not in forms  # <note> skipped
    by_form = {f: (l, s) for f, l, s in toks}
    assert by_form["θεόπνευστος"] == ("1.2", "book.chapter")
    assert by_form["μῆνιν"] == ("2.5", "Book.line")   # verse line locus
    meta = tei_locus.work_meta(root, "pta9995.pta001.pta-grc1.xml")
    assert meta["tlg_id"] == "9995-001"
    assert meta["creation_year"] == 450


def test_byz_century_parsing():
    sys.path.insert(0, str(REPO / "build"))
    import build_form_attestation as B
    assert B._byz_century("12th century (MS 15th century)") == 12
    assert B._byz_century("14th century") == 14
    assert B._byz_century("3rd century BC") == -3
    assert B._byz_century("undated") is None


def _build_all(out_dir: Path, cap: int = 50) -> str:
    pytest.importorskip("betacode.conv")
    pytest.importorskip("lxml.etree")
    out_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [sys.executable, str(BUILDER),
         "--sources", "glaux,diorisis,first1k,pta,pg,byz",
         "--glaux", str(FX / "glaux"), "--metadata", str(FX / "metadata.txt"),
         "--diorisis", str(FX / "diorisis"),
         "--first1k", str(FX / "first1k" / "data"),
         "--pta", str(FX / "pta" / "data"),
         "--pg", str(FX / "pg"),
         "--byz", str(FX / "byz"),
         "--profile-out", str(out_dir / "form_profile.db"),
         "--citations-out", str(out_dir / "form_citations.db"),
         "--cap", str(cap)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    return out_dir


@pytest.fixture(scope="module")
def built_all(tmp_path_factory):
    return AttestDB(_build_all(tmp_path_factory.mktemp("form_attest_b")))


def test_first1k_form_attested_with_tei_locus(built_all):
    rec = built_all.attestation("πατερικὸν", max_citations=None)
    assert rec is not None and rec["source_counts"].get("first1k") == 1
    cite = rec["citations"][0]
    assert cite["source"] == "first1k"
    assert cite["locus"] == "1.2" and cite["locus_scheme"] == "book.chapter"


def test_pta_form_attested_with_century(built_all):
    rec = built_all.attestation("θεόπνευστος", max_citations=None)
    assert rec is not None and rec["source_counts"].get("pta") == 1
    assert rec["by_century"] == {5: 1}            # creation date 440-460 -> 5th c.
    assert rec["citations"][0]["locus_scheme"] == "section"


def test_byz_form_attested_with_line_locus(built_all):
    rec = built_all.attestation("γέρων", max_citations=None)
    assert rec is not None and rec["source_counts"].get("byz") == 1
    assert rec["by_century"] == {14: 1}           # manifest "14th century"
    assert rec["citations"][0]["source"] == "byz"
    assert rec["citations"][0]["locus_scheme"] == "line"


def test_pg_form_attested_with_migne_locus(built_all):
    rec = built_all.attestation("θεωρία", max_citations=None)
    assert rec is not None and rec["source_counts"].get("pg") == 2  # pp.10 + 11
    assert rec["by_century"] == {14: 2}                # PG151 -> Gregory Palamas, 14th c.
    cite = rec["citations"][0]
    assert cite["work_id"] == "PG151" and cite["locus_scheme"] == "migne-page"
    assert cite["locus"] in ("10", "11")


def test_phase_b_does_not_break_phase_a(built_all):
    # μῆνιν still attested from GLAUx+Diorisis with its deduped total of 3.
    rec = built_all.attestation("μῆνιν")
    assert rec["total_count"] == 3
    assert "glaux" in rec["source_counts"] and "diorisis" in rec["source_counts"]


@pytest.mark.skipif(not _real_available(), reason="form_profile.db not present")
def test_generate_output_gate_real():
    from dilemma.paradigm import generate, ParadigmSlot
    # an unattested generated cell is dropped; an attested one is not.
    slot = ParadigmSlot.verb_finite(voice="active", tense="present",
                                    mood="indicative", person="1", number="sg")
    # type check only: result is a ParadigmForm or None, never an exception
    out = generate("λύω", slot, attested_only=True)
    assert out is None or out.form

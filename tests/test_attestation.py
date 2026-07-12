#!/usr/bin/env python3
"""Tests for the per-lemma corpus attestation artifact + Dilemma.attestation().

The artifact (data/lemma_attestation.json) is built by
build/build_lemma_attestation.py from GLAUx + Diorisis. Like the other large
data files it lives on HuggingFace, not in git, so the whole module skips when
it is absent.

Run with:
    python -m pytest tests/test_attestation.py -x -v
"""

import json
import sys
import unicodedata
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ATTESTATION_PATH = DATA_DIR / "lemma_attestation.json"
sys.path.insert(0, str(PROJECT_ROOT))

from dilemma import Dilemma

pytestmark = pytest.mark.skipif(
    not ATTESTATION_PATH.exists(),
    reason="data/lemma_attestation.json not found (lives on HuggingFace)",
)


@pytest.fixture(scope="module")
def d():
    return Dilemma(lang="all")


@pytest.fixture(scope="module")
def raw():
    with open(ATTESTATION_PATH, encoding="utf-8") as f:
        return json.load(f)


# --- _meta / schema --------------------------------------------------------

def test_meta_schema(raw):
    m = raw["_meta"]
    assert m["schema_version"] == 1
    assert m["sources"] == ["glaux", "diorisis", "oga"]
    assert len(m["genres"]) == 10
    assert m["n_lemmas"] == len(raw["lemmas"])
    # Union of works, not the naive sum: GLAUx + Diorisis-only works
    # (~17M) plus the ~16M tokens of works only cog's OGA export covers.
    assert 28_000_000 < m["total_tokens"] < 42_000_000
    assert "dedup" in m
    assert set(m["source_sha"]) == {
        "glaux_metadata", "glaux_xml", "diorisis_xml",
        "oga_export", "oga_pin"}
    # dilemma pins cog's export, not OGA upstream (cog pins upstream).
    assert m["source_sha"]["oga_pin"].startswith("cog export oga-")


def test_dedup_total_is_union_not_sum(raw):
    """total / by_* are the DEDUPED frequency (each work counted once);
    source_counts holds each source's INDEPENDENT count (overlapping). So the
    deduped total sits between GLAUx alone and the naive GLAUx+Diorisis sum, and
    both sources' evidence is preserved."""
    total = sc_glaux = sc_diorisis = sc_oga = 0
    for e in raw["lemmas"].values():
        total += e["total"]
        sc_glaux += e["source_counts"].get("glaux", 0)
        sc_diorisis += e["source_counts"].get("diorisis", 0)
        sc_oga += e["source_counts"].get("oga", 0)
    assert total == raw["_meta"]["total_tokens"]
    # union, not sum
    assert sc_glaux < total < sc_glaux + sc_diorisis + sc_oga
    assert sc_glaux > 10_000_000 and sc_diorisis > 5_000_000  # both preserved


def test_dimension_sums_are_consistent(raw):
    """by_genre partitions the deduped tokens (sums to total); by_century /
    by_dialect are subsets. source_counts is independent (need not sum to
    total) but is always present and non-empty."""
    for w, e in raw["lemmas"].items():
        t = e["total"]
        assert e["source_counts"], w
        assert sum(e["by_genre"].values()) == t, w
        assert sum(e["by_century"].values()) <= t, w
        if "by_dialect" in e:
            assert sum(e["by_dialect"].values()) <= t, w


def test_evidence_only_lemmas_preserved(raw):
    """The multi-source design keeps single-source readings of shared works
    (total 0, non-empty source_counts) instead of discarding them, and every
    lemma carries at least one source's evidence."""
    assert all(e["source_counts"] for e in raw["lemmas"].values())
    zero = [w for w, e in raw["lemmas"].items() if e["total"] == 0]
    assert zero, "expected some total=0 evidence-only lemmas"


def test_keys_are_lexical_greek(raw):
    """No residue keys: each starts with a Greek base letter, is not an
    all-caps geometry label, and contains only Greek letters + marks."""
    for w in raw["lemmas"]:
        assert unicodedata.category(w[0])[0] == "L", repr(w)
        letters = [c for c in w if unicodedata.category(c)[0] == "L"]
        assert letters, repr(w)
        assert not (len(letters) >= 2
                    and all(unicodedata.category(c) == "Lu" for c in letters)), \
            repr(w)
        for c in w:
            cat = unicodedata.category(c)
            assert cat[0] == "M" or (
                ("Ͱ" <= c <= "Ͽ" or "ἀ" <= c <= "῿")
                and cat[0] == "L"), repr(w)


def test_inner_dicts_are_canonically_ordered(raw):
    """Lemma keys sorted by code point; within each lemma, by_century is
    chronological, by_genre follows _meta.genres, source_counts follows
    _meta.sources, by_dialect is alphabetical (the brief's "sorted keys")."""
    genre_pos = {g: i for i, g in enumerate(raw["_meta"]["genres"])}
    source_pos = {s: i for i, s in enumerate(raw["_meta"]["sources"])}

    keys = list(raw["lemmas"])
    assert keys == sorted(keys), "top-level lemma keys not sorted"

    for w, e in raw["lemmas"].items():
        cents = [int(k) for k in e["by_century"]]
        assert cents == sorted(cents), (w, "by_century", cents)
        gp = [genre_pos[g] for g in e["by_genre"]]
        assert gp == sorted(gp), (w, "by_genre", list(e["by_genre"]))
        sp = [source_pos[s] for s in e["source_counts"]]
        assert sp == sorted(sp), (w, "source_counts", list(e["source_counts"]))
        if "by_dialect" in e:
            dk = list(e["by_dialect"])
            assert dk == sorted(dk), (w, "by_dialect", dk)


# --- public API ------------------------------------------------------------

def test_common_lemma_is_frequent(d):
    a = d.attestation("ἄνθρωπος")
    assert a is not None
    assert a["total"] > 1000
    assert a["dominant_pos"] == "noun"
    # attested independently by all three sources (agreement = confidence)
    assert set(a["source_counts"]) == {"glaux", "diorisis", "oga"}


def test_homeric_lemma_century_and_dialect(d):
    for w in ("μῆνις", "ἀείδω"):
        a = d.attestation(w)
        assert a is not None, w
        assert "-8" in a["by_century"], f"{w} should attest in the 8th c. BC"
        assert "Ionic/Epic" in a.get("by_dialect", {}), \
            f"{w} should attest in the Ionic/Epic dialect"


def test_dominant_pos_separates_noun_and_verb(d):
    assert d.attestation("ἀείδω")["dominant_pos"] == "verb"
    assert d.attestation("ἄνθρωπος")["dominant_pos"] == "noun"


def test_unattested_lemma_returns_none(d):
    assert d.attestation("ζζζζζ") is None


def test_lookup_is_nfc_nfd_robust(d):
    nfc = d.attestation(unicodedata.normalize("NFC", "ἄνθρωπος"))
    nfd = d.attestation(unicodedata.normalize("NFD", "ἄνθρωπος"))
    assert nfc is not None and nfd is not None
    assert nfc["total"] == nfd["total"]

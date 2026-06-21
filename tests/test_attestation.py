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
    assert m["sources"] == ["glaux", "diorisis"]
    assert len(m["genres"]) == 10
    assert m["n_lemmas"] == len(raw["lemmas"])
    assert m["total_tokens"] > 25_000_000          # ~27M
    assert set(m["source_sha"]) == {
        "glaux_metadata", "glaux_xml", "diorisis_xml"}


def test_dimension_sums_are_consistent(raw):
    """by_source and by_genre partition every token; by_century / by_dialect
    are subsets (a token may lack a dialect)."""
    for w, e in raw["lemmas"].items():
        t = e["total"]
        assert sum(e["by_source"].values()) == t, w
        assert sum(e["by_genre"].values()) == t, w
        assert sum(e["by_century"].values()) <= t, w
        if "by_dialect" in e:
            assert sum(e["by_dialect"].values()) <= t, w


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
    chronological, by_genre follows _meta.genres, by_source follows
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
        sp = [source_pos[s] for s in e["by_source"]]
        assert sp == sorted(sp), (w, "by_source", list(e["by_source"]))
        if "by_dialect" in e:
            dk = list(e["by_dialect"])
            assert dk == sorted(dk), (w, "by_dialect", dk)


# --- public API ------------------------------------------------------------

def test_common_lemma_is_frequent(d):
    a = d.attestation("ἄνθρωπος")
    assert a is not None
    assert a["total"] > 1000
    assert a["dominant_pos"] == "noun"
    assert set(a["by_source"]) <= {"glaux", "diorisis"}


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

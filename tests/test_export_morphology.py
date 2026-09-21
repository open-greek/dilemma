import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dilemma.form_sanitize import has_editorial_sigla
from export_morphology import (
    _derive_elision_pairs,
    _derive_nu_forms,
    _load_lemma_forms,
)


VERB_TAGS = [
    "third-person",
    "plural",
    "present",
    "indicative",
    "active",
]


def test_nu_derivation_rejects_editorial_sigla(tmp_path):
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(json.dumps([
        {
            "form": ")λπίζουσι",
            "lemma": ")λπίζω",
            "pos": "verb",
            "tags": VERB_TAGS,
        },
        {
            "form": "ἐλπίζουσι",
            "lemma": "ἐλπίζω",
            "pos": "verb",
            "tags": VERB_TAGS,
        },
    ], ensure_ascii=False), encoding="utf-8")

    forms = _derive_nu_forms([pairs_path])

    assert ")λπίζουσι" not in forms
    assert "ἐλπίζουσι" in forms
    assert not any(has_editorial_sigla(form) for form in forms)


def test_elision_loading_and_derivation_reject_editorial_sigla(tmp_path):
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(json.dumps([
        {"form": "λέγε", "lemma": "λέγω"},
        {"form": "λέγ᾽", "lemma": "λέγω"},
        {"form": ")φέρε", "lemma": "φέρω"},
        {"form": "φέρ᾽", "lemma": ")φέρω"},
    ], ensure_ascii=False), encoding="utf-8")

    lemma_forms = _load_lemma_forms([pairs_path], None)
    assert lemma_forms == {"λέγω": {"λέγε", "λέγ᾽"}}

    # Keep the derivation guard independent of the loader so callers passing
    # an already-built mapping cannot reintroduce contaminated entries.
    lemma_forms[")ἄγω"] = {"ἄγε", "ἄγ᾽"}
    lemma_forms["φέρω"] = {")φέρε", ")φέρ᾽"}
    pairs = _derive_elision_pairs(lemma_forms)

    assert pairs["λέγε"] == "λέγ᾽"
    assert not any(
        has_editorial_sigla(full) or has_editorial_sigla(elided)
        for full, elided in pairs.items()
    )

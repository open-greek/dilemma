"""Guards against ingesting the placeholders a treebank uses for "not a word".

Two annotation conventions were being read as ordinary data:

  * GLAUx marks an editorial gap, where the manuscript is damaged or
    unreadable, with ``relation="GAP"``, ``postag="z"`` and the gap character
    in the lemma field. The surviving letters stay in ``form``, so a token
    reads as Greek even though it is the remains of one. Before the filter,
    ``Gδου`` (what is left of the genitive of Hades after its first two
    letters were lost, Plutarch, Comparatio Cimonis et Luculli) reached
    ``lookup.db`` and ``Dilemma().lemmatize("Gδου")`` answered ``"G"``.

  * The cog standardized exports use the CoNLL-U convention of ``_`` for a
    field the annotator left empty. That is a plain string, so it passed
    every lemma gate: 1,108 Pedalion pairs carried it, and
    ``Dilemma().lemmatize("διατείνας")`` answered ``"_"`` for a real aorist
    participle of διατείνω.

Both are cheap to detect exactly, by reading the annotation the corpus
already provides, rather than by guessing from the characters.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build"))

import cog_annotations as C  # noqa: E402
import build_glaux_pairs as G  # noqa: E402


def _glaux_xml(tmp_path, words):
    """Write a one-sentence GLAUx file whose <word> elements are `words`."""
    body = "".join(
        "<word " + " ".join(f'{k}="{v}"' for k, v in w.items()) + "/>"
        for w in words
    )
    path = tmp_path / "0001-001.xml"
    path.write_text(
        f"<doc><sentence analysis='automatic'>{body}</sentence></doc>",
        encoding="utf-8",
    )
    return path


def _extract(tmp_path, words):
    _glaux_xml(tmp_path, words)
    pairs = G.extract_glaux(str(tmp_path))
    if isinstance(pairs, tuple):       # (pairs, stats) in some versions
        pairs = pairs[0]
    return {(p["form"], p["lemma"]) for p in pairs}


class TestGlauxEditorialGaps:

    def test_a_gap_token_is_not_ingested(self, tmp_path):
        got = _extract(tmp_path, [
            # The real annotation from Plutarch: relation GAP, postag z, and
            # the gap character as the lemma. Only `δου` survived.
            {"form": "Gδου", "lemma": "G", "postag": "z--------",
             "relation": "GAP"},
        ])
        assert got == set(), f"an editorial gap was ingested as a word: {got}"

    def test_either_marker_alone_is_enough(self, tmp_path):
        got = _extract(tmp_path, [
            {"form": "Gἦν", "lemma": "G", "postag": "z--------"},
            {"form": "τGῆς", "lemma": "G", "relation": "GAP"},
        ])
        assert got == set(), f"gap markers leaked: {got}"

    def test_ordinary_words_still_come_through(self, tmp_path):
        got = _extract(tmp_path, [
            {"form": "θεός", "lemma": "θεός", "postag": "n-s---mn-"},
            {"form": "ἀρχόμενος", "lemma": "ἄρχω", "postag": "v-sppemn-"},
            # A zeta-initial word: the gap test reads the POSTAG, not the form,
            # so a word that merely starts with zeta must survive.
            {"form": "ζῷον", "lemma": "ζῷον", "postag": "n-s---nn-"},
        ])
        assert got == {("θεός", "θεός"), ("ἀρχόμενος", "ἄρχω"),
                       ("ζῷον", "ζῷον")}

    def test_editorial_sigla_are_not_ingested(self, tmp_path):
        got = _extract(tmp_path, [
            {"form": ")λπίζουσι", "lemma": ")λπίζω",
             "postag": "v3ppia---"},
            {"form": "ἐλπίζουσι", "lemma": ")λπίζω",
             "postag": "v3ppia---"},
            {"form": "ἐλπίζουσι", "lemma": "ἐλπίζω",
             "postag": "v3ppia---"},
        ])
        assert got == {("ἐλπίζουσι", "ἐλπίζω")}

    def test_the_character_test_alone_would_not_have_caught_it(self):
        # Why the annotation is read rather than the characters: `is_greek`
        # asks whether ANY character is Greek, and the surviving letters of a
        # gap token are. This is the mechanism the filter exists to cover, so
        # if it ever changes the filter's justification should be revisited.
        assert G.is_greek("Gδου") is True
        assert G.is_greek("θεός") is True


class TestEmptyLemmaField:

    def test_underscore_is_not_a_lemma(self):
        assert C.is_clean_lemma("_") is False

    def test_real_lemmas_are_unaffected(self):
        for lemma in ("θεός", "ἄρχω", "διατείνω", "εἶπον"):
            assert C.is_clean_lemma(lemma) is True, lemma

    def test_empty_stays_rejected(self):
        assert C.is_clean_lemma("") is False

#!/usr/bin/env python3
"""Diorisis Ancient Greek corpus loader for the polytonic next-word LM.

Mirrors ``iter_glaux_sentences`` in ``train_lm.py`` so ``train_lm.py`` can
feed GLAUx and Diorisis through the same sentence-level pipeline without
per-corpus branching downstream.

Corpus
------

The Diorisis Ancient Greek Corpus ships as TEI-flavored XML with one
``<sentence>`` element per natural sentence; each ``<word>`` inside
carries a ``form`` attribute in **beta code** (TLG-style ASCII
transliteration, e.g. ``a)/ndra`` for ἄνδρα). Sentence boundaries are
explicit and trustworthy, so we use them directly rather than punctuation
heuristics. Diorisis does not record internal punctuation as separate
tokens, so ``</s>`` is always appended once at the end of each sentence.

Beta code conversion
--------------------

We rely on the ``betacode`` PyPI package for the ASCII -> Unicode mapping
(it handles breathings, accents, iota subscript, diaeresis, final sigma).
Output is normalized to NFC to match GLAUx's polytonic forms exactly. See
``_BETA_SANITY_CHECKS`` for the spot-checks we run on import to catch a
silent regression in the upstream package.

Token shape
-----------

Each yielded sentence is ``(sent_id, [BOS, tok1, tok2, ..., EOS])`` where
every intermediate token is the converted NFC form of a word with at
least one Greek letter (U+0370..U+03FF or U+1F00..U+1FFF). This matches
what ``iter_glaux_sentences`` in ``train_lm.py`` yields, so both loaders
flow into the same vocab / n-gram counters.

Author filter
-------------

Everything in the Diorisis drop is Ancient Greek literary text: classical
Attic, epic, Koine (New Testament, Septuagint), early Byzantine (Eusebius,
Basil, Julian). We do not drop any authors. Heavy papyrus lacunae would
matter, but Diorisis texts are all edited literary critical editions, so
lacuna-dominated words are rare.
"""

from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

# Reserved tokens. Keep in sync with train_lm.py. Import-from is avoided
# so this module stays independent and merge-friendly against parallel
# branches that add other corpora.
BOS_TOK = "<s>"
EOS_TOK = "</s>"

_BETA_CONV = None


def _get_beta_conv():
    """Lazy-import the ``betacode`` package so missing deps surface only
    when someone actually runs the Diorisis ingest."""
    global _BETA_CONV
    if _BETA_CONV is None:
        import betacode.conv  # type: ignore
        _BETA_CONV = betacode.conv.beta_to_uni
    return _BETA_CONV


# Spot checks, verified against Perseus / TLG reference forms.
# Run once on first call; if any of these regresses we want to fail
# loud rather than silently train on garbage.
_BETA_SANITY_CHECKS = [
    # (beta_input, expected_NFC_unicode)
    ("a)/ndra", "ἄνδρα"),
    ("mh=nin", "μῆνιν"),
    ("*)axilh=os", "Ἀχιλῆος"),
    ("ou)lome/nhn", "οὐλομένην"),
    ("*phlhi+a/dew", "Πηληϊάδεω"),
    ("qea/", "θεά"),
    ("a)/eide", "ἄειδε"),
    ("tw=nd'", "τῶνδ’"),
    # Elision after an unaccented α, ι or υ, which the package would
    # otherwise read as a macron (δῑ for δι’).
    ("di'", "δι’"),
    ("*di'", "Δι’"),
    ("le/gous'", "λέγουσ’"),
]

_SANITY_DONE = False


def _run_sanity_checks() -> None:
    global _SANITY_DONE
    if _SANITY_DONE:
        return
    failures = []
    for beta, expected in _BETA_SANITY_CHECKS:
        got = beta_to_nfc(beta)
        if got != expected:
            failures.append((beta, got, expected))
    if failures:
        msg = "betacode conversion sanity check failed:\n"
        for beta, got, expected in failures:
            msg += f"  {beta!r} -> {got!r} (expected {expected!r})\n"
        raise RuntimeError(msg)
    _SANITY_DONE = True


# In a small fraction (~0.13%) of Diorisis forms, elision after a
# consonant is encoded with a right paren ``)`` (the smooth-breathing
# marker) instead of an apostrophe ``'``. Example: ``par)`` for
# ``παρ’``. This confuses beta-to-Unicode into treating the ``)`` as a
# breathing mark on the final consonant, which is nonsensical and
# leaves the raw ``)`` in the output. We rewrite trailing
# ``<consonant>)`` to ``<consonant>'`` before conversion. Vowel + ``)``
# (e.g. ``ou)`` / ``ei)``) is legitimate smooth breathing and is left
# untouched.
_BETA_CONSONANTS = "bgdzqklmnprstfxcBGDZQKLMNPRSTFXC"
_ELISION_PAREN_RE = re.compile(
    rf"(?<=[{_BETA_CONSONANTS}])\)$"
)


def _fix_elision_artifact(beta: str) -> str:
    return _ELISION_PAREN_RE.sub("'", beta)


def beta_to_nfc(beta: str) -> str:
    """Convert a single beta-code token to polytonic Unicode (NFC).

    A trailing apostrophe marks elision. After an unaccented α, ι or υ the
    ``betacode`` package reads it as a macron on that vowel (``di'`` became
    ``δῑ`` for 8,136 Diorisis tokens of δι’), so the word is converted without
    it and the elision mark U+2019 appended, which is what the package
    already produces after a consonant (``par'`` -> ``παρ’``).
    """
    conv = _get_beta_conv()
    beta = _fix_elision_artifact(beta)
    if len(beta) > 1 and beta.endswith("'"):
        core = unicodedata.normalize("NFC", conv(beta[:-1]))
        # The converter writes a word-final sigma as ς; before an elision
        # mark the medial form is the one Greek uses (λέγουσ’, πᾶσ’).
        if core.endswith("ς"):
            core = core[:-1] + "σ"
        return core + "\u2019"
    return unicodedata.normalize("NFC", conv(beta))


def _is_greek_token(s: str) -> bool:
    """True iff the token has at least one Greek letter.

    Diorisis forms occasionally carry a trailing apostrophe / combining
    mark only, but those always co-occur with a Greek base letter, so
    this filter is sufficient.
    """
    return any(
        "\u0370" <= c <= "\u03FF" or "\u1F00" <= c <= "\u1FFF"
        for c in s
    )


def iter_diorisis_sentences(
    diorisis_dir: Path,
    max_files: int | None = None,
) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(sent_id, [tokens])`` for every Diorisis sentence.

    Contract matches ``iter_glaux_sentences`` in ``train_lm.py``:

      * tokens are NFC polytonic Unicode, not beta code;
      * tokens are bracketed by ``BOS_TOK`` / ``EOS_TOK``;
      * non-Greek tokens (numerals, stray Latin from marginalia) are
        dropped;
      * sentences with fewer than one real token are skipped.

    Sentence ids are namespaced as ``diorisis:<xml_stem>:<sentence_id>``
    so they are globally unique and deterministic across runs, which
    matters for the hash-based dev split in ``train_lm.py``. The
    ``diorisis:`` prefix also keeps GLAUx bucket assignments stable when
    this corpus is added (GLAUx sids stay unprefixed, matching the
    GLAUx-only baseline run).
    """
    _run_sanity_checks()
    xml_files = sorted(diorisis_dir.glob("*.xml"))
    if not xml_files:
        raise SystemExit(
            f"No Diorisis XML files found at {diorisis_dir}"
        )
    if max_files is not None:
        xml_files = xml_files[:max_files]

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            continue

        doc_id = xml_file.stem
        for sent in tree.findall(".//sentence"):
            sid = sent.get("id") or "?"
            global_sid = f"diorisis:{doc_id}:{sid}"
            tokens: list[str] = [BOS_TOK]
            for w in sent.findall("word"):
                beta = w.get("form") or ""
                if not beta:
                    continue
                form = beta_to_nfc(beta)
                if not _is_greek_token(form):
                    continue
                tokens.append(form)

            # Diorisis sentence boundaries are explicit; append </s>.
            tokens.append(EOS_TOK)

            # Need at least one real token between <s> and </s>.
            if len(tokens) >= 3:
                yield global_sid, tokens


if __name__ == "__main__":
    # Smoke test: convert a handful of sentences from the first XML
    # files and print them so a human can eyeball conversion quality.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(
        Path.home() / "Documents" / "dilemma" / "data" / "diorisis" / "xml"
    ))
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    d = Path(args.dir)
    for i, (sid, toks) in enumerate(
        iter_diorisis_sentences(d, max_files=2)
    ):
        print(sid)
        print("  " + " ".join(toks))
        if i + 1 >= args.n:
            break

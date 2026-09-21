"""Sanitise Greek surface forms so combining breathing marks don't end up stranded.

Two pathological patterns turn up in upstream OCR, treebank, and paradigm
expansion output:

  1. A leading combining psili/dasia (U+0313 / U+0314) or spacing psili/dasia
     (U+1FBF / U+1FFE) preceding the base letter. This is the wrong Unicode
     order for a breathing on that letter and NFC cannot fix it at read time
     (the mark precedes the base, so no composition happens).
  2. A trailing combining psili/dasia used as an apostrophe for elision.
     The correct mark is U+1FBD (GREEK KORONIS), a spacing character.
     Combining marks visually hang off the final vowel otherwise.

Both patterns are fixed up to the intended visual form:
  - Leading mark: reattach to the base letter and NFC-compose. If the base
    cannot carry a breathing (consonant aphaeresis, e.g. U+0313 + σ for
    'στί from ἐστί), emit U+1FBD as a leading spacing koronis instead.
  - Trailing mark: replace with U+1FBD. If the preceding base letter
    already carries that breathing (precomposed), drop the trailing mark
    as a redundant-breathing bug.

This module is imported by `export_hunspell.py` (final guard before the
Hunspell .dic is written), `build_lookup_db.py` (ingestion guard before
forms land in lookup.db), and the per-source extract_*.py scripts so that
each stage of the pipeline is self-consistent.

Lexicon and paradigm sources also use square brackets and parentheses as
editorial notation. Those delimiters are not Greek spelling. The one safe
mechanical resolution is a terminal ``(nu)`` marker, written ``(nu`` or
``(nu)`` in source data, which represents forms both without and with movable
nu. All other bracketed material remains ambiguous and is rejected.
"""
from __future__ import annotations

import unicodedata

_COMBINING_PSILI = 0x0313
_COMBINING_DASIA = 0x0314
_SPACING_PSILI = 0x1FBF
_SPACING_DASIA = 0x1FFE
_GREEK_KORONIS = "\u1FBD"
EDITORIAL_SIGLA = frozenset("[]()")
# Textual apostrophes used for Greek elision. U+02B9 MODIFIER LETTER PRIME is
# deliberately absent: in the LM vocabulary it normally marks a numeral
# (alpha-prime, beta-prime), not an elision.
FINAL_ELISION_APOSTROPHES = frozenset("'`\u02BC\u1FBD\u1FBF\u2019")


def has_editorial_sigla(s: str) -> bool:
    """Return whether *s* still carries bracket/parenthesis notation."""
    return any(char in EDITORIAL_SIGLA for char in s)


def canonicalize_final_elision(s: str) -> str:
    """Fold a word-final textual apostrophe onto the Greek koronis.

    Lookup and exported morphology use U+1FBD as their storage spelling,
    while corpora commonly use ASCII apostrophe, U+2019, U+02BC, or spacing
    psili. Numeral primes are intentionally not folded here.
    """
    if s and s[-1] in FINAL_ELISION_APOSTROPHES:
        return s[:-1] + _GREEK_KORONIS
    return s


def resolve_editorial_form(s: str) -> tuple[str, ...]:
    """Resolve safe editorial notation in a surface form.

    A final ``(nu`` or ``(nu)`` is Wiktionary's optional movable-nu marker,
    so both conventional spellings are returned. Any other square bracket or
    parenthesis is ambiguous Leiden notation and yields no spelling.
    """
    form = sanitize_form(s)
    if not has_editorial_sigla(form):
        return (form,) if form else ()

    if form.endswith("(\u03bd)"):
        stem = form[:-3]
    elif form.endswith("(\u03bd"):
        stem = form[:-2]
    else:
        return ()
    if not stem or has_editorial_sigla(stem):
        return ()
    return (stem, stem + "\u03bd")


def sanitize_form(s: str) -> str:
    """Return an NFC form with stray leading/trailing breathings fixed up."""
    if not s:
        return s
    out = s

    # Handle leading breathing character (at most one).
    if ord(out[0]) in (_COMBINING_PSILI, _COMBINING_DASIA,
                       _SPACING_PSILI, _SPACING_DASIA):
        mark_cp = ord(out[0])
        if mark_cp == _SPACING_PSILI:
            comb = chr(_COMBINING_PSILI)
        elif mark_cp == _SPACING_DASIA:
            comb = chr(_COMBINING_DASIA)
        else:
            comb = out[0]
        rest = out[1:]
        if not rest:
            out = ""
        else:
            base = rest[0]
            base_nfc = unicodedata.normalize("NFC", base)
            combined_nfc = unicodedata.normalize("NFC", base + comb)
            if len(combined_nfc) == len(base_nfc):
                out = unicodedata.normalize("NFC", base + comb + rest[1:])
            else:
                base_nfd = unicodedata.normalize("NFD", base)
                base_marks = {ord(c) for c in base_nfd
                              if unicodedata.category(c) == "Mn"}
                if ord(comb) in base_marks:
                    # Redundant mark on a base letter that already has it.
                    out = unicodedata.normalize("NFC", base + rest[1:])
                else:
                    # Consonant aphaeresis: prefix with GREEK KORONIS.
                    out = _GREEK_KORONIS + unicodedata.normalize(
                        "NFC", base + rest[1:])

    # Handle trailing combining breathing.
    if out and ord(out[-1]) in (_COMBINING_PSILI, _COMBINING_DASIA):
        prev = out[-2] if len(out) >= 2 else ""
        if prev:
            prev_nfd = unicodedata.normalize("NFD", prev)
            prev_marks = {ord(c) for c in prev_nfd
                          if unicodedata.category(c) == "Mn"}
            if prev_marks & {_COMBINING_PSILI, _COMBINING_DASIA}:
                out = out[:-1]
            else:
                out = out[:-1] + _GREEK_KORONIS
        else:
            out = out[:-1]

    out = unicodedata.normalize("NFC", out)
    # Canonicalize word-final sigma: Beta-Code-derived sources (e.g. Diorisis)
    # emit medial σ word-finally (γραφῆσ, εἷσ); Greek words always end in ς, so
    # fold so keys match standard-spelling queries.
    if out.endswith("σ"):
        out = out[:-1] + "ς"
    return out

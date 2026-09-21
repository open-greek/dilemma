#!/usr/bin/env python3
"""Export Ancient Greek boundary-rewrite morphology tables for Tonos.

Produces a single compact JSON at ``build/hunspell/grc_morph.json`` that
the Tonos iOS keyboard reads at install time. The file carries two
boundary-sensitive rewrite tables:

* ``nu`` - surface forms that take movable nu (``ἐστί`` -> ``ἐστίν``)
  when the next word is vowel-initial. Includes:

  - Verb 3sg active indicative past (imperfect / aorist / pluperfect /
    perfect) ending in ``-ε``;
  - Verb 3sg active indicative present ending in ``-σι`` / ``-τι``
    (``ἐστί``, ``δίδωσι``, ``τίθησι``);
  - Verb 3pl active indicative (present / future / perfect) ending in
    ``-σι`` (``λέγουσι``, ``γράφουσι``);
  - Noun / pronoun / adjective dative plural ending in ``-σι`` /
    ``-ξι`` / ``-ψι`` (``ἀνδράσι``, ``πᾶσι``);
  - A small closed list of numerals that historically take movable nu
    (``εἴκοσι``).

  Subjunctive, optative, imperative, infinitive, and participle forms
  are explicitly excluded because Ancient Greek movable nu never
  attaches to them.

* ``el`` - full-form -> elided-form pairs harvested from GLAUx,
  Diorisis, and the canonical dilemma lookup table. The elided form
  is stored in NFC with its final elision glyph canonicalised to
  U+1FBD GREEK KORONIS (same convention as
  ``HunspellCompiler.normalizeEntry`` and ``GreekStyle``'s existing
  hardcoded particle table). The Tonos output layer rewrites the
  koronis to the user's chosen elision glyph via
  ``GreekStyle.applyingElisionMark``.

The derivation pulls from:

* ``data/glaux_pairs.json`` + ``data/diorisis_pairs.json``: morpho-
  tagged token stream with ``pos`` and grammatical feature tags. This
  is where the nu-eligibility tags come from.
* ``data/lookup.db`` (grc src only): full form-to-lemma table, used to
  augment the elision pair list with forms that appear only in the
  Wiktionary paradigm expansion.

When the corpus offers multiple elided variants for a single full form
(e.g. ``ἀπ᾽`` vs ``ἀφ᾽`` for ``ἀπό``; different editions use different
glyphs), we score candidates and pick the one that best matches the
full form's casing and breathing profile. Canonical ten particles are
additionally pinned to a fixed mapping so ``ἀλλά`` always elides to
``ἀλλ᾽`` regardless of whatever minority spellings the corpus carries.

Output is a single JSON file; Tonos compresses it with LZMA as part of
``scripts/prebuild_archives.sh``. Total compressed size is under
300 KB as of this writing, against the app's 10 MB size budget for the
combined morphology feature.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from dilemma.form_sanitize import has_editorial_sigla

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "build" / "hunspell"

GLAUX_PAIRS = DATA / "glaux_pairs.json"
DIORISIS_PAIRS = DATA / "diorisis_pairs.json"
LOOKUP_DB = DATA / "lookup.db"

# Elision glyphs we treat as equivalent on the trailing position.
# U+0313 COMBINING COMMA ABOVE and U+1FBD GREEK KORONIS are the two
# canonical Greek editing conventions; U+02BC MODIFIER LETTER APOSTROPHE
# and U+2019 RIGHT SINGLE QUOTATION MARK appear in modern typeset
# editions (Oxford, Teubner) that reach for a generic apostrophe. All
# four are folded to U+1FBD for storage.
ELISION_GLYPHS = frozenset("\u0313\u1FBD\u02BC\u2019")
KORONIS = "\u1FBD"

# Unicode combining categories we strip when normalising a form for
# accent-blind prefix comparison.
_COMBINING = "Mn"

# The ten canonical particle / preposition mappings. These are pinned so
# the iconic entries (``ἀλλ᾽``, ``κατ᾽`` and friends) use the textbook
# spelling regardless of what variant wins the corpus vote. Listed as
# {full: elided} where both strings are NFC; the elided form carries
# U+1FBD GREEK KORONIS as its final scalar, matching the canonical
# storage glyph used by ``HunspellCompiler.normalizeEntry`` and by
# ``GreekStyle.elidableParticles``.
CANONICAL_ELISION_OVERRIDES: dict[str, str] = {
    "ἀλλά": "ἀλλ" + KORONIS,
    "διά":  "δι" + KORONIS,
    "ἐπί":  "ἐπ" + KORONIS,
    "κατά": "κατ" + KORONIS,
    "μετά": "μετ" + KORONIS,
    "παρά": "παρ" + KORONIS,
    "ὑπό":  "ὑπ" + KORONIS,
    "ἀπό":  "ἀπ" + KORONIS,
    "ἀντί": "ἀντ" + KORONIS,
    "δέ":   "δ" + KORONIS,
}

# Extra forms that classical grammar flags as movable-nu-eligible but
# that corpus tagging misses. ``εἴκοσι`` is the only one that survives
# cleanly, so we keep the list short.
EXTRA_NU_FORMS: frozenset[str] = frozenset([
    "εἴκοσι",
])


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _strip_lower(s: str) -> str:
    """Return a lowercased, diacritic-stripped copy of ``s``.

    Used for accent-blind prefix matching during elision pair
    derivation. The goal is to let ``κατά`` (full) match ``κατ᾽``
    (elided, stripped stem ``κατ``) whether or not either carries a
    breathing or final accent.
    """
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != _COMBINING).lower()


def _last_base_vowel(s: str) -> str | None:
    """Return the lowercased base vowel at the end of ``s`` once all
    trailing combining marks are skipped, or None if the last base
    character isn't one of the seven Greek vowels.
    """
    nfd = unicodedata.normalize("NFD", s)
    for c in reversed(nfd):
        if unicodedata.category(c) != _COMBINING:
            lower = c.lower()
            return lower if lower in "αεηιουω" else None
    return None


def _has_breathing(s: str) -> bool:
    nfd = unicodedata.normalize("NFD", s)
    return any(ord(c) in (0x0313, 0x0314) for c in nfd)


def _has_accent(s: str) -> bool:
    nfd = unicodedata.normalize("NFD", s)
    return any(ord(c) in (0x0300, 0x0301, 0x0342) for c in nfd)


def _canon_elided(s: str) -> str:
    """Normalise the trailing elision glyph to U+1FBD."""
    if not s:
        return s
    if s[-1] in ELISION_GLYPHS:
        return s[:-1] + KORONIS
    return s


def _score_elision_variant(full: str, elided: str) -> int:
    """Rank an elided candidate against its full form. Higher is better.

    Greek editing practice prefers the elided form's casing and
    breathing profile to track the full form. For ``Αὐτός`` (cap + smooth)
    we want ``Αὐτ᾽`` not ``αὐτ᾽``; for ``ἀλλά`` (lower + smooth) we want
    ``ἀλλ᾽`` not ``Ἀλλ᾽``. A secondary preference for an elided form
    that retains its own accent distinguishes corpus-legitimate
    ``μετ᾽`` / ``παρ᾽`` entries from noise-stripped ``μετ`` / ``παρ``
    variants that leaked in without their final accent.
    """
    score = 0
    # Same case on the first letter
    if full[:1].isupper() == elided[:1].isupper():
        score += 10
    # Breathing matches (both present or both absent)
    if _has_breathing(full) == _has_breathing(elided):
        score += 5
    # Prefer elided forms that kept an accent (attested typography)
    if _has_accent(elided):
        score += 2
    # Penalise elided forms that start with an elision glyph (OCR junk
    # where a leading combining mark wasn't reattached to its base)
    if elided[:1] in ELISION_GLYPHS:
        score -= 20
    return score


# Tags considered disqualifying for movable nu. Movable nu never
# attaches to subjunctive, optative, imperative, infinitive, or
# participle forms. Indicative past / present (where the letter
# conditions allow) is the carrier.
_NU_DISQUALIFIERS = frozenset([
    "subjunctive", "optative", "imperative", "infinitive", "participle",
])


def _derive_nu_forms(pairs_files: list[Path]) -> set[str]:
    """Union set of NFC surface forms eligible for movable nu.

    Linguistic rules applied (Smyth Greek Grammar 134):

      1. Verb 3sg active indicative past (imperfect / aorist /
         pluperfect / perfect) ending in ``-ε``.
      2. Verb 3sg active indicative present ending in ``-σι`` or
         ``-τι`` (``ἐστί``, ``δίδωσι``, ``τίθησι``).
      3. Verb 3pl active indicative (present / future / perfect)
         ending in ``-σι``.
      4. Noun / pronoun / adjective dative plural ending in ``-σι``,
         ``-ξι``, or ``-ψι``.

    Subjunctive, optative, imperative, infinitive, and participle tags
    on the same token disqualify it outright, since none of those
    moods / forms take movable nu even when the surface spelling ends
    in a qualifying letter. Relies on GLAUx / Diorisis morphological
    tagging; no heuristic falls back to raw frequency counts because
    corpus co-occurrence alone produces too many false positives on
    neuter nominative participles (e.g. ``γραφέν``) that share the
    ``-ε`` surface with a 3sg past.
    """
    nu_eligible: set[str] = set()
    for p in pairs_files:
        with open(p, encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            form = _nfc(entry.get("form", "").strip())
            if not form or len(form) < 2 or has_editorial_sigla(form):
                continue
            # Trailing nu never needs a second movable nu. Skip.
            if form.endswith("ν") or form.endswith("Ν"):
                continue
            tags = set(entry.get("tags", []))
            pos = entry.get("pos", "")

            # Disqualify: never-nu moods / forms.
            if tags & _NU_DISQUALIFIERS:
                continue

            last = _last_base_vowel(form)
            stripped = _strip_lower(form)

            # Rule 1: verb 3sg active indicative past (-ε)
            if (pos == "verb"
                    and "third-person" in tags and "singular" in tags
                    and "active" in tags and "indicative" in tags
                    and (tags & {"imperfect", "aorist",
                                  "pluperfect", "perfect"})
                    and last == "ε"):
                nu_eligible.add(form)
                continue

            # Rule 2: verb 3sg active indicative present (-σι / -τι)
            if (pos == "verb"
                    and "third-person" in tags and "singular" in tags
                    and "active" in tags and "indicative" in tags
                    and "present" in tags
                    and last == "ι"
                    and (stripped.endswith("σι")
                         or stripped.endswith("τι"))):
                nu_eligible.add(form)
                continue

            # Rule 3: verb 3pl active indicative (present / future / perfect) (-σι)
            if (pos == "verb"
                    and "third-person" in tags and "plural" in tags
                    and "active" in tags and "indicative" in tags
                    and (tags & {"present", "future", "perfect"})
                    and last == "ι" and stripped.endswith("σι")):
                nu_eligible.add(form)
                continue

            # Rule 4: noun / pron / adj dative plural (-σι / -ξι / -ψι)
            if (pos in ("noun", "pron", "adj")
                    and "dative" in tags and "plural" in tags):
                if stripped.endswith(("σι", "ξι", "ψι")):
                    nu_eligible.add(form)
                    continue

    # Closed-list numerals. Only ``εἴκοσι`` survives the morphological
    # audit; other -κοντα cardinals end in α, not ι / ε, so movable nu
    # doesn't apply.
    for w in EXTRA_NU_FORMS:
        nu_eligible.add(_nfc(w))

    return nu_eligible


def _load_lemma_forms(
    pairs_files: list[Path], lookup_db: Path | None,
) -> dict[str, set[str]]:
    """Collect {lemma -> set of NFC surface forms} from all sources.

    Merging morpho-tagged corpora and the Wiktionary-expanded lookup
    table gives us the widest possible net for elision-pair discovery.
    The lookup table contributes forms that corpus tagging misses,
    typically rarer inflections that a Wiktionary Lua paradigm
    generated. Malformed tokens (OCR artifacts with brackets or
    newlines) are dropped.
    """
    lemma_to_forms: dict[str, set[str]] = defaultdict(set)

    for p in pairs_files:
        with open(p, encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            form = _nfc(entry.get("form", "").strip())
            lemma = _nfc(entry.get("lemma", "").strip())
            if (not form or not lemma
                    or has_editorial_sigla(form)
                    or has_editorial_sigla(lemma)
                    or any(c in form or c in lemma for c in "{}<>\n")):
                continue
            lemma_to_forms[lemma].add(form)

    if lookup_db is not None and lookup_db.exists():
        conn = sqlite3.connect(str(lookup_db))
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT k.form, l.text FROM lookup k "
            "JOIN lemmas l ON k.lemma_id = l.id WHERE k.src='grc'"
        )
        for form, lemma in rows:
            nfc = _nfc(form)
            lemma_nfc = _nfc(lemma)
            if (not nfc or not lemma_nfc
                    or has_editorial_sigla(nfc)
                    or has_editorial_sigla(lemma_nfc)
                    or any(c in nfc or c in lemma_nfc for c in "{}<>\n")):
                continue
            lemma_to_forms[lemma_nfc].add(nfc)
        conn.close()

    return lemma_to_forms


def _derive_elision_pairs(
    lemma_to_forms: dict[str, set[str]],
) -> dict[str, str]:
    """Return {full_form_nfc -> elided_form_nfc_with_koronis}.

    For each lemma we partition its attested forms into *fulls* (end
    without an elision glyph) and *elideds* (end with one of the four
    elision glyphs; we canonicalise all four to U+1FBD on storage). A
    full form F matches an elided form E when:

      - ``strip(F)`` starts with ``strip(E_stem)`` where ``E_stem`` is
        E minus its final koronis;
      - ``strip(F)`` is exactly one character longer than
        ``strip(E_stem)``, i.e. F has exactly one extra base
        character relative to the elided stem;
      - that extra character is one of the seven Greek vowels, so F's
        final vowel is what elision dropped.

    When a full form matches multiple elided candidates we pick the
    one that best tracks the full form's casing and breathing profile
    via ``_score_elision_variant``. Canonical overrides for the ten
    iconic particles are applied last so the textbook forms always win
    regardless of corpus noise.
    """
    pairs: dict[str, str] = {}

    for lemma, forms in lemma_to_forms.items():
        if has_editorial_sigla(lemma):
            continue
        elideds = set()
        fulls: list[str] = []
        for f in forms:
            if not f or has_editorial_sigla(f):
                continue
            if f[-1] in ELISION_GLYPHS:
                elideds.add(_canon_elided(f))
            else:
                fulls.append(f)
        if not elideds:
            continue
        for full in fulls:
            if len(full) < 2:
                continue
            stripped_full = _strip_lower(full)
            # Find elided candidates whose stripped stem is exactly
            # one vowel shorter than the full form.
            candidates: list[str] = []
            for e in elideds:
                stem_e = e[:-1]  # drop koronis
                if not stem_e:
                    continue
                stripped_stem = _strip_lower(stem_e)
                if (stripped_full.startswith(stripped_stem)
                        and len(stripped_full) - len(stripped_stem) == 1):
                    if stripped_full[-1] in "αεηιουω":
                        candidates.append(e)
            if not candidates:
                continue
            best = max(
                candidates, key=lambda e: _score_elision_variant(full, e)
            )
            pairs[full] = best

    # Pin canonical overrides last. NFC everything for safety.
    for full, elided in CANONICAL_ELISION_OVERRIDES.items():
        pairs[_nfc(full)] = _nfc(elided)

    return pairs


def build(out_dir: Path) -> dict:
    """Drive the full morphology export end to end and write
    ``<out_dir>/grc_morph.json``. Returns a stats dict.
    """
    if not GLAUX_PAIRS.exists():
        print(
            f"ERROR: {GLAUX_PAIRS} not found. Download with "
            f"`huggingface-cli download ciscoriordan/dilemma --local-dir . "
            f"--include 'data/*'`",
            file=sys.stderr,
        )
        sys.exit(1)

    pairs_files = [GLAUX_PAIRS, DIORISIS_PAIRS]
    pairs_files = [p for p in pairs_files if p.exists()]
    print(
        f"Reading morphology tags from {len(pairs_files)} file(s): "
        f"{', '.join(p.name for p in pairs_files)}"
    )

    nu_forms = _derive_nu_forms(pairs_files)
    print(f"  nu-eligible surface forms: {len(nu_forms):,}")

    lemma_to_forms = _load_lemma_forms(
        pairs_files, LOOKUP_DB if LOOKUP_DB.exists() else None
    )
    print(f"  lemmas with attested forms: {len(lemma_to_forms):,}")

    elision_pairs = _derive_elision_pairs(lemma_to_forms)
    print(f"  elision full -> elided pairs: {len(elision_pairs):,}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grc_morph.json"
    payload = {
        "version": 1,
        "nu": sorted(nu_forms),
        "el": dict(sorted(elision_pairs.items())),
    }
    # Compact JSON: no extra whitespace, but pretty-ish for diffing.
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            payload, f, ensure_ascii=False, separators=(",", ":"),
            sort_keys=False,
        )
    size = out_path.stat().st_size
    print(f"  wrote {out_path} ({size:,} bytes)")

    return {
        "nu_count": len(nu_forms),
        "elision_count": len(elision_pairs),
        "bytes": size,
        "path": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir", default=str(OUT),
        help=f"Output directory (default: {OUT})",
    )
    args = ap.parse_args()
    build(Path(args.out_dir))


if __name__ == "__main__":
    main()

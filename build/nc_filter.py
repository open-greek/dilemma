"""Identify GLAUx source texts excluded from openly licensed builds.

GLAUx aggregates several dependency treebanks; most are CC BY-SA, but two
classes of texts are excluded from every build:

* NonCommercial source texts (CC BY-NC-SA / CC BY-NC-ND in SOURCE_LICENSE).
* PROIEL-derived treebank annotations (TREEBANK_ANNOTATIONS = 'PROIEL'):
  25 works / 364K tokens (the New Testament books and Herodotus' Historiae)
  whose trees GLAUx homogenized from the PROIEL treebank, which is
  CC BY-NC-SA and banned from this project in every role. The NT coverage
  is replaced by the Nestle-1904 macula trees (CC BY, already ingested) and
  Herodotus by Diorisis annotations.

GLAUx records both fields in metadata.txt, keyed by TLG, which equals the
GLAUx xml filename stem (e.g. ``0068-001`` -> ``0068-001.xml``). So the
returned stems can be matched directly against ``xmlfile.stem``.

``excluded_glaux_stems`` (NC + PROIEL-derived) is the single source of truth,
imported by the pair, frequency, and attestation builders. Gorman is excluded
at its own ingestion points as the held-out gold corpus (a separate input,
not part of GLAUx).
"""

import csv
import re
from pathlib import Path

# All NonCommercial CC variants are spelled "CC BY-NC-...". Match the BY-NC
# token specifically so permissive licenses (CC BY-SA, CC BY) are never caught.
_NC_RE = re.compile(r"BY-NC|NonCommercial", re.I)


def nc_glaux_stems(metadata_path) -> set:
    """Set of GLAUx file stems (== TLG) whose SOURCE_LICENSE is NonCommercial.

    Empty set if the metadata file is missing.
    """
    stems = set()
    p = Path(metadata_path)
    if not p.exists():
        return stems
    text = p.read_text(encoding="utf-8")
    for row in csv.DictReader(text.splitlines(), delimiter="\t"):
        if _NC_RE.search(row.get("SOURCE_LICENSE") or ""):
            tlg = (row.get("TLG") or "").strip()
            if tlg:
                stems.add(tlg)
    return stems


def proiel_glaux_stems(metadata_path) -> set:
    """Set of GLAUx file stems whose treebank annotation derives from the
    PROIEL treebank (TREEBANK_ANNOTATIONS = 'PROIEL'): re-exported PROIEL
    trees, banned from this project in every role.

    Empty set if the metadata file is missing.
    """
    stems = set()
    p = Path(metadata_path)
    if not p.exists():
        return stems
    text = p.read_text(encoding="utf-8")
    for row in csv.DictReader(text.splitlines(), delimiter="\t"):
        if "PROIEL" in (row.get("TREEBANK_ANNOTATIONS") or ""):
            tlg = (row.get("TLG") or "").strip()
            if tlg:
                stems.add(tlg)
    return stems


def excluded_glaux_stems(metadata_path) -> set:
    """All GLAUx stems excluded from builds: NonCommercial source texts plus
    PROIEL-derived treebank annotations. This is the set every GLAUx reader
    must skip."""
    return nc_glaux_stems(metadata_path) | proiel_glaux_stems(metadata_path)


def gorman_glaux_stems(metadata_path) -> set:
    """Set of GLAUx file stems whose MANUAL sentences derive from the Gorman
    treebanks (TREEBANK_ANNOTATIONS mentions Gorman).

    Gorman is this project's HELD-OUT GOLD corpus: no shipped artifact may
    derive from its trees. GLAUx homogenized Gorman's trees into these works'
    ``analysis="manual"`` sentences, so every GLAUx reader must skip the
    manual sentences of these stems (the auto sentences are model output,
    which is acceptable). Whole-work exclusion is NOT wanted here - the auto
    portions are large (e.g. Thucydides: 34K manual + 136K auto tokens) and
    openly licensed.
    """
    stems = set()
    p = Path(metadata_path)
    if not p.exists():
        return stems
    text = p.read_text(encoding="utf-8")
    for row in csv.DictReader(text.splitlines(), delimiter="\t"):
        if "Gorman" in (row.get("TREEBANK_ANNOTATIONS") or ""):
            tlg = (row.get("TLG") or "").strip()
            if tlg:
                stems.add(tlg)
    return stems


# A TEI file's per-text license lives in the publicationStmt; PTA records it as
# <licence target="https://creativecommons.org/licenses/by-nc-.../">. NC files
# carry a "by-nc" target URL (the human-readable wording is inconsistent, so we
# key on the canonical @target). Most PTA Greek editions are CC BY / BY-SA; this
# drops the handful (currently just pta0036) that are NonCommercial.
_TEI_NC_RE = re.compile(rb'<(?:\w+:)?licence\b[^>]*?target\s*=\s*"[^"]*by-nc',
                        re.I)


def tei_is_noncommercial(data: bytes) -> bool:
    """True if a TEI file's licence/@target is a CC BY-NC variant.

    Pass the raw file bytes; only the header region is scanned (the
    publicationStmt sits near the top of teiHeader). Used by the PTA freq and
    form-attestation readers to skip NonCommercial texts in commercial builds.
    """
    return bool(_TEI_NC_RE.search(data[:65536]))

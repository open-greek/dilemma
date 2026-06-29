"""Identify NonCommercial-licensed GLAUx source texts, for openly licensed builds.

GLAUx aggregates several dependency treebanks; most are CC BY-SA, but a handful
of source texts carry NonCommercial terms (CC BY-NC-SA / CC BY-NC-ND). For a
build whose artifacts may be used commercially, those texts are excluded.

GLAUx records the per-text license in metadata.txt (column SOURCE_LICENSE),
keyed by TLG, which equals the GLAUx xml filename stem (e.g. ``0068-001`` ->
``0068-001.xml``). So the returned stems can be matched directly against
``xmlfile.stem``.

This is the single source of truth for "which GLAUx texts are NonCommercial",
imported by the pair, frequency, and attestation builders. PROIEL, Gorman, and
the UD Perseus release are wholly NonCommercial and are excluded at their own
ingestion points (they are separate inputs, not part of GLAUx).
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

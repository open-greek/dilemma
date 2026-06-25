"""Locus-aware Greek token iterator for TEI corpora.

Unlike ``tei_tokenize`` (which returns accent-stripped, lowercased keys for the
form-frequency builders and discards position), this yields, for every Greek
word in a TEI ``<text>``, the EXACT NFC polytonic surface form together with its
passage locus, reconstructed from the enclosing citation structure:

  * the stack of ``<div type="textpart" subtype=... n=...>`` (book / chapter /
    section / ...), and
  * the nearest enclosing ``<l n=...>`` verse line.

``<lb>`` / ``<milestone>`` / ``<pb>`` are treated as edition/layout markers, not
citation units, so they don't enter the locus. Editorial / non-text elements
(``teiHeader``, ``note``, ``app``, ``rdg``, ``bibl``, ...) are skipped.

Used by ``build_form_attestation.py`` to add First1KGreek, canonical-greekLit
and PTA. Needs ``lxml``.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterator, Optional, Tuple

from lxml import etree

TEI_NS = "http://www.tei-c.org/ns/1.0"

# Elements whose textual content is editorial / non-lexical and must not be
# tokenized (mirrors build/tei_tokenize.py, plus heads/titles).
SKIP_TAGS = {
    "teiHeader", "note", "app", "rdg", "witDetail", "bibl", "fw",
    "abbr", "orig", "head", "title", "speaker", "label", "ref",
    "gap", "del", "figure", "figDesc",
}

# A run of Greek letters with optional internal/trailing elision apostrophe.
_GREEK_RUN = re.compile(r"[Ͱ-Ͽἀ-῿]+['’ʼ᾽]?")


def _local(tag) -> str:
    if isinstance(tag, str):
        return tag.rsplit("}", 1)[-1]
    return ""


def _locus(div_stack, line_n) -> Tuple[Optional[str], Optional[str]]:
    parts, scheme = [], []
    for sub, n in div_stack:
        if n:
            parts.append(n)
            scheme.append(sub or "div")
    if line_n:
        parts.append(line_n)
        scheme.append("line")
    if not parts:
        return None, None
    return ".".join(parts), ".".join(scheme)


def _emit(text, div_stack, line_n, out):
    if not text:
        return
    locus, scheme = _locus(div_stack, line_n)
    for m in _GREEK_RUN.finditer(text):
        form = unicodedata.normalize("NFC", m.group(0))
        out.append((form, locus, scheme))


def _walk(el, div_stack, state, out):
    tag = _local(el.tag)
    if tag in SKIP_TAGS or not isinstance(el.tag, str):  # skip comments/PIs too
        return
    if tag == "div" and el.get("type") == "textpart":
        div_stack = div_stack + [(el.get("subtype") or "div", el.get("n") or "")]
        state = {"line": None}  # fresh line context inside a new division
    elif tag == "l":
        state["line"] = el.get("n")
    if el.text:
        _emit(el.text, div_stack, state["line"], out)
    for child in el:
        _walk(child, div_stack, state, out)
        if child.tail:
            _emit(child.tail, div_stack, state["line"], out)


def iter_tokens(root) -> Iterator[Tuple[str, Optional[str], Optional[str]]]:
    """Yield ``(form, locus, locus_scheme)`` for every Greek word in the TEI
    body. ``form`` is exact NFC polytonic; lexical filtering is left to the
    caller (the builder's form filter), so punctuation-only runs never occur
    here but stray non-lexical Greek runs may."""
    body = None
    for el in root.iter():
        if isinstance(el.tag, str) and _local(el.tag) == "body":
            body = el
            break
    if body is None:
        return
    out: list = []
    _walk(body, [], {"line": None}, out)
    yield from out


# ---------------------------------------------------------------------------
# Work metadata
# ---------------------------------------------------------------------------


def _year_from_creation(root) -> Optional[int]:
    """Composition year from <profileDesc><creation><date>, or None.

    Uses @when, else the midpoint of @notBefore/@notAfter (PTA's convention).
    """
    for el in root.iter():
        if not isinstance(el.tag, str) or _local(el.tag) != "date":
            continue
        parent = el.getparent()
        if parent is None or _local(parent.tag) != "creation":
            continue
        when = el.get("when")
        nb, na = el.get("notBefore"), el.get("notAfter")
        try:
            if when:
                return int(when)            # int("0401") -> 401, int("-450") -> -450
            if nb and na:
                return (int(nb) + int(na)) // 2
            if nb:
                return int(nb)
            if na:
                return int(na)
        except ValueError:
            return None
    return None


_URN_TLG = re.compile(r"greekLit:tlg(\d{1,4})\.tlg(\d{1,3})")
_URN_PTA = re.compile(r":(pta\d{1,4}\.pta\d{1,3})")
_FK_TLG = re.compile(r"\btlg(\d{1,4})[-.]tlg(\d{1,3})\b")


def work_meta(root, filename: str) -> dict:
    """Best-effort work metadata: urn, a TLG author-work id (NNNN-NNN, to join
    GLAUx) when present, author/title, and a composition century when the header
    carries one (PTA does; Perseus/First1K usually don't)."""
    urn = None
    for el in root.iter():
        if (isinstance(el.tag, str) and _local(el.tag) == "div"
                and el.get("type") == "edition" and el.get("n")):
            urn = el.get("n")
            break
    tlg_id = None
    blob = f"{urn or ''} {filename}"
    # explicit <idno type="TLG"> (PTA maps patristic works to TLG)
    for el in root.iter():
        if (isinstance(el.tag, str) and _local(el.tag) == "idno"
                and (el.get("type") or "").upper() == "TLG" and el.text):
            blob += " " + el.text
            break
    m = _URN_TLG.search(blob) or _FK_TLG.search(blob)
    if m:
        tlg_id = f"{int(m.group(1)):04d}-{int(m.group(2)):03d}"

    def _first_text(localname, ancestor=None):
        for el in root.iter():
            if not isinstance(el.tag, str) or _local(el.tag) != localname:
                continue
            if ancestor is not None:
                p = el.getparent()
                if p is None or _local(p.tag) != ancestor:
                    continue
            if el.text and el.text.strip():
                return el.text.strip()
        return None

    return {
        "urn": urn,
        "tlg_id": tlg_id,
        "author": _first_text("author") or _first_text("persName"),
        "title": _first_text("title"),
        "creation_year": _year_from_creation(root),
    }


def parse_file(path: Path):
    """Return (root, work_meta) for a TEI file, or (None, None) on parse error."""
    try:
        root = etree.parse(str(path)).getroot()
    except (etree.XMLSyntaxError, OSError):
        return None, None
    return root, work_meta(root, path.name)

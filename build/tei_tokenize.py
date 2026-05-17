"""Greek-token extractor for TEI XML files.

Walks each `<TEI>/<text>` subtree, joins text nodes, splits on anything
that isn't a Greek letter or an apostrophe (so elision forms like δ',
ἀλλ' stay intact), and returns a Counter keyed by NFC accent-stripped
lowercase tokens — the same key shape `corpus_freq.json` uses.

Skips elements that hold non-text or editorial content (`teiHeader`,
`note`, `app`, `rdg`, `witDetail`, `bibl`, `cit/bibl`). Inside the body
we keep everything else; non-Greek tokens (Latin, German notes that
leaked through) are filtered out at the token level.

Used by build_first1kgreek_freq.py and build_pta_freq.py.
"""

import re
import unicodedata
from collections import Counter
from pathlib import Path

from lxml import etree

TEI_NS = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI_NS}

SKIP_TAGS = {
    f"{{{TEI_NS}}}{t}"
    for t in (
        "teiHeader", "note", "app", "rdg", "witDetail",
        "bibl", "fw", "abbr", "orig",
    )
}

GREEK_RUN = re.compile(
    r"["
    r"Ͱ-Ͽ"   # Greek and Coptic
    r"ἀ-῿"   # Greek Extended
    r"'’ʼ"   # apostrophes (elision)
    r"]+"
)
HAS_GREEK_LETTER = re.compile(r"[Ͱ-Ͽἀ-῿]")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def normalize_key(token: str) -> str:
    """Match the corpus_freq.json key convention: lowercase, accent-stripped,
    NFC. Trailing/leading apostrophes that don't sit between two letters get
    dropped (those are usually OCR/quotation marks, not elision)."""
    t = strip_accents(token).lower()
    t = unicodedata.normalize("NFC", t)
    if not t or not HAS_GREEK_LETTER.search(t):
        return ""
    # Normalize all apostrophe-like marks to U+2019 (right single quote),
    # which is what GLAUx and Diorisis store for elision. Keeps merge keys
    # aligned with the existing corpora.
    t = t.replace("'", "’").replace("ʼ", "’").replace("᾽", "’")
    return t


def _text_of(elem) -> str:
    """Concatenate text from an element, skipping SKIP_TAGS subtrees."""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        if child.tag not in SKIP_TAGS:
            parts.append(_text_of(child))
        if child.tail:
            parts.append(child.tail)
    return " ".join(parts)


def tokenize_tei(path: Path) -> Counter:
    """Parse one TEI file, return Counter of accent-stripped lowercase forms."""
    try:
        tree = etree.parse(str(path))
    except etree.XMLSyntaxError:
        return Counter()
    root = tree.getroot()
    counts: Counter = Counter()
    # Only iterate the <text> subtrees (skip <teiHeader> entirely).
    for text_elem in root.iter(f"{{{TEI_NS}}}text"):
        body_text = _text_of(text_elem)
        for run in GREEK_RUN.findall(body_text):
            for tok in run.split():
                key = normalize_key(tok)
                if key:
                    counts[key] += 1
    return counts

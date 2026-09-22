#!/usr/bin/env python3
"""Build a revision-pinned held-out corpus gate for the grc dictionary.

The fixture stores exact, accent-preserving surface counts rather than source
prose. Contextual grave is folded to acute and textual apostrophes are
canonicalized exactly as in the release audit. Forms rejected by the explicit
Hunspell orthography policy are counted in metadata but are not coverage
requirements.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from export_hunspell import (  # noqa: E402
    BARE_ELISION_STEMS,
    SPACING_DIACRITICS,
    grc_orthography_reason,
)
from scripts.audit_hunspell_frequency import lookup_form, sha256  # noqa: E402
from scripts.build_hunspell_compatibility_fixture import (  # noqa: E402
    read_version_sidecar,
)

DEFAULT_OGC = Path.home() / "Documents" / "open-greek-corpus"
DEFAULT_OUTPUT = ROOT / "tests" / "fixtures" / "hunspell_heldout.json.gz"
SKIP_LOCAL_NAMES = {
    "teiHeader", "note", "app", "rdg", "witDetail", "bibl", "fw",
    "abbr", "orig",
}
APOSTROPHES = frozenset("'`\u2019\u02bc\u1fbd")


def _is_greek_letter(char: str) -> bool:
    return (
        unicodedata.category(char).startswith("L")
        and "GREEK" in unicodedata.name(char, "")
    )


def exact_tokens(text: str) -> list[str]:
    """Extract Greek words without carrying adjacent punctuation."""
    tokens: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current and any(_is_greek_letter(char) for char in current):
            token = lookup_form(unicodedata.normalize("NFC", "".join(current)))
            if token:
                tokens.append(token)
        current.clear()

    for char in unicodedata.normalize("NFC", text):
        category = unicodedata.category(char)
        if _is_greek_letter(char) or (category == "Mn" and current):
            current.append(char)
            continue
        if ord(char) in SPACING_DIACRITICS:
            if current:
                current.append(char)
                flush()
            else:
                current.append(char)
            continue
        if char in APOSTROPHES and current:
            current.append(char)
            flush()
            continue
        flush()
    flush()
    return tokens


def _element_text(element: etree._Element) -> str:
    parts: list[str] = []
    if element.text:
        parts.append(element.text)
    for child in element:
        local = etree.QName(child).localname
        if local not in SKIP_LOCAL_NAMES:
            parts.append(_element_text(child))
        if child.tail:
            parts.append(child.tail)
    return " ".join(parts)


def tei_counts(path: Path, *, book_one: bool = False) -> Counter[str]:
    tree = etree.parse(str(path))
    if book_one:
        roots = tree.xpath(
            "//*[local-name()='div' and @n='1' and "
            "translate(@subtype, 'BOOK', 'book')='book']"
        )
        if len(roots) != 1:
            raise ValueError(f"expected one book 1 div in {path}, got {len(roots)}")
    else:
        roots = tree.xpath("//*[local-name()='text']")
        if not roots:
            raise ValueError(f"no TEI text element in {path}")
    counts: Counter[str] = Counter()
    for element in roots:
        counts.update(exact_tokens(_element_text(element)))
    return counts


def plain_counts(path: Path) -> Counter[str]:
    return Counter(exact_tokens(path.read_text(encoding="utf-8")))


def _manifest(paths: list[Path], root: Path) -> dict:
    rows = [
        {"path": str(path.relative_to(root)), "sha256": sha256(path)}
        for path in sorted(paths)
    ]
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {"files": rows, "manifest_sha256": digest.hexdigest()}


def _coverage(counts: Counter[str], dictionary) -> dict:
    accepted_types = 0
    accepted_tokens = 0
    for form, count in counts.items():
        if dictionary.lookup(form):
            accepted_types += 1
            accepted_tokens += count
    return {
        "accepted_types": accepted_types,
        "rejected_types": len(counts) - accepted_types,
        "accepted_tokens": accepted_tokens,
        "rejected_tokens": sum(counts.values()) - accepted_tokens,
    }


def _review_counts(raw: Counter[str]) -> tuple[Counter[str], dict[str, int]]:
    kept: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    for form, count in raw.items():
        reason = grc_orthography_reason(form)
        if form in BARE_ELISION_STEMS:
            reason = "bare_elision"
        if reason is None:
            kept[form] += count
        else:
            excluded[reason] += count
    return kept, dict(sorted(excluded.items()))


def _corpus_payload(counts: Counter[str], baseline) -> dict:
    reviewed, excluded = _review_counts(counts)
    return {
        "total_types": len(reviewed),
        "total_tokens": sum(reviewed.values()),
        "policy_excluded_tokens": excluded,
        "baseline": _coverage(reviewed, baseline),
        "forms": [[form, count] for form, count in sorted(reviewed.items())],
    }


def write_payload(path: Path, payload: dict) -> None:
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ogc-root", type=Path, default=DEFAULT_OGC)
    parser.add_argument(
        "--baseline", type=Path, required=True,
        help="path stem of the shipped grc_polytonic .dic/.aff/.version",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        from spylls.hunspell import Dictionary
    except ImportError as exc:
        raise RuntimeError("install spylls to build the fixture") from exc

    nt = sorted((
        args.ogc_root / "sources/perseus/data/tlg0031"
    ).glob("**/*.perseus-grc2.xml"))
    lxx = sorted((
        args.ogc_root / "sources/first1k/data/tlg0527"
    ).glob("**/*-grc1.xml"))
    iliad = (
        args.ogc_root / "sources/perseus/data/tlg0012/tlg001"
        / "tlg0012.tlg001.perseus-grc2.xml"
    )
    herodotus = (
        args.ogc_root / "sources/perseus/data/tlg0016/tlg001"
        / "tlg0016.tlg001.perseus-grc2.xml"
    )
    kath = ROOT / "data/benchmarks/katharevousa.txt"
    required = [iliad, herodotus, kath, *nt, *lxx]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing held-out source: " + str(missing[0]))

    version = read_version_sidecar(args.baseline)
    baseline = Dictionary.from_files(str(args.baseline))
    corpora = {
        "new_testament": _corpus_payload(
            sum((tei_counts(path) for path in nt), Counter()), baseline
        ),
        "septuagint": _corpus_payload(
            sum((tei_counts(path) for path in lxx), Counter()), baseline
        ),
        "iliad_book_1": _corpus_payload(
            tei_counts(iliad, book_one=True), baseline
        ),
        "herodotus_book_1": _corpus_payload(
            tei_counts(herodotus, book_one=True), baseline
        ),
        "katharevousa": _corpus_payload(plain_counts(kath), baseline),
    }
    payload = {
        "schema_version": 1,
        "normalization": (
            "NFC; textual elision apostrophe to U+1FBD; contextual grave "
            "to acute; case preserved"
        ),
        "baseline": {
            "version": version["version"],
            "commit": version["commit"],
            "dic_sha256": sha256(args.baseline.with_suffix(".dic")),
            "aff_sha256": sha256(args.baseline.with_suffix(".aff")),
        },
        "sources": {
            "new_testament": _manifest(nt, args.ogc_root),
            "septuagint": _manifest(lxx, args.ogc_root),
            "iliad_book_1": _manifest([iliad], args.ogc_root),
            "herodotus_book_1": _manifest([herodotus], args.ogc_root),
            "katharevousa": _manifest([kath], ROOT),
        },
        "corpora": corpora,
    }
    write_payload(args.output, payload)
    print(f"wrote {args.output}; sha256={sha256(args.output)}")
    for name, corpus in corpora.items():
        coverage = corpus["baseline"]
        print(
            f"{name}: {corpus['total_tokens']:,} tokens / "
            f"{corpus['total_types']:,} types; April rejects "
            f"{coverage['rejected_tokens']:,} tokens / "
            f"{coverage['rejected_types']:,} types"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

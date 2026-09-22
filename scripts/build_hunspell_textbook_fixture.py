#!/usr/bin/env python3
"""Build the pinned textbook-verb paradigm gate for Hunspell releases."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dilemma.form_sanitize import (  # noqa: E402
    has_editorial_sigla,
    resolve_editorial_form,
    sanitize_form,
)
from export_hunspell import (  # noqa: E402
    GRC_COMPLETE_PARADIGM_LEMMAS,
    add_contextual_acute_twins,
    grc_pair_orthography_reason,
)
from scripts.audit_hunspell_frequency import sha256  # noqa: E402

DEFAULT_SOURCE = ROOT / "data" / "ag_verb_paradigms.json"
DEFAULT_REVIEW = ROOT / "data" / "hunspell_grc_textbook_review.json"
DEFAULT_OUTPUT = ROOT / "data" / "hunspell_grc_textbook.json.gz"
VOWEL_LENGTH_MARKS = frozenset({0x0304, 0x0306})


def strip_vowel_length(form: str) -> str:
    """Drop macron and breve: dictionary notation, not spelling."""
    nfd = unicodedata.normalize("NFD", form)
    return unicodedata.normalize(
        "NFC", "".join(ch for ch in nfd if ord(ch) not in VOWEL_LENGTH_MARKS)
    )


def build_payload(source: Path, review_path: Path) -> dict:
    raw = json.loads(source.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("schema_version") != 1:
        raise ValueError(f"unsupported textbook review: {review_path}")
    paradigms: dict[str, list[str]] = {}
    for lemma in sorted(GRC_COMPLETE_PARADIGM_LEMMAS):
        entry = raw.get(lemma)
        if not isinstance(entry, dict) or not isinstance(entry.get("forms"), dict):
            raise ValueError(f"missing textbook paradigm for {lemma}")
        removed = set(review["remove"].get(lemma, {}))
        pairs: list[tuple[str, str]] = [(lemma, lemma)]
        for form in entry["forms"].values():
            if not isinstance(form, str) or not form:
                continue
            resolved = resolve_editorial_form(form)
            if not resolved and not has_editorial_sigla(form):
                resolved = (sanitize_form(form),)
            pairs.extend(
                (strip_vowel_length(value), lemma)
                for value in resolved
                if value and value not in removed
                and strip_vowel_length(value) not in removed
            )
        pairs.extend((form, lemma) for form in review["add"].get(lemma, {}))
        pairs = [
            pair for pair in pairs
            if grc_pair_orthography_reason(*pair) is None
        ]
        pairs, _added = add_contextual_acute_twins(pairs)
        paradigms[lemma] = sorted(
            {form for form, _lemma in pairs} - removed
        )
    return {
        "schema_version": 1,
        "source": {
            "path": "data/ag_verb_paradigms.json",
            "sha256": sha256(source),
        },
        "review": {
            "path": "data/hunspell_grc_textbook_review.json",
            "sha256": sha256(review_path),
        },
        "selection": (
            "structurally valid forms, without vowel-length marks, for one "
            "regular omega verb, three athematic verbs, and alpha/epsilon/"
            "omicron contract verbs, after the reviewed removals and "
            "corroborated additions"
        ),
        "paradigms": paradigms,
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
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build_payload(args.source, args.review)
    write_payload(args.output, payload)
    count = sum(len(forms) for forms in payload["paradigms"].values())
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"wrote {args.output}: {count:,} forms, sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

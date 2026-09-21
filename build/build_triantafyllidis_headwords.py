#!/usr/bin/env python3
"""Build a bare Triantafyllidis headword inventory from the MDC dataset.

The source archive also contains pronunciation, dictionary text, page numbers,
and source URLs. Dilemma needs none of those: the Parquet reader projects only
``lemma``, and the generated artifacts contain only cleaned headwords, source
identity, checksums, and aggregate counts.

The ``lemma`` values are dictionary display headings. Cleaning therefore
recognizes only conventions proven by an audit of this source: a decimal
homograph index immediately after the citation form, and trailing inflection
endings whose every token begins with a hyphen (with ``/`` as a separator).
Bound forms and ambiguous multiword headings are rejected rather than guessed.

Usage:
    python build/build_triantafyllidis_headwords.py SOURCE.tar.gz
    python build/build_triantafyllidis_headwords.py SOURCE.parquet --stats
    python build/build_triantafyllidis_headwords.py SOURCE.tar.gz \
        --rejections /tmp/triantafyllidis-rejections.tsv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import tarfile
import tempfile
import unicodedata
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_PATH = DATA_DIR / "triantafyllidis_headwords.json"
VALIDATION_PATH = DATA_DIR / "mg_headwords.json"

DATASET_ID = "cmo1sklpo00d0mk07taoepe1a"
DATASET_URL = (
    "https://mozilladatacollective.com/datasets/"
    "cmo1sklpo00d0mk07taoepe1a"
)
SOURCE_NAME = "Modern Greek Dictionary (Triantafyllidis)"
SOURCE_LICENSE = "CC BY-NC-ND 4.0 (as declared by the dataset provider)"

DISPLAY_INDEX_RE = re.compile(r"[1-9][0-9]*$")
LATIN_HOMOGLYPHS = str.maketrans({
    "A": "Α", "B": "Β", "E": "Ε", "H": "Η", "I": "Ι", "K": "Κ",
    "M": "Μ", "N": "Ν", "O": "Ο", "P": "Ρ", "T": "Τ", "X": "Χ",
    "Y": "Υ", "Z": "Ζ",
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _is_greek_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L") and \
        "GREEK" in unicodedata.name(ch, "")


def _is_greek_word(word: str) -> bool:
    """Return whether each hyphen-separated segment is Greek letters/marks."""
    segments = word.split("-")
    if any(not segment for segment in segments):
        return False
    for segment in segments:
        saw_letter = False
        for ch in segment:
            if _is_greek_letter(ch):
                saw_letter = True
            elif unicodedata.category(ch).startswith("M") and saw_letter:
                continue
            else:
                return False
        if not saw_letter:
            return False
    return True


def _is_inflection_tail(tokens: list[str]) -> bool:
    if not tokens:
        return False
    expect_ending = True
    for token in tokens:
        if token == "/":
            if expect_ending:
                return False
            expect_ending = True
            continue
        if not token.startswith("-") or not _is_greek_word(token[1:]):
            return False
        expect_ending = False
    return not expect_ending


def clean_headword(raw: object, drops: Counter[str], repairs: Counter[str],
                   homoglyph_targets: set[str]) -> str | None:
    """Return one conservative bare headword, recording every decision."""
    if not isinstance(raw, str):
        drops["missing_or_nonstring"] += 1
        return None

    normalized = unicodedata.normalize("NFC", raw)
    heading = normalized.strip()
    if heading != normalized:
        repairs["outer_whitespace_trimmed"] += 1
    if not heading:
        drops["empty"] += 1
        return None

    tokens = heading.split()
    candidate = tokens.pop(0)
    if tokens and DISPLAY_INDEX_RE.fullmatch(tokens[0]):
        tokens.pop(0)
        repairs["display_index_removed"] += 1
    if tokens:
        if not _is_inflection_tail(tokens):
            drops["ambiguous_multiword"] += 1
            return None
        repairs["inflection_tail_removed"] += 1

    if candidate.startswith("-") or candidate.endswith("-"):
        drops["bound_form"] += 1
        return None

    if not _is_greek_word(candidate):
        repaired = candidate.translate(LATIN_HOMOGLYPHS)
        if (repaired != candidate and _is_greek_word(repaired)
                and repaired in homoglyph_targets):
            candidate = repaired
            repairs["validated_latin_homoglyph"] += 1
        else:
            drops["non_greek_residue"] += 1
            return None

    return unicodedata.normalize("NFC", candidate)


def _load_parquet_lemmas(path: Path) -> list[object]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit(
            "error: pyarrow is required to read the source; "
            "install it in the build environment"
        )

    parquet = pq.ParquetFile(path)
    if "lemma" not in parquet.schema_arrow.names:
        raise ValueError(f"Parquet source has no 'lemma' column: {path}")
    # This projection is the data boundary: no entry text, pronunciation,
    # page number, or source URL is read into the build process.
    return parquet.read(columns=["lemma"]).column("lemma").to_pylist()


@contextmanager
def _parquet_source(source: Path) -> Iterator[Path]:
    if source.suffix.lower() == ".parquet":
        yield source
        return

    try:
        archive = tarfile.open(source, mode="r:*")
    except tarfile.TarError as exc:
        raise ValueError(
            f"source must be a Parquet file or tar archive: {source}"
        ) from exc

    with archive:
        members = [
            member for member in archive.getmembers()
            if member.isfile() and member.name.lower().endswith(".parquet")
        ]
        if len(members) != 1:
            raise ValueError(
                f"expected exactly one Parquet member in {source}, "
                f"found {len(members)}"
            )
        member_file = archive.extractfile(members[0])
        if member_file is None:
            raise ValueError(f"could not read {members[0].name} from {source}")
        with member_file, tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            shutil.copyfileobj(member_file, tmp)
            tmp.flush()
            yield Path(tmp.name)


def load_lemmas(source: Path) -> list[object]:
    """Read only the source's ``lemma`` Parquet column."""
    with _parquet_source(source) as parquet_path:
        return _load_parquet_lemmas(parquet_path)


def load_validation_headwords(path: Path = VALIDATION_PATH) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8") as fh:
        return set(json.load(fh))


def build(source: Path, *, homoglyph_targets: set[str] | None = None
          ) -> tuple[list[str], dict, list[tuple[str, str]]]:
    raw_lemmas = load_lemmas(source)
    if homoglyph_targets is None:
        homoglyph_targets = load_validation_headwords()
    drops: Counter[str] = Counter()
    repairs: Counter[str] = Counter()
    cleaned: list[str] = []
    rejections: list[tuple[str, str]] = []

    for raw in raw_lemmas:
        before = drops.copy()
        headword = clean_headword(raw, drops, repairs, homoglyph_targets)
        if headword is not None:
            cleaned.append(headword)
            continue
        reason = next(
            (name for name, count in drops.items()
             if count > before.get(name, 0)),
            "unknown",
        )
        rejections.append(("" if raw is None else str(raw), reason))

    headwords = sorted(set(cleaned))
    distinct_raw = len({
        (type(value).__name__, repr(value)) for value in raw_lemmas
    })
    stats = {
        "source_rows": len(raw_lemmas),
        "distinct_raw_lemmas": distinct_raw,
        "clean_headwords": len(headwords),
        "duplicate_clean_headwords": len(cleaned) - len(headwords),
        "repaired_total": sum(repairs.values()),
        "repaired_by_reason": dict(sorted(repairs.items())),
        "dropped_total": sum(drops.values()),
        "dropped_by_reason": dict(sorted(drops.items())),
    }
    return headwords, stats, rejections


def meta_path_for(output: Path) -> Path:
    return output.with_name(f"{output.stem}.meta.json")


def write_artifacts(source: Path, output: Path, headwords: list[str],
                    stats: dict, validation_path: Path = VALIDATION_PATH) -> None:
    meta = {
        "source": SOURCE_NAME,
        "dataset_id": DATASET_ID,
        "dataset_url": DATASET_URL,
        "source_filename": source.name,
        "source_sha256": sha256_file(source),
        "source_license": SOURCE_LICENSE,
        "fields_read": ["lemma"],
        "homoglyph_validation": {
            "source": validation_path.name,
            "sha256": sha256_file(validation_path),
        },
        "stats": stats,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(headwords, ensure_ascii=False, indent=0) + "\n",
        encoding="utf-8",
    )
    meta_path_for(output).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_rejections(path: Path, rejections: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(("lemma", "reason"))
        writer.writerows(rejections)


def _print_stats(stats: dict) -> None:
    print("Triantafyllidis headwords:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for reason, count in value.items():
                print(f"    {reason}: {count:,}")
        else:
            print(f"  {key}: {value:,}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", type=Path,
                        help="MDC .tar.gz archive or its Parquet member")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="flat JSON headword-list output")
    parser.add_argument("--rejections", type=Path,
                        help="optional TSV audit of rejected lemma strings")
    parser.add_argument("--stats", action="store_true",
                        help="print stats only; do not write JSON artifacts")
    args = parser.parse_args(argv)

    if not args.source.is_file():
        parser.error(f"source not found: {args.source}")

    headwords, stats, rejections = build(args.source)
    _print_stats(stats)
    if args.rejections:
        write_rejections(args.rejections, rejections)
        print(f"wrote {len(rejections):,} rejections -> {args.rejections}")
    if args.stats:
        return

    write_artifacts(args.source, args.output, headwords, stats)
    print(f"wrote {len(headwords):,} headwords -> {args.output}")
    print(f"wrote metadata -> {meta_path_for(args.output)}")


if __name__ == "__main__":
    main()

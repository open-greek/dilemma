#!/usr/bin/env python3
"""Audit the grc Hunspell export against the language-model frequency head.

The LM vocabulary contains structural sentinels, punctuation-bearing tokens,
and source defects as well as words. This gate ranks lexical tokens made only
of Greek letters and combining marks, folds contextual grave accents to acute,
and aggregates counts after that normalization. Elision-mark normalization is
a separate consumer concern and is deliberately outside this fixture.

The committed fixture lets CI enforce the coverage invariant without shipping
the complete LM build. Its source hashes make regeneration against a different
vocabulary or unigram table explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dilemma import grave_to_acute  # noqa: E402
from dilemma.nonlexical import classify_nonlexical  # noqa: E402

DEFAULT_VOCAB = ROOT / "build" / "lm" / "vocab.json"
DEFAULT_UNIGRAMS = ROOT / "build" / "lm" / "unigrams.json"
DEFAULT_DICTIONARY = ROOT / "build" / "hunspell" / "grc_polytonic"
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "hunspell_lm_top100.json"
DEFAULT_TOP_N = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_greek_lexical_form(form: str) -> bool:
    """Return whether *form* belongs in the frequency coverage invariant."""
    if not form or classify_nonlexical(form) is not None:
        return False
    for char in unicodedata.normalize("NFD", form):
        category = unicodedata.category(char)
        if category == "Mn":
            continue
        if category.startswith("L") and "GREEK" in unicodedata.name(char, ""):
            continue
        return False
    return True


def normalize_form(form: str) -> str:
    return grave_to_acute(unicodedata.normalize("NFC", form))


def ranked_forms(vocab_path: Path, unigrams_path: Path) -> list[tuple[str, int]]:
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    unigrams = json.loads(unigrams_path.read_text(encoding="utf-8"))
    if not isinstance(vocab, list) or not isinstance(unigrams, dict):
        raise ValueError("vocab must be a list and unigrams must be an object")

    counts: Counter[str] = Counter()
    for raw_index, raw_count in unigrams.items():
        index = int(raw_index)
        if index < 0 or index >= len(vocab):
            raise ValueError(f"unigram index {index} is outside the vocabulary")
        form = vocab[index]
        if not isinstance(form, str) or not is_greek_lexical_form(form):
            continue
        counts[normalize_form(form)] += int(raw_count)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def fixture_payload(
    vocab_path: Path,
    unigrams_path: Path,
    top_n: int = DEFAULT_TOP_N,
) -> dict:
    forms = ranked_forms(vocab_path, unigrams_path)[:top_n]
    if len(forms) != top_n:
        raise ValueError(f"only {len(forms)} lexical forms available for top {top_n}")
    return {
        "schema_version": 1,
        "top_n": top_n,
        "selection": "Greek letters and combining marks only; nonlexical tokens excluded",
        "normalization": "NFC; combining grave U+0300 folded to acute U+0301; counts aggregated",
        "sources": {
            "vocab.json": {"sha256": sha256(vocab_path)},
            "unigrams.json": {"sha256": sha256(unigrams_path)},
        },
        "forms": [{"form": form, "count": count} for form, count in forms],
    }


def audit_dictionary(dictionary_base: Path, fixture: dict) -> list[str]:
    try:
        from spylls.hunspell import Dictionary
    except ImportError as exc:  # pragma: no cover - exercised by CLI users
        raise RuntimeError("install spylls to audit the expanded dictionary") from exc
    dictionary = Dictionary.from_files(str(dictionary_base))
    return [
        row["form"] for row in fixture["forms"]
        if not dictionary.lookup(row["form"])
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--unigrams", type=Path, default=DEFAULT_UNIGRAMS)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()

    generated = fixture_payload(args.vocab, args.unigrams, args.top_n)
    if args.write_fixture:
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        args.fixture.write_text(
            json.dumps(generated, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.fixture}")
    else:
        committed = json.loads(args.fixture.read_text(encoding="utf-8"))
        if generated != committed:
            print("ERROR: frequency fixture is stale; regenerate with --write-fixture",
                  file=sys.stderr)
            return 1

    missing = audit_dictionary(args.dictionary, generated)
    total = sum(row["count"] for row in generated["forms"])
    print(f"top {args.top_n}: {args.top_n - len(missing)}/{args.top_n} accepted "
          f"({total:,} LM tokens)")
    if missing:
        print("missing: " + ", ".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

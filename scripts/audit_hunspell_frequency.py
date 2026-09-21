#!/usr/bin/env python3
"""Audit the grc Hunspell export against the language-model frequency head.

The LM vocabulary contains structural sentinels, punctuation-bearing tokens,
and source defects as well as words. This gate ranks lexical tokens made only
of Greek letters and combining marks, folds contextual grave accents to acute,
and aggregates counts after that normalization. Elision-mark normalization is
a separate consumer concern and is deliberately outside this fixture.

The committed fixture lets CI enforce the coverage invariant without shipping
the complete LM build. Its source hashes make regeneration against a different
vocabulary or unigram table explicit. JSON sources require a non-sanity
``stats.json``. The format-v2 LM binary is also accepted because it embeds the
exact exported vocabulary and unigram counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
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
DEFAULT_STATS = ROOT / "build" / "lm" / "stats.json"
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


def _ranked_forms(vocab: list[str], unigrams: dict[int, int]) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for index, raw_count in unigrams.items():
        if index < 0 or index >= len(vocab):
            raise ValueError(f"unigram index {index} is outside the vocabulary")
        form = vocab[index]
        if not isinstance(form, str) or not is_greek_lexical_form(form):
            continue
        counts[normalize_form(form)] += int(raw_count)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def ranked_forms(vocab_path: Path, unigrams_path: Path) -> list[tuple[str, int]]:
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    raw_unigrams = json.loads(unigrams_path.read_text(encoding="utf-8"))
    if not isinstance(vocab, list) or not isinstance(raw_unigrams, dict):
        raise ValueError("vocab must be a list and unigrams must be an object")
    return _ranked_forms(vocab, {
        int(index): int(count) for index, count in raw_unigrams.items()
    })


def _payload(forms: list[tuple[str, int]], top_n: int, sources: dict,
             training: dict) -> dict:
    selected = forms[:top_n]
    if len(selected) != top_n:
        raise ValueError(f"only {len(selected)} lexical forms available for top {top_n}")
    return {
        "schema_version": 1,
        "top_n": top_n,
        "selection": "Greek letters and combining marks only; nonlexical tokens excluded",
        "normalization": "NFC; combining grave U+0300 folded to acute U+0301; counts aggregated",
        "training": training,
        "sources": sources,
        "forms": [{"form": form, "count": count} for form, count in selected],
    }


def fixture_payload_from_json(
    vocab_path: Path,
    unigrams_path: Path,
    stats_path: Path,
    top_n: int = DEFAULT_TOP_N,
) -> dict:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats.get("sanity") is not False:
        raise ValueError(
            f"{stats_path} is a sanity or unclassified run; "
            "the Hunspell frequency gate requires a full LM run"
        )
    return _payload(
        ranked_forms(vocab_path, unigrams_path),
        top_n,
        {
            "vocab.json": {"sha256": sha256(vocab_path)},
            "unigrams.json": {"sha256": sha256(unigrams_path)},
            "stats.json": {"sha256": sha256(stats_path)},
        },
        {
            "kind": "full-json-run",
            "sanity": False,
            "total_tokens": int(stats["n_train_tokens"]),
        },
    )


def _read_lm_binary(path: Path) -> tuple[list[str], dict[int, int], dict]:
    data = path.read_bytes()
    if len(data) < 128 or data[:4] != b"GNLM":
        raise ValueError(f"{path} is not a Dilemma GNLM artifact")
    format_version = struct.unpack_from("<I", data, 4)[0]
    if format_version < 2:
        raise ValueError("LM binary must be format version 2 or newer")
    vocab_size = struct.unpack_from("<I", data, 16)[0]
    total_tokens = struct.unpack_from("<Q", data, 40)[0]
    offsets_at, pool_at, pool_size = struct.unpack_from("<QQQ", data, 48)
    offsets_end = offsets_at + 4 * (vocab_size + 1)
    pool_end = pool_at + pool_size
    counts_end = pool_end + 4 * vocab_size
    if offsets_end > len(data) or pool_end > len(data) or counts_end > len(data):
        raise ValueError(f"{path} has out-of-range vocabulary sections")

    offsets = struct.unpack_from(f"<{vocab_size + 1}I", data, offsets_at)
    if offsets[0] != 0 or offsets[-1] != pool_size:
        raise ValueError(f"{path} has invalid vocabulary offsets")
    pool = data[pool_at:pool_end]
    vocab = [
        pool[offsets[index]:offsets[index + 1]].decode("utf-8")
        for index in range(vocab_size)
    ]
    dense_counts = struct.unpack_from(f"<{vocab_size}I", data, pool_end)
    unigrams = {
        index: count for index, count in enumerate(dense_counts) if count
    }
    return vocab, unigrams, {
        "format_version": format_version,
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
    }


def fixture_payload_from_binary(
    binary_path: Path,
    version_path: Path | None,
    top_n: int = DEFAULT_TOP_N,
) -> dict:
    vocab, unigrams, metadata = _read_lm_binary(binary_path)
    sources = {"grc_ngram.bin": {"sha256": sha256(binary_path)}}
    training = {
        "kind": "full-format-v2-artifact",
        "sanity": False,
        **metadata,
    }
    if version_path is not None:
        version = json.loads(version_path.read_text(encoding="utf-8"))
        if int(version["total_tokens"]) != metadata["total_tokens"]:
            raise ValueError("LM binary and version sidecar disagree on total_tokens")
        sources["grc_ngram.version"] = {"sha256": sha256(version_path)}
        training["dilemma_commit"] = version.get("dilemma_commit")
        training["semver"] = version.get("semver")
    return _payload(
        _ranked_forms(vocab, unigrams), top_n, sources, training
    )


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
    parser.add_argument(
        "--source", choices=("json", "binary", "fixture"), default="json",
        help="frequency source; JSON mode rejects sanity runs",
    )
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--unigrams", type=Path, default=DEFAULT_UNIGRAMS)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--lm-binary", type=Path)
    parser.add_argument("--lm-version", type=Path)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()

    try:
        if args.source == "fixture":
            if args.write_fixture:
                parser.error("--write-fixture cannot be used with --source fixture")
            generated = json.loads(args.fixture.read_text(encoding="utf-8"))
        elif args.source == "binary":
            if args.lm_binary is None:
                parser.error("--lm-binary is required with --source binary")
            generated = fixture_payload_from_binary(
                args.lm_binary, args.lm_version, args.top_n
            )
        else:
            generated = fixture_payload_from_json(
                args.vocab, args.unigrams, args.stats, args.top_n
            )
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.write_fixture:
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        args.fixture.write_text(
            json.dumps(generated, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.fixture}")
    elif args.source != "fixture":
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

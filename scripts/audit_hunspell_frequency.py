#!/usr/bin/env python3
"""Audit grc Hunspell coverage at the frequency head and over all entries.

The LM vocabulary contains structural sentinels, punctuation-bearing tokens,
Modern Greek intrusions, and source defects as well as words. This gate ranks
every token containing Greek text, folds contextual grave accents to acute,
canonicalizes a final textual apostrophe to the lookup koronis, and aggregates
counts after normalization. A reviewed fixture records the nonwords that must
remain rejected; every other top-frequency form must be accepted.

The committed fixture lets CI enforce the coverage invariant without shipping
the complete LM build. Its source hashes make regeneration against a different
vocabulary or unigram table explicit. JSON sources require a non-sanity
``stats.json``. The format-v2 LM binary is also accepted because it embeds the
exact exported vocabulary and unigram counts.

The same command enforces the reviewed shipped acceptance surface, top LSJ9
citation headwords, complete textbook paradigms, grave/acute twins, structural
spelling hygiene, and pinned held-out corpus rejection ceilings.
"""
from __future__ import annotations

import argparse
import gzip
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
from dilemma.form_sanitize import (  # noqa: E402
    canonicalize_final_elision,
    sanitize_form,
)
from export_hunspell import (  # noqa: E402
    BARE_ELISION_STEMS,
    FORM_PROFILE_DB,
    GRC_CLOSED_LIST_FORMS,
    contextual_acute,
    filter_dominated_spelling_variants,
    grc_orthography_reason,
    grc_pinned_forms,
    load_canonical_ag_sets,
    load_form_profile_freq,
    load_grc_textbook_forms,
    load_lm_head_required_forms,
    load_top_lsj9_lemmas,
    new_form_structural_reason,
)

DEFAULT_VOCAB = ROOT / "build" / "lm" / "vocab.json"
DEFAULT_UNIGRAMS = ROOT / "build" / "lm" / "unigrams.json"
DEFAULT_STATS = ROOT / "build" / "lm" / "stats.json"
DEFAULT_DICTIONARY = ROOT / "build" / "hunspell" / "grc_polytonic"
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "hunspell_lm_top1000.json"
DEFAULT_EXCLUSIONS = (
    ROOT / "tests" / "fixtures" / "hunspell_lm_top1000_exclusions.json"
)
DEFAULT_COMPATIBILITY = ROOT / "data" / "hunspell_grc_shipped_compat.json.gz"
DEFAULT_HELDOUT = ROOT / "tests" / "fixtures" / "hunspell_heldout.json.gz"
DEFAULT_TOP_N = 1000
# Top-2,000 LSJ9 headwords that are not single dictionary words. Every other
# headword must be accepted, so a structural rule that rejects a real
# headword fails the audit instead of shrinking it.
CITATION_HEADWORD_EXCLUSIONS = {
    "οὐ μή": "two-word phrase",
    "μὴ οὐ": "two-word phrase",
    "καί τοι": "two-word spelling of καίτοι",
    "βίοςι": "LSJ9 extraction artifact",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contains_greek_text(form: str) -> bool:
    """Return whether a whitespace-free LM token contains a Greek letter."""
    if not form or any(char.isspace() for char in form):
        return False
    return any(
        unicodedata.category(char).startswith("L")
        and "GREEK" in unicodedata.name(char, "")
        for char in unicodedata.normalize("NFD", form)
    )


def normalize_form(form: str) -> str:
    return grave_to_acute(unicodedata.normalize("NFC", form))


def lookup_form(form: str) -> str:
    """Return the spelling stored in the exported Hunspell dictionary."""
    return canonicalize_final_elision(normalize_form(form))


def exact_attestation_payload(
    vocab: list[str],
    unigrams: dict[int, int],
    sources: dict,
    training: dict,
) -> dict:
    """Build the accent-preserving frequency map consumed by the exporter."""
    counts: Counter[str] = Counter()
    for index, raw_count in unigrams.items():
        if index < 0 or index >= len(vocab):
            raise ValueError(f"unigram index {index} is outside the vocabulary")
        form = vocab[index]
        if not isinstance(form, str):
            continue
        if not contains_greek_text(form):
            continue
        form = canonicalize_final_elision(sanitize_form(form))
        key = unicodedata.normalize("NFC", normalize_form(form).lower())
        counts[key] += int(raw_count)
    return {
        "_meta": {
            "normalization": (
                "NFC; final textual elision apostrophe to U+1FBD; "
                "grave-to-acute; lowercase; all other marks preserved"
            ),
            "training": training,
            "sources": sources,
        },
        "forms": dict(sorted(counts.items())),
    }


def write_gzip_json(path: Path, payload: dict) -> None:
    """Write deterministic compact UTF-8 JSON in gzip format."""
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=False
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(encoded)


def _ranked_forms(vocab: list[str], unigrams: dict[int, int]) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for index, raw_count in unigrams.items():
        if index < 0 or index >= len(vocab):
            raise ValueError(f"unigram index {index} is outside the vocabulary")
        form = vocab[index]
        if not isinstance(form, str) or not contains_greek_text(form):
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
        "selection": "whitespace-free LM tokens containing at least one Greek letter",
        "normalization": (
            "NFC; combining grave U+0300 folded to acute U+0301; "
            "counts aggregated; final textual apostrophe canonicalized only at lookup"
        ),
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


def load_exclusions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("exclusions")
    if payload.get("schema_version") != 1 or not isinstance(rows, list):
        raise ValueError(f"invalid Hunspell exclusion fixture: {path}")
    exclusions: dict[str, str] = {}
    for row in rows:
        form = row.get("form")
        reason = row.get("reason")
        if not isinstance(form, str) or not isinstance(reason, str) or not reason:
            raise ValueError(f"invalid Hunspell exclusion row: {row!r}")
        if form in exclusions:
            raise ValueError(f"duplicate Hunspell exclusion: {form}")
        exclusions[form] = reason
    return exclusions


def audit_dictionary(
    dictionary_base: Path,
    fixture: dict,
    exclusions: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    try:
        from spylls.hunspell import Dictionary
    except ImportError as exc:  # pragma: no cover - exercised by CLI users
        raise RuntimeError("install spylls to audit the expanded dictionary") from exc
    dictionary = Dictionary.from_files(str(dictionary_base))
    exclusions = exclusions or {}
    forms = {row["form"] for row in fixture["forms"]}
    unknown_exclusions = sorted(set(exclusions) - forms)
    if unknown_exclusions:
        raise ValueError(
            "exclusions are outside the frequency fixture: "
            + ", ".join(unknown_exclusions)
        )
    missing_required: list[str] = []
    accepted_exclusions: list[str] = []
    for row in fixture["forms"]:
        form = row["form"]
        accepted = dictionary.lookup(lookup_form(form))
        if form in exclusions:
            if accepted:
                accepted_exclusions.append(form)
        elif not accepted:
            missing_required.append(form)
    return missing_required, accepted_exclusions


def _affix_suffixes(dictionary_base: Path) -> dict[str, set[str]]:
    """Read the exporter's zero-strip numeric suffix rules."""
    suffixes: dict[str, set[str]] = {}
    aff_path = dictionary_base.with_suffix(".aff")
    for line in aff_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[0] != "SFX":
            continue
        _kind, flag, strip, add, condition = fields
        if strip != "0" or condition != ".":
            raise ValueError(
                f"unsupported Hunspell rule in artifact audit: {line}"
            )
        suffixes.setdefault(flag, set()).add("" if add == "0" else add)
    return suffixes


def expanded_export_forms(dictionary_base: Path) -> tuple[set[str], set[str]]:
    """Return expanded forms and flags whose stem lacks an identity rule."""
    suffixes = _affix_suffixes(dictionary_base)
    forms: set[str] = set()
    flags_without_identity: set[str] = set()
    dic_path = dictionary_base.with_suffix(".dic")
    for index, line in enumerate(dic_path.read_text(encoding="utf-8").splitlines()):
        if index == 0 or not line:
            continue
        head = line.split("\t", 1)[0]
        word, separator, raw_flags = head.rpartition("/")
        if (not separator
                or not all(flag.isdecimal() for flag in raw_flags.split(","))):
            word, separator, raw_flags = head, "", ""
        forms.add(word)
        if not separator:
            continue
        for flag in raw_flags.split(","):
            endings = suffixes.get(flag)
            if not endings:
                raise ValueError(f"dictionary entry uses undefined flag {flag}")
            if "" not in endings:
                flags_without_identity.add(flag)
            forms.update(word + ending for ending in endings)
    return forms, flags_without_identity


def audit_export_orthography(
    dictionary_base: Path,
    reviewed: set[str] | frozenset[str] = frozenset(),
) -> tuple[list[str], list[str]]:
    """Find structurally invalid forms and synthetic flagged dictionary bases.

    ``reviewed`` forms (the shipped baseline and the closed lists) are exempt
    from the structural rules, exactly as in the exporter; bare elision
    stems never are.
    """
    forms, flags_without_identity = expanded_export_forms(dictionary_base)
    invalid = sorted(
        form for form in forms
        if (form not in reviewed and grc_orthography_reason(form) is not None)
        or form in BARE_ELISION_STEMS
    )
    return invalid, sorted(flags_without_identity)


def load_gzip_fixture(path: Path, schema_version: int = 1) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != schema_version:
        raise ValueError(f"unsupported fixture schema in {path}")
    return payload


def audit_whole_artifact(
    dictionary_base: Path,
    compatibility_path: Path = DEFAULT_COMPATIBILITY,
) -> dict[str, list[str]]:
    """Audit all expanded entries, not only forms in the LM vocabulary."""
    forms, flags_without_identity = expanded_export_forms(dictionary_base)
    compatibility = load_gzip_fixture(compatibility_path)
    required = set(compatibility.get("forms", []))
    reviewed = required | set(GRC_CLOSED_LIST_FORMS)
    invalid = sorted(
        form for form in forms
        if (form not in reviewed and grc_orthography_reason(form) is not None)
        or form in BARE_ELISION_STEMS
    )
    # The acute twin of a reviewed grave inherits its review (see
    # export_hunspell.finalize_grc_pairs).
    inherited = {contextual_acute(form) for form in required}
    grave_without_acute = sorted(
        form for form in forms
        if (contextual_acute(form) != form
            and contextual_acute(form) not in forms
            and grc_orthography_reason(contextual_acute(form)) is None
            and new_form_structural_reason(contextual_acute(form)) is None)
    )
    citation_headwords = (
        load_top_lsj9_lemmas(2000) - set(CITATION_HEADWORD_EXCLUSIONS)
    )
    return {
        "compatibility_missing": sorted(required - forms),
        "invalid_forms": invalid,
        "grave_without_acute": grave_without_acute,
        "citation_headwords_missing": sorted(citation_headwords - forms),
        "flags_without_identity": sorted(flags_without_identity),
        "weak_respellings": weak_respellings(forms - required - inherited),
    }


def weak_respellings(new_forms: set[str]) -> list[str]:
    """New entries that fail the exporter's respelling-evidence rule.

    The exporter applies the rule before affix compression and again to the
    acute twins it derives; this checks the expanded result, so a later step
    cannot reintroduce ``ἑγώ`` beside ``ἐγώ``.
    """
    if not FORM_PROFILE_DB.exists():
        raise RuntimeError(
            f"{FORM_PROFILE_DB} is required for the respelling audit; "
            "run `python -m dilemma download --with-attestation`"
        )
    profile = load_form_profile_freq()
    canonical_forms, _canonical_lemmas = load_canonical_ag_sets()
    textbook_forms, _textbook_meta = load_grc_textbook_forms()
    _kept, rejected = filter_dominated_spelling_variants(
        [(form, form) for form in sorted(new_forms)],
        profile.exact, profile.dominant, profile.treebank,
        compatibility_forms=set(),
        protected_forms=grc_pinned_forms(
            canonical_forms, load_top_lsj9_lemmas(), textbook_forms,
            load_lm_head_required_forms(),
        ),
        treebank_confirmed=profile.treebank_confirmed,
    )
    return [form for form, _lemma in rejected]


def textbook_paradigm_forms() -> set[str]:
    """Return the pinned standard forms for the release-gate paradigms."""
    forms, _metadata = load_grc_textbook_forms()
    return forms


def audit_textbook_paradigms(dictionary_base: Path) -> list[str]:
    forms, _flags = expanded_export_forms(dictionary_base)
    return sorted(textbook_paradigm_forms() - forms)


def measure_heldout_corpora(
    dictionary_base: Path,
    fixture_path: Path = DEFAULT_HELDOUT,
) -> dict[str, dict[str, int]]:
    """Measure candidate and baseline rejection counts on held-out corpora."""
    try:
        from spylls.hunspell import Dictionary
    except ImportError as exc:  # pragma: no cover - exercised by CLI users
        raise RuntimeError("install spylls to audit held-out corpora") from exc
    dictionary = Dictionary.from_files(str(dictionary_base))
    fixture = load_gzip_fixture(fixture_path)
    metrics: dict[str, dict[str, int]] = {}
    for name, corpus in fixture["corpora"].items():
        rows = corpus["forms"]
        rejected_types = sum(
            1 for form, _count in rows if not dictionary.lookup(form)
        )
        rejected_tokens = sum(
            int(count) for form, count in rows if not dictionary.lookup(form)
        )
        baseline = corpus["baseline"]
        metrics[name] = {
            "rejected_types": rejected_types,
            "baseline_rejected_types": int(baseline["rejected_types"]),
            "rejected_tokens": rejected_tokens,
            "baseline_rejected_tokens": int(baseline["rejected_tokens"]),
        }
    return metrics


def heldout_regressions(
    metrics: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    return {
        name: row for name, row in metrics.items()
        if (row["rejected_types"] > row["baseline_rejected_types"]
            or row["rejected_tokens"] > row["baseline_rejected_tokens"])
    }


def compare_dictionary_coverage(
    baseline_base: Path,
    candidate_base: Path,
    vocab: list[str],
    unigrams: dict[int, int],
) -> dict:
    """Compare two expanded dictionaries over every Greek-bearing LM token."""
    try:
        from spylls.hunspell import Dictionary
    except ImportError as exc:  # pragma: no cover - exercised by CLI users
        raise RuntimeError("install spylls to compare dictionaries") from exc

    baseline = Dictionary.from_files(str(baseline_base))
    candidate = Dictionary.from_files(str(candidate_base))
    counts: Counter[str] = Counter()
    for index, raw_count in unigrams.items():
        if index < 0 or index >= len(vocab):
            raise ValueError(f"unigram index {index} is outside the vocabulary")
        raw_form = vocab[index]
        if isinstance(raw_form, str) and contains_greek_text(raw_form):
            counts[lookup_form(raw_form)] += int(raw_count)

    lost: list[dict[str, int | str]] = []
    gained: list[dict[str, int | str]] = []
    for form, count in counts.items():
        old = baseline.lookup(form)
        new = candidate.lookup(form)
        if old and not new:
            lost.append({"form": form, "count": count})
        elif new and not old:
            gained.append({"form": form, "count": count})
    lost.sort(key=lambda row: (-int(row["count"]), str(row["form"])))
    gained.sort(key=lambda row: (-int(row["count"]), str(row["form"])))
    return {
        "schema_version": 1,
        "baseline": {
            "base": str(baseline_base),
            "dic_sha256": sha256(baseline_base.with_suffix(".dic")),
            "aff_sha256": sha256(baseline_base.with_suffix(".aff")),
        },
        "candidate": {
            "base": str(candidate_base),
            "dic_sha256": sha256(candidate_base.with_suffix(".dic")),
            "aff_sha256": sha256(candidate_base.with_suffix(".aff")),
        },
        "lm_word_tokens": sum(counts.values()),
        "lm_forms": len(counts),
        "lost_token_count": sum(int(row["count"]) for row in lost),
        "gained_token_count": sum(int(row["count"]) for row in gained),
        "lost": lost,
        "gained": gained,
    }


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
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument(
        "--compatibility", type=Path, default=DEFAULT_COMPATIBILITY,
        help="reviewed whole-artifact acceptance baseline",
    )
    parser.add_argument(
        "--heldout", type=Path, default=DEFAULT_HELDOUT,
        help="accent-preserving held-out corpus fixture",
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--write-fixture", action="store_true")
    parser.add_argument(
        "--baseline-dictionary", type=Path,
        help="compare the candidate with a previously shipped dictionary base",
    )
    parser.add_argument(
        "--comparison-report", type=Path,
        help="write the full baseline comparison as JSON",
    )
    parser.add_argument(
        "--attestation-out", type=Path,
        help="write the exporter's deterministic exact-form frequency map",
    )
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

    if args.attestation_out is not None:
        if args.source == "fixture":
            parser.error("--attestation-out requires --source json or binary")
        if args.source == "binary":
            vocab, unigrams, _metadata = _read_lm_binary(args.lm_binary)
        else:
            vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
            raw = json.loads(args.unigrams.read_text(encoding="utf-8"))
            unigrams = {int(index): int(count) for index, count in raw.items()}
        write_gzip_json(
            args.attestation_out,
            exact_attestation_payload(
                vocab, unigrams, generated["sources"], generated["training"]
            ),
        )
        print(f"wrote {args.attestation_out}")

    if args.baseline_dictionary is not None:
        if args.source == "fixture":
            parser.error("--baseline-dictionary requires --source json or binary")
        if args.source == "binary":
            vocab, unigrams, _metadata = _read_lm_binary(args.lm_binary)
        else:
            vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
            raw = json.loads(args.unigrams.read_text(encoding="utf-8"))
            unigrams = {int(index): int(count) for index, count in raw.items()}
        comparison = compare_dictionary_coverage(
            args.baseline_dictionary, args.dictionary, vocab, unigrams
        )
        print(
            "baseline delta: "
            f"-{len(comparison['lost']):,} forms / "
            f"{comparison['lost_token_count']:,} LM tokens; "
            f"+{len(comparison['gained']):,} forms / "
            f"{comparison['gained_token_count']:,} LM tokens"
        )
        if args.comparison_report is not None:
            args.comparison_report.parent.mkdir(parents=True, exist_ok=True)
            args.comparison_report.write_text(
                json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {args.comparison_report}")

    exclusions = load_exclusions(args.exclusions)
    missing, accepted_exclusions = audit_dictionary(
        args.dictionary, generated, exclusions
    )
    whole = audit_whole_artifact(args.dictionary, args.compatibility)
    paradigm_missing = audit_textbook_paradigms(args.dictionary)
    heldout_metrics = measure_heldout_corpora(args.dictionary, args.heldout)
    heldout_failures = heldout_regressions(heldout_metrics)
    total = sum(row["count"] for row in generated["forms"])
    required = args.top_n - len(exclusions)
    print(f"top {args.top_n}: {required - len(missing)}/{required} required "
          f"forms accepted; {len(exclusions) - len(accepted_exclusions)}/"
          f"{len(exclusions)} reviewed nonwords rejected ({total:,} LM tokens)")
    compatibility_count = len(load_gzip_fixture(args.compatibility)["forms"])
    citation_count = len(
        load_top_lsj9_lemmas(2000) - set(CITATION_HEADWORD_EXCLUSIONS)
    )
    paradigm_count = len(textbook_paradigm_forms())
    print(
        "whole artifact: "
        f"{compatibility_count - len(whole['compatibility_missing']):,}/"
        f"{compatibility_count:,} reviewed shipped forms; "
        f"{citation_count - len(whole['citation_headwords_missing']):,}/"
        f"{citation_count:,} top citation headwords; "
        f"{paradigm_count - len(paradigm_missing):,}/{paradigm_count:,} "
        "textbook paradigm forms"
    )
    print(
        "whole artifact hygiene: "
        f"{len(whole['invalid_forms'])} invalid forms, "
        f"{len(whole['grave_without_acute'])} missing acute twins, "
        f"{len(whole['flags_without_identity'])} non-identity flags, "
        f"{len(whole['weak_respellings'])} new weak respellings; "
        f"{len(heldout_failures)} regression-corpus failures"
    )
    for name, metrics in heldout_metrics.items():
        print(
            f"regression corpus {name}: candidate rejects "
            f"{metrics['rejected_tokens']:,} tokens/"
            f"{metrics['rejected_types']:,} types; April "
            f"{metrics['baseline_rejected_tokens']:,}/"
            f"{metrics['baseline_rejected_types']:,}"
        )
    if missing:
        print("missing required: " + ", ".join(missing), file=sys.stderr)
    if accepted_exclusions:
        print("accepted reviewed nonwords: " + ", ".join(accepted_exclusions),
              file=sys.stderr)
    labels = {
        "compatibility_missing": "forms the shipped dictionary has and this lacks",
        "invalid_forms": "structurally invalid accepted forms",
        "grave_without_acute": "grave forms without acute twins",
        "citation_headwords_missing": "top citation headwords missing",
        "flags_without_identity": "flags without identity rules",
        "weak_respellings": "new weak respellings of a common spelling",
    }
    for key, label in labels.items():
        rows = whole[key]
        if rows:
            print(f"{label} ({len(rows)}): " + ", ".join(rows[:50]),
                  file=sys.stderr)
    if paradigm_missing:
        print(f"textbook paradigm forms missing ({len(paradigm_missing)}): "
              + ", ".join(paradigm_missing[:50]), file=sys.stderr)
    if heldout_failures:
        for name, metrics in heldout_failures.items():
            print(f"held-out regression {name}: {metrics}", file=sys.stderr)
    whole_failures = any(whole.values())
    if (missing or accepted_exclusions or whole_failures
            or paradigm_missing or heldout_failures):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

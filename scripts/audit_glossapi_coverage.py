#!/usr/bin/env python3
"""Audit revision-pinned GlossAPI corpora against Dilemma.

The manifest in ``data/glossapi_corpora.json`` is the reproducibility
contract: every corpus is pinned to a 40-character Hub commit and every file
to its byte size and SHA-256.  The audit measures structured ``guess=False``
coverage, unresolved lexical forms, Unicode/OCR defects, and forms whose
effective Modern- and Ancient-Greek lookup rows disagree.

The same samples can produce a balanced Modern Greek surface-frequency
experiment.  It is intentionally diagnostic: this script never overwrites
``data/mg_freq.txt`` and historical candidates never enter that experiment.

Examples:

    # Audit artifacts already present in the Hugging Face cache.
    python scripts/audit_glossapi_coverage.py --source cache

    # Stream pinned artifacts with authenticated HTTP range reads.
    python scripts/audit_glossapi_coverage.py --source stream

    # Defer a very large source and mark the frequency experiment incomplete.
    python scripts/audit_glossapi_coverage.py --source stream --exclude-corpus diavgeia

    # Fast smoke run of one corpus.
    python scripts/audit_glossapi_coverage.py --corpus mitos --max-tokens 20000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
import unicodedata
import zipfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dilemma.core import (  # noqa: E402
    LOOKUP_DB_PATH,
    Dilemma,
    strip_accents,
    to_monotonic,
    to_standard_sigma,
)


DEFAULT_MANIFEST = ROOT / "data" / "glossapi_corpora.json"
DEFAULT_REPORT = ROOT / "data" / "glossapi_coverage_report.json"
DEFAULT_FREQUENCY_REPORT = (
    ROOT / "data" / "glossapi_mg_frequency_experiment.json"
)
DEFAULT_MG_FREQUENCY = ROOT / "data" / "mg_freq.txt"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_URL_RE = re.compile(r"^(?:https?|ftp)://", re.IGNORECASE)
_WORD_JOINERS = {
    "'", "\u2019", "\u02bc", "\u1fbd", "\u0374", "\u02b9",
    "-", "\u2010", "\u2011",
}
_APOSTROPHES = {"'", "\u2019", "\u02bc", "\u1fbd", "\u0374", "\u02b9"}


class AuditError(RuntimeError):
    """A reproducibility, access, or input error in the corpus audit."""


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    """Load and validate the pinned corpus manifest."""
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict) -> None:
    """Reject floating revisions, weak file pins, and duplicate corpus keys."""
    if manifest.get("schema_version") != 1:
        raise AuditError("unsupported GlossAPI manifest schema")
    corpora = manifest.get("corpora")
    if not isinstance(corpora, list) or len(corpora) != 9:
        raise AuditError("GlossAPI manifest must contain exactly nine corpora")
    keys: set[str] = set()
    for corpus in corpora:
        key = corpus.get("key")
        if not key or key in keys:
            raise AuditError(f"missing or duplicate corpus key: {key!r}")
        keys.add(key)
        revision = corpus.get("revision", "")
        if not _COMMIT_RE.fullmatch(revision):
            raise AuditError(f"{key}: revision must be a full commit SHA")
        if corpus.get("role") not in {"mg_core", "mg_legal", "ogc_candidate"}:
            raise AuditError(f"{key}: invalid role {corpus.get('role')!r}")
        if corpus.get("lookup_lang") not in {"all", "el", "grc"}:
            raise AuditError(f"{key}: invalid lookup_lang")
        files = corpus.get("files")
        if not isinstance(files, list) or not files:
            raise AuditError(f"{key}: no pinned files")
        for artifact in files:
            if not artifact.get("path"):
                raise AuditError(f"{key}: artifact path is missing")
            if not isinstance(artifact.get("size"), int) or artifact["size"] <= 0:
                raise AuditError(f"{key}/{artifact.get('path')}: invalid size")
            if not _SHA256_RE.fullmatch(artifact.get("sha256", "")):
                raise AuditError(
                    f"{key}/{artifact.get('path')}: invalid SHA-256"
                )
        if corpus["role"] == "ogc_candidate":
            if corpus.get("frequency_token_cap") != 0:
                raise AuditError(f"{key}: OGC candidates cannot feed MG frequency")
            if not corpus.get("ogc_decision"):
                raise AuditError(f"{key}: OGC decision is missing")

    model = manifest.get("variety_model", {})
    if not _COMMIT_RE.fullmatch(model.get("revision", "")):
        raise AuditError("variety model revision must be a full commit SHA")
    weights = model.get("weights", {})
    if not _SHA256_RE.fullmatch(weights.get("sha256", "")):
        raise AuditError("variety model weights must have a SHA-256 pin")


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_local_artifact(path: Path, artifact: dict) -> None:
    size = path.stat().st_size
    if size != artifact["size"]:
        raise AuditError(
            f"{path}: size {size}, manifest requires {artifact['size']}"
        )
    digest = file_sha256(path)
    if digest != artifact["sha256"]:
        raise AuditError(
            f"{path}: sha256 {digest}, manifest requires {artifact['sha256']}"
        )


def _cached_or_downloaded_path(
    corpus: dict, artifact: dict, *, download: bool
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - core dependency
        raise AuditError("huggingface_hub is required") from exc
    try:
        resolved = hf_hub_download(
            repo_id=corpus["repo"],
            filename=artifact["path"],
            repo_type="dataset",
            revision=corpus["revision"],
            local_files_only=not download,
        )
    except Exception as exc:
        mode = "download" if download else "cache"
        raise AuditError(
            f"{corpus['key']}/{artifact['path']} unavailable in {mode} mode: {exc}"
        ) from exc
    path = Path(resolved)
    _verify_local_artifact(path, artifact)
    return path


def _verify_remote_artifact(corpus: dict, artifact: dict) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - core dependency
        raise AuditError("huggingface_hub is required") from exc
    try:
        infos = HfApi().get_paths_info(
            corpus["repo"],
            paths=[artifact["path"]],
            repo_type="dataset",
            revision=corpus["revision"],
            expand=True,
        )
    except Exception as exc:
        raise AuditError(
            f"cannot verify {corpus['key']}/{artifact['path']}: {exc}"
        ) from exc
    if len(infos) != 1:
        raise AuditError(f"remote artifact is missing: {corpus['key']}/{artifact['path']}")
    info = infos[0]
    remote_sha = info.lfs.sha256 if info.lfs else None
    if info.size != artifact["size"] or remote_sha != artifact["sha256"]:
        raise AuditError(
            f"remote bytes drifted for {corpus['key']}/{artifact['path']}"
        )


@contextmanager
def open_artifact(
    corpus: dict, artifact: dict, source: str
) -> Iterator[BinaryIO]:
    """Open a pinned local artifact or an authenticated range-readable stream."""
    if source in {"cache", "download"}:
        path = _cached_or_downloaded_path(
            corpus, artifact, download=source == "download"
        )
        with path.open("rb") as handle:
            yield handle
        return

    if source != "stream":
        raise AuditError(f"unknown artifact source mode: {source}")
    _verify_remote_artifact(corpus, artifact)
    try:
        from huggingface_hub import HfFileSystem
    except ImportError as exc:  # pragma: no cover - core dependency
        raise AuditError("huggingface_hub is required") from exc
    remote_path = (
        f"datasets/{corpus['repo']}@{corpus['revision']}/{artifact['path']}"
    )
    try:
        with HfFileSystem().open(remote_path, "rb") as handle:
            yield handle
    except Exception as exc:
        raise AuditError(
            f"cannot stream {corpus['key']}/{artifact['path']}: {exc}"
        ) from exc


def _top_level_columns(reader: dict) -> list[str] | None:
    fields = reader.get("text_fields", [])
    if any(field.get("path") == "*" for field in fields):
        return None
    columns = []
    for field in fields:
        top = field["path"].split(".", 1)[0]
        if top.endswith("[]"):
            top = top[:-2]
        if top not in columns:
            columns.append(top)
    return columns


def _sampled_row_groups(total: int, maximum: int = 32) -> list[int]:
    if total <= maximum:
        return list(range(total))
    # Midpoints of equal-width bins include the full file without favoring its
    # first row groups.  The set handles rare float-rounding duplicates.
    return sorted({min(total - 1, int((i + 0.5) * total / maximum))
                   for i in range(maximum)})


def iter_parquet_rows(
    handle: BinaryIO, reader: dict, *, max_documents: int = 4096
) -> Iterator[dict]:
    """Yield a deterministic row-group-stratified sample from a Parquet file."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise AuditError(
            "Parquet auditing requires: pip install -e '.[glossapi-audit]'"
        ) from exc

    parquet = pq.ParquetFile(handle)
    row_groups = _sampled_row_groups(parquet.metadata.num_row_groups)
    columns = _top_level_columns(reader)
    streams = [
        parquet.iter_batches(
            batch_size=1, row_groups=[row_group], columns=columns,
            use_threads=False,
        )
        for row_group in row_groups
    ]
    yielded = 0
    active = list(streams)
    while active and yielded < max_documents:
        next_active = []
        for batches in active:
            try:
                batch = next(batches)
            except StopIteration:
                continue
            next_active.append(batches)
            for row in batch.to_pylist():
                yield row
                yielded += 1
                if yielded >= max_documents:
                    return
        active = next_active


def iter_zip_texts(
    handle: BinaryIO, reader: dict, revision: str, *, max_documents: int = 4096
) -> Iterator[str]:
    """Yield deterministically ordered text members from a remote-safe ZIP."""
    suffixes = tuple(reader.get("member_suffixes", [".txt", ".md"]))
    with zipfile.ZipFile(handle) as archive:
        names = [
            info.filename for info in archive.infolist()
            if not info.is_dir() and info.filename.lower().endswith(suffixes)
        ]
        names.sort(
            key=lambda name: hashlib.sha256(
                f"{revision}\0{name}".encode("utf-8")
            ).digest()
        )
        for name in names[:max_documents]:
            raw = archive.read(name)
            yield raw.decode("utf-8", errors="replace")


def _path_values(value, parts: list[str]) -> Iterator[object]:
    if not parts:
        yield value
        return
    part = parts[0]
    is_list = part.endswith("[]")
    key = part[:-2] if is_list else part
    if not isinstance(value, dict) or key not in value:
        return
    child = value[key]
    if is_list:
        if isinstance(child, list):
            for item in child:
                yield from _path_values(item, parts[1:])
        return
    yield from _path_values(child, parts[1:])


def _leaf_strings(value) -> Iterator[str]:
    if isinstance(value, str):
        if value and not _URL_RE.match(value):
            yield value
    elif isinstance(value, list):
        for item in value:
            yield from _leaf_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _leaf_strings(item)


def _field_strings(value, fmt: str) -> Iterator[str]:
    if value is None:
        return
    if fmt == "json_values" and isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            pass
    if fmt not in {"plain", "json_values"}:
        raise AuditError(f"unsupported text field format: {fmt}")
    yield from _leaf_strings(value)


def extract_row_text(row: dict, reader: dict) -> str:
    """Extract natural-language values without counting JSON keys or URLs."""
    chunks: list[str] = []
    excluded = set(reader.get("exclude_fields", []))
    for field in reader.get("text_fields", []):
        path = field["path"]
        if path == "*":
            for key, value in row.items():
                if key not in excluded:
                    chunks.extend(_field_strings(value, field.get("format", "plain")))
            continue
        values = _path_values(row, path.split("."))
        for value in values:
            chunks.extend(_field_strings(value, field.get("format", "plain")))
    return "\n".join(chunks)


def _is_greek_letter(char: str) -> bool:
    code = ord(char)
    return (
        0x0370 <= code <= 0x03FF
        or 0x1F00 <= code <= 0x1FFF
        or 0xAB65 == code
    ) and unicodedata.category(char).startswith("L")


def _is_latin_letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L") and "LATIN" in unicodedata.name(
        char, ""
    )


def _is_word_char(char: str) -> bool:
    return unicodedata.category(char)[0] in {"L", "M", "N"} or char in _WORD_JOINERS


def iter_greek_tokens(text: str) -> Iterator[str]:
    """Yield word-like runs containing Greek while preserving raw Unicode."""
    run: list[str] = []

    def flush() -> str | None:
        if not run:
            return None
        token = "".join(run).strip("-\u2010\u2011")
        run.clear()
        if token and any(_is_greek_letter(char) for char in token):
            return token
        return None

    for char in text:
        if _is_word_char(char):
            run.append(char)
        else:
            token = flush()
            if token is not None:
                yield token
    token = flush()
    if token is not None:
        yield token


def token_defects(token: str) -> tuple[str, ...]:
    """Return conservative Unicode/OCR diagnostics for one observed token."""
    defects: list[str] = []
    if token != unicodedata.normalize("NFC", token):
        defects.append("non_nfc")
    if "\ufffd" in token:
        defects.append("replacement_character")
    if token and unicodedata.combining(token[0]):
        defects.append("leading_combining_mark")
    if any(unicodedata.category(char) == "Co" for char in token):
        defects.append("private_use_character")
    has_greek = any(_is_greek_letter(char) for char in token)
    has_latin = any(_is_latin_letter(char) for char in token)
    if has_greek and has_latin:
        defects.append("mixed_greek_latin")
    if has_greek and any(char.isdigit() for char in token):
        defects.append("mixed_greek_digit")

    previous_was_base_or_mark = False
    for char in token:
        category = unicodedata.category(char)
        if category.startswith("M"):
            if not previous_was_base_or_mark:
                defects.append("orphan_combining_mark")
                break
            previous_was_base_or_mark = True
        else:
            previous_was_base_or_mark = category.startswith("L")

    nfd = unicodedata.normalize("NFD", token)
    marks: set[str] = set()
    for char in nfd:
        if unicodedata.category(char).startswith("M"):
            if char in marks:
                defects.append("duplicate_combining_mark")
                break
            marks.add(char)
        else:
            marks.clear()

    if unicodedata.normalize("NFKC", token) != unicodedata.normalize("NFC", token):
        defects.append("compatibility_character")
    return tuple(dict.fromkeys(defects))


@dataclass
class RawSample:
    token_cap: int
    per_document_token_cap: int = 20_000
    documents: int = 0
    tokens: Counter[str] = field(default_factory=Counter)
    defect_tokens: Counter[str] = field(default_factory=Counter)
    defect_types: dict[str, set[str]] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return sum(self.tokens.values())

    def add_document(self, text: str, identity: str = "") -> None:
        if self.token_count >= self.token_cap:
            return
        observed = list(iter_greek_tokens(text))
        if not observed:
            return
        allowance = min(
            self.per_document_token_cap,
            self.token_cap - self.token_count,
        )
        if len(observed) > allowance:
            seed = hashlib.sha256(
                (identity + "\0" + text[:1024]).encode("utf-8", errors="replace")
            ).digest()
            start = int.from_bytes(seed[:8], "big") % (len(observed) - allowance + 1)
            observed = observed[start:start + allowance]
        self.documents += 1
        self.tokens.update(observed)
        for token in observed:
            for defect in token_defects(token):
                self.defect_tokens[defect] += 1
                self.defect_types.setdefault(defect, set()).add(token)

    def merge(self, other: "RawSample") -> None:
        self.documents += other.documents
        self.tokens.update(other.tokens)
        self.defect_tokens.update(other.defect_tokens)
        for defect, forms in other.defect_types.items():
            self.defect_types.setdefault(defect, set()).update(forms)

    def defect_report(self) -> dict:
        return {
            defect: {
                "tokens": count,
                "types": len(self.defect_types.get(defect, set())),
                "examples": sorted(
                    self.defect_types.get(defect, set()),
                    key=lambda token: (-self.tokens[token], token),
                )[:12],
            }
            for defect, count in sorted(self.defect_tokens.items())
        }


def collect_artifact_sample(
    corpus: dict,
    artifact: dict,
    *,
    source: str,
    token_cap: int,
    max_documents: int,
) -> RawSample:
    sample = RawSample(token_cap=token_cap)
    reader = corpus["reader"]
    with open_artifact(corpus, artifact, source) as handle:
        if reader["kind"] == "parquet":
            rows = iter_parquet_rows(handle, reader, max_documents=max_documents)
            for index, row in enumerate(rows):
                sample.add_document(
                    extract_row_text(row, reader),
                    identity=f"{corpus['key']}:{artifact['path']}:{index}",
                )
                if sample.token_count >= token_cap:
                    break
        elif reader["kind"] == "zip_text":
            texts = iter_zip_texts(
                handle, reader, corpus["revision"], max_documents=max_documents
            )
            for index, text in enumerate(texts):
                sample.add_document(
                    text,
                    identity=f"{corpus['key']}:{artifact['path']}:{index}",
                )
                if sample.token_count >= token_cap:
                    break
        else:
            raise AuditError(f"{corpus['key']}: unsupported reader kind")
    return sample


def collect_corpus_sample(
    corpus: dict,
    *,
    source: str,
    token_cap: int,
    max_documents: int = 4096,
) -> RawSample:
    """Collect an equal-budget sample from every pinned artifact in a corpus."""
    combined = RawSample(token_cap=token_cap)
    files = corpus["files"]
    per_file_cap = math.ceil(token_cap / len(files))
    for artifact in files:
        part = collect_artifact_sample(
            corpus,
            artifact,
            source=source,
            token_cap=per_file_cap,
            max_documents=max_documents,
        )
        combined.merge(part)
    # Per-file rounding can exceed the declared corpus cap by at most N-1.
    if combined.token_count > token_cap:
        excess = combined.token_count - token_cap
        for token in sorted(combined.tokens, reverse=True):
            removed = min(excess, combined.tokens[token])
            combined.tokens[token] -= removed
            excess -= removed
            if combined.tokens[token] == 0:
                del combined.tokens[token]
            if not excess:
                break
    return combined


class LookupConflictResolver:
    """Compare effective ``el`` and ``grc`` rows for observed lookup forms."""

    def __init__(self, path: Path = LOOKUP_DB_PATH):
        self.path = path

    @staticmethod
    def variants(form: str) -> list[str]:
        form = to_standard_sigma(unicodedata.normalize("NFC", form))
        lower = form.lower()
        values = [form, lower, to_monotonic(lower), strip_accents(lower)]
        return list(dict.fromkeys(values))

    def conflicts(self, forms: Counter[str]) -> dict:
        variants_by_form = {form: self.variants(form) for form in forms}
        all_variants = sorted({v for values in variants_by_form.values() for v in values})
        rows: dict[str, dict[str, tuple[str, str]]] = {}
        connection = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True)
        try:
            for offset in range(0, len(all_variants), 800):
                batch = all_variants[offset:offset + 800]
                placeholders = ",".join("?" for _ in batch)
                query = (
                    "SELECT k.form, k.lang, l.text, k.src "
                    "FROM lookup k JOIN lemmas l ON l.id = k.lemma_id "
                    f"WHERE k.form IN ({placeholders}) "
                    "AND k.lang IN ('all', 'el', 'grc')"
                )
                for form, lang, lemma, source in connection.execute(query, batch):
                    rows.setdefault(form, {})[lang] = (lemma, source)
        finally:
            connection.close()

        examples = []
        conflict_tokens = 0
        conflict_types = 0
        for form, count in forms.items():
            effective = {}
            for language in ("el", "grc"):
                for variant in variants_by_form[form]:
                    available = rows.get(variant, {})
                    hit = available.get(language) or available.get("all")
                    if hit:
                        effective[language] = {
                            "lemma": hit[0],
                            "source": hit[1],
                            "variant": variant,
                            "row_lang": language if language in available else "all",
                        }
                        break
            if (
                "el" in effective
                and "grc" in effective
                and effective["el"]["lemma"] != effective["grc"]["lemma"]
            ):
                conflict_types += 1
                conflict_tokens += count
                examples.append({"form": form, "count": count, **effective})
        examples.sort(key=lambda item: (-item["count"], item["form"]))
        return {
            "tokens": conflict_tokens,
            "types": conflict_types,
            "examples": examples[:40],
        }


def _normalize_counts(counter: Counter[str]) -> Counter[str]:
    normalized: Counter[str] = Counter()
    for form, count in counter.items():
        normalized[unicodedata.normalize("NFC", form)] += count
    return normalized


def _percent(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 4) if denominator else 0.0


def measure_coverage(
    sample: RawSample,
    *,
    lookup_lang: str,
    lookup_db: Path = LOOKUP_DB_PATH,
) -> dict:
    """Measure structured Dilemma coverage without invoking a model or identity."""
    forms = _normalize_counts(sample.tokens)
    dilemma = Dilemma(
        lang=lookup_lang,
        resolve_articles=True,
        skip_pos=True,
    )
    dilemma.preload()
    lexical_tokens = 0
    lexical_types = 0
    covered_tokens = 0
    covered_types = 0
    unknown: list[dict] = []
    source_tokens: Counter[str] = Counter()
    source_types: Counter[str] = Counter()
    nonlexical_tokens: Counter[str] = Counter()
    nonlexical_types: Counter[str] = Counter()

    for form, count in forms.items():
        nonlexical = dilemma.classify_nonlexical(form)
        if nonlexical:
            nonlexical_tokens[nonlexical] += count
            nonlexical_types[nonlexical] += 1
            continue
        lexical_tokens += count
        lexical_types += 1
        candidates = [
            candidate
            for candidate in dilemma.lemmatize_verbose(form, guess=False)
            if candidate.source != "nonlexical"
        ]
        if candidates:
            covered_tokens += count
            covered_types += 1
            source = candidates[0].source or "unknown"
            source_tokens[source] += count
            source_types[source] += 1
        else:
            unknown.append({
                "form": form,
                "count": count,
                "defects": list(token_defects(form)),
            })
    unknown.sort(key=lambda item: (-item["count"], item["form"]))
    conflicts = LookupConflictResolver(lookup_db).conflicts(forms)
    return {
        "lookup_lang": lookup_lang,
        "guess": False,
        "lexical_tokens": lexical_tokens,
        "lexical_types": lexical_types,
        "covered_tokens": covered_tokens,
        "covered_types": covered_types,
        "token_coverage_percent": _percent(covered_tokens, lexical_tokens),
        "type_coverage_percent": _percent(covered_types, lexical_types),
        "unknown_tokens": lexical_tokens - covered_tokens,
        "unknown_types": lexical_types - covered_types,
        "unknown_examples": unknown[:100],
        "first_candidate_sources": {
            source: {
                "tokens": source_tokens[source],
                "types": source_types[source],
            }
            for source in sorted(source_tokens)
        },
        "excluded_nonlexical": {
            label: {
                "tokens": nonlexical_tokens[label],
                "types": nonlexical_types[label],
            }
            for label in sorted(nonlexical_tokens)
        },
        "lookup_language_conflicts": conflicts,
    }


def _clean_mg_form(form: str) -> str | None:
    normalized = unicodedata.normalize("NFC", form)
    if token_defects(normalized):
        return None
    if Dilemma.classify_nonlexical(normalized):
        return None
    for char in normalized:
        if unicodedata.category(char).startswith("L") and not _is_greek_letter(char):
            return None
        if char.isdigit():
            return None
    normalized = to_monotonic(normalized.lower())
    return normalized if normalized else None


def mg_frequency_counts(sample: RawSample, token_cap: int) -> Counter[str]:
    """Create a clean monotonic MG form counter from a capped corpus sample."""
    result: Counter[str] = Counter()
    consumed = 0
    for raw, count in sample.tokens.most_common():
        clean = _clean_mg_form(raw)
        if clean is None:
            continue
        add = min(count, token_cap - consumed)
        if add <= 0:
            break
        result[clean] += add
        consumed += add
    return result


def load_frequency_ranks(path: Path = DEFAULT_MG_FREQUENCY) -> dict[str, int]:
    ranks = {}
    with path.open(encoding="utf-8") as handle:
        for rank, line in enumerate(handle, start=1):
            parts = line.split()
            if len(parts) >= 2 and parts[0] not in ranks:
                ranks[parts[0]] = rank
    return ranks


def build_frequency_experiment(
    corpora: list[dict],
    samples: dict[str, RawSample],
    *,
    existing_frequency: Path = DEFAULT_MG_FREQUENCY,
) -> dict:
    """Combine equal/capped source samples without mutating runtime data."""
    source_counts: dict[str, Counter[str]] = {}
    combined: Counter[str] = Counter()
    source_reports = {}
    for corpus in corpora:
        cap = corpus.get("frequency_token_cap", 0)
        if cap <= 0 or corpus["key"] not in samples:
            continue
        counts = mg_frequency_counts(samples[corpus["key"]], cap)
        source_counts[corpus["key"]] = counts
        combined.update(counts)
        source_reports[corpus["key"]] = {
            "role": corpus["role"],
            "declared_token_cap": cap,
            "eligible_tokens": sum(counts.values()),
            "types": len(counts),
            "top_forms": counts.most_common(100),
        }

    existing = load_frequency_ranks(existing_frequency)
    combined_rank = {
        form: rank
        for rank, (form, _count) in enumerate(combined.most_common(), start=1)
    }
    comparisons = {}
    for n in (100, 1000, 10000):
        old_top = {form for form, rank in existing.items() if rank <= n}
        new_top = {form for form, rank in combined_rank.items() if rank <= n}
        comparisons[f"top_{n}_overlap"] = len(old_top & new_top)
        comparisons[f"top_{n}_overlap_percent"] = _percent(
            len(old_top & new_top), min(n, len(new_top))
        )
    changes = []
    for form, new_rank in combined_rank.items():
        old_rank = existing.get(form)
        if old_rank is not None:
            changes.append({
                "form": form,
                "old_rank": old_rank,
                "experimental_rank": new_rank,
                "rank_change": old_rank - new_rank,
            })
    changes.sort(key=lambda item: (-abs(item["rank_change"]), item["form"]))

    return {
        "status": "experiment_only",
        "replaces_mg_freq": False,
        "method": (
            "Concatenate clean monotonic Greek form counts from equal 1M-token "
            "OpenGov/MITOS/EELLAK samples and 250k-token samples of each legal "
            "corpus; do not scale historical corpora or short samples upward."
        ),
        "sources": source_reports,
        "combined_tokens": sum(combined.values()),
        "combined_types": len(combined),
        "combined_top_forms": combined.most_common(1000),
        "existing_mg_freq_comparison": comparisons,
        "largest_absolute_rank_changes": changes[:100],
        "new_top_1000_forms": [
            form for form, _count in combined.most_common(1000)
            if form not in existing
        ],
    }


def audit_corpus(
    corpus: dict,
    *,
    source: str,
    max_tokens: int | None,
    max_documents: int,
) -> tuple[dict, RawSample]:
    declared_cap = corpus["audit_token_cap"]
    token_cap = min(declared_cap, max_tokens) if max_tokens else declared_cap
    sample = collect_corpus_sample(
        corpus,
        source=source,
        token_cap=token_cap,
        max_documents=max_documents,
    )
    if sample.token_count == 0:
        raise AuditError(
            f"{corpus['key']}: configured reader produced no Greek-bearing tokens"
        )
    coverage = measure_coverage(sample, lookup_lang=corpus["lookup_lang"])
    report = {
        "repo": corpus["repo"],
        "revision": corpus["revision"],
        "license": corpus["license"],
        "role": corpus["role"],
        "sample": {
            "method": "row_group_stratified_deterministic_windows",
            "declared_token_cap": declared_cap,
            "effective_token_cap": token_cap,
            "documents": sample.documents,
            "greek_bearing_tokens": sample.token_count,
            "greek_bearing_types": len(sample.tokens),
        },
        "unicode_ocr_defects": sample.defect_report(),
        "coverage": coverage,
    }
    if corpus.get("ogc_decision"):
        report["ogc_decision"] = corpus["ogc_decision"]
    if corpus.get("known_quality_risks"):
        report["known_quality_risks"] = corpus["known_quality_risks"]
    return report, sample


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _print_summary(key: str, report: dict) -> None:
    coverage = report["coverage"]
    sample = report["sample"]
    conflicts = coverage["lookup_language_conflicts"]
    print(
        f"{key}: {sample['greek_bearing_tokens']:,} tokens, "
        f"coverage {coverage['token_coverage_percent']:.2f}% tokens / "
        f"{coverage['type_coverage_percent']:.2f}% types, "
        f"unknown {coverage['unknown_types']:,} types, "
        f"lookup conflicts {conflicts['types']:,} types"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--source", choices=("cache", "stream", "download"), default="cache",
        help="read verified cache bytes, range-stream them, or download them",
    )
    parser.add_argument(
        "--corpus", action="append", default=[],
        help="audit only this manifest key (repeatable)",
    )
    parser.add_argument(
        "--exclude-corpus", action="append", default=[],
        help="explicitly defer this manifest key (repeatable)",
    )
    parser.add_argument(
        "--max-tokens", type=int,
        help="lower every corpus cap for smoke tests",
    )
    parser.add_argument("--max-documents", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--frequency-output", type=Path, default=DEFAULT_FREQUENCY_REPORT
    )
    parser.add_argument("--skip-frequency", action="store_true")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    all_corpora = manifest["corpora"]
    selected = all_corpora
    known = {corpus["key"] for corpus in all_corpora}
    if args.corpus:
        wanted = set(args.corpus)
        unknown = wanted - known
        if unknown:
            parser.error(f"unknown corpus key(s): {', '.join(sorted(unknown))}")
        selected = [corpus for corpus in selected if corpus["key"] in wanted]
    excluded = set(args.exclude_corpus)
    unknown = excluded - known
    if unknown:
        parser.error(f"unknown excluded corpus key(s): {', '.join(sorted(unknown))}")
    overlap = excluded & set(args.corpus)
    if overlap:
        parser.error(
            "corpus key(s) are both selected and excluded: "
            + ", ".join(sorted(overlap))
        )
    selected = [corpus for corpus in selected if corpus["key"] not in excluded]

    reports = {}
    samples = {}
    failures = {}
    for corpus in selected:
        print(f"auditing {corpus['key']}...", file=sys.stderr)
        try:
            report, sample = audit_corpus(
                corpus,
                source=args.source,
                max_tokens=args.max_tokens,
                max_documents=args.max_documents,
            )
        except AuditError as exc:
            failures[corpus["key"]] = str(exc)
            print(f"  unavailable: {exc}", file=sys.stderr)
            continue
        reports[corpus["key"]] = report
        samples[corpus["key"]] = sample
        _print_summary(corpus["key"], report)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha256(args.manifest),
        "source_mode": args.source,
        "corpora": reports,
        "unavailable": failures,
        "excluded": {
            key: "explicitly excluded by command line"
            for key in sorted(excluded)
        },
    }
    _write_json(args.output, payload)
    print(f"coverage report: {args.output}")

    if not args.skip_frequency:
        frequency = build_frequency_experiment(all_corpora, samples)
        missing_frequency_sources = [
            corpus["key"]
            for corpus in all_corpora
            if corpus["frequency_token_cap"] > 0
            and corpus["key"] not in samples
        ]
        frequency.update({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": payload["manifest_sha256"],
            "complete": not missing_frequency_sources,
            "missing_sources": missing_frequency_sources,
        })
        _write_json(args.frequency_output, frequency)
        print(f"frequency experiment: {args.frequency_output}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

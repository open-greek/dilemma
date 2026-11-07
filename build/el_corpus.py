"""Pinned, openly licensed monotonic Modern Greek corpora: manifest,
license policy, input verification, document readers, the document-level
train/dev split, token caps, and the compact shards the extractor writes.

``data/el_lm_corpora.json`` lists every source with its license, the
evidence for that license, pinned inputs (Hugging Face revision or URL,
byte size and sha256), its role, register, token cap and count weight, and
how to read it. :func:`load_manifest` refuses a manifest in which a source
that can be read carries a NonCommercial or NoDerivatives license, a
license outside :data:`ALLOWED_LICENSES`, or an input without a full pin.
Sources the policy keeps out are listed under ``excluded`` with the reason,
so the policy is visible in one place.

A document is the smallest unit a reader yields (a speech, a comment, an
article, a sentence of a sentence collection). Each document has a stable
``id`` and a ``group``: the unit the train/dev split hashes. Groups are
chosen so that text that repeats lands on one side: a parliamentary
sitting, a public consultation, an article, an entry. See
:func:`split_of`.

Shards are gzip-compressed JSON lines, one document per line::

    {"id": "parliament:row:123", "group": "parliament:...", "s": ["Ο κ. ..."]}

``s`` holds the kept sentences, each a string of tokens separated by single
spaces (tokens never contain spaces; see ``el_text``).
"""

from __future__ import annotations

import bz2
import csv
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "el_lm_corpora.json"
DEFAULT_OUT_DIR = ROOT / "build" / "el_corpus"

SPLIT_SEED = 4242
CAP_SEED = 1717
BUCKETS = 10_000

ROLES = {"el_lm_core", "el_lm_capped", "el_lm_conversational"}
# SPDX identifiers (and CC BY 2.0 FR, which SPDX lacks) that may be read.
ALLOWED_LICENSES = {
    "CC0-1.0", "CC-BY-2.0-FR", "CC-BY-3.0", "CC-BY-4.0",
    "CC-BY-SA-3.0", "CC-BY-SA-4.0",
}
_REFUSED_LICENSE_RE = re.compile(r"(^|-)(NC|ND)(-|$)", re.IGNORECASE)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class CorpusError(RuntimeError):
    """A manifest, license, pin or input problem."""


# --------------------------------------------------------------------------
# Manifest and license policy
# --------------------------------------------------------------------------

def license_refusal(license_id: str) -> str | None:
    """Why ``license_id`` may not be read, or None when it may."""
    if not license_id:
        return "no license recorded"
    if _REFUSED_LICENSE_RE.search(license_id):
        return f"{license_id} is NonCommercial or NoDerivatives"
    if license_id not in ALLOWED_LICENSES:
        return f"{license_id} is not on the allowed license list"
    return None


def _validate_fetch(key: str, fetch: dict) -> None:
    kind = fetch.get("kind")
    if kind == "hf_dataset":
        if not fetch.get("repo"):
            raise CorpusError(f"{key}: hf_dataset needs a repo")
        if not _COMMIT_RE.fullmatch(fetch.get("revision", "")):
            raise CorpusError(f"{key}: revision must be a full commit sha")
        files = fetch.get("files")
    elif kind == "url":
        if not fetch.get("url") or not fetch.get("filename"):
            raise CorpusError(f"{key}: url fetch needs url and filename")
        files = [fetch]
    elif kind == "local":
        if not fetch.get("base"):
            raise CorpusError(f"{key}: local fetch needs a base")
        files = fetch.get("files")
    else:
        raise CorpusError(f"{key}: unknown fetch kind {kind!r}")
    if not isinstance(files, list) or not files:
        raise CorpusError(f"{key}: no pinned files")
    for f in files:
        name = f.get("path") or f.get("filename")
        if not name:
            raise CorpusError(f"{key}: a pinned file has no path")
        if not isinstance(f.get("size"), int) or f["size"] <= 0:
            raise CorpusError(f"{key}/{name}: invalid size")
        if not _SHA256_RE.fullmatch(f.get("sha256", "")):
            raise CorpusError(f"{key}/{name}: invalid sha256")


def validate_manifest(manifest: dict) -> None:
    """Refuse floating pins, refused licenses and malformed entries."""
    if manifest.get("schema_version") != 1:
        raise CorpusError("unsupported el_lm_corpora schema")
    for field in ("max_sentence_repeats", "dedup_min_tokens"):
        value = manifest.get(field)
        if not isinstance(value, int) or value < 1:
            raise CorpusError(f"{field} must be an integer >= 1")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise CorpusError("manifest has no sources")
    keys: set[str] = set()
    for src in sources:
        key = src.get("key")
        if not key or not re.fullmatch(r"[a-z][a-z0-9_]*", key) or key in keys:
            raise CorpusError(f"missing, malformed or duplicate key {key!r}")
        keys.add(key)
        if src.get("role") not in ROLES:
            raise CorpusError(f"{key}: invalid role {src.get('role')!r}")
        if not src.get("license_evidence"):
            raise CorpusError(f"{key}: license_evidence is missing")
        refusal = license_refusal(src.get("license", ""))
        if src.get("enabled", True):
            if refusal:
                raise CorpusError(f"{key}: refused: {refusal}")
        elif not src.get("disabled_reason"):
            raise CorpusError(f"{key}: a disabled source needs disabled_reason")
        cap = src.get("token_cap")
        if cap is not None and (not isinstance(cap, int) or cap <= 0):
            raise CorpusError(f"{key}: token_cap must be null or > 0")
        weight = src.get("weight", 1)
        if not isinstance(weight, int) or weight < 1:
            raise CorpusError(f"{key}: weight must be an integer >= 1")
        frac = src.get("dev_fraction", manifest.get("split", {}).get(
            "dev_fraction", 0.02))
        if not (0 < frac < 0.5):
            raise CorpusError(f"{key}: dev_fraction out of range")
        _validate_fetch(key, src.get("fetch", {}))
        if not src.get("reader", {}).get("kind"):
            raise CorpusError(f"{key}: reader kind is missing")
    for ex in manifest.get("excluded", []):
        if not ex.get("key") or not ex.get("reason") or not ex.get("license"):
            raise CorpusError(f"excluded entry needs key, license and reason: {ex}")
        if ex["key"] in keys:
            raise CorpusError(f"{ex['key']} is both a source and excluded")


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    validate_manifest(manifest)
    return manifest


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def raw_dir() -> Path:
    """Where URL downloads live (outside git): ``$DILEMMA_EL_RAW_DIR`` or
    ``data/raw`` of the checkout."""
    env = os.environ.get("DILEMMA_EL_RAW_DIR")
    return Path(env) if env else ROOT / "data" / "raw"


def kaikki_dir() -> Path:
    """The kaikki.org Wiktionary dumps: ``$DILEMMA_KAIKKI_DIR``, else the
    checkout's ``kaikki/``, else the main checkout's (for worktrees)."""
    env = os.environ.get("DILEMMA_KAIKKI_DIR")
    if env:
        return Path(env)
    here = ROOT / "kaikki"
    if here.exists():
        return here
    try:
        common = subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return here
    return Path(common).parent / "kaikki"


def _verify(path: Path, pin: dict, verified: dict | None) -> None:
    size = path.stat().st_size
    if size != pin["size"]:
        raise CorpusError(f"{path}: size {size}, manifest pins {pin['size']}")
    cache_key = (str(path), size, path.stat().st_mtime_ns)
    if verified is not None and verified.get(cache_key) == pin["sha256"]:
        return
    digest = file_sha256(path)
    if digest != pin["sha256"]:
        raise CorpusError(
            f"{path}: sha256 {digest}, manifest pins {pin['sha256']}. The "
            f"upstream file changed or the download is incomplete; delete it "
            f"to download again, or update the pin after reviewing the change."
        )
    if verified is not None:
        verified[cache_key] = digest


def _download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"  downloading {url}", flush=True)
    with urllib.request.urlopen(url) as resp, open(part, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    part.replace(dest)


def resolve_inputs(src: dict, *, download: bool = False,
                   verified: dict | None = None) -> list[Path]:
    """Local paths of a source's pinned inputs, verified against the pins.

    Hugging Face files come from the HF cache (downloaded when ``download``
    is true); URL files from :func:`raw_dir`; local files from their base
    directory. Every file's size and sha256 must match the manifest.
    """
    fetch = src["fetch"]
    paths: list[Path] = []
    if fetch["kind"] == "hf_dataset":
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        from huggingface_hub import hf_hub_download

        for pin in fetch["files"]:
            try:
                p = Path(hf_hub_download(
                    repo_id=fetch["repo"], filename=pin["path"],
                    repo_type="dataset", revision=fetch["revision"],
                    local_files_only=not download))
            except Exception as exc:
                raise CorpusError(
                    f"{src['key']}: {fetch['repo']}/{pin['path']} is not "
                    f"available ({type(exc).__name__}); rerun with --download"
                ) from exc
            _verify(p, pin, verified)
            paths.append(p)
    elif fetch["kind"] == "url":
        p = raw_dir() / src["key"] / fetch["filename"]
        if not p.exists():
            if not download:
                raise CorpusError(
                    f"{src['key']}: {p} is missing; rerun with --download")
            _download(fetch["url"], p)
        _verify(p, fetch, verified)
        paths.append(p)
    elif fetch["kind"] == "local":
        if fetch["base"] != "kaikki":
            raise CorpusError(f"{src['key']}: unknown local base {fetch['base']}")
        base = kaikki_dir()
        for pin in fetch["files"]:
            p = base / pin["path"]
            if not p.exists():
                raise CorpusError(f"{src['key']}: {p} is missing")
            _verify(p, pin, verified)
            paths.append(p)
    return paths


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Document:
    id: str
    group: str
    text: str


def _iter_parliament_csv(src: dict, paths: list[Path]) -> Iterator[Document]:
    rd = src["reader"]
    key = src["key"]
    csv.field_size_limit(sys.maxsize)
    with zipfile.ZipFile(paths[0]) as zf, zf.open(rd["member"]) as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8",
                                                 newline=""))
        for row_no, row in enumerate(reader):
            sitting = ":".join(row.get(c) or "" for c in rd["group_columns"])
            yield Document(f"{key}:row:{row_no}", f"{key}:{sitting}",
                           row.get(rd["text_column"]) or "")


def _iter_opengov_comments(src: dict, paths: list[Path]) -> Iterator[Document]:
    import pyarrow.parquet as pq

    key = src["key"]
    for path in paths:
        table = pq.read_table(path, columns=["consultation_id", "articles"])
        cids = table.column("consultation_id").to_pylist()
        articles = table.column("articles").to_pylist()
        tag = Path(path).stem
        for cid, arts in zip(cids, articles):
            for art in arts or []:
                for com in art.get("comments") or []:
                    yield Document(
                        f"{key}:{tag}:{com['comment_row_id']}",
                        f"{key}:{cid}", com.get("content") or "")


def _iter_parquet_text(src: dict, paths: list[Path]) -> Iterator[Document]:
    import pyarrow.parquet as pq

    rd = src["reader"]
    key = src["key"]
    id_field, text_field = rd["id_field"], rd["text_field"]
    group_field = rd.get("group_field")
    skip_field = rd.get("skip_if_field_matches", {})
    skip_re = {f: re.compile(p) for f, p in skip_field.items()}
    columns = sorted({id_field, text_field, *( [group_field] if group_field else []),
                      *skip_re})
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2048, columns=columns):
            cols = {c: batch.column(c).to_pylist() for c in columns}
            for k in range(batch.num_rows):
                if any(isinstance(cols[f][k], str) and r.search(cols[f][k])
                       for f, r in skip_re.items()):
                    continue
                doc_id = f"{key}:{cols[id_field][k]}"
                group = (f"{key}:{cols[group_field][k]}" if group_field
                         else doc_id)
                yield Document(doc_id, group, cols[text_field][k] or "")


def _iter_tatoeba(src: dict, paths: list[Path]) -> Iterator[Document]:
    key = src["key"]
    with bz2.open(paths[0], "rt", encoding="utf-8", newline="\n") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sid = f"{key}:{parts[0]}"
            yield Document(sid, sid, parts[2])


def _iter_text_lines(src: dict, paths: list[Path]) -> Iterator[Document]:
    key = src["key"]
    with open(paths[0], encoding="utf-8", newline="\n") as fh:
        for line_no, line in enumerate(fh):
            text = line.strip()
            if text:
                did = f"{key}:line:{line_no}"
                yield Document(did, did, text)


def _iter_kaikki_examples(src: dict, paths: list[Path]) -> Iterator[Document]:
    """Usage examples of Modern Greek entries in the kaikki.org dumps.

    The group is the entry's word, lowercased, so every example filed under
    one word is on one side of the split whichever edition holds it.
    """
    key = src["key"]
    lang_code = src["reader"].get("lang_code", "el")
    for path in paths:
        edition = "el" if "-el-dictionary-" in path.name else "en"
        with open(path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh):
                if '"examples"' not in line:
                    continue
                entry = json.loads(line)
                if entry.get("lang_code") != lang_code:
                    continue
                word = (entry.get("word") or "").lower()
                for si, sense in enumerate(entry.get("senses") or []):
                    for ei, ex in enumerate(sense.get("examples") or []):
                        text = ex.get("text") or ""
                        if text:
                            yield Document(
                                f"{key}:{edition}:{line_no}:{si}:{ei}",
                                f"{key}:{word}", text)


READERS = {
    "parliament_csv": _iter_parliament_csv,
    "opengov_comments": _iter_opengov_comments,
    "parquet_text": _iter_parquet_text,
    "tatoeba_tsv_bz2": _iter_tatoeba,
    "text_lines": _iter_text_lines,
    "kaikki_examples": _iter_kaikki_examples,
}


def iter_documents(src: dict, paths: list[Path],
                   max_docs: int | None = None) -> Iterator[Document]:
    """Yield a source's documents in a fixed order (the input's order)."""
    reader = READERS.get(src["reader"]["kind"])
    if reader is None:
        raise CorpusError(f"{src['key']}: no reader {src['reader']['kind']!r}")
    for n, doc in enumerate(reader(src, paths)):
        if max_docs is not None and n >= max_docs:
            return
        yield doc


def text_options(src: dict) -> dict:
    """Keyword arguments for ``el_text.process_document``."""
    rd = src["reader"]
    return {
        "line_breaks": rd.get("line_breaks", "sentence"),
        "require_terminal_punctuation": bool(
            rd.get("require_terminal_punctuation", False)),
        "stop_at_lines": frozenset(rd.get("stop_at_lines", [])),
    }


# --------------------------------------------------------------------------
# Split and caps
# --------------------------------------------------------------------------

def _bucket(text: str) -> int:
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h[:4], "little") % BUCKETS


def split_of(group: str, dev_fraction: float, seed: int = SPLIT_SEED) -> str:
    """``"dev"`` or ``"train"`` for a document group.

    The hash is of the group, not of a sentence or document, so a sitting,
    a consultation or an article is never on both sides. Sentences that
    still repeat across groups are removed from train by the extractor.
    """
    return "dev" if _bucket(f"{seed}:{group}") < round(dev_fraction * BUCKETS) else "train"


def cap_bucket(doc_id: str, seed: int = CAP_SEED) -> int:
    """A document's position in [0, BUCKETS) for token caps.

    A cap keeps the training documents whose bucket is below the threshold
    the extractor records, so a capped source is a fixed, document-level
    sample that does not depend on reading order. The seed differs from
    the split seed, so the sample is independent of the split.
    """
    return _bucket(f"cap:{seed}:{doc_id}")


def cap_threshold(bucket_tokens: list[int], cap: int | None) -> tuple[int, int]:
    """The bucket threshold that keeps at most ``cap`` tokens, and the
    tokens it keeps. ``cap`` None keeps everything."""
    total = sum(bucket_tokens)
    if cap is None or total <= cap:
        return BUCKETS, total
    kept = 0
    for b, n in enumerate(bucket_tokens):
        if kept + n > cap:
            return b, kept
        kept += n
    return BUCKETS, kept


# --------------------------------------------------------------------------
# Shards
# --------------------------------------------------------------------------

class ShardWriter:
    """Write one source's documents of one split as rotating gzip shards.

    Output is byte-stable: gzip headers carry no name and no time, and the
    documents keep the reader's order.
    """

    def __init__(self, out_dir: Path, split: str, shard_tokens: int = 20_000_000):
        self.out_dir = Path(out_dir)
        self.split = split
        self.shard_tokens = shard_tokens
        self.shards: list[dict] = []
        self._fh = None
        self._raw = None
        self._cur: dict | None = None

    def _open(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{self.split}-{len(self.shards):05d}.jsonl.gz"
        path = self.out_dir / name
        self._raw = open(path, "wb")
        self._fh = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw,
                                 mtime=0, compresslevel=6)
        self._cur = {"file": name, "documents": 0, "sentences": 0, "tokens": 0}

    def write(self, doc_id: str, group: str, sentences: list[str], tokens: int) -> None:
        if self._fh is None:
            self._open()
        line = json.dumps({"id": doc_id, "group": group, "s": sentences},
                          ensure_ascii=False, separators=(",", ":"))
        self._fh.write(line.encode("utf-8") + b"\n")
        self._cur["documents"] += 1
        self._cur["sentences"] += len(sentences)
        self._cur["tokens"] += tokens
        if self._cur["tokens"] >= self.shard_tokens:
            self._close_current()

    def _close_current(self) -> None:
        if self._fh is None:
            return
        self._fh.close()
        self._raw.close()
        self._cur["sha256"] = file_sha256(self.out_dir / self._cur["file"])
        self.shards.append(self._cur)
        self._fh = self._raw = self._cur = None

    def close(self) -> list[dict]:
        self._close_current()
        return self.shards


def iter_shard_documents(path: Path) -> Iterator[dict]:
    """Documents of one shard: dicts with ``id``, ``group`` and ``s``."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def iter_corpus_sentences(corpus_dir: Path, source: str, split: str,
                          *, cap: bool = True) -> Iterator[tuple[str, list[str]]]:
    """``(sentence_id, tokens)`` for one source and split of an extracted
    corpus, applying the source's recorded token cap to train when ``cap``.

    Sentence ids are ``<document id>:<index>``, stable across runs.
    """
    corpus_dir = Path(corpus_dir)
    meta = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    entry = meta["sources"][source]
    threshold = BUCKETS
    if cap and split == "train":
        threshold = entry["cap"]["bucket_threshold"]
    for shard in entry["splits"][split]["shards"]:
        for doc in iter_shard_documents(corpus_dir / source / shard["file"]):
            if threshold < BUCKETS and cap_bucket(doc["id"]) >= threshold:
                continue
            for k, s in enumerate(doc["s"]):
                yield f"{doc['id']}:{k}", s.split(" ")

#!/usr/bin/env python3
"""Archive the pinned kaikki.org dumps on the Hub, and restore them from it.

``data/kaikki_inputs.json`` pins one snapshot of the Wiktionary extracts the
builders read (``build_data.py``, ``build/extract_el_dialect_sections.py``):
per file its source URL, kaikki's Last-Modified date, size, sha256, license,
and a Hub archive path. kaikki.org replaces its files in place, so a build
can only be repeated from such an archive.

    # Gzip and verify every pinned file into a staging directory, and record
    # each archive's size and sha256 in the pin file (the revision stays
    # "pending").
    python scripts/archive_kaikki_inputs.py stage --src KAIKKI_DIR --out-dir STAGE
    # After uploading STAGE/* to data/kaikki_inputs/<snapshot>/ on the Hub:
    python scripts/archive_kaikki_inputs.py pin-revision FULL_SHA --out-dir STAGE
    # Fetch the archives at the pinned revision into a flat dump directory
    # (the layout build_data.py --kaikki reads), verifying every sha256.
    python scripts/archive_kaikki_inputs.py download --out KAIKKI_DIR
    # Check a flat dump directory against the pins.
    python scripts/archive_kaikki_inputs.py verify KAIKKI_DIR

No file's content is changed: an archive decompresses to the pinned bytes,
and a file kaikki already serves gzipped is archived as it is.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "data" / "kaikki_inputs.json"
SCHEMA = 1
REVISION = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")


class PinError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gunzip_sha256(path: Path) -> tuple[int, str]:
    digest, size = hashlib.sha256(), 0
    with gzip.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def load_pins(path: Path = PINS) -> dict:
    pins = json.loads(path.read_text(encoding="utf-8"))
    validate(pins)
    return pins


def write_pins(pins: dict, path: Path = PINS) -> None:
    validate(pins)
    path.write_text(json.dumps(pins, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")


def validate(pins: dict) -> None:
    """Refuse a pin file that does not have the documented shape."""
    if pins.get("schema_version") != SCHEMA:
        raise PinError(f"schema_version must be {SCHEMA}")
    if not re.fullmatch(r"\d{8}", str(pins.get("snapshot", ""))):
        raise PinError("snapshot must be a YYYYMMDD date")
    if not pins.get("repo") or pins.get("repo_type") not in {"model", "dataset"}:
        raise PinError("repo and repo_type are required")
    files = pins.get("files")
    if not files:
        raise PinError("no files are pinned")
    prefix = f"data/kaikki_inputs/{pins['snapshot']}/"
    for name, rec in files.items():
        if "/" in name or not name.startswith("kaikki.org-"):
            raise PinError(f"{name}: not a flat kaikki dump name")
        if not isinstance(rec.get("bytes"), int) or rec["bytes"] <= 0:
            raise PinError(f"{name}: bytes must be a positive integer")
        if not SHA256.fullmatch(rec.get("sha256", "")):
            raise PinError(f"{name}: sha256 must be 64 lowercase hex digits")
        if "CC BY-SA 4.0" not in rec.get("license", ""):
            raise PinError(f"{name}: license must name CC BY-SA 4.0")
        if "derived" in rec:
            derived = rec["derived"]
            if derived.get("from") not in files or not derived.get("by"):
                raise PinError(f"{name}: derived files name their pinned input and builder")
        else:
            if not str(rec.get("url", "")).startswith("https://kaikki.org/"):
                raise PinError(f"{name}: url must be a kaikki.org URL")
            if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ",
                                rec.get("last_modified", "")):
                raise PinError(f"{name}: last_modified must be an ISO UTC time")
        archive = rec.get("archive") or {}
        gz = name.endswith(".gz")
        want = prefix + (name if gz else name + ".gz")
        if archive.get("path") != want:
            raise PinError(f"{name}: archive path must be {want}")
        if archive.get("compression") != ("none" if gz else "gzip"):
            raise PinError(f"{name}: archive compression must be "
                           + ("none" if gz else "gzip"))
        staged = archive.get("sha256") is not None
        if staged != (archive.get("bytes") is not None):
            raise PinError(f"{name}: archive bytes and sha256 are set together")
        if staged and not SHA256.fullmatch(archive["sha256"]):
            raise PinError(f"{name}: archive sha256 must be 64 lowercase hex digits")
        revision = archive.get("revision")
        if revision != "pending" and not REVISION.fullmatch(str(revision)):
            raise PinError(f"{name}: archive revision must be 'pending' or a full commit sha")
        if revision != "pending" and not staged:
            raise PinError(f"{name}: a pinned revision needs the archive's sha256")
        if gz and staged and (archive["bytes"], archive["sha256"]) != (rec["bytes"], rec["sha256"]):
            raise PinError(f"{name}: an uncompressed archive is the pinned file itself")


def check_file(path: Path, rec: dict) -> None:
    if not path.exists():
        raise PinError(f"{path} is missing")
    if path.stat().st_size != rec["bytes"] or sha256(path) != rec["sha256"]:
        raise PinError(f"{path} differs from its pin")


def selected(pins: dict, names) -> dict:
    files = pins["files"]
    unknown = sorted(set(names or ()) - set(files))
    if unknown:
        raise PinError("not pinned: " + ", ".join(unknown))
    return {n: r for n, r in files.items() if not names or n in names}


def stage(src: Path, out_dir: Path, pins_path: Path = PINS, names=None) -> dict:
    """Write each pinned file's archive into ``out_dir`` and verify it.

    A gzip archive is written with no name and mtime 0, so staging the same
    file twice gives the same bytes; it is decompressed and compared with the
    pin before it is kept.
    """
    pins = load_pins(pins_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rec in selected(pins, names).items():
        source = src / name
        check_file(source, rec)
        archive = rec["archive"]
        target = out_dir / Path(archive["path"]).name
        part = target.with_name(target.name + ".part")
        if archive["compression"] == "none":
            shutil.copyfile(source, part)
            check_file(part, rec)
        else:
            with source.open("rb") as raw_in, part.open("wb") as raw_out:
                with gzip.GzipFile(filename="", fileobj=raw_out, mode="wb",
                                   compresslevel=9, mtime=0) as stream:
                    shutil.copyfileobj(raw_in, stream, 1 << 20)
            if gunzip_sha256(part) != (rec["bytes"], rec["sha256"]):
                part.unlink()
                raise PinError(f"{name}: the archive does not decompress to the pinned file")
        part.replace(target)
        size, digest = target.stat().st_size, sha256(target)
        if archive.get("sha256") not in (None, digest) and archive["revision"] != "pending":
            raise PinError(f"{name}: staged archive differs from the one pinned at "
                           f"{archive['revision']}")
        archive.update({"bytes": size, "sha256": digest})
        print(f"  {target.name}: {size:,} bytes, sha256 {digest}")
    write_pins(pins, pins_path)
    return pins


def pin_revision(revision: str, out_dir: Path, pins_path: Path = PINS, names=None) -> dict:
    if not REVISION.fullmatch(revision):
        raise PinError("the revision must be a full Hub commit sha")
    pins = load_pins(pins_path)
    for name, rec in selected(pins, names).items():
        archive = rec["archive"]
        if archive["sha256"] is None:
            raise PinError(f"{name}: stage the archive first")
        check_file(out_dir / Path(archive["path"]).name,
                   {"bytes": archive["bytes"], "sha256": archive["sha256"]})
        archive["revision"] = revision
    write_pins(pins, pins_path)
    return pins


def _hub_fetch(pins: dict, path: str, revision: str) -> Path:
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=pins["repo"], filename=path,
                                repo_type=pins["repo_type"], revision=revision))


def download(out: Path, pins_path: Path = PINS, names=None, revision=None,
             fetch=_hub_fetch) -> dict:
    """Restore the pinned files into the flat directory ``out``.

    Each archive comes from the Hub at its pinned revision (or ``revision``),
    must match the archive's pinned sha256, and must decompress to the
    pinned file. A file already in ``out`` that matches its pin is kept.
    """
    pins = load_pins(pins_path)
    out.mkdir(parents=True, exist_ok=True)
    for name, rec in selected(pins, names).items():
        target = out / name
        if target.exists():
            check_file(target, rec)
            print(f"  {name}: present and verified")
            continue
        archive = rec["archive"]
        rev = revision or archive["revision"]
        if not REVISION.fullmatch(str(rev)):
            raise PinError(f"{name}: no Hub revision is pinned yet (revision {rev!r})")
        fetched = fetch(pins, archive["path"], rev)
        check_file(fetched, {"bytes": archive["bytes"], "sha256": archive["sha256"]})
        part = target.with_name(target.name + ".part")
        if archive["compression"] == "none":
            shutil.copyfile(fetched, part)
        else:
            with gzip.open(fetched, "rb") as stream, part.open("wb") as raw_out:
                shutil.copyfileobj(stream, raw_out, 1 << 20)
        try:
            check_file(part, rec)
        except PinError:
            part.unlink()
            raise
        part.replace(target)
        print(f"  {name}: restored from {archive['path']}@{rev[:12]}")
    sums = "".join(f"{rec['sha256']}  {name}\n"
                   for name, rec in sorted(pins["files"].items())
                   if (out / name).exists())
    (out / "SHA256SUMS").write_text(sums, encoding="utf-8")
    return pins


def verify(directory: Path, pins_path: Path = PINS, names=None) -> None:
    pins = load_pins(pins_path)
    for name, rec in selected(pins, names).items():
        check_file(directory / name, rec)
        print(f"  {name}: ok")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pins", type=Path, default=PINS)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("stage", help="gzip and verify the pinned files")
    p.add_argument("--src", type=Path, required=True, help="flat kaikki dump directory")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("files", nargs="*")
    p = sub.add_parser("pin-revision", help="pin the Hub commit that holds the archives")
    p.add_argument("revision")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="the staging directory the upload was made from")
    p.add_argument("files", nargs="*")
    p = sub.add_parser("download", help="restore the pinned files from the Hub")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--revision", help="override the pinned revision")
    p.add_argument("files", nargs="*")
    p = sub.add_parser("verify", help="check a flat dump directory against the pins")
    p.add_argument("directory", type=Path)
    p.add_argument("files", nargs="*")
    args = parser.parse_args(argv)
    try:
        if args.command == "stage":
            stage(args.src, args.out_dir, args.pins, args.files)
        elif args.command == "pin-revision":
            pin_revision(args.revision, args.out_dir, args.pins, args.files)
            print(f"Pinned the archives at {args.revision}")
        elif args.command == "download":
            download(args.out, args.pins, args.files, args.revision)
        else:
            verify(args.directory, args.pins, args.files)
    except PinError as exc:
        sys.exit(f"error: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

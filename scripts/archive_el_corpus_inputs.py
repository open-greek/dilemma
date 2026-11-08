#!/usr/bin/env python3
"""Archive pinned source documents, then pin an uploaded Hub revision.

No source text is corrected or normalized here. Wiktionary document ids and
groups remain unchanged so the corpus's held-out split is preserved.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dilemma_build import el_corpus as C

REPO = "ciscoriordan/dilemma"
ARCHIVE_DIR = ROOT / "data/el_lm_inputs"
FILES = {"tatoeba": "tatoeba_ell_sentences.tsv.bz2",
         "wiktionary_examples": "wiktionary_examples.jsonl.gz"}


def file_pin(path: Path) -> dict:
    return {"size": path.stat().st_size, "sha256": C.file_sha256(path)}


def stage(manifest_path: Path, out_dir: Path) -> dict:
    manifest = C.load_manifest(manifest_path)
    sources = {s["key"]: s for s in manifest["sources"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"schema_version": 1, "sources": {}}
    for key, filename in FILES.items():
        source = sources[key]
        if source["fetch"]["kind"] == "hf_model":
            raise C.CorpusError("Source is already archived; use its upstream pins to refresh it")
        paths = C.resolve_inputs(source)
        target = out_dir / filename
        part = target.with_name(target.name + ".part")
        if key == "tatoeba":
            path, = paths
            shutil.copyfile(path, part)
            documents = None
        else:
            if source["reader"]["kind"] != "kaikki_examples":
                raise C.CorpusError("Wiktionary source must use the pinned kaikki reader")
            documents = 0
            with part.open("wb") as raw:
                with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as fh:
                    for document in C.iter_documents(source, paths):
                        row = json.dumps(asdict(document), ensure_ascii=False,
                                         separators=(",", ":")) + "\n"
                        fh.write(row.encode("utf-8"))
                        documents += 1
        part.replace(target)
        provenance["sources"][key] = {
            "license": source["license"], "license_evidence": source["license_evidence"],
            "license_url": source.get("license_url"), "attribution": source["attribution"],
            "upstream_fetch": source["fetch"], "upstream_reader": source["reader"],
            "source_inputs": [{"file": p.name, **file_pin(p)} for p in paths],
            "archive": {"file": filename, **file_pin(target)},
            "documents": documents, "text_modified": False,
            "document_ids_and_groups_preserved": True}
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return provenance


def pin_revision(manifest_path: Path, out_dir: Path, revision: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise C.CorpusError("Archive revision must be a full Hub commit sha")
    provenance = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    manifest = C.load_manifest(manifest_path)
    for source in manifest["sources"]:
        key = source["key"]
        if key not in FILES:
            continue
        record = provenance["sources"][key]
        path = out_dir / FILES[key]
        if file_pin(path) != {k: record["archive"][k] for k in ("size", "sha256")}:
            raise C.CorpusError(f"Archive changed after staging: {key}")
        source["archive_source"] = {"fetch": record["upstream_fetch"],
                                     "reader": record["upstream_reader"]}
        source["fetch"] = {
            "kind": "hf_model", "repo": REPO, "revision": revision,
            "files": [{"path": "data/el_lm_inputs/" + path.name, **file_pin(path)}]}
        if key == "wiktionary_examples":
            source["reader"] = {"kind": "archived_documents",
                                 "note": "Unmodified usage examples, original document ids and groups"}
    C.validate_manifest(manifest)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=C.DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=ARCHIVE_DIR)
    parser.add_argument("--pin-revision", help="After hf_data.py push, pin this full Hub commit")
    args = parser.parse_args()
    if args.pin_revision:
        pin_revision(args.manifest, args.out_dir, args.pin_revision)
        print(f"Pinned the two corpus archives at {args.pin_revision}")
    else:
        result = stage(args.manifest, args.out_dir)
        print(json.dumps({k: v["archive"] for k, v in result["sources"].items()},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()

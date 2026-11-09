#!/usr/bin/env python3
"""Repair reviewed source-sense tags in the pinned 1.3.17 Modern snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_data as B
from dilemma.el_source_tenses import CANONICAL_LEMMAS, correct_sense_tense

BASE_REVISION = "c5ad6aff4c9358b0e4d997a7bb553db9afcd047a"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_evidence(kaikki):
    contexts, witnesses, hashes = [], [], {}
    for edition, name in B.DUMPS["el"].items():
        path = B.resolve_dump(name, kaikki)
        hashes[edition] = sha(path)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                if entry.get("pos") != "verb":
                    continue
                form = entry.get("word")
                if form == "άχθηκα":
                    for sense in entry.get("senses", []):
                        tags = set(entry.get("tags", [])) | set(sense.get("tags", []))
                        _, change = correct_sense_tense(form, edition, sense, tags)
                        if change:
                            contexts.append({"edition": edition, "form": form,
                                             "sense": sense, "tags": sorted(tags)})
                if form == "άγω":
                    for cell in entry.get("forms", []):
                        if cell.get("form") in {"άχθηκα", "αγόμουν"}:
                            witnesses.append({"edition": edition, "headword": form, "cell": cell})
                if form == "άγομαι" and edition == "el":
                    witnesses.append({"edition": edition, "headword": form,
                                      "senses": entry.get("senses", [])})
    if not contexts:
        raise ValueError("The source no longer has the reviewed error; replay the published source audit instead")
    return {"source_sha256": hashes, "source_contexts": contexts, "source_witnesses": witnesses}


def repair(rows, evidence):
    corrected_rows = list(rows)
    changes, source_changes = [], []
    for context in evidence["source_contexts"]:
        tags, source_change = correct_sense_tense(
            context["form"], context["edition"], context["sense"], context["tags"])
        if source_change is None:
            raise ValueError("The archived source context no longer matches the reviewed rule")
        source_changes.append(source_change)
        for index, row in enumerate(corrected_rows):
            if (row.get("pos") == "verb" and row["form"] == context["form"]
                    and row["lemma"] in CANONICAL_LEMMAS
                    and set(row.get("tags", [])) == set(context["tags"])):
                corrected = {**row, "tags": tags}
                changes.append({"index": index, "original": row, "corrected": corrected,
                                "source_change": source_change})
                corrected_rows[index] = corrected
    return corrected_rows, {**evidence, "source_changes": source_changes, "pair_changes": changes}


def undo(rows, audit):
    for change in audit["pair_changes"]:
        if rows[change["index"]] != change["corrected"]:
            raise ValueError("Pair changed after source correction")
    result = list(rows)
    for change in reversed(audit["pair_changes"]):
        result[change["index"]] = change["original"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--kaikki", type=Path)
    source.add_argument("--source-audit", type=Path)
    parser.add_argument("--download-base", action="store_true")
    parser.add_argument("--undo", action="store_true")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("--input and --output must be separate files")
    if args.undo:
        audit = json.loads(args.audit.read_text())
        if audit.get("kind") != "snapshot-repair" or sha(args.input) != audit["output_sha256"]:
            raise ValueError("The corrected snapshot differs from its audit")
        result = undo(json.loads(args.input.read_text()), audit)
    else:
        if not args.kaikki and not args.source_audit:
            parser.error("--kaikki or --source-audit is required")
        pins = json.loads(subprocess.check_output(
            ["git", "show", "1.3.17:data/hf_manifest.json"], cwd=ROOT))["files"]
        if args.download_base and not args.input.exists():
            from huggingface_hub import hf_hub_download
            import shutil
            args.input.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(hf_hub_download("ciscoriordan/dilemma", "data/mg_pairs.json",
                                           revision=BASE_REVISION), args.input)
        if sha(args.input) != pins["data/mg_pairs.json"]["sha256"]:
            raise ValueError("The base differs from the pinned 1.3.17 pairs")
        evidence = (source_evidence(args.kaikki) if args.kaikki
                    else json.loads(args.source_audit.read_text()))
        result, audit = repair(json.loads(args.input.read_text()), evidence)
        if not audit["pair_changes"]:
            raise ValueError("No matching incorrect analysis was found")
        audit = {**audit, "schema_version": 1, "language": "el", "kind": "snapshot-repair",
                 "base_git_ref": "1.3.17", "base_hub_revision": BASE_REVISION,
                 "input_sha256": sha(args.input),
                 "generator_sha256": {name: sha(ROOT / name) for name in (
                     "build_data.py", "dilemma/el_source_tenses.py", "scripts/repair_el_source_tenses.py")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.undo:
        if sha(args.output) != audit["input_sha256"]:
            raise ValueError("Restored pairs differ from the original pinned bytes")
    else:
        audit["output_sha256"] = sha(args.output)
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"changes": len(audit["pair_changes"]), "undo": args.undo}))


if __name__ == "__main__":
    main()

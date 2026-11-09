#!/usr/bin/env python3
"""Apply the verb snapshot's source evidence to the tracked POS overrides."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_data import WIKT_TO_UPOS


def repair(base, repairs, imperfect):
    result = {form: dict(readings) for form, readings in base.items()}
    changes = []

    def assign(form, pos, lemma, evidence):
        existed = form in result
        old = result.get(form, {}).get(pos)
        if old == lemma:
            return
        result.setdefault(form, {})[pos] = lemma
        changes.append({"form": form, "pos": pos, "form_existed": existed,
                        "original": old, "corrected": lemma, "evidence": evidence})

    for c in repairs["head_changes"]:
        for proof in c["evidence"]:
            pos = WIKT_TO_UPOS[proof["pos"]]
            old = result.get(c["form"], {}).get(pos)
            if old is None or old == c["original"][0]:
                assign(c["form"], pos, c["form"], proof)
    for c in imperfect["changes"]:
        row = c["corrected"]
        if result.get(row["form"]) and "VERB" not in result[row["form"]]:
            assign(row["form"], "VERB", row["lemma"], c)
    return result, changes


def undo(result, changes):
    if any(result.get(c["form"], {}).get(c["pos"]) != c["corrected"] for c in changes):
        raise ValueError("POS override changed after repair")
    for c in reversed(changes):
        if c["original"] is None:
            del result[c["form"]][c["pos"]]
            if not c["form_existed"] and not result[c["form"]]:
                del result[c["form"]]
        else:
            result[c["form"]][c["pos"]] = c["original"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path)
    p.add_argument("--repairs", type=Path)
    p.add_argument("--imperfect", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--base-ref", default="1.3.16", help="Git tag identifying the baseline POS file")
    p.add_argument("--undo", action="store_true")
    args = p.parse_args()
    if args.undo:
        audit = json.loads(args.audit.read_text())
        if sha(args.output) != audit["output_sha256"]:
            raise ValueError("POS file changed after repair")
        result = json.loads(args.output.read_text())
        undo(result, audit["changes"])
    else:
        if args.base is None or args.repairs is None or args.imperfect is None:
            p.error("--base, --repairs and --imperfect are required")
        if args.base.resolve() == args.output.resolve():
            p.error("--base and --output must be separate")
        baseline = subprocess.check_output(
            ["git", "show", args.base_ref + ":data/mg_pos_lookup.json"], cwd=ROOT)
        if hashlib.sha256(baseline).hexdigest() != sha(args.base):
            raise ValueError("The POS baseline differs from its declared Git reference")
        result, changes = repair(json.loads(args.base.read_text()),
                                 json.loads(args.repairs.read_text()),
                                 json.loads(args.imperfect.read_text()))
        audit = {"schema_version": 1, "language": "el", "rule": "source_scoped_verb_pos_repairs_v1",
                 "base_git_ref": args.base_ref, "input_sha256": sha(args.base),
                 "base_git_commit": subprocess.check_output(
                     ["git", "rev-parse", args.base_ref + "^{commit}"], cwd=ROOT, text=True).strip(),
                 "source_audit_sha256": sha(args.repairs), "imperfect_audit_sha256": sha(args.imperfect),
                 "generator_sha256": sha(Path(__file__)), "changes": changes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    if not args.undo:
        audit["output_sha256"] = sha(args.output)
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"changes": len(audit["changes"]), "undo": args.undo}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""How often the verb-form generation reproduces attested polytonic spellings.

``mg_polytonic_paradigms`` writes polytonic spellings for Modern Greek verb
forms from their monotonic spelling, the verb's attested polytonic forms and
the Ancient Greek words of the grc dictionary. This script holds out, in
turn, every verb form the polytonic Modern Greek slice attests (its training
sentences, polytonic documents, at least ``MIN_TOKENS`` lowercase tokens with
a dominant spelling), generates it from the monotonic spelling with that form
left out of the verb's evidence, and counts how often the generated spelling
is the attested one, contextual grave and acute counted as one. The
traditional subjunctive spelling of the second and third person singular
(πάρῃς beside πάρεις) is scored the same way, where the slice attests it.

Precision is reported by verb class (A γράφω, B1 μιλάω, B2 μπορώ, passive
-μαι, other) and by paradigm cell (present, imperfect, past, dependent,
imperative, participle).

Usage:
    python eval/eval_mg_paradigms.py [--json OUT] [--examples N]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import export_mg_polytonic as mg  # noqa: E402
import mg_polytonic_paradigms as P  # noqa: E402
from export_hunspell import contextual_acute  # noqa: E402

MIN_TOKENS = 3
MIN_SHARE = 0.5


def attested_spellings(counts, sources) -> dict[str, Counter]:
    """Monotonic key -> well-formed lowercase spellings and their tokens."""
    out: dict[str, Counter] = defaultdict(Counter)
    for (form, position), per_doc in counts.forms.items():
        if position != "lower":
            continue
        n = sum(c for d, c in per_doc.items() if d in sources)
        if n and mg.mg_orthography_reason(form) is None:
            out[P.monotonic_key(form)][form] += n
    return dict(out)


def dominant(spellings: Counter | None) -> str | None:
    if not spellings:
        return None
    s, n = spellings.most_common(1)[0]
    total = sum(spellings.values())
    if n < MIN_TOKENS or n < MIN_SHARE * total:
        return None
    return s


def evaluate(paradigms, attested, grc_words, examples: int = 0,
             frequencies: dict[str, int] | None = None) -> dict:
    evidence = P.Evidence(attested, grc_words, paradigms)
    print('contract subscript shares', evidence.subscript_share)
    table: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    wrong: list[tuple] = []
    for lemma, forms in paradigms.items():
        vclass = P.verb_class(lemma, forms)
        for form, tags in forms.items():
            key = P.monotonic_key(form)
            gold = dominant(attested.get(key))
            gold_variant = None
            if P.is_subjunctive_cell(tags):
                # The variant's key: the form's letters with -ει- as -η-.
                probe = P.subjunctive_variant(
                    P.polytonic(form, tags, vclass, [], evidence, lemma)
                    or "")
                if probe:
                    gold_variant = (P.monotonic_key(probe),
                                    dominant(attested.get(
                                        P.monotonic_key(probe))))
            if gold is None and not (gold_variant and gold_variant[1]):
                continue
            exclude = {key} | ({gold_variant[0]} if gold_variant else set())
            siblings = evidence.lemma_spellings(forms, frozenset(exclude))
            generated = P.polytonic(form, tags, vclass, siblings, evidence,
                                    lemma)
            frequent = frequencies is not None and \
                frequencies.get(form, 0) >= P.GENERATED_MIN_TOKENS
            group = (vclass, P.cell(tags) + (" (frequent)" if frequent
                                            else ""))
            if gold is not None:
                ok = generated is not None and \
                    contextual_acute(generated) == contextual_acute(gold)
                table[group + ("indicative",)]["n"] += 1
                table[group + ("indicative",)]["ok"] += ok
                if not ok:
                    wrong.append((lemma, form, generated, gold))
            if gold_variant and gold_variant[1] and generated:
                variant = P.subjunctive_variant(generated)
                ok = variant is not None and contextual_acute(variant) == \
                    contextual_acute(gold_variant[1])
                table[group + ("subjunctive",)]["n"] += 1
                table[group + ("subjunctive",)]["ok"] += ok
                if not ok:
                    wrong.append((lemma, form + " (subj.)", variant,
                                  gold_variant[1]))
    return {"table": table, "wrong": wrong[:examples] if examples else wrong}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", type=Path)
    ap.add_argument("--examples", type=int, default=40)
    args = ap.parse_args()

    counts = mg.count_corpus()
    attested = attested_spellings(counts, mg.source_documents(counts))
    paradigms = P.load_verb_paradigms()
    grc_words = mg.read_grc_words()
    result = evaluate(paradigms, attested, grc_words,
                      frequencies=P.load_form_frequencies())
    table = result["table"]

    def line(label, rows):
        n = sum(table[r]["n"] for r in rows)
        ok = sum(table[r]["ok"] for r in rows)
        if n:
            print(f"{label:34s} {ok:6,} / {n:6,}  {ok / n:6.1%}")

    keys = sorted(table)
    for kind in ("indicative", "subjunctive"):
        print(f"\n{kind} spellings, by verb class")
        for c in sorted({k[0] for k in keys}):
            line(c, [k for k in keys if k[0] == c and k[2] == kind])
        print(f"{kind} spellings, by cell")
        for c in sorted({k[1].replace(" (frequent)", "") for k in keys}):
            line(c, [k for k in keys if k[1].replace(" (frequent)", "") == c
                     and k[2] == kind])
        line(f"all {kind}", [k for k in keys if k[2] == kind])
        line(f"  of them forms generated (>= {P.GENERATED_MIN_TOKENS} "
             "monotonic tokens)",
             [k for k in keys if k[2] == kind and "(frequent)" in k[1]])
    print("\nsome misses (verb, form, generated, attested):")
    for row in result["wrong"][:args.examples]:
        print("  ", *row)
    if args.json:
        args.json.write_text(json.dumps(
            {"|".join(k): dict(v) for k, v in table.items()},
            ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

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

The forms the list generates are the ones the slice does not attest, so a
second number comes from a hand-checked sample of them
(``eval/mg_generated_sample.tsv``, stratified by class and cell): the share
of its spellings judged right, among those the built list still generates.

Usage:
    python eval/eval_mg_paradigms.py [--json OUT] [--examples N]
        [--sample TSV] [--list DIC]
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
SAMPLE = ROOT / "eval" / "mg_generated_sample.tsv"
BUILT_LIST = ROOT / "build" / "hunspell" / "grc_mg_polytonic.dic"


def sample_precision(sample: Path = SAMPLE, dic: Path = BUILT_LIST) -> dict:
    """The hand-checked sample's verdicts on the spellings ``dic`` still
    generates; a sampled monotonic form now generated with another spelling
    is counted apart, unchecked."""
    generated: dict[str, set[str]] = defaultdict(set)
    for line in dic.read_text("utf-8").splitlines()[1:]:
        form, _, fields = line.partition("\t")
        if "mg:generated" in fields:
            generated[P.monotonic_key(form)].add(form)
    out = Counter()
    for line in sample.read_text("utf-8").splitlines():
        if not line or line.startswith(("#", "spelling\t")):
            continue
        spelling, mono, *_rest = line.split("\t")
        verdict = _rest[4]
        now = generated.get(P.monotonic_key(spelling))
        if now and spelling in now:
            out[verdict] += 1
        elif now:
            out["respelled"] += 1
        else:
            out["no longer generated"] += 1
    return dict(out)


def attested_spellings(counts, sources, position: str = "lower"
                       ) -> dict[str, Counter]:
    """Monotonic key -> well-formed spellings and their tokens, of the
    words in lowercase (or, with ``position`` ``cap``, of the capitalized
    words inside a sentence)."""
    out: dict[str, Counter] = defaultdict(Counter)
    for (form, where), per_doc in counts.forms.items():
        if where != position:
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


def same(generated: str, gold: str) -> bool:
    """Whether a generated spelling is the attested one, the grave of a
    word in running text counted as its acute and a capital (a name, or a
    word at the start of a sentence) as its small letter."""
    return (contextual_acute(generated.lower())
            == contextual_acute(gold.lower()))


def evaluate(paradigms, attested, grc_words, examples: int = 0,
             frequencies: dict[str, int] | None = None,
             capitalized: dict[str, Counter] | None = None) -> dict:
    """Generated against attested spellings, by verb class or part of
    speech and cell; a form the generator declines (an unaccented word of
    several syllables, a clitic of one) is counted apart, as undecided. A
    capitalized form (a name, a month) is compared with the capitalized
    spellings inside a sentence, ``capitalized``, and left out without
    them: a lowercase spelling of its letters is another word's (the
    surname Γράψη, the subjunctive γράψῃ)."""
    evidence = P.Evidence(attested, grc_words, paradigms)
    table: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    wrong: list[tuple] = []
    for lemma, forms in paradigms.items():
        vclass = P.verb_class(lemma, forms)
        for form, tags in forms.items():
            key = P.monotonic_key(form)
            if form[:1].isupper():
                gold = dominant((capitalized or {}).get(key))
            else:
                gold = dominant(attested.get(key))
            gold_variant = None
            if P.is_subjunctive_cell(tags, form):
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
                P.form_tokens(form, frequencies) >= P.GENERATED_MIN_TOKENS
            group = (vclass, P.report_cell(tags) + (" (frequent)" if frequent
                                            else ""))
            if generated is None:
                # Not generated at all (an unaccented word of several
                # syllables, a clitic of one): no spelling to judge.
                table[group + ("indicative",)]["undecided"] += 1
                continue
            if gold is not None:
                ok = same(generated, gold)
                table[group + ("indicative",)]["n"] += 1
                table[group + ("indicative",)]["ok"] += ok
                if not ok:
                    wrong.append((lemma, form, generated, gold))
            if gold_variant and gold_variant[1] and generated:
                variant = P.subjunctive_variant(generated)
                ok = variant is not None and same(variant, gold_variant[1])
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
    ap.add_argument("--sample", type=Path, default=SAMPLE)
    ap.add_argument("--list", type=Path, default=BUILT_LIST)
    args = ap.parse_args()

    counts = mg.count_corpus()
    sources = mg.source_documents(counts)
    attested = attested_spellings(counts, sources)
    paradigms = P.load_paradigms()
    grc_words = mg.read_grc_words()
    result = evaluate(paradigms, attested, grc_words,
                      frequencies=P.load_form_frequencies(),
                      capitalized=attested_spellings(counts, sources, "cap"))
    table = result["table"]

    def line(label, rows):
        n = sum(table[r]["n"] for r in rows)
        ok = sum(table[r]["ok"] for r in rows)
        if n:
            print(f"{label:34s} {ok:6,} / {n:6,}  {ok / n:6.1%}")

    keys = sorted(table)
    for kind in ("indicative", "subjunctive"):
        print(f"\n{kind} spellings, by verb class or part of speech")
        for c in sorted({k[0] for k in keys}):
            line(c, [k for k in keys if k[0] == c and k[2] == kind])
        line("all verbs", [k for k in keys if k[2] == kind
                           and k[0] in P.VERB_CLASSES])
        line("all other parts of speech", [k for k in keys if k[2] == kind
                                           and k[0] not in P.VERB_CLASSES])
        print(f"{kind} spellings, by cell")
        for c in sorted({k[1].replace(" (frequent)", "") for k in keys}):
            line(c, [k for k in keys if k[1].replace(" (frequent)", "") == c
                     and k[2] == kind])
        line(f"all {kind}", [k for k in keys if k[2] == kind])
        line(f"  of them forms generated (>= {P.GENERATED_MIN_TOKENS} "
             "monotonic tokens)",
             [k for k in keys if k[2] == kind and "(frequent)" in k[1]])
        undecided = sum(table[k]["undecided"] for k in keys if k[2] == kind)
        if undecided:
            print(f"  not generated (no spelling to judge): {undecided:,}")
    if args.sample.exists() and args.list.exists():
        found = sample_precision(args.sample, args.list)
        checked = found.get("ok", 0) + found.get("wrong", 0)
        if checked:
            print(f"\nhand-checked sample of generated spellings: "
                  f"{found.get('ok', 0):,} / {checked:,} right "
                  f"{found.get('ok', 0) / checked:6.1%}; {found}")
    print("\nsome misses (verb, form, generated, attested):")
    for row in result["wrong"][:args.examples]:
        print("  ", *row)
    if args.json:
        args.json.write_text(json.dumps(
            {"|".join(k): dict(v) for k, v in table.items()},
            ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

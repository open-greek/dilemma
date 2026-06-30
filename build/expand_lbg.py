#!/usr/bin/env python3
"""Generate inflected forms for the Byzantine (LBG) headwords.

build_lookup_db.py ingests these as lowest-priority gap-fill behind two gates
(variant-collision + a corpus_freq ceiling): a raw exact-key gap-fill silently
shadows real tokens, because the runtime cascade also resolves the grave/lower/
monotonic/accent-stripped variants of a form, so a paradigm guess whose variant
already resolves to a different, more common lemma must be dropped. The gates
keep only the safe long tail (the ungated version was net-negative on the DBBE
Byzantine gold).


Reuses build/expand_lsj.py's Wiktionary grc-decl / grc-conj Lua machinery to
decline the Byzantine nouns (headword + gender) and conjugate the regular
Byzantine verbs (headword), so inflected Byzantine forms lemmatize back to
their headword. Only headword + gender are used; no copyrighted lexicon text.

Conservative by design:
  - Nouns: declined from headword + gender (genitive inferred for the regular
    1st/2nd-declension cases; entries where the Lua decliner errors or yields
    nothing are skipped, not guessed).
  - Verbs: the regular derived/contract classes conjugate from the present
    headword; entries that error or yield nothing are skipped.
  - Adjectives / adverbs / indeclinables (no captured gender, not a -ω verb)
    are skipped.

Input:  data/lbg_headwords.json   (list of {lemma, gender})
Output: data/lbg_pairs.json       ({inflected_form: lemma})

Run:    PYTHONPATH=.:build python build/expand_lbg.py
"""
import json
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "build"), str(ROOT)]

import lupa.lua53  # noqa: F401  force Lua 5.3 before expand_lsj imports the sandbox
import expand_lsj as E  # applies the 8 idempotent Lua-sandbox fixes on import

LBG_HEADWORDS = ROOT / "data" / "lbg_headwords.json"
LBG_PAIRS = ROOT / "data" / "lbg_pairs.json"

ARTICLE_TO_CODE = {"ὁ": "m", "ἡ": "f", "τό": "n"}
# Only ACTIVE verbs (-ω / -ῶ, incl. contracts -έω/-όω/-άω). expand_verb's
# stem classifier mishandles -ομαι/-οῦμαι deponents (it keeps a phantom
# `stem+ο` and conjugates a fake active), so deponents are skipped here. The
# active paradigm already yields the correct middle/passive forms.
VERB_ENDINGS = ("ω", "ῶ")
# 3rd-declension -ων nouns need the real genitive (ἐλαιών/-ῶνος vs
# δαίμων/-ονος, βουβῶν/-ῶνος); inferring it picks the wrong class, so skip
# them. Match on the accent-folded ending so -ων / -ών / -ῶν all skip.
def _strip_acc(s):
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if not unicodedata.combining(c))


def _forms_err(ret):
    """expand_noun returns (forms, err); expand_verb may return forms or
    (forms, err). Normalize to (set, err)."""
    if isinstance(ret, tuple):
        return (ret[0] or set()), (ret[1] if len(ret) > 1 else "")
    return (ret or set()), ""


def _expand_optional(form):
    """Expand a single ``(...)`` optional segment (the grc optional-nu, e.g.
    ``λύουσι(ν)`` -> {λύουσι, λύουσιν}; also handles the unclosed ``...σι(ν``
    artifact). Returns the clean variants; drops anything with leftover parens."""
    if "(" not in form:
        return {form} if ")" not in form else set()
    i = form.index("(")
    prefix, rest = form[:i], form[i + 1:]
    if ")" in rest:
        j = rest.index(")")
        inner, suffix = rest[:j], rest[j + 1:]
    else:
        inner, suffix = rest, ""
    out = {prefix + suffix, prefix + inner + suffix}
    return {f for f in out if "(" not in f and ")" not in f}


def main():
    data = json.load(open(LBG_HEADWORDS, encoding="utf-8"))
    print(f"Loaded {len(data):,} LBG headwords")
    wtp = E.get_wtp()
    print("wtp ready; expanding...", flush=True)

    pairs: dict[str, str] = {}
    n_noun = n_verb = n_skip = 0
    t0 = time.time()
    for i, e in enumerate(data):
        lemma = e.get("lemma")
        gender = e.get("gender")
        if not lemma:
            n_skip += 1
            continue
        try:
            if gender in ARTICLE_TO_CODE and not _strip_acc(lemma).endswith("ων"):
                forms, err = _forms_err(
                    E.expand_noun(wtp, lemma, ARTICLE_TO_CODE[gender]))
                kind = "noun"
            elif not gender and lemma.endswith(VERB_ENDINGS):
                forms, err = _forms_err(E.expand_verb(wtp, lemma))
                kind = "verb"
            else:
                n_skip += 1
                continue
        except Exception:
            n_skip += 1
            continue
        if err or not forms:
            n_skip += 1
            continue
        n_noun += kind == "noun"
        n_verb += kind == "verb"
        for raw in forms:
            for f in _expand_optional(unicodedata.normalize("NFC", raw).strip()):
                # inflected forms only (the headword self-map is added
                # separately); first writer wins on collisions (kept as
                # lowest-priority gap-fill).
                if f and f != lemma and f not in pairs:
                    pairs[f] = lemma
        if i % 2000 == 0:
            print(f"  {i:,}/{len(data):,}  nouns={n_noun:,} verbs={n_verb:,} "
                  f"skip={n_skip:,} pairs={len(pairs):,} ({time.time()-t0:.0f}s)",
                  flush=True)

    json.dump(pairs, open(LBG_PAIRS, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\nDONE: declined {n_noun:,} nouns + conjugated {n_verb:,} verbs, "
          f"skipped {n_skip:,}; wrote {len(pairs):,} inflected pairs to "
          f"{LBG_PAIRS} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

"""Present-active cells missing from independently defined Modern Greek verbs.

The source snapshots often define a verb but explicitly have no conjugation
table. These are reviewed inflection rules, not inferred corpus lemmas. Past,
passive and imperative forms are outside this completion pass.
"""
from __future__ import annotations

import unicodedata

from .el_paradigm_gaps import stripped

RULE = "el_present_active_v1"
PERSONS = (("first-person", "singular"), ("second-person", "singular"),
           ("third-person", "singular"), ("first-person", "plural"),
           ("second-person", "plural"), ("third-person", "plural"))
EVIDENCE = {
    "regular": "https://el.wiktionary.org/wiki/γράφω",
    "contracted_ao": "https://el.wiktionary.org/wiki/αγαπάω",
    "ακούω": "https://el.wiktionary.org/wiki/ακούω",
    "καίω": "https://el.wiktionary.org/wiki/καίω",
    "κλαίω": "https://el.wiktionary.org/wiki/κλαίω",
    "φταίω": "https://el.wiktionary.org/wiki/φταίω",
    "λέω": "https://el.wiktionary.org/wiki/λέω",
    "τρώω": "https://el.wiktionary.org/wiki/τρώω",
    "δρω": "https://el.wiktionary.org/wiki/δρω",
    "ζω": "https://el.wiktionary.org/wiki/ζω",
}
LEO_PREFIXES = frozenset(("", "αντι", "απο", "καλο", "κακο", "ξε", "ξανα",
                          "παρα", "πρωτο", "προ", "πολυ"))


def _unaccent_monosyllable(word: str) -> str:
    letters, accents, dialytika = [], [], []
    for letter in word:
        parts = unicodedata.normalize("NFD", letter)
        letters.append(parts[0])
        accents.append("\u0301" in parts)
        dialytika.append("\u0308" in parts)
    nuclei, i = 0, 0
    while i < len(letters):
        if letters[i] in "αεηιουω":
            nuclei += 1
            if (i + 1 < len(letters) and not accents[i] and not dialytika[i + 1]
                    and letters[i] + letters[i + 1] in {"αι", "ει", "οι", "ου", "υι", "αυ", "ευ", "ηυ"}):
                i += 1
        i += 1
    if nuclei == 1:
        return unicodedata.normalize("NFC", "".join(
            c for c in unicodedata.normalize("NFD", word) if c != "\u0301"))
    return word


def present_forms(lemma: str) -> tuple[str, tuple[str, ...]] | None:
    """Six standard indicative cells, or None outside the reviewed classes."""
    if (len(lemma) < 2 or lemma != unicodedata.normalize("NFC", lemma)
            or lemma != lemma.lower() or not lemma.endswith("ω")
            or not all("\u0370" <= c <= "\u03ff" and c.isalpha() for c in lemma)):
        return None
    if lemma == "δρω":
        return "δρω", ("δρω", "δρας", "δρα", "δρούμε", "δράτε", "δρουν")
    if lemma == "ζω":
        return "ζω", ("ζω", "ζεις", "ζει", "ζούμε", "ζείτε", "ζουν")
    if "\u0301" not in unicodedata.normalize("NFD", lemma):
        return None
    families = {
        "ακούω": ("ακούς", "ακούει", "ακούμε", "ακούτε", "ακούν"),
        "καίω": ("καις", "καίει", "καίμε", "καίτε", "καίνε"),
        "κλαίω": ("κλαις", "κλαίει", "κλαίμε", "κλαίτε", "κλαίνε"),
        "φταίω": ("φταις", "φταίει", "φταίμε", "φταίτε", "φταίνε"),
        "τρώω": ("τρως", "τρώει", "τρώμε", "τρώτε", "τρώνε"),
    }
    for root, cells in families.items():
        if lemma.endswith(root):
            prefix = lemma[:-len(root)]
            cells = list(cells)
            if prefix and root != "ακούω":
                cells[0] = {"καίω": "καίς", "κλαίω": "κλαίς", "φταίω": "φταίς",
                            "τρώω": "τρώς"}[root]
            return root, (lemma, *(prefix + f for f in cells))
    if lemma.endswith("λέω") and lemma[:-3] in LEO_PREFIXES:
        prefix = lemma[:-3]
        return "λέω", (lemma, prefix + ("λές" if prefix else "λες"),
                        *(prefix + f for f in ("λέει", "λέμε", "λέτε", "λένε")))
    if lemma.endswith("άω"):
        stem = lemma[:-2]
        return "contracted_ao", tuple(_unaccent_monosyllable(stem + ending) for ending in
                                       ("άω", "άς", "άει", "άμε", "άτε", "άνε"))
    if lemma.endswith(("όω", "ώω")):
        return None
    return "regular", tuple(lemma[:-1] + ending for ending in
                             ("ω", "εις", "ει", "ουμε", "ετε", "ουν"))


def complete_present_pairs(pairs: list[dict], lookup: dict, heads: dict) -> dict:
    """Append missing pairs beneath independent self-mapped heads only.

    Existing pairs and lookup readings are never replaced. Every insertion is
    returned with its prior value so the completion can be unapplied exactly.
    """
    known = {(r["form"], r["lemma"], r.get("pos")) for r in pairs}
    source_cells = set()
    for row in pairs:
        tags = set(row.get("tags", ()))
        if (row.get("pos") == "verb" and {"present", "indicative"} <= tags
                and not {"past", "aorist", "imperfect", "passive", "middle", "participle"} & tags):
            for person in PERSONS:
                if set(person) <= tags:
                    source_cells.add((row["lemma"], person))
    changes, unsupported, lookup_additions = [], [], []
    used_heads = {}
    for lemma in sorted(heads):
        if lookup.get(lemma, (None,))[0] != lemma:
            continue
        # A weather use does not erase a separate personal meaning (βρέχω).
        # Infer personal cells only when at least one independent reading
        # is not marked impersonal or defective.
        if heads[lemma] and not any(
                not {"impersonal", "defective"} & set(record.get("tags", ()))
                and (not record.get("senses") or any(
                    not {"impersonal", "defective"} & set(sense.get("tags", ()))
                    for sense in record["senses"])) for record in heads[lemma]):
            unsupported.append(lemma)
            continue
        result = present_forms(lemma)
        if result is None:
            if lemma.endswith("ω"):
                unsupported.append(lemma)
            continue
        pattern, forms = result
        for form, person in zip(forms, PERSONS):
            if (form, lemma, "verb") in known or (lemma, person) in source_cells:
                continue
            previous = lookup.get(form)
            if previous is not None and previous[0] != lemma:
                continue
            row = {"form": form, "lemma": lemma, "pos": "verb",
                   "tags": ["present", "active", "indicative", *person],
                   "generated_by": RULE}
            pairs.append(row)
            known.add((form, lemma, "verb"))
            for key in dict.fromkeys((form, stripped(form))):
                if key not in lookup:
                    lookup[key] = (lemma, 1)
                    lookup_additions.append({"form": key, "lemma": lemma,
                                             "original": None, "corrected": [lemma, 1]})
            changes.append({"original": None, "corrected": row, "pattern": pattern,
                            "rule_evidence": EVIDENCE[pattern],
                            "lookup_before": list(previous) if previous is not None else None,
                            "lookup_after": list(lookup[form])})
            used_heads[lemma] = heads[lemma]
    return {"schema_version": 1, "language": "el", "rule": RULE,
            "head_evidence": used_heads, "changes": changes,
            "lookup_additions": lookup_additions,
            "unsupported_heads": unsupported}


def unapply_present_pairs(pairs: list[dict], lookup: dict, audit: dict) -> None:
    """Undo this completion, refusing to erase unrelated subsequent edits."""
    generated = {(r["form"], r["lemma"], r.get("pos")): r for r in pairs
                 if r.get("generated_by") == RULE}
    remove = set()
    for change in audit["changes"]:
        row = change["corrected"]
        form = row["form"]
        key = (form, row["lemma"], row["pos"])
        if generated.get(key) != row or list(lookup.get(form, ())) != change["lookup_after"]:
            raise ValueError(f"Present completion changed after generation: {form}")
        remove.add(key)
    for change in audit["lookup_additions"]:
        if list(lookup.get(change["form"], ())) != change["corrected"]:
            raise ValueError(f"Present completion changed after generation: {change['form']}")
    pairs[:] = [r for r in pairs if not (
        r.get("generated_by") == RULE and (r["form"], r["lemma"], r.get("pos")) in remove)]
    for change in audit["lookup_additions"]:
        del lookup[change["form"]]

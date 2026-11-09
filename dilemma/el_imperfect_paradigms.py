"""Complete active imperfect cells from source-recorded past stems.

Never derive a past stem or augment from the present. A singular anchor
licenses only singular cells and the third plural; a plural anchor licenses
only the second plural. Unsupported verbs remain unchanged.
"""
from __future__ import annotations

from .el_paradigm_gaps import stripped
from .el_present_paradigms import PERSONS

RULE = "el_imperfect_from_source_stems_v1"
GRAMMAR = "https://www.ebooks.edu.gr/ebooks/v/html/8547/2334/Grammatiki-Neas-Ellinikis-Glossas_A-B-G-Gymnasiou_html-apli/index_C_06.html"
# The two singular stems are explicitly listed together by Foundalis.
# The school table confirms the unaugmented plural independently. These
# are linguistic inflection facts, not text added to the training corpus.
REVIEWED = {
    "συλλαμβάνω": {
        "singular": ("συλλάμβανα", "συνελάμβανα"),
        "plural": ("συλλαμβάναμε",),
        "evidence": (
            "https://www.foundalis.com/lan/grkvbirr.htm",
            "https://ekpaid-evi.gr/wp-content/uploads/2024/11/ΙΔΙΑΙΤΕΡΟΤΗΤΕΣ-ΣΤΑ-ΡΗΜΑΤΑ.pdf",
            "https://blogs.sch.gr/g2271/files/2023/10/ΓΡΑΜΜΑΤΙΚΗ-16ης-ΕΝΟΤΗΤΑΣ-ΓΛΩΣΣΑΣ-ΣΤ΄.pdf",
        ),
    },
}
OTHER_TENSES = {"present", "future", "aorist", "perfect", "pluperfect", "future-perfect"}
EXCLUDED = {"passive", "middle", "mediopassive", "participle", "imperative", "subjunctive", "optative"}
BLOCKED = OTHER_TENSES | EXCLUDED
PERSON_TAGS = {"first-person", "second-person", "third-person", "singular", "plural", "dual"}


def _personal(records):
    return not records or any(
        not {"impersonal", "defective"} & set(r.get("tags", ()))
        and (not r.get("senses") or any(
            not {"impersonal", "defective"} & set(s.get("tags", ()))
            for s in r["senses"])) for r in records)


def _active_imperfect(row):
    if row.get("pos") != "verb":
        return False
    tags = set(row.get("tags", ()))
    return "imperfect" in tags and not tags & BLOCKED


def complete_imperfect_pairs(pairs, lookup, heads):
    """Append missing, source-anchored cells and return reversible evidence."""
    readings = {(r["form"], r["lemma"], p) for r in pairs if _active_imperfect(r)
                for p in PERSONS if set(p) <= set(r.get("tags", ()))}
    source_cells = {(lemma, p) for _, lemma, p in readings}
    anchors = {}
    for row in pairs:
        if (not _active_imperfect(row) or row.get("generated_by") == RULE
                or {"form-of", "alt-of"} & set(row.get("tags", ()))):
            continue
        tags = set(row.get("tags", ()))
        if tags & PERSON_TAGS == {"first-person", "singular"} and row["form"].endswith("α"):
            anchors.setdefault(row["lemma"], []).append((row, "singular", {"source_pair": row}))
        if tags & PERSON_TAGS == {"first-person", "plural"} and row["form"].endswith("αμε"):
            anchors.setdefault(row["lemma"], []).append((row, "plural", {"source_pair": row}))
    for lemma, record in REVIEWED.items():
        for number in ("singular", "plural"):
            for form in record[number]:
                row = {"form": form, "lemma": lemma, "pos": "verb",
                       "tags": ["imperfect", "active", "indicative", "first-person", number]}
                anchors.setdefault(lemma, []).append((row, number, {"stem_evidence": list(record["evidence"])}))
    changes, additions, used_heads = [], [], {}
    for lemma in sorted(anchors):
        if (lemma not in heads or lookup.get(lemma, (None,))[0] != lemma
                or not lemma.endswith("ω") or not _personal(heads[lemma])):
            continue
        for anchor, number, evidence in anchors[lemma]:
            form = anchor["form"]
            # Existing source cells keep their spelling preferences. A
            # reviewed explicit stem may supply its independent variant.
            reviewed = "stem_evidence" in evidence
            cells = ([(form, PERSONS[0]), (form[:-1] + "ες", PERSONS[1]),
                      (form[:-1] + "ε", PERSONS[2]), (form + "ν", PERSONS[5])]
                     if number == "singular" else
                     [(form, PERSONS[3]), (form[:-2] + "τε", PERSONS[4])])
            for spelling, person in cells:
                if not reviewed and (lemma, person) in source_cells:
                    continue
                tags = sorted((set(anchor["tags"]) - PERSON_TAGS)
                              | {"imperfect", "active", "indicative", *person})
                # Do not duplicate a cell merely because a source omitted
                # voice or mood tags already established by this rule.
                if (spelling, lemma, person) in readings:
                    continue
                row = {"form": spelling, "lemma": lemma, "pos": "verb", "tags": tags,
                       "generated_by": RULE}
                pairs.append(row); readings.add((spelling, lemma, person))
                for lookup_key in dict.fromkeys((spelling, stripped(spelling))):
                    if lookup_key not in lookup:
                        target = lookup.get(spelling, (lemma, 1))
                        lookup[lookup_key] = target
                        additions.append({"form": lookup_key, "original": None,
                                          "corrected": list(target)})
                changes.append({"original": None, "corrected": row,
                                "rule_evidence": GRAMMAR, "anchor": anchor, **evidence})
                used_heads[lemma] = heads[lemma]
    return {"schema_version": 1, "language": "el", "rule": RULE,
            "head_evidence": used_heads, "changes": changes, "lookup_additions": additions}


def unapply_imperfect_pairs(pairs, lookup, audit):
    expected = [c["corrected"] for c in audit["changes"]]
    generated = {}
    for i, row in enumerate(pairs):
        if row.get("generated_by") == RULE:
            key = (row["form"], row["lemma"], tuple(sorted(row.get("tags", ()))))
            generated.setdefault(key, []).append((i, row))
    remove = set()
    for row in expected:
        key = (row["form"], row["lemma"], tuple(sorted(row.get("tags", ()))))
        matches = [(i, r) for i, r in generated.get(key, ()) if r == row]
        if len(matches) != 1 or matches[0][0] in remove:
            raise ValueError("Imperfect completion changed after generation")
        remove.add(matches[0][0])
    if any(list(lookup.get(c["form"], ())) != c["corrected"]
           for c in audit["lookup_additions"]):
        raise ValueError("Imperfect completion changed after generation")
    pairs[:] = [r for i, r in enumerate(pairs) if i not in remove]
    for c in audit["lookup_additions"]:
        del lookup[c["form"]]

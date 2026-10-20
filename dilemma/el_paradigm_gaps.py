"""Conservative completion of four source-confirmed Modern Greek paradigms.

The Wiktionary snapshots define these heads but omit the required tables.
These are reviewed inflection rules, never rules learned from corpus unknowns.
Only the present active and the masculine noun αιτών are supplied. No past,
passive, adjective, or Ancient Greek paradigm is inferred.
"""
from __future__ import annotations

import unicodedata

LEXICON = "https://www.greek-language.gr/greekLang/modern_greek/tools/lexica/"
EVIDENCE = {
    "εμπίπτω": [LEXICON + "triantafyllides/search.html?lq=εμπίπτω"],
    "απορρέω": [LEXICON + "triantafyllides/search.html?lq=απορρέω"],
    "συνιστώ": [LEXICON + "triantafyllides/search.html?lq=συνιστώ"],
    "αιτών": [LEXICON + "search.html?lq=αιτών"],
}


def stripped(word):
    return unicodedata.normalize("NFC", "".join(
        c for c in unicodedata.normalize("NFD", word.lower())
        if unicodedata.category(c) != "Mn"))


def paradigm_cells():
    cells = []
    persons = [("first-person", "singular"), ("second-person", "singular"),
               ("third-person", "singular"), ("first-person", "plural"),
               ("second-person", "plural"), ("third-person", "plural")]
    for head in ("εμπίπτω", "απορρέω", "συνιστώ"):
        if head == "συνιστώ":
            forms = ["συνιστώ", "συνιστάς", "συνιστά", "συνιστούμε", "συνιστάτε", "συνιστούν"]
        else:
            forms = [head[:-1] + ending for ending in ("ω", "εις", "ει", "ουμε", "ετε", "ουν")]
        for form, person in zip(forms, persons):
            cells.append({"form": form, "lemma": head, "pos": "VERB",
                          "tags": ["present", "active", "indicative", *person],
                          "evidence": EVIDENCE[head]})
    for form, case, number in [("αιτών", "nominative", "singular"),
                               ("αιτούντος", "genitive", "singular"),
                               ("αιτούντα", "accusative", "singular"),
                               ("αιτούντες", "nominative", "plural"),
                               ("αιτούντες", "accusative", "plural"),
                               ("αιτούντων", "genitive", "plural")]:
        cells.append({"form": form, "lemma": "αιτών", "pos": "NOUN",
                      "tags": ["masculine", case, number], "evidence": EVIDENCE["αιτών"]})
    return cells


def complete_lookup(lookup):
    """Fill absent keys only, and only beneath an existing self-mapped head."""
    added = []
    for cell in paradigm_cells():
        if lookup.get(cell["lemma"]) != cell["lemma"]:
            continue
        for form in dict.fromkeys((cell["form"], stripped(cell["form"]))):
            if form not in lookup:
                lookup[form] = cell["lemma"]
                added.append({**cell, "form": form})
    return added

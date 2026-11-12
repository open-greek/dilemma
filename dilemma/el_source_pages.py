"""Reviewed Greek Wiktionary pages that the Modern Greek extract omits.

The Modern Greek build reads the Greek-edition extract's ``Greek`` language
file. A page whose noun section the extract cannot read, or whose section a
later edit filed under another language code, has no record there, so its
headword gets no lookup key and its paradigm no pairs, while inflection pages
that name it keep pointing at a lemma that is absent.

Each record below restates one such page from the cited revision, in the
shape the extract gives its neighbours: the definition, the gender on the
headword line, and the cells of the declension table that revision renders.
``build_data.extract_pairs`` reads a record only while the extract has no
lexical entry for that word and part of speech, and then passes it through
the same head, table and reference checks as any other entry. A record whose
page the extract has regained is reported as retired and changes nothing.

Nothing here is inferred from an inflection page's reference alone: a target
that no page defines does not become a headword.
"""
from __future__ import annotations

RULE = "el_omitted_source_pages_v1"
REVIEWED_AT = "2026-11-11T18:20:00Z"
SOURCE = "el"
LICENSE = "CC BY-SA 4.0"
EXTRACT = {
    "name": "kaikki.org Greek-edition extract, language file Greek",
    "extracted": "2026-11-03",
    "dump": "2026-11-01",
}


def _revision(title: str, oldid: int) -> str:
    return f"https://el.wiktionary.org/w/index.php?title={title}&oldid={oldid}"


def _declension(template: str, gender: str, cells: dict) -> list:
    """The extract's rows for a rendered noun table and its headword line."""
    order = [("nominative", "singular"), ("nominative", "plural"),
             ("genitive", "singular"), ("genitive", "plural"),
             ("accusative", "singular"), ("accusative", "plural"),
             ("vocative", "singular"), ("vocative", "plural")]
    rows = [{"form": template, "tags": ["inflection-template"],
             "source": "declension"}]
    rows += [{"form": cells[cell], "tags": list(cell), "source": "declension"}
             for cell in order]
    rows.append({"form": cells["nominative", "singular"], "tags": [gender],
                 "source": "header"})
    return rows


PAGES = (
    {
        "entry": {
            "word": "χιονόνερο", "pos": "noun", "lang_code": "el",
            "senses": [{
                "glosses": ["ένα από τα είδη υετού που αποτελείται από μίξη "
                            "χιονιού και βροχής"],
                "raw_tags": ["μετεωρολογία"],
            }],
            "forms": _declension("el-κλίση-'σίδερο'", "neuter", {
                ("nominative", "singular"): "χιονόνερο",
                ("nominative", "plural"): "χιονόνερα",
                ("genitive", "singular"): "χιονόνερου",
                ("genitive", "plural"): "χιονόνερων",
                ("accusative", "singular"): "χιονόνερο",
                ("accusative", "plural"): "χιονόνερα",
                ("vocative", "singular"): "χιονόνερο",
                ("vocative", "plural"): "χιονόνερα",
            }),
        },
        "revision": {"oldid": 7663351, "timestamp": "2026-10-15T12:14:36Z"},
        "evidence": [_revision("χιονόνερο", 7663351)],
        "reason": ("The noun heading of this revision begins with a space, "
                   "so the page renders it as preformatted text and has no "
                   "noun section. The extract has the inflection pages "
                   "χιονόνερα, χιονόνερου and χιονόνερων, each a form of "
                   "χιονόνερο, and no entry for the headword."),
    },
    {
        "entry": {
            "word": "μπαλωθιά", "pos": "noun", "lang_code": "el-crt",
            "senses": [{"glosses": ["πυροβολισμός, τουφεκιά"]}],
            "forms": _declension("el-κλίση-'καρδιά'", "feminine", {
                ("nominative", "singular"): "μπαλωθιά",
                ("nominative", "plural"): "μπαλωθιές",
                ("genitive", "singular"): "μπαλωθιάς",
                ("genitive", "plural"): "μπαλωθιών",
                ("accusative", "singular"): "μπαλωθιά",
                ("accusative", "plural"): "μπαλωθιές",
                ("vocative", "singular"): "μπαλωθιά",
                ("vocative", "plural"): "μπαλωθιές",
            }),
        },
        "revision": {"oldid": 7363298, "timestamp": "2026-07-19T19:04:16Z"},
        "evidence": [_revision("μπαλωθιά", 7363298)],
        "reason": ("This revision heads the page as Cretan (el-crt), so the "
                   "extract files it outside the Greek language file, as a "
                   "headword without its declension table. The page defines "
                   "the noun, cites μπαλωθιά and μπαλωθιές from Myrivilis "
                   "(1949) and Moysidis (2006), and is independent of "
                   "μπαλοθιά: neither page names the other."),
    },
)


def audit_row(page: dict, status: str) -> dict:
    """One page's audit record: what was read, from where, and why."""
    entry = page["entry"]
    row = {"word": entry["word"], "pos": entry["pos"], "status": status,
           "source": SOURCE, "license": LICENSE,
           "revision": dict(page["revision"]),
           "evidence": list(page["evidence"]), "reason": page["reason"]}
    if status == "applied":
        row["entry"] = entry
    return row

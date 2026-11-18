"""Reviewed rows of the Ancient view (``lang="grc"``) only.

The Ancient view falls through to the combined table for a form it has no row
of its own for, so a spelling that only Modern Greek keys reads the Modern
lemma there. A rule cannot tell these forms from Modern words that merely
share letters with an Ancient verb (εδώ and the εδω of ἔδω, κάνει and the
κανει of καίνω), so each form is reviewed and listed.

The record is applied when the lookup is opened with ``lang="grc"``. The
combined view keeps its row, because the same spelling is current Modern
Greek there, and it does not join the Ancient table that ``lang="all"``
consults for its candidates.
"""

from __future__ import annotations

GRC_VIEW_ROW_RECORDS: dict[str, dict[str, str]] = {
    "βάλει": {
        "lemma": "βάλλω",
        "reason": "ancient_verb_under_modern_row",
        "evidence": "The only row is Modern (βάζω, wiktionary_el). The Ancient "
                    "reading is the spelling of βάλῃ, aorist subjunctive of "
                    "βάλλω, which the accent-stripped row βαλει also reads; "
                    "attested in the Byzantine vernacular corpus (#151).",
    },
}

GRC_VIEW_ROWS: dict[str, str] = {
    form: record["lemma"] for form, record in GRC_VIEW_ROW_RECORDS.items()
}

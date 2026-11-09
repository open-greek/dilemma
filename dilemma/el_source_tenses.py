"""Reviewed corrections to Modern Greek source-sense tense labels."""
from __future__ import annotations

RULE = "el_achthika_aorist_v1"
REVIEWED_AT = "2026-11-09T13:19:38Z"
CANONICAL_LEMMAS = frozenset({"άγω", "άγομαι"})
EVIDENCE = (
    "https://ebooks.edu.gr/ebooks/v/html/8547/2334/Grammatiki-Neas-Ellinikis-Glossas_A-B-G-Gymnasiou_html-apli/index_G_01.htm",
    "https://en.wiktionary.org/w/index.php?title=άγω&oldid=76677838",
)


def correct_sense_tense(form, source, sense, tags):
    """Correct only the reviewed first-singular reading of άγομαι.

    The original gloss is retained as source evidence. Other headwords,
    editions, grammatical persons, and already-correct readings are untouched.
    """
    original = set(tags)
    targets = {r.get("word") for r in sense.get("form_of", [])}
    if (source != "en" or form != "άχθηκα" or targets != {"άγομαι"}
            or not {"imperfect", "first-person", "singular"} <= original
            or original & {"present", "aorist", "perfect", "pluperfect", "future", "plural", "dual",
                           "second-person", "third-person", "imperative", "subjunctive", "participle", "infinitive"}):
        return list(tags), None
    corrected = sorted((original - {"imperfect"}) | {"aorist"})
    audit = {"rule": RULE, "reviewed_at": REVIEWED_AT, "form": form, "source": source,
             "original": {"tags": sorted(original), "source_record": sense},
             "corrected": {"tags": corrected}, "evidence": list(EVIDENCE),
             "evidence_note": "The school grammar and the headword table place άχθηκα in the passive aorist; αγόμουν is imperfect."}
    return corrected, audit


def restore_sense_tags(tags, audit):
    """Recover the original source tags, refusing a subsequently changed row."""
    if sorted(set(tags)) != audit["corrected"]["tags"]:
        raise ValueError("Source tags changed after correction")
    return list(audit["original"]["tags"])

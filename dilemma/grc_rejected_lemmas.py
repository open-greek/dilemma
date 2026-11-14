"""Reviewed Ancient Greek lemma values that are no citation form.

Each entry is a spelling that some source gives as a lemma but that no
dictionary heads, with the reason and the evidence it was reviewed on. The
lookup build (``build_lookup_db.py``) drops every reading whose lemma is
listed, ``build/build_glaux_pos_lookup.py`` skips GLAUx pairs with such a
lemma, and the runtime drops POS-table cells that target one, so tables
built before the record (the treebank and Wiktionary POS tables) cannot
return them either.

Only exact spellings are listed: ``τοῦτος`` (an LBG headword, the late
by-form of οὗτος) and ``ἄτη`` keep their own readings.
"""

from __future__ import annotations

REJECTED_LEMMA_TARGETS: dict[str, dict[str, str]] = {
    "Ταύτη": {
        "reason": "wiktionary_el_only_head",
        "evidence": "Only a Greek Wiktionary Ancient Greek page heads it "
                    "(grc_lexical_heads.json lists it under wiktionary_el_grc "
                    "alone); ταύτῃ and ταύτην are forms of οὗτος. GLAUx also "
                    "files four οὗτος forms under it as a name (#108).",
    },
    "Τοῦτος": {
        "reason": "wiktionary_el_only_head",
        "evidence": "Only a Greek Wiktionary Ancient Greek page heads it "
                    "(wiktionary_el_grc alone); GLAUx files Τούτου, Τούτοις, "
                    "Τοῦτο, Τοῦτον under it as a name. Lowercase τοῦτος (LBG) "
                    "is not affected (#108).",
    },
    "Ἡμων": {
        "reason": "wiktionary_el_only_head",
        "evidence": "Only a Greek Wiktionary Ancient Greek page heads it "
                    "(wiktionary_el_grc alone); ἡμῶν is the genitive of "
                    "ἡμεῖς (ἐγώ) (#108).",
    },
    "Ταύτης": {
        "reason": "glaux_name_lemma",
        "evidence": "GLAUx annotates Ταύτῃ, Ταύτην, ταύτῃ, ταύτης (forms of "
                    "οὗτος) as a proper noun with this lemma; no dictionary "
                    "heads it (#108).",
    },
    "Τοῦτης": {
        "reason": "glaux_name_lemma",
        "evidence": "GLAUx annotates Τοῦτο as a proper noun with this lemma; "
                    "no dictionary heads it (#108).",
    },
    "Τούτῖς": {
        "reason": "glaux_name_lemma",
        "evidence": "GLAUx annotates Τούτων (pron) with this lemma; no "
                    "dictionary heads it (#108).",
    },
    "Τουτό": {
        "reason": "glaux_name_lemma",
        "evidence": "GLAUx annotates Τοῦτο (adv) with this lemma; no "
                    "dictionary heads it (#108).",
    },
    "Αὕτῃ": {
        "reason": "glaux_name_lemma",
        "evidence": "GLAUx annotates Αὕτη (adv) with this lemma, a form of "
                    "οὗτος; no dictionary heads it (#108).",
    },
    "Τούτων": {
        "reason": "glaux_name_lemma",
        "evidence": "GLAUx annotates τούτων as a noun with this lemma, the "
                    "genitive plural of οὗτος; no dictionary heads it (#108).",
    },
}

REJECTED_LEMMAS = frozenset(REJECTED_LEMMA_TARGETS)

# Closed-class pronominal heads. A capitalised corpus form whose lowercase
# spelling already reads as one of these (from a dictionary source) takes
# that reading rather than the corpus's: GLAUx gives sentence-initial Αὕτη
# the lemma Ἄτη, which the build lowercases to the real but wrong head ἄτη.
CLOSED_PRONOMINAL_LEMMAS = frozenset({
    "οὗτος", "ὅδε", "ἐκεῖνος", "αὐτός", "ἑαυτοῦ", "ἐμαυτοῦ", "σεαυτοῦ",
    "ἐγώ", "σύ", "ἡμεῖς", "ὑμεῖς", "ὅς", "ὅστις", "τίς", "τις", "πᾶς",
    "ἅπας", "εἷς", "οὐδείς", "μηδείς", "ἄλλος", "ἕτερος", "ὅσος", "οἷος",
    "σός", "ἐμός", "ἀλλήλων", "τοιοῦτος", "τοσοῦτος",
})

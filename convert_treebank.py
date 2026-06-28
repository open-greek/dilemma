#!/usr/bin/env python3
"""AGDT/GLAUx treebank -> Universal Dependencies conversion for the tagger.

Pure helpers (no lxml, no Morpheus): parse a 9-character AGDT/Perseus postag
into a UPOS tag + UD feature dict, and normalize a feature dict onto the gold
treebank convention. Used by build/build_tagger_data.py (data prep) and
train_tagger.py (gold-feature normalization), so the Ancient-Greek tagger is
reproducible from the public repo with no Morpheus dependency.
"""

# AGDT POS (postag position 0) -> UD UPOS
_POS_TO_UPOS = {
    "n": "NOUN", "v": "VERB", "a": "ADJ", "d": "ADV",
    "l": "DET", "c": "CCONJ", "r": "ADP", "p": "PRON",
    "m": "NUM", "i": "INTJ", "e": "INTJ",
    "u": "PUNCT", "x": "X", "g": "PART",
    # GLAUx codes coordinators/particles (καί, δέ, τε, ἀλλά, ἤ, ...) as "b" in
    # its hand-annotated texts; the same lemmas are "c" (CCONJ) in the bulk
    # corpus, so map to CCONJ for convention consistency. The SCONJ-by-lemma
    # refinement below still reclassifies subordinators.
    "b": "CCONJ",
}

# Subordinating conjunction lemmas (AGDT "c" -> SCONJ instead of CCONJ)
_SCONJ_LEMMAS = frozenset({
    "εἰ", "ὡς", "ὅτε", "ὅτι", "ἵνα", "ὅπως", "ἐπεί", "ἐπειδή",
    "ἐάν", "ὅταν", "ἕως", "πρίν", "ὥστε", "μέχρι", "ἄν",
})

# Tense (position 3) -> Tense + Aspect features
_TENSE_MAP = {
    "p": {"Tense": "Pres"},
    "i": {"Tense": "Past", "Aspect": "Imp"},
    "r": {"Tense": "Pres", "Aspect": "Perf"},
    "l": {"Tense": "Pqp"},
    "f": {"Tense": "Fut"},
    "a": {"Tense": "Past", "Aspect": "Perf"},
    "t": {"Tense": "Fut", "Aspect": "Perf"},
}

# Mood (position 4) -> Mood + VerbForm features
_MOOD_MAP = {
    "i": {"Mood": "Ind", "VerbForm": "Fin"},
    "s": {"Mood": "Sub", "VerbForm": "Fin"},
    "o": {"Mood": "Opt", "VerbForm": "Fin"},
    "n": {"VerbForm": "Inf"},
    "m": {"Mood": "Imp", "VerbForm": "Fin"},
    "p": {"VerbForm": "Part"},
}

_PERSON_MAP = {"1": "1", "2": "2", "3": "3"}
_NUMBER_MAP = {"s": "Sing", "p": "Plur", "d": "Dual"}
_VOICE_MAP = {"a": "Act", "p": "Pass", "m": "Mid", "e": "Mid,Pass"}
_GENDER_MAP = {"m": "Masc", "f": "Fem", "n": "Neut"}
_CASE_MAP = {"n": "Nom", "g": "Gen", "d": "Dat", "a": "Acc", "v": "Voc"}
_DEGREE_MAP = {"c": "Cmp", "s": "Sup"}

# Per-UPOS feature whitelist, for normalize_feats (the gold treebank convention).
_ALLOWED_FEATS = {
    "NOUN": {"Case", "Gender", "Number"},
    "PROPN": {"Case", "Gender", "Number"},
    "ADJ": {"Case", "Degree", "Gender", "Number"},
    "DET": {"Case", "Gender", "Number"},
    "PRON": {"Case", "Gender", "Number"},
    "ADV": {"Degree"},
    "NUM": set(),
    "VERB": {"Aspect", "Case", "Gender", "Mood", "Number", "Person", "Tense",
             "VerbForm", "Voice"},
    "AUX": {"Aspect", "Case", "Gender", "Mood", "Number", "Person", "Tense",
            "VerbForm", "Voice"},
    "ADP": set(), "PART": set(), "INTJ": set(), "CCONJ": set(),
    "SCONJ": set(), "PUNCT": set(), "SYM": set(),
}


def _is_uppercase_greek(ch: str) -> bool:
    return ch.isalpha() and ch.isupper()


def convert_postag(postag: str, lemma: str = "") -> tuple[str, dict]:
    """Convert a 9-character AGDT postag to (upos, feats_dict).

    Position layout:
      0: part of speech    4: mood      8: degree
      1: person            5: voice
      2: number            6: gender
      3: tense             7: case
    """
    if not postag or len(postag) != 9 or postag == "---------":
        return ("X", {})

    upos = _POS_TO_UPOS.get(postag[0], "X")
    if upos == "CCONJ" and lemma in _SCONJ_LEMMAS:
        upos = "SCONJ"
    if upos == "NOUN" and lemma and _is_uppercase_greek(lemma[0]):
        upos = "PROPN"

    feats = {}
    person = _PERSON_MAP.get(postag[1])
    if person:
        feats["Person"] = person
    number = _NUMBER_MAP.get(postag[2])
    if number:
        feats["Number"] = number
    tense_feats = _TENSE_MAP.get(postag[3])
    if tense_feats:
        feats.update(tense_feats)
    mood_feats = _MOOD_MAP.get(postag[4])
    if mood_feats:
        feats.update(mood_feats)
    voice = _VOICE_MAP.get(postag[5])
    if voice:
        feats["Voice"] = voice
    gender = _GENDER_MAP.get(postag[6])
    if gender:
        feats["Gender"] = gender
    case = _CASE_MAP.get(postag[7])
    if case:
        feats["Case"] = case
    degree = _DEGREE_MAP.get(postag[8])
    if degree:
        feats["Degree"] = degree
    return (upos, feats)


def normalize_feats(upos: str, lemma: str, feats: dict) -> dict:
    """Project a feature dict onto the gold treebank convention: drop keys the
    gold never annotates for that UPOS, the positive degree (gold marks Degree
    only on comparatives/superlatives), and Voice on the copula εἰμί."""
    allowed = _ALLOWED_FEATS.get(upos)
    out = {}
    for k, v in feats.items():
        if allowed is not None and k not in allowed:
            continue
        if k == "Degree" and v == "Pos":
            continue
        out[k] = v
    if lemma == "εἰμί":
        out.pop("Voice", None)
    return out

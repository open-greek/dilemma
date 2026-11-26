"""Crasis resolution for Ancient Greek.

Crasis is the contraction of two words into one, typically an article
or καί + the following word. For example:
  τοὔνομα = τό + ὄνομα  -> lemma: ὄνομα
  κἀγώ = καί + ἐγώ      -> lemma: κἀγώ (has its own entry)
  τἀνδρός = τοῦ + ἀνδρός -> lemma: ἀνήρ

Rules:
  1. If the crasis form has its own Wiktionary entry with no clear
     main word, return the crasis form itself as the lemma.
  2. For τ- crasis (article + noun), return the lemma of the noun.
  3. For κ- crasis (καί + X), return the lemma of X if identifiable.
  4. For forms not in the table, return None (let caller handle it).
"""

import unicodedata

# Hand-curated crasis table: form -> lemma of the main content word.
# Sources: EN Wiktionary AG entries with "crasis" or "contraction of"
# in the gloss, plus standard Greek grammar references.
CRASIS_TABLE = {
    # τ- crasis: article + noun/adjective -> noun lemma
    "τοὔνομα": "ὄνομα",
    "τοὔνεκα": "τοὔνεκα",       # adverb, has own entry ("for that reason")
    "τοὐναντίον": "ἐναντίος",
    "τοὐμόν": "ἐμός",
    "τοὐμπαλιν": "ἔμπαλιν",
    "τἀργύριον": "ἀργύριον",
    "τἀδικήματα": "ἀδίκημα",
    "τἀναγκαῖα": "ἀναγκαῖος",
    "τἀνδρός": "ἀνήρ",
    "τἀνδρί": "ἀνήρ",
    "τἆλλα": "ἄλλος",
    "τἀκεῖ": "ἐκεῖ",
    "τὠνόματα": "ὄνομα",
    "τὠυτό": "αὐτός",
    "ταὐτός": "αὐτός",
    "ταὐτά": "αὐτός",
    "θάτερα": "ἕτερος",

    # κ- crasis: καί + X -> lemma of X or self if established word
    "κἀγώ": "ἐγώ",
    "κἀγαθός": "ἀγαθός",
    "κἀκ": "ἐκ",
    "κἀν": "ἐν",
    "κἄν": "ἄν",
    "κἀκείνων": "ἐκεῖνος",
    "κἀκεῖθεν": "ἐκεῖθεν",
    "κἀνταῦθα": "ἐνταῦθα",
    "κεἰς": "εἰς",
    "κᾷτα": "εἶτα",
    "κᾆτα": "εἶτα",
    "κἆτα": "εἶτα",
    "καὐτός": "αὐτός",

    # ὁ/ἡ + X crasis
    "αὑτή": "αὐτός",
    "αὑτοῦ": "αὐτός",
    "οὑκ": "ἐκ",
    "ἁνήρ": "ἀνήρ",
    "ὡνήρ": "ἀνήρ",
    "ὦνερ": "ἀνήρ",
    "ἅνδρες": "ἀνήρ",
    "ἇνδρες": "ἀνήρ",
    "ὧνδρες": "ἀνήρ",
    "ὦνδρες": "ἀνήρ",
    "ὤριστος": "ἄριστος",

    # μέντοι + ἄν
    "μεντἄν": "ἄν",
}


def resolve_crasis(form: str) -> str | None:
    """Resolve a crasis form to the lemma of its main content word.

    Returns the lemma if known, or None if the form is not recognized
    as crasis.
    """
    return CRASIS_TABLE.get(form)


# ---- Crasis restoration (#161, #162) -----------------------------------------
#
# A crasis of καί or of the article with a word that opens with a vowel writes
# the merged vowel with a coronis (a smooth breathing inside the word): κἀκεῖ
# (καὶ ἐκεῖ), κᾄπειτα (καὶ ἔπειτα), τοὐπιχειρεῖν (τὸ ἐπιχειρεῖν), τἀμφίβληστρα
# (τὰ ἀμφίβληστρα). The word's lemma is the second word's, which the table
# above lists for the common forms. crasis_restorations() gives the spellings
# the second word may have had, in tiers: the caller reads them against the
# lookup and takes an answer only when one tier gives a single headword.

_PSILI, _DASIA, _SUBSCRIPT = "\u0313", "\u0314", "\u0345"
_ACCENTS = ("\u0301", "\u0300", "\u0342")
_VOWELS = "αεηιουω"

# (consonant, coronis vowel or diphthong, circumflex?) -> tiers of the vowels
# the second word may have opened with. A subscript under the vowel is καί's
# ι: κᾀ is καὶ ἐ- far more often than καὶ ἀ- (κᾀγώ, κᾄπειτα, κᾀκεῖνος).
_KAI_TIERS = {
    "α": (("α", "ε"),), "α_sub": (("ε",), ("α",)), "α_circ": (("ει",), ("α",)),
    "η": (("η", "ε"),), "ω": (("ο", "ω"),),
    "ου": (("ου",),), "αυ": (("αυ",),), "ει": (("ει",),), "οι": (("οι",),),
}
_ARTICLE_TIERS = {
    "α": (("α", "ε"),), "α_sub": (("α", "ε"),), "α_circ": (("α",),),
    "ω": (("ο", "ω"),), "η": (("η", "ε"),),
    "ου": (("ο", "ε", "ου"),), "αυ": (("αυ",),),
}


def _units(word: str) -> list:
    units = []
    for ch in unicodedata.normalize("NFD", word):
        if unicodedata.combining(ch):
            if units:
                units[-1][1].append(ch)
        else:
            units.append([ch, []])
    return units


def is_crasis_shaped(word: str) -> bool:
    """Whether *word* opens with κ or τ and a vowel with a coronis."""
    return bool(crasis_restorations(word))


def crasis_restorations(word: str) -> tuple:
    """The spellings the second word of a crasis *word* may have had, as
    tiers of lowercase NFC spellings; an empty tuple when *word* is no
    crasis of καί or of the article. The second word had a smooth
    breathing: a rough one turns κ and τ into χ and θ (χἠ, θἄτερον)."""
    units = _units(unicodedata.normalize("NFC", word))
    if len(units) < 3 or units[0][0].lower() not in "κτ" or units[0][1]:
        return ()
    first, second = units[1], units[2]
    if first[0].lower() not in _VOWELS:
        return ()
    if _PSILI in first[1] and not first[1].count(_DASIA):
        coronis, marks, rest = first[0].lower(), first[1], units[2:]
    elif (_PSILI in second[1] and not first[1]
          and (first[0] + second[0]).lower() in ("ου", "αυ", "ει", "οι")):
        coronis, marks, rest = (first[0] + second[0]).lower(), second[1], units[3:]
    else:
        return ()
    if not rest or "̈" in marks:
        return ()
    accent = next((mark for mark in marks if mark in _ACCENTS), "")
    key = coronis
    if coronis == "α" and _SUBSCRIPT in marks:
        key = "α_sub"
    if coronis == "α" and accent == "͂":
        key = "α_circ"
    table = _KAI_TIERS if units[0][0].lower() == "κ" else _ARTICLE_TIERS
    tiers = table.get(key)
    if not tiers:
        return ()
    tail = "".join(base + "".join(m) for base, m in rest)
    out = []
    for vowels in tiers:
        tier = []
        for vowel in vowels:
            if accent == "͂" and vowel in ("ε", "ο"):
                continue
            spelling = unicodedata.normalize(
                "NFC", vowel + _PSILI + accent + tail).lower()
            if spelling not in tier:
                tier.append(spelling)
        if tier:
            out.append(tuple(tier))
    return tuple(out)

"""Join a compound verb lemma that a treebank writes with its preverbs apart.

AGDT names some compound verbs as ``preverb-verb`` (ἐπί-εἰσκομίζω,
σύν-καταφεύγω), with two preverbs as ``preverb,preverb-verb``
(ἀντί,κατά-ἕζομαι). That spelling is no citation form. ``joined_preverb_lemma``
writes the compound the way a lexicon does, with the preverb's elision,
aspiration and assimilation; ``preverb_lemma_head`` accepts the result only
when a headword list has it, and returns the head as the list spells it.
"""
import unicodedata

_SMOOTH, _ROUGH, _DIAERESIS = "̓", "̔", "̈"
_TONAL = frozenset("́̀͂")
_VOWELS = frozenset("αεηιουω")

# Each preverb as the treebank spells it apart: its stem before a consonant,
# its stem before a vowel, and its stem before a vowel with a rough breathing
# (ἀπο-, ἀπ-, ἀφ-). περί and πρό never elide.
_PREVERBS = {
    "ἀμφί": ("ἀμφι", "ἀμφ", "ἀμφ"),
    "ἀνά": ("ἀνα", "ἀν", "ἀν"),
    "ἀντί": ("ἀντι", "ἀντ", "ἀνθ"),
    "ἀπό": ("ἀπο", "ἀπ", "ἀφ"),
    "διά": ("δια", "δι", "δι"),
    "εἰς": ("εἰσ", "εἰσ", "εἰσ"),
    "εἰσ": ("εἰσ", "εἰσ", "εἰσ"),
    "ἐκ": ("ἐκ", "ἐξ", "ἐξ"),
    "ἐξ": ("ἐκ", "ἐξ", "ἐξ"),
    "ἐν": ("ἐν", "ἐν", "ἐν"),
    "ἐπί": ("ἐπι", "ἐπ", "ἐφ"),
    "κατά": ("κατα", "κατ", "καθ"),
    "μετά": ("μετα", "μετ", "μεθ"),
    "παρά": ("παρα", "παρ", "παρ"),
    "περί": ("περι", "περι", "περι"),
    "πρό": ("προ", "προ", "προ"),
    "πρός": ("προσ", "προσ", "προσ"),
    "πρόσ": ("προσ", "προσ", "προσ"),
    "σύν": ("συν", "συν", "συν"),
    "ὑπέρ": ("ὑπερ", "ὑπερ", "ὑπερ"),
    "ὑπό": ("ὑπο", "ὑπ", "ὑφ"),
}

# The final nu of ἐν and σύν before a consonant: συμπ-, συγκ-, συλλ-, συρρ-;
# σύν loses it before ζ and before σ with a consonant (συζ-, συστ-) and
# turns it into σ before σ with a vowel (συσσ-).
_NASAL = {**dict.fromkeys("βπφψμ", "μ"), **dict.fromkeys("γκχξ", "γ"),
          "λ": "λ", "ρ": "ρ"}


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _letters(word: str) -> list[list[str]]:
    """The word as letters, each a base character with its combining marks."""
    letters: list[list[str]] = []
    for char in unicodedata.normalize("NFD", word):
        if unicodedata.combining(char):
            if not letters:
                return []
            letters[-1].append(char)
        else:
            letters.append([char])
    return letters


def _attach(preverb: str, stem: str) -> str | None:
    """``stem`` with ``preverb`` in front of it, or None when the stem is no
    word the rule knows how to join (a vowel without a breathing)."""
    before_consonant, before_vowel, before_rough = _PREVERBS[preverb]
    letters = _letters(stem)
    if not letters or not all(letter[0].isalpha() for letter in letters):
        return None
    first = letters[0][0]
    if first in _VOWELS:
        # The breathing sits on the first vowel, or on the second of a
        # diphthong (εἰ-, αὐ-).
        marked = next((i for i in (0, 1) if i < len(letters)
                       and letters[i][0] in _VOWELS
                       and (_SMOOTH in letters[i] or _ROUGH in letters[i])),
                      None)
        if marked is None:
            return None
        rough = _ROUGH in letters[marked]
        letters[marked] = [mark for mark in letters[marked]
                           if mark not in (_SMOOTH, _ROUGH)]
        head = before_rough if rough else before_vowel
        if preverb == "πρό" and first in "ιυ" and marked == 0:
            # προϊστορέω, προϋποτίθημι: the vowel stays apart from the ο
            letters[0].insert(1, _DIAERESIS)
    elif first == "ρ":
        if _ROUGH not in letters[0]:
            return None
        letters[0] = [mark for mark in letters[0] if mark != _ROUGH]
        head = before_consonant
        if preverb in ("ἐν", "σύν"):
            head = head[:-1] + "ρ"
        elif head[-1] in _VOWELS:
            letters.insert(0, ["ρ"])
    else:
        if any(mark in (_SMOOTH, _ROUGH) for mark in letters[0]):
            return None
        head = before_consonant
        if preverb in ("ἐν", "σύν"):
            second = letters[1][0] if len(letters) > 1 else ""
            if first in _NASAL:
                head = head[:-1] + _NASAL[first]
            elif preverb == "σύν" and (first == "ζ" or (
                    first == "σ" and second not in _VOWELS)):
                head = head[:-1]
            elif preverb == "σύν" and first == "σ":
                head = head[:-1] + "σ"
    return _nfc(head + "".join("".join(letter) for letter in letters))


def joined_preverb_lemma(lemma: str) -> str | None:
    """The compound a ``preverb-verb`` lemma names, or None.

    ἐπί-εἰσκομίζω is ἐπεισκομίζω, σύν-καταφεύγω is συγκαταφεύγω,
    ἀντί,κατά-ἕζομαι is ἀντικαθέζομαι and πρό-ἱστορέω is προϊστορέω. The
    preverbs are attached from the verb outwards. The verb keeps its accent,
    which a short verb does not keep in composition (περί-εἶμι is περίειμι):
    ``preverb_lemma_head`` settles the accent from a headword list. None
    when the part before the hyphen is not a list of preverbs spelled as
    AGDT spells them, so a line-broken token's lemma (αὐ-τός, προσ-τίθημι)
    and OCR are left alone.
    """
    if not isinstance(lemma, str) or lemma.count("-") != 1:
        return None
    lemma = _nfc(lemma)
    front, stem = lemma.split("-")
    preverbs = front.split(",")
    if not stem or not all(preverb in _PREVERBS for preverb in preverbs):
        return None
    for preverb in reversed(preverbs):
        stem = _attach(preverb, stem)
        if stem is None:
            return None
    return stem


def accent_fold(word: str) -> str:
    """``word`` without its accents and diaereses; breathings, iota
    subscripts and letters stay."""
    return _nfc("".join(
        char for char in unicodedata.normalize("NFD", word)
        if char not in _TONAL and char != _DIAERESIS))


def heads_by_accent_fold(heads) -> dict[str, tuple[str, ...]]:
    """``heads`` indexed by ``accent_fold``, for ``preverb_lemma_head``."""
    index: dict[str, list[str]] = {}
    for head in heads:
        index.setdefault(accent_fold(head), []).append(head)
    return {key: tuple(sorted(found)) for key, found in index.items()}


def preverb_lemma_head(lemma: str, heads, folded_heads) -> str | None:
    """The listed headword a ``preverb-verb`` lemma names, or None.

    The joined compound stands when ``heads`` has it. Otherwise the single
    head that differs from it only in accent or diaeresis is taken, which
    gives the compound the lexicon's accent (περί-εἶμι -> περίειμι,
    κατά-προίημι -> καταπροΐημι). A compound no list has, or one that two
    heads spell, yields None.
    """
    joined = joined_preverb_lemma(lemma)
    if joined is None:
        return None
    if joined in heads:
        return joined
    same = folded_heads.get(accent_fold(joined), ())
    return same[0] if len(same) == 1 else None

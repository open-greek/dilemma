"""Paradigm generator orchestrator.

A thin Python facade in front of the scattered paradigm tables that
already exist in the wider Greek-tooling ecosystem (jtauber's
greek-inflexion output, the Morpheus stemsrc dump, dilemma's own
corpus-derived paradigm extracts). For a given Ancient Greek lemma and
inflection slot, pick the form by source precedence:

    jtauber > Morpheus > dilemma_corpus > template

For cells that none of the JSON sources cover, a small set of
template rules synthesises forms for the simplest regular cases
(thematic -ω verbs, vowel-stem 1st / 2nd-declension nouns,
three-termination -ος / -η / -ον adjectives). The templates
deliberately decline to handle anything that needs accent
re-placement or stem allomorphy: contract -άω / -έω / -όω, athematic
μι-verbs, suppletive verbs (φέρω, εἰμί, ...), 3rd-declension
consonant stems, etc. — those return None so callers can leave the
cell empty rather than ship a bogus form.

Returned forms are wrapped in `ParadigmForm(form=..., source=...)` so
build pipelines can decide whether to trust the cell. Pure-Python:
no torch, no onnxruntime, no DB queries, no network. Mirrors
`dilemma.morph_diff`'s shape.

Data discovery: the orchestrator looks for paradigm JSON files in
`$DILEMMA_PARADIGM_DATA`. Expected filenames:

    jtauber_ag_paradigms.json
    ag_verb_paradigms.json            (Morpheus-derived verbs)
    ag_noun_paradigms.json            (Morpheus-derived nouns/adjs)
    dilemma_ag_verb_paradigms.json    (corpus-derived verbs)
    dilemma_ag_noun_paradigms.json    (corpus-derived nouns/adjs)

If a file is missing, that source is silently skipped and the next
in the precedence chain takes over. If no sources are configured,
only the template fallback is available.

Usage:

    from dilemma.paradigm import (
        generate, generate_paradigm, ParadigmSlot, ParadigmForm,
    )

    slot = ParadigmSlot.verb_finite(
        voice="active", tense="aorist", mood="indicative",
        person="1", number="sg",
    )
    f = generate("γράφω", slot)
    # ParadigmForm(form="ἔγραψα", source="jtauber")

    paradigm = generate_paradigm("γράφω", pos="verb")
    # {"active_aorist_indicative_1sg": ParadigmForm(...), ...}
"""

from __future__ import annotations

import json
import os
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


__all__ = [
    "ParadigmSource",
    "ParadigmSlot",
    "ParadigmForm",
    "generate",
    "generate_paradigm",
    "iter_slots",
    "fill_canonical_dict",
    "reset_cache",
]


# ---------------------------------------------------------------------------
# Source enum + lightweight dataclasses
# ---------------------------------------------------------------------------


class ParadigmSource(str, Enum):
    """Which underlying table produced a form."""

    JTAUBER = "jtauber"
    MORPHEUS = "morpheus"
    DILEMMA_CORPUS = "dilemma_corpus"
    TEMPLATE = "template"


@dataclass(frozen=True)
class ParadigmSlot:
    """A single paradigm cell, identified by POS + grammatical features.

    The `key` property emits the canonical inflection-key shape that
    matches what jtauber's greek-inflexion library and the
    dilemma-derived paradigm extracts emit:

        verb finite:     <voice>_<tense>_<mood>_<person><number>
                         e.g. active_aorist_indicative_1sg
        verb infinitive: <voice>_<tense>_infinitive
                         e.g. active_aorist_infinitive
        verb participle: <voice>_<tense>_participle_<case>_<gender>_<number>
                         e.g. active_aorist_participle_nom_m_sg
        noun:            <case>_<number>           (e.g. genitive_pl)
        adj:             <case>_<gender>_<number>  (e.g. nominative_m_sg)
    """

    pos: str
    voice: Optional[str] = None
    tense: Optional[str] = None
    mood: Optional[str] = None  # "indicative" / "subjunctive" / ... / "infinitive" / "participle"
    person: Optional[str] = None
    number: Optional[str] = None
    case: Optional[str] = None
    gender: Optional[str] = None  # "m" / "f" / "n"

    @property
    def key(self) -> str:
        if self.pos == "verb":
            if self.mood == "participle":
                return (
                    f"{self.voice}_{self.tense}_participle_"
                    f"{self.case}_{self.gender}_{self.number}"
                )
            if self.mood == "infinitive":
                return f"{self.voice}_{self.tense}_infinitive"
            return (
                f"{self.voice}_{self.tense}_{self.mood}_"
                f"{self.person}{self.number}"
            )
        if self.pos == "noun":
            return f"{self.case}_{self.number}"
        if self.pos == "adj":
            return f"{self.case}_{self.gender}_{self.number}"
        raise ValueError(f"unknown pos: {self.pos!r}")

    # Convenience constructors for callers that don't want to remember
    # which fields apply to which shape.
    @classmethod
    def verb_finite(
        cls,
        *,
        voice: str,
        tense: str,
        mood: str,
        person: str,
        number: str,
    ) -> "ParadigmSlot":
        return cls(
            pos="verb", voice=voice, tense=tense, mood=mood,
            person=person, number=number,
        )

    @classmethod
    def verb_infinitive(cls, *, voice: str, tense: str) -> "ParadigmSlot":
        return cls(pos="verb", voice=voice, tense=tense, mood="infinitive")

    @classmethod
    def verb_participle(
        cls,
        *,
        voice: str,
        tense: str,
        case: str,
        gender: str,
        number: str,
    ) -> "ParadigmSlot":
        return cls(
            pos="verb", voice=voice, tense=tense, mood="participle",
            case=case, gender=gender, number=number,
        )

    @classmethod
    def noun(cls, *, case: str, number: str) -> "ParadigmSlot":
        return cls(pos="noun", case=case, number=number)

    @classmethod
    def adj(cls, *, case: str, gender: str, number: str) -> "ParadigmSlot":
        return cls(pos="adj", case=case, gender=gender, number=number)


@dataclass(frozen=True)
class ParadigmForm:
    """A generated form plus the source that produced it."""

    form: str
    source: str  # one of ParadigmSource values

    def __str__(self) -> str:
        return self.form


# ---------------------------------------------------------------------------
# Slot grids
# ---------------------------------------------------------------------------


_VERB_VOICES: Tuple[str, ...] = ("active", "middle", "passive", "mediopassive")

# Indicative ranges over the full 7 traditional tenses; subjunctive /
# optative / imperative each pick a subset (Greek doesn't form e.g. an
# imperfect subjunctive). Future-perfect indicative is rare in
# practice but jtauber/Morpheus emit it for the few verbs that have
# one, so we keep it in the grid.
_INDICATIVE_TENSES: Tuple[str, ...] = (
    "present", "imperfect", "future", "aorist",
    "perfect", "pluperfect", "future_perfect",
)
_SUBJUNCTIVE_TENSES: Tuple[str, ...] = ("present", "aorist", "perfect")
_OPTATIVE_TENSES: Tuple[str, ...] = ("present", "future", "aorist", "perfect")
_IMPERATIVE_TENSES: Tuple[str, ...] = ("present", "aorist", "perfect")
_INFINITIVE_TENSES: Tuple[str, ...] = ("present", "future", "aorist", "perfect")
_PARTICIPLE_TENSES: Tuple[str, ...] = (
    "present", "future", "aorist", "perfect", "future_perfect",
)

_PERSONS: Tuple[str, ...] = ("1", "2", "3")
_NUMBERS: Tuple[str, ...] = ("sg", "pl", "du")

_NOUN_CASES: Tuple[str, ...] = (
    "nominative", "genitive", "dative", "accusative", "vocative",
)
# Participle case keys use 3-letter abbreviations (matches canonical YAML).
_PARTICIPLE_CASES: Tuple[str, ...] = ("nom", "gen", "dat", "acc", "voc")
_GENDERS: Tuple[str, ...] = ("m", "f", "n")


def _verb_pn_combos(persons: Tuple[str, ...]) -> Iterator[Tuple[str, str]]:
    """Iterate (person, number) for finite verb forms.

    Greek has no first-person dual (1du), so it's filtered out. All
    other person × number combinations are yielded.
    """
    for p in persons:
        for n in _NUMBERS:
            if p == "1" and n == "du":
                continue
            yield p, n


def iter_slots(pos: str) -> Iterator[ParadigmSlot]:
    """Yield every canonical leaf slot for a POS.

    For verbs this is the cross-product of voice × mood/tense × p/n
    plus per-voice/tense infinitives and participles. For nouns and
    adjectives it's case × (gender) × number. The orchestrator uses
    this grid to compute "missing" cells against an existing
    inflections map; cells no source has and that templates can't fill
    just stay empty.
    """
    if pos == "verb":
        for voice in _VERB_VOICES:
            for tense in _INDICATIVE_TENSES:
                for person, number in _verb_pn_combos(_PERSONS):
                    yield ParadigmSlot.verb_finite(
                        voice=voice, tense=tense, mood="indicative",
                        person=person, number=number,
                    )
            for tense in _SUBJUNCTIVE_TENSES:
                for person, number in _verb_pn_combos(_PERSONS):
                    yield ParadigmSlot.verb_finite(
                        voice=voice, tense=tense, mood="subjunctive",
                        person=person, number=number,
                    )
            for tense in _OPTATIVE_TENSES:
                for person, number in _verb_pn_combos(_PERSONS):
                    yield ParadigmSlot.verb_finite(
                        voice=voice, tense=tense, mood="optative",
                        person=person, number=number,
                    )
            for tense in _IMPERATIVE_TENSES:
                # Imperative is only attested for 2nd / 3rd person.
                for person, number in _verb_pn_combos(("2", "3")):
                    yield ParadigmSlot.verb_finite(
                        voice=voice, tense=tense, mood="imperative",
                        person=person, number=number,
                    )
            for tense in _INFINITIVE_TENSES:
                yield ParadigmSlot.verb_infinitive(voice=voice, tense=tense)
            for tense in _PARTICIPLE_TENSES:
                for case in _PARTICIPLE_CASES:
                    for gender in _GENDERS:
                        for number in _NUMBERS:
                            yield ParadigmSlot.verb_participle(
                                voice=voice, tense=tense,
                                case=case, gender=gender, number=number,
                            )
        return
    if pos == "noun":
        for case in _NOUN_CASES:
            for number in _NUMBERS:
                yield ParadigmSlot.noun(case=case, number=number)
        return
    if pos == "adj":
        for case in _NOUN_CASES:
            for gender in _GENDERS:
                for number in _NUMBERS:
                    yield ParadigmSlot.adj(
                        case=case, gender=gender, number=number,
                    )
        return
    # Unknown POS: empty grid.
    return


# ---------------------------------------------------------------------------
# Source loading (lazy)
# ---------------------------------------------------------------------------


_SOURCE_FILES: Dict[str, str] = {
    # source_id -> filename in a candidate data dir
    "jtauber_verb": "jtauber_ag_paradigms.json",
    "morpheus_verb": "ag_verb_paradigms.json",
    "morpheus_noun": "ag_noun_paradigms.json",
    "dilemma_verb": "dilemma_ag_verb_paradigms.json",
    "dilemma_noun": "dilemma_ag_noun_paradigms.json",
}


def _candidate_data_dirs() -> List[Path]:
    """Return the directories to scan for paradigm JSON files.

    Currently a single entry: `$DILEMMA_PARADIGM_DATA` if set. Tests
    that need to pin the orchestrator at a specific directory monkey
    patch this function.
    """
    env = os.environ.get("DILEMMA_PARADIGM_DATA")
    if env:
        return [Path(env)]
    return []


_SOURCE_CACHE: Optional[Dict[str, Dict[str, dict]]] = None


def reset_cache() -> None:
    """Drop the in-memory paradigm-source cache.

    Tests that mutate $DILEMMA_PARADIGM_DATA between calls should
    invoke this to force a re-load.
    """
    global _SOURCE_CACHE
    _SOURCE_CACHE = None


def _load_sources() -> Dict[str, Dict[str, dict]]:
    """Load paradigm JSONs from the first candidate dir that has each.

    Returns a dict {source_id: {lemma: entry}}. Missing files yield
    an empty per-source dict so lookups uniformly return None
    rather than erroring on absent data.
    """
    global _SOURCE_CACHE
    if _SOURCE_CACHE is not None:
        return _SOURCE_CACHE
    loaded: Dict[str, Dict[str, dict]] = {k: {} for k in _SOURCE_FILES}
    for d in _candidate_data_dirs():
        if not d.exists() or not d.is_dir():
            continue
        for src, fname in _SOURCE_FILES.items():
            if loaded[src]:
                continue  # an earlier candidate already filled this slot
            p = d / fname
            if not p.exists():
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                loaded[src] = data
    _SOURCE_CACHE = loaded
    return loaded


def _lookup(source_data: Dict[str, dict], lemma: str, key: str) -> Optional[str]:
    if not source_data:
        return None
    entry = source_data.get(lemma)
    if not isinstance(entry, dict):
        return None
    forms = entry.get("forms")
    if not isinstance(forms, dict):
        return None
    val = forms.get(key)
    if isinstance(val, str):
        v = val.strip()
        if v:
            return v
    return None


# ---------------------------------------------------------------------------
# Template fallback
# ---------------------------------------------------------------------------


def _strip_diacritics(s: str) -> str:
    """Lower-case + strip every combining mark."""
    nfd = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in nfd if not unicodedata.combining(c)).lower()


def _starts_with_vowel(s: str) -> bool:
    base = _strip_diacritics(s)
    return bool(base) and base[0] in "αεηιουω"


# Verb templates: handle the simplest regular case only — present-system
# active forms for plain thematic -ω verbs. Contract -άω/-έω/-όω, all
# μ-verbs, mediopassive, future, aorist, perfect, and the like return
# None: their stem allomorphy / accent shifts are too irregular to
# synthesise without dedicated paradigm tables.

_THEMATIC_PRESENT_ACTIVE_INDICATIVE: Dict[Tuple[str, str], str] = {
    ("1", "sg"): "ω",
    ("2", "sg"): "εις",
    ("3", "sg"): "ει",
    ("1", "pl"): "ομεν",
    ("2", "pl"): "ετε",
    ("3", "pl"): "ουσι(ν)",
    ("2", "du"): "ετον",
    ("3", "du"): "ετον",
}

_THEMATIC_PRESENT_ACTIVE_SUBJUNCTIVE: Dict[Tuple[str, str], str] = {
    ("1", "sg"): "ω",
    ("2", "sg"): "ῃς",
    ("3", "sg"): "ῃ",
    ("1", "pl"): "ωμεν",
    ("2", "pl"): "ητε",
    ("3", "pl"): "ωσι(ν)",
    ("2", "du"): "ητον",
    ("3", "du"): "ητον",
}

_THEMATIC_PRESENT_ACTIVE_OPTATIVE: Dict[Tuple[str, str], str] = {
    ("1", "sg"): "οιμι",
    ("2", "sg"): "οις",
    ("3", "sg"): "οι",
    ("1", "pl"): "οιμεν",
    ("2", "pl"): "οιτε",
    ("3", "pl"): "οιεν",
    ("2", "du"): "οιτον",
    ("3", "du"): "οίτην",
}

_THEMATIC_PRESENT_ACTIVE_IMPERATIVE: Dict[Tuple[str, str], str] = {
    ("2", "sg"): "ε",
    ("3", "sg"): "έτω",
    ("2", "pl"): "ετε",
    ("3", "pl"): "όντων",
    ("2", "du"): "ετον",
    ("3", "du"): "έτων",
}


def _is_regular_thematic_omega(lemma: str) -> bool:
    """True for plain -ω verbs that aren't contracts or athematic.

    Filters out -άω / -έω / -όω contract verbs and -μι / -μαι
    athematic / mediopassive lemmas. These need stem-specific contract
    or athematic templates that we deliberately don't ship.
    """
    base = _strip_diacritics(lemma)
    if not base.endswith("ω"):
        return False
    if base.endswith(("αω", "εω", "οω")):
        return False
    if base.endswith(("μι", "μαι")):
        return False
    return True


def _verb_stem(lemma: str) -> Optional[str]:
    """Strip the final NFC code point of a -ω lemma to recover the stem.

    Preserves the lemma's accent and breathing marks on the stem so
    the synthesised forms inherit them; we never try to compute a
    fresh accent placement, which keeps the template's failure modes
    visible (the form looks slightly off rather than confidently
    wrong).
    """
    if not lemma:
        return None
    nfc = unicodedata.normalize("NFC", lemma)
    if not nfc.endswith("ω") and not nfc.endswith("ώ"):
        return None
    return nfc[:-1]


def _verb_template(lemma: str, slot: ParadigmSlot) -> Optional[str]:
    if slot.voice != "active":
        return None
    if not _is_regular_thematic_omega(lemma):
        return None
    stem = _verb_stem(lemma)
    if stem is None:
        return None
    if slot.mood == "infinitive" and slot.tense == "present":
        return stem + "ειν"
    if slot.tense != "present":
        return None
    pn = (slot.person or "", slot.number or "")
    table: Optional[Dict[Tuple[str, str], str]] = None
    if slot.mood == "indicative":
        table = _THEMATIC_PRESENT_ACTIVE_INDICATIVE
    elif slot.mood == "subjunctive":
        table = _THEMATIC_PRESENT_ACTIVE_SUBJUNCTIVE
    elif slot.mood == "optative":
        table = _THEMATIC_PRESENT_ACTIVE_OPTATIVE
    elif slot.mood == "imperative":
        table = _THEMATIC_PRESENT_ACTIVE_IMPERATIVE
    if table is None:
        return None
    ending = table.get(pn)
    if ending is None:
        return None
    return stem + ending


# Noun templates: vowel-stem 1st-decl (-α, -η) and 2nd-decl (-ος, -ον).
# 3rd-decl consonant stems return None (their stem is opaque from the
# nominative singular alone — δαίμων vs. ποιμήν vs. πούς — and Morpheus
# already covers them via dilemma_ag_noun_paradigms.json anyway).

_FIRST_DECL_LONG_A: Dict[Tuple[str, str], str] = {
    # For lemmas like χώρα where α stays through the singular.
    ("nominative", "sg"): "α",
    ("genitive", "sg"): "ας",
    ("dative", "sg"): "ᾳ",
    ("accusative", "sg"): "αν",
    ("vocative", "sg"): "α",
    ("nominative", "pl"): "αι",
    ("genitive", "pl"): "ῶν",
    ("dative", "pl"): "αις",
    ("accusative", "pl"): "ας",
    ("vocative", "pl"): "αι",
    ("nominative", "du"): "α",
    ("genitive", "du"): "αιν",
    ("dative", "du"): "αιν",
    ("accusative", "du"): "α",
    ("vocative", "du"): "α",
}

_FIRST_DECL_ETA: Dict[Tuple[str, str], str] = {
    # Lemmas like τέχνη.
    ("nominative", "sg"): "η",
    ("genitive", "sg"): "ης",
    ("dative", "sg"): "ῃ",
    ("accusative", "sg"): "ην",
    ("vocative", "sg"): "η",
    ("nominative", "pl"): "αι",
    ("genitive", "pl"): "ῶν",
    ("dative", "pl"): "αις",
    ("accusative", "pl"): "ας",
    ("vocative", "pl"): "αι",
    ("nominative", "du"): "α",
    ("genitive", "du"): "αιν",
    ("dative", "du"): "αιν",
    ("accusative", "du"): "α",
    ("vocative", "du"): "α",
}

_SECOND_DECL_OS: Dict[Tuple[str, str], str] = {
    # Masc/fem -ος, e.g. λόγος. (Vocative is -ε.)
    ("nominative", "sg"): "ος",
    ("genitive", "sg"): "ου",
    ("dative", "sg"): "ῳ",
    ("accusative", "sg"): "ον",
    ("vocative", "sg"): "ε",
    ("nominative", "pl"): "οι",
    ("genitive", "pl"): "ων",
    ("dative", "pl"): "οις",
    ("accusative", "pl"): "ους",
    ("vocative", "pl"): "οι",
    ("nominative", "du"): "ω",
    ("genitive", "du"): "οιν",
    ("dative", "du"): "οιν",
    ("accusative", "du"): "ω",
    ("vocative", "du"): "ω",
}

_SECOND_DECL_ON: Dict[Tuple[str, str], str] = {
    # Neuter -ον, e.g. δῶρον. nom/acc/voc collapse per neuter rule.
    ("nominative", "sg"): "ον",
    ("genitive", "sg"): "ου",
    ("dative", "sg"): "ῳ",
    ("accusative", "sg"): "ον",
    ("vocative", "sg"): "ον",
    ("nominative", "pl"): "α",
    ("genitive", "pl"): "ων",
    ("dative", "pl"): "οις",
    ("accusative", "pl"): "α",
    ("vocative", "pl"): "α",
    ("nominative", "du"): "ω",
    ("genitive", "du"): "οιν",
    ("dative", "du"): "οιν",
    ("accusative", "du"): "ω",
    ("vocative", "du"): "ω",
}


def _noun_decl_table(lemma: str) -> Optional[Dict[Tuple[str, str], str]]:
    """Pick a vowel-stem declension table from the lemma ending.

    Returns None for anything we don't confidently handle (3rd-decl,
    irregular, contract). We match on the *diacritic-stripped*
    ending, but slice on the NFC lemma so accents on the stem are
    preserved.
    """
    base = _strip_diacritics(lemma)
    if base.endswith("α"):
        return _FIRST_DECL_LONG_A
    if base.endswith("η"):
        return _FIRST_DECL_ETA
    if base.endswith("ον"):
        return _SECOND_DECL_ON
    if base.endswith("ος"):
        return _SECOND_DECL_OS
    return None


def _noun_stem(lemma: str, table: Dict[Tuple[str, str], str]) -> str:
    """Strip the lemma ending in NFC space.

    The NFC strip preserves the stem's accent and breathing; the
    `table` we picked already encoded which suffix length to remove
    (1 for α/η/ω, 2 for ος/ον). We re-derive that length from the
    table's nominative-singular ending so the two stay in sync.
    """
    nfc = unicodedata.normalize("NFC", lemma)
    nom_sg = table[("nominative", "sg")]
    if nfc.lower().endswith(_strip_diacritics(nom_sg)):
        return nfc[: -len(nom_sg)]
    return nfc


def _noun_template(lemma: str, slot: ParadigmSlot) -> Optional[str]:
    table = _noun_decl_table(lemma)
    if table is None:
        return None
    ending = table.get((slot.case or "", slot.number or ""))
    if ending is None:
        return None
    # The lemma's citation form (nominative singular) is the lemma
    # itself: returning it verbatim preserves whatever accent the
    # caller wrote. Derived forms get an approximate accent because
    # the template doesn't model Greek accent shifts.
    if (slot.case, slot.number) == ("nominative", "sg"):
        return unicodedata.normalize("NFC", lemma)
    stem = _noun_stem(lemma, table)
    return stem + ending


# Adj templates: 2-1-2 three-termination -ος / -η / -ον (the canonical
# textbook adjective shape). Picks the masculine table from
# _SECOND_DECL_OS / _SECOND_DECL_ON and the feminine from
# _FIRST_DECL_ETA. -ος/-α/-ον adjectives (where α follows ε/ι/ρ) and
# 2-2 contract / consonant-stem adjectives are intentionally not
# handled — Morpheus / dilemma_ag_noun_paradigms.json cover those.

_ADJ_M_BY_GENDER: Dict[str, Dict[Tuple[str, str], str]] = {
    "m": _SECOND_DECL_OS,
    "f": _FIRST_DECL_ETA,
    "n": _SECOND_DECL_ON,
}


def _adj_template(lemma: str, slot: ParadigmSlot) -> Optional[str]:
    base = _strip_diacritics(lemma)
    if not base.endswith("ος"):
        return None  # Only -ος three-term adjs are templatable.
    table = _ADJ_M_BY_GENDER.get(slot.gender or "")
    if table is None:
        return None
    ending = table.get((slot.case or "", slot.number or ""))
    if ending is None:
        return None
    nfc = unicodedata.normalize("NFC", lemma)
    # Citation form (masc nom sg) is the lemma. Returning the lemma
    # verbatim lets oxytones like καλός keep their accent.
    if (slot.case, slot.gender, slot.number) == ("nominative", "m", "sg"):
        return nfc
    stem = nfc[:-2]  # strip "ος"
    return stem + ending


def _template_form(lemma: str, slot: ParadigmSlot) -> Optional[str]:
    if slot.pos == "verb":
        return _verb_template(lemma, slot)
    if slot.pos == "noun":
        return _noun_template(lemma, slot)
    if slot.pos == "adj":
        return _adj_template(lemma, slot)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate(
    lemma: str,
    slot: ParadigmSlot,
    lang: str = "grc",
    *,
    allow_template: bool = True,
) -> Optional[ParadigmForm]:
    """Resolve `lemma`'s `slot` form by source precedence.

    Order: jtauber > Morpheus > dilemma_corpus > template. The first
    source that has a non-empty form wins. Returns None when no
    source has the form and the template fallback either can't safely
    produce one (irregular lemma, non-templatable cell) or
    `allow_template=False`.

    `lang` only accepts `"grc"` for now; modern Greek paradigms
    aren't generated by this orchestrator (kaikki MG coverage is
    already strong, so the brief defers MG support).
    """
    if lang != "grc":
        return None
    if not lemma or slot is None:
        return None
    sources = _load_sources()
    key = slot.key

    if slot.pos == "verb":
        order = (
            (ParadigmSource.JTAUBER.value, sources["jtauber_verb"]),
            (ParadigmSource.MORPHEUS.value, sources["morpheus_verb"]),
            (ParadigmSource.DILEMMA_CORPUS.value, sources["dilemma_verb"]),
        )
    else:
        order = (
            (ParadigmSource.MORPHEUS.value, sources["morpheus_noun"]),
            (ParadigmSource.DILEMMA_CORPUS.value, sources["dilemma_noun"]),
        )

    for src, data in order:
        f = _lookup(data, lemma, key)
        if f:
            return ParadigmForm(form=f, source=src)

    if not allow_template:
        return None
    f = _template_form(lemma, slot)
    if f:
        return ParadigmForm(form=f, source=ParadigmSource.TEMPLATE.value)
    return None


def generate_paradigm(
    lemma: str,
    pos: str,
    lang: str = "grc",
    *,
    allow_template: bool = True,
) -> Dict[str, ParadigmForm]:
    """Generate every leaf cell `iter_slots(pos)` covers.

    Returns a {key -> ParadigmForm} dict, keyed by canonical
    inflection-key shape. Cells nobody has and templates can't fill
    are simply absent from the dict.
    """
    if lang != "grc":
        return {}
    out: Dict[str, ParadigmForm] = {}
    for slot in iter_slots(pos):
        f = generate(lemma, slot, lang=lang, allow_template=allow_template)
        if f is not None:
            out[slot.key] = f
    return out


# ---------------------------------------------------------------------------
# Canonical-dict integration helper
# ---------------------------------------------------------------------------


def fill_canonical_dict(
    canonical: Dict[str, dict],
    *,
    allow_template: bool = False,
    dialect: str = "attic",
) -> Tuple[Dict[str, dict], Dict[str, int]]:
    """Fill missing inflection cells across a canonical-AG entry map.

    `canonical` is a mapping `{key -> entry}` where `entry` has at
    least `lemma`, `pos`, and (optionally) `inflections.<dialect>:
    {inflection_key: form}`. The keys of the outer dict are
    arbitrary - typically file paths from a downstream canonical
    builder's `pending_canonical`.

    For every verb / noun / adj entry, missing cells in the named
    `dialect` are filled via `generate(...)`. The mutated entry
    grows two new shapes:

      * inflections.<dialect>.<inflection_key>: <form>
      * inflections_source.<dialect>.<inflection_key>: <source>

    `allow_template=False` (the default) skips template-derived
    fills, matching the downstream canonical builder's default behaviour. Pass
    `allow_template=True` to opt in to the template fallback.

    Returns the mutated map plus a small stats dict (cells filled
    per source) so callers can log how many cells each source
    produced.
    """
    stats: Dict[str, int] = {
        ParadigmSource.JTAUBER.value: 0,
        ParadigmSource.MORPHEUS.value: 0,
        ParadigmSource.DILEMMA_CORPUS.value: 0,
        ParadigmSource.TEMPLATE.value: 0,
        "lemmas_touched": 0,
    }
    for entry in canonical.values():
        if not isinstance(entry, dict):
            continue
        lemma = entry.get("lemma")
        pos = entry.get("pos")
        if not isinstance(lemma, str) or not isinstance(pos, str):
            continue
        if pos not in ("verb", "noun", "adj"):
            continue
        inflections = entry.get("inflections")
        if not isinstance(inflections, dict):
            inflections = {}
            entry["inflections"] = inflections
        attic = inflections.get(dialect)
        if not isinstance(attic, dict):
            attic = {}
            inflections[dialect] = attic
        sources_block = entry.get("inflections_source")
        if not isinstance(sources_block, dict):
            sources_block = {}
            entry["inflections_source"] = sources_block
        attic_sources = sources_block.get(dialect)
        if not isinstance(attic_sources, dict):
            attic_sources = {}
            sources_block[dialect] = attic_sources
        any_filled = False
        for slot in iter_slots(pos):
            key = slot.key
            if key in attic:
                continue
            f = generate(lemma, slot, allow_template=allow_template)
            if f is None:
                continue
            attic[key] = f.form
            attic_sources[key] = f.source
            stats[f.source] = stats.get(f.source, 0) + 1
            any_filled = True
        if any_filled:
            stats["lemmas_touched"] += 1
        # Don't leave empty containers behind if nothing was added.
        if not attic_sources:
            sources_block.pop(dialect, None)
        if not sources_block:
            entry.pop("inflections_source", None)
        if not attic:
            inflections.pop(dialect, None)
        if not inflections:
            entry.pop("inflections", None)
    return canonical, stats


# ---------------------------------------------------------------------------
# CLI: read pending-canonical JSON, emit filled JSON
# ---------------------------------------------------------------------------


def _cli_fill(argv: List[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="dilemma.paradigm fill",
        description="Fill missing inflection cells in a canonical-AG dict.",
    )
    p.add_argument("--in", dest="in_path", required=True,
                   help="path to a JSON file (mapping {key: entry})")
    p.add_argument("--out", dest="out_path", required=True,
                   help="path to write the filled JSON to")
    p.add_argument("--with-templates", action="store_true",
                   help="include template-derived fills in addition to "
                        "the JSON-source ones")
    p.add_argument("--dialect", default="attic",
                   help="which dialect bucket to fill (default: attic)")
    args = p.parse_args(argv)
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    canonical = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(canonical, dict):
        print("input JSON must be an object {key: entry}", flush=True)
        return 2
    filled, stats = fill_canonical_dict(
        canonical,
        allow_template=args.with_templates,
        dialect=args.dialect,
    )
    out_path.write_text(
        json.dumps(filled, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = ", ".join(f"{k}={v}" for k, v in sorted(stats.items()))
    print(f"paradigm fill: {summary}", flush=True)
    return 0


def _cli(argv: List[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print("Usage: python -m dilemma.paradigm fill --in IN --out OUT "
              "[--with-templates] [--dialect attic]")
        return 0
    cmd, *rest = argv
    if cmd == "fill":
        return _cli_fill(rest)
    print(f"Unknown command: {cmd}", flush=True)
    return 2


if __name__ == "__main__":
    import sys

    sys.exit(_cli(sys.argv[1:]))

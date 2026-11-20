"""Dilemma - Greek lemmatizer.

Fast lookup table for known forms, custom transformer model for unknown forms.

Usage:
    from dilemma import Dilemma

    m = Dilemma()                        # loads lookup table; model stays lazy
    m.lemmatize("πάθης")                # -> "παθαίνω"
    m.lemmatize("πολεμούσαν")           # -> "πολεμώ"
    m.lemmatize_batch(["δώση", "σκότωσε"])  # -> ["δίνω", "σκοτώνω"]

    # Elision expansion (uses Wiktionary lookup)
    m.lemmatize("ἀλλ̓")                  # -> "ἀλλά"
    m.lemmatize("ἔφατ̓")                 # -> "φημί"

    # Verbose mode: returns all candidates with metadata
    m.lemmatize_verbose("ἔριδι")
    # -> [LemmaCandidate(lemma="ἔρις", lang="grc", proper=False),
    #     LemmaCandidate(lemma="Ἔρις", lang="grc", proper=True)]

    m.lemmatize_verbose("πόλεμο")
    # -> [LemmaCandidate(lemma="πόλεμος", lang="el"),
    #     LemmaCandidate(lemma="πόλεμος", lang="grc")]

    # Convention remapping: LSJ dictionary headwords
    m_lsj = Dilemma(convention="lsj")
    m_lsj.lemmatize("αἰνῶς")        # -> "αἰνός" (adverb -> adjective)
    m_lsj.lemmatize("εἶπον")        # -> "λέγω" (aorist -> present stem)
"""

import json
import hashlib
import math
import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .nonlexical import classify_nonlexical as _classify_nonlexical
from .nonlexical import is_lexical as _is_lexical
from .nonlexical import NONLEXICAL_POS as _NONLEXICAL_POS
from .nonlexical import NUMERAL as _NONLEXICAL_NUMERAL
from .citation import malformed_tonal_reason as _malformed_tonal_reason
from .form_sanitize import is_capitalized, sanitize_form
from .grc_rejected_lemmas import REJECTED_LEMMAS as _REJECTED_POS_LEMMAS
from .grc_view_rows import GRC_VIEW_ROWS as _GRC_VIEW_ROWS
from .grammatical_lemmas import (
    correct_grammatical_lemma, preferred_reviewed_lemma, reviewed_verb_grammar,
    reviewed_verb_readings,
)


def _newest_marker_mtime(d: Path, markers):
    """Newest mtime among `markers` (glob patterns relative to `d`), or None if
    `d` holds none of them. lookup.db and the model weights are written only by
    a rebuild or `dilemma download`, never by git (they are gitignored), so the
    mtime is a reliable "how fresh is this copy" signal."""
    best = None
    for pat in markers:
        for f in d.glob(pat):
            try:
                mt = f.stat().st_mtime
            except OSError:
                continue
            if best is None or mt > best:
                best = mt
    return best


def _resolve_dir(env_var: str, subdir: str, markers) -> Path:
    """Resolve a Dilemma data/model directory, preferring freshness over a fixed
    precedence so a locally rebuilt copy is never shadowed by a stale download
    cache (and vice-versa).

      1. $<env_var> (explicit override) if set and existing.
      2. Otherwise, among the dev tree (<repo-root>/<subdir>/), the download
         cache (~/.cache/dilemma/<subdir>/), and any bundled <package>/<subdir>/,
         pick the one whose data was built/downloaded most recently.
      3. If none holds data yet, fall back to whichever dir exists, then to
         ~/.cache/dilemma/<subdir>/ as a stable path to write to.
    """
    env = os.environ.get(env_var)
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    cache = Path.home() / ".cache" / "dilemma" / subdir
    dev = Path(__file__).resolve().parent.parent / subdir
    bundled = Path(__file__).resolve().parent / subdir
    scored = [(d, _newest_marker_mtime(d, markers)) for d in (dev, cache, bundled)]
    scored = [(d, mt) for d, mt in scored if mt is not None]
    if scored:
        return max(scored, key=lambda dm: dm[1])[0]
    for d in (cache, dev, bundled):
        if d.exists():
            return d
    return cache


def _resolve_data_dir() -> Path:
    """Directory holding Dilemma data files (lookup.db, spell_index.db, ...)."""
    return _resolve_dir("DILEMMA_DATA_DIR", "data", ["lookup.db"])


def _resolve_model_dir() -> Path:
    """Directory holding trained model files (<scale>/encoder.onnx | model.pt)."""
    return _resolve_dir("DILEMMA_MODEL_DIR", "model", ["*/encoder.onnx", "*/model.pt"])


DATA_DIR = _resolve_data_dir()
MODEL_DIR = _resolve_model_dir()
LOOKUP_DB_PATH = DATA_DIR / "lookup.db"
SPELL_INDEX_PATH = DATA_DIR / "spell_index.db"
LSJ9_POS_LOOKUP_PATH = DATA_DIR / "lsj9_pos_lookup.json"
LSJ9_INDECLINABLES_PATH = DATA_DIR / "lsj9_indeclinables.json"
LSJ9_FREQUENCY_PATH = DATA_DIR / "lsj9_frequency.json"
LOOKUP_PATH = DATA_DIR / "mg_lookup.json"
AG_LOOKUP_PATH = DATA_DIR / "ag_lookup.json"
MED_LOOKUP_PATH = DATA_DIR / "med_lookup.json"
MG_POS_LOOKUP_PATH = DATA_DIR / "mg_pos_lookup.json"
AG_POS_LOOKUP_PATH = DATA_DIR / "ag_pos_lookup.json"
TREEBANK_POS_LOOKUP_PATH = DATA_DIR / "treebank_pos_lookup.json"
TREEBANK_POS_AUDIT_PATH = DATA_DIR / "treebank_pos_lookup.audit.json"
GLAUX_POS_LOOKUP_PATH = DATA_DIR / "glaux_pos_lookup.json"
POS_CELL_CORRECTIONS_PATH = DATA_DIR / "pos_cell_corrections.json"
GRC_LEXICAL_HEADS_PATH = DATA_DIR / "grc_lexical_heads.json"
# Regular Ancient adverbs in -ως -> their adjective, plus the reviewed adverbs
# kept as headwords (build/build_grc_adverb_heads.py; issue #117).
GRC_ADVERB_HEADS_PATH = DATA_DIR / "grc_adverb_heads.json"


LSJ_HEADWORDS_PATH = DATA_DIR / "lsj_headwords.json"
LSJ10_HEADWORDS_PATH = DATA_DIR / "lsj10_headwords.json"
CUNLIFFE_HEADWORDS_PATH = DATA_DIR / "cunliffe_headwords.json"
MG_HEADWORDS_PATH = DATA_DIR / "mg_headwords.json"
TRIANTAFYLLIDIS_HEADWORDS_PATH = DATA_DIR / "triantafyllidis_headwords.json"
AG_HEADWORDS_PATH = DATA_DIR / "ag_headwords.json"
DGE_HEADWORDS_PATH = DATA_DIR / "dge_headwords.json"
LBG_HEADWORDS_PATH = DATA_DIR / "lbg_headwords.json"
VLG_HEADWORDS_PATH = DATA_DIR / "vlg_headwords.json"
PD_HEADWORDS_PATH = DATA_DIR / "pd_headwords.json"
WIP_HEADWORDS_PATH = DATA_DIR / "wip_headwords.json"
LGPN_NAMES_PATH = DATA_DIR / "lgpn_names.json"
LEMMA_EQUIVALENCES_PATH = DATA_DIR / "lemma_equivalences.json"
CORPUS_FREQ_PATH = DATA_DIR / "corpus_freq.json"
ATTESTATION_PATH = DATA_DIR / "lemma_attestation.json"
CONVENTION_DIR = DATA_DIR

_VALID_CONVENTIONS = {None, "lsj", "cunliffe", "triantafyllidis", "wiktionary"}
_VALID_CITATION_POLICIES = {"default", "strict_ag"}

# Map convention name -> headword file path for auto-derivation.
# Conventions not listed here use LSJ headwords as fallback.
_CONVENTION_HEADWORDS = {
    "lsj": (LSJ_HEADWORDS_PATH,),
    "cunliffe": (CUNLIFFE_HEADWORDS_PATH,),
    # The dictionary list is the convention authority. Wiktionary remains an
    # additive fallback for Modern Greek coverage outside that dictionary.
    "triantafyllidis": (TRIANTAFYLLIDIS_HEADWORDS_PATH, MG_HEADWORDS_PATH),
}

# Conventions that output monotonic Greek (apply to_monotonic to all results).
_MONOTONIC_CONVENTIONS = {"triantafyllidis"}


class DilemmaDataWarning(UserWarning):
    """A data file that lemmatization reads is missing from the data directory.

    Dilemma still loads, but the paths that read the file (compound accenting,
    grave-accent citation checks, convention mapping, the -ως adverb map) fall
    back without it, so lemmas differ from those of a full download.
    """


# Lists the runtime reads, by language (issue #118). Every reader treats a
# missing file as empty, so Dilemma names them once instead of answering
# differently in silence. Keep this in step with new runtime-read lists.
_AG_DATA_LISTS = tuple(path.name for path in (
    LSJ_HEADWORDS_PATH, LSJ10_HEADWORDS_PATH, CUNLIFFE_HEADWORDS_PATH,
    DGE_HEADWORDS_PATH, LBG_HEADWORDS_PATH, VLG_HEADWORDS_PATH,
    PD_HEADWORDS_PATH, WIP_HEADWORDS_PATH, AG_HEADWORDS_PATH,
    GRC_LEXICAL_HEADS_PATH, GRC_ADVERB_HEADS_PATH, POS_CELL_CORRECTIONS_PATH))
_MG_DATA_LISTS = (TRIANTAFYLLIDIS_HEADWORDS_PATH.name, MG_HEADWORDS_PATH.name)
_CONVENTION_DATA_FILES = {
    "lsj": (LEMMA_EQUIVALENCES_PATH.name, "convention_lsj.json"),
    "cunliffe": (LEMMA_EQUIVALENCES_PATH.name, "convention_cunliffe.json"),
    "triantafyllidis": (LEMMA_EQUIVALENCES_PATH.name,
                        "convention_triantafyllidis.json"),
}
_WARNED_MISSING_DATA: set[tuple[str, tuple[str, ...]]] = set()


def missing_data_files(lang: str = "all", convention: str | None = None,
                       data_dir: Path | str | None = None) -> list[str]:
    """The data files a Dilemma(lang, convention) reads that ``data_dir`` lacks."""
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    names: list[str] = []
    if lang in ("all", "grc"):
        names += _AG_DATA_LISTS
    if lang in ("all", "el"):
        names += _MG_DATA_LISTS
    names += _CONVENTION_DATA_FILES.get(convention, ())
    return sorted({name for name in names if not (data_dir / name).exists()})


def _warn_missing_data_files(lang: str, convention: str | None) -> None:
    """Warn once per data directory and missing set (DilemmaDataWarning)."""
    missing = missing_data_files(lang, convention)
    key = (str(DATA_DIR), tuple(missing))
    if not missing or key in _WARNED_MISSING_DATA:
        return
    _WARNED_MISSING_DATA.add(key)
    import warnings
    warnings.warn(
        f"Dilemma data directory {DATA_DIR} lacks {len(missing)} file(s) that "
        f"lemmatization reads: {', '.join(missing)}. Compound accenting, "
        "grave-accent citation checks and convention mapping fall back without "
        "them, so lemmas differ from a full download. Fetch them with "
        "`python -m dilemma download --no-tagger`.",
        DilemmaDataWarning, stacklevel=3)


@lru_cache(maxsize=8)
def _bound_pos_rejections(audit_path, audit_size, audit_mtime,
                          heads_path, heads_size, heads_mtime):
    """Read the generated negative evidence only with its exact lexical input."""
    audit_file, heads_file = Path(audit_path), Path(heads_path)
    try:
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
        digest = hashlib.sha256()
        with heads_file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        current = {"size": heads_size, "sha256": digest.hexdigest()}
        rejected = audit.get("rejected_corpus_targets")
        if (audit.get("schema_version") != 2 or audit.get("language") != "grc"
                or audit.get("inputs", {}).get("grc_lexical_heads") != current
                or not isinstance(rejected, dict)
                or any(not isinstance(lemma, str) or not lemma
                       or not isinstance(sources, dict) or not sources
                       or any(not isinstance(source, str) or not source
                              or type(count) is not int or count <= 0
                              for source, count in sources.items())
                       for lemma, sources in rejected.items())
                or (audit_file.stat().st_size, audit_file.stat().st_mtime_ns)
                   != (audit_size, audit_mtime)
                or (heads_file.stat().st_size, heads_file.stat().st_mtime_ns)
                   != (heads_size, heads_mtime)):
            raise ValueError("unbound or malformed POS rejection evidence")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        raise ValueError("Ancient Greek POS data mismatch: refresh "
                         "treebank_pos_lookup.audit.json and grc_lexical_heads.json "
                         "together with python -m dilemma download --no-tagger") from error
    return frozenset(grave_to_acute(unicodedata.normalize("NFC", lemma))
                     for lemma in rejected)

# Independent ancient/Byzantine lexicon inventories used to prove that a
# grave-accented lookup value can be normalized to a real citation form. Do not
# include AG_HEADWORDS_PATH: that file is Wiktionary-derived and is known to
# contain the same grave-headword contamination this gate is meant to block.
_TRUSTED_AG_HEADWORD_PATHS = (
    LSJ_HEADWORDS_PATH, LSJ10_HEADWORDS_PATH, LBG_HEADWORDS_PATH,
    DGE_HEADWORDS_PATH, VLG_HEADWORDS_PATH, CUNLIFFE_HEADWORDS_PATH,
    PD_HEADWORDS_PATH, WIP_HEADWORDS_PATH,
)


@lru_cache(maxsize=2)
def _lexical_head_lists(path, size, mtime):
    """The lexical heads of grc_lexical_heads.json, by language, parsed once."""
    with Path(path).open(encoding="utf-8") as stream:
        evidence = json.load(stream)
    ancient = frozenset(evidence["heads"])
    vernacular = frozenset(evidence.get("vernacular_heads", ()))
    return {"grc": ancient, "el": vernacular, "all": ancient | vernacular}


def _lexical_heads(lang: str) -> frozenset[str]:
    """Ancient ("grc"), vernacular ("el") or all lexical heads; none without the file."""
    try:
        stat = GRC_LEXICAL_HEADS_PATH.stat()
    except OSError:
        return frozenset()
    return _lexical_head_lists(
        str(GRC_LEXICAL_HEADS_PATH), stat.st_size, stat.st_mtime_ns
    ).get(lang, frozenset())


_POS_CORRECTION_TABLES = frozenset({"treebank", "glaux", "ag", "mg", "lsj9"})


@lru_cache(maxsize=4)
def _pos_cell_correction_rows(path, size, mtime):
    """The reviewed POS-cell corrections, by table, parsed and checked once.

    Each row maps (form, upos) to (target, corrected): ``target`` is the
    cell the row expects (None for an ``add`` row, which expects no cell)
    and ``corrected`` the replacement (None for ``drop``).
    """
    with Path(path).open(encoding="utf-8") as stream:
        record = json.load(stream)
    by_table: dict[str, dict[tuple[str, str], tuple[str | None, str | None]]] = {}
    for row in record["corrections"]:
        table, action = row["table"], row["action"]
        corrected = row.get("corrected")
        target = row.get("target")
        if (table not in _POS_CORRECTION_TABLES
                or action not in ("drop", "correct", "add")
                or not all(isinstance(row.get(field), str) and row[field]
                           for field in ("form", "upos", "reason"))
                or (action == "add") != (target is None)
                or (action != "add" and not (isinstance(target, str) and target))
                or (action in ("correct", "add")) != isinstance(corrected, str)
                or (action in ("correct", "add") and not corrected)):
            raise ValueError(f"malformed POS cell correction: {row!r}")
        by_table.setdefault(table, {})[(row["form"], row["upos"])] = (
            target, corrected)
    return by_table


def _apply_pos_cell_corrections(table_name: str, table: dict,
                                path: Path | None = None) -> dict:
    """Apply data/pos_cell_corrections.json to one freshly loaded POS table.

    A row changes a cell only while the table still holds the row's target
    for that form and tag, so a rebuilt table that fixed the cell is left
    alone. A dropped cell lets a lower-priority table supply the tag. An
    ``add`` row supplies a cell only while the table has none for that
    form and tag (#128: a Modern reading that only a since-dropped
    paradigm fragment kept in the table).
    """
    path = POS_CELL_CORRECTIONS_PATH if path is None else path
    try:
        stat = path.stat()
    except OSError:
        return table
    rows = _pos_cell_correction_rows(str(path), stat.st_size, stat.st_mtime_ns)
    for (form, upos), (target, corrected) in rows.get(table_name, {}).items():
        entry = table.get(form)
        if target is None:
            if not entry or upos not in entry:
                table.setdefault(form, {})[upos] = corrected
            continue
        if not entry or entry.get(upos) != target:
            continue
        if corrected is None:
            del entry[upos]
            if not entry:
                del table[form]
        else:
            entry[upos] = corrected
    return table


@lru_cache(maxsize=2)
def _grc_adverb_tables(path, size, mtime):
    """The adverb -> adjective map and the kept adverbs, parsed once."""
    with Path(path).open(encoding="utf-8") as stream:
        record = json.load(stream)
    return dict(record.get("heads", {})), frozenset(record.get("keep", ()))


def _grc_adverb_data() -> tuple[dict[str, str], frozenset[str]]:
    """grc_adverb_heads.json as (heads, keep); empty without the file."""
    try:
        stat = GRC_ADVERB_HEADS_PATH.stat()
    except OSError:
        return {}, frozenset()
    return _grc_adverb_tables(
        str(GRC_ADVERB_HEADS_PATH), stat.st_size, stat.st_mtime_ns)


def _without_quantity_marks(word: str) -> str:
    nfd = unicodedata.normalize("NFD", word)
    return unicodedata.normalize("NFC", "".join(
        c for c in nfd if ord(c) not in (0x0304, 0x0306)))


def _starts_with_combining_mark(word: str) -> bool:
    return bool(word) and bool(unicodedata.combining(word[0]))


def _has_overline_mark(word: str) -> bool:
    return "\u0305" in unicodedata.normalize("NFD", word)


_CITATION_FINAL_ELISION_MARKS = {"\u2019", "\u02bc", "'", "\u1fbd", "`",
                                 "\u1fbf", "\u0313", "\u0314"}
_CITATION_FINAL_KERAIA_OR_PRIME_MARKS = {"\u0374", "\u02b9"}


def _citation_final_mark_reason(word: str) -> str | None:
    if not word:
        return None
    if word[-1] in _CITATION_FINAL_KERAIA_OR_PRIME_MARKS:
        return "final_keraia_or_prime"
    if word[-1] in _CITATION_FINAL_ELISION_MARKS:
        return "final_elision_mark"
    return None


def trusted_ag_citation_headwords(
        paths: tuple[Path, ...] = _TRUSTED_AG_HEADWORD_PATHS) -> set[str]:
    """Independent AG/Byzantine citation forms trusted for output validation.

    This deliberately excludes the Wiktionary-derived AG headword list: that
    source is useful for coverage, but it can carry the same citation-form
    contamination the validator is meant to catch.
    """
    targets: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        entries = raw.values() if isinstance(raw, dict) else raw
        for entry in entries:
            word = entry.get("lemma") if isinstance(entry, dict) else entry
            if not isinstance(word, str) or not word:
                continue
            word = unicodedata.normalize("NFC", word)
            targets.add(word)
            targets.add(_without_quantity_marks(word))
            if (not any(ch.isspace() for ch in word)
                    and "\u0300" in unicodedata.normalize("NFD", word)):
                acute = grave_to_acute(word)
                targets.add(acute)
                targets.add(_without_quantity_marks(acute))
    return targets


def _load_convention_headwords(convention: str) -> set[str]:
    """Load and union every headword inventory for a lemma convention."""
    headwords: set[str] = set()
    for path in _CONVENTION_HEADWORDS.get(convention, ()):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        headwords.update(raw)
        # LSJ-family lists may carry vowel quantity. Keeping the plain form
        # beside the display headword also makes comparison source-neutral.
        for headword in raw:
            nfd = unicodedata.normalize("NFD", headword)
            stripped = "".join(
                ch for ch in nfd if ord(ch) not in (0x0304, 0x0306)
            )
            headwords.add(unicodedata.normalize("NFC", stripped))
    return headwords


_POLYTONIC_STRIP = {0x0313, 0x0314, 0x0345, 0x0306, 0x0304}
_POLYTONIC_TO_ACUTE = {0x0300, 0x0342}

# Elision mark: U+0313 COMBINING COMMA ABOVE (repurposed as apostrophe
# in polytonic Greek text). Also handle right single quote U+2019 and
# modifier letter apostrophe U+02BC.
#
# The spacing marks here cover the six apostrophe codepoints real Greek text
# uses for elision - U+2019, ASCII ', U+02BC, U+1FBD, U+0060 grave (seen in
# some OCR / older digitizations), U+02B9 - kept in step with the sibling Open
# Greek prosodia engine's recognized set so the two tools agree on what an
# elision mark is; plus dilemma's combining U+0313 and the U+1FBF psili variant.
_ELISION_MARKS = {"\u0313", "\u2019", "\u02BC", "'", "\u1FBD",
                  "`", "\u02B9", "\u1FBF"}

# Vowels to try when expanding elision (ordered by frequency in AG text)
_GREEK_VOWELS = "αεοιηυω"

# Article and pronoun resolution: maps forms to canonical lemma.
# Used when resolve_articles=True (for treebank evaluation).
_ARTICLE_LEMMA = "ὁ"
_ARTICLE_FORMS = {
    # Polytonic
    "ὁ", "ἡ", "τό", "τοῦ", "τῆς", "τῶν", "τόν", "τήν",
    "τά", "τοῖς", "ταῖς", "τῷ", "τῇ", "τούς", "τάς", "τοῖν", "ταῖν",
    "οἱ", "αἱ", "τώ",
    # Grave variants, plus αἵ and οἵ (the article accented before an enclitic)
    "τὸ", "τοὺς", "τὰ", "τὸν", "τὴν", "τὰς", "αἵ", "οἵ",
    # Monotonic
    "ο", "η", "το", "του", "της", "των", "τον", "την",
    "τα", "τους", "τοις", "οι", "αι",
    # Stripped (no accents/breathings)
    "τω", "ται",
}
_MARKED_ARTICLE_FORMS = {
    form for form in _ARTICLE_FORMS
    if any(unicodedata.combining(char)
           for char in unicodedata.normalize("NFD", form))
}

# A bare letter tagged NOUN or ADJ is a numeral, siglum or list label
# (#116). The POS tables carry one-letter keys from dictionary abbreviations
# and stray corpus tokens (α ADJ -> Λητογενής, ι NOUN -> ζευγίτης), so
# those tags return the letter itself instead of trusting a cell.
_BARE_LETTER_UPOS = frozenset({"NOUN", "ADJ"})


def _bare_letter_nominal(word: str, upos: str | None) -> str | None:
    """Return ``word`` (NFC) when it is one unmarked Greek letter tagged
    NOUN or ADJ, else None. Accented or breathed letters (ἡ, ὁ, ά) are not
    bare and keep the article and pronoun logic."""
    if upos not in _BARE_LETTER_UPOS or not word:
        return None
    nfd = unicodedata.normalize("NFD", word)
    if len(nfd) != 1:
        return None
    code = ord(nfd)
    if (0x0391 <= code <= 0x03A9 or 0x03B1 <= code <= 0x03C9
            or 0x03D8 <= code <= 0x03E1 or code in (0x03F2, 0x03F9)):
        return unicodedata.normalize("NFC", word)
    return None


_PRONOUN_LEMMAS = {
    # 1st person -> ἐγώ
    "μοι": "ἐγώ", "μοί": "ἐγώ", "μου": "ἐγώ", "με": "ἐγώ",
    "ἐμοί": "ἐγώ", "ἐμοῦ": "ἐγώ", "ἐμέ": "ἐγώ",
    "ἡμεῖς": "ἐγώ", "ἡμῶν": "ἐγώ", "ἡμῖν": "ἐγώ", "ἡμᾶς": "ἐγώ",
    # 2nd person -> σύ
    "σοι": "σύ", "σοί": "σύ", "σου": "σύ", "σε": "σύ",
    "σοῦ": "σύ",
    "ὑμεῖς": "σύ", "ὑμῶν": "σύ", "ὑμῖν": "σύ", "ὑμᾶς": "σύ",
    # Personal/intensive pronoun, including the ordinary dual forms.
    **dict.fromkeys((
        "αὐτός", "αὐτή", "αὐτό", "αὐτοῦ", "αὐτῆς", "αὐτῷ", "αὐτῇ",
        "αὐτόν", "αὐτήν", "αὐτοί", "αὐταί", "αὐτά", "αὐτῶν",
        "αὐτοῖς", "αὐταῖς", "αὐτούς", "αὐτάς", "αὐτώ", "αὐτοῖν", "αὐταῖν",
    ), "αὐτός"),
    # Accent distinguishes the indefinite from the interrogative pronoun.
    **dict.fromkeys((
        "τις", "τι", "τινός", "τινί", "τινά", "τινές", "τινῶν",
        "τισί", "τισίν", "τινάς", "τινέ", "τινοῖν",
    ), "τις"),
    **dict.fromkeys((
        "τίς", "τί", "τίνος", "τίνι", "τίνα", "τίνες", "τίνων",
        "τίσι", "τίσιν", "τίνας", "τίνε", "τίνοιν",
    ), "τίς"),
}

# Modern Greek closed-class resolution (monotonic).
# Used when convention='triantafyllidis' to intercept MG function words
# before the AG lookup cascade, which otherwise misresolves them
# (e.g. στη -> ἵστημι, σε -> σύ, τις -> τις self-map).
_MG_ARTICLE_FORMS = {
    "ο", "η", "το", "τον", "την", "του", "της", "τα", "τους", "τις",
    "των", "τη", "οι",
}
_MG_ARTICLE_LEMMA = "ο"

# ---- Feature: Particle/enclitic suffix stripping ----
# Ancient Greek appends particles like -περ, -γε, -δε, and deictic -ι to
# words without changing the lemma. Ordered longest-first for greedy match.
_PARTICLE_SUFFIXES = ["περ", "γε", "δε"]

# Deictic -ι is only stripped from demonstrative pronoun stems.
# These are accent-stripped stems of common demonstratives (οὗτος, ὅδε, etc.)
_DEICTIC_STEMS = {
    "τουτο", "τουτω", "τουτοισ", "τουτου", "τουτων", "τουτοι",
    "ταυτη", "ταυτα", "ταυτησ", "ταυτων", "ταυται", "ταυταισ",
    "τουτουσ", "τουτοισι", "ταυτασ", "ταυτησι",
    "τοδ", "τηδ", "τωδ", "τονδ", "τηνδ",
    "ενθαδ",  # ἐνθάδε is a real word but ἐνθαδί also exists
}

# ---- Feature: Article-agreement disambiguation ----
# Maps article forms to (gender, number, case) features for ranking
# ambiguous lemma candidates. "m"=masculine, "f"=feminine, "n"=neuter;
# "s"=singular, "p"=plural, "d"=dual; case abbreviated.
_ARTICLE_FEATURES = {
    # Masculine
    "ὁ": ("m", "s", "nom"), "τοῦ": ("m", "s", "gen"),
    "τῷ": ("m", "s", "dat"), "τόν": ("m", "s", "acc"),
    "οἱ": ("m", "p", "nom"), "τῶν": ("m", "p", "gen"),
    "τοῖς": ("m", "p", "dat"), "τούς": ("m", "p", "acc"),
    # Feminine
    "ἡ": ("f", "s", "nom"), "τῆς": ("f", "s", "gen"),
    "τῇ": ("f", "s", "dat"), "τήν": ("f", "s", "acc"),
    "αἱ": ("f", "p", "nom"),
    "ταῖς": ("f", "p", "dat"), "τάς": ("f", "p", "acc"),
    # Neuter
    "τό": ("n", "s", "nom"), "τά": ("n", "p", "nom"),
    # Grave variants
    "τὸ": ("n", "s", "nom"), "τὰ": ("n", "p", "nom"),
    "τὸν": ("m", "s", "acc"), "τὴν": ("f", "s", "acc"),
    "τοὺς": ("m", "p", "acc"), "τὰς": ("f", "p", "acc"),
}

# Gender of common lemma endings (heuristic for AG nouns/adjectives)
_LEMMA_GENDER_HINTS = {
    # Masculine endings
    "ος": "m", "ής": "m", "εύς": "m", "ηρ": "m", "ων": "m",
    "ας": "m", "ής": "m",
    # Feminine endings
    "η": "f", "α": "f", "ις": "f", "ύς": "f",
    # Neuter endings
    "ον": "n", "ος": "m", "μα": "n", "ον": "n",
}

# Vernacular contractions of σε with the article (στο, στον, στην, στὸ,
# στὴν, ...), accent-stripped. With lang="all" or "el" they give the
# article under DET and σε under ADP, in monotonic and in polytonic
# spelling (#121, #134); their Ancient readings (στόν as ἀστός, στῆς as
# ἵστημι) stand under the other tags.
_VERNACULAR_ARTICLE_CONTRACTIONS = frozenset({
    "στο", "στον", "στη", "στην", "στα", "στου", "στης", "στων", "στους",
    "στις",
})

_MG_CLOSED_CLASS = {
    # Preposition contractions σε + article -> σε
    "στη": "σε", "στην": "σε", "στο": "σε", "στον": "σε",
    "στου": "σε", "στους": "σε", "στις": "σε", "στα": "σε",
    "στης": "σε", "στων": "σε",
    # Indefinite article -> ένας
    "ένα": "ένας", "μία": "ένας", "μια": "ένας",
    "ενός": "ένας", "μίας": "ένας", "μιας": "ένας",
    # Copula -> είμαι
    "είναι": "είμαι", "ήταν": "είμαι", "ήμουν": "είμαι",
    "ήσουν": "είμαι", "ήμαστε": "είμαι", "ήσαστε": "είμαι",
    # Common auxiliaries -> έχω
    "έχει": "έχω", "είχε": "έχω", "έχουν": "έχω",
    "είχαν": "έχω", "έχεις": "έχω", "είχες": "έχω",
    "έχουμε": "έχω", "είχαμε": "έχω",
    # Demonstrative/personal pronoun αυτός (MG Wiktionary has bad
    # mappings like αυτών->τα due to article/pronoun conflation)
    "αυτό": "αυτός", "αυτόν": "αυτός", "αυτήν": "αυτός",
    "αυτού": "αυτός", "αυτής": "αυτός", "αυτών": "αυτός",
    "αυτούς": "αυτός", "αυτές": "αυτός", "αυτοί": "αυτός",
    "αυτά": "αυτός",
}


@dataclass
class LemmaCandidate:
    """A lemma candidate with metadata for disambiguation."""
    lemma: str
    lang: str = ""       # "el" (SMG), "grc" (AG), "med" (Medieval), "" (unknown)
    proper: bool = False  # True if lemma is a proper noun (capitalized headword)
    source: str = ""      # "lookup", "elision", "crasis", "model", "article"
    score: float = 1.0    # confidence (1.0 for lookup, lower for model)
    via: str = ""         # how the lookup matched: "exact", "lower", "mono",
                          # "stripped", "elision:ε" (which vowel expanded), etc.
    attestation: dict | None = None  # corpus attestation of the matched surface
                          # form, populated only by lemmatize_verbose(
                          # with_attestation=True); None otherwise.
    tag: str = ""         # UD POS refinement; "X" for a NON-LEXICAL token
                          # (apparatus mark, numeral, ref, abbreviation, siglum)
                          # classified by dilemma.nonlexical. "" otherwise.
    citation: str = ""    # Citation-form validation note, e.g.
                          # "trusted_headword", "grave_to_trusted_headword".
    grammar: dict | None = None  # Reviewed grammatical lemma, analyses and
                                # related headwords; independent of convention.

    @property
    def is_lexical(self) -> bool:
        """False iff this candidate is a NON-LEXICAL token (source
        ``nonlexical``), i.e. not a real word to be lemmatized."""
        return self.source != "nonlexical"


@dataclass(frozen=True)
class _CitationLemmaCheck:
    lemma: str | None
    reason: str
    normalized_from: str = ""


def to_monotonic(s: str) -> str:
    """Convert polytonic Greek to monotonic."""
    nfd = unicodedata.normalize("NFD", s)
    out = []
    for ch in nfd:
        cp = ord(ch)
        if cp in _POLYTONIC_STRIP:
            continue
        if cp in _POLYTONIC_TO_ACUTE:
            out.append("\u0301")
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def grave_to_acute(s: str) -> str:
    """Convert grave accents to acute, preserving all other diacritics.

    In Greek orthography, grave (βαρεῖα) is a positional variant of acute —
    it appears on the last syllable when followed by another word. So ὣς = ὡς,
    τὸν = τόν, etc. This is a lighter normalization than to_monotonic(), which
    also strips breathings and circumflex.
    """
    nfd = unicodedata.normalize("NFD", s)
    out = []
    for ch in nfd:
        if ord(ch) == 0x0300:  # COMBINING GRAVE ACCENT
            out.append("\u0301")  # COMBINING ACUTE ACCENT
        else:
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


# Preserve accent placement when accepting monotonic closed-class spellings.
_MONOTONIC_ARTICLE_FORMS = {to_monotonic(form) for form in _ARTICLE_FORMS}
_MONOTONIC_PRONOUN_LEMMAS = {
    to_monotonic(form): lemma for form, lemma in _PRONOUN_LEMMAS.items()
}


def strip_accents(s: str) -> str:
    """Strip all accents for fuzzy matching."""
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if unicodedata.category(c) != "Mn"))


def to_standard_sigma(s: str) -> str:
    """Normalize lunate sigma (ϲ U+03F2 / Ϲ U+03F9) to standard σ/ς.

    Lunate sigma is a pure glyph variant used throughout many ancient-Greek
    editions (papyri, inscriptions, scholia, much of First1KGreek); it carries
    no lexical distinction. Without this, forms like ``ὡϲ``/``τῆϲ`` never match
    the standard-sigma lookup, so whole editions silently fail to lemmatize.
    Map lunate -> medial σ / capital Σ, then fix word-final σ -> ς. The
    word-final σ -> ς fix always runs (Greek words never end in medial σ), so
    Beta-Code-derived keys/queries like ``γραφῆσ``/``εἷσ`` canonicalize to the
    final-ς spelling and match.
    """
    if "ϲ" in s or "Ϲ" in s:
        s = s.replace("ϲ", "σ").replace("Ϲ", "Σ")
    if s.endswith("σ"):
        s = s[:-1] + "ς"
    return s


# A lowercase rho with or without a breathing, composed or decomposed. Two
# of them in a row inside a word are a double rho however they are marked:
# ῤῥ, the half-marked ρῥ and ῤρ, and the ῥῥ, ῤῤ, ῥρ and ῥῤ that the OCR of
# the same editions produces. Not at either end of the word: a doubled
# initial ῥ is a misprint, not this spelling, and a marked final rho is an
# elision.
_LOWERCASE_RHO = "(?:\u1fe4|\u1fe5|\u03c1[\u0313\u0314]?)"
_DOUBLE_RHO = re.compile(
    "(?<=.)" + _LOWERCASE_RHO + _LOWERCASE_RHO + "(?=[^\u0313\u0314])")


def to_plain_double_rho(s: str) -> str:
    """Write a double rho printed with breathings (ῤῥ) as plain ρρ.

    Older editions print a medial double rho with a smooth breathing on the
    first rho and a rough one on the second (παῤῥησία, ἄῤῥητος, Βοῤῥᾶς).
    The marks are an editorial convention and carry no lexical distinction:
    the dictionaries and the lookup spell these words with plain ρρ. In the
    form-attestation corpora one double-rho token in six is printed with a
    breathing on either rho (13,450 of 78,989, mostly Migne and
    First1KGreek): ῤῥ in 12,936, and in the rest the pair is half-marked
    or its breathings are confused (ρῥ, ῤρ, ῥῥ, ῤῤ, ῥρ, ῥῤ). Without this
    the word has no row of its own and is read, if at all, through its
    accent-stripped key (ἀπεῤῥάγησαν had none, #141), and a tagged lookup
    reads the Modern Greek table (Βοῤῥᾶ PROPN -> Βορράς, #134).

    A word-initial ῥ and the elision mark on a final rho (γάῤ) are left
    alone: only a rho pair inside the word is rewritten.
    """
    if not any(ch in s for ch in "\u1fe4\u1fe5\u0313\u0314"):
        return s
    return _DOUBLE_RHO.sub("ρρ", s)


def _is_self_map(form: str, lemma: str) -> bool:
    """Check if a lookup entry is a trivial self-map (form ≈ lemma)."""
    return (form == lemma
            or strip_accents(form.lower()) == strip_accents(lemma.lower()))


def _levenshtein(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1,
                            prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


# OCR confusion pairs: (char_a, char_b) -> substitution cost (0.0 to 1.0).
# Normal substitution costs 1.0. OCR-common confusions cost less.
# Built from GCV analysis of LSJ supplement OCR output.
_OCR_CONFUSIONS: dict[tuple[str, str], float] = {}


def _build_ocr_confusions():
    """Build OCR confusion cost matrix (lazy, called once)."""
    if _OCR_CONFUSIONS:
        return

    pairs = [
        # Greek/Latin script mixing (cost ~0, same glyph)
        ("ο", "o", 0.1), ("Ο", "O", 0.1),
        ("ρ", "p", 0.1), ("Ρ", "P", 0.1),
        ("ν", "v", 0.1), ("Ν", "N", 0.1),
        ("τ", "t", 0.1), ("Τ", "T", 0.1),
        ("κ", "k", 0.1), ("Κ", "K", 0.1),
        ("α", "a", 0.1), ("Α", "A", 0.1),
        ("η", "n", 0.2), ("Η", "H", 0.1),
        ("ε", "e", 0.1), ("Ε", "E", 0.1),
        ("ι", "i", 0.1),
        ("υ", "u", 0.1), ("Υ", "Y", 0.1),
        ("χ", "x", 0.1), ("Χ", "X", 0.1),
        ("ω", "w", 0.2),
        ("β", "B", 0.2),
        ("γ", "y", 0.3),  # GCV-specific
        ("δ", "d", 0.3),  # GCV-specific

        # Cyrillic contamination (GCV-specific, same glyph)
        ("ο", "\u043e", 0.0),  # Cyrillic о
        ("α", "\u0430", 0.0),  # Cyrillic а
        ("ε", "\u0435", 0.0),  # Cyrillic е
        ("υ", "\u0443", 0.0),  # Cyrillic у
        ("κ", "\u043a", 0.0),  # Cyrillic к
        ("ρ", "\u0440", 0.0),  # Cyrillic р

        # Common OCR letter confusions (similar shapes)
        ("ο", "σ", 0.5),  # round shapes
        ("θ", "δ", 0.5),  # similar with crossbar
        ("η", "π", 0.5),  # similar verticals
        ("ν", "μ", 0.7),  # similar verticals
        ("ρ", "β", 0.7),  # descender confusion
        ("ι", "ΐ", 0.3),  # diaeresis confusion
        ("υ", "ΰ", 0.3),

        # Number/letter confusions (GCV Roman numeral issue)
        ("1", "I", 0.1),
        ("1", "l", 0.1),

        # GCV descender confusions: J/j for ψ/ὑ
        ("ψ", "J", 0.2), ("ψ", "j", 0.2),
        ("υ", "J", 0.3), ("υ", "j", 0.3),
        ("ὑ", "J", 0.2), ("ὑ", "j", 0.2),
        # GCV: Q/q for θ/σ
        ("θ", "Q", 0.3), ("θ", "q", 0.3),
        ("σ", "Q", 0.3), ("σ", "q", 0.3),
    ]

    for a, b, cost in pairs:
        _OCR_CONFUSIONS[(a, b)] = cost
        _OCR_CONFUSIONS[(b, a)] = cost


def _weighted_levenshtein(a: str, b: str) -> float:
    """Weighted Levenshtein distance using OCR confusion costs.

    Decomposes to NFD first so that combining diacritics are handled
    separately from base characters:
    - Combining diacritics: insert/delete costs 0.1 (nearly free)
    - OCR-confused base character pairs: uses cost matrix (0.1-0.5)
    - Normal substitutions: cost 1.0

    This makes breathing/accent errors nearly free while correctly
    penalizing real letter substitutions.
    """
    _build_ocr_confusions()

    a_nfd = unicodedata.normalize("NFD", a)
    b_nfd = unicodedata.normalize("NFD", b)

    if len(a_nfd) < len(b_nfd):
        return _weighted_levenshtein(b, a)
    if not b_nfd:
        return float(len(a_nfd))

    def _char_cost(c: str) -> float:
        """Cost to insert or delete a character."""
        if unicodedata.combining(c):
            return 0.1  # diacritics are nearly free
        return 1.0

    prev = [0.0]
    for j in range(len(b_nfd)):
        prev.append(prev[-1] + _char_cost(b_nfd[j]))

    for i, ca in enumerate(a_nfd):
        ins_cost_a = _char_cost(ca)
        curr = [prev[0] + ins_cost_a]
        for j, cb in enumerate(b_nfd):
            if ca == cb:
                sub_cost = 0.0
            elif unicodedata.combining(ca) and unicodedata.combining(cb):
                sub_cost = 0.1  # swapping one diacritic for another
            else:
                sub_cost = _OCR_CONFUSIONS.get((ca, cb), 1.0)
            ins_cost_b = _char_cost(cb)
            curr.append(min(
                curr[j] + ins_cost_b,     # insert b[j]
                prev[j + 1] + ins_cost_a,  # delete a[i]
                prev[j] + sub_cost,        # substitute
            ))
        prev = curr
    return prev[-1]


# Greek diphthongs whose second vowel may legitimately carry the breathing
_DIPHTHONGS = {"αι", "ει", "οι", "υι", "αυ", "ευ", "ου", "ηυ", "ωυ"}


def _psili_is_diphthong_breathing(nfd: str, base_idx: int, mark_idx: int) -> bool:
    """True iff the combining psili at ``mark_idx`` is a genuine smooth
    breathing on the second vowel of a word-initial diphthong (οὐ, εἰ, εὖ)
    rather than an elision apostrophe. A real breathing is the FIRST
    combining mark on its vowel and the diphthong opens the word; an
    elision apostrophe instead follows any real diacritics (elided οἷ̓ has
    rough breathing + circumflex before the psili)."""
    if mark_idx != base_idx + 1:
        return False  # psili after other marks = elision apostrophe
    if base_idx != 1:
        return False  # the diphthong must open the word
    return (nfd[base_idx - 1].lower() + nfd[base_idx].lower()) in _DIPHTHONGS


_SMOOTH_BREATHING = "\u0313"
_ROUGH_BREATHING = "\u0314"
# Initial diphthongs that carry the breathing on their second vowel (εὐ, οἱ).
_BREATHING_DIPHTHONGS = frozenset(
    {"αι", "ει", "οι", "υι", "αυ", "ευ", "ηυ", "ου"})


def _initial_breathing(word: str) -> str | None:
    """The breathing of a word that opens with a vowel or rho.

    Returns U+0313 or U+0314, "" when the opening vowel, diphthong or rho
    has none (ευσεβής, ρῆσις), and None for a word that opens otherwise.
    """
    nfd = unicodedata.normalize("NFD", word.lower())
    if not nfd or nfd[0] not in _GREEK_VOWELS + "ρ":
        return None
    bases = 0
    for ch in nfd:
        if unicodedata.combining(ch):
            if ch in (_SMOOTH_BREATHING, _ROUGH_BREATHING):
                return ch
            continue
        bases += 1
        if bases > 2 or (bases == 2 and (nfd[0] == "ρ" or ch not in "ιυ")):
            break
    return ""


def _has_bare_initial_rho(lemma: str | None) -> bool:
    """Whether *lemma* opens with a rho that has no breathing (ρῆσις)."""
    nfd = unicodedata.normalize("NFD", lemma or "")
    return nfd[:1] in ("ρ", "Ρ") and nfd[1:2] not in (
        _SMOOTH_BREATHING, _ROUGH_BREATHING)


def _with_initial_breathing(lemma: str, mark: str) -> str | None:
    """*lemma* with *mark* on its opening vowel or rho.

    The breathing goes on the second vowel of an opening diphthong (εὐ-),
    which needs an unaccented first vowel and no diaeresis on the second.
    """
    nfd = unicodedata.normalize("NFD", lemma)
    bases = [index for index, ch in enumerate(nfd)
             if not unicodedata.combining(ch)]
    if not bases:
        return None
    position = bases[0]
    low = nfd.lower()
    if (len(bases) > 1 and low[0] != "ρ"
            and low[bases[0]] + low[bases[1]] in _BREATHING_DIPHTHONGS
            and "\u0308" not in nfd[bases[1]:bases[2] if len(bases) > 2 else None]
            and not any(ord(ch) in _TONAL_ACCENT_MARKS
                        for ch in nfd[bases[0]:bases[1]])):
        position = bases[1]
    return unicodedata.normalize(
        "NFC", nfd[:position + 1] + mark + nfd[position + 1:])


_TONAL_ACCENT_MARKS = frozenset({0x0301, 0x0300, 0x0342})


def _strip_elision(word: str) -> str | None:
    """Strip trailing elision mark from an elided word form.

    Returns the consonant stem, or None if no elision detected.
    The elision mark in polytonic text is U+0313 (COMBINING COMMA ABOVE)
    attached to the final consonant, e.g. ἀλλ + U+0313 for ἀλλ̓.

    IMPORTANT: U+0313 also serves as smooth breathing at the START of
    polytonic words (ἐ = ε + U+0313). We only treat it as elision when
    it appears after the first base character cluster (i.e. not on the
    initial letter), and - via _psili_is_diphthong_breathing - not when
    it is the genuine breathing on the second vowel of a word-initial
    diphthong (οὐ, εἰ, εὖ). Distinguishing signal: a real breathing is
    the FIRST combining mark on its vowel, while an elision apostrophe
    follows any real diacritics (elided οἷ̓ = ι + rough breathing +
    circumflex + psili).
    """
    nfd = unicodedata.normalize("NFD", word)
    if len(nfd) < 2:
        return None

    # Count base (non-combining) characters
    base_count = sum(1 for ch in nfd if unicodedata.category(ch) != "Mn")

    if base_count <= 1:
        # Single base char (like δ̓, τ̓, γ̓): the U+0313 IS the elision
        # mark, not a breathing. Return the bare consonant as stem.
        if nfd[-1] in _ELISION_MARKS:
            stem = unicodedata.normalize("NFC", nfd[:-1])
            if stem:
                return stem
        return None

    # Multi-char word: U+0313 is only elision when it's on the LAST
    # base character. Anywhere else (initial letter, diphthong like ου)
    # it's a smooth breathing mark.
    #
    # Find the last base character, then check if U+0313 follows it
    # with no more base characters after.
    last_base_idx = -1
    for i in range(len(nfd) - 1, -1, -1):
        if unicodedata.category(nfd[i]) != "Mn":
            last_base_idx = i
            break

    if last_base_idx < 0:
        return None

    # Check combining marks after the last base char for elision mark
    for i in range(last_base_idx + 1, len(nfd)):
        if nfd[i] in _ELISION_MARKS:
            if (nfd[i] == "̓"
                    and _psili_is_diphthong_breathing(nfd, last_base_idx, i)):
                continue
            stem = unicodedata.normalize("NFC", nfd[:i])
            if stem:
                return stem

    # Also check non-combining elision marks (right quote, modifier apostrophe)
    # at the very end of the string
    if nfd[-1] in _ELISION_MARKS and unicodedata.category(nfd[-1]) != "Mn":
        stem = unicodedata.normalize("NFC", nfd[:-1])
        if stem:
            return stem

    return None


# The single canonical elision mark used as the lookup.db key for elided
# forms (U+1FBD GREEK KORONIS). Every apostrophe variant a text might use
# (U+2019, U+02BC, ASCII ', combining U+0313) is folded onto this key so
# the resolved lemma no longer depends on which codepoint the source chose.
_KORONIS = "᾽"

_GREEK_VOWEL_BASES = set("αεηιουωΑΕΗΙΟΥΩ")


def _is_elided_junk_value(lemma: str) -> bool:
    """True iff a lookup VALUE is itself an elided fragment (ends in a
    spacing apostrophe/koronis): ἀλλ᾽, μηδʼ, ἡνίκ'. A real headword never
    ends in an elision mark, so such a value carries no lemma information -
    but it does signal that its KEY is an elided stem, which callers use to
    route the word through the elision machinery instead."""
    return bool(lemma) and lemma[-1] in _ELISION_MARKS \
        and unicodedata.category(lemma[-1]) != "Mn"


def _is_explicit_numeral(word: str) -> bool:
    """True iff ``word`` is structurally an explicitly marked Greek numeral.

    This includes lower-keraia thousands forms (͵α) that do not end in a mark
    but still must be claimed before elision expansion can invent a word.
    """
    return _classify_nonlexical(word) == _NONLEXICAL_NUMERAL


def _is_consonant_psili_elision(word: str) -> bool:
    """True iff ``word`` ends in a combining psili (U+0313) whose base is a
    CONSONANT (γ̓, μ̓, ἄγετ̓). A breathing can only sit on a vowel (or
    initial ρ), so on a final consonant U+0313 is always an elision
    apostrophe - unlike a diphthong carrying a genuine breathing on its
    second vowel (οὐ, εἰ), which must stay eligible for plain lookup."""
    nfd = unicodedata.normalize("NFD", word)
    if not nfd.endswith("̓"):
        return False
    for ch in reversed(nfd[:-1]):
        if not unicodedata.combining(ch):
            return ch not in _GREEK_VOWEL_BASES
    return False


def _is_marked_vowel_psili_elision(word: str) -> bool:
    """True iff ``word`` ends in a combining psili (U+0313) that follows
    another diacritic on its final vowel (αἶ̓, οἷ̓, βαί̓, ὁποῖ̓).

    A breathing is the first mark on its vowel, so a psili typed after a
    breathing or an accent there is an elision apostrophe. A psili that is
    the vowel's only mark (ἐ, αἰ, οὐ) stays a breathing, as in
    ``_strip_elision``.
    """
    nfd = unicodedata.normalize("NFD", word)
    if not nfd.endswith("\u0313") or len(nfd) < 3:
        return False
    marks = 0
    for ch in reversed(nfd[:-1]):
        if unicodedata.combining(ch):
            marks += 1
            continue
        return marks > 0 and ch in _GREEK_VOWEL_BASES
    return False


def _is_bare_vowel_psili_final(word: str) -> bool:
    """True iff ``word`` ends in a vowel whose only mark is a psili (ἐ, οὐ,
    δἐ): a breathing, not the elision apostrophe of
    ``_is_marked_vowel_psili_elision``. lemmatize() reads such a word in the
    lookup before the elision expander, so lemmatize_batch() does too."""
    nfd = unicodedata.normalize("NFD", word)
    return (len(nfd) >= 2 and nfd.endswith("\u0313")
            and nfd[-2] in _GREEK_VOWEL_BASES)


def _canon_elision(word: str) -> str | None:
    """Rewrite a trailing elision mark to the canonical U+1FBD key used by lookup.db.

    Returns ``stem + U+1FBD`` (e.g. ``ὅτ᾽`` for any of ``ὅτ’``/``ὅτʼ``/``ὅτ'``),
    or None if ``word`` carries no elision mark.
    """
    stem = _strip_elision(word)
    return stem + _KORONIS if stem else None


# LSJ9 indeclinable category -> UPOS tag mapping
_INDECL_TO_UPOS = {
    "adverb": "ADV",
    "preposition": "ADP",
    "conjunction": "CCONJ",
    "particle": "PART",
    "interjection": "INTJ",
}

# Subordinating conjunctions get SCONJ instead of CCONJ
_SUBORDINATING = {
    "ὅτι", "ὁτιή", "ἐπεί", "ἐπάν", "ὄφρα", "πρίν", "διότι",
    "μέχριπερ", "ἠΰτε",
}


def _indeclinables_to_pos(raw: dict[str, str]) -> dict[str, dict[str, str]]:
    """Convert {lemma: category} indeclinables dict to POS lookup format.

    Returns {form: {UPOS: lemma}} where form == lemma (indeclinable).
    For conjunctions, subordinating forms get SCONJ; others get CCONJ.
    """
    result: dict[str, dict[str, str]] = {}
    for lemma, category in raw.items():
        upos = _INDECL_TO_UPOS.get(category)
        if not upos:
            continue
        if category == "conjunction" and lemma in _SUBORDINATING:
            upos = "SCONJ"
        result[lemma] = {upos: lemma}
    return result


class LookupDB:
    """Dict-like wrapper around SQLite lookup table.

    Provides .get(key) that queries SQLite instead of loading the full
    12.5M-entry dict into memory. Supports lazy bulk-load into a dict
    for batch operations.

    The DB has two tables:
      - lemmas(id, text): deduplicated lemma strings (~700K)
      - lookup(form, lemma_id, lang): form->lemma mappings (~12.5M)
        lang='all' for combined, lang='grc' for AG-only overrides,
        lang='el' for MG-only overrides
    """

    def __init__(self, db_path, lang='all', reviewed_rows=None):
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
        self._lang = lang
        # Reviewed rows that answer before the table (dilemma/grc_view_rows.py)
        self._reviewed_rows = reviewed_rows or {}
        self._dict = None  # lazy-loaded for batch ops
        self._cache = None  # per-query result cache (enabled via enable_cache)
        self._query = (
            "SELECT l.text FROM lookup k JOIN lemmas l ON k.lemma_id = l.id "
            "WHERE k.form = ? AND k.lang = ? LIMIT 1"
        )
        # Separate spell index (stripped -> form mapping)
        self._spell_conn = None
        if SPELL_INDEX_PATH.exists():
            self._spell_conn = sqlite3.connect(str(SPELL_INDEX_PATH))
            self._spell_conn.execute("PRAGMA mmap_size=134217728")  # 128MB mmap

    def get(self, key, default=None):
        """Dict-compatible .get() backed by SQLite."""
        if key in self._reviewed_rows:
            return self._reviewed_rows[key]
        if self._dict is not None:
            return self._dict.get(key, default)
        # Check per-query cache before hitting SQLite
        if self._cache is not None:
            _sentinel = object()
            cached = self._cache.get(key, _sentinel)
            if cached is not _sentinel:
                return cached if cached is not None else default
        row = self._conn.execute(self._query, (key, self._lang)).fetchone()
        if row:
            if self._cache is not None:
                self._cache[key] = row[0]
            return row[0]
        # AG-only table ('grc') falls through to combined ('all') for entries
        # where AG agrees with combined (not stored separately to save space).
        # Same logic for MG-only ('el').
        if self._lang in ('grc', 'el'):
            row = self._conn.execute(self._query, (key, 'all')).fetchone()
            result = row[0] if row else None
            if self._cache is not None:
                self._cache[key] = result
            return result if result is not None else default
        if self._cache is not None:
            self._cache[key] = None
        return default

    def __contains__(self, key):
        if self._dict is not None:
            return key in self._dict
        if self._cache is not None:
            _sentinel = object()
            cached = self._cache.get(key, _sentinel)
            if cached is not _sentinel:
                return cached is not None
        # Delegate to .get() which handles caching
        return self.get(key) is not None

    def __getitem__(self, key):
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def __bool__(self):
        return True

    def items(self):
        """Iterate all entries (loads nothing extra into memory)."""
        cursor = self._conn.execute(
            "SELECT k.form, l.text FROM lookup k JOIN lemmas l ON k.lemma_id = l.id "
            "WHERE k.lang = ?", (self._lang,))
        return cursor

    def __iter__(self):
        cursor = self._conn.execute(
            "SELECT form FROM lookup WHERE lang = ?", (self._lang,))
        for row in cursor:
            yield row[0]

    def __len__(self):
        row = self._conn.execute(
            "SELECT COUNT(*) FROM lookup WHERE lang = ?", (self._lang,)).fetchone()
        return row[0]

    def bulk_load(self):
        """Load entire table into a dict for fast batch operations."""
        if self._dict is not None:
            return
        self._dict = {}
        for form, lemma in self.items():
            self._dict[form] = lemma
        self._dict.update(self._reviewed_rows)

    def enable_cache(self):
        """Enable per-query result caching for repeated lookups.

        Unlike bulk_load() which loads the entire table into memory,
        this only caches results for forms that are actually queried.
        Useful for large tables (e.g. lang='all' with 12M+ entries)
        where bulk_load would use too much memory but the working set
        of queried forms is much smaller.
        """
        if self._dict is not None:
            return  # already fully loaded, cache not needed
        if self._cache is None:
            self._cache = {}
            # Boost SQLite page cache for the initial cold lookups
            self._conn.execute("PRAGMA cache_size=-65536")  # 64MB

    @staticmethod
    def _parse_spell_forms(forms_blob: str,
                           src_filter: str | None = None) -> list[str]:
        """Parse the compact forms blob from spell_index.db.

        Each line is "form\\tsrc" or just "form". Returns the form
        strings, optionally filtered by src.
        """
        results = []
        for line in forms_blob.split("\n"):
            if "\t" in line:
                form, src = line.split("\t", 1)
                if src_filter and src != src_filter:
                    continue
            else:
                form = line
            results.append(form)
        return results

    def spell_lookup_stripped(self, candidates: set[str],
                             src_filter: str = None
                             ) -> dict[str, list[str]]:
        """Look up stripped forms, return {stripped: [original_forms]}.

        Queries the separate spell_index.db for fast batch lookup.
        Falls back to the main lookup table if spell_index.db is
        not available (legacy DBs with a stripped column).

        Args:
            candidates: Set of accent-stripped forms to look up.
            src_filter: If set, only return forms from this source
                (e.g., 'grc' for AG-sourced forms only).
        """
        if not candidates:
            return {}
        result: dict[str, list[str]] = {}
        candidate_list = list(candidates)

        if self._spell_conn is not None:
            for i in range(0, len(candidate_list), 900):
                batch = candidate_list[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                query = (
                    f"SELECT stripped, forms FROM spell "
                    f"WHERE stripped IN ({placeholders})"
                )
                for stripped, forms_blob in self._spell_conn.execute(
                        query, batch):
                    forms = self._parse_spell_forms(forms_blob, src_filter)
                    if forms:
                        result[stripped] = forms
            return result

        # Legacy: stripped column in main lookup table
        for i in range(0, len(candidate_list), 900):
            batch = candidate_list[i:i + 900]
            placeholders = ",".join("?" * len(batch))
            if src_filter:
                query = (
                    f"SELECT DISTINCT stripped, form FROM lookup "
                    f"WHERE stripped IN ({placeholders}) AND src = ?"
                )
                batch = batch + [src_filter]
            elif self._lang in ('grc', 'el'):
                query = (
                    f"SELECT DISTINCT stripped, form FROM lookup "
                    f"WHERE stripped IN ({placeholders}) "
                    f"AND lang IN ('{self._lang}', 'all')"
                )
            else:
                query = (
                    f"SELECT DISTINCT stripped, form FROM lookup "
                    f"WHERE stripped IN ({placeholders}) AND lang = ?"
                )
                batch = batch + [self._lang]
            for stripped, form in self._conn.execute(query, batch):
                if stripped not in result:
                    result[stripped] = []
                result[stripped].append(form)
        return result

    def has_stripped(self, stripped: str) -> bool:
        """Check if a stripped form exists."""
        if self._spell_conn is not None:
            row = self._spell_conn.execute(
                "SELECT 1 FROM spell WHERE stripped = ? LIMIT 1",
                (stripped,)).fetchone()
        elif self._lang in ('grc', 'el'):
            row = self._conn.execute(
                "SELECT 1 FROM lookup WHERE stripped = ? AND lang IN (?, 'all') LIMIT 1",
                (stripped, self._lang)).fetchone()
        else:
            row = self._conn.execute(
                "SELECT 1 FROM lookup WHERE stripped = ? AND lang = ? LIMIT 1",
                (stripped, self._lang)).fetchone()
        return row is not None

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


# Numeral letters for 1-9, as an ordinal printed with a superscript ending
# flattens to them: Η with ον for ὄγδοον. Higher letters stay out because the
# same shape spells common words (Τον, Σον, Ιον).
_ORDINAL_NUMERAL_LETTERS = frozenset("ΑΒΓΔΕϚϛΖΗΘ")


def is_numeral_ordinal(word: str) -> bool:
    """Whether *word* is a numeral letter for 1-9 followed by ον, as printed."""
    word = unicodedata.normalize("NFC", word or "")
    return (len(word) == 3 and word.endswith("ον")
            and word[0] in _ORDINAL_NUMERAL_LETTERS)


class Dilemma:
    def __init__(self, lang="all", device=None, scale=None,
                 resolve_articles=False, normalize=False, period=None,
                 dialect=None, convention=None, skip_pos=False,
                 citation_policy: str = "default",
                 skip_numeral_ordinals: bool = False):
        """Initialize Dilemma.

        Args:
            lang: "all" (default) for MG+AG+Medieval combined,
                  "el" for MG only, "grc" for AG only
            device: "cpu", "cuda", etc. Auto-detected if None.
            scale: Model scale (0-4). None auto-detects the best available.
                   Larger scales = more training data = better generalization
                   on unseen forms. Lookup table is the same for all scales.
            resolve_articles: if True, resolve article forms (τῆς, τόν,
                  etc.) to the canonical lemma ὁ, and pronoun clitics
                  (μοι, σοι) to their pronoun lemma (ἐγώ, σύ). Default
                  False, which keeps articles/pronouns as self-mappings
                  (better for alignment where you want surface-form
                  matching). Set True for evaluation against treebanks
                  like DiGreC/AGDT which use ὁ as the article lemma.
            normalize: if True, enable orthographic normalization for
                  Byzantine/papyrological texts. Generates candidate
                  spellings (fixing itacism, missing iota subscripta,
                  etc.) and checks them against the lookup table.
                  Also enabled implicitly when dialect is set.
            period: Historical period for normalization rule weights.
                  One of: "hellenistic", "late_antique", "byzantine",
                  "all" (default). Only used when normalize=True.
            dialect: Ancient Greek dialect normalization. Maps dialect
                  forms to Attic equivalents for lookup. One of:
                  "ionic" (Herodotus, Hippocrates, etc.),
                  "doric" (Pindar, Theocritus, etc.),
                  "aeolic" (Sappho, Alcaeus, etc.),
                  "koine" (NT, papyri - overlaps with period rules),
                  "auto" (try all dialect rules),
                  None (default, no dialect normalization).
                  Setting a dialect implicitly enables normalization.
            convention: Lemma convention for output remapping. Controls
                  which citation form is returned when multiple conventions
                  exist for the same word (e.g. LSJ vs Wiktionary headwords).
                  One of: None (default, no remapping), "wiktionary" (same
                  as None), "lsj" (remap to LSJ dictionary headwords),
                  "cunliffe" (remap to Cunliffe Homeric Lexicon headwords),
                  "triantafyllidis" (remap to Modern Greek monotonic
                  dictionary forms, using MG Wiktionary headwords as
                  reference).
            citation_policy: "default" preserves Dilemma's broad lookup/model
                  coverage while rejecting unproven grave-accent citation
                  lemmas. "strict_ag" additionally requires Ancient/Byzantine
                  Greek citation outputs to be independently headword-backed;
                  use it for dictionary/headword audits where no lemma is
                  better than an unverified guessed citation form.
            skip_numeral_ordinals: If True, return no lemma for a numeral
                  letter for 1-9 followed by ον, as printed (Ηον, Γον, ϛον):
                  an ordinal whose superscript ending was flattened. Such a
                  token otherwise reaches word rows (Ηον gives εἶμι through
                  ηον). Βον is covered too, although it has a row of its own.
                  Τον, Σον and Ιον spell common words and are not.
        """
        if convention not in _VALID_CONVENTIONS:
            raise ValueError(
                f"Unknown convention {convention!r}. "
                f"Valid values: {sorted(c for c in _VALID_CONVENTIONS if c)}, or None."
            )
        if citation_policy not in _VALID_CITATION_POLICIES:
            raise ValueError(
                f"Unknown citation_policy {citation_policy!r}. "
                f"Valid values: {sorted(_VALID_CITATION_POLICIES)}."
            )
        if lang == "both":
            lang = "all"
        if lang not in ("all", "el", "grc"):
            raise ValueError(
                f"Unknown language {lang!r}. Valid values: 'all', 'el', 'grc'."
            )
        self.lang = lang
        self._skip_numeral_ordinals = skip_numeral_ordinals
        self._citation_policy = citation_policy
        self._scale = scale
        # Triantafyllidis convention always needs article resolution:
        # MG text lemmatizes articles to ο, pronouns to αυτός, etc.
        if convention == "triantafyllidis" and not resolve_articles:
            resolve_articles = True
        self._resolve_articles = resolve_articles
        self._convention_name = convention
        self._model = None
        self._backend_checked = False
        self._vocab = None
        self._device = device
        self._normalizer = None
        # Setting a dialect implicitly enables normalization
        if normalize or dialect is not None:
            from .normalize import Normalizer
            self._normalizer = Normalizer(period=period, dialect=dialect)

        # Lookup tables: SQLite-backed (instant startup) or dict (JSON fallback)
        self._mg_lookup = {}
        self._med_lookup = {}
        self._ag_lookup = {}
        self._lookup = {}
        self._using_db = False

        # POS-indexed disambiguation table: {form: {upos: lemma}}
        self._pos_lookup: dict[str, dict[str, str]] = {}
        # AG-only POS lookup for polytonic-first disambiguation
        self._pos_ag_lookup: dict[str, dict[str, str]] = {}

        # Headword frequency table (lazy-loaded from lsj9_frequency.json)
        self._hw_frequency: dict[str, int] | None = None

        # Headword sets by convention name (lazy-loaded)
        self._headword_sets: dict[str, set[str]] = {}

        # Per-lemma corpus attestation profiles (lazy-loaded)
        self._attestation: dict[str, dict] | None = None

        # Canonical AG headword inventory for repair_lemma():
        # (all_headwords, stripped_index, lsj_only). Lazy-loaded.
        self._canonical_hw: tuple[set, dict, set] | None = None
        # Accent-stripped skeleton -> headword spellings, one index per
        # dictionary, for accenting compound reconstructions. Lazy-loaded.
        self._compound_headword_indexes: dict[str, dict[str, list[str]]] = {}
        self._trusted_grave_targets: set[str] | None = None
        self._elision_lexical_heads: frozenset[str] | None = None
        self._ancient_lexical_heads: frozenset[str] | None = None
        self._vernacular_lexical_heads: frozenset[str] | None = None
        self._fallback_readings = None
        self._fallback_exact_readings = None
        self._fallback_name_lemmas = frozenset()
        self._fallback_verb_lemmas = None
        self._ancient_pronoun_readings = None
        self._pos_rejected_targets = None

        # Form-keyed corpus attestation (form_profile.db / form_citations.db),
        # for the "attested only" gate + form_attestation(). Lazy-opened.
        self._attest = None

        self._load_lookups(skip_pos=skip_pos)
        self._convention_map = self._build_convention_map(convention)
        self._convention_monotonic = convention in _MONOTONIC_CONVENTIONS
        _warn_missing_data_files(lang, convention)

    def _check_backend(self):
        """Check native dependencies only when model inference is requested."""
        model_dir = self._find_model_dir()
        has_onnx_files = (model_dir / "encoder.onnx").exists()
        has_pt_file = (model_dir / "model.pt").exists()
        if has_onnx_files:
            try:
                import onnxruntime  # noqa: F401
                return
            except ImportError:
                pass
        if has_pt_file:
            try:
                import torch  # noqa: F401
                return
            except ImportError:
                pass
        import warnings
        warnings.warn(
            "No model backend available. Install onnxruntime (~50 MB) or "
            "torch (~2 GB) for unseen-word inference. Lookup table still "
            "works, but unknown forms will return unchanged. "
            "pip install onnxruntime",
            stacklevel=3,
        )

    def preload(self):
        """Optimize lookup tables for batch processing.

        Call this before processing a large corpus to avoid per-query
        SQLite overhead. Enables per-query caching on all SQLite-backed
        lookup tables so repeated lookups for the same form skip SQLite
        entirely. The Iliad's working set (~20K unique forms) stays
        small even though the full table has 12M+ entries.

        Safe to call multiple times (idempotent). Does not affect the
        public API - all existing methods work the same, just faster.
        """
        if not self._using_db:
            return  # JSON fallback already uses dicts
        for table in [self._lookup, self._ag_lookup,
                      self._mg_lookup, self._med_lookup]:
            if isinstance(table, LookupDB):
                table.enable_cache()

    def export_cache(self) -> dict[str, str]:
        """Export the full lookup table as a {form: lemma} dict.

        Useful for passing to downstream tools (e.g. Tagger's lemma_cache
        parameter) so they can skip Dilemma calls for known forms.
        Convention remapping is applied if a convention is set.
        """
        cache: dict[str, str] = {}
        if self._using_db:
            for form, lemma in self._lookup.items():
                out = self._apply_convention(lemma)
                if out:
                    cache[form] = out
        else:
            for form, lemma in self._lookup.items():
                out = self._apply_convention(lemma)
                if out:
                    cache[form] = out
        return cache

    def _load_lookups(self, skip_pos=False):
        """Load lookup tables from SQLite (instant) or JSON (fallback).

        SQLite path (lookup.db): pre-merged combined table with AG priority,
        plus AG-only overrides for polytonic disambiguation. Near-instant
        startup, 0.05ms/query, supports lazy bulk_load() for batch ops.

        JSON fallback: loads all three JSON files and merges at init (~11s).
        """
        if LOOKUP_DB_PATH.exists():
            # SQLite: instant startup for all language modes
            if self.lang == "grc":
                self._lookup = LookupDB(LOOKUP_DB_PATH, lang='grc',
                                        reviewed_rows=_GRC_VIEW_ROWS)
                self._ag_lookup = self._lookup
            elif self.lang == "el":
                self._lookup = LookupDB(LOOKUP_DB_PATH, lang='all')
                self._ag_lookup = LookupDB(LOOKUP_DB_PATH, lang='grc')
                self._mg_lookup = LookupDB(LOOKUP_DB_PATH, lang='el')
            else:
                self._lookup = LookupDB(LOOKUP_DB_PATH, lang='all')
                self._ag_lookup = LookupDB(LOOKUP_DB_PATH, lang='grc')
                # Load MG lookup for triantafyllidis convention: MG entries
                # should take priority over AG for monotonic MG forms.
                if self._convention_name == "triantafyllidis":
                    self._mg_lookup = LookupDB(LOOKUP_DB_PATH, lang='el')
            self._using_db = True
        elif LOOKUP_PATH.exists() or AG_LOOKUP_PATH.exists():
            # JSON fallback (dev mode before lookup.db is built)
            self._load_lookups_json()
        else:
            raise FileNotFoundError(
                f"Dilemma data not found at {DATA_DIR}. "
                "Download with: python -m dilemma download  "
                "(or set DILEMMA_DATA_DIR to point at an existing copy)."
            )

        if not skip_pos:
            self._load_pos_lookups()

    def _load_lookups_json(self):
        """Fallback: load JSON lookup tables and merge (~11s)."""
        if LOOKUP_PATH.exists():
            with open(LOOKUP_PATH, encoding="utf-8") as f:
                self._mg_lookup = json.load(f)
        if MED_LOOKUP_PATH.exists():
            with open(MED_LOOKUP_PATH, encoding="utf-8") as f:
                self._med_lookup = json.load(f)
        if AG_LOOKUP_PATH.exists():
            with open(AG_LOOKUP_PATH, encoding="utf-8") as f:
                self._ag_lookup = json.load(f)

        def _is_self_map(form, lemma):
            return (form == lemma
                    or strip_accents(form.lower()) == strip_accents(lemma.lower()))

        if self.lang == "all":
            for data in [self._ag_lookup, self._med_lookup, self._mg_lookup]:
                for k, v in data.items():
                    if k not in self._lookup:
                        self._lookup[k] = v
                    elif _is_self_map(k, self._lookup[k]) and not _is_self_map(k, v):
                        self._lookup[k] = v
                    elif (_is_self_map(k, self._lookup[k])
                          and _is_self_map(k, v) and v == k
                          and self._lookup[k] != k):
                        self._lookup[k] = v
        elif self.lang == "el":
            for data in [self._mg_lookup, self._med_lookup]:
                for k, v in data.items():
                    if k not in self._lookup:
                        self._lookup[k] = v
                    elif _is_self_map(k, self._lookup[k]) and not _is_self_map(k, v):
                        self._lookup[k] = v
        elif self.lang == "grc":
            self._ag_lookup = {**self._ag_lookup, **_GRC_VIEW_ROWS}
            self._lookup = dict(self._ag_lookup)

    def _load_pos_lookups(self, *, include_treebank=True, data_dir=None):
        """Load POS disambiguation tables from JSON.

        Builds two tables:
        - _pos_ag_lookup: AG-only sources (treebank, GLAUx, AG Wiktionary, LSJ9)
        - _pos_lookup: combined (AG sources + MG Wiktionary)

        For polytonic input (breathing marks, circumflex), lemmatize_pos() and
        lemmatize_batch_pos() check _pos_ag_lookup first before the combined
        table, mirroring the AG-first logic in the main lookup.

        Builders may omit the treebank overlay and read another data directory
        to compare corpus votes with this same lower-priority lookup cascade.

        Priority within each table: treebank (gold) > LSJ9 indeclinables
        (unambiguous POS) > GLAUx (corpus) > MG Wiktionary (combined only) >
        AG Wiktionary > LSJ9 grammar.
        """
        def _path(default):
            return Path(data_dir) / default.name if data_dir is not None else default

        corrections = _path(POS_CELL_CORRECTIONS_PATH)

        def _add_to(target, source_data, overwrite=False):
            """Merge source_data into target POS dict.

            A cell whose lemma is a reviewed non-lemma (Ταύτη, Τοῦτος, Ἡμων;
            dilemma/grc_rejected_lemmas.py) is skipped, so tables built
            before the record cannot return it.
            """
            for form, upos_lemmas in source_data.items():
                kept = {upos: lemma for upos, lemma in upos_lemmas.items()
                        if lemma not in _REJECTED_POS_LEMMAS}
                if not kept and upos_lemmas:
                    continue
                cells = target.setdefault(form, {})
                for upos, lemma in kept.items():
                    if overwrite or upos not in cells:
                        cells[upos] = lemma

        # 1. Treebank POS lookup (gold-annotated, highest priority for AG)
        if (include_treebank and self.lang in ("all", "grc")
                and _path(TREEBANK_POS_LOOKUP_PATH).exists()):
            with open(_path(TREEBANK_POS_LOOKUP_PATH), encoding="utf-8") as f:
                tb_pos = json.load(f)
            _apply_pos_cell_corrections("treebank", tb_pos, corrections)
            _add_to(self._pos_ag_lookup, tb_pos, overwrite=True)
            _add_to(self._pos_lookup, tb_pos, overwrite=True)

        # 1b. LSJ9 indeclinables (adverbs, prepositions, conjunctions,
        #     particles, interjections) - POS is unambiguous for these
        if self.lang in ("all", "grc") and _path(LSJ9_INDECLINABLES_PATH).exists():
            with open(_path(LSJ9_INDECLINABLES_PATH), encoding="utf-8") as f:
                indecl_raw = json.load(f)
            indecl_pos = _indeclinables_to_pos(indecl_raw)
            _add_to(self._pos_ag_lookup, indecl_pos)
            _add_to(self._pos_lookup, indecl_pos)

        # 2. GLAUx POS lookup (corpus-derived, 8.7K entries)
        if self.lang in ("all", "grc") and _path(GLAUX_POS_LOOKUP_PATH).exists():
            with open(_path(GLAUX_POS_LOOKUP_PATH), encoding="utf-8") as f:
                glaux_pos = json.load(f)
            _apply_pos_cell_corrections("glaux", glaux_pos, corrections)
            _add_to(self._pos_ag_lookup, glaux_pos)
            _add_to(self._pos_lookup, glaux_pos)

        # 3. MG POS lookup (Wiktionary-derived, combined table only)
        if self.lang in ("all", "el") and _path(MG_POS_LOOKUP_PATH).exists():
            with open(_path(MG_POS_LOOKUP_PATH), encoding="utf-8") as f:
                mg_pos = json.load(f)
            _apply_pos_cell_corrections("mg", mg_pos, corrections)
            _add_to(self._pos_lookup, mg_pos)  # combined only, not AG

        # 4. AG Wiktionary POS lookup (fills remaining gaps)
        if self.lang in ("all", "grc") and _path(AG_POS_LOOKUP_PATH).exists():
            with open(_path(AG_POS_LOOKUP_PATH), encoding="utf-8") as f:
                ag_pos = json.load(f)
            _apply_pos_cell_corrections("ag", ag_pos, corrections)
            _add_to(self._pos_ag_lookup, ag_pos)
            _add_to(self._pos_lookup, ag_pos)

        # 5. LSJ9 grammar-derived POS (407K forms with NOUN/ADJ from
        #    the grammar field: ὁ/ἡ/τό -> NOUN, ον/ές -> ADJ)
        if self.lang in ("all", "grc") and _path(LSJ9_POS_LOOKUP_PATH).exists():
            with open(_path(LSJ9_POS_LOOKUP_PATH), encoding="utf-8") as f:
                lsj9_pos = json.load(f)
            _apply_pos_cell_corrections("lsj9", lsj9_pos, corrections)
            _add_to(self._pos_ag_lookup, lsj9_pos)
            _add_to(self._pos_lookup, lsj9_pos)

    def _build_convention_map(self, convention: str | None) -> dict[str, str]:
        """Build a lemma remapping dict for the given convention.

        For "lsj"/"cunliffe", each equivalence group is resolved to the
        first member that appears in the convention's headword list. All
        other members map to it.

        For "triantafyllidis", the headword set combines the dictionary's
        bare lemma column with Modern Greek Wiktionary coverage. Members are
        also checked after to_monotonic() conversion. The
        _convention_monotonic flag ensures all output is converted to
        monotonic in _apply_convention().

        For None or "wiktionary", returns an empty dict (no remapping).

        After auto-deriving from equivalences, explicit overrides from
        data/convention_{name}.json are applied (if the file exists).
        The override file format is: {"mappings": {"from_lemma": "to_lemma"}}.
        """
        if convention is None or convention == "wiktionary":
            return {}

        remap = {}

        headwords = _load_convention_headwords(convention)
        if convention == "triantafyllidis" and not headwords:
            # Fallback: derive from MG lookup JSON if no headword file.
            if LOOKUP_PATH.exists():
                with open(LOOKUP_PATH, encoding="utf-8") as f:
                    headwords = set(json.load(f).values())

        # Auto-derive from equivalence groups
        if LEMMA_EQUIVALENCES_PATH.exists():
            with open(LEMMA_EQUIVALENCES_PATH, encoding="utf-8") as f:
                equiv_data = json.load(f)

            is_monotonic = convention in _MONOTONIC_CONVENTIONS
            # An Ancient regular adverb is cited under its adjective (#117),
            # so it never becomes a group's canonical member: otherwise
            # ἄδικος would map back to ἀδίκως where neither is a headword.
            adverbs = ({} if is_monotonic or self.lang not in ("all", "grc")
                       else _grc_adverb_data()[0])

            for group in equiv_data.get("groups", []):
                if len(group) < 2:
                    continue
                if adverbs and any(member in adverbs for member in group):
                    group = ([m for m in group if m not in adverbs]
                             + [m for m in group if m in adverbs])

                # Find the canonical for this convention: first member
                # in the headword list. Fall back to the first member
                # of the group if none match.
                canonical = group[0]
                found = False
                for member in group:
                    if member in headwords and member not in adverbs:
                        canonical = member
                        found = True
                        break

                # For monotonic conventions, also try to_monotonic()
                # on each member and check against the headword set.
                if not found and is_monotonic:
                    for member in group:
                        mono = to_monotonic(member)
                        if mono in headwords:
                            canonical = mono
                            found = True
                            break

                for member in group:
                    if member != canonical:
                        remap[member] = canonical

        # For monotonic conventions, add systematic morphological remappings
        # from AG lemma forms to their MG equivalents. These cover productive
        # patterns like neuter -ον -> -ο, abstract -σις -> -ση, and
        # contracted verbs -έω -> -ώ / -άω -> -ώ.
        if convention in _MONOTONIC_CONVENTIONS and headwords:
            # Load AG lemmas from pre-extracted headword file (fast) or
            # fall back to loading the full AG lookup JSON (slow).
            ag_lemmas = set()
            if AG_HEADWORDS_PATH.exists():
                with open(AG_HEADWORDS_PATH, encoding="utf-8") as f:
                    ag_lemmas = set(json.load(f))
            elif AG_LOOKUP_PATH.exists():
                with open(AG_LOOKUP_PATH, encoding="utf-8") as f:
                    ag_lemmas = set(json.load(f).values())
            patterns = [
                # (AG suffix, replacement, description)
                ("ον", "ο", "neuter -ον -> -ο"),
                ("σις", "ση", "abstract -σις -> -ση"),
                ("ξις", "ξη", "abstract -ξις -> -ξη"),
                ("ψις", "ψη", "abstract -ψις -> -ψη"),
                ("έω", "ώ", "contracted -έω -> -ώ"),
                ("άω", "ώ", "contracted -άω -> -ώ"),
                ("όω", "ώ", "contracted -όω -> -ώ"),
            ]
            for al in ag_lemmas:
                if al in remap:
                    continue  # already mapped by equivalence group
                mono = to_monotonic(al)
                for suffix, replacement, _desc in patterns:
                    if mono.endswith(suffix):
                        candidate = mono[:-len(suffix)] + replacement
                        if candidate in headwords and candidate != mono:
                            remap[al] = candidate
                            break

        # Apply explicit overrides from convention file
        override_path = CONVENTION_DIR / f"convention_{convention}.json"
        if override_path.exists():
            with open(override_path, encoding="utf-8") as f:
                overrides = json.load(f)
            for from_lemma, to_lemma in overrides.get("mappings", {}).items():
                remap[from_lemma] = to_lemma

        return remap

    def _trusted_grave_citation_targets(self) -> set[str]:
        """Headwords allowed as targets for grave-to-acute citation repair.

        The lookup table itself is not a trustworthy authority here: its lemma
        values are the source of the contamination. This uses independent
        lexicon headword lists, and also indexes macron/breve-stripped spellings
        so LSJ quantity marks do not block a plain Dilemma lemma spelling.
        """
        if self._trusted_grave_targets is not None:
            return self._trusted_grave_targets

        self._trusted_grave_targets = trusted_ag_citation_headwords()
        return self._trusted_grave_targets

    def _strict_ag_citation_applies(self, lang: str = "",
                                    source: str = "") -> bool:
        """Whether this candidate must be independently AG-headword-backed."""
        if self._citation_policy != "strict_ag":
            return False
        if source == "nonlexical":
            return False
        if self._convention_name == "triantafyllidis":
            return False
        if self.lang == "el" or lang == "el":
            return False
        return self.lang in ("all", "grc")

    def _check_citation_lemma(self, lemma: str | None, *,
                              lang: str = "", source: str = "",
                              strict: bool | None = None
                              ) -> _CitationLemmaCheck:
        """Validate/normalize a candidate citation lemma.

        The default policy blocks citation-form artifacts that are never
        dictionary headwords: stranded leading combining marks, overline
        abbreviation marks, and grave-accented citation forms unless the
        corresponding acute spelling is independently attested as a headword.
        The strict AG policy also requires Greek citation lemmas to be backed
        by the same independent AG/Byzantine headword inventory.
        """
        if not lemma:
            return _CitationLemmaCheck(None, "empty")

        original = unicodedata.normalize("NFC", lemma)
        checked = original
        reason = "unchanged"

        if source == "nonlexical":
            return _CitationLemmaCheck(checked, "nonlexical")

        if _starts_with_combining_mark(checked):
            return _CitationLemmaCheck(
                None, "leading_combining", normalized_from=original)

        if _has_overline_mark(checked):
            return _CitationLemmaCheck(
                None, "overline", normalized_from=original)

        tonal_reason = _malformed_tonal_reason(checked)
        if tonal_reason:
            return _CitationLemmaCheck(
                None, tonal_reason, normalized_from=original)

        final_mark_reason = _citation_final_mark_reason(checked)
        if final_mark_reason:
            return _CitationLemmaCheck(
                None, final_mark_reason, normalized_from=original)

        if "\u0300" in unicodedata.normalize("NFD", checked):
            acute = grave_to_acute(checked)
            if acute != checked and acute in self._trusted_grave_citation_targets():
                checked = acute
                reason = "grave_to_trusted_headword"
            else:
                return _CitationLemmaCheck(
                    None, "untrusted_grave", normalized_from=original)

        if strict is None:
            strict = self._strict_ag_citation_applies(lang, source)
        if not strict:
            return _CitationLemmaCheck(
                checked, reason,
                normalized_from=original if checked != original else "")

        # Non-Greek passthrough is not an AG citation lemma, so strict AG
        # headword validation does not apply.
        has_greek = any(0x0370 <= ord(c) <= 0x03ff
                        or 0x1f00 <= ord(c) <= 0x1fff for c in checked)
        if not has_greek:
            return _CitationLemmaCheck(
                checked, "non_greek",
                normalized_from=original if checked != original else "")

        if self.lang in ("all", "el") and self._is_mg_citation_form(checked):
            return _CitationLemmaCheck(
                checked, "modern_greek",
                normalized_from=original if checked != original else "")

        if checked in self._trusted_grave_citation_targets():
            strict_reason = (reason if reason != "unchanged"
                             else "trusted_headword")
            return _CitationLemmaCheck(
                checked, strict_reason,
                normalized_from=original if checked != original else "")

        return _CitationLemmaCheck(
            None, "untrusted_headword", normalized_from=original)

    def _normalize_grave_citation_lemma(self, lemma: str) -> str | None:
        """Return a citation-safe lemma, or None for an unproven grave value.

        A grave accent is positional in running text and should not survive in a
        dictionary citation form. We only rewrite it to an acute when that exact
        acute spelling is a headword in an independent lexicon inventory.
        Otherwise the candidate is treated as a bad lookup/model value.
        """
        return self._check_citation_lemma(lemma, strict=False).lemma

    def _apply_convention_check(self, lemma: str | None, *,
                                lang: str = "", source: str = ""
                                ) -> _CitationLemmaCheck:
        """Apply convention remapping and citation-form validation."""
        check = self._check_citation_lemma(
            lemma, lang=lang, source=source, strict=False)
        if check.lemma is None:
            return check

        lemma = check.lemma
        adjective = self._grc_adverb_head(lemma)
        if adjective is not None:
            # A regular adverb in -ως is cited under its adjective in Ancient
            # Greek (καλῶς -> καλός, issue #117). The adjective goes through
            # the same convention and citation checks; the adverb stands when
            # the adjective fails them.
            converted = self._apply_convention_check(
                adjective, lang=lang, source=source)
            if converted.lemma is not None:
                return _CitationLemmaCheck(
                    converted.lemma,
                    check.reason if check.reason != "unchanged" else "convention",
                    normalized_from=check.normalized_from or lemma)
        convention_changed = False
        before = lemma
        if self._convention_map:
            lemma = self._convention_map.get(lemma, lemma)
        if self._convention_monotonic:
            lemma = to_monotonic(lemma)
        convention_changed = lemma != before

        final = self._check_citation_lemma(lemma, lang=lang, source=source)
        if final.lemma is None:
            return final
        if check.reason != "unchanged" and final.reason in (
                "unchanged", "trusted_headword"):
            return _CitationLemmaCheck(
                final.lemma, check.reason,
                normalized_from=check.normalized_from)
        if convention_changed and final.reason in ("unchanged",
                                                   "trusted_headword"):
            return _CitationLemmaCheck(
                final.lemma, "convention",
                normalized_from=check.normalized_from or before)
        return final

    def _apply_convention(self, lemma: str | None, *,
                          lang: str = "", source: str = "") -> str | None:
        """Remap a lemma according to the active convention.

        For monotonic conventions (e.g. triantafyllidis), the result is
        converted to monotonic Greek after any explicit remapping, so
        polytonic lemmas like ὁ become ο automatically.

        In Ancient Greek (lang "all" or "grc") under every polytonic
        convention, a regular adverb in -ως is cited under its adjective
        (καλῶς -> καλός, ταχέως -> ταχύς), as the annotated corpora and
        LSJ file it; see ``_grc_adverb_head``.
        """
        return self._apply_convention_check(
            lemma, lang=lang, source=source).lemma

    def citation_status(self, lemma: str, *, lang: str = "",
                        source: str = "") -> dict[str, str | bool | None]:
        """Explain whether a lemma is safe to emit as a citation form.

        The returned ``normalized`` value is the lemma Dilemma would emit after
        grave repair and convention mapping, or ``None`` if this instance's
        citation policy would reject it.
        """
        check = self._apply_convention_check(lemma, lang=lang, source=source)
        return {
            "lemma": lemma,
            "normalized": check.lemma,
            "ok": check.lemma is not None,
            "reason": check.reason,
            "normalized_from": check.normalized_from or None,
            "policy": self._citation_policy,
        }

    def _grc_adverb_head(self, lemma: str | None) -> str | None:
        """The adjective of a regular Ancient adverb in -ως, or None.

        Applies in Ancient Greek (lang "all" or "grc") under every polytonic
        convention. Modern Greek (lang "el", the monotonic triantafyllidis
        convention) keeps its adverbs as headwords, and so do the adverbs with
        no adjective base (οὕτως, πῶς, ὄντως, πάντως) and the reviewed
        lexicalized ones (ἴσως, ἰδίως, ἄλλως; data/grc_adverb_review.json).
        """
        if (not lemma or self.lang not in ("all", "grc")
                or self._convention_monotonic
                or self._convention_name == "triantafyllidis"
                or not lemma.endswith(("ως", "ῶς"))):
            return None
        adjective = _grc_adverb_data()[0].get(lemma)
        return adjective if adjective and adjective != lemma else None

    def _get_headword_set(self, convention: str = "lsj") -> set[str]:
        """Lazy-load and cache the headword set for a given convention.

        Returns a set of headword strings. Strips vowel-length marks
        (macron/breve) so matching works against plain forms.
        """
        if convention in self._headword_sets:
            return self._headword_sets[convention]

        headwords = _load_convention_headwords(convention)

        self._headword_sets[convention] = headwords
        return headwords

    def is_headword(self, word: str, convention: str = "lsj") -> bool:
        """Check if a word is a known headword in the given convention.

        Args:
            word: The Greek word to check.
            convention: Which headword list to check against.
                "lsj" (default), "cunliffe", or "triantafyllidis".

        Returns:
            True if the word is in the convention's headword list.
        """
        return word in self._get_headword_set(convention)

    def _build_sorted_headword_index(self):
        """Build sorted (sort_key, headword) list from all headword sources.

        Lazy-loaded on first call to headwords_between(). Combines:
        LSJ, Cunliffe, DGE, LGPN, VLG, and ag_headwords (Wiktionary-derived).
        """
        if hasattr(self, "_sorted_hw_index"):
            return
        all_hws = set()
        PD_HW_PATH = DATA_DIR / "pd_headwords.json"
        VLG_HW_PATH = DATA_DIR / "vlg_headwords.json"
        for path in [LSJ_HEADWORDS_PATH, CUNLIFFE_HEADWORDS_PATH,
                     AG_HEADWORDS_PATH, DGE_HEADWORDS_PATH, LGPN_NAMES_PATH,
                     PD_HW_PATH, VLG_HW_PATH]:
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    all_hws.update(json.load(f))
        # Build sorted list by sort key
        def _sort_key(hw):
            nfkd = unicodedata.normalize("NFKD", hw.lower().replace("-", ""))
            return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
        self._sorted_hw_pairs = sorted(
            ((_sort_key(h), h) for h in all_hws if h),
            key=lambda x: x[0])
        self._sorted_hw_keys = [p[0] for p in self._sorted_hw_pairs]
        self._sorted_hw_index = True

    def headwords_between(self, hw_a: str, hw_b: str) -> list[str]:
        """Return all known headwords sorting alphabetically between hw_a and hw_b.

        Uses accent-stripped, case-folded sort keys for comparison.
        Returns original polytonic headword forms.

        Args:
            hw_a: Lower bound headword (exclusive).
            hw_b: Upper bound headword (exclusive).

        Returns:
            List of headwords between hw_a and hw_b, sorted alphabetically.
        """
        import bisect
        self._build_sorted_headword_index()

        def _sort_key(hw):
            nfkd = unicodedata.normalize("NFKD", hw.lower().replace("-", ""))
            return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")

        key_a = _sort_key(hw_a)
        key_b = _sort_key(hw_b)
        if key_a > key_b:
            key_a, key_b = key_b, key_a

        lo = bisect.bisect_right(self._sorted_hw_keys, key_a)
        hi = bisect.bisect_left(self._sorted_hw_keys, key_b)
        return [self._sorted_hw_pairs[i][1] for i in range(lo, hi)]

    def _get_lemma_set(self) -> set[str]:
        """Get the set of all lemmata (citation forms) in the lookup table.

        Unlike headword sets which come from specific dictionaries, this
        returns all unique lemma values from the lookup table itself -
        the right-hand side of form->lemma mappings. Cached after first call.
        """
        if hasattr(self, "_lemma_set"):
            return self._lemma_set

        lemmas: set[str] = set()
        if self._using_db:
            cursor = self._lookup._conn.execute(
                "SELECT DISTINCT text FROM lemmas")
            lemmas = {row[0] for row in cursor}
        else:
            lemmas = set(self._lookup.values()) if self._lookup else set()

        self._lemma_set = lemmas
        return self._lemma_set

    def _find_model_dir(self):
        """Find the best available model directory.

        Search order: {lang}/ (full model), then {lang}-s3/-s2/-s1 (legacy),
        then combined/ as fallback if no language-specific model exists.
        """
        lang_dir = {"el": "el", "grc": "grc", "all": "combined"}[self.lang]

        # Explicit scale requested
        if self._scale is not None:
            for prefix in [lang_dir, "combined"]:
                # New naming: {lang}-test or {lang} (full)
                if str(self._scale) == "test":
                    candidate = MODEL_DIR / f"{prefix}-test"
                else:
                    candidate = MODEL_DIR / prefix
                if (candidate / "encoder.onnx").exists() or (candidate / "model.pt").exists():
                    return candidate
                # Legacy naming: -s1, -s2, -s3
                candidate = MODEL_DIR / f"{prefix}-s{self._scale}"
                if (candidate / "encoder.onnx").exists() or (candidate / "model.pt").exists():
                    return candidate

        # Auto-detect: prefer {lang}/ (full), then legacy -s3/-s2/-s1
        for prefix in [lang_dir, "combined"]:
            candidate = MODEL_DIR / prefix
            if (candidate / "encoder.onnx").exists() or (candidate / "model.pt").exists():
                return candidate
            for s in [3, 2, 1]:
                candidate = MODEL_DIR / f"{prefix}-s{s}"
                if (candidate / "encoder.onnx").exists() or (candidate / "model.pt").exists():
                    return candidate

        return MODEL_DIR / lang_dir

    def _load_model(self):
        """Lazy-load the model on first use. Prefers ONNX, falls back to PyTorch."""
        if self._model is not None:
            return

        if not self._backend_checked:
            self._backend_checked = True
            self._check_backend()

        model_path = self._find_model_dir()

        # Try ONNX first (no PyTorch dependency, ~50MB vs ~2GB)
        if (model_path / "encoder.onnx").exists():
            self._load_onnx(model_path)
            return

        # Fall back to PyTorch
        self._load_pytorch(model_path)

    def _load_onnx(self, model_path):
        """Load ONNX model and lightweight vocab."""
        from .onnx_inference import OnnxLemmaModel, CharVocabLight
        vocab_path = model_path / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"No vocab.json at {vocab_path}. "
                f"Run: python export_onnx.py"
            )
        self._vocab = CharVocabLight(vocab_path)
        self._model = OnnxLemmaModel(model_path)
        self._device = "cpu"
        self._use_onnx = True

    def _load_pytorch(self, model_path):
        """Load PyTorch model (original path).

        Detects and loads morphology heads (POS, nominal, verbal) when
        present in the checkpoint. Head label mappings are stored in
        self._head_labels for inference use.
        """
        import torch
        from .model import CharVocab, LemmaTransformer

        pt_path = model_path / "model.pt"
        if not pt_path.exists():
            raise FileNotFoundError(
                f"No trained model at {pt_path}. "
                f"Run: python train.py --lang {self.lang}"
            )

        device = self._device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        checkpoint = torch.load(pt_path, map_location=device, weights_only=False)
        self._vocab = CharVocab()
        self._vocab.load_state_dict(checkpoint["vocab"])
        cfg = checkpoint["config"]

        # Detect morphology heads from state dict keys
        state = checkpoint["model_state_dict"]
        head_cfg = checkpoint.get("head_config", {})

        # Infer head dimensions from weights if head_config is missing
        num_pos = cfg.get("num_pos_tags", 0)
        if not num_pos and "pos_head.weight" in state:
            num_pos = state["pos_head.weight"].shape[0]
            cfg["num_pos_tags"] = num_pos

        self._model = LemmaTransformer(**cfg)

        # Create nom/verb heads if weights exist
        d_model = cfg.get("d_model", 256)
        if "nom_head.weight" in state:
            num_nom = state["nom_head.weight"].shape[0]
            import torch.nn as nn
            self._model.nom_head = nn.Linear(d_model, num_nom)
        if "verb_head.weight" in state:
            num_verb = state["verb_head.weight"].shape[0]
            import torch.nn as nn
            self._model.verb_head = nn.Linear(d_model, num_verb)

        self._model.load_state_dict(state)
        self._model.to(device)
        self._model.eval()
        self._use_onnx = False

        # Store label mappings for inference
        self._head_labels = {
            "pos": {int(k): v for k, v in head_cfg.get("pos_labels", {}).items()},
            "nom": {int(k): v for k, v in head_cfg.get("nom_labels", {}).items()},
            "verb": {int(k): v for k, v in head_cfg.get("verb_labels", {}).items()},
        }
        # Build fallback POS label map if not saved
        if num_pos and not self._head_labels["pos"]:
            _POS_FALLBACK = {
                0: "verb", 1: "noun", 2: "adj", 3: "adv", 4: "name",
                5: "pron", 6: "num", 7: "prep", 8: "article", 9: "character",
            }
            self._head_labels["pos"] = {
                i: _POS_FALLBACK.get(i, f"tag{i}") for i in range(num_pos)
            }

    @staticmethod
    def _is_ancient_article_form(word: str) -> bool:
        # Accent stripping would confuse ἤ "or" with the article ἡ.
        lower = word.lower()
        if word in _ARTICLE_FORMS or lower in _ARTICLE_FORMS:
            return True
        # An elided word is no article (οἷ̓ is οἷος, αἶ̓ αἶα); its trailing
        # psili is the apostrophe, not a breathing.
        if _is_marked_vowel_psili_elision(word):
            return False
        # The plural articles οἱ/αἱ have a rough breathing, so a smooth one on
        # αι/οι marks another word (αἴ, αἰ, αἶ, οἴ), even though the monotonic
        # projection drops the breathing and matches αἵ/οἵ. Byzantine ὀ for
        # ὁ and τἠς for τῆς are scribal breathings and stay articles.
        if (strip_accents(lower) in ("αι", "οι")
                and "\u0313" in unicodedata.normalize("NFD", word)):
            return False
        return (to_monotonic(word) in _MONOTONIC_ARTICLE_FORMS
                or to_monotonic(lower) in _MONOTONIC_ARTICLE_FORMS)

    def _article_pos_lemma(self, word: str, upos: str) -> str | None:
        """Preserve requested marked articles despite a nominal tagging error."""
        contraction = self._vernacular_contraction_pos_lemma(word, upos)
        if contraction is not None:
            return contraction
        if (self._resolve_articles and self.lang in ("all", "grc")
                and self._convention_name != "triantafyllidis"
                and upos in ("NOUN", "PROPN") and word):
            normalized = to_standard_sigma(unicodedata.normalize("NFC", word)).lower()
            if normalized in _MARKED_ARTICLE_FORMS:
                return _ARTICLE_LEMMA
        if upos != "DET" or not self._is_ancient_article_form(word):
            return None
        if not self._resolve_articles:
            retained = self._resolve_closed_class(word)
            if retained is not None:
                return retained
        lemma = self._pos_table_lookup(word, upos)
        if lemma in (None, _ARTICLE_LEMMA, _MG_ARTICLE_LEMMA):
            return None
        if (self._convention_name == "triantafyllidis"
                and word.lower() in _MG_ARTICLE_FORMS):
            return _MG_ARTICLE_LEMMA
        return _ARTICLE_LEMMA

    def _vernacular_contraction_pos_lemma(self, word: str,
                                          upos: str) -> str | None:
        """The DET or ADP lemma of a contraction of σε + article.

        With lang="all" or "el", στο, στον, στην, στους, ... tagged DET
        gives the Modern article ο and tagged ADP gives σε, under every
        convention, in monotonic spelling (#134) as in polytonic (στὸ,
        στὴν; #121). Neither tag fits an Ancient reading of these letters
        (στῆν is a verb form, στόν the aphaeresis of ἀστόν), and the Modern
        POS table records the article for seven of the ten forms only,
        three of them behind a case-twin family (στο DET gave σε, στον
        στον, στην ἵστημι). The spelling may carry accents (acute, grave
        or circumflex) or none: a breathing, iota subscript or diaeresis
        is no vernacular contraction. lang="grc" reads Ancient Greek only.
        """
        if upos not in ("DET", "ADP") or self.lang not in ("all", "el") \
                or not word:
            return None
        nfd = unicodedata.normalize("NFD", word)
        marks = {ord(ch) for ch in nfd if unicodedata.combining(ch)}
        if not marks <= {0x0300, 0x0301, 0x0342}:
            return None
        if strip_accents(to_standard_sigma(word).lower()) \
                not in _VERNACULAR_ARTICLE_CONTRACTIONS:
            return None
        return _MG_ARTICLE_LEMMA if upos == "DET" else "σε"

    def _resolve_closed_class(self, word: str) -> str | None:
        """Recognize Ancient articles and optionally canonicalize closed classes.

        When convention='triantafyllidis', MG closed-class resolution
        takes priority: MG articles -> ο, preposition contractions
        (στη, στον, ...) -> σε, copula forms -> είμαι, etc. AG pronoun
        resolution (σε -> σύ, με -> ἐγώ) is skipped since those forms
        serve as prepositions/particles in MG.
        """
        if not self._resolve_articles:
            lower = unicodedata.normalize("NFC", word).lower()
            if self.lang in ("all", "grc") and lower in _MARKED_ARTICLE_FORMS:
                # Recognition does not depend on the lookup's lemma choice.
                # Preserve the article form, normalizing positional graves.
                return grave_to_acute(lower)
            return None

        # MG closed-class resolution (triantafyllidis convention)
        if self._convention_name == "triantafyllidis":
            lower = word.lower()
            # MG articles -> ο
            if lower in _MG_ARTICLE_FORMS:
                return _MG_ARTICLE_LEMMA
            # MG preposition contractions, copula, auxiliaries, αυτός
            if lower in _MG_CLOSED_CLASS:
                return _MG_CLOSED_CLASS[lower]
            # For triantafyllidis, skip AG pronoun resolution entirely.
            # AG pronouns (σε -> σύ, με -> ἐγώ) conflict with MG
            # prepositions/particles. Fall through to AG article
            # resolution only for polytonic article forms.

        lower = word.lower()
        if self._is_ancient_article_form(word):
            return _ARTICLE_LEMMA

        # Skip AG pronoun resolution when triantafyllidis is active
        if self._convention_name == "triantafyllidis":
            return None

        normalized = grave_to_acute(lower)
        if normalized in _PRONOUN_LEMMAS:
            return _PRONOUN_LEMMAS[normalized]
        return _MONOTONIC_PRONOUN_LEMMAS.get(to_monotonic(normalized))

    @staticmethod
    def _loose_keys(word: str, *, monotonic: bool = True,
                    stripped: bool = True) -> list[str]:
        """The lowercase monotonic and accent-stripped keys of *word*."""
        lower = word.lower()
        keys = []
        if monotonic:
            keys.append(to_monotonic(lower))
        if stripped:
            keys.append(strip_accents(lower))
        return list(dict.fromkeys(keys))

    @staticmethod
    def _capitalized_loose_keys(word: str, *, monotonic: bool = True,
                                stripped: bool = True) -> list[str]:
        """The monotonic and accent-stripped keys of a capitalized word, with its capital.

        The loose keys are lowercase, as the Wiktionary-derived rows are
        keyed. The generated dictionary paradigms key the accent-stripped
        spelling of a capitalized form with its capital (Σαμαρειτις for
        Σαμαρεῖτις), so a capitalized word that no lowercase key answers
        tries these keys too. No key for a lowercase word.
        """
        if word == word.lower():
            return []
        keys = []
        if monotonic:
            keys.append(to_monotonic(word))
        if stripped:
            keys.append(strip_accents(word))
        lowercase = Dilemma._loose_keys(word)
        return [key for key in dict.fromkeys(keys) if key not in lowercase]

    @staticmethod
    def _is_capitalized_ancient_head(lemma: str) -> bool:
        """Whether *lemma* is written as a capitalized Ancient citation form.

        Capitalized, accented, and with a breathing when it opens with a
        vowel or rho: the head of a generated paradigm (Σαμαρεῖτις), not a
        Modern Greek name (Ολυμπιακούς), a corpus lemma printed without
        accents (Εζεκιος) or a letter name.
        """
        if not is_capitalized(lemma) or _initial_breathing(lemma) == "":
            return False
        return any(mark in unicodedata.normalize("NFD", lemma)
                   for mark in "\u0300\u0301\u0342")

    def _capitalized_key_reading(self, table, keys: list[str]) -> str | None:
        """The first capitalized Ancient head *table* holds under *keys*."""
        for key in keys:
            hit = table.get(key)
            hit = self._normalize_grave_citation_lemma(hit) if hit else None
            if hit and self._is_capitalized_ancient_head(hit):
                return hit
        return None

    # The steps of the _lookup_word cascade, tightest first: the word's own
    # rows (as written, lowercase, grave folded to acute), its monotonic
    # key (the accent stays in place; breathing, accent type and iota
    # subscript are dropped) and its accent-stripped key.
    _OWN_ROWS, _MONOTONIC_KEY, _STRIPPED_KEY = 0, 1, 2

    def _lookup_word(self, word: str, *,
                     prefer_modern: bool | None = None,
                     step: int | None = None) -> str | None:
        """Try lookup cascade: exact -> lowercase -> grave_to_acute -> monotonic -> stripped.

        Grave-to-acute is tried before monotonic because it preserves
        breathings and circumflex (lighter normalization). ὣς → ὡς works
        here without losing the breathing that monotonic would strip.

        For polytonic input (breathings/circumflex present), AG lookup is
        tried first to avoid MG lemma forms (biblion vs biblion).

        For lang="el" (MG mode), MG-specific entries are checked first
        (monotonic input -> el lookup), then falls back to combined.

        Skips mono/stripped matches that are trivially short (1-2 chars)
        and map to themselves - these are usually false positives from
        accent stripping on elided or particle forms.

        *prefer_modern* overrides whether the MG-specific entries are
        checked first; a caller that wants the Ancient reading whatever
        the convention passes False.

        *step* runs one step of the cascade alone (_OWN_ROWS,
        _MONOTONIC_KEY or _STRIPPED_KEY): a caller that has already
        changed the spelling asks, for all its respellings, first for a
        row that needs no further change.
        """
        own = step in (None, self._OWN_ROWS)
        if own and self.lang in ("all", "grc"):
            grammar = reviewed_verb_grammar(word)
            if grammar is not None:
                return grammar["lemma"]
        loose_keys, capitalized_keys = [], []
        if step != self._OWN_ROWS:
            kinds = {"monotonic": step in (None, self._MONOTONIC_KEY),
                     "stripped": step in (None, self._STRIPPED_KEY)}
            loose_keys = self._loose_keys(word, **kinds)
            # Read after the lowercase keys, in the Ancient rows only.
            capitalized_keys = self._capitalized_loose_keys(word, **kinds)

        # For MG mode or triantafyllidis convention, try MG-specific
        # lookup first. MG entries take priority over AG when there's a
        # conflict (e.g. χρήσης -> χρήση in MG vs χράω in AG, or
        # κοντά -> κοντά in MG vs κοντός in AG).
        if prefer_modern is None:
            prefer_modern = (self.lang == "el"
                             or self._convention_name == "triantafyllidis")
        if prefer_modern and self._mg_lookup:
            tbl = self._mg_lookup
            hit = (tbl.get(word) or tbl.get(word.lower())) if own else None
            hit = self._normalize_grave_citation_lemma(hit) if hit else None
            if not hit:
                for variant in loose_keys:
                    hit = tbl.get(variant)
                    hit = self._normalize_grave_citation_lemma(hit) if hit else None
                    if hit and not (len(variant) <= 2 and hit == variant):
                        break
                    hit = None
            if hit:
                return hit

        # For polytonic input (breathings/circumflex), try AG-only lookup
        # first. Even though AG has priority in the combined table, the
        # normalization cascade (mono/stripped) can still land on MG entries.
        nfd = unicodedata.normalize("NFD", word)
        has_poly = any(ord(ch) in _POLYTONIC_STRIP | _POLYTONIC_TO_ACUTE
                       for ch in nfd)
        if has_poly and self.lang == "all" and self._ag_lookup:
            tbl = self._ag_lookup
            hit = (tbl.get(word) or tbl.get(word.lower())) if own else None
            hit = self._normalize_grave_citation_lemma(hit) if hit else None
            if own and not hit:
                acute = grave_to_acute(word)
                if acute != word:
                    hit = tbl.get(acute) or tbl.get(acute.lower())
                    hit = self._normalize_grave_citation_lemma(hit) if hit else None
            if not hit:
                for variant in loose_keys:
                    hit = tbl.get(variant)
                    hit = self._normalize_grave_citation_lemma(hit) if hit else None
                    # A loose key may not drop the form's breathing (#109).
                    hit = self._restore_breathing(word, hit) if hit else None
                    if hit and not (len(variant) <= 2 and hit == variant):
                        break
                    hit = None
            if not hit:
                hit = self._capitalized_key_reading(tbl, capitalized_keys)
            if hit:
                return hit

        if own:
            lemma = self._lookup.get(word) or self._lookup.get(word.lower())
            lemma = self._normalize_grave_citation_lemma(lemma) if lemma else None
            if lemma:
                return lemma
            # Grave → acute (lightest normalization, preserves breathings)
            acute = grave_to_acute(word)
            if acute != word:
                lemma = self._lookup.get(acute) or self._lookup.get(acute.lower())
                lemma = self._normalize_grave_citation_lemma(lemma) if lemma else None
                if lemma:
                    return lemma
        for variant in loose_keys:
            hit = self._lookup.get(variant)
            hit = self._normalize_grave_citation_lemma(hit) if hit else None
            if hit and not prefer_modern:
                # A loose key may not drop the form's breathing (#109).
                hit = self._restore_breathing(word, hit)
            if hit and not (len(variant) <= 2 and hit == variant):
                return hit
        return self._capitalized_key_reading(self._lookup, capitalized_keys)

    def _positive_elision_target(self, lemma: str) -> bool:
        """Whether merged lexical definitions support an exact elision target."""
        if self._elision_lexical_heads is None:
            self._elision_lexical_heads = _lexical_heads(self.lang)
        normalized = _without_quantity_marks(sanitize_form(lemma)).replace("-", "")
        return normalized in self._elision_lexical_heads

    def _lookup_elided(self, word: str) -> str | None:
        """Exact corpus-derived entry for an elided form (ὅτ᾽->ὅτε, κ᾽->ἄν, μ᾽->ἐγώ).

        The lemma for an elided form is genuine corpus evidence about which
        word was elided, so it must beat the frequency-ranked vowel expander
        (which otherwise lets the common homograph win: ὅτι over ὅτε, καί over
        ἄν, ὁ over τε). lookup.db stores these under one canonical U+1FBD key,
        so any apostrophe codepoint the text uses resolves the same way.

        Rejects self-maps (ἀλλ᾽->ἀλλ᾽) and entries whose value is itself an
        elided form; those carry no lemma information and must fall through to
        the expander, whose function-word allow-list settles them (ἀλλ᾽->ἀλλά).

        Known residual (context-only, unfixable by a static lookup): elided
        ὅθ᾽ maps to the temporal ὅτε here, but a handful of Homeric ὅθ᾽ tokens
        are the locative ὅθι "where" (e.g. Iliad 10.520, 17.54, 2.572, 20.320,
        4.217, 7.143). Disambiguating ὅτε vs ὅθι needs syntactic context; both
        readings share this one surface form. This still corrects the far more
        common prior error (ὅθ᾽ -> ὅτι, wrong for every token).
        """
        canon = _canon_elision(word)
        if not canon:
            return None
        lemma = self._lookup.get(canon) or self._lookup.get(canon.lower())
        if not lemma or lemma == canon or _strip_elision(lemma) is not None:
            return None
        return self._normalize_grave_citation_lemma(lemma)

    def _expand_elision(self, word: str) -> str | None:
        """Try to resolve an elided form by expanding with vowels.

        Strips the elision mark, appends each Greek vowel, and checks
        if the expanded form is in the lookup table. Prefers expansions
        where the lemma is a real word (differs from the expanded form)
        over self-mapping headwords.
        """
        candidates = self._expand_elision_all(word)
        if not candidates:
            return None
        # _expand_elision_all returns results pre-sorted by vowel frequency
        return candidates[0][1]

    def _expand_elision_all(self, word: str) -> list[tuple[str, str, str]]:
        """Return ALL valid elision expansions as (expanded, lemma, vowel) triples.

        For polytonic input (has breathings/circumflex), prioritizes AG lookup.
        Elision is overwhelmingly an AG phenomenon; MG monotonic forms would
        pollute results with false matches.

        Results are sorted by vowel frequency in elision contexts (ε, α, ο
        most common), then by lemma length. This ensures both lemmatize()
        and lemmatize_verbose() get the same ranking.
        """
        stem = _strip_elision(word)
        if not stem:
            return []

        # Detect if input is polytonic (has AG-style diacritics).
        # Also treat any non-combining elision mark (U+1FBD KORONIS,
        # U+02BC, U+2019) as indicating AG context, since elision is
        # overwhelmingly an AG phenomenon.
        nfd = unicodedata.normalize("NFD", word)
        has_polytonic = (any(ord(ch) in _POLYTONIC_STRIP | _POLYTONIC_TO_ACUTE
                            for ch in nfd)
                         or any(ch in _ELISION_MARKS
                                and unicodedata.category(ch) != "Mn"
                                for ch in word))

        # Choose which lookups to search
        if has_polytonic:
            tables = [(self._ag_lookup, "grc")]
        else:
            tables = [(self._mg_lookup, "el"),
                      (self._med_lookup, "med"),
                      (self._ag_lookup, "grc")]

        results = []
        seen_lemmas = set()

        all_vowels = list(_GREEK_VOWELS)
        # Also try accented vowels
        accented = {"α": "ά", "ε": "έ", "ι": "ί", "ο": "ό",
                    "η": "ή", "υ": "ύ", "ω": "ώ"}
        all_candidates = [(v, v) for v in all_vowels]
        all_candidates += [(acc, v) for v, acc in accented.items()]

        # Consonant de-assimilation: before rough breathing, Greek
        # assimilates voiceless stops to aspirates (τ->θ, π->φ, κ->χ).
        # Reverse this to recover the original stem for prepositions:
        # καθ' -> κατ' (κατά), ἐφ' -> ἐπ' (ἐπί), ἀφ' -> ἀπ' (ἀπό)
        _DEASSIMILATE = {"θ": "τ", "φ": "π", "χ": "κ"}
        stems_to_try = [stem]
        if stem and stem[-1] in _DEASSIMILATE:
            stems_to_try.append(stem[:-1] + _DEASSIMILATE[stem[-1]])

        for try_stem in stems_to_try:
            for table, lang in tables:
                for suffix, vowel_name in all_candidates:
                    expanded = try_stem + suffix
                    # Try full normalization cascade
                    lemma = None
                    for variant in (expanded, expanded.lower(),
                                    grave_to_acute(expanded),
                                    to_monotonic(expanded.lower()),
                                    strip_accents(expanded.lower())):
                        lemma = table.get(variant)
                        lemma = (self._normalize_grave_citation_lemma(lemma)
                                 if lemma else None)
                        if lemma:
                            break
                    if lemma and lemma not in seen_lemmas:
                        seen_lemmas.add(lemma)
                        results.append((expanded, lemma, suffix))

        # Common elided function words - these should always win over
        # content words when ambiguous. Includes both original stems
        # and assimilated forms (καθ from κατά, αφ from ἀπό, etc.)
        _ELISION_PRIORITY = {
            "αλλ", "μετ", "παρ", "κατ", "δι", "απ", "επ",
            "υπ", "αφ", "εφ", "υφ", "μηδ", "ουδ", "αντ", "περ",
            # Assimilated forms (θ<-τ, φ<-π, χ<-κ before rough breathing)
            "καθ", "εφ", "αφ", "υφ", "μεθ", "παρ", "ανθ",
        }
        stem_lower = strip_accents(stem.lower())

        # Sort by: (1) function word priority, (2) vowel frequency, (3) lemma length
        _VOWEL_RANK = {v: i for i, v in enumerate("εαοιηυω")}
        _ACC_VOWEL_RANK = {"ά": 1, "έ": 0, "ί": 4, "ό": 2,
                           "ή": 3, "ύ": 5, "ώ": 6}

        # Function word lemmas (prepositions, particles, conjunctions)
        _FUNCTION_LEMMAS = {
            "ἀλλά", "μετά", "παρά", "κατά", "διά", "ἀπό", "ἐπί",
            "ὑπό", "ἀπό", "ἐπί", "ὑπό", "μηδέ", "οὐδέ", "ἀντί",
            "περί", "δέ", "γε", "τε", "ἄρα", "ἔτι",
        }

        def _rank(item):
            expanded, lemma, vowel = item
            # Prefer function words when stem is a known elision stem
            is_function = 0 if (stem_lower in _ELISION_PRIORITY
                                and lemma in _FUNCTION_LEMMAS) else 1
            # Deprioritize proper nouns
            is_proper = 1 if is_capitalized(lemma) else 0
            # Use corpus frequency: more frequent lemmas are more likely
            # to be the correct resolution (κατά >> κάθω). Ranked things are
            # LEMMAS, so query the lemma-keyed attestation table - the
            # form-keyed corpus_freq table would count every unrelated
            # homograph of the accent-stripped spelling (lemma εἷς stripped
            # to "εις" picked up the preposition εἰς's 400k tokens, which is
            # how μ᾽ once resolved to εἷς instead of ἐγώ).
            #
            # Count only the HAND-CHECKED corpora (GLAUx, Diorisis): the
            # all-auto OGA source attests its own lemmatizer's junk (ρε,
            # μηνιν), so including it lets ghost entries escape the
            # unattested tier - exactly how ῥ᾽ regressed onto ρε instead
            # of ἄρα when OGA first joined the attestation artifact.
            att = self.attestation(lemma)
            freq = 0
            if att:
                sc = att.get("source_counts") or {}
                freq = sc.get("glaux", 0) + sc.get("diorisis", 0)
            neg_freq = -freq  # negate so higher frequency sorts first
            vrank = _ACC_VOWEL_RANK.get(vowel, _VOWEL_RANK.get(vowel, 10))
            # Corpus-attested lemmas before unattested ones (a lemma with
            # zero attestation is usually a junk table entry: ῥ᾽ must not
            # land on the ghost entry ρε when ἄρα is available), then vowel
            # plausibility BEFORE lemma frequency: which vowel was elided is
            # phonological evidence (ε/α/ο elide constantly, η/ω almost
            # never), while global lemma frequency is not evidence about
            # elision at all - ranked freq-first, γ᾽ resolved to the
            # frequent ἐγώ (via the aphaeresis entry γώ) instead of γε.
            return (is_function, is_proper, freq == 0, vrank, neg_freq,
                    len(lemma))

        results.sort(key=_rank)
        return results

    def _lang_of(self, form: str) -> str:
        """Determine which language table a form comes from."""
        if self._using_db:
            return self._lang_of_db(form)
        lower = form.lower()
        mono = to_monotonic(lower)
        stripped = strip_accents(lower)
        for variant in [form, lower, mono, stripped]:
            if variant in self._mg_lookup:
                return "el"
            if variant in self._med_lookup:
                return "med"
            if variant in self._ag_lookup:
                return "grc"
        return ""

    def _lang_of_db(self, form: str) -> str:
        """SQLite-backed language detection via the src column."""
        _SRC_TO_LANG = {'grc': 'grc', 'el': 'el', 'med': 'med'}
        conn = self._lookup._conn
        lower = form.lower()
        mono = to_monotonic(lower)
        stripped = strip_accents(lower)
        query = "SELECT src FROM lookup WHERE form = ? AND lang = 'all' LIMIT 1"
        for variant in [form, lower, mono, stripped]:
            row = conn.execute(query, (variant,)).fetchone()
            if row:
                return _SRC_TO_LANG.get(row[0], "")
        return ""

    def _is_proper(self, lemma: str) -> bool:
        """Check if a lemma is a proper noun (capitalized headword)."""
        return is_capitalized(lemma)

    def _is_mg_citation_form(self, lemma: str) -> bool:
        """True iff ``lemma`` is a Modern Greek citation form: the MG table
        maps the exact string to itself. Used by repair_lemma so a valid
        Demotic lemma is never "repaired" onto an AG headword."""
        if self._using_db:
            row = self._lookup._conn.execute(
                "SELECT l.text FROM lookup k JOIN lemmas l ON l.id = k.lemma_id"
                " WHERE k.form = ? AND k.lang = 'all' AND k.src = 'el' LIMIT 1",
                (lemma,)).fetchone()
            return bool(row) and row[0] == lemma
        return self._mg_lookup.get(lemma) == lemma

    # ---- Morphology head inference ----

    # POS tag to UPOS mapping for POS-lookup integration
    _POS_TO_UPOS = {
        "verb": "VERB", "noun": "NOUN", "adj": "ADJ", "adv": "ADV",
        "name": "PROPN", "pron": "PRON", "num": "NUM", "prep": "ADP",
        "article": "DET", "character": "PROPN",
    }

    def predict_pos_tag(self, word: str) -> str:
        """Predict POS tag for a Greek word using the model's POS head.

        Returns a Wiktionary-style POS label ("verb", "noun", "adj", etc.)
        or "" if no POS head is available.

        Requires the model to be loaded (lazy-loads on first call).
        """
        self._load_model()
        if hasattr(self._model, 'has_pos_head') and self._model.has_pos_head:
            # ONNX path
            src, mask = self._encode_word(word)
            tags = self._model.predict_pos(src, mask)
            return tags[0]
        return ""

    def predict_pos_batch(self, words: list[str]) -> list[str]:
        """Predict POS tags for a batch of Greek words.

        Returns list of Wiktionary-style POS labels.
        """
        self._load_model()
        if hasattr(self._model, 'has_pos_head') and self._model.has_pos_head:
            src, mask = self._encode_words(words)
            return self._model.predict_pos(src, mask)
        return [""] * len(words)

    def _encode_word(self, word: str):
        """Encode a single word for ONNX inference. Returns (src, mask) arrays."""
        import numpy as np
        ids = self._vocab.encode(word)
        max_len = 48  # ONNX_MAX_LEN
        ids = ids + [0] * (max_len - len(ids))
        src = np.array([ids[:max_len]], dtype=np.int64)
        mask = (src == 0)
        return src, mask

    def _encode_words(self, words: list[str]):
        """Encode a batch of words for ONNX inference."""
        import numpy as np
        max_len = 48
        batch = []
        for w in words:
            ids = self._vocab.encode(w)
            ids = ids + [0] * (max_len - len(ids))
            batch.append(ids[:max_len])
        src = np.array(batch, dtype=np.int64)
        mask = (src == 0)
        return src, mask

    # ---- Byzantine normalization fallback ----

    def _load_fallback_readings(self):
        """Load compact grammatical evidence independently of POS selection.

        ``skip_pos`` disables POS disambiguation, but must not turn a noun
        lookup into evidence for an augmented verb. Keep all source readings
        at a normalized form rather than choosing one homograph here. The
        full pair files and expanded nominal POS table are not needed.

        Cells that target a reviewed non-lemma (Ταύτη, Τοῦτος;
        dilemma/grc_rejected_lemmas.py) are skipped as in
        ``_load_pos_lookups``, so the Byzantine-normalisation lookups and
        the participle nominative fallback cannot return one either (#133).
        """
        if getattr(self, "_fallback_readings", None) is not None:
            return
        readings = {}
        exact_readings = {}
        verb_lemmas = set()
        name_lemmas = set()
        ancient_pronouns = {}
        paths = []
        if self.lang in ("all", "grc"):
            paths.extend((AG_POS_LOOKUP_PATH, GLAUX_POS_LOOKUP_PATH,
                          TREEBANK_POS_LOOKUP_PATH))
        if self.lang in ("all", "el"):
            paths.append(MG_POS_LOOKUP_PATH)
        corrected_tables = {AG_POS_LOOKUP_PATH: "ag", GLAUX_POS_LOOKUP_PATH: "glaux",
                            TREEBANK_POS_LOOKUP_PATH: "treebank",
                            MG_POS_LOOKUP_PATH: "mg"}
        for path in paths:
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as stream:
                table = json.load(stream)
            if path in corrected_tables:
                _apply_pos_cell_corrections(corrected_tables[path], table)
            for form, by_pos in table.items():
                kept = {pos: lemma for pos, lemma in by_pos.items()
                        if lemma not in _REJECTED_POS_LEMMAS}
                if not kept and by_pos:
                    # Like _load_pos_lookups: a form whose every cell is a
                    # rejected non-lemma leaves no (empty) exact entry that
                    # would hide its other accentual homographs.
                    continue
                key = strip_accents(form.lower())
                entry = readings.setdefault(key, {})
                exact = exact_readings.setdefault(grave_to_acute(form), {})
                for pos, lemma in kept.items():
                    lemma = self._normalize_grave_citation_lemma(lemma)
                    if (not lemma or _is_elided_junk_value(lemma)
                            or lemma in _REJECTED_POS_LEMMAS):
                        continue
                    entry.setdefault(pos, set()).add(lemma)
                    exact.setdefault(pos, set()).add(lemma)
                    if pos in ("VERB", "AUX"):
                        verb_lemmas.add(lemma)
                    elif pos == "PROPN":
                        name_lemmas.add(lemma)
                    if pos == "PRON" and path != MG_POS_LOOKUP_PATH:
                        ancient_pronouns.setdefault(key, set()).add(lemma)
        self._ancient_pronoun_readings = ancient_pronouns
        self._fallback_readings = readings
        self._fallback_exact_readings = exact_readings
        self._fallback_verb_lemmas = verb_lemmas
        self._fallback_name_lemmas = frozenset(name_lemmas)

    def _grammatical_readings(self, word: str) -> dict[str, set[str]]:
        """Prefer an exact accented reading to other accentual homographs."""
        self._load_fallback_readings()
        word = grave_to_acute(unicodedata.normalize("NFC", word))
        for variant in (word, word.lower()):
            if variant in self._fallback_exact_readings:
                return self._fallback_exact_readings[variant]
        return self._fallback_readings.get(strip_accents(word.lower()), {})

    def _grammatical_targets(self, word: str, *, verb_only=False) -> set[str]:
        """Return all retained grammatical targets for a form."""
        by_pos = self._grammatical_readings(word)
        if verb_only:
            return by_pos.get("VERB", set()) | by_pos.get("AUX", set())
        return set().union(*by_pos.values()) if by_pos else set()

    def _lookup_grammatical_word(self, word: str, *, verb_only=False,
                                 normalization=False,
                                 source_word: str | None = None) -> str | None:
        """Resolve a transformed form without promoting a noun to a verb."""
        targets = self._grammatical_targets(word, verb_only=verb_only)
        lemma = self._lookup_word(word)
        if normalization:
            return lemma if lemma and self._normalization_reading_supported(
                word, lemma, targets=targets, source_word=source_word) else None
        if lemma in targets:
            return lemma
        if verb_only and lemma in self._fallback_verb_lemmas:
            # A known verb already resolves the transformed spelling. A
            # conflicting corpus annotation must not replace that reading.
            return lemma
        if verb_only and lemma:
            readings = self._grammatical_readings(word)
            head_readings = self._grammatical_readings(lemma)
            if (_is_self_map(word, lemma)
                    or any(lemma in values for pos, values in readings.items()
                           if pos not in ("VERB", "AUX"))
                    or (lemma not in self._fallback_verb_lemmas
                        and any(lemma in values for pos, values in head_readings.items()
                                if pos not in ("VERB", "AUX")))):
                # Stripping an apparent augment is not a POS tag. A genuine
                # noun/adjective or identity reading must not be replaced by
                # an unrelated verb merely sharing its unaccented letters.
                return None
        if len(targets) == 1:
            return next(iter(targets))
        if targets:
            # The sources disagree and the ordinary lookup does not select
            # any of their readings. Do not invent a tiebreak for a fallback.
            return None
        if verb_only:
            # Ordinary verb forms are omitted from the compact POS tables
            # when they need no disambiguation. Their lexical verb target
            # still establishes the grammatical class of the lookup result.
            if self._grammatical_readings(word):
                return None
            return lemma if lemma in self._fallback_verb_lemmas else None
        return lemma

    def _normalization_reading_supported(self, word: str, lemma: str,
                                          *, targets=None,
                                          source_word: str | None = None) -> bool:
        """Keep ordinary normalization, rejecting contradictory homographs.

        The compact tables omit unambiguous forms. Absence alone therefore
        cannot invalidate an ordinary lookup. For a clearly verbal input,
        it also cannot establish the nominal reading of a head that is
        independently a verb inflection: φάσκον is both a noun and a form
        of φάσκω, but does not justify normalizing φάσκονται to φάσκοντε.
        """
        if targets is None:
            targets = self._grammatical_targets(word)
        if lemma in targets:
            return True
        # These middle/passive endings are not ordinary nominal inflections.
        # The ending alone never rejects a lookup: the target must also be a
        # source-backed nominal/verbal homograph with no matching form reading.
        if (not source_word or not strip_accents(source_word.lower()).endswith(
                ("ονται", "ωνται", "ομεθα", "ωμεθα", "εσθε", "ησθε",
                 "σθαι", "σθω", "σθων"))):
            return True
        head = self._grammatical_readings(lemma)
        nominal = any(lemma in head.get(pos, ())
                      for pos in ("NOUN", "PROPN", "ADJ"))
        verbs = head.get("VERB", set()) | head.get("AUX", set())
        return not (nominal and verbs and lemma not in self._fallback_verb_lemmas)

    # Iota adscript -> subscript mappings (Byzantine texts often write
    # the iota next to rather than under the vowel).
    _IOTA_ADSCRIPT_MAP = {
        'ωι': 'ῳ', 'ηι': 'ῃ',
        'ῶι': 'ῷ', 'ῆι': 'ῇ',
        'ώι': 'ῴ', 'ήι': 'ῄ',
        'ὼι': 'ῲ', 'ὴι': 'ῂ',
    }

    # Known Greek verbal/nominal prefixes for compound decomposition.
    _KNOWN_PREFIXES = [
        'ἀντι', 'ἐπι', 'ὑπερ', 'παρα', 'κατα', 'ἀπο', 'μετα',
        'περι', 'προσ', 'ἀνα', 'δια', 'ὑπο', 'ἐξ', 'ἐκ',
        'εἰσ', 'εἰς', 'συν', 'συμ', 'συγ', 'συλ', 'ἐν',
        'προ',
    ]

    @staticmethod
    def _other_breathing(word: str) -> str | None:
        """*word* with the breathing of its initial vowel or diphthong
        changed to the other one; None when it has none there."""
        nfd = unicodedata.normalize("NFD", word)
        for at, char in enumerate(nfd):
            if char in "\u0313\u0314":
                swapped = "\u0314" if char == "\u0313" else "\u0313"
                return unicodedata.normalize(
                    "NFC", nfd[:at] + swapped + nfd[at + 1:])
            if char.lower() not in "αεηιουω" and not unicodedata.combining(char):
                return None
        return None

    def _normalize_byzantine(self, word: str,
                             use_normalizer: bool = True) -> str | None:
        """Try Byzantine-specific normalizations to find a lookup match.

        Handles common scribal/orthographic features of Byzantine Greek
        manuscripts that differ from the classical forms stored in the
        lookup table:
          - Final medial sigma (σ instead of ς at word end)
          - Iota adscript (ωι, ηι instead of subscript ῳ, ῃ)
          - Missing breathings on initial vowels
          - The wrong breathing on an initial vowel (with the normalizer)
          - Hyphens and editorial marks in transcriptions

        When use_normalizer=True, also chains with the orthographic
        normalizer to handle itacism, geminate simplification, etc.
        This is more aggressive and can produce false positives.

        When use_normalizer=False, only the precise/safe normalizations
        above are applied.

        Called as a fallback after lookup, normalizer, and model have
        all failed. Returns a lemma if any normalization yields a
        lookup hit, otherwise None.
        """
        # 1. Final sigma: σ at end of word -> ς
        if word.endswith('σ'):
            fixed = word[:-1] + 'ς'
            lemma = self._lookup_word(fixed)
            if lemma:
                return lemma

        # 2. Iota adscript -> subscript
        fixed = word
        for src, dst in self._IOTA_ADSCRIPT_MAP.items():
            fixed = fixed.replace(src, dst)
        if fixed != word:
            lemma = self._lookup_word(fixed)
            if lemma:
                return lemma

        # 3. Hyphen removal (editorial hyphens in line-broken words)
        if '-' in word:
            fixed = word.replace('-', '')
            lemma = self._lookup_word(fixed)
            if lemma:
                return lemma

        # 4. Lowercase initial capital (Byzantine sentence-initial caps)
        if is_capitalized(word):
            lowered = word[0].lower() + word[1:]
            lemma = self._lookup_word(lowered)
            if lemma:
                return lemma
            # Also try lowercase + iota adscript fix
            fixed = lowered
            for src, dst in self._IOTA_ADSCRIPT_MAP.items():
                fixed = fixed.replace(src, dst)
            if fixed != lowered:
                lemma = self._lookup_word(fixed)
                if lemma:
                    return lemma
            # Also try lowercase + final sigma fix
            if lowered.endswith('σ'):
                fixed = lowered[:-1] + 'ς'
                lemma = self._lookup_word(fixed)
                if lemma:
                    return lemma

        # 5. Missing breathing on initial vowel: try adding smooth or rough
        lower = word.lower()
        nfd = unicodedata.normalize("NFD", lower)
        if nfd and nfd[0] in 'αεηιουω':
            # Check if there's already a breathing mark
            has_breathing = (len(nfd) > 1
                            and ord(nfd[1]) in (0x0313, 0x0314))
            if not has_breathing:
                # Try smooth breathing (U+0313), then rough (U+0314)
                for breathing in ('\u0313', '\u0314'):
                    candidate = unicodedata.normalize(
                        "NFC", nfd[0] + breathing + nfd[1:])
                    lemma = self._lookup_word(candidate)
                    if lemma:
                        return lemma

        if not use_normalizer:
            return None

        # 6. Chain: apply Byzantine fixes, then run the orthographic
        #    normalizer on the result. This handles cases like
        #    ἀναγυνώσκοντα (ι/υ itacism + no iota adscript issue)
        #    where the normalizer alone would have worked if the
        #    form had standard orthographic features.
        #    Use existing normalizer if available, otherwise create a
        #    lightweight Byzantine normalizer on first use.
        #    Only applied to longer words (>= 6 chars) to avoid false
        #    matches on short common words.
        stripped_lower = strip_accents(word.lower())
        if len(stripped_lower) >= 6:
            if not hasattr(self, '_byz_normalizer'):
                from .normalize import Normalizer
                self._byz_normalizer = Normalizer(period="byzantine")
            norm = self._normalizer or self._byz_normalizer

            # A breathing slip changes no letter and no accent, which is
            # less than any respelling below: ἅπατον is ἄπατον before it
            # is ἅπαθον (τ for θ) or the letters of ἀπάτων (ο for ω).
            other = self._other_breathing(word)
            if other:
                lemma = self._lookup_word(other, step=self._OWN_ROWS)
                if lemma and self._normalization_reading_supported(
                        other, lemma, source_word=word):
                    return lemma

            # Collect all forms to try: original + Byzantine-fixed variants
            forms_to_try = [word]
            # Final sigma fix
            if word.endswith('σ'):
                forms_to_try.append(word[:-1] + 'ς')
            # Iota adscript fix
            fixed = word
            for src, dst in self._IOTA_ADSCRIPT_MAP.items():
                fixed = fixed.replace(src, dst)
            if fixed != word:
                forms_to_try.append(fixed)
            # Lowercase
            if is_capitalized(word):
                forms_to_try.append(word[0].lower() + word[1:])

            # Only run normalizer if it wasn't already run pre-model
            # (avoid double-normalizing). For pre-model normalizer,
            # only the original form was tried, so skip it here.
            start_idx = 1 if self._normalizer else 0
            # NFC, so that a row is found as it is written (the tagged
            # entry points pass an oxia through).
            respellings = list(dict.fromkeys(
                unicodedata.normalize("NFC", candidate)
                for variant in forms_to_try[start_idx:]
                for candidate in norm.normalize(variant)))
            # Every respelling changes the spelling of one sound, and the
            # normalizer ranks them by how common that confusion is. The
            # lookup cascade can then change the accents as well: the
            # monotonic key drops breathing, accent type and subscript,
            # the accent-stripped key also moves the accent. That is a
            # second change, so it ranks after every respelling that
            # reads a row as it is written, whatever the rule weights:
            # αναγνώστει is αναγνώστη (ει for η), not the letters of
            # ἀναγνωστέοι reached through αναγνώστεοι (ι for οι).
            answered = set()
            for step in (self._OWN_ROWS, self._MONOTONIC_KEY,
                         self._STRIPPED_KEY):
                for candidate in respellings:
                    if candidate in answered:
                        continue
                    lemma = self._lookup_word(candidate, step=step)
                    if not lemma:
                        continue
                    # The first row the cascade reaches decides for this
                    # respelling, as in a single lookup of it.
                    answered.add(candidate)
                    if not self._normalization_reading_supported(
                            candidate, lemma, source_word=word):
                        continue
                    lemma_s = strip_accents(lemma.lower())
                    cand_s = strip_accents(candidate.lower())
                    # Guard: only accept if the lookup found a real
                    # lemma (different from the candidate form).
                    # This prevents accepting identity lookups where
                    # the normalizer just tweaked the form slightly.
                    if lemma_s != cand_s:
                        return lemma

        return None

    def _prefix_strip_lookup(self, word: str) -> str | None:
        """Try stripping known Greek prefixes and looking up the base.

        For compound verbs and adjectives where the prefixed form is not
        in the lookup table but the base form is. Only returns a result
        if the reconstructed prefix+base_lemma compound is itself a known
        form in the lookup table (very conservative to avoid false hits).
        """
        lower = word.lower()
        stripped = strip_accents(lower)

        for prefix in self._KNOWN_PREFIXES:
            prefix_s = strip_accents(prefix)
            if not stripped.startswith(prefix_s):
                continue
            if len(stripped) <= len(prefix_s) + 2:
                continue

            remainder = stripped[len(prefix_s):]
            base_lemma = self._lookup_grammatical_word(remainder)
            if not base_lemma:
                continue

            base_lemma_s = strip_accents(base_lemma.lower())

            # Guard: skip identity lookups
            if base_lemma_s == remainder:
                continue

            # Guard: skip suspiciously short base lemmas
            if len(base_lemma_s) < 2:
                continue

            # Reconstruct compound: prefix + base_lemma
            compound = prefix_s + base_lemma_s

            # Validate: the reconstructed compound must itself be
            # a known form in the lookup table. This avoids producing
            # nonsense compounds like υπεραραρισκω.
            compound_lemma = self._lookup_word(compound)
            if compound_lemma:
                return compound_lemma

        return None

    # ---- Compound decomposition ----

    # Linking vowels at the junction of Greek compounds
    _COMPOUND_LINK_VOWELS = set("οιυ")
    _MIN_COMPOUND_PREFIX = 2   # e.g. εὐ-, τρι-
    _MIN_COMPOUND_BASE = 3     # need enough for inflection
    # The second element and its lemma begin alike (προφήτου, προφήτης;
    # ἡγουμένην, ἡγέομαι). A base that resolves to an unrelated word, such as
    # the crasis κἠν (καὶ ἐν) or a stray verb form, is not a compound's head:
    # μοναχικὴν is not μοναχι- + ἐν and Βανδίνην is not βανδι- + νάω.
    _MIN_COMPOUND_SHARED_START = 2

    @classmethod
    def _compound_base_matches(cls, base: str, base_lemma: str) -> bool:
        n = cls._MIN_COMPOUND_SHARED_START
        return len(base) >= n and base[:n] == base_lemma[:n]

    # Dictionaries that can accent a reconstructed compound, in priority
    # order. Ancient lists serve polytonic or Ancient input; Modern Greek
    # lists serve monotonic input when Modern Greek is enabled.
    _COMPOUND_AG_HEADWORD_SOURCES = ("lsj", "lsj10", "cunliffe", "dge",
                                     "lbg", "ag")
    _COMPOUND_MG_HEADWORD_SOURCES = ("triantafyllidis", "mg")

    def _compound_headword_index(self, source: str) -> dict[str, list[str]]:
        idx = self._compound_headword_indexes.get(source)
        if idx is not None:
            return idx
        path = DATA_DIR / f"{source}_headwords.json"
        idx = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            for entry in raw:
                if isinstance(entry, dict):
                    entry = entry.get("lemma")
                if not isinstance(entry, str) or not entry:
                    continue
                entry = unicodedata.normalize("NFC", entry)
                spellings = idx.setdefault(strip_accents(entry.lower()), [])
                if entry not in spellings:
                    spellings.append(entry)
        self._compound_headword_indexes[source] = idx
        return idx

    def _compound_headword(self, skeleton: str, word: str,
                           base_lemma: str | None = None) -> str | None:
        """Accent a reconstructed compound skeleton from a dictionary.

        Compound reconstruction joins accent-stripped pieces
        (ολεθρο + τοκος), which is not a citation form. The skeleton is
        matched against the accent-stripped headword lists, and the first
        dictionary that has it supplies the accented spelling
        (ὀλεθροτόκος). Without a headword the reconstruction abstains
        (issue #101), since no dictionary vouches for the spelling and an
        Ancient prefix would lack its breathing.

        A caller that is allowed to guess passes *base_lemma*: monotonic
        Modern Greek input then keeps the accent of its monotonic base
        lemma (μονο + ακόρεστος -> μονοακόρεστος). That spelling is
        unverified, and more often a non-word than a word (βραβευ + σός,
        issue #103), so a no-guess caller leaves *base_lemma* out.
        """
        def _polytonic(s: str) -> bool:
            return any(ord(ch) in _POLYTONIC_STRIP | _POLYTONIC_TO_ACUTE
                       for ch in unicodedata.normalize("NFD", s))
        polytonic = _polytonic(word)
        if self.lang == "grc" or (self.lang == "all" and polytonic):
            sources = self._COMPOUND_AG_HEADWORD_SOURCES
        elif self.lang == "el":
            sources = self._COMPOUND_MG_HEADWORD_SOURCES
        else:
            sources = (self._COMPOUND_MG_HEADWORD_SOURCES
                       + self._COMPOUND_AG_HEADWORD_SOURCES)
        upper = is_capitalized(word)
        for source in sources:
            spellings = self._compound_headword_index(source).get(skeleton)
            if not spellings:
                continue
            # A common word's compound is lowercase; only a capitalized
            # input may take a proper-noun headword.
            matching = [h for h in spellings if is_capitalized(h) == upper]
            if not matching and upper:
                matching = spellings
            if not matching:
                continue
            if len(matching) == 1:
                return matching[0]

            def _count(h):
                info = self.attestation(h) or {}
                return info.get("total", 0) or 0
            return min(matching, key=lambda h: (-_count(h), h))
        if (self.lang in ("el", "all") and not polytonic and base_lemma
                and skeleton.isalpha() and not _polytonic(base_lemma)
                and not is_capitalized(base_lemma)):
            base = unicodedata.normalize("NFC", base_lemma.lower())
            stripped_base = strip_accents(base)
            if skeleton.endswith(stripped_base) and base != stripped_base:
                return skeleton[:len(skeleton) - len(stripped_base)] + base
        return None

    def _decompose_compound(self, word: str, *,
                            guess: bool = True) -> str | None:
        """Try to lemmatize an unknown compound by splitting at linking vowels.

        Greek compounds: first-stem + linking-vowel (ο/ι/υ) + second-element.
        The second element inflects like its standalone form. Strategy: split
        at each linking vowel (left to right, preferring longer bases), look
        up the base, and reconstruct prefix + base_lemma.

        Returns the reconstructed compound lemma, or None. With
        ``guess=False`` only a lemma that a dictionary lists is returned.
        """
        lower = word.lower()
        stripped = strip_accents(lower)

        if len(stripped) < self._MIN_COMPOUND_PREFIX + self._MIN_COMPOUND_BASE + 1:
            return None

        # Try split points left to right (longest base first = most reliable)
        for i in range(self._MIN_COMPOUND_PREFIX - 1,
                       len(stripped) - self._MIN_COMPOUND_BASE):
            if stripped[i] not in self._COMPOUND_LINK_VOWELS:
                continue

            prefix = stripped[:i + 1]   # includes linking vowel
            base = stripped[i + 1:]

            if len(base) < self._MIN_COMPOUND_BASE:
                continue

            # Look up the base in the lookup table
            base_lemma = self._lookup_grammatical_word(base)
            if not base_lemma:
                continue

            base_lemma_s = strip_accents(base_lemma.lower())

            # Guard: skip if lookup returned identity (no real lemmatization)
            if base_lemma_s == base:
                continue

            # Guard: skip if base_lemma is suspiciously short (false match)
            if len(base_lemma_s) < 2:
                continue

            # Guard: base_lemma should be shorter or equal to base
            # (lemmatization removes inflection, doesn't add length)
            if len(base_lemma_s) > len(base) + 2:
                continue

            if not self._compound_base_matches(base, base_lemma_s):
                continue

            # Reconstruct the compound and accent it from a dictionary
            # headword; an unaccented skeleton is not a citation form.
            compound = self._compound_headword(
                prefix + base_lemma_s, word, base_lemma if guess else None)
            if compound:
                return compound

        return None

    def _decompose_compound_all(self, word: str, *, guess: bool = True
                                ) -> list[tuple[str, str, str]]:
        """Return ALL valid compound decompositions.

        Returns list of (compound_lemma, base_lemma, prefix) triples,
        ordered by base length descending (longest base = most specific).
        Used by lemmatize_verbose for multi-candidate output. With
        ``guess=False`` only lemmas that a dictionary lists are returned.
        """
        lower = word.lower()
        stripped = strip_accents(lower)
        results = []
        seen = set()

        if len(stripped) < self._MIN_COMPOUND_PREFIX + self._MIN_COMPOUND_BASE + 1:
            return results

        for i in range(self._MIN_COMPOUND_PREFIX - 1,
                       len(stripped) - self._MIN_COMPOUND_BASE):
            if stripped[i] not in self._COMPOUND_LINK_VOWELS:
                continue

            prefix = stripped[:i + 1]
            base = stripped[i + 1:]

            if len(base) < self._MIN_COMPOUND_BASE:
                continue

            base_lemma = self._lookup_grammatical_word(base)
            if not base_lemma:
                continue

            base_lemma_s = strip_accents(base_lemma.lower())
            if base_lemma_s == base or len(base_lemma_s) < 2:
                continue
            if len(base_lemma_s) > len(base) + 2:
                continue
            if not self._compound_base_matches(base, base_lemma_s):
                continue

            compound = self._compound_headword(
                prefix + base_lemma_s, word, base_lemma if guess else None)
            if compound and compound not in seen:
                seen.add(compound)
                results.append((compound, base_lemma, prefix))

        return results

    # ---- Feature: Particle/enclitic suffix stripping ----

    def _strip_particle_suffix(self, word: str) -> str | None:
        """Try stripping enclitic particles (-per, -ge, -de, deictic -i).

        Only fires as a fallback when the full form is not in the lookup
        table. Returns the base form's lemma (not the base form itself),
        or None if no valid stripping is found.

        Suffix priority: -per (safest, most common), then -ge, then -de.
        Deictic -i is only stripped from known demonstrative pronoun stems.
        For -de, only strips if the resulting base form is found in lookup
        (since -de is also a real word ending in many cases).
        """
        lower = word.lower()
        stripped_word = strip_accents(lower)

        # Try -per, -ge, -de suffixes
        for suffix in _PARTICLE_SUFFIXES:
            suffix_stripped = strip_accents(suffix)
            if not stripped_word.endswith(suffix_stripped):
                continue
            base = stripped_word[:-len(suffix_stripped)]
            if len(base) < 2:
                continue

            # For the original (accented) form, try cutting the suffix
            # from the actual word. We try multiple cuts since accent
            # position may shift. Order matters: accented variants are
            # tried before the accent-stripped fallback, since accented
            # forms disambiguate homographs like ἔμοι (dat. of ἐγώ) vs
            # εμοι (which the lookup also resolves to a verb of ἐμέω).
            # Use a list with explicit dedup rather than a set, since
            # set iteration order is hash-seed-dependent.
            candidates: list[str] = []
            def _add(c: str) -> None:
                if c and c not in candidates:
                    candidates.append(c)

            # Accented variants first (more specific)
            for suf_len in (len(suffix), len(suffix_stripped)):
                if len(lower) > suf_len:
                    _add(lower[:-suf_len])
            # Accent-stripped fallback
            _add(base)

            # After stripping a suffix, a medial sigma (σ) that was
            # word-internal may now be at the end of the base. Convert
            # to final sigma (ς) so lookup matches dictionary forms
            # (e.g. ὅσπερ -> ὅσ needs to find ὅς). Append in the same
            # priority order as the source candidates.
            for c in list(candidates):
                if c.endswith("σ"):
                    _add(c[:-1] + "ς")

            for candidate in candidates:
                lemma = self._lookup_word(candidate)
                if lemma:
                    return lemma

        # Deictic -i: only strip from demonstrative pronoun forms
        if stripped_word.endswith("ι") and len(stripped_word) > 2:
            base = stripped_word[:-1]
            if base in _DEICTIC_STEMS:
                # Try looking up the base (without the deictic -i)
                # Try the accent-stripped base and original minus last char
                for candidate in (base, lower[:-1] if len(lower) > 1 else ""):
                    if not candidate:
                        continue
                    lemma = self._lookup_word(candidate)
                    if lemma:
                        return lemma

        return None

    # ---- Feature: Verb morphology stripping (augment, reduplication) ----

    # Temporal augment mappings: augmented vowel -> original stem-initial vowel(s)
    _TEMPORAL_AUGMENT = {
        "η": ["ε", "α"],     # η- can be augment of ε- or α-initial stems
        "ω": ["ο"],           # ω- augment of ο-initial stems
        "ηυ": ["ευ"],         # ηυ- augment of ευ-initial stems
        "ῃ": ["ε", "α"],     # with iota subscript
        "ει": ["ε"],          # spurious diphthong augment
        "ηι": ["αι"],         # ηι- augment of αι-initial stems (rare)
    }

    # An oblique case of a -μενος participle (accent-stripped), and the
    # thematic or contract vowel that marks a present, future or aorist
    # middle participle. Neither kind ever carries an augment.
    _PARTICIPLE_OBLIQUE = re.compile(
        r"μεν(?:ου|ω|ον|ε|οι|ων|οις|ους|η|ης|ην|αι|αις|ας|α|οιν|αιν)$")
    _UNAUGMENTED_PARTICIPLE = re.compile(
        r"(?:ο|ου|ω|α)μεν(?:ος|ου|ω|ον|ε|οι|ων|οις|ους|η|ης|ην|αι|αις|ας|α|οιν|αιν)$")

    def _participle_nominative_lemma(self, word: str) -> str | None:
        """Resolve an oblique -μενος participle through its nominative.

        Participles are usually listed in the masculine nominative singular
        only, so ἐλευσομένῳ has no row while ἐλευσόμενος maps to ἔρχομαι.
        The nominative is rebuilt with the recessive accent (ἐλευσόμενος),
        then with the perfect's -μένος (ἐσταλμένος). It must be an exact
        lookup key whose reading is a verb; a self-map is not evidence.
        """
        lower = word.lower()
        stripped = strip_accents(lower)
        match = self._PARTICIPLE_OBLIQUE.search(stripped)
        if not match or len(stripped) < 7:
            return None
        nfd = unicodedata.normalize("NFD", lower)
        letters = [i for i, ch in enumerate(nfd) if not unicodedata.combining(ch)]
        if len(letters) != len(stripped):
            return None
        # Cut before μεν, keeping breathings and subscripts but no accent.
        stem = "".join(ch for ch in nfd[:letters[len(stripped) - len(match.group(0))]]
                       if ch not in "\u0301\u0300\u0342")
        nominatives = []
        for i in range(len(stem) - 1, -1, -1):
            if stem[i] in "αεηιουω":
                j = i + 1
                while j < len(stem) and unicodedata.combining(stem[j]):
                    j += 1
                nominatives.append(stem[:j] + "\u0301" + stem[j:] + "μενος")
                break
        nominatives.append(stem + "μένος")
        lemmas = []
        for nominative in nominatives:
            nominative = unicodedata.normalize("NFC", nominative)
            if (self._lookup.get(nominative) is None
                    and self._ag_lookup.get(nominative) is None):
                continue
            lemma = self._lookup_grammatical_word(nominative, verb_only=True)
            if lemma and not _is_self_map(nominative, lemma) and lemma not in lemmas:
                lemmas.append(lemma)
        if len(lemmas) > 1:
            return self._participle_owner_by_cells(stem, lemmas)
        return lemmas[0] if lemmas else None

    # Endings of a -μενος participle as the POS tables spell them: with the
    # accent on -μέν-, and with the recessive accent where the ultima allows it.
    _PARTICIPLE_CELL_ENDINGS = ("ος", "ου", "ῳ", "ον", "ε", "οι", "ων", "οις",
                                "ους", "η", "ης", "ῃ", "ην", "αι", "αις", "ας",
                                "α", "ᾳ")
    _RECESSIVE_PARTICIPLE_ENDINGS = ("ος", "ον", "ε", "οι", "α", "αι")

    def _participle_owner_by_cells(self, stem: str, lemmas: list[str]) -> str:
        """The verb whose participle this is when two nominatives compete.

        ἱερώμενος (ἱεράομαι) and ἱερωμένος (ἱερόω) share their oblique cells,
        so ἱερωμέναις fits both. The verb the POS tables name for more cells
        of this participle wins; on a tie the first nominative stands, as
        before (#151). *stem* is the NFD stem without accent.
        """
        self._load_fallback_readings()
        spellings = {stem + "μέν" + ending
                     for ending in self._PARTICIPLE_CELL_ENDINGS}
        for index in range(len(stem) - 1, -1, -1):
            if stem[index] in "αεηιουω":
                end = index + 1
                while end < len(stem) and unicodedata.combining(stem[end]):
                    end += 1
                accented = stem[:end] + "\u0301" + stem[end:]
                spellings.update(accented + "μεν" + ending
                                 for ending in self._RECESSIVE_PARTICIPLE_ENDINGS)
                break
        cells = dict.fromkeys(lemmas, 0)
        for spelling in spellings:
            readings = self._fallback_exact_readings.get(
                unicodedata.normalize("NFC", spelling), {})
            verbs = readings.get("VERB", set()) | readings.get("AUX", set())
            for lemma in cells:
                cells[lemma] += lemma in verbs
        best = max(cells.values())
        return next(lemma for lemma in lemmas if cells[lemma] == best)

    def _aorist_participle_lemma(self, word: str) -> str | None:
        """Resolve a nominative aorist participle through its feminine.

        A verb known from the corpora may have εὐτονήσασα and εὐτονῆσαι but
        no row for the masculine εὐτονήσας (#151). The feminine must be a
        lookup row that reads a verb.
        """
        if self.lang == "el":
            return None
        lower = unicodedata.normalize("NFC", word.lower())
        if not lower.endswith("σας") or len(lower) < 6:
            return None
        lemma = self._ag_lookup.get(lower[:-2] + "ασα") if self._ag_lookup else None
        if lemma and self._is_verb_shaped_lemma(lemma):
            return lemma
        return None

    def _strip_verb_morphology(self, word: str) -> str | None:
        """Try decomposing unknown verb forms by stripping augment or reduplication.

        Handles three patterns:
        1. Syllabic augment (ε-prefix): strip leading ε- and look up
        2. Temporal augment (η- for ε-stems, ω- for ο-stems, etc.)
        3. Reduplication (λε-λυ- type perfect forms): strip the reduplicated
           consonant+vowel prefix

        The transformed form must have a supported verbal reading or resolve
        to a known verb lemma. A noun homograph is not evidence for a verb.
        """
        lower = word.lower()
        stripped = strip_accents(lower)

        # Skip very short words
        if len(stripped) < 4:
            return None

        # A present, future or aorist participle has no augment (ἐλευσομένῳ
        # is not an augmented λευσομένῳ); only reduplication can apply.
        participle = bool(self._UNAUGMENTED_PARTICIPLE.search(stripped))

        # 1. Syllabic augment: strip leading ε- (most common augment)
        # The augmented form starts with ε and the unaugmented stem
        # should be a recognizable verb form.
        nfd = unicodedata.normalize("NFD", lower)
        base_chars = [ch for ch in nfd if unicodedata.category(ch) != "Mn"]
        if not participle and base_chars and base_chars[0] == "ε":
            # Strip the initial epsilon (with any diacritics)
            # Find the end of the first character cluster (base + combiners)
            idx = 0
            for i, ch in enumerate(nfd):
                if unicodedata.category(ch) != "Mn":
                    if idx > 0:
                        break
                    idx = 1
                    cut_point = i + 1
                else:
                    cut_point = i + 1
            remainder = unicodedata.normalize("NFC", nfd[cut_point:])
            if len(remainder) >= 3:
                lemma = self._lookup_grammatical_word(remainder, verb_only=True)
                if lemma and not _is_self_map(remainder, lemma):
                    return lemma

        # 2. Temporal augment: try replacing augmented initial vowel
        # with the original stem vowel(s)
        for augmented, originals in self._TEMPORAL_AUGMENT.items():
            aug_stripped = strip_accents(augmented)
            if participle or not stripped.startswith(aug_stripped):
                continue
            remainder_stripped = stripped[len(aug_stripped):]
            if len(remainder_stripped) < 3:
                continue
            for original in originals:
                candidate = original + remainder_stripped
                lemma = self._lookup_grammatical_word(candidate, verb_only=True)
                # The lemma must begin with the restored vowel: εἰκόνισεν
                # is not ἐκόνισεν, itself an augmented form of κονίω.
                if (lemma and not _is_self_map(candidate, lemma)
                        and strip_accents(lemma.lower()).startswith(original)):
                    return lemma

        # 3. Reduplication: perfect forms have consonant+vowel prefix
        # that duplicates the stem-initial consonant. Pattern: CV-stem
        # where C matches the first consonant of the stem.
        if len(stripped) >= 5:
            # Check if first two chars look like reduplication (C + ε/vowel)
            first_char = stripped[0]
            if first_char.isalpha() and stripped[1] in "εα":
                rest = stripped[2:]
                # The stem should start with the same consonant
                if rest and rest[0] == first_char:
                    lemma = self._lookup_grammatical_word(rest, verb_only=True)
                    if lemma and not _is_self_map(rest, lemma):
                        return lemma

        return None

    # ---- Feature: Article-agreement disambiguation ----

    def _rank_by_article_agreement(self, candidates: list[LemmaCandidate],
                                   prev_word: str | None) -> list[LemmaCandidate]:
        """Re-rank candidates using article gender/number agreement.

        If prev_word is a Greek article, boost candidates whose lemma
        gender matches the article's gender. This only re-ranks, never
        excludes candidates.

        Args:
            candidates: List of LemmaCandidate objects to rank.
            prev_word: The preceding word (may be an article or None).

        Returns:
            Re-ranked list of candidates (same elements, possibly reordered).
        """
        if not prev_word or not candidates or len(candidates) < 2:
            return candidates

        # Normalize the article form (try exact, then grave-to-acute)
        features = _ARTICLE_FEATURES.get(prev_word)
        if not features:
            features = _ARTICLE_FEATURES.get(grave_to_acute(prev_word))
        if not features:
            return candidates

        article_gender = features[0]  # "m", "f", or "n"

        def _guess_gender(lemma: str) -> str | None:
            """Guess gender from lemma ending (heuristic)."""
            # Check longest suffixes first
            for suffix, gender in sorted(_LEMMA_GENDER_HINTS.items(),
                                         key=lambda x: -len(x[0])):
                if lemma.endswith(suffix):
                    return gender
            return None

        # Score each candidate: 0 if gender matches, 1 if not
        def _gender_score(c: LemmaCandidate) -> int:
            g = _guess_gender(strip_accents(c.lemma.lower()))
            if g == article_gender:
                return 0
            return 1

        # Stable sort: preserve original order within same gender score
        candidates_sorted = sorted(candidates,
                                   key=lambda c: (_gender_score(c), c.proper, -c.score))
        return candidates_sorted

    def _is_skipped_ordinal(self, word: str) -> bool:
        return (getattr(self, "_skip_numeral_ordinals", False)
                and is_numeral_ordinal(word))

    def _without_skipped_ordinals(self, words, results):
        if not getattr(self, "_skip_numeral_ordinals", False):
            return results
        return [None if is_numeral_ordinal(word) else result
                for word, result in zip(words, results)]

    def lemmatize(self, word: str, *, attested_only: bool = False,
                  guess: bool = True) -> str | None:
        """Lemmatize a single Greek word.

        Resolution order:
          1. Article/pronoun resolution (if resolve_articles=True)
          2. Crasis table (small, hand-curated)
          3. Lookup table (instant, 5M+ forms)
          4. Elision expansion (strip mark, try vowels against lookup)
          5. Particle suffix stripping (-per, -ge, -de, deictic -i)
          6. Verb morphology stripping (augment, reduplication)
          7. Normalizer (orthographic variants)
          8. Compound decomposition (split at linking vowel, look up base)
          9. Model with beam search + headword filter

        If a convention is set, the output lemma is remapped accordingly.

        When ``attested_only=True``, the input surface form must occur in the
        corpus (exact NFC polytonic, or via elision expansion); otherwise this
        returns ``None`` rather than a guessed lemma. Digits and empty input are
        passed through unchanged. Needs ``form_profile.db`` (download it with
        ``python -m dilemma download``).

        When ``guess=False``, the transformer fallback (step 9) is skipped and
        a word that nothing structured can resolve returns ``None`` instead of
        the model's unconstrained prediction or an echo of the input. For a
        dictionary-lookup caller a wrong lemma is worse than no lemma (a
        guessed headword puts a false definition in front of the reader), and
        an input echoed back is indistinguishable from a successful identity
        lemmatization - ``guess=False`` is the way to say "I don't know".
        Non-Greek tokens and NON-LEXICAL tokens (apparatus marks, numerals,
        refs; see ``classify_nonlexical``) still pass through unchanged: those
        are classifications, not lemma guesses. For a per-candidate confidence
        signal, use ``lemmatize_verbose`` and its ``score``/``source`` fields.
        """
        if not word:
            return word
        # Pass through tokens with no Greek letters (punctuation, numerals,
        # Latin) unchanged - they have no Greek lemma. Checking word.isdigit()
        # alone missed punctuation like "." / "," / ";", which then fell through
        # to the model and produced garbage lemmas (. -> ὅ, , -> Ϛ).
        if not any(0x0370 <= ord(c) <= 0x03ff or 0x1f00 <= ord(c) <= 0x1fff
                   for c in word):
            return word
        # NFC-normalize: some sources (e.g. DBBE) emit decomposed (NFD) forms
        # that won't match the NFC lookup keys otherwise.
        word = unicodedata.normalize("NFC", word)
        # Lunate sigma (ϲ) -> standard σ/ς, so editions that use it still match.
        word = to_standard_sigma(word)

        if self._is_skipped_ordinal(word):
            return None
        # Attested-only gate: drop forms that don't occur in the corpus.
        if attested_only and self._resolve_attested_form(word) is None:
            return None
        # ῤῥ is an editorial spelling of ρρ (after the gate, which matches
        # the corpus spelling as printed).
        word = to_plain_double_rho(word)

        # Resolve articles/pronouns to canonical lemma
        closed = self._resolve_closed_class(word)
        if closed is not None:
            return self._apply_convention(closed)

        # Check crasis first (before lookup, since crasis forms are
        # Wiktionary headwords that self-map in the lookup)
        from .crasis import resolve_crasis
        crasis_result = resolve_crasis(word) or resolve_crasis(to_monotonic(word))
        if crasis_result is not None:
            return self._apply_convention(crasis_result)

        # Lookup: exact -> lowercase -> monotonic -> accent-stripped.
        # Two guards divert an elided form to the elided-entry / expander
        # path instead of accepting a junk lookup hit:
        #   * a result that is itself an elided form (a self-map like
        #     ἀλλ᾽->ἀλλ᾽, which build_lookup_db.py now drops but older DBs
        #     still carry) - a real headword never ends in a spacing
        #     apostrophe/koronis;
        #   * a word ending in combining psili on a CONSONANT (γ̓, μ̓),
        #     which is always an elision apostrophe and whose accent-
        #     stripped variant false-matches letter-name headwords (γ̓->Γ).
        # Both tests deliberately avoid _strip_elision, whose combining-mark
        # rule misfires on diphthongs that carry a breathing on the second
        # vowel (οὐ, εἰ) and would wrongly discard those correct lookups.
        #
        # Explicit numerals (͵α, ιβʹ, κζ’) are classified NON-LEXICAL before
        # any of this machinery runs: their thousands/keraia/apostrophe marks
        # would otherwise be stripped or vowel-expanded into a word.
        if _is_explicit_numeral(word):
            return word
        # A psili after the final vowel's own marks is an elision apostrophe
        # (αἶ̓, οἷ̓): the elided entry and the expander come first, as in
        # lemmatize_batch(), since the loose lookup keys drop the mark and
        # read the bare letters (αι -> αἴ).
        if _is_marked_vowel_psili_elision(word):
            elided = self._lookup_elided(word) or self._expand_elision(word)
            if elided:
                return self._apply_convention(elided)
        if not _is_consonant_psili_elision(word):
            lemma = self._lookup_word(word)
            if lemma and not _is_elided_junk_value(lemma):
                return self._apply_convention(lemma)
            if lemma:
                # The junk value marks the KEY as a bare elided stem whose
                # apostrophe was tokenized off (key ἀλλ -> value "ἀλλ'"):
                # restore the mark and run the elided machinery, so the stem
                # gets its real lemma (ἀλλ -> ἀλλά, οὐδ -> οὐδέ) instead of
                # falling to the model (εἴθ -> ἔρχομαι) or echoing junk.
                bridged = (self._lookup_elided(word + _KORONIS)
                           or self._expand_elision(word + _KORONIS))
                if bridged:
                    return self._apply_convention(bridged)

        # Exact elided entry (ὅτ᾽->ὅτε, μ᾽->ἐγώ): a corpus-derived lemma keyed
        # under the canonical U+1FBD mark, so any apostrophe codepoint resolves
        # identically. Runs before the frequency-ranked expander so the elided
        # word's real lemma wins over the common homograph.
        exact_elided = self._lookup_elided(word)
        if exact_elided:
            return self._apply_convention(exact_elided)

        # Elision expansion (after lookup, so known words like εἰ/οὐ
        # aren't falsely caught by smooth-breathing-as-elision)
        elision_lemma = self._expand_elision(word)
        if elision_lemma:
            return self._apply_convention(elision_lemma)

        # Particle suffix stripping (after lookup, before model)
        particle_lemma = self._strip_particle_suffix(word)
        if particle_lemma:
            return self._apply_convention(particle_lemma)

        # Verb morphology stripping (augment, reduplication - after
        # particle stripping, before model)
        verb_morph_lemma = (self._participle_nominative_lemma(word)
                            or self._aorist_participle_lemma(word)
                            or self._strip_verb_morphology(word))
        if verb_morph_lemma:
            return self._apply_convention(verb_morph_lemma)

        # Normalizer: try orthographic variants against lookup
        if self._normalizer:
            for candidate in self._normalizer.normalize(word):
                lemma = self._lookup_grammatical_word(
                    candidate, normalization=True, source_word=word)
                if lemma and not _is_elided_junk_value(lemma):
                    return self._apply_convention(lemma)

        # Attested-key-gated orthographic fallbacks: editorial/manuscript
        # surface variants (dropped iota-subscript θεῷ->θεῶ, movable-nu,
        # dieresis). Accept a variant only if it itself resolves in the
        # lookup, so this never invents a lemma.
        for candidate in self._ortho_fallback_variants(word):
            lemma = self._lookup_word(candidate)
            if lemma and not _is_elided_junk_value(lemma):
                return self._apply_convention(lemma)

        # Non-lexical classification (BEFORE the expensive model): apparatus
        # marks (γρ), Greek numerals, editorial refs, abbreviations, and
        # vowel-less sigla are not words, so don't send them to the transformer
        # (which manufactures spurious lemmas like δτι->δ). Return the token
        # unchanged, matching the non-Greek passthrough; callers distinguish a
        # non-word from a failed real word via is_lexical()/classify_nonlexical()
        # or lemmatize_verbose()'s "nonlexical" source.
        if _classify_nonlexical(word) is not None:
            return word

        # Fall back to model (skipped with guess=False: the transformer is
        # an unconstrained guesser, so a no-guess call must not use it).
        pred = None
        if guess:
            try:
                self._load_model()
                pred = self._predict([word])[0]
            except (RuntimeError, IndexError, ImportError, FileNotFoundError):
                # Model inference can fail on unusual inputs (empty tensors,
                # single-char forms, etc.) or if no backend is installed.
                # Fall through to identity fallback.
                return self._apply_convention(word)

        # Fallback strategies when the model returns identity (model
        # couldn't lemmatize) or was skipped (guess=False). Uses
        # accent-stripped comparison since the model may return slight
        # accent variants of the input.
        if pred is None or strip_accents(pred.lower()) == strip_accents(word.lower()):
            # 1. Light Byzantine normalization (final sigma, iota
            #    adscript, breathing, hyphen - very precise, few false
            #    positives)
            byz = self._normalize_byzantine(word, use_normalizer=False)
            if byz:
                return self._apply_convention(byz)

            # 2. Compound decomposition (linking vowel split). Without a
            #    dictionary headword the reconstruction is a guess.
            compound = self._decompose_compound(word, guess=guess)
            if compound:
                return self._apply_convention(compound)

            # 3. Prefix stripping (known prefixes + base lookup)
            prefix_hit = self._prefix_strip_lookup(word)
            if prefix_hit:
                return self._apply_convention(prefix_hit)

            # 4. Heavy Byzantine normalization (uses the orthographic
            #    normalizer to handle itacism, geminate simplification,
            #    etc. - more aggressive, may produce false positives)
            byz_heavy = self._normalize_byzantine(word, use_normalizer=True)
            if byz_heavy:
                return self._apply_convention(byz_heavy)

        if pred is None:
            # guess=False and nothing structured resolved: say "I don't know"
            # rather than echoing the input as if it were a lemma.
            return None
        return self._apply_convention(pred)

    def _ortho_fallback_variants(self, word: str):
        """Yield editorial/manuscript orthographic spelling variants of a
        surface form, to try against the lookup as a last resort before the
        model. These are alternate spellings, not guesses: the caller accepts
        a variant only if it itself resolves in the lookup.

        Covers a dropped iota-subscript on a long vowel (the dat. θεῷ written
        θεῶ; preserved-vs-not is safe because we only accept resolving
        variants), a movable-nu (-εν/-ιν/-σιν), a stray combining dieresis,
        and digamma (ϝ).
        """
        seen = {word}
        nfd = unicodedata.normalize("NFD", word)

        # Digamma (ϝ/Ϝ): a real archaic letter, but in OCR'd or hand-keyed
        # text most often noise for ν (ϝέκταρ = νέκταρ, πρόσθεϝ = πρόσθεν).
        # Try the ν-replacement first, then dropping it entirely (genuine
        # digamma spellings are lemmatized without it: ϝάναξ -> ἄναξ).
        if "ϝ" in word or "Ϝ" in word:
            for v in (word.replace("ϝ", "ν").replace("Ϝ", "Ν"),
                      word.replace("ϝ", "").replace("Ϝ", "")):
                if v and v not in seen:
                    seen.add(v)
                    yield v

        # Movable-nu: drop a final ν after ε/ι (3sg -ε(ν), dat./3pl -σι(ν)),
        # never after the case-ending vowels -ων/-ην/-αν.
        if len(word) >= 2 and word[-1] == "ν":
            base = strip_accents(word[-2].lower())
            if base in ("ε", "ι"):
                v = word[:-1]
                if v not in seen:
                    seen.add(v)
                    yield v

        # Strip a combining dieresis (U+0308): ἀρχϊεράρχης -> ἀρχιεράρχης.
        if "̈" in nfd:
            v = unicodedata.normalize("NFC", nfd.replace("̈", ""))
            if v not in seen:
                seen.add(v)
                yield v

        # Restore a dropped iota-subscript: add U+0345 after each long vowel
        # (α/η/ω) base + its combining marks, one position at a time.
        chars = list(nfd)
        for i, c in enumerate(chars):
            if c in ("α", "η", "ω"):
                j = i + 1
                while j < len(chars) and unicodedata.combining(chars[j]):
                    j += 1
                if "ͅ" not in chars[i:j]:
                    v = unicodedata.normalize(
                        "NFC", "".join(chars[:j]) + "ͅ" + "".join(chars[j:]))
                    if v not in seen:
                        seen.add(v)
                        yield v

    def _ancient_pos_context(self, word: str, upos: str | None) -> bool:
        if (not upos or self.lang == "el"
                or getattr(self, "_convention_name", None) == "triantafyllidis"):
            return False
        if not any(unicodedata.category(char).startswith("L")
                   and (0x0370 <= ord(char) <= 0x03FF or 0x1F00 <= ord(char) <= 0x1FFF)
                   for char in word):
            return False
        polytonic = any(ord(char) in _POLYTONIC_STRIP | _POLYTONIC_TO_ACUTE
                        for char in unicodedata.normalize("NFD", word))
        return self.lang == "grc" or (self.lang == "all" and polytonic)

    def _pos_rejections(self, word: str, upos: str | None) -> frozenset[str]:
        """Bind the source-target veto for explicit Ancient Greek POS calls."""
        if not self._ancient_pos_context(word, upos):
            return frozenset()
        cached = getattr(self, "_pos_rejected_targets", None)
        if cached is not None:
            return cached
        try:
            audit_stat = TREEBANK_POS_AUDIT_PATH.stat()
            heads_stat = GRC_LEXICAL_HEADS_PATH.stat()
        except OSError as error:
            raise ValueError("Ancient Greek POS data mismatch: refresh "
                "treebank_pos_lookup.audit.json and grc_lexical_heads.json "
                "together with python -m dilemma download --no-tagger") from error
        rejected = _bound_pos_rejections(
            str(TREEBANK_POS_AUDIT_PATH), audit_stat.st_size, audit_stat.st_mtime_ns,
            str(GRC_LEXICAL_HEADS_PATH), heads_stat.st_size, heads_stat.st_mtime_ns)
        self._pos_rejected_targets = rejected
        return rejected

    def _checked_pos_result(self, word: str, upos: str | None,
                            lemma: str | None) -> str | None:
        if lemma is not None and grave_to_acute(unicodedata.normalize("NFC", lemma)) \
                in self._pos_rejections(word, upos):
            return None
        return lemma

    def _checked_pos_candidates(self, word, upos, candidates):
        rejected = self._pos_rejections(word, upos)
        return [candidate for candidate in candidates
                if grave_to_acute(unicodedata.normalize("NFC", candidate.lemma))
                not in rejected]

    def _checked_pos_table_lookup(self, word, upos):
        lemma = self._pos_table_lookup(word, upos)
        checked = self._checked_pos_result(word, upos, lemma)
        if lemma is not None and checked is None:
            # A rejected higher-priority entry cannot hide a unique retained
            # reading from another exact grammatical source. Preserve ties.
            targets = {candidate for candidate in self._grammatical_readings(word).get(upos, ())
                       if self._checked_pos_result(word, upos, candidate) is not None}
            if len(targets) == 1:
                return next(iter(targets))
        return checked

    def _prefer_exact_pos_citation(self, word, upos, candidate, pos_lemma):
        if self.lang == "el" and word.lower() == candidate.lower():
            readings = (getattr(self, "_pos_lookup", {}).get(word)
                        or getattr(self, "_pos_lookup", {}).get(word.lower()) or {})
            # An independently defined conjunction can share its bare
            # spelling with an accented noun. Honor an explicit noun tag
            # when the table records both owners, rather than treating
            # their accents as interchangeable citations.
            if (readings.get(upos) == pos_lemma
                    and any(pos != upos and lemma.lower() == candidate.lower()
                            for pos, lemma in readings.items())):
                return pos_lemma
        # A self-map that differs from a nominal POS target only by an iota
        # subscript is that noun's dative (σιγῇ beside σιγή, βίᾳ beside βία),
        # not an independent word; the adverb row still answers under ADV.
        if (upos in ("NOUN", "PROPN")
                and self._ancient_pos_context(word, upos)
                and grave_to_acute(word).lower() == grave_to_acute(candidate).lower()
                and (unicodedata.normalize("NFD", candidate).count("\u0345")
                     > unicodedata.normalize("NFD", pos_lemma).count("\u0345"))):
            return pos_lemma
        # An exact lookup self-map can be an independent word rather than an
        # inflection of the differently accented POS target (οὖ versus οὐ).
        if (self._ancient_pos_context(word, upos)
                and grave_to_acute(word).lower() != grave_to_acute(candidate).lower()
                and grave_to_acute(word).lower() != grave_to_acute(pos_lemma).lower()):
            return pos_lemma
        return candidate

    # Tags one Ancient lemma can carry across tagsets and POS sources: a
    # proper noun or substantivized adjective beside NOUN, a pronominal
    # adjective (ἄλλος, σός) or a numeral (εἷς) beside PRON or DET, an
    # adverbial neuter beside ADJ, a particle beside ADV, the two kinds of
    # conjunction, an auxiliary beside VERB.
    _POS_COMPATIBLE_TAGS = (
        frozenset({"NOUN", "PROPN", "ADJ"}),
        frozenset({"ADJ", "PRON", "DET", "NUM"}),
        frozenset({"ADJ", "ADV"}),
        frozenset({"ADV", "PART"}),
        frozenset({"CCONJ", "SCONJ"}),
        frozenset({"VERB", "AUX"}),
    )
    # Tags of the small closed word classes. Modern Greek function words
    # (να, με, σε, σαν, ας, στους) carry them, and no verb form does.
    _CLOSED_CLASS_TAGS = frozenset(
        {"PRON", "DET", "ADP", "CCONJ", "SCONJ", "PART", "INTJ"})
    # The last key variant of _pos_table_hit that is the word's own
    # spelling (as written, lowercased, grave folded to acute).
    _OWN_SPELLING_POS_VARIANT = 2
    _TONAL_ACCENTS = frozenset({0x0301, 0x0300, 0x0342})

    @classmethod
    def _pos_tags_compatible(cls, first: str, second: str) -> bool:
        return first == second or any(
            first in group and second in group
            for group in cls._POS_COMPATIBLE_TAGS)

    def _is_ancient_headword(self, lemma: str) -> bool:
        """Whether an Ancient dictionary or lexical-head list has *lemma*."""
        if lemma in self._get_headword_set("lsj"):
            return True
        if self._ancient_lexical_heads is None:
            self._ancient_lexical_heads = _lexical_heads("grc")
        normalized = _without_quantity_marks(sanitize_form(lemma)).replace("-", "")
        return normalized in self._ancient_lexical_heads

    def _restore_breathing(self, word: str, lemma: str) -> str | None:
        """The lemma of a loose-key lookup hit, with the breathing *word* has.

        A form written with a breathing that has no row of its own can
        reach a row through its monotonic or accent-stripped key, and the
        rows there are often Modern Greek (ευσεβεστατη -> ευσεβής). When
        such a lemma lacks the breathing the form has, the Ancient headword
        with the same letters is returned instead: the form's breathing
        first, then the other one (always rough on rho), each with the
        accent as written and with an acute turned circumflex. A spelling
        the Ancient lookup maps to itself counts with its accent unchanged.
        A lemma that is itself an Ancient head written without breathings
        (the Sophocles lexicon's οφφίκιον) takes the form's breathing.
        A capitalized lemma (a name only Modern Greek Wiktionary lists,
        Ραχίλ) is returned as it is for a capitalized form; a lowercase
        form is not that name (ἀνάβη is not the surname Ανάβης, nor the
        fragment εἴπ the abbreviation ΕΙΠ). With ``lang="all"``, a Modern
        Greek lexical head is returned as it is too: polytonic vernacular Greek
        reads as Modern there (1.3.24, #104), so ἀκόμη keeps ακόμη and
        ἀξίζει αξίζω. A Modern ``-ώ`` head that spells one Ancient contract
        verb gives that verb first (ἀστοχήσει: αστοχώ, ἀστοχέω). Otherwise
        there is no Ancient lemma with those letters: None (ἀκόμη with
        ``lang="grc"``).
        A lemma that keeps a breathing, or a form without one, is returned
        unchanged.
        """
        if _initial_breathing(word) not in (_SMOOTH_BREATHING, _ROUGH_BREATHING) \
                or _initial_breathing(lemma) != "":
            return lemma
        if unicodedata.normalize("NFD", lemma.lower())[0] == "ρ":
            marks = (_ROUGH_BREATHING,)
        else:
            own = _initial_breathing(word)
            marks = (own, _ROUGH_BREATHING if own == _SMOOTH_BREATHING
                     else _SMOOTH_BREATHING)
        tried = []
        for mark in marks:
            base = _with_initial_breathing(lemma, mark)
            if not base:
                continue
            spellings = [base]
            nfd = unicodedata.normalize("NFD", base)
            spellings.extend(
                unicodedata.normalize("NFC", nfd[:index] + "\u0342" + nfd[index + 1:])
                for index, ch in enumerate(nfd) if ch == "\u0301")
            for spelling in spellings:
                tried.append(spelling)
                if self._is_ancient_headword(spelling) or (
                        spelling == base and self._ag_lookup
                        and self._ag_lookup.get(spelling) == spelling):
                    return spelling
        if tried and self._is_ancient_headword(lemma):
            return tried[0]
        contract = self._ancient_contract_head(word, lemma, marks)
        if contract:
            return contract
        if is_capitalized(lemma) and is_capitalized(word):
            # A name has one citation form in both languages; one that
            # only Modern Greek lists (Ραχίλ for Ῥαχὶλ) is still the name.
            # A lowercase word is not a name the lookup lists without its
            # breathing: the loose keys of ἀνάβη, ὅτ and εἴπ are rows of
            # the surname Ανάβης and of the abbreviations Οτ and ΕΙΠ.
            return lemma
        if self.lang == "all":
            if self._vernacular_lexical_heads is None:
                self._vernacular_lexical_heads = _lexical_heads("el")
            if lemma in self._vernacular_lexical_heads:
                return lemma
        return None

    def _ancient_contract_head(self, word: str, lemma: str, marks) -> str | None:
        """The Ancient contract verb a Modern ``-ώ`` head spells (#151).

        αστοχώ is the Modern citation of ἀστοχέω, so a form written with a
        breathing (ἀστοχήσει) reads the Ancient verb. Modern ``-ώ`` heads
        include nouns and conjunctions (ηχώ, ενώ), so the word must show the
        lengthened vowel and σ of the verb's future or aorist after the
        head's stem. The stem must make exactly one Ancient headword in -έω,
        -άω or -όω; otherwise there is no telling which verb it is and None
        is returned.
        """
        if not lemma.endswith("ώ") or len(lemma) < 3:
            return None
        stem = strip_accents(lemma[:-1])
        rest = strip_accents(word.lower())
        if (not rest.startswith(stem)
                or rest[len(stem):len(stem) + 2] not in ("ησ", "ασ", "ωσ")
                or len(rest) <= len(stem) + 2):
            return None
        heads = set()
        for ending in ("έω", "άω", "όω"):
            for mark in marks:
                head = _with_initial_breathing(lemma[:-1] + ending, mark)
                if head and self._is_ancient_headword(head):
                    heads.add(head)
        return heads.pop() if len(heads) == 1 else None

    def _ancient_lookup_has_spelling(self, word: str) -> bool:
        """Whether the Ancient lookup holds *word* with its own accents.

        A reading reached only through the monotonic or accent-stripped
        variant, or only in the combined lookup, is a reading of some
        other spelling. The shipped SQLite lookup serves its Ancient view
        from rows stored for both languages as well, some of Modern
        origin, so a vernacular spelling held there counts as known and
        keeps the lookup's answer.
        """
        if reviewed_verb_grammar(word) is not None:
            return True
        table = self._ag_lookup
        if not table:
            return False
        acute = grave_to_acute(word)
        return any(
            self._normalize_grave_citation_lemma(table.get(variant) or "")
            for variant in dict.fromkeys(
                (word, word.lower(), acute, acute.lower())))

    _SYLLABLE_NUCLEUS = re.compile("[αεο]ι|υι|[αεηο]υ|[αεηιουω]")

    @classmethod
    def _without_enclitic_accent(cls, word: str) -> str:
        """Drop the second accent a following enclitic adds (ἄνθρωπός τις).

        Only the two patterns an enclitic produces count: an acute on the
        final syllable of a word accented on the antepenult, or of one
        with a circumflex on the penult (δῶρόν). In ἄεί the second accent
        is the word's own.
        """
        nfd = unicodedata.normalize("NFD", word)
        accents = [index for index, ch in enumerate(nfd)
                   if ord(ch) in cls._TONAL_ACCENTS]
        if len(accents) != 2 or nfd[accents[1]] != "\u0301":
            return word
        first, last = accents
        syllables_after = {"\u0301": 2, "\u0342": 1}.get(nfd[first])

        def nuclei(text: str) -> int:
            return len(cls._SYLLABLE_NUCLEUS.findall("".join(
                ch for ch in text.lower() if not unicodedata.combining(ch))))

        if nuclei(nfd[first + 1:last]) != syllables_after \
                or nuclei(nfd[last + 1:]):
            return word
        return unicodedata.normalize("NFC", nfd[:last] + nfd[last + 1:])

    @staticmethod
    def _monosyllable_by_synizesis(word: str) -> bool:
        """Whether *word* is one syllable, counting synizesis.

        A diphthong (αι ει οι υι αυ ευ ηυ ου) needs an unaccented first
        vowel and no diaeresis on the second. The word is a monosyllable
        with one nucleus, or with two adjacent nuclei of which the first is
        an unaccented ι, υ, ει or οι that does not open the word (μιὰ,
        ποιὸς, γριὰ). Hiatus is not synizesis: προϊόν, πρωΐας and Κάιν
        are not monosyllables, nor ξύλα.
        """
        units = []
        for ch in unicodedata.normalize("NFD", word.lower()):
            if unicodedata.combining(ch):
                if units:
                    units[-1][1].add(ord(ch))
            else:
                units.append((ch, set()))
        tonal = Dilemma._TONAL_ACCENTS
        nuclei = []  # (letters, accented, first unit, last unit)
        index = 0
        while index < len(units):
            base, marks = units[index]
            if base not in _GREEK_VOWELS:
                index += 1
                continue
            start, letters, accented = index, base, bool(marks & tonal)
            if (index + 1 < len(units) and not accented
                    and base + units[index + 1][0] in _BREATHING_DIPHTHONGS
                    and 0x0308 not in units[index + 1][1]):
                index += 1
                letters += units[index][0]
                accented = bool(units[index][1] & tonal)
            nuclei.append((letters, accented, start, index))
            index += 1
        if len(nuclei) <= 1:
            return True
        (letters, accented, start, end), second = nuclei[0], nuclei[1]
        return (len(nuclei) == 2 and letters in ("ι", "υ", "ει", "οι")
                and not accented and start > 0 and second[2] == end + 1)

    def _modern_pos_entry_keeps_accent(self, word: str, upos: str,
                                       modern: str) -> bool:
        """Whether the Modern reading is recorded for the word's accent.

        Monotonic spelling writes no accent on a monosyllable, so there the
        bare key is the word's own spelling (νὰ is να, μιὰ is μια). A word
        counts as one by _monosyllable_by_synizesis, which admits synizesis
        (ποιὸς) but not hiatus (προϊόν, #123).
        """
        lower = word.lower()
        stripped = strip_accents(lower)
        if self._monosyllable_by_synizesis(lower):
            return True
        return any(
            variant != stripped
            and (self._pos_lookup.get(variant) or {}).get(upos) == modern
            for variant in (word, lower, grave_to_acute(lower),
                            to_monotonic(lower)))

    @staticmethod
    def _capitalized_ancient_word(word: str, upos: str, modern: str,
                                  ancient: str) -> bool:
        """Whether a capitalized PROPN form is a capitalized Ancient word.

        A capitalized form tagged PROPN whose Ancient reading is a
        lowercase word keeps that reading rather than a Modern name
        (Ὀρᾷς is ὁράω, not Όρας; Γένῃ is γίγνομαι, not Γένη; #123).
        """
        return (upos == "PROPN" and is_capitalized(word)
                and is_capitalized(modern) and not is_capitalized(ancient))

    def _modern_pos_reading_applies(self, word: str, upos: str,
                                    modern: str) -> bool:
        """Whether a Modern Greek POS reading may stand for polytonic input.

        Polytonic spelling does not make a word Ancient: vernacular
        Byzantine and pre-1982 Modern Greek are written the same way. The
        Modern reading stands only when the Ancient side offers nothing of
        the requested kind:

        * no reading at all, from the lookup or any structured resolver;
        * an unaccented echo of the input that is no Ancient headword (the
          misaccented ἀλλην);
        * a reading the Ancient POS tables record for the form only under
          a conflicting part of speech (νὰ as νή when a conjunction is
          requested, κυροῦ as the verb κυρόω when a noun is). For an
          open-class tag the Ancient side must also not know the spelling
          itself (as a headword, or as an accented key of the Ancient POS
          tables), so a mistagged Ancient form keeps its lemma (ὄντων ADJ
          stays εἰμί);
        * a reading with no recorded part of speech, of a spelling the
          Ancient side does not know, that was found only by discarding
          the form's accents while the Modern reading is recorded for
          them (ὧρες, matched to ὄρνυμι through ώρες, is ώρες), or that
          the lookup holds under the form's own accents but that has a
          verb ending when a closed-class tag is requested (σεῖς PRON is
          not a form of σέω). A form written with a smooth breathing whose
          rough-breathing spelling the Ancient lookup holds under the same
          reading counts as a spelling the Ancient side knows (ἄπαν NOUN
          stays ἅπας, like ἅπαν);
        * a capitalized name with no recorded part of speech, when the
          word and the Modern lemma are lowercase and the tag is not a
          noun's (κακὸ ADJ is not the name Κακό, nor πιὰ ADV Πῖα).

        Once an Ancient reading exists, the Modern one never stands when
        it is the same word in monotonic spelling (ἄλλος, άλλος), when the
        Ancient reading's part of speech is compatible (Ἑλλήνων PROPN
        keeps Ἕλλην, not Έλληνας), or when it is a capitalized name for a
        lowercase word. A second accent from a following enclitic is not
        part of the spelling: ἄνθρωπός is judged as ἄνθρωπος.
        """
        word = self._without_enclitic_accent(
            unicodedata.normalize("NFC", word))
        # The Ancient reading, whatever the convention prefers.
        ancient = self._lookup_word(word, prefer_modern=False)
        if not ancient:
            # The article, elision and crasis resolvers read forms the
            # lookup does not hold (οἱ); their reading is an Ancient one.
            return not self._lemmatize_verbose_impl(word, guess=False)
        ancient_key = strip_accents(ancient.lower())
        name_only = (is_capitalized(ancient) and not is_capitalized(word)
                     and not is_capitalized(modern)
                     and upos not in ("PROPN", "NOUN"))
        if ancient_key == strip_accents(modern.lower()) and not name_only:
            return False
        if is_capitalized(modern) and not is_capitalized(word):
            return False
        lower = word.lower()
        accented = (word, lower, grave_to_acute(lower))
        table = self._pos_ag_lookup
        tags = {tag
                for variant in (*accented, to_monotonic(lower),
                                strip_accents(lower))
                for tag, lemma in (table.get(variant) or {}).items()
                if lemma and strip_accents(lemma.lower()) == ancient_key}
        if any(self._pos_tags_compatible(upos, tag) for tag in tags):
            return False
        echo = ancient_key == strip_accents(lower)
        headword = echo and self._is_ancient_headword(ancient)
        if echo and not headword and not any(
                ord(ch) in self._TONAL_ACCENTS
                for ch in unicodedata.normalize("NFD", word)):
            return True
        if self._capitalized_ancient_word(word, upos, modern, ancient):
            return False
        known = headword or any(table.get(variant) for variant in accented)
        closed_class = upos in self._CLOSED_CLASS_TAGS
        if tags:
            return closed_class or not known
        if name_only:
            return True
        if known:
            return False
        if not self._ancient_lookup_has_spelling(word) \
                and not self._rough_spelling_reads(word, ancient):
            # Neither reading is one of this spelling. A Modern entry
            # reached the same way, under the bare key, is no better
            # evidence than the Ancient one (ἐφη VERB stays φημί).
            return self._modern_pos_entry_keeps_accent(word, upos, modern)
        return closed_class and ancient_key.endswith(("ω", "μι", "μαι"))

    def _rough_spelling_reads(self, word: str, lemma: str) -> bool:
        """Whether the Ancient lookup reads *word* with a rough breathing as *lemma*.

        A smooth breathing written for a rough one is the commonest slip
        of Byzantine and later polytonic spelling, and no evidence for
        Modern Greek. When the Ancient lookup lacks the form as written
        and holds it with the rough breathing, under the lemma the loose
        key gave, the Ancient side knows the spelling: ἄπαν NOUN is the
        ἅπας of ἅπαν, as ἅπαν NOUN is, not the Modern noun άπαν (#134).
        The reverse does not hold: a rough breathing is written on
        purpose, and ὧρες is the Modern ώρες, not the Ancient ὦρες.
        """
        if _initial_breathing(word) != _SMOOTH_BREATHING or not self._ag_lookup:
            return False
        nfd = unicodedata.normalize("NFD", word)
        swapped = unicodedata.normalize(
            "NFC", nfd.replace(_SMOOTH_BREATHING, _ROUGH_BREATHING, 1))
        acute = grave_to_acute(swapped)
        return any(
            self._normalize_grave_citation_lemma(
                self._ag_lookup.get(variant) or "") == lemma
            for variant in dict.fromkeys(
                (swapped, swapped.lower(), acute, acute.lower())))

    def _lowercase_ancient_twin(self, word: str, modern: str) -> str | None:
        """The lowercase Ancient headword a Modern POS reading stands for.

        When the Modern reading stands against an Ancient reading that is
        a capitalized name (εὐήνεμός, read as Εὐήνεμός through its
        enclitic accent, gives the name Εὐήνεμος), the lowercase headword
        with the name's spelling is the Ancient word the Modern lemma
        spells monotonically (εὐήνεμος, not ευήνεμος; #123). The word and
        the Modern lemma must be lowercase and the headword must have the
        Modern lemma's letters.
        """
        if is_capitalized(word) or is_capitalized(modern):
            return None
        written = unicodedata.normalize("NFC", word)
        for form in dict.fromkeys((written, self._without_enclitic_accent(written))):
            ancient = self._lookup_word(form, prefer_modern=False)
            if not ancient or not is_capitalized(ancient):
                continue
            twin = ancient[0].lower() + ancient[1:]
            if (strip_accents(twin) == strip_accents(modern.lower())
                    and self._is_ancient_headword(twin)):
                return twin
        return None

    def _pos_table_lookup(self, word: str, upos: str) -> str | None:
        """Look up POS-specific lemma from POS disambiguation tables.

        For polytonic input, checks the AG-only POS table first to avoid
        MG lemma overrides on Ancient Greek text. Returns None if no match.
        """
        return self._pos_table_hit(word, upos)[0]

    # Index of the loose variant in _pos_table_hit's key cascade (word,
    # lower, grave folded to acute, monotonic, accent-stripped). The
    # monotonic key keeps the accent's place, so a misspelt breathing or
    # accent type still finds its word there (ἰδῶν VERB -> ὁράω through
    # ιδών); the accent-stripped key is shared by other words (#105).
    _LOOSE_POS_VARIANT = 4

    def _pos_table_hit(self, word: str, upos: str) -> tuple[str | None, int | None]:
        """The POS-table lemma and the index of the key variant it was found at.

        Variants 0-2 are the form's own spelling (as written, lowercased,
        grave folded to acute), 3 its monotonic key and 4 its accent-stripped
        key, which other words share (see _pos_entry_is_loose). A reviewed
        verb grammar counts as the form's own spelling.
        """
        if self.lang in ("all", "grc") and upos in ("VERB", "AUX"):
            grammar = reviewed_verb_grammar(word)
            if grammar is not None:
                return grammar["lemma"], 0
        if self.lang in ("all", "grc") and upos == "ADV":
            # A reviewed lexicalized adverb (ἴσως, ἰδίως) is its own
            # headword, whatever adjective a treebank cell names (#117).
            # It counts as the form's own spelling, never a loose entry.
            adverb = grave_to_acute(word.lower())
            if adverb in _grc_adverb_data()[1]:
                return adverb, 0
        lower = word.lower()
        acute = grave_to_acute(lower)
        mono = to_monotonic(lower)
        stripped = strip_accents(lower)
        variants = (word, lower, acute, mono, stripped)

        # A POS-table VALUE that is itself a corpus artifact - an elided
        # fragment (ἀλλ᾽) or an abbreviation overline (οὐδ̅, U+0305) - is
        # not a headword; lemmatize_pos trusts the table over a single
        # lookup candidate, so junk here becomes the output. Skip such
        # entries (the builders filter them too; this guards stale data).
        def _usable(lemma: str | None) -> bool:
            lemma = self._normalize_grave_citation_lemma(lemma) if lemma else None
            return bool(lemma) and not _is_elided_junk_value(lemma) \
                and "̅" not in unicodedata.normalize("NFD", lemma)

        def _target(lemma):
            lemma = self._normalize_grave_citation_lemma(lemma)
            if self.lang in ("all", "grc") and upos in ("VERB", "AUX"):
                return correct_grammatical_lemma(word, lemma)
            return lemma

        # The Ancient corpus tables hold spellings from texts that write an
        # initial rho without its breathing (ρήσεις NOUN -> ρῆσις, VERB ->
        # ρέω). For Ancient input such a target is skipped and the lookup
        # candidates decide (ῥήσεις NOUN -> ῥῆσις; #107). Modern entries
        # (ρε) are not affected.
        ancient_context = self._ancient_pos_context(word, upos)

        def _breathingless_ancient(variant, lemma):
            return (ancient_context and _has_bare_initial_rho(lemma)
                    and (getattr(self, "_pos_ag_lookup", {}).get(variant)
                         or {}).get(upos) == lemma)

        # For polytonic input, try AG-only POS first
        nfd = unicodedata.normalize("NFD", word)
        has_poly = any(ord(ch) in _POLYTONIC_STRIP | _POLYTONIC_TO_ACUTE
                       for ch in nfd)
        if has_poly and self.lang == "all":
            for index, variant in enumerate(variants):
                pos_entry = self._pos_ag_lookup.get(variant)
                if pos_entry and upos in pos_entry \
                        and _usable(pos_entry[upos]) \
                        and not _breathingless_ancient(variant, pos_entry[upos]):
                    return _target(pos_entry[upos]), index
            # The combined table adds only Modern Greek entries to the
            # AG-only one, and its monotonic and stripped variants would
            # turn polytonic Ancient forms into monotonic headwords
            # (ἄλλην -> άλλος, Ἑλλήνων -> Έλληνας; issue #100). Only the
            # first Modern entry for the tag is considered, and it stands
            # only where _modern_pos_reading_applies finds no Ancient
            # reading of the requested kind (νὰ CCONJ -> να, κυροῦ NOUN
            # -> κυρός); otherwise the lookup candidates decide.
            # The Modern Greek entries carry no SCONJ tag, so an SCONJ
            # request reads the Modern CCONJ entry when the Ancient table
            # records no CCONJ entry for the form (νὰ SCONJ -> να; #120).
            tags = (upos,)
            if upos == "SCONJ" and not any(
                    "CCONJ" in (self._pos_ag_lookup.get(variant) or {})
                    for variant in variants):
                tags = ("SCONJ", "CCONJ")
            for tag in tags:
                for index, variant in enumerate(variants):
                    pos_entry = self._pos_lookup.get(variant)
                    if pos_entry and tag in pos_entry \
                            and _usable(pos_entry[tag]) \
                            and not _breathingless_ancient(variant, pos_entry[tag]):
                        if self._modern_pos_reading_applies(
                                word, upos, pos_entry[tag]):
                            twin = self._lowercase_ancient_twin(word, pos_entry[tag])
                            return _target(twin or pos_entry[tag]), index
                        return None, None
            return None, None

        # Combined POS lookup
        for index, variant in enumerate(variants):
            pos_entry = self._pos_lookup.get(variant)
            if pos_entry and upos in pos_entry \
                    and _usable(pos_entry[upos]) \
                    and not _breathingless_ancient(variant, pos_entry[upos]):
                return _target(pos_entry[upos]), index

        return None, None

    def _one_reading_against_modern_entry(self, word: str, upos: str,
                                          candidates) -> bool:
        """Whether several candidates are one lemma facing a Modern POS entry.

        True when every candidate names the same lemma and the POS-table
        entry for (word, upos) came from the Modern Greek side of
        _pos_table_hit (polytonic input with lang="all" and no usable
        Ancient-only entry for the tag), i.e. the polytonic rule already
        judged the Modern reading against that Ancient one (#119). VERB
        is left to the case-twin rule, which already treats one lemma as
        one reading.
        """
        if self.lang != "all" or upos == "VERB" or len(candidates) < 2 \
                or len({c.lemma for c in candidates}) != 1:
            return False
        nfd = unicodedata.normalize("NFD", word)
        if not any(ord(ch) in _POLYTONIC_STRIP | _POLYTONIC_TO_ACUTE
                   for ch in nfd):
            return False
        lower = word.lower()
        variants = (word, lower, grave_to_acute(lower), to_monotonic(lower),
                    strip_accents(lower))
        table = getattr(self, "_pos_ag_lookup", None) or {}
        return not any((table.get(variant) or {}).get(upos)
                       for variant in variants)

    # Lookup matches made under the form's own spelling (see
    # _lemmatize_verbose_impl); "+case_alt" twins count with their match.
    _EXACT_LOOKUP_VIAS = frozenset({"exact", "lower", "reviewed_inflection",
                                    "elision-exact"})
    # Tags under which a loose POS entry may not replace an exact nominal
    # reading (#105). Other tags keep the override: Byzantine misspellings
    # rely on it (οῦν PART, τῆδε PRON, μᾶλα ADV).
    _LOOSE_OVERRIDE_BLOCKED_TAGS = frozenset({"NOUN", "PROPN"})
    # Tags under which it may not replace one either when every such
    # reading is a lemma an Ancient dictionary lists (#130: ταύτην ADJ
    # keeps οὗτος, not the ταὐτός of ταὐτήν). A row whose lemma no
    # dictionary lists is a corpus artifact of a misaccented form and
    # still yields (μεγά ADJ -> μέγας, not the row's μεγός). lang="el" and
    # the triantafyllidis convention prefer Modern rows and keep the
    # override. DET and NUM are left out: the entries they would stop are
    # right (τοῖϊν DET -> ὁ, ιέ NUM -> ιε).
    _LOOSE_OVERRIDE_LISTED_TAGS = frozenset({"ADJ", "PRON"})

    def _pos_entry_is_loose(self, word: str, upos: str, pos_lemma: str) -> bool:
        """Whether *pos_lemma* was found only under the accent-stripped key.

        The accent-stripped key of ὄρους holds ὀρός, a reading of ὀροῦς;
        such an entry is evidence about the letters, not about this
        spelling. An entry this method cannot trace back to the
        tables (a substituted reading, a model without tables) is treated
        as exact, so it keeps its previous authority.
        """
        try:
            lemma, index = self._pos_table_hit(word, upos)
        except AttributeError:
            return False
        return (lemma is not None and lemma == pos_lemma and index is not None
                and index >= self._LOOSE_POS_VARIANT)

    @classmethod
    def _is_exact_lookup_row(cls, candidate) -> bool:
        via = (getattr(candidate, "via", "") or "").split("+")[0]
        return (getattr(candidate, "source", "") == "lookup"
                and via in cls._EXACT_LOOKUP_VIAS)

    @staticmethod
    def _is_closed_class_reading(candidate) -> bool:
        return getattr(candidate, "source", "") in ("article", "crasis")

    def _candidate_owns_spelling(self, word: str, candidate) -> bool:
        """Whether *candidate* reads the form's own spelling.

        True for an article or crasis reading and for a lookup row found
        under the form as written. A row that only re-accents the input's
        letters (Ὦρα -> ὤρα) is itself a guess about the diacritics and
        does not count.
        """
        if self._is_closed_class_reading(candidate):
            return True
        if not self._is_exact_lookup_row(candidate):
            return False
        own = grave_to_acute(self._without_enclitic_accent(
            unicodedata.normalize("NFC", word))).lower()
        lemma = grave_to_acute(unicodedata.normalize("NFC", candidate.lemma)).lower()
        return lemma == own or strip_accents(lemma) != strip_accents(own)

    def _is_attested_name(self, lemma: str) -> bool:
        """Whether a POS table gives *lemma* as a proper noun."""
        self._load_fallback_readings()
        return lemma in self._fallback_name_lemmas

    def _is_verb_shaped_lemma(self, lemma: str) -> bool:
        self._load_fallback_readings()
        return (lemma in self._fallback_verb_lemmas
                or strip_accents(lemma.lower()).endswith(("ω", "μαι", "μι")))

    def _pos_respell_blocked(self, word: str, upos: str, pos_lemma: str,
                             candidate) -> bool:
        """A loose POS entry may not respell a reading of the form's own
        spelling (ὄρους NOUN keeps ὄρος, not the ὀρός of ὀρους; #105)."""
        return (self._candidate_owns_spelling(word, candidate)
                and self._pos_entry_is_loose(word, upos, pos_lemma))

    def _pos_override_blocked(self, word: str, upos: str, pos_lemma: str,
                              candidates, *, case_twin_family: bool = False) -> bool:
        """Whether a loose POS entry may not replace the candidates (#105).

        It may not replace an article or crasis reading, nor, under NOUN or
        PROPN, exact lookup rows that inflect to lemmas without a verb
        shape (ὅλων NOUN keeps ὅλος, Ἄργους PROPN keeps Ἄργος). An exact
        verb row still yields to a nominal tag (the Byzantine μοχθῶ NOUN ->
        μόχθος). Under ADJ or PRON the same holds when every such row
        reads a lemma an Ancient dictionary lists (#130: ταύτην ADJ keeps
        οὗτος, ἀθληταί ADJ ἀθλητής), except with lang="el" or the
        triantafyllidis convention; a row whose lemma no dictionary lists
        still yields (μεγά ADJ -> μέγας), and so does the row of a
        capitalized word when the row of its lowercase spelling reads the
        entry's lemma (see _lowercase_row_reads). Under PRON the row
        must also be a reading the tables record as a pronoun's for
        these letters (see _tables_record_pronoun). Under the VERB
        case-twin rule (#42) a loose entry may not replace an exact
        case-twin family (πόθῳ VERB keeps πόθος, not the πίνω of ποθῶ).

        An unaccented word has no spelling apart from its accent-stripped
        key, so under PRON an entry it finds there yields to a row the
        tables record as a pronoun of these letters (ταυτης PRON keeps
        οὗτος, not the αὐτός of ταὐτῆς; συ PRON keeps σύ; #151). A row
        that only repeats the form does not stop a noun entry: ὅρη NOUN is
        ὄρος and οἴκοι NOUN οἶκος, though both spellings are listed. A
        PROPN entry for another word does not replace a common word and
        its name twin (see _name_twin_reads_form).
        """
        if any(self._is_closed_class_reading(c) for c in candidates):
            return self._pos_entry_is_loose(word, upos, pos_lemma)
        exact = [c for c in candidates if self._is_exact_lookup_row(c)]
        if not exact:
            return False
        if not case_twin_family and self._name_twin_reads_form(
                word, upos, pos_lemma, exact):
            return True
        rows = []
        if not case_twin_family:
            listed_only = upos not in self._LOOSE_OVERRIDE_BLOCKED_TAGS
            if listed_only and (
                    upos not in self._LOOSE_OVERRIDE_LISTED_TAGS
                    or self.lang == "el"
                    or self._convention_name == "triantafyllidis"):
                return False
            stripped = strip_accents(word.lower())
            inflected = [c for c in exact
                         if strip_accents(c.lemma.lower()) != stripped]
            if not inflected:
                inflected = self._unaccented_pronoun_rows(word, upos, exact)
                if not inflected:
                    return False
            if any(self._is_verb_shaped_lemma(c.lemma) for c in inflected):
                return False
            if listed_only and (
                    not all(self._is_ancient_headword(c.lemma)
                            for c in inflected)
                    or self._lowercase_row_reads(word, pos_lemma)):
                return False
            if upos == "PRON" and not all(
                    self._tables_record_pronoun(word, c.lemma)
                    for c in inflected):
                return False
            rows = inflected
        return (self._pos_entry_is_loose(word, upos, pos_lemma)
                or self._pronoun_entry_at_bare_key(word, upos, pos_lemma, rows))

    def _prefers_modern_rows(self) -> bool:
        return (self.lang == "el"
                or getattr(self, "_convention_name", None) == "triantafyllidis")

    @staticmethod
    def _is_unaccented(word: str) -> bool:
        lower = word.lower()
        return lower == strip_accents(lower)

    def _pronoun_entry_at_bare_key(self, word: str, upos: str,
                                   pos_lemma: str, rows) -> bool:
        """Whether a PRON entry of an unaccented word yields to *rows*.

        The tables store accent-stripped keys, so ταυτης finds the cell
        that ταὐτῆς left under ταυτης as if it were the word's own. Only
        PRON is treated this way: the pronouns are a closed class the tables
        cover, while a noun or adjective cell at such a key is often the
        better reading of the letters. Each row must be recorded under PRON
        itself: α PRON stays ὅς (ἅ), though the article ὁ is recorded for
        these letters under DET. The record must be an Ancient table's: the
        Modern cell του PRON του does not make that row the reading of
        Ancient του, which stays τις. An entry with a row's letters stands,
        as the missing accent cannot choose between the two (οιαν PRON keeps
        οἷος beside the row οἶος).
        """
        if upos != "PRON" or not rows or self._prefers_modern_rows() \
                or not self._is_unaccented(word):
            return False
        entry_letters = strip_accents(pos_lemma.lower())
        if any(strip_accents(c.lemma.lower()) == entry_letters for c in rows):
            return False
        self._load_fallback_readings()
        recorded = (getattr(self, "_ancient_pronoun_readings", None) or {}).get(
            strip_accents(word.lower()), ())
        if not all(c.lemma in recorded for c in rows):
            return False
        try:
            lemma, index = self._pos_table_hit(word, upos)
        except AttributeError:
            return False
        return lemma is not None and lemma == pos_lemma and index is not None

    def _unaccented_pronoun_rows(self, word: str, upos: str, exact) -> list:
        """The rows reading an unaccented lowercase PRON as a listed headword.

        συ has no accent to tell it from σύ, so the row with its letters is
        the word's own reading. A case alternate only repeats the row it
        came from, and a row no Ancient dictionary lists may be a corpus
        artifact. _pronoun_entry_at_bare_key then asks that the tables
        record the row under PRON.
        """
        if (upos != "PRON" or self._prefers_modern_rows()
                or not word[:1].islower() or not self._is_unaccented(word)):
            return []
        return [c for c in exact
                if not getattr(c, "via", "").endswith("+case_alt")
                and self._is_ancient_headword(c.lemma)]

    # Accented spellings of Ζεύς whose letters are also another word's: Δῖός
    # (Sophocles, Ajax 137) beside δῖος, Doric Δᾶν (Theocritus 4.17, 7.39)
    # beside Δάν. Δὰν is the biblical Dan and is not listed.
    _NAME_ENTRY_SPELLINGS = frozenset({"Δῖός", "Δᾶν"})

    def _name_twin_reads_form(self, word: str, upos: str, pos_lemma: str,
                              exact) -> bool:
        """Whether a common word and its name twin both spell the form.

        Ἡδὺς reads ἡδύς beside the name Ἡδύς, so the lookup already holds
        a name for it under PROPN. A PROPN entry with other letters (the
        Ἡδεῖα that the table files under ἡδύς) names a different person
        and does not replace them (#151). Only a capitalized form with its
        own accent counts: unaccented Διος and Αρη are the names Ζεύς and
        Ἄρης, whatever the rows of their bare letters say. A spelling with
        only an acute may be monotonic, and there the name entry stands
        too (Βάρδα, of Βάρδας). An entry the form is the genitive of also
        stands (Ἀράχνης, of Ἀράχνη), as does the entry of a spelling
        reviewed as that name.
        """
        spelling = unicodedata.normalize("NFC", word)
        if (upos != "PROPN" or self._prefers_modern_rows()
                or not is_capitalized(word) or self._is_unaccented(word)
                or to_monotonic(spelling) == spelling
                or spelling in self._NAME_ENTRY_SPELLINGS):
            return False
        letters = strip_accents(word.lower())
        entry_letters = strip_accents(pos_lemma.lower())
        if (entry_letters == letters or entry_letters + "ς" == letters
                or any(strip_accents(c.lemma.lower()) != letters for c in exact)):
            return False
        return (any(is_capitalized(c.lemma) for c in exact)
                and any(c.lemma[:1].islower() for c in exact))

    def _tables_record_pronoun(self, word: str, lemma: str) -> bool:
        """Whether the POS tables record *lemma* for the word's letters under PRON or DET.

        The pronouns are a closed class the tables cover, so a row that
        stands against a pronoun entry must itself be a recorded pronoun
        reading of these letters (οὕ PRON keeps ἕ: the tables hold οὗ
        PRON ἕ; ταύτὴν PRON keeps οὗτος). A row that reads another kind
        of word does not stand in a pronoun's place: a word tagged PRON
        that is misaccented or has the wrong breathing still reads its
        pronoun (ὁδὲ PRON is ὅδε, not the ὁδός of its row; ἐκάτερος
        PRON is ἑκάτερος, not ἐκάς; the Modern μάς PRON is μας, not μής).
        An adjective's row is such another word (κεινὸς PRON is ἐκεῖνος,
        not the κενός the tables record under ADJ).
        """
        self._load_fallback_readings()
        readings = self._fallback_readings.get(strip_accents(word.lower()), {})
        return any(lemma in readings.get(tag, ()) for tag in ("PRON", "DET"))

    def _lowercase_row_reads(self, word: str, lemma: str) -> bool:
        """Whether the lookup reads a capitalized word's lowercase spelling as *lemma*.

        A capitalized word tagged other than PROPN may be a sentence-initial
        common word, so the row of its lowercase spelling is a reading of
        the word too. A loose POS entry that row agrees with is supported
        by the form's letters and by one of its spellings (Ἀντίαν ADJ gives
        ἀντίος with ἀντίαν, not the ἀντία of the capitalized row).
        """
        lower = word.lower()
        if lower == word:
            return False
        target = strip_accents(lemma.lower())
        tables = []
        if self.lang in ("all", "el"):
            tables += [self._mg_lookup, self._med_lookup]
        if self.lang in ("all", "grc"):
            tables.append(self._ag_lookup)
        for table in tables:
            if not table:
                continue
            for variant in dict.fromkeys((lower, grave_to_acute(lower))):
                hit = table.get(variant)
                if hit and strip_accents(hit.lower()) == target:
                    return True
        return False

    def _closed_class_entry_decides(self, word: str, upos: str, pos_lemma: str,
                                    candidates) -> bool:
        """Whether a closed-class POS entry outranks a case-twin family.

        A lookup reading and its case alternate are one reading (αἶα
        beside Αἶα for αἶ, τίσις beside Τίσις for τίσι), and a lone
        reading yields to a supported POS entry. Under a tag of the small
        closed classes the pair yields as well, when the entry stands
        under the word's own spelling: no noun or name is a pronoun, a
        preposition or a conjunction, so αἶ CCONJ is εἰ, ἐμέο PRON ἐγώ
        and ἄνα ADP ἀνά. An entry of the monotonic or the accent-stripped
        key, which other words share, leaves the lookup reading. So does
        an entry whose lemma the lookup reads as a form of that reading:
        it names the same word under another citation form (πᾶσι PRON
        keeps πᾶς against the cell's πᾶν, οἷα CCONJ οἷος against οἷα).
        """
        if upos not in self._CLOSED_CLASS_TAGS or len(candidates) < 2:
            return False
        try:
            lemma, index = self._pos_table_hit(word, upos)
        except AttributeError:
            return False
        if (lemma is None or lemma != pos_lemma or index is None
                or index > self._OWN_SPELLING_POS_VARIANT):
            return False
        family = {c.lemma.lower() for c in candidates}
        read = self._lookup_word(pos_lemma)
        return not (read and read.lower() in family)

    def _is_recorded_verb_lemma(self, lemma: str) -> bool:
        """Whether *lemma* has a verb's ending and a verb entry in the POS tables.

        The ending alone also fits nouns (the Modern ποτάμι, ἠχώ), and a
        verb entry alone can be a mistagged cell of another word.
        """
        if not strip_accents(lemma.lower()).endswith(("ω", "μαι", "μι")):
            return False
        self._load_fallback_readings()
        return lemma in self._fallback_verb_lemmas

    def _name_entry_decides(self, word: str, upos: str, pos_lemma: str,
                            candidates) -> bool:
        """Whether a PROPN entry outranks the capitalized twin of a verb row.

        Under PROPN a lookup reading answers as its capitalized twin. When
        the row reads a verb (φάων as the participle of φάω, μέλητον as a
        dual of μέλω), that twin (Φάω, Μέλω) is a name that only shares
        the verb's citation spelling: no form of a verb is a form of a
        name. An entry under the word's own spelling that names a
        capitalized lemma then decides (Φάων, Μέλητος), as it already does
        against a verb row that has no capitalized twin (#138).

        A row that reads a noun or an adjective keeps its twin (ἵππων
        PROPN stays Ἵππος, not the entry's Ἵππων): the form may be a case
        of that name, and on the names of the Perseus, Diorisis and GLAUx
        pairs the entry was wrong about as often as right there, and
        more often for capitalized words. So does a reading whose own row
        is a capitalized lemma: the lookup holds a name for the form.
        """
        if upos != "PROPN" or len(candidates) < 2 or not is_capitalized(pos_lemma):
            return False
        direct = [c for c in candidates
                  if not getattr(c, "via", "").endswith("+case_alt")]
        if not direct or any(is_capitalized(c.lemma)
                             or not self._is_recorded_verb_lemma(c.lemma)
                             for c in direct):
            return False
        try:
            lemma, index = self._pos_table_hit(word, upos)
        except AttributeError:
            return False
        return (lemma is not None and lemma == pos_lemma and index is not None
                and index <= self._OWN_SPELLING_POS_VARIANT)

    @staticmethod
    def _is_name_row_family(word: str, candidates) -> bool:
        """Whether a lowercase word reads only a capitalized lemma and its twin.

        The lookup row of νῖκος reads the name Νίκη, and the lowercase head
        νίκη joins it only as that lemma's case alternate. For a lowercase
        word that is one reading, as it was before νίκη had a row of its
        own, so a supported POS entry (νῖκος NOUN -> νῖκος) still decides.
        A family whose row reads the lowercase lemma (ἐλάττονος: μικρός
        beside Μικρός) is not one: there the lookup reading stands.
        """
        if len(candidates) < 2 or not word[:1].islower():
            return False
        if len({c.lemma.lower() for c in candidates}) != 1:
            return False
        direct = [c for c in candidates
                  if not getattr(c, "via", "").endswith("+case_alt")]
        return (0 < len(direct) < len(candidates)
                and all(is_capitalized(c.lemma) for c in direct))

    # Cells of a first- or second-declension nominal by citation ending,
    # without accents or iota subscript.
    _NOMINAL_PARADIGM_CELLS = {
        "ος": ("ος", "ου", "ω", "ον", "ε", "οι", "ων", "οις", "ους",
               "η", "ης", "ην", "α", "ας", "αν", "αι", "αις"),
        "ον": ("ον", "ου", "ω", "α", "ων", "οις"),
        "α": ("α", "ας", "αν", "αι", "ων", "αις", "η", "ης", "ην"),
        "η": ("η", "ης", "ην", "αι", "ων", "αις"),
    }

    @classmethod
    def _is_nominal_paradigm_cell(cls, word: str, lemma: str) -> bool:
        """Whether *word* has the letters of a regular cell of *lemma*."""
        form = strip_accents(word.lower())
        head = strip_accents(lemma.lower())
        for ending, cells in cls._NOMINAL_PARADIGM_CELLS.items():
            stem = head[:-len(ending)]
            if head.endswith(ending) and len(stem) >= 2:
                return any(form == stem + cell for cell in cells)
        return False

    # Noun and adjective that share spellings and that a POS tag separates,
    # reviewed pair by pair. The POS tables split about 1,900 form/tag pairs
    # this way, and most of the others would follow a tagging slip to a rare
    # word (ζωήν ADJ to ζωός, βασιλείαν ADJ to βασίλειος).
    _REVIEWED_NOUN_ADJECTIVE_SPLITS = frozenset({
        frozenset({"ἀξία", "ἄξιος"}),
        frozenset({"ἱερός", "ἱερόν"}),
    })

    def _pos_split_decides(self, word: str, upos: str, pos_lemma: str,
                           candidates) -> bool:
        """Whether the POS tables split the word between a noun and an adjective.

        The row of ἀξίαν reads ἄξιος, and the river Ἄξιος joins it only as
        that lemma's case alternate. The tables hold both readings of the
        spelling, ἄξιος under ADJ and ἀξία under NOUN, so the tag chooses
        (ἱερὰ likewise: ἱερός ADJ, ἱερόν NOUN; #151). The row's lemma and the
        requested one must be a reviewed pair, sit under NOUN and ADJ in the
        same entry, and the word must be a regular cell of the requested
        lemma. Any other cell stays a disagreement with the row (πάντα NOUN
        keeps πᾶς against Πάν, ἐλάττονος ADJ keeps μικρός against ἐλαχύς).
        """
        other = {"NOUN": "ADJ", "ADJ": "NOUN"}.get(upos)
        if other is None or len(candidates) < 2:
            return False
        direct = [c for c in candidates
                  if not getattr(c, "via", "").endswith("+case_alt")]
        if len(direct) != 1:
            return False
        row = direct[0].lemma
        if (frozenset({row, pos_lemma}) not in self._REVIEWED_NOUN_ADJECTIVE_SPLITS
                or not self._is_nominal_paradigm_cell(word, pos_lemma)):
            return False
        table = getattr(self, "_pos_lookup", None) or {}
        lower = word.lower()
        return any((table.get(variant) or {}).get(upos) == pos_lemma
                   and (table.get(variant) or {}).get(other) == row
                   for variant in dict.fromkeys((word, lower, grave_to_acute(lower))))

    # Oxytone second-declension cells that show which nominal an -οῖς form
    # belongs to.
    _OXYTONE_SIBLING_ENDINGS = ("ῷ", "οῦ", "όν", "ούς", "ῶν", "οί")

    @staticmethod
    def _is_denominative_of(verb: str, nominal: str) -> bool:
        """Whether *verb* is the contract verb of the -ος word *nominal*."""
        stem = strip_accents(nominal.lower())
        if not stem.endswith("ος") or len(stem) < 4:
            return False
        stem = stem[:-2]
        return strip_accents(verb.lower()) in {
            stem + ending for ending in ("εω", "αω", "οω",
                                         "εομαι", "αομαι", "οομαι")}

    def _nominal_reading(self, word: str, upos: str, candidates) -> str | None:
        """A nominal reading for a word the POS tables do not cover (#151).

        Asked only without a POS entry for the tag. Under NOUN or ADJ, a
        lone row of a verb derived from an -ος nominal gives way to that
        nominal (βαλῶ stays with βάλλω, which βαλός does not derive):

        - the row of the same form with its iota subscript, when no POS table
          knows the verb (μοναχῶ, written for the dative μοναχῷ: μοναχός;
          ποιῶ and καλῶ stay verbs whatever the tag says);
        - the oxytone -ός headword of an -οῖς form, when the lookup reads
          another case of it as that headword (ποδηγοῖς: ποδηγῷ is ποδηγός).

        Under NOUN or PROPN, a lowercase word with no lookup row of its own
        reads the row of its capitalized spelling when the POS tables name
        that lemma as a PROPN (θεοφάνην: Θεοφάνην is Θεοφάνης). A capitalized
        row alone is no evidence of a name: the generated paradigms hold
        Κανδιδάτος and Μόριος.
        """
        table = getattr(self, "_ag_lookup", None)
        if self.lang not in ("all", "grc") or not word or not table:
            return None
        lemma = None
        if upos in ("NOUN", "PROPN") and word[:1].islower() and not any(
                getattr(c, "source", "") == "lookup" for c in candidates):
            row = table.get(word[0].title() + word[1:])
            if (row and self._is_capitalized_ancient_head(row)
                    and self._is_attested_name(row)):
                lemma = row
        if lemma is None and upos in ("NOUN", "ADJ") and len(candidates) == 1 \
                and self._is_exact_lookup_row(candidates[0]) \
                and self._is_verb_shaped_lemma(candidates[0].lemma):
            lower = unicodedata.normalize("NFC", word.lower())
            if lower[-1:] in ("ῶ", "ῆ", "ᾶ"):
                row = table.get(unicodedata.normalize("NFC", lower + "\u0345"))
                if (row and row == row.lower()
                        and candidates[0].lemma not in self._fallback_verb_lemmas
                        and self._is_denominative_of(candidates[0].lemma, row)):
                    lemma = row
            elif lower.endswith("οῖς") and len(lower) > 5:
                stem = lower[:-3]
                head = stem + "ός"
                if self._is_ancient_headword(head) and self._is_denominative_of(
                        candidates[0].lemma, head) and any(
                            table.get(stem + ending) == head
                            for ending in self._OXYTONE_SIBLING_ENDINGS):
                    lemma = head
        if lemma is None:
            return None
        lemma = self._apply_convention(lemma)
        return self._checked_pos_result(word, upos, lemma)

    def _capital_row_decides(self, word: str, upos: str | None,
                             candidates) -> bool:
        """Whether a capitalized word's own row outranks its lowercase twin.

        Asked only where no POS entry chose a candidate; a POS entry that
        matched one decides by itself (Πῶς ADV -> πῶς, Χριστὸς NOUN ->
        Χριστός). Ancient editions capitalize names, so there the row of a
        capitalized word tagged other than PROPN stands (Εὐξένου NOUN
        keeps Εὔξενος). Modern text capitalizes every sentence, in
        polytonic spelling as in monotonic: under lang="el" or the
        triantafyllidis convention a capitalized word keeps the lowercase
        reading (Χέρι NOUN -> χέρι, Αν SCONJ -> αν, Καρδίᾳ NOUN ->
        καρδία), as it did before the row's own reading was consulted.
        Under the other settings a word written with polytonic marks is
        Ancient, and a monotonic one is Modern when its lowercase twin is
        a Modern citation form. PROPN and lowercase words never ask: the
        tiebreak settles them before it looks at the rows.
        """
        if not word or upos == "PROPN" or word[:1].islower():
            return False
        if (self.lang == "el"
                or getattr(self, "_convention_name", None) == "triantafyllidis"):
            return False
        if any(ord(ch) in _POLYTONIC_STRIP | _POLYTONIC_TO_ACUTE
               for ch in unicodedata.normalize("NFD", word)):
            return True
        return not any(c.lemma[:1].islower()
                       and self._is_mg_citation_form(c.lemma)
                       for c in candidates)

    @staticmethod
    def _prefer_case_twin(word: str, upos: str | None, lemma: str,
                          candidates: list[LemmaCandidate],
                          extra: tuple[str, ...] = (), *,
                          row_decides: bool = False) -> str:
        """Capitalization-agreement tiebreak between case-twin lemmas.

        Many common lemmas have a capitalized proper-noun twin as a
        separate headword (θυμός "spirit" vs Θυμός the personification,
        ἔρις vs Ἔρις). The lookup surfaces both (via "+case_alt") and
        the POS tables may carry either spelling, so the POS-matching
        loops can land on the twin that disagrees with the input's
        capitalization. When the chosen ``lemma`` and a case twin of it
        are both available, pick the twin whose capitalization agrees
        with the input form, conditioned on the requested POS:

          * ``upos == "PROPN"``: the capitalized twin wins.
          * lowercase input with any other POS: the lowercase twin wins
            (a token neither written nor tagged as a proper noun should
            get the common-word lemma).
          * capitalized or all-caps input with a non-PROPN POS:
            ambiguous (sentence-initial, titles) - keep the incoming
            choice, so either twin stays reachable. This is what a POS
            entry that matched a candidate gets, in lemmatize_pos() as
            in lemmatize_batch_pos().
          * the same input with ``row_decides``: the reading of the
            word's own lookup row stands against a twin that is only its
            case alternate. The callers ask for this only where no POS
            entry chose a candidate (see _capital_row_decides).

        This is a re-rank, not a filter: it can only swap ``lemma`` for
        a case twin drawn from ``candidates`` or ``extra`` (the POS
        table's own suggestion); any other input returns ``lemma``
        unchanged.
        """
        if not word or not lemma:
            return lemma
        low = lemma.lower()
        pool = [lemma]
        pool.extend(c.lemma for c in candidates)
        pool.extend(extra)
        lower_twin = upper_twin = None
        for t in pool:
            if not t or t.lower() != low:
                continue
            if t[:1].islower():
                if lower_twin is None:
                    lower_twin = t
            elif is_capitalized(t):
                if upper_twin is None:
                    upper_twin = t
        if lower_twin is None or upper_twin is None:
            return lemma  # no case-twin pair in play
        if upos == "PROPN":
            return upper_twin
        if word[:1].islower():
            return lower_twin
        if not row_decides:
            return lemma
        # Capitalized input no POS entry decided: the lookup row's own
        # reading stands against a twin that is only its case alternate
        # (Εὐξένου NOUN keeps Εὔξενος).
        direct = {c.lemma for c in candidates
                  if getattr(c, "source", "") == "lookup"
                  and not getattr(c, "via", "").endswith("+case_alt")}
        if upper_twin in direct and lower_twin not in direct:
            return upper_twin
        if lower_twin in direct and upper_twin not in direct:
            return lower_twin
        return lemma

    def _fix_mg_selfmap(self, word: str, candidates: list[LemmaCandidate],
                        upos: str | None = None) -> str | None:
        """Fix MG self-map problem for adjective/verb inflections.

        When _prefer_mg is true (triantafyllidis or lang="el"), the MG
        lookup sometimes returns self-maps for inflected forms instead
        of the citation form (masculine nominative for ADJ, infinitive
        for VERB). Examples:
          - ανθρώπινα -> ανθρώπινα (self-map), should be ανθρώπινος
          - περιέχει -> περιέχει (self-map), should be περιέχω

        When POS is known (from upos parameter), uses it directly.
        Adverbs and nouns keep their MG self-maps unchanged.

        Also fixes adjectives where MG returns a feminine nominative
        (e.g. παλαιολιθική) instead of the masculine nominative
        (παλαιολιθικός), by checking if a -ος form exists as a headword.

        Returns the corrected lemma, or None if no fix applies.
        """
        _prefer_mg = (self.lang == "el"
                      or self._convention_name == "triantafyllidis")
        if not _prefer_mg or not candidates:
            return None

        top = candidates[0]
        form_lower = word.lower()
        top_lower = top.lemma.lower()

        # ADJ/VERB self-map fix: when MG returns a self-map for an
        # inflected form, prefer a non-self-map citation form from
        # the combined/AG lookup.
        is_selfmap = _is_self_map(word, top.lemma)

        if is_selfmap and upos in ("ADJ", "VERB", None):
            # Find non-self-map candidates
            for c in candidates:
                if _is_self_map(word, c.lemma):
                    continue
                c_lower = c.lemma.lower()
                # ADJ: prefer masculine nominative (-ος, -ής, -ύς)
                if upos == "ADJ" or upos is None:
                    if (c_lower.endswith("ος") or c_lower.endswith("ής")
                            or c_lower.endswith("ύς")):
                        if not c.proper:
                            return c.lemma
                # VERB: prefer infinitive/1sg (-ω, -ώ, -μαι)
                if upos == "VERB" or upos is None:
                    if (c_lower.endswith("ω") or c_lower.endswith("ώ")
                            or c_lower.endswith("μαι")):
                        if not c.proper:
                            return c.lemma

        # ADJ feminine-to-masculine fix: when the top result is a
        # feminine nominative adjective (ending in -η/-ή/-ια/-α), check
        # if there is a masculine form (-ος) available as a headword.
        if upos == "ADJ" and not is_selfmap:
            mono_lemma = to_monotonic(top_lower)
            masc_form = None
            # Common feminine -> masculine adjective mappings
            if mono_lemma.endswith("ική") or mono_lemma.endswith("ικη"):
                masc_form = mono_lemma[:-1] + "ός"
            elif mono_lemma.endswith("ή"):
                masc_form = mono_lemma[:-1] + "ός"
            elif mono_lemma.endswith("η"):
                masc_form = mono_lemma[:-1] + "ος"
            elif mono_lemma.endswith("ια"):
                masc_form = mono_lemma[:-2] + "ος"
            elif mono_lemma.endswith("ιά"):
                masc_form = mono_lemma[:-2] + "ός"

            if masc_form:
                # Check if the masculine form exists as a headword
                # (self-mapping entry) in any table
                for tbl in (self._mg_lookup, self._ag_lookup, self._lookup):
                    hit = tbl.get(masc_form)
                    if hit and strip_accents(hit.lower()) == strip_accents(masc_form.lower()):
                        return self._apply_convention(hit)

        return None

    @staticmethod
    def _tagged_spelling(word):
        """The spelling the tagged entry points read the POS tables under.

        lemmatize() and lemmatize_batch() normalize to NFC and write a
        lunate sigma as σ/ς before they read the lookup; the tagged paths
        do the same, so a form printed with ϲ or in decomposed form finds
        its POS cell (#154).
        """
        if not isinstance(word, str) or not any(
                0x0370 <= ord(c) <= 0x03ff or 0x1f00 <= ord(c) <= 0x1fff
                for c in word):
            # Tokens without Greek letters pass through as given.
            return word
        return to_standard_sigma(unicodedata.normalize("NFC", word))

    def lemmatize_pos(self, word: str, upos: str,
                      *, attested_only: bool = False,
                      guess: bool = True) -> str | None:
        """Lemmatize with POS-aware disambiguation.

        POS is used to disambiguate among multiple candidates, or to
        override the single candidate when curated POS tables indicate
        a different lemma for the given POS tag.

        Algorithm:
          1. Run regular lemmatize_verbose() to get all candidates.
          2. Check POS tables for a POS-specific lemma. If it matches
             any candidate, return that candidate.
          3. If there is only one candidate and the POS table suggests
             a different lemma, trust the POS table (curated sources:
             treebank, GLAUx, Wiktionary). This handles cases like
             σκέψει+NOUN -> σκέψις where the default lookup only has
             the verb mapping σκέπτομαι.
          4. For MG self-map fix: when _prefer_mg is true and POS is
             ADJ/VERB, check if the top candidate is an MG self-map and
             prefer a citation-form alternative from combined/AG lookup.
          5. If no POS match, return the top candidate (same as regular
             lookup).

        Args:
            word: Greek word form.
            upos: Universal POS tag (NOUN, VERB, ADJ, etc.).
            attested_only: If True and the input form is unattested in the
                corpus, return None instead of a lemma.
            guess: If False, skip the transformer fallback and return None
                when nothing structured resolves, instead of a model guess
                or an echo of the input (see ``lemmatize``). A curated
                POS-table entry still counts as evidence.

        Returns:
            The lemma string, or None when gated out by ``attested_only``
            or unresolved with ``guess=False``.
        """
        # The tables are read under the spelling lemmatize() reads: NFC,
        # with a lunate sigma as σ/ς (Δῖόϲ PROPN like Δῖός, #154).
        word = self._tagged_spelling(word)
        self._pos_rejections(word, upos)
        def checked(lemma):
            return self._checked_pos_result(word, upos, lemma)

        if self._is_skipped_ordinal(word):
            return None
        # Attested-only gate: drop forms that don't occur in the corpus.
        if attested_only and word and not word.isdigit() \
                and self._resolve_attested_form(word) is None:
            return None
        plain = to_plain_double_rho(word)
        if plain != word:
            # ῤῥ is an editorial spelling of ρρ: the tables are read under
            # the plain spelling (Βοῤῥᾶ PROPN like Βορρᾶ). The gate above
            # has matched the corpus spelling as printed.
            word, attested_only = plain, False

        article_lemma = self._article_pos_lemma(word, upos)
        if article_lemma is not None:
            return checked(self._apply_convention(article_lemma))
        letter = _bare_letter_nominal(word, upos)
        if letter is not None:
            # The rule answers for the letter itself: the corpus evidence
            # against α as a lemma is about stray tokens, not about a numeral
            # or a list label (#116, #151).
            return letter

        # Get all candidates from regular lookup
        candidates = self._checked_pos_candidates(
            word, upos, self._lemmatize_verbose_impl(word, guess=guess))

        if not candidates:
            # Reachable when guess=False suppressed the model/identity
            # candidates (with guess=True verbose always adds identity).
            # A curated POS-table entry is still evidence, so honor it.
            pos_lemma = self._checked_pos_table_lookup(word, upos)
            if pos_lemma is not None:
                return checked(self._apply_convention(pos_lemma))
            nominal = self._nominal_reading(word, upos, candidates)
            if nominal is not None:
                return nominal
            return checked(self.lemmatize(word, attested_only=attested_only,
                                          guess=guess))

        # Use POS tables to disambiguate or override
        pos_lemma = self._checked_pos_table_lookup(word, upos)
        if pos_lemma is not None:
            pos_lemma_conv = self._apply_convention(pos_lemma)
            if pos_lemma_conv is None:
                pos_lemma = None
            else:
                # Check if any candidate matches the POS-specific lemma.
                # The case-twin tiebreak keeps a lowercase non-PROPN token
                # from landing on the capitalized proper-noun twin (and
                # vice versa for PROPN).
                for c in candidates:
                    if c.lemma == pos_lemma_conv:
                        return checked(self._prefer_case_twin(
                            word, upos, c.lemma, candidates,
                            extra=(pos_lemma_conv,)))
                # Also check with accent-stripped comparison (POS tables and
                # lookup tables may use slightly different accent conventions).
                # Stripping also erases the case distinction, so the matched
                # candidate may be the case twin of the POS lemma - pass the
                # POS lemma into the tiebreak so it can win the re-rank.
                pos_stripped = strip_accents(pos_lemma_conv.lower())
                for c in candidates:
                    if strip_accents(c.lemma.lower()) == pos_stripped:
                        if self._pos_respell_blocked(word, upos, pos_lemma, c):
                            return checked(self._prefer_case_twin(
                                word, upos, c.lemma, candidates))
                        preferred = self._prefer_exact_pos_citation(
                            word, upos, c.lemma, pos_lemma_conv)
                        return checked(self._prefer_case_twin(
                            word, upos, preferred, candidates,
                            extra=(pos_lemma_conv,)))
                # POS lemma not among candidates (e.g., single candidate from
                # lookup maps to a different headword). For an explicit verb
                # tag, two case twins still represent one competing lexical
                # reading, so they cannot block the supported verb target
                # (#42). So do a name and its lowercase case alternate for a
                # lowercase word (_is_name_row_family: νῖκος NOUN reads Νίκη
                # and νίκη, and the treebank's νῖκος still wins, as it does
                # against Νίκη alone). Any other twin family keeps the
                # lookup reading under a non-verb tag.
                # Under convention="triantafyllidis" the Modern and Ancient
                # lookups read one row, so a Modern entry the polytonic rule
                # let stand meets the same lemma twice (σεῖς PRON: σέω from
                # both sides); that is one competing reading (#119).
                if len(candidates) == 1 or self._one_reading_against_modern_entry(
                        word, upos, candidates):
                    if not self._pos_override_blocked(word, upos, pos_lemma,
                                                      candidates):
                        return checked(pos_lemma_conv)
                elif ((upos == "VERB"
                         or self._is_name_row_family(word, candidates)
                         or self._pos_split_decides(
                             word, upos, pos_lemma, candidates)
                         or self._closed_class_entry_decides(
                             word, upos, pos_lemma, candidates)
                         or self._name_entry_decides(
                             word, upos, pos_lemma, candidates))
                        and len({c.lemma.lower() for c in candidates}) == 1
                        and not self._pos_override_blocked(
                            word, upos, pos_lemma, candidates,
                            case_twin_family=upos == "VERB")):
                    return checked(pos_lemma_conv)
                # Also trust the POS table when every candidate is just a
                # case/accent variant of the input form (i.e. nothing in
                # the candidate list disagrees with the POS table). This
                # catches forms like αυτού (homograph: pronoun gen sg of
                # αυτός vs adverb "there") where the lookup has only the
                # adverb's self-map and the αυτός pron form-of resolution
                # got hidden under it.
                word_stripped = strip_accents(word.lower())
                if all(strip_accents(c.lemma.lower()) == word_stripped
                       for c in candidates) and not self._pos_override_blocked(
                           word, upos, pos_lemma, candidates):
                    return checked(pos_lemma_conv)

        if pos_lemma is None:
            nominal = self._nominal_reading(word, upos, candidates)
            if nominal is not None:
                return nominal

        if len(candidates) > 1:
            # MG self-map fix: when the top candidate is an MG self-map for
            # an ADJ/VERB inflection, prefer the citation form from
            # combined/AG.
            mg_fix = self._fix_mg_selfmap(word, candidates, upos=upos)
            if mg_fix is not None:
                return checked(mg_fix)

        # No POS match among candidates: the regular lemmatize_batch()
        # answer, as lemmatize_batch_pos() keeps it.
        return checked(self._unmatched_pos_result(
            word, upos, candidates, pos_lemma,
            attested_only=attested_only, guess=guess))

    @staticmethod
    def _recase_disagrees(word: str, upos: str, baseline: str,
                          candidate: str) -> bool:
        """Whether *candidate* only re-cases *baseline* against the tag.

        A single candidate that re-accents the baseline replaces it, but
        one that differs from it only in case and whose case disagrees
        with the form and tag does not: lowercase γ VERB keeps γ rather
        than the capital Γ (#128), while PROPN still takes a capital.
        """
        if candidate == baseline or candidate.lower() != baseline.lower():
            return False
        if upos == "PROPN":
            return candidate[:1].islower() and not baseline[:1].islower()
        return (word[:1].islower() and is_capitalized(candidate)
                and not is_capitalized(baseline))

    def _unmatched_pos_result(self, word: str, upos: str, candidates,
                              pos_lemma: str | None, *,
                              attested_only: bool, guess: bool) -> str | None:
        """What lemmatize_pos() returns when no POS entry chose a candidate.

        It is the answer lemmatize_batch_pos() keeps: the lemmatize_batch()
        baseline (not the verbose list's first candidate, which ranks
        differently under convention="triantafyllidis", προῖὸν NOUN), with
        that method's adjustments. Without a POS entry a single candidate
        that only re-accents the baseline replaces it; a list of case twins
        goes to the case-twin tiebreak; a baseline the POS check rejects
        gives way to the first candidate; and a lemma whose case disagrees
        with the form or tag is re-ranked between case twins.
        """
        def first():
            return self._prefer_case_twin(
                word, upos, candidates[0].lemma, candidates,
                row_decides=self._capital_row_decides(word, upos, candidates))

        baseline = self.lemmatize_batch([word], attested_only=attested_only,
                                        guess=guess)[0]
        rejected = (baseline is not None
                    and self._checked_pos_result(word, upos, baseline) is None)
        lemma = None if rejected else baseline
        if (pos_lemma is None and len(candidates) == 1 and lemma is not None
                and strip_accents(candidates[0].lemma.lower())
                == strip_accents(lemma.lower())
                and not self._recase_disagrees(word, upos, lemma,
                                               candidates[0].lemma)):
            return candidates[0].lemma
        if len(candidates) > 1 and all(
                c.lemma.lower() == candidates[0].lemma.lower()
                for c in candidates):
            lemma = first()
        if lemma is None:
            return first() if rejected and candidates else None
        need = (lemma[:1].islower() if upos == "PROPN"
                else word[:1].islower() and is_capitalized(lemma))
        if need:
            extra_lemma = (self._apply_convention(pos_lemma)
                           if pos_lemma is not None else None)
            extra = (extra_lemma,) if extra_lemma is not None else ()
            lemma = self._prefer_case_twin(word, upos, lemma, candidates,
                                           extra=extra)
        return lemma

    def lemmatize_batch_pos(self, words: list[str], upos_tags: list[str],
                            *, attested_only: bool = False,
                            guess: bool = True) -> list[str | None]:
        """Lemmatize a batch of words with POS-aware disambiguation.

        POS is used to disambiguate among multiple candidates, or to
        override the single candidate when curated POS tables indicate
        a different lemma. Preserves the batch model optimization from
        lemmatize_batch() while applying POS corrections.

        Algorithm:
          1. Run lemmatize_batch() to get baseline results (efficient,
             batches model inference for unknown words).
          2. For each word where POS tables suggest a different lemma,
             call lemmatize_verbose() to get all candidates and check
             if the POS-specific lemma is among them.
          3. If a single candidate doesn't match the POS lemma, trust
             the POS table (curated sources).
          4. For MG self-map fix: when _prefer_mg is true and POS is
             ADJ/VERB, check if the baseline result is an MG self-map
             and prefer a citation-form alternative.
          5. If a match is found, use it. Otherwise keep the baseline.

        Args:
            words: List of Greek word forms.
            upos_tags: List of UPOS tags, one per word.

        Returns:
            List of lemma strings.
        """
        assert len(words) == len(upos_tags), (
            f"words and upos_tags must have same length: {len(words)} vs {len(upos_tags)}"
        )
        # ῤῥ is an editorial spelling of ρρ: the tables are read under the
        # plain spelling, as in lemmatize_pos(). The attested-only gate
        # matches the corpus spelling as printed.
        # NFC and a lunate sigma as σ/ς come first, as in lemmatize() (#154).
        printed = [self._tagged_spelling(word) for word in words]
        words = [to_plain_double_rho(word) for word in printed]
        for word, upos in zip(words, upos_tags):
            self._pos_rejections(word, upos)

        # Step 1: Get baseline results from regular batch lemmatization
        results = self.lemmatize_batch(printed, attested_only=attested_only,
                                       guess=guess)
        rejected_baselines = [lemma is not None
            and self._checked_pos_result(word, upos, lemma) is None
            for word, upos, lemma in zip(words, upos_tags, results)]
        results = [None if rejected else lemma
                   for rejected, lemma in zip(rejected_baselines, results)]

        # Step 2: For each word, check if POS could improve the result
        resolved_articles = set()
        # Bare letters answer for themselves, past the POS rejection (#151).
        bare_letters = set()
        # Words whose loose POS entry was kept from respelling or replacing
        # the candidates (#105); step 3 must not offer it as a case twin.
        loose_blocked = set()
        for i, (word, upos) in enumerate(zip(words, upos_tags)):
            # Don't resurrect a gated-out (None) result via POS correction.
            # (A guess=False None is different: a curated POS-table entry is
            # evidence, so it MAY resurrect that - re-check the gate to tell
            # the two kinds of None apart, matching lemmatize_pos().)
            if attested_only and results[i] is None and (
                    self._resolve_attested_form(printed[i]) is None
                    or (guess and not rejected_baselines[i])):
                continue
            article_lemma = self._article_pos_lemma(word, upos)
            if article_lemma is not None:
                results[i] = self._apply_convention(article_lemma)
                resolved_articles.add(i)
                continue
            letter = _bare_letter_nominal(word, upos)
            if letter is not None:
                results[i] = letter
                resolved_articles.add(i)
                bare_letters.add(i)
                continue
            pos_lemma = self._checked_pos_table_lookup(word, upos)
            if pos_lemma is not None:
                pos_lemma_conv = self._apply_convention(pos_lemma)
                if pos_lemma_conv is None:
                    continue
                if (pos_lemma_conv == results[i]
                        and strip_accents(pos_lemma_conv.lower())
                        != strip_accents(word.lower())):
                    # A non-self result already agrees. Self-maps still need
                    # the scalar API's candidate and citation comparison.
                    continue

                # POS suggests a different lemma than baseline. Check if
                # the POS lemma is among the valid candidates.
                candidates = self._checked_pos_candidates(
                    word, upos, self._lemmatize_verbose_impl(word, guess=guess))

                # Check if any candidate matches the POS-specific lemma
                pos_stripped = strip_accents(pos_lemma_conv.lower())
                matched = False
                for c in candidates:
                    if c.lemma == pos_lemma_conv:
                        results[i] = c.lemma
                        matched = True
                        break
                if not matched:
                    # Try accent-stripped comparison
                    for c in candidates:
                        if strip_accents(c.lemma.lower()) == pos_stripped:
                            if self._pos_respell_blocked(word, upos, pos_lemma, c):
                                results[i] = c.lemma
                                loose_blocked.add(i)
                            else:
                                results[i] = self._prefer_exact_pos_citation(
                                    word, upos, c.lemma, pos_lemma_conv)
                            matched = True
                            break
                one_reading = not matched and self._one_reading_against_modern_entry(
                    word, upos, candidates)
                if not matched and (len(candidates) <= 1 or one_reading
                        or ((upos == "VERB"
                             or self._is_name_row_family(word, candidates)
                             or self._pos_split_decides(
                             word, upos, pos_lemma, candidates)
                             or self._closed_class_entry_decides(
                                 word, upos, pos_lemma, candidates)
                             or self._name_entry_decides(
                                 word, upos, pos_lemma, candidates))
                            and len({c.lemma.lower() for c in candidates}) == 1)):
                    # A case-twin family is one competing reading when a
                    # supported explicit verb tag selects a different lemma,
                    # and so is a name with its lowercase case alternate for
                    # a lowercase word. So is one lemma met from both sides
                    # of the lookup under a Modern entry (#119; as in
                    # lemmatize_pos). A loose entry leaves the candidates to
                    # decide (#105).
                    if candidates and self._pos_override_blocked(
                            word, upos, pos_lemma, candidates,
                            case_twin_family=upos == "VERB"
                            and len(candidates) > 1 and not one_reading):
                        loose_blocked.add(i)
                    else:
                        results[i] = pos_lemma_conv
                        matched = True
                if not matched and all(
                        strip_accents(c.lemma.lower()) == strip_accents(word.lower())
                        for c in candidates):
                    # Match the scalar POS override when the candidates only
                    # repeat the input with case or accent variations.
                    if self._pos_override_blocked(word, upos, pos_lemma, candidates):
                        loose_blocked.add(i)
                    else:
                        results[i] = pos_lemma_conv
                        matched = True
                if matched:
                    continue

            # MG self-map fix: when the baseline result is an MG self-map
            # for an ADJ/VERB inflection, prefer the citation form.
            # Also handles ADJ feminine-to-masculine fix (not a self-map
            # but still a wrong lemma form).
            if results[i] is not None and upos in ("ADJ", "VERB"):
                candidates = self._checked_pos_candidates(
                    word, upos, self._lemmatize_verbose_impl(word, guess=guess))
                if len(candidates) > 1:
                    mg_fix = self._fix_mg_selfmap(word, candidates, upos=upos)
                    if mg_fix is not None:
                        results[i] = mg_fix
                        continue

            if pos_lemma is None:
                # Compare structured readings even without a POS entry. Do
                # not repeat model inference for unresolved batch words;
                # the compound step still follows the caller's guess.
                candidates = self._checked_pos_candidates(
                    word, upos, self._lemmatize_verbose_impl(
                        word, guess=False, compound_guess=guess))
                nominal = self._nominal_reading(word, upos, candidates)
                if nominal is not None:
                    results[i] = nominal
                    continue
                if (len(candidates) == 1 and results[i] is not None
                        and strip_accents(candidates[0].lemma.lower())
                        == strip_accents(results[i].lower())
                        and not self._recase_disagrees(
                            word, upos, results[i], candidates[0].lemma)):
                    results[i] = candidates[0].lemma
                    continue

            if (len(candidates) > 1
                    and all(c.lemma.lower() == candidates[0].lemma.lower()
                            for c in candidates)):
                # An unmatched or absent POS entry does not disambiguate
                # case twins. Preserve the scalar API's candidate order and
                # existing PROPN/capitalization tiebreak.
                results[i] = self._prefer_case_twin(
                    word, upos, candidates[0].lemma, candidates,
                    row_decides=self._capital_row_decides(
                        word, upos, candidates))

        # A rejected batch baseline cannot hide a retained structured
        # candidate. Do not run the model again to recover that reading.
        for i, (word, upos) in enumerate(zip(words, upos_tags)):
            if rejected_baselines[i] and results[i] is None:
                candidates = self._checked_pos_candidates(
                    word, upos, self._lemmatize_verbose_impl(
                        word, guess=False, compound_guess=guess))
                if candidates:
                    results[i] = self._prefer_case_twin(
                        word, upos, candidates[0].lemma, candidates,
                        row_decides=self._capital_row_decides(
                            word, upos, candidates))

        # Step 3: capitalization-agreement tiebreak between case-twin
        # lemmas (θυμός vs Θυμός), mirroring lemmatize_pos(). Only
        # tokens whose result disagrees in case with the form/POS pay
        # for the candidate list.
        for i, (word, upos) in enumerate(zip(words, upos_tags)):
            if i in resolved_articles:
                continue  # A requested canonical article is not a name twin.
            res = results[i]
            if not res or not word:
                continue
            if upos == "PROPN":
                need = res[:1].islower()
            else:
                need = word[:1].islower() and is_capitalized(res)
            if not need:
                continue
            candidates = self._checked_pos_candidates(
                word, upos, self._lemmatize_verbose_impl(word, guess=guess))
            pos_lemma = (None if i in loose_blocked
                         else self._checked_pos_table_lookup(word, upos))
            extra_lemma = (self._apply_convention(pos_lemma)
                           if pos_lemma is not None else None)
            extra = (extra_lemma,) if extra_lemma is not None else ()
            results[i] = self._prefer_case_twin(word, upos, res,
                                                candidates, extra=extra)

        return self._without_skipped_ordinals(words, [
            lemma if i in bare_letters
            else self._checked_pos_result(word, upos, lemma)
            for i, (word, upos, lemma)
            in enumerate(zip(words, upos_tags, results))])

    @staticmethod
    def classify_nonlexical(token: str) -> str | None:
        """Return a NON-LEXICAL class label for ``token``, or ``None`` if it
        looks like a real lexical word.

        Recognizes the editorial/typographic residue that leaks into OCR'd
        corpora - γράφεται variant marks (γρ), Greek numerals (κζ', ,αφ'),
        bracket references ([76], [49-59]), Latin/citation abbreviations (fr.,
        Herod.), lone punctuation/sigla, and vowel-less consonant fragments -
        so a caller can exclude them from a lemmatization-failure count instead
        of treating them as real words that failed to resolve. The label is one
        of ``dilemma.nonlexical.NONLEXICAL_CLASSES``. Pure/stdlib, no model.
        """
        return _classify_nonlexical(token)

    @staticmethod
    def is_lexical(token: str) -> bool:
        """True iff ``token`` is a candidate lexical word (not classified
        NON-LEXICAL and not empty/whitespace). The complement of
        ``classify_nonlexical`` for non-empty tokens."""
        return _is_lexical(token)

    def lemmatize_verbose(self, word: str,
                          prev_word: str | None = None,
                          *,
                          attested_only: bool = False,
                          with_attestation: bool = False,
                          guess: bool = True,
                          ) -> list[LemmaCandidate]:
        """Return all candidate lemmas with metadata.

        Unlike lemmatize(), this returns multiple candidates for
        ambiguous forms, tagged with language, proper noun status,
        and source. Useful for downstream tools that can use context
        to disambiguate.

        Reviewed inflections also populate ``grammar`` with their grammatical
        lemma, analyses, related headwords, and evidence. Dictionary convention
        remapping affects ``lemma`` while preserving that grammatical record.

        Args:
            word: The Greek word to lemmatize.
            prev_word: Optional preceding word. If this is a Greek article
                (e.g. ὁ, τήν, τῶν), it is used to rank candidates by
                gender/number agreement. Only affects ranking, not
                filtering - all candidates are still returned.
            attested_only: If True and the input surface form is not attested
                in the corpus (exact NFC, or via elision expansion), return an
                empty list instead of candidates. The gate is on the FORM, so a
                real lemma can be dropped because the particular inflection typed
                is unattested.
            with_attestation: If True, set each candidate's ``attestation`` to
                the corpus attestation of the matched surface form (None when
                the form is unattested). Needs ``form_profile.db``.
            guess: If False, skip the transformer fallback and never append
                the ``identity``-source echo candidate; a word that nothing
                structured can resolve returns an empty list. Each returned
                candidate's ``score``/``source`` are the confidence signal.

        Examples:
            lemmatize_verbose("ἔριδι")
            -> [LemmaCandidate(lemma="Ἔρις", lang="grc", proper=True, ...),
                LemmaCandidate(lemma="ἔρις", lang="grc", proper=False, ...)]

            lemmatize_verbose("πόλεμο")
            -> [LemmaCandidate(lemma="πόλεμος", lang="el", ...),
                LemmaCandidate(lemma="πόλεμος", lang="grc", ...)]

            lemmatize_verbose("ἀλλ̓")
            -> [LemmaCandidate(lemma="ἀλλά", lang="grc", source="elision", via="elision:ά")]
        """
        if self._is_skipped_ordinal(word):
            return []
        # Digits / empty are not Greek forms; pass through ungated.
        attest_key = None
        if (attested_only or with_attestation) and word and not word.isdigit():
            attest_key = self._resolve_attested_form(word)
            if attested_only and attest_key is None:
                return []
        candidates = self._lemmatize_verbose_impl(word, prev_word, guess=guess)
        if with_attestation and attest_key is not None:
            payload = self.form_attestation(attest_key)
            for c in candidates:
                c.attestation = payload
        return candidates

    def _lemmatize_verbose_impl(self, word: str,
                                prev_word: str | None = None,
                                *, guess: bool = True,
                                compound_guess: bool | None = None,
                                ) -> list[LemmaCandidate]:
        """Build the candidate list (no attestation gate); see lemmatize_verbose.

        *compound_guess* decides whether the compound step may return a
        lemma no dictionary lists; it follows *guess* unless given. A
        caller that skips the model with ``guess=False`` on behalf of a
        guessing caller passes its own caller's setting.
        """
        candidates = []
        seen = set()  # Citation, language, and grammatical identity.
        supported_elisions = set()

        def _add(lemma, lang="", source="", via="", score=1.0, tag="", grammar=None):
            supported_elision = (via == "elision-exact"
                                  and self._positive_elision_target(lemma))
            check = self._apply_convention_check(
                lemma, lang=lang, source=source)
            if check.lemma is None:
                return
            lemma = check.lemma
            key = (lemma, lang, grammar["lemma"] if grammar else None)
            if key not in seen:
                seen.add(key)
                candidates.append(LemmaCandidate(
                    lemma=lemma,
                    lang=lang or self._lang_of(lemma),
                    proper=self._is_proper(lemma),
                    source=source,
                    score=score,
                    via=via,
                    tag=tag,
                    citation=check.reason,
                    grammar=grammar,
                ))
                if supported_elision:
                    supported_elisions.add(id(candidates[-1]))

        # 0. Digit-only passthrough
        if word.isdigit():
            _add(word, source="identity")
            return candidates
        # Lunate sigma (ϲ) -> standard σ/ς, so editions that use it still match.
        word = to_standard_sigma(word)
        # ῤῥ is an editorial spelling of ρρ.
        word = to_plain_double_rho(word)

        if self.lang in ("all", "grc"):
            readings = reviewed_verb_readings(word)
            if readings:
                original = self._lookup_word(word)
                owners = {reading["lemma"] for reading in readings}
                related = {head["lemma"] for reading in readings
                           for head in reading["related_headwords"]}
                if original and original not in owners | related:
                    # A reviewed verbal reading does not erase another
                    # exact lexical owner, e.g. the adverb ἦ beside its
                    # verbal homographs. Its unreviewed grammar stays None.
                    _add(original, lang="grc", source="lookup", via="exact")
                preferred = preferred_reviewed_lemma(word, original)
                readings.sort(key=lambda reading: reading["lemma"] != preferred)
                for grammar in readings:
                    _add(grammar["lemma"], lang="grc", source="lookup",
                         via="reviewed_inflection", tag="VERB", grammar=grammar)
                return candidates

        # 1. Article/pronoun
        closed = self._resolve_closed_class(word)
        if closed is not None:
            if self._resolve_articles:
                _add(closed, source="article")
            else:
                _add(closed, lang="grc", source="article", via="self", tag="DET")
            return candidates

        # 2. Crasis
        from .crasis import resolve_crasis
        cr = resolve_crasis(word) or resolve_crasis(to_monotonic(word))
        if cr:
            _add(cr, source="crasis")
            return candidates

        # 2.4 Explicit numerals (͵α, ιβʹ, κζ’) — classified NON-LEXICAL before
        #     elision can strip/expand their marks into words; matches
        #     lemmatize().
        if _is_explicit_numeral(word):
            _add(word, source="nonlexical", via=_classify_nonlexical(word),
                 score=0.0, tag=_NONLEXICAL_POS)
            return candidates

        # 2.5 Exact elided entry — the corpus-derived lemma for an elided form
        #     (ὅτ᾽->ὅτε, κ᾽->ἄν, μ᾽->ἐγώ), keyed under the canonical U+1FBD mark.
        #     Added BEFORE the vowel expander so it becomes candidates[0]; this
        #     is what makes lemmatize_pos()/the Tagger agree with lemmatize()
        #     (both otherwise let the frequency-ranked homograph, ὅτι, win).
        #     The lang is resolved eagerly so the dedupe key matches the
        #     expander's when both find the same lemma (no double ὅτε).
        exact_elided = self._lookup_elided(word)
        if exact_elided:
            _add(exact_elided, lang=self._lang_of(exact_elided) or "grc",
                 source="lookup", via="elision-exact")

        # 3. Elision expansion — collect ALL valid expansions (before
        #    lookup, since elided forms false-match letter headwords)
        elision_results = self._expand_elision_all(word)
        for expanded, lemma, vowel in elision_results:
            lang = self._lang_of(expanded) or self._lang_of(lemma)
            _add(lemma, lang=lang, source="elision", via=f"elision:{vowel}")

        # 4. Lookup — collect from ALL language tables. Skipped entirely for
        #    a consonant-psili elision (γ̓, μ̓): its accent-stripped variant
        #    false-matches letter-name headwords (γ̓ -> Γ), and the elision
        #    candidates above are the only legitimate readings.
        lower = word.lower()
        mono = to_monotonic(lower)
        stripped = strip_accents(lower)
        variants = [] if _is_consonant_psili_elision(word) else [
            (word, "exact"), (lower, "lower"),
            (mono, "mono"), (stripped, "stripped"),
        ]
        # The capitalized loose keys of the generated paradigms, after
        # the lowercase ones and for an Ancient head only, as in
        # _lookup_word.
        capitalized_keys = set()
        if variants:
            tried = {variant for variant, _via in variants}
            for via, kinds in (("mono", {"stripped": False}),
                               ("stripped", {"monotonic": False})):
                for variant in self._capitalized_loose_keys(word, **kinds):
                    if variant not in tried:
                        variants.append((variant, via))
                        capitalized_keys.add(variant)

        # Respect self.lang: when constructed with lang='el' (MG mode),
        # don't leak AG candidates (and vice versa). This matters for
        # lemmatize_pos() which returns the first matching candidate:
        # αυτό in MG mode must not match AG's αὐτός, and είναι (MG copula)
        # must not match AG's εἰμί.
        _tables: list[tuple[object, str]] = []
        if self.lang in ("all", "el"):
            _tables.append((self._mg_lookup, "el"))
            _tables.append((self._med_lookup, "med"))
        if self.lang in ("all", "grc"):
            _tables.append((self._ag_lookup, "grc"))
        junk_value_seen = False
        prefer_modern = (self.lang == "el"
                         or self._convention_name == "triantafyllidis")
        for table, lang in _tables:
            for variant, via in variants:
                lemma = table.get(variant)
                if lemma and variant in capitalized_keys and not (
                        lang == "grc"
                        and self._is_capitalized_ancient_head(lemma)):
                    continue
                if lemma:
                    # Skip trivial short self-mappings (accent artifacts)
                    if len(variant) <= 2 and lemma == variant and via in ("mono", "stripped"):
                        continue
                    # A loose key may not drop the form's breathing (#109).
                    if via in ("mono", "stripped") and not prefer_modern:
                        lemma = self._restore_breathing(word, lemma)
                        if not lemma:
                            continue
                    # An elided junk VALUE (key ἡνίκ -> "ἡνίκ'") is not a
                    # headword; note it for the elided-stem bridging below
                    # and reject this table's hit, matching lemmatize().
                    if _is_elided_junk_value(lemma):
                        junk_value_seen = True
                        break
                    _add(lemma, lang=lang, source="lookup", via=via)
                    # Also check if the OTHER case variant is a headword
                    # (Ἔρις the goddess vs ἔρις strife)
                    if is_capitalized(lemma):
                        alt = lemma[0].lower() + lemma[1:]
                    else:
                        # Titlecase keeps Greek prosgegrammeni in one initial
                        # character; uppercase expands ᾅ into the all-caps ἍΙ.
                        alt = lemma[0].title() + lemma[1:]
                    # The alt must be a self-mapping headword (not just
                    # a form that maps elsewhere)
                    alt_lemma = table.get(alt)
                    if alt_lemma == alt:
                        _add(alt, lang=lang, source="lookup",
                             via=via + "+case_alt")
                    break  # first matching variant wins per language

        # 4b. Bare elided stem bridging (matches lemmatize()): a junk value
        #     marks the key as an elided stem whose apostrophe was tokenized
        #     off; restore the mark and collect the elided readings
        #     (ἀλλ -> ἀλλά, οὐδ -> οὐδέ).
        if junk_value_seen and not candidates:
            restored = word + _KORONIS
            bridged_exact = self._lookup_elided(restored)
            if bridged_exact:
                _add(bridged_exact,
                     lang=self._lang_of(bridged_exact) or "grc",
                     source="lookup", via="elision-exact")
            for expanded, el_lemma, vowel in self._expand_elision_all(restored):
                el_lang = self._lang_of(expanded) or self._lang_of(el_lemma)
                _add(el_lemma, lang=el_lang, source="elision",
                     via=f"elision:{vowel}")

        # 5. Particle suffix stripping (after lookup, before model)
        if not candidates:
            particle_lemma = self._strip_particle_suffix(word)
            if particle_lemma:
                lang = self._lang_of(particle_lemma) or "grc"
                _add(particle_lemma, lang=lang, source="particle_strip",
                     via="suffix_strip", score=0.9)

        # 6. Verb morphology: an oblique participle through its nominative,
        # then augment and reduplication stripping
        if not candidates:
            participle_lemma = (self._participle_nominative_lemma(word)
                                or self._aorist_participle_lemma(word))
            if participle_lemma:
                lang = self._lang_of(participle_lemma) or "grc"
                _add(participle_lemma, lang=lang, source="verb_morphology",
                     via="participle_nominative", score=0.85)
        if not candidates:
            verb_morph_lemma = self._strip_verb_morphology(word)
            if verb_morph_lemma:
                lang = self._lang_of(verb_morph_lemma) or "grc"
                _add(verb_morph_lemma, lang=lang, source="verb_morphology",
                     via="augment_strip", score=0.85)

        # 7. Normalizer: try orthographic variants against all tables
        if not candidates and self._normalizer:
            for norm_candidate in self._normalizer.normalize(word):
                norm_lower = norm_candidate.lower()
                norm_mono = to_monotonic(norm_lower)
                norm_stripped = strip_accents(norm_lower)
                norm_variants = [
                    (norm_candidate, "normalize"),
                    (norm_lower, "normalize+lower"),
                    (norm_mono, "normalize+mono"),
                    (norm_stripped, "normalize+stripped"),
                ]
                for table, lang in _tables:
                    for variant, via in norm_variants:
                        lemma = table.get(variant)
                        if lemma:
                            if len(variant) <= 2 and lemma == variant and via.endswith(("mono", "stripped")):
                                continue
                            if via.endswith(("mono", "stripped")) and not prefer_modern:
                                lemma = self._restore_breathing(norm_candidate, lemma)
                                if not lemma:
                                    continue
                            if _is_elided_junk_value(lemma):
                                continue
                            if not self._normalization_reading_supported(
                                    norm_candidate, lemma, source_word=word):
                                continue
                            _add(lemma, lang=lang, source="normalize", via=via)
                            break

        # 7a. Attested-key-gated orthographic fallbacks, matching lemmatize():
        #     editorial/manuscript spelling variants (dropped iota subscript,
        #     movable nu, stray dieresis, digamma). A variant is accepted only
        #     if it itself resolves in the lookup, so this never invents a
        #     lemma. Before this step the verbose path (which lemmatize_pos()
        #     and the Tagger use) silently lacked these fallbacks, so the two
        #     paths disagreed on forms like πρόσθεϝ.
        if not candidates:
            for ortho_candidate in self._ortho_fallback_variants(word):
                lemma = self._lookup_word(ortho_candidate)
                if lemma and not _is_elided_junk_value(lemma):
                    _add(lemma, source="ortho_fallback",
                         via=f"variant:{ortho_candidate}", score=0.8)
                    break

        # 7b. Non-lexical classification (BEFORE the model): if nothing has
        #     resolved and the token is an apparatus mark / numeral / editorial
        #     ref / abbreviation / vowel-less siglum, tag it NON-LEXICAL (POS X)
        #     and stop. This keeps non-words out of the transformer and lets a
        #     caller tell "not a word" from "failed to lemmatize a real word".
        if not candidates:
            nonlex = _classify_nonlexical(word)
            if nonlex is not None:
                _add(word, source="nonlexical", via=nonlex, score=0.0,
                     tag=_NONLEXICAL_POS)
                return candidates

        # 8. Model fallback (if no candidates yet). With guess=False the
        #    transformer is skipped; the identity-fallback heuristics below
        #    (steps 9-12) still run, limited to what the lookup or a
        #    dictionary headword supports.
        model_identity = False
        if not candidates:
            if not guess:
                model_identity = True
            else:
                try:
                    self._load_model()
                    pred = self._predict([word])[0]
                    if strip_accents(pred.lower()) != strip_accents(word.lower()):
                        _add(pred, source="model", score=0.5)
                    else:
                        model_identity = True
                except (FileNotFoundError, RuntimeError, ImportError, IndexError):
                    model_identity = True

        # 9. Light Byzantine normalization (only when model returned identity)
        if model_identity and not candidates:
            byz = self._normalize_byzantine(word, use_normalizer=False)
            if byz:
                _add(byz, source="byzantine_norm", score=0.7)

        # 10. Compound decomposition (only when model returned identity)
        if model_identity:
            for compound, base_lemma, prefix in self._decompose_compound_all(
                    word, guess=(guess if compound_guess is None
                                 else compound_guess)):
                _add(compound, source="compound",
                     via=f"{prefix}+{base_lemma}", score=0.65)

        # 11. Prefix stripping (only when model returned identity)
        if model_identity and not candidates:
            prefix_hit = self._prefix_strip_lookup(word)
            if prefix_hit:
                _add(prefix_hit, source="prefix_strip", score=0.6)

        # 12. Heavy Byzantine normalization with normalizer
        if model_identity and not candidates:
            byz_heavy = self._normalize_byzantine(
                word, use_normalizer=True)
            if byz_heavy:
                _add(byz_heavy, source="byzantine_norm", score=0.5)

        # If still nothing, return the word itself - unless guess=False, in
        # which case an empty list is the honest answer ("I don't know"): an
        # identity echo is indistinguishable from a successful identity
        # lemmatization to the caller.
        if not candidates:
            if not guess:
                return candidates
            _add(word, source="identity", score=0.0)

        # Refine lookup candidate scores using corpus frequency.
        # Only activated when there are multiple lookup candidates to rank.
        lookup_candidates = [c for c in candidates if c.source == "lookup"]
        if len(lookup_candidates) > 1:
            # Collect frequencies for all lookup candidates' lemmas. These
            # are LEMMAS, so use the lemma-keyed attestation table; the
            # form-keyed corpus_freq table would hand every candidate the
            # summed count of all homographs of its accent-stripped spelling
            # (and erased the Ἔρις/ἔρις case distinction entirely).
            freq_pairs = []
            for c in lookup_candidates:
                # Hand-checked corpora only (GLAUx, Diorisis): the all-auto
                # OGA source lowercases every lemma, so counting it would
                # bias case-twin scoring toward the lowercase twin.
                att = self.attestation(c.lemma)
                freq = 0
                if att:
                    sc = att.get("source_counts") or {}
                    freq = sc.get("glaux", 0) + sc.get("diorisis", 0)
                freq_pairs.append((c, freq))
            # Only refine if at least one candidate has frequency data
            has_freq = any(f > 0 for _, f in freq_pairs)
            if has_freq:
                max_log = max(math.log1p(f) for _, f in freq_pairs)
                for c, freq in freq_pairs:
                    if freq > 0:
                        # Scale log(1+freq) into [0.5, 1.0]
                        c.score = 0.5 + 0.5 * (math.log1p(freq) / max_log)
                    else:
                        # No corpus data: slight penalty vs attested forms
                        c.score = 0.55

        # Independently supported exact elisions precede reconstructed
        # expansions, including names. Other candidates retain their order.
        candidates.sort(key=lambda c: (id(c) not in supported_elisions,
                                       c.proper, -c.score))

        # Dedupe after convention/citation normalization. _add() already applies
        # this, but the pass is cheap and keeps older in-place mutations honest.
        if self._convention_map or self._convention_name == "lsj" \
                or self._convention_monotonic or self._citation_policy != "default":
            seen_remapped = set()
            remapped = []
            for c in candidates:
                check = self._apply_convention_check(
                    c.lemma, lang=c.lang, source=c.source)
                if check.lemma is None:
                    continue
                c.lemma = check.lemma
                c.citation = check.reason
                key = (c.lemma, c.lang)
                if key not in seen_remapped:
                    seen_remapped.add(key)
                    remapped.append(c)
            candidates = remapped

        # Article-agreement disambiguation: if prev_word is a Greek article,
        # re-rank candidates by gender agreement (only re-ranks, never excludes)
        if prev_word is not None and len(candidates) > 1:
            candidates = self._rank_by_article_agreement(candidates, prev_word)

        return candidates

    def lemmatize_batch(self, words: list[str], *,
                        attested_only: bool = False,
                        guess: bool = True) -> list[str | None]:
        """Lemmatize a batch of words. Uses model only for unknowns.

        If a convention is set, all output lemmas are remapped accordingly.

        When ``attested_only=True``, any word whose surface form is unattested
        in the corpus (exact NFC, or via elision expansion) yields ``None`` at
        its position, preserving positional alignment with the input. Digits and
        empty strings pass through unchanged. Needs ``form_profile.db``.

        When ``guess=False``, the transformer fallback is skipped and any word
        that nothing structured can resolve yields ``None`` at its position
        instead of a model guess or an echo of the input (see ``lemmatize``).
        """
        results = []
        model_indices = []
        model_words = []

        for i, word in enumerate(words):
            # Pass through tokens with no Greek letters (punctuation, digits)
            # unchanged - matches lemmatize().
            if not word or not any(
                    0x0370 <= ord(c) <= 0x03ff or 0x1f00 <= ord(c) <= 0x1fff
                    for c in word):
                results.append(word)
                continue
            # NFC-normalize (sources may emit NFD), then lunate sigma -> σ/ς
            # and the editorial ῤῥ -> ρρ.
            word = unicodedata.normalize("NFC", word)
            word = to_standard_sigma(word)
            word = to_plain_double_rho(word)
            # Article/pronoun resolution
            closed = self._resolve_closed_class(word)
            if closed is not None:
                results.append(closed)
                continue

            # Crasis
            from .crasis import resolve_crasis
            cr = resolve_crasis(word) or resolve_crasis(to_monotonic(word))
            if cr:
                results.append(cr)
                continue

            # Explicit numerals (͵α, ιβʹ, κζ’) pass through before elision can
            # strip/expand their marks into words; matches lemmatize().
            if _is_explicit_numeral(word):
                results.append(word)
                continue

            # A final vowel whose only mark is a psili carries a breathing
            # (ἐ, δἐ), so the lookup reads it first, as in lemmatize(); the
            # expander would read the breathing as an elision mark
            # (δἐ -> δέω instead of δέ).
            if (_is_bare_vowel_psili_final(word)
                    and not _is_marked_vowel_psili_elision(word)):
                lemma = self._lookup_word(word)
                if lemma and not _is_elided_junk_value(lemma):
                    results.append(lemma)
                    continue

            # Exact elided entry (ὅτ᾽->ὅτε, μ᾽->ἐγώ), before the expander —
            # matches lemmatize(): the corpus-derived lemma must beat the
            # frequency-ranked homograph (ὅτι/καί/ὁ/εἷς).
            exact_elided = self._lookup_elided(word)
            if exact_elided:
                results.append(exact_elided)
                continue

            # Elision expansion (before lookup — elided forms can
            # false-match letter headwords)
            elision_lemma = self._expand_elision(word)
            if elision_lemma:
                results.append(elision_lemma)
                continue

            # Lookup (with the same elided-form guards as lemmatize():
            # no accent-stripped letter-name junk for consonant-psili
            # elisions, no elided junk values; a junk value routes the bare
            # elided stem through the elision machinery, ἀλλ -> ἀλλά)
            lemma = None
            if not _is_consonant_psili_elision(word):
                lemma = self._lookup_word(word)
                if lemma and _is_elided_junk_value(lemma):
                    lemma = (self._lookup_elided(word + _KORONIS)
                             or self._expand_elision(word + _KORONIS))
            if lemma:
                results.append(lemma)
                continue

            # Particle suffix stripping
            particle_lemma = self._strip_particle_suffix(word)
            if particle_lemma:
                results.append(particle_lemma)
                continue

            # Verb morphology stripping (participle nominative, augment,
            # reduplication)
            verb_morph_lemma = (self._participle_nominative_lemma(word)
                                or self._aorist_participle_lemma(word)
                                or self._strip_verb_morphology(word))
            if verb_morph_lemma:
                results.append(verb_morph_lemma)
                continue

            # Normalizer: try orthographic variants against lookup
            if self._normalizer:
                norm_hit = None
                for candidate in self._normalizer.normalize(word):
                    norm_hit = self._lookup_grammatical_word(
                        candidate, normalization=True, source_word=word)
                    if norm_hit and _is_elided_junk_value(norm_hit):
                        norm_hit = None
                        continue
                    if norm_hit:
                        break
                if norm_hit:
                    results.append(norm_hit)
                    continue

            # Attested-key-gated orthographic fallbacks (matches lemmatize()).
            ortho_hit = None
            for candidate in self._ortho_fallback_variants(word):
                ortho_hit = self._lookup_word(candidate)
                if ortho_hit and _is_elided_junk_value(ortho_hit):
                    ortho_hit = None
                    continue
                if ortho_hit:
                    break
            if ortho_hit:
                results.append(ortho_hit)
                continue

            # Non-lexical tokens (apparatus marks, numerals, refs, sigla) are
            # not words: return them unchanged instead of sending them to the
            # model (matches lemmatize()).
            if _classify_nonlexical(word) is not None:
                results.append(word)
                continue

            results.append(None)
            model_indices.append(i)
            model_words.append(word)

        if model_words:
            if guess:
                try:
                    self._load_model()
                    predictions = self._predict(model_words)
                except (RuntimeError, IndexError, ImportError,
                        FileNotFoundError):
                    predictions = model_words  # identity fallback
            else:
                # guess=False: skip the transformer entirely; the structured
                # fallbacks below still run, and a word nothing resolves
                # yields None instead of an echo (matches lemmatize()).
                predictions = [None] * len(model_words)
            for idx, word, pred in zip(model_indices, model_words, predictions):
                # Fallback strategies when model returns identity or was skipped
                if pred is None or (strip_accents(pred.lower())
                                    == strip_accents(word.lower())):
                    # 1. Light Byzantine normalization (safe/precise)
                    byz = self._normalize_byzantine(
                        word, use_normalizer=False)
                    if byz:
                        pred = byz
                    # 2. Compound decomposition (linking vowel split)
                    elif (compound := self._decompose_compound(
                            word, guess=guess)):
                        pred = compound
                    # 3. Prefix stripping (known prefixes + base)
                    elif (pfx := self._prefix_strip_lookup(word)):
                        pred = pfx
                    # 4. Heavy Byzantine normalization (with normalizer)
                    else:
                        byz_heavy = self._normalize_byzantine(
                            word, use_normalizer=True)
                        if byz_heavy:
                            pred = byz_heavy
                results[idx] = pred

        # Apply convention remapping and citation-form validation to all
        # results. This always runs, even without a named convention, because
        # lookup/model artifacts can contain non-citation grave lemmas.
        remapped_results = []
        for original, result in zip(words, results):
            if not result:
                remapped_results.append(result)
                continue
            source = ""
            normalized_original = unicodedata.normalize("NFC", original)
            normalized_original = to_standard_sigma(normalized_original)
            if (result == normalized_original
                    and _classify_nonlexical(normalized_original) is not None):
                source = "nonlexical"
            remapped_results.append(
                self._apply_convention(result, source=source))
        results = remapped_results

        # Attested-only gate: null out words whose surface form is unattested.
        # (Digits / empty pass through; they are not Greek forms.)
        if attested_only:
            results = [
                None if (w and not w.isdigit()
                         and self._resolve_attested_form(w) is None) else r
                for w, r in zip(words, results)
            ]

        return self._without_skipped_ordinals(words, results)

    # ---- Spelling correction ----

    # Greek lowercase letters for ED1 candidate generation
    _GREEK_LETTERS = "αβγδεζηθικλμνξοπρσςτυφχψω"

    def _build_spell_index(self):
        """Build the accent-stripped index for spelling correction.

        With SQLite backend: no-op (queries use the indexed 'stripped' column).
        With JSON fallback: builds in-memory norm map from dict.
        """
        if hasattr(self, "_spell_norm_map"):
            return
        if self._using_db:
            # SQLite path: no in-memory index needed
            self._spell_norm_map = None
            self._spell_norm_set = None
            return
        norm_map: dict[str, set[str]] = {}
        for form in self._lookup:
            stripped = strip_accents(form.lower())
            if stripped not in norm_map:
                norm_map[stripped] = set()
            norm_map[stripped].add(form)
        self._spell_norm_map = norm_map
        self._spell_norm_set = set(norm_map.keys())

    @staticmethod
    def _edits1(word: str) -> set[str]:
        """Generate all strings within edit distance 1 of word.

        Operations: deletes, transposes, replaces, inserts.
        Uses Greek lowercase alphabet for replacements and insertions.
        """
        letters = Dilemma._GREEK_LETTERS
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def suggest_spelling(self, word: str, max_distance: int = 2,
                         ocr_mode: bool = False,
                         headwords_only: str | None = None,
                         lemmata_only: bool = False,
                         ) -> list[tuple[str, int | float]]:
        """Suggest spelling corrections for an unknown Greek word.

        Returns a list of (correct_form, edit_distance) tuples, sorted
        by edit distance then alphabetically. Uses a two-layer approach:

        1. Strip diacritics from the input and the dictionary, reducing
           8-11M entries to ~1-3M unique base forms
        2. Find ED0/ED1/ED2 matches on the stripped forms
        3. Return the original polytonic forms, ranked by actual
           Levenshtein distance to the input

        This means diacritic errors (wrong accent, missing breathing)
        cost 0 in the first layer and are corrected for free, while
        letter-level errors (θ/δ, ρ/ν) use standard edit distance.

        Args:
            word: The possibly-misspelled Greek word.
            max_distance: Maximum edit distance (1 or 2). Default 2.
            ocr_mode: If True, use weighted Levenshtein distance that
                gives lower cost to OCR-common confusions (Greek/Latin
                script mixing, Cyrillic contamination, θ/δ, ο/σ).
                This produces better rankings for OCR post-correction.
            headwords_only: If set to a convention name (e.g. "lsj",
                "cunliffe"), only return forms that are known headwords
                in that dictionary. This filters out inflected forms and
                headwords from other dictionaries, reducing false
                positives when resolving to a specific lexicon.
            lemmata_only: If True, only return forms that are lemmata
                (citation forms) in the lookup table, filtering out
                inflected forms. Less restrictive than headwords_only
                since it includes all lemmata, not just those in a
                specific dictionary.

        Returns:
            List of (corrected_form, distance) tuples. Empty if no
            suggestions found within max_distance.
        """
        self._build_spell_index()
        query_stripped = strip_accents(word.lower())

        if self._using_db:
            results = self._suggest_spelling_db(word, query_stripped,
                                                max_distance, ocr_mode)
        else:
            results = self._suggest_spelling_mem(word, query_stripped,
                                                 max_distance, ocr_mode)

        if headwords_only:
            hw_set = self._get_headword_set(headwords_only)
            results = [(form, dist) for form, dist in results
                       if form in hw_set]
        elif lemmata_only:
            lemma_set = self._get_lemma_set()
            results = [(form, dist) for form, dist in results
                       if form in lemma_set]

        return results

    def _suggest_spelling_db(self, word: str, query_stripped: str,
                             max_distance: int, ocr_mode: bool = False
                             ) -> list[tuple[str, int | float]]:
        """SQLite-backed spelling suggestion. No in-memory index needed.

        Generates ED1/ED2 candidate strings, then batch-queries the
        indexed 'stripped' column. ~1000 candidates for ED1, checked
        in one SQL query.
        """
        # For AG mode, filter to AG-sourced forms only (src='grc')
        # to avoid suggesting monotonic MG forms
        src_filter = 'grc' if self.lang == 'grc' else None

        # Collect candidate stripped forms at each distance level
        candidates: set[str] = set()

        # ED0: just the query itself
        candidates.add(query_stripped)

        # ED1: all edits of the stripped query
        if max_distance >= 1:
            ed1 = self._edits1(query_stripped)
            candidates.update(ed1)

        # Look up which candidates actually exist in the DB
        hits = self._lookup.spell_lookup_stripped(candidates,
                                                  src_filter=src_filter)

        # ED2: if few hits so far, expand
        if max_distance >= 2 and len(hits) < 3:
            ed2_candidates: set[str] = set()
            for e1 in ed1:
                ed2_candidates.update(self._edits1(e1))
            # Remove already-checked candidates
            ed2_candidates -= candidates
            ed2_hits = self._lookup.spell_lookup_stripped(ed2_candidates,
                                                          src_filter=src_filter)
            hits.update(ed2_hits)

        if not hits:
            return []

        return self._rank_spell_results(word, query_stripped, hits, ocr_mode)

    def _suggest_spelling_mem(self, word: str, query_stripped: str,
                              max_distance: int, ocr_mode: bool = False
                              ) -> list[tuple[str, int | float]]:
        """In-memory spelling suggestion (JSON fallback)."""
        norm_hits: set[str] = set()

        if query_stripped in self._spell_norm_set:
            norm_hits.add(query_stripped)

        if not norm_hits or max_distance >= 1:
            for candidate in self._edits1(query_stripped):
                if candidate in self._spell_norm_set:
                    norm_hits.add(candidate)

        if max_distance >= 2 and len(norm_hits) < 3:
            for e1 in self._edits1(query_stripped):
                for candidate in self._edits1(e1):
                    if candidate in self._spell_norm_set:
                        norm_hits.add(candidate)

        if not norm_hits:
            return []

        # Convert to {stripped: [original_forms]} format
        hits = {n: list(self._spell_norm_map[n]) for n in norm_hits}
        return self._rank_spell_results(word, query_stripped, hits, ocr_mode)

    @staticmethod
    def _has_breathing(s: str) -> bool:
        """Check if a string contains Greek breathing marks (polytonic)."""
        nfd = unicodedata.normalize("NFD", s)
        return "\u0313" in nfd or "\u0314" in nfd

    def _get_frequency(self, headword: str) -> int:
        """Get reference count for a headword from lsj9 frequency data.

        Lazy-loads lsj9_frequency.json on first call. Returns 0 for
        unknown headwords.
        """
        if self._hw_frequency is None:
            self._hw_frequency = {}
            if LSJ9_FREQUENCY_PATH.exists():
                with open(LSJ9_FREQUENCY_PATH, encoding="utf-8") as f:
                    self._hw_frequency = json.load(f)
        return self._hw_frequency.get(headword, 0)

    def attestation(self, lemma: str) -> dict | None:
        """Corpus attestation profile for an Ancient Greek lemma, or None.

        Returns the per-lemma record from ``data/lemma_attestation.json``: the
        deduped frequency ``total`` with its ``by_genre`` (10 bins),
        ``by_century`` (signed century integer, where ``-8`` = 8th c. BC and
        ``2`` = 2nd c. AD), and ``by_dialect`` (GLAUx only; present only when
        known) breakdowns, plus ``source_counts`` (each corpus's independent
        token count -- overlapping, not summed; agreement is a confidence
        signal) and ``dominant_pos``. Because GLAUx and Diorisis annotate the
        same texts, counts are deduplicated per work (GLAUx preferred), so a
        lemma seen only in a non-preferred source's reading of a shared work has
        ``total`` 0 but a non-empty ``source_counts``. Returns None for a lemma
        not attested in GLAUx or Diorisis.

        The argument is matched (after NFC normalization) against the corpora's
        own polytonic lemma annotation; it is not accent-stripped. Because the
        corpora are not sense-disambiguated, there is one profile per lemma
        string -- LSJ (A)/(B) homographs are not separated; use
        ``dominant_pos`` to pick a noun-vs-verb homograph. The artifact's
        ``_meta`` documents the full schema. Lazily loaded on first call.
        """
        if self._attestation is None:
            self._attestation = {}
            if ATTESTATION_PATH.exists():
                with open(ATTESTATION_PATH, encoding="utf-8") as f:
                    self._attestation = json.load(f).get("lemmas", {})
        return self._attestation.get(unicodedata.normalize("NFC", lemma))

    # ---- Lemma annotation repair ----

    def _canonical_ag_headwords(self) -> tuple[set, dict, set]:
        """Lazy canonical AG headword inventory for ``repair_lemma``.

        Returns ``(all_headwords, stripped_index, lsj_only)`` where
        ``all_headwords`` is the union of the LSJ and Cunliffe headword
        sets, ``stripped_index`` maps each accent-stripped lowercase
        skeleton to its headwords (so diaeresis/accent/breathing variants
        and edit-distance candidates can be matched), and ``lsj_only`` is
        the LSJ subset (used as a tiebreak toward the primary lexicon).
        """
        if self._canonical_hw is None:
            hw: set[str] = set()
            lsj: set[str] = set()
            for path, bucket in ((LSJ_HEADWORDS_PATH, lsj),
                                 (CUNLIFFE_HEADWORDS_PATH, None)):
                if not path.exists():
                    continue
                with open(path, encoding="utf-8") as f:
                    raw = json.load(f)
                entries = raw if isinstance(raw, list) else list(raw)
                hw.update(entries)
                if bucket is not None:
                    bucket.update(entries)
            idx: dict[str, list[str]] = {}
            for h in hw:
                idx.setdefault(strip_accents(h.lower()), []).append(h)
            self._canonical_hw = (hw, idx, lsj)
        return self._canonical_hw

    def _canonicalize_headword(self, s: str) -> str | None:
        """Map a (possibly corrupt) lemma string onto a canonical AG headword.

        Tries, in order: exact membership; digamma normalization (ϝ -> ν,
        then ϝ dropped); an accent-stripped exact match (absorbs diaeresis,
        breathing, and accent-convention differences, ἐυμμελίης ->
        ἐϋμμελίης); and finally edit-distance-1 on the accent-stripped
        skeleton (OCR-grade damage: μάκαπ -> μάκαρ, διαπέταμαι ->
        διαπέτομαι, ἐπιτροχάδη -> ἐπιτροχάδην). Multiple matches are ranked
        by case agreement with the input, corpus attestation, then LSJ
        membership. Returns None when nothing canonical is within reach.
        """
        canon, idx, lsj = self._canonical_ag_headwords()
        if not canon:
            return None

        def _lsj_spelling(h: str) -> str:
            # Prefer the LSJ spelling of the same accent-stripped skeleton
            # for a Cunliffe-only diacritic variant (ἐυμμελίης ->
            # ἐϋμμελίης). Case must agree so a common noun is never
            # remapped onto a proper-noun twin.
            if h in lsj:
                return h
            twins = [t for t in idx.get(strip_accents(h.lower()), ())
                     if t in lsj and is_capitalized(t) == is_capitalized(h)]
            return min(twins) if twins else h

        if s in canon:
            return _lsj_spelling(s)

        # Digamma noise (also handled for forms in _ortho_fallback_variants)
        if "ϝ" in s or "Ϝ" in s:
            for v in (s.replace("ϝ", "ν").replace("Ϝ", "Ν"),
                      s.replace("ϝ", "").replace("Ϝ", "")):
                if v in canon:
                    return v
                hit = self._canonicalize_headword(v) if v != s else None
                if hit:
                    return hit

        key = strip_accents(s.lower())
        matches = list(idx.get(key, ()))
        if not matches and len(key) >= 4:
            # ED1 on the stripped skeleton against the canonical inventory.
            # Short skeletons are excluded: at 1-3 letters a single edit
            # reaches half the particle inventory, which is noise, not repair.
            for cand_key in self._edits1(key):
                matches.extend(idx.get(cand_key, ()))
        if not matches:
            return None

        def _rank(h: str) -> tuple:
            case_mismatch = (is_capitalized(h) != is_capitalized(s))
            att = self.attestation(h)
            freq = att["total"] if att else 0
            return (case_mismatch, -freq, h not in lsj, h)

        return _lsj_spelling(min(matches, key=_rank))

    def repair_lemma(self, lemma: str, *, form: str | None = None) -> str | None:
        """Validate or repair a lemma annotation, optionally guided by the
        surface form it was assigned to.

        Lemma annotations in treebanks and OCR-derived data are corrupt in
        learnable ways: OCR letter confusions (μάκαπ for μάκαρ), digamma
        noise (ϝέκταρ for νέκταρ, πρόσθεϝ for πρόσθεν), diaeresis and
        gemination variation (ἐυμελίης for ἐϋμμελίης), and truncation
        (ἐπιτροχάδη for ἐπιτροχάδην). This repairs them against the
        canonical AG headword inventory (LSJ + Cunliffe), preferring
        evidence from the surface ``form`` - which is usually correct even
        when the annotation is not - over the damaged ``lemma`` string.

        Resolution order:
          1. A ``lemma`` that is already a canonical headword is returned
             directly, normalized onto its LSJ spelling when the input is a
             Cunliffe-only diacritic variant (this API repairs damaged
             strings; it does not second-guess a well-formed annotation -
             use the lemmatizer for that).
          2. With ``form`` given: the form is lemmatized without guessing
             (``guess=False``) and each candidate is canonicalized
             (exact / digamma / diaeresis / edit-distance-1); the first
             canonical hit wins. A form that is itself a canonical headword
             (indeclinables: ἐπιτροχάδην) also counts.
          3. The ``lemma`` string itself is canonicalized the same way.
          4. Whatever the lookup resolves ``lemma`` to is canonicalized.

        Returns the canonical headword, or None when nothing trustworthy
        resolves - never an unconstrained model guess, and never the
        corrupt input echoed back. This is an ANCIENT GREEK repair: a valid
        Modern Greek citation form is returned unchanged when an MG table
        is loaded, but on a grc-only instance MG input may be "repaired"
        onto an AG headword. Raises RuntimeError if the LSJ/Cunliffe
        headword files are missing (``python -m dilemma download``).

        Examples:
            repair_lemma("μάκαπ")                      -> "μάκαρ"
            repair_lemma("ϝέκταρ")                     -> "νέκταρ"
            repair_lemma("Πειραί", form="Πειραΐδης")   -> "Πειραΐδης"
            repair_lemma("ξζψχ")                       -> None
        """
        if not lemma:
            return None
        lemma = unicodedata.normalize("NFC", to_standard_sigma(lemma.strip()))
        canon, _idx, _lsj = self._canonical_ag_headwords()
        if not canon:
            raise RuntimeError(
                "repair_lemma needs the LSJ/Cunliffe headword files "
                "(data/lsj_headwords.json, data/cunliffe_headwords.json). "
                "Download them with `python -m dilemma download`.")
        if lemma in canon:
            # Already canonical; normalize a Cunliffe-only diacritic variant
            # onto its LSJ spelling when one exists (ἐυμμελίης -> ἐϋμμελίης).
            return self._canonicalize_headword(lemma)
        # A valid Modern Greek citation form is not corrupt Ancient Greek:
        # do not "repair" the Demotic σπήλαιο onto the AG σπήλαιον. (Only
        # applies when MG data is loaded, i.e. lang='all'/'el'.)
        if self.lang in ("all", "el") and self._is_mg_citation_form(lemma):
            return lemma

        if form:
            form = unicodedata.normalize("NFC", to_standard_sigma(form.strip()))
            for c in self._lemmatize_verbose_impl(form, guess=False):
                if c.source == "nonlexical":
                    continue
                hit = self._canonicalize_headword(c.lemma)
                if hit:
                    return hit
            # An indeclinable surface form can itself be the headword the
            # annotation truncated (ἐπιτροχάδην -> ἐπιτροχάδη).
            if form in canon:
                return form

        hit = self._canonicalize_headword(lemma)
        if hit:
            return hit

        resolved = self._lookup_word(lemma)
        if resolved and resolved != lemma:
            return self._canonicalize_headword(resolved)
        return None

    # ---- Form-keyed corpus attestation ("attested only" + references) ----

    def _attestdb(self):
        """Lazily open the form-attestation DBs (form_profile / form_citations).

        Returns the shared :class:`dilemma._attest_db.AttestDB`. The DBs are
        resolved from the same data dir as the rest of Dilemma's artifacts.
        """
        if self._attest is None:
            from ._attest_db import AttestDB
            # No explicit dir: AttestDB scans the candidate data dirs
            # ($DILEMMA_DATA_DIR, dev tree, download cache), so the form DBs are
            # found wherever `dilemma download --with-attestation` placed them.
            self._attest = AttestDB()
        return self._attest

    def _resolve_attested_form(self, word: str) -> str | None:
        """The attested surface key for ``word``, or None if it is unattested.

        Returns the exact NFC form when that occurs in the corpus, else an
        elision expansion that does (an elided token like ``ἀλλ̓`` is
        canonically attested as its full form ``ἀλλά``; a strict exact check on
        the raw elided token would miss it). This is the input-direction gate's
        match: exact NFC polytonic, plus elision resolution.
        """
        db = self._attestdb()
        if db.is_attested(word):
            return unicodedata.normalize("NFC", word)
        for expanded, _lemma, _vowel in self._expand_elision_all(word):
            if db.is_attested(expanded):
                return expanded
        return None

    def form_attestation(self, form: str, *, max_citations: int | None = 20):
        """Corpus attestation for an exact polytonic surface FORM, or None.

        The form-keyed sibling of :meth:`attestation` (which is lemma-keyed).
        Returns a dict ``{form, attested, total_count, n_works, source_counts,
        by_century, by_genre, by_dialect, by_century_genre, dominant_pos,
        citations}`` for a form attested in GLAUx or Diorisis, or ``None`` if
        the exact NFC form does not occur in the corpus. ``total_count`` and the
        distributions reflect the full evidence; ``max_citations`` bounds only
        the returned ``citations`` list (``None`` returns every stored citation,
        ``0`` skips them). ``citations`` is empty (with a ``citations_note``)
        unless the opt-in ``form_citations.db`` has been downloaded.

        Raises if ``form_profile.db`` is not available; download it with
        ``python -m dilemma download``.
        """
        return self._attestdb().attestation(form, max_citations=max_citations)

    def _rank_spell_results(self, word: str, query_stripped: str,
                            hits: dict[str, list[str]],
                            ocr_mode: bool = False
                            ) -> list[tuple[str, int | float]]:
        """Rank spelling suggestions by edit distance.

        Sorts by (stripped_dist, full_dist, -frequency, form) so that
        at the same stripped distance, forms closer to the original
        polytonic input are preferred, then more common headwords win.
        Polytonic forms (with breathing marks) are preferred over
        monotonic when the input is polytonic or when using AG mode.

        In ocr_mode, uses weighted Levenshtein that gives lower cost
        to OCR-common character confusions (Greek/Latin script mixing,
        Cyrillic contamination, theta/delta, omicron/sigma).
        """
        prefer_polytonic = self.lang == "grc" or self._has_breathing(word)
        dist_fn = _weighted_levenshtein if ocr_mode else _levenshtein

        results: list[tuple[str, float, float]] = []
        for norm, originals in hits.items():
            stripped_dist = _levenshtein(query_stripped, norm)
            for original in originals:
                full_dist = dist_fn(word.lower(), original.lower())
                results.append((original, stripped_dist, full_dist))

        # Deduplicate: keep best (lowest stripped_dist, then full_dist)
        best: dict[str, tuple[float, float]] = {}
        for form, sd, fd in results:
            if form not in best or (sd, fd) < best[form]:
                best[form] = (sd, fd)

        if prefer_polytonic:
            by_sd: dict[float, list[str]] = {}
            for form, (sd, fd) in best.items():
                by_sd.setdefault(sd, []).append(form)
            for sd, forms in by_sd.items():
                poly = [f for f in forms if self._has_breathing(f)]
                if poly:
                    for f in forms:
                        if not self._has_breathing(f):
                            del best[f]

        # Sort by: stripped_dist, full_dist, then prefer common headwords
        return sorted(
            [(form, sd) for form, (sd, fd) in best.items()],
            key=lambda x: (x[1], best[x[0]][1],
                           -self._get_frequency(x[0]), x[0]),
        )

    def _model_headword_set(self) -> set[str]:
        """Headword inventory used to constrain model fallback candidates."""
        strict = self._strict_ag_citation_applies(source="model")
        cache_name = ("_strict_model_headwords" if strict
                      else "_model_headwords")
        cached = getattr(self, cache_name, None)
        if cached is not None:
            return cached

        if strict:
            headwords = set(self._trusted_grave_citation_targets())
        else:
            headwords = set(self._trusted_grave_citation_targets())
            # Broad default mode keeps Wiktionary/self-map coverage, but still
            # runs every lookup-derived headword through the grave citation gate.
            for k, v in self._lookup.items():
                if k != v:
                    continue
                checked = self._normalize_grave_citation_lemma(v)
                if checked:
                    headwords.add(checked)

        setattr(self, cache_name, headwords)
        return headwords

    def _accepted_model_candidate(self, candidate: str) -> str | None:
        """Return the validated headword spelling for a model beam, if any."""
        if not candidate:
            return None
        headwords = self._model_headword_set()
        strict = self._strict_ag_citation_applies(source="model")
        variants = (
            candidate,
            candidate.lower(),
            to_monotonic(candidate),
            to_monotonic(candidate).lower(),
            candidate[0].upper() + candidate[1:],
        )
        for variant in variants:
            check = self._check_citation_lemma(
                variant, source="model", strict=strict)
            if check.lemma and check.lemma in headwords:
                return check.lemma
        return None

    def _predict(self, words: list[str], num_beams=4) -> list[str]:
        """Reuse context-independent model guesses across lemma and tier calls."""
        from collections import OrderedDict
        cache = getattr(self, "_prediction_cache", None)
        if cache is None:
            cache = self._prediction_cache = OrderedDict()
        answers = {word: cache[(word, num_beams)] for word in words
                   if (word, num_beams) in cache}
        missing = list(dict.fromkeys(word for word in words if word not in answers))
        if missing:
            predictions = self._predict_uncached(missing, num_beams=num_beams)
            if len(predictions) != len(missing):
                raise RuntimeError("Model changed the number of predictions")
            for word, prediction in zip(missing, predictions):
                answers[word] = prediction
                cache[(word, num_beams)] = prediction
        for word in words:
            key = (word, num_beams)
            if key in cache:
                cache.move_to_end(key)
        while len(cache) > 65536:
            cache.popitem(last=False)
        return [answers[word] for word in words]

    def _predict_uncached(self, words: list[str], num_beams=4) -> list[str]:
        """Run model inference with beam search + headword filtering.

        Generates multiple candidates via beam search. Picks the
        highest-scoring candidate that validates as a known citation
        headword. If no candidate is a headword, returns the input word
        unchanged (better than a confidently wrong answer).

        Works with both PyTorch and ONNX backends transparently.
        """
        if not words:
            return []

        max_len = max(len(w) for w in words) + 1
        src_ids = []
        for w in words:
            ids = self._vocab.encode(w)
            ids = ids + [0] * (max_len - len(ids))
            src_ids.append(ids)

        if getattr(self, '_use_onnx', False):
            import numpy as np
            # ONNX MHA reshapes require consistent sequence lengths.
            # Pad all inputs to a fixed max to avoid shape mismatches.
            ONNX_MAX_LEN = 48
            padded = []
            for ids in src_ids:
                if len(ids) < ONNX_MAX_LEN:
                    ids = ids + [0] * (ONNX_MAX_LEN - len(ids))
                padded.append(ids[:ONNX_MAX_LEN])
            src = np.array(padded, dtype=np.int64)
            src_pad_mask = (src == 0)
            beam_results = self._model.generate(
                src, src_key_padding_mask=src_pad_mask, num_beams=num_beams)
        else:
            import torch
            src = torch.tensor(src_ids, dtype=torch.long, device=self._device)
            src_pad_mask = (src == 0)
            with torch.no_grad():
                beam_results = self._model.generate(
                    src, src_key_padding_mask=src_pad_mask, num_beams=num_beams)

        results = []
        greedy = (num_beams == 1)
        onnx_mode = getattr(self, '_use_onnx', False)
        for i, candidates in enumerate(beam_results):
            if greedy and not onnx_mode:
                # num_beams=1 with PyTorch: beam_results is (batch, seq_len)
                # tensor, each row is token IDs with no score
                decoded = [self._vocab.decode(candidates)]
            else:
                # ONNX always returns list of (ids, score) tuples;
                # PyTorch beam search (num_beams>1) does the same
                decoded = [self._vocab.decode(ids) for ids, score in candidates]
            chosen = None
            for d in decoded:
                chosen = self._accepted_model_candidate(d)
                if chosen:
                    break
            if chosen is None:
                chosen = words[i]
            results.append(chosen)

        return results

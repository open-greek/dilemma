#!/usr/bin/env python3
"""Build SQLite lookup database from per-language lookup tables.

Reads from raw_lookups.db (SQLite, written by build_data.py) when available,
falling back to JSON files. Combines AG, MG, and Medieval lookups with
AG-first priority merging.

Output:
    data/lookup.db       - main form->lemma lookup (~1.1 GB)
    data/spell_index.db  - stripped form->original form index for spell checking

Startup: near-instant via mmap (vs ~11s for JSON loading)

Usage:
    python build_lookup_db.py
"""

import csv
import hashlib
import json
import os
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from dilemma.form_sanitize import (  # noqa: E402
    MIN_POLYTONIC_SHARE, has_editorial_sigla, polytonic_share, sanitize_form,
)
from dilemma.citation import malformed_tonal_reason  # noqa: E402
from dilemma import grave_to_acute, to_monotonic  # noqa: E402
from dilemma.core import trusted_ag_citation_headwords  # noqa: E402
from dilemma.nonlexical import classify_nonlexical  # noqa: E402
from dilemma.elision_tails import (  # noqa: E402
    REASON as ELISION_TAIL_REASON,
    find_elision_tails,
    open_profile_attestation,
)
from dilemma.el_paradigm_gaps import complete_lookup  # noqa: E402
from dilemma.grc_adjective_gaps import complete_ikos_adjectives  # noqa: E402
from dilemma.grc_noun_gaps import (  # noqa: E402
    SOURCE as GRC_NOUN_GAPS_SOURCE, complete_noun_gaps, select_noun_gap_lemmas)
from dilemma.grc_rejected_lemmas import (  # noqa: E402
    CLOSED_PRONOMINAL_LEMMAS, REJECTED_LEMMAS,
)
from dilemma.grammatical_lemmas import (  # noqa: E402
    reviewed_verb_forms, reviewed_verb_lemmas, reviewed_verb_readings,
)

DATA_DIR = SCRIPT_DIR / "data"
DB_PATH = DATA_DIR / "lookup.db"
SPELL_DB_PATH = DATA_DIR / "spell_index.db"
RAW_DB_PATH = DATA_DIR / "raw_lookups.db"
CITATION_REJECTIONS_PATH = DATA_DIR / "citation_hygiene_rejections.tsv"

AG_PATH = DATA_DIR / "ag_lookup.json"
LOOKUP_SOURCES_PATH = DATA_DIR / "lookup_sources.json"
GRC_IKOS_ADJECTIVES_PATH = DATA_DIR / "grc_ikos_adjectives.json"
# Inputs that say which layer of a merged file supplied a reading.
AG_WIKTIONARY_BASE_PATH = DATA_DIR / "ag_lookup_wikt.json"
AG_EXPANSION_MANIFEST_PATH = DATA_DIR / "ag_expansion_reference.json"
AG_EXPANSION_REFERENCE_ENV = "DILEMMA_AG_EXPANSION_REFERENCE"
CHANGES_DIR = DATA_DIR / "changes"
EL_RULE_PARADIGM_RECORDS = ("el_present_completion.json",
                            "el_imperfect_completion.json")
AG_WIKTIONARY_LAYER = "ag_lookup.json#wiktionary"
AG_EXPANSION_LAYER = "ag_lookup.json#expansion"
EL_RULE_PARADIGM_LAYER = "mg_lookup.json#rule-paradigms"
LSJ_HEADWORDS_PATH = DATA_DIR / "lsj_headwords.json"
CUNLIFFE_HEADWORDS_PATH = DATA_DIR / "cunliffe_headwords.json"
AG_HEADWORDS_PATH = DATA_DIR / "ag_headwords.json"
DGE_HEADWORDS_PATH = DATA_DIR / "dge_headwords.json"
LGPN_NAMES_PATH = DATA_DIR / "lgpn_names.json"
PD_HEADWORDS_PATH = DATA_DIR / "pd_headwords.json"
VLG_HEADWORDS_PATH = DATA_DIR / "vlg_headwords.json"
WIP_HEADWORDS_PATH = DATA_DIR / "wip_headwords.json"
LSJ10_HEADWORDS_PATH = DATA_DIR / "lsj10_headwords.json"
LBG_HEADWORDS_PATH = DATA_DIR / "lbg_headwords.json"
LBG_PAIRS_PATH = DATA_DIR / "lbg_pairs.json"
CORPUS_FREQ_PATH = DATA_DIR / "corpus_freq.json"
# Surface-form corpus attestation (build/build_form_attestation.py). An
# opt-in download, so the corpus word-tail filter below states plainly when
# it is missing rather than silently changing what the build ships.
FORM_PROFILE_PATH = DATA_DIR / "form_profile.db"
MG_PATH = DATA_DIR / "mg_lookup.json"
MED_PATH = DATA_DIR / "med_lookup.json"
GLAUX_PAIRS_PATH = DATA_DIR / "glaux_pairs.json"
DIORISIS_PAIRS_PATH = DATA_DIR / "diorisis_pairs.json"
# Openly-licensed Koine NT (Nestle 1904 lowfat, macula-greek, CC BY 4.0;
# Nestle 1904 base text is public domain). Open replacement for the dropped
# CC BY-NC-SA PROIEL NT. Built by build/extract_nt.py.
NT_PAIRS_PATH = DATA_DIR / "nt_pairs.json"
# Lowest-priority manual-gold gap-fill from cog's standardized exports
# (extract_ptnk.py / extract_tagnt.py / extract_pedalion.py). Each only fills
# forms no earlier source has; none can override Wiktionary/LSJ/GLAUx/Diorisis.
COG_GAPFILL_PATHS = [
    ("PTNK", DATA_DIR / "ptnk_pairs.json"),
    ("TAGNT", DATA_DIR / "tagnt_pairs.json"),
    ("Pedalion", DATA_DIR / "pedalion_pairs.json"),
]
PERSEUS_PAIRS_PATH = DATA_DIR / "perseus_pairs.json"
ETYMOLOGY_BRIDGES_PATH = DATA_DIR / "etymology_bridges.json"
LSJGR_BRIDGES_PATH = DATA_DIR / "lsjgr_bridges.json"
RELATED_LEMMAS_PATH = DATA_DIR / "related_lemmas.json"
HNC_PAIRS_PATH = DATA_DIR / "hnc_pairs.json"
# The lexical heads every Ancient reading is checked against (built by
# build/build_grc_lexical_heads.py, published on the Hub).
GRC_LEXICAL_HEADS_PATH = DATA_DIR / "grc_lexical_heads.json"

_TRUSTED_BUILD_HEADWORD_PATHS = (
    LSJ_HEADWORDS_PATH, LSJ10_HEADWORDS_PATH, LBG_HEADWORDS_PATH,
    DGE_HEADWORDS_PATH, VLG_HEADWORDS_PATH, CUNLIFFE_HEADWORDS_PATH,
    PD_HEADWORDS_PATH, WIP_HEADWORDS_PATH,
)

# The lookup is openly licensed by default: PROIEL (CC BY-NC-SA) is excluded
# entirely (not even used for evaluation); Perseus (the CC BY-SA AGDT
# original) is kept; the Gorman treebanks (CC BY-SA 4.0) are deliberately NOT
# ingested - they are the project's held-out gold corpus
# (eval/eval_gorman_gold.py); glaux_pairs.json is built with the
# NonCommercial GLAUx texts already filtered out (build/nc_filter.py +
# build/build_glaux_pairs.py). See NOTICE.


def strip_accents(s):
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if unicodedata.category(c) != "Mn"))


def _is_self_map(form, lemma):
    return (form == lemma
            or strip_accents(form.lower()) == strip_accents(lemma.lower()))


def _has_grave(s: str) -> bool:
    return "\u0300" in unicodedata.normalize("NFD", s)


# Readings the 2026-11 kaikki dumps change for the worse, restored or
# corrected from the treebanks (GLAUx form-lemma counts in parentheses). The
# dumps are right that accented ἤν is ἐάν (GLAUx 1,045), but the unaccented
# and monotonic keys belong to the far commoner ἦν; a new adverb page made
# μεγάλα its own lemma; εἴσομαι and ἑψόμενα fell to εἶμι and ἕπω; τοῖ
# (1.3.25: τέῳ, the dumps: ὁ) is Doric σοί; κἆτα is καὶ εἶτα, not ἄτη (the
# GLAUx annotation of its two tokens is wrong; κᾆτα: εἶτα 103).
KAIKKI_2026_11_READING_OVERRIDES = {
    "ην": "εἰμί", "ήν": "εἰμί",                 # ἦν 29,294
    "μεγάλα": "μέγας", "μεγαλα": "μέγας",       # 1,638
    "εἴσομαι": "οἶδα", "είσομαι": "οἶδα", "εισομαι": "οἶδα",  # 43 (ἔρχομαι 2)
    "ἑψόμενα": "ἕψω",                            # 42
    "τοῖ": "σύ",                                 # 25
    "κἆτα": "εἶτα",                              # crasis καὶ εἶτα
    # The new page ἐπαινέειν ("uncontracted present active infinitive of
    # ἐπαινέω") is self-mapped by the base and takes the contracted
    # infinitive with it. GLAUx: ἐπαινεῖν -> ἐπαινέω 437, ἐπαινέειν 5.
    "ἐπαινεῖν": "ἐπαινέω", "επαινείν": "ἐπαινέω", "επαινειν": "ἐπαινέω",
    "ἐπαινέειν": "ἐπαινέω", "επαινέειν": "ἐπαινέω", "επαινεειν": "ἐπαινέω",
    # The rebuilt merged layer and the regenerated LSJ verb paradigms read
    # these forms as homographs of rare words, against the treebanks and the
    # 1.3.25 readings: νοερῶν and νοεροῖς as the verb νοερόω (GLAUx: νοερός
    # 11 and 5), παστάδα and παστάδος as παστός (παστάς 4 and 6), θεανθρώπου
    # and οἰκιοῦντες as θεανθρωπόω and οἰκίον, μηνας as the name Μηνᾶς
    # (μῆνας of μείς), ἑψομένης as ἕπω (ἕψω 5), ἠμφεσβήτησαν as the
    # generated ἀμφιαμφισβητέω.
    "νοερῶν": "νοερός", "νοεροῖς": "νοερός",
    "παστάδα": "παστάς", "παστάδος": "παστάς",
    "θεανθρώπου": "θεάνθρωπος", "οἰκιοῦντες": "οἰκίζω", "μηνας": "μείς",
    "ἑψομένης": "ἕψω", "ἠμφεσβήτησαν": "ἀμφισβητέω",
    # Pinned 1.3.1 resolutions the expansion exclusions still hold, which
    # the corpus gap-fill now supplies: GLAUx files ἀπεικός under ἀφίημι
    # (LSJ ἀπεικός, of ἀπέοικα), the superlative πορρωτάτω under the
    # comparative προσωτέρω (GLAUx: πόρρω 182) and σαλπισταῖς under σάλπιγξ.
    "ἀπεικός": "ἀπεικός",
    "πορρωτάτω": "πόρρω", "ποῤῥωτάτω": "πόρρω",
    "σαλπισταῖς": "σαλπιστής",
}

# Reviewed readings that are written even when no source has the key. An
# override changes only a key the tables hold, so a form whose single
# reading the build rejects would otherwise be left without an answer.
REVIEWED_ABSENT_KEY_READINGS = {
    # The citation-hygiene sanitizer drops the dirty μὴτ value of the bare
    # elided stem before the overrides run. The clean stem stays as an
    # explicit bridge to μήτε, like the other apostrophe-tokenized stems.
    "μήτ": "μήτε",
}

# The input every reviewed reading is recorded under (lookup_sources.json).
LOOKUP_OVERRIDES_INPUT = "lookup_overrides"

# The Ancient corpus pair files. Their gap-fill also writes a pair into the
# el table when the Ancient table agrees with it or lacks the form.
ANCIENT_CORPUS_PAIR_INPUTS = frozenset(path.name for path in (
    PERSEUS_PAIRS_PATH, GLAUX_PAIRS_PATH, DIORISIS_PAIRS_PATH, NT_PAIRS_PATH,
    *(path for _, path in COG_GAPFILL_PATHS)))

# Forms whose reviewed reading also displaces an el-only row of a corpus
# gap-fill. The override corrects the combined and the Ancient table; the
# corpus pair the gap-fill copied into the el table before that would
# otherwise be written as an el row and answer lang="el" with the reading
# the review rejected (εἴσομαι -> εἶμι, ἀπεικός -> ἀφίημι, Μάγων -> Μάγων).
# Limited to the readings reviewed for the 2026-11 rebuild: about fifty
# older el-only corpus rows disagree with an override too (ᾔει, ᾔδετε), and
# the published tables answer lang="el" with those.
EL_CORPUS_ROW_OVERRIDE_FORMS = (
    frozenset(KAIKKI_2026_11_READING_OVERRIDES)
    | frozenset(REVIEWED_ABSENT_KEY_READINGS) | {"Μάγων"})


def apply_lookup_overrides(overrides: dict, *, combined: dict,
                           combined_sources: dict, ag: dict, ag_sources: dict,
                           el: dict, el_sources: dict,
                           absent_keys=REVIEWED_ABSENT_KEY_READINGS,
                           el_corpus_forms=EL_CORPUS_ROW_OVERRIDE_FORMS
                           ) -> dict[str, int]:
    """Write the reviewed readings into the assembled tables, in place.

    A reading replaces the combined and the Ancient reading of a key the
    tables hold. A key no table holds is added to the combined table only
    when ``absent_keys`` lists it. For a form in ``el_corpus_forms``, an el
    reading that an Ancient corpus gap-fill supplied and that differs from
    the reviewed one is removed, so the combined row answers lang="el" too;
    el readings from any other source stay. Returns how many combined
    readings changed, how many keys were added and how many el readings
    were removed.
    """
    counts = {"changed": 0, "added": 0, "el_removed": 0}
    for form, lemma in overrides.items():
        if form in combined:
            if combined[form] != lemma:
                combined[form] = lemma
                combined_sources[form] = LOOKUP_OVERRIDES_INPUT
                counts["changed"] += 1
        elif form in absent_keys:
            combined[form] = lemma
            combined_sources[form] = LOOKUP_OVERRIDES_INPUT
            counts["added"] += 1
        # Also fix in ag dict for the grc-only table
        if form in ag and ag[form] != lemma:
            ag[form] = lemma
            ag_sources[form] = LOOKUP_OVERRIDES_INPUT
        if (form in el_corpus_forms and form in el and el[form] != lemma
                and el_sources.get(form) in ANCIENT_CORPUS_PAIR_INPUTS):
            del el[form]
            el_sources.pop(form, None)
            counts["el_removed"] += 1
    return counts


_FINAL_ELISION_MARKS = {"\u2019", "\u02bc", "'", "\u1fbd", "`",
                        "\u1fbf", "\u0313", "\u0314"}
_FINAL_KERAIA_OR_PRIME_MARKS = {"\u0374", "\u02b9"}


def _normalize_grave_citation_lemma(lemma: str,
                                    trusted_targets: set[str]) -> str | None:
    """Return a citation-safe lemma, or None for an unproven grave value."""
    if not isinstance(lemma, str) or not _has_grave(lemma):
        return lemma
    acute = grave_to_acute(lemma)
    if acute != lemma and acute in trusted_targets:
        return acute
    return None


# Annotation placeholders that corpus exports put in the lemma field where no
# lemma was assigned: the CoNLL-U empty marker, an ellipsis, and the gap
# marker a treebank uses for a damaged or unreadable passage. None is a lemma.
PLACEHOLDER_LEMMAS = frozenset({"_", "...", "G"})


def _citation_artifact_reason(lemma: str) -> str | None:
    """Return why a lemma value is not a citation-form candidate, if any."""
    if not isinstance(lemma, str) or not lemma:
        return None
    if unicodedata.combining(lemma[0]):
        return "leading_combining"
    if "\u0305" in unicodedata.normalize("NFD", lemma):
        return "overline"
    tonal_reason = malformed_tonal_reason(lemma)
    if tonal_reason:
        return tonal_reason
    if lemma[-1] in _FINAL_KERAIA_OR_PRIME_MARKS:
        nonlexical = classify_nonlexical(lemma)
        if nonlexical:
            return f"nonlexical_{nonlexical}"
        return "final_keraia_or_prime"
    if lemma[-1] in _FINAL_ELISION_MARKS:
        nonlexical = classify_nonlexical(lemma)
        if nonlexical:
            return f"nonlexical_{nonlexical}"
        return "final_elision_mark"
    return None


def _bare_letters(text: str) -> str:
    """The letters of a spelling, case kept, every combining mark removed."""
    return unicodedata.normalize("NFC", "".join(
        c for c in unicodedata.normalize("NFD", text)
        if not unicodedata.combining(c)))


def load_expansion_lemma_heads(path: Path) -> tuple[frozenset, dict]:
    """The heads an unaccented expansion lemma may stand for.

    Returns every Ancient and vernacular head of grc_lexical_heads.json,
    and the Ancient heads indexed by their bare letters, the index
    ``unaccented_expansion_lemma`` repairs a lemma through.
    """
    evidence = json.loads(path.read_text(encoding="utf-8"))
    ancient = evidence["heads"]
    heads = frozenset(ancient) | frozenset(evidence.get("vernacular_heads", ()))
    by_letters: dict[str, set] = {}
    for head in ancient:
        by_letters.setdefault(_bare_letters(head), set()).add(head)
    return heads, by_letters


def unaccented_expansion_lemma(lemma: str, heads, heads_by_letters,
                               is_citation=None
                               ) -> tuple[str, str | None] | None:
    """Judge a dictionary-expansion lemma that carries no diacritic at all.

    The expansion layer holds lemmas that are source-text artefacts, not
    citation forms: Latin words read as Beta Code (αελιυς, σερvιλια), OCR
    garble (λαβλυμι, λαβ’μι), hyphenated composition lemmas
    (προ-ανακηρυσσω) and unaccented doublets of real heads (συνοχος). A
    lemma with any combining mark, or one a head list has (τε, που, κρεω),
    is kept: None. Otherwise the reading is repaired to the single Ancient
    head with the same letters, ``("unaccented_expansion_lemma_repaired",
    head)``, or dropped, ``("unaccented_expansion_lemma_unlisted", None)``.
    With ``is_citation``, the head must also pass it (the build asks for a
    head the lookup maps to itself): the head lists carry inflected
    cross-reference entries (κλεῖδα, κλεῖδες, of κλείς), and a reading
    repaired to one of them is another non-lemma, so it is dropped instead,
    ``("unaccented_expansion_lemma_head_not_citation", None)``.
    """
    if not isinstance(lemma, str) or not lemma:
        return None
    if any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", lemma)):
        return None
    if lemma in heads:
        return None
    same_letters = heads_by_letters.get(lemma, ())
    if len(same_letters) == 1:
        head = next(iter(same_letters))
        if is_citation is not None and not is_citation(head):
            return "unaccented_expansion_lemma_head_not_citation", None
        return "unaccented_expansion_lemma_repaired", head
    return "unaccented_expansion_lemma_unlisted", None


def capitalized_pronoun_twin(form: str, lemma: str, tables) -> str | None:
    """The pronoun a capitalised corpus form's lowercase spelling reads as.

    GLAUx and Diorisis give some sentence-initial pronouns a capitalised
    name lemma, which the build lowercases into another real head: Αὕτη
    becomes ἄτη, Τούτου τοῦτος, Πάντα πάν, Ἡμῶν ἡμός. When the form's
    lowercase spelling (first letter lowered) already reads, from a
    non-corpus source, as a closed-class pronominal lemma
    (``CLOSED_PRONOMINAL_LEMMAS``) and the corpus pair names another,
    capitalised-free lemma, return that pronoun so the pair is skipped and
    the capitalised form falls back to its lowercase reading. ``tables`` is
    a sequence of (readings, is_corpus) pairs to look the twin up in.
    """
    if not form or not lemma or not form[0].isupper() or lemma[0].isupper():
        return None
    lower = form[0].lower() + form[1:]
    if lower == form:
        return None
    for readings, is_corpus in tables:
        twin = readings.get(lower)
        if (twin in CLOSED_PRONOMINAL_LEMMAS and twin != lemma
                and not is_corpus(lower)):
            return twin
    return None


def _load_from_sqlite(table: str) -> dict:
    """Load a lookup table from raw_lookups.db."""
    if not RAW_DB_PATH.exists():
        return {}
    conn = sqlite3.connect(str(RAW_DB_PATH))
    try:
        rows = conn.execute(f"SELECT form, lemma FROM {table}").fetchall()
        conn.close()
        return dict(rows)
    except sqlite3.OperationalError:
        conn.close()
        return {}


def _load_from_json(path: Path) -> dict:
    """Load a lookup table from JSON (fallback)."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_lookup(table: str, json_path: Path,
                 label: str) -> tuple[dict, str]:
    """Load lookup, preferring whichever source has more entries.

    raw_lookups.db has base Wiktionary entries (~2.36M AG), while the
    JSON files may have LSJ-expanded entries (~9.97M AG). Always prefer
    the larger source to avoid losing LSJ forms from the spell index.
    """
    t0 = time.time()

    sqlite_data = _load_from_sqlite(table)
    json_data = _load_from_json(json_path)

    if sqlite_data and json_data:
        if len(json_data) > len(sqlite_data):
            print(f"  {label}: {len(json_data):,} entries from JSON "
                  f"(preferred over SQLite's {len(sqlite_data):,}) "
                  f"({time.time()-t0:.1f}s)")
            return json_data, json_path.name
        else:
            print(f"  {label}: {len(sqlite_data):,} entries from SQLite ({time.time()-t0:.1f}s)")
            return sqlite_data, f"{RAW_DB_PATH.name}:{table}"
    elif sqlite_data:
        print(f"  {label}: {len(sqlite_data):,} entries from SQLite ({time.time()-t0:.1f}s)")
        return sqlite_data, f"{RAW_DB_PATH.name}:{table}"
    elif json_data:
        print(f"  {label}: {len(json_data):,} entries from JSON ({time.time()-t0:.1f}s)")
        return json_data, json_path.name
    else:
        print(f"  {label}: no data found")
        return {}, f"{RAW_DB_PATH.name}:{table}|{json_path.name}"


def _write_citation_rejection_report(rejections: list[dict],
                                     path: Path = CITATION_REJECTIONS_PATH
                                     ) -> None:
    """Write deterministic source-aware diagnostics for rejected lemmas."""
    ordered = sorted(
        rejections,
        key=lambda item: (
            item["source"], item["reason"], item["table"],
            item["lemma"], item["form"],
        ),
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("table", "source", "reason", "form", "lemma"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(ordered)


def drop_corpus_word_tails(tables, *, is_attested, record):
    """Strip corpus word-tails from every assembled table, in place.

    `tables` is an ordered sequence of `(name, table, source_of)`, where
    `table` maps form to lemma and `source_of` names the ingestion path a form
    arrived by. Every table is filtered, not just the merged one: `lookup.db`
    writes grc-only and el-only rows from the `ag` and `el` tables wherever
    those differ from `combined`, so dropping a tail from `combined` alone
    would put it straight back as a row of its own.

    Each table is judged against the forms in ALL of them, not only its own,
    so a tail cannot survive by having its full word in a different table.

    `record` receives one audit row per rejection, with the sibling the tail
    was cut from as its evidence. Returns `{name: {form: sibling}}`.

    The rule and the measurements behind it live in `dilemma/elision_tails.py`.
    """
    dropped = {}
    # Every table is also a sibling source for every other one. A tail in `ag`
    # whose full word is only in `el` is judged against that full word, so it
    # cannot be dropped from `combined` and then written back by the
    # Ancient-Greek-only row pass.
    sibling_tables = [table for _, table, _ in tables]
    for name, table, source_of in tables:
        tails = find_elision_tails(table, is_attested=is_attested,
                                   sibling_tables=sibling_tables)
        for form, sibling in sorted(tails.items()):
            record(table=name, source=source_of(form), form=form,
                   lemma=table[form],
                   reason=f"{ELISION_TAIL_REASON}_of_{sibling}")
            del table[form]
        if tails:
            listed = ", ".join(f"{f} (end of {t})"
                               for f, t in sorted(tails.items()))
            print(f"  Dropped {len(tails):,} {name} corpus word-tails: "
                  f"{listed}")
        dropped[name] = tails
    return dropped


LOOKUP_SCHEMA_VERSION = 3


def lookup_content_sha256(conn) -> str:
    """Hash the lookup's logical rows: form, lemma, src, lang and source code.

    Independent of page layout, row order, ids and indexes, so two files with
    the same readings and provenance share it and a consumer can pin the
    content instead of the file bytes.
    """
    digest = hashlib.sha256()
    for row in conn.execute(
            "SELECT l.form, m.text, l.src, l.lang, s.code FROM lookup l "
            "JOIN lemmas m ON m.id = l.lemma_id "
            "JOIN sources s ON s.id = l.source_id "
            "ORDER BY l.form, l.lang, m.text, l.src, s.code"):
        digest.update("\t".join(row).encode("utf-8") + b"\n")
    return digest.hexdigest()


def load_lookup_sources(path: Path = None) -> tuple[list[dict], dict[str, str]]:
    """The reviewed source registry and the map from build input to source code."""
    registry = json.loads((path or LOOKUP_SOURCES_PATH).read_text(encoding="utf-8"))
    sources = registry["sources"]
    by_input: dict[str, str] = {}
    codes = set()
    for source in sources:
        code = source["code"]
        if code in codes:
            raise ValueError(f"lookup source {code!r} is listed twice")
        codes.add(code)
        for missing in {"name", "inputs", "share_alike", "components",
                        "attribution"} - source.keys():
            raise ValueError(f"lookup source {code!r} has no {missing!r}")
        for component in source["components"]:
            if not component.get("license"):
                raise ValueError(f"lookup source {code!r} has a component "
                                 "without a license")
        for label in source["inputs"]:
            if by_input.setdefault(label, code) != code:
                raise ValueError(f"input {label!r} belongs to two sources")
    return sources, by_input


def ag_reading_layers(ag: dict, base: dict, reference: dict) -> dict[str, str]:
    """The layer of ag_lookup.json that recorded each reading, where known.

    ag_lookup.json merges Wiktionary's inflection tables with paradigms
    generated from dictionary headwords. Two byte-pinned files tell the layers
    apart (data/ag_expansion_reference.json): the Wiktionary-only base and the
    reference whose keys absent from that base are the dictionary expansion.
    A reading equal to the base's is Wiktionary's; a reading equal to the
    expansion layer's is the expansion's. Any other reading (a newer
    Wiktionary revision, a regenerated paradigm, a recovered alias) keeps the
    merged file as its source, because no pinned record says which it is.
    """
    layers = {}
    for form, lemma in ag.items():
        recorded = base.get(form)
        if recorded is not None:
            if recorded == lemma:
                layers[form] = AG_WIKTIONARY_LAYER
        elif reference.get(form) == lemma:
            layers[form] = AG_EXPANSION_LAYER
    return layers


def read_ag_layer_records() -> tuple[dict, dict]:
    """The pinned Wiktionary-only base and expansion reference, verified."""
    import overlay_lsj
    for path in (AG_EXPANSION_MANIFEST_PATH, AG_WIKTIONARY_BASE_PATH):
        if not path.exists():
            raise SystemExit(f"{path.name} not found; it is needed to record "
                             "which layer of ag_lookup.json supplied a reading")
    manifest = overlay_lsj.load_manifest(AG_EXPANSION_MANIFEST_PATH)
    overlay_lsj.verify_file(AG_WIKTIONARY_BASE_PATH, manifest["reference_base"],
                            "expansion reference base")
    explicit = os.environ.get(AG_EXPANSION_REFERENCE_ENV)
    reference_path = overlay_lsj.resolve_reference(
        manifest, Path(explicit) if explicit else None)
    with open(AG_WIKTIONARY_BASE_PATH, encoding="utf-8") as handle:
        base = json.load(handle)
    with open(reference_path, encoding="utf-8") as handle:
        reference = json.load(handle)
    return base, reference


def load_ag_reading_layers(ag: dict) -> dict[str, str]:
    """Read the pinned layer files and classify ``ag``'s readings."""
    base, reference = read_ag_layer_records()
    return ag_reading_layers(ag, base, reference)


# The Wiktionary extracts of grc_lexical_heads.json. A head they list has an
# entry of its own that defines it, not only an inflection or alternative-form
# entry (build/build_grc_lexical_heads.py).
WIKTIONARY_HEAD_SOURCES = frozenset({"wiktionary_en_grc", "wiktionary_el_grc"})


def lexical_heads_by_kind(path: Path) -> tuple[frozenset[str], frozenset[str]]:
    """The Ancient heads of grc_lexical_heads.json: (Wiktionary, all).

    The first set holds the heads a Wiktionary extract defines in an entry of
    their own; the second every listed head, dictionaries included. A head
    list without sources (older files and test fixtures) yields two empty
    sets.
    """
    heads = json.loads(path.read_text(encoding="utf-8"))["heads"]
    if not isinstance(heads, dict):
        return frozenset(), frozenset()
    return (frozenset(head for head, sources in heads.items()
                      if WIKTIONARY_HEAD_SOURCES.intersection(sources)),
            frozenset(heads))


def displaced_head_candidates(ag: dict, wiktionary_heads, listed_heads=(),
                              base: dict = None) -> dict[str, str]:
    """Heads whose Ancient reading is another lemma: {head: lemma}.

    Every Wiktionary head qualifies; another listed head only where the
    pinned Wiktionary-only base reads it as itself.
    """
    base = base or {}
    heads = set(wiktionary_heads) | {head for head in listed_heads
                                     if base.get(head) == head}
    return {head: ag[head] for head in sorted(heads)
            if isinstance(ag.get(head), str) and ag[head] != head}


def corpus_form_evidence(pairs, forms) -> dict[str, dict[str, int]]:
    """Token counts by lemma of each listed form in counted corpus pairs."""
    evidence: dict[str, dict[str, int]] = {}
    for pair in pairs:
        form = pair["form"]
        if form in forms:
            counts = evidence.setdefault(form, {})
            counts[pair["lemma"]] = counts.get(pair["lemma"], 0) + int(pair.get("count", 1))
    return evidence


def head_alias_keys(head: str) -> set[str]:
    """The looser keys of a head: lowercase, monotonic and unaccented."""
    return {head.lower(), to_monotonic(head), to_monotonic(head).lower(),
            strip_accents(head.lower())} - {head}


def corpus_key_evidence(pairs, keys) -> dict[str, dict[str, int]]:
    """Token counts by lemma of every counted form that folds to a key.

    A form folds to its lowercase, monotonic and unaccented spellings, the
    keys a displaced head's aliases use (γυναικῶν and γυναικών both fold
    to γυναικων).
    """
    evidence: dict[str, dict[str, int]] = {}
    for pair in pairs:
        form = pair["form"]
        for key in {form, *head_alias_keys(form)} & keys:
            counts = evidence.setdefault(key, {})
            counts[pair["lemma"]] = counts.get(pair["lemma"], 0) + int(pair.get("count", 1))
    return evidence


def corpus_lemma_totals(pairs) -> dict[str, int]:
    """Token counts of each lemma over all counted corpus pairs."""
    totals: dict[str, int] = {}
    for pair in pairs:
        totals[pair["lemma"]] = totals.get(pair["lemma"], 0) + int(pair.get("count", 1))
    return totals


# A head only a dictionary lists needs at least this many corpus tokens read
# as itself: one token is a lemmatizer's choice, not evidence (ἐγώγε, τός).
LISTED_HEAD_MIN_TOKENS = 2


def restore_displaced_lexical_heads(ag: dict, *, wiktionary_heads, base: dict,
                                    evidence: dict, listed_heads=(),
                                    eligible=None, key_evidence=None,
                                    lemma_totals=None,
                                    rejected=None) -> list[dict]:
    """Give a lexical head back its own reading when another lemma took it.

    The merged Ancient table can read a word that has an entry of its own as
    a form of another word: a homograph entry on the same Wiktionary page
    (κλαίω is also a future of κλείω, ἄπειρος a Doric spelling of ἤπειρος),
    a name page's lowercase key (δῖος -> Δῖος), or a page that lists the word
    in its table (γραφή -> γραφεύς). Both readings are real, and the POS
    table keeps both; the lookup holds one, so the head takes it back where
    the evidence prefers it:

    - the counted corpus pairs (GLAUx) read the form as the head more often
      than as the lemma that displaced it (ἕως 5,399 to 1, against ἠώς), or
    - for a head Wiktionary defines, they are silent or tied and the pinned
      Wiktionary-only base reads the head as itself.

    A head only a dictionary lists (``listed_heads``: σπαθίον, κατέναντι)
    is considered where the base reads it as itself, and needs the corpus
    majority of at least ``LISTED_HEAD_MIN_TOKENS`` tokens: those lists and
    the base also hold inflected forms. Where the corpus prefers the other
    lemma (τοῦ, ἴσον, θρόνον), or neither source supports the head, the
    reading stays.

    A case twin is no other lemma: a capitalized head never takes a key
    from its own lowercase lemma (Υἱός, Πατήρ keep υἱός, πατήρ), and a
    lowercase head takes one from its capitalized twin only where the
    corpus uses the lowercase lemma at least as often overall
    (``lemma_totals``: ψυχή, τύχη, δῖος, but not φίλιππος beside Φίλιππος,
    whose new self-map would turn every Φιλίππου into the adjective).

    A restored head's unaccented, lowercase and monotonic keys follow it
    when they hold the same displaced lemma, are no listed head themselves,
    no other restored head claims them, and the corpus forms that fold to
    the key (``key_evidence``) read it as the head more often than as any
    other lemma, or the corpus has no form for the key at all: γυναικων
    stays γυνή beside γυναικών, because γυναικῶν folds to it too, and ρα
    stays ῥεῖα beside ῥᾶ, because the corpus reads ῥα as ἄρα. A
    capitalized head's lowercase keys never follow it, and a key that maps
    to itself keeps its reading.

    ``rejected(form, head)``, when given, names a reading a later step of
    the build drops (the article filter); such a head keeps the displaced
    lemma, so its keys are not left without a row (ὄ, ό keep ὅς).

    ``eligible(form)``, when given, limits the change to keys whose reading
    came from the merged file. Changes ``ag`` in place and returns one
    record per key: form, displaced lemma, restored head and the counts.
    """
    eligible = eligible or (lambda form: True)
    rejected = rejected or (lambda form, head: False)
    key_evidence = key_evidence or {}
    lemma_totals = lemma_totals or {}
    all_heads = set(wiktionary_heads) | set(listed_heads)
    restored = []
    alias_claims: dict[str, set] = {}
    for head, lemma in displaced_head_candidates(
            ag, wiktionary_heads, listed_heads, base).items():
        if not eligible(head) or rejected(head, head):
            continue
        if head.lower() == lemma.lower():
            if not lemma[:1].isupper() or (lemma_totals.get(head, 0)
                                           < lemma_totals.get(lemma, 0)):
                continue
        counts = evidence.get(head, {})
        own, other = counts.get(head, 0), counts.get(lemma, 0)
        if head in wiktionary_heads:
            supported = own > other or (own == other and base.get(head) == head)
        else:
            supported = own > other and own >= LISTED_HEAD_MIN_TOKENS
        if not supported:
            continue
        restored.append({"form": head, "lemma": lemma, "head": head,
                         "head_count": own, "lemma_count": other})
        for key in head_alias_keys(head):
            if key == lemma or ag.get(key) != lemma or key in all_heads:
                continue
            if head[:1].isupper() and not key[:1].isupper():
                continue
            if not eligible(key) or rejected(key, head):
                continue
            folded = key_evidence.get(key)
            if folded and folded.get(head, 0) <= max(
                    (count for other_lemma, count in folded.items()
                     if other_lemma != head), default=0):
                continue
            alias_claims.setdefault(key, set()).add((head, lemma))
    for record in restored:
        ag[record["form"]] = record["head"]
    for key, claims in sorted(alias_claims.items()):
        if len(claims) == 1:
            head, lemma = next(iter(claims))
            ag[key] = head
            restored.append({"form": key, "lemma": lemma, "head": head,
                             "head_count": None, "lemma_count": None})
    return sorted(restored, key=lambda record: record["form"])


def el_rule_paradigm_forms(el: dict) -> set[str]:
    """Forms whose reading a Modern Greek paradigm rule generated.

    The completion records under data/changes list every form the present
    and imperfect rules added; a form counts while the lookup still holds the
    reading the record gave it.
    """
    forms = set()
    for name in EL_RULE_PARADIGM_RECORDS:
        path = CHANGES_DIR / name
        if not path.exists():
            raise SystemExit(f"changes/{name} not found; it is needed to record "
                             "which Modern Greek forms a paradigm rule generated")
        record = json.loads(path.read_text(encoding="utf-8"))
        for row in record["lookup_additions"]:
            if el.get(row["form"]) == row["corrected"][0]:
                forms.add(row["form"])
    return forms


def write_lookup_sources(conn, sources: list[dict]) -> dict[str, int]:
    """Copy the registry into ``sources`` and return each code's id."""
    conn.execute("CREATE TABLE sources (id INTEGER PRIMARY KEY, "
                 "code TEXT NOT NULL UNIQUE, name TEXT NOT NULL, "
                 "share_alike INTEGER NOT NULL, attribution TEXT NOT NULL, "
                 "components TEXT NOT NULL)")
    ids = {}
    for index, source in enumerate(sources, start=1):
        conn.execute("INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?)", (
            index, source["code"], source["name"], int(bool(source["share_alike"])),
            source["attribution"],
            json.dumps(source["components"], ensure_ascii=False, sort_keys=True)))
        ids[source["code"]] = index
    return ids


def write_lookup_meta(conn) -> dict:
    """Stamp lookup.db with what built it and what it holds.

    ``meta(key, value)`` carries the schema version, the Dilemma version of
    the checkout that built the file, the row counts and the content hash.
    It has no clock time or commit, so equal inputs still give equal bytes.
    """
    version_path = SCRIPT_DIR / "VERSION"
    meta = {
        "schema_version": str(LOOKUP_SCHEMA_VERSION),
        "dilemma_version": (version_path.read_text(encoding="utf-8").strip()
                            if version_path.exists() else "unknown"),
        "lemmas": str(conn.execute("SELECT COUNT(*) FROM lemmas").fetchone()[0]),
        "lookup_rows": str(conn.execute("SELECT COUNT(*) FROM lookup").fetchone()[0]),
        "lookup_sha256": lookup_content_sha256(conn),
        "sources_sha256": hashlib.sha256(json.dumps(
            [list(row) for row in conn.execute(
                "SELECT code, name, share_alike, attribution, components "
                "FROM sources ORDER BY id")],
            ensure_ascii=False).encode("utf-8")).hexdigest(),
    }
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) "
                 "WITHOUT ROWID")
    conn.executemany("INSERT INTO meta VALUES (?, ?)", sorted(meta.items()))
    return meta


def build():
    t0 = time.time()

    print("Loading lookup tables...")
    ag, ag_default_source = _load_lookup("ag", AG_PATH, "AG")
    el, el_default_source = _load_lookup("mg", MG_PATH, "MG")
    completed = complete_lookup(el)
    print(f"  Reviewed Modern Greek paradigm gaps: +{len(completed)} keys")
    med, med_default_source = _load_lookup("med", MED_PATH, "Med")
    lookup_sources, source_of_input = load_lookup_sources()
    ag_source_overrides: dict[str, str] = {}
    el_source_overrides: dict[str, str] = {}
    for cell in completed:
        el_source_overrides[cell["form"]] = "el_paradigm_gaps"
    # A merged file names one source per layer where pinned records tell the
    # layers apart; every other reading keeps the file itself as its source.
    ag_base = None
    if ag_default_source == AG_PATH.name:
        ag_base, reference = read_ag_layer_records()
        ag_source_overrides.update(ag_reading_layers(ag, ag_base, reference))
        del reference
        counts = {layer: 0 for layer in (AG_WIKTIONARY_LAYER, AG_EXPANSION_LAYER)}
        for layer in ag_source_overrides.values():
            counts[layer] += 1
        print(f"  Ancient reading layers: {counts[AG_WIKTIONARY_LAYER]:,} Wiktionary, "
              f"{counts[AG_EXPANSION_LAYER]:,} dictionary expansion, "
              f"{len(ag) - len(ag_source_overrides):,} unassigned")
    if el_default_source == MG_PATH.name:
        generated = el_rule_paradigm_forms(el) - el_source_overrides.keys()
        for form in generated:
            el_source_overrides[form] = EL_RULE_PARADIGM_LAYER
        print(f"  Modern Greek rule-generated paradigm forms: {len(generated):,}")
    citation_rejections: list[dict] = []

    def _record_citation_rejection(*, table: str, source: str,
                                   form: str, lemma: str,
                                   reason: str) -> None:
        citation_rejections.append({
            "table": table,
            "source": source,
            "form": form,
            "lemma": lemma,
            "reason": reason,
        })

    # The counted GLAUx pairs: evidence for the displaced-head rule below,
    # then a gap-fill source of their own.
    glaux_pairs = None
    if GLAUX_PAIRS_PATH.exists():
        with open(GLAUX_PAIRS_PATH, encoding="utf-8") as f:
            glaux_pairs = json.load(f)

    # Merge med into el: vernacular medieval Greek is the ancestor of
    # Modern Greek, and EL Wiktionary's "Medieval Greek" category contains
    # early MG vocabulary, not Byzantine literary Greek.
    med_merged = 0
    for k, v in med.items():
        if k not in el:
            el[k] = v
            el_source_overrides[k] = med_default_source
            med_merged += 1
    print(f"  Merged {med_merged:,} med entries into el ({len(el):,} total)")

    # Add HNC Golden Corpus pairs (gold-standard MG annotations).
    # Lower priority than Wiktionary: only add where form is not already present.
    hnc_added = 0
    if HNC_PAIRS_PATH.exists():
        t_h = time.time()
        with open(HNC_PAIRS_PATH, encoding="utf-8") as f:
            hnc_pairs = json.load(f)
        for form, lemma in hnc_pairs.items():
            if form not in el:
                el[form] = lemma
                el_source_overrides[form] = HNC_PAIRS_PATH.name
                hnc_added += 1
        print(f"  HNC: +{hnc_added:,} to el "
              f"({len(hnc_pairs):,} total, {len(hnc_pairs) - hnc_added:,} already present) "
              f"({time.time()-t_h:.1f}s)")
    else:
        print(f"  HNC: no hnc_pairs.json found, skipping")

    # PROIEL is excluded entirely: CC BY-NC-SA 3.0 (NonCommercial) at both the
    # UD release and the original proiel-treebank, with no permissive version.

    # The Gorman treebanks (CC BY-SA 4.0, hand-annotated, 18 classical
    # authors) are deliberately NOT ingested: they are the project's
    # HELD-OUT GOLD corpus (eval/eval_gorman_gold.py). GLAUx/Diorisis
    # cover the same texts, so holding Gorman out costs ~135 of 9.7M
    # lookup entries and nothing measurable on any benchmark, while
    # keeping a genuinely independent 554K-token gold standard.

    # Article and pronoun forms excluded from the lookup so that
    # resolve_articles=True/False in Dilemma controls their resolution.
    # Without this, fresh Wiktionary data maps τοῦ -> ὁ etc. in the
    # lookup itself, bypassing the resolve_articles flag.
    #
    # Only AG article mappings are excluded (form -> polytonic ὁ/ἡ/τό
    # or a polytonic article form). MG article self-maps like ο -> ο
    # must stay in the lookup so MG lemmatization of function words
    # works without requiring resolve_articles=True. The AG vs MG
    # distinction is made by checking for breathing marks / polytonic
    # diacritics on the lemma.
    _EXCLUDED_ARTICLE_MAPS = {
        "ὁ", "ἡ", "τό", "τοῦ", "τῆς", "τῶν", "τόν", "τήν",
        "τά", "τοῖς", "ταῖς", "τῷ", "τῇ", "τούς", "τάς", "τοῖν", "ταῖν",
        "οἱ", "αἱ", "τώ",
        "τὸ", "τοὺς", "τὰ", "τὸν", "τὴν", "τὰς", "αἵ", "οἵ",
    }
    _ARTICLE_LEMMA = "ὁ"
    _excluded_stripped = {strip_accents(a.lower()) for a in _EXCLUDED_ARTICLE_MAPS}

    def _has_polytonic(s):
        """True if the string carries a breathing or circumflex
        (i.e. it's an AG polytonic form, not MG monotonic)."""
        nfd = unicodedata.normalize("NFD", s)
        for ch in nfd:
            cp = ord(ch)
            # Combining smooth/rough breathing, circumflex, iota subscript,
            # psili/dasia precomposed glyphs
            if cp in (0x0313, 0x0314, 0x0342, 0x0345, 0x1FBD, 0x1FBF,
                      0x1FFE, 0x1FC0, 0x1FC1):
                return True
        return False

    def _is_article_map(form, lemma):
        """True for form -> polytonic-AG-article mappings only.

        Exclude mappings like ο -> ὁ, τοῦ -> ὁ (AG article paradigm),
        but KEEP MG monotonic self-maps like ο -> ο so that MG
        lemmatization of function words works without requiring
        resolve_articles=True. The AG vs MG distinction is made by
        checking for breathing marks / polytonic diacritics on the
        lemma - MG lemmas are always monotonic.
        """
        if strip_accents(form.lower()) not in _excluded_stripped:
            return False
        if strip_accents(lemma.lower()) != strip_accents(_ARTICLE_LEMMA.lower()):
            return False
        # Lemma stripped matches ὁ (i.e. "ο"). Keep the entry if the
        # lemma is a monotonic MG form (no breathings) - that's the
        # legitimate MG self-map we want to preserve. Exclude only
        # when the lemma is polytonic AG (ὁ, ὁ̓, ὅ, etc).
        return _has_polytonic(lemma)

    _VOWEL_ONLY_ARTICLE_LETTERS = {"ο", "η", "οι", "αι"}

    def _fills_ag_gap(form, lemma):
        """True when a corpus pair may set ``ag[form]``.

        Corpus sources only fill gaps: the first one to name a form keeps
        it. The exception is a grave ὃ, ἣ, οἳ or αἳ that an earlier source
        filed under the article, which the article filter below then drops.
        The article ὁ, ἡ, οἱ, αἱ is a proclitic: it has no accent of its
        own, and before an enclitic it takes an acute (ὅ γε), never a
        grave. A grave spelling of these letters is the relative or Homer's
        demonstrative, and AGDT, read first, files the demonstrative under
        ὁ, so the article entry shut out GLAUx's relative ὅς and the forms
        had no row at all. Such an entry gives way to a later source's
        other reading. Every other article spelling (τὸ, τὴν, the dual
        τὼ) stays the article's, and the filter keeps it out as before.
        """
        if form not in ag:
            return True
        return (_is_article_map(form, ag[form])
                and not _is_article_map(form, lemma)
                and "\u0300" in unicodedata.normalize("NFD", form)
                and strip_accents(form.lower()) in _VOWEL_ONLY_ARTICLE_LETTERS)

    # A word with an entry of its own takes back the reading another lemma's
    # homograph entry or table took from it in the merged table, where GLAUx,
    # or for a Wiktionary head the pinned Wiktionary-only base, prefers it
    # (restore_displaced_lexical_heads; ἄπειρος, κλαίω, ἕως, γραφή).
    if ag_base is not None and GRC_LEXICAL_HEADS_PATH.exists():
        if glaux_pairs is None:
            print(f"  NOTE: {GLAUX_PAIRS_PATH.name} not found; displaced "
                  "heads are restored only where Wiktionary defines them and "
                  "the pinned base reads them as themselves")
        wiktionary_heads, listed_heads = lexical_heads_by_kind(GRC_LEXICAL_HEADS_PATH)
        candidates = displaced_head_candidates(ag, wiktionary_heads, listed_heads,
                                               ag_base)
        alias_keys = set().union(*map(head_alias_keys, candidates)) if candidates else set()
        restored_heads = restore_displaced_lexical_heads(
            ag, wiktionary_heads=wiktionary_heads, listed_heads=listed_heads,
            base=ag_base,
            evidence=corpus_form_evidence(glaux_pairs or (), candidates.keys()),
            key_evidence=corpus_key_evidence(glaux_pairs or (), alias_keys),
            lemma_totals=corpus_lemma_totals(glaux_pairs or ()),
            rejected=_is_article_map,
            eligible=lambda form: ag_source_overrides.get(form) in (
                None, AG_WIKTIONARY_LAYER, AG_EXPANSION_LAYER))
        for record in restored_heads:
            _record_citation_rejection(
                table="AG", source=ag_source_overrides.get(
                    record["form"], ag_default_source),
                form=record["form"], lemma=record["lemma"],
                reason=f"displaced_lexical_head_restored_to_{record['head']}")
            ag_source_overrides[record["form"]] = AG_WIKTIONARY_LAYER
        print(f"  Displaced lexical heads restored: "
              f"{sum(r['head_count'] is not None for r in restored_heads):,} "
              f"heads, {len(restored_heads):,} keys "
              f"({len(candidates):,} heads read as another lemma)")
    ag_base = None

    # Expand AG with AGDT/Perseus treebank pairs (the 33 Greek AGDT works:
    # Sophocles, Aeschylus, Homer, Hesiod, Herodotus, Thucydides, Plutarch,
    # Polybius, Athenaeus). Sourced from the AGDT original
    # (PerseusDL/treebank_data), CC BY-SA 3.0 US -- not the NC UD repackaging.
    perseus_added_ag = 0
    if PERSEUS_PAIRS_PATH.exists():
        t_pe = time.time()
        with open(PERSEUS_PAIRS_PATH, encoding="utf-8") as f:
            perseus_pairs = json.load(f)
        for p in perseus_pairs:
            form, lemma = p["form"], p["lemma"]
            if _fills_ag_gap(form, lemma):
                ag[form] = lemma
                ag_source_overrides[form] = PERSEUS_PAIRS_PATH.name
                perseus_added_ag += 1
        print(f"  Perseus (AGDT): +{perseus_added_ag:,} to AG "
              f"({len(perseus_pairs):,} total, "
              f"{len(perseus_pairs) - perseus_added_ag:,} already present) "
              f"({time.time()-t_pe:.1f}s)")
    else:
        print(f"  Perseus: no perseus_pairs.json found, skipping")

    # Load AG headwords early: used both for corpus lemma validation
    # and for protecting AG self-maps from EL overrides later.
    # AG headwords that self-map (e.g. καθάπερ -> καθάπερ) are correct
    # citation forms and should not be replaced by EL form-of redirects.
    ag_headwords = set()
    ag_headwords_exact = set()  # original forms only (for lemma validation)
    if AG_HEADWORDS_PATH.exists():
        with open(AG_HEADWORDS_PATH, encoding="utf-8") as f:
            ag_headwords_exact = set(json.load(f))
        ag_headwords = set(ag_headwords_exact)
        ag_headwords |= {h.lower() for h in ag_headwords}
        ag_headwords |= {strip_accents(h.lower()) for h in ag_headwords}
        print(f"  AG headwords: {len(ag_headwords):,} (for self-map protection)")

    if DGE_HEADWORDS_PATH.exists():
        with open(DGE_HEADWORDS_PATH, encoding="utf-8") as f:
            dge_raw = set(json.load(f))
        dge_new = dge_raw - ag_headwords_exact
        ag_headwords_exact |= dge_raw
        ag_headwords |= dge_raw
        ag_headwords |= {h.lower() for h in dge_raw}
        ag_headwords |= {strip_accents(h.lower()) for h in dge_raw}
        print(f"  DGE headwords: {len(dge_new):,} new (for spell-check coverage)")

    if LGPN_NAMES_PATH.exists():
        with open(LGPN_NAMES_PATH, encoding="utf-8") as f:
            lgpn_raw = set(json.load(f))
        lgpn_new = lgpn_raw - ag_headwords_exact
        ag_headwords_exact |= lgpn_raw
        ag_headwords |= lgpn_raw
        ag_headwords |= {h.lower() for h in lgpn_raw}
        ag_headwords |= {strip_accents(h.lower()) for h in lgpn_raw}
        print(f"  LGPN names: {len(lgpn_new):,} new (proper noun coverage)")

    if PD_HEADWORDS_PATH.exists():
        with open(PD_HEADWORDS_PATH, encoding="utf-8") as f:
            pd_raw = set(json.load(f))
        pd_new = pd_raw - ag_headwords_exact
        ag_headwords_exact |= pd_raw
        ag_headwords |= pd_raw
        ag_headwords |= {h.lower() for h in pd_raw}
        ag_headwords |= {strip_accents(h.lower()) for h in pd_raw}
        print(f"  PD headwords (L&S, Pape, Bailly, etc.): {len(pd_new):,} new")

    if VLG_HEADWORDS_PATH.exists():
        with open(VLG_HEADWORDS_PATH, encoding="utf-8") as f:
            vlg_raw = set(json.load(f))
        vlg_new = vlg_raw - ag_headwords_exact
        ag_headwords_exact |= vlg_raw
        ag_headwords |= vlg_raw
        ag_headwords |= {h.lower() for h in vlg_raw}
        ag_headwords |= {strip_accents(h.lower()) for h in vlg_raw}
        # Add self-map entries to AG lookup so these headwords are
        # recognized by Dilemma's spell-checker (form -> lemma = itself).
        vlg_lookup_added = 0
        # Set iteration changes with PYTHONHASHSEED. Keep every headword
        # insertion ordered so unchanged inputs produce identical databases.
        for h in sorted(vlg_raw):
            if h not in ag:
                ag[h] = h
                ag_source_overrides[h] = VLG_HEADWORDS_PATH.name
                vlg_lookup_added += 1
        print(f"  VLG headwords: {len(vlg_new):,} new, "
              f"+{vlg_lookup_added:,} self-maps to AG lookup")

    # Words in Progress (Aristarchus, supplementary lexicon of new/rare AG words,
    # directed by Montanari & Perrone): curated headwords + morphology. Use the
    # single-token clean headwords as AG self-maps, same as the VLG block.
    if WIP_HEADWORDS_PATH.exists():
        with open(WIP_HEADWORDS_PATH, encoding="utf-8") as f:
            wip_raw = {h for h in json.load(f) if " " not in h}
        wip_new = wip_raw - ag_headwords_exact
        ag_headwords_exact |= wip_raw
        ag_headwords |= wip_raw
        ag_headwords |= {h.lower() for h in wip_raw}
        ag_headwords |= {strip_accents(h.lower()) for h in wip_raw}
        wip_lookup_added = 0
        for h in sorted(wip_raw):
            if h not in ag:
                ag[h] = h
                ag_source_overrides[h] = WIP_HEADWORDS_PATH.name
                wip_lookup_added += 1
        print(f"  WiP headwords: {len(wip_new):,} new, "
              f"+{wip_lookup_added:,} self-maps to AG lookup")

    # LSJ 10th ed. headwords (Liddell-Scott-Jones, from the LSJ10 app database
    # via build/build_lsj10_headwords.py). Clean single-token NFC polytonic
    # headwords used as AG self-maps, same as the VLG/WiP blocks.
    if LSJ10_HEADWORDS_PATH.exists():
        with open(LSJ10_HEADWORDS_PATH, encoding="utf-8") as f:
            lsj10_raw = {h for h in json.load(f) if " " not in h}
        lsj10_new = lsj10_raw - ag_headwords_exact
        ag_headwords_exact |= lsj10_raw
        ag_headwords |= lsj10_raw
        ag_headwords |= {h.lower() for h in lsj10_raw}
        ag_headwords |= {strip_accents(h.lower()) for h in lsj10_raw}
        lsj10_lookup_added = 0
        for h in sorted(lsj10_raw):
            if h not in ag:
                ag[h] = h
                ag_source_overrides[h] = LSJ10_HEADWORDS_PATH.name
                lsj10_lookup_added += 1
        print(f"  LSJ10 headwords: {len(lsj10_new):,} new, "
              f"+{lsj10_lookup_added:,} self-maps to AG lookup")


    # Expand AG and Med with GLAUx corpus pairs (644K forms from
    # 8th c. BC - 4th c. AD Greek texts). These are corpus-derived
    # so lower confidence than Wiktionary, but fill coverage gaps.
    def _normalize_corpus_lemma(lemma, headwords_exact):
        """Normalize corpus lemmas with spurious capitalization.

        Corpora like GLAUx sometimes capitalize sentence-initial lemmas
        (e.g. Εἰμί instead of εἰμί). If the lowercase version is a known
        headword, prefer it. A reviewed non-lemma (REJECTED_LEMMAS) stays
        as it is, so the rejection still catches it: GLAUx's Ταύτης for
        Ταύτῃ would otherwise pass as the lowercase ταύτης, which some head
        list carries.
        """
        if lemma in REJECTED_LEMMAS:
            return lemma
        if lemma and lemma[0].isupper():
            lower = lemma[0].lower() + lemma[1:]
            if lower in headwords_exact:
                return lower
        return lemma

    _CORPUS_PAIR_SOURCES = {GLAUX_PAIRS_PATH.name, DIORISIS_PAIRS_PATH.name}

    def _pronoun_twin_tables(ag_snapshot):
        """Where a capitalised corpus form's lowercase twin is looked up:
        the AG table as it stood before the corpus block, then the el table,
        each with a test for readings an earlier corpus block supplied."""
        return (
            (ag_snapshot, lambda f: ag_source_overrides.get(f)
             in _CORPUS_PAIR_SOURCES),
            (el, lambda f: el_source_overrides.get(f) in _CORPUS_PAIR_SOURCES),
        )

    # Unaccented dictionary-expansion lemmas are checked against the lexical
    # heads (#103). Without them the check cannot run, and the build would
    # ship the transliterated-Latin and OCR lemmas it exists to remove.
    # Every expansion lemma is judged once, here, before any corpus pair is
    # read: the verdict drops or repairs the expansion reading in
    # _sanitize_table, and it applies to a corpus pair naming the same lemma
    # too, since GLAUx and Diorisis carry some of these artefacts (λαβῃ ->
    # λαβλυμι) and ag_headwords.json lists them, so a corpus pair would
    # otherwise put the dropped lemma back. A repair goes only to a head the
    # Ancient or Modern table maps to itself.
    expansion_heads = expansion_heads_by_letters = None
    if GRC_LEXICAL_HEADS_PATH.exists():
        expansion_heads, expansion_heads_by_letters = (
            load_expansion_lemma_heads(GRC_LEXICAL_HEADS_PATH))
    elif AG_EXPANSION_LAYER in ag_source_overrides.values():
        sys.exit(f"{GRC_LEXICAL_HEADS_PATH.name} is missing; it is needed to "
                 "check the dictionary-expansion lemmas. Fetch it with "
                 "python -m dilemma download.")
    expansion_verdicts: dict[str, tuple[str, str | None]] = {}
    if expansion_heads is not None:
        def _is_citation_head(head):
            return ag.get(head) == head or el.get(head) == head
        for form, layer in ag_source_overrides.items():
            if layer != AG_EXPANSION_LAYER or not isinstance(ag.get(form), str):
                continue
            lemma = sanitize_form(ag[form])
            if lemma in expansion_verdicts:
                continue
            verdict = unaccented_expansion_lemma(
                lemma, expansion_heads, expansion_heads_by_letters,
                _is_citation_head)
            if verdict is not None:
                expansion_verdicts[lemma] = verdict
    corpus_expansion_dropped: dict[str, int] = {}
    corpus_expansion_repaired: dict[str, int] = {}

    def _corpus_expansion_lemma(form, lemma, source):
        """A corpus pair's lemma after the #103 verdict on the same expansion
        lemma: the repaired head, the lemma unchanged, or None (dropped)."""
        verdict = expansion_verdicts.get(sanitize_form(lemma))
        if verdict is None:
            return lemma
        reason, head = verdict
        counts = corpus_expansion_repaired if head else corpus_expansion_dropped
        counts[source] = counts.get(source, 0) + 1
        _record_citation_rejection(
            table="AG", source=source, form=form, lemma=lemma,
            reason=f"{reason}_to_{head}" if head else reason)
        return head

    # Reject pairs whose lemma is not a known AG headword to filter
    # out annotation errors (e.g. Ἔστι -> Ἔσθι, corrupt -δήποτε lemmas).
    # Reviewed non-lemmas (dilemma/grc_rejected_lemmas.py) are rejected too.
    glaux_added_ag = 0
    glaux_added_med = 0
    glaux_skipped_med = 0
    glaux_bad_lemma = 0
    if glaux_pairs is not None:
        t_g = time.time()
        # Snapshot AG keys before GLAUx expansion, so we can check
        # whether a form had an original (Wiktionary) AG entry.
        ag_original = dict(ag)
        glaux_twin_skipped = 0
        twin_tables = _pronoun_twin_tables(ag_original)
        for p in glaux_pairs:
            form, lemma = p["form"], p["lemma"]
            # Normalize capitalized lemmas (GLAUx sentence-initial convention)
            lemma = _normalize_corpus_lemma(lemma, ag_headwords_exact)
            lemma = _corpus_expansion_lemma(form, lemma, GLAUX_PAIRS_PATH.name)
            if lemma is None:
                glaux_bad_lemma += 1
                continue
            if capitalized_pronoun_twin(form, lemma, twin_tables):
                glaux_twin_skipped += 1
                continue
            # Validate lemma against known AG headwords
            if (lemma in REJECTED_LEMMAS
                    or (ag_headwords_exact and lemma not in ag_headwords_exact)):
                glaux_bad_lemma += 1
                continue
            # Add to AG if not already present
            if _fills_ag_gap(form, lemma):
                ag[form] = lemma
                ag_source_overrides[form] = GLAUX_PAIRS_PATH.name
                glaux_added_ag += 1
            # Selectively add to el: only when the pair won't cause
            # a priority override conflict in the combined merge.
            if form not in el:
                if form not in ag_original:
                    el[form] = lemma
                    el_source_overrides[form] = GLAUX_PAIRS_PATH.name
                    glaux_added_med += 1
                elif ag_original[form] == lemma:
                    el[form] = lemma
                    el_source_overrides[form] = GLAUX_PAIRS_PATH.name
                    glaux_added_med += 1
                else:
                    glaux_skipped_med += 1
        print(f"  GLAUx: +{glaux_added_ag:,} to AG, "
              f"+{glaux_added_med:,} to el, "
              f"{glaux_skipped_med:,} el conflicts skipped, "
              f"{glaux_bad_lemma:,} bad lemmas rejected, "
              f"{glaux_twin_skipped:,} capitalised pronoun twins skipped "
              f"({time.time()-t_g:.1f}s)")
        glaux_pairs = None

    # Expand AG and el with Diorisis corpus pairs (456K forms from
    # 10.2M tokens of ancient Greek texts). Lower confidence than GLAUx
    # (91.4% vs 98.8% lemma accuracy), so lowest priority: only added
    # when not already present from Wiktionary, LSJ, or GLAUx.
    # Same lemma validation as GLAUx.
    dior_added_ag = 0
    dior_added_el = 0
    dior_skipped_el = 0
    dior_skipped_ag = 0
    dior_bad_lemma = 0
    if DIORISIS_PAIRS_PATH.exists():
        t_d = time.time()
        with open(DIORISIS_PAIRS_PATH, encoding="utf-8") as f:
            diorisis_pairs = json.load(f)
        # Snapshot AG keys before Diorisis expansion (includes Wiktionary + GLAUx)
        ag_before_dior = dict(ag)
        dior_twin_skipped = 0
        twin_tables = _pronoun_twin_tables(ag_before_dior)
        for p in diorisis_pairs:
            form, lemma = p["form"], p["lemma"]
            # Normalize capitalized lemmas
            lemma = _normalize_corpus_lemma(lemma, ag_headwords_exact)
            lemma = _corpus_expansion_lemma(form, lemma, DIORISIS_PAIRS_PATH.name)
            if lemma is None:
                dior_bad_lemma += 1
                continue
            if capitalized_pronoun_twin(form, lemma, twin_tables):
                dior_twin_skipped += 1
                continue
            # Validate lemma against known AG headwords
            if (lemma in REJECTED_LEMMAS
                    or (ag_headwords_exact and lemma not in ag_headwords_exact)):
                dior_bad_lemma += 1
                continue
            # Add to AG if not already present from any source
            if _fills_ag_gap(form, lemma):
                ag[form] = lemma
                ag_source_overrides[form] = DIORISIS_PAIRS_PATH.name
                dior_added_ag += 1
            else:
                dior_skipped_ag += 1
            # Selectively add to el: only when the pair won't cause
            # a priority override conflict in the combined merge.
            if form not in el:
                if form not in ag_before_dior:
                    el[form] = lemma
                    el_source_overrides[form] = DIORISIS_PAIRS_PATH.name
                    dior_added_el += 1
                elif ag_before_dior[form] == lemma:
                    el[form] = lemma
                    el_source_overrides[form] = DIORISIS_PAIRS_PATH.name
                    dior_added_el += 1
                else:
                    dior_skipped_el += 1
        print(f"  Diorisis: +{dior_added_ag:,} to AG ({dior_skipped_ag:,} skipped), "
              f"+{dior_added_el:,} to el, "
              f"{dior_skipped_el:,} el conflicts skipped, "
              f"{dior_bad_lemma:,} bad lemmas rejected, "
              f"{dior_twin_skipped:,} capitalised pronoun twins skipped "
              f"({time.time()-t_d:.1f}s)")

    # Koine NT (Nestle 1904 lowfat, CC BY 4.0): lowest-priority gap-fill of
    # Koine form->lemma coverage, the open replacement for the dropped PROIEL
    # NT. Same headword validation; only fills forms no earlier source has.
    nt_added_ag = nt_bad_lemma = 0
    if NT_PAIRS_PATH.exists():
        t_n = time.time()
        with open(NT_PAIRS_PATH, encoding="utf-8") as f:
            nt_pairs = json.load(f)
        ag_before_nt = dict(ag)
        for p in nt_pairs:
            form, lemma = p["form"], p["lemma"]
            lemma = _normalize_corpus_lemma(lemma, ag_headwords_exact)
            lemma = _corpus_expansion_lemma(form, lemma, NT_PAIRS_PATH.name)
            if lemma is None:
                nt_bad_lemma += 1
                continue
            if (lemma in REJECTED_LEMMAS
                    or (ag_headwords_exact and lemma not in ag_headwords_exact)):
                nt_bad_lemma += 1
                continue
            if _fills_ag_gap(form, lemma):
                ag[form] = lemma
                ag_source_overrides[form] = NT_PAIRS_PATH.name
                nt_added_ag += 1
            if form not in el and form not in ag_before_nt:
                el[form] = lemma
                el_source_overrides[form] = NT_PAIRS_PATH.name
        print(f"  Koine NT: +{nt_added_ag:,} to AG, "
              f"{nt_bad_lemma:,} bad lemmas rejected ({time.time()-t_n:.1f}s)")

    # Manual-gold gap-fill from cog's standardized exports (PTNK / TAGNT /
    # Pedalion). Same headword validation and gap-fill-only guard as NT; placed
    # last so they are strictly lowest priority (never override earlier sources).
    for label, path in COG_GAPFILL_PATHS:
        if not path.exists():
            continue
        t_g = time.time()
        with open(path, encoding="utf-8") as f:
            gap_pairs = json.load(f)
        ag_before = dict(ag)
        added_ag = bad_lemma = 0
        for p in gap_pairs:
            form, lemma = p["form"], p["lemma"]
            # "_" is the CoNLL-U marker for a field the annotator left empty,
            # not a lemma. The extractors reject it now, but a pair file
            # written before that fix is still on disk and would put it back,
            # so the guard lives here as well as at extraction.
            if lemma in PLACEHOLDER_LEMMAS:
                bad_lemma += 1
                continue
            lemma = _normalize_corpus_lemma(lemma, ag_headwords_exact)
            lemma = _corpus_expansion_lemma(form, lemma, path.name)
            if lemma is None:
                bad_lemma += 1
                continue
            if (lemma in REJECTED_LEMMAS
                    or (ag_headwords_exact and lemma not in ag_headwords_exact)):
                bad_lemma += 1
                continue
            if _fills_ag_gap(form, lemma):
                ag[form] = lemma
                ag_source_overrides[form] = path.name
                added_ag += 1
            if form not in el and form not in ag_before:
                el[form] = lemma
                el_source_overrides[form] = path.name
        print(f"  {label}: +{added_ag:,} to AG, "
              f"{bad_lemma:,} bad lemmas rejected ({time.time()-t_g:.1f}s)")


    for label, counts in (("dropped", corpus_expansion_dropped),
                          ("repaired", corpus_expansion_repaired)):
        if counts:
            print(f"  Corpus pairs naming a judged expansion lemma (#103), {label}: "
                  + ", ".join(f"{source} {count:,}"
                              for source, count in sorted(counts.items())))

    trusted_grave_targets = trusted_ag_citation_headwords(
        _TRUSTED_BUILD_HEADWORD_PATHS)
    if trusted_grave_targets:
        print(f"  Trusted citation headwords for grave repair: "
              f"{len(trusted_grave_targets):,}")
    else:
        print("  Trusted citation headwords unavailable; "
              "leaving grave lemma validation to runtime")

    # Sanitise every form and lemma so a stray combining breathing mark
    # (leading U+0313/U+0314/U+1FBF/U+1FFE, or trailing U+0313/U+0314 used
    # as an apostrophe) cannot leak into lookup.db from any upstream source.
    # LSJ paradigm expansion produces leading combining-psili forms like
    # `̓Αβαρικός`, and treebank exports encode elision as trailing psili
    # (`μετ̓`). See form_sanitize.sanitize_form for the rules.

    def _sanitize_table(name: str, table: dict, *, default_source: str,
                        source_overrides: dict[str, str],
                        sources_out: dict[str, str]) -> dict:
        out: dict = {}
        changed = 0
        dropped_elided = 0
        grave_normalized = 0
        grave_dropped = 0
        artifact_dropped = {}
        expansion_repaired = expansion_dropped = 0
        for k, v in table.items():
            source = source_overrides.get(k, default_source)
            sk = sanitize_form(k)
            sv = sanitize_form(v) if isinstance(v, str) else v
            if not sk:
                continue
            if (has_editorial_sigla(sk)
                    or (isinstance(sv, str) and has_editorial_sigla(sv))):
                artifact_dropped["editorial_siglum"] = (
                    artifact_dropped.get("editorial_siglum", 0) + 1)
                _record_citation_rejection(
                    table=name, source=source, form=k, lemma=sv,
                    reason="editorial_siglum")
                continue
            artifact_reason = (_citation_artifact_reason(sv)
                               if isinstance(sv, str) else None)
            if artifact_reason:
                artifact_dropped[artifact_reason] = (
                    artifact_dropped.get(artifact_reason, 0) + 1)
                _record_citation_rejection(
                    table=name, source=source, form=k, lemma=sv,
                    reason=artifact_reason)
                continue
            if isinstance(sv, str) and sv in REJECTED_LEMMAS:
                artifact_dropped["reviewed_rejected_target"] = (
                    artifact_dropped.get("reviewed_rejected_target", 0) + 1)
                _record_citation_rejection(
                    table=name, source=source, form=k, lemma=sv,
                    reason="reviewed_rejected_target")
                continue
            if source == AG_EXPANSION_LAYER:
                verdict = expansion_verdicts.get(sv)
                if verdict is not None:
                    reason, head = verdict
                    if head is None:
                        expansion_dropped += 1
                        _record_citation_rejection(
                            table=name, source=source, form=k, lemma=sv,
                            reason=reason)
                        continue
                    expansion_repaired += 1
                    _record_citation_rejection(
                        table=name, source=source, form=k, lemma=sv,
                        reason=f"{reason}_to_{head}")
                    sv = head
            if trusted_grave_targets and isinstance(sv, str) and _has_grave(sv):
                safe = _normalize_grave_citation_lemma(
                    sv, trusted_grave_targets)
                if safe is None:
                    grave_dropped += 1
                    _record_citation_rejection(
                        table=name, source=source, form=k, lemma=sv,
                        reason="untrusted_grave")
                    continue
                if safe != sv:
                    sv = safe
                    grave_normalized += 1
            # Elided forms may be encoded with any of several apostrophe
            # codepoints (U+2019 right single quote, U+02BC modifier letter,
            # U+0027 ascii, U+0060 grave, U+02B9 modifier prime). Canonicalize
            # the trailing mark onto the single U+1FBD GREEK KORONIS key so the
            # runtime resolves an elided form the same way regardless of which
            # codepoint the text used. This recognized set matches the sibling
            # Open Greek prosodia engine's, so the two tools agree on what an
            # elision mark is.
            #
            # Then drop ONLY the entries that carry no lemma evidence: a
            # self-map (ἀλλ᾽ -> ἀλλ᾽, created when lemma validation rejected
            # the source lemma) or a value that is itself an elided form.
            # Those must fall through to the runtime elision expander (which
            # settles them via its function-word allow-list, ἀλλ᾽ -> ἀλλά).
            # Genuine mappings (ὅτ᾽ -> ὅτε, κ᾽ -> ἄν, μ᾽ -> ἐγώ) are KEPT.
            _APOS = ("’", "'", "ʼ", "`", "ʹ")
            if sk[-1] in _APOS:
                sk = sk[:-1] + "᾽"          # canonicalize onto one key
            if sk[-1] == "᾽" and (not isinstance(sv, str) or sv == sk
                                  or sv.endswith(("᾽",) + _APOS)):
                dropped_elided += 1
                continue
            if sk != k:
                changed += 1
            # If sanitisation collides two keys, keep the first-seen value
            # (the tables are already merged by priority upstream).
            if sk not in out:
                out[sk] = sv
                sources_out[sk] = source
        if changed:
            print(f"  Sanitised {changed:,} {name} forms "
                  f"(misplaced breathing marks)")
        if dropped_elided:
            print(f"  Dropped {dropped_elided:,} {name} trailing-apostrophe "
                  f"elided forms (resolved via elision layer)")
        if grave_normalized:
            print(f"  Normalized {grave_normalized:,} {name} grave citation "
                  f"lemmas to trusted acute headwords")
        if grave_dropped:
            print(f"  Dropped {grave_dropped:,} {name} untrusted grave "
                  f"citation lemmas")
        for reason, count in sorted(artifact_dropped.items()):
            print(f"  Dropped {count:,} {name} {reason} citation artifacts")
        if expansion_dropped or expansion_repaired:
            print(f"  Unaccented dictionary-expansion lemmas: dropped "
                  f"{expansion_dropped:,} {name} readings, repaired "
                  f"{expansion_repaired:,} to their accented head")
        return out

    # Feminine and neuter-plural cells of -ικός adjectives that the pinned
    # expansion declined as masculine nouns (dilemma/grc_adjective_gaps.py).
    # They are generated by rule, so they come after every other Ancient
    # source and fill only the keys none of them supplied.
    if GRC_IKOS_ADJECTIVES_PATH.exists():
        listed = json.loads(GRC_IKOS_ADJECTIVES_PATH.read_text(encoding="utf-8"))
        adjective_cells = complete_ikos_adjectives(
            ag, [row["lemma"] for row in listed["lemmas"]])
        for cell in adjective_cells:
            ag_source_overrides[cell["form"]] = "grc_adjective_gaps"
        print(f"  Ancient -ικός adjective cells: +{len(adjective_cells):,} keys")
    else:
        print(f"  NOTE: {GRC_IKOS_ADJECTIVES_PATH.name} not found; -ικός "
              "adjectives keep only their masculine and neuter cells")

    print("\nSanitising form-and-lemma tables...")
    # The input that supplied each key's reading, through every later step.
    ag_sources: dict[str, str] = {}
    el_sources: dict[str, str] = {}
    combined_sources: dict[str, str] = {}
    ag = _sanitize_table(
        "AG", ag, default_source=ag_default_source,
        source_overrides=ag_source_overrides, sources_out=ag_sources)
    el = _sanitize_table(
        "EL", el, default_source=el_default_source,
        source_overrides=el_source_overrides, sources_out=el_sources)

    # Build combined lookup (AG-first priority)
    print("\nBuilding combined lookup (AG-first)...")
    combined = {}
    ag_protected = 0
    article_excluded = 0
    for data, data_sources in [(ag, ag_sources), (el, el_sources)]:
        for k, v in data.items():
            if _is_article_map(k, v):
                article_excluded += 1
                continue
            if k not in combined:
                combined[k] = v
                combined_sources[k] = data_sources[k]
            elif _is_self_map(k, combined[k]) and not _is_self_map(k, v):
                # EL non-self-map overrides AG self-map, UNLESS the AG
                # self-map is a known AG headword (correct citation form).
                if k in ag_headwords or combined[k] in ag_headwords:
                    ag_protected += 1
                else:
                    combined[k] = v
                    combined_sources[k] = data_sources[k]
            elif (_is_self_map(k, combined[k])
                  and _is_self_map(k, v) and v == k
                  and combined[k] != k):
                combined[k] = v
                combined_sources[k] = data_sources[k]
    if article_excluded:
        print(f"  Article forms excluded (controlled by resolve_articles): {article_excluded:,}")
    if ag_protected:
        print(f"  AG headword self-maps protected: {ag_protected:,}")

    # Byzantine Greek headwords (classicizing literary vocabulary of the
    # 9th-12th c.): single-token NFC polytonic lemmas, JSON list of
    # {lemma, gender}. Added here, AFTER the AG/EL merge, as lowest-priority
    # gap-fill self-maps: only forms not already resolved are added, so these
    # can never change an existing AG/EL resolution. (Adding them to `ag`
    # before the merge perturbed the AG/MG self-map arbitration and leaked
    # function words like ή -> ὅ.) Gender rides along in the source for future
    # POS use and is not consumed here.
    lbg_added = 0
    lbg_grave_normalized = 0
    lbg_grave_dropped = 0
    lbg_artifact_dropped = {}
    if LBG_HEADWORDS_PATH.exists():
        with open(LBG_HEADWORDS_PATH, encoding="utf-8") as f:
            lbg_raw = {sanitize_form(e["lemma"]) for e in json.load(f)
                       if e.get("lemma") and " " not in e["lemma"]}
        for h in sorted(lbg_raw):
            if has_editorial_sigla(h):
                lbg_artifact_dropped["editorial_siglum"] = (
                    lbg_artifact_dropped.get("editorial_siglum", 0) + 1)
                _record_citation_rejection(
                    table="combined", source=LBG_HEADWORDS_PATH.name,
                    form=h, lemma=h, reason="editorial_siglum")
                continue
            artifact_reason = _citation_artifact_reason(h)
            if artifact_reason:
                lbg_artifact_dropped[artifact_reason] = (
                    lbg_artifact_dropped.get(artifact_reason, 0) + 1)
                _record_citation_rejection(
                    table="combined", source=LBG_HEADWORDS_PATH.name,
                    form=h, lemma=h, reason=artifact_reason)
                continue
            if trusted_grave_targets and _has_grave(h):
                safe = _normalize_grave_citation_lemma(
                    h, trusted_grave_targets)
                if safe is None:
                    lbg_grave_dropped += 1
                    _record_citation_rejection(
                        table="combined", source=LBG_HEADWORDS_PATH.name,
                        form=h, lemma=h, reason="untrusted_grave")
                    continue
                if safe != h:
                    h = safe
                    lbg_grave_normalized += 1
            if h not in combined:
                combined[h] = h
                combined_sources[h] = LBG_HEADWORDS_PATH.name
                lbg_added += 1
        print(f"  Byzantine headwords: +{lbg_added:,} gap-fill self-maps (lang=all)")
        if lbg_grave_normalized:
            print(f"  Byzantine headwords: normalized "
                  f"{lbg_grave_normalized:,} grave citation headwords")
        if lbg_grave_dropped:
            print(f"  Byzantine headwords: dropped {lbg_grave_dropped:,} "
                  f"untrusted grave citation headwords")
        for reason, count in sorted(lbg_artifact_dropped.items()):
            print(f"  Byzantine headwords: dropped {count:,} {reason} "
                  f"citation artifacts")

    # Byzantine INFLECTED forms (build/expand_lbg.py -> data/lbg_pairs.json):
    # paradigm cells declined/conjugated from the headword + gender. Added as
    # lowest-priority gap-fill, but a raw exact-key gap-fill silently shadows
    # real tokens, because the runtime cascade also resolves the grave/lower/
    # monotonic/accent-stripped variants of a form. Two gates keep only the safe
    # long tail and drop the paradigm-guess collisions:
    #   (a) variant-collision: drop a generated form if ANY normalized variant
    #       already resolves to a *different* lemma (the existing lexicon wins).
    #   (b) frequency ceiling: drop ultra-common surface forms (function-word
    #       collisions like ποτε), via corpus_freq.
    lbg_forms_added = 0
    lbg_bare_added = 0
    lbg_pair_grave_normalized = 0
    lbg_pair_grave_dropped = 0
    lbg_pair_artifact_dropped = {}
    if LBG_PAIRS_PATH.exists():
        with open(LBG_PAIRS_PATH, encoding="utf-8") as f:
            lbg_pairs = json.load(f)
        lbg_share = polytonic_share(lbg_pairs)
        if lbg_share < MIN_POLYTONIC_SHARE:
            print(f"  WARNING: only {lbg_share:.1%} of {LBG_PAIRS_PATH.name}'s "
                  f"{len(lbg_pairs):,} forms carry a diacritic; its lemmas will "
                  "have no accented inflections. Rerun build/expand_lbg.py.")
        cf = {}
        if CORPUS_FREQ_PATH.exists():
            cf = json.load(open(CORPUS_FREQ_PATH, encoding="utf-8")).get(
                "forms", {})
        LBG_FREQ_CEILING = 20_000
        for form, lemma in lbg_pairs.items():
            form = sanitize_form(form)
            lemma = sanitize_form(lemma)
            if not form or form in combined:
                continue
            if has_editorial_sigla(form) or has_editorial_sigla(lemma):
                lbg_pair_artifact_dropped["editorial_siglum"] = (
                    lbg_pair_artifact_dropped.get(
                        "editorial_siglum", 0) + 1)
                _record_citation_rejection(
                    table="combined", source=LBG_PAIRS_PATH.name,
                    form=form, lemma=lemma, reason="editorial_siglum")
                continue
            artifact_reason = _citation_artifact_reason(lemma)
            if artifact_reason:
                lbg_pair_artifact_dropped[artifact_reason] = (
                    lbg_pair_artifact_dropped.get(artifact_reason, 0) + 1)
                _record_citation_rejection(
                    table="combined", source=LBG_PAIRS_PATH.name,
                    form=form, lemma=lemma, reason=artifact_reason)
                continue
            if trusted_grave_targets and _has_grave(lemma):
                safe = _normalize_grave_citation_lemma(
                    lemma, trusted_grave_targets)
                if safe is None:
                    lbg_pair_grave_dropped += 1
                    _record_citation_rejection(
                        table="combined", source=LBG_PAIRS_PATH.name,
                        form=form, lemma=lemma, reason="untrusted_grave")
                    continue
                if safe != lemma:
                    lemma = safe
                    lbg_pair_grave_normalized += 1
            variants = {form, form.lower(), grave_to_acute(form),
                        to_monotonic(form.lower()), strip_accents(form.lower())}
            if any(v in combined and combined[v] != lemma for v in variants):
                continue  # (a) an existing form->lemma mapping wins
            fv = cf.get(strip_accents(form.lower()))
            if fv and fv[0] >= LBG_FREQ_CEILING:
                continue  # (b) ultra-common surface -> function-word collision
            combined[form] = lemma
            combined_sources[form] = LBG_PAIRS_PATH.name
            lbg_forms_added += 1
            # The pairs are polytonic. Unaccented input reaches a form
            # through its accentless key, so keep that too where it is free.
            # A capitalized form also answers lowercase input, and the
            # headword gets its own accentless key the same way.
            bare = strip_accents(form)
            for key in (bare, bare.lower(), strip_accents(lemma),
                        strip_accents(lemma).lower()):
                if key not in combined:
                    combined[key] = lemma
                    combined_sources[key] = LBG_PAIRS_PATH.name
                    lbg_bare_added += 1
        print(f"  Byzantine inflected forms: +{lbg_forms_added:,} gated "
              f"gap-fill pairs (lang=all), +{lbg_bare_added:,} accentless keys")
        if lbg_pair_grave_normalized:
            print(f"  Byzantine inflected forms: normalized "
                  f"{lbg_pair_grave_normalized:,} grave citation lemmas")
        if lbg_pair_grave_dropped:
            print(f"  Byzantine inflected forms: dropped "
                  f"{lbg_pair_grave_dropped:,} untrusted grave citation lemmas")
        for reason, count in sorted(lbg_pair_artifact_dropped.items()):
            print(f"  Byzantine inflected forms: dropped {count:,} {reason} "
                  f"citation artifacts")

    # Corpus word-tails: forms that are the END of another form of the same
    # lemma, which a corpus tokenizer split off at an elision mark and then
    # annotated with the whole word's lemma (νοντ᾽ under βαρύνω, the end of
    # βαρύνοντ᾽). Deciding this needs the lemma's other forms, so it cannot run
    # at each pair's point of entry the way the citation-hygiene checks above
    # do. It runs here instead, once per assembled table, after every ingestion
    # path has contributed and before anything is written.
    def _merge_source(form: str) -> str:
        """Best-known provenance label for a form in the merged table."""
        if form in ag_source_overrides:
            return ag_source_overrides[form]
        if form in el_source_overrides:
            return el_source_overrides[form]
        if form in ag:
            return ag_default_source
        if form in el:
            return el_default_source
        # Only the two Byzantine (LBG) blocks write straight into `combined`,
        # and only the headword one writes self-maps.
        return (LBG_HEADWORDS_PATH.name if combined.get(form) == form
                else LBG_PAIRS_PATH.name)

    if FORM_PROFILE_PATH.exists():
        drop_corpus_word_tails(
            [
                ("combined", combined, _merge_source),
                ("AG", ag, lambda f: ag_source_overrides.get(
                    f, ag_default_source)),
                ("EL", el, lambda f: el_source_overrides.get(
                    f, el_default_source)),
            ],
            is_attested=open_profile_attestation(FORM_PROFILE_PATH),
            record=_record_citation_rejection,
        )
    else:
        # Loud on purpose. form_profile.db is a 361 MB OPT-IN download: it is
        # not fetched by `python -m dilemma download`, it is not in
        # scripts/hf_data.py TRACKED, and it is not a step of the documented
        # rebuild order. Without it this filter is a no-op and the build
        # quietly ships the word-tails, so the same code and the same corpora
        # produce two different artifacts depending on what is on disk. A
        # single line in a build log that runs for hours is not enough notice.
        banner = "!" * 72
        print(f"\n{banner}")
        print("  CORPUS WORD-TAIL FILTER INACTIVE")
        print(f"  {FORM_PROFILE_PATH.name} is missing, so corpus tokenizer "
              f"word-tails will be")
        print("  written to lookup.db as if they were words.")
        print("  Fix: python -m dilemma download --with-attestation")
        print(f"{banner}\n")

    # Regular cells of Ancient -σις nouns and s-stem neuters that no source
    # declined (dilemma/grc_noun_gaps.py): headword-only -σις nouns (σύναρσις)
    # and the Attic contracted cells of neuters whose tables are Ionic only
    # (στέφους beside στέφεος). Generated by rule, so they run on the final
    # combined table, after every source (the Byzantine headwords reach only
    # `combined`) and after the word-tail filter, and fill absent keys only;
    # the reviewed overrides below still win. Keys the merge left out on
    # purpose (article forms) stay out.
    # The selection weighs each row by its source (a Modern Greek row is no
    # evidence of an Ancient declension), and a cell is not added when its
    # lowercase, grave-to-acute or monotonic spelling already reads as
    # another lemma.
    noun_sis, noun_sstem = select_noun_gap_lemmas(
        (form, lemma, source_of_input.get(combined_sources.get(form)))
        for form, lemma in combined.items())
    left_out = {k for table in (ag, el) for k in table if k not in combined}
    noun_cells = complete_noun_gaps(combined, noun_sis, noun_sstem, taken=left_out)
    for cell in noun_cells:
        combined_sources[cell["form"]] = GRC_NOUN_GAPS_SOURCE
    for kind, label in (("sis", "-σις noun"), ("sstem", "s-stem neuter")):
        cells = [c for c in noun_cells if c["kind"] == kind]
        print(f"  Ancient {label} cells ({GRC_NOUN_GAPS_SOURCE}): +{len(cells):,} "
              f"keys for {len({c['lemma'] for c in cells}):,} lemmas")

    _write_citation_rejection_report(citation_rejections)
    print(f"  Citation rejection report: {CITATION_REJECTIONS_PATH} "
          f"({len(citation_rejections):,} rows)")

    # NOTE: Corpus self-map and consensus overrides were tried here but
    # proved too aggressive, overriding correct Wiktionary entries with
    # corpus convention preferences (e.g. δεῖ -> δέομαι instead of δέω).
    # The targeted manual overrides below handle specific known issues.

    # Manual corrections for known lookup bugs.
    # These override wrong entries from Wiktionary or pipeline errors.
    _LOOKUP_OVERRIDES = {
        **reviewed_verb_lemmas(),
        **{form.title(): lemma for form, lemma in reviewed_verb_lemmas().items()},
        "φασιν": "φημί",          # was Φᾶσις (proper noun beats common verb)
        "Ἔστι": "εἰμί",          # was Ἔσθι (GLAUx annotation error)
        "σκέπτεσθαι": "σκέπτομαι",  # was self-map (infinitive as headword)
        "οἷον": "οἷος",           # was self-map (adverb use as headword)
        # Proper nouns beating common words (Herodotus benchmark)
        "μένοντας": "μένω",       # was Μένων (proper noun)
        "δευτέρης": "δεύτερος",   # was Δευτέρης (proper noun)
        "Μάγων": "μάγος",         # was Μάγων (proper noun beats common noun)
        "τέῳ": "τίς",            # was Τέως (place name beats pronoun)
        "ταφῆς": "ταφή",         # was Ταφῆς (place name beats common noun)
        "πάτρας": "πάτρα",       # was Πάτραι (place name beats common noun)
        "ἥρων": "ἥρως",          # was Ἥρων (proper noun beats common noun)
        # Wrong verb or noun picked for ambiguous forms
        "κατέχει": "κατέχω",      # was καταχέω
        "ἐπιστάμενος": "ἐπίσταμαι",  # was ἐφίστημι
        "ἁρπαγῇ": "ἁρπαγή",      # was ἁρπάζω (noun dative, not verb subj.)
        "ἀγορῇ": "ἀγορά",        # was ἀγοράζω (noun dative, not verb subj.)
        "φρουρῇ": "φρουρά",      # was φρουρέω (noun dative, not verb subj.)
        "φανεροῦ": "φανερός",    # was φανερόω (adj genitive, not verb)
        # AG-classical ambiguous forms: the more common noun/relative/
        # adjective/lexicalized-adverb reading, not the verb/numeral/adverb.
        "σκέψει": "σκέψις",      # was σκέπτομαι (noun dative, not verb fut.)
        "σε": "σύ",              # was σῦς (pig!) - enclitic 2sg pronoun acc.
        "σέ": "σύ",              # was σός - accented 2sg pronoun acc.
        "ποτέ": "ποτέ",          # was ποτός (drink) - lexicalized adverb
        "κοτέ": "ποτέ",          # Ionic ποτέ - was ποτός (drink)
        "κοτε": "ποτέ",          # Ionic ποτέ, unaccented/enclitic
        "κως": "πως",            # Ionic πως - was the island Κῶς
        "κου": "που",            # Ionic που - was ποῦ (interrogative)
        "ἕως": "ἕως",            # was ἠώς (Epic dawn); LSJ heads both the
                                 # "dawn" and the far commoner "until" at ἕως
        "ἑπτὰ": "ἑπτά",          # was ἑπτάς (heptad) - indeclinable numeral
        "Ἑπτά": "ἑπτά",          # capitalized variant of the same bug
        "οἰκίας": "οἰκία",       # was οἰκίον (rare diminutive; not its form)
        "οἰκίαν": "οἰκία",       # was οἰκίον
        "οἰκίᾳ": "οἰκία",        # was οἰκίον
        # Bare elided stems (apostrophe tokenized off by corpora): the only
        # possible reading is the elided function word, but the corpus pair
        # is a junk self-map or accent variant.
        "ἀλλ": "ἀλλά",           # was the self-map ἀλλ
        # μήτ (was the grave junk μὴτ) is in REVIEWED_ABSENT_KEY_READINGS.
        # NB: do NOT "fix" corpus-derived elided entries that reflect a
        # genuine ambiguity or lemmatization convention: λίπ᾽ is BOTH the
        # adverb λίπα (λίπ᾽ ἐλαίῳ) and elided λίπε (λείπω), and treebanks
        # lemmatize hortatory ἄγετ᾽ under the LSJ particle headword ἄγε,
        # not ἄγω. Overriding these regressed AGDT agreement.
        "ἧς": "ὅς",              # was εἷς (relative pron gen, not numeral)
        "ἐπιφανέστερον": "ἐπιφανής",  # was ἐπιφανῶς (adj comparative, not adverb)
        "πάντως": "πάντως",      # was πᾶς (lexicalized adverb; cf. καλῶς/οὕτως)
        "φανερῶν": "φανερός",    # was φανερόω (adj genitive pl., not verb)
        "βάθεος": "βαθύς",       # was βάθος (adj, not noun)
        "ἀσκοῦ": "ἀσκός",       # was ἀσκέω (noun genitive, not verb)
        "κατάρας": "κατάρα",     # was καταίρω (noun, not verb)
        "προσῆλθε": "προσέρχομαι",  # was πρόσειμι
        "ἀπεδέχθη": "ἀποδέχομαι",  # was ἀποδείκνυμι
        "ὑπάρξει": "ὑπάρχω",    # was ὕπαρξις (verb, not noun)
        "παύσει": "παύω",        # was παῦσις (verb, not noun)
        "ἀνήκω": "ἀνήκω",       # was ἀνίημι (self-map should win)
        "ἀνήκετε": "ἀνήκω",     # was ἀνίημι
        "δευτέρους": "δεύτερος", # was δύο (adj, not numeral)
        "θέης": "θέα",           # was θέω (noun, not verb)
        "σταθμῶν": "σταθμός",   # was στάθμη (common noun)
        "πλέοντας": "πλέω",      # was πολύς (verb, not adj)
        "πρόειπε": "προλέγω",    # was προαγορεύω
        "χωρῇ": "χώρα",         # was χωρέω (noun dative more common)
        # The merged layer of ag_lookup.json gives the γραφή paradigm to
        # γραφεύς (an Attic dual γραφή under γραφεύς) and the forms of σύ to
        # σῦς (form-of rows σεῖο -> σῦς); the Wiktionary base, every
        # dictionary and the treebanks have them under γραφή and σύ (#110).
        # These are exactly the keys the base reads as γραφή or σύ that the
        # merged layer moved to γραφεύς or σῦς.
        **{form: "γραφή" for form in (
            "γραφή", "γραφῆς", "γραφήν", "γραφαί", "γραφάς", "γραφαῖς",
            "γραφῶν", "γραφά", "γραφαῖν", "γραφη", "γραφην", "γραφαι",
            "γραφας", "γραφαις", "γραφα", "γραφαιν", "γραφαίν", "γραφαίς",
            "γραφών")},
        **{form: "σύ" for form in (
            "σύ", "συ", "σοῦ", "σού", "σου", "σεῖο", "σείο", "σειο", "σέο",
            "σεο", "σεῦ", "σεύ", "σευ", "σέθεν", "σεθεν", "τεΐν", "τειν")},
        "ἱερέων": "ἱερεύς",      # was ἱερός (gen. pl. of ἱερεύς; ἱερός has ἱερῶν)
        "ἄριστα": "ἀγαθός",      # was ἄριστον (as ἄριστος, ἀρίστους -> ἀγαθός)
        "ἀρίστοις": "ἀγαθός",    # was ἄριστον
        "ἀρὰν": "ἀρά",           # was ἀρή (expansion); ἀράν -> ἀρά
        **KAIKKI_2026_11_READING_OVERRIDES,
        **REVIEWED_ABSENT_KEY_READINGS,
    }
    # Also fix corrupt -δήποτε lemmas from pipeline
    for k, v in list(combined.items()):
        if "δήποτε" in k and "δήποτε" in v and v != k:
            stem = k.split("δήποτε")[0]
            if stem:
                _LOOKUP_OVERRIDES[k] = "ὁστισδήποτε"
    overridden = apply_lookup_overrides(
        _LOOKUP_OVERRIDES, combined=combined, combined_sources=combined_sources,
        ag=ag, ag_sources=ag_sources, el=el, el_sources=el_sources)
    if any(overridden.values()):
        print(f"  Lookup overrides applied: {overridden['changed']:,} readings "
              f"changed, {overridden['added']:,} absent keys added, "
              f"{overridden['el_removed']:,} el-only corpus readings removed")

    print(f"  Combined: {len(combined):,} entries")

    # Write SQLite database
    print(f"\nWriting {DB_PATH}...")
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    # Build-time only: the DB is regenerated from scratch, so crash
    # consistency buys nothing. journal_mode=OFF avoids the multi-GB
    # rollback journal whose final fsync intermittently fails with
    # "disk I/O error" on large builds; synchronous=OFF for the same
    # reason. The finished file is fully synced on close.
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA page_size=4096")

    # Deduplicated lemma table
    all_lemmas = sorted(set(combined.values()) | set(ag.values()) | set(el.values()))
    lemma_to_id = {lemma: i for i, lemma in enumerate(all_lemmas)}
    conn.execute("CREATE TABLE lemmas (id INTEGER PRIMARY KEY, text TEXT NOT NULL)")
    conn.executemany("INSERT INTO lemmas (id, text) VALUES (?, ?)",
                     enumerate(all_lemmas))
    print(f"  lemmas: {len(all_lemmas):,} distinct")

    # Main lookup: form -> lemma_id
    conn.execute("""CREATE TABLE lookup (
        form TEXT NOT NULL,
        lemma_id INTEGER NOT NULL,
        src TEXT NOT NULL,
        lang TEXT NOT NULL DEFAULT 'all',
        source_id INTEGER NOT NULL,
        FOREIGN KEY (lemma_id) REFERENCES lemmas(id),
        FOREIGN KEY (source_id) REFERENCES sources(id)
    )""")
    source_ids = write_lookup_sources(conn, lookup_sources)
    unlisted: dict[str, list[str]] = {}

    def source_id(label: str | None, form: str) -> int:
        """The registry id of the input that supplied a row's reading."""
        code = source_of_input.get(label)
        if code is None:
            unlisted.setdefault(str(label), []).append(form)
            return 0
        return source_ids[code]

    # Track which source provided each combined entry
    src_map = {}
    for data, src_label in [(ag, 'grc'), (el, 'el')]:
        for k in data:
            if k not in src_map:
                src_map[k] = src_label
            elif _is_self_map(k, combined.get(k, '')) and not _is_self_map(k, data[k]):
                src_map[k] = src_label

    conn.executemany(
        "INSERT INTO lookup (form, lemma_id, src, lang, source_id) "
        "VALUES (?, ?, ?, 'all', ?)",
        ((k, lemma_to_id[v], src_map.get(k, 'grc'),
          source_id(combined_sources.get(k), k)) for k, v in combined.items()))
    print(f"  combined: {len(combined):,} rows")

    # AG-only entries where AG differs from combined (for polytonic-first lookup)
    ag_extra = 0
    for k, v in ag.items():
        if combined.get(k) != v and not _is_article_map(k, v):
            conn.execute("INSERT INTO lookup (form, lemma_id, src, lang, source_id) "
                         "VALUES (?, ?, 'grc', 'grc', ?)",
                         (k, lemma_to_id[v], source_id(ag_sources.get(k), k)))
            ag_extra += 1
    print(f"  grc-only (differs from combined): {ag_extra:,} rows")

    # MG-only entries where MG differs from combined (for lang="el" mode)
    mg_extra = 0
    for k, v in el.items():
        if combined.get(k) != v and not _is_article_map(k, v):
            conn.execute("INSERT INTO lookup (form, lemma_id, src, lang, source_id) "
                         "VALUES (?, ?, 'el', 'el', ?)",
                         (k, lemma_to_id[v], source_id(el_sources.get(k), k)))
            mg_extra += 1
    print(f"  el-only (differs from combined): {mg_extra:,} rows")
    if unlisted:
        detail = "; ".join(f"{label}: {len(forms):,} rows, e.g. {forms[0]}"
                           for label, forms in sorted(unlisted.items()))
        raise SystemExit(f"lookup rows from inputs not in {LOOKUP_SOURCES_PATH.name}: "
                         f"{detail}")
    for code, count in conn.execute(
            "SELECT s.code, COUNT(*) FROM lookup l JOIN sources s ON s.id = l.source_id "
            "GROUP BY s.code ORDER BY COUNT(*) DESC"):
        print(f"  source {code}: {count:,} rows")

    conn.execute("CREATE INDEX idx_lookup_form_lang ON lookup (form, lang)")
    # Every form of a lemma without scanning the table (inflection indexes
    # for e-readers and apps read the lookup lemma-first).
    conn.execute("CREATE INDEX idx_lookup_lemma ON lookup (lemma_id)")
    conn.execute("CREATE INDEX idx_lemmas_text ON lemmas (text)")

    # Etymology bridges: MG lemma -> AG ancestor lemma
    conn.execute("""CREATE TABLE bridges (
        el_lemma_id INTEGER NOT NULL,
        grc_lemma_id INTEGER NOT NULL,
        FOREIGN KEY (el_lemma_id) REFERENCES lemmas(id),
        FOREIGN KEY (grc_lemma_id) REFERENCES lemmas(id)
    )""")
    bridge_pairs = set()
    bridge_skipped = 0
    if ETYMOLOGY_BRIDGES_PATH.exists():
        with open(ETYMOLOGY_BRIDGES_PATH, encoding="utf-8") as f:
            bridges = json.load(f)
        for mg_lemma, ag_ancestors in bridges.items():
            el_id = lemma_to_id.get(mg_lemma)
            if el_id is None:
                bridge_skipped += 1
                continue
            for ag_lemma in ag_ancestors:
                grc_id = lemma_to_id.get(ag_lemma)
                if grc_id is None:
                    bridge_skipped += 1
                    continue
                bridge_pairs.add((el_id, grc_id))
        etym_count = len(bridge_pairs)
        print(f"  etymology bridges: {etym_count:,} pairs ({bridge_skipped:,} skipped)")
    else:
        etym_count = 0
        print(f"  etymology bridges: not found")

    lsjgr_new = 0
    lsjgr_skipped = 0
    lsjgr_dup = 0
    if LSJGR_BRIDGES_PATH.exists():
        with open(LSJGR_BRIDGES_PATH, encoding="utf-8") as f:
            lsjgr_bridges = json.load(f)
        for mg_lemma, ag_ancestors in lsjgr_bridges.items():
            el_id = lemma_to_id.get(mg_lemma)
            if el_id is None:
                lsjgr_skipped += 1
                continue
            for ag_lemma in ag_ancestors:
                grc_id = lemma_to_id.get(ag_lemma)
                if grc_id is None:
                    lsjgr_skipped += 1
                    continue
                pair = (el_id, grc_id)
                if pair in bridge_pairs:
                    lsjgr_dup += 1
                else:
                    bridge_pairs.add(pair)
                    lsjgr_new += 1
        print(f"  lsjgr bridges: {lsjgr_new:,} new, {lsjgr_dup:,} dup, {lsjgr_skipped:,} skipped")
    else:
        print(f"  lsjgr bridges: not found")

    for el_id, grc_id in bridge_pairs:
        conn.execute("INSERT INTO bridges (el_lemma_id, grc_lemma_id) VALUES (?, ?)",
                     (el_id, grc_id))
    print(f"  bridges total: {len(bridge_pairs):,} (etymology: {etym_count:,}, lsjgr: {lsjgr_new:,})")
    conn.execute("CREATE INDEX idx_bridges_el ON bridges (el_lemma_id)")
    conn.execute("CREATE INDEX idx_bridges_grc ON bridges (grc_lemma_id)")

    # Related lemmas
    conn.execute("""CREATE TABLE related_lemmas (
        lemma_id INTEGER NOT NULL,
        related_id INTEGER NOT NULL,
        FOREIGN KEY (lemma_id) REFERENCES lemmas(id),
        FOREIGN KEY (related_id) REFERENCES lemmas(id)
    )""")
    related_count = 0
    related_skipped = 0
    if RELATED_LEMMAS_PATH.exists():
        with open(RELATED_LEMMAS_PATH, encoding="utf-8") as f:
            related_data = json.load(f)
        seen_pairs = set()
        for lemma, related_list in related_data.items():
            lemma_id = lemma_to_id.get(lemma)
            if lemma_id is None:
                related_skipped += 1
                continue
            for related in related_list:
                related_id = lemma_to_id.get(related)
                if related_id is None:
                    related_skipped += 1
                    continue
                pair = (lemma_id, related_id)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    conn.execute("INSERT INTO related_lemmas (lemma_id, related_id) VALUES (?, ?)",
                                 (lemma_id, related_id))
                    related_count += 1
        print(f"  related_lemmas: {related_count:,} pairs ({related_skipped:,} skipped)")
    else:
        print(f"  related_lemmas: not found")
    conn.execute("CREATE INDEX idx_related_lemma ON related_lemmas (lemma_id)")
    conn.execute("CREATE INDEX idx_related_related ON related_lemmas (related_id)")

    # A single lookup target is a citation preference. Keep every reviewed
    # grammatical reading separately, including lexical ambiguity and roles.
    # Older databases remain supported by the packaged runtime registry.
    conn.execute("CREATE TABLE grammatical_readings ("
                 "form TEXT PRIMARY KEY, readings TEXT NOT NULL)")
    conn.executemany("INSERT INTO grammatical_readings VALUES (?, ?)", (
        (form, json.dumps(reviewed_verb_readings(form), ensure_ascii=False,
                          sort_keys=True, separators=(",", ":")))
        for form in reviewed_verb_forms()))
    write_lookup_meta(conn)
    conn.commit()
    conn.close()

    # Compact main DB
    print("\nOptimizing lookup.db...")
    # Re-open for ANALYZE/VACUUM (avoids resource exhaustion on large DBs)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("ANALYZE")
    except sqlite3.OperationalError as e:
        print(f"  ANALYZE skipped: {e}")
    try:
        conn.execute("VACUUM")
    except sqlite3.OperationalError as e:
        print(f"  VACUUM skipped: {e}")
    conn.close()

    size_mb = DB_PATH.stat().st_size / 1e6
    print(f"  lookup.db: {size_mb:.1f} MB")

    # Build separate spell index (stripped form -> original forms)
    # Groups all polytonic variants under each stripped form, with src
    # tags for AG-mode filtering. Uses a compact single-row-per-stripped
    # format to minimize size.
    print("\nBuilding spell_index.db...")
    if SPELL_DB_PATH.exists():
        SPELL_DB_PATH.unlink()

    spell_conn = sqlite3.connect(str(SPELL_DB_PATH))
    spell_conn.execute("PRAGMA journal_mode=DELETE")
    spell_conn.execute("PRAGMA page_size=4096")
    # Each stripped form gets one row. `forms` is newline-separated list
    # of "form\tsrc" pairs (or just "form" when src is empty).
    spell_conn.execute("""CREATE TABLE spell (
        stripped TEXT PRIMARY KEY,
        forms TEXT NOT NULL
    ) WITHOUT ROWID""")

    # Read all forms from main DB and group by stripped form
    main_conn = sqlite3.connect(str(DB_PATH))
    main_conn.execute("PRAGMA mmap_size=268435456")
    rows = main_conn.execute(
        "SELECT DISTINCT form, src FROM lookup").fetchall()
    main_conn.close()

    grouped: dict[str, list[tuple[str, str]]] = {}
    for form, src in rows:
        stripped = strip_accents(form.lower())
        if stripped not in grouped:
            grouped[stripped] = []
        grouped[stripped].append((form, src))

    # Deduplicate within each group
    spell_rows = []
    for stripped, pairs in grouped.items():
        seen: set[str] = set()
        parts = []
        for form, src in pairs:
            if form not in seen:
                seen.add(form)
                parts.append(f"{form}\t{src}" if src else form)
        spell_rows.append((stripped, "\n".join(parts)))

    spell_conn.executemany(
        "INSERT INTO spell (stripped, forms) VALUES (?, ?)", spell_rows)
    print(f"  unique stripped forms: {len(spell_rows):,}")

    spell_conn.commit()
    spell_conn.execute("ANALYZE")
    try:
        spell_conn.execute("VACUUM")
    except sqlite3.OperationalError as e:
        print(f"  VACUUM skipped: {e}")
    spell_conn.close()

    spell_mb = SPELL_DB_PATH.stat().st_size / 1e6
    elapsed = time.time() - t0
    print(f"  spell_index.db: {spell_mb:.1f} MB")
    print(f"\nDone ({elapsed:.1f}s, total: {size_mb + spell_mb:.1f} MB)")


if __name__ == "__main__":
    # The build is openly licensed by default (no NonCommercial sources, no
    # variants): PROIEL dropped, Gorman + AGDT-Perseus kept, NC-filtered
    # glaux_pairs.json. Writes lookup.db + spell_index.db.
    build()

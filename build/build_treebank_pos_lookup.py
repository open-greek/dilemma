"""Build POS-indexed disambiguation lookup from gold treebank data.

Reads the AGDT originals (PerseusDL treebank_data .tb.xml, CC BY-SA),
plus the openly licensed lemmatized corpora (GLAUx, Diorisis,
Nestle-1904 NT). Extracts genuinely ambiguous forms: same surface form
maps to different lemmas depending on UPOS tag. The NonCommercial UD
treebanks (Perseus UD repackaging, PROIEL) are not read, and the Gorman
treebanks are excluded as the project's held-out gold corpus.

Output: data/treebank_pos_lookup.json
Format: {form: {UPOS: lemma, ...}, ...}
Companion audit: data/treebank_pos_lookup.audit.json (input hashes, rejected
corpus targets, tied owner votes, and complete capital name/ethnic owner votes
collected before the sparse runtime-table filter).

Only forms that are genuinely ambiguous (2+ distinct UPOS->lemma mappings)
are included, plus single corpus votes the runtime needs: one that disagrees
with the bare lookup, and one that agrees with it while a lower-priority
table or a case variant would answer differently, when Wiktionary defines the
lemma for the tag or when the form is itself a head an edited lexicon lists.
Monotonic and lowercase variants are added for lookup cascade. A capitalized
name that GLAUx attests well keeps its NOUN vote under its exact spelling.
"""

import argparse
import hashlib
import json
import sqlite3
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dilemma.form_sanitize import is_capitalized
from dilemma.grammatical_lemmas import correct_grammatical_lemma

TREEBANKS_DIR = Path(__file__).parent.parent / "data" / "treebanks"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "treebank_pos_lookup.json"
KAIKKI_DIR = None
LSJ9_HEADWORDS = None
AUDIT_PATH = None

# Openly licensed sources only. UD_Ancient_Greek-Perseus and -PROIEL are
# CC BY-NC-SA (NonCommercial) and are NOT used; the AGDT POS data comes from the
# Perseus original (.tb.xml, CC BY-SA 3.0 US) instead. DiGreC is dropped pending
# license verification.
import os
AGDT_DIR = Path(os.environ.get(
    "DILEMMA_AGDT_DIR", str(TREEBANKS_DIR / "treebank_data")))
TREEBANK_DIRS: list = []   # no CoNLL-U sources (the NC UD treebanks are excluded)


# Openly-licensed lemmatized corpora (already NC-filtered): GLAUx (CC BY-SA)
# and Diorisis (CC BY 4.0). They carry per-form POS + lemma, so they supply
# POS-keyed disambiguations the treebanks miss (θεῶ -> NOUN θεός / VERB
# θεάομαι, γραφῆς -> NOUN γραφή). GLAUx is weighted higher (98.8% vs 91.4%
# lemma accuracy). Used only via lemmatize_pos (POS-gated), so this never
# changes a bare, context-free lemmatize() result.
DATA_DIR = TREEBANKS_DIR.parent
GLAUX_PAIRS_PATH = DATA_DIR / "glaux_pairs.json"
DIORISIS_PAIRS_PATH = DATA_DIR / "diorisis_pairs.json"
# Openly-licensed Koine NT (Nestle 1904 lowfat, macula-greek, CC BY 4.0) -
# the open replacement for the dropped CC BY-NC-SA PROIEL NT.
NT_PAIRS_PATH = DATA_DIR / "nt_pairs.json"
# Manual-gold POS edges from cog's standardized exports. PTNK (UD, Koine LXX)
# and Pedalion (AGDT, classical) are hand-annotated, so - unlike all-auto OGA
# (rejected below) - their POS edges are trustworthy. TAGNT is intentionally
# absent: its Robinson tagset is not the AGDT/UD tagset this table trusts.
PTNK_PAIRS_PATH = DATA_DIR / "ptnk_pairs.json"
PEDALION_PAIRS_PATH = DATA_DIR / "pedalion_pairs.json"
LOOKUP_DB_PATH = DATA_DIR / "lookup.db"
_CORPUS_POS_TO_UPOS = {
    "verb": "VERB", "noun": "NOUN", "adj": "ADJ", "adv": "ADV",
    "pron": "PRON", "num": "NUM", "prep": "ADP", "conj": "CCONJ",
    "intj": "INTJ", "article": "DET", "particle": "PART",
}
# NT is gold human annotation (CC BY 4.0), weighted like GLAUx. OGA is
# all-auto model output (cog export): lowest weight, gap-fill only.
_CORPUS_WEIGHT = {"glaux": 2, "diorisis": 1, "nt": 2, "oga": 1,
                  "ptnk": 2, "pedalion": 2}

# AGDT POS code (position 1 of postag) -> UD UPOS
_AGDT_TO_UPOS = {
    "n": "NOUN", "v": "VERB", "a": "ADJ", "p": "PRON",
    "d": "ADV", "c": "CCONJ", "r": "ADP", "l": "DET",
    "g": "INTJ", "m": "NUM", "x": "X", "u": "PUNCT",
}

# Reuse Dilemma's monotonic conversion
_POLYTONIC_STRIP = {0x0313, 0x0314, 0x0345, 0x0306, 0x0304}
_POLYTONIC_TO_ACUTE = {0x0300, 0x0342}


def to_monotonic(s: str) -> str:
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


def strip_accents(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if unicodedata.category(c) != "Mn"))


def parse_conllu(path: Path):
    """Yield (form, lemma, upos) tuples from a CoNLL-U file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 4:
                continue
            tok_id = cols[0]
            # Skip multiword tokens (1-2) and empty nodes (1.1)
            if "-" in tok_id or "." in tok_id:
                continue
            form = cols[1]
            lemma = cols[2]
            upos = cols[3]
            # Skip punctuation
            if upos == "PUNCT":
                continue
            yield form, lemma, upos


def parse_gorman_xml(path: Path):
    """Yield (form, lemma, upos) from a Gorman AGDT XML treebank file."""
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return
    for word in tree.findall(".//word"):
        form = word.get("form", "")
        lemma = word.get("lemma", "")
        postag = word.get("postag", "")
        if not form or not lemma or not postag or len(postag) < 1:
            continue
        upos = _AGDT_TO_UPOS.get(postag[0], "")
        if not upos or upos == "PUNCT":
            continue
        yield form, lemma, upos


_JUNK_LEMMA_FINALS = tuple("᾽'’ʼ`ʹ")


def _clean_lemma(lemma: str) -> bool:
    """Reject junk lemma values: elided fragments (ἀλλ᾽) and abbreviation
    overlines (οὐδ̅, U+0305) are corpus artifacts, not headwords. A junk
    POS edge is worse than none - lemmatize_pos trusts the table over a
    single lookup candidate."""
    if not lemma or lemma.endswith(_JUNK_LEMMA_FINALS):
        return False
    return "̅" not in unicodedata.normalize("NFD", lemma)


def _lexical_target_key(lemma):
    """Match tonal/quantity variants without merging distinct lexical marks."""
    from dilemma_build.build_grc_lexical_heads import normalize_headword

    head = normalize_headword(lemma)
    if head is None:
        return None
    return unicodedata.normalize("NFC", "".join(
        char for char in unicodedata.normalize("NFD", head)
        if ord(char) not in {0x0300, 0x0301, 0x0342}
    ))


def _input_fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size": path.stat().st_size, "sha256": digest.hexdigest()}


# A listing in one of these does not show that a lower-case word is a
# dictionary head of its own: pd_headwords also lists inflected fragments
# (ὁρώ, ἵε) and lgpn_names lists personal names. Wiktionary is consulted
# separately, with its own test that the page defines the lemma for the tag.
_HEAD_SOURCES_NOT_COUNTED = frozenset({"pd_headwords", "lgpn_names"})
# Closed-class words are left to the reviewed grammatical lemmas.
_HEAD_VOTE_UPOS = frozenset({"NOUN", "ADJ", "ADV", "VERB"})


# Tests for the NOUN cell of a capitalized name, on GLAUx token counts: the
# capitalized spelling has enough tokens, nearly all of them are this name,
# and the name is not rare beside the lower-case noun of the same letters, so
# that capitals at the start of a sentence cannot pass for the name.
_NAME_MIN_FORM_TOKENS = 10
_NAME_MIN_SHARE = (9, 10)
_NAME_PER_LOWERCASE_NOUN = (1, 4)


def _listed_by_lexicon(sources):
    return any(not source.startswith("wiktionary_")
               and source not in _HEAD_SOURCES_NOT_COUNTED for source in sources)


def _load_lexical_targets():
    """Generated lookup membership cannot establish a corpus lemma's status."""
    path = DATA_DIR / "grc_lexical_heads.json"
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if (payload.get("schema_version") != 2
            or not isinstance(payload.get("heads"), dict)):
        raise ValueError(f"invalid merged lexical-head evidence: {path}")
    keys = {_lexical_target_key(head) for head, sources in payload["heads"].items()
            if isinstance(sources, list) and sources}
    keys.discard(None)
    listed = frozenset(head for head, sources in payload["heads"].items()
                       if isinstance(sources, list) and _listed_by_lexicon(sources))
    return keys, _input_fingerprint(path), listed


def _is_own_listed_head(form, upos, lemma, listed_heads):
    """The corpus reads a lower-case head of an edited lexicon as itself."""
    if (upos not in _HEAD_VOTE_UPOS or is_capitalized(form)
            or lemma not in listed_heads):
        return False
    # A final grave is the acute of running text. Unaccented spellings and
    # single letters are fragments and sigla, whatever a lexicon lists.
    spelled = unicodedata.normalize("NFD", form).replace("\u0300", "\u0301")
    if unicodedata.normalize("NFC", spelled) != lemma:
        return False
    letters = sum(1 for char in spelled if not unicodedata.combining(char))
    return letters >= 2 and any(char in "\u0301\u0342" for char in spelled)


def _unique_owner(counts):
    """A tied annotation is evidence for every owner, not for its first row."""
    highest = max(counts.values())
    owners = [lemma for lemma, count in counts.items() if count == highest]
    return owners[0] if len(owners) == 1 else None


def _record_capital_owner(votes, form, lemma, upos, source, weight, lexical_targets):
    """Keep defined name readings before the sparse POS-table selection."""
    if (upos not in {"NOUN", "PROPN", "ADJ"} or not is_capitalized(form)
            or not is_capitalized(lemma) or not _clean_lemma(lemma)
            or _lexical_target_key(lemma) not in lexical_targets):
        return
    form = unicodedata.normalize("NFC", form)
    lemma = unicodedata.normalize("NFC", lemma)
    sources = votes.setdefault(form, {}).setdefault((upos, lemma), {})
    sources[source] = sources.get(source, 0) + weight


def _capital_owner_audit(votes):
    """Serialize every supported owner, including tied and distinct readings."""
    return {
        form: [{"upos": upos, "lemma": lemma,
                "source_votes": dict(sorted(sources.items()))}
               for (upos, lemma), sources in sorted(owners.items())]
        for form, owners in sorted(votes.items())
    }


def _lower_priority_pos_models():
    """Use the runtime cascade, without the corpus table being rebuilt."""
    from dilemma.core import Dilemma

    models = []
    for lang in ("grc", "all"):
        model = Dilemma(lang=lang, scale=0, skip_pos=True)
        model._load_pos_lookups(include_treebank=False, data_dir=DATA_DIR)
        model.preload()
        models.append(model)
    return models


def _different_owner(lemma, other):
    return bool(other) and strip_accents(lemma) != strip_accents(other)


def _different_word(lemma, other):
    """Another word, not the same letters in the other case."""
    return (bool(other)
            and strip_accents(lemma).lower() != strip_accents(other).lower())


def _defined_pos_targets(wanted):
    """Require fresh independent definitions for additional corpus overrides."""
    if KAIKKI_DIR is None:
        return set()
    from build_data import WIKT_TO_UPOS, is_independent_lexical_entry, strip_length_marks

    from dilemma_build.kaikki_dumps import DUMPS, resolve_dump

    found = set()
    for source in ("en", "el"):
        path = resolve_dump(DUMPS["grc"][source], Path(KAIKKI_DIR))
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                target = (strip_length_marks(entry.get("word", "")),
                          WIKT_TO_UPOS.get(entry.get("pos")))
                if (entry.get("lang_code") == "grc" and target in wanted
                        and is_independent_lexical_entry(entry, source)):
                    found.add(target)
    return found



def _defined_projection_targets(wanted):
    """Use fresh definitions and fingerprint-matched merged LSJ evidence."""
    found = _defined_pos_targets(wanted)
    if not wanted or LSJ9_HEADWORDS is None:
        return found
    import hashlib
    from dilemma_build.build_grc_lexical_heads import normalize_headword

    merged = json.loads((DATA_DIR / "grc_lexical_heads.json").read_text())
    expected = merged["inputs"]["lsj9_headwords"]
    raw = Path(LSJ9_HEADWORDS).read_bytes()
    if (len(raw) != expected["size"]
            or hashlib.sha256(raw).hexdigest() != expected["sha256"]):
        raise ValueError("LSJ headwords do not match the merged lexical evidence")
    grammar_pos = {"ὁ": "NOUN", "ἡ": "NOUN", "τό": "NOUN",
                   "ον": "ADJ", "ές": "ADJ"}
    for entry in json.loads(raw):
        head = normalize_headword(entry.get("headword", ""))
        pair = (head, grammar_pos.get(entry.get("grammar")))
        if pair in wanted and "lsj9_headwords" in merged["heads"].get(head, []):
            found.add(pair)
    return found


def _project_pos_entries(lookup, conflicts=None, *, defined_targets=None):
    """Project only unique POS owners; retain every distinct POS reading."""
    result = {form: dict(values) for form, values in lookup.items()}
    proposals = defaultdict(lambda: defaultdict(set))
    for form, upos_lemmas in lookup.items():
        for key in {form.lower(), to_monotonic(form.lower())} - {form}:
            for upos, lemma in upos_lemmas.items():
                if upos in lookup.get(key, {}):
                    continue
                proposals[key][upos].add(lemma)
    # Adding a different POS beside an exact spelling broadens the prior
    # table's scope. Require independent lexical POS evidence for that reading:
    # inherited corpus labels can otherwise turn e.g. πείραι into ἀπολάω.
    wanted = {(lemma, upos) for key, values in proposals.items() if key in lookup
              for upos, lemmas in values.items() for lemma in lemmas
              if len(lemmas) == 1}
    defined = (_defined_projection_targets(wanted) if defined_targets is None and wanted
               else set() if defined_targets is None else defined_targets)
    for key, values in proposals.items():
        unique = {upos: next(iter(lemmas)) for upos, lemmas in sorted(values.items())
                  if len(lemmas) == 1 and (key not in lookup or
                     (next(iter(lemmas)), upos) in defined)}
        if unique:
            result.setdefault(key, {}).update(unique)
    for form, ambiguous_pos in (conflicts or {}).items():
        if form not in result:
            continue
        remaining = {upos: lemma for upos, lemma in result[form].items()
                     if upos in lookup.get(form, {}) or upos not in ambiguous_pos}
        if remaining:
            result[form] = remaining
        else:
            del result[form]
    return result


def _projection_conflicts(corpus, protected):
    """Retain competing corpus owners when new votes acquire aliases."""
    claims = defaultdict(lambda: defaultdict(set))
    for form, upos_counts in corpus.items():
        for key in {form.lower(), to_monotonic(form.lower())} & protected:
            for upos, counts in upos_counts.items():
                claims[key][upos].update(counts)
    return {key: {upos for upos, lemmas in values.items() if len(lemmas) > 1}
            for key, values in claims.items()}


def _scalar_query_forms(form):
    """Case variants of the literal and projected source form."""
    forms = {form, form.lower(), to_monotonic(form.lower())}
    return tuple(sorted(forms | {value[:1].upper() + value[1:] for value in forms}
                        | {value.upper() for value in forms}))


def _scalar_query_conflict(model, form, upos, lemma):
    """A matching lower-case owner may hide a different capital owner."""
    for query in _scalar_query_forms(form):
        if not _different_owner(lemma, model._lookup_word(query)):
            continue
        pos_lemma = model._pos_table_lookup(query, upos)
        if not pos_lemma or _different_owner(lemma, pos_lemma):
            return True
    return False


class _NameTokens:
    """GLAUx token counts behind the capitalized-name tests."""

    def __init__(self):
        self.spelling = defaultdict(int)     # capitalized spelling, any reading
        self.name = defaultdict(int)         # (spelling, capitalized noun lemma)
        self.lower_noun = defaultdict(int)   # lower-case spelling as a common noun

    def add(self, form, lemma, upos, tokens):
        if is_capitalized(form):
            self.spelling[form] += tokens
            if upos == "NOUN" and is_capitalized(lemma):
                self.name[form, lemma] += tokens
        elif upos == "NOUN" and not is_capitalized(lemma):
            self.lower_noun[form] += tokens

    def evidence(self, form, lemma):
        """The counts, when they show the capitalized spelling is this name."""
        total = self.spelling.get(form, 0)
        name = self.name.get((form, lemma), 0)
        lower = self.lower_noun.get(form.lower(), 0)
        share, ratio = _NAME_MIN_SHARE, _NAME_PER_LOWERCASE_NOUN
        if (total < _NAME_MIN_FORM_TOKENS or share[1] * name < share[0] * total
                or ratio[1] * name < ratio[0] * lower):
            return None
        return {"form_tokens": total, "name_tokens": name,
                "lowercase_noun_tokens": lower}


def _capital_name_cells(lookup, candidates, tokens, cascade):
    """Exact NOUN cells for capitalized names the corpus attests well.

    A capitalized spelling without a cell of its own is answered through the
    lower-case keys, by its lower-case twin or by another word (Θέρμον NOUN
    gave θέρμος). Keep the name's agreeing vote where GLAUx shows that the
    capitalized spelling is the name and every language scope reads it so
    without a tag. The cell is not projected: a lower-case or monotonic key
    belongs to the common word.
    """
    attested = {}
    for form, (upos, lemma) in candidates.items():
        if (upos != "NOUN" or form in lookup or not is_capitalized(form)
                or not is_capitalized(lemma)):
            continue
        evidence = tokens.evidence(form, lemma)
        if evidence is not None:
            attested[form] = (lemma, evidence)
    if not attested:
        return {}
    models = cascade.overlay(lookup)
    return {form: {"lemma": lemma, **evidence}
            for form, (lemma, evidence) in sorted(attested.items())
            if _own_row(models, form, lemma) and any(
                _different_owner(lemma, model._pos_table_lookup(form, "NOUN"))
                for model in models)}


def _own_row(models, form, lemma):
    """Every language scope reads the spelling as this lemma without a tag."""
    return all(model._lookup_word(form) == lemma for model in models)


class _LowerCascade:
    """The runtime's lower-priority tables, loaded once and only when needed."""

    def __init__(self):
        self._loaded = None

    def models(self):
        if self._loaded is None:
            models = _lower_priority_pos_models()
            self._loaded = (models, [(model._pos_ag_lookup, model._pos_lookup)
                                     for model in models])
        return self._loaded[0]

    def overlay(self, projected):
        """Put the table being built in front of the lower tables."""
        models, lower = self.models(), self._loaded[1]
        for model, tables in zip(models, lower):
            combined = []
            for table in tables:
                overlaid = table.copy()
                for form, values in projected.items():
                    overlaid[form] = {**table.get(form, {}), **values}
                combined.append(overlaid)
            model._pos_ag_lookup, model._pos_lookup = combined
        return models


def _restore_shadowed_corpus_votes(lookup, candidates, corpus, *, projection_corpus=None,
                                   head_votes=frozenset(), kept_heads=None,
                                   cascade=None):
    """Restore supported votes hidden by POS or case-sensitive scalar entries.

    A vote is supported when Wiktionary defines its lemma for the tag, or when
    its form is in head_votes: a lower-case head of an edited lexicon that the
    corpus and the bare lookup both read as itself. The second ground alone
    keeps a vote only where every language scope reads the form as that head
    without a tag and a lower table answers the form itself with another
    word. A differently cased query is no reason, nor is an answer that is
    the head's capitalized twin, which the runtime settles by the case of the
    form. kept_heads, when given, receives the votes kept on that ground.
    """
    unambiguous = {}
    for form, (upos, lemma) in candidates.items():
        counts = corpus[form][upos]
        highest = max(counts.values())
        owners = {owner for owner, count in counts.items() if count == highest}
        if owners == {lemma}:
            unambiguous[form] = (upos, lemma)
    # Keep every tied claim in corpus for projection-conflict checks, but do
    # not let an arbitrary ordering authorize an additional scalar override.
    candidates = unambiguous
    wanted = {(lemma, upos) for upos, lemma in candidates.values()}
    defined = _defined_pos_targets(wanted)
    listed_only = {form for form, value in candidates.items()
                   if form in head_votes and (value[1], value[0]) not in defined}
    candidates = {form: value for form, value in candidates.items()
                  if (value[1], value[0]) in defined or form in listed_only}
    if not candidates:
        return 0, {}
    cascade = cascade or _LowerCascade()
    added = 0
    protected = set()
    conflicts = {}
    while candidates:
        models = cascade.overlay(_project_pos_entries(lookup, conflicts))
        corrections = {}
        for form, (upos, lemma) in candidates.items():
            if form in listed_only:
                needed = (_own_row(models, form, lemma) and any(
                    _different_word(lemma, model._pos_table_lookup(form, upos))
                    for model in models))
            else:
                needed = any(
                    _different_owner(lemma, model._pos_table_lookup(form, upos))
                    or _scalar_query_conflict(model, form, upos, lemma)
                    for model in models)
            if needed:
                corrections[form] = {upos: lemma}
        if not corrections:
            break
        # Apply one simultaneous round so traversal order cannot change which
        # exact corpus votes are compared with the same projected table.
        lookup.update(corrections)
        added += len(corrections)
        if kept_heads is not None:
            for form in sorted(listed_only & set(corrections)):
                (upos, lemma), = corrections[form].items()
                kept_heads[form] = {"upos": upos, "lemma": lemma}
        for form in corrections:
            protected.update({form.lower(), to_monotonic(form.lower())} - {form})
            del candidates[form]
        conflicts = _projection_conflicts(
            corpus if projection_corpus is None else projection_corpus, protected)
    return added, conflicts


def build_lookup():
    # Load required evidence before touching either output. A SQLite lemma can
    # itself be an inferred or corrupt corpus target, so it is only a second
    # gate after the merged independent lexical-head inventory.
    lexical_targets, lexical_fingerprint, listed_heads = _load_lexical_targets()
    audit = {
        "schema_version": 2,
        "language": "grc",
        "inputs": {"grc_lexical_heads": lexical_fingerprint,
                   "builder": _input_fingerprint(Path(__file__))},
        "rejected_corpus_targets": {},
        "tied_owners": {},
        "capital_name_owners": {},
        "listed_head_votes": {},
        "capital_name_cells": {},
    }
    for relative in ("dilemma/core.py", "dilemma/form_sanitize.py",
                     "build/build_grc_lexical_heads.py", "build_data.py"):
        audit["inputs"][relative] = _input_fingerprint(ROOT / relative)
    # The recovery pass compares corpus votes with the runtime's lower
    # priority tables, so their exact input bytes belong in the audit too.
    for name in ("lsj9_indeclinables.json", "glaux_pos_lookup.json",
                 "mg_pos_lookup.json", "ag_pos_lookup.json", "lsj9_pos_lookup.json"):
        path = DATA_DIR / name
        if path.exists():
            audit["inputs"][name] = _input_fingerprint(path)
    if KAIKKI_DIR is not None:
        from dilemma_build.kaikki_dumps import DUMPS, resolve_dump
        for source, edition in (("wiktionary_en_grc", "en"), ("wiktionary_el_grc", "el")):
            path = resolve_dump(DUMPS["grc"][edition], Path(KAIKKI_DIR))
            if path.exists():
                audit["inputs"][source] = _input_fingerprint(path)
    if LSJ9_HEADWORDS is not None:
        audit["inputs"]["lsj9_headwords"] = _input_fingerprint(Path(LSJ9_HEADWORDS))

    # Collect all (form, upos) -> {lemma: count} from treebanks
    # form_upos_lemmas[form][upos][lemma] = count
    form_upos_lemmas = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    capital_owner_votes = {}
    total_tokens = 0

    # CoNLL-U treebanks
    for treebank_dir in TREEBANK_DIRS:
        if not treebank_dir.exists():
            print(f"  Skipping {treebank_dir.name} (not found)")
            continue
        conllu_files = sorted(treebank_dir.glob("*.conllu"))
        for f in conllu_files:
            audit["inputs"][f"treebank/{treebank_dir.name}/{f.name}"] = _input_fingerprint(f)
            count = 0
            for form, lemma, upos in parse_conllu(f):
                if not _clean_lemma(lemma):
                    continue
                form_upos_lemmas[form][upos][lemma] += 1
                _record_capital_owner(capital_owner_votes, form, lemma, upos,
                                      f"treebank/{treebank_dir.name}", 1, lexical_targets)
                count += 1
            total_tokens += count
            print(f"  {f.name}: {count} tokens")

    # AGDT original (.tb.xml, CC BY-SA 3.0 US) - the openly licensed Perseus
    # POS source, replacing the NonCommercial UD_Ancient_Greek-Perseus.
    agdt_files = []
    for _ver in ("v2.1", "v2.0", "v1.6"):
        _texts = AGDT_DIR / _ver / "Greek" / "texts"
        if _texts.exists():
            agdt_files = sorted(_texts.glob("*.tb.xml"))
            break
    if agdt_files:
        agdt_tokens = 0
        for f in agdt_files:
            audit["inputs"][f"agdt/{f.relative_to(AGDT_DIR).as_posix()}"] = _input_fingerprint(f)
            for form, lemma, upos in parse_gorman_xml(f):
                if not _clean_lemma(lemma):
                    continue
                form_upos_lemmas[form][upos][lemma] += 1
                _record_capital_owner(capital_owner_votes, form, lemma, upos,
                                      "agdt", 1, lexical_targets)
                agdt_tokens += 1
        total_tokens += agdt_tokens
        print(f"  AGDT (Perseus original): {len(agdt_files)} files, "
              f"{agdt_tokens:,} tokens")
    else:
        print("  Skipping AGDT (not found; set DILEMMA_AGDT_DIR)")

    # The Gorman treebanks are deliberately NOT read: they are the
    # project's held-out gold corpus (eval/eval_gorman_gold.py), so no
    # shipped artifact may derive from them.

    print(f"\nTotal tokens: {total_tokens}")
    print(f"Unique forms (treebanks): {len(form_upos_lemmas)}")

    # Openly licensed corpus POS edges must name an independent lexical head
    # as well as a built lemma. Keep them separate so they only fill UPOS gaps
    # the gold treebanks do not cover; per UPOS, treebank votes always win.
    corpus_upos_lemmas = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int)))
    # A rejected exact target still documents a competing source reading.
    # Keep those claims to prevent a newly projected spelling from silently
    # absorbing a different source form when its target lacks lexical support.
    projection_corpus = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    name_tokens = _NameTokens()
    lemma_set = set()
    if LOOKUP_DB_PATH.exists():
        audit["inputs"]["lookup"] = _input_fingerprint(LOOKUP_DB_PATH)
        _c = sqlite3.connect(str(LOOKUP_DB_PATH))
        lemma_set = {r[0] for r in _c.execute("SELECT text FROM lemmas")}
        _c.close()
    for src, path in (("glaux", GLAUX_PAIRS_PATH),
                      ("diorisis", DIORISIS_PAIRS_PATH),
                      ("nt", NT_PAIRS_PATH),
                      ("ptnk", PTNK_PAIRS_PATH),
                      ("pedalion", PEDALION_PAIRS_PATH)):
        if not path.exists():
            print(f"  Skipping {src} corpus ({path.name} not found)")
            continue
        audit["inputs"][src] = _input_fingerprint(path)
        w = _CORPUS_WEIGHT[src]
        n = 0
        seen_edges = set()
        for p in json.load(open(path, encoding="utf-8")):
            form, lemma = p.get("form"), p.get("lemma")
            upos = _CORPUS_POS_TO_UPOS.get(p.get("pos"))
            if src == "glaux" and form and lemma:
                name_tokens.add(form, lemma, upos, int(p.get("count", 1)))
            if not form or not lemma or not upos:
                continue
            if not _clean_lemma(lemma):
                continue
            # Distinct morphological readings may share one POS edge.
            # They preserve source ambiguity, not independent votes.
            edge = (form, upos, lemma)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            projection_corpus[form][upos][lemma] += w
            if _lexical_target_key(lemma) not in lexical_targets:
                counts = audit["rejected_corpus_targets"].setdefault(lemma, {})
                counts[src] = counts.get(src, 0) + 1
                continue
            _record_capital_owner(capital_owner_votes, form, lemma, upos,
                                  src, w, lexical_targets)
            if lemma_set and lemma not in lemma_set:
                continue
            corpus_upos_lemmas[form][upos][lemma] += w
            n += 1
        print(f"  {src} corpus: +{n:,} POS edges (weight {w})")

    # OGA edges were evaluated and REJECTED (2026-07-12): even gap-fill-only,
    # lowercase-form-only edges from cog's OGA export cost 0.3pp on the
    # Byzantine gold-POS benchmark (91.8 -> 91.5) and gained nothing
    # measurable elsewhere - all-auto annotation is too noisy for a table
    # that lemmatize_pos trusts over lookup candidates. OGA participates in
    # the attestation artifacts only.

    # Filter to genuinely ambiguous forms:
    # This independent audit keeps unambiguous names too. The export needs
    # lexical ownership even when a name does not need POS disambiguation.
    audit["capital_name_owners"] = _capital_owner_audit(capital_owner_votes)

    # A form is ambiguous if it has multiple DISTINCT (upos -> lemma) mappings,
    # meaning different UPOS tags lead to different lemmas.
    # For each UPOS, require a unique most frequent lemma. Retain all tied
    # owners in the audit instead of making input order select one of them.
    lookup = {}
    for form in sorted(set(form_upos_lemmas) | set(corpus_upos_lemmas)):
        tb = form_upos_lemmas.get(form, {})
        co = corpus_upos_lemmas.get(form, {})
        # For each UPOS, pick the most frequent lemma; gold treebank wins, the
        # corpus only fills UPOS the treebank doesn't cover for this form.
        resolved = {}
        for upos in sorted(set(tb) | set(co)):
            counts = tb[upos] if upos in tb else co[upos]
            owner = _unique_owner(counts)
            if owner is not None:
                resolved[upos] = owner
            else:
                audit["tied_owners"].setdefault(form, {})[upos] = {
                    "source": "treebank" if upos in tb else "corpus",
                    "votes": dict(sorted(counts.items())),
                }

        # Only keep forms where different UPOS tags map to different lemmas
        if len(set(resolved.values())) < 2:
            continue

        lookup[form] = resolved

    print(f"Ambiguous forms (different UPOS -> different lemma): {len(lookup)}")

    # Corpus-disagreement entries (the "kept-source-dropped" case): a form the
    # ambiguity filter dropped (one resolved UPOS) whose well-supported corpus
    # lemma disagrees with the current single-value lookup - e.g. γραφῆς, where
    # the lookup kept Wiktionary's γραφεύς and discarded the corpus γραφή. Add
    # it as a POS-keyed candidate so lemmatize_pos surfaces the corpus reading.
    # A matching bare lookup does not make the corpus vote redundant when a
    # lower-priority POS table would return a different lemma. Check both
    # language scopes with the runtime's own source order and normalization.
    # POS-gated, so bare context-free lemmatize() is unaffected.
    bare = {}
    if LOOKUP_DB_PATH.exists():
        _c = sqlite3.connect(str(LOOKUP_DB_PATH))
        for _f, _l in _c.execute(
                "SELECT k.form, l.text FROM lookup k "
                "JOIN lemmas l ON k.lemma_id = l.id WHERE k.lang = 'all'"):
            if _f not in bare:
                bare[_f] = _l
        _c.close()
    fallback_candidates = {}
    head_votes = set()
    disagree_added = 0
    for form, upos_dict in corpus_upos_lemmas.items():
        if form in lookup or form not in bare:
            continue
        strongest = max(max(counts.values()) for counts in upos_dict.values())
        strongest_pos = [upos for upos, counts in upos_dict.items()
                         if max(counts.values()) == strongest]
        if len(strongest_pos) != 1:
            continue
        upos = strongest_pos[0]
        counts = upos_dict[upos]
        lemma = _unique_owner(counts)
        if lemma is None:
            continue
        if counts[lemma] < 2:          # require GLAUx (w=2) or GLAUx+Diorisis
            continue
        cur = bare[form]
        if _different_owner(lemma, cur):
            lookup[form] = {upos: lemma}
            disagree_added += 1
        else:
            fallback_candidates[form] = (upos, lemma)
            # The lookup's own row must be the head too, accent for accent:
            # a tagged answer that leaves the untagged one helps nobody.
            if cur == lemma and _is_own_listed_head(form, upos, lemma, listed_heads):
                head_votes.add(form)
    cascade = _LowerCascade()
    restored, conflicts = _restore_shadowed_corpus_votes(
        lookup, fallback_candidates, corpus_upos_lemmas,
        projection_corpus=projection_corpus, head_votes=head_votes,
        kept_heads=audit["listed_head_votes"], cascade=cascade)
    print(f"Corpus-disagreement POS entries added: {disagree_added}; "
          f"defined votes retained against POS fallback: {restored} "
          f"({len(audit['listed_head_votes'])} as listed heads)")

    lookup = _project_pos_entries(lookup, conflicts)
    print(f"After adding lowercase/monotonic variants: {len(lookup)}")

    # After the projection, so that the lower-case keys a capitalized
    # spelling falls back to are the ones the runtime will read.
    audit["capital_name_cells"] = _capital_name_cells(
        lookup, fallback_candidates, name_tokens, cascade)
    for form, cell in audit["capital_name_cells"].items():
        lookup[form] = {"NOUN": cell["lemma"]}
    print(f"Capitalized names kept under their exact spelling: "
          f"{len(audit['capital_name_cells'])}")

    for form, readings in lookup.items():
        for upos in ("VERB", "AUX"):
            if upos in readings:
                readings[upos] = correct_grammatical_lemma(form, readings[upos])

    # Sort for stable output
    lookup = dict(sorted(lookup.items()))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False, indent=1)

    audit_path = AUDIT_PATH or OUTPUT_PATH.with_suffix(".audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, sort_keys=True,
                                    indent=1) + "\n", encoding="utf-8")

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Corpus target audit: {audit_path}")
    return lookup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaikki", type=Path,
                        help="Fresh Kaikki directory for independent POS target definitions")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--audit-output", type=Path,
                        help="Deterministic source/target audit (default: OUTPUT.audit.json)")
    parser.add_argument("--lsj9-headwords", type=Path,
                        help="Optional LSJ export matching the merged lexical-head fingerprint")
    args = parser.parse_args()
    KAIKKI_DIR = args.kaikki
    LSJ9_HEADWORDS = args.lsj9_headwords
    OUTPUT_PATH = args.output
    AUDIT_PATH = args.audit_output
    build_lookup()

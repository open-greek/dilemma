"""Build POS-indexed disambiguation lookup from gold treebank data.

Reads the AGDT originals (PerseusDL treebank_data .tb.xml, CC BY-SA),
plus the openly licensed lemmatized corpora (GLAUx, Diorisis,
Nestle-1904 NT). Extracts genuinely ambiguous forms: same surface form
maps to different lemmas depending on UPOS tag. The NonCommercial UD
treebanks (Perseus UD repackaging, PROIEL) are not read, and the Gorman
treebanks are excluded as the project's held-out gold corpus.

Output: data/treebank_pos_lookup.json
Format: {form: {UPOS: lemma, ...}, ...}

Only forms that are genuinely ambiguous (2+ distinct UPOS->lemma mappings)
are included. Monotonic and lowercase variants are added for lookup cascade.
"""

import argparse
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

TREEBANKS_DIR = Path(__file__).parent.parent / "data" / "treebanks"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "treebank_pos_lookup.json"
KAIKKI_DIR = None

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


def _defined_pos_targets(wanted):
    """Require fresh independent definitions for additional corpus overrides."""
    if KAIKKI_DIR is None:
        return set()
    from build_data import WIKT_TO_UPOS, is_independent_lexical_entry, strip_length_marks

    found = set()
    for source, relative in (
            ("en", "en-el/kaikki.org-dictionary-AncientGreek.jsonl"),
            ("el", "el/kaikki.org-dictionary-AncientGreek.jsonl")):
        path = Path(KAIKKI_DIR) / relative
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                target = (strip_length_marks(entry.get("word", "")),
                          WIKT_TO_UPOS.get(entry.get("pos")))
                if (entry.get("lang_code") == "grc" and target in wanted
                        and is_independent_lexical_entry(entry, source)):
                    found.add(target)
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
    # table's scope. Require an independent definition for that new reading:
    # inherited corpus labels can otherwise turn e.g. πείραι into ἀπολάω.
    wanted = {(lemma, upos) for key, values in proposals.items() if key in lookup
              for upos, lemmas in values.items() for lemma in lemmas
              if len(lemmas) == 1}
    defined = (_defined_pos_targets(wanted) if defined_targets is None and wanted
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


def _restore_shadowed_corpus_votes(lookup, candidates, corpus):
    """Restore defined votes hidden by POS or case-sensitive scalar entries."""
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
    candidates = {form: value for form, value in candidates.items()
                  if (value[1], value[0]) in defined}
    if not candidates:
        return 0, {}
    models = _lower_priority_pos_models()
    lower = [(model._pos_ag_lookup, model._pos_lookup) for model in models]
    added = 0
    protected = set()
    conflicts = {}
    while candidates:
        projected = _project_pos_entries(lookup, conflicts)
        for model, tables in zip(models, lower):
            combined = []
            for table in tables:
                overlaid = table.copy()
                for form, values in projected.items():
                    overlaid[form] = {**table.get(form, {}), **values}
                combined.append(overlaid)
            model._pos_ag_lookup, model._pos_lookup = combined
        corrections = {}
        for form, (upos, lemma) in candidates.items():
            if any(_different_owner(lemma, model._pos_table_lookup(form, upos))
                   or _scalar_query_conflict(model, form, upos, lemma)
                   for model in models):
                corrections[form] = {upos: lemma}
        if not corrections:
            break
        # Apply one simultaneous round so traversal order cannot change which
        # exact corpus votes are compared with the same projected table.
        lookup.update(corrections)
        added += len(corrections)
        for form in corrections:
            protected.update({form.lower(), to_monotonic(form.lower())} - {form})
            del candidates[form]
        conflicts = _projection_conflicts(corpus, protected)
    return added, conflicts


def build_lookup():
    # Collect all (form, upos) -> {lemma: count} from treebanks
    # form_upos_lemmas[form][upos][lemma] = count
    form_upos_lemmas = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    total_tokens = 0

    # CoNLL-U treebanks
    for treebank_dir in TREEBANK_DIRS:
        if not treebank_dir.exists():
            print(f"  Skipping {treebank_dir.name} (not found)")
            continue
        conllu_files = sorted(treebank_dir.glob("*.conllu"))
        for f in conllu_files:
            count = 0
            for form, lemma, upos in parse_conllu(f):
                if not _clean_lemma(lemma):
                    continue
                form_upos_lemmas[form][upos][lemma] += 1
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
            for form, lemma, upos in parse_gorman_xml(f):
                if not _clean_lemma(lemma):
                    continue
                form_upos_lemmas[form][upos][lemma] += 1
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

    # Openly-licensed corpus POS edges (GLAUx + Diorisis), validated against
    # the built lemma set. Kept SEPARATE so they only fill UPOS gaps the gold
    # treebanks don't cover - per UPOS, treebank votes always win.
    corpus_upos_lemmas = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int)))
    lemma_set = set()
    if LOOKUP_DB_PATH.exists():
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
        w = _CORPUS_WEIGHT[src]
        n = 0
        seen_edges = set()
        for p in json.load(open(path, encoding="utf-8")):
            form, lemma = p.get("form"), p.get("lemma")
            upos = _CORPUS_POS_TO_UPOS.get(p.get("pos"))
            if not form or not lemma or not upos:
                continue
            if lemma_set and lemma not in lemma_set:
                continue
            if not _clean_lemma(lemma):
                continue
            # Distinct morphological readings may share one POS edge.
            # They preserve source ambiguity, not independent votes.
            edge = (form, upos, lemma)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
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
    # A form is ambiguous if it has multiple DISTINCT (upos -> lemma) mappings,
    # meaning different UPOS tags lead to different lemmas.
    # For each UPOS, pick the most frequent lemma.
    lookup = {}
    for form in sorted(set(form_upos_lemmas) | set(corpus_upos_lemmas)):
        tb = form_upos_lemmas.get(form, {})
        co = corpus_upos_lemmas.get(form, {})
        # For each UPOS, pick the most frequent lemma; gold treebank wins, the
        # corpus only fills UPOS the treebank doesn't cover for this form.
        resolved = {}
        for upos in sorted(set(tb) | set(co)):
            counts = tb[upos] if upos in tb else co[upos]
            resolved[upos] = max(counts, key=counts.get)

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
    disagree_added = 0
    for form, upos_dict in corpus_upos_lemmas.items():
        if form in lookup or form not in bare:
            continue
        upos = max(upos_dict, key=lambda u: max(upos_dict[u].values()))
        counts = upos_dict[upos]
        lemma = max(counts, key=counts.get)
        if counts[lemma] < 2:          # require GLAUx (w=2) or GLAUx+Diorisis
            continue
        cur = bare[form]
        if _different_owner(lemma, cur):
            lookup[form] = {upos: lemma}
            disagree_added += 1
        else:
            fallback_candidates[form] = (upos, lemma)
    restored, conflicts = _restore_shadowed_corpus_votes(
        lookup, fallback_candidates, corpus_upos_lemmas)
    print(f"Corpus-disagreement POS entries added: {disagree_added}; "
          f"defined votes retained against POS fallback: {restored}")

    lookup = _project_pos_entries(lookup, conflicts)
    print(f"After adding lowercase/monotonic variants: {len(lookup)}")

    # Sort for stable output
    lookup = dict(sorted(lookup.items()))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False, indent=1)

    print(f"\nSaved to {OUTPUT_PATH}")
    return lookup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaikki", type=Path,
                        help="Fresh Kaikki directory for independent POS target definitions")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    KAIKKI_DIR = args.kaikki
    OUTPUT_PATH = args.output
    build_lookup()

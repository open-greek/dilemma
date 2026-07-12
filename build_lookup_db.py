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

import json
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from dilemma.form_sanitize import sanitize_form  # noqa: E402
from dilemma import grave_to_acute, to_monotonic  # noqa: E402

DATA_DIR = SCRIPT_DIR / "data"
DB_PATH = DATA_DIR / "lookup.db"
SPELL_DB_PATH = DATA_DIR / "spell_index.db"
RAW_DB_PATH = DATA_DIR / "raw_lookups.db"

AG_PATH = DATA_DIR / "ag_lookup.json"
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
MG_PATH = DATA_DIR / "mg_lookup.json"
MED_PATH = DATA_DIR / "med_lookup.json"
GLAUX_PAIRS_PATH = DATA_DIR / "glaux_pairs.json"
DIORISIS_PAIRS_PATH = DATA_DIR / "diorisis_pairs.json"
# Openly-licensed Koine NT (Nestle 1904 lowfat, macula-greek, CC BY 4.0;
# Nestle 1904 base text is public domain). Open replacement for the dropped
# CC BY-NC-SA PROIEL NT. Built by build/extract_nt.py.
NT_PAIRS_PATH = DATA_DIR / "nt_pairs.json"
PERSEUS_PAIRS_PATH = DATA_DIR / "perseus_pairs.json"
ETYMOLOGY_BRIDGES_PATH = DATA_DIR / "etymology_bridges.json"
LSJGR_BRIDGES_PATH = DATA_DIR / "lsjgr_bridges.json"
RELATED_LEMMAS_PATH = DATA_DIR / "related_lemmas.json"
HNC_PAIRS_PATH = DATA_DIR / "hnc_pairs.json"

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


def _load_lookup(table: str, json_path: Path, label: str) -> dict:
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
            return json_data
        else:
            print(f"  {label}: {len(sqlite_data):,} entries from SQLite ({time.time()-t0:.1f}s)")
            return sqlite_data
    elif sqlite_data:
        print(f"  {label}: {len(sqlite_data):,} entries from SQLite ({time.time()-t0:.1f}s)")
        return sqlite_data
    elif json_data:
        print(f"  {label}: {len(json_data):,} entries from JSON ({time.time()-t0:.1f}s)")
        return json_data
    else:
        print(f"  {label}: no data found")
        return {}


def build():
    t0 = time.time()

    print("Loading lookup tables...")
    ag = _load_lookup("ag", AG_PATH, "AG")
    el = _load_lookup("mg", MG_PATH, "MG")
    med = _load_lookup("med", MED_PATH, "Med")

    # Merge med into el: vernacular medieval Greek is the ancestor of
    # Modern Greek, and EL Wiktionary's "Medieval Greek" category contains
    # early MG vocabulary, not Byzantine literary Greek.
    med_merged = 0
    for k, v in med.items():
        if k not in el:
            el[k] = v
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
            if form not in ag:
                ag[form] = lemma
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
        for h in vlg_raw:
            if h not in ag:
                ag[h] = h
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
        for h in wip_raw:
            if h not in ag:
                ag[h] = h
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
        for h in lsj10_raw:
            if h not in ag:
                ag[h] = h
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
        headword, prefer it.
        """
        if lemma and lemma[0].isupper():
            lower = lemma[0].lower() + lemma[1:]
            if lower in headwords_exact:
                return lower
        return lemma

    # Reject pairs whose lemma is not a known AG headword to filter
    # out annotation errors (e.g. Ἔστι -> Ἔσθι, corrupt -δήποτε lemmas).
    glaux_added_ag = 0
    glaux_added_med = 0
    glaux_skipped_med = 0
    glaux_bad_lemma = 0
    if GLAUX_PAIRS_PATH.exists():
        t_g = time.time()
        with open(GLAUX_PAIRS_PATH, encoding="utf-8") as f:
            glaux_pairs = json.load(f)
        # Snapshot AG keys before GLAUx expansion, so we can check
        # whether a form had an original (Wiktionary) AG entry.
        ag_original = dict(ag)
        for p in glaux_pairs:
            form, lemma = p["form"], p["lemma"]
            # Normalize capitalized lemmas (GLAUx sentence-initial convention)
            lemma = _normalize_corpus_lemma(lemma, ag_headwords_exact)
            # Validate lemma against known AG headwords
            if ag_headwords_exact and lemma not in ag_headwords_exact:
                glaux_bad_lemma += 1
                continue
            # Add to AG if not already present
            if form not in ag:
                ag[form] = lemma
                glaux_added_ag += 1
            # Selectively add to el: only when the pair won't cause
            # a priority override conflict in the combined merge.
            if form not in el:
                if form not in ag_original:
                    el[form] = lemma
                    glaux_added_med += 1
                elif ag_original[form] == lemma:
                    el[form] = lemma
                    glaux_added_med += 1
                else:
                    glaux_skipped_med += 1
        print(f"  GLAUx: +{glaux_added_ag:,} to AG, "
              f"+{glaux_added_med:,} to el, "
              f"{glaux_skipped_med:,} el conflicts skipped, "
              f"{glaux_bad_lemma:,} bad lemmas rejected "
              f"({time.time()-t_g:.1f}s)")

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
        for p in diorisis_pairs:
            form, lemma = p["form"], p["lemma"]
            # Normalize capitalized lemmas
            lemma = _normalize_corpus_lemma(lemma, ag_headwords_exact)
            # Validate lemma against known AG headwords
            if ag_headwords_exact and lemma not in ag_headwords_exact:
                dior_bad_lemma += 1
                continue
            # Add to AG if not already present from any source
            if form not in ag:
                ag[form] = lemma
                dior_added_ag += 1
            else:
                dior_skipped_ag += 1
            # Selectively add to el: only when the pair won't cause
            # a priority override conflict in the combined merge.
            if form not in el:
                if form not in ag_before_dior:
                    el[form] = lemma
                    dior_added_el += 1
                elif ag_before_dior[form] == lemma:
                    el[form] = lemma
                    dior_added_el += 1
                else:
                    dior_skipped_el += 1
        print(f"  Diorisis: +{dior_added_ag:,} to AG ({dior_skipped_ag:,} skipped), "
              f"+{dior_added_el:,} to el, "
              f"{dior_skipped_el:,} el conflicts skipped, "
              f"{dior_bad_lemma:,} bad lemmas rejected "
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
            if ag_headwords_exact and lemma not in ag_headwords_exact:
                nt_bad_lemma += 1
                continue
            if form not in ag:
                ag[form] = lemma
                nt_added_ag += 1
            if form not in el and form not in ag_before_nt:
                el[form] = lemma
        print(f"  Koine NT: +{nt_added_ag:,} to AG, "
              f"{nt_bad_lemma:,} bad lemmas rejected ({time.time()-t_n:.1f}s)")

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

    # Sanitise every form and lemma so a stray combining breathing mark
    # (leading U+0313/U+0314/U+1FBF/U+1FFE, or trailing U+0313/U+0314 used
    # as an apostrophe) cannot leak into lookup.db from any upstream source.
    # LSJ paradigm expansion produces leading combining-psili forms like
    # `̓Αβαρικός`, and treebank exports encode elision as trailing psili
    # (`μετ̓`). See form_sanitize.sanitize_form for the rules.
    def _sanitize_table(name: str, table: dict) -> dict:
        out: dict = {}
        changed = 0
        dropped_elided = 0
        for k, v in table.items():
            sk = sanitize_form(k)
            sv = sanitize_form(v) if isinstance(v, str) else v
            if not sk:
                continue
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
        if changed:
            print(f"  Sanitised {changed:,} {name} forms "
                  f"(misplaced breathing marks)")
        if dropped_elided:
            print(f"  Dropped {dropped_elided:,} {name} trailing-apostrophe "
                  f"elided forms (resolved via elision layer)")
        return out

    print("\nSanitising form-and-lemma tables...")
    ag = _sanitize_table("AG", ag)
    el = _sanitize_table("EL", el)

    # Build combined lookup (AG-first priority)
    print("\nBuilding combined lookup (AG-first)...")
    combined = {}
    ag_protected = 0
    article_excluded = 0
    for data in [ag, el]:
        for k, v in data.items():
            if _is_article_map(k, v):
                article_excluded += 1
                continue
            if k not in combined:
                combined[k] = v
            elif _is_self_map(k, combined[k]) and not _is_self_map(k, v):
                # EL non-self-map overrides AG self-map, UNLESS the AG
                # self-map is a known AG headword (correct citation form).
                if k in ag_headwords or combined[k] in ag_headwords:
                    ag_protected += 1
                else:
                    combined[k] = v
            elif (_is_self_map(k, combined[k])
                  and _is_self_map(k, v) and v == k
                  and combined[k] != k):
                combined[k] = v
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
    if LBG_HEADWORDS_PATH.exists():
        with open(LBG_HEADWORDS_PATH, encoding="utf-8") as f:
            lbg_raw = {e["lemma"] for e in json.load(f)
                       if e.get("lemma") and " " not in e["lemma"]}
        for h in lbg_raw:
            if h not in combined:
                combined[h] = h
                lbg_added += 1
        print(f"  Byzantine headwords: +{lbg_added:,} gap-fill self-maps (lang=all)")

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
    if LBG_PAIRS_PATH.exists():
        with open(LBG_PAIRS_PATH, encoding="utf-8") as f:
            lbg_pairs = json.load(f)
        cf = {}
        if CORPUS_FREQ_PATH.exists():
            cf = json.load(open(CORPUS_FREQ_PATH, encoding="utf-8")).get(
                "forms", {})
        LBG_FREQ_CEILING = 20_000
        for form, lemma in lbg_pairs.items():
            if not form or form in combined:
                continue
            variants = {form, form.lower(), grave_to_acute(form),
                        to_monotonic(form.lower()), strip_accents(form.lower())}
            if any(v in combined and combined[v] != lemma for v in variants):
                continue  # (a) an existing form->lemma mapping wins
            fv = cf.get(strip_accents(form.lower()))
            if fv and fv[0] >= LBG_FREQ_CEILING:
                continue  # (b) ultra-common surface -> function-word collision
            combined[form] = lemma
            lbg_forms_added += 1
        print(f"  Byzantine inflected forms: +{lbg_forms_added:,} gated "
              f"gap-fill pairs (lang=all)")

    # NOTE: Corpus self-map and consensus overrides were tried here but
    # proved too aggressive, overriding correct Wiktionary entries with
    # corpus convention preferences (e.g. δεῖ -> δέομαι instead of δέω).
    # The targeted manual overrides below handle specific known issues.

    # Manual corrections for known lookup bugs.
    # These override wrong entries from Wiktionary or pipeline errors.
    _LOOKUP_OVERRIDES = {
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
        # Bare elided stems (apostrophe tokenized off by corpora): the only
        # possible reading is the elided function word, but the corpus pair
        # is a junk self-map or accent variant.
        "ἀλλ": "ἀλλά",           # was the self-map ἀλλ
        "μήτ": "μήτε",           # was the grave junk μὴτ
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
    }
    # Also fix corrupt -δήποτε lemmas from pipeline
    for k, v in list(combined.items()):
        if "δήποτε" in k and "δήποτε" in v and v != k:
            stem = k.split("δήποτε")[0]
            if stem:
                _LOOKUP_OVERRIDES[k] = "ὁστισδήποτε"
    override_count = 0
    for form, lemma in _LOOKUP_OVERRIDES.items():
        if form in combined and combined[form] != lemma:
            combined[form] = lemma
            override_count += 1
        # Also fix in ag dict for the grc-only table
        if form in ag and ag[form] != lemma:
            ag[form] = lemma
    if override_count:
        print(f"  Lookup overrides applied: {override_count:,}")

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
        FOREIGN KEY (lemma_id) REFERENCES lemmas(id)
    )""")

    # Track which source provided each combined entry
    src_map = {}
    for data, src_label in [(ag, 'grc'), (el, 'el')]:
        for k in data:
            if k not in src_map:
                src_map[k] = src_label
            elif _is_self_map(k, combined.get(k, '')) and not _is_self_map(k, data[k]):
                src_map[k] = src_label

    conn.executemany("INSERT INTO lookup (form, lemma_id, src, lang) VALUES (?, ?, ?, 'all')",
                     ((k, lemma_to_id[v], src_map.get(k, 'grc')) for k, v in combined.items()))
    print(f"  combined: {len(combined):,} rows")

    # AG-only entries where AG differs from combined (for polytonic-first lookup)
    ag_extra = 0
    for k, v in ag.items():
        if combined.get(k) != v and not _is_article_map(k, v):
            conn.execute("INSERT INTO lookup (form, lemma_id, src, lang) VALUES (?, ?, 'grc', 'grc')",
                         (k, lemma_to_id[v]))
            ag_extra += 1
    print(f"  grc-only (differs from combined): {ag_extra:,} rows")

    # MG-only entries where MG differs from combined (for lang="el" mode)
    mg_extra = 0
    for k, v in el.items():
        if combined.get(k) != v and not _is_article_map(k, v):
            conn.execute("INSERT INTO lookup (form, lemma_id, src, lang) VALUES (?, ?, 'el', 'el')",
                         (k, lemma_to_id[v]))
            mg_extra += 1
    print(f"  el-only (differs from combined): {mg_extra:,} rows")

    conn.execute("CREATE INDEX idx_lookup_form_lang ON lookup (form, lang)")
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

    # Compact main DB
    print("\nOptimizing lookup.db...")
    conn.commit()
    conn.close()

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

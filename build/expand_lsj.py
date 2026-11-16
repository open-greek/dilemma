#!/usr/bin/env python3
"""Expand LSJ headwords into inflected forms using Wiktionary Lua modules.

Uses wikitextprocessor to run Wiktionary's grc-decl and grc-conj templates
on LSJ headwords that don't have Wiktionary articles. The 14K+ overlap
between LSJ and Wiktionary serves as validation.

Phase 1: nouns (40K+ LSJ entries have gender from article in entry text)
Phase 2: adjectives
Phase 3: verbs

Usage:
    python expand_lsj.py --setup          # build Wiktionary module database (first run)
    python expand_lsj.py --test           # test on overlap entries
    python expand_lsj.py --expand         # expand LSJ-only entries
    python expand_lsj.py --test-one λύπη  # test a single word
"""

import argparse
import bisect
import hashlib
import json
import os
import re
import sqlite3
import sys
import unicodedata
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dilemma.form_sanitize import (  # noqa: E402
    has_editorial_sigla,
    resolve_editorial_form,
    sanitize_form,
)
from dilemma._attest_db import norm_key  # noqa: E402
from dilemma.grc_rejected_lemmas import REJECTED_LEMMAS  # noqa: E402
from lsj_principal_parts import (  # noqa: E402
    INDICATIVE_ONLY, INFERRED_FAMILIES, compound_parts, compound_simplexes,
    derive_grc_conj_args, ends_at_label, other_voice, parse_principal_parts,
    derived_cells, future_stems, participle_cases, regular_compound_parts,
    regular_parts, regular_rivals, short_verb_compound,
)

DATA_DIR = SCRIPT_DIR / "data"
from kaikki_dumps import DUMPS, env_dump_dir, resolve_dump  # noqa: E402

# LSJ9 exports: ``$DILEMMA_LSJ9_DIR`` (or ``--lsj9-dir``), else ~/Documents/lsj9.
LSJ9_DIR = Path(os.environ.get("DILEMMA_LSJ9_DIR")
                or Path.home() / "Documents" / "lsj9")
LSJ9_FORMS = LSJ9_DIR / "lsj9_forms.tsv"
LSJ9_HEADWORDS = LSJ9_DIR / "lsj9_headwords.json"
LSJ9_GLOSSES = LSJ9_DIR / "lsj9_glosses.jsonl"
# Earlier export of the same glosses, read only for principal parts.
LSJ9_GLOSSES_SUPPLEMENT = LSJ9_DIR / "lsj9_glosses_earlier.jsonl"
# kaikki dumps: ``--kaikki``, else ``$DILEMMA_KAIKKI_DIR`` (or the older
# ``$KAIKKI_DIR``), else the checkout's kaikki/; flat or nested layout.
KAIKKI_DIR = env_dump_dir() or SCRIPT_DIR / "kaikki"
KAIKKI_AG = resolve_dump(DUMPS["grc"]["en"], KAIKKI_DIR)
KAIKKI_AG_EL = resolve_dump(DUMPS["grc"]["el"], KAIKKI_DIR)


def configure_inputs(kaikki: Path | None = None, lsj9_dir: Path | None = None) -> None:
    """Point the dump and LSJ9 paths at other directories (the ``--kaikki``
    and ``--lsj9-dir`` options of this script and expand_sophocles.py)."""
    global KAIKKI_DIR, KAIKKI_AG, KAIKKI_AG_EL
    global LSJ9_DIR, LSJ9_FORMS, LSJ9_HEADWORDS, LSJ9_GLOSSES
    global LSJ9_GLOSSES_SUPPLEMENT
    if kaikki is not None:
        KAIKKI_DIR = Path(kaikki)
        KAIKKI_AG = resolve_dump(DUMPS["grc"]["en"], KAIKKI_DIR)
        KAIKKI_AG_EL = resolve_dump(DUMPS["grc"]["el"], KAIKKI_DIR)
    if lsj9_dir is not None:
        LSJ9_DIR = Path(lsj9_dir)
        LSJ9_FORMS = LSJ9_DIR / "lsj9_forms.tsv"
        LSJ9_HEADWORDS = LSJ9_DIR / "lsj9_headwords.json"
        LSJ9_GLOSSES = LSJ9_DIR / "lsj9_glosses.jsonl"
        LSJ9_GLOSSES_SUPPLEMENT = LSJ9_DIR / "lsj9_glosses_earlier.jsonl"


def add_input_arguments(parser) -> None:
    parser.add_argument("--kaikki", type=Path, default=None,
                        help="kaikki dump directory, flat or nested layout "
                             "(default: $DILEMMA_KAIKKI_DIR, else ./kaikki)")
    parser.add_argument("--lsj9-dir", type=Path, default=None,
                        help="LSJ9 export directory (default: $DILEMMA_LSJ9_DIR, "
                             "else ~/Documents/lsj9)")


WTP_DB = DATA_DIR / "wtp.db"

# --- Lua runtime compatibility for running Wiktionary's grc modules via wtp ---
# lupa's default runtime here is PUC Lua 5.5, which is too strict for the
# Wiktionary modules: its numeric for-loop variable is read-only (Module:scripts
# reassigns it) and string.char rejects floats. Lua 5.1/5.2 are too old for
# wtp's OWN sandbox (it uses `load(string)` and `\u` escapes). Lua 5.3 is the one
# version that satisfies both, together with the Module:string/char override
# below (which handles the float->int issue 5.3 still enforces). Force it.
try:
    import lupa.lua53 as _lua53
    import wikitextprocessor.luaexec as _wtp_luaexec
    _wtp_luaexec.LuaRuntime = _lua53.LuaRuntime
except Exception:
    pass

# Three idempotent patches to wtp's bundled sandbox Lua files (gaps that break
# the grc presentation modules). Applied at import; no-ops once applied.
try:
    import wikitextprocessor as _wtp_pkg
    _luadir = Path(_wtp_pkg.__file__).parent / "lua"

    # (a) mw_text.lua: mw.text.unstrip calls the misspelled `untripNoWiki`
    # (defined as `unstripNoWiki`), crashing the grc link path; and two per-call
    # debug print()s would flood a full expansion.
    _mt = _luadir / "mw_text.lua"
    _s = _mt.read_text(encoding="utf-8")
    _f = (_s.replace("mw.text.untripNoWiki", "mw.text.unstripNoWiki")
            .replace('   print("mw.text.unstripNoWiki called")\n', "")
            .replace('   print("mw.text.killMarkers called")\n', ""))
    if _f != _s:
        _mt.write_text(_f, encoding="utf-8")

    # (b) _sandbox_phase1.lua: expose `package` to modules (Module:load,
    # Module:require when needed, labels, ... read package.loaded/loaders).
    _p1 = _luadir / "_sandbox_phase1.lua"
    _s = _p1.read_text(encoding="utf-8")
    if 'env["package"]' not in _s:
        _anchor = '    env["require"] = _orig_new_require'
        _pkg = (_anchor + '\n    env["package"] = { loaded = _orig_package.loaded,'
                ' loaders = { [2] = function(m) return function() return'
                ' _orig_new_require(m) end end },'
                ' searchers = { [2] = function(m) return function() return'
                ' _orig_new_require(m) end end }, preload = {} }')
        if _anchor in _s:
            _p1.write_text(_s.replace(_anchor, _pkg, 1), encoding="utf-8")

    # (c) mw_title.lua: mw.title.new should accept a title object (it has
    # __tostring) rather than asserting it is a string.
    _mti = _luadir / "mw_title.lua"
    _s = _mti.read_text(encoding="utf-8")
    _old = '   assert(type(text) == "string")'
    _new = '   if type(text) ~= "string" then text = tostring(text) end'
    if _old in _s:
        _mti.write_text(_s.replace(_old, _new, 1), encoding="utf-8")
except Exception:
    pass

# The current Wiktionary Module:load caches loaders/data via Lua's
# `package.loaded` and `package.loaders[2]`, which wtp's module sandbox does not
# expose (modules see `package` as nil). That breaks every module that loads
# data through it (grc-decl -> grc-decl/table -> languages -> load_data), so the
# paradigm expansion silently produces zero forms. We override Module:load with
# a sandbox-compatible shim providing the same public API (safe_require,
# load_data, safe_load_data) backed by `require` + `mw.loadData` and plain-table
# caching. Loaded after the tarball modules so it wins.
_LOAD_MODULE_SHIM = """\
local export = {}
local require, loadData = require, mw.loadData
local req_cache, data_cache = {}, {}

function export.safe_require(modname)
\tlocal c = req_cache[modname]
\tif c ~= nil then if c == false then return nil end return c end
\tlocal ok, mod = pcall(require, modname)
\tif ok then req_cache[modname] = mod return mod end
\treq_cache[modname] = false
\treturn nil
end

function export.load_data(modname)
\tlocal c = data_cache[modname]
\tif c ~= nil and c ~= false then return c end
\tlocal data = loadData(modname)
\tdata_cache[modname] = data
\treturn data
end

function export.safe_load_data(modname)
\tlocal c = data_cache[modname]
\tif c ~= nil then if c == false then return nil end return c end
\tlocal ok, data = pcall(loadData, modname)
\tif ok then data_cache[modname] = data return data end
\tdata_cache[modname] = false
\treturn nil
end

return export
"""

# Replacement for Wiktionary's Module:string/char. The original manually
# UTF-8-encodes codepoints with float division and calls string.char on the
# float bytes, which Lua 5.3+ rejects. This returns the same thing (a function
# mapping codepoints -> UTF-8 string) using utf8.char with integer coercion.
_STRING_CHAR_SHIM = """\
local char, floor, unpack = string.char, math.floor, table.unpack
return function(...)
\tlocal n = select("#", ...)
\tif n == 0 then return end
\tlocal bytes, b = {}, 0
\tfor i = 1, n do
\t\tlocal cp = floor((select(i, ...)))  -- integer cp so // and % stay integer
\t\tif cp < 0x80 then
\t\t\tb = b + 1; bytes[b] = cp
\t\telseif cp < 0x800 then
\t\t\tbytes[b+1] = 0xC0 + cp // 0x40
\t\t\tbytes[b+2] = 0x80 + cp % 0x40
\t\t\tb = b + 2
\t\telseif cp < 0x10000 then
\t\t\tbytes[b+1] = 0xE0 + cp // 0x1000
\t\t\tbytes[b+2] = 0x80 + cp // 0x40 % 0x40
\t\t\tbytes[b+3] = 0x80 + cp % 0x40
\t\t\tb = b + 3
\t\telse
\t\t\tbytes[b+1] = 0xF0 + cp // 0x40000
\t\t\tbytes[b+2] = 0x80 + cp // 0x1000 % 0x40
\t\t\tbytes[b+3] = 0x80 + cp // 0x40 % 0x40
\t\t\tbytes[b+4] = 0x80 + cp % 0x40
\t\t\tb = b + 4
\t\tend
\tend
\treturn char(unpack(bytes, 1, b))
end
"""

# grc-decl computes the inflected forms, then wraps each in Module:links.full_link
# for display, which drags in the headword/title/maintenance-category machinery
# that wtp's sandbox only partially implements. We only need the forms, and
# full_link's `term` IS the form, so replace Module:links with a shim whose link
# functions return the bare term and whose every other function is a no-op.
_LINKS_SHIM = """\
local export = {}
local function term_of(data)
  if type(data) == "table" then return data.alt or data.term or "" end
  if data == nil then return "" end
  return tostring(data)
end
function export.full_link(data) return term_of(data) end
function export.language_link(data) return term_of(data) end
function export.plain_link(data) return term_of(data) end
return setmetatable(export, { __index = function() return function() return "" end end })
"""

# Pure-decoration modules that only add maintenance/hidden categories or other
# page chrome we don't want. Their page-context machinery (Module:pages page-type
# lookups, etc.) crashes in the synthetic sandbox, so replace each with a
# no-op: every function returns "".
_NOOP_MODULE_SHIM = """\
return setmetatable({}, { __index = function() return function() return "" end end })
"""
_NOOP_MODULES = ["Module:maintenance category", "Module:pages"]

# Article -> gender mapping
GENDER_MAP = {
    "ὁ": "m", "ἡ": "f", "τό": "n",
    "τά": "n", "οἱ": "m", "αἱ": "f",
    "ἁ": "f",  # Doric feminine article
}

# Nominative ending -> likely genitive ending (for regular nouns)
# Used when LSJ doesn't provide an explicit genitive
REGULAR_GENITIVE = {
    # 2nd declension
    ("ος", "m"): "ου",
    ("ος", "f"): "ου",
    ("ον", "n"): "ου",
    # 1st declension
    ("η", "f"): "ης",
    ("ᾱ", "f"): "ᾱς",
    ("α", "f"): "ας",  # short alpha (ambiguous - could be -ης)
    ("ά", "f"): "άς",
    ("ή", "f"): "ῆς",
    # 1st declension masculine
    ("ης", "m"): "ου",
    ("ᾱς", "m"): "ου",
    ("ας", "m"): "ου",
    # 3rd declension regular patterns
    ("μα", "n"): "ματος",
    ("ξ", "m"): "κος",
    ("ξ", "f"): "κος",
    ("ψ", "m"): "πος",
    ("ψ", "f"): "πος",
}

# 3rd declension patterns that need stem analysis (ending, gender) -> genitive suffix
# These replace the entire ending, not just append
THIRD_DECL_GENITIVE = {
    # -ις / -εως (πόλις type) vs -ις / -ιδος (ἐλπίς type)
    # Ambiguous: needs itype or Wiktionary cross-ref to disambiguate
    # -υς patterns
    ("ευς", "m"): "εως",       # βασιλεύς -> βασιλέως
    # -ων patterns
    ("ων", "m"): "ονος",       # λέων -> λέοντος is irregular, but δαίμων -> δαίμονος
    ("ων", "f"): "ονος",
    # -ηρ patterns
    ("ηρ", "m"): "ηρος",       # πατήρ -> πατρός is irregular, σωτήρ -> σωτῆρος
    ("ωρ", "m"): "ορος",       # ῥήτωρ -> ῥήτορος
    ("ωρ", "f"): "ορος",
    # -ης 3rd decl (proper nouns, Σωκράτης -> Σωκράτους)
    # Covered by itype usually
}


def strip_length_marks(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        ''.join(c for c in nfd if ord(c) not in (0x0306, 0x0304)))


def strip_diacritics(s: str) -> str:
    """Strip all combining diacritics for accent-free comparison."""
    nfd = unicodedata.normalize("NFD", s)
    return ''.join(c for c in nfd if not unicodedata.combining(c))


# Explicit LSJ abbreviation contracts: (nominative ending, supplied genitive
# suffix, number of nominative letters replaced). Never guess the replacement
# length from an accidental overlap. More specific endings precede shorter ones.
_GENITIVE_SUFFIX_RULES = (
    ("ευς", "εως", 3), ("ευς", "ηος", 3),
    ("πους", "ποδος", 4), ("πους", "οδος", 3),
    ("οδους", "οδοντος", 5), ("δους", "οντος", 3),
    ("θριξ", "τριχος", 4), ("παις", "παιδος", 4),
    ("χειρ", "χειρος", 4), ("κυων", "κυνος", 4),
    ("φρων", "φρονος", 4), ("ανηρ", "ανδρος", 4),
    ("ιγξ", "ιγγος", 3), ("υγξ", "υγγος", 3),
    ("μα", "ατος", 1),
    ("ος", "ου", 2), ("ον", "ου", 2),
    ("ης", "ου", 2), ("ας", "ου", 2),
    ("η", "ης", 1), ("α", "ας", 1), ("α", "ης", 1),
    ("ις", "εως", 2), ("υς", "εως", 2),
    ("ης", "ους", 2), ("ης", "ητος", 2),
    ("ως", "ωος", 2), ("ως", "ωτος", 2),
    ("ων", "ωνος", 2), ("ων", "ονος", 2), ("ων", "οντος", 2),
    ("ηρ", "ηρος", 2), ("ηρ", "ερος", 2),
    ("ωρ", "ορος", 2), ("ωρ", "ωρος", 2),
    ("ην", "ενος", 2), ("ην", "ηνος", 2),
    ("ις", "ιδος", 2), ("ις", "ιος", 2), ("ις", "ινος", 2),
    ("ις", "ιθος", 2), ("ις", "ιτος", 2),
    ("υς", "υος", 2), ("υς", "υδος", 2), ("υς", "υθος", 2),
    ("ας", "αδος", 2), ("ας", "αντος", 2), ("ας", "ατος", 2),
    ("αξ", "ακος", 2), ("αξ", "αγος", 2), ("αξ", "ακτος", 2),
    ("ιξ", "ικος", 2), ("ιξ", "ιγος", 2), ("ιξ", "ιχος", 2),
    ("υξ", "υγος", 2), ("υξ", "υκος", 2), ("υξ", "υχος", 2),
    ("ηξ", "ηγος", 2), ("ηξ", "ηκος", 2),
    ("ωξ", "ωγος", 2), ("ωξ", "ωκος", 2),
    ("ωψ", "ωπος", 2), ("οψ", "οπος", 2),
    ("ιψ", "ιβος", 2), ("αψ", "απος", 2),
    ("λι", "ιτος", 1), ("αν", "ανος", 2),
    ("αρ", "αρος", 2), ("αρ", "ατος", 2),
    ("ος", "εος", 2), ("ος", "ους", 2), ("ες", "εος", 2),
    ("ω", "ους", 1),
)


def _letter_clusters(word):
    clusters = []
    for char in unicodedata.normalize("NFD", word):
        if unicodedata.combining(char) and clusters:
            clusters[-1] += char
        else:
            clusters.append(char)
    return clusters


def _drop_letters(word, count):
    """Drop whole final Greek letters, retaining marks on the lexical prefix."""
    return unicodedata.normalize("NFC", "".join(_letter_clusters(word)[:-count]))


def build_genitive_from_itype(headword, itype):
    """Complete only explicit, supported LSJ genitive abbreviations."""
    hw_plain, it_plain = strip_diacritics(headword), strip_diacritics(itype)
    for nom_end, gen_suffix, count in _GENITIVE_SUFFIX_RULES:
        if hw_plain.endswith(nom_end) and it_plain == gen_suffix:
            prefix = _drop_letters(headword, count)
            if prefix and any(c in "\u0313\u0314" for c in unicodedata.normalize("NFD", itype)):
                # A breathing on an abbreviated suffix would become medial.
                # Preserve the source field for review; do not correct it here.
                return ""
            # Repeated letters retain their lexical diaeresis, breathing,
            # subscript, and length. Tone placement belongs to the module.
            ending = _letter_clusters(itype)
            for index, (old, new) in enumerate(zip(_letter_clusters(headword)[-count:], ending)):
                if old[0] != new[0]:
                    break
                ending[index] += "".join(c for c in old[1:]
                    if c not in "\u0300\u0301\u0342" and c not in new)
            itype = unicodedata.normalize("NFC", "".join(ending))
            # A supplied accent takes precedence over the nominative accent.
            # Lua then adjusts it for the generated case and syllable length.
            if any(c in "\u0300\u0301\u0342" for c in unicodedata.normalize("NFD", itype)):
                prefix = _unaccented_verb_stem(prefix)
            return unicodedata.normalize("NFC", prefix + itype)
    return ""


def parse_lsj_entries():
    """Parse LSJ entries from lsj9 exports.

    Priority: lsj9_forms.tsv (63K entries with explicit grammar) is loaded
    first, then lsj9_headwords.json fills in remaining entries (those without
    grammar in forms.tsv but present in the full headword list).
    """
    # Start with lsj9 forms data (entries with explicit grammar)
    entries = parse_lsj9_entries()

    # Fill in from headwords.json for entries not covered by forms.tsv
    hw_entries = _parse_lsj9_headwords()
    hw_only = 0
    gen_fills = 0
    for hw, entry in hw_entries.items():
        if hw not in entries:
            entries[hw] = entry
            hw_only += 1
        elif not entries[hw]["genitive"] and entry.get("genitive"):
            # forms.tsv entry exists but lacks genitive - fill from headwords
            entries[hw]["genitive"] = entry["genitive"]
            entries[hw]["itype"] = entry.get("itype", "")
            gen_fills += 1

    if hw_only:
        print(f"  headwords-only: {hw_only:,} additional entries")
    if gen_fills:
        print(f"  genitive fills from headwords: {gen_fills:,}")
    print(f"  Total: {len(entries):,} entries")

    return entries


def _parse_lsj9_headwords():
    """Load all LSJ entries from lsj9_headwords.json.

    Returns entries keyed by length-mark-stripped headword, with gender and
    genitive extracted from the structured JSON.
    """
    entries = {}
    if not LSJ9_HEADWORDS.exists():
        print(f"  lsj9_headwords.json not found at {LSJ9_HEADWORDS}")
        return entries

    with open(LSJ9_HEADWORDS, encoding="utf-8") as f:
        headwords_list = json.load(f)

    _grammar_to_info = {
        "ὁ": ("ὁ", "m"),
        "ἡ": ("ἡ", "f"),
        "τό": ("τό", "n"),
        "ον": ("ον", ""),
        "ές": ("ές", ""),
    }

    for e in headwords_list:
        hw_orig = e["headword"]
        hw = strip_length_marks(hw_orig)
        grammar = e.get("grammar", "")
        genitive = e.get("genitive", "")

        article, gender = _grammar_to_info.get(grammar, ("", ""))
        itype = genitive if grammar in ("ὁ", "ἡ", "τό") and genitive else ""

        if hw not in entries:
            entries[hw] = {
                "headword": hw,
                "orth_orig": hw_orig,
                "article": article,
                "gender": gender,
                "itype": itype,
                "genitive": genitive if gender else "",
            }

    print(f"  lsj9_headwords: {len(entries):,} entries")
    return entries


def parse_lsj9_entries(forms_path: Path | None = None) -> dict:
    """Load entries from lsj9_forms.tsv (an upstream LSJ9 export).

    This provides explicit grammar (ὁ/ἡ/τό/ον/ές) and pre-extracted
    genitive endings for 63K entries.

    Returns same format as parse_lsj_entries(): {headword: {headword,
    orth_orig, article, gender, itype, genitive}}.
    """
    forms_path = forms_path or LSJ9_FORMS
    if not forms_path.exists():
        print(f"  lsj9_forms.tsv not found at {forms_path}")
        return {}

    # Grammar -> (article, gender) mapping
    _grammar_to_article = {
        "ὁ": ("ὁ", "m"),
        "ἡ": ("ἡ", "f"),
        "τό": ("τό", "n"),
        "ον": ("ον", ""),    # adjective, no article/gender
        "ές": ("ές", ""),    # adjective, no article/gender
    }

    entries = {}
    with open(forms_path, encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            hw_raw, grammar, genitive, etymology = parts
            hw = strip_length_marks(hw_raw)

            article_info = _grammar_to_article.get(grammar, ("", ""))
            article, gender = article_info

            # For adjectives (ον/ές), try to infer genitive from pattern
            itype = ""
            if grammar == "ον":
                # 2-termination adjective: -ος/-ον type
                # genitive would be -ου (same as masculine)
                itype = "ον"
            elif grammar == "ές":
                # -ής/-ές type adjective
                itype = "ές"
            elif genitive:
                # Use the extracted genitive as itype
                itype = genitive

            if hw not in entries:
                entries[hw] = {
                    "headword": hw,
                    "orth_orig": hw_raw,
                    "article": article,
                    "gender": gender,
                    "itype": itype,
                    "genitive": genitive if grammar in ("ὁ", "ἡ", "τό") else "",
                }

    print(f"  lsj9: {len(entries):,} entries from {forms_path.name}")
    return entries


def load_wiktionary_forms():
    """Load form sets from Wiktionary kaikki for overlap validation."""
    wikt = {}
    if not KAIKKI_AG.exists():
        return wikt
    with open(KAIKKI_AG, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            word = e.get("word", "")
            if not word:
                continue
            forms = e.get("forms", [])
            real_forms = set()
            for fe in forms:
                tags = fe.get("tags", [])
                if "table-tags" in tags or "inflection-template" in tags:
                    continue
                form = strip_length_marks(fe.get("form", ""))
                if form and any('\u0370' <= c <= '\u03FF' or
                                '\u1F00' <= c <= '\u1FFF' for c in form):
                    real_forms.add(form)
            if word not in wikt or len(real_forms) > len(wikt[word]["forms"]):
                wikt[word] = {
                    "pos": e.get("pos", ""),
                    "forms": real_forms,
                }
    return wikt


def load_wiktionary_genitives():
    """Merge every exact genitive variant from both Wiktionary editions.

    Read the original extracts: training pairs also contain synthetic monotonic
    spellings. An omitted number is singular in these noun table exports;
    explicitly plural and dual cells are excluded. No variant wins by order.
    """
    genitives = {}
    for path in (KAIKKI_AG, KAIKKI_AG_EL):
        if not path.exists():
            raise FileNotFoundError(f"both Wiktionary genitive sources are required: {path}")
        with path.open(encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                if e.get("pos") not in ("noun", "adj", "name"):
                    continue
                word = strip_length_marks(e.get("word", ""))
                for fe in e.get("forms", []):
                    tags = set(fe.get("tags", []))
                    form = strip_length_marks(fe.get("form", ""))
                    if ("genitive" not in tags or tags & {"plural", "dual"}
                            or not form or _genitive_is_junk(form)):
                        continue
                    genitives.setdefault(word, set()).add(form)
    return {word: tuple(sorted(forms)) for word, forms in genitives.items()}


def noun_generation_input_hashes(include_lsj=True):
    """Pin the exact extraction and Lua inputs needed by a noun audit."""
    paths = {"wiktionary_en": KAIKKI_AG, "wiktionary_el": KAIKKI_AG_EL,
             "wtp": WTP_DB, "wtp_pickle": WTP_DB.with_suffix(".db.pickle")}
    if include_lsj:
        paths.update(lsj9_forms=LSJ9_FORMS, lsj9_headwords=LSJ9_HEADWORDS)
    hashes = {}
    for key, path in paths.items():
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            hashes[key] = digest.hexdigest()
    return hashes


def _genitive_variants(headword, genitives):
    value = (genitives or {}).get(strip_length_marks(headword), ())
    return (value,) if isinstance(value, str) else tuple(value)


def infer_genitive(headword, gender, wikt_genitives=None):
    """Infer genitive singular from nominative ending + gender.

    Priority:
    1. Wiktionary cross-reference (gold standard)
    2. Regular 1st/2nd declension patterns
    3. 3rd declension heuristics (lower confidence)
    """
    # Check Wiktionary cross-reference first
    headword = unicodedata.normalize("NFC", headword)
    variants = _genitive_variants(headword, wikt_genitives)
    if variants:
        return variants[0] if len(variants) == 1 else ""

    hw_plain = strip_diacritics(headword)

    # Regular patterns (1st/2nd declension - high confidence)
    for (ending, g), gen_ending in REGULAR_GENITIVE.items():
        if gender == g and hw_plain.endswith(ending):
            stem = headword[:-len(ending)]
            return stem + gen_ending

    # 3rd declension patterns (moderate confidence)
    for (ending, g), gen_ending in THIRD_DECL_GENITIVE.items():
        if gender == g and hw_plain.endswith(ending):
            stem = headword[:-len(ending)]
            return stem + gen_ending

    return ""


def setup_wtp():
    """Set up wikitextprocessor database from kaikki.org module/template tarballs.

    Downloads tarballs if needed, loads modules and templates into a SQLite db.
    Only needs to run once.
    """
    import tarfile
    import urllib.request
    from wikitextprocessor import Wtp

    print("Setting up wikitextprocessor database...")

    modules_url = "https://kaikki.org/dictionary/wiktionary-modules.tar.gz"
    templates_url = "https://kaikki.org/dictionary/wiktionary-templates.tar.gz"

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for url, name in [(modules_url, "wiktionary-modules"),
                      (templates_url, "wiktionary-templates")]:
        dest = DATA_DIR / f"{name}.tar.gz"
        if not dest.exists():
            print(f"  Downloading {name}...")
            urllib.request.urlretrieve(url, dest)
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  {size_mb:.1f} MB")

    if WTP_DB.exists():
        WTP_DB.unlink()

    wtp = Wtp(cache_file=str(WTP_DB))  # wtp >=0.4.x renamed db_path -> cache_file

    NS_MODULE = 828
    NS_TEMPLATE = 10

    def tar_to_title(member_name, namespace):
        """Convert tar path like 'Module/grc-decl.txt' to 'Module:grc-decl'."""
        name = member_name
        if name.startswith(f"{namespace}/"):
            name = name[len(f"{namespace}/"):]
        if name.endswith(".txt"):
            name = name[:-4]
        return f"{namespace}:{name}"

    for ns_name, ns_id, model in [("Module", NS_MODULE, "Scribunto"),
                                   ("Template", NS_TEMPLATE, "wikitext")]:
        tarball = DATA_DIR / f"wiktionary-{ns_name.lower()}s.tar.gz"
        print(f"  Loading {ns_name.lower()}s...")
        count = 0
        with tarfile.open(tarball, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                body = f.read().decode("utf-8", errors="replace")
                title = tar_to_title(member.name, ns_name)
                # wtp >=0.4.x: add_page(model, title, text) - no namespace_id arg
                wtp.add_page(model, title, body)
                count += 1
        print(f"  Loaded {count:,} {ns_name.lower()}s")

    # Override Wiktionary's Module:load (which needs Lua `package`, absent in the
    # wtp sandbox) with a sandbox-compatible shim. See _LOAD_MODULE_SHIM.
    wtp.add_page("Scribunto", "Module:load", _LOAD_MODULE_SHIM)
    wtp.add_page("Scribunto", "Module:string/char", _STRING_CHAR_SHIM)
    wtp.add_page("Scribunto", "Module:links", _LINKS_SHIM)
    for _m in _NOOP_MODULES:
        wtp.add_page("Scribunto", _m, _NOOP_MODULE_SHIM)
    print("  Patched Module:load + string/char + links + "
          f"{len(_NOOP_MODULES)} no-op decoration module(s) for the wtp sandbox.")

    # wtp >=0.4.x: analyze_templates() persists the template-transclusion index
    # to the cache file. Without it, a fresh Wtp opened from wtp.db (the separate
    # --expand process) cannot resolve any {{template}} and every paradigm
    # expands to an error, so the expansion silently adds ~0 forms.
    print("  Analyzing templates (persists transclusion index to cache)...")
    wtp.analyze_templates()

    # add_page() only INSERTs; the commit otherwise happens lazily in
    # close_db_conn(). Without committing here, a standalone --setup exits with
    # the transaction uncommitted and the WAL-mode db rolls back to empty.
    wtp.db_conn.commit()
    wtp.db_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    print("  Database ready.")
    return wtp


_WTP_INSTANCE = None


def get_wtp():
    """Get wikitextprocessor instance, reusing existing DB if available."""
    global _WTP_INSTANCE
    if _WTP_INSTANCE is not None:
        return _WTP_INSTANCE
    if WTP_DB.exists():
        from wikitextprocessor import Wtp
        print(f"Loading existing wtp database from {WTP_DB}...")
        _WTP_INSTANCE = Wtp(cache_file=str(WTP_DB))
    else:
        _WTP_INSTANCE = setup_wtp()
    return _WTP_INSTANCE


MACRON = "\u0304"  # combining macron
BREVE = "\u0306"   # combining breve

def mark_alpha_length(word):
    """Add macron to final alpha if it follows ι, ρ, or ε (long alpha rule).
    Add breve for short alpha patterns."""
    base = strip_diacritics(word)
    if not base.endswith("α") and not base.endswith("ας"):
        return word
    # Find the character before the alpha
    alpha_pos = len(base) - 1 if base.endswith("α") else len(base) - 2
    if alpha_pos < 1:
        return word
    preceding = base[alpha_pos - 1]
    if preceding in ("ι", "ρ", "ε"):
        # Long alpha after ι, ρ, ε - insert macron after the α
        # Find the actual α in the original word (may have accents)
        nfd = unicodedata.normalize("NFD", word)
        # Find the alpha at the right position and add macron after it
        result = []
        base_idx = 0
        for ch in nfd:
            if not unicodedata.combining(ch):
                base_idx += 1
            if base_idx == alpha_pos + 1 and ch == "α":
                result.append(ch)
                result.append(MACRON)
                base_idx_done = True
            else:
                result.append(ch)
        return unicodedata.normalize("NFC", "".join(result))
    return word


# Cache only identical lexical requests. Nouns with matching bare endings can
# have different accent and breathing patterns; another noun's forms cannot
# supply their marked spellings by replacing a stripped stem.
_NOUN_CACHE: dict[tuple[str, str, str], frozenset[str]] = {}


_GENITIVE_ARTICLES = {
    "ἡ", "ὁ", "τό", "τὸ", "τά", "τὰ", "τοῦ", "τῆς", "τῶν", "οἱ", "αἱ",
    "ἥ", "ὅ", "οὐ", "οὐκ", "οὐχ", "ὦ", "δέ", "δὲ", "καί", "καὶ",
}
_GREEK_GENITIVE = re.compile(r"^[Ͱ-Ͽἀ-῿̀-ͅ\-]+$")


def _genitive_is_junk(gen):
    """True if a parsed LSJ genitive is clearly not a usable genitive form
    (markup like '**', a dialect note such as 'Ion.', an article/particle,
    Latin, digits, internal whitespace, or too short). Such fields come from
    mis-parsing the LSJ entry; we infer the genitive from the nominative
    instead. A valid-but-irregular genitive is NOT junk and is left as-is, so we
    never substitute a wrong standard-rule paradigm for an irregular noun.
    """
    g = gen.strip()
    if len(g) < 2:
        return True
    if g in _GENITIVE_ARTICLES:
        return True
    if re.search(r"[\s*().,;:\[\]/0-9A-Za-z]", g):
        return True
    return not _GREEK_GENITIVE.match(g)


def resolve_noun_genitive(headword, gender, supplied="", wikt_genitives=None):
    """Resolve a full genitive without silently choosing a dialect/homograph.

    LSJ supplies both complete forms and abbreviated endings. The Lua module
    requires a complete form; its generic consonant declension otherwise
    mistakes an abbreviation for the entire stem.
    """
    headword = unicodedata.normalize("NFC", headword)
    supplied = unicodedata.normalize("NFC", supplied.strip())
    if any(c.isspace() or c in ",;" for c in headword):
        return "", "nonlexical-noun-headword"
    variants = _genitive_variants(headword, wikt_genitives)
    if not supplied or _genitive_is_junk(supplied):
        if len(variants) > 1:
            return "", "ambiguous-wiktionary-genitive"
        inferred = infer_genitive(headword, gender, wikt_genitives)
        return (inferred, "") if inferred else ("", "missing-genitive")

    # Independent exact agreement also proves irregular complete forms whose
    # oblique stem has no simple correspondence (for example Ζεύς / Διός).
    if supplied in variants:
        return supplied, ""
    hw_plain, gen_plain = strip_diacritics(headword), strip_diacritics(supplied)
    reconstructed = build_genitive_from_itype(headword, supplied)
    full_regular = any(
        hw_plain.endswith(nom_end) and gen_plain.endswith(gen_suffix)
        and hw_plain[:-count] + gen_suffix == gen_plain
        for nom_end, gen_suffix, count in _GENITIVE_SUFFIX_RULES
    )
    full_ending = (
        hw_plain.endswith(("ς", "ν", "ρ", "ξ", "ψ")) and gen_plain.endswith("ος")
        and not (hw_plain.endswith("ος") and not gen_plain.endswith("εος"))
    ) or (hw_plain.endswith("ω") and gender == "f" and gen_plain.endswith("ους"))
    if (not reconstructed and (full_regular or full_ending) and len(gen_plain) > 2
            and len(gen_plain) >= len(hw_plain) - 1
            and hw_plain[:2] == gen_plain[:2]
            and hw_plain.replace("-", "") != gen_plain.replace("-", "")):
        return supplied, ""

    if not reconstructed:
        return "", "unsupported-genitive-abbreviation"
    if variants:
        compatible = tuple(form for form in variants
                           if strip_diacritics(form).endswith(gen_plain)
                           and (not any(c in "\u0300\u0301\u0342" for c in unicodedata.normalize("NFD", supplied))
                                or strip_length_marks(form).endswith(strip_length_marks(supplied))))
        if len(compatible) != 1:
            return "", "conflicting-wiktionary-genitive"
        return compatible[0], ""
    return reconstructed, ""


def noun_expansion_error(headword, genitive):
    """Withhold evidenced irregular families unsupported by the generic route.

    A complete source genitive does not repair the module's irregular vocative
    and dative cells. This applies equally to full, inferred and completed input.
    """
    nom, gen = strip_diacritics(headword).lower(), strip_diacritics(genitive).lower()
    if ((nom.endswith("κυων") and gen.endswith(("κυνος", "κυονος")))
            or (nom.endswith("ανηρ") and gen.endswith(("ανδρος", "ανηρος")))
            or (nom.endswith("πατηρ") and gen.endswith("πατρος"))):
        return "unsupported-irregular-noun"
    return ""


def expand_noun(wtp, headword, gender, genitive="", wikt_genitives=None):
    """Expand a noun using grc-decl, caching identical lexical requests."""
    headword = unicodedata.normalize("NFC", headword)
    genitive, error = resolve_noun_genitive(headword, gender, genitive, wikt_genitives)
    error = error or noun_expansion_error(headword, genitive)
    if error:
        return set(), error

    # Include the resolved genitive: explicit and inferred identical requests
    # share a result, while differently marked lexical inputs remain distinct.
    cache_key = (headword, gender, genitive)
    if cache_key in _NOUN_CACHE:
        return set(_NOUN_CACHE[cache_key]), ""

    # grc-decl gender codes
    gender_code = {"m": "M", "f": "F", "n": "N"}.get(gender, "")

    # Mark alpha length for disambiguation
    hw_marked = mark_alpha_length(headword)
    gen_marked = mark_alpha_length(genitive) if genitive else ""

    parts = [hw_marked]
    if gen_marked:
        parts.append(gen_marked)
    else:
        parts.append("")

    form_param = f"|form={gender_code}" if gender_code else ""
    template = "{{grc-decl|" + "|".join(parts) + form_param + "}}"

    try:
        wtp.start_page(headword)
        html = wtp.expand(template)
    except Exception as e:
        return set(), str(e)

    # The table's notes describe historical clusters such as *-δς. They are
    # not paradigm cells and must never become dictionary forms.
    lines = []
    in_notes = False
    for line in html.splitlines(keepends=True):
        if line.startswith(('! class="notes-header"', '| class="notes"')):
            in_notes = True
        if in_notes and line.startswith(("|-", "|}")):
            in_notes = False
        if not in_notes:
            lines.append(line)
    forms = parse_html_forms("".join(lines), headword)

    if forms:
        _NOUN_CACHE[cache_key] = frozenset(forms)

    return forms, "" if forms else "empty-noun-paradigm"


# Present-system cache for identical lexical requests, with all marks intact.
_VERB_CACHE: dict[tuple[str, str, str, str], frozenset[str]] = {}


# Cache: principal-parts grc-conj-args tuple -> set of forms.
# Keyed on (tense_code, *stems) so verbs that share a passive stem (e.g.
# all -ευω verbs producing "-ευθ" passives) reuse the expansion.
_PP_CACHE: dict[tuple, set[str]] = {}


_LSJ_HEAD_TEXTS: dict[str, str] | None = None
_LSJ_SUPPLEMENT_HEAD_TEXTS: dict[str, str] | None = None


LSJ9_GLOSSES_PIN = DATA_DIR / "lsj9_glosses_pin.json"
SUPPLEMENT_PIN_KEY = "principal_parts_supplement"


def _verify_pinned_export(path: Path, pin: dict[str, object]) -> dict[str, object]:
    expected = {"sha256": pin["sha256"], "size": pin["size"]}
    if not path.is_file():
        raise SystemExit(f"{path.name} not found in {path.parent}; the verb "
                         "pass needs the pinned LSJ9 glosses export "
                         f"({LSJ9_GLOSSES_PIN.name})")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    actual = {"sha256": digest.hexdigest(), "size": path.stat().st_size}
    if actual != expected:
        raise SystemExit(f"{path} does not match {LSJ9_GLOSSES_PIN.name} "
                         f"(sha256 {actual['sha256'][:12]}, {actual['size']:,} bytes; "
                         f"pinned {expected['sha256'][:12]}, {expected['size']:,})")
    return actual


def verify_lsj_glosses() -> dict[str, object]:
    """Check the LSJ9 glosses exports against their pins, or stop.

    The verb pass reads principal parts and its entry rejections from each
    headword's LSJ text, so another export (or none) silently changes
    ag_lookup.json: an export without the ἡμι- entry's text once turned 38
    bare endings into forms of the prefix. data/lsj9_glosses_pin.json
    records the export's sha256 and size, and the same for the earlier
    export that supplies principal parts the current text cuts off.
    """
    pin = json.loads(LSJ9_GLOSSES_PIN.read_text(encoding="utf-8"))
    actual = _verify_pinned_export(LSJ9_GLOSSES, pin)
    supplement = pin.get(SUPPLEMENT_PIN_KEY)
    if supplement:
        actual["supplement"] = _verify_pinned_export(
            LSJ9_GLOSSES_SUPPLEMENT, supplement)
    return actual


def _read_head_texts(path: Path) -> dict[str, str]:
    heads: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            hw = e.get("headword")
            if not hw:
                continue
            if "level" in e or "number" in e:
                continue
            if hw not in heads:
                heads[hw] = e.get("text", "")
    return heads


def load_lsj_head_texts() -> dict[str, str]:
    """Load the leading paragraph of every LSJ entry (gloss without
    `level` / `number` is the entry head, which carries the
    principal-part header before the English definition starts).

    Cached after first call.
    """
    global _LSJ_HEAD_TEXTS
    if _LSJ_HEAD_TEXTS is not None:
        return _LSJ_HEAD_TEXTS
    heads: dict[str, str] = {}
    if not LSJ9_GLOSSES.exists():
        print(f"  lsj9_glosses.jsonl not found at {LSJ9_GLOSSES}")
        _LSJ_HEAD_TEXTS = heads
        return heads
    heads = _read_head_texts(LSJ9_GLOSSES)
    print(f"  lsj9 head texts: {len(heads):,}")
    _LSJ_HEAD_TEXTS = heads
    return heads


def _without_iota_subscript(s: str) -> str:
    return unicodedata.normalize("NFC", "".join(
        c for c in unicodedata.normalize("NFD", s) if ord(c) != 0x0345))


def head_text_finder(head_texts: dict[str, str]):
    """A function from a headword to its entry head, "" when it has none.

    The headword list drops quantity marks, and prints θρώσκω where the
    entry is θρῴσκω, while the glosses keep the entry's own spelling
    (ἀγῑνέω), so the exact spelling misses the text of some 6,900 verb
    heads. The exact spelling comes first; then the one entry that has the
    same letters without quantity marks, then the one that has them without
    iota subscript either. A spelling two entries share (θᾱλέω, θᾰλέω)
    stays without text, since it does not say which entry is meant.
    """
    by_quantity: dict[str, list[str]] = {}
    by_subscript: dict[str, list[str]] = {}
    for key in head_texts:
        plain = strip_length_marks(key)
        by_quantity.setdefault(plain, []).append(key)
        by_subscript.setdefault(_without_iota_subscript(plain), []).append(key)

    def find(headword: str) -> str:
        if headword in head_texts:
            return head_texts[headword]
        plain = strip_length_marks(headword)
        for index, folded in ((by_quantity, plain),
                              (by_subscript, _without_iota_subscript(plain))):
            keys = index.get(folded)
            if keys:
                return head_texts[keys[0]] if len(keys) == 1 else ""
        return ""

    return find


def load_lsj_supplement_head_texts() -> dict[str, str]:
    """Entry heads of the earlier glosses export named in the pin.

    The current export ends many verb entries' head text at the first
    citation (κατατέμνω: "... fut. -τεμῶ Ar.Ach.301: aor."), before the
    remaining principal parts; the earlier export keeps them and drops the
    citations. Empty when the pin names no supplement or the file is absent.
    """
    global _LSJ_SUPPLEMENT_HEAD_TEXTS
    if _LSJ_SUPPLEMENT_HEAD_TEXTS is not None:
        return _LSJ_SUPPLEMENT_HEAD_TEXTS
    heads: dict[str, str] = {}
    pin = json.loads(LSJ9_GLOSSES_PIN.read_text(encoding="utf-8"))
    if pin.get(SUPPLEMENT_PIN_KEY) and LSJ9_GLOSSES_SUPPLEMENT.exists():
        heads = _read_head_texts(LSJ9_GLOSSES_SUPPLEMENT)
        print(f"  lsj9 supplement head texts: {len(heads):,}")
    _LSJ_SUPPLEMENT_HEAD_TEXTS = heads
    return heads


# The three aorist slots are read together (an unnumbered aorist stands for
# the first or the second depending on which numbered one is present), so
# the supplement fills them only as a group.
_AORIST_PART_KEYS = frozenset({"aor", "aor1", "aor2"})


def merged_principal_parts(headword: str, head_text: str,
                           supplement_text: str = "") -> dict[str, str]:
    """Principal parts of the entry text, completed from the supplement.

    Every part the current text gives stands. The earlier export adds only
    the tenses the current text lacks. Where the current text stops right
    after a label (ἀνοιδέω: "... fut. -ήσω: aor."), the earlier one opens
    with the form that label announces ("ἀνῴδησα , e: pf. ἀνῴδηκα"), so the
    two are read as one text. The aorists of the earlier text are taken only
    as a group: when the current text has none, or when the earlier text
    repeats every one the current text gives, which then stopped among them
    (ἀναθάλλω: "aor. ἀνέθηλα ...: aor. 2" beside "aor. ἀνέθηλα : aor. 2
    ἀνέθαλον").
    """
    parts = dict(parse_principal_parts(head_text, headword)) if head_text else {}
    if not supplement_text:
        return parts
    if (head_text and ends_at_label(head_text)
            and re.match(r"\s*[\u0370-\u03ff\u1f00-\u1fff]", supplement_text)):
        supplement_text = head_text.rstrip() + " " + supplement_text
    supplement = parse_principal_parts(supplement_text, headword)
    aorists = {key: form for key, form in parts.items()
               if key in _AORIST_PART_KEYS}
    # "aor" stands for the numbered aorist where no plain label is printed,
    # so it is compared only when it is the one aorist given.
    printed = {key: form for key, form in aorists.items()
               if key != "aor"} or aorists
    continued = bool(printed) and all(
        supplement.get(key) == form for key, form in printed.items())
    for key, form in supplement.items():
        if key in _AORIST_PART_KEYS and aorists and not continued:
            continue
        parts.setdefault(key, form)
    return parts


def _perfect_participle_cells(table: str) -> str:
    """Keep the participle token from the pinned Lua periphrastic cells.

    In the perfect middle subjunctive/optative, the module incorrectly
    prefixes the lexical stem to the auxiliary. The first token in each
    affected table cell is the correctly generated participle. Retain it
    without inferring another form, and omit the malformed auxiliary and
    its alternatives. Other table cells and arbitrary Greek text are intact.
    """
    middle = False
    periphrastic = False
    output = []
    for line in table.splitlines(keepends=True):
        if line.startswith('! rowspan="4" | '):
            middle = "middle/" in line
            periphrastic = False
        elif line.startswith("! ") and not line.startswith("! rowspan"):
            periphrastic = middle and line.strip() in {"! subjunctive", "! optative"}
        prefix = '| class="grc-conj-finite-form" | '
        if periphrastic and line.startswith(prefix):
            content = line[len(prefix):]
            first_token = content.split()[0] if content.split() else ""
            # These are the actual cell tokens, not generated inflections.
            # Fail closed if the pinned table no longer supplies one.
            plain = strip_diacritics(first_token)
            participle = first_token if plain.endswith(("μενος", "μενοι", "μενω")) else ""
            line = prefix + participle + ("\n" if line.endswith("\n") else "")
        output.append(line)
    return "".join(output)


def _indicative_cells(table: str) -> str:
    """Keep the indicative rows of a conjugation table and nothing else.

    Used for a past tense whose unaugmented stem is unknown: the module
    builds the other moods, the infinitive and the participle on the stem
    without the augment, and would print them with it.
    """
    indicative = False
    output = []
    for line in table.splitlines(keepends=True):
        if line.startswith("! ") and not line.startswith(
                ("! rowspan", "! colspan", "! class")):
            indicative = line.strip() == "! indicative"
        if line.startswith("|") and "grc-conj-nonfinite-form" in line:
            continue
        if line.startswith('| class="grc-conj-finite-form"') and not indicative:
            continue
        output.append(line)
    return "".join(output)


def _expand_principal_part(wtp, headword: str, tense_args: list[str]
                            ) -> set[str]:
    """Expand a single ``{{grc-conj|...}}`` invocation derived from
    LSJ principal parts, returning the resulting set of forms.

    Caches the result keyed on the argument tuple so verbs that share
    e.g. a passive stem don't pay the Lua cost twice.
    """
    cache_key = tuple(tense_args)
    if cache_key in _PP_CACHE:
        return _PP_CACHE[cache_key]

    # Build the {{grc-conj|...}} call.
    indicative_only = INDICATIVE_ONLY in tense_args
    tpl = "{{grc-conj|" + "|".join(
        arg for arg in tense_args if arg != INDICATIVE_ONLY) + "}}"
    try:
        wtp.start_page(headword)
        html = wtp.expand(tpl)
    except Exception:
        _PP_CACHE[cache_key] = set()
        return set()
    if tense_args[0] == "perf":
        html = _perfect_participle_cells(html)
    if indicative_only:
        html = _indicative_cells(html)
    forms = parse_html_forms(html, headword)
    _PP_CACHE[cache_key] = forms
    return forms


def expand_principal_parts(wtp, headword: str, head_text: str,
                           supplement_text: str = "") -> set[str]:
    """Extract LSJ principal parts and expand each into a paradigm.

    Returns the union of forms produced by the per-tense grc-conj
    invocations. Returns an empty set if no parts could be parsed
    (caller should fall back to present-only expansion).
    """
    parts = merged_principal_parts(headword, head_text, supplement_text)
    if not parts:
        return set()
    args_by_tense = derive_grc_conj_args(parts, headword)
    if not args_by_tense:
        return set()
    forms: set[str] = set()
    for tense_args in args_by_tense.values():
        forms |= _expand_principal_part(wtp, headword, tense_args)
    return forms


def derived_principal_part_cells(wtp, headword: str, head_text: str,
                                 supplement_text: str = "") -> set[str]:
    """The principal-part cells of an entry that stand only on a stem the
    entry does not print (:func:`derived_cells`): καταδέξας beside the
    printed κατέδεξα, not κατέδεξε, and not a cell that a printed stem
    gives as well (λύσω of the aorist subjunctive and of the future)."""
    headword = unicodedata.normalize("NFC", headword)
    parts = merged_principal_parts(headword, head_text, supplement_text)
    derived: set[str] = set()
    printed: set[str] = set()
    calls = derive_grc_conj_args(parts, headword)
    printed_stems = future_stems(calls)
    for tense, tense_args in calls.items():
        cells = _expand_principal_part(wtp, headword, tense_args)
        on_derived = derived_cells(tense, tense_args, cells, printed_stems)
        derived |= on_derived
        printed |= cells - on_derived
    return derived - printed


# Common verbal preverb endings (preposition-prefixes) used in compound -μι
# verbs. We list the diacritic-stripped, lowercased forms only. The
# splitter below matches the longest known prefix that leaves at least the
# expected base ending (e.g. "ειμι"). Composite prefixes are constructed
# automatically from the building blocks below; we don't enumerate every
# attested combination by hand.
_AG_PREVERB_ATOMS = (
    # Two-vowel atoms first so longest-match works at the atom level
    "παρα", "περι", "ἀνα", "δια", "κατα", "μετα", "ἀπο", "ὑπο", "ὑπερ",
    "ἀντι", "ἐπι", "προ", "προσ", "ἐκ", "ἐξ", "ἐν", "ἐμ", "εἰσ", "συμ", "συν",
    "συγ", "ἀν", "ἀπ", "ἀφ", "δι", "ἐπ", "ἐφ", "καθ", "κατ", "μετ",
    "μεθ", "ἀνθ", "ἀντ", "παρ", "ὑπ", "ὑφ", "ὑπε", "ὑπεκ", "ὑπεξ",
    "ὑπεισ", "ἀνταπο", "ἀντεκ", "ἀντεξ", "ἀντεπι", "ἀντεπε", "ἀντεπεισ",
    "ἀντεπαπο", "ἀντιπαρα", "ἀντιπερι", "ἀντιπρο", "ἀντιπροσ", "ἀντικα",
    "ἀντικαθ", "ἀντιμε", "ἀντιδια", "ἀνθυπο", "ἀνθυπ",
    "ἐπανα", "ἐπαν", "ἐπικα", "ἐπικατα", "ἐπιπαρα", "ἐπιπαρ",
    "ἐπιπροσ", "ἐπισυν", "ἐπισυμπαρ",
    "παραπρο", "παρακα", "παρακατα",
    "προαπο", "προεκ", "προεξ", "προεπι", "προεν", "προε",
    "προπαρ", "προπερι", "προσπαρ", "προσυν", "προσυπ", "προυπ",
    "συγκα", "συγκατα", "συμπαρα", "συμπερι", "συμπροσ", "συμπρο",
    "συναπο", "συνδια", "συνδιεξ", "συνδιολ", "συνεισ", "συνεξ",
    "συνεπει", "συνεπι", "συνεπε", "συνπαρα", "συνπρο", "συνυπ",
    "ὑπεραπο", "ὑπερεν", "ὑποκατ", "ὑποπαρα", "ὑποπερι", "ὑποπρο",
    "ἐκπερι", "εἰσανα", "εἰσαν",
    "ἐναπο", "ἐνδια",
)

# Recognised base-stems for the irregular dispatcher. The splitter's job
# is to find the longest preverb such that what's left starts with one
# of these (diacritic-stripped, lowercased) bases.
_IRREGULAR_BASE_STEMS = ("ειμι", "οιδα", "χρη", "φημι")


def _strip_diacritics_lower(s):
    return strip_diacritics(s).lower()


def _split_preverb(headword):
    """Split a headword into (preverb, base) by stripping a recognised
    irregular base ending (e.g. ``ειμι``).

    The headword's diacritic-stripped suffix must be one of
    ``_IRREGULAR_BASE_STEMS``. The portion before that suffix is the
    preverb. Returns (preverb_orig, base_orig) preserving the original
    diacritics. If no base matches, returns ("", headword).
    """
    hw_plain = _strip_diacritics_lower(headword)
    for base in sorted(_IRREGULAR_BASE_STEMS, key=len, reverse=True):
        if hw_plain.endswith(base):
            n = len(hw_plain) - len(base)
            if n == 0:
                # Bare base headword - no preverb
                return "", headword
            # Validate preverb consists of recognised atoms (in any
            # combination, longest-first). A non-recognised prefix
            # (e.g. random OCR junk) is rejected so we don't blindly
            # generate forms.
            preverb_plain = hw_plain[:n]
            if _is_valid_preverb(preverb_plain):
                return headword[:n], headword[n:]
            # Otherwise fall through and try a shorter base
    return "", headword


_AG_PREVERB_ATOMS_PLAIN = None


def _get_preverb_atoms_plain():
    global _AG_PREVERB_ATOMS_PLAIN
    if _AG_PREVERB_ATOMS_PLAIN is None:
        _AG_PREVERB_ATOMS_PLAIN = sorted(
            {strip_diacritics(a).lower() for a in _AG_PREVERB_ATOMS},
            key=len, reverse=True)
    return _AG_PREVERB_ATOMS_PLAIN


def _is_valid_preverb(preverb_plain):
    """Check that a (lowercased, diacritic-stripped) preverb decomposes
    into recognised _AG_PREVERB_ATOMS via *backtracking* longest-match.

    Greedy longest-first match fails on cases like ``αντεπεξ`` where the
    longest atom (``αντεπε``) leaves a residue ``ξ`` that no atom
    matches; the right segmentation is ``αντ + επ + εξ``.
    """
    atoms = _get_preverb_atoms_plain()

    def _try(s):
        if not s:
            return True
        for a in atoms:
            if s.startswith(a) and _try(s[len(a):]):
                return True
        return False

    return _try(preverb_plain)


# A handful of irregular AG verbs whose paradigms cannot be generated by the
# Wiktionary `Module:grc-conj` rule-based dispatch (they map to
# `pres-irreg` on Wiktionary, with every form filled in by hand). We list a
# minimal but useful Attic core paradigm for each exact literal base. Compound
# accent, augment, and lexical-family rules need independent evidence and are
# deliberately not generated by prepending a stem.
#
# Each value is a set of bare Attic forms (without preverb prefix). We
# deliberately keep this list small: only the forms that show up in real
# corpora and that the lookup table needs in order to map them back to a
# headword. Augmented past-tense forms include the augmented stem (so a
# compound prepends *to* the augmented form, e.g. παρ + ῆν -> παρῆν).
_IRREGULAR_BASE_FORMS = {
    # εἰμί 'to be' (Attic core: present, imperfect, future, optative,
    # subjunctive, imperative, infinitive, participle)
    "εἰμί": {
        # Present indicative active
        "εἰμί", "εἶ", "ἐστί", "ἐστίν", "ἐσμέν", "ἐστέ", "εἰσί", "εἰσίν",
        "ἐστόν",
        # Subjunctive
        "ὦ", "ᾖς", "ᾖ", "ὦμεν", "ἦτε", "ὦσι", "ὦσιν", "ἦτον",
        # Optative
        "εἴην", "εἴης", "εἴη", "εἶμεν", "εἶτε", "εἶεν", "εἴητε",
        "εἴημεν", "εἴησαν", "εἴητον", "εἰήτην",
        # Imperative
        "ἴσθι", "ἔστω", "ἔστε", "ἔστων", "ἔστωσαν", "ἤτω", "ὄντων",
        # Imperfect (augmented)
        "ἦν", "ἦσθα", "ἦς", "ἦμεν", "ἦτε", "ἦσαν", "ἦστον", "ἤστην",
        # Future
        "ἔσομαι", "ἔσῃ", "ἔσει", "ἔσται", "ἐσόμεθα", "ἔσεσθε",
        "ἔσονται", "ἔσεσθαι",
        # Infinitive
        "εἶναι",
        # Participles (m/f/n nom sg)
        "ὤν", "οὖσα", "ὄν",
        "ὄντος", "οὔσης", "ὄντι", "οὔσῃ", "ὄντα", "οὖσαν",
        "ὄντες", "οὖσαι", "ὄντα", "ὄντων", "οὐσῶν", "οὖσι", "οὔσαις",
    },
    # εἶμι 'to go' (Attic core)
    "εἶμι": {
        # Present indicative active
        "εἶμι", "εἶ", "εἶσι", "εἶσιν", "ἴμεν", "ἴτε", "ἴᾱσι", "ἴᾱσιν",
        "ἴτον",
        # Subjunctive
        "ἴω", "ἴῃς", "ἴῃ", "ἴωμεν", "ἴητε", "ἴωσι", "ἴωσιν",
        # Optative
        "ἴοιμι", "ἴοις", "ἴοι", "ἴοιμεν", "ἴοιτε", "ἴοιεν", "ἰοίην",
        "ἴοιτον", "ἰοίτην",
        # Imperative
        "ἴθι", "ἴτω", "ἴτε", "ἴτων", "ἴτωσαν", "ἰόντων",
        # Imperfect (augmented)
        "ᾖα", "ᾔειν", "ᾔεις", "ᾔει", "ᾔειν", "ᾖμεν", "ᾔειμεν",
        "ᾖτε", "ᾔειτε", "ᾖσαν", "ᾔεσαν", "ᾔειτον", "ᾐείτην",
        # Infinitive
        "ἰέναι", "ἴναι",
        # Participles
        "ἰών", "ἰοῦσα", "ἰόν",
        "ἰόντος", "ἰούσης", "ἰόντι", "ἰούσῃ", "ἰόντα", "ἰοῦσαν",
        "ἰόντες", "ἰοῦσαι", "ἰόντα", "ἰόντων", "ἰουσῶν", "ἰοῦσι",
    },
    # οἶδα 'know' (perfect-with-present meaning)
    "οἶδα": {
        # Indicative
        "οἶδα", "οἶσθα", "οἶδε", "οἶδεν", "ἴσμεν", "ἴστε", "ἴσασι",
        "ἴσασιν", "ἴστον",
        # Subjunctive
        "εἰδῶ", "εἰδῇς", "εἰδῇ", "εἰδῶμεν", "εἰδῆτε", "εἰδῶσι",
        "εἰδῶσιν",
        # Optative
        "εἰδείην", "εἰδείης", "εἰδείη", "εἰδεῖμεν", "εἰδεῖτε",
        "εἰδεῖεν", "εἰδείημεν", "εἰδείητε", "εἰδείησαν",
        # Imperative
        "ἴσθι", "ἴστω", "ἴστε", "ἴστων", "ἴστωσαν",
        # Pluperfect (augmented)
        "ᾔδη", "ᾔδειν", "ᾔδεις", "ᾔδεισθα", "ᾔδει", "ᾔδειν",
        "ᾔδεμεν", "ᾔδειμεν", "ᾔδετε", "ᾔδειτε", "ᾔδεσαν", "ᾔδεισαν",
        # Future (εἴσομαι)
        "εἴσομαι", "εἴσῃ", "εἴσει", "εἴσεται", "εἰσόμεθα", "εἴσεσθε",
        "εἴσονται",
        # Infinitive
        "εἰδέναι",
        # Participle
        "εἰδώς", "εἰδυῖα", "εἰδός",
        "εἰδότος", "εἰδυίας", "εἰδότι", "εἰδυίᾳ", "εἰδότα",
        "εἰδυῖαν", "εἰδότες", "εἰδυῖαι", "εἰδότων", "εἰδυιῶν",
        "εἰδόσι",
    },
    # χρή 'it is necessary' (defective; only 3sg-style forms)
    "χρή": {
        "χρή", "χρῇ", "χρῆν", "χρῆναι", "χρεών", "χρῆται",
        "χρῷη", "χρείη",
    },
    # φημί 'say' requires this explicit paradigm. The regular -ημι
    # template produces φῆμι, and compound accent/augment rules are not
    # supported by the preverb joiner.
    "φημί": {
        # Indicative
        "φημί", "φῄς", "φησί", "φησίν", "φαμέν", "φατέ", "φᾱσί",
        "φᾱσίν", "φατόν",
        # Subjunctive
        "φῶ", "φῇς", "φῇ", "φῶμεν", "φῆτε", "φῶσι", "φῶσιν",
        # Optative
        "φαίην", "φαίης", "φαίη", "φαῖμεν", "φαῖτε", "φαῖεν",
        # Imperative
        "φαθί", "φάθι", "φάτω", "φάτε", "φάντων", "φάτωσαν",
        # Imperfect (no preverb augment to apply)
        "ἔφην", "ἔφης", "ἔφη", "ἔφαμεν", "ἔφατε", "ἔφασαν",
        # Infinitive / participle
        "φάναι", "φάς", "φᾶσα", "φάν",
    },
}


def _expand_irregular_compound(headword):
    """Return only a reviewed literal irregular base paradigm.

    The former prefix joiner lost iota subscripts, misplaced accents, and
    combined the distinct εἰμί and εἶμι paradigms. A compound needs its own
    accent, augment, and lexical-family evidence; it must not inherit an
    approximate union. Independent source mappings remain available.
    """
    headword = unicodedata.normalize("NFC", headword)
    base_forms = _IRREGULAR_BASE_FORMS.get(headword)
    if base_forms is None:
        return set()
    return set(base_forms) | {strip_diacritics(form) for form in base_forms}


# Trivial OCR-corruption guard. Drop any headword that contains a combining
# underdot / diaeresis-below mark (often inserted in scanned papyrus
# editions) or that has an obviously ill-formed consonant cluster
# (e.g. "ννν", "λλλ", "μμμ"). These are not real Greek words and they
# routinely trigger Lua errors when fed to grc-conj.
_OCR_BAD_COMBINING = {0x0323, 0x0324, 0x032E, 0x0325}
_BAD_CONSONANT_CLUSTERS = ("ννν", "λλλ", "μμμ", "ρρρ", "σσσ")


def _is_corrupt_headword(headword):
    if not headword or len(headword) < 3:
        return True
    nfd = unicodedata.normalize("NFD", headword)
    for ch in nfd:
        if ord(ch) in _OCR_BAD_COMBINING:
            return True
    plain = strip_diacritics(headword.lower())
    for c in _BAD_CONSONANT_CLUSTERS:
        if c in plain:
            return True
    # Guard against headwords containing whitespace or commas
    # (e.g. 'προεῖναι, πρόειμι', 'πρόσειμι εἰμί') which are LSJ
    # cross-references mis-extracted as headwords.
    if any(ch in headword for ch in " ,;"):
        return True
    return False


def _unaccented_verb_stem(stem):
    """Remove tone accents while retaining lexical and vowel-length marks.

    The conjugation module places the accent, but still needs breathings,
    diaeresis, iota subscript, macrons, and breves from the lexical stem.
    """
    return unicodedata.normalize("NFC", "".join(
        char for char in unicodedata.normalize("NFD", stem)
        if char not in {"\u0300", "\u0301", "\u0342"}
    ))


def _classify_verb(headword):
    """Classify a verb into conjugation type and stem. Returns (conj_type, stem) or (None, None).

    The returned ``conj_type`` is the parameter-1 string that
    ``Module:grc-conj`` understands. We support identifiable rule-based
    dispatches the Lua module exposes (pres-con-{a,e,o},
    pres-{ami,emi,omi,numi,lumi}, plain pres). For genuinely
    irregular athematic verbs whose Wiktionary entries use
    ``pres-irreg`` (εἰμί, εἶμι, οἶδα, χρή, and their compounds),
    we return the sentinel ``conj_type='irreg'`` so the caller can
    dispatch to ``_expand_irregular_compound`` instead of Lua. Only exact
    reviewed base paradigms are supported; compounds are withheld.
    """
    # Slice endings only after canonical composition: in NFD an accented
    # final vowel occupies more than one code point.
    headword = unicodedata.normalize("NFC", headword)
    hw_plain = strip_diacritics(headword)

    if headword in _IRREGULAR_BASE_FORMS:
        return "irreg", ""
    elif hw_plain.endswith("φημι"):
        # Surface -φημι is not enough to identify a φημί compound, and
        # the preverb joiner does not preserve its accent or augment.
        return None, None
    elif hw_plain.endswith("εομαι"):
        return "pres-con-e", _unaccented_verb_stem(headword[:-5])
    elif hw_plain.endswith("αομαι"):
        return "pres-con-a", _unaccented_verb_stem(headword[:-5])
    elif hw_plain.endswith("οομαι"):
        return "pres-con-o", _unaccented_verb_stem(headword[:-5])
    elif hw_plain.endswith("ουμαι"):
        # The contracted ending does not distinguish ε- and ο-contracts.
        return None, None
    elif hw_plain.endswith("ομαι"):
        return "pres", _unaccented_verb_stem(headword[:-4])
    elif hw_plain.endswith("μαι"):
        # Athematic, perfect, and contracted middles need more evidence
        # than this ending supplies. Keep their independent base entries.
        return None, None
    elif hw_plain.endswith("εω"):
        return "pres-con-e", _unaccented_verb_stem(headword[:-2])
    elif hw_plain.endswith("αω"):
        return "pres-con-a", _unaccented_verb_stem(headword[:-2])
    elif hw_plain.endswith("οω"):
        return "pres-con-o", _unaccented_verb_stem(headword[:-2])
    # -όλλυμι (ὄλλυμι and compounds) uses pres-lumi, NOT pres-numi.
    # The Wiktionary stem strips back to '-ολ' (e.g. ἀπόλλυμι -> ἀπολ).
    # Detect *before* the more general -ννυμι rule which would otherwise
    # mis-route these to pres-numi.
    elif hw_plain.endswith("ολλυμι"):
        return "pres-lumi", _unaccented_verb_stem(headword[:-4])
    elif hw_plain.endswith("ννυμι"):
        # Lua appends νυμι, so the first ν belongs to the lexical stem.
        return "pres-numi", _unaccented_verb_stem(headword[:-4])
    elif hw_plain.endswith("νυμι"):
        return "pres-numi", _unaccented_verb_stem(headword[:-4])
    # Irregular -ειμι compounds (εἰμί 'to be' / εἶμι 'to go' families).
    # These need pres-irreg with hand-coded forms; we route to the
    # irregular dispatcher rather than failing into a Lua error.
    elif hw_plain.endswith(("ειμι", "οιδα", "χρη")):
        return "irreg", ""
    # φημί has already been handled by its explicit paradigm above.
    elif hw_plain.endswith("ημι"):
        return "pres-emi", _unaccented_verb_stem(headword[:-3])
    elif hw_plain.endswith("ωμι"):
        return "pres-omi", _unaccented_verb_stem(headword[:-3])
    elif hw_plain.endswith("αμι"):
        return "pres-ami", _unaccented_verb_stem(headword[:-3])
    elif hw_plain.endswith("μι"):
        # No remaining rule-based dispatch handles a generic '-μι'
        # ending. Most surviving cases are mis-OCR'd or unusual; route
        # to the irregular dispatcher which will return empty if no
        # known base matches.
        return "irreg", ""
    elif hw_plain.endswith("ω"):
        return "pres", _unaccented_verb_stem(headword[:-1])
    return None, None


def verb_entry_rejection(head_text: str) -> str:
    """Keep nonparadigmatic entries out of unrestricted Lua expansion.

    A headword's ending alone does not make it a verb citation form. Keep
    ordinary verb entries that begin by documenting a person or tense: an
    imperative reference must explicitly identify another verb as its owner.
    """
    lead = head_text.lstrip(" ,*")
    if re.match(r"(?:insep\.\s+)?prefix\b", lead, re.IGNORECASE):
        return "prefix-headword"
    if re.match(
        r"(?:\[[^\]]*\][,\s]*)?"
        r"(?:(?:late|earlier|epic)\s+form\s+(?:for|of)\s+[^,]+,\s*)?"
        r"[123]\s*(?:sg|pl|du)\.\s*(?:(?:sg|pl|du)\.\s*)?"
        r"imper(?:at)?\.\s*(?:(?:pf|pres|aor|pass)\.\s*)*"
        r"(?:of|from)\s+",
        lead, re.IGNORECASE,
    ):
        return "inflected-imperative-headword"
    # Some defective verbs explicitly license only particular persons.
    # A regular template would invent the rest of their paradigm; retain
    # source-attested forms through extraction instead. Ignore Markdown
    # emphasis in the dictionary text when recognizing this restriction.
    if re.search(
        r"\botherwise\s+only\s+in\s+[123]\s*(?:sg|pl|du)\.",
        head_text.replace("*", ""), re.IGNORECASE,
    ):
        return "person-limited-verb-entry"
    return ""


def expand_verb(wtp, headword, head_text: str = "", supplement_text: str = "",
                principal_part_forms: set[str] | None = None):
    """Expand a verb using grc-conj template. Returns set of forms.

    Always expands the present-system paradigm. When ``head_text`` is
    provided (the lead paragraph of the LSJ entry, see
    :func:`load_lsj_head_texts`), additional principal parts (fut.,
    aor., pf., aor. p., etc.) are extracted from the entry text and
    their paradigms are unioned in. Without ``head_text`` the function
    behaves exactly as before (present-system only). ``supplement_text``
    is the same entry's head in the earlier glosses export; it adds only
    the principal parts ``head_text`` lacks. With ``principal_part_forms``
    (a set), the forms of the principal parts go into it instead of the
    result, for a caller that ranks them below the present system.

    Memoizes only identical present-system requests. Other verbs of the
    same conjugation still require their own marked paradigm from Lua.
    """
    rejection = verb_entry_rejection(head_text)
    if rejection:
        return set(), rejection
    # Reject obviously corrupt OCR'd headwords up front: they would only
    # produce Lua errors and zero forms.
    if _is_corrupt_headword(headword):
        return set(), "corrupt-headword"

    headword = unicodedata.normalize("NFC", headword)
    conj_type, stem = _classify_verb(headword)
    if conj_type is None:
        plain = strip_diacritics(headword)
        if plain.endswith("φημι"):
            return set(), "unsupported-phemi-family"
        if plain.endswith("μαι"):
            return set(), "unsupported-middle-ending"
        return set(), f"unknown-ending:{headword[-3:]}"
    if not stem and conj_type != "irreg":
        # The whole headword is the conjugation ending (the prefix ἡμι-),
        # so there is no stem to inflect. This holds even when the source
        # description that marks it as a prefix is missing from the export.
        return set(), "empty-verb-stem"

    # Literal irregular bases use their explicit paradigms. A compound
    # requires independently supported accent, augment, and family rules;
    # keep its source mappings instead of inventing a marked paradigm.
    if conj_type == "irreg":
        forms = _expand_irregular_compound(headword)
        if forms:
            return forms, ""
        preverb, base = _split_preverb(headword)
        if preverb and _strip_diacritics_lower(base) in _IRREGULAR_BASE_STEMS:
            return set(), "unsupported-irregular-compound"
        return set(), "irregular-no-base-match"

    # Present-system expansion (with cache).
    forms: set[str] = set()
    err = ""
    # A middle headword supports only its middle/passive present system;
    # the default template would also invent an active paradigm.
    voice = "mp" if headword.endswith("μαι") else ""
    cache_key = (headword, conj_type, stem, voice)
    if cache_key in _VERB_CACHE:
        forms = set(_VERB_CACHE[cache_key])
    if not forms:
        # Cache miss or empty cached result - call Lua.
        template = ("{{grc-conj|" + conj_type + "|" + stem
                    + ("|form=" + voice if voice else "") + "}}")
        try:
            wtp.start_page(headword)
            html = wtp.expand(template)
            forms = parse_html_forms(html, headword)
        except Exception as e:
            err = str(e)
        if forms:
            _VERB_CACHE[cache_key] = frozenset(forms)

    if err and not forms:
        return set(), err

    # Principal-parts expansion (additive). When the LSJ head text
    # exposes principal parts beyond the present system, expand each
    # tense via grc-conj and union the resulting forms in. Best
    # effort: any failure is silent and falls back to the present-only
    # paradigm above.
    if head_text or supplement_text:
        try:
            extra = expand_principal_parts(wtp, headword, head_text,
                                           supplement_text)
        except Exception:
            extra = set()
        if extra and principal_part_forms is not None:
            principal_part_forms |= extra - forms
        elif extra:
            forms = forms | extra

    return forms, ""


ARTICLES = {"ὁ", "ἡ", "τό", "τοῦ", "τῆς", "τῷ", "τῇ", "τόν", "τήν",
            "τών", "τῶν", "τοῖς", "ταῖς", "τούς", "τάς", "τά",
            "τοῖν", "ταῖν", "τώ", "τὼ", "αἱ", "οἱ",
            "τὰς", "τὴν", "τὸ", "τὸν", "τοὺς", "τὰ"}

def parse_html_forms(html, headword):
    """Extract Greek word forms from expanded HTML table."""
    forms = set()
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', html)
    for token in re.split(r'[\s,/;]+', text):
        token = token.strip('.,;:()[]—– ')
        # Handle wikilink pipe artifacts like "Greek|λύπη"
        if '|' in token:
            token = token.split('|')[-1]
        # Strip anchor fragments like "λύπη#Ancient&#95"
        if '#' in token:
            token = token.split('#')[0]
        token = strip_length_marks(token)
        if (token and len(token) > 1 and
            any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF'
                for c in token)
            and token not in ARTICLES
            and not token.isupper()
            and token[0] != '-'):
            forms.add(token)
    return forms


def test_one(word):
    """Test expansion on a single word."""
    lsj_entries = parse_lsj_entries()
    wikt = load_wiktionary_forms()

    entry = lsj_entries.get(word)
    if not entry:
        print(f"{word} not in LSJ")
        return

    print(f"LSJ: {entry}")
    if word in wikt:
        print(f"Wiktionary: pos={wikt[word]['pos']}, {len(wikt[word]['forms'])} forms")
    else:
        print("Not in Wiktionary")

    wtp = get_wtp()

    if entry["gender"]:
        forms, err = expand_noun(wtp, word, entry["gender"], entry["genitive"])
        print(f"\ngrc-decl result: {len(forms)} forms" + (f" (error: {err})" if err else ""))
    else:
        head_text = head_text_finder(load_lsj_head_texts())(word)
        supplement_text = head_text_finder(
            load_lsj_supplement_head_texts())(word)
        if head_text or supplement_text:
            parts = merged_principal_parts(word, head_text, supplement_text)
            print(f"LSJ principal parts: {parts}")
        forms, err = expand_verb(wtp, word, head_text=head_text,
                                 supplement_text=supplement_text)
        print(f"\ngrc-conj result: {len(forms)} forms" + (f" (error: {err})" if err else ""))

    if forms:
        print(f"Sample forms: {sorted(forms)[:20]}")

    if word in wikt:
        wikt_forms = wikt[word]["forms"]
        overlap = forms & wikt_forms
        lua_only = forms - wikt_forms
        wikt_only = wikt_forms - forms
        print(f"\nOverlap with Wiktionary: {len(overlap)}")
        print(f"Lua-only: {len(lua_only)}")
        print(f"Wiktionary-only: {len(wikt_only)}")
        if wikt_only:
            print(f"  Missing: {sorted(wikt_only)[:10]}")


def test_overlap():
    """Test expansion accuracy on overlap entries."""
    print("Loading data...")
    lsj_entries = parse_lsj_entries()
    wikt = load_wiktionary_forms()
    wikt_genitives = load_wiktionary_genitives()
    print(f"  {len(wikt_genitives):,} Wiktionary genitives loaded")
    overlap = {w for w in lsj_entries if w in wikt and lsj_entries[w]["gender"]}

    print(f"Overlap nouns with gender: {len(overlap)}")

    wtp = get_wtp()

    results = {"success": 0, "partial": 0, "fail": 0, "error": 0}
    total_recall = []
    total_precision = []

    sample_size = min(200, len(overlap))  # test a sample first
    import random
    random.seed(42)
    sample = random.sample(sorted(overlap), sample_size)

    for i, word in enumerate(sample):
        entry = lsj_entries[word]
        forms, err = expand_noun(wtp, word, entry["gender"], entry["genitive"],
                                 wikt_genitives=wikt_genitives)

        if err:
            results["error"] += 1
            continue

        wikt_forms = wikt[word]["forms"]
        if not wikt_forms:
            continue

        recall = len(forms & wikt_forms) / len(wikt_forms) if wikt_forms else 0
        precision = len(forms & wikt_forms) / len(forms) if forms else 0
        total_recall.append(recall)
        total_precision.append(precision)

        if recall > 0.8:
            results["success"] += 1
        elif recall > 0.3:
            results["partial"] += 1
        else:
            results["fail"] += 1

        if (i + 1) % 50 == 0:
            avg_r = sum(total_recall) / len(total_recall)
            avg_p = sum(total_precision) / len(total_precision)
            print(f"  {i+1}/{sample_size}: recall={avg_r:.2f} precision={avg_p:.2f}")

    avg_recall = sum(total_recall) / len(total_recall) if total_recall else 0
    avg_precision = sum(total_precision) / len(total_precision) if total_precision else 0

    print(f"\nResults ({sample_size} nouns):")
    print(f"  Avg recall: {avg_recall:.1%}")
    print(f"  Avg precision: {avg_precision:.1%}")
    print(f"  Success (>80% recall): {results['success']}")
    print(f"  Partial (30-80%): {results['partial']}")
    print(f"  Fail (<30%): {results['fail']}")
    print(f"  Error: {results['error']}")


AG_LOOKUP = DATA_DIR / "ag_lookup.json"


def expand_all():
    """Expand LSJ-only nouns and merge into ag_lookup.json."""
    import time

    input_hashes = noun_generation_input_hashes()
    print("Loading data...")
    lsj_entries = parse_lsj_entries()
    wikt = load_wiktionary_forms()
    print("Loading Wiktionary genitives for cross-reference...")
    wikt_genitives = load_wiktionary_genitives()
    print(f"  {len(wikt_genitives):,} Wiktionary genitives loaded")

    # Load existing lookup
    print(f"Loading {AG_LOOKUP}...")
    with open(AG_LOOKUP, encoding="utf-8") as f:
        lookup = json.load(f)
    original_size = len(lookup)
    print(f"  {original_size:,} existing entries")

    # Find LSJ-only nouns with gender
    candidates = []
    genitive_audit = {}
    withheld = Counter()
    superseded = set()
    for hw, entry in lsj_entries.items():
        if has_editorial_sigla(hw):
            continue
        if hw in wikt:
            continue  # already covered by Wiktionary
        if not entry["gender"]:
            continue  # no gender = can't decline
        supplied = entry["genitive"]
        resolved, error = resolve_noun_genitive(hw, entry["gender"], supplied,
                                               wikt_genitives)
        error = error or noun_expansion_error(hw, resolved)
        if (error == "unsupported-irregular-noun" or
                (supplied and not _genitive_is_junk(supplied) and (error or resolved != supplied))):
            superseded.add(sanitize_form(hw))
        if error or resolved != supplied:
            genitive_audit[hw] = {"supplied": supplied, "resolved": resolved,
                                  "reason": error or "completed-genitive"}
        if error:
            withheld[error] += 1
            continue
        candidates.append(hw)

    print(f"LSJ-only nouns to expand: {len(candidates):,}")

    wtp = get_wtp()

    stats = {"expanded": 0, "failed": 0, "new_forms": 0, "collisions": 0}
    failures = Counter()
    t0 = time.time()

    for i, hw in enumerate(candidates):
        entry = lsj_entries[hw]
        forms, err = expand_noun(wtp, hw, entry["gender"], entry["genitive"],
                                 wikt_genitives=wikt_genitives)

        if err or not forms:
            stats["failed"] += 1
            failures[err or "empty-noun-paradigm"] += 1
            continue

        stats["expanded"] += 1

        # Wiktionary's Lua modules occasionally emit forms with a misplaced
        # leading combining psili (U+0313 + base letter) for proper-noun
        # lemmas whose citation form starts with U+1FBF. sanitize_form()
        # reattaches the breathing to the base letter and NFC-composes.
        hw_clean = sanitize_form(hw)
        for raw_form in sorted(forms):
            for form in resolve_editorial_form(raw_form):
                # Accented version
                if form not in lookup:
                    lookup[form] = hw_clean
                    stats["new_forms"] += 1
                elif lookup[form] != hw_clean:
                    stats["collisions"] += 1

                # Accent-stripped version
                plain = strip_diacritics(form)
                if plain != form:
                    if plain not in lookup:
                        lookup[plain] = hw_clean
                        stats["new_forms"] += 1
                    elif lookup[plain] != hw_clean:
                        stats["collisions"] += 1

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(candidates) - i - 1) / rate
            print(f"  {i+1:,}/{len(candidates):,} "
                  f"({stats['expanded']:,} ok, {stats['failed']:,} fail, "
                  f"{stats['new_forms']:,} new forms) "
                  f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Expanded: {stats['expanded']:,} / {len(candidates):,} nouns")
    print(f"  Failed: {stats['failed']:,}")
    print(f"  Withheld before Lua: {dict(sorted(withheld.items()))}")
    print(f"  Lua failures: {dict(sorted(failures.items()))}")
    print(f"  New forms added: {stats['new_forms']:,}")
    print(f"  Collisions (kept existing): {stats['collisions']:,}")
    print(f"  Lookup size: {original_size:,} -> {len(lookup):,}")

    # Save
    out_path = AG_LOOKUP
    print(f"\nSaving to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  {size_mb:.1f} MB written")
    audit_path = out_path.with_name("ag_noun_generation_audit.json")
    audit_path.write_text(json.dumps({
        "schema_version": 1,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": input_hashes,
        "superseded_lemmas": sorted(superseded),
        "withheld": dict(sorted(withheld.items())),
        "failures": dict(sorted(failures.items())),
        "resolutions": dict(sorted(genitive_audit.items())),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# Corpus tables the verb pass checks its principal-part cells against. They
# admit nothing: the attested forms decide which cells of an inferred part
# are kept, and the treebank readings which cells are left alone.
FORM_PROFILE = DATA_DIR / "form_profile.db"
LEXICAL_HEADS = DATA_DIR / "grc_lexical_heads.json"
VERB_AUDIT = DATA_DIR / "ag_verb_generation_audit.json"
TREEBANK_PAIRS = tuple(DATA_DIR / name for name in (
    "perseus_pairs.json", "glaux_pairs.json", "diorisis_pairs.json",
    "nt_pairs.json", "ptnk_pairs.json", "tagnt_pairs.json",
    "pedalion_pairs.json"))


def bare_letters(word: str) -> str:
    """Lowercase letters without any mark, for comparing spellings."""
    return strip_diacritics(word).lower().replace("ς", "σ")


def load_attested_forms(path: Path) -> tuple[frozenset[str], list[str]]:
    """The attested forms of ``form_profile.db``.

    Returns their keys as the output gate compares them (grave folded to
    acute, case folded) and the sorted list of their bare letters, which
    answers whether any attested form begins with a stem. The letters are
    those of the form as spelled: case folding writes an iota subscript as
    a letter (ᾐ as ἠι), which no stem has.
    """
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        keys, spellings = set(), set()
        for form, key in connection.execute(
                "SELECT form, form_norm FROM forms"):
            keys.add(key)
            spellings.add(bare_letters(form))
    finally:
        connection.close()
    return frozenset(keys), sorted(spellings)


def _begins_attested(spellings: list[str], prefix: str) -> bool:
    index = bisect.bisect_left(spellings, prefix)
    return index < len(spellings) and spellings[index].startswith(prefix)


def stems_attested(tense_args: list[str], spellings: list[str]) -> bool:
    """Whether an attested form begins with a stem of this template call.

    Every cell of a call begins with one of its stems, except that the
    last consonant of a perfect middle stem changes before the ending
    (γεγραφ-, γέγραμμαι), so that one is tested without it. A call that
    fails has no attested cell and need not be expanded.
    """
    stems = [arg for arg in tense_args[1:]
             if arg and "=" not in arg and arg != INDICATIVE_ONLY]
    for position, stem in enumerate(tense_args[1:3]):
        if (position == 1 and stem and tense_args[0] in ("perf", "plup")
                and bare_letters(stem)[-1] not in "αεηιουω"):
            stems.append(stem[:-1])
    return any(_begins_attested(spellings, bare_letters(stem))
               for stem in stems)


def inferred_part_sets(headword: str, own_parts: dict[str, str],
                       verbs: dict[str, list[str]], parts_of
                       ) -> tuple[str, str, list[dict[str, str]]]:
    """Principal parts an entry does not print, as candidate sets.

    A compound takes the parts its simplex prints (``parts_of(simplex)``),
    or the regular ones of the simplex's class; any other verb the regular
    ones of its own class, with the augment at the front and after each
    preverb its letters allow. The tenses the entry prints itself
    (``own_parts``) are left out. Returns the kind of inference
    ("compound", "regular" or ""), the verb the parts come from ("" for
    the entry's own class) and the sets.

    Nothing is inferred for a capitalised headword (Βακχεύω, which the
    corpora file under βακχεύω), for a verb whose other voice has an entry
    of its own (θεάω beside θεάομαι), or, from a class, for a verb with a
    rival that would share the tenses (:func:`regular_rivals`) or one that
    may be a compound of a short verb (ἀπο-θέω).
    """
    kind, source, variants = "", "", []
    if (headword[:1] != headword[:1].lower()
            or other_voice(bare_letters(headword)) in verbs):
        return "", "", []

    def unrivalled(verb):
        return not any(rival in verbs for rival in regular_rivals(verb))

    simplexes = compound_simplexes(headword, verbs)
    if simplexes:
        printed = [(simplex, parts) for simplex in simplexes
                   if (parts := parts_of(simplex))]
        if len(printed) == 1:
            kind, source = "compound", printed[0][0]
            variants = [compound_parts(headword, *printed[0])]
        if not printed or variants == [{}]:
            regular = [(simplex, sets) for simplex in simplexes
                       if (sets := regular_parts(simplex))]
            kind, source, variants = "", "", []
            if (len(regular) == 1 and unrivalled(headword)
                    and unrivalled(regular[0][0])):
                kind, source = "regular", regular[0][0]
                variants = [compound_parts(headword, source, parts)
                            for parts in regular[0][1]]
    elif unrivalled(headword) and not short_verb_compound(headword):
        kind = "regular"
        variants = regular_parts(headword) + regular_compound_parts(headword)
    own_families = [family for family in INFERRED_FAMILIES
                    if any(key in own_parts for key in family)]
    kept = []
    for variant in variants:
        variant = {key: form for key, form in variant.items()
                   if not any(key in family for family in own_families)}
        if variant and variant not in kept:
            kept.append(variant)
    if not kept:
        return "", "", []
    return kind, source, kept


def expand_inferred_parts(wtp, headword: str, variants: list[dict[str, str]],
                          attested: frozenset[str], spellings: list[str],
                          stats: Counter | None = None) -> set[str]:
    """The attested cells of principal parts that no entry prints.

    Each set of ``variants`` is expanded like printed parts, one template
    call per tense, but only the cells whose exact spelling a corpus has
    are returned (``attested``, the keys of :func:`load_attested_forms`).
    A cell with a movable nu is two spellings (ἔλυσε, ἔλυσεν), each kept
    on its own attestation. A participle in -μενος stands for its whole
    declension, which the template does not give: it is also kept when a
    corpus has another case of it (ἐσκοτισμένους for ἐσκοτισμένος).
    """
    forms: set[str] = set()
    seen: set[tuple] = set()
    for parts in variants:
        for tense_args in derive_grc_conj_args(parts, headword).values():
            call = tuple(tense_args)
            if call in seen:
                continue
            seen.add(call)
            if not stems_attested(tense_args, spellings):
                if stats is not None:
                    stats["inferred_calls_skipped"] += 1
                continue
            if stats is not None:
                stats["inferred_calls"] += 1
            cells = _expand_principal_part(wtp, headword, tense_args)
            if stats is not None:
                stats["inferred_cells"] += len(cells)
            for cell in cells:
                for form in resolve_editorial_form(cell):
                    if norm_key(form) in attested:
                        forms.add(form)
                    elif form not in forms and any(
                            norm_key(case) in attested
                            for case in participle_cases(form)):
                        forms.add(form)
                        if stats is not None:
                            stats["inferred_participles_by_case"] += 1
    return forms


def _pair_lemma(lemma: str) -> str:
    """A treebank lemma without its homograph number."""
    return unicodedata.normalize("NFC", lemma).rstrip("0123456789")


def reading_key(form: str) -> str:
    """A spelling as the treebank readings are keyed: the grave folded to
    the acute, the case kept."""
    folded = unicodedata.normalize("NFD", form).replace("\u0300", "\u0301")
    return unicodedata.normalize("NFC", folded)


def load_lexical_heads(path: Path) -> frozenset[str]:
    """The headwords of every dictionary the build knows
    (``grc_lexical_heads.json``), without the reviewed non-lemmas. The
    lookup build accepts a treebank lemma on the same grounds."""
    with open(path, encoding="utf-8") as f:
        heads = json.load(f)["heads"]
    return frozenset(heads) - REJECTED_LEMMAS


def load_treebank_readings(paths, forms) -> dict[str, list[tuple[str, str]]]:
    """The readings the treebank pair files give each of ``forms``.

    The lookup build lets the pair files fill gaps in the order of
    ``paths``, and within a file the first pair of a form does. The result
    holds, for each form (:func:`reading_key`), the lemmas and parts of
    speech of the first file that has it, in the file's order. A file that
    has the form only with a capital (sentence-initial Κατεδέξατο) speaks
    for it when no file has it as spelled, unless the lemma is capitalised
    too: Φιαλέων of Φιαλεύς is not φιαλέων.
    """
    wanted = {reading_key(form) for form in forms}
    folded_wanted = {key.casefold() for key in wanted}
    exact: dict[str, list[tuple[str, str]]] = {}
    folded: dict[str, list[tuple[str, str]]] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            pairs = json.load(f)
        found_exact: dict[str, list[tuple[str, str]]] = {}
        found_folded: dict[str, list[tuple[str, str]]] = {}
        for pair in pairs:
            lemma = pair.get("lemma")
            if not lemma:
                continue
            key = reading_key(pair["form"])
            if key in wanted:
                if key in exact:
                    continue
                found = found_exact.setdefault(key, [])
            else:
                key = key.casefold()
                if (key not in folded_wanted or key in folded
                        or lemma[:1] != lemma[:1].lower()):
                    continue
                found = found_folded.setdefault(key, [])
            reading = (_pair_lemma(lemma), pair.get("pos") or "")
            if reading not in found:
                found.append(reading)
        exact.update(found_exact)
        folded.update(found_folded)
        del pairs
    readings = {}
    for key in wanted:
        found = exact.get(key) or folded.get(key.casefold())
        if found:
            readings[key] = found
    return readings


def treebank_reading(form: str, lemma: str, readings, lexical_heads
                     ) -> tuple[str, str]:
    """The treebank reading that counts for a cell of the verb ``lemma``.

    Returns ``lemma`` itself where the treebanks read the form as this
    verb, another lemma (with its part of speech) where they read it
    otherwise, and ``("", "")`` where they have nothing usable. As in the
    gap fill of the lookup build, the first usable reading decides: one
    under this verb's letters (ἐνθρῴσκω for ἐνθρώσκω), under a headword of
    some dictionary, or under this verb in its other voice (ἐπιλανθάνομαι
    for ἐπιλανθάνω). A lemma that no dictionary heads (σύν-εἶδον,
    νεπεπυρίζω) is passed over.
    """
    own = bare_letters(lemma)
    for other, pos in readings.get(reading_key(form), ()):
        letters = bare_letters(other)
        if letters == own:
            return lemma, pos
        if (other in lexical_heads
                or other[:1].lower() + other[1:] in lexical_heads
                or other_voice(letters) == own):
            return other, pos
    return "", ""


def treebank_agrees(form: str, lemma: str, readings, lexical_heads) -> bool:
    """Whether the treebank reading of ``form`` is this verb."""
    return treebank_reading(form, lemma, readings, lexical_heads)[0] == lemma


def left_to_treebank(form: str, lemma: str, readings, lexical_heads) -> bool:
    """Whether a cell on a stem no entry prints gives way to the treebank
    reading: a cell of an inferred part, or one on the unaugmented stem
    derived for a printed past tense.

    It does when the treebanks read the form as another word
    (:func:`treebank_reading`): ἠβουλήθη is the aorist of βούλομαι, not of
    ἀβουλέω, and ἐπελάθοντο stays with ἐπιλανθάνομαι.
    """
    reading = treebank_reading(form, lemma, readings, lexical_heads)[0]
    return bool(reading) and reading != lemma


def left_to_rival(form: str, lemma: str, readings, lexical_heads,
                  yields: dict[str, set[str]]) -> bool:
    """Whether a cell on a stem the entry prints gives way to another word.

    The dictionary states such a cell, so another verb in a treebank does
    not move it by itself (ἀμφιέσομαι stays the future that LSJ prints
    under ἀμφιέννυμι). It gives way where the verb the treebanks read has
    the cell among its own principal parts, printed or inferred
    (``yields``: the verbs, by their letters, for each
    :func:`reading_key`): κατεδέξατο is the aorist of καταδέχομαι there, a
    compound of δέχομαι, while LSJ prints κατέδεξα as the Ionic aorist of
    καταδείκνυμι. It also gives way where the treebanks read the form as
    no verb at all (φιαλέων is the genitive of φιάλη before it is a
    participle of the future φιαλῶ), or as this verb in its other voice
    (ἐνόσφισεν under νοσφίζομαι): that is the same verb.
    """
    reading, pos = treebank_reading(form, lemma, readings, lexical_heads)
    if not reading or reading == lemma:
        return False
    if pos not in ("verb", ""):
        return True
    letters = bare_letters(reading)
    return (letters in yields.get(reading_key(form), ())
            or other_voice(letters) == bare_letters(lemma))


# An inferred cell that a noun also spells is kept only where a treebank
# reads it as the verb: the future in -σει or -σεις is a case of the action
# noun in -σις (ἐπισκιάσει, ἀμοιρήσεις), and the participle of an Attic
# future in -ιῶν, or of its uncontracted -ιέων, a genitive plural (κοσμιῶν,
# πεδιέων).
_ACTION_NOUN_ENDINGS = ("σει", "ξει", "ψει", "σεισ", "ξεισ", "ψεισ")
_GENITIVE_PLURAL_ENDINGS = ("ιῶν", "ιέων")


def noun_twin(form: str) -> bool:
    """Whether a noun paradigm spells this verb cell too."""
    return (bare_letters(form).endswith(_ACTION_NOUN_ENDINGS)
            or unicodedata.normalize("NFC", form).endswith(
                _GENITIVE_PLURAL_ENDINGS))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expand_verbs(corpus_checks: bool = True):
    """Expand LSJ verbs and merge into ag_lookup.json.

    With ``corpus_checks`` (the published build), the pass reads
    ``form_profile.db`` and the treebank pair files: principal parts that
    no entry prints are inferred and their attested cells kept, and a
    principal-part cell that the treebanks read as another verb is left
    out (:func:`left_to_treebank`, :func:`left_to_rival`). Without it
    neither happens.
    """
    import time

    if corpus_checks:
        missing = [path.name for path in (FORM_PROFILE, LEXICAL_HEADS,
                                           *TREEBANK_PAIRS)
                   if not path.exists()]
        if missing:
            sys.exit(
                f"{', '.join(missing)}: missing from {DATA_DIR}. The verb "
                "pass checks its principal-part cells against them (python "
                "-m dilemma download --with-attestation fetches "
                "form_profile.db; build/build_grc_lexical_heads.py writes "
                "grc_lexical_heads.json). --no-corpus-checks runs without "
                "them, which is not the published build.")

    print("Loading data...")
    lsj_entries = parse_lsj_entries()
    wikt = load_wiktionary_forms()
    print("Loading LSJ entry head texts (for principal-parts extraction)...")
    pinned = verify_lsj_glosses()
    print(f"  {LSJ9_GLOSSES.name}: sha256 {pinned['sha256'][:12]} (pinned)")
    if "supplement" in pinned:
        print(f"  {LSJ9_GLOSSES_SUPPLEMENT.name}: sha256 "
              f"{pinned['supplement']['sha256'][:12]} (pinned)")
    head_texts = load_lsj_head_texts()
    supplement_texts = load_lsj_supplement_head_texts()
    find_head_text = head_text_finder(head_texts)
    find_supplement_text = head_text_finder(supplement_texts)

    print(f"Loading {AG_LOOKUP}...")
    with open(AG_LOOKUP, encoding="utf-8") as f:
        lookup = json.load(f)
    original_size = len(lookup)
    print(f"  {original_size:,} existing entries")

    # Find LSJ-only verbs (entries without gender = likely verbs)
    candidates = []
    for hw, entry in lsj_entries.items():
        if has_editorial_sigla(hw):
            continue
        if hw in wikt:
            continue
        if entry["gender"]:
            continue  # has gender = noun/adj, not verb
        dp = strip_diacritics(hw)
        if (dp.endswith("ω") or dp.endswith("μι")
                or dp.endswith("μαι")):
            candidates.append(hw)

    print(f"LSJ-only verbs to expand: {len(candidates):,}")

    wtp = get_wtp()

    stats = Counter({"expanded": 0, "failed": 0, "new_forms": 0,
                     "collisions": 0, "with_pp": 0, "with_supplement_pp": 0})
    failure_reasons = Counter()
    failure_examples = {}
    t0 = time.time()

    # The keys this pass has written itself, as against those it found.
    written: set[str] = set()

    def add_forms(forms, hw_clean):
        # See sanitize_form() rationale in expand_all() above.
        for raw_form in sorted(forms):
            for form in resolve_editorial_form(raw_form):
                if form not in lookup:
                    lookup[form] = hw_clean
                    written.add(form)
                    stats["new_forms"] += 1
                elif lookup[form] != hw_clean:
                    stats["collisions"] += 1

                plain = strip_diacritics(form)
                if plain != form:
                    if plain not in lookup:
                        lookup[plain] = hw_clean
                        written.add(plain)
                        stats["new_forms"] += 1
                    elif lookup[plain] != hw_clean:
                        stats["collisions"] += 1

    # The present system of every verb goes in before any principal part:
    # a cell a headword spells out itself (φιαλοῦμαι of φιαλόω) outranks
    # one derived from another verb's printed part (the future φιαλῶ of
    # φιάλλω gives the same form). A headword that another verb prints as
    # its principal part (ὑπομνήσω under ὑπομιμνήσκω) is that verb's tense,
    # so its cells go in last.
    deferred = []
    late = []
    inferable = []
    texts = {hw: (find_head_text(hw), find_supplement_text(hw))
             for hw in candidates}
    printed_parts = set()
    for hw, (head_text, supplement_text) in texts.items():
        for form in merged_principal_parts(
                hw, head_text, supplement_text).values():
            printed_parts.add(strip_length_marks(form))

    for i, hw in enumerate(candidates):
        # Entry selection strips quantity marks; the entry's text is under
        # its own spelling (head_text_finder), and a spelling two entries
        # share gets none.
        head_text, supplement_text = texts[hw]
        # The original spelling still identifies the source description of
        # a shared one (ἡμῐ is a prefix) for the eligibility check.
        description = head_texts.get(lsj_entries[hw].get("orth_orig"), head_text)
        rejection = verb_entry_rejection(description)
        if rejection:
            forms, err = set(), rejection
        else:
            extra = set()
            forms, err = expand_verb(
                wtp, hw, head_text=head_text,
                supplement_text=supplement_text, principal_part_forms=extra)

        if err or not forms:
            stats["failed"] += 1
            reason = err.splitlines()[0][:160] if err else "empty-paradigm"
            failure_reasons[reason] += 1
            examples = failure_examples.setdefault(reason, [])
            if len(examples) < 3:
                examples.append(hw)
            continue

        stats["expanded"] += 1
        # Track verbs that contributed at least one extra principal
        # part. We re-parse here rather than threading the result
        # through expand_verb to keep that signature simple.
        own_parts = merged_principal_parts(hw, head_text)
        all_parts = merged_principal_parts(hw, head_text, supplement_text)
        if all_parts:
            stats["with_pp"] += 1
        if len(all_parts) > len(own_parts):
            stats["with_supplement_pp"] += 1

        hw_clean = sanitize_form(hw)
        if hw in printed_parts and not all_parts:
            late.append((forms, hw_clean))
        else:
            add_forms(forms, hw_clean)
        if extra:
            derived = (derived_principal_part_cells(
                wtp, hw, head_text, supplement_text) if corpus_checks
                else set())
            deferred.append((extra, hw_clean, derived))
        inferable.append((hw, hw_clean, all_parts))

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(candidates) - i - 1) / rate
            print(f"  {i+1:,}/{len(candidates):,} "
                  f"({stats['expanded']:,} ok, {stats['failed']:,} fail, "
                  f"{stats['new_forms']:,} new forms) "
                  f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]")

    inferred = []
    if corpus_checks:
        # Principal parts that no entry prints, kept where a corpus has the
        # exact cell. A cell two verbs infer is left out.
        print("Inferring the principal parts that no entry prints...")
        attested, spellings = load_attested_forms(FORM_PROFILE)
        print(f"  {FORM_PROFILE.name}: {len(attested):,} attested forms")
        verbs: dict[str, list[str]] = {}
        for hw, entry in lsj_entries.items():
            if entry["gender"] or has_editorial_sigla(hw):
                continue
            if strip_diacritics(hw).endswith(("ω", "μι", "μαι")):
                verbs.setdefault(bare_letters(hw), []).append(hw)
        simplex_parts: dict[str, dict[str, str]] = {}

        def parts_of(simplex):
            if simplex not in simplex_parts:
                simplex_parts[simplex] = merged_principal_parts(
                    simplex, find_head_text(simplex),
                    find_supplement_text(simplex))
            return simplex_parts[simplex]

        heads = set(lsj_entries) | set(wikt)
        claims: dict[str, set[str]] = {}
        inferred_from: dict[str, tuple[str, str]] = {}
        for hw, hw_clean, own_parts in inferable:
            kind, source, variants = inferred_part_sets(
                hw, own_parts, verbs, parts_of)
            if not variants:
                continue
            inferred_from[hw_clean] = (kind, source)
            stats["inferred_" + kind + "_verbs"] += 1
            # The table has every present system by now; a cell it already
            # has, under this verb or another, is not this pass's to give,
            # and neither is a spelling that is a headword itself.
            cells = {form for form in expand_inferred_parts(
                         wtp, hw, variants, attested, spellings, stats)
                     if form not in lookup and form not in heads}
            if cells:
                stats["inferred_" + kind + "_verbs_attested"] += 1
            for form in cells:
                claims.setdefault(form, set()).add(hw_clean)
        by_verb: dict[str, set[str]] = {}
        shared = {}
        for form, claimants in claims.items():
            if len(claimants) > 1:
                stats["inferred_shared"] += 1
                shared[form] = sorted(claimants)
                continue
            by_verb.setdefault(next(iter(claimants)), set()).add(form)
        inferred = sorted(by_verb.items())
        print(f"  {time.time() - t0:.0f}s elapsed")

        # A principal-part cell does not take a form the treebanks read as
        # another verb. Each spelling of a cell is checked by itself
        # (ἔλυσε, ἔλυσεν).
        def spellings_of(cells):
            return {form for cell in cells
                    for form in resolve_editorial_form(cell)}

        deferred = [(spellings_of(forms), hw_clean, spellings_of(derived))
                    for forms, hw_clean, derived in deferred]
        wanted = {form for forms, _, _ in deferred for form in forms}
        wanted |= {form for _, forms in inferred for form in forms}
        readings = load_treebank_readings(TREEBANK_PAIRS, wanted)
        lexical_heads = load_lexical_heads(LEXICAL_HEADS)
        # The verbs that have a form among their own principal parts.
        yields: dict[str, set[str]] = {}
        for forms, hw_clean, _ in deferred:
            for form in forms:
                yields.setdefault(reading_key(form), set()).add(
                    bare_letters(hw_clean))
        for form, claimants in claims.items():
            yields.setdefault(reading_key(form), set()).update(
                bare_letters(verb) for verb in claimants)
        withheld = []
        over_present = []

        def uncontested(forms, hw_clean, kind, derived=frozenset()):
            kept = set()
            for form in forms:
                # A cell on a stem the entry prints is the dictionary's; one
                # on a derived stem is the generator's, like an inferred one.
                part = "derived" if form in derived else kind
                if part == "printed":
                    gives_way = left_to_rival(
                        form, hw_clean, readings, lexical_heads, yields)
                else:
                    gives_way = left_to_treebank(
                        form, hw_clean, readings, lexical_heads)
                if gives_way:
                    stats[part + "_withheld"] += 1
                    withheld.append((form, hw_clean, part, treebank_reading(
                        form, hw_clean, readings, lexical_heads)[0]))
                elif (kind == "inferred" and noun_twin(form)
                      and not treebank_agrees(
                          form, hw_clean, readings, lexical_heads)):
                    # Only a treebank tells such a cell from the noun.
                    stats["inferred_noun_twins"] += 1
                else:
                    kept.add(form)
            return kept

        deferred = [(uncontested(forms, hw_clean, "printed", derived),
                     hw_clean, derived)
                    for forms, hw_clean, derived in deferred]
        inferred = [(hw_clean, uncontested(forms, hw_clean, "inferred"))
                    for hw_clean, forms in inferred]

        # So far this pass has written present systems only. Where another
        # verb's present holds the cell of a printed part (περιειλόμεθα of
        # περιείλω and of the aorist περιειλόμην under περιαιρέω) and the
        # treebanks read the form as the verb of the part, that verb takes
        # it.
        for forms, hw_clean, _ in deferred:
            for form in sorted(forms):
                held = lookup.get(form)
                if (held is None or held == hw_clean or form not in written
                        or not treebank_agrees(
                            form, hw_clean, readings, lexical_heads)):
                    continue
                lookup[form] = hw_clean
                plain = strip_diacritics(form)
                if plain in written and lookup.get(plain) == held:
                    lookup[plain] = hw_clean
                stats["printed_over_present"] += 1
                over_present.append((form, hw_clean, held))

    for forms, hw_clean, _ in deferred:
        add_forms(forms, hw_clean)
    for forms, hw_clean in late:
        add_forms(forms, hw_clean)
    before_inferred = stats["new_forms"]
    for hw_clean, forms in inferred:
        stats["inferred_kept"] += len(forms)
        add_forms(forms, hw_clean)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Expanded: {stats['expanded']:,} / {len(candidates):,} verbs")
    print(f"  With LSJ principal parts: {stats['with_pp']:,}")
    print("  With principal parts from the earlier export: "
          f"{stats['with_supplement_pp']:,}")
    if corpus_checks:
        print("  With inferred principal parts: "
              f"{stats['inferred_compound_verbs']:,} from a simplex "
              f"({stats['inferred_compound_verbs_attested']:,} with an "
              f"attested cell), {stats['inferred_regular_verbs']:,} from a "
              f"regular class ({stats['inferred_regular_verbs_attested']:,})")
        print(f"    Template calls: {stats['inferred_calls']:,} made, "
              f"{stats['inferred_calls_skipped']:,} skipped (no attested "
              "form begins with a stem)")
        print(f"    Cells: {stats['inferred_cells']:,} generated, "
              f"{stats['inferred_kept']:,} attested and kept, "
              f"{stats['inferred_shared']:,} left out as inferred by two "
              f"verbs, {stats['inferred_noun_twins']:,} as cells that a "
              "noun also spells; participles attested in another case only: "
              f"{stats['inferred_participles_by_case']:,}; keys added: "
              f"{stats['new_forms'] - before_inferred:,}")
        print("  Cells left to a treebank reading: "
              f"{stats['printed_withheld']:,} on a printed stem (the other "
              f"verb has the cell too), {stats['derived_withheld']:,} on a "
              f"derived stem, {stats['inferred_withheld']:,} of inferred "
              "parts")
        print("  Present cells that gave way to a printed part the treebanks "
              f"read: {stats['printed_over_present']:,}")
        for form, hw_clean, _, reading in sorted(withheld)[:40]:
            print(f"    {form} ({hw_clean}): {reading}")
    print(f"  Failed: {stats['failed']:,}")
    for reason, count in failure_reasons.most_common():
        print(f"    {reason}: {count:,} ({', '.join(failure_examples[reason])})")
    print(f"  New forms added: {stats['new_forms']:,}")
    print(f"  Collisions (kept existing): {stats['collisions']:,}")
    print(f"  Lookup size: {original_size:,} -> {len(lookup):,}")

    print(f"\nSaving to {AG_LOOKUP}...")
    with open(AG_LOOKUP, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False)
    size_mb = AG_LOOKUP.stat().st_size / (1024 * 1024)
    print(f"  {size_mb:.1f} MB written")

    if corpus_checks:
        # What the corpus tables decided, for review: the inferred cells
        # by verb, the cells two verbs inferred, the cells left to a
        # treebank reading.
        connection = sqlite3.connect(f"file:{FORM_PROFILE}?mode=ro", uri=True)
        try:
            row = connection.execute(
                "SELECT value FROM meta WHERE key = 'content_hash'").fetchone()
        except sqlite3.Error:
            row = None
        finally:
            connection.close()
        audit = {
            "schema_version": 1,
            "inputs": {
                FORM_PROFILE.name: {"content_hash": row[0] if row else None},
                **{path.name: {"sha256": sha256_file(path)}
                   for path in (LEXICAL_HEADS, *TREEBANK_PAIRS)},
            },
            "counts": {key: stats[key] for key in sorted(stats)
                       if key.startswith(("inferred_", "printed_",
                                          "derived_"))},
            "inferred": {
                hw_clean: {"kind": inferred_from[hw_clean][0],
                           "from": inferred_from[hw_clean][1],
                           "cells": sorted(forms)}
                for hw_clean, forms in inferred if forms},
            "shared": dict(sorted(shared.items())),
            "left_to_treebank": [
                {"form": form, "verb": hw_clean, "part": part,
                 "treebank": reading}
                for form, hw_clean, part, reading in sorted(withheld)],
            "taken_from_a_present": [
                {"form": form, "verb": hw_clean, "present_of": held}
                for form, hw_clean, held in sorted(over_present)],
        }
        with open(VERB_AUDIT, "w", encoding="utf-8") as f:
            json.dump(audit, f, ensure_ascii=False, indent=1, sort_keys=True)
            f.write("\n")
        print(f"  {VERB_AUDIT.name} written")


def main():
    parser = argparse.ArgumentParser(description="Expand LSJ headwords via Wiktionary Lua")
    parser.add_argument("--setup", action="store_true",
                        help="Set up wikitextprocessor database (first run)")
    parser.add_argument("--test", action="store_true",
                        help="Test on overlap entries")
    parser.add_argument("--test-one", type=str, default=None,
                        help="Test a single word")
    parser.add_argument("--expand", action="store_true",
                        help="Expand LSJ-only noun entries and add to lookup")
    parser.add_argument("--expand-verbs", action="store_true",
                        help="Expand LSJ-only verb entries and add to lookup")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --expand, show stats but don't save")
    parser.add_argument("--no-corpus-checks", action="store_true",
                        help="With --expand-verbs, run without "
                             "form_profile.db and the treebank pair files: "
                             "no inferred principal parts, no treebank "
                             "check (not the published build)")
    add_input_arguments(parser)
    args = parser.parse_args()
    configure_inputs(args.kaikki, args.lsj9_dir)

    if args.setup:
        setup_wtp()
    elif args.test_one:
        test_one(args.test_one)
    elif args.test:
        test_overlap()
    elif args.expand:
        expand_all()
    elif args.expand_verbs:
        expand_verbs(corpus_checks=not args.no_corpus_checks)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

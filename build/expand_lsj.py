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
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dilemma.form_sanitize import sanitize_form  # noqa: E402
from lsj_principal_parts import (  # noqa: E402
    parse_principal_parts, derive_grc_conj_args,
)

DATA_DIR = SCRIPT_DIR / "data"
LSJ9_DIR = Path.home() / "Documents" / "lsj9"
LSJ9_FORMS = LSJ9_DIR / "lsj9_forms.tsv"
LSJ9_HEADWORDS = LSJ9_DIR / "lsj9_headwords.json"
LSJ9_GLOSSES = LSJ9_DIR / "lsj9_glosses.jsonl"
KAIKKI_DIR = Path(os.environ.get(
    "KAIKKI_DIR", Path.home() / "Documents" / "kaikki"))
# Try nested layout first (en-el/), then flat layout
_KAIKKI_AG_NESTED = KAIKKI_DIR / "en-el" / "kaikki.org-dictionary-AncientGreek.jsonl"
_KAIKKI_AG_FLAT = KAIKKI_DIR / "kaikki.org-en-dictionary-AncientGreek.jsonl"
KAIKKI_AG = _KAIKKI_AG_NESTED if _KAIKKI_AG_NESTED.exists() else _KAIKKI_AG_FLAT
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


def build_genitive_from_itype(headword, itype):
    """Build genitive form from headword + LSJ itype (genitive suffix).

    The itype replaces a portion of the headword's ending. The number of
    characters to strip depends on the nominative ending pattern.
    """
    if not itype:
        return ""

    hw_plain = strip_diacritics(headword)
    it_plain = strip_diacritics(itype)

    # (nominative ending, itype ending) → chars to strip from headword
    # Ordered so more specific patterns match first
    STRIP_RULES = [
        # -ευς / -εως: strip 3 (the ε is shared between stem and itype)
        ("ευς", "εως", 3),
        # 1st/2nd declension
        ("ος", "ου", 2),
        ("ον", "ου", 2),
        ("ης", "ου", 2),
        ("ας", "ου", 2),
        ("η", "ης", 1),
        # 3rd declension -ις/-εως (biggest: 4451 entries, πόλις-type)
        ("ις", "εως", 2),
        # 3rd declension -υς/-εως (πῆχυς-type, NOT -ευς which is above)
        ("υς", "εως", 2),
        # 3rd declension -ης/-ους (Attic, e.g. Σωκράτης)
        ("ης", "ους", 2),
        # 3rd declension -ης/-ητος
        ("ης", "ητος", 2),
        # 3rd declension -ως/-ω (Attic, e.g. ἥρως)
        ("ως", "ωος", 2),
    ]

    for nom_end, gen_itype, strip_n in STRIP_RULES:
        if hw_plain.endswith(nom_end) and it_plain == strip_diacritics(gen_itype):
            return headword[:-strip_n] + itype

    # Default: strip 1 char (the final case marker) and append itype.
    # Works for most 3rd declension patterns where itype starts from
    # the oblique stem consonant:
    #   -μα + ατος → -ματος (strip α, append ατος)
    #   -ίς + ίδος → -ίδος (strip ς, append... wait, ίδος)
    #   -ήρ + ῆρος → -ῆρος (strip ρ... hmm)
    #
    # Actually for consonant stems, strip 1 often leaves extra chars.
    # Try strip 2 if itype starts with a char that matches the second-to-last
    # char of the headword (accent-free).
    if len(hw_plain) >= 2 and len(it_plain) >= 1:
        if hw_plain[-2] == it_plain[0]:
            # The itype "restarts" from a character that's already in the headword
            # e.g. ἐλπίς + ίδος: ί matches → strip 2, append ίδος
            return headword[:-2] + itype
    return headword[:-1] + itype


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


def parse_lsj9_entries(forms_path: Path = LSJ9_FORMS) -> dict:
    """Load entries from lsj9_forms.tsv (an upstream LSJ9 export).

    This provides explicit grammar (ὁ/ἡ/τό/ον/ές) and pre-extracted
    genitive endings for 63K entries.

    Returns same format as parse_lsj_entries(): {headword: {headword,
    orth_orig, article, gender, itype, genitive}}.
    """
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
    """Load genitive forms from Wiktionary for cross-referencing with LSJ.

    Returns {headword: genitive_form} for nouns/adjectives that have
    genitive singular forms tagged in Wiktionary.
    """
    genitives = {}
    if not KAIKKI_AG.exists():
        return genitives
    with open(KAIKKI_AG, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            word = e.get("word", "")
            pos = e.get("pos", "")
            if pos not in ("noun", "adj"):
                continue
            forms = e.get("forms", [])
            for fe in forms:
                tags = fe.get("tags", [])
                form = strip_length_marks(fe.get("form", ""))
                if "genitive" in tags and "singular" in tags and form:
                    # Strip article prefix if present (e.g. "τῆς κυνός" -> "κυνός")
                    if " " in form:
                        form = form.split()[-1]
                    genitives[strip_length_marks(word)] = form
                    break
    return genitives


def infer_genitive(headword, gender, wikt_genitives=None):
    """Infer genitive singular from nominative ending + gender.

    Priority:
    1. Wiktionary cross-reference (gold standard)
    2. Regular 1st/2nd declension patterns
    3. 3rd declension heuristics (lower confidence)
    """
    # Check Wiktionary cross-reference first
    if wikt_genitives:
        hw_clean = strip_length_marks(headword)
        if hw_clean in wikt_genitives:
            return wikt_genitives[hw_clean]

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


# Cache: (gender, nom_ending, gen_ending) -> list of full forms from reference word
# Nouns with the same declension pattern produce forms that differ only in the stem.
# We store (ref_stem, ref_forms) and apply by swapping stems.
_NOUN_CACHE = {}


def _classify_noun(headword, gender, genitive):
    """Classify a noun into declension type for caching.

    Returns (cache_key, stem_len) where cache_key = (gender, nom_ending, gen_ending).
    stem_len is how many chars of the headword form the invariant stem.
    """
    hw_plain = strip_diacritics(headword.lower())
    gen_plain = strip_diacritics(genitive.lower()) if genitive else ""

    # Find the longest common prefix = invariant stem
    # Then trim back to exclude the thematic/connecting vowel, which
    # is shared between nom and gen but changes in other cases
    # (e.g. ανθρωπ-ος/ου share 'ο' but vocative is ανθρωπ-ε)
    stem_len = 0
    for i in range(min(len(hw_plain), len(gen_plain)) if gen_plain else 0):
        if hw_plain[i] == gen_plain[i]:
            stem_len = i + 1
        else:
            break

    # Trim back thematic vowel: if the last shared char is a vowel
    # and it's the first char of both endings, exclude it from stem
    vowels = set("αεηιουω")
    if stem_len > 1 and hw_plain[stem_len - 1] in vowels:
        stem_len -= 1

    # Ensure at least 1 char as ending
    stem_len = min(stem_len, len(hw_plain) - 1)

    nom_ending = hw_plain[stem_len:]
    gen_ending = gen_plain[stem_len:] if gen_plain else ""

    return (gender, nom_ending, gen_ending), stem_len


def _apply_noun_cache(headword, stem_len, ref_stem_plain, ref_forms):
    """Apply cached declension forms to a new headword by swapping the stem.

    ref_forms are the full accented forms from the reference word.
    We strip the reference stem prefix and replace it with the new stem.
    """
    new_stem = strip_diacritics(headword[:stem_len].lower())
    forms = set()
    for form in ref_forms:
        form_plain = strip_diacritics(form.lower())
        if form_plain.startswith(ref_stem_plain):
            ending = form_plain[len(ref_stem_plain):]
            new_form = new_stem + ending
            if len(new_form) > 1:
                forms.add(new_form)
    return forms


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


def expand_noun(wtp, headword, gender, genitive="", wikt_genitives=None):
    """Expand a noun using grc-decl template. Returns set of forms.

    Uses a cache keyed by declension pattern: nouns with the same
    (gender, nom_ending, gen_ending) produce the same inflection endings.
    """
    # A junk LSJ genitive (article, dialect note, "**" markup, fragment) can't
    # match a declension, so discard it and infer from the nominative instead.
    if genitive and _genitive_is_junk(genitive):
        genitive = ""
    if not genitive:
        genitive = infer_genitive(headword, gender, wikt_genitives)

    # Check cache
    if genitive:
        cache_key, stem_len = _classify_noun(headword, gender, genitive)
        if cache_key in _NOUN_CACHE:
            ref_stem, ref_forms = _NOUN_CACHE[cache_key]
            forms = _apply_noun_cache(headword, stem_len, ref_stem, ref_forms)
            if forms:
                return forms, ""

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

    forms = parse_html_forms(html, headword)

    # Cache: store (ref_stem_plain, ref_forms) for this declension pattern
    if forms and genitive:
        cache_key, stem_len = _classify_noun(headword, gender, genitive)
        if cache_key not in _NOUN_CACHE:
            ref_stem = strip_diacritics(headword[:stem_len].lower())
            _NOUN_CACHE[cache_key] = (ref_stem, forms)

    return forms, ""


# Cache: (conj_type, stem_len) -> list of suffix offsets extracted from a reference expansion
# Each cached entry is [(suffix, strip_n), ...] where form = stem[:-strip_n] + suffix
_VERB_CACHE = {}


# Cache: principal-parts grc-conj-args tuple -> set of forms.
# Keyed on (tense_code, *stems) so verbs that share a passive stem (e.g.
# all -ευω verbs producing "-ευθ" passives) reuse the expansion.
_PP_CACHE: dict[tuple, set[str]] = {}


_LSJ_HEAD_TEXTS: dict[str, str] | None = None


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
    with open(LSJ9_GLOSSES, encoding="utf-8") as f:
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
    print(f"  lsj9 head texts: {len(heads):,}")
    _LSJ_HEAD_TEXTS = heads
    return heads


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
    tpl = "{{grc-conj|" + "|".join(tense_args) + "}}"
    try:
        wtp.start_page(headword)
        html = wtp.expand(tpl)
    except Exception:
        _PP_CACHE[cache_key] = set()
        return set()
    forms = parse_html_forms(html, headword)
    _PP_CACHE[cache_key] = forms
    return forms


def expand_principal_parts(wtp, headword: str, head_text: str) -> set[str]:
    """Extract LSJ principal parts and expand each into a paradigm.

    Returns the union of forms produced by the per-tense grc-conj
    invocations. Returns an empty set if no parts could be parsed
    (caller should fall back to present-only expansion).
    """
    parts = parse_principal_parts(head_text, headword)
    if not parts:
        return set()
    args_by_tense = derive_grc_conj_args(parts, headword)
    if not args_by_tense:
        return set()
    forms: set[str] = set()
    for tense_args in args_by_tense.values():
        forms |= _expand_principal_part(wtp, headword, tense_args)
    return forms


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
# minimal but useful Attic core paradigm here so compound forms (πάρειμι,
# σύνειμι, εἴσειμι, ...) can be expanded by stem-prepending.
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
    # φημί 'say' — the existing dispatcher routes φημί to pres-emi, but
    # Wiktionary uses pres-ami for it (stem 'φ'). We keep the explicit
    # paradigm here as a robust fallback.
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


def _join_preverb(preverb, base_form):
    """Prepend `preverb` to `base_form` for an AG compound verb.

    Strategy: we generate two surface candidates - one with the
    diacritics stripped from the base form (so the natural-accent
    placement falls out from downstream NFC + accent rules) and one
    that keeps the base's diacritics. We add both to the set the
    caller will store: the lookup table indexes both accented and
    diacritic-stripped keys, so a slightly imperfect accent on the
    accented version is acceptable as long as the stripped key is
    correct.

    The preverb's trailing vowel elides before a vowel-initial base
    form; the base's leading breathing is dropped in the joined form.
    Aspirate-mutation (κατά + ἵστημι -> καθίστημι) is approximated
    by mapping τ/π/κ -> θ/φ/χ when the base had a rough breathing.
    """
    if not preverb:
        return base_form
    nfd_base = unicodedata.normalize("NFD", base_form)
    base_first = None
    base_had_rough = False
    base_first_idx = None
    for i, ch in enumerate(nfd_base):
        if not unicodedata.combining(ch):
            base_first = ch
            base_first_idx = i
            break
    # Detect rough breathing on first base char
    if base_first_idx is not None:
        for ch in nfd_base[base_first_idx + 1:]:
            if not unicodedata.combining(ch):
                break
            if ord(ch) == 0x0314:
                base_had_rough = True

    # Strip ALL combining marks from the first base char (breathing + accent)
    # so the preverb's accent or syllable structure dictates the surface.
    if base_first_idx is not None:
        out = []
        i = 0
        # Keep chars before the first base char (none usually)
        out.extend(nfd_base[:base_first_idx + 1])
        i = base_first_idx + 1
        # Drop combining marks immediately after the first base char
        while i < len(nfd_base) and unicodedata.combining(nfd_base[i]):
            i += 1
        out.extend(nfd_base[i:])
        base_clean = unicodedata.normalize("NFC", "".join(out))
    else:
        base_clean = base_form

    base_starts_vowel = base_first is not None and base_first in "αεηιουωΑΕΗΙΟΥΩ"

    pv_nfd = unicodedata.normalize("NFD", preverb)
    pv_chars = list(pv_nfd)

    if base_starts_vowel:
        # Find last base char of preverb
        last_base_idx = None
        for i in range(len(pv_chars) - 1, -1, -1):
            if not unicodedata.combining(pv_chars[i]):
                last_base_idx = i
                break
        if last_base_idx is not None:
            last = pv_chars[last_base_idx]
            if last in "αεηιουωΑΕΗΙΟΥΩ":
                # Elide trailing vowel + its combining marks
                end = last_base_idx
                while end + 1 < len(pv_chars) and unicodedata.combining(
                        pv_chars[end + 1]):
                    end += 1
                pv_chars = pv_chars[:last_base_idx] + pv_chars[end + 1:]
            elif base_had_rough:
                aspirate_map = {"τ": "θ", "π": "φ", "κ": "χ"}
                if last in aspirate_map:
                    pv_chars[last_base_idx] = aspirate_map[last]
    pv_str = unicodedata.normalize("NFC", "".join(pv_chars))
    return pv_str + base_clean


def _expand_irregular_compound(headword):
    """Try to expand an irregular -μι compound (e.g. πάρειμι, εἴσειμι).

    Strategy: identify a known preverb prefix and a recognised
    irregular base (εἰμί, εἶμι, οἶδα, χρή, φημί). If it matches,
    generate forms by joining each base form with the preverb.
    Returns the set of accented forms (and their stripped variants)
    or empty set if no base matches.
    """
    preverb, base = _split_preverb(headword)
    base_plain = _strip_diacritics_lower(base)
    base_aliases = {
        "ειμι": ["εἰμί", "εἶμι"],   # ambiguous: emit forms for both
        "οιδα": ["οἶδα"],
        "χρη": ["χρή"],
        "φημι": ["φημί"],
    }
    bases = base_aliases.get(base_plain, [])
    if not bases:
        return set()
    forms = set()
    for canon in bases:
        for f in _IRREGULAR_BASE_FORMS.get(canon, ()):
            joined = _join_preverb(preverb, f)
            forms.add(joined)
            # Also emit the stripped form; lookup keys are stripped too
            forms.add(strip_diacritics(joined))
    return forms


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


def _classify_verb(headword):
    """Classify a verb into conjugation type and stem. Returns (conj_type, stem) or (None, None).

    The returned ``conj_type`` is the parameter-1 string that
    ``Module:grc-conj`` understands. We support all rule-based
    dispatches the Lua module exposes (pres-con-{a,e,o},
    pres-{ami,emi,omi,numi,lumi}, plain pres). For genuinely
    irregular athematic verbs whose Wiktionary entries use
    ``pres-irreg`` (εἰμί, εἶμι, οἶδα, χρή, and their compounds),
    we return the sentinel ``conj_type='irreg'`` so the caller can
    dispatch to ``_expand_irregular_compound`` instead of Lua.
    """
    hw_plain = strip_diacritics(headword)

    if hw_plain.endswith("εω"):
        return "pres-con-e", strip_diacritics(headword[:-2])
    elif hw_plain.endswith("αω"):
        return "pres-con-a", strip_diacritics(headword[:-2])
    elif hw_plain.endswith("οω"):
        return "pres-con-o", strip_diacritics(headword[:-2])
    # -όλλυμι (ὄλλυμι and compounds) uses pres-lumi, NOT pres-numi.
    # The Wiktionary stem strips back to '-ολ' (e.g. ἀπόλλυμι -> ἀπολ).
    # Detect *before* the more general -ννυμι rule which would otherwise
    # mis-route these to pres-numi.
    elif hw_plain.endswith("ολλυμι"):
        return "pres-lumi", strip_diacritics(headword[:-4])
    elif hw_plain.endswith("ννυμι"):
        return "pres-numi", strip_diacritics(headword[:-5])
    elif hw_plain.endswith("νυμι"):
        return "pres-numi", strip_diacritics(headword[:-4])
    # Irregular -ειμι compounds (εἰμί 'to be' / εἶμι 'to go' families).
    # These need pres-irreg with hand-coded forms; we route to the
    # irregular dispatcher rather than failing into a Lua error.
    elif hw_plain.endswith("ειμι"):
        return "irreg", ""
    # -ημι: most are pres-emi (τίθημι, ἵημι, ἵστημι), but φημί is pres-ami
    # with stem 'φ'. The single-consonant-stem case is the tell.
    elif hw_plain.endswith("ημι"):
        if len(hw_plain) == 3:
            # φ + ημι would be 4 chars; this is only ημι itself
            return "pres-emi", strip_diacritics(headword[:-3])
        return "pres-emi", strip_diacritics(headword[:-3])
    elif hw_plain.endswith("ωμι"):
        return "pres-omi", strip_diacritics(headword[:-3])
    elif hw_plain.endswith("αμι"):
        return "pres-ami", strip_diacritics(headword[:-3])
    elif hw_plain == "φημι" or hw_plain.endswith("φημι"):
        # φημί and compounds (συμφημι, σύμφημι, πρόφημι) use pres-ami
        # with consonant stem 'φ'. Only routes here if we didn't match
        # the longer -ημι suffix above (we did, so this branch is mostly
        # safety; the explicit irregular base for φημί covers it too).
        return "irreg", ""
    elif hw_plain.endswith("μι"):
        # No remaining rule-based dispatch handles a generic '-μι'
        # ending. Most surviving cases are mis-OCR'd or unusual; route
        # to the irregular dispatcher which will return empty if no
        # known base matches.
        return "irreg", ""
    elif hw_plain.endswith("ω"):
        return "pres", strip_diacritics(headword[:-1])
    elif hw_plain.endswith("μαι"):
        return "pres", strip_diacritics(headword[:-3])
    return None, None


def _build_suffix_cache(forms, stem):
    """Extract suffix patterns from expanded forms relative to the stem.

    Returns list of suffixes where each form = stem_prefix + suffix.
    The stem_prefix is the longest common prefix between stem and form.
    """
    stem_plain = strip_diacritics(stem.lower())
    suffixes = []
    for form in forms:
        form_plain = strip_diacritics(form.lower())
        # Find how much of the stem matches the beginning of the form
        match_len = 0
        for i in range(min(len(stem_plain), len(form_plain))):
            if stem_plain[i] == form_plain[i]:
                match_len = i + 1
            else:
                break
        if match_len > 0:
            suffixes.append(form_plain[match_len:])
    return suffixes


def _apply_suffix_cache(stem, suffixes, headword):
    """Apply cached suffix patterns to a new stem. Returns set of forms."""
    stem_plain = strip_diacritics(stem.lower())
    forms = set()
    for suffix in suffixes:
        form = stem_plain + suffix
        if len(form) > 1:
            forms.add(form)
    return forms


def expand_verb(wtp, headword, head_text: str = ""):
    """Expand a verb using grc-conj template. Returns set of forms.

    Always expands the present-system paradigm. When ``head_text`` is
    provided (the lead paragraph of the LSJ entry, see
    :func:`load_lsj_head_texts`), additional principal parts (fut.,
    aor., pf., aor. p., etc.) are extracted from the entry text and
    their paradigms are unioned in. Without ``head_text`` the function
    behaves exactly as before (present-system only).

    Uses a cache keyed by conjugation type: if we've already expanded a
    verb with the same conj_type, apply the suffix pattern instead of
    calling Lua.
    """
    # Reject obviously corrupt OCR'd headwords up front: they would only
    # produce Lua errors and zero forms.
    if _is_corrupt_headword(headword):
        return set(), "corrupt-headword"

    conj_type, stem = _classify_verb(headword)
    if conj_type is None:
        return set(), f"unknown-ending:{headword[-3:]}"

    # Irregular verbs (εἰμί / εἶμι / οἶδα / χρή / φημί and compounds): use
    # the hand-coded paradigm + preverb-prepending rather than Lua.
    # Principal-parts expansion is skipped for these because the
    # hand-coded paradigm already covers all attested tenses.
    if conj_type == "irreg":
        forms = _expand_irregular_compound(headword)
        if forms:
            return forms, ""
        return set(), "irregular-no-base-match"

    # Present-system expansion (with cache).
    forms: set[str] = set()
    err = ""
    if conj_type in _VERB_CACHE:
        forms = _apply_suffix_cache(stem, _VERB_CACHE[conj_type], headword)
    if not forms:
        # Cache miss or empty cached result - call Lua.
        template = "{{grc-conj|" + conj_type + "|" + stem + "}}"
        try:
            wtp.start_page(headword)
            html = wtp.expand(template)
            forms = parse_html_forms(html, headword)
        except Exception as e:
            err = str(e)
            # For -μι verbs that fail Lua dispatch, fall back to
            # pres-emi (a real conjugation type), then plain 'pres'
            # for thematic.
            if conj_type.startswith("pres-") and conj_type != "pres":
                for fallback in ["pres-emi", "pres"]:
                    if fallback == conj_type:
                        continue
                    try:
                        template = ("{{grc-conj|" + fallback + "|"
                                    + stem + "}}")
                        wtp.start_page(headword)
                        html = wtp.expand(template)
                        fallback_forms = parse_html_forms(html, headword)
                        if fallback_forms:
                            _VERB_CACHE[conj_type] = _build_suffix_cache(
                                fallback_forms, stem)
                            forms = fallback_forms
                            err = ""
                            break
                    except Exception:
                        continue
        if forms and conj_type not in _VERB_CACHE:
            _VERB_CACHE[conj_type] = _build_suffix_cache(forms, stem)

    if err and not forms:
        return set(), err

    # Principal-parts expansion (additive). When the LSJ head text
    # exposes principal parts beyond the present system, expand each
    # tense via grc-conj and union the resulting forms in. Best
    # effort: any failure is silent and falls back to the present-only
    # paradigm above.
    if head_text:
        try:
            extra = expand_principal_parts(wtp, headword, head_text)
        except Exception:
            extra = set()
        if extra:
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
        head_texts = load_lsj_head_texts()
        head_text = head_texts.get(word, "")
        if head_text:
            from lsj_principal_parts import parse_principal_parts as _ppp
            parts = _ppp(head_text, word)
            print(f"LSJ principal parts: {parts}")
        forms, err = expand_verb(wtp, word, head_text=head_text)
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
    for hw, entry in lsj_entries.items():
        if hw in wikt:
            continue  # already covered by Wiktionary
        if not entry["gender"]:
            continue  # no gender = can't decline
        if not entry["genitive"] and not infer_genitive(hw, entry["gender"], wikt_genitives):
            continue  # no genitive info at all
        candidates.append(hw)

    print(f"LSJ-only nouns to expand: {len(candidates):,}")

    wtp = get_wtp()

    stats = {"expanded": 0, "failed": 0, "new_forms": 0, "collisions": 0}
    t0 = time.time()

    for i, hw in enumerate(candidates):
        entry = lsj_entries[hw]
        forms, err = expand_noun(wtp, hw, entry["gender"], entry["genitive"],
                                 wikt_genitives=wikt_genitives)

        if err or not forms:
            stats["failed"] += 1
            continue

        stats["expanded"] += 1

        # Wiktionary's Lua modules occasionally emit forms with a misplaced
        # leading combining psili (U+0313 + base letter) for proper-noun
        # lemmas whose citation form starts with U+1FBF. sanitize_form()
        # reattaches the breathing to the base letter and NFC-composes.
        hw_clean = sanitize_form(hw)
        for raw_form in forms:
            form = sanitize_form(raw_form)
            if not form:
                continue
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


def expand_verbs():
    """Expand LSJ verbs and merge into ag_lookup.json."""
    import time

    print("Loading data...")
    lsj_entries = parse_lsj_entries()
    wikt = load_wiktionary_forms()
    print("Loading LSJ entry head texts (for principal-parts extraction)...")
    head_texts = load_lsj_head_texts()

    print(f"Loading {AG_LOOKUP}...")
    with open(AG_LOOKUP, encoding="utf-8") as f:
        lookup = json.load(f)
    original_size = len(lookup)
    print(f"  {original_size:,} existing entries")

    # Find LSJ-only verbs (entries without gender = likely verbs)
    candidates = []
    for hw, entry in lsj_entries.items():
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

    stats = {"expanded": 0, "failed": 0, "new_forms": 0, "collisions": 0,
             "with_pp": 0}
    t0 = time.time()

    for i, hw in enumerate(candidates):
        head_text = head_texts.get(hw, "")
        forms, err = expand_verb(wtp, hw, head_text=head_text)

        if err or not forms:
            stats["failed"] += 1
            continue

        stats["expanded"] += 1
        if head_text:
            # Track verbs that contributed at least one extra principal
            # part. We re-parse here rather than threading the result
            # through expand_verb to keep that signature simple.
            from lsj_principal_parts import parse_principal_parts as _ppp
            if _ppp(head_text, hw):
                stats["with_pp"] += 1

        # See sanitize_form() rationale in expand_all() above.
        hw_clean = sanitize_form(hw)
        for raw_form in forms:
            form = sanitize_form(raw_form)
            if not form:
                continue
            if form not in lookup:
                lookup[form] = hw_clean
                stats["new_forms"] += 1
            elif lookup[form] != hw_clean:
                stats["collisions"] += 1

            plain = strip_diacritics(form)
            if plain != form:
                if plain not in lookup:
                    lookup[plain] = hw_clean
                    stats["new_forms"] += 1
                elif lookup[plain] != hw_clean:
                    stats["collisions"] += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(candidates) - i - 1) / rate
            print(f"  {i+1:,}/{len(candidates):,} "
                  f"({stats['expanded']:,} ok, {stats['failed']:,} fail, "
                  f"{stats['new_forms']:,} new forms) "
                  f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Expanded: {stats['expanded']:,} / {len(candidates):,} verbs")
    print(f"  With LSJ principal parts: {stats['with_pp']:,}")
    print(f"  Failed: {stats['failed']:,}")
    print(f"  New forms added: {stats['new_forms']:,}")
    print(f"  Collisions (kept existing): {stats['collisions']:,}")
    print(f"  Lookup size: {original_size:,} -> {len(lookup):,}")

    print(f"\nSaving to {AG_LOOKUP}...")
    with open(AG_LOOKUP, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False)
    size_mb = AG_LOOKUP.stat().st_size / (1024 * 1024)
    print(f"  {size_mb:.1f} MB written")


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
    args = parser.parse_args()

    if args.setup:
        setup_wtp()
    elif args.test_one:
        test_one(args.test_one)
    elif args.test:
        test_overlap()
    elif args.expand:
        expand_all()
    elif args.expand_verbs:
        expand_verbs()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

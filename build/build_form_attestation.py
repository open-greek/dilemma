#!/usr/bin/env python3
"""Build a form-keyed corpus attestation profile + example citations.

The surface-FORM sibling of ``build_lemma_attestation.py``: keys by the exact
NFC polytonic surface form and records, per form, its usage distribution and the
passages (work + locus) it occurs in. Powers two runtime features (see
``dilemma/_attest_db.py``):

  * an "attested only" gate (input: exact form; output: grave/case-folded form);
  * ``Dilemma.form_attestation(form)`` -> usage distribution + example citations.

Sources, in dedup priority order:

  glaux > diorisis > first1k > pta > byz   (canonical-greekLit optional)

GLAUx + Diorisis are lemmatized treebanks (Phase A); First1KGreek, PTA and the
byzantine-vernacular corpus are raw-text (Phase B), added for late-antique /
patristic / Byzantine coverage. Each TLG work is counted once in the deduped
frequency (the highest-priority source that has it wins); ``source_counts`` keeps
every source's independent count; ``citations`` keep every source's passages. So
``total_count`` is intentionally NOT ``SUM(citations.count)``. TLG-bearing works
inherit GLAUx's century/genre/dialect; others use a per-source genre bucket plus
a composition century when the header (PTA) or manifest (byzantine) carries one.

Two SQLite artifacts under ``data/``: ``form_profile.db`` (base: forms, the
distribution, works, meta) and ``form_citations.db`` (opt-in: up to ``--cap``
example loci per form). Reproducibility is asserted via a logical content hash
(``meta.content_hash``) plus SHA-256 of every input.

Usage:
    python build/build_form_attestation.py                 # full build
    python build/build_form_attestation.py --stats         # report, no write
    python build/build_form_attestation.py --limit 5       # smoke (N files/source)
    python build/build_form_attestation.py --cap 200       # per-form citation cap
    python build/build_form_attestation.py --sources glaux,diorisis  # subset
"""

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent       # build/
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))
from build_lemma_attestation import (  # noqa: E402
    GENRE_ORDER, GLAUX_GENRE_MAP, DIORISIS_GENRE_MAP,
    GLAUX_POS_MAP, DIORISIS_POS_MAP,
    year_to_century, dominant_pos, is_lexical_greek, diorisis_work_id,
    fold_file_hash,
)
from extract_diorisis_lm import beta_to_nfc  # noqa: E402
import tei_locus  # noqa: E402
from dilemma._attest_db import nfc_key, norm_key, SCHEMA_VERSION  # noqa: E402

DEFAULT_GLAUX_DIR = Path.home() / "Documents" / "glaux" / "xml"
DEFAULT_METADATA = Path.home() / "Documents" / "glaux" / "metadata.txt"
DEFAULT_DIORISIS_DIR = DATA_DIR / "diorisis" / "xml"
CORPORA = Path.home() / "Documents" / "corpora"
DEFAULT_FIRST1K_DIR = CORPORA / "First1KGreek" / "data"
DEFAULT_PTA_DIR = CORPORA / "pta_data" / "data"
DEFAULT_CANONICAL_DIR = CORPORA / "canonical-greekLit" / "data"
DEFAULT_BYZ_DIR = Path.home() / "Documents" / "byzantine-vernacular-corpus" / "texts"
PROFILE_OUT = DATA_DIR / "form_profile.db"
CITATIONS_OUT = DATA_DIR / "form_citations.db"
DEFAULT_CAP = 200

# Dedup priority + default genre bucket for a source's non-GLAUx works.
ALL_SOURCES = ["glaux", "diorisis", "first1k", "pta", "byz", "canonical"]
DEFAULT_SOURCES = ["glaux", "diorisis", "first1k", "pta", "byz"]
TEI_GENRE = {"first1k": "other", "pta": "religion", "canonical": "other"}

# GLAUx div_* attributes are CUMULATIVE (the finest already embeds its
# ancestors), so the locus is the value of the single finest present division.
FINEST_FIRST_DIVS = [
    "line", "div_subsection", "div_section", "div_stephanus_section",
    "div_chapter", "div_strophe", "div_poem", "div_fragment",
    "div_stephanus_page", "div_jebb_page", "div_bekker_page",
    "div_perseus_section", "div_stephpage", "div_page", "div_letter",
    "div_part", "div_tetralogy", "div_book", "div_volume", "div_sentence",
]

# Word-final elision is written with one of these apostrophe-like marks.
_ELISION_MARKS = {"’", "'", "’", "ʼ", "᾽", "̓", "ʹ"}
_GREEK_RUN = re.compile(r"[Ͱ-Ͽἀ-῿]+['’ʼ᾽]?")


def _is_lexical_form(form: str) -> bool:
    core = form[:-1] if form and form[-1] in _ELISION_MARKS else form
    return bool(core) and is_lexical_greek(core)


def _short(div: str) -> str:
    return div[4:] if div.startswith("div_") else div


def glaux_locus(attrs) -> tuple:
    for d in FINEST_FIRST_DIVS:
        v = attrs.get(d)
        if v:
            return v, _short(d)
    for k in sorted(attrs):
        if k.startswith("div_") and attrs[k]:
            return attrs[k], _short(k)
    return None, None


class FormProfile:
    """Per-form accumulator mirroring lemma_attestation's per-lemma record."""
    __slots__ = ("total", "src_counts", "by_genre", "by_century",
                 "by_dialect", "cg", "pos")

    def __init__(self):
        self.total = 0
        self.src_counts = Counter()
        self.by_genre = Counter()
        self.by_century = Counter()
        self.by_dialect = Counter()
        self.cg = defaultdict(Counter)
        self.pos = Counter()

    def observe(self, source, pos):
        self.src_counts[source] += 1
        self.pos[pos] += 1

    def add_deduped(self, genre, century, dialect):
        self.total += 1
        self.by_genre[genre] += 1
        if century is not None:
            self.by_century[century] += 1
            self.cg[century][genre] += 1
        if dialect:
            self.by_dialect[dialect] += 1


def century_year(century, lo):
    if century is None:
        return None
    if century > 0:
        return (century - 1) * 100 + (1 if lo else 100)
    return -((-century) * 100) + (1 if lo else 100)


def load_glaux_works(path, source_sha):
    import csv
    raw = path.read_bytes()
    source_sha["glaux_metadata"] = hashlib.sha256(raw).hexdigest()
    works = {}
    reader = csv.DictReader(raw.decode("utf-8").splitlines(), delimiter="\t")
    for row in reader:
        stem = (row.get("TLG") or "").strip()
        if not stem:
            continue
        try:
            start = int((row.get("STARTDATE") or "").strip())
            end = int((row.get("ENDDATE") or "").strip())
            century = year_to_century((start + end) // 2)
        except (ValueError, TypeError):
            start = end = century = None
        works[stem] = {
            "work_id": stem, "id_scheme": "tlg", "source": "glaux",
            "author": (row.get("AUTHOR_STANDARD") or "").strip() or None,
            "title": (row.get("TITLE_STANDARD") or "").strip() or None,
            "genre": GLAUX_GENRE_MAP.get((row.get("GENRE_STANDARD") or "").strip(), "other"),
            "dialect": (row.get("DIALECT") or "").strip() or None,
            "century": century, "start_year": start, "end_year": end,
        }
    return works


def _intern(form, forms, form_ids):
    fid = form_ids.get(form)
    if fid is None:
        fid = len(forms)
        forms.append(form)
        form_ids[form] = fid
    return fid


def _ordered(counter, key):
    return {str(k): counter[k] for k in sorted(counter, key=key)}


# ---------------------------------------------------------------------------
# Citation sink (streams to a temp SQLite _raw table to bound memory)
# ---------------------------------------------------------------------------


class CiteSink:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.conn.executescript(
            "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;"
            "CREATE TABLE _raw (form_id INTEGER, work_id TEXT, source TEXT,"
            " locus TEXT, locus_scheme TEXT, count INTEGER, century INTEGER);"
        )
        self.buf = []
        self.n = 0

    def add(self, rows):
        self.buf.extend(rows)
        self.n += len(rows)
        if len(self.buf) >= 500_000:
            self.flush()

    def flush(self):
        if self.buf:
            self.conn.executemany("INSERT INTO _raw VALUES (?,?,?,?,?,?,?)", self.buf)
            self.conn.commit()
            self.buf.clear()


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def process_glaux(glaux_dir, glaux_works, forms, form_ids, profiles, works,
                  sink, claimed, limit, stats):
    h = hashlib.sha256()
    files = sorted(glaux_dir.glob("*.xml"))
    if limit:
        files = files[:limit]
    print(f"GLAUx: {len(files)} files")
    for i, xf in enumerate(files):
        stem = xf.stem
        data = xf.read_bytes()
        fold_file_hash(h, stem, data)
        meta = glaux_works.get(stem)
        genre = meta["genre"] if meta else "other"
        century = meta["century"] if meta else None
        dialect = meta["dialect"] if meta else None
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            stats["parse_errors"] += 1
            continue
        claimed.add(stem)  # GLAUx is top priority: claims every work it has
        file_cites = Counter()
        for w in root.iter("word"):
            postag = w.get("postag", "")
            if postag and postag[0] == "u":
                continue
            form = nfc_key(w.get("form", ""))
            if not _is_lexical_form(form):
                stats["glaux_nonlexical"] += 1
                continue
            fid = _intern(form, forms, form_ids)
            pos = GLAUX_POS_MAP.get(postag[0] if postag else "", "other")
            p = profiles[fid]
            p.observe("glaux", pos)
            p.add_deduped(genre, century, dialect)
            locus, scheme = glaux_locus(w.attrib)
            file_cites[(fid, locus, scheme)] += 1
            stats["glaux_tokens"] += 1
        sink.add([(fid, stem, "glaux", loc, sch, c, century)
                  for (fid, loc, sch), c in file_cites.items()])
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}, {stats['glaux_tokens']:,} tokens, "
                  f"{len(forms):,} forms", flush=True)
    return h.hexdigest()


def process_diorisis(diorisis_dir, forms, form_ids, profiles, works,
                     sink, claimed, limit, stats):
    h = hashlib.sha256()
    files = sorted(diorisis_dir.glob("*.xml"))
    if limit:
        files = files[:limit]
    print(f"Diorisis: {len(files)} files")
    for i, xf in enumerate(files):
        data = xf.read_bytes()
        fold_file_hash(h, xf.name, data)
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            stats["parse_errors"] += 1
            continue
        wid = diorisis_work_id(root, xf.name)
        deferred = wid is not None and wid in claimed
        genre_el = root.find(".//genre")
        genre = DIORISIS_GENRE_MAP.get(
            (genre_el.text or "").strip() if genre_el is not None else "", "other")
        date_el = root.find(".//creation/date")
        century = None
        if date_el is not None and date_el.text:
            try:
                century = year_to_century(int(date_el.text.strip()))
            except ValueError:
                stats["diorisis_bad_date"] += 1
        if wid and not deferred and wid not in works:
            a = root.find(".//titleStmt/author") or root.find(".//author")
            t = root.find(".//titleStmt/title") or root.find(".//title")
            works[wid] = {
                "work_id": wid, "id_scheme": "tlg", "source": "diorisis",
                "author": (a.text or "").strip() if a is not None else None,
                "title": (t.text or "").strip() if t is not None else None,
                "genre": genre, "dialect": None, "century": century,
                "start_year": century_year(century, lo=True),
                "end_year": century_year(century, lo=False),
            }
        if wid and not deferred:
            claimed.add(wid)
        cite_work = wid or xf.stem
        file_cites = Counter()
        for sent in root.iter("sentence"):
            locus = sent.get("location") or None
            for w in sent.findall("word"):
                form = beta_to_nfc(w.get("form") or "")
                if not _is_lexical_form(form):
                    stats["diorisis_nonlexical"] += 1
                    continue
                lem = w.find("lemma")
                pos = DIORISIS_POS_MAP.get(
                    (lem.get("POS") or "").lower() if lem is not None else "", "other")
                fid = _intern(form, forms, form_ids)
                pr = profiles[fid]
                pr.observe("diorisis", pos)
                if deferred:
                    stats["diorisis_evidence_tokens"] += 1
                else:
                    pr.add_deduped(genre, century, None)
                    stats["diorisis_tokens"] += 1
                file_cites[(fid, locus, "diorisis-sentence")] += 1
        sink.add([(fid, cite_work, "diorisis", loc, sch, c, century)
                  for (fid, loc, sch), c in file_cites.items()])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)}, {stats['diorisis_tokens']:,} tokens, "
                  f"{len(forms):,} forms", flush=True)
    return h.hexdigest()


def process_tei(source, tei_dir, default_genre, glaux_works, forms, form_ids,
                profiles, works, sink, claimed, limit, stats):
    """First1KGreek / canonical-greekLit / PTA: raw-text TEI with CTS loci."""
    h = hashlib.sha256()
    files = sorted(tei_dir.glob("*/*/*grc*.xml"))
    if limit:
        files = files[:limit]
    print(f"{source}: {len(files)} Greek TEI files")
    for i, xf in enumerate(files):
        data = xf.read_bytes()
        fold_file_hash(h, str(xf.relative_to(tei_dir)), data)
        root, meta = tei_locus.parse_file(xf)
        if root is None:
            stats["parse_errors"] += 1
            continue
        tlg = meta["tlg_id"]
        if tlg and tlg in glaux_works:
            gw = glaux_works[tlg]
            work_id, id_scheme = tlg, "tlg"
            genre, century, dialect = gw["genre"], gw["century"], gw["dialect"]
            author, title = gw["author"], gw["title"]
            start_y, end_y = gw["start_year"], gw["end_year"]
        else:
            work_id = tlg or (meta["urn"] or "").rsplit(":", 1)[-1] or xf.stem
            id_scheme = "tlg" if tlg else source
            century = (year_to_century(meta["creation_year"])
                       if meta["creation_year"] is not None else None)
            genre, dialect = default_genre, None
            author, title = meta["author"], meta["title"]
            start_y, end_y = century_year(century, True), century_year(century, False)
        deferred = work_id in claimed
        if not deferred and work_id not in works:
            works[work_id] = {
                "work_id": work_id, "id_scheme": id_scheme, "source": source,
                "author": author, "title": title, "genre": genre,
                "dialect": dialect, "century": century,
                "start_year": start_y, "end_year": end_y,
            }
        if not deferred:
            claimed.add(work_id)
        file_cites = Counter()
        for form, locus, scheme in tei_locus.iter_tokens(root):
            if not _is_lexical_form(form):
                stats[f"{source}_nonlexical"] += 1
                continue
            fid = _intern(form, forms, form_ids)
            p = profiles[fid]
            p.observe(source, "other")  # raw text: no POS
            if deferred:
                stats[f"{source}_evidence_tokens"] += 1
            else:
                p.add_deduped(genre, century, dialect)
                stats[f"{source}_tokens"] += 1
            file_cites[(fid, locus, scheme)] += 1
        sink.add([(fid, work_id, source, loc, sch, c, century)
                  for (fid, loc, sch), c in file_cites.items()])
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}, {stats[source + '_tokens']:,} tokens, "
                  f"{len(forms):,} forms", flush=True)
    return h.hexdigest()


_CENTURY_RE = re.compile(r"(\d+)\s*(?:st|nd|rd|th)?\s*c(?:entury|\.)", re.I)


def _byz_century(date_str):
    """'12th century (MS 15th century)' -> 12; 'BC' negates. None if unparseable."""
    if not date_str:
        return None
    m = _CENTURY_RE.search(date_str)
    if not m:
        return None
    c = int(m.group(1))
    return -c if re.search(r"\bbc\b", date_str, re.I) else c


def process_byzantine(byz_dir, forms, form_ids, profiles, works,
                      sink, claimed, limit, stats):
    from extract_byzantine import _line_is_metadata, _fix_latin_homoglyphs, _polytonic_share
    manifest_path = byz_dir / "manifest.json"
    try:
        manifest = {e["file"]: e for e in json.loads(manifest_path.read_text("utf-8"))}
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        manifest = {}
    h = hashlib.sha256()
    files = sorted(f for f in byz_dir.glob("*.txt") if f.name in manifest)
    if limit:
        files = files[:limit]
    print(f"byz: {len(files)} polytonic vernacular files")
    for xf in files:
        data = xf.read_bytes()
        text = data.decode("utf-8", "replace")
        if _polytonic_share(text) < 0.02:   # skip the monotonic editions
            stats["byz_monotonic_skipped"] += 1
            continue
        fold_file_hash(h, xf.name, data)
        entry = manifest.get(xf.name, {})
        century = _byz_century(entry.get("date"))
        work_id = xf.stem
        works[work_id] = {
            "work_id": work_id, "id_scheme": "byz", "source": "byz",
            "author": None, "title": entry.get("title"),
            "genre": "poetry", "dialect": None, "century": century,
            "start_year": century_year(century, True),
            "end_year": century_year(century, False),
        }
        claimed.add(work_id)
        file_cites = Counter()
        verse = 0
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line or _line_is_metadata(line):
                continue
            verse += 1
            for m in _GREEK_RUN.finditer(line):
                form = nfc_key(_fix_latin_homoglyphs(m.group(0)))
                if not _is_lexical_form(form):
                    stats["byz_nonlexical"] += 1
                    continue
                fid = _intern(form, forms, form_ids)
                p = profiles[fid]
                p.observe("byz", "other")
                p.add_deduped("poetry", century, None)
                stats["byz_tokens"] += 1
                file_cites[(fid, str(verse), "line")] += 1
        sink.add([(fid, work_id, "byz", loc, sch, c, century)
                  for (fid, loc, sch), c in file_cites.items()])
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_dbs(forms, profiles, works, sink, cit_tmp, cap, source_sha,
              profile_out=PROFILE_OUT, citations_out=CITATIONS_OUT):
    genre_idx = {g: i for i, g in enumerate(GENRE_ORDER)}
    sink.flush()
    cc = sink.conn

    # n_works per form (distinct works among its citations, across all sources).
    n_works = dict(cc.execute(
        "SELECT form_id, COUNT(DISTINCT work_id) FROM _raw GROUP BY form_id"))

    # --- form_profile.db ---
    tmp_profile = profile_out.with_suffix(".db.tmp")
    if tmp_profile.exists():
        tmp_profile.unlink()
    pc = sqlite3.connect(tmp_profile)
    pc.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE works (
          work_id TEXT, id_scheme TEXT, source TEXT, author TEXT, title TEXT,
          genre TEXT, dialect TEXT, century INTEGER, start_year INTEGER,
          end_year INTEGER, PRIMARY KEY (work_id, id_scheme));
        CREATE TABLE forms (
          form_id INTEGER PRIMARY KEY, form TEXT NOT NULL, form_norm TEXT NOT NULL);
        CREATE TABLE form_profile (
          form_id INTEGER PRIMARY KEY, total_count INTEGER, n_works INTEGER,
          source_counts_json TEXT, by_century_json TEXT, by_genre_json TEXT,
          by_dialect_json TEXT, century_genre_json TEXT, dominant_pos TEXT);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
    """)
    pc.executemany(
        "INSERT INTO works VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(w["work_id"], w["id_scheme"], w["source"], w["author"], w["title"],
          w["genre"], w["dialect"], w["century"], w["start_year"], w["end_year"])
         for w in sorted(works.values(), key=lambda w: (w["work_id"], w["id_scheme"]))],
    )
    pc.executemany(
        "INSERT INTO forms VALUES (?,?,?)",
        [(fid, f, norm_key(f)) for fid, f in enumerate(forms)],
    )
    prof_rows = []
    for fid, p in profiles.items():
        cg = {str(cent): _ordered(p.cg[cent], lambda g: genre_idx.get(g, 99))
              for cent in sorted(p.cg)}
        prof_rows.append((
            fid, p.total, n_works.get(fid, 0),
            json.dumps(_ordered(p.src_counts, lambda s: s), ensure_ascii=False),
            json.dumps(_ordered(p.by_century, int), ensure_ascii=False),
            json.dumps(_ordered(p.by_genre, lambda g: genre_idx.get(g, 99)), ensure_ascii=False),
            json.dumps(_ordered(p.by_dialect, lambda d: d), ensure_ascii=False),
            json.dumps(cg, ensure_ascii=False),
            dominant_pos(p.pos),
        ))
    pc.executemany("INSERT INTO form_profile VALUES (?,?,?,?,?,?,?,?,?)", prof_rows)
    pc.execute("CREATE INDEX idx_forms_form ON forms(form)")
    pc.execute("CREATE INDEX idx_forms_norm ON forms(form_norm)")
    pc.commit()

    # --- form_citations.db (cap per form, chronologically) ---
    cc.execute("CREATE TABLE citations (form_id INTEGER, work_id TEXT, source TEXT,"
               " locus TEXT, locus_scheme TEXT, count INTEGER)")
    cc.execute(
        "INSERT INTO citations "
        "SELECT form_id, work_id, source, locus, locus_scheme, count FROM ("
        "  SELECT *, ROW_NUMBER() OVER ("
        "    PARTITION BY form_id "
        "    ORDER BY (century IS NULL), century, work_id, locus) AS rn "
        "  FROM _raw) WHERE rn <= ?",
        (cap,),
    )
    cc.execute("DROP TABLE _raw")
    cc.execute("CREATE INDEX idx_citations_form ON citations(form_id)")
    cc.commit()

    # --- logical content hash (over what was actually stored) ---
    content = hashlib.sha256()
    for row in pc.execute("SELECT work_id,id_scheme,source,author,title,genre,"
                          "dialect,century,start_year,end_year FROM works "
                          "ORDER BY work_id,id_scheme"):
        content.update(repr(row).encode("utf-8"))
    for row in pc.execute("SELECT form,form_norm FROM forms ORDER BY form_id"):
        content.update(repr(row).encode("utf-8"))
    for row in pc.execute("SELECT total_count,n_works,source_counts_json,"
                          "by_century_json,by_genre_json,by_dialect_json,"
                          "century_genre_json,dominant_pos FROM form_profile "
                          "ORDER BY form_id"):
        content.update(repr(row).encode("utf-8"))
    for row in cc.execute("SELECT form_id,work_id,source,locus,locus_scheme,count "
                          "FROM citations ORDER BY form_id,work_id,locus,source"):
        content.update(repr(row).encode("utf-8"))
    content_hash = content.hexdigest()

    meta = {
        "schema_version": str(SCHEMA_VERSION),
        "sources": json.dumps(sorted({w["source"] for w in works.values()})),
        "genres": json.dumps(GENRE_ORDER),
        "citation_cap": str(cap),
        "source_sha": json.dumps(source_sha, sort_keys=True),
        "content_hash": content_hash,
        "n_forms": str(len(forms)),
        "notes": ("form-keyed corpus attestation. total/by_* are DEDUPED at the "
                  "work level by source priority; source_counts are each source's "
                  "independent count; citations keep every source. So total_count "
                  "!= SUM(citations.count). by_century is the usage-by-year axis; "
                  "century_genre_json is the heatmap joint."),
    }
    pc.executemany("INSERT INTO meta VALUES (?,?)", sorted(meta.items()))
    pc.commit()
    pc.execute("VACUUM")
    cc.execute("VACUUM")
    pc.close()
    cc.close()
    tmp_profile.replace(profile_out)
    cit_tmp.replace(citations_out)
    return content_hash


def report(stats, forms, sources):
    print(f"\nForms: {len(forms):,}")
    total = sum(stats.get(f"{s}_tokens", 0) for s in sources)
    print(f"Deduped tokens (total, counted once per work): {total:,}")
    for s in sources:
        kept = stats.get(f"{s}_tokens", 0)
        ev = stats.get(f"{s}_evidence_tokens", 0)
        nonlex = stats.get(f"{s}_nonlexical", 0)
        extra = f", {ev:,} shared-work evidence" if ev else ""
        print(f"  {s:9} {kept:>11,} tokens{extra}  (non-lexical {nonlex:,})")
    print(f"  parse errors: {stats['parse_errors']:,}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--glaux", type=Path, default=DEFAULT_GLAUX_DIR)
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--diorisis", type=Path, default=DEFAULT_DIORISIS_DIR)
    p.add_argument("--first1k", type=Path, default=DEFAULT_FIRST1K_DIR)
    p.add_argument("--pta", type=Path, default=DEFAULT_PTA_DIR)
    p.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL_DIR)
    p.add_argument("--byz", type=Path, default=DEFAULT_BYZ_DIR)
    p.add_argument("--sources", default=",".join(DEFAULT_SOURCES),
                   help="comma list from " + ",".join(ALL_SOURCES))
    p.add_argument("--cap", type=int, default=DEFAULT_CAP)
    p.add_argument("--profile-out", type=Path, default=PROFILE_OUT)
    p.add_argument("--citations-out", type=Path, default=CITATIONS_OUT)
    p.add_argument("--stats", action="store_true", help="report only, no write")
    p.add_argument("--limit", type=int, default=0, help="first N files per source")
    args = p.parse_args()
    sources = [s for s in args.sources.split(",") if s]

    t0 = time.time()
    stats = Counter()
    source_sha = {}
    forms: list = []
    form_ids: dict = {}
    profiles: dict = defaultdict(FormProfile)
    claimed: set = set()

    print("Loading GLAUx metadata...", flush=True)
    glaux_works = load_glaux_works(args.metadata, source_sha)
    works = {stem: dict(m) for stem, m in glaux_works.items()} if "glaux" in sources else {}
    print(f"  {len(glaux_works)} GLAUx works")

    cit_tmp = args.citations_out.with_suffix(".db.tmp")
    if cit_tmp.exists():
        cit_tmp.unlink()
    sink = CiteSink(str(cit_tmp))

    if "glaux" in sources:
        source_sha["glaux_xml"] = process_glaux(
            args.glaux, glaux_works, forms, form_ids, profiles, works,
            sink, claimed, args.limit, stats)
    if "diorisis" in sources:
        source_sha["diorisis_xml"] = process_diorisis(
            args.diorisis, forms, form_ids, profiles, works,
            sink, claimed, args.limit, stats)
    for src, d in (("first1k", args.first1k), ("pta", args.pta),
                   ("canonical", args.canonical)):
        if src in sources:
            source_sha[f"{src}_xml"] = process_tei(
                src, d, TEI_GENRE[src], glaux_works, forms, form_ids,
                profiles, works, sink, claimed, args.limit, stats)
    if "byz" in sources:
        source_sha["byz_txt"] = process_byzantine(
            args.byz, forms, form_ids, profiles, works,
            sink, claimed, args.limit, stats)

    report(stats, forms, sources)
    sink.flush()
    print(f"Citation rows: {sink.n:,}; works: {len(works):,}")

    if args.stats:
        sink.conn.close()
        cit_tmp.unlink(missing_ok=True)
        print(f"\n(stats only, {time.time()-t0:.1f}s)")
        return 0

    print(f"\nWriting DBs (cap={args.cap})...", flush=True)
    hsh = write_dbs(forms, profiles, works, sink, cit_tmp, args.cap, source_sha,
                    profile_out=args.profile_out, citations_out=args.citations_out)
    pmb = args.profile_out.stat().st_size / 1e6
    cmb = args.citations_out.stat().st_size / 1e6
    print(f"Wrote {args.profile_out.name} ({pmb:.1f} MB) + "
          f"{args.citations_out.name} ({cmb:.1f} MB) in {time.time()-t0:.1f}s")
    print(f"content_hash: {hsh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

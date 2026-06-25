#!/usr/bin/env python3
"""Build a form-keyed corpus attestation profile + example citations.

This is the surface-FORM sibling of ``build_lemma_attestation.py``: where that
builder keys by lemma, this one keys by the exact NFC polytonic surface form and
additionally records, per form, the passages (work + locus) it occurs in. It
powers two runtime features (see ``dilemma/_attest_db.py``):

  * an "attested only" gate (input: exact form; output: grave/case-folded form);
  * ``Dilemma.form_attestation(form)`` -> usage distribution + example citations.

Two SQLite artifacts are emitted under ``data/``:

  * ``form_profile.db``   - base install. ``forms`` (exact / norm / stripped keys),
    ``form_profile`` (uncapped per-form usage distribution: total, source_counts,
    by_century, by_genre, by_dialect, a century x genre joint for the heatmap,
    dominant_pos), ``works`` (metadata), and ``meta``.
  * ``form_citations.db`` - opt-in. ``citations``: up to ``--cap`` example loci
    per form (the full distribution lives uncapped in form_profile, so capping
    never degrades the graphs, only the number of example passages shown).

Like the lemma builder this reads only GLAUx + Diorisis (the two corpora with
clean per-text metadata AND extractable loci). The frequency distribution
(total / by_*) is DEDUPED at the work level (GLAUx preferred) so shared works
are counted once; ``source_counts`` keeps each lemmatizer's independent count;
and ``citations`` keep BOTH sources' passages (more evidence). So
``total_count`` is intentionally NOT ``SUM(citations.count)``.

Reproducibility: a .db is not byte-identical across runs, so determinism is
asserted via a SHA-256 over a canonical ORDER BY dump of the logical rows
(``meta.content_hash``), plus SHA-256 of every input file (``meta.source_sha``).

Usage:
    python build/build_form_attestation.py                 # full build
    python build/build_form_attestation.py --stats         # report, no write
    python build/build_form_attestation.py --limit 5       # smoke test
    python build/build_form_attestation.py --cap 200       # per-form citation cap
"""

import argparse
import hashlib
import json
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent       # build/
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"

# Reuse the proven lemma-builder helpers (genre/POS maps, dedup id logic, ...).
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))
from build_lemma_attestation import (  # noqa: E402
    GENRE_ORDER, GLAUX_GENRE_MAP, DIORISIS_GENRE_MAP,
    GLAUX_POS_MAP, DIORISIS_POS_MAP,
    year_to_century, dominant_pos, is_lexical_greek, diorisis_work_id,
    fold_file_hash,
)
from extract_diorisis_lm import beta_to_nfc  # noqa: E402
from dilemma._attest_db import (  # noqa: E402
    nfc_key, norm_key, SCHEMA_VERSION,
)

DEFAULT_GLAUX_DIR = Path.home() / "Documents" / "glaux" / "xml"
DEFAULT_METADATA = Path.home() / "Documents" / "glaux" / "metadata.txt"
DEFAULT_DIORISIS_DIR = DATA_DIR / "diorisis" / "xml"
PROFILE_OUT = DATA_DIR / "form_profile.db"
CITATIONS_OUT = DATA_DIR / "form_citations.db"
DEFAULT_CAP = 200

# GLAUx locus assembly. The div_* attributes are CUMULATIVE: each finer level's
# value already embeds its ancestors (div_book="1", div_chapter="1.1",
# div_section="1.1.2"; and line="1.1" is already book.line). So the locus is the
# value of the single FINEST present division, not a concatenation. This list is
# ordered finest -> coarsest; the first one present wins. Any unknown div_*
# (sorted, finest-assumed) is tried before the coarse book/volume fallbacks so a
# locus is never dropped.
FINEST_FIRST_DIVS = [
    "line", "div_subsection", "div_section", "div_stephanus_section",
    "div_chapter", "div_strophe", "div_poem", "div_fragment",
    "div_stephanus_page", "div_jebb_page", "div_bekker_page",
    "div_perseus_section", "div_stephpage", "div_page", "div_letter",
    "div_part", "div_tetralogy", "div_book", "div_volume", "div_sentence",
]


# Word-final elision is written with one of these apostrophe-like marks. A form
# that ends in one (δ’, ἀλλ’, τῶνδ’) is lexical and must be kept; is_lexical_greek
# alone rejects it because the apostrophe is not a Greek letter.
_ELISION_MARKS = {"’", "'", "’", "ʼ", "᾽", "̓", "ʹ"}


def _is_lexical_form(form: str) -> bool:
    core = form[:-1] if form and form[-1] in _ELISION_MARKS else form
    return bool(core) and is_lexical_greek(core)


def _short(div: str) -> str:
    return div[4:] if div.startswith("div_") else div


def glaux_locus(attrs) -> tuple:
    """(locus, scheme) for a GLAUx <word>: the finest present cumulative
    division's value, or (None, None) if it carries none. e.g. div_section
    "1.1.2" -> ("1.1.2", "section"); line "1.1" -> ("1.1", "line")."""
    for d in FINEST_FIRST_DIVS:
        v = attrs.get(d)
        if v:
            return v, _short(d)
    # Fallback: any other div_* we didn't enumerate (don't silently drop locus).
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
        self.cg = defaultdict(Counter)   # century -> genre -> count (heatmap)
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


def load_glaux_works(path, source_sha):
    """stem (TLG id) -> work metadata dict. Folds file bytes into source_sha."""
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


def process_glaux(glaux_dir, glaux_works, forms, form_ids, profiles,
                  raw_cites, limit, stats):
    glaux_hash = hashlib.sha256()
    work_ids = set()
    files = sorted(glaux_dir.glob("*.xml"))
    if limit:
        files = files[:limit]
    print(f"GLAUx: {len(files)} files")
    for i, xf in enumerate(files):
        stem = xf.stem
        data = xf.read_bytes()
        fold_file_hash(glaux_hash, stem, data)
        meta = glaux_works.get(stem)
        genre = meta["genre"] if meta else "other"
        century = meta["century"] if meta else None
        dialect = meta["dialect"] if meta else None
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            stats["parse_errors"] += 1
            continue
        work_ids.add(stem)
        file_cites = Counter()  # (form_id, locus, scheme) -> count, for this work
        for w in root.iter("word"):
            postag = w.get("postag", "")
            if postag and postag[0] == "u":
                continue  # punctuation
            form = w.get("form", "")
            if not form:
                continue
            form = nfc_key(form)
            if not _is_lexical_form(form):
                stats["glaux_nonlexical"] += 1
                continue
            fid = _intern(form, forms, form_ids)
            pos = GLAUX_POS_MAP.get(postag[0] if postag else "", "other")
            p = profiles[fid]
            p.observe("glaux", pos)
            p.add_deduped(genre, century, dialect)  # GLAUx preferred everywhere
            locus, scheme = glaux_locus(w.attrib)
            file_cites[(fid, locus, scheme)] += 1
            stats["glaux_tokens"] += 1
        for (fid, locus, scheme), c in file_cites.items():
            raw_cites.append((fid, stem, "glaux", locus, scheme, c, century))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)} files, {stats['glaux_tokens']:,} tokens, "
                  f"{len(forms):,} forms", flush=True)
    return glaux_hash.hexdigest(), work_ids


def century_year(century, lo):
    """Representative start/end year for a signed century (for the works table)."""
    if century is None:
        return None
    if century > 0:
        return (century - 1) * 100 + (1 if lo else 100)
    return -((-century) * 100) + (1 if lo else 100)


def _intern(form, forms, form_ids):
    fid = form_ids.get(form)
    if fid is None:
        fid = len(forms)
        forms.append(form)
        form_ids[form] = fid
    return fid


def _ordered(counter, key):
    return {str(k): counter[k] for k in sorted(counter, key=key)}


def write_dbs(forms, profiles, works, raw_cites, cap, source_sha,
              profile_out=PROFILE_OUT, citations_out=CITATIONS_OUT):
    genre_idx = {g: i for i, g in enumerate(GENRE_ORDER)}

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
          form_id INTEGER PRIMARY KEY, form TEXT NOT NULL,
          form_norm TEXT NOT NULL);
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
    # n_works per form = distinct works among its citations (both sources).
    n_works = defaultdict(set)
    for (fid, work, _src, _loc, _sch, _c, _cent) in raw_cites:
        n_works[fid].add(work)
    prof_rows = []
    for fid, p in profiles.items():
        cg = {str(cent): _ordered(p.cg[cent], lambda g: genre_idx.get(g, 99))
              for cent in sorted(p.cg)}
        prof_rows.append((
            fid, p.total, len(n_works.get(fid, ())),
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

    # --- form_citations.db (capped per form) ---
    tmp_cit = citations_out.with_suffix(".db.tmp")
    if tmp_cit.exists():
        tmp_cit.unlink()
    cc = sqlite3.connect(tmp_cit)
    cc.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE _raw (form_id INTEGER, work_id TEXT, source TEXT,
          locus TEXT, locus_scheme TEXT, count INTEGER, century INTEGER);
        CREATE TABLE citations (form_id INTEGER, work_id TEXT, source TEXT,
          locus TEXT, locus_scheme TEXT, count INTEGER);
    """)
    cc.executemany("INSERT INTO _raw VALUES (?,?,?,?,?,?,?)", raw_cites)
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
        "sources": json.dumps(["glaux", "diorisis"]),
        "genres": json.dumps(GENRE_ORDER),
        "citation_cap": str(cap),
        "source_sha": json.dumps(source_sha, sort_keys=True),
        "content_hash": content_hash,
        "n_forms": str(len(forms)),
        "notes": ("form-keyed corpus attestation. total/by_* are DEDUPED at the "
                  "work level (GLAUx preferred); source_counts are each source's "
                  "independent count; citations keep both sources. So "
                  "total_count != SUM(citations.count). by_century is the "
                  "usage-by-year axis; century_genre_json is the heatmap joint."),
    }
    pc.executemany("INSERT INTO meta VALUES (?,?)", sorted(meta.items()))
    pc.commit()
    pc.execute("VACUUM")
    cc.execute("VACUUM")
    pc.close()
    cc.close()
    tmp_profile.replace(profile_out)
    tmp_cit.replace(citations_out)
    return content_hash


def report(stats, forms, total_tokens):
    print(f"\nForms: {len(forms):,}")
    print(f"Deduped tokens (total): {total_tokens:,}")
    print(f"  glaux:    {stats['glaux_tokens']:,}")
    print(f"  diorisis: {stats['diorisis_tokens']:,} (GLAUx-absent works)")
    print(f"  diorisis evidence on shared works: {stats['diorisis_evidence_tokens']:,}")
    print(f"  glaux non-lexical:    {stats['glaux_nonlexical']:,}")
    print(f"  diorisis non-lexical: {stats['diorisis_nonlexical']:,}")
    print(f"  parse errors:         {stats['parse_errors']:,}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--glaux", type=Path, default=DEFAULT_GLAUX_DIR)
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--diorisis", type=Path, default=DEFAULT_DIORISIS_DIR)
    p.add_argument("--cap", type=int, default=DEFAULT_CAP,
                   help="max example citations stored per form")
    p.add_argument("--profile-out", type=Path, default=PROFILE_OUT)
    p.add_argument("--citations-out", type=Path, default=CITATIONS_OUT)
    p.add_argument("--stats", action="store_true", help="report only, no write")
    p.add_argument("--limit", type=int, default=0, help="first N files per corpus")
    args = p.parse_args()

    t0 = time.time()
    stats = Counter()
    source_sha = {}
    forms: list = []
    form_ids: dict = {}
    profiles: dict = defaultdict(FormProfile)
    raw_cites: list = []

    print("Loading GLAUx metadata...", flush=True)
    glaux_works = load_glaux_works(args.metadata, source_sha)
    works = {stem: dict(m) for stem, m in glaux_works.items()}
    print(f"  {len(glaux_works)} GLAUx works")

    source_sha["glaux_xml"], glaux_work_ids = process_glaux(
        args.glaux, glaux_works, forms, form_ids, profiles, raw_cites,
        args.limit, stats)
    # Re-parse Diorisis files with a sentence index so words know their locus.
    source_sha["diorisis_xml"] = _process_diorisis_with_index(
        args.diorisis, forms, form_ids, profiles, works, raw_cites,
        args.limit, stats, glaux_work_ids)

    total_tokens = stats["glaux_tokens"] + stats["diorisis_tokens"]
    report(stats, forms, total_tokens)
    # works actually cited (keep only referenced works small + honest)
    print(f"Raw citation rows: {len(raw_cites):,}; works: {len(works):,}")

    if args.stats:
        print(f"\n(stats only, {time.time()-t0:.1f}s)")
        return 0

    print(f"\nWriting DBs (cap={args.cap})...", flush=True)
    h = write_dbs(forms, profiles, works, raw_cites, args.cap, source_sha,
                  profile_out=args.profile_out, citations_out=args.citations_out)
    pmb = args.profile_out.stat().st_size / 1e6
    cmb = args.citations_out.stat().st_size / 1e6
    print(f"Wrote {args.profile_out.name} ({pmb:.1f} MB) + "
          f"{args.citations_out.name} ({cmb:.1f} MB) in {time.time()-t0:.1f}s")
    print(f"content_hash: {h}")
    return 0


def _process_diorisis_with_index(diorisis_dir, forms, form_ids, profiles, works,
                                 raw_cites, limit, stats, glaux_work_ids):
    """Wrap process_diorisis so each file's sentence index is built first."""
    dio_hash = hashlib.sha256()
    files = sorted(diorisis_dir.glob("*.xml"))
    if limit:
        files = files[:limit]
    print(f"Diorisis: {len(files)} files")
    for i, xf in enumerate(files):
        data = xf.read_bytes()
        fold_file_hash(dio_hash, xf.name, data)
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            stats["parse_errors"] += 1
            continue
        wid = diorisis_work_id(root, xf.name)
        deferred = wid is not None and wid in glaux_work_ids
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
            a = root.find(".//titleStmt/author")
            if a is None:
                a = root.find(".//author")
            t = root.find(".//titleStmt/title")
            if t is None:
                t = root.find(".//title")
            works[wid] = {
                "work_id": wid, "id_scheme": "tlg", "source": "diorisis",
                "author": (a.text or "").strip() if a is not None else None,
                "title": (t.text or "").strip() if t is not None else None,
                "genre": genre, "dialect": None, "century": century,
                "start_year": century_year(century, lo=True),
                "end_year": century_year(century, lo=False),
            }
        cite_work = wid or xf.stem
        file_cites = Counter()
        for sent in root.iter("sentence"):
            locus = sent.get("location") or None
            for w in sent.findall("word"):
                beta = w.get("form") or ""
                if not beta:
                    continue
                form = beta_to_nfc(beta)
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
        for (fid, locus, scheme), c in file_cites.items():
            raw_cites.append((fid, cite_work, "diorisis", locus, scheme, c, century))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)} files, {stats['diorisis_tokens']:,} tokens, "
                  f"{len(forms):,} forms", flush=True)
    return dio_hash.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

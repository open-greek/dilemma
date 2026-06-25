"""Pure-stdlib loader for the form-attestation SQLite artifacts.

Shared by :mod:`dilemma.core` (the input gate + ``Dilemma.form_attestation``)
and :mod:`dilemma.paradigm` (the output gate). This module imports nothing
heavy: only ``sqlite3`` + ``unicodedata`` from the stdlib, so ``paradigm.py``
stays free of torch / onnxruntime / network and ``generate()`` keeps working
standalone.

Two artifacts live under the data dir:

  * ``form_profile.db``   - ships in the base install. Holds ``forms``,
    ``form_profile`` (the per-form usage distribution), ``works`` and ``meta``.
    This drives the gate, the totals, and the usage-by-year / heatmap data.
  * ``form_citations.db`` - opt-in download. Holds ``citations``: the capped
    list of example loci (work + locus) per form.

``form_attestation()`` returns the profile whenever ``form_profile.db`` is
present and fills ``citations`` only when ``form_citations.db`` is too.

The three key functions below are the single source of truth for how a surface
form is canonicalized; the builder (``build/build_form_attestation.py``) imports
the very same functions, so the keys it stores and the keys the runtime queries
can never drift.
"""

from __future__ import annotations

import json
import os
import sqlite3
import unicodedata
from pathlib import Path

PROFILE_DB = "form_profile.db"
CITATIONS_DB = "form_citations.db"

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Canonical keys (single source of truth; the builder imports these)
# ---------------------------------------------------------------------------


def nfc_key(form: str) -> str:
    """The INPUT-gate key: the exact NFC polytonic surface form, unchanged.

    Real corpus text is matched against real corpus text, so nothing is
    folded here beyond NFC normalization.
    """
    return unicodedata.normalize("NFC", form)


def norm_key(form: str) -> str:
    """The OUTPUT-gate key: grave -> acute, then casefold.

    Generated paradigm forms are citation-style: always acute (never the
    positional grave variant), always lowercase, never sentence-initial
    capitalized. Corpus surface forms, by contrast, carry grave on ~23% of
    tokens and a capital initial on ~8%. Folding the grave to acute and
    casefolding neutralizes exactly those two systematic differences while
    preserving every other accent/breathing distinction (so it does NOT
    collapse genuine minimal pairs the way full accent-stripping would).

    This intentionally mirrors ``dilemma.core.grave_to_acute`` followed by
    ``str.casefold``; ``tests/test_form_attestation.py`` asserts they agree so
    the two definitions cannot diverge.
    """
    nfd = unicodedata.normalize("NFD", form)
    nfd = nfd.replace("̀", "́")  # COMBINING GRAVE -> COMBINING ACUTE
    return unicodedata.normalize("NFC", nfd).casefold()


def stripped_key(form: str) -> str:
    """Accent-stripped, casefolded key. A utility reserved for a future looser
    match mode; not stored in the DB and not used by the default gates.
    """
    nfd = unicodedata.normalize("NFD", form)
    base = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", base).casefold()


# ---------------------------------------------------------------------------
# Path resolution (mirrors core._resolve_data_dir without importing core)
# ---------------------------------------------------------------------------


def _candidate_data_dirs() -> list[Path]:
    dirs: list[Path] = []
    env = os.environ.get("DILEMMA_DATA_DIR")
    if env:
        dirs.append(Path(env).expanduser())
    dirs.append(Path(__file__).resolve().parent.parent / "data")  # dev tree
    dirs.append(Path.home() / ".cache" / "dilemma" / "data")      # download cache
    dirs.append(Path(__file__).resolve().parent / "data")        # bundled
    return dirs


def _find(db_name: str, data_dir: Path | None = None) -> Path | None:
    # An explicit data_dir is authoritative: look only there (clean isolation
    # for tests and for a pinned DILEMMA_DATA_DIR). Only when none is given do
    # we scan the candidate dirs.
    if data_dir is not None:
        p = Path(data_dir) / db_name
        return p if p.exists() else None
    for d in _candidate_data_dirs():
        p = d / db_name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class AttestDBMissing(RuntimeError):
    """Raised when a gated/attestation call needs ``form_profile.db`` but it
    has not been downloaded."""


class AttestDB:
    """Lazy read-only accessor over the two attestation artifacts.

    One instance caches its SQLite connections and (on first gated call) an
    in-memory presence set, so batch gating is a dict lookup rather than a
    query per word. Like ``LookupDB``, a single instance is meant for a single
    thread; the connections open with ``check_same_thread=False`` so a
    read-only multi-thread caller does not crash, but heavy concurrency should
    use one ``Dilemma`` per thread.
    """

    def __init__(self, data_dir: Path | None = None):
        self._data_dir = Path(data_dir) if data_dir else None
        self._profile_path = _find(PROFILE_DB, self._data_dir)
        self._citations_path = _find(CITATIONS_DB, self._data_dir)
        self._profile_conn: sqlite3.Connection | None = None
        self._citations_conn: sqlite3.Connection | None = None
        self._exact_set: set[str] | None = None
        self._norm_set: set[str] | None = None
        self._works: dict[str, dict] | None = None

    # -- availability ------------------------------------------------------

    @property
    def available(self) -> bool:
        """True iff ``form_profile.db`` (the gate + distribution data) exists."""
        return self._profile_path is not None

    @property
    def citations_available(self) -> bool:
        """True iff the opt-in ``form_citations.db`` (example loci) exists."""
        return self._citations_path is not None

    def require(self) -> None:
        if not self.available:
            raise AttestDBMissing(
                "form_profile.db not found. The 'attested only' / attestation "
                "feature needs it; download with "
                "`python -m dilemma download --with-attestation` (or set "
                "DILEMMA_DATA_DIR to a dir containing form_profile.db)."
            )

    # -- connections -------------------------------------------------------

    @staticmethod
    def _connect_ro(path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(
            f"file:{path}?mode=ro", uri=True, check_same_thread=False
        )
        conn.execute("PRAGMA query_only=ON")
        return conn

    def _profile(self) -> sqlite3.Connection:
        self.require()
        if self._profile_conn is None:
            self._profile_conn = self._connect_ro(self._profile_path)
        return self._profile_conn

    def _citations(self) -> sqlite3.Connection | None:
        if not self.citations_available:
            return None
        if self._citations_conn is None:
            self._citations_conn = self._connect_ro(self._citations_path)
        return self._citations_conn

    # -- gate (presence) ---------------------------------------------------

    def is_attested(self, form: str) -> bool:
        """Input gate: exact NFC polytonic surface form occurs in the corpus."""
        if self._exact_set is None:
            self._exact_set = {
                r[0] for r in self._profile().execute("SELECT form FROM forms")
            }
        return nfc_key(form) in self._exact_set

    def is_attested_norm(self, form: str) -> bool:
        """Output gate: grave/case-folded form occurs in the corpus."""
        if self._norm_set is None:
            self._norm_set = {
                r[0]
                for r in self._profile().execute("SELECT form_norm FROM forms")
            }
        return norm_key(form) in self._norm_set

    # -- works lookup ------------------------------------------------------

    def _works_map(self) -> dict[str, dict]:
        if self._works is None:
            self._works = {}
            cur = self._profile().execute(
                "SELECT work_id, id_scheme, source, author, title, genre, "
                "dialect, century, start_year, end_year FROM works"
            )
            for (wid, scheme, source, author, title, genre, dialect,
                 century, sy, ey) in cur:
                self._works[wid] = {
                    "work_id": wid, "id_scheme": scheme, "source": source,
                    "author": author, "title": title, "genre": genre,
                    "dialect": dialect, "century": century,
                    "start_year": sy, "end_year": ey,
                }
        return self._works

    # -- full attestation record ------------------------------------------

    def attestation(self, form: str, *, max_citations: int | None = 20):
        """Return the per-form attestation record for the EXACT NFC form, or
        ``None`` if unattested.

        ``max_citations`` bounds only the returned ``citations`` list
        (``None`` = all stored). ``total_count`` and the distributions always
        reflect the full evidence. ``citations`` is ``[]`` (with
        ``citations_note``) when the opt-in citations DB is absent.
        """
        row = self._profile().execute(
            "SELECT form_id, form FROM forms WHERE form = ?", (nfc_key(form),),
        ).fetchone()
        if row is None:
            return None
        return self._record(row[0], row[1], max_citations)

    def attestation_by_norm(self, form: str, *, max_citations: int | None = 20):
        """Attestation for a grave/case-folded form (the OUTPUT-gate match).

        Generated paradigm forms are citation-style (acute, lowercase), so they
        match corpus surface forms only after folding. Among the exact forms
        sharing this norm key, prefer one equal to the input, else the most
        frequent, and return its record. ``None`` if the norm is unattested.
        """
        nkey = norm_key(form)
        rows = self._profile().execute(
            "SELECT f.form_id, f.form, p.total_count "
            "FROM forms f JOIN form_profile p ON p.form_id = f.form_id "
            "WHERE f.form_norm = ?",
            (nkey,),
        ).fetchall()
        if not rows:
            return None
        exact = nfc_key(form)
        rows.sort(key=lambda r: (r[1] != exact, -(r[2] or 0)))
        form_id, exact_form, _ = rows[0]
        return self._record(form_id, exact_form, max_citations)

    def _record(self, form_id, form, max_citations):
        row = self._profile().execute(
            "SELECT total_count, n_works, source_counts_json, by_century_json, "
            "by_genre_json, by_dialect_json, century_genre_json, dominant_pos "
            "FROM form_profile WHERE form_id = ?",
            (form_id,),
        ).fetchone()
        (total, n_works, src_json, cent_json, genre_json, dial_json,
         cg_json, dom_pos) = row
        out = {
            "form": form,
            "attested": True,
            "total_count": total,
            "n_works": n_works,
            "source_counts": json.loads(src_json or "{}"),
            "by_century": {int(k): v for k, v in json.loads(cent_json or "{}").items()},
            "by_genre": json.loads(genre_json or "{}"),
            "by_dialect": json.loads(dial_json or "{}"),
            "by_century_genre": {
                int(k): v for k, v in json.loads(cg_json or "{}").items()
            },
            "dominant_pos": dom_pos,
            "citations": [],
        }
        if max_citations == 0:
            return out
        cconn = self._citations()
        if cconn is None:
            out["citations_note"] = (
                "form_citations.db not downloaded; run "
                "`python -m dilemma download --with-citations` for example loci"
            )
            return out
        out["citations"] = self._fetch_citations(cconn, form_id, max_citations)
        return out

    def _fetch_citations(self, cconn, form_id, max_citations):
        works = self._works_map()
        cites = []
        # The build-time cap already stores a chronologically-spread sample
        # (<= cap rows per form), so fetching all stored rows is cheap; we
        # sort by century here so the returned slice reads in date order.
        for work_id, source, locus, scheme, count in cconn.execute(
            "SELECT work_id, source, locus, locus_scheme, count FROM citations "
            "WHERE form_id = ?",
            (form_id,),
        ):
            w = works.get(work_id, {})
            cites.append({
                "work_id": work_id,
                "author": w.get("author"),
                "title": w.get("title"),
                "source": source,  # the annotating corpus for THIS citation
                "century": w.get("century"),
                "locus": locus,
                "locus_scheme": scheme,
                "count": count,
            })
        cites.sort(key=lambda c: (
            c["century"] if c["century"] is not None else 9999,
            c["work_id"] or "", c["locus"] or "",
        ))
        if max_citations is not None:
            cites = cites[:max_citations]
        return cites

    def close(self) -> None:
        for conn in (self._profile_conn, self._citations_conn):
            if conn is not None:
                conn.close()
        self._profile_conn = self._citations_conn = None

"""Where the kaikki.org Wiktionary dumps live, for every builder that reads them.

A dump directory has one of two layouts, and every reader accepts both:

* flat, as ``scripts/archive_kaikki_inputs.py download`` writes it and
  ``build_data.py --kaikki`` has always read it::

      <dir>/kaikki.org-en-dictionary-AncientGreek.jsonl

* nested, as kaikki.org's per-edition download pages name the files::

      <dir>/en-el/kaikki.org-dictionary-AncientGreek.jsonl

A reader names a dump by its flat file name (``DUMPS``) and calls
``resolve_dump``. The module is pure stdlib so a builder can import it
without loading ``build_data``.
"""
from __future__ import annotations

import os
from pathlib import Path

# Flat file names of the dumps build_data.py reads, per language and edition.
DUMPS = {
    "el": {
        "en": "kaikki.org-en-dictionary-Greek.jsonl",
        "el": "kaikki.org-el-dictionary-Greek.jsonl",
    },
    "grc": {
        "en": "kaikki.org-en-dictionary-AncientGreek.jsonl",
        "el": "kaikki.org-el-dictionary-AncientGreek.jsonl",
    },
    "mgr": {
        "el": "kaikki.org-el-dictionary-MedievalGreek.jsonl",
    },
}

# The Greek Wiktionary heads some Modern Greek pages as Cretan or Cypriot. The
# extract files them as languages of their own, outside the Greek file;
# build/extract_el_dialect_sections.py collects them into this one.
EL_DIALECT_DUMP = "kaikki.org-el-dictionary-GreekDialects.jsonl"

# Flat name -> path in the nested layout.
NESTED_MAP = {
    DUMPS["el"]["en"]: "en-el/kaikki.org-dictionary-Greek-words.jsonl",
    DUMPS["grc"]["en"]: "en-el/kaikki.org-dictionary-AncientGreek.jsonl",
    DUMPS["el"]["el"]: "el/kaikki.org-dictionary-Greek.jsonl",
    DUMPS["grc"]["el"]: "el/kaikki.org-dictionary-AncientGreek.jsonl",
    DUMPS["mgr"]["el"]: "el/kaikki.org-dictionary-MedievalGreek.jsonl",
    EL_DIALECT_DUMP: "el/kaikki.org-dictionary-GreekDialects.jsonl",
}

# The environment variable naming the dump directory. ``KAIKKI_DIR`` is the
# older name and is still read when the new one is unset.
ENV_VAR = "DILEMMA_KAIKKI_DIR"
LEGACY_ENV_VAR = "KAIKKI_DIR"


def resolve_dump(filename: str, dump_dir: Path) -> Path:
    """The path of ``filename`` (a flat name) in ``dump_dir``, flat layout
    first, then nested. A dump present in neither resolves to its flat path,
    so the caller's missing-file message names the documented layout."""
    dump_dir = Path(dump_dir)
    direct = dump_dir / filename
    if direct.exists():
        return direct
    nested = NESTED_MAP.get(filename)
    if nested is not None and (dump_dir / nested).exists():
        return dump_dir / nested
    return direct


def relative_dump_path(filename: str, dump_dir: Path) -> str:
    """``resolve_dump`` relative to ``dump_dir``, for portable audit records."""
    return resolve_dump(filename, dump_dir).relative_to(Path(dump_dir)).as_posix()


def layout_names(filename: str) -> set[str]:
    """Every relative path a dump may have: its flat and its nested name."""
    return {filename} | ({NESTED_MAP[filename]} if filename in NESTED_MAP else set())


def env_dump_dir() -> Path | None:
    """The dump directory the environment names, or None."""
    value = os.environ.get(ENV_VAR) or os.environ.get(LEGACY_ENV_VAR)
    return Path(value) if value else None

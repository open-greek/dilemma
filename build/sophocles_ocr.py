"""Readings of the Sophocles OCR that are settled before any head rule runs.

The digitized Sophocles lexicon (``data/sophocles/chunks``) is OCR output.
Three kinds of misreading are put right in the text itself, so that every
reader of the chunks (build/build_grc_lexical_heads.py,
build/expand_sophocles.py) and of a table an earlier expansion made from
them (overlay_lsj.py) sees the same letters:

* ``greek_mu``: the OCR prints part of its mu as the micro sign (U+00B5).
  NFC leaves that character alone, it is no Greek letter, and a key
  spelled with it matches no Greek text.
* A koronis (U+1FBD) before a capital is the smooth breathing the OCR
  usually gives as U+1FBF; ``settle_orth`` writes it that way, and the
  shared spelling rules then join it to the capital.
* ``load_head_corrections``: the reviewed corrections of single heads in
  ``data/sophocles_head_corrections.json``. Each names the ``<orth>`` text
  as the chunk prints it (micro signs aside), the head it stands for, the
  chunk and the reason. They cover what no rule can decide: an accent the
  OCR misread, and letters it gave as Latin look-alikes, as the phi
  symbol (U+03D5), as a guillemet, or dropped.
* ``load_proper_names``: the reviewed proper names of the same file. The
  lexicon capitalizes names and some common words alike, and the head
  rules read a capitalized head as a common word when the other lexica
  list only its lowercase spelling. A name that shares its spelling with a
  common word (Νοητός, Σκόπελος) is listed here, and the rows the earlier
  tables have under the name stay under it.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import unicodedata

MICRO_SIGN = "µ"
KORONIS, PSILI = "᾽", "᾿"
HEAD_CORRECTIONS = (Path(__file__).resolve().parents[1] / "data"
                    / "sophocles_head_corrections.json")
SCHEMA = 1
_SOURCE = re.compile(r"^sophocles/\d{4}\.xml$")
_ACCENTS = frozenset("̀́͂")


def greek_mu(text: str) -> str:
    """``text`` with every micro sign written as the Greek letter mu."""
    return text.replace(MICRO_SIGN, "μ") if MICRO_SIGN in text else text


def _is_greek_word(word: str) -> bool:
    return bool(word) and all(
        unicodedata.combining(char) or "GREEK" in unicodedata.name(char, "")
        for char in unicodedata.normalize("NFD", word))


def load_head_corrections(path: Path | None = None) -> dict[str, dict]:
    """The reviewed head corrections: ``{orth text: record}``.

    A record holds ``head`` (the corrected headword), ``source`` (the chunk
    that prints the entry) and ``reason``. The head must be one NFC Greek
    word with a single accent that is no grave, and the key the trimmed
    ``<orth>`` text with its micro signs already written as mu; anything
    else is refused, as is a record without a reason. A missing file means
    no corrections.
    """
    path = path or HEAD_CORRECTIONS
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA:
        raise SystemExit(f"unsupported head-correction record: {path}")
    corrections = payload.get("corrections") or {}
    for orth, record in corrections.items():
        head = record.get("head", "")
        accents = [char for char in unicodedata.normalize("NFD", head)
                   if char in _ACCENTS]
        if (not orth or orth != orth.strip() or MICRO_SIGN in orth
                or not _is_greek_word(head) or MICRO_SIGN in head
                or unicodedata.normalize("NFC", head) != head
                or len(accents) != 1 or accents[0] == "̀"
                or not _SOURCE.match(record.get("source", ""))
                or not record.get("reason", "").strip()):
            raise SystemExit(f"invalid head correction for {orth!r} in {path}")
    return corrections


def load_proper_names(path: Path | None = None) -> dict[str, dict]:
    """The reviewed proper names: ``{head: record}``.

    The head is the capitalized headword as the head rules read it (after
    any correction, its breathing joined to the capital), one NFC Greek
    word; the record holds ``source`` (the chunk that prints the entry) and
    ``reason`` (what the entry says the name is). Anything else is refused.
    A missing file or a file without the section means no names.
    """
    path = path or HEAD_CORRECTIONS
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA:
        raise SystemExit(f"unsupported head-correction record: {path}")
    names = payload.get("names") or {}
    for head, record in names.items():
        if (not _is_greek_word(head) or not head[:1].isupper()
                or head[1:] != head[1:].lower()
                or unicodedata.normalize("NFC", head) != head
                or not _SOURCE.match(record.get("source", ""))
                or not record.get("reason", "").strip()):
            raise SystemExit(f"invalid proper name {head!r} in {path}")
    return names


def corrections_sha256(corrections: dict[str, dict]) -> str:
    """Digest of the corrections as loaded (independent of file layout)."""
    import hashlib
    return hashlib.sha256(json.dumps(
        corrections, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def settle_orth(text: str, corrections: dict[str, dict] | None = None) -> str:
    """The text of an ``<orth>`` as the head rules read it.

    A reviewed correction replaces the whole text by its head. Otherwise the
    micro signs become mu and a koronis before an initial capital becomes
    the spacing smooth breathing.
    """
    text = greek_mu(text)
    record = corrections.get(text.strip()) if corrections else None
    if record:
        return record["head"]
    lead = len(text) - len(text.lstrip())
    if text[lead:lead + 1] == KORONIS and text[lead + 1:lead + 2].isupper():
        text = text[:lead] + PSILI + text[lead + 1:]
    return text

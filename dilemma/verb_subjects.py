"""Source-backed subject restrictions, separate from inflectional morphology.

The optional table records explicitly labeled dictionary senses, not estimated
probabilities or a pronoun choice for an unseen sentence. Missing evidence is
unknown. Importing this module never loads models or accesses the network.
"""

from __future__ import annotations

import copy
import json
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

FILENAME = "verb_subjects.json"
SCHEMA_VERSION = 1
VOICES = frozenset({"active", "middle", "passive", "mediopassive", "unspecified"})
SUBJECT_TYPES = frozenset({"person", "thing", "either"})


def lemma_key(lemma: str) -> str:
    """Normalize Unicode and vowel length without conflating Greek accents."""
    nfd = unicodedata.normalize("NFD", lemma)
    return unicodedata.normalize("NFC", nfd.replace("\u0304", "").replace("\u0306", ""))


def _find(data_dir: str | Path | None) -> Path | None:
    if data_dir is not None:
        candidates = [Path(data_dir).expanduser()]
    elif os.environ.get("DILEMMA_DATA_DIR"):
        candidates = [Path(os.environ["DILEMMA_DATA_DIR"]).expanduser()]
    else:
        candidates = [Path(__file__).resolve().parents[1] / "data",
                      Path.home() / ".cache" / "dilemma" / "data"]
    for directory in candidates:
        path = directory / FILENAME
        if path.is_file():
            return path
    return None


@lru_cache(maxsize=8)
def _load(path: str, mtime_ns: int, size: int) -> dict:
    # The stat values invalidate the cache when a rebuilt table replaces it.
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_inventory(data)
    return data


def validate_inventory(data: dict) -> None:
    """Reject unsupported claims or provenance instead of interpreting bad data."""
    if not isinstance(data, dict) or data.get("_meta", {}).get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported verb-subject evidence schema; rebuild the table")
    sources = data["_meta"].get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("verb-subject evidence requires source snapshots")
    for source in sources.values():
        if (not isinstance(source, dict)
                or re.fullmatch(r"[0-9a-f]{64}", source.get("sha256", "")) is None
                or not source.get("name") or not source.get("license")):
            raise ValueError("invalid verb-subject source provenance")
    for lang in ("el", "grc"):
        if not isinstance(data.get(lang), dict):
            raise ValueError(f"invalid verb-subject language table: {lang}")
        for lemma, readings in data[lang].items():
            if not lemma or not isinstance(readings, list):
                raise ValueError("invalid verb-subject lemma readings")
            for reading in readings:
                if (not isinstance(reading, dict)
                        or reading.get("voice") not in VOICES
                        or reading.get("subject_type") not in SUBJECT_TYPES | {None}
                        or reading.get("construction") not in {"personal", "impersonal", None}
                        or (reading.get("construction") == "impersonal" and reading.get("subject_type") is not None)
                        or not reading.get("sense_id")
                        or reading.get("source_id") not in sources
                        or not isinstance(reading.get("reference"), str)
                        or not reading["reference"].startswith("https://")
                        or not isinstance(reading.get("conditions"), dict)
                        or set(reading["conditions"]) - {"tense", "mood", "person", "number"}
                        or not isinstance(reading.get("evidence"), list)
                        or not reading["evidence"]
                        or any(not isinstance(label, str) or not label for label in reading["evidence"])):
                    raise ValueError(f"invalid verb-subject reading: {lemma}")


def summarize(readings: list[dict]) -> dict | None:
    """Keep every reading; the summary covers documented senses only."""
    if not readings:
        return None
    types = {r["subject_type"] for r in readings if r["subject_type"] is not None}
    unknown = any(r["subject_type"] is None and r["construction"] != "impersonal"
                  for r in readings)
    both = "either" in types or len(types) > 1
    subject_type = "either" if both else (None if unknown else next(iter(types), None))
    return {
        "subject_type": subject_type,
        "subject_types": sorted((types - {"either"}) | ({"person", "thing"} if "either" in types else set())),
        "has_unknown_readings": unknown,
        "constructions": sorted({r["construction"] for r in readings if r["construction"] is not None}),
        "coverage": "documented_senses",
        "readings": copy.deepcopy(readings),
    }


def verb_subjects(
    lemma: str, *, lang: str = "grc", voice: str = "passive",
    sense_id: str | None = None, data_dir: str | Path | None = None,
    slot=None,
) -> dict | None:
    """Return optional subject evidence for the requested voice and sense.

    ``subject_type`` is ``person``, ``thing``, ``either``, or None. An
    impersonal construction has no referential subject and is recorded as
    ``construction="impersonal"`` with a None type, never as a thing.
    ``either`` can reflect distinct senses or disagreeing sources; consult
    ``readings`` for the individual claims and provenance. These are lexical
    observations, not an exhaustive restriction on the verb or a frequency.

    Voice-unspecified readings require ``voice="unspecified"``; they do not
    silently apply to a passive. ``slot`` optionally restricts evidence to a
    ParadigmSlot, including third-person singular for finite impersonals.
    An unclassified personal reading prevents an exclusive type summary;
    ``subject_types`` still lists the explicitly supported alternatives.
    Missing tables and unsupported lemmas return None; invalid tables raise.
    """
    if lang not in {"el", "grc"}:
        raise ValueError("verb subject language must be 'el' or 'grc'")
    if voice not in VOICES:
        raise ValueError(f"unknown verb voice: {voice!r}")
    if slot is not None and (slot.pos != "verb" or slot.voice != voice):
        return None
    path = _find(data_dir)
    if path is None:
        return None
    stat = path.stat()
    data = _load(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    readings = []
    for reading in data.get(lang, {}).get(lemma_key(lemma), []):
        if reading["voice"] != voice:
            continue
        if sense_id is not None and reading["sense_id"] != sense_id:
            continue
        conditions = reading["conditions"]
        if slot is not None:
            if any(getattr(slot, key, None) != value
                   for key, value in conditions.items()):
                continue
            if (reading["construction"] == "impersonal"
                    and slot.mood not in {"infinitive", "participle"}
                    and (slot.person, slot.number) != ("3", "sg")):
                continue
        enriched = dict(reading)
        enriched["source"] = data["_meta"]["sources"][reading["source_id"]]
        readings.append(enriched)
    return summarize(readings)


def annotate_verb_entry(lemma: str, entry: dict, *, data_dir=None) -> None:
    """Add voice-keyed evidence to an Ancient verb without replacing fields."""
    if "subject_evidence" in entry and not isinstance(entry["subject_evidence"], dict):
        return
    for voice in ("active", "middle", "passive", "mediopassive", "unspecified"):
        evidence = verb_subjects(lemma, voice=voice, data_dir=data_dir)
        if evidence is not None:
            entry.setdefault("subject_evidence", {}).setdefault(voice, evidence)

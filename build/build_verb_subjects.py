#!/usr/bin/env python3
"""Extract explicit verb-subject labels from English and Greek Wiktionary.

Usage: python build/build_verb_subjects.py --kaikki kaikki

Writes data/verb_subjects.json. It does not change lookup data or paradigms.
All input editions are required for each selected language. Hashes identify
the exact source snapshots; the output has no clock-dependent fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_data import DUMPS, resolve_dump
from dilemma.verb_subjects import SCHEMA_VERSION, lemma_key, validate_inventory


_PREFIX = re.compile(r"^\s*\(([^()]*)\)\s*")
_VOICES = re.compile(r"\b(active|middle|passive|mediopassive)(?:\s+voice)?\b")
_TYPE = re.compile(
    r"^of (?:a |the )?(person|persons|people|thing|things|objects)"
    r"(?: (?:and|or) (?:a |the )?(person|persons|people|thing|things|objects))?"
    r"(?: as (?:the )?subject)?$"
)
_SUBJECT = re.compile(r"^with (?:a |the )?(person|thing) as (?:the )?subject$")
_TENSES = frozenset({"present", "imperfect", "future", "aorist", "perfect", "pluperfect"})


def _subject_type(label: str) -> str | None:
    match = _TYPE.fullmatch(label) or _SUBJECT.fullmatch(label)
    if not match:
        return None
    kinds = {"person" if word in {"person", "persons", "people"} else "thing"
             for word in match.groups() if word}
    return "either" if len(kinds) > 1 else next(iter(kinds))


def extract_sense(sense: dict) -> list[dict]:
    """Read scoped usage labels, never argument case or English gloss guesses.

    Kaikki's raw_glosses is the ordered path through a nested definition.
    A new voice starts a new subject scope: an active object's 'of things'
    label must not leak into a passive subdefinition. A child's explicit
    subject label replaces its parent's, instead of flattening their tags.
    """
    tags = set(sense.get("tags", []))
    if (sense.get("form_of") or sense.get("alt_of")
            or tags & {"form-of", "alt-of", "alternative"}):
        return []
    voices = []
    kind = None
    explicit_subject = False
    construction = None
    conditions = {}
    evidence = []
    raw_glosses = sense.get("raw_glosses") or []
    if not raw_glosses:
        # Normalized sense tags have no ancestor scope when there is no raw
        # definition path. 'personal' means non-impersonal, not human.
        voices = sorted(tags & {"active", "middle", "passive", "mediopassive"})
        evidence.extend("tag:" + voice for voice in voices)
        if "impersonal" in tags:
            construction = "impersonal"
            evidence.append("tag:impersonal")
        elif "personal" in tags:
            construction = "personal"
            evidence.append("tag:personal")
        for label in sense.get("raw_tags", []):
            if label == "για πρόσωπα" and "passive" in voices:
                kind = "person"
                construction = "personal"
                evidence.append(label)
            elif label == "για πράγματα" and "passive" in voices:
                kind = "thing"
                construction = "personal"
                evidence.append(label)
        for key, choices in {
            "tense": _TENSES,
            "mood": {"indicative", "subjunctive", "optative", "imperative", "infinitive", "participle"},
            "person": {"first-person", "second-person", "third-person"},
            "number": {"singular", "plural", "dual"},
        }.items():
            found = tags & choices
            if len(found) == 1:
                value = next(iter(found))
                conditions[key] = {"first-person": "1", "second-person": "2", "third-person": "3",
                                   "singular": "sg", "plural": "pl", "dual": "du"}.get(value, value)
    for raw in raw_glosses:
        remaining = raw
        while match := _PREFIX.match(remaining):
            label = match[1].lower()
            remaining = remaining[match.end():]
            # Only usage labels at the beginning of a definition count.
            # 'with passive sense', a quotation, or 'never impersonal' is
            # not an assertion that the current form is a passive impersonal.
            voice_label = re.match(
                r"^(?:(?:in |first |second )?(?:present |imperfect |future |aorist |perfect |pluperfect )?"
                r"(?:active|middle|passive|mediopassive)\b)", label)
            new_voices = sorted(set(_VOICES.findall(label))) if voice_label else []
            if new_voices:
                if new_voices != voices:
                    kind, construction, conditions, evidence = None, None, {}, []
                    explicit_subject = False
                voices = new_voices
                evidence.append(match[0].strip())
                tenses = _TENSES & set(label.split())
                if len(tenses) == 1:
                    conditions["tense"] = next(iter(tenses))
            for part in (p.strip() for p in label.split(",")):
                subject_type = _subject_type(part)
                # A bare 'of people' under an unmarked/transitive definition
                # can describe its object. Only a passive scope or an explicit
                # 'as subject' label establishes the grammatical subject.
                if subject_type and ("passive" in voices or "as subject" in part
                                     or "as the subject" in part):
                    kind = subject_type
                    explicit_subject = "as subject" in part or "as the subject" in part
                    construction = "personal"
                    evidence.append(match[0].strip())
                elif part.startswith("of ") and "passive" in voices:
                    # An unsupported child restriction, such as 'of animals',
                    # must not inherit its parent's human subject label.
                    kind, explicit_subject, construction = None, False, None
                    evidence.append(match[0].strip())
                if part in {"impersonal", "impersonal use"}:
                    kind, construction = None, "impersonal"
                    evidence.append(match[0].strip())
                elif part in {"personal", "personal use", "never impersonal"}:
                    construction = "personal"
                    evidence.append(match[0].strip())
    if kind is None and construction != "impersonal" and not voices:
        return []
    sense_id = sense.get("id")
    if not sense_id:
        # Stable identity for editions that omit Wiktextract sense IDs.
        body = json.dumps(sense, ensure_ascii=False, sort_keys=True).encode()
        sense_id = "sha256:" + hashlib.sha256(body).hexdigest()
    return [{
        "subject_type": kind if voice == "passive" or explicit_subject else None,
        "construction": construction if voice == "passive" or explicit_subject or kind is None else None,
        "voice": voice,
        "sense_id": sense_id,
        "conditions": conditions,
        "evidence": list(dict.fromkeys(evidence)),
    } for voice in voices or ["unspecified"]]


def build_inventory(paths: dict, only: set[str] | None = None) -> dict:
    """Merge labeled senses from every supplied edition without majority votes."""
    out = {"_meta": {"schema_version": SCHEMA_VERSION, "sources": {}},
           "el": {}, "grc": {}}
    for (lang, edition), path in sorted(paths.items()):
        path = Path(path)
        digest = hashlib.sha256()
        source_id = f"wiktionary-{edition}-{lang}"
        base = f"https://{edition}.wiktionary.org/wiki/"
        with path.open("rb") as stream:
            for raw in stream:
                digest.update(raw)
                entry = json.loads(raw)
                if entry.get("pos") != "verb" or entry.get("lang_code") != lang:
                    continue
                lemma = lemma_key(entry.get("word", ""))
                if not lemma or (only is not None and lemma not in only):
                    continue
                for sense in entry.get("senses", []):
                    for reading in extract_sense(sense):
                        reading["source_id"] = source_id
                        reading["reference"] = base + quote(entry["word"], safe="")
                        readings = out[lang].setdefault(lemma, [])
                        if reading not in readings:
                            readings.append(reading)
        out["_meta"]["sources"][source_id] = {
            "name": f"{'English' if edition == 'en' else 'Greek'} Wiktionary ({lang}), via Kaikki/Wiktextract",
            "file": path.name,
            "sha256": digest.hexdigest(),
            "url": base,
            "license": "https://creativecommons.org/licenses/by-sa/4.0/",
        }
    for lang in ("el", "grc"):
        for readings in out[lang].values():
            readings.sort(key=lambda r: json.dumps(r, ensure_ascii=False, sort_keys=True))
    validate_inventory(out)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaikki", type=Path, required=True,
                        help="directory containing both English and Greek edition dumps")
    parser.add_argument("--lang", choices=("el", "grc", "all"), default="all")
    parser.add_argument("--only", help="comma-separated lemmas for a sanity check")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "verb_subjects.json")
    args = parser.parse_args(argv)
    languages = ("el", "grc") if args.lang == "all" else (args.lang,)
    paths = {(lang, edition): resolve_dump(filename, args.kaikki)
             for lang in languages for edition, filename in DUMPS[lang].items()}
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        parser.error("missing source editions: " + ", ".join(missing))
    only = {lemma_key(word.strip()) for word in args.only.split(",")} if args.only else None
    data = build_inventory(paths, only)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8")
    print(f"Verb subject evidence: {len(data['grc'])} Ancient, {len(data['el'])} Modern lemmas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

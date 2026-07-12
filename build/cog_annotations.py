"""Reader for cog's standardized annotation exports.

cog (the Open Greek Corpus project) owns making external annotation corpora
usable and standardized: encoding normalization (NFC, U+2019 apostrophes,
standard sigma), per-source license/provenance auditing, and upstream pinning
(version DOI + checksums). dilemma consumes the standardized export and pins
it by the manifest's ``pin_line`` instead of parsing upstream treebank
formats. See the export's ``manifest.json`` for the full contract.

The first export is OGA (Opera Graeca Adnotata) v0.2.0 as ``oga-v1``:
per-work gzipped JSONL token records with form/lemma/pos/morph/head/deprel/
locus/sentence_id, all ``analysis="auto"`` (Trankit + GreTa model output -
acceptable-but-dispreferred evidence, never gold).

Consumption policy applied HERE (dilemma-side, not cog-side):
* Works annotated by the Gorman treebanks (``data/gorman_work_ids.json``)
  are skipped entirely: OGA's models trained on Gorman, so its annotations
  of those works are near-copies of the held-out gold and would contaminate
  ``eval/eval_gorman_gold.py``.
* Ellipsis placeholder tokens are skipped.
* Lemma homograph digits (λέγω3) are stripped for attestation keying,
  matching the digit-stripping the evals apply to AGDT-style lemmas.
"""

import gzip
import json
import os
import re
from pathlib import Path

DEFAULT_EXPORT = Path(os.environ.get(
    "DILEMMA_COG_OGA",
    str(Path.home() / "Documents" / "corpus-of-open-greek" / "data"
        / "annotations" / "oga" / "oga-v1")))

GORMAN_WORK_IDS_PATH = (Path(__file__).resolve().parent.parent
                        / "data" / "gorman_work_ids.json")

_TLG_RE = re.compile(r"^tlg(\d{4})\.tlg(\d{3})")
_TRAILING_DIGITS = re.compile(r"\d+$")
# OGA ellipsis placeholders surface as bracketed forms ([0], [1], ...)
_ELLIPSIS_RE = re.compile(r"^\[\d+\]$")


def load_manifest(export_dir=DEFAULT_EXPORT) -> dict | None:
    p = Path(export_dir) / "manifest.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def pin_line(manifest: dict) -> str:
    """The export identity dilemma records in its artifacts' _meta."""
    return manifest["export"]["pin_line"]


def work_tlg_stem(work_id: str) -> str | None:
    """Map a cog work_id (tlg0012.tlg001.perseus-grc2) to the GLAUx-stem
    TLG key (0012-001) dilemma's work-level dedup uses. None for non-TLG
    namespaces (ggm/pta/stoa), which cannot collide with GLAUx/Diorisis."""
    m = _TLG_RE.match(work_id)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def gorman_work_ids() -> set:
    """TLG stems of every Gorman-annotated work (held-out gold; see module
    docstring). Empty set if the committed list is missing."""
    if not GORMAN_WORK_IDS_PATH.exists():
        return set()
    with open(GORMAN_WORK_IDS_PATH, encoding="utf-8") as f:
        return set(json.load(f)["work_ids"])


def strip_homograph_digits(lemma: str) -> str:
    return _TRAILING_DIGITS.sub("", lemma)


def iter_work_tokens(export_dir, work_entry):
    """Yield token record dicts for one manifest works[] entry, skipping
    ellipsis placeholders."""
    path = Path(export_dir) / "works" / work_entry["file"]
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if _ELLIPSIS_RE.match(rec.get("form") or ""):
                continue
            yield rec


def load_oga_dating(export_dir=DEFAULT_EXPORT) -> dict:
    """Signed composition century per tlgAUTHOR.tlgWORK, derived from cog's
    OGA dating artifact (data/oga_dating.json two levels above the export).
    Empty dict when unavailable; callers fall back to GLAUx metadata or None.
    """
    p = Path(export_dir).parent.parent.parent / "oga_dating.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        works = json.load(f).get("works", {})
    out = {}
    for key, entry in works.items():
        rng = entry.get("formatted_work_date") or ""
        m = re.match(r"([+-]\d{4})-\d{2}/([+-]\d{4})-\d{2}", rng)
        if not m:
            continue
        mid = (int(m.group(1)) + int(m.group(2))) // 2
        if mid == 0:
            continue
        century = (mid - 1) // 100 + 1 if mid > 0 else -((-mid - 1) // 100 + 1)
        out[key] = century
    return out

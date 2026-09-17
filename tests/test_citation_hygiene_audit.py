"""Regression guards for lookup citation-form residue."""

import csv

from pathlib import Path

from scripts.audit_citation_hygiene import audit
from build_lookup_db import CITATION_REJECTIONS_PATH
from dilemma.core import LOOKUP_DB_PATH


def test_lookup_has_no_rejected_citation_artifact_buckets():
    """The generated lookup must not ship known non-citation lemma residue."""
    assert Path(LOOKUP_DB_PATH).exists()
    report = audit(LOOKUP_DB_PATH, example_limit=0)
    for flag in (
        "grave",
        "overline",
        "leading_combining",
        "final_elision_mark",
        "final_keraia_or_prime",
        "duplicate_tonal_marks",
        "orphaned_tonal_mark",
    ):
        assert report["counts"].get(flag, 0) == 0, flag


def test_build_rejection_report_preserves_source_provenance():
    with CITATION_REJECTIONS_PATH.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert rows
    assert rows == sorted(
        rows,
        key=lambda item: (
            item["source"], item["reason"], item["table"],
            item["lemma"], item["form"],
        ),
    )
    for item in rows:
        assert item["source"]
        assert item["table"] in {"AG", "EL", "combined"}
        assert item["form"]
        assert item["lemma"]
        assert item["reason"]

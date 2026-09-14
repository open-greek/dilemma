"""Regression guards for lookup citation-form residue."""

from pathlib import Path

from scripts.audit_citation_hygiene import audit
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
    ):
        assert report["counts"].get(flag, 0) == 0, flag

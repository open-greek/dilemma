"""Frequency summaries count exact spellings separately from ranked words."""
from __future__ import annotations

import json
import sys

import pytest

from scripts import audit_hunspell_frequency as audit


@pytest.mark.parametrize(
    ("dictionary_forms", "accepted", "rejected", "exit_code"),
    [
        (["καί", "καὶ"], 2, 1, 0),
        (["καί"], 1, 1, 1),
        (["καί", "καὶ", "στὴν"], 2, 0, 1),
    ],
)
def test_frequency_summary_counts_exact_spellings(
    tmp_path, monkeypatch, capsys, dictionary_forms, accepted, rejected, exit_code
):
    pytest.importorskip("spylls.hunspell")
    fixture = audit._payload(
        [("καί", 10), ("καὶ", 8), ("στὴν", 1)], 2, {}, {}
    )
    fixture_path = tmp_path / "frequency.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    exclusions_path = tmp_path / "exclusions.json"
    exclusions_path.write_text(
        json.dumps({
            "schema_version": 1,
            "exclusions": [{"form": "στὴν", "reason": "Modern Greek"}],
        }),
        encoding="utf-8",
    )
    dictionary_base = tmp_path / "dictionary"
    dictionary_base.with_suffix(".aff").write_text("SET UTF-8\n", encoding="utf-8")
    dictionary_base.with_suffix(".dic").write_text(
        f"{len(dictionary_forms)}\n" + "\n".join(dictionary_forms) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", [
        "audit_hunspell_frequency.py", "--source", "fixture",
        "--fixture", str(fixture_path), "--exclusions", str(exclusions_path),
        "--dictionary", str(dictionary_base),
    ])
    # Keep the real exact-spelling frequency audit. Other artifact gates are
    # independent of this summary and need the full release data.
    monkeypatch.setattr(audit, "audit_whole_artifact", lambda *_: {
        "compatibility_missing": [], "invalid_forms": [],
        "grave_without_acute": [], "citation_headwords_missing": [],
        "flags_without_identity": [], "weak_respellings": [],
    })
    monkeypatch.setattr(audit, "audit_textbook_paradigms", lambda *_: [])
    monkeypatch.setattr(audit, "measure_heldout_corpora", lambda *_: {})
    monkeypatch.setattr(audit, "load_top_lsj9_lemmas", lambda *_: set())
    monkeypatch.setattr(audit, "textbook_paradigm_forms", lambda: set())

    assert audit.main() == exit_code
    output = capsys.readouterr()
    assert output.out.splitlines()[0] == (
        "top 2 words (3 spellings): "
        f"{accepted}/2 required spellings accepted; "
        f"{rejected}/1 reviewed nonword spellings rejected (19 LM tokens)"
    )
    if accepted == 1:
        assert "missing required: καὶ" in output.err
    if rejected == 0:
        assert "accepted reviewed nonwords: στὴν" in output.err

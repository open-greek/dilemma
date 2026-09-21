"""Regression tests for OGC public-lexicon frequency import."""

import json
import sys

from build import build_cog_public_freq as cog


def test_importer_excludes_bare_elision_keys_after_stripping(tmp_path, monkeypatch):
    lexicon = tmp_path / "public_lexicon.tsv"
    lexicon.write_text(
        "\n".join((
            "δ\t10",                 # detached/lost elision mark
            "καθ\u0313\t20",           # combining koronis stripped by _key
            "ἐπ\u0313\t30",            # same, with a precomposed accent
            "δ’\t40",                 # marked elision remains a distinct key
            "τε\t50",                 # valid unaccented enclitic
            "περ\t60",                # valid unaccented particle
            "λέγω\t70",               # ordinary lexical form
            "",
        )),
        encoding="utf-8",
    )
    out = tmp_path / "cog_public_freq.json"
    monkeypatch.setattr(cog, "OUT", out)
    monkeypatch.setattr(sys, "argv", ["build_cog_public_freq.py", str(lexicon)])

    assert cog.main() == 0

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["_total_tokens"] == 220
    assert data["forms"] == {
        "δ’": [40],
        "τε": [50],
        "περ": [60],
        "λεγω": [70],
    }

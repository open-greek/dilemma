"""Build artifacts must not depend on Python's per-process hash seed."""

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_lookup_build_is_identical_across_hash_seeds(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    sources = {
        "ag_lookup.json": {"λόγος": "λόγος", "λέγει": "λέγω"},
        "mg_lookup.json": {"λόγος": "λόγος", "λέει": "λέω", "λέγει": "λέω"},
        "med_lookup.json": {},
        "vlg_headwords.json": ["ἀρχή", "βίος", "γῆ", "δῶρον", "ἔργον"],
        "wip_headwords.json": ["ζυγόν", "ἦθος", "θεός", "ἰδέα", "κόσμος"],
        "lsj10_headwords.json": ["λίθος", "μέτρον", "νόμος", "ὁδός", "πῦρ"],
        "lbg_headwords.json": [{"lemma": word} for word in
                               ("ῥόδον", "σῶμα", "τέλος", "ὕδωρ", "φῶς")],
        "etymology_bridges.json": {"λέω": ["λέγω"], "λόγος": ["λόγος"]},
        "related_lemmas.json": {"λέγω": ["λόγος"]},
    }
    for name, value in sources.items():
        (data / name).write_text(json.dumps(value, ensure_ascii=False))

    # Run the real builder with only the small fixture available. Redirect
    # all its inputs, outputs and audit report; no production data is read
    # or overwritten by this test.
    script = """
from functools import partial
from pathlib import Path
import sys
import build_lookup_db as builder

data = Path(sys.argv[1])
original_data = builder.DATA_DIR
def relocate(value):
    if isinstance(value, Path) and value.parent == original_data:
        return data / value.name
    if isinstance(value, (tuple, list)):
        return type(value)(relocate(item) for item in value)
    return value
for name, value in list(vars(builder).items()):
    setattr(builder, name, relocate(value))
builder.DATA_DIR = data
builder._write_citation_rejection_report = partial(
    builder._write_citation_rejection_report,
    path=builder.CITATION_REJECTIONS_PATH,
)
builder.build()
"""
    results = []
    for seed in (1, 17, 73):
        result = subprocess.run(
            [sys.executable, "-c", script, str(data)], cwd=ROOT,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        results.append({
            name: hashlib.sha256((data / name).read_bytes()).hexdigest()
            for name in ("lookup.db", "spell_index.db",
                         "citation_hygiene_rejections.tsv")
        })
    assert results[1:] == results[:-1]

    # Verify that all four independent headword sources actually took part,
    # and that language-specific rows and bridge tables were built as well.
    with sqlite3.connect(data / "lookup.db") as conn:
        rows = dict(conn.execute(
            "SELECT form, lemmas.text FROM lookup JOIN lemmas "
            "ON lookup.lemma_id = lemmas.id WHERE lang = 'all'"
        ))
        for name in ("vlg_headwords.json", "wip_headwords.json",
                     "lsj10_headwords.json"):
            assert all(rows[word] == word for word in sources[name])
        assert all(rows[row["lemma"]] == row["lemma"]
                   for row in sources["lbg_headwords.json"])
        assert conn.execute("SELECT count(*) FROM bridges").fetchone()[0] == 2
        assert conn.execute("SELECT count(*) FROM related_lemmas").fetchone()[0] == 1
        assert conn.execute(
            "SELECT lemmas.text FROM lookup JOIN lemmas "
            "ON lookup.lemma_id = lemmas.id WHERE form = ? AND lang = 'el'",
            ("λέγει",),
        ).fetchone() == ("λέω",)

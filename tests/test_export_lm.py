#!/usr/bin/env python3
"""Guards for the n-gram LM exporter (``export_lm.py``).

The exporter cuts every context to its most frequent continuations, and
keeps past that cut the spellings of homograph sets (ἢ and ἡ, ἐκείνῃ and
ἐκείνη) seen at least ``KEEP_HOMOGRAPH_MIN_COUNT`` times in the context, so
a keyboard choosing between two spellings of the same letters scores both
from the same row. These tests pin the grouping, the cut, and an end-to-end
export of a small hand-made set of counts, read back with ``eval_lm.py``'s
reader. They need no corpus on disk.

Run with:
    python -m pytest tests/test_export_lm.py -x -v
"""

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import export_lm  # noqa: E402
from eval_lm import NgramLM  # noqa: E402
from export_lm import (  # noqa: E402
    bare_letters,
    homograph_spelling_ids,
    keep_past_cut,
    spelling_key,
)


def test_bare_letters_is_what_a_bare_keyboard_types():
    assert bare_letters("ἢ") == bare_letters("ἡ") == bare_letters("Ἡ") == "η"
    assert bare_letters("ἐκείνῃ") == bare_letters("ἐκείνη") == "εκεινη"
    assert bare_letters("λόγος") == "λογοσ"
    assert bare_letters("δ’") == bare_letters("δ᾽") == "δ"
    assert bare_letters("προϊών") == "προιων"


def test_spelling_key_folds_grave_case_and_elision_glyph_only():
    assert spelling_key("καὶ") == spelling_key("καί") == spelling_key("Καί")
    assert spelling_key("δ᾽") == spelling_key("δ’")
    assert spelling_key("ἢ") != spelling_key("ἡ")
    assert spelling_key("ἐκείνῃ") != spelling_key("ἐκείνη")
    assert spelling_key("αὐτοῦ") != spelling_key("αὑτοῦ")


def test_homograph_spellings_differ_by_more_than_grave_case_or_glyph():
    vocab = ["</s>", "<PAD>", "<UNK>", "<s>",
             "ἢ", "ἡ", "Ἡ", "καί", "καὶ", "δ’", "δ᾽",
             "ἐκείνη", "ἐκείνῃ", "λόγος"]
    ids = homograph_spelling_ids(vocab)
    assert {vocab[i] for i in ids} == {"ἢ", "ἡ", "Ἡ", "ἐκείνη", "ἐκείνῃ"}


def test_keep_past_cut_appends_kept_spellings_in_count_order():
    entries = [(10, 50), (11, 40), (12, 9), (13, 5), (14, 3), (15, 2)]
    assert keep_past_cut(entries, 2) == entries[:2]
    assert keep_past_cut(entries, 2, {13, 14, 15}, 0) == entries[:2]
    assert keep_past_cut(entries, 2, {13, 14, 15}, 3) == [
        (10, 50), (11, 40), (13, 5), (14, 3)]
    assert keep_past_cut(entries, 2, {13, 14, 15}, 1) == [
        (10, 50), (11, 40), (13, 5), (14, 3), (15, 2)]
    # A kept id inside the cut is not repeated.
    assert keep_past_cut(entries, 2, {10}, 1) == entries[:2]


# --- end to end ---------------------------------------------------------------

# After τῇ: 31 ordinary continuations fill the bigram row past its cut of
# 30, then the dative ἐκείνῃ (3 times) and the plain word λόγῳ (3 times).
# The nominative ἐκείνη follows τῇ once; it and the dative are the
# homograph pair. After ἐν τῇ the same holds for the trigram row (cut 15).
LETTERS = "αβγδεζηθικλμνξοπρστυφχψω"
FILLERS = ([f"φ{c}" for c in LETTERS] + [f"ψ{c}" for c in LETTERS])[:31]


def _write_counts(root: Path) -> None:
    vocab = ["<PAD>", "<UNK>", "<s>", "</s>", "ἐν", "τῇ", "ἐκείνῃ",
             "ἐκείνη", "λόγῳ"] + FILLERS
    ids = {t: i for i, t in enumerate(vocab)}
    bigrams = {(ids["ἐν"], ids["τῇ"]): 400}
    trigrams = {}
    for n, w in enumerate(FILLERS):
        bigrams[(ids["τῇ"], ids[w])] = 100 - n
        if n < 16:
            trigrams[(ids["ἐν"], ids["τῇ"], ids[w])] = 20 - n
    bigrams[(ids["τῇ"], ids["ἐκείνῃ"])] = 3
    bigrams[(ids["τῇ"], ids["λόγῳ"])] = 3
    bigrams[(ids["τῇ"], ids["ἐκείνη"])] = 1
    trigrams[(ids["ἐν"], ids["τῇ"], ids["ἐκείνῃ"])] = 3
    trigrams[(ids["ἐν"], ids["τῇ"], ids["λόγῳ"])] = 3
    unigrams = {i: 5 for i in range(1, len(vocab))}
    unigrams[ids["τῇ"]] = 3000
    unigrams[ids["ἐν"]] = 1000
    (root / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False),
                                     encoding="utf-8")
    (root / "unigrams.json").write_text(json.dumps(unigrams), encoding="utf-8")
    with gzip.open(root / "bigrams.tsv.gz", "wt", encoding="utf-8") as f:
        for (a, b), c in bigrams.items():
            f.write(f"{a}\t{b}\t{c}\n")
    with gzip.open(root / "trigrams.tsv.gz", "wt", encoding="utf-8") as f:
        for (a, b, c), k in trigrams.items():
            f.write(f"{a}\t{b}\t{c}\t{k}\n")


def _export(root: Path, name: str, *extra: str) -> tuple[Path, dict]:
    out = root / f"{name}.bin"
    version = root / f"{name}.version"
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "export_lm.py"),
         "--in", str(root), "--out", str(out), "--version-out", str(version),
         *extra],
        check=True, capture_output=True,
    )
    return out, json.loads(version.read_text(encoding="utf-8"))


def _rows(path: Path) -> tuple[list[str], list[str]]:
    lm = NgramLM(path)
    try:
        tok = lm._vocab
        bi = [tok[w] for w, _ in lm._bigram_lookup(lm.id_of("τῇ"))]
        tri = [tok[w] for w, _ in
               lm._trigram_lookup(lm.id_of("ἐν"), lm.id_of("τῇ"))]
    finally:
        lm.close()
    return bi, tri


@pytest.fixture(scope="module")
def exports(tmp_path_factory):
    root = tmp_path_factory.mktemp("lm")
    _write_counts(root)
    return {
        "default": _export(root, "default"),
        "off": _export(root, "off", "--keep-homograph-min-count", "0"),
        "one": _export(root, "one", "--keep-homograph-min-count", "1"),
    }


def test_default_export_keeps_homograph_spellings_seen_three_times(exports):
    bi, tri = _rows(exports["default"][0])
    assert len(bi) == export_lm.TOP_K_BI + 1 and bi[-1] == "ἐκείνῃ"
    assert len(tri) == export_lm.TOP_K_TRI + 1 and tri[-1] == "ἐκείνῃ"
    # An ordinary word with the same count is still cut, and the
    # nominative, seen once, is below the default minimum.
    assert "λόγῳ" not in bi and "λόγῳ" not in tri and "ἐκείνη" not in bi
    info = exports["default"][1]
    assert info["keep_homograph_min_count"] == export_lm.KEEP_HOMOGRAPH_MIN_COUNT == 3
    assert info["homograph_spellings"] == 2


def test_minimum_count_one_keeps_every_seen_homograph_spelling(exports):
    bi, _ = _rows(exports["one"][0])
    assert bi[export_lm.TOP_K_BI:] == ["ἐκείνῃ", "ἐκείνη"]


def test_rule_off_exports_the_plain_top_k(exports):
    bi, tri = _rows(exports["off"][0])
    assert bi == FILLERS[:export_lm.TOP_K_BI]
    assert tri == FILLERS[:export_lm.TOP_K_TRI]
    assert exports["off"][1]["homograph_spellings"] == 0


def test_kept_entries_leave_the_top_k_and_its_order_unchanged(exports):
    """Next-word suggestions read a row's first entries, so they must be
    the same with and without the rule."""
    bi_off, tri_off = _rows(exports["off"][0])
    bi_on, tri_on = _rows(exports["default"][0])
    assert bi_on[:export_lm.TOP_K_BI] == bi_off
    assert tri_on[:export_lm.TOP_K_TRI] == tri_off
    lm = NgramLM(exports["default"][0])
    try:
        probs = [p for _, p in lm._bigram_lookup(lm.id_of("τῇ"))]
    finally:
        lm.close()
    assert probs == sorted(probs, reverse=True)

#!/usr/bin/env python3
"""Guards for the n-gram LM exporter (``export_lm.py``).

The exporter cuts every context to its most frequent continuations, and
keeps past that cut the spellings of homograph sets (ἢ and ἡ, ἐκείνῃ and
ἐκείνη) seen at least ``KEEP_HOMOGRAPH_MIN_COUNT`` times in the context, so
a keyboard choosing between two spellings of the same letters scores both
from the same row. It also appends a table of exact training counts for the
out-of-vocabulary spellings a dictionary proposes against another spelling
of the same letters, so such spellings are no longer all tied at nothing.
These tests pin the grouping, the cut, the table's selection and layout,
and an end-to-end export of a small hand-made set of counts and dictionary,
read back with ``eval_lm.py``'s reader. They need no corpus on disk.

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
    contested_lookup_keys,
    fnv1a64,
    homograph_spelling_ids,
    keep_past_cut,
    lookup_key,
    read_hunspell_words,
    read_oov_unigram_table,
    select_oov_unigrams,
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


# --- out-of-vocabulary unigram table -----------------------------------------

def test_fnv1a64_matches_the_reference_vectors():
    assert fnv1a64("") == 0xCBF29CE484222325
    assert fnv1a64("a") == 0xAF63DC4C8601EC8C
    assert fnv1a64("foobar") == 0x85944171F73967E8


def test_lookup_key_lowercases_and_writes_elision_as_the_corpus_does():
    assert lookup_key("Ἡμέρᾳ") == "ἡμέρᾳ"
    assert lookup_key("Δ᾽") == lookup_key("δ'") == "δ\u2019"


def _write_dictionary(root: Path) -> Path:
    """θεά/θεάς come from one flagged stem, θέα/θέας are plain lines, so
    the contested pairs cross the affix expansion; ἀγαθός has no twin."""
    (root / "d.aff").write_text(
        "SET UTF-8\nFLAG num\nSFX 1000 Y 2\nSFX 1000 0 0 .\n"
        "SFX 1000 0 ς .\n", encoding="utf-8")
    dic = root / "d.dic"
    dic.write_text("6\nθεά/1000\tfr:R\nθέα\tfr:C\nθέας\tfr:C\n"
                   "ἡμέρᾳ\tfr:C\nἡμέρα\tfr:C\nἀγαθός\tfr:C\n",
                   encoding="utf-8")
    return dic


def test_dictionary_words_are_expanded_through_the_affix_rules(tmp_path):
    words = read_hunspell_words(_write_dictionary(tmp_path))
    assert words == {"θεά", "θεάς", "θέα", "θέας", "ἡμέρᾳ", "ἡμέρα", "ἀγαθός"}


def test_contested_spellings_share_letters_with_another_spelling(tmp_path):
    words = read_hunspell_words(_write_dictionary(tmp_path))
    assert contested_lookup_keys(words) == {
        "θεά", "θεάς", "θέα", "θέας", "ἡμέρᾳ", "ἡμέρα"}
    # A grave or a capital is not another spelling.
    assert contested_lookup_keys({"καί", "καὶ", "Καί"}) == set()


# Training counts: in-vocabulary ἐκείνη, the contested out-of-vocabulary
# spellings (one capitalized), the uncontested ἀγαθός, and a θεάς count
# over the u8 cap.
TYPE_COUNTS = [("ἐκείνη", 5), ("θεάς", 300), ("ἀγαθός", 9), ("θέας", 7),
               ("ἡμέρᾳ", 4), ("Ἡμέρᾳ", 3), ("ἡμέρα", 1)]


def test_table_selects_contested_out_of_vocabulary_spellings(tmp_path):
    keys = contested_lookup_keys(read_hunspell_words(_write_dictionary(tmp_path)))
    table = select_oov_unigrams(TYPE_COUNTS, {"ἐκείνη"}, 1, keys)
    assert [h for h, _ in table] == sorted(h for h, _ in table)
    assert dict(table) == {fnv1a64("θεάς"): 255, fnv1a64("θέας"): 7,
                           fnv1a64("ἡμέρᾳ"): 4, fnv1a64("Ἡμέρᾳ"): 3,
                           fnv1a64("ἡμέρα"): 1}
    two = select_oov_unigrams(TYPE_COUNTS, {"ἐκείνη"}, 2, keys)
    assert fnv1a64("ἡμέρα") not in dict(two) and len(two) == 4
    assert select_oov_unigrams(TYPE_COUNTS, {"ἐκείνη"}, 0, keys) == []


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
    with gzip.open(root / "type_counts.tsv.gz", "wt", encoding="utf-8") as f:
        for tok, c in TYPE_COUNTS:
            f.write(f"{tok}\t{c}\n")
    _write_dictionary(root)


def _export(root: Path, name: str, *extra: str) -> tuple[Path, dict]:
    out = root / f"{name}.bin"
    version = root / f"{name}.version"
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "export_lm.py"),
         "--in", str(root), "--out", str(out), "--version-out", str(version),
         "--oov-dictionary", str(root / "d.dic"), *extra],
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
        "no_table": _export(root, "no_table", "--oov-min-count", "0"),
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


def test_default_export_appends_the_table_after_an_unchanged_model(exports):
    with_table = exports["default"][0].read_bytes()
    without = exports["no_table"][0].read_bytes()
    table = read_oov_unigram_table(with_table)
    assert table == {fnv1a64("θεάς"): 255, fnv1a64("θέας"): 7,
                     fnv1a64("ἡμέρᾳ"): 4, fnv1a64("Ἡμέρᾳ"): 3,
                     fnv1a64("ἡμέρα"): 1}
    assert read_oov_unigram_table(without) == {}
    # The header and every section before the table are byte-identical,
    # so a reader that does not know the table reads the same model.
    assert with_table[:len(without)] == without
    padding = -len(without) % 8
    assert len(with_table) == len(without) + padding + 9 * 5 + 16
    info = exports["default"][1]
    assert info["oov_unigram_min_count"] == export_lm.OOV_UNIGRAM_MIN_COUNT
    assert info["oov_unigrams"] == 5
    assert exports["no_table"][1]["oov_unigrams"] == 0
    lm = NgramLM(exports["default"][0])
    try:
        assert lm._vocab[lm.id_of("τῇ")] == "τῇ"
    finally:
        lm.close()


def test_export_without_type_counts_stops_unless_the_table_is_off(tmp_path):
    _write_counts(tmp_path)
    (tmp_path / "type_counts.tsv.gz").unlink()
    run = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "export_lm.py"),
         "--in", str(tmp_path), "--out", str(tmp_path / "x.bin"),
         "--version-out", str(tmp_path / "x.version"),
         "--oov-dictionary", str(tmp_path / "d.dic")],
        capture_output=True, text=True,
    )
    assert run.returncode != 0 and "type_counts.tsv.gz" in run.stderr
    _export(tmp_path, "x", "--oov-min-count", "0")

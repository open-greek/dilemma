"""Spelling rules, syllable counting, rating edges, admission routes and the
output format of the monotonic Modern Greek word list
(build/el_lexicon.py, build/el_type_counts.py, export_el_monotonic.py).

Every input is a small file written by the test: a few Wiktionary rows, a
lookup.db with three rows, a FrequencyWords file and type-count tables
with chosen counts. Nothing is downloaded and no corpus is read.

Run with:
    python -m pytest tests/test_el_monotonic_list.py -x -v
"""

import gzip
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import export_el_monotonic as ex  # noqa: E402
from dilemma_build import el_lexicon as L  # noqa: E402
from dilemma_build import el_type_counts as TC  # noqa: E402

A = L.APOSTROPHE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -- spelling ----------------------------------------------------------------

@pytest.mark.parametrize("form", [
    "είναι", "ή", "πού", "πώς", "ποιος", "μια", "γεια", "σπίτια", "κορόιδο",
    "προϊόν", "ταΐζω", "Ευρωπαϊκή", "τσάι", "ρολόι", "Αθήνα", "άνθρωπος",
    "και", "να", "σ" + A, "απ" + A, "είν" + A, A + "ναι", A + "θελες",
])
def test_spelling_accepts(form):
    assert L.orthography_reason(form) is None


@pytest.mark.parametrize("form,reason", [
    ("ναί", "accented_monosyllable"),
    ("δέν", "accented_monosyllable"),
    ("θά", "accented_monosyllable"),
    ("ποιός", "accented_synizesis"),
    ("μιά", "accented_synizesis"),
    ("όνομά", "double_accent"),
    ("ειναι", "unaccented_polysyllable"),
    ("vα", "latin"),
    ("ϊσος", "dialytika_without_vowel_before"),
    ("τσάϊ", "unneeded_dialytika"),
    ("πρωϊνό", "unneeded_dialytika"),
    ("ΗΠΑ", "all_caps"),
    ("καλόσ", "final_sigma"),
    ("καλςο", "medial_final_sigma"),
    ("ἄνθρωπος", "not_monotonic_greek"),
    ("έυκολο", "accent_too_early"),
    ("ναιιι", "repeated_letters"),
    ("μεταξ" + A, "unaccented_elision"),
    ("απ" + A + "το", "apostrophe_inside"),
    ("ΑθήΝα", "mixed_case"),
])
def test_spelling_rejects(form, reason):
    assert L.orthography_reason(form) == reason


def test_one_letter_elisions_need_the_mark():
    assert L.orthography_reason("σ" + A) is None
    assert L.orthography_reason("σ") == "final_sigma"
    assert L.orthography_reason("τ") == "no_vowel"


def test_lexicon_exceptions_for_synizesis_and_loanwords():
    wiktionary = {"ποιος", "μια"}
    assert ex._spelling_ok("ιός", wiktionary) is None          # no ιος twin
    assert ex._spelling_ok("ποιός", wiktionary) == "accented_synizesis"
    assert ex._spelling_ok("κάμεραμαν", wiktionary) is None
    assert L.orthography_reason("κάμεραμαν") == "accent_too_early"


# -- syllables ---------------------------------------------------------------

@pytest.mark.parametrize("word,syllables", [
    ("κορόιδο", 3), ("χάιδεψε", 3), ("ρολόι", 3), ("τσάι", 2), ("Μάιος", 2),
    ("σπίτια", 2), ("βοήθεια", 3), ("μια", 1), ("μία", 2), ("ποιος", 1),
    ("γεια", 1), ("πιο", 1), ("δυο", 1), ("προϊόν", 3), ("άνθρωπος", 3),
    ("υιός", 2), ("ουρανός", 3), ("αυτοκίνητο", 5),
])
def test_syllable_count_with_synizesis(word, syllables):
    assert L.syllable_count(word) == syllables


def test_accent_position():
    assert L.accent_position("άνθρωπος") == 3
    assert L.is_proparoxytone("άνθρωπος")
    assert L.accent_position("σπίτια") == 2
    assert L.accent_position("ουρανός") == 1


@pytest.mark.parametrize("a,b,same", [
    ("κάθισε", "κάθησε", True), ("γλυκιά", "γλυκειά", True),
    ("νιώθεις", "νοιώθεις", True), ("σέξι", "σέξυ", True),
    ("Ελλάδα", "Ελάδα", True), ("τσάι", "τσε", False),
    ("προϊόν", "προιόν", False),
])
def test_sound_key(a, b, same):
    assert (L.sound_key(a) == L.sound_key(b)) is same


def test_bare_key_is_the_stripped_key():
    assert L.bare_key("Προϊόντα") == "προιοντα"
    assert L.bare_key("καλός") == "καλοσ"


# -- ratings -----------------------------------------------------------------

@pytest.mark.parametrize("rate,bucket", [
    (1000.0, "C"), (37.0, "C"), (36.999, "M"), (3.7, "M"), (3.699, "R"),
    (0.0001, "R"), (0.0, "X"),
])
def test_bucket_edges(rate, bucket):
    assert L.bucket_for_rate(rate) == bucket


def test_rate_per_million():
    assert L.rate_per_million(37, 1_000_000) == 37.0
    assert L.bucket_for_rate(L.rate_per_million(37, 1_000_000)) == "C"
    assert L.bucket_for_rate(L.rate_per_million(36, 1_000_000)) == "M"


def test_wilson_lower_bound():
    assert L.wilson_lower(0, 0) == 0.0
    assert 0.80 < L.wilson_lower(24, 25) < 0.81
    assert L.wilson_lower(10, 10) < 1.0


# -- reviewed data files -----------------------------------------------------

def test_every_function_word_passes_the_spelling_check():
    words = ex.load_function_words(ex.FUNCTION_WORDS)
    assert {"και", "ο", "ή", "να", "θα", "πού", "πώς"} <= set(words)
    for word in words:
        assert L.orthography_reason(word) is None, word


def test_review_file_loads_and_explains_every_entry():
    data = json.loads(ex.REVIEW.read_text(encoding="utf-8"))
    review = ex.load_review(ex.REVIEW)
    assert set(review) == {"rejects", "corpus_accepts", "elision_accepts",
                           "abbreviation_rejects", "acronym_rejects"}
    for key in review:
        for entry in data.get(key, []):
            assert entry["reason"].strip(), entry


# -- elisions ------------------------------------------------------------------

def neighbors(next_word=0, next_vowel=0, next_clitic=0, prev_word=0, prev_vowel=0):
    return [next_word, next_vowel, next_clitic, prev_word, prev_vowel]


@pytest.fixture
def elision_tables():
    attested = {"από", "είναι", "έχει", "εκεί", "κατά", "άδηλος", "εδώ",
                "καλό", "μεταξύ", "εξ", "τους", "έγραψα"}
    return ex.elision_index(attested)


def test_elision_needs_its_full_word_and_a_vowel_after(elision_tables):
    attested, after = elision_tables
    admitted = attested
    good = neighbors(next_word=100, next_vowel=95)
    assert ex.elision_reason("απ" + A, attested, after, admitted, set(), good) is None
    assert ex.elision_reason("κατ" + A, attested, after, admitted, set(), good) is None
    assert ex.elision_reason("είν" + A, attested, after, admitted, set(), good) is None
    # μεταξ’: unaccented, from a word accented on the dropped vowel
    assert ex.elision_reason("μεταξ" + A, attested, after, admitted, set(), good) is None
    # colloquial elision before a weak pronoun counts as an elision context
    clitic = neighbors(next_word=100, next_vowel=40, next_clitic=55)
    assert ex.elision_reason("απ" + A, attested, after, admitted, set(), clitic) is None
    # a numbering mark: stands before anything
    numbering = neighbors(next_word=40, next_vowel=8, next_clitic=2)
    assert (ex.elision_reason("καλ" + A, attested, after, admitted, set(), numbering)
            == "elision_not_before_a_vowel")
    # η and ω are not elided: εδ’ is no elision of εδώ
    assert (ex.elision_reason("εδ" + A, attested, after, admitted, set(), good)
            == "elision_without_full_word")
    # a word that does not elide, unless reviewed
    assert (ex.elision_reason("εξ" + A, attested, after, admitted, set(), good)
            == "elision_of_a_word")
    # a reviewed acceptance waives only that check; the full word is still needed
    assert (ex.elision_reason("εξ" + A, attested, after, admitted, {"εξ" + A}, good)
            == "elision_without_full_word")
    # a closing quotation mark after a word in ς
    assert (ex.elision_reason("τουσ" + A, attested, after, admitted, set(), good)
            == "quotation_mark_after_a_word")


def test_aphaeresis_needs_its_full_word_and_a_vowel_before(elision_tables):
    attested, after = elision_tables
    after_vowel = neighbors(prev_word=40, prev_vowel=40)
    assert ex.elision_reason(A + "ναι", attested, after, attested, set(), after_vowel) is None
    assert ex.elision_reason(A + "χει", attested, after, attested, set(), after_vowel) is None
    assert ex.elision_reason(A + "κει", attested, after, attested, set(), after_vowel) is None
    assert ex.elision_reason(A + "γραψα", attested, after, attested, set(),
                             after_vowel) is None
    # an opening quotation mark follows anything
    quote = neighbors(prev_word=100, prev_vowel=65)
    assert (ex.elision_reason(A + "χει", attested, after, attested, set(), quote)
            == "aphaeresis_not_after_a_vowel")
    # the accent must match what is left of the full word: άδηλος -> δηλος
    assert (ex.elision_reason(A + "δήλος", attested, after, attested, set(), after_vowel)
            == "elision_without_full_word")


# -- abbreviation shapes ---------------------------------------------------------

@pytest.mark.parametrize("spelling,rejected", [
    ("π.χ.", False), ("κ.λπ.", False), ("σελ.", False), ("μ.Χ.", False),
    ("κ.", False), ("ά.", True), ("ς.", True), ("του.Β.", True),
])
def test_abbreviation_shapes(spelling, rejected):
    words = {"του", "λόγο", "σελίδα"}
    assert (ex.abbreviation_shape_reason(spelling, words) is not None) is rejected


# -- type counts -----------------------------------------------------------------

def test_type_counts_record_elision_neighbors(tmp_path):
    docs = [{"s": ["απ’ όλα τα καλά", "πάρ’ το τώρα", "να ’χει δίκιο"]},
            {"s": ["απ’ όλους"]}]
    counts = TC.count_documents("x", "train", docs)
    assert counts.count["απ’"] == 2 and counts.docs["απ’"] == 2
    assert counts.context["απ’"] == [2, 2, 0, 0, 0]
    assert counts.context["πάρ’"] == [1, 0, 1, 0, 0]
    assert counts.context["’χει"] == [1, 0, 0, 1, 1]
    assert counts.initial["απ’"] == 2
    path = tmp_path / "t.tsv.gz"
    TC.write_table(counts, path)
    back = TC.read_table(path)
    assert back.count == counts.count and back.context == counts.context
    first = path.read_bytes()
    TC.write_table(counts, path)
    assert path.read_bytes() == first


# -- end to end ----------------------------------------------------------------------

SHARD = {"train-00000.jsonl.gz": "ab" * 32}
WORDS = 1_000_000


def table(source, rows, context=None, words=WORDS):
    """``rows``: token -> (count, initial, docs)."""
    tc = TC.TypeCounts(source, "train", words=words, tokens=words, sentences=1000,
                       documents=100, shards=dict(SHARD))
    for token, (n, ini, docs) in rows.items():
        tc.count[token] = n
        if ini:
            tc.initial[token] = ini
        tc.docs[token] = docs
    tc.context.update(context or {})
    return tc


def write_fixture(tmp_path):
    pairs = []

    def paradigm(lemma, pos, forms):
        for form, tags in forms:
            pairs.append({"form": form, "lemma": lemma, "pos": pos, "tags": tags})

    paradigm("καλός", "adj", [
        ("καλός", ["masculine", "nominative", "singular"]),
        ("καλή", ["feminine", "nominative", "singular"]),
        ("καλό", ["neuter", "nominative", "singular"]),
        ("καλοί", ["masculine", "nominative", "plural"]),
    ])
    paradigm("σπίτι", "noun", [
        ("σπίτι", ["nominative", "singular"]), ("σπιτιού", ["genitive", "singular"]),
        ("σπίτια", ["nominative", "plural"]),
    ])
    paradigm("βουλή", "noun", [("βουλή", ["nominative", "singular"]),
                               ("βουλής", ["genitive", "singular"])])
    paradigm("Αθήνα", "name", [("Αθήνα", ["nominative", "singular"]),
                               ("Αθήνας", ["genitive", "singular"])])
    paradigm("ζαβολιάρης", "noun", [("ζαβολιάρης", ["nominative", "singular"]),
                                    ("ζαβολιάρη", ["genitive", "singular"])])
    paradigm("γράφω", "verb", [
        ("γράφω", ["first-person", "present", "singular"]),
        ("γράφει", ["third-person", "present", "singular"]),
        ("έγραψα", ["first-person", "past", "singular"]),
        ("γράφοντάς", ["gerund"]),
    ])
    mg_pairs = tmp_path / "mg_pairs.json"
    mg_pairs.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")

    db = tmp_path / "lookup.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE lemmas (id INTEGER PRIMARY KEY, text TEXT)")
    con.execute("CREATE TABLE lookup (form TEXT, lemma_id INTEGER, src TEXT)")
    con.execute("INSERT INTO lemmas VALUES (1, 'γράφω'), (2, 'καλός')")
    con.executemany("INSERT INTO lookup VALUES (?, ?, ?)", [
        ("γράφουμε", 1, "el"), ("γραφω", 1, "el"), ("Καλός", 2, "el")])
    con.commit()
    con.close()

    freq = tmp_path / "mg_freq.txt"
    freq.write_text("\n".join([
        "zzzz 999000", "καλός 500", "σπίτι 5000", "ρωτάω 500", "σπιτί 60",
        "και 400", "γράφουμε 20", "απ 10",
    ]) + "\n", encoding="utf-8")

    function_words = tmp_path / "function_words.json"
    function_words.write_text(json.dumps({
        "schema_version": 1, "description": "fixture",
        "categories": {"conjunction": ["και", "ή"], "preposition": ["από"],
                       "particle": ["να"], "article": ["του"]}}, ensure_ascii=False), encoding="utf-8")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({
        "schema_version": 1, "rejects": [], "corpus_accepts": [],
        "elision_accepts": [], "abbreviation_rejects": [],
        "acronym_rejects": []}), encoding="utf-8")

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shards = [{"file": f, "sha256": d} for f, d in SHARD.items()]
    (corpus / "manifest.json").write_text(json.dumps({"sources": {
        "wiki": {"splits": {"train": {"shards": shards}, "dev": {"shards": shards}}}}}),
        encoding="utf-8")
    rows = {
        "καλός": (100, 0, 50), "καλή": (10, 0, 5), "καλό": (1, 0, 1),
        "σπίτι": (500, 0, 80), "σπίτια": (5, 0, 3),
        "βουλή": (10, 0, 5), "Βουλή": (40, 0, 20),
        "Αθήνα": (40, 0, 5), "Αθήνας": (3, 0, 2),
        "γράφω": (40, 0, 10), "έγραψα": (4, 0, 2), "γράφοντάς": (10, 0, 3),
        "ρωτάω": (20, 0, 4), "σπιτί": (6, 0, 2),
        "Κηφισιά": (40, 0, 4),
        "και": (900, 50, 99), "Και": (60, 60, 40), "από": (300, 0, 90),
        "απ’": (50, 0, 10), "καλ’": (20, 0, 5), "’γραψα": (15, 0, 6),
        "π.χ.": (30, 0, 5), "σελ.": (20, 0, 4), "του.Β.": (15, 0, 3),
        "ΟΗΕ": (40, 0, 10), "ΚΑΙ": (30, 0, 6),
    }
    context = {"απ’": [50, 48, 1, 45, 40], "καλ’": [20, 3, 1, 20, 15],
               "’γραψα": [15, 2, 3, 15, 15]}
    out_dir = tmp_path / "type_counts"
    TC.write_table(table("wiki", rows, context), TC.table_path(out_dir, "wiki", "train"))
    return {"mg_pairs": mg_pairs, "lookup_db": db, "mg_freq": freq,
            "function_words": function_words, "review": review, "corpus": corpus,
            "type_counts": out_dir}


def run_export(paths, out_dir, *extra):
    argv = ["--out-dir", str(out_dir), "--corpus", str(paths["corpus"]),
            "--type-counts", str(paths["type_counts"]),
            "--mg-pairs", str(paths["mg_pairs"]), "--lookup-db", str(paths["lookup_db"]),
            "--mg-freq", str(paths["mg_freq"]),
            "--function-words", str(paths["function_words"]),
            "--review", str(paths["review"]),
            "--reviewed-loanwords", str(paths["review"].with_name("none.json")),
            "--allow-unpinned", "--workers", "1",
            "--version", "9.9.9", "--commit", "c" * 40, *extra]
    assert ex.main(argv) == 0


def read_dic(path):
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    assert lines[-1] == ""
    body = lines[1:-1]
    assert int(lines[0]) == len(body)
    return body


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("el_monotonic")
    paths = write_fixture(tmp)
    out = tmp / "out"
    run_export(paths, out)
    return paths, out


def test_dic_format(exported):
    _, out = exported
    body = read_dic(out / "el_monotonic.dic")
    assert body == sorted(body, key=lambda s: s.encode("utf-8"))
    pattern = re.compile(r"^[^\t ]+\tfr:([CMRX])( mg:generated)?$")
    for line in body:
        m = pattern.match(line)
        assert m, line
        if m.group(2):
            assert m.group(1) == "X"
        assert L.orthography_reason(line.split("\t")[0]) is None


def test_routes_and_ratings(exported):
    _, out = exported
    ratings = dict(line.split("\t") for line in read_dic(out / "el_monotonic.dic"))
    # paradigm of an attested lemma, rated per million in its best source
    assert ratings["καλός"] == "fr:C"
    assert ratings["καλή"] == "fr:M"
    assert ratings["καλό"] == "fr:R"
    assert ratings["καλοί"] == "fr:X mg:generated"
    # a lookup.db form mg_pairs lacks joins its lemma; aliases do not
    assert ratings["γράφουμε"] == "fr:M"          # 20 FrequencyWords tokens
    assert "γραφω" not in ratings and "Καλός" not in ratings
    # no double-accented enclitic forms
    assert "γράφοντάς" not in ratings
    # an unattested lemma stays out
    assert "ζαβολιάρης" not in ratings
    # a name with its own evidence keeps its capital
    assert ratings["Αθήνα"] == "fr:C" and "Αθήνας" in ratings
    # function words
    assert "και" in ratings and "ή" in ratings and "από" in ratings
    assert "Και" not in ratings
    # corpus word attested in both sources
    assert "ρωτάω" in ratings
    # a misplaced accent of a far commoner word is no corpus word
    assert "σπιτί" not in ratings
    # names from the corpus; institutional capitals of common words are not
    assert "Κηφισιά" in ratings and "Βουλή" not in ratings
    # elisions with U+2019: a real one, an aphaeresis, not a numbering mark
    assert "απ’" in ratings and "’γραψα" in ratings and "καλ’" not in ratings
    # an elided word written without its mark is not listed
    assert "απ" not in ratings


def test_companion_lists(exported):
    _, out = exported
    abbreviations = (out / "el_monotonic_abbreviations.txt").read_text(
        encoding="utf-8").split()
    assert "π.χ." in abbreviations and "σελ." in abbreviations
    assert "του.Β." not in abbreviations
    acronyms = (out / "el_monotonic_acronyms.txt").read_text(encoding="utf-8").split()
    assert acronyms == ["ΟΗΕ"]


def test_aff_and_version(exported):
    paths, out = exported
    aff = (out / "el_monotonic.aff").read_text(encoding="utf-8").splitlines()
    rules = [line for line in aff if line and not line.startswith("#")]
    assert rules[:3] == ["SET UTF-8", "LANG el", "FLAG num"]
    assert rules[3].startswith("TRY ")
    letters = rules[3][4:]
    assert set(letters) == set(L.LOWER_LETTERS) and len(letters) == len(L.LOWER_LETTERS)
    assert len(rules) == 4
    version = dict(line.split(": ", 1) for line in
                   (out / "el_monotonic.version").read_text(encoding="utf-8").splitlines())
    assert version["version"] == "9.9.9"
    assert version["commit"] == "c" * 40
    assert version["variant"] == "el"
    body = read_dic(out / "el_monotonic.dic")
    assert int(version["entries"]) == len(body)
    buckets = Counter(line.split("\tfr:")[1][0] for line in body)
    assert version["buckets"] == " ".join(f"{b}={buckets[b]}" for b in "CMRX")
    assert version["dic_sha256"] == sha(out / "el_monotonic.dic")
    assert version["mg_pairs_sha256"] == sha(paths["mg_pairs"])
    assert version["mg_freq_sha256"] == sha(paths["mg_freq"])
    assert version["lookup_db_sha256"] == sha(paths["lookup_db"])
    assert version["review_sha256"] == sha(paths["review"])
    assert version["function_words_sha256"] == sha(paths["function_words"])
    assert "type_counts_sha256" in version and "corpus_manifest_sha256" in version


def test_rerun_is_byte_identical_and_inputs_change_the_version(exported, tmp_path):
    paths, out = exported
    again = tmp_path / "again"
    run_export(paths, again)
    for name in ("el_monotonic.dic", "el_monotonic.aff", "el_monotonic.version",
                 "el_monotonic_abbreviations.txt", "el_monotonic_acronyms.txt",
                 "el_monotonic.audit.json", "el_monotonic.evidence.tsv.gz"):
        assert (again / name).read_bytes() == (out / name).read_bytes(), name
    freq = paths["mg_freq"].read_text(encoding="utf-8")
    changed = dict(paths)
    changed["mg_freq"] = tmp_path / "mg_freq.txt"
    changed["mg_freq"].write_text(freq + "καλή 1\n", encoding="utf-8")
    third = tmp_path / "third"
    run_export(changed, third)
    assert ((third / "el_monotonic.version").read_bytes()
            != (out / "el_monotonic.version").read_bytes())


def test_partial_run_is_refused_in_the_shipping_directory(exported):
    paths, _ = exported
    before = sorted(ex.OUT.glob("el_monotonic*")) if ex.OUT.exists() else []
    stamps = {p: p.stat().st_mtime_ns for p in before}
    with pytest.raises(ex.ExportError, match="max-lemmas"):
        ex.run(ex_args(paths, ex.OUT, max_lemmas=2))
    after = sorted(ex.OUT.glob("el_monotonic*")) if ex.OUT.exists() else []
    assert after == before
    assert {p: p.stat().st_mtime_ns for p in after} == stamps


def ex_args(paths, out_dir, **overrides):
    import argparse
    args = argparse.Namespace(
        out_dir=out_dir, corpus=paths["corpus"], type_counts=paths["type_counts"],
        mg_pairs=paths["mg_pairs"], lookup_db=paths["lookup_db"], no_lookup=False,
        mg_freq=paths["mg_freq"], function_words=paths["function_words"],
        review=paths["review"], reviewed_loanwords=paths["review"].with_name("none.json"),
        max_lemmas=None, allow_unpinned=True, workers=1, version="9.9.9",
        commit="c" * 40)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_unpinned_inputs_are_refused(exported, tmp_path, monkeypatch):
    paths, _ = exported
    manifest = tmp_path / "hf_manifest.json"
    manifest.write_text(json.dumps({"files": {"data/mg_pairs.json": {
        "sha256": "0" * 64}}}), encoding="utf-8")
    digest = sha(paths["mg_pairs"])
    mismatched = ex.check_hf_pins({"mg_pairs": paths["mg_pairs"]},
                                  {"mg_pairs": digest}, manifest)
    assert len(mismatched) == 1 and "data/mg_pairs.json" in mismatched[0]
    assert ex.check_hf_pins({"mg_pairs": paths["mg_pairs"]},
                            {"mg_pairs": "0" * 64}, manifest) == []
    monkeypatch.setattr(ex, "check_hf_pins", lambda *args, **kwargs: mismatched)
    with pytest.raises(ex.ExportError, match="hf_manifest"):
        ex.run(ex_args(paths, tmp_path / "pinned", allow_unpinned=False))

"""Guards for the corpus word-tail filter (``dilemma/elision_tails.py``).

A corpus tokenizer that splits one printed word at its elision mark and gives
the second half the whole word's lemma puts a non-word into the lookup
(``νοντ᾽`` under βαρύνω, the end of βαρύνοντ᾽). The filter rejects those and
must leave real elided words alone.

Everything here runs off a synthetic fixture, so no corpus has to be on disk.
The fixture carries the real ``form_profile.db`` schema and the real
attestation counts, both asserted against the shipped artifact when it happens
to be present, so the fixture cannot quietly drift away from what it stands in
for. The (form, lemma) tables are the same shape ``build_lookup_db`` assembles:
a plain mapping of surface form to lemma.
"""

import sqlite3

import pytest

from build_lookup_db import drop_corpus_word_tails
from dilemma.elision_tails import (
    ELISION_MARKS,
    MIN_CUT_CHARACTERS,
    elision_stem,
    find_elision_tails,
    open_profile_attestation,
)

KORONIS = "᾽"          # the canonical elision key
SPACING_PSILI = "᾿"    # the spacing smooth breathing, as ἀλλ᾿ is written

# The real form_profile.db schema, copied from build/build_form_attestation.py.
# test_fixture_schema_matches_the_shipped_profile_database checks the copy.
PROFILE_SCHEMA = """
    CREATE TABLE works (
      work_id TEXT, id_scheme TEXT, source TEXT, author TEXT, title TEXT,
      genre TEXT, dialect TEXT, century INTEGER, start_year INTEGER,
      end_year INTEGER, PRIMARY KEY (work_id, id_scheme));
    CREATE TABLE forms (
      form_id INTEGER PRIMARY KEY, form TEXT NOT NULL, form_norm TEXT NOT NULL);
    CREATE TABLE form_profile (
      form_id INTEGER PRIMARY KEY, total_count INTEGER, n_works INTEGER,
      source_counts_json TEXT, by_century_json TEXT, by_genre_json TEXT,
      by_dialect_json TEXT, century_genre_json TEXT, dominant_pos TEXT);
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""

# The four word-tails measured in data/lookup.db, with the form each was cut
# from. Every one is the ending of another form of the same lemma, is written
# by no corpus, and is cut mid-stem rather than at an augment.
TAILS = {
    "νοντ" + KORONIS: ("βαρύνω", "βαρύνοντ" + KORONIS),
    "εντ" + KORONIS: ("πευκήεις", "πυκάεντ" + KORONIS),
    "λευμ" + KORONIS: ("γαμήλευμα", "γαμήλευμ" + KORONIS),
    "γάσασθ" + KORONIS: ("ἐργάζομαι", "ἠργάσασθ" + KORONIS),
}

# Real elided words the filter must never touch, each with the sibling of the
# same lemma it ends (where lookup.db has one) and the corpus count measured in
# form_profile.db, summed over the spellings of the mark. Together they cover
# every way a real word survives:
#   - it is the ending of nothing else its lemma inflects to (δ᾽, σ᾽, κατ᾽);
#   - the one character between it and its sibling is the AUGMENT, so it is an
#     unaugmented form and a word (γράψατ᾽, φης᾽), which is what the
#     cut-length test is for;
#   - the two characters between it and its sibling are a REDUPLICATION, which
#     the cut length cannot tell from a tokenizer's split, so only its corpus
#     attestation saves it (δοῦσ᾽, μνήσομ᾽, θεῖσ᾽).
KNOWN_GOOD = {
    "μ" + KORONIS: ("ἐγώ", 2674, "ἐμ" + KORONIS),
    "σ" + KORONIS: ("σύ", 1591, None),
    "ν" + KORONIS: ("ἐν", 274, "έν"),
    "κ" + KORONIS: ("ἄν", 1514, None),
    "τιν" + KORONIS: ("τις", 789, None),
    "ποτ" + KORONIS: ("ποτέ", 2719, None),
    "φησ" + KORONIS: ("φημί", 17, None),
    "σφ" + KORONIS: ("σφεῖς", 123, "ἄσφ" + KORONIS),
    KORONIS + "ς": ("εἰς", 0, None),
    KORONIS + "ν": ("ἐν", 0, None),
    "κατ" + KORONIS: ("κατά", 28397, None),
    "δ" + KORONIS: ("δέ", 161949, None),
    "ἀλλ" + SPACING_PSILI: ("ἀλλά", 82327, "μἀλλ" + KORONIS),
    "γράψατ" + KORONIS: ("γράφω", 0, "ἐγράψατ" + KORONIS),
    "φης" + KORONIS: ("φημί", 0, "έφης"),
    "βούλεσθ" + KORONIS: ("βούλομαι", 35, "ἐβούλεσθ" + KORONIS),
    # Unreduplicated forms, cut from their reduplicated sibling by exactly the
    # two characters of the reduplication. The cut-length test cannot save
    # these; only the corpus can.
    "δοῦσ" + KORONIS: ("δίδωμι", 4, "διδοῦσ" + KORONIS),
    "μνήσομ" + KORONIS: ("μιμνήσκω", 13, "μεμνήσομ"),
    "θεῖσ" + KORONIS: ("τίθημι", 1, "τιθεῖσ" + KORONIS),
}

# The three of those that the cut-length test alone would reject.
REDUPLICATED = {"δοῦσ" + KORONIS, "μνήσομ" + KORONIS,
                "θεῖσ" + KORONIS}


def _write_profile(path, counts):
    """A form_profile.db with the real schema and the given per-form counts."""
    connection = sqlite3.connect(path)
    connection.executescript(PROFILE_SCHEMA)
    for form_id, (form, total) in enumerate(sorted(counts.items()), start=1):
        connection.execute("INSERT INTO forms VALUES (?, ?, ?)",
                           (form_id, form, form.casefold()))
        connection.execute(
            "INSERT INTO form_profile VALUES (?, ?, 1, '{}', '{}', '{}',"
            " '{}', '{}', NULL)", (form_id, total))
    connection.commit()
    connection.close()
    return path


@pytest.fixture
def fixture_table():
    """Every tail and every known-good word, with the siblings that judge them."""
    table = {}
    for form, (lemma, sibling) in TAILS.items():
        table[form] = lemma
        table[sibling] = lemma
    for form, (lemma, _count, sibling) in KNOWN_GOOD.items():
        table[form] = lemma
        if sibling is not None:
            table[sibling] = lemma
    return table


@pytest.fixture
def is_attested(tmp_path):
    """The real corpus counts for the fixture's forms, keyed as the real DB is."""
    counts = {form: count
              for form, (_lemma, count, _sib) in KNOWN_GOOD.items()
              if count}
    # The bare stem of a tail IS in the corpus: the tokenizer error that made
    # the lookup row also left the bare stem in the text it annotated. The
    # filter must not read that as evidence that the elided form is a word.
    counts["νοντ"] = 4
    return open_profile_attestation(
        _write_profile(tmp_path / "form_profile.db", counts))


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------

def test_the_four_measured_tails_are_rejected(fixture_table, is_attested):
    assert set(find_elision_tails(fixture_table, is_attested=is_attested)) == \
        set(TAILS)


def test_every_known_good_elided_word_survives(fixture_table, is_attested):
    tails = find_elision_tails(fixture_table, is_attested=is_attested)
    assert not set(KNOWN_GOOD) & set(tails)


def test_the_rejection_names_the_form_the_tail_was_cut_from(
        fixture_table, is_attested):
    tails = find_elision_tails(fixture_table, is_attested=is_attested)
    for form, (_lemma, sibling) in TAILS.items():
        assert tails[form] == elision_stem(sibling)


def test_an_unaugmented_form_survives_only_because_of_the_cut_length(
        fixture_table, is_attested):
    """γράψατ᾽ and φης᾽ pass every other test; the augment is what saves them.

    Both are the ending of another form of the same lemma and neither is in
    the corpus, so lowering the cut length to one character rejects them -
    which is precisely why the rule requires two.
    """
    augmented = {"γράψατ" + KORONIS, "φης" + KORONIS}
    kept = find_elision_tails(fixture_table, is_attested=is_attested)
    assert not augmented & set(kept)

    relaxed = find_elision_tails(fixture_table, is_attested=is_attested,
                                 min_cut=1)
    assert augmented <= set(relaxed)
    assert MIN_CUT_CHARACTERS == 2


def test_an_unreduplicated_form_survives_only_because_the_corpus_wrote_it(
        fixture_table, is_attested):
    """δοῦσ᾽, μνήσομ᾽ and θεῖσ᾽ are cut by two characters, like the tails.

    The two characters are a reduplication, not a tokenizer's split, so the
    cut-length test cannot tell them apart and the corpus has to. Declaring
    nothing attested rejects all three, which is exactly what the attestation
    test is holding back.
    """
    assert not REDUPLICATED & set(
        find_elision_tails(fixture_table, is_attested=is_attested))
    assert REDUPLICATED <= set(
        find_elision_tails(fixture_table, is_attested=lambda stem: False))


def test_the_bare_stem_of_a_tail_is_not_treated_as_attestation(is_attested):
    """νοντ (no mark) is in the corpus 4 times; νοντ᾽ is still a tail."""
    assert not is_attested("νοντ")


def test_a_word_that_ends_nothing_else_is_never_reached(is_attested):
    """Most elided words are the ending of no other form of their lemma."""
    table = {"δ" + KORONIS: "δέ", "δέ": "δέ", "κατ" + KORONIS: "κατά"}
    assert find_elision_tails(table, is_attested=is_attested) == {}


# --------------------------------------------------------------------------
# What counts as an elision mark
# --------------------------------------------------------------------------

# Spelled out rather than taken from ELISION_MARKS, so that dropping one from
# the set fails the behavior below instead of quietly skipping its case.
REQUIRED_MARKS = (
    "\u1FBD",  # GREEK KORONIS
    "\u2019",  # RIGHT SINGLE QUOTATION MARK
    "\u1FBF",  # GREEK PSILI
    "\u02BC",  # MODIFIER LETTER APOSTROPHE
    "\u0313",  # COMBINING COMMA ABOVE
    "\u0027",  # APOSTROPHE
    "\u0060",  # GRAVE ACCENT
    "\u02B9",  # MODIFIER LETTER PRIME
)


def test_the_mark_set_holds_every_spelling_a_source_uses():
    assert set(REQUIRED_MARKS) == set(ELISION_MARKS)


@pytest.mark.parametrize("mark", REQUIRED_MARKS)
def test_every_spelling_of_the_mark_is_read_the_same_way(mark, is_attested):
    """A tail spelled with any apostrophe codepoint is still a tail.

    U+0313 COMBINING COMMA ABOVE matters most: it is how treebank exports
    write elision before form_sanitize rewrites it to the koronis, so leaving
    it out of the set lets a tail through from that source.
    """
    table = {"νοντ" + mark: "βαρύνω", "βαρύνοντ" + mark: "βαρύνω"}
    assert set(find_elision_tails(table, is_attested=is_attested)) == \
        {"νοντ" + mark}


def test_the_rough_breathing_is_not_an_elision_mark():
    """U+1FFE GREEK DASIA is a stranded breathing, not an apostrophe."""
    assert "῾" not in ELISION_MARKS
    assert elision_stem("νοντ῾") is None


def test_a_leading_mark_is_aphaeresis_and_never_a_tail(is_attested):
    """᾽ς and ᾽ν lose their FRONT, so they cannot be the end of anything."""
    assert elision_stem(KORONIS + "ς") is None
    table = {KORONIS + "ς": "εἰς", "εἰς": "εἰς", "ἐς": "εἰς"}
    assert find_elision_tails(table, is_attested=is_attested) == {}


def test_a_lone_mark_is_not_a_form():
    assert elision_stem(KORONIS) is None
    assert elision_stem("") is None
    assert elision_stem("βαρύνοντα") is None


# --------------------------------------------------------------------------
# The wiring in build_lookup_db
# --------------------------------------------------------------------------

def test_all_three_assembled_tables_are_filtered(fixture_table, is_attested):
    """lookup.db writes rows from `ag` and `el` as well as from `combined`.

    Dropping a tail from the merged table alone would put it straight back,
    because the grc-only and el-only rows come from the per-language tables
    wherever those differ from the merged one.
    """
    tables = {name: dict(fixture_table)
              for name in ("combined", "AG", "EL")}
    rows = []
    dropped = drop_corpus_word_tails(
        [(name, tables[name], lambda form: "ag_lookup.json")
         for name in ("combined", "AG", "EL")],
        is_attested=is_attested,
        record=lambda **row: rows.append(row),
    )
    for name, table in tables.items():
        assert set(dropped[name]) == set(TAILS), name
        assert not set(TAILS) & set(table), name
        assert set(KNOWN_GOOD) <= set(table), name
    assert len(rows) == 3 * len(TAILS)


def test_the_audit_row_carries_the_evidence_for_the_rejection(
        fixture_table, is_attested):
    """Each drop is reversible from the report: form, lemma, source, sibling."""
    rows = []
    drop_corpus_word_tails(
        [("combined", dict(fixture_table), lambda form: "glaux_pairs.json")],
        is_attested=is_attested, record=lambda **row: rows.append(row))
    by_form = {row["form"]: row for row in rows}
    for form, (lemma, sibling) in TAILS.items():
        row = by_form[form]
        assert row["table"] == "combined"
        assert row["source"] == "glaux_pairs.json"
        assert row["lemma"] == lemma
        assert row["reason"].startswith("corpus_word_tail")
        assert elision_stem(sibling) in row["reason"]


def test_the_builder_filters_every_table_it_writes_rows_from():
    """The call site must cover `combined`, `ag` and `el`, in that order."""
    import inspect

    import build_lookup_db

    source = inspect.getsource(build_lookup_db.build)
    call = source[source.index("drop_corpus_word_tails("):]
    call = call[:call.index("record=")]
    assert '("combined", combined,' in call
    assert '("AG", ag,' in call
    assert '("EL", el,' in call


# --------------------------------------------------------------------------
# Ties to the real artifacts (skipped when they are not downloaded)
# --------------------------------------------------------------------------

def _data_file(name):
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data" / name
    return path if path.exists() else None


@pytest.mark.skipif(_data_file("form_profile.db") is None,
                    reason="form_profile.db not downloaded")
def test_fixture_schema_matches_the_shipped_profile_database(tmp_path):
    """The stand-in must have the same tables and columns as the real one."""
    real = sqlite3.connect(f"file:{_data_file('form_profile.db')}?mode=ro",
                           uri=True)
    fake = sqlite3.connect(_write_profile(tmp_path / "p.db", {"δ᾽": 1}))
    for table in ("works", "forms", "form_profile", "meta"):
        assert [row[1:3] for row in
                fake.execute(f"PRAGMA table_info({table})")] == \
               [row[1:3] for row in
                real.execute(f"PRAGMA table_info({table})")], table


@pytest.mark.skipif(_data_file("form_profile.db") is None,
                    reason="form_profile.db not downloaded")
def test_fixture_attestation_counts_match_the_shipped_corpus():
    """The counts the fixture asserts on are the real corpus counts."""
    real = open_profile_attestation(_data_file("form_profile.db"))
    for form, (_lemma, count, _sib) in KNOWN_GOOD.items():
        stem = elision_stem(form)
        if stem is None:
            continue
        assert real(stem) is bool(count), form
    # The bare stem of a tail is in the corpus; the elided form is not.
    assert real("νοντ") is False


@pytest.mark.skipif(
    _data_file("lookup.db") is None or _data_file("form_profile.db") is None,
    reason="lookup.db or form_profile.db not downloaded")
def test_the_shipped_lookup_holds_no_tail_beyond_the_four_measured_ones():
    """A subset assertion, so it holds before and after the next rebuild.

    The lookup built before this filter existed carries exactly the four;
    the next one built with it carries none. Anything else is a tail the
    measurement did not account for, and wants looking at.
    """
    connection = sqlite3.connect(f"file:{_data_file('lookup.db')}?mode=ro",
                                 uri=True)
    is_real = open_profile_attestation(_data_file("form_profile.db"))
    for src in ("grc", "el"):
        table = dict(connection.execute(
            "SELECT l.form, m.text FROM lookup l "
            "JOIN lemmas m ON m.id = l.lemma_id WHERE l.src = ?", (src,)))
        assert set(find_elision_tails(table, is_attested=is_real)) <= set(TAILS)


def test_a_tail_cannot_hide_by_having_its_full_word_in_another_table():
    """The builder writes Ancient-Greek-only and Modern-Greek-only rows from
    the `ag` and `el` tables wherever those differ from `combined`. If each
    table were judged only against its own forms, a tail in one table whose
    full word sits in the other would be dropped from `combined` and then
    written straight back as a row of its own.
    """
    tail = "\u03bd\u03bf\u03bd\u03c4" + KORONIS
    whole = "\u03b2\u03b1\u03c1\u03cd\u03bd\u03bf\u03bd\u03c4" + KORONIS
    lemma = "\u03b2\u03b1\u03c1\u03cd\u03bd\u03c9"
    ag = {tail: lemma}
    el = {whole: lemma}
    never_attested = lambda stem: False

    assert find_elision_tails(ag, is_attested=never_attested) == {}, (
        "fixture assumption: judged alone, the tail has no sibling to be cut from")
    assert find_elision_tails(
        ag, is_attested=never_attested, sibling_tables=[ag, el]
    ) == {tail: "\u03b2\u03b1\u03c1\u03cd\u03bd\u03bf\u03bd\u03c4"}

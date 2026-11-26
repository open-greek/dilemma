"""Shipped treebank POS table: votes for a listed head and for an attested name.

#141: a lower-case head that an edited lexicon lists keeps its own corpus
vote where a lower table would answer with another word, so the tagged
answer is the untagged one. #138: a capitalized name that GLAUx attests well
has a NOUN cell under its exact spelling; the grave and the acute spelling
of a word are counted together, and a lower-case adjective counts against
the name like a lower-case noun. These need the table rebuilt by
build/build_treebank_pos_lookup.py.
"""
import json
from pathlib import Path

import pytest

from dilemma import core
from dilemma.grc_reviewed_respellings import respelled_lemma

ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "data" / "treebank_pos_lookup.json"
AUDIT_PATH = ROOT / "data" / "treebank_pos_lookup.audit.json"


@pytest.fixture(scope="module", params=["grc", "all"])
def shipped_model(request):
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    return core.Dilemma(lang=request.param)


LISTED_HEADS = [
    ("ταμεῖον", "NOUN"), ("ψηφίς", "NOUN"), ("συρίγγιον", "NOUN"),
    ("σπαθίον", "NOUN"), ("δοξάριον", "NOUN"), ("σαῦρος", "NOUN"),
    ("κρίκος", "NOUN"), ("φάλαινα", "NOUN"), ("ξενοδόχος", "NOUN"),
    ("καταπέλτης", "NOUN"), ("μεγάλος", "ADJ"), ("ἁμός", "ADJ"),
    ("ἔγγειος", "ADJ"), ("ἠΰτε", "ADV"), ("ἰθύνω", "VERB"),
]


@pytest.mark.parametrize("head,upos", LISTED_HEADS)
def test_listed_head_reads_itself_with_and_without_its_tag(shipped_model, head, upos):
    assert shipped_model.lemmatize(head, guess=False) == head
    assert shipped_model.lemmatize_pos(head, upos, guess=False) == head
    assert shipped_model.lemmatize_batch_pos([head], [upos], guess=False) == [head]


@pytest.mark.parametrize("form,upos,lemma", [
    ("ταμείον", "NOUN", "ταμεῖον"),     # the monotonic key follows the head
    ("ψηφὶς", "NOUN", "ψηφίς"),         # and so does the grave spelling
    ("ψῆφος", "NOUN", "ψῆφος"),         # the other word keeps its own forms
    ("ψήφου", "NOUN", "ψῆφος"),
    ("ταμιεῖον", "NOUN", "ταμιεῖον"),
    ("σύριγγος", "NOUN", "σῦριγξ"),
])
def test_neighbours_of_the_listed_heads(shipped_model, form, upos, lemma):
    assert shipped_model.lemmatize_pos(form, upos, guess=False) == lemma


CAPITALIZED_NAMES = [
    ("Θέρμον", "Θέρμος"), ("Εὐριπίδου", "Εὐριπίδης"), ("Συρίας", "Συρία"),
    ("Ἕκτωρ", "Ἕκτωρ"), ("Πέτρος", "Πέτρος"), ("Τρύφων", "Τρύφων"),
    # The grave and the acute spelling share the token counts, and each has
    # its own cell: Αἰακός and Πενθεύς have under ten tokens of their own.
    ("Αἰακὸς", "Αἰακός"), ("Αἰακός", "Αἰακός"),
    ("Πενθεὺς", "Πενθεύς"), ("Πενθεύς", "Πενθεύς"),
]


@pytest.mark.parametrize("form,name", CAPITALIZED_NAMES)
def test_attested_capitalized_name_keeps_the_name_under_noun(shipped_model, form, name):
    assert shipped_model.lemmatize(form, guess=False) == name
    assert shipped_model.lemmatize_pos(form, "NOUN", guess=False) == name
    assert shipped_model.lemmatize_batch_pos([form], ["NOUN"], guess=False) == [name]


@pytest.mark.parametrize("form,lemma", [
    ("θέρμον", "θέρμος"), ("πέτρος", "πέτρος"), ("ἕκτωρ", "ἕκτωρ"),
    ("συρίας", "συρία"),
])
def test_lower_case_spelling_keeps_the_common_noun(shipped_model, form, lemma):
    assert shipped_model.lemmatize_pos(form, "NOUN", guess=False) == lemma


@pytest.mark.parametrize("form,lemma", [
    # Under a quarter as many name tokens as the lower-case noun has.
    ("Τύχη", "τύχη"), ("Τύχης", "τύχη"), ("Τύχην", "τύχη"),
    ("Νίκη", "νίκη"), ("Νίκης", "νίκη"), ("Νίκην", "νίκη"),
    # The name in 27 of 38 capitalized tokens.
    ("Ἔρις", "ἔρις"),
    # The lower-case spelling is an adjective with over four times the
    # name's tokens (φίλων 2,036 beside Φίλων 60; νέων 644 beside Νέων 15).
    ("Φίλων", "φίλος"), ("Νέων", "νέος"),
])
def test_names_that_fail_the_token_tests_are_left(shipped_model, form, lemma):
    assert shipped_model.lemmatize_pos(form, "NOUN", guess=False) == lemma
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    assert form not in audit["capital_name_cells"]


@pytest.mark.parametrize("form,name", [("Φίλων", "Φίλων"), ("Νέων", "Νέων"),
                                       ("Τύχη", "Τύχη")])
def test_a_name_without_a_cell_keeps_its_other_readings(shipped_model, form, name):
    assert shipped_model.lemmatize(form, guess=False) == name
    assert shipped_model.lemmatize_pos(form, "PROPN", guess=False) == name


def test_audit_lists_the_kept_votes_with_their_evidence():
    if not AUDIT_PATH.exists() or not TABLE_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    table = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    assert audit["listed_head_votes"]["ψηφίς"] == {"upos": "NOUN", "lemma": "ψηφίς"}
    assert audit["capital_name_cells"]["Θέρμον"] == {
        "lemma": "Θέρμος", "form_tokens": 10, "name_tokens": 10,
        "lowercase_nominal_tokens": 4}
    # Αἰακὸς 24 and Αἰακός 7 tokens, counted as one spelling.
    assert audit["capital_name_cells"]["Αἰακός"] == audit[
        "capital_name_cells"]["Αἰακὸς"] == {
            "lemma": "Αἰακός", "form_tokens": 31, "name_tokens": 31,
            "lowercase_nominal_tokens": 0}
    for form, vote in audit["listed_head_votes"].items():
        assert table[form][vote["upos"]] == vote["lemma"]
        assert not form[:1].isupper()
    for form, cell in audit["capital_name_cells"].items():
        assert table[form]["NOUN"] == cell["lemma"]
        assert cell["form_tokens"] >= 10
        assert 10 * cell["name_tokens"] >= 9 * cell["form_tokens"]
        assert 4 * cell["name_tokens"] >= cell["lowercase_nominal_tokens"]
        # ᾍδης begins with a titlecase letter, so compare with the lower case.
        assert len(form) > 1 and form[:1] != form[:1].lower()
        assert cell["lemma"][:1] != cell["lemma"][:1].lower()
        # Exact spelling only: the lower-case key is not the name's.
        assert table.get(form.lower(), {}).get("NOUN") != cell["lemma"]


# A verb the tables would otherwise not name as one keeps a single vote.

def _witness_fixture(tmp_path, monkeypatch):
    from dilemma_build import build_treebank_pos_lookup as builder

    (tmp_path / "ag_pos_lookup.json").write_text(
        json.dumps({"λύει": {"VERB": "λύω", "NOUN": "λύσις"}}), encoding="utf-8")
    monkeypatch.setattr(builder, "DATA_DIR", tmp_path)
    monkeypatch.setattr(builder, "_lexical_target_key", lambda lemma: lemma)
    return builder


def test_verb_witness_keeps_one_exact_vote_per_unknown_verb(tmp_path, monkeypatch):
    builder = _witness_fixture(tmp_path, monkeypatch)
    lookup = {"γράφει": {"VERB": "γράφω", "NOUN": "γραφεύς"}}
    candidates = {
        "προσκαθῆσθαι": ("VERB", "προσκάθημαι"),
        "προσκαθημένους": ("VERB", "προσκάθημαι"),
        "γράφειν": ("VERB", "γράφω"),          # known from the table itself
        "λύειν": ("VERB", "λύω"),              # known from a lower table
        "συναποθανεῖν": ("VERB", "συναποθνῄσκω"),  # the lookup spells it otherwise
        "ξύλου": ("NOUN", "ξύλον"),             # not a verb vote
        "βλαψάτω": ("VERB", "βλάπτωω"),         # no listed head
    }
    corpus = {
        "προσκαθῆσθαι": {"VERB": {"προσκάθημαι": 3}},
        "προσκαθημένους": {"VERB": {"προσκάθημαι": 5}},
        "γράφειν": {"VERB": {"γράφω": 9}},
        "λύειν": {"VERB": {"λύω": 9}},
        "συναποθανεῖν": {"VERB": {"συναποθνῄσκω": 4}},
        "ξύλου": {"NOUN": {"ξύλον": 7}},
        "βλαψάτω": {"VERB": {"βλάπτωω": 2}},
    }
    bare = {"προσκαθῆσθαι": "προσκάθημαι", "προσκαθημένους": "προσκάθημαι",
            "γράφειν": "γράφω", "λύειν": "λύω",
            "συναποθανεῖν": "συναποθνήσκω", "ξύλου": "ξύλον",
            "βλαψάτω": "βλάπτωω"}
    listed = {"προσκάθημαι", "γράφω", "λύω", "συναποθνῄσκω", "ξύλον"}
    cells = builder._verb_witness_cells(lookup, candidates, corpus, listed, bare)
    assert cells == {"προσκαθημένους": {"upos": "VERB", "lemma": "προσκάθημαι",
                                         "votes": 5}}


def test_verb_witness_breaks_a_vote_tie_by_sort_order(tmp_path, monkeypatch):
    builder = _witness_fixture(tmp_path, monkeypatch)
    candidates = {"φρυκτωροῦσι": ("VERB", "φρυκτωρέω"),
                  "φρυκτωρεῖν": ("VERB", "φρυκτωρέω")}
    corpus = {form: {"VERB": {"φρυκτωρέω": 2}} for form in candidates}
    bare = {form: "φρυκτωρέω" for form in candidates}
    cells = builder._verb_witness_cells({}, candidates, corpus, {"φρυκτωρέω"}, bare)
    assert list(cells) == [min(candidates)]


def test_shipped_table_keeps_verbs_known_by_an_agreeing_vote(shipped_model):
    """A verb whose every vote agrees with the lookup is still a known verb,
    so an oblique participle with no row resolves through its nominative."""
    if not AUDIT_PATH.exists():
        pytest.skip("The corpus target audit is not installed")
    cells = json.loads(AUDIT_PATH.read_text(encoding="utf-8")).get(
        "verb_witness_cells")
    if not cells:
        pytest.skip("The table predates the verb witness cells")
    table = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    for form, cell in cells.items():
        assert table[form] == {cell["upos"]: cell["lemma"]}
        # The reviewed readings correct a vote (προεπίστανται is
        # προεπίσταμαι, not προεφίστημι), a vote's lemma loses the accent
        # of an enclitic (ἀποκτίννυμί; #159), and a malformed one gives
        # way to the headword it misspells (λε͂γω λέγω; #162).
        want = core.Dilemma._without_enclitic_accent(
            respelled_lemma(form, cell["lemma"]))
        want = shipped_model._repaired_lemma(want) or want
        assert shipped_model.lemmatize(form, guess=False) == want
    assert shipped_model.lemmatize("προσκαθημένας", guess=False) == "προσκάθημαι"

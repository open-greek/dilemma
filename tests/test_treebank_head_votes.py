"""Shipped treebank POS table: votes for a listed head and for an attested name.

#141: a lower-case head that an edited lexicon lists keeps its own corpus
vote where a lower table would answer with another word, so the tagged
answer is the untagged one. #138: a capitalized name that GLAUx attests well
has a NOUN cell under its exact spelling. These need the table rebuilt by
build/build_treebank_pos_lookup.py.
"""
import json
from pathlib import Path

import pytest

from dilemma import core

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
])
def test_names_that_fail_the_token_tests_are_left(shipped_model, form, lemma):
    assert shipped_model.lemmatize_pos(form, "NOUN", guess=False) == lemma
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    assert form not in audit["capital_name_cells"]


def test_audit_lists_the_kept_votes_with_their_evidence():
    if not AUDIT_PATH.exists() or not TABLE_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    table = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    assert audit["listed_head_votes"]["ψηφίς"] == {"upos": "NOUN", "lemma": "ψηφίς"}
    assert audit["capital_name_cells"]["Θέρμον"] == {
        "lemma": "Θέρμος", "form_tokens": 10, "name_tokens": 10,
        "lowercase_noun_tokens": 4}
    for form, vote in audit["listed_head_votes"].items():
        assert table[form][vote["upos"]] == vote["lemma"]
        assert not form[:1].isupper()
    for form, cell in audit["capital_name_cells"].items():
        assert table[form]["NOUN"] == cell["lemma"]
        assert cell["form_tokens"] >= 10
        assert 10 * cell["name_tokens"] >= 9 * cell["form_tokens"]
        assert 4 * cell["name_tokens"] >= cell["lowercase_noun_tokens"]
        # Exact spelling only: the lower-case key is not the name's.
        assert table.get(form.lower(), {}).get("NOUN") != cell["lemma"]

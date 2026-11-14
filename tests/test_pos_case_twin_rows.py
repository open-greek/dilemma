"""Homographs whose lookup row has a capitalized twin (#110).

The POS selection does not let an entry of a tag other than VERB override
a lookup reading together with its capitalized twin (#42 is VERB-only;
extending it measured worse on held-out Gorman). The wrong answers come
from single-valued lookup.db rows (γραφῆς -> γραφεύς, σεῖο -> σῦς, ...)
that the 1.3.25 lookup rebuild corrects. Remove the xfail marks once the
rebuilt lookup is pinned.
"""
import pytest

from dilemma import core


@pytest.fixture(scope="module", params=["grc", "all"])
def shipped_model(request):
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    return core.Dilemma(lang=request.param, resolve_articles=True)


def both(model, word, upos):
    scalar = model.lemmatize_pos(word, upos, guess=False)
    batch = model.lemmatize_batch_pos([word], [upos], guess=False)
    assert batch == [scalar], (word, upos, scalar, batch)
    return scalar


SHIPPED_AFTER_REBUILD = [
    ("γραφῆς", "NOUN", "γραφή"), ("γραφήν", "NOUN", "γραφή"),
    ("γραφή", "NOUN", "γραφή"), ("γραφαί", "NOUN", "γραφή"),
    ("σεῖο", "PRON", "σύ"), ("σέο", "PRON", "σύ"),
    ("ἱερέων", "NOUN", "ἱερεύς"), ("ἄριστα", "ADJ", "ἀγαθός"),
    ("ἀρίστοις", "ADJ", "ἀγαθός"), ("ἀρὰν", "NOUN", "ἀρά"),
]
SHIPPED_AFTER_REBUILD_UNTAGGED = [
    ("σύ", "σύ"), ("σεῖο", "σύ"), ("σέθεν", "σύ"), ("γραφή", "γραφή"),
]


@pytest.mark.xfail(strict=False, reason="needs the 1.3.25 lookup rebuild")
@pytest.mark.parametrize("word,upos,lemma", SHIPPED_AFTER_REBUILD)
def test_shipped_lookup_rows_after_rebuild(shipped_model, word, upos, lemma):
    assert both(shipped_model, word, upos) == lemma


@pytest.mark.xfail(strict=False, reason="needs the 1.3.25 lookup rebuild")
@pytest.mark.parametrize("word,lemma", SHIPPED_AFTER_REBUILD_UNTAGGED)
def test_shipped_untagged_lookup_rows_after_rebuild(shipped_model, word, lemma):
    assert shipped_model.lemmatize(word, guess=False) == lemma


@pytest.mark.parametrize("word,upos,lemma", [
    ("γραφὴν", "NOUN", "γραφή"), ("ἀράν", "NOUN", "ἀρά"), ("ἱερά", "ADJ", "ἱερός"),
])
def test_shipped_110_controls(shipped_model, word, upos, lemma):
    assert both(shipped_model, word, upos) == lemma

"""Closed-class conventions must not discard independent lexical readings."""

import pytest

from dilemma.core import Dilemma, _ARTICLE_FORMS, to_monotonic


def resolver(*, enabled=True, convention=None):
    instance = object.__new__(Dilemma)
    instance._resolve_articles = enabled
    instance._convention_name = convention
    return instance


@pytest.mark.parametrize("form, expected", [
    ("τι", "τις"), ("τί", "τίς"), ("τὶ", "τίς"),
    ("ΤΙ", "τις"), ("ΤΊ", "τίς"),
    ("τινός", "τις"), ("τίνος", "τίς"),
    ("τινῶν", "τις"), ("τίνων", "τίς"),
    ("τισὶν", "τις"), ("τίσιν", "τίς"),
    ("αὐτοῦ", "αὐτός"), ("Αὐτοῦ", "αὐτός"), ("αυτού", "αὐτός"),
    ("αὐτὴν", "αὐτός"), ("αὐτῇ", "αὐτός"), ("αὐτοῖν", "αὐτός"),
    ("αὐτὰς", "αὐτός"), ("ΑΥΤΩΝ", None),
    ("ΜΟΙ", "ἐγώ"), ("ἐμοὶ", "ἐγώ"), ("ΣΟῪ", "σύ"),
])
def test_canonical_pronouns_preserve_accent_distinctions(form, expected):
    assert resolver()._resolve_closed_class(form) == expected


@pytest.mark.parametrize("form", ["τι", "τί", "αὐτοῦ", "μοι"])
def test_default_leaves_readings_to_the_lookup(form):
    assert resolver(enabled=False)._resolve_closed_class(form) is None


@pytest.mark.parametrize("form", sorted({to_monotonic(f) for f in _ARTICLE_FORMS}))
def test_opt_in_accepts_normalized_forms_of_every_article(form):
    assert resolver()._resolve_closed_class(form) == "ὁ"
    assert resolver()._resolve_closed_class(form.upper()) == "ὁ"
    assert resolver(enabled=False)._resolve_closed_class(form) is None


@pytest.mark.parametrize("form", ["ἤ", "ή", "ὦ", "ώ", "τῇδε", "τῆλε"])
def test_article_normalization_does_not_remove_distinctive_accents(form):
    assert resolver()._resolve_closed_class(form) is None


@pytest.mark.parametrize("form, expected", [
    ("τι", None), ("τί", None), ("αὐτοῦ", None),
    ("σε", None), ("αυτού", "αυτός"), ("τις", "ο"),
])
def test_modern_convention_retains_its_own_closed_classes(form, expected):
    assert resolver(convention="triantafyllidis")._resolve_closed_class(form) == expected

"""No lemma begins with a double rho (#132).

The doubled rho of an augmented past tense (ἔρριψα) was kept on the lemma of
a few generated and corpus rows: ἐρρίφθαι read ρροπίω, as did ἐῤῥίφθαι. The
lemmas are reviewed non-lemmas, so the lookup build drops their rows and the
forms read their accent-stripped key (ερριφθαι: ῥίπτω).
"""
import pytest

from dilemma import core
from dilemma.grc_rejected_lemmas import REJECTED_LEMMA_TARGETS, REJECTED_LEMMAS

DOUBLE_RHO = {"ρροπίω", "ρραστωνάω", "ρρυφάνω", "ρρωζμαίω", "ρρωμαίζω",
              "ρρωννύω", "ρρᾳθύμνυηι", "ρρῃίννυμι"}


def test_double_rho_lemmas_are_reviewed_non_lemmas():
    assert DOUBLE_RHO <= REJECTED_LEMMAS
    for lemma in DOUBLE_RHO:
        entry = REJECTED_LEMMA_TARGETS[lemma]
        assert entry["reason"] == "double_rho_augment_artifact"
        assert entry["evidence"]


def test_heads_with_rho_are_not_rejected():
    assert not {"ῥίπτω", "ῥᾳστωνέω", "ῥώννυμι", "ἔρρω"} & REJECTED_LEMMAS


# Needs the lookup rebuilt with the record (the central data build).
@pytest.mark.parametrize("lang", ["all", "grc"])
@pytest.mark.parametrize("form", ["ἐρρίφθαι", "ἐῤῥίφθαι"])
def test_shipped_forms_read_their_verb(lang, form):
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    assert core.Dilemma(lang=lang).lemmatize(form, guess=False) == "ῥίπτω"

"""A treebank lemma written ``preverb-verb`` is read as the compound verb."""
import pytest

from dilemma.preverb_lemma import (accent_fold, heads_by_accent_fold,
                                   joined_preverb_lemma, preverb_lemma_head)


@pytest.mark.parametrize("lemma,joined", [
    # elision before a vowel; the verb loses its breathing
    ("ἐπί-εἰσκομίζω", "ἐπεισκομίζω"),
    ("διά-ἐγείρω", "διεγείρω"),
    ("παρά-ἀναπίπτω", "παραναπίπτω"),
    ("ἀντί-αὐγάζω", "ἀνταυγάζω"),
    ("ἀνά-ἴσχω", "ἀνίσχω"),
    ("ἀμφί-ἔπω", "ἀμφέπω"),
    # aspiration before a rough breathing
    ("κατά-ἕζομαι", "καθέζομαι"),
    ("ἐπί-ἁμιλλάομαι", "ἐφαμιλλάομαι"),
    ("ἀπό-ἱστάνω", "ἀφιστάνω"),
    ("ἀντί-ἵστημι", "ἀνθίστημι"),
    ("μετά-ἵστημι", "μεθίστημι"),
    ("ὑπό-αἱρέω", "ὑφαιρέω"),
    # no aspirate to take: σύν, ἀνά, διά
    ("σύν-ἱστάνω", "συνιστάνω"),
    ("ἀνά-ὁρμάω", "ἀνορμάω"),
    # before a consonant the preverb keeps its vowel
    ("κατά-κληρουχέω", "κατακληρουχέω"),
    ("ἐπί-τροπεύω", "ἐπιτροπεύω"),
    # περί and πρό do not elide; ι and υ after πρό take a diaeresis
    ("περί-ἵστημι", "περιίστημι"),
    ("περί-εἶδον", "περιεῖδον"),
    ("πρό-ἐνείρω", "προενείρω"),
    ("πρό-ἱστορέω", "προϊστορέω"),
    ("πρό-ὑποτίθημι", "προϋποτίθημι"),
    ("πρό-ἵημι", "προΐημι"),
    ("πρό-εἶπον", "προεῖπον"),
    # ἐκ is ἐξ before a vowel
    ("ἐκ-ἀναπείθω", "ἐξαναπείθω"),
    ("ἐκ-πλάσσω", "ἐκπλάσσω"),
    # the nu of ἐν and σύν
    ("ἐν-περιλαμβάνω", "ἐμπεριλαμβάνω"),
    ("ἐν-χειρέω", "ἐγχειρέω"),
    ("ἐν-μένω", "ἐμμένω"),
    ("ἐν-λείπω", "ἐλλείπω"),
    ("ἐν-ὑπάρχω", "ἐνυπάρχω"),
    ("ἐν-τίθημι", "ἐντίθημι"),
    ("σύν-καταφεύγω", "συγκαταφεύγω"),
    ("σύν-παλαίω", "συμπαλαίω"),
    ("σύν-λέγω", "συλλέγω"),
    ("σύν-ναυαγέω", "συνναυαγέω"),
    ("σύν-διατίθημι", "συνδιατίθημι"),
    ("σύν-σιτέω", "συσσιτέω"),
    ("σύν-στρατεύω", "συστρατεύω"),
    ("σύν-ζεύγνυμι", "συζεύγνυμι"),
    # initial rho
    ("ἀνά-ῥιπίζω", "ἀναρριπίζω"),
    ("σύν-ῥέω", "συρρέω"),
    ("ἐν-ῥάπτω", "ἐρράπτω"),
    ("ἐκ-ῥέω", "ἐκρέω"),
    # εἰς and πρός as AGDT spells them
    ("εἰσ-κρούω", "εἰσκρούω"),
    ("πρόσ-στρατοπεδεύω", "προσστρατοπεδεύω"),
    ("πρόσ-ἀμπέχω", "προσαμπέχω"),
    # two preverbs, attached from the verb outwards
    ("ἀντί,κατά-ἕζομαι", "ἀντικαθέζομαι"),
    ("σύν,ἐν-κλείω", "συνεγκλείω"),
    ("ἐν,κατά-κλείω", "ἐγκατακλείω"),
    ("παρά,εἰσ-δύω", "παρεισδύω"),
    # the verb keeps its own accent; a headword list settles it
    ("περί-εἶμι", "περιεῖμι"),
    ("ἐν-εἰμί", "ἐνειμί"),
])
def test_preverb_lemma_is_joined_with_elision_and_assimilation(lemma, joined):
    assert joined_preverb_lemma(lemma) == joined


@pytest.mark.parametrize("lemma", [
    # a line-broken token lemmatized as it stands
    "αὐ-τός", "πήν-τος", "προσ-τίθημι", "προ-πιέω", "προὐ-τίθημι",
    # OCR and misread tokens
    "διί-λύω", "μέι-οἰυρίζω", "ὕπ-οἰλαβω", "ἐπά-ναρθόω",
    # a lexicon's divided head
    "Βακχ-εία", "καθ-ηδύνω",
    # no hyphen, two hyphens, other marks
    "ἐπεισκομίζω", "ἐπί-εἰσ-κομίζω", "ἐπί—εἰσκομίζω", "ἐπί-",
    "ἐπί-εἰσ­κομίζω",
    # a vowel without a breathing after the hyphen
    "ἐπί-αγω",
    # a capitalized verb
    "ἐπί-Ἀγω",
    "", None,
])
def test_other_hyphenated_lemmas_are_left_alone(lemma):
    assert joined_preverb_lemma(lemma) is None


HEADS = frozenset({
    "ἐπεισκομίζω", "συγκαταφεύγω", "περίειμι", "ἔνειμι", "παρείσειμι",
    "ὑποστορέννυμι", "καταπροΐημι", "καθέζομαι",
})
FOLDED = heads_by_accent_fold(HEADS)


@pytest.mark.parametrize("lemma,head", [
    ("ἐπί-εἰσκομίζω", "ἐπεισκομίζω"),
    ("σύν-καταφεύγω", "συγκαταφεύγω"),
    ("κατά-ἕζομαι", "καθέζομαι"),
    # the list's accent, where a short verb throws it back
    ("περί-εἶμι", "περίειμι"),
    ("ἐν-εἰμί", "ἔνειμι"),
    ("παρά,εἰσ-εἰμί", "παρείσειμι"),
    ("ὑπό-στόρεννυμι", "ὑποστορέννυμι"),
    # and the list's diaeresis
    ("κατά-προίημι", "καταπροΐημι"),
    # no list has the compound
    ("παρά,εἰσ-δυάω", None),
    ("κατά-ἐνίζω", None),
    ("ἀμφί-ἀμφιέννυμι", None),
    # not a preverb lemma
    ("αὐ-τός", None),
    ("περίειμι", None),
])
def test_joined_compound_must_be_a_listed_headword(lemma, head):
    assert preverb_lemma_head(lemma, HEADS, FOLDED) == head


def test_exact_compound_stands_beside_another_spelling_of_it():
    # one list writes the hiatus with a diaeresis, another without
    heads = frozenset({"καταπροίημι", "καταπροΐημι"})
    assert preverb_lemma_head("κατά-προίημι", heads,
                              heads_by_accent_fold(heads)) == "καταπροίημι"


def test_two_heads_that_differ_only_in_accent_give_no_head():
    # the second spelling is a mis-accented twin of the kind an OCR list
    # carries; neither is the compound as joined (ὑποστόρεννυμι)
    heads = frozenset({"ὑποστορέννυμι", "ὑποστορεννύμι"})
    assert preverb_lemma_head("ὑπό-στόρεννυμι", heads,
                              heads_by_accent_fold(heads)) is None


def test_accent_fold_keeps_breathing_and_subscript():
    assert accent_fold("καταπροΐημι") == "καταπροιημι"
    assert accent_fold("ἔνειμι") == "ἐνειμι"
    assert accent_fold("ἀναθρῴσκω") == "ἀναθρῳσκω"
    assert accent_fold("ὑφαιρέω") != accent_fold("ὑπαιρέω")
    assert FOLDED["περιειμι"] == ("περίειμι",)

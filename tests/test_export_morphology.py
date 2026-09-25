import json
import sys
import unicodedata

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dilemma.form_sanitize import has_editorial_sigla
from export_morphology import (
    _derive_dative_keys,
    _derive_elision_pairs,
    _is_oxytone,
    _derive_nu_forms,
    _load_lemma_forms,
)


VERB_TAGS = [
    "third-person",
    "plural",
    "present",
    "indicative",
    "active",
]


def test_nu_derivation_rejects_editorial_sigla(tmp_path):
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(json.dumps([
        {
            "form": ")λπίζουσι",
            "lemma": ")λπίζω",
            "pos": "verb",
            "tags": VERB_TAGS,
        },
        {
            "form": "ἐλπίζουσι",
            "lemma": "ἐλπίζω",
            "pos": "verb",
            "tags": VERB_TAGS,
        },
    ], ensure_ascii=False), encoding="utf-8")

    forms = _derive_nu_forms([pairs_path])

    assert ")λπίζουσι" not in forms
    assert "ἐλπίζουσι" in forms
    assert not any(has_editorial_sigla(form) for form in forms)


def test_elision_loading_and_derivation_reject_editorial_sigla(tmp_path):
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(json.dumps([
        {"form": "λέγε", "lemma": "λέγω"},
        {"form": "λέγ᾽", "lemma": "λέγω"},
        {"form": ")φέρε", "lemma": "φέρω"},
        {"form": "φέρ᾽", "lemma": ")φέρω"},
    ], ensure_ascii=False), encoding="utf-8")

    lemma_forms = _load_lemma_forms([pairs_path], None)
    assert lemma_forms == {"λέγω": {"λέγε", "λέγ᾽"}}

    # Keep the derivation guard independent of the loader so callers passing
    # an already-built mapping cannot reintroduce contaminated entries.
    lemma_forms[")ἄγω"] = {"ἄγε", "ἄγ᾽"}
    lemma_forms["φέρω"] = {")φέρε", ")φέρ᾽"}
    pairs = _derive_elision_pairs(lemma_forms)

    assert pairs["λέγε"] == "λέγ᾽"
    assert not any(
        has_editorial_sigla(full) or has_editorial_sigla(elided)
        for full, elided in pairs.items()
    )


def test_an_elided_form_keeps_its_own_full_form_stem_marks():
    # εἶπε and εἰπέ elide to different spellings, so neither one's corpus
    # count says anything about which belongs to which full form.
    lemma_forms = {"λέγω": {"Εἶπε", "Εἰπέ", "εἶπ᾽", "εἴπ᾽"}}
    pairs = _derive_elision_pairs(lemma_forms)
    assert pairs["Εἶπε"] == "Εἶπ᾽"
    assert pairs["Εἰπέ"] == "Εἴπ᾽"


def test_the_rules_not_the_corpus_pick_the_spelling():
    # Εἴτε keeps its smooth breathing whatever the counts say: εἵτ᾽ is
    # another spelling, not a variant of this one.
    lemma_forms = {"εἴτε": {"Εἴτε", "εἴτ᾽", "εἵτ᾽"}}
    assert _derive_elision_pairs(lemma_forms)["Εἴτε"] == "Εἴτ᾽"


def test_a_candidate_the_orthography_rules_reject_loses():
    # τὸτ᾽ carries a grave on an elided word.
    lemma_forms = {"τότε": {"Τότε", "τότ᾽", "τὸτ᾽"}}
    assert _derive_elision_pairs(lemma_forms)["Τότε"] == "Τότ᾽"


def test_the_table_does_not_depend_on_set_iteration_order():
    # The candidates arrive from a set, whose iteration order varies with
    # the interpreter's hash seed, so ties have to break on the spelling.
    lemma_forms = {"ζζύω": {"Ζζύε", "ζζύ᾽", "ζζὺ᾽"}}
    first = _derive_elision_pairs(lemma_forms)
    again = _derive_elision_pairs({"ζζύω": set(reversed(sorted(
        lemma_forms["ζζύω"])))})
    assert first == again


def test_an_elided_oxytone_throws_its_accent_back():
    # Smyth 174: ἀνδρί loses its final vowel and the accent goes back onto
    # the penult as an acute. The full form's own stem carries no mark, so
    # the stem-marks test alone would prefer the bare spelling.
    lemma_forms = {"ἀνήρ": {"ἀνδρί", "ἄνδρ᾽", "ἀνδρ᾽"}}
    assert _derive_elision_pairs(lemma_forms)["ἀνδρί"] == "ἄνδρ᾽"


def test_prepositions_and_conjunctions_lose_the_accent_instead():
    # The other half of Smyth 174. οὐδέ is not one of the pinned ten, so
    # nothing else would keep it bare.
    lemma_forms = {"οὐδέ": {"οὐδέ", "οὐδ᾽", "οὔδ᾽"}}
    assert _derive_elision_pairs(lemma_forms)["οὐδέ"] == "οὐδ᾽"


def test_a_form_that_is_not_oxytone_keeps_its_marks_where_they_were():
    # The retraction rule has to say nothing here, or it would override the
    # stem-marks test for every paroxytone and properispomenon.
    lemma_forms = {"ἄλλος": {"ἄλλε", "ἄλλ᾽", "ἆλλ᾽"}}
    assert _derive_elision_pairs(lemma_forms)["ἄλλε"] == "ἄλλ᾽"


def test_a_grave_oxytone_retracts_like_an_acute_one():
    # The grave is only the contextual spelling of an oxytone, and it is
    # the spelling a word carries mid-sentence, which is where elision
    # happens. Reported by Tonos: 59 entries were left bare.
    lemma_forms = {"αὐτός": {"αὐτὸ", "αὔτ᾽", "αὐτ᾽"}}
    assert _derive_elision_pairs(lemma_forms)["αὐτὸ"] == "αὔτ᾽"


def test_a_word_elision_cannot_touch_gets_no_entry():
    # Elision removes a short final vowel. η and ω are always long, a
    # circumflex or iota subscript marks length, and the second element of
    # a diphthong goes with the first.
    for full, elided in [("αὐτῷ", "αὐτ᾽"), ("αὐτῇ", "αὐτ᾽"),
                         ("δεῖ", "δέ᾽"), ("ἤδη", "ἠδ᾽"),
                         ("λόγοι", "λόγ᾽"), ("λόγου", "λόγ᾽")]:
        pairs = _derive_elision_pairs({"x": {full, elided}})
        assert full not in pairs, f"{full} cannot elide"


def test_words_attic_never_elides_get_no_entry():
    # ὅτ᾽ reads as ὅτε, δι᾽ as διά, and οὐχί ends in a long ί.
    lemma_forms = {"ὅτι": {"ὅτι", "ὅτ᾽"}, "περί": {"περί", "περὶ", "περ᾽"},
                   "διό": {"διό", "δι᾽"}, "οὐχί": {"οὐχί", "οὔχ᾽"},
                   "ὅτε": {"ὅτε", "ὅτ᾽"}}
    pairs = _derive_elision_pairs(lemma_forms)
    for word in ("ὅτι", "περί", "περὶ", "διό", "οὐχί"):
        assert word not in pairs, word
    assert pairs["ὅτε"] == "ὅτ᾽"


def test_a_hiatus_or_a_written_out_subscript_is_read_as_such():
    from export_morphology import _can_elide
    # An accented ε before a final ι is a hiatus: βασιλέι is βασιλέϊ, whose
    # short ι could elide were it not a dative.
    assert _can_elide("βασιλέι")
    # An ι after η or ω is the iota subscript written out, and a macron
    # marks a long vowel.
    assert not _can_elide("λόγωι")
    assert not _can_elide("τῆι")
    assert not _can_elide("χώρᾱ")
    # With a diaeresis the ι is a vowel of its own, not a subscript.
    assert _can_elide("ἥρωϊ")


def test_the_dative_ending_does_not_elide(tmp_path):
    # Smyth 72: the dative -ι and -σι elide only in epic, and the elided
    # spelling the corpora attest is another case's (ἄνδρ᾽ is ἄνδρα). The
    # first tagged file that has a form decides, so GLAUx's verb λέγουσι
    # outvotes Diorisis's dative participle, and Diorisis's stray vocative
    # of ἀνδρίς does not save ἀνδρί. τῷδε is a dative too, but it loses the
    # ε of δε.
    glaux = tmp_path / "glaux.json"
    glaux.write_text(json.dumps([
        {"form": "ἀνδρί", "lemma": "ἀνήρ", "tags": ["singular", "dative"]},
        {"form": "λέγουσι", "lemma": "λέγω",
         "tags": ["third-person", "plural", "indicative"]},
        {"form": "τῷδε", "lemma": "ὅδε", "tags": ["singular", "dative"]},
        {"form": "Χερσὶ", "lemma": "Χείρ", "tags": ["plural", "feminine"]},
        {"form": "χερσὶ", "lemma": "χείρ", "tags": ["plural", "dative"]},
    ], ensure_ascii=False), encoding="utf-8")
    diorisis = tmp_path / "diorisis.json"
    diorisis.write_text(json.dumps([
        {"form": "λέγουσι", "lemma": "λέγω",
         "tags": ["plural", "dative", "participle"]},
        {"form": "πᾶσι", "lemma": "πᾶς", "tags": ["plural", "dative"]},
        {"form": "ἀνδρί", "lemma": "ἀνδρίς",
         "tags": ["feminine", "singular", "vocative"]},
    ], ensure_ascii=False), encoding="utf-8")
    datives = _derive_dative_keys([glaux, diorisis])
    assert {key for key, dative in datives.items() if dative} == {
        "ἀνδρί", "τῷδε", "πᾶσι", "χερσί"}

    lemma_forms = {"χείρ": {"χερσὶ", "Χερσὶ", "χέρσ᾽"},
                   "ἀνήρ": {"ἀνδρί", "ἀνδρὶ", "ἄνδρ᾽"},
                   "λέγω": {"λέγουσι", "λέγουσ᾽"},
                   "ὅδε": {"τῷδε", "τῷδ᾽"},
                   "πᾶς": {"πᾶσι", "πᾶσ᾽"}}
    pairs = _derive_elision_pairs(lemma_forms, datives)
    # The capitalized Χερσὶ, analyzed without a case, is read through its
    # lowercase spelling, and does not keep the dative eliding.
    for word in ("ἀνδρί", "ἀνδρὶ", "πᾶσι", "χερσὶ", "Χερσὶ"):
        assert word not in pairs, word
    assert pairs["λέγουσι"] == "λέγουσ᾽"
    assert pairs["τῷδε"] == "τῷδ᾽"


def test_a_participles_dative_plural_takes_movable_nu(tmp_path):
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(json.dumps([
        {"form": "οὖσι", "lemma": "εἰμί", "pos": "verb",
         "tags": ["plural", "present", "participle", "dative"]},
        {"form": "οὖσα", "lemma": "εἰμί", "pos": "verb",
         "tags": ["singular", "present", "participle", "nominative"]},
        {"form": "ὦσι", "lemma": "εἰμί", "pos": "verb",
         "tags": ["third-person", "plural", "present", "subjunctive"]},
        {"form": "ἔχουσἰ", "lemma": "ἔχω", "pos": "verb",
         "tags": ["plural", "present", "participle", "dative"]},
        {"form": "ἒστι", "lemma": "εἰμί", "pos": "verb",
         "tags": ["third-person", "singular", "present", "indicative",
                  "active"]},
    ], ensure_ascii=False), encoding="utf-8")
    forms = _derive_nu_forms([pairs_path])
    assert "οὖσι" in forms
    assert "ὦσι" in forms       # the subjunctive's third plural (Smyth 134)
    assert "οὖσα" not in forms
    # Malformed spellings get no ν appended; a prodelided one is not
    # malformed.
    assert "ἔχουσἰ" not in forms
    assert "ἒστι" not in forms
    assert "στι" in forms


def test_a_pair_is_dropped_when_every_candidate_is_junk():
    # Ranking only picks the least bad one, and writing that into
    # someone's text is worse than offering nothing.
    from export_hunspell import grc_orthography_reason
    assert grc_orthography_reason("ὃσδ᾽") is not None
    assert "ὅσδε" not in _derive_elision_pairs({"ὅσδε": {"ὅσδε", "ὃσδ᾽"}})


def test_a_form_two_lemmas_claim_is_ranked_once():
    # The corpora carry a capitalized lemma Κατά whose one elided token
    # opened a sentence. Assigned lemma by lemma, whichever came last won,
    # and κατὰ took Κατ᾽ from it.
    lemma_forms = {"κατά": {"κατὰ", "κατ᾽", "κάτ᾽"},
                   "Κατά": {"κατὰ", "Κατ᾽"},
                   "Ἑλλάς": {"ἑλλάδα", "Ἑλλάδ᾽"}}
    pairs = _derive_elision_pairs(lemma_forms)
    assert pairs["κατὰ"] == "κατ᾽"
    assert pairs["ἑλλάδα"] == "ἑλλάδ᾽"
    reordered = dict(reversed(list(lemma_forms.items())))
    assert _derive_elision_pairs(reordered) == pairs


def test_the_elided_form_takes_the_full_forms_case():
    lemma_forms = {"αὐτός": {"Αὐτὸ", "αὔτ᾽"}}
    assert _derive_elision_pairs(lemma_forms)["Αὐτὸ"] == "Αὔτ᾽"


def test_an_enclitics_accent_is_not_the_words_own():
    # χεῖρά τε: the acute on the last syllable came from the enclitic and
    # leaves with the elided vowel, so this is no oxytone to retract.
    lemma_forms = {"χείρ": {"χεῖρά", "χεῖρ᾽", "χείρ᾽"},
                   "λέγω": {"εἶπέ", "εἶπ᾽", "εἴπ᾽"},
                   "ἄλλος": {"ἄλλὰ", "ἄλλ᾽", "ἀλλ᾽"}}
    pairs = _derive_elision_pairs(lemma_forms)
    assert pairs["χεῖρά"] == "χεῖρ᾽"
    assert pairs["εἶπέ"] == "εἶπ᾽"
    assert pairs["ἄλλὰ"] == "ἄλλ᾽"


def test_no_entry_when_the_rules_spelling_is_unattested():
    # αἰτία ends in a long α, which cannot elide, and its only candidate is
    # the elided neuter plural αἴτια's. γυναικί would retract to γυναίκ᾽,
    # which no text writes; γυναῖκ᾽ is the accusative's. Either value would
    # replace the user's word with another one.
    lemma_forms = {"αἰτία": {"αἰτία", "αἴτι᾽"},
                   "γυνή": {"γυναικί", "γυναῖκ᾽"}}
    pairs = _derive_elision_pairs(lemma_forms)
    assert "αἰτία" not in pairs
    assert "γυναικί" not in pairs


def test_the_epic_preposition_eni_elides_bare_and_the_numeral_retracts():
    # The corpora file ἐνί under εἷς as well as under ἐν, which is how it
    # took the numeral's ἕν᾽; the answer must not depend on which comes last.
    lemma_forms = {"ἐν": {"ἐνί", "ἐν᾽"},
                   "εἷς": {"ἐνί", "ἑνί", "ἕν᾽"}}
    for order in (lemma_forms, dict(reversed(list(lemma_forms.items())))):
        pairs = _derive_elision_pairs(order)
        assert pairs["ἐνί"] == "ἐν᾽"
        assert pairs["ἑνί"] == "ἕν᾽"


def test_a_rule_spelling_the_orthography_rules_reject_is_not_emitted():
    # ἂρα carries a grave before its last syllable, and the rule keeps it.
    assert "ἂρα" not in _derive_elision_pairs({"ἄρα": {"ἂρα", "ἂρ᾽"}})


def test_a_key_that_opens_with_a_mark_is_skipped():
    # A prodelided ᾽κεῖνα is not a word a keyboard reads.
    lemma_forms = {"ἐκεῖνος": {"᾽κεῖνα", "᾽κεῖν᾽"}}
    assert "᾽κεῖνα" not in _derive_elision_pairs(lemma_forms)


def test_tote_is_an_adverb_that_retracts():
    # τοτέ, "at times", is accented and no enclitic; the lemma τότε lists it.
    lemma_forms = {"τοτέ": {"τοτέ", "τοτ᾽"}, "τότε": {"τότε", "τοτέ", "τότ᾽"}}
    assert _derive_elision_pairs(lemma_forms)["τοτέ"] == "τότ᾽"


def test_dialect_prepositions_and_crasis_elide_bare():
    lemma_forms = {"προτί": {"προτί", "προτ᾽", "πρότ᾽"},
                   "ἐπί": {"τἀπί", "τἀπ᾽", "τἄπ᾽"},
                   "ὑπό": {"ὑπά", "ὑπ᾽", "ὕπ᾽"}}
    pairs = _derive_elision_pairs(lemma_forms)
    assert pairs["προτί"] == "προτ᾽"
    assert pairs["τἀπί"] == "τἀπ᾽"
    assert pairs["ὑπά"] == "ὑπ᾽"


def test_a_circumflex_before_the_penult_hosts_no_enclitic_accent():
    # ὦγαθέ is the crasis ὦ ἀγαθέ, an oxytone, and prints as ὦγάθ᾽.
    lemma_forms = {"ἀγαθός": {"ὦγαθέ", "ὦγάθ᾽"}}
    assert _derive_elision_pairs(lemma_forms)["ὦγαθέ"] == "ὦγάθ᾽"


def test_long_final_vowels_the_spelling_hides_do_not_elide():
    # The deictic -ί (Smyth 333g), the Attic accusative -έᾱ of a noun in
    # -εύς (Smyth 276), a contracted neuter plural in -ᾱ, and a monosyllable
    # not in ε (Smyth 72).
    lemma_forms = {"ὅδε": {"τόνδε", "τονδί", "τόνδ᾽"},
                   "βασιλεύς": {"βασιλέα", "βασιλέ᾽"},
                   "κρέας": {"κρέα", "κρέ᾽"},
                   "σός": {"σά", "σ᾽"},
                   "ἄρα": {"ῥα", "ῥ᾽"}}
    pairs = _derive_elision_pairs(lemma_forms)
    for word in ("τονδί", "βασιλέα", "κρέα", "σά"):
        assert word not in pairs, word
    assert pairs["τόνδε"] == "τόνδ᾽"
    assert pairs["ῥα"] == "ῥ᾽"


def test_a_grave_before_the_last_syllable_is_not_an_accent():
    # μὲτὰ is a malformed μετά, not a proparoxytone carrying an enclitic's
    # accent the way χεῖρά is, so it stays an oxytone.
    assert _is_oxytone("μὲτὰ")
    assert not _is_oxytone("χεῖρά")


ARTIFACT = ROOT / "build" / "hunspell" / "grc_morph.json"


@pytest.mark.skipif(not ARTIFACT.exists(), reason="grc_morph.json not built")
def test_the_built_elision_table_holds_its_invariants():
    """The keyboard rewrites the user's text with this table.

    Tonos's dictionary gate never reads this file, so three separate
    defects reached it before anything here checked them.
    """
    import export_morphology as em
    from export_hunspell import (exact_form_key, grc_orthography_reason,
                                 load_form_profile_freq)

    table = json.loads(ARTIFACT.read_text(encoding="utf-8"))["el"]
    assert table

    non_elidable = [k for k in table if not em._can_elide(k)]
    assert not non_elidable, f"keys elision cannot touch: {non_elidable[:10]}"

    invalid = {k: v for k, v in table.items()
               if grc_orthography_reason(v) is not None}
    assert not invalid, f"values the orthography rules reject: {list(invalid)[:10]}"

    # The keyboard writes the value into the user's text as it stands, so
    # a capital here puts one mid-sentence (κατὰ -> Κατ᾽ did, 226,495
    # occurrences).
    recased = {k: v for k, v in table.items()
               if k[:1].isupper() != v[:1].isupper()}
    assert not recased, f"values that change the key's case: {list(recased.items())[:10]}"

    # Elision drops the last vowel and may move the accent, nothing else. A
    # value that changes a letter, a breathing or an iota subscript is the
    # elision of another word (ἐνί took the numeral's ἕν᾽), and a key whose
    # accent sits before its last vowel keeps that accent where it was
    # (αἰτία took αἴτι᾽, the neuter plural's). An oxytone may add the acute
    # it throws back (ὦγαθέ gives ὦγάθ᾽), and that is all it may add.
    accents = {"\u0300", "\u0301", "\u0342"}

    def keeps(kept, own):
        return kept == own or any(
            kept[:i] + kept[i + 1:] == own
            for i, c in enumerate(kept) if c == "\u0301")

    def stem(form):
        nfd = unicodedata.normalize("NFD", form)
        last = max(i for i, c in enumerate(nfd) if c.lower() in "αεηιουω")
        return nfd[:last]

    def unaccented(nfd):
        return "".join(c for c in nfd if c not in accents).lower()

    other_word, moved = {}, {}
    for k, v in table.items():
        kept, own = unicodedata.normalize("NFD", v)[:-1], stem(k)
        if unaccented(kept) != unaccented(own):
            other_word[k] = v
        elif any(c in accents for c in own) and not keeps(kept, own):
            moved[k] = v
    assert not other_word, f"values that spell another word: {list(other_word.items())[:10]}"
    assert not moved, f"values that move the key's own accent: {list(moved.items())[:10]}"

    # Oxytones are pinned by corpus weight rather than by count, because
    # weight is what separates a rule that stopped firing from the tail of
    # the known-bare class. That tail is crasis forms (κἀπί), dialect
    # particles (ποκά, πεδά, προτί) and OCR fragments, and stands at 56
    # entries carrying 11,679 occurrences. The regression Tonos reported
    # carried 119,486.
    freq = load_form_profile_freq().exact
    if not freq:
        pytest.skip("form_profile.db not downloaded")
    bare = [k for k, v in table.items()
            if grc_orthography_reason(k) is None
            and em._is_oxytone(k)
            and not em._elides_bare(k)
            and not em._has_accent(v)]
    weight = sum(freq.get(exact_form_key(k), 0) for k in bare)
    assert weight < 25_000, (
        f"elided oxytones left bare carry {weight:,} occurrences: "
        f"{sorted(bare, key=lambda k: -freq.get(exact_form_key(k), 0))[:10]}"
    )

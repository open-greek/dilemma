"""Grammatical evidence for transformations and explicit verb selection."""

import json

import pytest

from dilemma import core


@pytest.fixture
def fallback_model(tmp_path, monkeypatch):
    """Use small source tables without loading the shipped lookup or model."""
    monkeypatch.setattr(core.Dilemma, "_load_lookups", lambda *a, **k: None)
    monkeypatch.setattr(core.Dilemma, "_check_backend", lambda *a: None)
    for name in ("AG_POS_LOOKUP_PATH", "GLAUX_POS_LOOKUP_PATH",
                 "TREEBANK_POS_LOOKUP_PATH", "MG_POS_LOOKUP_PATH"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(core, name, path)

    def make(lookup, readings):
        core.AG_POS_LOOKUP_PATH.write_text(
            json.dumps(readings, ensure_ascii=False), encoding="utf-8")
        model = core.Dilemma(lang="grc", skip_pos=True)
        model._lookup = lookup
        model._ag_lookup = lookup
        return model

    return make


@pytest.mark.parametrize("word", ["ἔπιναι", "ἠπιναι", "πεπιναι"])
def test_augment_and_reduplication_cannot_use_a_noun_homograph(fallback_model, word):
    model = fallback_model(
        {"πιναι": "πίνη", "πίναι": "πίνη"},
        {"πίναι": {"NOUN": "πίνη"}, "πίνῃ": {"VERB": "πίνω"}},
    )
    assert model._strip_verb_morphology(word) is None
    assert model._lookup_word("πίναι") == "πίνη"


def test_augment_prefers_evidenced_verb_beside_noun(fallback_model):
    model = fallback_model(
        {"λεγον": "λέγος"},
        {"λέγον": {"VERB": "λέγω"}},
    )
    assert model._strip_verb_morphology("ἔλεγον") == "λέγω"
    assert model._lookup_word("λεγον") == "λέγος"


def test_temporal_augment_does_not_promote_a_supported_nominal_reading(fallback_model):
    model = fallback_model(
        {"οργην": "ὀργή"},
        {"οργην": {"NOUN": "ὀργή", "VERB": "ὀργάω"}},
    )
    assert model._strip_verb_morphology("ωργην") is None


def test_temporal_augment_does_not_replace_a_nominal_selfmap(fallback_model):
    model = fallback_model({"οικος": "οἶκος"},
                           {"οικος": {"VERB": "ἔοικα"}})
    assert model._strip_verb_morphology("ωικος") is None


def test_ordinary_verb_target_supports_a_missing_compact_form(fallback_model):
    model = fallback_model(
        {"ποίησεν": "ποιέω"}, {"ποιῶν": {"VERB": "ποιέω"}},
    )
    assert model._strip_verb_morphology("ἐποίησεν") == "ποιέω"


def test_missing_grammatical_sources_cannot_authorize_augment(fallback_model):
    model = fallback_model({"ποίησεν": "ποιέω"}, {})
    assert model._strip_verb_morphology("ἐποίησεν") is None


def test_fallback_evidence_is_lazy_cached_and_does_not_enable_pos(fallback_model):
    model = fallback_model({"ποίησεν": "ποιέω"},
                           {"ποιῶν": {"VERB": "ποιέω"}})
    assert model._fallback_readings is None
    assert model._lookup_word("ποίησεν") == "ποιέω"
    assert model._fallback_readings is None
    assert model._strip_verb_morphology("ἐποίησεν") == "ποιέω"
    cached = model._fallback_readings
    assert model._strip_verb_morphology("ἐποίησεν") == "ποιέω"
    assert model._fallback_readings is cached
    assert model._pos_lookup == {}


def test_normalized_noun_match_needs_form_reading_evidence(fallback_model):
    model = fallback_model({"φάσκοντε": "φάσκον"},
                           {"φάσκον": {"NOUN": "φάσκον", "VERB": "φάσκω"}})
    assert model._normalize_byzantine("φάσκονται") is None
    assert model._lookup_word("φάσκοντε") == "φάσκον"


def test_normalized_noun_with_source_evidence_survives(fallback_model):
    model = fallback_model({"γυναίκαις": "γυνή"},
                           {"γυναίκαις": {"NOUN": "γυνή"}})
    assert model._normalize_byzantine("γυναίκες") == "γυνή"


def test_unambiguous_normalization_does_not_require_a_sparse_pos_entry(fallback_model):
    model = fallback_model({"γυναίκαις": "γυνή"}, {})
    assert model._normalize_byzantine("γυναίκες") == "γυνή"


@pytest.mark.parametrize("form,nominal,verbal_form,verb", [
    ("κύνας", "κύων", "κυνᾶς", "κυνάω"),
    ("νίκην", "νίκη", "νικῆν", "νικάω"),
])
def test_accented_nominal_reading_does_not_borrow_a_verbal_homograph(
        fallback_model, form, nominal, verbal_form, verb):
    model = fallback_model({form: nominal},
                           {form: {"NOUN": nominal}, verbal_form: {"VERB": verb}})
    assert model._lookup_grammatical_word(form, verb_only=True) is None
    assert model._lookup_word(form) == nominal


def test_compound_uses_verbal_reading_of_independent_noun_homograph(fallback_model):
    model = fallback_model(
        {"ηγουμενην": "ἡγούμενος", "ἡγούμενος": "ἡγούμενος"},
        {"ἡγουμένην": {"VERB": "ἡγέομαι"},
         "ἡγούμενος": {"NOUN": "ἡγούμενος", "VERB": "ἡγέομαι"}},
    )
    _headwords(model, "διηγέομαι")
    assert model._decompose_compound("διηγουμένην") == "διηγέομαι"
    assert model._decompose_compound_all("διηγουμένην") == [
        ("διηγέομαι", "ἡγέομαι", "δι")]
    assert model._lookup_word("ἡγούμενος") == "ἡγούμενος"


def test_nominal_compound_is_not_rejected_by_verb_ending_rules(fallback_model):
    model = fallback_model({"λογους": "λόγος"},
                           {"λόγους": {"NOUN": "λόγος"}})
    _headwords(model, "πολύλογος")
    assert model._decompose_compound("πολυλόγους") == "πολύλογος"


def _headwords(model, *headwords, source="lsj"):
    """Give the compound accenting step a small dictionary of its own."""
    index = {}
    for h in headwords:
        index.setdefault(core.strip_accents(h.lower()), []).append(h)
    model._compound_headword_indexes = {
        s: (index if s == source else {})
        for s in (model._COMPOUND_AG_HEADWORD_SOURCES
                  + model._COMPOUND_MG_HEADWORD_SOURCES)}


def test_compound_lemma_is_accented_from_a_headword(fallback_model):
    # Issue #101: ολεθρο + τόκος was returned as the bare skeleton.
    model = fallback_model({"τοκον": "τόκος"}, {"τόκον": {"NOUN": "τόκος"}})
    _headwords(model, "ὀλεθροτόκος", source="lbg")
    assert model._decompose_compound("ὀλεθροτόκον") == "ὀλεθροτόκος"
    assert model._decompose_compound_all("ὀλεθροτόκον") == [
        ("ὀλεθροτόκος", "τόκος", "ολεθρο")]


def test_compound_without_a_headword_abstains(fallback_model):
    model = fallback_model({"τοκον": "τόκος"}, {"τόκον": {"NOUN": "τόκος"}})
    _headwords(model)
    assert model._decompose_compound("ὀλεθροτόκον") is None
    assert model._decompose_compound_all("ὀλεθροτόκον") == []


def test_compound_prefers_the_earlier_dictionary(fallback_model):
    model = fallback_model({"σωσει": "σῴζω"}, {"σώσει": {"VERB": "σῴζω"}})
    _headwords(model)
    model._compound_headword_indexes["lsj"] = {"αποσωζω": ["ἀποσῴζω"]}
    model._compound_headword_indexes["ag"] = {"αποσωζω": ["ἀποσώζω"]}
    assert model._decompose_compound("ἀποσώσει") == "ἀποσῴζω"


def test_lowercase_compound_does_not_take_a_proper_noun_headword(fallback_model):
    model = fallback_model({"τοκον": "τόκος"}, {"τόκον": {"NOUN": "τόκος"}})
    _headwords(model, "Ὀλεθροτόκος")
    assert model._decompose_compound("ὀλεθροτόκον") is None


@pytest.mark.parametrize("word,base,base_lemma", [
    ("μοναχικὴν", "κην", "ἐν"),        # the crasis κἠν (καὶ ἐν)
    ("βανδίνην", "νην", "νάω"),
])
def test_compound_base_must_begin_like_its_lemma(fallback_model, word, base, base_lemma):
    model = fallback_model({base: base_lemma}, {})
    assert model._decompose_compound(word) is None
    assert model._decompose_compound_all(word) == []


def test_prefix_reconstruction_uses_supported_reading(fallback_model):
    model = fallback_model(
        {"λεγοντων": "λέγος", "επιλεγω": "ἐπιλέγω"},
        {"λεγόντων": {"VERB": "λέγω"}},
    )
    assert model._prefix_strip_lookup("ἐπιλεγόντων") == "ἐπιλέγω"


def test_competing_source_readings_are_not_arbitrarily_selected(fallback_model):
    model = fallback_model({"λεγον": "λέγος"},
                           {"λέγον": {"VERB": "λέγω"}})
    core.GLAUX_POS_LOOKUP_PATH.write_text(
        json.dumps({"λεγόν": {"VERB": "ἄλλω"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    assert model._lookup_grammatical_word("λεγον", verb_only=True) is None


def pos_model(lemmas, pos_lemma):
    model = core.Dilemma.__new__(core.Dilemma)
    model.lang = "grc"
    model._convention_name = None
    model._pos_rejected_targets = frozenset()  # This fixture supplies no vetoed targets.
    candidates = [core.LemmaCandidate(lemma=value, source="lookup",
                  proper=value[:1].isupper()) for value in lemmas]
    model._lemmatize_verbose_impl = lambda word, guess: candidates
    model._pos_table_lookup = lambda word, upos: pos_lemma
    model._article_pos_lemma = lambda word, upos: None
    model._apply_convention = lambda lemma: lemma
    model.lemmatize_batch = lambda words, **kwargs: [lemmas[0]] * len(words)
    model.lemmatize = lambda word, **kwargs: lemmas[0]
    return model


@pytest.mark.parametrize("word,lemmas,verb", [
    ("Λάμπε", ["λάμπος", "Λάμπος"], "λάμπω"),
    ("Λήθειν", ["ληθώ", "Ληθώ"], "λανθάνω"),
])
def test_explicit_verb_overrides_a_single_case_twin_family(word, lemmas, verb):
    model = pos_model(lemmas, verb)
    assert model.lemmatize_pos(word, "VERB", guess=False) == verb
    assert model.lemmatize_batch_pos([word], ["VERB"], guess=False) == [verb]


def test_explicit_verb_does_not_discard_distinct_competing_lemmas():
    model = pos_model(["πρῶτος", "δεύτερος"], "ἄλλω")
    assert model.lemmatize_pos("τύπος", "VERB", guess=False) == "πρῶτος"
    assert model.lemmatize_batch_pos(["τύπος"], ["VERB"], guess=False) == ["πρῶτος"]


def test_unmatched_noun_does_not_override_an_independent_case_family():
    model = pos_model(["ωρολόγιο", "Ωρολόγιο"], "ὡρολόγιον")
    assert model.lemmatize_pos("Ωρολογίου", "NOUN", guess=False) == "ωρολόγιο"
    assert model.lemmatize_batch_pos(["Ωρολογίου"], ["NOUN"], guess=False) == ["ωρολόγιο"]


@pytest.fixture(scope="module")
def shipped_models():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    return [core.Dilemma(lang="grc", skip_pos=skip) for skip in (False, True)]


@pytest.mark.parametrize("normalize", [False, True])
def test_shipped_noun_homographs_and_supported_fallbacks(shipped_models, normalize):
    from dilemma.normalize import Normalizer
    for model in shipped_models:
        model._normalizer = Normalizer(period="byzantine") if normalize else None
        assert model.lemmatize("φάσκονται", guess=False) is None
        assert model._strip_verb_morphology("ἔπιναι") is None
        # The old augment path returned the noun πίνη. The normalizer may
        # still resolve ἐπεῖναι, an attested infinitive of ἔπειμι.
        assert model.lemmatize("ἔπιναι", guess=False) == "ἔπειμι"
        assert model.lemmatize("διηγουμένην", guess=False) == "διηγέομαι"
        assert model.lemmatize_batch(["φάσκονται", "διηγουμένην"], guess=False) == [
            None, "διηγέομαι"]
        assert model.lemmatize("πίναι", guess=False) == "πίνη"
        assert model.lemmatize("ἡγούμενος", guess=False) == "ἡγούμενος"
        assert model._strip_verb_morphology("ἐποίησεν") == "ποιέω"
        assert model.lemmatize_verbose("φάσκονται", guess=False) == []
        assert all(c.lemma != "πίνη" for c in
                   model.lemmatize_verbose("ἔπιναι", guess=False))


@pytest.mark.parametrize("form,lemma", [
    ("λεγιν", "λέγω"), ("ανθροπων", "ἄνθρωπος"),
    ("λωγου", "λόγος"), ("σωματως", "σῶμα"), ("εχιν", "ἔχω"),
    ("αυτωις", "αὐτός"), ("γυνεκα", "γυνή"),
    ("προτην", "πρῶτος"), ("μεντωι", "μέντοι"),
    ("ωυτως", "οὕτως"), ("ωικος", "οἶκος"), ("ωικου", "οἶκος"),
    ("ωργην", "ὀργή"), ("ωργανον", "ὄργανον"),
    ("αλιθες", "ἀληθής"), ("πρωβατα", "πρόβατον"),
    ("χωριωυ", "χωρίον"), ("ωργανων", "ὄργανον"),
])
def test_shipped_normalizer_retains_ordinary_grammatical_readings(
        shipped_models, form, lemma):
    from dilemma.normalize import Normalizer
    for model in shipped_models:
        model._normalizer = Normalizer(period="byzantine")
        assert model.lemmatize(form, guess=False) == lemma
        assert model.lemmatize_batch([form], guess=False) == [lemma]
        assert model.lemmatize_verbose(form, guess=False)[0].lemma == lemma


def test_shipped_explicit_verbs_preserve_independent_names(shipped_models):
    model = shipped_models[0]
    assert model.lemmatize_batch_pos(["Λάμπε", "Λήθειν"], ["VERB", "VERB"],
                                     guess=False) == ["λάμπω", "λανθάνω"]
    for word, lemma in [("Λάμπε", "λάμπω"), ("Λήθειν", "λανθάνω")]:
        assert model.lemmatize_pos(word, "VERB", guess=False) == lemma
    assert model.lemmatize("Λάμπε", guess=False) == "Λάμπος"
    assert model.lemmatize("Λήθειν", guess=False) == "Ληθώ"


def _all_model(fallback_model, lookup, ancient_pos, modern_pos, headwords=()):
    """A lang="all" model with separate Ancient and Modern Greek POS entries."""
    model = fallback_model(lookup, {})
    model.lang = "all"
    model._pos_ag_lookup = ancient_pos
    combined = {form: dict(by_pos) for form, by_pos in ancient_pos.items()}
    for form, by_pos in modern_pos.items():
        for upos, lemma in by_pos.items():
            combined.setdefault(form, {}).setdefault(upos, lemma)
    model._pos_lookup = combined
    model._headword_sets["lsj"] = set(headwords)
    model._ancient_lexical_heads = set()
    return model


def test_polytonic_form_does_not_take_a_modern_pos_reading(fallback_model):
    # Issue #100: with lang="all" the combined POS table matched the
    # monotonic variant of a polytonic form and returned a Modern Greek
    # headword (ἄλλην -> άλλος).
    model = _all_model(
        fallback_model, {"ἄλλην": "ἄλλος"},
        {"ἀπαστράπτων": {"VERB": "ἀπαστράπτω"}},
        {"άλλην": {"PRON": "άλλος"}, "απαστράπτων": {"VERB": "απαστράπτω"}})
    assert model._pos_table_lookup("ἄλλην", "PRON") is None
    assert model._pos_table_lookup("ἀπαστράπτων", "VERB") == "ἀπαστράπτω"
    # Monotonic input still reads the Modern Greek entries.
    assert model._pos_table_lookup("άλλην", "PRON") == "άλλος"


def test_same_word_in_monotonic_spelling_never_stands(fallback_model):
    # Even a conflicting Ancient tag does not turn ἄλλος into άλλος.
    model = _all_model(
        fallback_model, {"ἄλλην": "ἄλλος"},
        {"ἄλλην": {"VERB": "ἄλλος"}}, {"άλλην": {"PRON": "άλλος"}})
    assert model._pos_table_lookup("ἄλλην", "PRON") is None


def test_only_the_first_modern_entry_is_considered(fallback_model):
    # 1.3.22 returned the first usable entry among the form's variants;
    # that entry is the one judged. A rejected monotonic entry does not
    # hand the form to the accent-stripped one.
    model = _all_model(
        fallback_model, {"ἄλλην": "ἄλλος"}, {"ἄλλην": {"VERB": "ἄλλος"}},
        {"άλλην": {"PRON": "άλλος"}, "αλλην": {"PRON": "αλλήνα"}})
    assert model._pos_table_lookup("ἄλλην", "PRON") is None
    del model._pos_lookup["άλλην"]
    assert model._pos_table_lookup("ἄλλην", "PRON") == "αλλήνα"


def test_modern_entry_is_checked_like_any_table_value(fallback_model):
    # An elided fragment is no headword: the next variant's entry is used.
    model = _all_model(
        fallback_model, {"νά": "νή"}, {"νά": {"NOUN": "νή"}},
        {"νά": {"CCONJ": "ν᾽"}, "να": {"CCONJ": "να"}})
    assert model._pos_table_lookup("νὰ", "CCONJ") == "να"
    # A grave citation form is returned with its acute.
    model = _all_model(
        fallback_model, {"κυροῦ": "κυρόω"}, {"κυρου": {"VERB": "κυρόω"}},
        {"κυρου": {"NOUN": "κυρὸς"}})
    model._trusted_grave_citation_targets = lambda: {"κυρός"}
    assert model._pos_table_lookup("κυροῦ", "NOUN") == "κυρός"


def test_decomposed_input_is_judged_like_composed_input(fallback_model):
    import unicodedata
    model = _all_model(
        fallback_model, {"ὄντων": "εἰμί"}, {"ὄντων": {"VERB": "εἰμί"}},
        {"οντων": {"ADJ": "αρχαΐζων"}})
    decomposed = unicodedata.normalize("NFD", "ὄντων")
    assert decomposed != "ὄντων"
    assert model._pos_table_lookup("ὄντων", "ADJ") is None
    assert model._pos_table_lookup(decomposed, "ADJ") is None
    # The Ancient POS table knows the spelling by its composed key.
    model = _all_model(
        fallback_model, {"σέ": "σέω"}, {"σὲ": {"NOUN": "σής"}}, {})
    for spelling in ("σὲ", unicodedata.normalize("NFD", "σὲ")):
        assert not model._modern_pos_reading_applies(spelling, "PRON", "εσύ")


@pytest.mark.parametrize("word,lookup,ancient_pos,modern_pos,upos", [
    # Ἕλλην is recorded as a noun; a PROPN request must not bring Έλληνας.
    ("Ἑλλήνων", {"Ἑλλήνων": "Ἕλλην"}, {"Ἑλλήνων": {"NOUN": "Ἕλλην"}},
     {"ελλήνων": {"PROPN": "Έλληνας"}}, "PROPN"),
    # A pronominal adjective answers a pronoun request (σός, not σας).
    ("σὰς", {"σάς": "σός"}, {"σάς": {"ADJ": "σός"}},
     {"σας": {"PRON": "σας"}}, "PRON"),
    # An adverbial neuter keeps its adjective.
    ("ὑψηλὰ", {"ὑψηλά": "ὑψηλός"}, {"ὑψηλά": {"ADJ": "ὑψηλός"}},
     {"υψηλα": {"ADV": "ψηλά"}}, "ADV"),
    # One compatible tag among the recorded ones is enough.
    ("σὰς", {"σάς": "σός"}, {"σάς": {"NOUN": "σός", "ADJ": "σός"}},
     {"σας": {"PRON": "σας"}}, "PRON"),
    # The POS tables may spell the lookup's lemma with other accents or
    # capitals (Δαμασκηνός, δαμασκηνός).
    ("δαμασκηνοῦ", {"δαμασκηνού": "Δαμασκηνός"},
     {"δαμασκηνου": {"ADJ": "δαμασκηνός"}},
     {"δαμασκηνου": {"NOUN": "δαμάσκηνο"}}, "NOUN"),
])
def test_compatible_ancient_reading_keeps_a_modern_headword_out(
        fallback_model, word, lookup, ancient_pos, modern_pos, upos):
    model = _all_model(fallback_model, lookup, ancient_pos, modern_pos)
    assert model._pos_table_lookup(word, upos) is None


def test_ancient_reading_without_recorded_pos_is_kept(fallback_model):
    model = _all_model(
        fallback_model, {"δαμασκηνοῦ": "Δαμασκηνός"}, {},
        {"δαμασκηνου": {"NOUN": "δαμάσκηνο"}})
    assert model._pos_table_lookup("δαμασκηνοῦ", "NOUN") is None


@pytest.mark.parametrize("word,lookup,ancient_pos,modern_pos,upos,lemma", [
    # A closed-class tag conflicts with the only Ancient reading, even for
    # a spelling the Ancient tables know (νά as νή).
    ("νὰ", {"νά": "νή"}, {"νά": {"NOUN": "νή"}},
     {"να": {"CCONJ": "να"}}, "CCONJ", "να"),
    # An open-class tag conflicts and the Ancient POS table does not know
    # the accented spelling at all.
    ("κυροῦ", {"κυροῦ": "κυρόω"}, {"κυρου": {"VERB": "κυρόω"}},
     {"κυρου": {"NOUN": "κυρός"}}, "NOUN", "κυρός"),
])
def test_modern_reading_stands_on_a_tag_conflict(
        fallback_model, word, lookup, ancient_pos, modern_pos, upos, lemma):
    model = _all_model(fallback_model, lookup, ancient_pos, modern_pos)
    assert model._pos_table_lookup(word, upos) == lemma


@pytest.mark.parametrize("upos,stands", [
    ("PRON", True), ("DET", True), ("ADP", True), ("CCONJ", True),
    ("SCONJ", True), ("PART", True), ("INTJ", True),
    ("VERB", False), ("AUX", False), ("ADV", False), ("NUM", False),
])
def test_only_closed_class_tags_displace_a_known_spelling(
        fallback_model, upos, stands):
    model = _all_model(
        fallback_model, {"νά": "νή"}, {"νά": {"NOUN": "νή"}},
        {"να": {upos: "να"}})
    assert model._pos_table_lookup("νὰ", upos) == ("να" if stands else None)


def test_only_the_ancient_reading_s_own_tags_count(fallback_model):
    # A tag the Ancient table records for another lemma of the form says
    # nothing about the lookup's reading (σέω).
    model = _all_model(
        fallback_model, {"σεῖς": "σέω"}, {"σεις": {"ADJ": "σός"}},
        {"σεις": {"PRON": "εσείς"}})
    assert model._pos_table_lookup("σεῖς", "PRON") == "εσείς"
    # Nor does a Modern Greek entry that names the Ancient lemma.
    model = _all_model(
        fallback_model, {"νά": "νή"}, {"νά": {"NOUN": "νή"}},
        {"να": {"CCONJ": "να", "SCONJ": "νη"}})
    assert model._pos_table_lookup("νὰ", "CCONJ") == "να"


@pytest.mark.parametrize("word,lookup,ancient_pos,modern_pos,upos,headwords", [
    # A mistagged Ancient form whose spelling the Ancient POS table knows.
    ("ὄντων", {"ὄντων": "εἰμί"}, {"ὄντων": {"VERB": "εἰμί"}},
     {"οντων": {"ADJ": "αρχαΐζων"}}, "ADJ", ()),
    # A self-mapped Ancient headword, recorded only under another tag.
    ("οὖν", {"οὖν": "οὖν"}, {"ουν": {"ADV": "οὖν"}},
     {"ουν": {"VERB": "σαλπάρω"}}, "VERB", ("οὖν",)),
    # The same for a headword met with a grave accent.
    ("ἐπεὶ", {"ἐπεί": "ἐπεί"}, {"επει": {"SCONJ": "ἐπεί"}},
     {"επει": {"VERB": "πείθω"}}, "VERB", ("ἐπεί",)),
])
def test_open_class_conflict_does_not_displace_a_known_ancient_form(
        fallback_model, word, lookup, ancient_pos, modern_pos, upos, headwords):
    model = _all_model(fallback_model, lookup, ancient_pos, modern_pos,
                       headwords=headwords)
    assert model._pos_table_lookup(word, upos) is None


def test_known_ancient_form_without_a_recorded_tag_is_kept(fallback_model):
    # ἄνω is a headword whose part of speech nothing records.
    model = _all_model(
        fallback_model, {"ἄνω": "ἄνω"}, {}, {"ανω": {"ADP": "πάνω"}},
        headwords={"ἄνω"})
    assert model._pos_table_lookup("ἄνω", "ADP") is None
    # The Ancient POS table holds the accented spelling for another lemma.
    model = _all_model(
        fallback_model, {"σεῖς": "σέω"}, {"σεῖς": {"NOUN": "σής"}},
        {"σεις": {"PRON": "εσείς"}})
    assert model._pos_table_lookup("σεῖς", "PRON") is None


def test_modern_reading_stands_when_nothing_reads_the_form(fallback_model):
    model = _all_model(fallback_model, {}, {}, {"αλλην": {"PRON": "άλλος"}})
    assert model._pos_table_lookup("ἀλλην", "PRON") == "άλλος"
    # A reading from the article, elision or crasis resolvers is one.
    model._lemmatize_verbose_impl = lambda word, guess=True: [object()]
    assert model._pos_table_lookup("ἀλλην", "PRON") is None


def test_unaccented_echo_that_is_no_headword_yields(fallback_model):
    modern = {"αλλην": {"PRON": "άλλος"}}
    model = _all_model(fallback_model, {"ἀλλην": "ἀλλην"}, {}, modern)
    assert model._pos_table_lookup("ἀλλην", "PRON") == "άλλος"
    assert model._pos_table_lookup("Ἀλλην", "PRON") == "άλλος"
    # The same echo is kept once a dictionary lists it.
    model = _all_model(fallback_model, {"ἀλλην": "ἀλλην"}, {}, modern,
                       headwords={"ἀλλην"})
    assert model._pos_table_lookup("ἀλλην", "PRON") is None


@pytest.mark.parametrize("word,lemma,upos,modern", [
    ("Σολομὼν", "Σολομών", "PROPN", "Σολομώντας"),
    ("Ἀγάθων", "Ἀγάθων", "NOUN", "αγαθός"),
    ("πρωτοπαπᾶς", "πρωτοπαπᾶς", "NOUN", "πρωτοπρεσβύτερος"),
])
def test_accented_self_mapped_word_is_not_an_echo(
        fallback_model, word, lemma, upos, modern):
    # A word with a grave, acute or circumflex that no dictionary list
    # holds keeps its lemma (Σολομών, not the Modern Σολομώντας).
    model = _all_model(
        fallback_model, {lemma: lemma}, {},
        {core.strip_accents(word.lower()): {upos: modern}})
    assert model._pos_table_lookup(word, upos) is None


def test_lexical_head_list_counts_as_ancient_headword(fallback_model):
    model = _all_model(
        fallback_model, {"ἀλλην": "ἀλλην"}, {}, {"αλλην": {"PRON": "άλλος"}})
    model._ancient_lexical_heads = {"ἀλλην"}
    assert model._pos_table_lookup("ἀλλην", "PRON") is None
    # The list holds its heads without quantity marks.
    model = _all_model(
        fallback_model, {"ἰσος": "ἰ̄σος"}, {}, {"ισος": {"ADJ": "ίσιος"}})
    assert model._pos_table_lookup("ἰσος", "ADJ") == "ίσιος"
    model._ancient_lexical_heads = {"ἰσος"}
    assert model._pos_table_lookup("ἰσος", "ADJ") is None


def test_reviewed_verb_form_is_an_ancient_spelling(fallback_model):
    # _lookup_word reads reviewed verb forms before any lookup table. The
    # Modern entry is keyed by the monotonic spelling, so only the
    # reviewed-form clause keeps the verb.
    model = _all_model(fallback_model, {}, {}, {"απήα": {"NOUN": "απήα"}})
    assert model._lookup_word("ἀπῇα") == "ἄπειμι"
    assert model._ancient_lookup_has_spelling("ἀπῇα")
    assert model._pos_table_lookup("ἀπῇα", "NOUN") is None


def test_reading_found_only_without_accents_yields(fallback_model):
    # ὧρες is no Ancient spelling; ὄρνυμι was matched through ώρες.
    modern = {"ώρες": {"NOUN": "ώρα"}}
    model = _all_model(fallback_model, {"ώρες": "ὄρνυμι"}, {}, modern)
    assert model._pos_table_lookup("ὧρες", "NOUN") == "ώρα"
    model = _all_model(fallback_model, {"ὧρες": "ὄρνυμι"}, {}, modern)
    assert model._pos_table_lookup("ὧρες", "NOUN") is None


def test_closed_class_tag_does_not_keep_a_verb_lemma(fallback_model):
    model = _all_model(
        fallback_model, {"σεῖς": "σέω"}, {},
        {"σεις": {"PRON": "εσείς", "NOUN": "σείση"}})
    assert model._pos_table_lookup("σεῖς", "PRON") == "εσείς"
    # An open-class tag gives no such evidence.
    assert model._pos_table_lookup("σεῖς", "NOUN") is None
    # Nor does a reading that is no verb lemma.
    model = _all_model(
        fallback_model, {"σεῖς": "σεύς"}, {}, {"σεις": {"PRON": "εσείς"}})
    assert model._pos_table_lookup("σεῖς", "PRON") is None


def test_lowercase_word_does_not_take_a_capitalized_modern_name(fallback_model):
    model = _all_model(
        fallback_model, {"γένῃ": "γίγνομαι"}, {"γενη": {"VERB": "γίγνομαι"}},
        {"γενη": {"PROPN": "Γένη", "NOUN": "γένι"}})
    assert model._pos_table_lookup("γένῃ", "PROPN") is None
    assert model._pos_table_lookup("γένῃ", "NOUN") == "γένι"
    assert model._pos_table_lookup("Γένῃ", "PROPN") == "Γένη"


def test_pos_tag_compatibility():
    compatible = core.Dilemma._pos_tags_compatible
    for first, second in [("PROPN", "NOUN"), ("NOUN", "ADJ"), ("PRON", "ADJ"),
                          ("DET", "PRON"), ("ADV", "ADJ"), ("PART", "ADV"),
                          ("SCONJ", "CCONJ"), ("AUX", "VERB"), ("ADP", "ADP")]:
        assert compatible(first, second) and compatible(second, first)
    for first, second in [("NOUN", "VERB"), ("CCONJ", "PART"), ("NOUN", "PRON"),
                          ("ADP", "PRON"), ("CCONJ", "NOUN"), ("PART", "ADJ")]:
        assert not compatible(first, second)


def test_same_word_with_another_accent_never_stands(fallback_model):
    # μά and μα have the same letters; the accent does not make them two
    # words, even under a conflicting closed-class tag.
    model = _all_model(
        fallback_model, {"μά": "μά"}, {"μά": {"PART": "μά"}},
        {"μα": {"CCONJ": "μα"}})
    assert model._pos_table_lookup("μὰ", "CCONJ") is None


def test_same_word_is_recognized_under_a_capital(fallback_model):
    # Ἀθηνᾶ and Αθηνά are one name.
    model = _all_model(
        fallback_model, {"αθηνα": "Ἀθηνᾶ"}, {}, {"αθηνα": {"PROPN": "Αθηνά"}})
    assert model._pos_table_lookup("Ἀθηνᾶ", "PROPN") is None


def test_unaccented_echo_does_not_take_a_capitalized_modern_name(
        fallback_model):
    model = _all_model(
        fallback_model, {"ἀλλην": "ἀλλην"}, {},
        {"αλλην": {"PROPN": "Αλλήνα"}})
    assert model._pos_table_lookup("ἀλλην", "PROPN") is None
    assert model._pos_table_lookup("Ἀλλην", "PROPN") == "Αλλήνα"


def test_headword_written_without_its_accent_is_a_known_spelling(
        fallback_model):
    # ἀνω has the letters of the headword ἄνω, which the lookup reaches
    # through the accent-stripped key.
    model = _all_model(
        fallback_model, {"ανω": "ἄνω"}, {}, {"ανω": {"ADP": "πάνω"}},
        headwords={"ἄνω"})
    assert model._pos_table_lookup("ἀνω", "ADP") is None


def test_headword_lemma_does_not_make_its_inflected_form_known(
        fallback_model):
    # κυρόω is a headword; κυροῦ is not one.
    model = _all_model(
        fallback_model, {"κυροῦ": "κυρόω"}, {"κυρου": {"VERB": "κυρόω"}},
        {"κυρου": {"NOUN": "κυρός"}}, headwords={"κυρόω"})
    assert model._pos_table_lookup("κυροῦ", "NOUN") == "κυρός"


def test_capitalized_form_is_known_by_its_lowercase_key(fallback_model):
    model = _all_model(
        fallback_model, {"οὕς": "ὅς"},
        {"οὓς": {"PRON": "ὅς"}, "ους": {"PRON": "ὅς"}},
        {"ους": {"PROPN": "Ους"}})
    assert model._pos_table_lookup("Οὓς", "PROPN") is None


def test_conflicting_tag_under_the_capitalized_key_counts(fallback_model):
    # The Ancient POS table holds the sentence-initial spelling itself.
    model = _all_model(
        fallback_model, {"σέ": "σύ"}, {"Σὲ": {"PRON": "σύ"}},
        {"σε": {"ADP": "σε"}})
    assert model._pos_table_lookup("Σὲ", "ADP") == "σε"


def test_monotonic_key_gives_the_reading_s_tag_but_not_the_spelling(
        fallback_model):
    # κυρού records the tag of κυρόω; it is not the spelling κυροῦ.
    model = _all_model(
        fallback_model, {"κυροῦ": "κυρόω"}, {"κυρού": {"VERB": "κυρόω"}},
        {"κυρου": {"NOUN": "κυρός"}})
    assert model._pos_table_lookup("κυροῦ", "NOUN") == "κυρός"


def test_capitalized_grave_form_is_known_by_its_lowercase_acute_key(
        fallback_model):
    model = _all_model(
        fallback_model, {"ἐπεί": "ἐπεί"},
        {"ἐπεί": {"SCONJ": "ἐπεί"}, "επει": {"SCONJ": "ἐπεί"}},
        {"επει": {"VERB": "πείθω"}})
    assert model._pos_table_lookup("Ἐπεὶ", "VERB") is None


@pytest.mark.parametrize("word,lookup,ancient_pos,modern_pos,upos", [
    # The Ancient POS table spells the lemma with its capital.
    ("Ἑλλήνων", {"ελληνων": "Ἕλλην"}, {"ελληνων": {"NOUN": "Ἕλλην"}},
     {"ελληνων": {"PROPN": "Έλληνας"}}, "PROPN"),
    # The lookup has the enclitic τις, the POS table the accented τίς.
    ("τινὰ", {"τινα": "τις"}, {"τινα": {"PRON": "τίς"}},
     {"τινα": {"DET": "κάτι"}}, "DET"),
])
def test_recorded_tag_matches_the_lemma_by_its_letters(
        fallback_model, word, lookup, ancient_pos, modern_pos, upos):
    model = _all_model(fallback_model, lookup, ancient_pos, modern_pos)
    assert model._pos_table_lookup(word, upos) is None


@pytest.mark.parametrize("word,ancient,upos,modern", [
    ("στῆς", "ἵστημι", "DET", "ο"), ("ἄντε", "ἄντομαι", "INTJ", "άντε"),
    # A contract lemma: the ending is read without its accent.
    ("τούτῃ", "τουτῶ", "PRON", "τούτος"),
])
def test_closed_class_tag_does_not_keep_a_mi_or_middle_verb_lemma(
        fallback_model, word, ancient, upos, modern):
    model = _all_model(
        fallback_model, {word: ancient}, {},
        {core.strip_accents(word): {upos: modern}})
    assert model._pos_table_lookup(word, upos) == modern


def test_spelling_in_the_pos_table_is_known_without_a_lookup_entry(
        fallback_model):
    # The lookup reaches σέω only through the stripped key, but the
    # Ancient POS table holds σεῖς itself.
    model = _all_model(
        fallback_model, {"σεις": "σέω"}, {"σεῖς": {"NOUN": "σής"}},
        {"σεις": {"PRON": "εσείς"}})
    assert model._pos_table_lookup("σεῖς", "PRON") is None


def test_ancient_headword_check_normalizes_like_the_head_list(fallback_model):
    model = _all_model(fallback_model, {}, {}, {})
    model._ancient_lexical_heads = {"ἰσος", "ἀντιλέγω"}
    assert model._is_ancient_headword("ἰσοσ")       # word-final medial sigma
    assert model._is_ancient_headword("ἀντι-λέγω")  # hyphenated head
    assert not model._is_ancient_headword("ἰσον")


def test_lexical_head_list_is_read_from_the_file_s_ancient_heads(
        fallback_model, tmp_path, monkeypatch):
    path = tmp_path / "grc_lexical_heads.json"
    path.write_text(json.dumps(
        {"heads": {"ἀλλην": ["lsj10_headwords"]},
         "vernacular_heads": {"ἀλλιν": ["kriaras"]}}, ensure_ascii=False),
        encoding="utf-8")
    monkeypatch.setattr(core, "GRC_LEXICAL_HEADS_PATH", path)
    model = _all_model(
        fallback_model, {"ἀλλην": "ἀλλην", "ἀλλιν": "ἀλλιν"}, {},
        {"αλλην": {"PRON": "άλλος"}, "αλλιν": {"PRON": "άλλος"}})
    model._ancient_lexical_heads = None
    assert model._pos_table_lookup("ἀλλην", "PRON") is None
    # A vernacular head is no Ancient headword.
    assert model._pos_table_lookup("ἀλλιν", "PRON") == "άλλος"


def test_capitalized_headword_echo_is_kept(fallback_model):
    model = _all_model(
        fallback_model, {"Ἀλλην": "Ἀλλην"}, {}, {"αλλην": {"PRON": "άλλος"}},
        headwords={"Ἀλλην"})
    assert model._pos_table_lookup("Ἀλλην", "PRON") is None


def test_reading_only_in_the_combined_lookup_is_no_ancient_spelling(
        fallback_model):
    # In-memory tables. The shipped SQLite lookup also serves its Ancient
    # view from rows stored for both languages, so there a spelling held
    # only in such a row counts as known.
    modern = {"ώρες": {"NOUN": "ώρα"}}
    model = _all_model(fallback_model, {"ώρες": "ὄρνυμι"}, {}, modern)
    model._lookup = {"ώρες": "ὄρνυμι", "ὧρες": "ὥρα"}
    assert model._pos_table_lookup("ὧρες", "NOUN") == "ώρα"
    # No Ancient lookup at all: every reading is the combined lookup's.
    model = _all_model(fallback_model, {}, {}, modern)
    model._lookup = {"ώρες": "ὄρνυμι"}
    assert model._lookup_word("ὧρες") == "ὄρνυμι"
    assert model._pos_table_lookup("ὧρες", "NOUN") == "ώρα"


@pytest.mark.parametrize("key", ["Σὰν", "σὰν", "Σάν", "σάν"])
def test_ancient_spelling_is_matched_like_the_lookup_cascade(
        fallback_model, key):
    # _lookup_word reads the form, its lowercase, and both with the grave
    # turned acute; a reading under any of them is one of this spelling.
    model = _all_model(
        fallback_model, {key: "σύν"}, {}, {"σαν": {"CCONJ": "σαν"}})
    assert model._lookup_word("Σὰν") == "σύν"
    assert model._pos_table_lookup("Σὰν", "CCONJ") is None


def test_unusable_lookup_value_is_no_ancient_spelling(fallback_model):
    # _lookup_word passes over a grave-accented value no headword list
    # backs and reaches ὄρνυμι through ώρες.
    model = _all_model(
        fallback_model, {"ὧρες": "ὡρὲς", "ώρες": "ὄρνυμι"}, {},
        {"ώρες": {"NOUN": "ώρα"}})
    model._trusted_grave_citation_targets = lambda: set()
    assert model._lookup_word("ὧρες") == "ὄρνυμι"
    assert model._pos_table_lookup("ὧρες", "NOUN") == "ώρα"


def test_first_modern_entry_is_the_one_nearest_the_form_s_spelling(
        fallback_model):
    model = _all_model(
        fallback_model, {}, {},
        {"Ἀλλην": {"PRON": "άλλος"}, "ἀλλην": {"PRON": "αλλήνα"},
         "αλλην": {"PRON": "αλλού"}})
    assert model._pos_table_lookup("Ἀλλην", "PRON") == "άλλος"
    del model._pos_lookup["Ἀλλην"]
    assert model._pos_table_lookup("Ἀλλην", "PRON") == "αλλήνα"


def test_pos_tag_compatibility_is_exactly_the_documented_pairs():
    tags = ("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
            "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X")
    pairs = {frozenset(pair) for pair in [
        ("NOUN", "PROPN"), ("NOUN", "ADJ"), ("PROPN", "ADJ"),
        ("ADJ", "PRON"), ("ADJ", "DET"), ("ADJ", "NUM"), ("PRON", "DET"),
        ("PRON", "NUM"), ("DET", "NUM"), ("ADJ", "ADV"), ("ADV", "PART"),
        ("CCONJ", "SCONJ"), ("VERB", "AUX")]}
    for first in tags:
        for second in tags:
            expected = first == second or frozenset((first, second)) in pairs
            assert core.Dilemma._pos_tags_compatible(first, second) \
                is expected, (first, second)


@pytest.mark.parametrize("upos", ["PROPN", "NOUN", "ADJ", "X", "SYM"])
def test_open_or_unknown_tag_does_not_displace_a_known_spelling(
        fallback_model, upos):
    model = _all_model(
        fallback_model, {"νά": "νάω"}, {"νά": {"VERB": "νάω"}},
        {"να": {upos: "να"}})
    assert model._pos_table_lookup("νὰ", upos) is None


@pytest.mark.parametrize("word", ["ἁλλην", "ἀλλῃ", "ἀϊλλην"])
def test_breathing_subscript_and_diaeresis_are_no_accent(fallback_model, word):
    # Only an acute, grave or circumflex keeps a self-mapped word no
    # dictionary lists from being an unaccented echo.
    model = _all_model(
        fallback_model, {word: word}, {},
        {core.strip_accents(word): {"PRON": "άλλος"}})
    assert model._pos_table_lookup(word, "PRON") == "άλλος"


def test_unread_lowercase_word_takes_a_capitalized_modern_name(fallback_model):
    # With no Ancient reading at all, the Modern entry is the only one.
    model = _all_model(
        fallback_model, {}, {}, {"πατουλιας": {"PROPN": "Πατουλιάς"}})
    assert model._pos_table_lookup("πατουλιὰς", "PROPN") == "Πατουλιάς"


@pytest.mark.parametrize("word,lookup,modern_pos,upos", [
    # ἐφη is ἔφη without its accent, and έφη is found the same way.
    ("ἐφη", {"εφη": "φημί"}, {"εφη": {"VERB": "έφη"}}, "VERB"),
    # ἄερα for ἀέρα; αέρας is recorded for another accent.
    ("ἄερα", {"αερα": "ἀήρ"}, {"αερα": {"NOUN": "αέρας"}}, "NOUN"),
    # A second accent that no enclitic gives.
    ("τοῦτοῦ", {"τουτου": "οὗτος"}, {"τουτου": {"PRON": "τούτος"}}, "PRON"),
])
def test_modern_entry_under_the_bare_key_needs_the_form_s_accent(
        fallback_model, word, lookup, modern_pos, upos):
    # Both readings were reached by discarding the accents, so neither is
    # evidence for this spelling and the lookup candidates decide.
    model = _all_model(fallback_model, lookup, {}, modern_pos)
    assert model._pos_table_lookup(word, upos) is None


@pytest.mark.parametrize("word,ancient,upos,modern", [
    ("στοὺς", "ἵστημι", "DET", "ο"), ("μιὰ", "μιαίνω", "DET", "ένας"),
    ("ποιὸς", "ποιέω", "PRON", "ποιος"),
])
def test_monosyllable_is_recorded_under_its_bare_key(
        fallback_model, word, ancient, upos, modern):
    # Monotonic spelling writes no accent on a monosyllable.
    key = core.strip_accents(word)
    model = _all_model(fallback_model, {key: ancient}, {}, {key: {upos: modern}})
    assert model._pos_table_lookup(word, upos) == modern


@pytest.mark.parametrize("word,base", [
    ("ἄνθρωπός", "ἄνθρωπος"), ("δῶρόν", "δῶρον"), ("Πᾶνά", "Πᾶνα"),
    ("οἶκοί", "οἶκοι"), ("δίκαιός", "δίκαιος"), ("Βορῆά", "Βορῆα"),
    ("ἅλιά", "ἅλια"), ("ἑπόμενός", "ἑπόμενος"), ("εἴδωλόν", "εἴδωλον"),
    # υι is one syllable; case does not change the count.
    ("Εἰλείθυιά", "Εἰλείθυια"), ("ἌΝΘΡΩΠΌΣ", "ἌΝΘΡΩΠΟΣ"),
])
def test_enclitic_accent_is_dropped(word, base):
    import unicodedata
    assert core.Dilemma._without_enclitic_accent(word) == base
    assert core.Dilemma._without_enclitic_accent(
        unicodedata.normalize("NFD", word)) == base


@pytest.mark.parametrize("word", [
    "καλός", "ἄνθρωπος", "νὰ", "ΑΝΘΡΩΠΟΣ",
    # Accents no enclitic gives: on the penult and the final syllable, on
    # one diphthong, too far apart, a final grave or circumflex, a second
    # accent that is not final, a grave first.
    "λόγός", "ἄεί", "Κάἵν", "ἀνθρωποφάγός", "ἄνθρωποφαγός", "ἄνθρωπὸς",
    "τοῦτοῦ", "ἄέρα", "τὸνά", "δῶρατά",
    # A second acute at the right distance that is not on the final
    # syllable (οἶκόνδε), and a grave two syllables before a final acute.
    "οἶκόνδε", "Οὔλυμπόνδε", "οἷόσπερ", "κεῖμένου", "ἣδιστόν", "ἀθὰνατός",
])
def test_other_accents_are_left_alone(word):
    assert core.Dilemma._without_enclitic_accent(word) == word


def test_modern_entry_for_the_accent_must_carry_the_requested_tag(fallback_model):
    # άλογο is recorded for the form's accent only as an adjective; the
    # noun entry sits under the bare key.
    model = _all_model(
        fallback_model, {"αλογο": "ἄλογος"}, {},
        {"άλογο": {"ADJ": "άλογο"}, "αλογο": {"NOUN": "άλογο"}})
    assert model._pos_table_lookup("ἄλογο", "NOUN") is None
    assert model._pos_table_lookup("ἄλογο", "ADJ") == "άλογο"


def test_capitalized_form_is_judged_by_its_lowercase_spelling(fallback_model):
    # A capital vowel is still a syllable, so sentence-initial Ἐφη is no
    # monosyllable and stays with the lookup.
    model = _all_model(
        fallback_model, {"εφη": "φημί"}, {}, {"εφη": {"VERB": "έφη"}})
    assert model._pos_table_lookup("Ἐφη", "VERB") is None
    assert model._pos_table_lookup("ἐφη", "VERB") is None
    # The Modern entry for the accent is keyed in lowercase.
    model = _all_model(
        fallback_model, {"μηνες": "μαίνομαι"}, {}, {"μήνες": {"NOUN": "μήνας"}})
    assert model._pos_table_lookup("Μῆνες", "NOUN") == "μήνας"
    assert model._pos_table_lookup("μῆνες", "NOUN") == "μήνας"


def test_proper_noun_request_keeps_the_ancient_name_for_a_lowercase_lemma(
        fallback_model):
    # The Modern lemma is lowercase, so only the tag condition of the
    # name clause answers the PROPN request.
    model = _all_model(
        fallback_model, {"κακό": "Κακό"}, {},
        {"κακό": {"ADJ": "κακός", "PROPN": "κακός"}}, headwords={"Κακό"})
    assert model._pos_table_lookup("κακὸ", "ADJ") == "κακός"
    assert model._pos_table_lookup("κακὸ", "PROPN") is None


def test_enclitic_second_accent_is_judged_as_the_base_spelling(fallback_model):
    # Πᾶνά τε: the lookup holds Πᾶνα, not the form with both accents.
    model = _all_model(
        fallback_model, {"Πᾶνα": "Πάν"}, {}, {"πανα": {"PROPN": "Πάνας"}})
    assert model._pos_table_lookup("Πᾶνά", "PROPN") is None
    assert model._pos_table_lookup("Πᾶνα", "PROPN") is None
    # The Ancient POS table knows ἑπόμενος as a verb form.
    model = _all_model(
        fallback_model, {"ἑπόμενος": "ἕπομαι"},
        {"ἑπόμενος": {"VERB": "ἕπομαι"}},
        {"επομενος": {"ADJ": "επόμενος"}})
    assert model._pos_table_lookup("ἑπόμενός", "ADJ") is None
    assert model._pos_table_lookup("ἑπόμενος", "ADJ") is None


def test_enclitic_accent_on_a_vernacular_word_keeps_its_modern_reading(
        fallback_model):
    # τὸ ἄλογό του: ἄλογο is judged, and άλογο is recorded for its accent.
    model = _all_model(
        fallback_model, {"αλογο": "ἄλογος"}, {},
        {"άλογο": {"NOUN": "άλογο"}, "αλογο": {"NOUN": "άλογο"}})
    assert model._pos_table_lookup("ἄλογό", "NOUN") == "άλογο"
    assert model._pos_table_lookup("ἄλογο", "NOUN") == "άλογο"
    # An entry for that accent with another lemma is not this reading.
    model._pos_lookup["άλογο"] = {"NOUN": "αλόγα"}
    assert model._pos_table_lookup("ἄλογό", "NOUN") is None


def test_compatible_recorded_tag_keeps_an_unaccented_echo(fallback_model):
    # ἀει echoes itself, but the Ancient POS table records its letters as
    # the adverb ἀεί, which a particle request does not contradict.
    modern = {"αει": {"PART": "άγιος", "VERB": "άγω"}}
    model = _all_model(
        fallback_model, {"ἀει": "ἀει"}, {"αει": {"ADV": "ἀεί"}}, modern)
    assert model._pos_table_lookup("ἀει", "PART") is None
    # A conflicting tag leaves the echo clause in charge.
    assert model._pos_table_lookup("ἀει", "VERB") == "άγω"


def test_ancient_reading_does_not_depend_on_the_convention(fallback_model):
    # convention="triantafyllidis" makes the lookup prefer Modern Greek
    # entries; the rule still compares against the Ancient reading.
    model = _all_model(
        fallback_model, {"νά": "νή"}, {"νά": {"NOUN": "νή"}},
        {"να": {"CCONJ": "να"}})
    model._mg_lookup = {"να": "νά"}
    assert model._pos_table_lookup("νὰ", "CCONJ") == "να"
    model._convention_name = "triantafyllidis"
    assert model._lookup_word("νὰ") == "νά"
    assert model._lookup_word("νὰ", prefer_modern=False) == "νή"
    assert model._pos_table_lookup("νὰ", "CCONJ") == "να"


def test_lowercase_word_is_not_read_as_a_capitalized_ancient_name(
        fallback_model):
    modern = {"κακό": {"ADJ": "κακός", "NOUN": "κακό", "PROPN": "Κακός"}}
    model = _all_model(
        fallback_model, {"κακό": "Κακό"}, {}, modern, headwords={"Κακό"})
    assert model._pos_table_lookup("κακὸ", "ADJ") == "κακός"
    # A noun request may mean the name, and so does a capital.
    assert model._pos_table_lookup("κακὸ", "NOUN") is None
    assert model._pos_table_lookup("κακὸ", "PROPN") is None
    assert model._pos_table_lookup("Κακὸ", "ADJ") is None
    # A capitalized Modern lemma is no better than the Ancient name.
    model._pos_lookup["κακό"]["ADJ"] = "Κακός"
    assert model._pos_table_lookup("κακὸ", "ADJ") is None
    # The name's letters are the Modern word's: πιὰ is not Πῖα.
    model = _all_model(
        fallback_model, {"πια": "Πῖα"}, {}, {"πια": {"ADV": "πια"}})
    assert model._pos_table_lookup("πιὰ", "ADV") == "πια"
    # A compatible recorded part of speech keeps the name.
    model = _all_model(
        fallback_model, {"κακό": "Κακό"}, {"κακό": {"NOUN": "Κακό"}}, modern)
    assert model._pos_table_lookup("κακὸ", "ADJ") is None
    # A conflicting one is judged like any tag conflict.
    model = _all_model(
        fallback_model, {"κακό": "Κακό"}, {"κακό": {"VERB": "Κακό"}}, modern)
    assert model._pos_table_lookup("κακὸ", "ADJ") is None


def test_lexical_heads_are_parsed_once_and_split_by_language(
        fallback_model, tmp_path, monkeypatch):
    path = tmp_path / "grc_lexical_heads.json"
    path.write_text(json.dumps(
        {"heads": {"ἀλλην": ["lsj10_headwords"]},
         "vernacular_heads": {"ἀλλιν": ["kriaras"]}}, ensure_ascii=False),
        encoding="utf-8")
    monkeypatch.setattr(core, "GRC_LEXICAL_HEADS_PATH", path)
    core._lexical_head_lists.cache_clear()
    expected = {"grc": (True, False), "el": (False, True), "all": (True, True),
                "med": (False, False)}
    for lang, (ancient, vernacular) in expected.items():
        model = fallback_model({}, {})
        model.lang = lang
        assert model._positive_elision_target("ἀλλην") is ancient
        assert model._positive_elision_target("ἀλλιν") is vernacular
        # The Ancient headword check reads the Ancient heads in every mode.
        assert model._is_ancient_headword("ἀλλην")
        assert not model._is_ancient_headword("ἀλλιν")
    assert core._lexical_head_lists.cache_info().misses == 1
    core._lexical_head_lists.cache_clear()
    # Without the file there are no heads.
    monkeypatch.setattr(core, "GRC_LEXICAL_HEADS_PATH", tmp_path / "missing")
    model = fallback_model({}, {})
    assert not model._positive_elision_target("ἀλλην")
    assert not model._is_ancient_headword("ἀλλην")


@pytest.fixture(scope="module")
def shipped_all_model():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    return core.Dilemma(lang="all", resolve_articles=True)


_TRIANTAFYLLIDIS_MODEL = []


def _shipped_triantafyllidis_model():
    if not _TRIANTAFYLLIDIS_MODEL:
        _TRIANTAFYLLIDIS_MODEL.append(
            core.Dilemma(lang="all", convention="triantafyllidis"))
    return _TRIANTAFYLLIDIS_MODEL[0]


def test_shipped_all_keeps_ancient_headwords_for_polytonic_pos(shipped_all_model):
    forms = ["ἄλλην", "εἰμι", "ἀπαστράπτων", "ἐργαζομένους", "ὁποῖον",
             "Ἑλλήνων", "Ἀρκάδας"]
    tags = ["PRON", "VERB", "VERB", "VERB", "PRON", "PROPN", "PROPN"]
    assert shipped_all_model.lemmatize_batch_pos(forms, tags, guess=False) == [
        "ἄλλος", "εἰμί", "ἀπαστράπτω", "ἐργάζομαι", "ὁποῖος", "Ἕλλην", "Ἀρκάς"]


@pytest.mark.parametrize("form,lemma", [
    ("ὀλεθροτόκον", "ὀλεθροτόκος"), ("ἡδύφθογγον", "ἡδύφθογγος"),
    ("ἐξύφανας", "ἐξυφαίνω"), ("γλυκίου", "γλύκιος"),
    ("προδειχθεὶς", "προδείκνυμι"), ("νοοτρόφους", "νοοτρόφος"),
    ("ἀποσώσῃ", "ἀποσῴζω"), ("προλάμψαν", "προλάμπω"), ("πονεόντι", "πονέω"),
])
def test_shipped_compound_lemmas_are_accented(shipped_all_model, form, lemma):
    # Issue #101: the compound fallback returned accent-stripped skeletons.
    assert shipped_all_model.lemmatize(form, guess=False) == lemma
    assert all(c.lemma == lemma for c in
               shipped_all_model.lemmatize_verbose(form, guess=False)
               if c.source == "compound")


def test_monotonic_compound_keeps_its_base_accent_without_a_headword(
        fallback_model):
    model = fallback_model({"ακορεστα": "ακόρεστος"}, {})
    model.lang = "el"
    _headwords(model)
    assert model._decompose_compound("μονοακόρεστα") == "μονοακόρεστος"
    # Polytonic input has no such fallback: the prefix lacks its breathing.
    model.lang = "all"
    model._lookup = model._ag_lookup = {"τοκον": "τόκος"}
    assert model._decompose_compound("ὀλεθροτόκον") is None


@pytest.mark.parametrize("lang", ["all", "el"])
@pytest.mark.parametrize("word,guessed", [
    ("βράβευσον", "βραβευσός"), ("μονοακόρεστα", "μονοακόρεστος")])
def test_compound_no_dictionary_lists_is_a_guess(
        fallback_model, monkeypatch, lang, word, guessed):
    # Issue #103: joining a prefix to the accented base lemma is unverified
    # (βραβευ + σός is no word), so a no-guess call abstains on every path.
    monkeypatch.setattr(core.Dilemma, "_load_model", lambda self: None)
    monkeypatch.setattr(core.Dilemma, "_predict",
                        lambda self, words: list(words))
    model = fallback_model({"σον": "σός", "ακορεστα": "ακόρεστος"}, {})
    model.lang = lang
    _headwords(model)
    assert model._decompose_compound(word, guess=False) is None
    assert model._decompose_compound_all(word, guess=False) == []
    assert model.lemmatize(word, guess=False) is None
    assert model.lemmatize_batch([word], guess=False) == [None]
    assert model.lemmatize_verbose(word, guess=False) == []
    # The transformer left the word as it was; the reconstruction is the
    # guess that replaces the echo.
    assert model.lemmatize(word) == guessed
    assert model.lemmatize_batch([word]) == [guessed]
    assert [(c.lemma, c.source) for c in model.lemmatize_verbose(word)] == [
        (guessed, "compound")]


def test_batch_pos_no_guess_reads_keep_the_callers_compound_guess(
        fallback_model, monkeypatch):
    # lemmatize_batch_pos re-reads the structured candidates without the
    # model and adopts their spelling of the model's answer. With
    # guess=True that read still holds the reconstruction, as in 1.3.23.
    monkeypatch.setattr(core.Dilemma, "_load_model", lambda self: None)
    monkeypatch.setattr(core.Dilemma, "_predict",
                        lambda self, words: ["Μονοακόρεστος" for _ in words])
    model = fallback_model({"ακορεστα": "ακόρεστος"}, {})
    model.lang = "el"
    _headwords(model)
    word = "μονοακόρεστα"
    assert model._lemmatize_verbose_impl(word, guess=False) == []
    assert [(c.lemma, c.source) for c in model._lemmatize_verbose_impl(
        word, guess=False, compound_guess=True)] == [
            ("μονοακόρεστος", "compound")]
    assert model.lemmatize_pos(word, "ADJ") == "Μονοακόρεστος"
    assert model.lemmatize_batch_pos([word], ["ADJ"]) == ["μονοακόρεστος"]
    assert model.lemmatize_batch_pos([word], ["ADJ"], guess=False) == [None]


def test_compound_a_dictionary_lists_needs_no_guess(fallback_model):
    model = fallback_model({"ακορεστα": "ακόρεστος"}, {})
    model.lang = "el"
    _headwords(model, "μονοακόρεστος", source="mg")
    assert model._decompose_compound("μονοακόρεστα", guess=False) == (
        "μονοακόρεστος")
    assert model.lemmatize("μονοακόρεστα", guess=False) == "μονοακόρεστος"
    assert model.lemmatize_batch(["μονοακόρεστα"], guess=False) == [
        "μονοακόρεστος"]
    assert [(c.lemma, c.source) for c in model.lemmatize_verbose(
        "μονοακόρεστα", guess=False)] == [("μονοακόρεστος", "compound")]


@pytest.mark.parametrize("form,upos,invented", [
    ("βράβευσον", "VERB", "βραβευσός"),
    ("προσηύδασε", "VERB", "προσηυδασός"),
    ("τεθαύμακαν", "VERB", "τεθαυμακάω"),
    ("φρυκτωρουμένους", "VERB", "φρυκτωρουμένος"),
    ("μαθητεύσατα", "VERB", "μαθητευσάτον"),
    ("σελίδεσιν", "NOUN", "σελιδέσις"),
    ("θεόγραφα", "ADJ", "θεογραφεύς"),
])
def test_shipped_no_guess_call_invents_no_compound(
        shipped_all_model, form, upos, invented):
    # Issue #103: none of these reconstructions is in a dictionary.
    model = shipped_all_model
    assert model.lemmatize(form, guess=False) != invented
    assert model.lemmatize_batch([form], guess=False) != [invented]
    assert model.lemmatize_pos(form, upos, guess=False) != invented
    assert model.lemmatize_batch_pos([form], [upos], guess=False) != [invented]
    assert all(c.lemma != invented
               for c in model.lemmatize_verbose(form, guess=False))


@pytest.mark.parametrize("form,upos,lemma", [
    # Vernacular forms in polytonic spelling keep their Modern reading.
    ("νὰ", "CCONJ", "να"), ("κυροῦ", "NOUN", "κυρός"),
    ("ἀλλην", "PRON", "άλλος"), ("μὲ", "ADP", "με"), ("σὲ", "ADP", "σε"),
    ("σὰν", "CCONJ", "σαν"), ("σεῖς", "PRON", "εσείς"),
    ("στοὺς", "DET", "ο"), ("ὧρες", "NOUN", "ώρα"),
    # Ancient readings a Modern headword must not replace: the same word,
    # a compatible or unrecorded part of speech, names outside LSJ,
    # mistagged Ancient forms, and the article.
    ("ἄλλην", "PRON", "ἄλλος"), ("Ἑλλήνων", "PROPN", "Ἕλλην"),
    ("σὰς", "PRON", "σός"), ("πλουτῶ", "VERB", "πλουτέω"),
    ("δαμασκηνοῦ", "NOUN", "Δαμασκηνός"), ("Σολομὼν", "PROPN", "Σολομών"),
    ("Ἀμφιτρύων", "NOUN", "Ἀμφιτρύων"), ("ὄντων", "ADJ", "εἰμί"),
    ("ἀεὶ", "PART", "ἀεί"), ("ὑψηλὰ", "ADV", "ὑψηλός"),
    ("Οὓς", "PROPN", "ὅς"), ("ἔφη", "PROPN", "φημί"), ("οἱ", "PART", "ὁ"),
    # A lowercase word is not a capitalized Ancient name.
    ("κακὸ", "ADJ", "κακός"), ("πιὰ", "ADV", "πια"), ("μιλᾶς", "VERB", "μιλάω"),
    # Ancient forms with a missing or misplaced accent, or with the second
    # accent of a following enclitic, are not read through a bare key.
    ("ἐφη", "VERB", "φημί"), ("τοῦτοῦ", "PRON", "οὗτος"),
    ("ἄερα", "NOUN", "ἀήρ"), ("Πᾶνά", "PROPN", "Πάν"),
    ("αὔλακά", "NOUN", "αὖλαξ"), ("ἑπόμενός", "ADJ", "ἕπομαι"),
    # A verb reading found by discarding the accents is kept under a
    # closed-class tag; one the lookup holds for the spelling is not, also
    # when the lemma is a contract verb in -ῶ.
    ("ἀκούσον", "INTJ", "ἀκούω"), ("τούτῃ", "PRON", "τούτος"),
    # Known gaps (docs/usage.md): no Modern entry carries SCONJ, and a
    # contraction read as an Ancient form keeps that reading or gets none.
    ("νὰ", "SCONJ", "νή"), ("νὰ", "PART", "να"),
    ("στὸν", "DET", "ἀστός"), ("στὸ", "DET", None),
])
def test_shipped_polytonic_pos_readings(shipped_all_model, form, upos, lemma):
    assert shipped_all_model.lemmatize_pos(form, upos, guess=False) == lemma
    assert shipped_all_model.lemmatize_batch_pos(
        [form], [upos], guess=False) == [lemma]


@pytest.mark.parametrize("form,upos,lemma", [
    ("νὰ", "CCONJ", "νά"), ("πᾶνε", "VERB", "πηγαίνω"),
    ("καταγῆς", "ADV", "καταγής"), ("ἀλλην", "PRON", "άλλος"),
    # Known gap: the entry the rule lets stand is not returned when the
    # lookup offers one lemma from both its sides.
    ("σεῖς", "PRON", "σέω"), ("κυροῦ", "NOUN", "κυρώ"),
])
def test_shipped_polytonic_pos_readings_under_triantafyllidis(form, upos, lemma):
    # The convention's lookup prefers Modern Greek entries; the rule must
    # still see the Ancient reading it weighs the Modern one against.
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    model = _shipped_triantafyllidis_model()
    assert model.lemmatize_pos(form, upos, guess=False) == lemma
    assert model.lemmatize_batch_pos([form], [upos], guess=False) == [lemma]

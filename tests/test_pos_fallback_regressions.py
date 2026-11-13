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


def test_polytonic_form_does_not_take_a_modern_pos_reading(fallback_model):
    # Issue #100: with lang="all" the combined POS table matched the
    # monotonic variant of a polytonic form and returned a Modern Greek
    # headword (ἄλλην -> άλλος).
    model = fallback_model({}, {})
    model.lang = "all"
    model._pos_ag_lookup = {"ἀπαστράπτων": {"VERB": "ἀπαστράπτω"}}
    model._pos_lookup = {"άλλην": {"PRON": "άλλος"},
                         "απαστράπτων": {"VERB": "απαστράπτω"},
                         "ἀπαστράπτων": {"VERB": "ἀπαστράπτω"}}
    assert model._pos_table_lookup("ἄλλην", "PRON") is None
    assert model._pos_table_lookup("ἀπαστράπτων", "VERB") == "ἀπαστράπτω"
    # Monotonic input still reads the Modern Greek entries.
    assert model._pos_table_lookup("άλλην", "PRON") == "άλλος"


@pytest.fixture(scope="module")
def shipped_all_model():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    return core.Dilemma(lang="all", resolve_articles=True)


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

"""Ancient adverbs in -ως are cited under their adjective (issue #117).

καλῶς -> καλός and ὀρθῶς -> ὀρθός in the tagged and the untagged calls, under
every Ancient convention; lexicalized adverbs (ἴσως, ἰδίως) and adverbs with
no adjective base (οὕτως, πῶς, ὄντως) keep their own headword, and Modern
Greek (lang="el", convention="triantafyllidis") is unchanged.
"""
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import pytest

from dilemma import core

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

_spec = importlib.util.spec_from_file_location(
    "build_grc_adverb_heads", ROOT / "build" / "build_grc_adverb_heads.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


# --- generator rules, on in-memory fixtures ---------------------------------

@pytest.mark.parametrize("word, position", [
    ("καλῶς", 0), ("δικαίως", 1), ("ἔκπλεως", 2), ("ἀνάπλεως", 2),
    ("καλός", 0), ("δίκαιος", 2), ("ἀΐδιος", 2), ("ἄϋπνος", 2),
])
def test_accent_position_counts_diphthongs_and_diaeresis(word, position):
    assert gen.accent_from_end(word) == position


@pytest.mark.parametrize("word, monotonic", [
    ("ακριβώς", True), ("ειλικρινώς", True), ("ραγδαίως", True),
    ("εὐλόγως", False), ("ῥᾳδίως", False), ("ἴσως", False),
    ("αἰσχρῶς", False), ("κυρίως", False),
])
def test_monotonic_keys_are_recognized(word, monotonic):
    assert gen.is_monotonic_key(word) is monotonic


def _fixture_inputs():
    ag = {
        "καλῶς": {"ADJ": "καλός", "ADV": "καλῶς"},
        "ὅμως": {"ADJ": "ὁμός"},                  # accent: ὁμός gives ὁμῶς
        "ἴσως": {"ADJ": "ἴσος"},                  # reviewed keep
        "ἀνάπλεως": {"ADJ": "ἀνάπλεος"},          # Attic declension
        "ἀνίδρως": {"ADJ": "ἄνιδρος"},            # dental-stem noun
        "δικαίως": {"ADJ": "δίκαιος"},
    }
    treebank = {
        "ἀεικελίως": {"ADV": "ἀεικής"},           # stem does not derive
        "κακῶς": {"ADV": "κακός", "PROPN": "Κακῶς"},
        "σῶς": {"ADJ": "σῶς"},
        "σαφῶς": {"ADV": "σαφής"},
        "ταχέως": {"ADV": "ταχύς"},
        "ακριβώς": {"ADV": "ακριβής"},            # monotonic key
        "ἀληθῶς": {"ADV": "ἀληθῶς", "ADJ": "ἀληθής"},
    }
    tables = {name: {} for name in gen.POS_TABLES}
    tables["ag_pos_lookup.json"] = ag
    tables["treebank_pos_lookup.json"] = treebank
    tables["glaux_pos_lookup.json"] = {"x": {"ADJ": "ἀεικής"},
                                       "y": {"ADJ": "ἀκριβής"}}
    headwords = {"καλός", "ὁμός", "ἴσος", "ἀνάπλεος", "ἄνιδρος", "δίκαιος",
                 "ἀεικής", "κακός", "σαφής", "ταχύς", "ἀκριβής", "ἀληθής",
                 "σῶς", "ὑποτυπωτικός", "σοφός"}
    tables["lsj9_pos_lookup.json"] = {"σαφής": {"ADJ": "σαφής"},
                                      "ταχύς": {"ADJ": "ταχύς"},
                                      "κακός": {"ADJ": "κακός"}}
    forms = {"ἀνίδρως": ["ἀνίδρωτα", "ἀνίδρως"],
             "ὑποτυπωτικῶς": ["ὑποτυπωτικῶς"],
             "καλῶς": ["καλῶς", "καλλιόνως"]}
    return tables, headwords, forms, {"ἴσως"}


def test_generator_rules_on_a_fixture():
    tables, headwords, forms, keep = _fixture_inputs()
    heads = gen.build_heads(tables, headwords, forms, keep)["heads"]
    assert heads == {
        "δικαίως": "δίκαιος",
        "κακῶς": "κακός",          # a PROPN self-cell does not block it
        "καλῶς": "καλός",
        "σαφῶς": "σαφής",
        "ταχέως": "ταχύς",
        "ἀληθῶς": "ἀληθής",        # treebank ADV self-cell, ADJ cell used
        "ὑποτυπωτικῶς": "ὑποτυπωτικός",   # lookup headword fallback
    }
    for left_out in ("ὅμως", "ἴσως", "ἀνάπλεως", "ἀνίδρως", "ἀεικελίως",
                     "σῶς", "ακριβώς"):
        assert left_out not in heads


def test_dental_check_needs_the_noun_forms():
    tables, headwords, forms, keep = _fixture_inputs()
    del forms["ἀνίδρως"]
    assert gen.build_heads(tables, headwords, forms, keep)["heads"]["ἀνίδρως"] == "ἄνιδρος"


def test_generator_output_is_byte_stable():
    tables, headwords, forms, keep = _fixture_inputs()
    first = json.dumps(gen.build_heads(tables, headwords, forms, keep),
                       ensure_ascii=False)
    shuffled = {}
    for name, table in tables.items():
        items = list(table.items())
        random.Random(117).shuffle(items)
        shuffled[name] = dict(items)
    second = json.dumps(gen.build_heads(shuffled, set(headwords), dict(
        reversed(list(forms.items()))), set(keep)), ensure_ascii=False)
    assert first == second


def test_review_entries_need_a_reason(tmp_path):
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"entries": [
        {"adverb": "ἴσως", "decision": "keep", "reason": " "}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        gen.read_review(path)


# --- the committed review and generated file --------------------------------

def test_review_file_gives_every_entry_a_reason():
    review = json.loads((DATA / gen.REVIEW_NAME).read_text(encoding="utf-8"))
    adverbs = [entry["adverb"] for entry in review["entries"]]
    assert len(adverbs) == len(set(adverbs))
    assert {"ἴσως", "ἰδίως"} <= set(adverbs)    # pinned by the gate fixtures
    for entry in review["entries"]:
        assert entry["decision"] == "keep"
        assert len(entry["reason"].split()) >= 5, entry


def test_generated_file_matches_its_review_and_rules():
    review_path = DATA / gen.REVIEW_NAME
    record = json.loads((DATA / gen.OUTPUT_NAME).read_text(encoding="utf-8"))
    assert record["rule"] == gen.RULE
    assert record["inputs"][gen.REVIEW_NAME] == hashlib.sha256(
        review_path.read_bytes()).hexdigest(), "rerun build/build_grc_adverb_heads.py"
    assert set(record["inputs"]) == {*gen.POS_TABLES, *gen.HEADWORD_LISTS,
                                     "lookup.db", gen.REVIEW_NAME}
    assert record["keep"] == sorted(gen.read_review(review_path))
    heads = record["heads"]
    assert not set(record["keep"]) & set(heads)
    for adverb, adjective in heads.items():
        assert adverb.endswith(("ως", "ῶς")) and not gen.is_monotonic_key(adverb)
        assert gen.accent_from_end(adverb) < 2
        assert not adjective.endswith(("ως", "ῶς")), adverb
    for word, adjective in {"καλῶς": "καλός", "ὀρθῶς": "ὀρθός", "ταχέως": "ταχύς",
                            "σαφῶς": "σαφής", "ὑποτυπωτικῶς": "ὑποτυπωτικός"}.items():
        assert heads[word] == adjective
    for word in ("οὕτως", "πῶς", "ὅμως", "ὄντως", "πάντως", "εἰκότως", "σῶς",
                 "ἅλως", "ἔκπλεως", "καταγέλως", "ἔρως"):
        assert word not in heads


# --- runtime, on small tables ---------------------------------------------

@pytest.fixture
def adverb_model(tmp_path, monkeypatch):
    monkeypatch.setattr(core.Dilemma, "_load_lookups", lambda *a, **k: None)
    monkeypatch.setattr(core.Dilemma, "_check_backend", lambda *a: None)
    heads = tmp_path / "grc_adverb_heads.json"
    heads.write_text(json.dumps({
        "heads": {"καλῶς": "καλός", "ἀδίκως": "ἄδικος", "κυρίως": "κύριος"},
        "keep": ["ἰδίως"]}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(core, "GRC_ADVERB_HEADS_PATH", heads)
    equivalences = tmp_path / "lemma_equivalences.json"
    equivalences.write_text(json.dumps({"groups": [["ἀδίκως", "ἄδικος"]]},
                                       ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(core, "LEMMA_EQUIVALENCES_PATH", equivalences)
    monkeypatch.setattr(core, "CONVENTION_DIR", tmp_path)
    cunliffe = tmp_path / "cunliffe_headwords.json"
    cunliffe.write_text("[]", encoding="utf-8")
    monkeypatch.setitem(core._CONVENTION_HEADWORDS, "cunliffe", (cunliffe,))

    def make(lang="grc", convention=None):
        lookup = {"καλῶς": "καλῶς", "ἰδίως": "ἰδίως", "κυρίως": "κυρίως",
                  "ἀδίκως": "ἀδίκως", "ἄδικον": "ἄδικος"}
        model = core.Dilemma(lang=lang, convention=convention, skip_pos=True)
        model._lookup = lookup
        model._ag_lookup = lookup
        model._pos_ag_lookup = {"ἰδίως": {"ADV": "ἴδιος"},
                                "καλῶς": {"ADV": "καλός"}}
        model._pos_lookup = dict(model._pos_ag_lookup)
        return model

    return make


@pytest.mark.parametrize("lang, convention", [
    ("grc", None), ("all", None), ("grc", "lsj"), ("all", "wiktionary")])
def test_runtime_cites_the_adjective(adverb_model, lang, convention):
    model = adverb_model(lang, convention)
    assert model.lemmatize("καλῶς", guess=False) == "καλός"
    assert model.lemmatize_batch(["καλῶς", "ἰδίως"], guess=False) == ["καλός", "ἰδίως"]
    status = model.citation_status("καλῶς")
    assert (status["normalized"], status["reason"], status["normalized_from"]) \
        == ("καλός", "convention", "καλῶς")


def test_runtime_keeps_a_reviewed_adverb_under_adv(adverb_model):
    model = adverb_model()
    assert model._pos_table_lookup("ἰδίως", "ADV") == "ἰδίως"
    assert model._pos_table_lookup("Ἰδίως", "ADV") == "ἰδίως"
    assert model._pos_table_lookup("καλῶς", "ADV") == "καλός"


def test_modern_greek_keeps_its_adverbs(adverb_model):
    model = adverb_model("el")
    assert model.lemmatize("κυρίως", guess=False) == "κυρίως"
    assert model._grc_adverb_head("κυρίως") is None


def test_equivalence_group_never_resolves_to_the_adverb(adverb_model):
    model = adverb_model("grc", "cunliffe")
    assert model._convention_map.get("ἄδικος", "ἄδικος") == "ἄδικος"
    assert model.lemmatize("ἀδίκως", guess=False) == "ἄδικος"
    assert model.lemmatize("ἄδικον", guess=False) == "ἄδικος"


def test_missing_file_changes_nothing(adverb_model, tmp_path, monkeypatch):
    monkeypatch.setattr(core, "GRC_ADVERB_HEADS_PATH", tmp_path / "absent.json")
    model = adverb_model()
    assert model.lemmatize("καλῶς", guess=False) == "καλῶς"


# --- the shipped data -----------------------------------------------------

ISSUE_ADVERBS = {
    "καλῶς": "καλός", "κακῶς": "κακός", "δικαίως": "δίκαιος", "σαφῶς": "σαφής",
    "ταχέως": "ταχύς", "σοφῶς": "σοφός", "ἡδέως": "ἡδύς", "ὀρθῶς": "ὀρθός",
    "ἀληθῶς": "ἀληθής", "συντόμως": "σύντομος", "προσφάτως": "πρόσφατος",
    "λυγρῶς": "λυγρός", "ὑποτυπωτικῶς": "ὑποτυπωτικός",
}
SELF_ADVERBS = ("ἴσως", "ἰδίως", "οὕτως", "πῶς", "πάντως", "ὄντως", "ὅμως")
UNTOUCHED = {"καλὼς": "καλός", "Καλῶς": "καλός", "σῶς": "σῶς", "ἔκπλεως": "ἔκπλεως", "σκέψεως": "σκέψις",
             "καταγέλωτ᾽": "καταγέλως"}


@pytest.fixture(scope="module")
def shipped():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    return {lang: core.Dilemma(lang=lang, resolve_articles=True)
            for lang in ("all", "grc")}


@pytest.mark.parametrize("lang", ["all", "grc"])
def test_shipped_adverbs_agree_tagged_and_untagged(shipped, lang):
    model = shipped[lang]
    expected = {**ISSUE_ADVERBS, **{w: w for w in SELF_ADVERBS}, **UNTOUCHED}
    words = list(expected)
    wanted = [expected[w] for w in words]
    assert model.lemmatize_batch(words, guess=False) == wanted
    assert model.lemmatize_batch_pos(words, ["ADV"] * len(words),
                                     guess=False)[:len(ISSUE_ADVERBS) + len(SELF_ADVERBS)] \
        == wanted[:len(ISSUE_ADVERBS) + len(SELF_ADVERBS)]
    for word in list(ISSUE_ADVERBS) + list(SELF_ADVERBS):
        assert model.lemmatize(word, guess=False) == expected[word]
        assert model.lemmatize_pos(word, "ADV", guess=False) == expected[word]


def test_shipped_lsj_convention_cites_the_adjective():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    model = core.Dilemma(lang="grc", convention="lsj", resolve_articles=True)
    words = ["δεινῶς", "κακῶς", "ὀρθῶς", "αἰνῶς"]
    wanted = ["δεινός", "κακός", "ὀρθός", "αἰνός"]
    assert model.lemmatize_batch(words, guess=False) == wanted
    assert model.lemmatize_batch_pos(words, ["ADV"] * 4, guess=False) == wanted


def test_shipped_modern_greek_keeps_its_adverbs():
    if not core.LOOKUP_DB_PATH.exists():
        pytest.skip("Shipped Dilemma data is not installed")
    model = core.Dilemma(lang="el", convention="triantafyllidis", skip_pos=True)
    assert [model.lemmatize(w, guess=False) for w in ("κυρίως", "διεθνώς")] \
        == ["κυρίως", "διεθνώς"]

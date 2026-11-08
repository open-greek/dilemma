"""Accent restoration with the monotonic word list and next-word model
(build/el_accent.py, eval/eval_el_accent_restoration.py).

Every input is written by the test: the word list and model come from the
fixture of ``tests/test_el_ngram.py``. Nothing is downloaded.

Run with:
    python -m pytest tests/test_el_accent_restoration.py -x -v
"""

import math
import random
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

import eval_el_accent_restoration as E  # noqa: E402
from dilemma_build import el_accent as A  # noqa: E402
from dilemma_build.gnlm_reader import GNLMReader, normalize  # noqa: E402


# --- marks ---------------------------------------------------------------------

@pytest.mark.parametrize("word, bare", [
    ("Όταν", "Οταν"), ("προϊόν", "προιον"), ("ταΐζω", "ταιζω"),
    ("καλημέρα", "καλημερα"), ("και", "και"), ("είν’", "ειν’"),
])
def test_stripping_removes_tonos_and_dialytika_and_keeps_case(word, bare):
    assert A.strip_marks(word) == bare


@pytest.mark.parametrize("typed, spelling, out", [
    ("Οταν", "όταν", "Όταν"), ("προιον", "προϊόν", "προϊόν"),
    ("ταιζω", "ταΐζω", "ταΐζω"), ("αθηνα", "Αθήνα", "αθήνα"),
    ("ποτε", "ποτέ", "ποτέ"),
])
def test_marks_move_onto_the_typed_letters(typed, spelling, out):
    assert A.transfer_marks(typed, spelling) == out


def test_marks_need_letters_of_the_same_length():
    with pytest.raises(ValueError):
        A.transfer_marks("ποτε", "πότες")


# --- candidates ----------------------------------------------------------------

@pytest.fixture
def index():
    return A.AccentIndex([
        ("πότε", "C"), ("ποτέ", "C"), ("που", "C"), ("πού", "C"),
        ("η", "C"), ("ή", "C"), ("καλημέρα", "C"), ("Νίκη", "M"),
        ("νίκη", "C"), ("Άρτα", "M"), ("αρτά", "X"), ("Μαρία", "M"),
        ("όνομα", "C"), ("και", "C"),
    ])


def test_candidates_are_grouped_by_lowercase_spelling(index):
    opts = index.options("Νικη")
    assert [o.lower for o in opts] == ["νίκη"]
    assert opts[0].spellings == ("Νίκη", "νίκη") and opts[0].rank == 3


def test_a_lowercase_word_never_takes_a_names_accents(index):
    assert [o.lower for o in index.options("αρτα")] == ["αρτά"]
    assert sorted(o.lower for o in index.options("Αρτα")) == ["άρτα", "αρτά"]
    # With no lowercase entry the name is the only candidate.
    assert [o.lower for o in index.options("μαρια")] == ["μαρία"]


@pytest.mark.parametrize("rating", ["R", "X"])
def test_lowercase_slips_do_not_take_the_accents_of_a_rare_name(rating):
    index = A.AccentIndex([("Μάι", rating)])
    assert index.options("μαι") == ()
    assert A.decide("μαι", index.options("μαι"), None, margin=1).output == "μαι"
    # A deliberately capitalized name remains available, even when rare.
    assert [o.lower for o in index.options("Μαι")] == ["μάι"]
    assert A.decide("Μαι", index.options("Μαι"), None, margin=1).output == "Μάι"
    restorer = E.Restorer(index, None, margin=1, policy="keep")
    assert restorer.decide("μαι", [], 0).kind == "none"


@pytest.mark.parametrize("rating", ["C", "M"])
def test_common_names_can_restore_lowercase_input(rating):
    index = A.AccentIndex([("Μαρία", rating)])
    assert A.decide("μαρια", index.options("μαρια"), None, margin=1).output == "μαρία"


def test_rare_lowercase_words_keep_their_accents_and_exclude_common_names():
    index = A.AccentIndex([("αγωνιά", "R"), ("Αγωνία", "C")])
    assert [o.lower for o in index.options("αγωνια")] == ["αγωνιά"]


def fixed(scores):
    return lambda option: scores.get(option.lower, -math.inf)


def test_no_candidate_and_one_candidate(index):
    assert A.decide("ξυλο", index.options("ξυλο"), None, margin=1).kind == "none"
    d = A.decide("καλημερα", index.options("καλημερα"), None, margin=1)
    assert (d.output, d.kind) == ("καλημέρα", "unique")
    d = A.decide("και", index.options("και"), None, margin=1)
    assert (d.output, d.kind) == ("και", "same")


def test_several_candidates_need_a_clear_lead(index):
    opts = index.options("ποτε")
    d = A.decide("ποτε", opts, fixed({"πότε": -2.0, "ποτέ": -5.0}), margin=1)
    assert (d.output, d.kind) == ("πότε", "model") and d.gap == pytest.approx(3)
    d = A.decide("ποτε", opts, fixed({"πότε": -2.0, "ποτέ": -2.5}), margin=1)
    assert (d.output, d.kind) == ("ποτε", "kept")
    d = A.decide("ποτε", opts, fixed({"πότε": -2.0, "ποτέ": -2.5}), margin=1,
                 policy="top")
    assert (d.output, d.kind) == ("πότε", "top")
    # Neither known: no lead at all.
    d = A.decide("ποτε", opts, fixed({}), margin=0)
    assert d.kind == "kept"
    # Only one known: an infinite lead.
    d = A.decide("ποτε", opts, fixed({"ποτέ": -9.0}), margin=5)
    assert (d.output, d.gap) == ("ποτέ", math.inf)


def test_a_typed_word_is_replaced_only_past_the_word_margin(index):
    opts = index.options("που")
    scores = fixed({"πού": -1.0, "που": -3.0})
    assert A.decide("που", opts, scores, margin=1).output == "πού"
    d = A.decide("που", opts, scores, margin=1, word_margin=3)
    assert (d.output, d.kind) == ("που", "kept")
    # Policy top never replaces a typed word below the margin.
    d = A.decide("που", opts, scores, margin=1, word_margin=math.inf, policy="top")
    assert (d.output, d.kind) == ("που", "kept")
    # A sentence-initial capital keeps its case.
    assert A.decide("Η", index.options("Η"), fixed({"ή": 0.0, "η": -4.0}),
                    margin=1).output == "Ή"


# --- enclitic second accent ------------------------------------------------------

@pytest.mark.parametrize("word, out", [
    ("όνομα", "όνομά"), ("άνθρωπος", "άνθρωπός"), ("Άκουσε", "Άκουσέ"),
    ("πρόεδρος", "πρόεδρός"), ("άνθρωποι", "άνθρωποί"), ("διάβασε", "διάβασέ"),
    ("αυτοκίνητο", "αυτοκίνητό"), ("σπίτια", None), ("δωμάτιο", None),
    ("καλή", None), ("σπίτι", None), ("όνομά", None), ("είν’", None),
])
def test_the_enclitic_accent_goes_on_the_last_syllable_of_proparoxytones(word, out):
    assert A.enclitic_accent(word) == out


@pytest.mark.parametrize("word, call", [
    ("δωμάτιο", True), ("σπίτια", True), ("οικογένεια", True),
    ("άνθρωπος", False), ("όνομα", False), ("καλή", False),
])
def test_words_whose_glide_decides_are_recognized(word, call):
    assert A.needs_synizesis_call(word) is call


def test_article_and_preposition_forms_take_the_accent_only_at_the_end():
    assert A.takes_enclitic_accent("μου", "ήρθε")
    assert A.takes_enclitic_accent("σας", None)
    assert not A.takes_enclitic_accent("του", "σχολείου")
    assert A.takes_enclitic_accent("του", None)
    assert not A.takes_enclitic_accent("με", "τον")
    assert A.takes_enclitic_accent("το", None)
    assert not A.takes_enclitic_accent("και", None)


@pytest.mark.parametrize("host, twin", [
    ("όνομα", "όνομά"), ("Άνθρωπος", "Άνθρωπός"), ("δωμάτιο", "δωμάτιό"),
    ("σπίτια", "σπίτιά"), ("σπίτι", None), ("καλή", None), ("όνομά", None),
    ("ΟΝΟΜΑ", None), ("είν’", None), ("και", None),
])
def test_the_keyboard_twin_of_a_host(host, twin):
    assert A.enclitic_twin(host) == twin


class FakeLM:
    """``knows`` and ``score`` of a reader, from a table of
    ``(prev, curr, word) -> score``; every other trigram scores ``default``."""

    def __init__(self, vocab, scores, default=-10.0):
        self.vocab, self.scores, self.default = set(vocab), scores, default

    def knows(self, word):
        return word in self.vocab

    def score(self, word, prev, curr):
        return self.scores.get((prev, curr, word), self.default)


VOCAB = {"η", "απόφαση", "απόφασή", "του", "να", "υπουργού", "όνομα", "όνομά",
         "μου", "δωμάτιο", "δωμάτιό"}


def test_enclitic_evidence_compares_the_two_readings():
    ev = A.enclitic_evidence
    # the enclitic after the twin is far likelier than after the host
    lm = FakeLM(VOCAB, {("η", "απόφασή", "του"): -2.0})
    assert ev(lm, "απόφαση", "απόφασή", "του", None, "<s>", "η") is True
    # and the other way round
    lm = FakeLM(VOCAB, {("η", "απόφαση", "του"): -2.0})
    assert ev(lm, "απόφαση", "απόφασή", "του", None, "<s>", "η") is False
    # a difference under the margin says nothing
    lm = FakeLM(VOCAB, {("η", "απόφασή", "του"): -9.5})
    assert ev(lm, "απόφαση", "απόφασή", "του", None, "<s>", "η") is None
    # a form the model has never seen says nothing either
    lm = FakeLM(VOCAB - {"απόφασή"}, {("η", "απόφασή", "του"): -2.0})
    assert ev(lm, "απόφαση", "απόφασή", "του", None, "<s>", "η") is None
    # the next word settles a tie when both readings score it
    lm = FakeLM(VOCAB, {("απόφασή", "του", "να"): -1.0})
    assert ev(lm, "απόφαση", "απόφασή", "του", "να", "<s>", "η") is True
    assert ev(lm, "απόφαση", "απόφασή", "του", None, "<s>", "η") is None
    # and is left out when one reading cannot score it
    lm = FakeLM(VOCAB, {("απόφασή", "του", "να"): -1.0,
                        ("απόφαση", "του", "να"): -math.inf})
    assert ev(lm, "απόφαση", "απόφασή", "του", "να", "<s>", "η") is None
    # a reading the model cannot score loses to one it can
    lm = FakeLM(VOCAB, {("η", "απόφαση", "του"): -math.inf})
    assert ev(lm, "απόφαση", "απόφασή", "του", None, "<s>", "η") is True


def test_the_keyboard_decision_without_evidence():
    decide = A.keyboard_enclitic_decision
    # a pronoun: the syllable rule, which leaves out synizesis words
    assert decide(None, "όνομα", "μου", "είναι", "<s>", "το") is True
    assert decide(None, "δωμάτιο", "μου", None, "<s>", "το") is False
    # an article homograph: the rule at the end of the sentence, and before
    # a word only after a word that marks the enclitic
    assert decide(None, "απόφαση", "του", None, "<s>", "η") is True
    assert decide(None, "απόφαση", "του", "υπουργού", "<s>", "η") is False
    assert decide(None, "απόφαση", "του", "να", "<s>", "η") is True
    # no twin, or no enclitic of the keyboard
    assert decide(None, "σπίτι", "μου", None, "<s>", "το") is False
    assert decide(None, "όνομα", "με", None, "<s>", "το") is False


def test_the_keyboard_decision_follows_the_model():
    decide = A.keyboard_enclitic_decision
    # evidence with the next word overrides the syllable rule for a pronoun
    lm = FakeLM(VOCAB, {("όνομα", "μου", "είναι"): -1.0})
    assert decide(lm, "όνομα", "μου", "είναι", "<s>", "το") is False
    # with no say from the next word, the evidence without it decides
    lm = FakeLM(VOCAB, {("το", "δωμάτιό", "μου"): -1.0})
    assert decide(lm, "δωμάτιο", "μου", "είναι", "<s>", "το") is True
    # an article homograph before a word that is no possessive signal
    lm = FakeLM(VOCAB, {("απόφασή", "του", "υπουργού"): -1.0})
    assert decide(lm, "απόφαση", "του", "υπουργού", "<s>", "η") is True
    lm = FakeLM(VOCAB, {("απόφαση", "του", "να"): -1.0})
    assert decide(lm, "απόφαση", "του", "να", "<s>", "η") is False


# --- typing slips -----------------------------------------------------------------

def test_slips_respect_digraphs_and_final_sigma():
    rng = random.Random(1)
    seen = set()
    for _ in range(400):
        slipped, kind = E.make_slip("ευχαριστουμε", rng)
        seen.add(kind)
        assert slipped != "ευχαριστουμε" and "ς" not in slipped[:-1]
    assert {"i_spelling", "e_ai", "neighbor_key"} <= seen
    for _ in range(100):
        slipped, _ = E.make_slip("θαλασσας", rng)
        assert slipped.endswith("ς") or not slipped.endswith("σ")
    assert E.make_slip("σ’", rng) is None


def test_a_slip_keeps_a_capital_initial():
    rng = random.Random(3)
    for _ in range(50):
        slipped, _ = E.make_slip("Καλημερα", rng)
        assert slipped[0].isupper()


# --- end to end on the fixture model ----------------------------------------------

@pytest.fixture(scope="module")
def model(tmp_path_factory):
    import test_el_ngram as T
    root = tmp_path_factory.mktemp("el_accent")
    corpus, dic = T.make_corpus(root), T.make_dictionary(root)
    in_dir = T.train(root, "lm_el", "--vocab-size", "20", corpus=corpus, dic=dic)
    proc, out, _ = T.export(root, in_dir, "el_ngram")
    assert proc.returncode == 0, proc.stderr
    entries = E.load_entries(dic)
    return {"lm": GNLMReader(out), "entries": entries, "dic": dic,
            "index": A.AccentIndex(entries), "spellings": {s for s, _ in entries}}


def test_restoration_uses_left_context(model):
    r = E.Restorer(model["index"], model["lm"], margin=0.5, policy="keep")
    toks = ["Δεν", "ξέρω", "πότε", "θα", "έρθει"]
    typed = [A.strip_marks(t) for t in toks]
    out, decisions, _ = r.restore(typed, toks)
    assert out == toks
    assert [d.kind for d in decisions] == ["same", "unique", "model", "same", "unique"]


def test_the_enclitic_accent_is_added_after_restoration(model):
    r = E.Restorer(model["index"], model["lm"], margin=0.5, policy="keep")
    toks = ["Το", "όνομά", "μου", "είναι", "Νίκη"]
    out, _, added = r.restore([A.strip_marks(t) for t in toks], toks)
    assert out == toks and added == {1}
    r = E.Restorer(model["index"], model["lm"], margin=0.5, policy="keep",
                   enclitics=False)
    out, _, added = r.restore([A.strip_marks(t) for t in toks], toks)
    assert out[1] == "όνομα" and not added


def test_damage_counts_only_listed_words_typed_right(model):
    st = E.AccentStats()
    toks = ["που", "απο"]
    typed = toks[:]
    decisions = [A.Decision("πού", "model", ("που", "πού"), 2.0),
                 A.Decision("από", "unique", ("από",))]
    st.add(toks, typed, ["πού", "από"], decisions, set(), {"που", "πού", "από"})
    s = st.summary()
    assert s["damage"] == 1 and s["unlisted_changed"] == 1
    assert s["damage_rate"] == 1.0 and s["accuracy"] == 0.0


def test_fast_completions_match_the_reader(model):
    lm = model["lm"]
    comp = E.Completer(lm)
    cases = [(None, "<s>", "θ"), ("<s>", "θα", "π"), ("δεν", "ξέρω", "π"),
             (None, "δεν", "ξ"), ("η", "βουλή", "ψ"), (None, "<s>", "α")]
    assert comp.check(cases) == 0
    assert comp.complete(None, lm.context_id("<s>"), normalize("ζζ"), 3) == []


def test_keystroke_savings_count_taps_and_letters(model):
    from dilemma_build.el_ngram import Resolver, load_dictionary
    lm = model["lm"]
    r = E.Restorer(model["index"], lm, margin=0.5, policy="keep")
    entries = load_dictionary(model["dic"])
    dev = {"news": [["Δεν", "ξέρω", "πότε", "θα", "έρθει"]]}
    ks = E.run_keystrokes(r, E.Completer(lm), Resolver(entries), entries, dev)
    s = ks["pooled"]["all"]
    assert s["words"] == 5
    assert s["baseline_keystrokes"] == sum(len(t) + 1 for t in dev["news"][0])
    assert 0 < s["savings_three_completions"] < 1
    assert s["savings_three_completions"] >= s["savings_predictions_only"]
    assert s["savings_three_completions"] >= s["savings_commit_aware"]
    assert s["savings_three_completions"] >= s["savings_completions_only"]
    assert ks["completion_check"]["mismatches"] == 0


def test_completions_only_counts_words_a_prediction_had(model):
    from dilemma_build.el_ngram import Resolver, load_dictionary
    lm = model["lm"]
    r = E.Restorer(model["index"], lm, margin=0.5, policy="keep")
    entries = load_dictionary(model["dic"])
    # "θα" is the model's first prediction at the start of a sentence
    # in the fixture, and its first completion after one letter.
    dev = {"news": [["θα"]]}
    s = E.run_keystrokes(r, E.Completer(lm), Resolver(entries), entries, dev)["pooled"]["all"]
    assert s["prediction_hit_share"] == 1.0
    assert s["savings_predictions_only"] == pytest.approx(2 / 3)
    assert s["savings_completions_only"] == pytest.approx(1 / 3)


def test_the_next_word_widens_a_correct_lead(model):
    toks = ["Δεν", "ξέρω", "που", "είναι"]
    typed = [A.strip_marks(t) for t in toks]
    gaps = {}
    for look in (False, True):
        r = E.Restorer(model["index"], model["lm"], margin=0.5, policy="keep",
                       lookahead=look)
        out, decisions, _ = r.restore(typed, toks)
        assert out == toks
        gaps[look] = decisions[2].gap
    assert gaps[True] > gaps[False] > 0


# --- the enclitic accent against corpus spellings ---------------------------------

def test_enclitic_pairs_read_the_corpus_spelling():
    pairs = list(E.enclitic_pairs(
        ["Το", "όνομά", "μου", "είναι", "ο", "δάσκαλος", "του", "σχολείου",
         "και", "ΟΝΟΜΑ", "μου", "σπίτι", "μου"]))
    assert pairs == [("όνομα", "μου", True, "είναι"),
                     ("δάσκαλος", "του", False, "σχολείου"),
                     ("σπίτι", "μου", False, None)]
    assert E.host_class("όνομα") == "proparoxytone"
    assert E.host_class("δωμάτιο") == "synizesis"
    assert E.host_class("σπίτι") == "other"


def test_enclitic_summary_gives_precision_and_recall():
    s = E.enclitic_summary({"μου": [8, 2, 1, 89], "του": [1, 9, 0, 90]})
    assert s["always"] == {"hosts": 100, "corpus_accented": 9, "rule_adds": 10,
                           "precision": 0.8, "recall": 8 / 9, "agreement": 0.97}
    assert s["article"]["precision"] == 0.1 and s["preposition"]["hosts"] == 0


def test_the_enclitic_table_counts_rule_and_corpus():
    sents = [("news", ["Το", "όνομά", "μου", "είναι", "Νίκη"]),
             ("news", ["Το", "όνομα", "μου", "είναι", "Νίκη"]),
             ("news", ["Ο", "δάσκαλος", "του", "σχολείου"]),
             ("news", ["Άκουσέ", "το"])]
    out = E.run_enclitics(sents, None, 5)
    assert out["by_class"]["proparoxytone"]["μου"]["cells"] == [1, 1, 0, 0]
    assert out["keyboard_rule"]["του"]["cells"] == [0, 0, 0, 1]
    assert out["keyboard_rule"]["το"]["cells"] == [1, 0, 0, 0]
    evidence = {("όνομα", "μου"): [0, 9]}
    out = E.run_enclitics(sents, evidence, 5)
    assert out["with_training_evidence"]["by_enclitic"]["μου"]["cells"] == [0, 0, 1, 1]


def test_watched_groups_sum_every_group_with_both_spellings():
    from collections import Counter
    st = E.AccentStats()
    st.groups["η/ή"] = Counter(n=10, right=8, kept=2)
    st.group_out["η/ή"] = Counter({("η", "η"): 8, ("ή", "η"): 2})
    st.groups["ή/η/ηι"] = Counter(n=2, right=2)
    st.group_out["ή/η/ηι"] = Counter({("η", "η"): 2})
    rows = {r["group"]: r for r in st.watched_groups()}
    assert rows["η/ή"]["n"] == 12
    assert rows["η/ή"]["accuracy"] == pytest.approx(10 / 12)
    assert rows["η/ή"]["outcomes"][0] == ["η", "η", 10]
    assert rows["ποτέ/πότε"]["n"] == 0

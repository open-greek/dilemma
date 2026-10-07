"""Guards for the polytonic spellings generated for Modern Greek verb forms
(``mg_polytonic_paradigms``) and for their place in the polytonic Modern
Greek list (``export_mg_polytonic.generated_verb_forms``). They need no
corpus on disk."""
from __future__ import annotations

import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import export_mg_polytonic as mg  # noqa: E402
import mg_polytonic_paradigms as P  # noqa: E402
from export_mg_polytonic import CorpusCounts, DocumentInfo  # noqa: E402

GRC = ["ἀνάγκη", "ἀναβαίνω", "ἀναλύω", "ἀνήρ", "ἀνατέλλω", "ἀνά",
       "εὑρίσκω", "εὑρέθη", "εὑρέθην", "εὑρεῖν", "εὑρέσεις", "εὑρετής",
       "εὗρον", "ηὗρον", "εὐχαριστέω", "εὐχή", "εὔχομαι", "εὐθύς",
       "εὐλογέω", "εὐγενής", "εὐκαιρία", "εὐρύς", "εὐδαίμων", "εὐνοῦχος",
       "εὐσεβής", "εὐτυχία", "εὐχαριστία", "εὐχαρις", "εὐχάριστος"]
EVIDENCE = P.Evidence({}, GRC)


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def gen(mono: str, tags=(), vclass: str = "A", siblings=(), lemma: str = "",
        evidence: P.Evidence = EVIDENCE) -> str | None:
    return P.polytonic(mono, frozenset(tags), vclass, list(siblings),
                       evidence, lemma)


@pytest.mark.parametrize("mono, tags, vclass, expected", [
    # Accent on the penult: long vowel before a short last syllable takes
    # the circumflex, otherwise the acute.
    ("έρθεις", {"dependent", "second-person", "singular"}, "other", "ἔρθεις"),
    ("μιλήσεις", {"dependent", "second-person", "singular"}, "B1",
     "μιλήσεις"),
    ("καταλάβεις", {"dependent"}, "A", "καταλάβεις"),
    ("δώσε", {"imperative"}, "A", "δῶσε"),
    ("ήρθα", {"past"}, "other", "ἦρθα"),
    ("είσαι", {"present"}, "passive", "εἶσαι"),
    ("πούμε", {"dependent"}, "A", "ποῦμε"),
    # A long accented last syllable of a verb is a contraction: circumflex.
    ("μπορείς", {"present"}, "B2", "μπορεῖς"),
    ("δοθεί", {"dependent", "passive"}, "A", "δοθεῖ"),
    ("ευχαριστώ", {"present"}, "B2", "εὐχαριστῶ"),
    # The -άω verbs: their contracted α is long.
    ("μιλάτε", {"present"}, "B1", "μιλᾶτε"),
    ("μιλάς", {"present"}, "B1", "μιλᾶς"),
    # ...but not the α of an aorist stem (σκάσε), which is short.
    ("σκάσε", {"imperative"}, "B1", "σκάσε"),
    # Monotonic writing leaves a monosyllable unaccented; its last vowel
    # takes the accent.
    ("πιεις", {"dependent"}, "A", "πιεῖς"),
    ("πω", {"dependent"}, "A", "πῶ"),
    ("πες", {"imperative"}, "A", "πές"),
    # Ancient syllables place the accent: ἤ-πι-ε is a proparoxytone.
    ("ήπιε", {"past"}, "A", "ἤπιε"),
    # The imperative's -α counts long (ζήτα, φεύγα), as does -ᾶτε's α.
    ("ζήτα", {"imperative"}, "B1", "ζήτα"),
    ("φεύγα", {"imperative"}, "A", "φεύγα"),
    ("ελάτε", {"imperative", "plural"}, "other", "ἐλᾶτε"),
    # The gerund is generated with the acute the texts write most.
    ("μιλώντας", {"participle", "present"}, "B1", "μιλώντας"),
    # Breathings: an initial ρ is rough, υ is rough, a diphthong carries it
    # on its second vowel, and grc words with the same first letters decide.
    ("ρίξε", {"imperative"}, "A", "ῥίξε"),
    ("υπάρχει", {"present"}, "A", "ὑπάρχει"),
    ("αναρωτιέμαι", {"present"}, "passive", "ἀναρωτιέμαι"),
    ("ευρέθηκα", {"past"}, "A", "εὑρέθηκα"),
])
def test_marks_follow_the_rules(mono, tags, vclass, expected):
    assert gen(mono, tags, vclass) == nfc(expected)


def test_the_augment_takes_the_smooth_breathing():
    # ἦβρε, not ἧβρε as ἥβη would suggest; ηὗρα keeps its stem's breathing.
    assert gen("ήβρε", {"past"}, lemma="βρίσκω") == nfc("ἦβρε")
    assert gen("έγραψα", {"past"}, lemma="γράφω") == nfc("ἔγραψα")
    assert gen("ηύρα", {"past"}, lemma="βρίσκω",
               siblings=[("ηὗρε", 5)]) == nfc("ηὗρα")


def test_a_verbs_own_forms_decide_what_the_rules_cannot():
    # The length of α, ι, υ: the verb's forms decide (κᾶνε would be wrong,
    # but if the texts write it so, it follows them).
    assert gen("κάνε", {"imperative"}) == nfc("κάνε")
    assert gen("κάνε", {"imperative"}, siblings=[("κᾶνε", 5)]) == nfc("κᾶνε")
    # ...but a contraction's long α is not the stem's (γελᾶτε, γελάστε).
    assert gen("γελάστε", {"imperative", "perfective"}, "B1",
               siblings=[("γελᾶτε", 5), ("γελᾶς", 5)]) == nfc("γελάστε")
    assert gen("φάμε", {"dependent", "perfective"}, "other",
               siblings=[("φᾶς", 5)]) == nfc("φᾶμε")
    # The breathing: the verb's forms outweigh the grc words.
    assert gen("ενώσει", {"dependent"},
               siblings=[("ἑνώνει", 7), ("ἐνώνει", 1)]) == nfc("ἑνώσει")
    # A stem's iota subscript carries over only with the same next letter.
    assert gen("σώσει", {"dependent"},
               siblings=[("σῴσῃ", 4)]) == nfc("σῴσει")
    assert gen("αγαπάει", {"present"}, "B1",
               siblings=[("ἀγαπᾷ", 9)]) == nfc("ἀγαπάει")


def test_contracted_endings_take_the_subscript_the_texts_mostly_write():
    attested = {"ζητάς": Counter({"ζητᾷς": 8, "ζητᾶς": 2}),
                "ζητά": Counter({"ζητᾷ": 9})}
    paradigms = {"ζητάω": {
        "ζητάς": frozenset({"present", "second-person", "singular"}),
        "ζητά": frozenset({"present", "third-person", "singular"})}}
    evidence = P.Evidence(attested, GRC, paradigms)
    assert evidence.subscript_share["ᾶς"] == 0.8
    assert gen("ρωτάς", {"present"}, "B1", evidence=evidence) == nfc("ῥωτᾷς")
    # A word of one syllable takes none (πᾶς).
    assert gen("πας", {"present"}, "other", evidence=evidence) == nfc("πᾶς")


def test_the_subjunctive_writes_ει_as_ῃ():
    assert P.subjunctive_variant(nfc("πάρεις")) == nfc("πάρῃς")
    assert P.subjunctive_variant(nfc("μιλήσει")) == nfc("μιλήσῃ")
    assert P.subjunctive_variant(nfc("δοθεῖ")) == nfc("δοθῇ")
    assert P.subjunctive_variant(nfc("πιεῖς")) == nfc("πιῇς")
    # After a vowel too, as the texts write it most (ζητάῃ 92, -άη 49).
    assert P.subjunctive_variant(nfc("ζητάει")) == nfc("ζητάῃ")
    assert P.subjunctive_variant(nfc("ἀκούει")) == nfc("ἀκούῃ")
    assert P.subjunctive_variant(nfc("πάρετε")) is None
    assert P.is_subjunctive_cell(frozenset(
        {"dependent", "second-person", "singular"}))
    assert not P.is_subjunctive_cell(frozenset(
        {"past", "second-person", "singular"}))


def test_monotonic_keys_drop_the_accent_of_a_monosyllable():
    assert P.monotonic_key(nfc("πιῶ")) == nfc("πιω")
    assert P.monotonic_key(nfc("πὲς")) == nfc("πες")
    assert P.monotonic_key(nfc("ἔρθῃς")) == nfc("έρθης")
    assert P.monotonic_key(nfc("ἀγαπᾷς")) == nfc("αγαπάς")


def test_verb_classes():
    assert P.verb_class("ζητάω", {}) == "B1"
    assert P.verb_class("μπορώ", {"μπορείς": frozenset(
        {"present", "second-person", "singular"})}) == "B2"
    assert P.verb_class("γράφω", {}) == "A"
    assert P.verb_class("έρχομαι", {}) == "passive"


PARADIGMS = {
    "έρχομαι": {
        "έρθεις": frozenset({"dependent", "perfective", "second-person",
                             "singular", "finite"}),
        "έρθει": frozenset({"dependent", "perfective", "third-person",
                            "singular", "finite"}),
        "έρθουν": frozenset({"dependent", "perfective", "third-person",
                             "plural", "finite"}),
    },
}
FREQUENCIES = {"έρθεις": 900, "έρθει": 2000, "έρθουν": 30}


def test_generation_covers_frequent_forms_with_both_subjunctives():
    attested = {"έρθει": Counter({"ἔρθει": 7}), "έρθουν": Counter(
        {"ἔρθουν": 20})}
    out = P.generate(PARADIGMS, FREQUENCIES, P.Evidence(attested, GRC),
                     min_tokens=50)
    spellings = {(g.spelling, g.kind) for g in out}
    assert (nfc("ἔρθεις"), "indicative") in spellings
    assert (nfc("ἔρθῃς"), "subjunctive") in spellings
    assert (nfc("ἔρθῃ"), "subjunctive") in spellings
    # Below the frequency floor nothing is generated.
    assert not any(g.mono == "έρθουν" for g in out)


def _counts(rows):
    counts = CorpusCounts()
    counts.documents = [DocumentInfo(f"d{i}", a, f"t{i}", words=1000)
                        for i, a in enumerate(("Α", "Β", "Γ"))]
    for form, position, per_doc in rows:
        counts.forms[(nfc(form), position)].update(per_doc)
    return counts


def test_the_list_marks_generated_forms_and_fills_only_gaps(
        tmp_path, monkeypatch):
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (PARADIGMS, FREQUENCIES))
    monkeypatch.setattr(mg, "MONOTONIC_BUCKETS", "")
    counts = _counts([
        ("ἔρθει", "lower", {0: 4, 1: 3}),
        ("ἔρθουν", "lower", {0: 10, 1: 10}),
    ])
    lst = mg.select_list(counts, {0, 1, 2}, set(GRC), generate_verbs=True)
    assert nfc("ἔρθει") in lst.entries            # attested, listed
    assert nfc("ἔρθεις") in lst.generated         # the missing cell
    assert nfc("ἔρθῃς") in lst.generated
    assert nfc("ἔρθῃ") in lst.generated
    # The attested ἔρθει covers its monotonic form: nothing competes with it.
    assert nfc("ἔρθει") not in lst.generated
    # grc covers a form the same way.
    lst = mg.select_list(counts, {0, 1, 2}, set(GRC) | {nfc("ἔρθῃς")},
                         generate_verbs=True)
    assert nfc("ἔρθῃς") not in lst.generated
    mg.write_selection(lst, tmp_path, variant="grc-mg", source="test")
    lines = (tmp_path / "grc_mg_polytonic.dic").read_text("utf-8").splitlines()
    assert nfc("ἔρθεις\tfr:R mg:generated") in lines
    assert nfc("ἔρθει\tfr:R") in lines
    version = (tmp_path / "grc_mg_polytonic.version").read_text("utf-8")
    assert "generated: 2\n" in version and "attested: 2\n" in version


def test_generation_is_off_unless_asked(monkeypatch):
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (PARADIGMS, FREQUENCIES))
    counts = _counts([("ἔρθει", "lower", {0: 4, 1: 3})])
    assert mg.select_list(counts, {0, 1, 2}, set(GRC)).generated == frozenset()


def test_only_an_attested_spelling_of_the_same_letters_stops_a_form(
        monkeypatch):
    paradigms = {"μαθαίνω": {
        "μάθεις": frozenset({"dependent", "perfective", "second-person",
                             "singular", "finite"}),
        "μάθει": frozenset({"dependent", "perfective", "third-person",
                            "singular", "finite"}),
    }}
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (
        paradigms, {"μάθεις": 900, "μάθει": 900, "μαθεί": 2000}))
    monkeypatch.setattr(mg, "_VERB_INPUTS", {})
    # grc's μαθεῖς, an Ancient word of the same letters, does not stop the
    # Modern μάθεις; the list's attested μαθεῖ, whose monotonic form is
    # more than twice as frequent, stops the generated μάθει.
    counts = _counts([("μαθεῖ", "lower", {0: 2, 1: 2})])
    lst = mg.select_list(counts, {0, 1, 2}, set(GRC) | {nfc("μαθεῖς")},
                         generate_verbs=True)
    assert nfc("μαθεῖ") in lst.entries
    assert nfc("μάθεις") in lst.generated
    assert nfc("μάθει") not in lst.generated
    # Its subjunctive spelling has other letters, and is generated.
    assert nfc("μάθῃ") in lst.generated


IMPERATIVES = {"σταματάω": {
    "σταμάτα": frozenset({"imperative", "second-person", "singular",
                          "finite", "active"}),
    "σταματά": frozenset({"present", "third-person", "singular", "finite",
                          "active"}),
}, "ξεκινάω": {
    "ξεκινά": frozenset({"present", "third-person", "singular", "finite",
                         "active"}),
    "ξεκίνα": frozenset({"imperative", "second-person", "singular",
                         "finite", "active"}),
}, "ανεβαίνω": {
    "ανέβουμε": frozenset({"dependent", "first-person", "plural", "finite",
                           "active"}),
    "ανεβούμε": frozenset({"dependent", "first-person", "plural", "finite",
                           "active"}),
}, "τρέχω": {
    "τρέχατε": frozenset({"imperfect", "second-person", "plural", "finite",
                          "active"}),
    "τρεχάτε": frozenset({"imperative", "second-person", "plural", "finite",
                          "active"}),
}, "χαϊδεύω": {
    "χάιδεψε": frozenset({"past", "third-person", "singular", "finite",
                          "active"}),
}}


def test_another_cell_of_the_same_letters_goes_in_when_frequent(
        monkeypatch):
    freq = {"σταμάτα": 93_732, "σταματά": 2_994, "ξεκινά": 3_725,
            "ξεκίνα": 8_995, "ανέβουμε": 987, "ανεβούμε": 793,
            "τρέχατε": 157, "τρεχάτε": 129, "χάιδεψε": 195}
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (IMPERATIVES, freq))
    monkeypatch.setattr(mg, "_VERB_INPUTS", {})
    monkeypatch.setattr(mg, "MONOTONIC_BUCKETS", "")
    counts = _counts([("σταματᾷ", "lower", {0: 2, 1: 2}),
                      ("ξεκίνα", "lower", {0: 2, 1: 2}),
                      ("ἀνεβοῦμε", "lower", {0: 2, 1: 2}),
                      ("τρεχᾶτε", "lower", {0: 2, 1: 2}),
                      ("χάϊδεψε", "lower", {0: 2, 1: 2})])
    lst = mg.select_list(counts, {0, 1, 2}, set(GRC), generate_verbs=True)
    assert {nfc("σταματᾷ"), nfc("ξεκίνα"), nfc("ἀνεβοῦμε"), nfc("τρεχᾶτε"),
            nfc("χάϊδεψε")} <= set(lst.entries)
    # The imperative, a cell of its own and 31 times as frequent, goes in
    # beside the attested present.
    assert nfc("σταμάτα") in lst.generated
    # The present beside the attested imperative is under half as frequent.
    assert nfc("ξεκινᾷ") not in lst.generated
    # Another spelling of the same cell competes with the attested one.
    assert nfc("ἀνέβουμε") not in lst.generated
    # Nor does a form only 1.2 times as frequent (157 to 129).
    assert nfc("τρέχατε") not in lst.generated
    # A diaeresis aside, χάιδεψε is the attested χάϊδεψε.
    assert nfc("χάιδεψε") not in lst.generated
    monkeypatch.setattr(mg, "LETTERS_SHARED_ACROSS_CELLS", None)
    lst = mg.select_list(counts, {0, 1, 2}, set(GRC), generate_verbs=True)
    assert nfc("σταμάτα") not in lst.generated


def test_buckets_read_the_monotonic_counts(tmp_path, monkeypatch):
    total = 10_000_000
    freq = {"έρθεις": 400, "έρθει": 50, "έρθουν": 10, "πολύ": total - 460}
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (PARADIGMS, freq))
    monkeypatch.setattr(mg, "_VERB_INPUTS", {})
    assert mg.monotonic_bucket(370, total) == "C"
    assert mg.monotonic_bucket(369, total) == "M"
    assert mg.monotonic_bucket(37, total) == "M"
    assert mg.monotonic_bucket(36, total) == "R"
    # Monotonic writing spells the traditional subjunctive -εις.
    assert mg.monotonic_count(nfc("ἔρθῃς"), freq) == 400
    assert mg.monotonic_count(nfc("Ἔρθει"), freq) == 50
    assert mg.monotonic_count("κι" + mg.ELISION_MARKS[3], {"κι": 9}) == 0
    counts = _counts([("ἔρθει", "lower", {0: 4, 1: 3}),
                      ("ἔρθουν", "lower", {0: 10, 1: 10})])
    for setting, attested in (("all", "M"), ("generated", "R")):
        monkeypatch.setattr(mg, "MONOTONIC_BUCKETS", setting)
        lst = mg.select_list(counts, {0, 1, 2}, set(GRC),
                             generate_verbs=True)
        # ἔρθει: slice R, monotonic 5 per million, M; ἔρθεις generated,
        # 40 per million, C.
        assert lst.buckets[nfc("ἔρθει")] == attested
        assert lst.buckets[nfc("ἔρθεις")] == "C"
        out = tmp_path / setting
        mg.write_selection(lst, out, variant="grc-mg", source="test")
        lines = (out / "grc_mg_polytonic.dic").read_text("utf-8").splitlines()
        assert nfc("ἔρθεις\tfr:C mg:generated") in lines
        assert nfc(f"ἔρθει\tfr:{attested}") in lines
        version = (out / "grc_mg_polytonic.version").read_text("utf-8")
        assert "bucket_source: attested entries: " in version
        assert "data/mg_freq.txt" in version
    monkeypatch.setattr(mg, "MONOTONIC_BUCKETS", "")
    assert mg.select_list(counts, {0, 1, 2}, set(GRC),
                          generate_verbs=True).buckets is None


class _Corpora:
    """A stand-in for ``AncientCorpora`` with treebank tokens by letters."""
    def __init__(self, letters):
        self.letters = letters

    def letters_tokens(self, letters):
        return self.letters.get(letters, 0)


def test_letters_only_an_ancient_word_has_cap_the_bucket():
    corpora = _Corpora({"τακ": 22, "θε": 8})
    grc_letters = {"θε", "πολυ"}
    # τάκ: grc has no τακ, and the treebanks write τἀκ 22 times.
    assert mg.ancient_letters(nfc("τάκ"), grc_letters, corpora)
    # θέ: grc has θε, so a keyboard does not take its letters for Modern.
    assert not mg.ancient_letters(nfc("θέ"), grc_letters, corpora)
    assert not mg.ancient_letters(nfc("τώρα"), grc_letters, corpora)
    assert mg._without_diaeresis(nfc("χάϊδεψε")) == nfc("χάιδεψε")

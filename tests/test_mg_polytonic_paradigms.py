"""Guards for the polytonic spellings generated for Modern Greek verb forms
(``mg_polytonic_paradigms``) and for their place in the polytonic Modern
Greek list (``export_mg_polytonic.generated_verb_forms``). They need no
corpus on disk."""
from __future__ import annotations

import re
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
    hebe = P.Evidence({}, GRC + ["ἥβη", "ἥβης", "ἥβῃ", "ἥβην", "ἡβάω",
                                 "ἡβῶν"])
    assert gen("ήβρε", {"past"}, lemma="βρίσκω",
               evidence=hebe) == nfc("ἦβρε")
    # Without an augment the grc words decide: ἡβ- is rough.
    assert gen("ηβρε", {"present"}, evidence=hebe) is None
    assert gen("ήβρε", {"present"}, lemma="βρίσκω",
               evidence=hebe) == nfc("ἧβρε")
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
    # An accented ει becomes a circumflexed ῃ whatever its accent was.
    assert P.subjunctive_variant(nfc("δοθεί")) == nfc("δοθῇ")
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


# --------------------------------------------------------------------------
# The rules the review of the generated forms tightened
# --------------------------------------------------------------------------

def test_the_longest_shared_prefix_decides_the_breathing():
    # εστι- is split by ἐστί (smooth) and ἑστία (rough); εστια- is rough
    # throughout, and the longer prefix decides. Neither the split εστι-
    # nor the smooth εσ- below it may override it.
    words = ["ἐστί", "ἐστίν", "ἐστιν", "ἐστιγμένος", "ἐστίγμαι", "ἐστιχίζω",
             "ἑστία", "ἑστίας", "ἑστίαι", "ἑστιάω", "ἑστιάτωρ", "ἑστιατόριον",
             "ἐσθίω", "ἐσθλός", "ἔσχατος", "ἐσθής", "ἔσοπτρον", "ἐσπέρα"]
    ev = P.Evidence({}, words)
    assert gen("εστιάζω", {"present"}, evidence=ev) == nfc("ἑστιάζω")
    # A split prefix leaves the default, smooth, even where a shorter one
    # has a rough majority: ορφα- is split, ορ- is 90% rough.
    words = (["ὀρφανός", "ὀρφανή", "ὀρφανοῦ", "ὁρφαα", "ὁρφαβ"]
             + [f"ὁρ{c}{d}" for c in "αβγδε" for d in "κλμνξπ"])
    ev = P.Evidence({}, words)
    assert gen("ορφάνεψε", {"past"}, evidence=ev) == nfc("ὀρφάνεψε")
    assert gen("ορκίζω", {"present"}, evidence=ev) == nfc("ὁρκίζω")


def test_the_participle_acute_is_the_aorist_passives_alone():
    participle = {"masculine", "nominative", "singular"}
    assert gen("γραφείς", participle | {"participle"}) == nfc("γραφείς")
    assert gen("επιζών", participle, "other") == nfc("ἐπιζῶν")
    assert gen("πυροδοτούν", {"neuter", "nominative", "singular"},
               "other") == nfc("πυροδοτοῦν")
    # The finite dependent form follows its vowel.
    assert gen("γραφείς", {"dependent", "second-person", "singular",
                           "finite"}) == nfc("γραφεῖς")


def test_a_perfective_stem_reads_only_forms_with_its_next_letter():
    # The contracted κοιτᾶς and κοιτᾶτε say nothing of κοιτάχτε's stem.
    assert gen("κοιτάχτε", {"imperative", "perfective"}, "B1",
               siblings=[("κοιτᾶς", 9), ("κοιτᾶτε", 9)]) == nfc("κοιτάχτε")
    # A contracted ending after it reads them (φᾶμε beside φᾶς).
    assert gen("φάμε", {"dependent", "perfective"}, "other",
               siblings=[("φᾶς", 5)]) == nfc("φᾶμε")


def test_a_perfective_acute_does_not_shorten_a_contraction():
    siblings = [P.Sibling("σκάσε", 30, frozenset({"imperative",
                                                   "perfective"}))]
    assert gen("σκας", {"present", "imperfective"}, "B1",
               siblings=siblings) == nfc("σκᾶς")
    assert gen("σκάνε", {"present", "imperfective"}, "B1",
               siblings=siblings) == nfc("σκᾶνε")
    # The perfective form itself keeps its short α.
    assert gen("σκάστε", {"imperative", "perfective"}, "B1",
               siblings=siblings) == nfc("σκάστε")


def test_a_spelling_below_the_lists_floor_decides_no_length():
    # One token of γελᾶστε is a misspelling, not evidence.
    assert gen("γελάστε", {"imperative", "perfective"}, "B1",
               siblings=[("γελᾶστε", 1)]) == nfc("γελάστε")
    assert gen("γελάστε", {"imperative", "perfective"}, "B1",
               siblings=[("γελᾶστε", 3)]) == nfc("γελᾶστε")
    # ...nor a subscript, but still a breathing.
    assert gen("ενώσει", {"dependent"}, siblings=[("ἑνώνει", 1)]) == \
        nfc("ἑνώσει")
    assert gen("σώσει", {"dependent"}, siblings=[("σῴσῃ", 2)]) == \
        nfc("σώσει")


def test_contraction_defaults_of_polytonic_and_passive_lemmas():
    assert gen("μιλάνε", {"form-of"}, "other", lemma="μιλῶ") == nfc("μιλᾶνε")
    assert gen("αυταπατάσαι", {"present", "second-person", "singular",
                               "finite"}, "passive",
               lemma="αυταπατώμαι") == nfc("αὐταπατᾶσαι")
    assert gen("θυμάται", {"present", "third-person", "singular", "finite"},
               "passive", lemma="θυμάμαι") == nfc("θυμᾶται")


def test_a_present_form_and_a_bare_form_of_get_the_subjunctive_twin():
    tags = frozenset({"present", "third-person", "singular", "finite"})
    assert P.is_subjunctive_cell(tags)
    # Tagged only as a form of its verb, the -ει(ς) shape decides.
    assert P.is_subjunctive_cell(frozenset({"form-of"}), "ξαναδεί")
    assert P.is_subjunctive_cell(frozenset(), "οδηγείς")
    assert not P.is_subjunctive_cell(frozenset({"form-of"}), "οδηγούσε")
    assert not P.is_subjunctive_cell(frozenset({"past"}), "έγραφει")
    paradigms = {"γράφω": {"γράφει": tags}, "ξαναβλέπω": {
        "ξαναδεί": frozenset({"form-of"})}}
    out = P.generate(paradigms, {"γράφει": 500, "ξαναδεί": 500},
                     P.Evidence({}, GRC))
    spellings = {g.spelling for g in out}
    assert {nfc("γράφῃ"), nfc("ξαναδῇ")} <= spellings


def test_a_subjunctive_twin_with_another_words_letters_is_left_out():
    # Monotonic text writes υπολογιστή, the noun, far more than the
    # subjunctive's -ει; σταματήσης is only its misspelling.
    paradigms = {"υπολογίζω": {"υπολογιστεί": frozenset({"dependent",
                 "third-person", "singular", "finite"})},
                 "σταματάω": {"σταματήσεις": frozenset({"dependent",
                 "second-person", "singular", "finite"})}}
    freq = {"υπολογιστεί": 2000, "υπολογιστή": 12664,
            "σταματήσεις": 26000, "σταματήσης": 80}
    out = {g.spelling for g in P.generate(paradigms, freq,
                                          P.Evidence({}, GRC))}
    assert nfc("ὑπολογιστεῖ") in out and nfc("ὑπολογιστῇ") not in out
    assert nfc("σταματήσῃς") in out


def test_the_generation_floor():
    paradigms = {"γράφω": {"γράφεις": frozenset({"present", "finite",
                                                "second-person",
                                                "singular"}),
                           "γράφουν": frozenset({"present", "finite",
                                                "third-person", "plural"})}}
    freq = {"γράφεις": 50, "γράφουν": 49}
    out = {g.mono for g in P.generate(paradigms, freq, P.Evidence({}, GRC))}
    assert out == {"γράφεις"}


def test_a_form_of_several_verbs_gets_one_spelling():
    present = frozenset({"present", "first-person", "plural", "finite"})
    paradigms = {"πουλάω": {"πουλάμε": present, "πουλάς": frozenset(
                     {"present", "second-person", "singular", "finite"})},
                 "πωλώ": {"πουλάμε": present, "πωλείς": frozenset(
                     {"present", "second-person", "singular", "finite"})},
                 "μιλάνος": {"μιλάν": frozenset()},
                 "μιλάω": {"μιλάν": frozenset({"present", "third-person",
                                              "plural", "finite"})}}
    freq = {"πουλάμε": 1831, "πουλάς": 900, "πωλείς": 60, "μιλάν": 257}
    out = P.generate(paradigms, freq, P.Evidence({}, GRC))
    by_form = Counter(g.mono for g in out if g.kind == "indicative")
    assert by_form["πουλάμε"] == 1 and by_form["μιλάν"] == 1
    spellings = {g.mono: g.spelling for g in out}
    # The home verb is the -άω verb it shares most letters with, and a
    # verb rather than the name's page.
    assert spellings["πουλάμε"] == nfc("πουλᾶμε")
    assert spellings["μιλάν"] == nfc("μιλᾶν")
    # Most frequent first.
    assert [g.mono for g in out][:2] == ["πουλάμε", "πουλάς"]


def test_attested_spellings_tie_in_alphabetical_order():
    ev = P.Evidence({"δει": Counter({nfc("δῇ"): 4, nfc("δεῖ"): 4})})
    assert [s.spelling for s in ev.lemma_spellings({"δει": frozenset()})] \
        == [nfc("δεῖ"), nfc("δῇ")]


def test_a_shared_form_reads_every_verbs_attested_spellings():
    # επόμενες is έπομαι's (a verb before a participle's page), but only
    # επόμενος's forms are attested: ἑπόμενη.
    tags = frozenset({"feminine", "nominative", "plural"})
    paradigms = {"επόμενος": {"επόμενη": tags, "επόμενες": tags},
                 "έπομαι": {"επόμενες": tags}}
    ev = P.Evidence({"επόμενη": Counter({nfc("ἑπόμενη"): 9})}, GRC)
    out = {g.mono: g.spelling for g in P.generate(
        paradigms, {"επόμενες": 4558, "επόμενη": 900}, ev)}
    assert out["επόμενες"] == nfc("ἑπόμενες")


def _write_pairs(tmp_path, rows):
    import json
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps([dict(form=f, lemma=l, pos=pos, tags=t)
                                for f, l, pos, t in rows],
                               ensure_ascii=False), encoding="utf-8")
    return path


def test_pseudo_lemmas_rare_cells_and_table_fragments(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "FRAGMENT_SHARE", 0.3)
    rows = []
    # Many passive verbs whose present first person plural ends -όμαστε.
    for stem in ("αγαπ", "γραφ", "δεχ", "λυπ", "τρεχ", "κρατ", "χαιρ"):
        rows.append((stem + "όμαστε", stem + "άω", "verb",
                     ["present", "first-person", "plural", "passive"]))
    rows += [
        # A table fragment: an ending without its stem.
        ("όμαστε", "σαλπάρω", "verb",
         ["present", "first-person", "plural", "passive"]),
        ("σαλπάρω", "σαλπάρω", "verb", ["present", "first-person"]),
        # Suppletive forms stay.
        ("φάγατε", "τρώω", "verb", ["past", "second-person", "plural"]),
        ("ήρθα", "έρχομαι", "verb", ["past", "first-person", "singular"]),
        # An abbreviation is no verb; a heading-cased page is read in
        # lowercase.
        ("δες", "ΔΕΣ", "verb", []),
        ("ρώτησα", "Ερωτώ", "verb", ["past", "first-person", "singular"]),
        # A rare cell is kept unless another word has its letters.
        ("πιάνου", "πιάνω", "verb", ["imperative", "rare"]),
        ("πιάνου", "πιάνο", "noun", ["genitive"]),
        ("υπακούουν", "υπακούω", "verb", ["present", "archaic"]),
    ]
    par = P.load_verb_paradigms(_write_pairs(tmp_path, rows))
    assert "όμαστε" not in par["σαλπάρω"] and "σαλπάρω" in par["σαλπάρω"]
    assert "φάγατε" in par["τρώω"] and "ήρθα" in par["έρχομαι"]
    assert "ΔΕΣ" not in par and "δες" not in {
        f for forms in par.values() for f in forms}
    assert "ρώτησα" in par["ερωτώ"]
    assert "πιάνου" not in par.get("πιάνω", {})
    assert "υπακούουν" in par["υπακούω"]


def test_generated_spellings_of_the_same_letters_compete(monkeypatch):
    past = frozenset({"past", "third-person", "plural", "finite"})
    paradigms = {
        "πεθαίνω": {"πέθαναν": past, "πεθάναν": past},
        "ξεπερνάω": {"ξεπερνά": frozenset({"present", "third-person",
                                           "singular", "finite"}),
                     "ξεπέρνα": frozenset({"imperative", "second-person",
                                           "singular", "finite"})},
        "πουλάω": {"πούλα": frozenset({"imperative", "second-person",
                                       "singular", "finite"}),
                   "πουλά": frozenset({"present", "third-person",
                                       "singular", "finite"})},
    }
    freq = {"πέθαναν": 8404, "πεθάναν": 61, "ξεπερνά": 918, "ξεπέρνα": 71,
            "πούλα": 797, "πουλά": 442, "ανέβουμε": 987, "ανεβούμε": 793}
    paradigms["ανεβαίνω"] = {f: frozenset({"dependent", "first-person",
                                          "plural", "finite"})
                             for f in ("ανέβουμε", "ανεβούμε")}
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (paradigms, freq))
    monkeypatch.setattr(mg, "_VERB_INPUTS", {})
    monkeypatch.setattr(mg, "MONOTONIC_BUCKETS", "")
    lst = mg.select_list(_counts([]), {0, 1, 2}, set(GRC),
                         generate_verbs=True)
    # The same cell: the commoner stress only, however close the counts.
    assert nfc("πέθαναν") in lst.generated
    assert nfc("πεθάναν") not in lst.generated
    assert nfc("ἀνέβουμε") in lst.generated
    assert nfc("ἀνεβοῦμε") not in lst.generated
    # Another cell, at less than half: the commoner only. (No attested
    # contraction here to give the -ᾶ its subscript.)
    assert nfc("ξεπερνᾶ") in lst.generated
    assert nfc("ξεπέρνα") not in lst.generated
    # Another cell, as common: both.
    assert {nfc("πούλα"), nfc("πουλᾶ")} <= lst.generated


def test_a_name_neither_covers_a_verb_form_nor_lends_it_its_count(
        monkeypatch):
    third = frozenset({"imperfect", "third-person", "singular", "finite"})
    paradigms = {"δρω": {"δρούσε": third}, "κινάω": {"κινά": frozenset(
        {"present", "third-person", "singular", "finite"})}}
    freq = {"δρούσε": 173, "κινά": 2000, "πολύ": 100_000_000}
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (paradigms, freq))
    monkeypatch.setattr(mg, "_VERB_INPUTS", {})
    monkeypatch.setattr(mg, "MONOTONIC_BUCKETS", "all")
    # The slice writes Κινᾶ mostly capitalized inside a sentence: a name.
    # Κίνα is a name; κίνα, listed, is rarer in lowercase.
    counts = _counts([("Κινᾶ", "cap", {0: 5, 1: 5}),
                      ("Κίνα", "cap", {0: 9, 1: 9}),
                      ("κίνα", "lower", {0: 2, 1: 2})])
    freq["κίνα"] = 500
    lst = mg.select_list(counts, {0, 1, 2}, set(GRC) | {nfc("Δροῦσε")},
                         generate_verbs=True)
    assert nfc("δροῦσε") in lst.generated
    assert lst.buckets[nfc("δροῦσε")] == "R"
    # κινᾶ's 2,000 monotonic tokens are mostly the name's.
    assert lst.buckets[nfc("κινᾶ")] == "R"
    assert mg.slice_names(counts, {0, 1, 2}) == {nfc("κινά"), nfc("κίνα")}
    # The listed lowercase κίνα keeps its slice bucket, R: its monotonic
    # count, M, is mostly the name Κίνα's.
    assert lst.buckets[nfc("κίνα")] == "R"
    # grc's names spell no lowercase letters.
    assert mg._grc_letters({nfc("Τάκ"), nfc("θέ")}) == {"θε"}

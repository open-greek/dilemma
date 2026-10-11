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


@pytest.fixture(autouse=True)
def nominal_forms_from_test_paradigms(monkeypatch):
    # Unit fixtures are self-contained; do not read the on-disk corpus.
    def forms():
        inputs = mg._verb_inputs()
        return frozenset(f for rows in (inputs[0] if inputs else {}).values()
                         for f, tags in rows.items() if not P.is_verb(tags))
    monkeypatch.setattr(mg, "_nominal_forms", forms)


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


def test_another_cell_of_the_same_letters_goes_in(monkeypatch):
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
    # Another cell of the same letters goes in beside the attested
    # spelling, whatever their counts: the imperative σταμάτα beside the
    # present σταματᾷ, the present ξεκινᾶ beside the imperative ξεκίνα (at
    # less than half its count), the imperfect τρέχατε beside the
    # imperative τρεχᾶτε.
    assert {nfc("σταμάτα"), nfc("τρέχατε")} <= lst.generated
    assert {nfc("ξεκινᾶ"), nfc("ξεκινᾷ")} & lst.generated
    # Another spelling of the same cell gives way to the attested one.
    assert nfc("ἀνέβουμε") not in lst.generated
    # A diaeresis aside, χάιδεψε is the attested χάϊδεψε.
    assert nfc("χάιδεψε") not in lst.generated


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
        # The buckets line counts the attested and the generated entries.
        fields = dict(line.split(": ", 1) for line in version.splitlines())

        def tally(name):
            return {b: int(n) for b, n in (part.split("=")
                                           for part in fields[name].split())}
        both, attested_n, generated_n = (tally(k) for k in (
            "buckets", "attested_buckets", "generated_buckets"))
        assert generated_n["C"] >= 1
        assert both == {b: attested_n[b] + generated_n[b] for b in "CMR"}
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


def test_owned_noun_uses_paradigm_support_before_own_page_penalty(tmp_path):
    # Two independent source tables: feminine "shadow", and the dated
    # masculine "robber". The latter's genitive is not the common word.
    rows = [(form, "σκιά", "noun", tags) for form, tags in [
        ("σκιά", ["nominative", "singular"]),
        ("σκιά", ["accusative", "singular"]),
        ("σκιά", ["vocative", "singular"]),
        ("σκιάς", ["genitive", "singular"]),
        ("σκιές", ["nominative", "plural"]),
        ("σκιών", ["genitive", "plural"])]]
    rows += [(form, "σκιάς", "noun", tags) for form, tags in [
        ("σκιάς", ["nominative", "singular"]),
        ("σκιά", ["genitive", "singular"]),
        ("σκιά", ["accusative", "singular"]),
        ("σκιά", ["vocative", "singular"]),
        ("σκιάδες", ["nominative", "plural"])]]
    par = P.load_paradigms(_write_pairs(tmp_path, rows))
    frequencies = {"σκιά": 5853, "σκιάς": 557, "σκιές": 2734,
                   "σκιών": 941, "σκιάδες": 51}
    generated = P.generate(par, frequencies, P.Evidence({}, (), par))
    shadow = [g for g in generated if g.mono == "σκιά"]
    assert {g.lemma for g in shadow} == {"σκιά"}
    assert {g.spelling for g in shadow} == {nfc("σκιά")}
    # The independent masculine table and its distinct plural survive.
    assert "genitive" in par["σκιάς"]["σκιά"]
    assert any(g.mono == "σκιάδες" and g.lemma == "σκιάς"
               for g in generated)
    # A genuinely better-supported non-headword analysis can still win.
    totals = {"σκιά": 10085, "σκιάς": 20000}
    assert P._home_lemma("σκιά", ["σκιά", "σκιάς"], par, totals) == "σκιάς"


@pytest.mark.parametrize("other_pos, own_tags", [
    ("adj", ["nominative", "singular"]),
    ("noun", ["form-of"]),
    ("verb", ["nominative", "singular"]),
])
def test_noun_support_order_requires_independent_noun_heads(
        tmp_path, other_pos, own_tags):
    rows = [("σκιά", "σκιά", "noun", ["nominative", "singular"]),
            ("σκιάς", "σκιάς", other_pos, own_tags),
            ("σκιά", "σκιάς", other_pos, ["genitive", "singular"])]
    par = P.load_paradigms(_write_pairs(tmp_path, rows))
    # Mixed parts of speech or an unsupported self head retain the old
    # preference for the analysis beyond a word's own page.
    assert P._home_lemma("σκιά", ["σκιά", "σκιάς"], par,
                         {"σκιά": 10000, "σκιάς": 1}) == "σκιάς"


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
    # Another cell: both, at any ratio (περπάτα and περπατᾷ; here ξεπερνᾶ,
    # 918, and ξεπέρνα, 71). (No attested contraction here to give the -ᾶ
    # its subscript.)
    assert {nfc("ξεπερνᾶ"), nfc("ξεπέρνα")} <= lst.generated
    assert {nfc("πούλα"), nfc("πουλᾶ")} <= lst.generated


def test_a_name_neither_covers_a_verb_form_nor_lends_it_its_count(
        monkeypatch):
    third = frozenset({"imperfect", "third-person", "singular", "finite"})
    paradigms = {"δρω": {"δρούσε": third}, "κινάω": {"κινά": frozenset(
        {"present", "third-person", "singular", "finite"})},
        "μασάω": {"μάσα": frozenset({"imperative", "second-person",
                                     "singular", "finite"})}}
    freq = {"δρούσε": 173, "κινά": 2000, "μάσα": 2000, "πολύ": 100_000_000}
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (paradigms, freq))
    monkeypatch.setattr(mg, "_VERB_INPUTS", {})
    monkeypatch.setattr(mg, "MONOTONIC_BUCKETS", "all")
    # The slice writes Κινᾶ mostly capitalized inside a sentence: a name.
    # Κίνα is a name; κίνα, listed, is rarer in lowercase.
    counts = _counts([("Κινᾶ", "cap", {0: 5, 1: 5}),
                      ("Μάσα", "cap", {0: 5, 1: 5}),
                      ("Κίνα", "cap", {0: 9, 1: 9}),
                      ("κίνα", "lower", {0: 2, 1: 2})])
    freq["κίνα"] = 500
    lst = mg.select_list(counts, {0, 1, 2}, set(GRC) | {nfc("Δροῦσε")},
                         generate_verbs=True)
    assert nfc("δροῦσε") in lst.generated
    assert lst.buckets[nfc("δροῦσε")] == "R"
    # κινᾶ's 2,000 monotonic tokens are mostly the name's, and μάσα's.
    assert lst.buckets[nfc("κινᾶ")] == "R"
    assert lst.buckets[nfc("μάσα")] == "R"
    assert mg.slice_names(counts, {0, 1, 2}) == {
        nfc("κινά"), nfc("κίνα"), nfc("μάσα")}
    # A handful of capitals is no name: 4 Τσάντα beside 2 τσάντα, or one
    # Μαλάκας beside none.
    few = _counts([("Τσάντα", "cap", {0: 2, 1: 2}),
                   ("τσάντα", "lower", {0: 1, 1: 1}),
                   ("Μαλάκας", "cap", {0: 1})])
    assert mg.slice_names(few, {0, 1, 2}) == set()
    # The listed lowercase κίνα keeps its slice bucket, R: its monotonic
    # count, M, is mostly the name Κίνα's.
    assert lst.buckets[nfc("κίνα")] == "R"
    # grc's names spell no lowercase letters.
    assert mg._grc_letters({nfc("Τάκ"), nfc("θέ")}) == {"θε"}


def _pos(pos: str, *tags: str) -> frozenset:
    return frozenset({P.POS_PREFIX + pos, *tags})


@pytest.mark.parametrize("mono, tags, lemma, expected", [
    # Proparoxytones and the plain oxytone take the acute.
    ("τηλέφωνο", _pos("noun", "neuter", "nominative", "singular"),
     "τηλέφωνο", "τηλέφωνο"),
    ("κινητό", _pos("noun", "neuter", "nominative", "singular"), "κινητό",
     "κινητό"),
    ("ενημέρωση", _pos("noun", "feminine", "nominative", "singular"),
     "ενημέρωση", "ἐνημέρωση"),
    # A long penult before a short last syllable takes the circumflex: the
    # neuters in -ο and -μα (a noun in -μα is neuter when its tags do not
    # say), the plural -α.
    ("φαρμακείο", _pos("noun", "neuter", "nominative", "singular"),
     "φαρμακείο", "φαρμακεῖο"),
    ("νήμα", _pos("noun", "nominative", "singular"), "νήμα", "νῆμα"),
    ("σχολεία", _pos("noun", "neuter", "nominative", "plural"), "σχολείο",
     "σχολεῖα"),
    # A final -ι is long: the neuters from -ιον keep the acute.
    ("τσεκούρι", _pos("noun", "neuter", "nominative", "singular"),
     "τσεκούρι", "τσεκούρι"),
    # The feminine -α is long, but short in the -εια of an adjective in
    # -ύς (ταχεῖα), and the acute stays (χώρα).
    ("χώρα", _pos("noun", "feminine", "nominative", "singular"), "χώρα",
     "χώρα"),
    ("ταχεία", _pos("adj", "feminine", "nominative", "singular"), "ταχύς",
     "ταχεῖα"),
    # A nominative singular in -α is a feminine's whatever lemma it is
    # filed under (σκούπα under the neuter σκουπόχορτο).
    ("σκούπα", _pos("noun", "nominative", "singular"), "σκουπόχορτο",
     "σκούπα"),
    # Synizesis: the -ιά of a feminine is long, so its genitive takes the
    # circumflex; a neuter plural's -ιά is short.
    ("καρδιά", _pos("noun", "feminine", "nominative", "singular"),
     "καρδιά", "καρδιά"),
    ("καρδιάς", _pos("noun", "feminine", "genitive", "singular"),
     "καρδιά", "καρδιᾶς"),
    ("καρδιών", _pos("noun", "feminine", "genitive", "plural"), "καρδιά",
     "καρδιῶν"),
    ("παιδιά", _pos("noun", "neuter", "nominative", "plural"), "παιδί",
     "παιδιά"),
    ("καλής", _pos("adj", "feminine", "genitive", "singular"), "καλός",
     "καλῆς"),
    # The nouns in -άς keep the circumflex in every case.
    ("μπαμπάς", _pos("noun", "nominative", "singular"), "μπαμπάς",
     "μπαμπᾶς"),
    ("μπαμπά", _pos("noun", "genitive", "singular"), "μπαμπάς", "μπαμπᾶ"),
    # The contracted plural -εῖς, but not an aorist passive participle
    # filed under its own form.
    ("γονείς", _pos("noun", "nominative", "plural"), "γονέας", "γονεῖς"),
    ("γραφείς", _pos("noun", "nominative", "singular"), "γραφείς",
     "γραφείς"),
    # An s-stem adjective: the acute in the nominative singular only.
    ("ασθενής", _pos("adj", "nominative", "singular", P.S_STEM), "ασθενής",
     "ἀσθενής"),
    ("ασθενή", _pos("adj", "accusative", "singular", P.S_STEM), "ασθενής",
     "ἀσθενῆ"),
    ("ζερβή", _pos("adj", "accusative", "singular"), "ζερβής", "ζερβή"),
    # A form Wiktionary files with no cell, in -ού, is a genitive.
    ("καλού", _pos("adj", "form-of"), "καλός", "καλοῦ"),
    # An adverb's -ά is short.
    ("ξανά", _pos("adv"), "ξανά", "ξανά"),
    # A name keeps its capital.
    ("Λονδίνο", _pos("name", "neuter"), "λονδίνο", "Λονδίνο"),
])
def test_nouns_and_adjectives_follow_the_ancient_rules(mono, tags, lemma,
                                                       expected):
    assert P.polytonic(mono, tags, P.verb_class(lemma, {mono: tags}), [],
                       EVIDENCE, lemma) == nfc(expected)


def test_every_part_of_speech_is_read(tmp_path):
    rows = [
        ("τηλέφωνο", "τηλέφωνο", "noun", ["nominative", "singular"]),
        ("τηλεφώνου", "τηλέφωνο", "noun", ["genitive", "singular"]),
        ("τηλεφωνώ", "τηλεφωνώ", "verb", ["present", "first-person"]),
        # A page filed only as a form of another, under a lemma in sentence
        # case, where the word's own table places the form.
        ("τηλεφώνου", "Τηλέφωνο", "noun", ["form-of"]),
        # A name with the letters of another word, one without, and a
        # foreign name Wiktionary does not decline.
        ("Τηλεφώνου", "Τηλέφωνος", "name", ["genitive"]),
        ("Λονδίνο", "Λονδίνο", "name", ["nominative", "singular"]),
        ("Λονδίνου", "Λονδίνο", "name", ["genitive", "singular"]),
        ("Λόνδρα", "Λονδίνο", "name", ["alt-of"]),
        ("Τέσα", "Τέσα", "name", ["feminine"]),
        # A polytonic spelling filed as a form.
        ("ζᾶ", "ζώο", "noun", ["form-of"]),
        # The adjectives in -ής: an s-stem, and one of another declension.
        ("ασθενής", "ασθενής", "adj", ["nominative", "singular"]),
        ("ασθενές", "ασθενής", "adj", ["neuter", "nominative", "singular"]),
        ("ζερβής", "ζερβής", "adj", ["nominative", "singular"]),
        ("ζερβί", "ζερβής", "adj", ["neuter", "nominative", "singular"]),
        # Its feminine plural in -ιές is no s-stem's neuter in -ές.
        ("ζερβιές", "ζερβής", "adj", ["feminine", "nominative", "plural"]),
        # A romanization is no Greek word.
        ("tilefono", "τηλέφωνο", "romanization", []),
    ]
    par = P.load_paradigms(_write_pairs(tmp_path, rows))
    assert set(par["τηλέφωνο"]) == {"τηλέφωνο", "τηλεφώνου"}
    assert P.is_verb(par["τηλεφωνώ"]["τηλεφωνώ"])
    assert P.pos_of(par["τηλέφωνο"]["τηλεφώνου"]) == "noun"
    assert "τηλέφωνο" not in par or "form-of" not in \
        par["τηλέφωνο"]["τηλεφώνου"]
    assert "τηλέφωνος" not in par and "Λονδίνο" in par["λονδίνο"]
    assert "τέσα" not in par and "Λόνδρα" in par["λονδίνο"]
    assert "ζώο" not in par
    assert P.S_STEM in par["ασθενής"]["ασθενής"]
    assert P.S_STEM not in par["ζερβής"]["ζερβής"]
    assert P.verb_class("λονδίνο", par["λονδίνο"]) == "name"
    # The verbs alone, as the verb evaluation reads them.
    assert set(P.load_verb_paradigms(_write_pairs(tmp_path, rows))) == {
        "τηλεφωνώ"}


def test_a_verbs_twin_gives_way_to_a_far_commoner_noun(monkeypatch):
    # The subjunctive twin ἐνημερώσῃ has the letters of the noun
    # ἐνημέρωση, which monotonic text writes far more often than the
    # twin's own letters (ενημερώση): the twin is left out, the noun and
    # the indicative ἐνημερώσει go in.
    paradigms = {
        "ενημερώνω": {"ενημερώσει": frozenset({
            "dependent", "third-person", "singular", "finite", "perfective",
            P.VERB})},
        "ενημέρωση": {"ενημέρωση": _pos("noun", "feminine", "nominative",
                                        "singular")},
    }
    # The verb form is commoner than the noun, its twin's own letters are
    # not: the twin is ranked by the latter.
    freq = {"ενημερώσει": 50_000, "ενημέρωση": 40_000, "ενημερώση": 4}
    order = [g.spelling for g in P.generate(paradigms, freq, EVIDENCE)]
    # The commonest spelling of the letters comes first.
    assert order.index(nfc("ἐνημέρωση")) < order.index(nfc("ἐνημερώσῃ"))
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (paradigms, freq))
    monkeypatch.setattr(mg, "_VERB_INPUTS", {})
    monkeypatch.setattr(mg, "MONOTONIC_BUCKETS", "")
    lst = mg.select_list(_counts([]), {0, 1, 2}, set(GRC),
                         generate_verbs=True)
    assert {nfc("ἐνημέρωση"), nfc("ἐνημερώσει")} <= lst.generated
    assert nfc("ἐνημερώσῃ") not in lst.generated
    # A noun's form in -εις is no subjunctive, even filed with no cell: no
    # twin (πόλεις, not πόλῃς).
    nouns = {"πόλη": {"πόλεις": _pos("noun", "form-of")}}
    assert {g.spelling for g in P.generate(nouns, {"πόλεις": 5000},
                                           EVIDENCE)} == {nfc("πόλεις")}


def test_a_spelling_with_another_words_letters_gives_way(monkeypatch):
    # εὑρῶ, the Ancient subjunctive, has the letters of εὐρώ, the currency.
    paradigms = {"ευρώ": {"ευρώ": _pos("noun", "neuter", "invariable")}}
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (paradigms,
                                                     {"ευρώ": 4311}))
    monkeypatch.setattr(mg, "_VERB_INPUTS", {})
    monkeypatch.setattr(mg, "MONOTONIC_BUCKETS", "")
    counts = _counts([("εὑρῶ", "lower", {0: 2, 1: 2, 2: 1})])
    lst = mg.select_list(counts, {0, 1, 2}, set(GRC), generate_verbs=True)
    assert nfc("εὑρῶ") not in lst.entries
    # Its rough breathing does not pass to the generated spelling.
    assert nfc("εὐρώ") in lst.generated
    lst = mg.select_list(counts, {0, 1, 2}, set(GRC) | {nfc("εὑρῶ")},
                         generate_verbs=True)
    assert nfc("εὐρώ") in lst.generated and nfc("εὑρῶ") in lst.avoid


def test_datives_capitals_and_the_bucket_ceiling(monkeypatch):
    noun = _pos("noun", "neuter", "nominative", "singular")
    paradigms = {
        "κέντρο": {"κέντρο": noun},
        "Ιούνιος": {"Ιούνιο": _pos("name", "accusative", "singular")},
        "γράψης": {"Γράψη": _pos("name", "genitive", "singular")},
        "τζίμας": {"Τζίμα": _pos("name", "genitive", "singular")},
        "λονδίνο": {"Λονδίνο": _pos("name", "nominative", "singular")},
        "μιλάω": {"μίλα": frozenset({
            "imperative", "second-person", "singular", "finite", P.VERB})},
        "κοιτάζω": {"κοίταξε": frozenset({
            "past", "third-person", "singular", "finite", P.VERB})},
    }
    total = 100_000_000
    freq = {"κέντρο": 60_000, "ιούνιο": 3_000, "γράψη": 80, "μίλα": 9_000,
            "μιλά": 1_000, "κοίταξε": 6_000, "τζίμα": 85, "λονδίνο": 14_283,
            "πολύ": total - 93_448}
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (paradigms, freq))
    monkeypatch.setattr(mg, "_VERB_INPUTS", {})
    monkeypatch.setattr(mg, "MONOTONIC_BUCKETS", "all")
    counts = _counts([("κέντρῳ", "lower", {0: 3, 1: 3}),
                      ("γράψῃ", "lower", {0: 3, 1: 3}),
                      ("μιλᾶ", "lower", {0: 3, 1: 3}),
                      ("κοίταξέ", "lower", {0: 2, 1: 2})])
    lst = mg.select_list(counts, {0, 1, 2}, set(GRC), generate_verbs=True)
    # A Katharevousa dative does not cover the plain spelling.
    assert nfc("κέντρῳ") in lst.entries and nfc("κέντρο") in lst.generated
    assert lst.buckets[nfc("κέντρο")] == "C"
    # A month is generated with its capital, at R; a surname with the
    # letters of a lowercase word is not.
    assert nfc("Ἰούνιο") in lst.generated
    assert lst.buckets[nfc("Ἰούνιο")] == "R"
    assert not any(s.lower() == nfc("γράψη") for s in lst.generated)
    # A name with the plain acute is generated where monotonic text has it
    # often enough to rank (Λονδίνο), not where it is rare (Τζίμα).
    assert nfc("Λονδίνο") in lst.generated
    assert nfc("Τζίμα") not in lst.generated
    # The imperative μίλα, C by its count, ranks no higher than the
    # attested μιλᾶ of its letters, M.
    assert nfc("μίλα") in lst.generated
    assert lst.buckets[nfc("μιλᾶ")] == "M"
    assert lst.buckets[nfc("μίλα")] == "M"
    # The enclitic's second accent makes no other word: κοίταξέ, R by its
    # count, sets no ceiling for κοίταξε.
    assert lst.buckets[nfc("κοίταξέ")] == "R"
    assert lst.buckets[nfc("κοίταξε")] == "C"


def test_a_clitic_of_one_syllable_is_not_generated():
    # Polytonic writing leaves an article or a weak pronoun unaccented
    # where it leans on its neighbor; a paradigm does not say where.
    assert P.polytonic("τσου", _pos("article"), "article", [], EVIDENCE,
                       "τσι") is None
    assert P.polytonic("ντα", _pos("pron"), "pron", [], EVIDENCE,
                       "είντα") is None
    # A noun of one syllable takes its accent.
    assert P.polytonic("φως", _pos("noun", "nominative", "singular"),
                       "noun", [], EVIDENCE, "φως") == nfc("φῶς")


def test_a_first_declension_dative_and_a_loanword(monkeypatch):
    noun = _pos("noun", "feminine", "nominative", "singular")
    paradigms = {
        "πρωτοβουλία": {"πρωτοβουλία": noun},
        "μπάσκετ": {"μπάσκετ": _pos("noun", "neuter", "invariable")},
        "γράψης": {"Γράψη": _pos("name", "genitive", "singular")},
    }
    freq = {"πρωτοβουλία": 5000, "μπάσκετ": 3000, "γράψη": 5000,
            "πολύ": 100_000_000}
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (paradigms, freq))
    monkeypatch.setattr(mg, "_VERB_INPUTS", {})
    monkeypatch.setattr(mg, "MONOTONIC_BUCKETS", "")
    counts = _counts([("πρωτοβουλίᾳ", "lower", {0: 3, 1: 3}),
                      ("γράψῃ", "lower", {0: 3, 1: 3})])
    lst = mg.select_list(counts, {0, 1, 2}, set(GRC), generate_verbs=True)
    # The Katharevousa dative covers no plain spelling of its letters.
    assert nfc("πρωτοβουλίᾳ") in lst.entries
    assert nfc("πρωτοβουλία") in lst.generated
    # A loanword may end in a consonant.
    assert nfc("μπάσκετ") in lst.generated
    # A surname with the letters of a lowercase subjunctive is not
    # generated, however common its letters.
    assert nfc("Γράψη") not in lst.generated


def test_grc_spellings_compete_by_their_monotonic_counts(monkeypatch):
    paradigms = {
        # The same form of the same word: grc's φακέλους, far commoner,
        # keeps out φάκελους; χειρουργός, commoner, keeps out the
        # misaccented χειρούργος; the rarer grc spelling keeps out nothing.
        "φάκελος": {"φάκελους": _pos("noun", "accusative", "plural"),
                    "φακέλους": _pos("noun", "accusative", "plural")},
        "χειρουργός": {"χειρούργος": _pos("noun", "nominative", "singular"),
                       "χειρουργός": _pos("noun", "nominative",
                                          "singular")},
        "επίπεδος": {"επίπεδου": _pos("adj", "genitive", "singular"),
                     "επιπέδου": _pos("adj", "genitive", "singular")},
        # The same case and number of one word filed under the noun and the
        # adjective is one form.
        # A capitalized form gives way as its lowercase one does.
        "υπάλληλος": {"υπάλληλους": _pos("adj", "accusative", "plural"),
                      "Υπάλληλους": _pos("adj", "accusative", "plural"),
                      "υπαλλήλους": _pos("noun", "accusative", "plural")},
        # Another word of the same letters: both.
        "ένοχος": {"ένοχη": _pos("adj", "feminine", "nominative",
                                  "singular")},
        "ενοχή": {"ενοχή": _pos("noun", "nominative", "singular")},
    }
    freq = {"φάκελους": 59, "φακέλους": 2712, "χειρούργος": 1286,
            "χειρουργός": 1643, "επίπεδου": 400, "επιπέδου": 300,
            "υπάλληλους": 126, "υπαλλήλους": 2388, "ένοχη": 3072,
            "ενοχή": 3213}
    grc = set(GRC) | {nfc(w) for w in ("φακέλους", "χειρουργός",
                                       "ἐπιπέδου", "ὑπαλλήλους", "ἐνοχή")}
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (paradigms, freq))
    monkeypatch.setattr(mg, "_VERB_INPUTS", {})
    monkeypatch.setattr(mg, "MONOTONIC_BUCKETS", "")
    lst = mg.select_list(_counts([]), {0, 1, 2}, grc, generate_verbs=True)
    assert not {nfc("φάκελους"), nfc("χειροῦργος"),
                nfc("ὑπάλληλους"), nfc("Ὑπάλληλους")} & lst.generated
    assert {nfc("ἐπίπεδου"), nfc("ἔνοχη")} <= lst.generated


def test_shared_spellings_get_a_modern_rating(tmp_path, monkeypatch):
    total = 100_000_000
    paradigms = {
        "θέση": {"θέση": _pos("noun", "feminine", "nominative",
                              "singular")},
        "κοιτάζω": {"κοίτα": frozenset({"imperative", "second-person",
                                        "singular", "finite", P.VERB})},
        "θέτω": {"θέσει": frozenset({"dependent", "third-person",
                                     "singular", "finite", P.VERB})},
    }
    freq = {"θέση": 30_000, "κοίτα": 20_000, "θέσει": 9_000,
            "πολύ": total - 59_000}
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (paradigms, freq))
    monkeypatch.setattr(mg, "_VERB_INPUTS", {})
    monkeypatch.setattr(mg, "MONOTONIC_BUCKETS", "all")
    grc = set(GRC) | {nfc("θέση"), nfc("κοίτα")}
    # θέση is attested in the slice; κοίτα only generated; grc has both.
    counts = _counts([("θέση", "lower", {0: 3, 1: 3}),
                      ("θέσῃ", "lower", {0: 3, 1: 3}),
                      ("θέσει", "lower", {0: 27, 1: 27})])
    lst = mg.select_list(counts, {0, 1, 2}, grc, generate_verbs=True)
    assert lst.shared == {nfc("θέση"): "C", nfc("κοίτα"): "C"}
    # The subjunctive θέσῃ is rated by its own use: the slice writes these
    # letters θέσῃ 6 times and θέσει 54, so θέσῃ takes 7/62 of θέσει's 9,000
    # tokens, M, not all of them, C.
    assert lst.buckets[nfc("θέσῃ")] == "M"
    mg.write_selection(lst, tmp_path, variant="grc-mg", source="test")
    lines = (tmp_path / "grc_mg_polytonic.dic").read_text("utf-8")
    assert nfc("θέση\tfr:C mg:shared\n") in lines
    version = (tmp_path / "grc_mg_polytonic.version").read_text("utf-8")
    assert "mg_shared: 2\n" in version
    assert "shared_buckets: C=2 M=0\n" in version


def test_no_shared_rating_for_a_spelling_the_slice_writes_otherwise(
        monkeypatch):
    total = 100_000_000
    paradigms = {"εγώ": {"εμάς": _pos("pron", "accusative", "plural")}}
    freq = {"εμάς": 30_000, "πολύ": total - 30_000}
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (paradigms, freq))
    monkeypatch.setattr(mg, "_VERB_INPUTS", {})
    monkeypatch.setattr(mg, "MONOTONIC_BUCKETS", "all")
    grc = set(GRC) | {nfc("ἐμάς"), nfc("ἐμᾶς")}
    # The slice writes ἐμᾶς; grc's ἐμάς, the generated spelling, gets no
    # Modern Greek rating.
    counts = _counts([("ἐμᾶς", "lower", {0: 3, 1: 3})])
    lst = mg.select_list(counts, {0, 1, 2}, grc, generate_verbs=True)
    assert lst.shared == {nfc("ἐμᾶς"): "C"}


def test_a_subscript_spelling_takes_its_share_of_the_monotonic_count():
    frequencies = {"θέσει": 2110, "σταματήσει": 25119}
    assert mg.subscript_count(nfc("θέσῃ"), frequencies,
                              {nfc("θέσῃ"): 19, nfc("θέσει"): 19}) == 1055
    # The slice's -ει spelling of a perispomenon takes the circumflex on
    # its ι (μπεῖ beside μπῇ).
    assert mg.subscript_count(nfc("μπῇ"), {"μπει": 800},
                              {nfc("μπῇ"): 2, nfc("μπεῖ"): 4}) == 300
    # A twin the slice has in neither spelling takes half.
    assert mg.subscript_count(nfc("σταματήσῃς"), {"σταματήσεις": 800},
                              {}) == 400


def test_a_name_whose_only_reading_is_a_dative(monkeypatch):
    paradigms = {
        "ρωσία": {"Ρωσία": _pos("name", "nominative", "singular"),
                  "Ρωσίας": _pos("name", "genitive", "singular")},
        "δανία": {"Δανία": _pos("name", "nominative", "singular"),
                  "Δανίας": _pos("name", "genitive", "singular")},
    }
    freq = {"ρωσία": 4147, "ρωσίας": 646, "δανία": 676, "δανίας": 214,
            "πολύ": 100_000_000}
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (paradigms, freq))
    monkeypatch.setattr(mg, "_VERB_INPUTS", {})
    monkeypatch.setattr(mg, "MONOTONIC_BUCKETS", "")
    grc = set(GRC) | {nfc("Ῥωσίᾳ"), nfc("ῥωσίᾳ")}
    lst = mg.select_list(_counts([]), {0, 1, 2}, grc, generate_verbs=True)
    # A dative in -ᾳ covers no name; a name declined in two frequent forms
    # is generated though its count stays under M.
    assert {nfc("Ῥωσία"), nfc("Ῥωσίας"), nfc("Δανία"),
            nfc("Δανίας")} <= lst.generated


@pytest.mark.parametrize("dative_in_grc", [False, True])
def test_the_plain_spelling_beside_a_listed_dative(tmp_path, monkeypatch,
                                                  dative_in_grc):
    total = 100_000_000
    paradigms = {"λαμβάνω": {"λάβει": frozenset({
        "dependent", "third-person", "singular", "finite", P.VERB})}}
    freq = {"μόσχα": 1_000, "πινακοθήκη": 139, "ζυρίχη": 592,
            "βυζαντινή": 32, "λάβη": 10, "σιγηλή": 0,
            "πολύ": total - 1_773}
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (paradigms, freq))
    monkeypatch.setattr(mg, "_VERB_INPUTS", {})
    monkeypatch.setattr(mg, "_nominal_forms", lambda: frozenset(
        {"Μόσχα", "πινακοθήκη", "Ζυρίχη", "βυζαντινή", "λαβή", "σιγηλή"}))
    monkeypatch.setattr(mg, "MONOTONIC_BUCKETS", "all")
    grc = set(GRC) | {nfc(w) for w in ("Μόσχα", "πινακοθήκη", "λάβη",
                                       "σιγηλή")}
    if dative_in_grc:
        grc |= {nfc("πινακοθήκῃ"), nfc("Μόσχᾳ")}
    counts = _counts([("Μόσχᾳ", "cap", {0: 3, 1: 3}),
                      ("πινακοθήκῃ", "lower", {0: 3, 1: 3}),
                      ("Ζυρίχῃ", "cap", {0: 3, 1: 3}),
                      ("βυζαντινῇ", "lower", {0: 3, 1: 3}),
                      ("σιγηλῇ", "lower", {0: 3, 1: 3}),
                      ("λάβῃ", "lower", {0: 3, 1: 3})])
    lst = mg.select_list(counts, {0, 1, 2}, grc, generate_verbs=True)
    # The list attests the datives; the nominative and accusative of the
    # same letters, which grc has, get a Modern Greek rating, R as well,
    # and a name's capitalized (Μόσχα: 10 per million, M).
    assert lst.shared[nfc("Μόσχα")] == "M"
    assert lst.shared[nfc("πινακοθήκη")] == "R"
    # Those neither list has are generated, with their oxytone twins.
    assert {nfc("Ζυρίχη"), nfc("βυζαντινή"), nfc("βυζαντινὴ")} \
        <= lst.generated
    # Not the letters of a subjunctive (λάβει), nor where monotonic text
    # never writes the plain letters.
    assert nfc("λάβη") not in lst.shared
    assert nfc("σιγηλή") not in lst.shared
    mg.write_selection(lst, tmp_path, variant="grc-mg", source="test")
    lines = (tmp_path / "grc_mg_polytonic.dic").read_text("utf-8")
    assert nfc("πινακοθήκη\tfr:R mg:shared\n") in lines
    assert nfc("Ζυρίχη\tfr:R mg:generated\n") in lines
    version = (tmp_path / "grc_mg_polytonic.version").read_text("utf-8")
    assert "shared_buckets: C=0 M=1 R=1\n" in version


@pytest.mark.parametrize("dative, plain", [
    ("πινακοθήκῃ", "πινακοθήκη"),
    ("Μόσχᾳ", "Μόσχα"),
    ("βυζαντινῇ", "βυζαντινή"),
    ("καρδιᾷ", "καρδιά"),
    ("γῇ", "γῆ"),           # grc writes the plain letters with it
    ("ἁπλῇ", "ἁπλή"),      # Ancient contracted ἁπλῆ is another spelling
    ("τριπλῇ", "τριπλή"),
    ("λόγῳ", None),         # the second declension's dative
    ("πινακοθήκη", None),
])
def test_plain_twin(dative, plain):
    assert mg.plain_twin(nfc(dative), {nfc("γῆ"), nfc("ἁπλῆ"), nfc("τριπλῆ")}) == (
        nfc(plain) if plain else None)


def test_a_real_nominal_dative_twin_is_not_a_verbs_subjunctive(monkeypatch):
    paradigms = {
        "κλίνω": {"κλίσει": _pos("verb", "dependent", "third-person", "singular")},
        "λύνω": {"λύσει": _pos("verb", "dependent", "third-person", "singular")},
        "κατακλίνω": {"κατακλίσει": _pos("verb", "dependent", "third-person", "singular")},
    }
    freq = {"κλίση": 27, "λύση": 14532, "κατάκλιση": 27,
            "πολύ": 100_000_000}
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (paradigms, freq))
    monkeypatch.setattr(mg, "_VERB_INPUTS", {})
    monkeypatch.setattr(mg, "_nominal_forms", lambda: frozenset(
        {"κλίση", "λύση", "κατάκλιση"}))
    shared, generated = mg.dative_twins(
        {nfc(s): 6 for s in ("κλίσῃ", "λύσῃ", "κατάκλισῃ", "κατακλίσῃ")}, {},
        {nfc("λύση")}, set(), frozenset(), Counter())
    assert shared == {nfc("λύση"): 14532}
    assert generated[nfc("κλίση")] == 27
    assert generated[nfc("κατάκλιση")] == 27
    assert nfc("κατακλίση") not in generated


def test_nominal_rows_preserve_genitive_accusative_and_vocative(tmp_path):
    rows = [("καθηγητή", "καθηγητής", "noun", [case, "singular"])
            for case in ("genitive", "accusative", "vocative")]
    par = P.load_paradigms(_write_pairs(tmp_path, rows))
    freq = {"καθηγητή": 2_000}
    generated = P.generate(par, freq, P.Evidence({}, set(), par))
    assert {(g.spelling, g.cell) for g in generated} == {
        (nfc("καθηγητῆ"), "noun genitive singular"),
        (nfc("καθηγητή"), "noun accusative singular"),
        (nfc("καθηγητή"), "noun vocative singular"),
    }


def test_nominal_syncretic_row_preserves_each_accent(tmp_path):
    rows = [("μαθητή", "μαθητής", "noun",
             ["genitive", "accusative", "vocative", "singular"])]
    par = P.load_paradigms(_write_pairs(tmp_path, rows))
    generated = P.generate(par, {"μαθητή": 1_000}, P.Evidence({}, set(), par))
    assert {g.spelling for g in generated} == {nfc("μαθητῆ"), nfc("μαθητή")}


@pytest.mark.parametrize("noun_grc, noun_new", [
    ("καθηγητῆ", "καθηγητή"), ("καθηγητή", "καθηγητῆ")])
@pytest.mark.parametrize("modern_in_grc", [False, True])
def test_nominal_coverage_and_shared_counts_follow_polytonic_cells(
        tmp_path, monkeypatch, noun_grc, noun_new, modern_in_grc):
    rows = [("καθηγητή", "καθηγητής", "noun", [case, "singular"])
            for case in ("genitive", "accusative", "vocative")]
    rows += [("απλή", "απλός", "adj", ["nominative", "singular", "feminine"]),
             ("απλοί", "απλός", "adj", ["nominative", "plural", "masculine"]),
             ("διεθνή", "διεθνής", "adj", ["accusative", "singular"]),
             ("διεθνές", "διεθνής", "adj", ["nominative", "singular", "neuter"])]
    par = P.load_paradigms(_write_pairs(tmp_path, rows))
    freq = {"καθηγητή": 20_000, "απλή": 10_641, "απλοί": 1_407,
            "διεθνή": 10_000, "πολύ": 100_000_000}
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (par, freq))
    monkeypatch.setattr(mg, "_VERB_INPUTS", {})
    monkeypatch.setattr(mg, "_nominal_forms", lambda: frozenset())
    monkeypatch.setattr(mg, "MONOTONIC_BUCKETS", "all")
    grc = set(GRC) | {nfc(s) for s in (noun_grc, "ἁπλῆ", "ἁπλοῖ", "διεθνῆ")}
    if modern_in_grc:
        grc |= {nfc("ἁπλή"), nfc("ἁπλοί")}
    counts = _counts([("ἁπλῆ", "lower", {0: 3, 1: 3}),
                      ("ἁπλοῖ", "lower", {0: 3, 1: 3})])
    lst = mg.select_list(counts, {0, 1, 2}, grc, generate_verbs=True)
    assert nfc(noun_new) in lst.generated
    if modern_in_grc:
        assert lst.shared[nfc("ἁπλή")] == "C"
        assert lst.shared[nfc("ἁπλοί")] == "M"
    else:
        assert {nfc("ἁπλή"), nfc("ἁπλοί")} <= lst.generated
    assert nfc("διεθνή") not in lst.generated
    assert lst.shared[nfc(noun_grc)] == "C"
    assert lst.shared[nfc("διεθνῆ")] == "C"
    assert nfc("ἁπλῆ") not in lst.shared
    assert nfc("ἁπλοῖ") not in lst.shared
    attested_only = mg.select_list(counts, {0, 1, 2}, grc, generate_verbs=False)
    assert nfc("ἁπλῆ") not in attested_only.shared
    assert nfc("ἁπλοῖ") not in attested_only.shared


def test_plain_twin_keeps_a_real_contracted_modern_name(monkeypatch):
    par = {"θωμάς": {"Θωμά": _pos("name", "accusative", "singular")}}
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (par, {"θωμά": 27}))
    monkeypatch.setattr(mg, "_VERB_INPUTS", {})
    monkeypatch.setattr(mg, "_nominal_forms", lambda: frozenset({"Θωμά"}))
    shared, generated = mg.dative_twins(
        {nfc("Θωμᾷ"): 6}, {}, {nfc("Θωμᾶ")}, set(), frozenset(), Counter(),
        written={nfc("Θωμᾶ")})
    assert shared == {nfc("Θωμᾶ"): 27}
    assert nfc("Θωμά") not in generated


def test_shared_circumflex_does_not_block_its_own_plain_twin(monkeypatch):
    par = {"γη": {"γη": _pos("noun", "nominative", "singular")}}
    monkeypatch.setattr(mg, "_verb_inputs", lambda: (par, {"γη": 27}))
    monkeypatch.setattr(mg, "_VERB_INPUTS", {})
    monkeypatch.setattr(mg, "_nominal_forms", lambda: frozenset({"γη"}))
    shared, generated = mg.dative_twins(
        {nfc("γῇ"): 6}, {}, {nfc("γῆ")}, set(), frozenset(), Counter(),
        realizations={nfc("γῆ"): frozenset({("γη", "noun nominative singular")})},
        written={nfc("γῆ")})
    assert shared == {nfc("γῆ"): 27}
    assert not generated


@pytest.mark.parametrize("written", ["σκιά", "σκιὰ"])
def test_shared_spelling_blocks_a_conflicting_dative_twin(monkeypatch, written):
    monkeypatch.setattr(mg, "_verb_inputs", lambda: ({}, {
        "σκια": 2_000, "πολύ": 100_000_000}))
    monkeypatch.setattr(mg, "_VERB_INPUTS", {})
    monkeypatch.setattr(mg, "_nominal_forms", lambda: frozenset({"σκια"}))
    monkeypatch.setattr(mg, "_verb_form_keys", lambda: frozenset())
    counts = _counts([("σκιᾷ", "lower", {0: 3, 1: 3}),
                      (written, "lower", {0: 27, 1: 27})])
    grc = {nfc(s) for s in ("σκιά", "σκιὰ", "σκιᾷ", "σκιᾶ")}
    result = mg.select_list(counts, {0, 1, 2}, grc, generate_verbs=True)
    assert nfc("σκιᾶ") not in result.shared
    assert nfc(written) in result.shared
    assert result.report["dative_twin_written_otherwise"] == 1


def test_inferred_nominals_do_not_demote_attested_accents():
    # These are the real shared-rating regressions: the generator's vowel
    # length, stress or nominal homograph is no reason to demote an
    # attested Modern spelling. The final Ancient contraction is distinct.
    alternatives = {
        "γυναῖκα": "γυναίκα", "ἔρωτα": "ἐρωτά", "ναί": "ναῖ",
        "ναὶ": "ναῖ", "μαχητής": "μαχητῆς",
        "χορευτής": "χορευτῆς", "ἁπλῆ": "ἁπλή",
    }
    frequencies = {P.monotonic_key(nfc(s)): 20_000 for s in alternatives}
    frequencies["πολύ"] = 100_000_000
    realizations = {nfc(s): frozenset({("word", "noun nominative singular")})
                    for s in alternatives.values()}
    got = mg.shared_lines(
        {nfc(s): 6 for s in alternatives}, {}, frequencies, set(),
        realizations=realizations)
    assert {nfc(s): got[nfc(s)] for s in alternatives if s != "ἁπλῆ"} == {
        nfc(s): "C" for s in alternatives if s != "ἁπλῆ"}
    assert nfc("ἁπλῆ") not in got


def test_a_final_circumflex_cell_keeps_its_own_rating():
    realizations = {nfc(s): frozenset({("καθηγητής", cell)}) for s, cell in (
        ("καθηγητῆ", "noun genitive singular"),
        ("καθηγητή", "noun accusative singular"))}
    got = mg.shared_lines({nfc("καθηγητῆ"): 6}, {},
                          {"καθηγητή": 20_000, "πολύ": 100_000_000}, set(),
                          realizations=realizations)
    assert got[nfc("καθηγητῆ")] == "C"

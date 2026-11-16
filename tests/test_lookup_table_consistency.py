"""Lookup build: keys a filter emptied, treebank copies in the el table,
Medieval entries spelled like Ancient forms and reviewed readings (#127,
#141).

The builder runs on a small fixture in a temporary data directory; no
production data is read or written.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_lookup_db as B  # noqa: E402
import test_lookup_build_corrections as T  # noqa: E402


# Pure helpers --------------------------------------------------------------

def test_citation_head_refuses_only_a_head_read_as_another_lemma():
    ag = {"σύνοχος": "σύνοχος", "κλεῖδα": "κλείς", "πέδιος": "Πέδιος"}
    el = {"λόγος": "λόγος", "κλείδα": "κλειδί"}
    assert B.citation_head("σύνοχος", ag, el)
    assert B.citation_head("λόγος", ag, el)
    # A name the dictionaries list and the lookup has no row for.
    assert B.citation_head("Βάβυρσα", ag, el)
    assert not B.citation_head("κλεῖδα", ag, el)
    assert not B.citation_head("πέδιος", ag, el)
    assert not B.citation_head("κλείδα", ag, el)
    # A reviewed non-lemma, although no table has a row for it.
    assert "μός" in B.REJECTED_LEMMAS
    assert not B.citation_head("μός", ag, el)


def test_lowercase_twin_readings_need_a_modern_row_and_an_ancient_twin():
    ag = {"πάσας": "πᾶς", "τίνας": "τίς", "ταύτην": "οὗτος",
          "Τούτων": "οὗτος", "τούτων": "οὗτος"}
    el = {"Πάσας": "Πάσας", "Τίνας": "Τίνας", "Τούτων": "Τούτων",
          "Αὕτη": "Αὕτη", "ταύτην": "αυτός"}
    readings = B.lowercase_twin_readings(
        ["Πάσας", "Τίνας", "Ταύτην", "Τούτων", "Αὕτη", "ταύτην"], ag=ag, el=el)
    # Ταύτην: no el row, the lookup-time fallback answers. Τούτων: the
    # Ancient table still reads it. Αὕτη: no Ancient lowercase reading.
    assert readings == {"Πάσας": "πᾶς", "Τίνας": "τίς"}


def test_medieval_readings_held_back_for_attested_ancient_forms():
    med = {"παρέξω": "παρέξω", "παρεξω": "παρέξω", "ἄνδρας": "ἄνδρας",
           "ἀκούεις": "ἀκούεις", "κουβαλῶ": "κουβαλῶ", "δόσις": "δόσιμον",
           "ῥάπτης": "ῥάπτης", "θεωρῶ": "θεωρῶ", "κωκυτός": "κωκυτός"}
    ag = {"παρέξω": "παρέχω", "παρεξω": "παρέχω", "ἄνδρας": "ἀνήρ",
          "ἀκούεις": "ἀκούω", "δόσις": "δόσις", "ῥάπτης": "ῥάπτω",
          "θεωρῶ": "θεωρέω", "κωκυτός": "Κωκυτός"}
    el = {"θεωρώ": "θεωρώ", "άνδρας": "άνδρας", "δόσις": "δόση"}
    evidence = {"παρέξω": {"παρέχω": 45}, "ἄνδρας": {"ἀνήρ": 4039},
                "ἀκούεις": {"ἀκούω": 60}, "ῥάπτης": {"ῥάπτω": 1},
                "θεωρῶ": {"θεωρέω": 30}, "κωκυτός": {"Κωκυτός": 9}}
    modern = {B.modern_word_key(word) for word in ("θεωρώ", "άνδρας")}
    held = B.medieval_readings_held_back(
        med, el=el, ag=ag, evidence=evidence, modern_words=modern)
    # ἄνδρας and θεωρῶ are Modern words, ῥάπτης has one token, κουβαλῶ no
    # Ancient reading, the el table already has δόσις, and κωκυτός is the
    # Ancient lemma's own word.
    assert held == {"παρέξω": "παρέχω", "παρεξω": "παρέχω",
                    "ἀκούεις": "ἀκούω"}


def test_drop_el_corpus_readings_keeps_other_sources_and_agreeing_rows():
    glaux, wikt = B.GLAUX_PAIRS_PATH.name, "mg_lookup.json"
    combined = {"χαίρετ᾽": "χαίρεω", "ᾔει": "εἶμι", "λόγον": "λόγος",
                "μεγάλα": "μέγας"}
    el = {"χαίρετ᾽": "χαίρω", "ᾔει": "ἔρχομαι", "λόγον": "λόγος",
          "μεγάλα": "μεγάλος", "οἳ": "ὅς"}
    sources = {"χαίρετ᾽": glaux, "ᾔει": B.DIORISIS_PAIRS_PATH.name,
               "λόγον": glaux, "μεγάλα": wikt, "οἳ": glaux}
    removed = B.drop_el_corpus_readings(
        combined=combined, el=el, el_sources=sources)
    assert sorted(removed) == [("χαίρετ᾽", "χαίρω", glaux),
                               ("ᾔει", "ἔρχομαι", B.DIORISIS_PAIRS_PATH.name)]
    # An agreeing copy, a Modern reading and a key the combined table lacks
    # (the article filter) stay.
    assert el == {"λόγον": "λόγος", "μεγάλα": "μεγάλος", "οἳ": "ὅς"}
    assert set(sources) == set(el)


def test_listed_head_takes_its_reading_back_from_a_capitalised_twin():
    ag = {"θύτης": "Θύτης", "θυτης": "Θύτης",       # the name page's keys
          "αἴας": "Αἴας",                           # the name is commoner
          "ἠπητής": "Ἠπητής",                       # one token only
          "αὖθι": "αὐτόθι",                         # no twin, base differs
          "Δέρρις": "δέρρις"}                       # a capitalised head
    listed = {"θύτης", "αἴας", "ἠπητής", "αὖθι", "Δέρρις"}
    base = {"θύτης": "Θύτης", "αἴας": "Αἴας", "αὖθι": "αὐτόθι"}
    assert set(B.displaced_head_candidates(ag, set(), listed, base)) == {
        "θύτης", "αἴας", "ἠπητής"}
    evidence = {"θύτης": {"θύτης": 8}, "αἴας": {"αἴας": 5},
                "ἠπητής": {"ἠπητής": 1}, "αὖθι": {"αὖθι": 115},
                "Δέρρις": {"Δέρρις": 3}}
    totals = {"θύτης": 35, "Θύτης": 0, "αἴας": 5, "Αἴας": 737,
              "ἠπητής": 2, "Ἠπητής": 0}
    restored = B.restore_displaced_lexical_heads(
        ag, wiktionary_heads=set(), listed_heads=listed, base=base,
        evidence=evidence, lemma_totals=totals,
        key_evidence={"θυτης": {"θύτης": 8}})
    assert ag["θύτης"] == ag["θυτης"] == "θύτης"
    assert ag["αἴας"] == "Αἴας" and ag["ἠπητής"] == "Ἠπητής"
    assert ag["αὖθι"] == "αὐτόθι" and ag["Δέρρις"] == "δέρρις"
    assert {r["form"] for r in restored} == {"θύτης", "θυτης"}


def test_reviewed_treebank_readings_name_lemmas():
    readings = B.REVIEWED_TREEBANK_READINGS
    assert readings["Νέαν"] == readings["Νέᾳ"] == "νέος"
    assert readings["Νεῶν"] == readings["νεῶν"] == "ναῦς"
    assert readings["Νέαις"] == "Νέαι"
    assert readings["σοι"] == readings["σοί"] == "σύ" and readings["ὥς"] == "ὡς"
    assert "νέη" not in readings.values()
    assert readings["χαίρετ\u1fbd"] == "χαίρω"
    assert readings["οἰκείοις"] == "οἰκεῖος"
    assert all(len(key) > 1 for key in readings)
    assert B.ANCIENT_FORM_LEMMA_READINGS == {"μου": "ἐγώ"}
    assert B.MODERN_READING_OVERRIDES["μου"] == "μου"


# Builds on the fixture -----------------------------------------------------

def test_emptied_keys_are_not_left_to_a_modern_row_or_a_junk_reading(
        tmp_path, monkeypatch):
    # #127. Βάβυρσά and Κέλσου hold expansion readings the #103 check
    # judges: Βαβυρσα is repaired to the listed name, which has no row of
    # its own, and the Latin ξελσυς is dropped, after which the GLAUx pair
    # takes the key (a Diorisis pair does not: λαβῃ stays empty). Πάσας and
    # Τούτος are emptied by the pronoun-twin rule and the reviewed
    # non-lemma record while a Modern name row holds the same key.
    monkeypatch.setitem(T.EXPANSION, "Βάβυρσά", "Βαβυρσα")
    monkeypatch.setitem(T.EXPANSION, "Κέλσου", "ξελσυς")
    glaux = [{"form": "Κέλσου", "lemma": "Κέλσος", "pos": "noun", "count": 176}]
    diorisis = [{"form": "Πάσας", "lemma": "πάσσω", "pos": "verb"},
                {"form": "λαβῃ", "lemma": "λαμβάνω", "pos": "verb"}]
    data, env = T._fixture(tmp_path, {
        "ag": {"πάσας": "πᾶς", "τούτος": "τοῦτος", "Τούτος": "Τοῦτος"},
        "mg_lookup.json": {"λόγος": "λόγος", "Πάσας": "Πάσας",
                           "Τούτος": "Τούτος", "Κέλσου": "Κέλσος"},
        "grc_lexical_heads.json": {
            "heads": ["σύνοχος", "γε", "λόγος", "μοναχικός", "λαβή",
                      "κλεῖδα", "Βάβυρσα"],
            "vernacular_heads": []},
        "ag_headwords.json": ["Κέλσος", "πάσσω", "πᾶς", "λαμβάνω", "λόγος"],
        "glaux_pairs.json": glaux,
        "diorisis_pairs.json": diorisis,
    })
    result = T._build(data, env)
    assert result.returncode == 0, result.stdout + result.stderr
    combined, grc, el = (T._rows(data), T._rows(data, "grc"),
                         T._rows(data, "el"))
    assert combined["Βάβυρσά"] == "Βάβυρσα"
    assert combined["Κέλσου"] == "Κέλσος" and "λαβῃ" not in combined
    assert combined["Πάσας"] == "πᾶς" and el["Πάσας"] == "Πάσας"
    # The Modern name keeps lang="all"; lang="grc" reads the Ancient twin.
    assert combined["Τούτος"] == "Τούτος" and grc["Τούτος"] == "τοῦτος"
    rejected = T._rejections(data)
    layer = B.AG_EXPANSION_LAYER
    assert ("AG", layer, "unaccented_expansion_lemma_unlisted", "Κέλσου",
            "ξελσυς") in rejected
    assert ("AG", layer, "unaccented_expansion_lemma_repaired_to_Βάβυρσα",
            "Βάβυρσά", "Βαβυρσα") in rejected
    assert "1 dropped expansion readings replaced" in result.stdout
    with T.sqlite3.connect(data / "lookup.db") as conn:
        sources = dict(conn.execute(
            "SELECT l.form, s.code FROM lookup l JOIN sources s "
            "ON s.id = l.source_id WHERE l.lang = 'all'"))
    assert sources["Κέλσου"] == "glaux"


def test_el_rows_follow_the_ancient_reading_of_ancient_forms(tmp_path):
    # #141. The GLAUx spelling ἴθ’ folds onto the dictionary key ἴθ᾽,
    # which reads otherwise, and its el copy used to answer lang="el". The Medieval adverb παρέξω took the el row of the future
    # of παρέχω; ἄνδρας, a Modern word, keeps its Medieval row. GLAUx's
    # name lemma Νέη is lowercased to the head νέη, a form of νέος.
    # Wiktionary's περάσῃσ, a truncated form, folds onto περάσῃς; the
    # GLAUx pair for the word itself takes the key.
    glaux = [{"form": "ἴθ’", "lemma": "ἔρχομαι", "pos": "verb", "count": 59},
             {"form": "περάσῃς", "lemma": "περάω", "pos": "verb", "count": 3},
             {"form": "παρέξω", "lemma": "παρέχω", "pos": "verb", "count": 45},
             {"form": "ἄνδρας", "lemma": "ἀνήρ", "pos": "noun", "count": 4039},
             {"form": "Νέαν", "lemma": "Νέη", "pos": "noun", "count": 24}]
    data, env = T._fixture(tmp_path, {
        "ag": {"ἴθ᾽": "εἶμι", "περάσῃσ": "πέρνημι", "κοτέῃσ": "κοτέω",
               "παρέξω": "παρέχω", "παρεξω": "παρέχω",
               "ἄνδρας": "ἀνήρ", "ἐμοῦ": "μου", "μου": "μου", "σοι": "σός"},
        "mg_lookup.json": {"λόγος": "λόγος", "άνδρας": "άνδρας", "μου": "μου"},
        "med_lookup.json": {"παρέξω": "παρέξω", "παρεξω": "παρέξω",
                            "ἄνδρας": "ἄνδρας"},
        "ag_headwords.json": ["ἔρχομαι", "παρέχω", "ἀνήρ", "νέη", "νέος",
                              "λόγος", "περάω"],
        "glaux_pairs.json": glaux,
    })
    result = T._build(data, env)
    assert result.returncode == 0, result.stdout + result.stderr
    combined, grc, el = (T._rows(data), T._rows(data, "grc"),
                         T._rows(data, "el"))
    assert combined["ἴθ᾽"] == "εἶμι" and "ἴθ᾽" not in el
    assert combined["περάσῃς"] == "περάω" and "περάσῃς" not in el
    assert combined["κοτέῃς"] == "κοτέω"   # no key of its own: kept
    assert ("AG", "ag_lookup.json", "final_medial_sigma_key", "περάσῃς",
            "πέρνημι") in T._rejections(data)
    assert combined["παρέξω"] == combined["παρεξω"] == "παρέχω"
    assert "παρέξω" not in el and "παρεξω" not in el
    assert combined["ἄνδρας"] == "ἀνήρ" and el["ἄνδρας"] == "ἄνδρας"
    assert ("EL", "med_lookup.json",
            "medieval_reading_of_ancient_form_παρέχω", "παρέξω",
            "παρέξω") in T._rejections(data)
    assert combined["Νέαν"] == "νέος"
    assert combined["ἐμοῦ"] == "ἐγώ" and combined["σοι"] == "σύ"
    # The Modern weak pronoun keeps the combined row; lang="grc" reads ἐγώ.
    assert combined["μου"] == "μου" and grc["μου"] == "ἐγώ"

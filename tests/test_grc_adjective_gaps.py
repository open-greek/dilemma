"""Ancient -ικός adjectives declined as masculine nouns regain their feminines,
and s-stem adjectives in -ής with Ionic tables gain their contracted cells
(#131)."""
import json
from pathlib import Path
import sys

import pytest

from dilemma.grc_adjective_gaps import (
    ES_KIND, SOURCE, complete_ikos_adjectives, es_cells, es_paradigms,
    ikos_cells, select_es_adjectives)
from dilemma.grc_noun_gaps import fill_gap_cells

ROOT = Path(__file__).resolve().parent.parent


def test_ikos_cells_decline_the_feminine_and_neuter_plural():
    forms = [form for form, _ in ikos_cells("μοναχικός")]
    assert forms == ["μοναχική", "μοναχικῆς", "μοναχικῇ", "μοναχικήν", "μοναχικαί",
                     "μοναχικῶν", "μοναχικαῖς", "μοναχικάς", "μοναχικά", "μοναχικαῖν"]
    with pytest.raises(ValueError):
        ikos_cells("οἶκος")


def test_completion_fills_absent_keys_only_and_needs_a_head():
    lookup = {"μοναχικός": "μοναχικός", "μοναχικῶν": "μοναχικός",
              "μοναχικά": "μοναχικά",          # an existing reading stays
              "ἀγγελτικοῦ": "ἀγγελτικός"}      # no self-mapped head
    added = complete_ikos_adjectives(lookup, ["μοναχικός", "ἀγγελτικός"])
    assert lookup["μοναχικήν"] == lookup["μοναχικην"] == "μοναχικός"
    assert lookup["μοναχικά"] == "μοναχικά"
    assert lookup["μοναχικα"] == "μοναχικός"
    assert "ἀγγελτική" not in lookup
    assert {row["lemma"] for row in added} == {"μοναχικός"}
    assert "μοναχικῶν" not in {row["form"] for row in added}


def test_capitalized_adjective_is_keyed_like_the_capitalized_heads():
    lookup = {"Πλατωνικός": "Πλατωνικός"}
    complete_ikos_adjectives(lookup, ["Πλατωνικός"])
    assert (lookup["Πλατωνικήν"] == lookup["πλατωνικήν"] == lookup["πλατωνικην"]
            == "Πλατωνικός")
    assert "Πλατωνικην" not in lookup


def test_sophocles_parser_reads_ocr_adjective_endings():
    sys.path.insert(0, str(ROOT / "build"))
    import expand_sophocles as S
    assert S.adjective_endings(", ή, όν, announcing") == "three"
    assert S.adjective_endings(", ἦ, ov, (ἀδικέω)") == "three"
    assert S.adjective_endings(", ή, 6v, (ἄβυσσος)") == "three"
    assert S.adjective_endings(", ov, unjust") == "two"
    assert S.adjective_endings(", ἡ, the deep") is None
    assert S.adjective_endings(", οῦ, ὁ, a monk") is None


def test_finder_lists_supported_ikos_adjectives(tmp_path):
    sys.path.insert(0, str(ROOT / "build"))
    import find_grc_ikos_adjectives as F
    data = tmp_path
    (data / "ag_lookup.json").write_text(json.dumps({
        "μοναχικός": "μοναχικός", "μοναχικοῦ": "μοναχικός",       # Sophocles: three endings
        "ἀγγελτικός": "ἀγγελτικός",                               # corpus: adjective
        "βολιστικός": "βολιστικός", "βολιστική": "βολιστικός",    # already has a feminine
        "δαρεικός": "δαρεικός",                                   # a noun: no evidence
        "*σχολαστικός": "*σχολαστικός",                           # OCR debris
        "ἄδικος": "ἄδικος",                                       # not -ικός oxytone
        "ἀνικός": "ἀνικός",                                       # Sophocles: two endings
        "φρουρικός": "φρουρικός",                                 # review: adjective
        "Πλατωνικός": "Πλατωνικός",                               # review: adjective
        "Γρανικός": "Γρανικός",                                   # capitalized, unreviewed
        "παθικός": "παθικός",                                     # review: exclude
    }, ensure_ascii=False))
    (data / "lsj9_pos_lookup.json").write_text("{}")
    (data / "lemma_attestation.json").write_text(json.dumps({"lemmas": {
        "ἀγγελτικός": {"dominant_pos": "adj"}, "ἀνικός": {"dominant_pos": "adj"},
        "παθικός": {"dominant_pos": "adj"}, "Γρανικός": {"dominant_pos": "adj"},
        "δαρεικός": {"dominant_pos": "noun"}}}, ensure_ascii=False))
    chunks = data / "sophocles" / "chunks"
    chunks.mkdir(parents=True)
    (chunks / "0001.xml").write_text(
        "<root><entry><orth>μοναχικός</orth>, ή, όν, monastic.</entry>"
        "<entry><orth>ἀνικός</orth>, όν, x.</entry></root>", encoding="utf-8")
    (data / "grc_ikos_review.json").write_text(json.dumps({"entries": [
        {"lemma": "φρουρικός", "decision": "adjective", "reason": "x"},
        {"lemma": "Πλατωνικός", "decision": "adjective", "reason": "x"},
        {"lemma": "παθικός", "decision": "exclude", "reason": "x"},
    ]}, ensure_ascii=False))
    import expand_sophocles
    chunk_dir = expand_sophocles.SOPH_DIR
    record = F.find(data)
    # the finder leaves the generator's chunk directory as it found it
    assert expand_sophocles.SOPH_DIR == chunk_dir
    assert sorted(record["lemmas"], key=lambda r: r["lemma"]) == sorted([
        {"lemma": "μοναχικός", "evidence": ["sophocles"]},
        {"lemma": "ἀγγελτικός", "evidence": ["corpus"]},
        {"lemma": "φρουρικός", "evidence": ["review"]},
        {"lemma": "Πλατωνικός", "evidence": ["review"]},
    ], key=lambda r: r["lemma"])
    assert len(record["inputs"]) == 5


def test_every_review_entry_has_a_decision_and_a_reason():
    review = json.loads((ROOT / "data" / "grc_ikos_review.json").read_text(encoding="utf-8"))
    lemmas = [entry["lemma"] for entry in review["entries"]]
    assert len(lemmas) == len(set(lemmas))
    for entry in review["entries"]:
        assert entry["decision"] in ("adjective", "exclude") and entry["reason"]
    listed = {row["lemma"]: row["evidence"] for row in json.loads(
        (ROOT / "data" / "grc_ikos_adjectives.json").read_text(encoding="utf-8"))["lemmas"]}
    for entry in review["entries"]:
        assert (entry["lemma"] in listed) == (entry["decision"] == "adjective")


# s-stem adjectives in -ής ---------------------------------------------------

def test_es_cells_are_the_attic_contracted_cells():
    assert [form for form, _ in es_cells("δυσπενθής")] == [
        "δυσπενθοῦς", "δυσπενθεῖ", "δυσπενθῆ", "δυσπενθές", "δυσπενθεῖς",
        "δυσπενθῶν", "δυσπενθέσι", "δυσπενθέσιν"]
    assert not any("dual" in cell or cell.startswith("vocative")
                   for _, cell in es_cells("δυσπενθής"))
    # After a vowel the accusative is -ᾶ as often as -ῆ: that cell is left out.
    assert [form for form, _ in es_cells("ὑγιής")] == [
        "ὑγιοῦς", "ὑγιεῖ", "ὑγιές", "ὑγιεῖς", "ὑγιῶν", "ὑγιέσι", "ὑγιέσιν"]
    assert "ἐνδεῆ" not in dict(es_cells("ἐνδεής"))
    # Other words are not declined, nor a barytone accented further back.
    assert es_cells("ἔμφλογος") == [] and es_cells("ἄκρατης") == []


def test_barytone_es_cells_keep_the_stem_accent():
    # σύνηθες recedes and εὐῶδες does not: the neuter singular is left out.
    # The genitive plural is given both ways (συνήθων, συνηθῶν).
    assert [form for form, _ in es_cells("συνήθης")] == [
        "συνήθους", "συνήθει", "συνήθη", "συνήθεις", "συνήθων", "συνήθεσι",
        "συνήθεσιν", "συνηθῶν"]
    assert [form for form, _ in es_cells("βληχηματώδης")][:4] == [
        "βληχηματώδους", "βληχηματώδει", "βληχηματώδη", "βληχηματώδεις"]
    assert not any("neuter nominative/accusative/vocative singular" in cell
                   for _, cell in es_cells("συνήθης"))


def test_listed_es_adjectives_need_no_genitive_row():
    rows = [
        ("βληχηματώδης", "βληχηματώδης", "lbg_headwords"),
        ("ἀκαμπής", "ἀκαμπής", "lsj"),
        ("εὐπαθής", "Εὐπαθής", "wiktionary_grc"),         # the name's key
        ("ἐγκρατής", "ἐγκρατής", "wiktionary_grc"),        # not listed
        # Listed, but a dictionary table declines it in the first declension.
        ("πολίτης", "πολίτης", "wiktionary_grc"),
        ("πολίτου", "πολίτης", "wiktionary_grc"),
        ("πολίτην", "πολίτης", "wiktionary_grc"),
    ]
    listed = ["βληχηματώδης", "ἀκαμπής", "εὐπαθής", "πολίτης", "ἀβλαβής"]
    assert select_es_adjectives(rows, listed=listed) == sorted([
        "ἀκαμπής", "βληχηματώδης", "εὐπαθής"])
    assert select_es_adjectives(rows) == []


def test_es_finder_reads_the_lexica(tmp_path):
    sys.path.insert(0, str(ROOT / "build"))
    import find_grc_es_adjectives as F
    assert F.head_text_class("ες, gen. εος, contr. ους") == "es"
    assert F.head_text_class("[ᾰ], ές, true") == "es"
    assert F.head_text_class("ὁ, a poet") == "noun"
    lsj9 = tmp_path / "lsj9"
    lsj9.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    (lsj9 / "lsj9_headwords.json").write_text(json.dumps([
        {"headword": "ἀληθής", "grammar": "ές"},
        {"headword": "ποιητής", "grammar": "ὁ"},
        {"headword": "συνήθης"}], ensure_ascii=False), encoding="utf-8")
    (lsj9 / "lsj9_glosses.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in [
            {"headword": "συνήθης", "text": "ες, gen. εος, contr. ους"},
            {"headword": "ποιητής", "text": "οῦ, ὁ, maker"}]), encoding="utf-8")
    (lsj9 / "lsj9_headword_pos.json").write_text(json.dumps(
        {"ἀκαμπής": "ADJ", "ποιητής": "NOUN"}, ensure_ascii=False),
        encoding="utf-8")
    (data / "lbg_headwords.json").write_text(json.dumps([
        {"lemma": "βληχηματώδης"}, {"lemma": "γυμνίτης"},
        {"lemma": "ἐργάτης", "gender": "ὁ"}], ensure_ascii=False),
        encoding="utf-8")
    sophocles = {"θεοειδής": {"adjective_class": "es"},
                 "ἀληθής": {"adjective_class": None, "gender_stated": True}}
    (data / "sophocles" / "chunks").mkdir(parents=True)
    (data / "sophocles_head_corrections.json").write_text("{}")
    record = F.find(lsj9, data, sophocles)
    assert record["lemmas"] == {
        "lbg": ["βληχηματώδης"], "lsj9": sorted(["ἀκαμπής", "συνήθης"]),
        "sophocles": ["θεοειδής"]}
    assert record["counts"] == {"oxytone": 2, "barytone": 2}
    assert "sophocles/chunks" in record["inputs"]


def test_committed_es_list():
    from dilemma.grc_adjective_gaps import read_es_adjectives
    record = json.loads((ROOT / "data" / "grc_es_adjectives.json").read_text(
        encoding="utf-8"))
    lemmas = read_es_adjectives(record)
    assert len(lemmas) == sum(record["counts"].values()) > 5000
    assert {"βληχηματώδης", "συνήθης", "ἀληθής"} <= set(lemmas)
    assert all(es_cells(lemma) for lemma in lemmas)


def test_sophocles_adjective_class():
    sys.path.insert(0, str(ROOT / "build"))
    import expand_sophocles as S
    assert S.adjective_class(", ἡ, όν, belonging to") == "adjective_eta"
    assert S.adjective_class(", ά, όν, x") == "adjective_alpha"
    assert S.adjective_class(", ον, (τηρέω) divinely") == "adjective"
    assert S.adjective_class(", ες, --- βληχώδης") == "es"
    # The OCR's ov for a noun's genitive ου, before the article.
    assert S.adjective_class(", ov, 6, L. cursus") is None
    assert S.adjective_class(", ον, ο, frog") is None


def test_es_selection_needs_an_ancient_genitive_in_eos():
    rows = [
        ("δυσπενθής", "δυσπενθής", "vlg_headwords"),
        ("δυσπενθέος", "δυσπενθής", "grc_lexicon_expansion"),
        # A corpus pair is evidence too: the genitive names the declension.
        ("ὁμοζυγέος", "ὁμοζυγής", "diorisis"),
        # A first-declension masculine keeps its own cells.
        ("ποιητοῦ", "ποιητής", "wiktionary_grc"),
        ("ποιητήν", "ποιητής", "wiktionary_grc"),
        ("ποιητέος", "ποιητής", "glaux"),
        # A Modern Greek row, an accentless row and a capitalized row are no
        # evidence; a name and a lemma without its breathing are not declined.
        ("συνεχέος", "συνεχής", "wiktionary_el"),
        ("σαφεος", "σαφής", "grc_lexicon_expansion"),
        ("Σαφέος", "σαφής", "glaux"),
        ("Διογενέος", "Διογενής", "glaux"),
        ("ευτυχέος", "ευτυχής", "grc_lexicon_expansion"),
        # A row without a source counts as an Ancient dictionary row.
        ("εὐπειθέος", "εὐπειθής"),
    ]
    assert select_es_adjectives(rows) == ["δυσπενθής", "εὐπειθής", "ὁμοζυγής"]
    assert select_es_adjectives(reversed(rows)) == select_es_adjectives(rows)


def test_es_cells_fill_absent_keys_and_keep_other_readings():
    lookup = {"δυσπενθής": "δυσπενθής", "δυσπενθέος": "δυσπενθής",
              "δυσπενθῶν": "δυσπενθέω", "δυσπενθων": "δυσπενθέω",
              "δυσπενθει": "δυσπενθέω"}
    added = fill_gap_cells(lookup, es_paradigms(["δυσπενθής"]), strict={ES_KIND})
    assert lookup["δυσπενθοῦς"] == lookup["δυσπενθεῖς"] == "δυσπενθής"
    assert lookup["δυσπενθές"] == "δυσπενθής"
    # The accented key only.
    assert "δυσπενθους" not in lookup and "δυσπενθες" not in lookup
    # The participle keeps δυσπενθῶν, and the dative that is the verb's form
    # but for its accent is not added.
    assert lookup["δυσπενθῶν"] == "δυσπενθέω"
    assert "δυσπενθεῖ" not in lookup and lookup["δυσπενθει"] == "δυσπενθέω"
    assert {row["kind"] for row in added} == {ES_KIND}
    assert SOURCE == "grc_adjective_gaps"

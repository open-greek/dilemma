"""Source-sense corrections cannot rewrite unrelated grammatical readings."""
import copy
import json

import pytest

import build_data as B
from dilemma.el_source_tenses import correct_sense_tense, restore_sense_tags
from scripts.repair_el_source_tenses import repair, undo


def source_sense():
    return {"glosses": ["first-person singular imperfect of άγομαι (ágomai)"],
            "tags": ["first-person", "form-of", "imperfect", "singular"],
            "form_of": [{"word": "άγομαι"}]}


def test_real_form_page_is_corrected_before_extraction_and_keeps_original_evidence(tmp_path):
    entry = {"word": "άχθηκα", "pos": "verb", "senses": [source_sense()]}
    original = copy.deepcopy(entry)
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps(entry, ensure_ascii=False) + "\n")
    audit = []
    pairs, *_ = B.extract_pairs(path, "el-en", source_tense_corrections=audit)
    assert pairs == [{"form": "άχθηκα", "lemma": "άγομαι", "pos": "verb",
                      "tags": ["aorist", "first-person", "form-of", "singular"]}]
    assert entry == original
    assert audit[0]["original"]["source_record"] == original["senses"][0]
    assert audit[0]["reviewed_at"] and len(audit[0]["evidence"]) == 2
    assert restore_sense_tags(pairs[0]["tags"], audit[0]) == sorted(source_sense()["tags"])
    with pytest.raises(ValueError, match="changed after correction"):
        restore_sense_tags(["present"], audit[0])


@pytest.mark.parametrize("form,edition,target,replace_tag", [
    ("άχθηκες", "en", "άγομαι", None),
    ("άχθηκα", "el", "άγομαι", None),
    ("άχθηκα", "en", "άχθομαι", None),
    ("άχθηκα", "en", "άγομαι", "aorist"),
    ("άχθηκα", "en", "άγομαι", "present"),
])
def test_unreviewed_contexts_and_already_correct_sources_remain_unchanged(form, edition, target, replace_tag):
    sense = source_sense()
    sense["form_of"] = [{"word": target}]
    tags = [replace_tag if t == "imperfect" and replace_tag else t for t in sense["tags"]]
    result, audit = correct_sense_tense(form, edition, sense, tags)
    assert result == tags and audit is None


def test_ancient_extraction_does_not_use_the_modern_correction(tmp_path):
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps({"word": "άχθηκα", "pos": "verb", "senses": [source_sense()]}) + "\n")
    audit = []
    pairs, *_ = B.extract_pairs(path, "grc-en", source_tense_corrections=audit)
    assert not audit and all("aorist" not in r["tags"] for r in pairs)


def test_shared_imperfect_and_aorist_readings_of_paresyra_are_retained(tmp_path):
    senses = [{"form_of": [{"word": "παρασύρω"}],
               "tags": ["active", "first-person", "form-of", tense, "singular"]}
              for tense in ("imperfect", "past")]
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps({"word": "παρέσυρα", "pos": "verb", "senses": senses}) + "\n")
    audit = []
    pairs, *_ = B.extract_pairs(path, "el-en", source_tense_corrections=audit)
    assert not audit and len(pairs) == 2
    assert {frozenset(r["tags"]) & {"imperfect", "past"} for r in pairs} == {
        frozenset({"imperfect"}), frozenset({"past"})}


def test_snapshot_repair_preserves_all_other_rows_and_undo_refuses_later_edits():
    rows = [{"form": "άχθηκα", "lemma": "άγω", "pos": "verb", "tags": source_sense()["tags"]},
            {"form": "άχθηκα", "lemma": "άγω", "pos": "verb", "tags": ["past", "perfective", "passive"]},
            {"form": "παρέσυρα", "lemma": "παρασύρω", "pos": "verb", "tags": ["imperfect"]},
            {"form": "παρέσυρα", "lemma": "παρασύρω", "pos": "verb", "tags": ["aorist"]},
            {"form": "άχθηκα", "lemma": "άχθομαι", "pos": "verb", "tags": source_sense()["tags"]}]
    original = copy.deepcopy(rows)
    evidence = {"source_contexts": [{"edition": "en", "form": "άχθηκα", "sense": source_sense(),
                                     "tags": source_sense()["tags"]}]}
    fixed, audit = repair(rows, evidence)
    assert rows == original and fixed[1:] == rows[1:]
    assert len(audit["pair_changes"]) == 1 and "aorist" in fixed[0]["tags"]
    assert undo(fixed, audit) == original
    replayed, _ = repair(fixed, evidence)
    assert replayed == fixed
    fixed[0] = {**fixed[0], "tags": ["present"]}
    with pytest.raises(ValueError, match="changed after source correction"):
        undo(fixed, audit)


def test_published_snapshot_has_no_imperfect_achthika():
    path = B.DATA_DIR / "mg_pairs.json"
    if not path.exists():
        pytest.skip("Published Modern Greek data is not installed")
    rows = [r for r in json.loads(path.read_text()) if r["form"] == "άχθηκα" and r.get("pos") == "verb"]
    assert rows and all("imperfect" not in r.get("tags", []) for r in rows)
    assert any("aorist" in r.get("tags", []) for r in rows)

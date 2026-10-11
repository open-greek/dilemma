"""Periphrastic table auxiliaries must not train unrelated verb lemmas."""
import json

import pytest

from build_data import extract_pairs


def extract(tmp_path, entry, lang="grc-en"):
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps(entry, ensure_ascii=False) + "\n")
    return extract_pairs(path, lang)


def entry(word, form, tags, pos="verb"):
    return {"word": word, "pos": pos,
            "senses": [{"glosses": ["An independent word."]}],
            "forms": [{"form": form, "tags": tags}]}


@pytest.mark.parametrize("lang", ["grc-en", "grc-el"])
@pytest.mark.parametrize("form", ["εἶτε", "είτε", "είμεν", "είεν"])
def test_auxiliary_table_claim_never_reaches_pairs_or_lookup(tmp_path, lang, form):
    pairs, lookup, _, pos_map, *_ = extract(
        tmp_path, entry("τύπτω", form,
                        ["middle", "perfect", "optative", "second-person", "plural"]), lang)
    assert not any(row["form"] in {form, "είτε"} and row["lemma"] == "τύπτω"
                   for row in pairs)
    assert form not in lookup and "είτε" not in lookup
    assert form not in pos_map and "είτε" not in pos_map
    assert lookup["τύπτω"][0] == "τύπτω"


@pytest.mark.parametrize("word,form,tags", [
    ("ἵημι", "εἷμεν", ["active", "aorist", "optative", "first-person", "plural"]),
    ("ἵημι", "εἷτε", ["active", "aorist", "optative", "second-person", "plural"]),
    ("ἵστημι", "ἑστῶ", ["active", "perfect", "imperative", "third-person", "singular"]),
    ("ἵστημι", "ἔστω", ["middle", "aorist", "indicative", "second-person", "singular"]),
    ("εἰμί", "εἶτε", ["active", "present", "optative", "second-person", "plural"]),
    ("εἶμι", "είτε", ["active", "perfect", "optative", "second-person", "plural"]),
])
def test_independent_forms_and_their_monotonic_copies_survive(tmp_path, word, form, tags):
    from build_data import to_monotonic
    pairs, lookup, *_ = extract(tmp_path, entry(word, form, tags))
    for spelling in {form, to_monotonic(form)}:
        assert any(row["form"] == spelling and row["lemma"] == word for row in pairs)
        assert lookup[spelling][0] == word


@pytest.mark.parametrize("lang,pos", [("el-en", "verb"), ("mgr-el", "verb"),
                                      ("grc-en", "conj")])
def test_ancient_verb_filter_does_not_change_other_languages_or_pos(tmp_path, lang, pos):
    pairs, lookup, *_ = extract(tmp_path, entry(
        "τύπτω", "είτε", ["perfect", "optative"], pos), lang)
    assert any(row["form"] == "είτε" for row in pairs)
    assert "είτε" in lookup


def test_inferred_table_tense_is_available_before_auxiliary_check(tmp_path):
    data = entry("τύπτω", "είτε", ["middle", "optative", "second-person", "plural"])
    data["forms"].insert(0, {"form": "Perfect", "tags": ["table-tags"]})
    pairs, lookup, *_ = extract(tmp_path, data)
    assert not any(row["form"] == "είτε" for row in pairs)
    assert "είτε" not in lookup

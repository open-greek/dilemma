"""Same-cell evidence for quotation marks before plain sentence punctuation."""

import xml.etree.ElementTree as ET

import pytest

from build.diorisis_source import DiorisisSourceContext, restore_source_beta


MORPHS = ("imperf ind act 3rd pl (doric aeolic)",
          "imperf ind act 1st sg (doric aeolic)",
          "imperf ind act 3rd pl (homeric ionic)",
          "imperf ind act 1st sg (homeric ionic)")


def word(beta, *, morphs=MORPHS, lemma="ἐπαίω", pos="verb", wid="4"):
    result = ET.Element("word", {"form": beta, "id": wid})
    entry = ET.SubElement(result, "lemma", {"entry": lemma, "POS": pos})
    for morph in morphs:
        ET.SubElement(entry, "analysis", {"morph": morph})
    return result


def source(*, continuation="petali/das", punct=".", witness=True):
    root = ET.Element("text")
    quoted = ET.SubElement(root, "sentence", {"id": "11452", "location": "9.18"})
    token = word("e)pa/ion'")
    quoted.append(token)
    ET.SubElement(quoted, "punct", {"mark": punct})
    if continuation is not None:
        following = ET.SubElement(root, "sentence", {"id": "11453"})
        following.append(word(continuation, lemma="πεταλίς", pos="noun", morphs=("fem acc pl",)))
    if witness:
        other = ET.SubElement(root, "sentence", {"id": "13227", "location": "10.35"})
        other.append(word("e)pa/ion", wid="3"))
    return root, token


def test_complete_athenaeus_cell_resolves_terminal_quote_reversibly():
    root, token = source()
    context = DiorisisSourceContext(root)
    assert context.beta(token) == "e)pa/ion"
    assert token.get("form") == "e)pa/ion'"
    row = context.audit[0]
    assert row["reason"] == "terminal_quote_with_same_source_cell"
    assert row["quote_boundary"] == "plain_sentence_punctuation"
    assert row["following_beta"] == "petali/das"
    assert len(row["evidence"]) == 4
    assert row["evidence"][0]["witnesses"][0]["location"] == "10.35"
    assert restore_source_beta(context.beta(token), row) == "e)pa/ion'"


@pytest.mark.parametrize("continuation", ["ei)=nai", "*)/andres", None, "123",
                                        "123j", "j", "!petali/das"])
def test_vowel_or_missing_continuation_cannot_resolve_elision(continuation):
    root, token = source(continuation=continuation, witness=False)
    if continuation is not None:
        root.append(ET.Element("sentence"))
        root[-1].append(word("e)pa/ion", wid="3"))
    context = DiorisisSourceContext(root)
    assert context.beta(token) == "e)pa/ion'"
    assert context.audit == []


def test_legitimate_aristotle_elision_before_vowel_is_unchanged():
    root = ET.fromstring("<text><sentence id='1486'/></text>")
    sentence = root[0]
    token = word("w)/iet'", lemma="οἴομαι", morphs=("imperf ind mp 3rd sg",))
    sentence.append(token)
    sentence.append(word("ei)=nai", lemma="εἰμί", morphs=("pres inf act",)))
    ET.SubElement(sentence, "punct", {"mark": ","})
    context = DiorisisSourceContext(root)
    assert context.beta(token) == "w)/iet'"
    assert context.audit == []


@pytest.mark.parametrize("punct", [",", "—", ".>\n", "”"])
def test_other_punctuation_does_not_expand_the_rule(punct):
    root, token = source(punct=punct)
    context = DiorisisSourceContext(root)
    assert context.beta(token) == "e)pa/ion'"
    assert context.audit == []


def test_no_witness_preserves_and_audits_the_uncertain_reading():
    root, token = source(witness=False)
    context = DiorisisSourceContext(root)
    assert context.beta(token) == "e)pa/ion'"
    assert context.audit[0]["reason"] == "unresolved_terminal_quote_or_elision"


def test_every_analysis_must_have_a_complete_witness():
    root, token = source()
    root[-1][0][0].remove(root[-1][0][0][-1])
    context = DiorisisSourceContext(root)
    assert context.beta(token) == "e)pa/ion'"
    assert len(context.audit[0]["unresolved_readings"]) == 1


def test_internal_clause_punctuation_does_not_resolve_a_quote():
    root, token = source()
    root[0].append(word("petali/das", lemma="πεταλίς", pos="noun"))
    context = DiorisisSourceContext(root)
    assert context.beta(token) == "e)pa/ion'"
    assert context.audit == []


def test_quotation_candidates_cannot_corroborate_each_other():
    root, token = source(witness=False)
    other = ET.SubElement(root, "sentence", {"id": "2"})
    other.append(word("e)pa/ion'"))
    ET.SubElement(other, "punct", {"mark": "."})
    following = ET.SubElement(root, "sentence", {"id": "3"})
    following.append(word("petali/das", pos="noun"))
    context = DiorisisSourceContext(root)
    assert all(row["normalized_beta"] == "e)pa/ion'" for row in context.audit)


def test_pair_builder_preserves_analyses_and_genuine_elision(tmp_path):
    pytest.importorskip("beta_code")
    from build.build_diorisis_pairs import extract_diorisis

    root, _token = source()
    sentence = ET.SubElement(root, "sentence", {"id": "1486"})
    sentence.append(word("w)/iet'", lemma="οἴομαι", morphs=("imperf ind mp 3rd sg",)))
    sentence.append(word("ei)=nai", lemma="εἰμί", morphs=("pres inf act",)))
    (tmp_path / "sample.xml").write_text(ET.tostring(root, encoding="unicode"))
    audit = []
    pairs = extract_diorisis(tmp_path, token_audit=audit)
    assert "ἐπάιον᾽" not in {row["form"] for row in pairs}
    complete = [row for row in pairs if row["form"] == "ἐπάιον"]
    assert len(complete) == 2
    assert {tuple(row["tags"]) for row in complete} == {
        ("plural", "imperfect", "indicative", "active", "third-person"),
        ("singular", "imperfect", "indicative", "active", "first-person"),
    }
    assert any(row["form"] == "ὤιετ᾽" for row in pairs)
    record = audit[0]["records"][0]
    assert record["reason"] == "terminal_quote_with_same_source_cell"
    assert len(record["readings"]) == 4
    assert len(audit[0]["source_sha256"]) == 64

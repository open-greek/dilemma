"""Context, exact source evidence, and reversible Diorisis quote handling."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from build.diorisis_source import DiorisisSourceContext, restore_source_beta


def _word(beta, entry="Βάκχος", pos="proper", morph="masc gen sg", wid="1"):
    return (f'<word form="{beta}" id="{wid}"><lemma entry="{entry}" POS="{pos}">'
            f'<analysis morph="{morph}"/></lemma></word>')


def _root(witness, quoted, punctuation=".’"):
    return ET.fromstring(
        f'<text><sentence id="1" location="1">{witness}</sentence>'
        f'<sentence id="2" location="2">{quoted}<punct mark="{punctuation}"/>'
        '</sentence></text>')


@pytest.mark.parametrize("beta,morph", [
    ("*ba/kxou", "masc gen sg"),
    ("*ba/kxw|", "masc dat sg"),
])
def test_duplicate_closing_quote_requires_the_same_complete_cell(beta, morph):
    root = _root(_word(beta, morph=morph), _word(beta + "'", morph=morph))
    context = DiorisisSourceContext(root)
    quoted = root.findall(".//word")[-1]
    assert context.beta(quoted) == beta
    assert quoted.get("form") == beta + "'"  # the original XML is unchanged
    row = context.audit[0]
    assert row["original_beta"] == beta + "'"
    assert row["normalized_beta"] == beta
    assert row["following_punctuation"] == ".’"
    assert row["evidence"][0]["witnesses"][0]["sentence_id"] == "1"
    assert row["evidence"][0]["witnesses"][0]["original_beta"] == beta
    assert restore_source_beta(context.beta(quoted), row) == quoted.get("form")
    with pytest.raises(ValueError):
        restore_source_beta("different", row)


def test_elision_immediately_before_a_closing_quote_is_preserved():
    root = _root(_word("par'", "παρά", "preposition", "indeclform (prep)"),
                 _word("par'", "παρά", "preposition", "indeclform (prep)"))
    context = DiorisisSourceContext(root)
    assert context.beta(root.findall(".//word")[-1]) == "par'"
    assert context.audit[0]["reason"] == "unresolved_terminal_quote_or_elision"


def test_a_duplicate_outer_quote_can_be_removed_without_losing_real_elision():
    root = _root(_word("par'", "παρά", "preposition", "indeclform (prep)"),
                 _word("par''", "παρά", "preposition", "indeclform (prep)"))
    context = DiorisisSourceContext(root)
    assert context.beta(root.findall(".//word")[-1]) == "par'"


@pytest.mark.parametrize("witness", [
    _word("*ba/kxou", "ἄλλος"),
    _word("*ba/kxou", morph="masc dat sg"),
    _word("*ba/kxou", pos="verb"),
    _word("ba/kxou"),  # keep case-preserving exact evidence
    '<word form="*ba/kxou"><lemma entry="Βάκχος" POS="proper"/></word>',
])
def test_other_owner_cell_case_or_missing_morphology_does_not_license_stripping(witness):
    root = _root(witness, _word("*ba/kxou'"))
    context = DiorisisSourceContext(root)
    assert context.beta(root.findall(".//word")[-1]) == "*ba/kxou'"
    assert not context.audit[0]["evidence"]


def test_quote_punctuation_alone_and_quotation_candidates_cannot_supply_evidence():
    root = _root(_word("*ba/kxou'") + '<punct mark=".’"/>', _word("*ba/kxou'"))
    context = DiorisisSourceContext(root)
    assert all(context.beta(w) == "*ba/kxou'" for w in root.findall(".//word"))


def test_a_supported_reading_does_not_erase_an_unsupported_alternative():
    quoted = _word("*ba/kxou'").replace(
        '</lemma>', '<analysis morph="masc dat sg"/></lemma>')
    root = _root(_word("*ba/kxou"), quoted)
    context = DiorisisSourceContext(root)
    assert context.beta(root.findall(".//word")[-1]) == "*ba/kxou'"
    assert context.audit[0]["evidence"]
    assert context.audit[0]["unresolved_readings"] == [
        {"lemma": "Βάκχος", "pos": "proper", "morph": "masc dat sg"}]


def test_each_distinct_lemma_reading_needs_complete_cell_evidence():
    quoted = _word("*ba/kxou'").replace(
        '</word>', '<lemma entry="ἄλλος" POS="proper">'
        '<analysis morph="masc gen sg"/></lemma></word>')
    root = _root(_word("*ba/kxou"), quoted)
    context = DiorisisSourceContext(root)
    assert context.beta(root.findall(".//word")[-1]) == "*ba/kxou'"
    # Supplying the alternative cell makes the decision order-independent.
    root.find('sentence').append(ET.fromstring(_word("*ba/kxou", "ἄλλος")))
    context = DiorisisSourceContext(root)
    assert context.beta(root.findall(".//word")[-1]) == "*ba/kxou"


@pytest.mark.parametrize("alternative", [
    '<lemma/>',
    '<lemma entry="ἄλλος" POS="proper"/>',
    '<lemma entry="Βάκχος"><analysis morph="masc gen sg"/></lemma>',
    '<lemma POS="proper"><analysis morph="masc gen sg"/></lemma>',
])
def test_incomplete_alternatives_are_recorded_and_block_global_normalization(alternative):
    quoted = _word("*ba/kxou'").replace('</word>', alternative + '</word>')
    root = _root(_word("*ba/kxou"), quoted)
    context = DiorisisSourceContext(root)
    assert context.beta(root.findall(".//word")[-1]) == "*ba/kxou'"
    assert context.audit[0]["unresolved_readings"]


@pytest.mark.parametrize("punctuation", [".", ";", "…", "”"])
def test_unrelated_punctuation_does_not_remove_an_apostrophe(punctuation):
    root = _root(_word("*ba/kxou"), _word("*ba/kxou'"), punctuation)
    context = DiorisisSourceContext(root)
    assert context.beta(root.findall(".//word")[-1]) == "*ba/kxou'"
    assert context.audit == []


def test_alternate_paren_elision_is_not_a_closing_quote():
    root = _root(_word("par", "παρά", "preposition", "indeclform (prep)"),
                 _word("par)", "παρά", "preposition", "indeclform (prep)"))
    context = DiorisisSourceContext(root)
    assert context.beta(root.findall(".//word")[-1]) == "par)"


def test_multiple_quote_levels_and_contextual_grave_are_reversible():
    root = _root(_word("*ba/kxou"), _word("*ba\\kxou''"), ".’’")
    context = DiorisisSourceContext(root)
    assert context.beta(root.findall(".//word")[-1]) == "*ba\\kxou"
    assert context.audit[0]["original_beta"] == "*ba\\kxou''"


def test_reader_paths_share_the_same_contextual_forms(tmp_path):
    pytest.importorskip("beta_code")
    pytest.importorskip("betacode")
    from build.build_diorisis_pairs import extract_diorisis
    from build.build_diorisis_freq import betacode_to_unicode
    from build.corpus_freq_key import corpus_freq_key
    from extract_diorisis_lm import iter_diorisis_sentences

    root = _root(_word("*ba/kxou") + _word("*ba/kxw|", morph="masc dat sg", wid="2"),
                 _word("*ba/kxou'"))
    (tmp_path / "sample.xml").write_text(ET.tostring(root, encoding="unicode"))
    audit = []
    rows = extract_diorisis(tmp_path, token_audit=audit)
    assert {row["form"] for row in rows} == {"Βάκχου", "Βάκχῳ"}
    assert audit[0]["source"] == "sample.xml"
    assert len(audit[0]["source_sha256"]) == 64
    sentences = list(iter_diorisis_sentences(tmp_path))
    assert sentences[-1][1] == ["<s>", "Βάκχου", "</s>"]
    context = DiorisisSourceContext(root)
    assert corpus_freq_key(betacode_to_unicode(context.beta(root.findall(".//word")[-1]))) == "βακχου"

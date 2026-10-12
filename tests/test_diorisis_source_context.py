"""Context, exact source evidence, and reversible Diorisis quote handling."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import subprocess

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


@pytest.mark.parametrize("error", [OSError("git unavailable"),
                                   subprocess.CalledProcessError(128, "git")])
def test_source_archives_can_write_audits_without_git_metadata(monkeypatch, error):
    pytest.importorskip("beta_code")
    from build import build_diorisis_pairs as builder

    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(builder.subprocess, "check_output", unavailable)
    assert builder.source_revision() is None


@pytest.mark.parametrize("quoted_first", [False, True])
def test_normalized_cells_preserve_distinct_pos_in_both_token_orders(tmp_path, quoted_first):
    pytest.importorskip("beta_code")
    from build.build_diorisis_pairs import extract_diorisis

    noun = _word("o(lkw=|", "ὁλκός", "noun", "masc dat sg")
    adjective = _word("o(lkw=|", "ὁλκός", "adjective", "masc dat sg")
    quoted = _word("o(lkw=|'", "ὁλκός", "adjective", "masc dat sg")
    root = _root(noun + adjective, quoted)
    if quoted_first:
        root[:] = list(reversed(root[:]))
    (tmp_path / "sample.xml").write_text(ET.tostring(root, encoding="unicode"))
    rows = extract_diorisis(tmp_path)
    assert {(r["form"], r["lemma"], r["pos"], tuple(r["tags"])) for r in rows} == {
        ("ὁλκῷ", "ὁλκός", "noun", ("masculine", "singular", "dative")),
        ("ὁλκῷ", "ὁλκός", "adj", ("masculine", "singular", "dative")),
    }
    assert len(rows) == 2


def test_later_document_normalization_restores_all_earlier_analyses(tmp_path):
    pytest.importorskip("beta_code")
    from build.build_diorisis_pairs import extract_diorisis

    ordinary = _word("o(lkw=|", "ὁλκός", "noun", "masc dat sg").replace(
        "</word>", '<lemma entry="ὁλκός" POS="adjective">'
        '<analysis morph="masc dat sg"/><analysis morph="neut dat sg"/>'
        '</lemma></word>')
    (tmp_path / "a.xml").write_text(f"<text><sentence>{ordinary}</sentence></text>")
    root = _root(_word("o(lkw=|", "ὁλκός", "adjective", "masc dat sg"),
                 _word("o(lkw=|'", "ὁλκός", "adjective", "masc dat sg"))
    (tmp_path / "b.xml").write_text(ET.tostring(root, encoding="unicode"))
    rows = extract_diorisis(tmp_path)
    assert [(r["pos"], r["tags"][0]) for r in rows] == [
        ("noun", "masculine"), ("adj", "masculine"), ("adj", "neuter")]
    assert {r["form"] for r in rows} == {"ὁλκῷ"}


def test_normalized_token_keeps_multiple_lemmas_and_analyses(tmp_path):
    pytest.importorskip("beta_code")
    from build.build_diorisis_pairs import extract_diorisis

    ordinary = _word("*ba/kxou").replace(
        "</word>", '<lemma entry="ἄλλος" POS="adjective">'
        '<analysis morph="masc gen sg"/><analysis morph="neut gen sg"/>'
        '</lemma></word>')
    quoted = ordinary.replace('form="*ba/kxou"', 'form="*ba/kxou\'"')
    root = _root(ordinary, quoted)
    (tmp_path / "sample.xml").write_text(ET.tostring(root, encoding="unicode"))
    rows = extract_diorisis(tmp_path)
    assert [(r["lemma"], r["pos"], r["tags"][0]) for r in rows] == [
        ("Βάκχος", "noun", "masculine"),
        ("ἄλλος", "adj", "masculine"),
        ("ἄλλος", "adj", "neuter"),
    ]
    assert {r["form"] for r in rows} == {"Βάκχου"}


def test_untouched_cells_keep_legacy_first_analysis(tmp_path):
    pytest.importorskip("beta_code")
    from build.build_diorisis_pairs import extract_diorisis

    ordinary = _word("o(lkw=|", "ὁλκός", "noun", "masc dat sg").replace(
        "</lemma>", '<analysis morph="neut dat sg"/></lemma>').replace(
        "</word>", '<lemma entry="ἄλλος" POS="adjective">'
        '<analysis morph="masc dat sg"/></lemma></word>')
    (tmp_path / "sample.xml").write_text(f"<text><sentence>{ordinary}</sentence></text>")
    assert extract_diorisis(tmp_path) == [{"form": "ὁλκῷ", "lemma": "ὁλκός",
                                         "pos": "noun",
                                         "tags": ["masculine", "singular", "dative"]}]

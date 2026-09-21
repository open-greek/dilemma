#!/usr/bin/env python3
"""Guards for the GLAUx reader behind the next-word prediction LM.

``train_lm.iter_glaux_sentences`` feeds ``build/lm/grc_ngram.bin``, which
ships inside the Tonos keyboard, so it must apply ``build/nc_filter.py``
exactly as the pair, frequency, attestation and tagger readers do: the
NonCommercial and PROIEL-derived works are dropped whole, and a
Gorman-derived work keeps only its automatic sentences.

The first group builds a four-work mini GLAUx in a temp directory (one
NonCommercial, one PROIEL-derived, one Gorman-derived with a manual and an
automatic sentence, one openly licensed), so it runs on CI with no corpus
on disk. Two variations on it stand beside that corpus: a metadata file
describing the same four works with their roles moved around, which is
how the --glaux-metadata override is told apart from the default path,
and a three-work corpus whose Gorman-derived work sorts last, which is
how the progress line is checked against the --max-files cut. The second
group checks the same wiring against the real GLAUx metadata and skips
when the corpus is absent.

Run with:
    python -m pytest tests/test_lm_corpus.py -x -v
"""

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "build"))

import train_lm  # noqa: E402
from nc_filter import excluded_glaux_stems, gorman_glaux_stems  # noqa: E402

GLAUX_DIR = Path.home() / "Documents" / "glaux" / "xml"
GLAUX_METADATA = Path.home() / "Documents" / "glaux" / "metadata.txt"

# TLG -> (SOURCE_LICENSE, TREEBANK_ANNOTATIONS). 9990 stands in for the
# 7 NonCommercial source texts, 9991 for the 25 PROIEL-derived works,
# 9992 for the 40 works whose manual sentences are Gorman's trees.
PERSEUS = "Perseus Ancient Greek Dependency Treebank"
MINI_WORKS = {
    "9990-001": ("CC BY-NC-SA 3.0", PERSEUS),
    "9991-001": ("CC BY-SA 4.0", "PROIEL"),
    "9992-001": ("CC BY-SA 4.0", "Gorman Trees"),
    "9993-001": ("CC BY-SA 4.0", PERSEUS),
}

# The same four works with their roles moved around: the NonCommercial
# one is openly licensed here, the openly licensed one is PROIEL-derived,
# and so on. A metadata file saying this yields exclusion sets that share
# no work with the ones MINI_WORKS yields, which is what makes it usable
# as a test of whether --glaux-metadata is read at all.
SWAPPED_WORKS = {
    "9990-001": ("CC BY-SA 4.0", PERSEUS),
    "9991-001": ("CC BY-SA 4.0", "Gorman Trees"),
    "9992-001": ("CC BY-NC-SA 3.0", PERSEUS),
    "9993-001": ("CC BY-SA 4.0", "PROIEL"),
}

# A corpus whose Gorman-derived work sorts last, so that --max-files 1
# leaves it unread. Used to check that the progress line counts the files
# the run actually reads.
GORMAN_SORTS_LAST_WORKS = {
    "8001-001": ("CC BY-SA 4.0", PERSEUS),
    "8002-001": ("CC BY-NC-SA 3.0", PERSEUS),
    "8003-001": ("CC BY-SA 4.0", "Gorman Trees"),
}

# One manual and one automatic sentence per work, each with a distinctive
# token so a leak can be traced back to the work and the analysis kind.
SENTENCE_FORMS = {
    ("9990-001", "manual"): ["μῆνιν", "ἄειδε", "θεά"],
    ("9990-001", "auto"): ["ἄνδρα", "μοι", "ἔννεπε"],
    ("9991-001", "manual"): ["ἐν", "ἀρχῇ", "ἦν"],
    ("9991-001", "auto"): ["πάντα", "δι", "αὐτοῦ"],
    ("9992-001", "manual"): ["Θουκυδίδης", "Ἀθηναῖος", "ξυνέγραψεν"],
    ("9992-001", "auto"): ["τὸν", "πόλεμον", "τῶν"],
    ("9993-001", "manual"): ["ἄνθρωποι", "φύσει", "ὀρέγονται"],
    ("9993-001", "auto"): ["πάντες", "ἄνθρωποι", "εἰδέναι"],
}


def _write_metadata(path: Path, works: dict) -> Path:
    """Write a GLAUx metadata.txt describing `works` and return its path."""
    header = ["GLAUX_TEXT_ID", "TLG", "AUTHOR_STANDARD", "TITLE_STANDARD",
              "SOURCE_LICENSE", "TOKENS", "TREEBANK_ANNOTATIONS"]
    rows = ["\t".join(header)]
    for i, (tlg, (license_, annotations)) in enumerate(works.items(), 1):
        rows.append("\t".join([str(i), tlg, "TestAuthor", "TestTitle",
                               license_, "999", annotations]))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _sentence_forms(tlg: str, analysis: str) -> list:
    """Tokens of one sentence. The four MINI_WORKS carry distinctive words
    so a leak can be traced back to the work and the annotation kind it
    came from; any other work gets filler, since those corpora are built
    to check counting rather than what reaches the model."""
    return SENTENCE_FORMS.get((tlg, analysis), ["λόγος", "ἔργον", "ἀριθμός"])


def _write_mini_glaux(root: Path, works: dict = MINI_WORKS) -> Path:
    """Write a GLAUx-shaped corpus (xml/ plus metadata.txt) and return xml/."""
    xml_dir = root / "xml"
    xml_dir.mkdir(parents=True)

    _write_metadata(root / "metadata.txt", works)

    for tlg in works:
        sentences = []
        for n, analysis in enumerate(("manual", "auto"), 1):
            words = []
            for w, form in enumerate(_sentence_forms(tlg, analysis), 1):
                words.append(f'      <word id="{w}" form="{form}" '
                             f'lemma="{form}" postag="n-s---fa-"/>')
            words.append(f'      <word id="{len(words) + 1}" form="." '
                         f'lemma="." postag="u--------"/>')
            sentences.append(
                f'  <sentence id="{n}" document_id="{tlg}" '
                f'analysis="{analysis}">\n' + "\n".join(words)
                + "\n  </sentence>")
        (xml_dir / f"{tlg}.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<treebank version="2" xml:lang="grc">\n'
            + "\n".join(sentences) + "\n</treebank>\n", encoding="utf-8")
    return xml_dir


@pytest.fixture(scope="module")
def mini_glaux(tmp_path_factory):
    return _write_mini_glaux(tmp_path_factory.mktemp("glaux_lm"))


@pytest.fixture(scope="module")
def read(mini_glaux):
    """{work: {analysis kind: [token, ...]}} as the LM reader sees it.

    The reader yields ``<doc_id>:<sentence id>``; sentence 1 is the manual
    sentence and sentence 2 the automatic one in every mini work.
    """
    out: dict = {}
    for sent_id, tokens in train_lm.iter_glaux_sentences(mini_glaux):
        work, _, n = sent_id.partition(":")
        analysis = "manual" if n == "1" else "auto"
        out.setdefault(work, {})[analysis] = tokens
    return out


# --- the mini corpus -------------------------------------------------------

def test_openly_licensed_work_is_read_in_full(read):
    """The filter must not swallow the works it has no claim on."""
    assert set(read["9993-001"]) == {"manual", "auto"}
    assert read["9993-001"]["manual"] == [
        train_lm.BOS_TOK, "ἄνθρωποι", "φύσει", "ὀρέγονται", train_lm.EOS_TOK]


def test_noncommercial_work_is_dropped_whole(read):
    assert "9990-001" not in read


def test_proiel_derived_work_is_dropped_whole(read):
    assert "9991-001" not in read


def test_gorman_work_keeps_auto_sentences_and_drops_manual(read):
    """Gorman is held-out gold, but only its manual sentences are its
    trees; whole-work exclusion would cost the large automatic portions."""
    assert set(read["9992-001"]) == {"auto"}
    assert "Θουκυδίδης" not in read["9992-001"]["auto"]
    assert read["9992-001"]["auto"] == [
        train_lm.BOS_TOK, "τὸν", "πόλεμον", "τῶν", train_lm.EOS_TOK]


def test_no_excluded_token_reaches_the_model(read):
    """Whole-corpus restatement of the three tests above: nothing from an
    excluded work or a held-out-gold sentence may reach the n-gram counts."""
    ingested = {tok for work in read.values()
                for tokens in work.values() for tok in tokens}
    banned = set()
    for (tlg, analysis), forms in SENTENCE_FORMS.items():
        if tlg in ("9990-001", "9991-001") or analysis == "manual":
            banned.update(forms)
    # 9993-001's own manual sentence is legitimately ingested.
    banned -= set(SENTENCE_FORMS[("9993-001", "manual")])
    assert not (ingested & banned)


def test_missing_metadata_is_refused(tmp_path):
    """An absent metadata.txt gives nc_filter empty exclusion sets, so the
    reader has to stop rather than train on everything."""
    xml_dir = _write_mini_glaux(tmp_path / "corpus")
    (tmp_path / "corpus" / "metadata.txt").unlink()
    with pytest.raises(SystemExit, match="metadata.txt"):
        list(train_lm.iter_glaux_sentences(xml_dir))


def test_metadata_override_is_honored(tmp_path, mini_glaux, read):
    """--glaux-metadata has to be the file the exclusion sets come from.

    The override below describes the same four works with their licenses
    and treebank provenance moved around (SWAPPED_WORKS), so it yields a
    different set of exclusions from the metadata.txt sitting next to the
    corpus. Reading the same corpus under it therefore has to return a
    disjoint set of works - which is the only way to tell that the
    override was read rather than ignored in favor of the default path.
    """
    override = _write_metadata(tmp_path / "elsewhere.txt", SWAPPED_WORKS)

    by_work: dict = {}
    for sent_id, tokens in train_lm.iter_glaux_sentences(
            mini_glaux, metadata_path=override):
        work, _, n = sent_id.partition(":")
        by_work.setdefault(work, set()).add(
            "manual" if n == "1" else "auto")

    # Under the default metadata this same corpus gives 9992-001 and
    # 9993-001 (the `read` fixture); under the override it must not.
    assert set(read) == {"9992-001", "9993-001"}
    assert set(by_work) == {"9990-001", "9991-001"}

    # The roles moved with the file: 9990-001 is openly licensed here and
    # is read whole, while 9991-001 is the Gorman-derived work and keeps
    # only its automatic sentences.
    assert by_work["9990-001"] == {"manual", "auto"}
    assert by_work["9991-001"] == {"auto"}


def test_progress_line_counts_the_files_actually_read(tmp_path, capsys):
    """The reader prints what the license filter did, and --max-files cuts
    the list after that filter, so the counts have to be taken after the
    cut. Taken before it, a two-file sanity pass reported all 40 of the
    corpus' Gorman-derived works as being read.

    The corpus here has one openly licensed work, one NonCommercial work
    and one Gorman-derived work that sorts last, so --max-files 1 reads
    exactly the openly licensed one and no Gorman-derived text at all.
    """
    xml_dir = _write_mini_glaux(tmp_path / "corpus", GORMAN_SORTS_LAST_WORKS)
    sentences = list(train_lm.iter_glaux_sentences(xml_dir, max_files=1))
    line = capsys.readouterr().out

    assert {sid.split(":")[0] for sid, _ in sentences} == {"8001-001"}
    gorman = re.search(r"(\d+) Gorman", line)
    assert gorman, f"the progress line no longer reports a count: {line!r}"
    assert gorman.group(1) == "0", line


# --- the real corpus -------------------------------------------------------

_HAVE_GLAUX = GLAUX_DIR.is_dir() and GLAUX_METADATA.exists()

# Small real works, one per case, symlinked into a temp corpus so the check
# runs in a second instead of reading all 1,421 files.
REAL_NONCOMMERCIAL = "0253-001"    # SOURCE_LICENSE is a CC BY-NC variant
REAL_PROIEL = "0031-025"           # TREEBANK_ANNOTATIONS = 'PROIEL'
REAL_GORMAN_MIXED = "0032-004"     # manual + automatic sentences
REAL_OPEN = "0367-001"             # neither excluded nor Gorman-derived


def _manual_and_auto_ids(stem):
    """Sentence ids of a real GLAUx work, split by annotation kind."""
    import xml.etree.ElementTree as ET
    manual, auto = set(), set()
    for sent in ET.parse(GLAUX_DIR / f"{stem}.xml").findall(".//sentence"):
        sid = sent.get("id") or sent.get("struct_id") or "?"
        (manual if sent.get("analysis") == "manual" else auto).add(sid)
    return manual, auto


@pytest.mark.skipif(not _HAVE_GLAUX,
                    reason="GLAUx corpus not present (~/Documents/glaux)")
def test_real_glaux_metadata_yields_the_documented_exclusions():
    """The mini corpus proves the mechanism; the sets have to be non-empty
    on the corpus that is actually read, or the filter is a no-op."""
    excluded = excluded_glaux_stems(GLAUX_METADATA)
    assert len(excluded) >= 32          # 7 NonCommercial + 25 PROIEL-derived
    assert REAL_NONCOMMERCIAL in excluded
    assert REAL_PROIEL in excluded
    assert "0016-001" in excluded       # Herodotus, PROIEL-derived
    assert len(gorman_glaux_stems(GLAUX_METADATA)) >= 40
    assert REAL_GORMAN_MIXED in gorman_glaux_stems(GLAUX_METADATA)


@pytest.mark.skipif(not _HAVE_GLAUX,
                    reason="GLAUx corpus not present (~/Documents/glaux)")
def test_real_glaux_works_are_filtered_as_the_policy_says(tmp_path):
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    for stem in (REAL_NONCOMMERCIAL, REAL_PROIEL,
                 REAL_GORMAN_MIXED, REAL_OPEN):
        (xml_dir / f"{stem}.xml").symlink_to(GLAUX_DIR / f"{stem}.xml")

    by_work: dict = {}
    for sent_id, tokens in train_lm.iter_glaux_sentences(
            xml_dir, metadata_path=GLAUX_METADATA):
        work, _, sid = sent_id.partition(":")
        by_work.setdefault(work, set()).add(sid)

    assert REAL_NONCOMMERCIAL not in by_work
    assert REAL_PROIEL not in by_work
    assert by_work.get(REAL_OPEN), "the openly licensed work should be read"

    manual, auto = _manual_and_auto_ids(REAL_GORMAN_MIXED)
    assert manual and auto, "fixture work no longer has both kinds"
    assert not (by_work[REAL_GORMAN_MIXED] & manual)
    assert by_work[REAL_GORMAN_MIXED] <= auto
    # Sentences of one real token are dropped for other reasons, so this
    # is a subset relation, not equality - but most automatic ones survive.
    assert len(by_work[REAL_GORMAN_MIXED]) > 0.8 * len(auto)

"""Reject corpus word-tails that entered the lookup as words of their own.

Some corpus tokenizers split one printed word at its elision mark and hand the
second half the whole word's lemma. The tail then arrives at ingestion looking
exactly like an elided word, so every per-pair filter passes it, and it lands
in ``lookup.db`` as a form:

    νοντ᾽    -> βαρύνω       (the end of βαρύνοντ᾽)
    εντ᾽     -> πευκήεις     (the end of πυκάεντ᾽)
    λευμ᾽    -> γαμήλευμα    (the end of γαμήλευμ᾽)
    γάσασθ᾽  -> ἐργάζομαι    (the end of ἠργάσασθ᾽)

Nothing about the pair on its own gives the tail away: ``νοντ᾽`` is a plausible
elided word shape, and ``βαρύνω`` is a real lemma. What gives it away is the
rest of the lemma's paradigm. ``βαρύνοντ᾽`` is in the table under the same
lemma, and ``νοντ᾽`` is its ending. That makes this a WHOLE-TABLE test: it
needs every form of the lemma, so it cannot run as a streaming filter at each
pair's point of entry. It runs once over each assembled table instead, just
before the tables are written.

Three further conditions keep it off real words. Measured over the 10,814
Ancient Greek rows of ``data/lookup.db`` whose form ends in an elision mark:

  1. Being a proper ending of another form of the same lemma cuts 10,814 to
     234 (228 once the six single-letter stems are set aside). Most elided
     words - ``δ᾽``, ``σ᾽``, ``κ᾽``, ``τιν᾽``, ``ποτ᾽``, ``κατ᾽`` - are the
     ending of nothing else their lemma inflects to, so the test never
     reaches them at all.
  2. Having no corpus attestation cuts 234 to 8. A tail is a tokenizer's
     invention, so no corpus ever wrote it, while a real word leaves traces.
  3. Being cut from its sibling by two or more characters cuts 8 to 4, and
     the 4 are exactly the tails above.

Tests 2 and 3 catch different things, and the lookup needs both, because
Ancient Greek builds two kinds of word-pair that look exactly like a word and
its tail.

The AUGMENT makes a one-character pair. An unaugmented form is a real word and
is also, by construction, its augmented sibling minus one leading vowel:
``γράψατ᾽`` (aorist imperative of γράφω) is ``ἐγράψατ᾽`` minus ``ἐ``, and
``φης᾽`` is ``έφης`` minus its augment. Neither is in the corpus, so test 2
lets both through; the cut length is the only thing that saves them. A
tokenizer's split lands mid-stem, two to four characters in, never on a lone
augment vowel.

REDUPLICATION makes a two-character pair, which the cut length cannot separate.
``δοῦσ᾽`` is ``διδοῦσ᾽`` minus the reduplicating ``δι``, ``μνήσομ᾽`` is
``μεμνήσομ`` minus ``με``, ``θεῖσ᾽`` is ``τιθεῖσ᾽`` minus ``τι``. All are real
words cut by exactly two characters, so only their corpus attestation saves
them. Of the 23 rows cut by two or more characters, 4 are the tails and the
other 19 are real words held by test 2 alone. Reduplication is the commonest
reason a real word lands in that group, but not the only one: crasis and an
augment before a doubled consonant both produce the same two-character cut.

The rule is therefore deliberately conservative: any form some corpus wrote is
kept, whatever its shape. ``σαντ᾽`` under κέλλω, the ending of ``κέλσαντ``,
looks like a tail and is kept because one corpus wrote it once.

The stem is looked up in the corpus with the elision mark still attached and
never bare. The bare stem is not evidence that the elided word exists: the
same tokenizer error that created the lookup row also left the bare stem in
the corpus it annotated (``νοντ`` occurs 4 times in ``form_profile.db``).

Two residues this module deliberately does not claim. ``G2α᾽`` (lemma ``G``)
and ``G?᾽`` (lemma ``_``) are optical-character-recognition placeholders, not
tails; they are cut from their siblings by one character and so survive test 3.
They need a test on the characters themselves - a form or lemma that is not
Greek - which is a different rule and belongs with the other per-pair citation
hygiene checks in ``build_lookup_db._citation_artifact_reason``.

The module is pure stdlib. ``find_elision_tails`` takes the attestation test as
a callable, so a caller can run the rule without any corpus on disk;
``open_profile_attestation`` builds that callable from ``form_profile.db``.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping

# Every spelling of the elision mark that can reach an assembled table.
#
# U+1FBD is the canonical key that `build_lookup_db` folds onto, and U+0313 is
# how treebank exports write elision before `form_sanitize` rewrites it. The
# rest are the spellings the source texts use directly: U+2019 is what GLAUx
# and Diorisis write, and U+0060/U+02B9 are in the apostrophe set
# `build_lookup_db` already folds. Listing all of them means the rule reads the
# same form the same way no matter which stage it is called from.
#
# U+1FFE GREEK DASIA is deliberately absent. It is the ROUGH breathing, and
# elision leaves a smooth one, so a word-final dasia is a stranded breathing
# mark rather than an apostrophe. Reading it as one would have this rule judge
# forms it has no business judging. No row of the shipped lookup ends in it.
ELISION_MARKS = frozenset(
    "\u1FBD"   # GREEK KORONIS - the canonical key
    "\u2019"   # RIGHT SINGLE QUOTATION MARK - GLAUx, Diorisis
    "\u1FBF"   # GREEK PSILI - spacing smooth breathing used as an apostrophe
    "\u02BC"   # MODIFIER LETTER APOSTROPHE
    "\u0313"   # COMBINING COMMA ABOVE - how treebank exports write elision
    "\u0027"   # APOSTROPHE - plain ASCII
    "\u0060"   # GRAVE ACCENT - a typewriter apostrophe
    "\u02B9"   # MODIFIER LETTER PRIME
)

# How many characters a form must be cut from its sibling by before the rule
# will call it a tail. One character is the augment, which makes a real word;
# see the module docstring.
MIN_CUT_CHARACTERS = 2

REASON = "corpus_word_tail"


def elision_stem(form: str) -> str | None:
    """The form without its trailing elision mark, or None if it has none.

    Returns None for a form that is only a mark, and for a form whose mark is
    leading rather than trailing (``᾽ς`` for εἰς is aphaeresis, which takes the
    FRONT off a word, so such a form can never be the tail of one).
    """
    if not isinstance(form, str) or len(form) < 2:
        return None
    if form[-1] not in ELISION_MARKS:
        return None
    stem = form[:-1]
    return stem or None


def _folded(form: str) -> str:
    """The form with a trailing elision mark removed, whatever its spelling.

    Used to compare a candidate against its siblings: which spelling of the
    apostrophe each of them happens to carry is not part of the question.
    """
    return elision_stem(form) or form


def find_elision_tails(
    table: Mapping[str, str],
    *,
    is_attested: Callable[[str], bool],
    min_cut: int = MIN_CUT_CHARACTERS,
    sibling_tables: Iterable[Mapping[str, str]] | None = None,
) -> dict[str, str]:
    """Return the ``{form: sibling}`` of every corpus word-tail in ``table``.

    ``table`` maps a surface form to its lemma. ``is_attested`` is called with
    a candidate's stem and must report whether any corpus wrote that stem with
    an elision mark on it; the caller supplies it so this stays testable
    without a corpus. ``min_cut`` is exposed for tests only - a caller that
    lowers it to 1 will reject unaugmented forms along with the tails.

    ``sibling_tables`` is where the full words are looked for, and defaults to
    ``table`` itself. The caller passes every table it is about to write when
    more than one exists: the builder writes Ancient-Greek-only and
    Modern-Greek-only rows from the `ag` and `el` tables wherever those differ
    from `combined`, so a tail whose full word sits in a DIFFERENT table than
    the tail itself would otherwise be dropped from one table and written back
    from another.

    The sibling in the returned mapping is the shortest form the tail was cut
    from, which is the evidence for the rejection.
    """
    # Pass 1: the candidates, and the lemmas worth collecting a paradigm for.
    # Only a form that ends in an elision mark can be a tail, which is 10,814
    # of the shipped lookup's 7.1M Ancient Greek rows.
    candidates: dict[str, tuple[str, str]] = {}
    for form, lemma in table.items():
        if not isinstance(lemma, str) or not lemma:
            continue
        stem = elision_stem(form)
        if stem is not None:
            candidates[form] = (lemma, stem)
    if not candidates:
        return {}

    wanted = {lemma for lemma, _ in candidates.values()}

    # Pass 2: the other forms of those few hundred lemmas. Collecting the whole
    # table by lemma would cost gigabytes; collecting only the lemmas a
    # candidate claims costs a few thousand strings.
    # Scanning each source once per filtered table costs a few minutes on a
    # build that already takes hours, and buys the cross-table guarantee; a
    # shared index would be faster and would couple the caller to this
    # module's internals.
    siblings: dict[str, set[str]] = defaultdict(set)
    for source in (sibling_tables if sibling_tables is not None else (table,)):
        for form, lemma in source.items():
            if isinstance(lemma, str) and lemma in wanted:
                siblings[lemma].add(_folded(form))

    # Pass 3: the three tests, cheapest first. The attestation test is a
    # database query, so it runs only on what the in-memory tests have left.
    tails: dict[str, str] = {}
    for form, (lemma, stem) in candidates.items():
        cut, source = _shortest_cut(stem, siblings[lemma])
        if source is None or cut < min_cut:
            continue
        if is_attested(stem):
            continue
        tails[form] = source
    return tails


def _shortest_cut(stem: str, sibling_stems: Iterable[str]) -> tuple[int, str | None]:
    """The fewest characters any sibling has to lose to become ``stem``.

    Only a strictly longer sibling counts, so a form is never judged against
    itself or against a differently spelled copy of itself.
    """
    best_cut = 0
    best: str | None = None
    for sibling in sibling_stems:
        if len(sibling) <= len(stem) or not sibling.endswith(stem):
            continue
        cut = len(sibling) - len(stem)
        if best is None or cut < best_cut or (cut == best_cut and sibling < best):
            best_cut, best = cut, sibling
    return best_cut, best


def open_profile_attestation(
    path: str | Path,
    *,
    marks: Iterable[str] = tuple(sorted(ELISION_MARKS)),
) -> Callable[[str], bool]:
    """Build the ``is_attested`` callable from ``data/form_profile.db``.

    A stem counts as attested when any corpus behind ``form_profile.db`` wrote
    it with any spelling of the elision mark attached. The bare stem is not
    queried; see the module docstring for why.

    The connection is read-only and stays open for the life of the callable.
    """
    marks = tuple(dict.fromkeys(marks))
    connection = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    placeholders = ",".join("?" * len(marks))
    query = (
        "SELECT 1 FROM forms f JOIN form_profile p ON p.form_id = f.form_id "
        f"WHERE f.form IN ({placeholders}) AND p.total_count > 0 LIMIT 1"
    )

    def is_attested(stem: str) -> bool:
        spellings = [stem + mark for mark in marks]
        return connection.execute(query, spellings).fetchone() is not None

    return is_attested

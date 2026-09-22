#!/usr/bin/env python3
"""Export Dilemma lookup tables as Hunspell .dic + .aff pairs.

This produces a compact Hunspell-format artifact for the Tonos iOS
keyboard extension. Two variants are emitted under build/hunspell/:

    el_GR_monotonic.dic / .aff   Modern Greek, monotonic
    grc_polytonic.dic  / .aff    Ancient + Medieval, polytonic

Both are accompanied by a sidecar .version file carrying a semver and
the dilemma commit hash the artifact was built from. Tonos reads this
to detect updates.

Approach
--------

Greek inflection is highly accent-sensitive, so exact SFX flag-sharing
across lemmas produces very long tails (~325K distinct signatures for
~426K MG lemmas). Rather than force every form through a shared affix
class, we take a pragmatic middle ground:

1. Group forms by lemma (from lookup.db).
2. For each lemma, compute the longest common prefix (stem) across
   its inflected forms and the multiset of suffixes.
3. If two or more lemmas share the exact same suffix-signature we emit
   one SFX rule-set and point both lemma stems at it.
4. Lemmas with a unique signature still get stem-compression:
   one .dic line per lemma, with a per-lemma SFX flag. We cap the
   suffix-rule expansion cost and fall back to plain wordlist entries
   for pathological cases.
5. Singleton lemmas (exactly one form) go in the .dic as plain words,
   no flag.

Frequency scoring
-----------------

Each .dic line carries a morphological field ``fr:<bucket>`` where
bucket is one of C, M, R, X (Common, Medium, Rare, eXtremely rare /
unseen in corpus). Buckets are computed from corpus counts found in
data/corpus_freq.json (GLAUx + Diorisis, 27M AG tokens) for AG forms
and data/mg_form_freq.json (Wiktionary + HNC-derived) for MG. Lookup
is done on the stripped-accent form so polytonic forms can inherit the
monotonic corpus count. Bucket edges:

    C: count >= 1000
    M: count >= 100
    R: count >= 1
    X: count == 0  (lemma-generated forms never observed)

Tonos can parse ``fr:`` to rank spelling candidates (e.g. prefer
suggesting 'και' over some obscure homograph).

Limitations
-----------

- Hunspell does not natively handle Greek iota subscript / polytonic
  breathings as separate graphemes, but since we store and match
  against NFC-normalized strings this is fine.
- Some Wiktionary-generated forms may be spurious for real usage.
  They are kept but tagged X so the consumer can filter.
- Capitalization is represented via the KEEPCASE flag. Proper names
  are tagged with a capital-only flag so 'σεραφειμ' is not falsely
  accepted for 'Σεραφείμ'. We err on the side of acceptance.

Usage
-----

    python3 export_hunspell.py                 # full export
    python3 export_hunspell.py --sanity 10000  # 10K-lemma sanity pass
    python3 export_hunspell.py --variant el    # only MG
    python3 export_hunspell.py --variant grc   # only AG
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sqlite3
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

from dilemma.form_sanitize import (
    canonicalize_final_elision,
    has_editorial_sigla,
    sanitize_form,
)

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "build" / "hunspell"

# AG function-word forms that dilemma.py resolves via hardcoded rules or that
# are conventionally written without an accent in enclitic use. These need to
# be added to the AG polytonic artifact explicitly or the keyboard will reject
# extremely common words like τὸ, τὴν, μοι, τε, and γε. Mapping: form -> lemma.
AG_FUNCTION_WORDS = {
    # Definite article (ὁ)
    "ὁ": "ὁ", "ἡ": "ὁ", "τό": "ὁ", "τοῦ": "ὁ", "τῆς": "ὁ",
    "τῶν": "ὁ", "τόν": "ὁ", "τήν": "ὁ", "τά": "ὁ", "τοῖς": "ὁ",
    "ταῖς": "ὁ", "τῷ": "ὁ", "τῇ": "ὁ", "τούς": "ὁ", "τάς": "ὁ",
    "τοῖν": "ὁ", "ταῖν": "ὁ", "οἱ": "ὁ", "αἱ": "ὁ", "τώ": "ὁ",
    # Grave variants
    "τὸ": "ὁ", "τοὺς": "ὁ", "τὰ": "ὁ", "τὸν": "ὁ", "τὴν": "ὁ",
    "τὰς": "ὁ", "αἵ": "ὁ", "οἵ": "ὁ",
    # 1st/2nd person pronouns
    "μοι": "ἐγώ", "μοί": "ἐγώ", "μου": "ἐγώ", "με": "ἐγώ",
    "ἐμοί": "ἐγώ", "ἐμοῦ": "ἐγώ", "ἐμέ": "ἐγώ", "ἐγώ": "ἐγώ",
    "ἡμεῖς": "ἐγώ", "ἡμῶν": "ἐγώ", "ἡμῖν": "ἐγώ", "ἡμᾶς": "ἐγώ",
    "σοι": "σύ", "σοί": "σύ", "σου": "σύ", "σε": "σύ",
    "σοῦ": "σύ", "σύ": "σύ",
    "ὑμεῖς": "σύ", "ὑμῶν": "σύ", "ὑμῖν": "σύ", "ὑμᾶς": "σύ",
    # Orthographically unaccented enclitic particles and pronouns. This is a
    # closed grammatical list, not a general exception for stripped forms.
    "τε": "τε", "γε": "γε", "τις": "τις", "τι": "τις",
    "τινος": "τις", "τινι": "τις", "τινα": "τις", "τινε": "τις",
    "τινοιν": "τις", "τινες": "τις", "τινων": "τις", "τισι": "τις",
    "τισιν": "τις", "τινας": "τις",
    "φημι": "φημί", "φησι": "φημί", "φησιν": "φημί",
    "φαμεν": "φημί", "φατε": "φημί", "φασι": "φημί", "φασιν": "φημί",
    "μιν": "ὅς", "νιν": "ὅς", "σφε": "σφεῖς", "σφι": "σφεῖς",
    "σφιν": "σφεῖς", "σφων": "σφεῖς", "σφας": "σφεῖς",
    "σφισι": "σφεῖς", "σφισιν": "σφεῖς",
    "ποτε": "ποτε", "που": "που", "πως": "πως", "πω": "πω",
    "πη": "πη", "πῃ": "πῃ", "κῃ": "πῃ", "ποθι": "ποθι", "ποθεν": "ποθεν",
    "περ": "περ", "τοι": "τοι", "νυν": "νυν", "νυ": "νυ",
    "θην": "θην", "κε": "κε", "κεν": "κε",
    # Enclitic forms of εἰμί. These are legitimately accentless in
    # connected text; the accented spellings remain in the ordinary paradigm.
    "ἐν": "ἐν", "εἰς": "εἰς", "ἐς": "ἐς", "ἐκ": "ἐκ", "ἐξ": "ἐξ",
    "οὐ": "οὐ", "ὡς": "ὡς", "εἰ": "εἰ",
    "εἰμι": "εἰμί", "ἐστι": "εἰμί", "ἐστιν": "εἰμί",
    "ἐσμεν": "εἰμί", "ἐστε": "εἰμί", "εἰσι": "εἰμί", "εἰσιν": "εἰμί",
    "ποι": "ποι", "οὑ": "οὑ", "ἑ": "ἑ", "σφεας": "σφεῖς",
    "σφωε": "σφεῖς", "σφωϊν": "σφεῖς", "ῥα": "ῥα", "κα": "κα",
    "μευ": "ἐγώ", "σευ": "σύ", "τευ": "σύ", "τυ": "σύ",
    "κοτε": "ποτε", "κου": "που", "κως": "πως", "κω": "πω",
    "ποκα": "ποτε",
    # The consonant-final contextual forms of οὐ are complete words, not
    # bare elision fallbacks, even though marked οὐκ᾽/οὐχ᾽ also occur.
    "οὐκ": "οὐ", "οὐχ": "οὐ",
    # Dialect proclitics (Doric article ἁ, Doric/Aeolic αἰ for εἰ, epic
    # εἰν), dialect enclitic forms of εἰμί, and crasis of καί or the article
    # with a proclitic, which stays unaccented (κἀν = καὶ ἐν/ἄν).
    "ἁ": "ὁ", "αἰ": "εἰ", "εἰν": "ἐν",
    "ἐντι": "εἰμί", "ἐσσι": "εἰμί", "εἰμεν": "εἰμί", "ἐστον": "εἰμί",
    "κἀν": "καί", "κᾀν": "καί", "κἀκ": "καί", "κἀξ": "καί",
    "κοὐ": "καί", "κοὐκ": "καί", "κοὐχ": "καί",
    "κεἰ": "καί", "κεἰς": "καί", "χὠ": "καί", "τἀν": "ὁ",
}

# High-frequency spellings resolved by Dilemma's grammatical/POS layers but
# absent from lookup.db. Keep this list narrow and independently attested by
# the pinned full LM; it is not a second lexicon.
AG_EXPORT_OVERRIDES = {
    # Runtime grammar/POS forms absent from lookup.db.
    "τ᾽": "τε",
    "μεθ᾽": "μετά",
    # The shipped dictionary has this acute twin of a reviewed grave form;
    # keep it even though the source compatibility fixture excludes it as a
    # respelling and the initial smooth upsilon is only a heuristic warning.
    "ὐπό": "ὑπό",
    "εἶνε": "εἵνω",
    "μαῦρον": "μαυρός",
    "μαῦροι": "μαυρός",
    "ἦτο": "εἰμί",
    "χἠ": "καί",
    "μιάν": "μία",
    "μιάς": "μία",
    "ἑκατέρως": "ἑκάτερος",
    # Exact mappings retained in authoritative AG source artifacts but lost
    # from lookup.db through source collisions or rejection of a malformed
    # competing lemma.
    "ἀγαποῦσα": "ἀγαπάω",
    "ἀγαποῦν": "ἀγαπάω",
    "ὀκτωβρίου": "Ὀκτώβριος",
    "τοιονδί": "τοιόσδε",
    "Φασαήλου": "Φασαήλος",
    "εὐρωπαίων": "Εὐρωπαῖος",
    "τοσουτονί": "τοσοῦτος",
    "λυχνίας": "λυχνία",
    "μαλακίων": "μαλάκιον",
    "προηγμένα": "προάγω",
    "μαύρους": "Μαῦρος",
    "πληρέστατα": "πλήρης",
    "ἔναι": "ναίω",
    "κατάβα": "καταβαίνω",
}

# Homeric prepositions shortened by apocope and subsequent assimilation.
# These are complete words despite ending in consonants that usually signal
# an elision fallback. Keep both acute and contextual-grave spellings.
HOMERIC_SHORT_PREPOSITIONS = {
    "κὰτ": "κατά", "κάτ": "κατά",
    "κὰδ": "κατά", "κάδ": "κατά",
    "κὰκ": "κατά", "κάκ": "κατά",
    "κὰπ": "κατά", "κάπ": "κατά",
    "κὰμ": "κατά", "κάμ": "κατά",
    "κὰγ": "κατά", "κάγ": "κατά",
    "ἂμ": "ἀνά", "ἄμ": "ἀνά",
}

# Complete words, not elision fallbacks, that end in a consonant no other
# Greek word ends in: ἔκ is ἐκ with the accent of anastrophe, and παρέκ is the
# adverb and preposition "beside, beyond".
CONSONANT_FINAL_WORDS = {
    "ἔκ": "ἐκ", "παρέκ": "παρέκ", "παρὲκ": "παρέκ",
    # Homeric double prepositions.
    "ὑπέκ": "ὑπέκ", "ὑπὲκ": "ὑπέκ", "διέκ": "διέκ", "διὲκ": "διέκ",
    # Indeclinable Hebrew loanwords of the Septuagint.
    "χερουβίμ": "χερουβίμ", "χερουβὶμ": "χερουβίμ",
    "χερουβείμ": "χερουβίμ", "χερουβεὶμ": "χερουβίμ",
    "σεραφίμ": "σεραφίμ", "σεραφὶμ": "σεραφίμ", "σεραφεὶμ": "σεραφίμ",
    "ἐφούδ": "ἐφούδ", "ἐφοὺδ": "ἐφούδ", "φασέκ": "φασέκ", "φασὲκ": "φασέκ",
    "αἰλάμ": "αἰλάμ", "αἰλὰμ": "αἰλάμ", "σαβαώθ": "σαβαώθ",
    "σαβαὼθ": "σαβαώθ", "ναθινίμ": "ναθινίμ", "ναθινὶμ": "ναθινίμ",
}

# Crasis whose coronis or accent falls where the structural rules do not
# expect one: after the first syllable (ἐγᾦμαι = ἐγὼ οἶμαι, μέντἄν =
# μέντοι ἄν, καλοκἀγαθία) or on vocative ὦ (ὦνθρωπε = ὦ ἄνθρωπε); and the
# Homeric demonstratives of ὅδε, whose fused -δε leaves one accent.
GRC_CRASIS_EXCEPTIONS = {
    "τοῖσιδε": "ὅδε", "τοιοῖσιδε": "τοιόσδε", "τοῖσδεσσι": "ὅδε",
    "ἐγᾦμαι": "οἴομαι", "ἐγᾦδα": "οἶδα",
    "μέντἄν": "μέντοι", "μεντἄν": "μέντοι",
    "μέντἂν": "μέντοι", "μεντἂν": "μέντοι",
    "ταὧς": "ταὧς",
    "καλοκἀγαθία": "καλοκἀγαθία", "καλοκἀγαθίας": "καλοκἀγαθία",
    "καλοκἀγαθίαν": "καλοκἀγαθία", "καλοκἀγαθίᾳ": "καλοκἀγαθία",
    "καλοκἀγαθίαι": "καλοκἀγαθία", "καλοκἀγαθίαις": "καλοκἀγαθία",
    "καλοκἀγαθιῶν": "καλοκἀγαθία", "Καλοκἀγαθία": "καλοκἀγαθία",
    "ὦνθρωπε": "ἄνθρωπος", "ὦνδρες": "ἀνήρ", "ὦγαθέ": "ἀγαθός",
    "ὦγαθοί": "ἀγαθός", "Ὦπολλον": "Ἀπόλλων",
}

# Every closed-list grc form, added after the lookup and frequency gates.
GRC_CLOSED_LIST_FORMS = (
    AG_FUNCTION_WORDS | AG_EXPORT_OVERRIDES | HOMERIC_SHORT_PREPOSITIONS
    | CONSONANT_FINAL_WORDS | GRC_CRASIS_EXCEPTIONS
)

# These lookup forms are not acceptable polytonic spellings. In particular,
# tau + omega + iota-subscript is not an unaccented variant of τῷ.
GRC_REJECT_FORMS = frozenset({
    # Unaccented article spellings must remain misspellings so Tonos can
    # correct them to the polytonic forms.
    "του", "τῳ",
    # Corpus/editorial errors which collide with very common words and rank
    # ahead of the correct spelling in downstream suggestions.  These are not
    # merely unattested generated cells: each was traced to a bad lookup row
    # or noisy corpus spelling, so the exporter must not treat it as proof.
    "τού", "τῷν", "τής", "αὐτου", "ταίς", "πᾶντα", "εἴπε", "ἆλλος",
    "ἵσον", "στό", "στά",
})

# lookup.db deliberately keeps these bare stems as tolerant lemmatizer
# fallbacks for elided words. They are not standalone spellings. They must not
# enter the word list or become accepted synthetic stems during affix
# compression.
BARE_ELISION_STEMS = frozenset({
    "δ", "ἀλλ", "δι", "καθ", "κατ", "παρ", "ἐπ", "ἐφ", "οὐδ", "ὑπ",
    "ἀπ", "μεθ",
    # Additional historical fallbacks observed in the 1.3.4 candidate. The
    # mechanism in is_bare_elision_fallback covers same-lemma cases; this
    # reviewed set also catches malformed self-headwords such as ἵν.
    "ἀφ", "ὑφ", "ἵν", "τοῦτ", "ταῦτ", "ὅτ", "ἀνθ", "οὔτ", "μήτ",
    "ἔπειτ", "εἶτ", "ἀντ", "μετ", "γ", "μ", "σ", "θ",
    "ἄπ", "δῖ", "λόγ",
})

# DGE and Cunliffe are independently edited headword inventories rather than
# generated paradigm tables. Their overlap is not required: either is enough
# to prove that a spelling which also resembles an elision stem is a real
# standalone headword (notably ἄν).
INDEPENDENT_AG_HEADWORD_PATHS = (
    DATA / "dge_headwords.json",
    DATA / "cunliffe_headwords.json",
)

LOOKUP_DB = DATA / "lookup.db"
CORPUS_FREQ = DATA / "corpus_freq.json"
MG_FORM_FREQ = DATA / "mg_form_freq.json"
FORM_PROFILE_DB = DATA / "form_profile.db"
LSJ9_FREQUENCY = DATA / "lsj9_frequency.json"
GRC_COMPATIBILITY_FORMS = DATA / "hunspell_grc_april_compat.json.gz"
GRC_TEXTBOOK_FORMS = DATA / "hunspell_grc_textbook.json.gz"
# Curated iconic AG polytonic surface forms and lemmas that are always
# promoted to bucket C, regardless of raw corpus token count. See the
# file's own _comment field for rationale.
CANONICAL_AG_FORMS = DATA / "canonical_ag_forms.json"
# The reviewed head of the language-model vocabulary: its 1,000 most frequent
# Greek forms, minus reviewed nonwords. The release audit requires every one,
# so the exporter pins them. This includes polytonic Modern spellings such as
# the article τή, which the Ancient Greek corpora behind form_profile.db
# barely attest.
LM_HEAD_FIXTURE = ROOT / "tests" / "fixtures" / "hunspell_lm_top1000.json"
LM_HEAD_EXCLUSIONS = (
    ROOT / "tests" / "fixtures" / "hunspell_lm_top1000_exclusions.json"
)

# Complete paradigms used as a stable textbook morphology gate.  Sparse cells
# of these paradigms are useful even when no corpus in form_profile.db happens
# to contain the exact surface form.  The set covers a regular omega verb,
# three athematic types, and one contract verb of each class.
GRC_COMPLETE_PARADIGM_LEMMAS = frozenset({
    "λύω", "παιδεύω", "τίθημι", "δίδωμι", "ἵστημι",
    "τιμάω", "ποιέω", "δηλόω",
})

# A new (post-April) spelling that differs only in its marks from a more
# common well-formed spelling needs treebank support, rising as its share of
# the common spelling's count falls: (share below, GLAUx or Diorisis tokens
# required). Only a dominant spelling with at least
# PROFILE_VARIANT_DOMINANCE_MIN tokens triggers the check. Greek has many
# genuine pairs that differ only in accent or iota subscript (φυγή/φυγῇ,
# βεβαία/βέβαια), and beside a rare word the rarer spelling is about as often
# a real word as a typo; beside a common word it is usually a typo, and one a
# keyboard user is likely to type. The check is not applied to the shipped
# compatibility baseline, pinned forms, or productive enclitic second accents.
PROFILE_VARIANT_DOMINANCE_MIN = 1000
PROFILE_VARIANT_TREEBANK_FLOORS = ((0.01, 25), (0.05, 5))
# Sources in form_profile.db that are lemmatized treebanks of edited texts,
# as opposed to digitized or OCR editions.
TREEBANK_PROFILE_SOURCES = ("glaux", "diorisis")

# Greek combining marks considered polytonic (absent in monotonic text)
POLYTONIC_MARKS = {0x0313, 0x0314, 0x0342, 0x0345}
# Grave accent (U+0300) is technically monotonic-absent too, but we
# treat it as polytonic for classification purposes.
POLYTONIC_MARKS_EXT = POLYTONIC_MARKS | {0x0300}
TONAL_MARKS = frozenset({0x0300, 0x0301, 0x0342})
# Characters of the Greek blocks that are editorial or numeral signs, not
# letters of a normalized spelling: the numeral sign, the spacing iota
# subscript, and lunate sigma (ϲφόδρα for σφόδρα).
GREEK_EDITORIAL_SIGNS = frozenset({0x0374, 0x037A, 0x03F2, 0x03F9})
BREATHING_MARKS = frozenset({0x0313, 0x0314})
ALLOWED_GREEK_COMBINING_MARKS = frozenset({
    0x0300, 0x0301, 0x0304, 0x0306, 0x0308,
    0x0313, 0x0314, 0x0342, 0x0345,
})
GREEK_DIPHTHONGS = frozenset({
    "αι", "ει", "οι", "υι", "αυ", "ευ", "ηυ", "ου", "ωυ",
})

# Spacing (non-combining) characters that Greek orthography uses as a
# genuine mark on a word. Unicode files them as Sk (modifier symbol),
# not Mn (nonspacing mark), so a combining-category test on its own
# reports a fully marked form such as 'δ᾽' as bare. Members:
#   U+1FBD GREEK KORONIS - the elision and crasis apostrophe. Now that
#     build_lookup_db.py runs sanitize_form at ingestion, a trailing
#     combining psili used as an apostrophe is rewritten to this
#     character before the exporter ever sees it, so elided forms that
#     lose their accent along with the elided syllable (δ᾽, κατ᾽, μετ᾽,
#     μηδ᾽) reach this gate carrying no combining mark at all.
#   U+1FBF GREEK PSILI and U+1FFE GREEK DASIA - the spacing breathings.
#     sanitize_form reattaches a leading one to its base letter but
#     leaves a trailing one alone, so a source that writes elision with
#     a spacing psili (κατ᾿, μετ᾿) reaches the exporter carrying it.
#     Dasia does not currently occur in that position in lookup.db; it
#     is listed so the pair stays symmetric if a source emits one.
# Excluded on purpose, because accepting them would defeat this gate
# rather than repair it:
#   U+00B4 ACUTE ACCENT and U+0384 GREEK TONOS - no Greek word is
#     spelled with a free-standing acute. Where one turns up it is
#     either an accent that failed to combine or the numeral sign
#     keraia, as in the Milesian numerals 'τ΄' and 'ϟ΄'; accepting it
#     would ship both as correct polytonic spellings.
#   U+0375 GREEK LOWER NUMERAL SIGN - a numeral marker, not a mark on
#     a word.
#   U+2019 RIGHT SINGLE QUOTATION MARK and U+02BC MODIFIER LETTER
#     APOSTROPHE - ordinary punctuation and a modifier letter. Some
#     sources do write elision with them, but they are also what a
#     stray quotation mark in noisy text looks like, and neither
#     carries Greek-specific meaning the way the koronis does. A form
#     that needs one should have it normalized to U+1FBD at ingestion.
SPACING_DIACRITICS = {0x1FBD, 0x1FBF, 0x1FFE}

# Greek vowel letters, both cases. Final sigma is a consonant, so the
# aphaeresized '᾽ς' is not affected by the rule below.
GREEK_VOWELS = frozenset("αεηιουωΑΕΗΙΟΥΩ")

# Leaving the three characters above out of SPACING_DIACRITICS does not
# by itself keep numerals out of the artifact, because a source that
# prints the keraia as a koronis writes the Milesian numerals with a
# character that IS in the set: lookup.db holds 'α᾽' (1) under lemma
# α, 'ε᾽' (5) under πέντε and 'ο᾽' (70) under lemma ο. It also holds
# vowel-initial spellings that simply lost their breathing on the way
# in, such as 'ημειβετ᾿' beside the correct 'ἠμείβετ᾿'.
#
# What separates those from a real elision is the first letter, not the
# length - 'δ᾽', 'μ᾽', 'σ᾽', 'κ᾽' and 'ν᾽' are one-letter elisions
# that have to keep passing. Polytonic Greek writes a breathing over
# every word-initial vowel, and neither elision, which drops the end of
# a word, nor aphaeresis, whose koronis stands in front of the letters
# that survive, can take that breathing away. sanitize_form has already
# reattached any leading spacing breathing to its vowel before a form
# reaches this gate, so a form whose only mark is a spacing one and
# whose first letter is a bare vowel is not an elided word at all.
#
# A numeral whose letters start with a consonant still passes, and must:
# 'δ᾽' is the numeral 4 as well as the elided δέ, 'κ᾽' is 20 as well as
# the elided κε, and no rule on the spelling can tell those apart. That
# costs nothing, because the keyboard has to accept the string either
# way.


def _first_letter(s: str) -> str:
    """Return the first letter of s, decomposed so a precomposed vowel
    is reported by its base letter, or '' if s has no letter at all."""
    for c in unicodedata.normalize("NFD", s):
        if unicodedata.category(c)[0] == "L":
            return c
    return ""


def _is_greek_letter(c: str) -> bool:
    return "Ͱ" <= c <= "Ͽ" or "ἀ" <= c <= "῿"


def marked_by_spacing_diacritic(s: str) -> bool:
    """True if s is spelled the way an elided or aphaeresized word is:
    its only mark is a spacing koronis or breathing, and it opens on a
    Greek consonant rather than on a bare vowel. See the comment above
    SPACING_DIACRITICS for why the opening letter is the test."""
    if not any(ord(c) in SPACING_DIACRITICS for c in s):
        return False
    first = _first_letter(s)
    return bool(first) and _is_greek_letter(first) and first not in GREEK_VOWELS


def has_polytonic(s: str) -> bool:
    """True if the string carries any mark exclusive to polytonic script
    (breathings, circumflex, grave, iota subscript)."""
    nfd = unicodedata.normalize("NFD", s)
    return any(ord(c) in POLYTONIC_MARKS_EXT for c in nfd)


def has_any_diacritic(s: str) -> bool:
    """True if the string carries a mark that makes it AG orthography.
    Used to decide whether a form is 'accented enough' to ship in the
    AG polytonic variant. We keep acute-only forms (e.g. 'ζωή') because
    they are the canonical AG lexicon entries, and elided or
    aphaeresized forms whose only mark is a spacing koronis or
    breathing (e.g. 'δ᾽', 'κατ᾿', '᾽ς'), but we drop fully-stripped
    fallback keys the DB carries for case-insensitive lookup. A
    combining mark settles it on its own; a spacing mark counts only on
    a form spelled the way an elided word is, which is what
    marked_by_spacing_diacritic decides."""
    nfd = unicodedata.normalize("NFD", s)
    if any(unicodedata.category(c) == "Mn" for c in nfd):
        return True
    return marked_by_spacing_diacritic(s)


def has_required_initial_breathing(form: str) -> bool:
    """Return whether a grc spelling obeys the initial-breathing rule.

    A vowel-initial word must carry smooth or rough breathing on its opening
    vowel or on the second vowel of an opening diphthong. Initial rho likewise
    requires a breathing. Consonant-initial and aphaeresized forms are outside
    this rule and pass unchanged.
    """
    nfd = unicodedata.normalize("NFD", form)
    bases: list[tuple[str, set[int]]] = []
    for char in nfd:
        if unicodedata.category(char) == "Mn":
            if bases:
                bases[-1][1].add(ord(char))
            continue
        if _is_greek_letter(char):
            bases.append((char, set()))
        elif bases:
            break

    if not bases:
        return True
    first, first_marks = bases[0]
    if first.lower() == "ρ":
        return bool(first_marks & BREATHING_MARKS)
    if first not in GREEK_VOWELS:
        return True
    if first_marks & BREATHING_MARKS:
        return True
    if len(bases) < 2:
        return False
    second, second_marks = bases[1]
    first_blocks_diphthong = first_marks & (
        TONAL_MARKS | {0x0308, 0x0345}
    )
    return bool(
        (first + second).lower() in GREEK_DIPHTHONGS
        and not first_blocks_diphthong
        and second_marks & BREATHING_MARKS
    )


def _greek_bases(form: str) -> list[tuple[str, frozenset[int]]]:
    """Return Greek base letters with the combining marks on each base."""
    bases: list[tuple[str, set[int]]] = []
    for char in unicodedata.normalize("NFD", form):
        if unicodedata.category(char) == "Mn":
            if bases:
                bases[-1][1].add(ord(char))
            continue
        if _is_greek_letter(char):
            bases.append((char, set()))
    return [(base, frozenset(marks)) for base, marks in bases]


def _syllable_base_indexes(
    bases: list[tuple[str, frozenset[int]]],
) -> list[tuple[int, ...]]:
    """Return the base-letter indexes belonging to each Greek syllable.

    This intentionally needs only enough syllabification for the universal
    three-syllable accent window.  A recognised diphthong is one nucleus unless
    the second vowel carries diaeresis.
    """
    syllables: list[tuple[int, ...]] = []
    index = 0
    while index < len(bases):
        base, _marks = bases[index]
        if base not in GREEK_VOWELS:
            index += 1
            continue
        members = [index]
        if index + 1 < len(bases):
            next_base, next_marks = bases[index + 1]
            pair = (base + next_base).lower()
            diphthong = (
                pair in GREEK_DIPHTHONGS
                and not (_marks & (TONAL_MARKS | BREATHING_MARKS
                                   | frozenset({0x0345})))
            )
            # Iota adscript after a long α, η or ω: the marks sit on the
            # long vowel and the ι carries none (ζῶια, τῆιδε, ῥάιδιον).
            adscript = (
                base.lower() in "αηω"
                and next_base.lower() == "ι"
                and not next_marks
            )
            if (next_base in GREEK_VOWELS
                    and (diphthong or adscript)
                    and 0x0308 not in next_marks):
                members.append(index + 1)
                index += 1
        syllables.append(tuple(members))
        index += 1
    return syllables


def mark_skeleton(form: str) -> str:
    """Return the letters of ``form`` with every combining mark removed.

    The key is built from :func:`exact_form_key`, so case and contextual grave
    are already folded. Accents, breathings, diaeresis, iota subscript, and
    vowel-length marks are then all removed, while the spacing elision mark
    stays: ``ἑγώ``, ``ἐγώ``, and ``ἐγὼ`` share a key, but ``Ἀπόλλων᾽`` (elided
    ``Ἀπόλλωνα``) does not compete with ``Ἀπόλλων``.
    """
    nfd = unicodedata.normalize("NFD", exact_form_key(form))
    return unicodedata.normalize(
        "NFC", "".join(char for char in nfd if not unicodedata.combining(char))
    )


def grc_orthography_reason(form: str) -> str | None:
    """Return why ``form`` cannot be a standalone polytonic word.

    The checks are structural, not corpus-frequency heuristics.  They reject
    punctuation-bearing tokens and generator joins which leave a simplex
    breathing inside a compound, while retaining initial diphthongs and the
    common crasis shapes (``κἀγώ``, ``τοὔνομα``).
    """
    nfd = unicodedata.normalize("NFD", form)
    acute_form = unicodedata.normalize(
        "NFC", "".join("\u0301" if char == "\u0300" else char for char in nfd)
    )
    # A grave is not itself a rejected spelling.  It is the contextual form
    # of an oxytone and must remain available (for example στὸ/στὰ); only its
    # acute twin may be an explicitly reviewed rejection (στό/στά).
    if form in GRC_REJECT_FORMS:
        return "explicit_reject"
    if form in GRC_CRASIS_EXCEPTIONS:
        return None
    if form in BARE_ELISION_STEMS or acute_form in BARE_ELISION_STEMS:
        return "bare_elision"
    if not form:
        return "empty"

    for index, char in enumerate(form):
        if _is_greek_letter(char) and ord(char) not in GREEK_EDITORIAL_SIGNS:
            continue
        if (unicodedata.category(char) == "Mn"
                and ord(char) in ALLOWED_GREEK_COMBINING_MARKS):
            continue
        if (ord(char) in SPACING_DIACRITICS
                and index in {0, len(form) - 1}):
            continue
        return "nonword_character"

    if not has_required_initial_breathing(form):
        return "missing_initial_breathing"

    bases = _greek_bases(form)
    if not bases:
        return "no_greek_letters"
    for index, (base, _marks) in enumerate(bases):
        if base == "ς" and index != len(bases) - 1:
            return "medial_final_sigma"
    if bases[-1][0] == "σ":
        return "final_nonfinal_sigma"

    syllables = _syllable_base_indexes(bases)
    first_nucleus = set(syllables[0]) if syllables else set()
    breathing_indexes = [
        index for index, (_base, marks) in enumerate(bases)
        if marks & BREATHING_MARKS
    ]
    # A breathing may occur on the first syllable's nucleus: on its first
    # vowel, on the second vowel of an initial diphthong, or on the first
    # vowel after a consonant in crasis. Anything later is a joined-word
    # artifact. This also rejects a second breathing rather than allowing a
    # stale simplex mark to survive a generated compound.
    allowed_breathing_indexes = first_nucleus
    if bases[0][0].lower() == "ρ":
        allowed_breathing_indexes = allowed_breathing_indexes | {0}
    if (len(breathing_indexes) > 1
            or any(index not in allowed_breathing_indexes
                   for index in breathing_indexes)):
        return "internal_breathing"

    base_to_syllable = {
        base_index: syllable_index
        for syllable_index, members in enumerate(syllables)
        for base_index in members
    }
    tonal_syllables: list[tuple[int, set[int]]] = []
    tonal_per_syllable: dict[int, int] = defaultdict(int)
    for base_index, (_base, marks) in enumerate(bases):
        tonal = set(marks & TONAL_MARKS)
        if not tonal:
            continue
        if len(tonal) > 1:
            return "duplicate_tonal_marks"
        syllable_index = base_to_syllable.get(base_index)
        if syllable_index is None:
            return "tonal_mark_on_consonant"
        tonal_syllables.append((syllable_index, tonal))
        tonal_per_syllable[syllable_index] += len(tonal)
    if any(count > 1 for count in tonal_per_syllable.values()):
        return "duplicate_tonal_marks_on_syllable"
    # A grave is the contextual form of a final acute, so it is written only
    # on the ultima. Earlier graves come from joined words (καὶτοὺς) or from
    # a rough breathing misread as a grave (ὓστερον for ὕστερον).
    if any(0x0300 in tonal and syllable_index != len(syllables) - 1
           for syllable_index, tonal in tonal_syllables):
        return "grave_before_ultima"
    # An elided oxytone throws its accent back as an acute, and elided
    # prepositions and conjunctions lose it (Smyth 174): never γὰρ᾽ or ἂλλ᾽.
    if (ord(form[-1]) in SPACING_DIACRITICS
            and any(0x0300 in tonal for _index, tonal in tonal_syllables)):
        return "grave_on_elided_word"

    if (not tonal_syllables
            and not any(ord(char) in SPACING_DIACRITICS for char in form)
            and form.lower() not in AG_FUNCTION_WORDS):
        if not syllables:
            return "no_vowel"
        return "missing_tonal_accent"
    if len(tonal_syllables) > 1:
        if not has_enclitic_second_accent(bases, syllables, tonal_syllables):
            return "misplaced_second_accent"
        return None
    fused = fused_enclitic_syllables(bases, syllables)
    for syllable_index, tonal in tonal_syllables:
        syllables_after = len(syllables) - syllable_index - 1
        if form and ord(form[-1]) in SPACING_DIACRITICS:
            syllables_after += 1
        # A fused enclitic leaves the host's own accent in place, so the
        # window is measured on the host: οὗτινος, ᾧτινι, τοῖσιδε.
        if fused and syllable_index < len(syllables) - fused:
            syllables_after -= fused
        if syllables_after > 2:
            return "accent_before_antepenult"
        if 0x0342 in tonal and syllables_after > 1:
            return "circumflex_before_penult"
    return None


# The fused -τις of ὅστις, whose first part keeps its own accent: οὗτινος,
# ᾧτινι. Fused -δε is not included: a locative such as πόλεμόνδε takes the
# enclitic's second accent, while the demonstratives of ὅδε do not (τοῖσιδε),
# so those are listed in GRC_CRASIS_EXCEPTIONS instead.
FUSED_ENCLITIC_TAILS = (
    ("τινος", 2), ("τινι", 2), ("τινα", 2), ("τινες", 2), ("τινων", 2),
    ("τισιν", 2), ("τισι", 2), ("τινας", 2), ("τινε", 2), ("τινοιν", 3),
)


def fused_enclitic_syllables(
    bases: list[tuple[str, frozenset[int]]],
    syllables: list[tuple[int, ...]],
) -> int:
    """Syllables at the end of the word that belong to a fused enclitic."""
    letters = "".join(base for base, _marks in bases).lower()
    letters = letters.replace("ς", "σ")
    for tail, count in FUSED_ENCLITIC_TAILS:
        tail = tail.replace("ς", "σ")
        if letters.endswith(tail) and len(syllables) > count:
            # Only the unaccented tail letters may form the enclitic.
            start = len(bases) - len(tail)
            if not any(marks & TONAL_MARKS for _b, marks in bases[start:]):
                return count
    return 0


# Enclitics that fuse with a preceding word and keep its second accent on
# the syllable before them: Αἴγυπτόνδε, τοῖόνδε, οἷοίπερ, Ionic ὁκοῖόντι.
FUSED_ENCLITIC_ENDINGS = ("δε", "ζε", "περ", "γε", "τε", "τι", "τις", "τοι")


def has_enclitic_second_accent(
    bases: list[tuple[str, frozenset[int]]],
    syllables: list[tuple[int, ...]],
    tonal_syllables: list[tuple[int, set[int]]],
) -> bool:
    """Return whether a word with two accents has the enclitic shape.

    Only a proparoxytone or properispomenon takes a second accent, an acute
    (Smyth 183d): on its own ultima before a separate enclitic (ἄνθρωπόν,
    δῶρόν), or on the syllable before an enclitic fused onto it
    (Αἴγυπτόνδε). A grave never occurs in such a word, and no word has three
    accents; anything else is two words run together or a misaccented copy.
    """
    if len(tonal_syllables) != 2:
        return False
    (first, first_mark), (second, second_mark) = tonal_syllables
    if first_mark not in ({0x0301}, {0x0342}) or second_mark != {0x0301}:
        return False
    # The host ends with the syllable that carries the second accent.
    host_after_first = second - first
    if not ((first_mark == {0x0301} and host_after_first == 2)
            or (first_mark == {0x0342} and host_after_first == 1)):
        return False
    if second == len(syllables) - 1:
        return True
    # A fused enclitic follows: the letters after the accented syllable's
    # vowel nucleus must spell one of the fused endings.
    tail_start = syllables[second][-1] + 1
    tail = "".join(base for base, _marks in bases[tail_start:]).lower()
    tail = tail.replace("ς", "σ")
    return any(
        tail.endswith(ending.replace("ς", "σ"))
        and len(tail) - len(ending) <= 1
        for ending in FUSED_ENCLITIC_ENDINGS
    )


def grc_pair_orthography_reason(form: str, lemma: str) -> str | None:
    """Apply word-level checks which need the form's lemma provenance."""
    reason = grc_orthography_reason(form)
    if reason is not None:
        return reason
    if any(ord(char) in SPACING_DIACRITICS for char in form):
        return None
    bases = _greek_bases(form)
    final = bases[-1][0].lower()
    if (final not in GREEK_VOWELS | frozenset("νρςξψ")
            and form not in HOMERIC_SHORT_PREPOSITIONS
            and form not in CONSONANT_FINAL_WORDS
            and form.lower() not in AG_FUNCTION_WORDS
            and exact_form_key(form) != exact_form_key(lemma)):
        return "impossible_final_consonant"
    return None


def contextual_acute(form: str) -> str:
    """Fold contextual grave accents to their citation/before-pause acute."""
    nfd = unicodedata.normalize("NFD", form)
    return unicodedata.normalize(
        "NFC", "".join("\u0301" if char == "\u0300" else char for char in nfd)
    )


def add_contextual_acute_twins(
    form_lemma: list[tuple[str, str]],
    reviewed: set[str] | frozenset[str] = frozenset(),
) -> tuple[list[tuple[str, str]], int]:
    """Add the acute counterpart of every retained grave surface form.

    This is a defined Greek orthographic normalization, not a general accent
    rewrite: a grave is the contextual spelling of an oxytone acute.  Keeping
    the acute twin is required for citation spelling and before punctuation.
    The twin of a ``reviewed`` grave inherits its review, so it is checked
    only against the lemma-free structural rules (the baseline ``Ὃκ`` keeps
    ``Ὅκ`` whatever lemma lookup.db assigns it).
    """
    out = list(form_lemma)
    seen = set(form_lemma)
    added = 0
    for form, lemma in form_lemma:
        acute = contextual_acute(form)
        pair = (acute, lemma)
        reason = (
            grc_orthography_reason(acute) if form in reviewed
            else grc_pair_orthography_reason(acute, lemma)
        )
        if (acute != form and pair not in seen and reason is None):
            seen.add(pair)
            out.append(pair)
            added += 1
    return out, added


def is_productive_second_accent(
    form: str,
    exact_freq: dict[str, int],
) -> bool:
    """Return whether ``form`` is a regular enclitic second-accent spelling.

    The form must have exactly two tonal marks on separate syllables, with an
    acute on the ultima.  Removing that final acute must yield an independently
    attested host spelling, as in ``θάλασσάν`` -> ``θάλασσαν``.  Only a
    proparoxytone (``ἄνθρωπός τις``) or a properispomenon (``δῶρόν τι``)
    takes the second accent (Smyth 183d); a paroxytone host keeps a single
    accent, so ``λύκοί`` and ``ὕπνῴ`` are not productive spellings.
    """
    nfd = list(unicodedata.normalize("NFD", form))
    tonal_indexes = [
        index for index, char in enumerate(nfd) if ord(char) in TONAL_MARKS
    ]
    if len(tonal_indexes) != 2 or nfd[tonal_indexes[-1]] != "\u0301":
        return False
    bases = _greek_bases(form)
    syllables = _syllable_base_indexes(bases)
    if not syllables:
        return False
    last_tonal = tonal_indexes[-1]
    base_before = ""
    for char in nfd[:last_tonal]:
        if unicodedata.category(char) != "Mn":
            base_before = char
    if base_before not in GREEK_VOWELS:
        return False
    # The final tonal mark must attach to the word's ultima nucleus.
    last_vowel = max(
        index for members in syllables for index in members
    )
    tonal_base_index = -1
    seen_bases = -1
    for char in nfd[:last_tonal]:
        if _is_greek_letter(char):
            seen_bases += 1
            tonal_base_index = seen_bases
    if tonal_base_index not in syllables[-1] or last_vowel not in syllables[-1]:
        return False
    del nfd[last_tonal]
    host = unicodedata.normalize("NFC", "".join(nfd))
    if not is_proparoxytone_or_properispomenon(host):
        return False
    return exact_freq_lookup(host, exact_freq) > 0


def is_proparoxytone_or_properispomenon(form: str) -> bool:
    """Return whether ``form`` has one accent: an acute on the antepenult or
    a circumflex on the penult."""
    bases = _greek_bases(form)
    syllables = _syllable_base_indexes(bases)
    base_to_syllable = {
        base_index: syllable_index
        for syllable_index, members in enumerate(syllables)
        for base_index in members
    }
    accents = [
        (base_to_syllable.get(index), marks & TONAL_MARKS)
        for index, (_base, marks) in enumerate(bases)
        if marks & TONAL_MARKS
    ]
    if len(accents) != 1 or accents[0][0] is None:
        return False
    syllable_index, tonal = accents[0]
    syllables_after = len(syllables) - syllable_index - 1
    return ((tonal == {0x0301} and syllables_after == 2)
            or (tonal == {0x0342} and syllables_after == 1))


def required_treebank_support(exact: int, dominant: int) -> int:
    """Treebank tokens a new spelling needs beside a more common spelling.

    ``exact`` is the new spelling's corpus count and ``dominant`` the count of
    the most frequent well-formed spelling with the same letters. The weaker
    the new spelling's share, the more likely it is a corrupted copy, so the
    required GLAUx or Diorisis support rises as the share falls.
    """
    if dominant < PROFILE_VARIANT_DOMINANCE_MIN or exact >= dominant:
        return 0
    share = exact / dominant
    for below, required in PROFILE_VARIANT_TREEBANK_FLOORS:
        if share < below:
            return required
    return 0


def filter_dominated_spelling_variants(
    form_lemma: list[tuple[str, str]],
    exact_freq: dict[str, int],
    dominant_freq: dict[str, int],
    treebank_freq: dict[str, int],
    compatibility_forms: set[str],
    protected_forms: set[str] | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Reject weak new respellings of strongly attested spellings.

    A respelling differs from a more common spelling only in its marks:
    accent, breathing, diaeresis, or iota subscript (``ἑγώ`` beside
    ``ἐγώ``, ``ταΐς`` beside ``ταῖς``, ``ὄσα`` beside ``ὅσα``). OCR and
    edition typos produce such spellings in proportion to the common word's
    frequency, so a raw corpus count cannot vouch for them; support in the
    lemmatized treebanks can. Genuine dialect spellings such as Doric ``τᾷ``
    carry that support. The reviewed April surface, pinned forms, and
    productive enclitic second accents are exempt.
    """
    protected = compatibility_forms | (protected_forms or set())
    kept: list[tuple[str, str]] = []
    rejected: list[tuple[str, str]] = []
    for pair in form_lemma:
        form, _lemma = pair
        if (form in protected
                or is_productive_second_accent(form, exact_freq)):
            kept.append(pair)
            continue
        required = required_treebank_support(
            exact_freq_lookup(form, exact_freq),
            dominant_freq.get(mark_skeleton(form), 0),
        )
        if required and exact_freq_lookup(form, treebank_freq) < required:
            rejected.append(pair)
        else:
            kept.append(pair)
    return kept, rejected


def filter_unattested_new_forms(
    form_lemma: list[tuple[str, str]],
    exact_freq: dict[str, int],
    compatibility_forms: set[str],
    protected_forms: set[str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Require reviewable evidence for every form added after April.

    Historical paradigm generators contain valuable sparse morphology, but
    they also produce joins which are orthographically possible and therefore
    cannot be caught by spelling-shape checks alone. A new form must be
    attested exactly, belong to a pinned citation/textbook/grammar fixture, or
    be a productive enclitic second-accent spelling. The reviewed April
    surface is never subjected to this new evidence requirement.
    """
    kept: list[tuple[str, str]] = []
    rejected: list[tuple[str, str]] = []
    for pair in form_lemma:
        form, _lemma = pair
        acute = contextual_acute(form)
        if (form in compatibility_forms
                or form in protected_forms
                or acute in protected_forms
                or exact_freq_lookup(form, exact_freq) > 0
                or is_productive_second_accent(form, exact_freq)):
            kept.append(pair)
        else:
            rejected.append(pair)
    return kept, rejected


def filter_grc_orthography(
    form_lemma: list[tuple[str, str]],
    compatibility_forms: set[str] | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Reject structurally impossible standalone polytonic spellings."""
    compatibility = compatibility_forms or set()
    kept: list[tuple[str, str]] = []
    rejected: list[tuple[str, str]] = []
    for pair in form_lemma:
        if pair[0] in compatibility or grc_pair_orthography_reason(*pair) is None:
            kept.append(pair)
        else:
            rejected.append(pair)
    return kept, rejected


def filter_new_tonos_structural_forms(
    form_lemma: list[tuple[str, str]],
    compatibility_forms: set[str],
    protected_forms: set[str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Keep reviewed baseline forms but block new Hunspell junk classes.

    Hunspell's zero-strip flags can expose a real-looking dictionary stem as
    a standalone word even when it was only a generator fragment. The
    compiled Tonos gate treats new lower-case consonant-final fragments and
    smooth-initial upsilon forms as structural junk, so apply those rules to
    the post-April surface while retaining reviewed baseline and explicit
    textbook/citation/function-word evidence.
    """
    reviewed = compatibility_forms | protected_forms
    kept: list[tuple[str, str]] = []
    rejected: list[tuple[str, str]] = []
    for pair in form_lemma:
        if pair[0] not in reviewed and new_form_structural_reason(pair[0]):
            rejected.append(pair)
        else:
            kept.append(pair)
    return kept, rejected


def new_form_structural_reason(form: str) -> str | None:
    """Return why a new (post-April) form is structural junk downstream.

    These two classes are stricter than :func:`grc_orthography_reason`
    because they would also reject a few valid baseline words: a lower-case
    word ending in a consonant no complete word ends in (a stem that lost its
    elision mark, exposed by a zero-strip affix flag), and a smooth breathing
    on an initial upsilon, which only Lesbian Aeolic writes.
    """
    bases = _greek_bases(form)
    if not bases:
        return None
    # An elided word legitimately ends in a consonant plus ᾽ (γ᾽, παρ᾽).
    if ord(form[-1]) not in SPACING_DIACRITICS:
        final = bases[-1][0].lower()
        if (not form[0].isupper()
                and final not in GREEK_VOWELS | frozenset("νρςξψ")
                and form.lower() not in AG_FUNCTION_WORDS
                and form not in HOMERIC_SHORT_PREPOSITIONS
                and form not in CONSONANT_FINAL_WORDS):
            return "truncated_fragment"
    first, marks = bases[0]
    if first.lower() == "υ" and 0x0313 in marks:
        return "smooth_initial_upsilon"
    return None


def grc_pinned_forms(
    canonical_forms: set[str],
    top_lsj9_lemmas: set[str],
    textbook_forms: set[str],
    lm_head_forms: set[str],
) -> set[str]:
    """Forms exempt from the corpus-evidence filters: curated canonical
    forms, top citation headwords, textbook paradigms, the reviewed LM head,
    and the closed grammatical lists. Textbook cells and closed-list forms
    are not also pinned to frequency bucket C (see run_export)."""
    return (
        canonical_forms | top_lsj9_lemmas | textbook_forms | lm_head_forms
        | set(GRC_CLOSED_LIST_FORMS)
    )


def add_grc_reviewed_forms(
    form_lemma: list[tuple[str, str]],
    export_overrides: dict[str, str],
    textbook_paradigms: dict[str, list[str]],
    compatibility_forms: set[str],
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Add closed-list, textbook-paradigm, and reviewed April forms.

    These bypass the lookup and frequency gates, so they are added after
    those gates and before sanitization and the structural filters.
    """
    out = list(form_lemma)
    added = {"overrides": 0, "textbook": 0, "compatibility": 0}
    existing_forms = {form for form, _lemma in out}
    for form, lemma in export_overrides.items():
        if form not in existing_forms:
            out.append((form, lemma))
            added["overrides"] += 1
    existing_pairs = set(out)
    for lemma, forms in textbook_paradigms.items():
        for form in forms:
            if (form, lemma) not in existing_pairs:
                existing_pairs.add((form, lemma))
                out.append((form, lemma))
                added["textbook"] += 1
    existing_forms = {form for form, _lemma in out}
    for form in sorted(compatibility_forms - existing_forms):
        out.append((form, form))
        added["compatibility"] += 1
    return out, added


def finalize_grc_pairs(
    form_lemma: list[tuple[str, str]],
    *,
    evidence: FormProfileEvidence,
    compatibility_forms: set[str],
    textbook_forms: set[str],
    export_overrides: dict[str, str],
    protected_forms: set[str],
) -> tuple[list[tuple[str, str]], dict]:
    """Apply the grc structural and evidence filters in release order.

    The reviewed April surface (``compatibility_forms``) is exempt from every
    rule a structural reviewer has already passed; ``protected_forms`` (top
    citation headwords, textbook paradigms, closed lists) is exempt from the
    corpus-evidence rules. Acute twins are generated last. A twin shares its
    grave's exact-form key and so its corpus evidence, and the twin of a
    reviewed baseline grave inherits that review: the grave and the acute are
    one word, and the downstream oxytone check requires both. Twins are only
    checked again for structure.
    """
    structural_protected = (
        textbook_forms | set(export_overrides) | set(AG_FUNCTION_WORDS)
    )
    report: dict = {}
    form_lemma, invalid = filter_grc_orthography(
        form_lemma, compatibility_forms | structural_protected
    )
    report["invalid"] = dict(sorted(
        Counter(
            grc_pair_orthography_reason(form, lemma) or "unknown"
            for form, lemma in invalid
        ).items()
    ))
    form_lemma, new_structural = filter_new_tonos_structural_forms(
        form_lemma, compatibility_forms, structural_protected,
    )
    report["new_structural"] = len(new_structural)
    form_lemma, unattested = filter_unattested_new_forms(
        form_lemma, evidence.exact, compatibility_forms, protected_forms,
    )
    report["unattested"] = len(unattested)
    form_lemma, dominated = filter_dominated_spelling_variants(
        form_lemma, evidence.exact, evidence.dominant, evidence.treebank,
        compatibility_forms, protected_forms,
    )
    report["dominated"] = len(dominated)
    before_twins = len(form_lemma)
    form_lemma, acute_twins = add_contextual_acute_twins(
        form_lemma, compatibility_forms
    )
    report["acute_twins"] = acute_twins
    form_lemma, _ = filter_new_tonos_structural_forms(
        form_lemma, compatibility_forms, structural_protected,
    )
    report["twin_dropped"] = before_twins + acute_twins - len(form_lemma)
    return form_lemma, report


def strip_accents(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize(
        "NFC", "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    ).lower()


def freq_lookup(form: str, freq_map: dict[str, int]) -> int:
    """Look up a form's corpus count under several normalization keys.

    The MG frequency map stores keys with monotonic accents preserved
    and lowercase (e.g. 'παπαδοπούλου'), while the AG corpus_freq map
    uses fully-stripped lowercase keys. We try: the form as-is,
    lowercased, and accent-stripped + lowercased, returning the first
    non-zero hit (or 0).
    """
    if form in freq_map:
        return freq_map[form]
    lower = form.lower()
    if lower in freq_map:
        return freq_map[lower]
    stripped = strip_accents(form)
    if stripped in freq_map:
        return freq_map[stripped]
    return 0


def exact_form_key(form: str) -> str:
    """Accent-preserving key used for Ancient Greek form attestation.

    Contextual grave is folded to acute and case is folded, but breathings,
    circumflexes, diaereses, vowel length, and iota subscripts are preserved.
    In particular, ``αυτός`` cannot borrow attestation from ``αὐτός``.
    """
    form = canonicalize_final_elision(sanitize_form(form))
    nfd = unicodedata.normalize("NFD", form)
    acute = "".join("\u0301" if char == "\u0300" else char for char in nfd)
    # lower(), unlike casefold(), preserves U+0345 COMBINING GREEK
    # YPOGEGRAMMENI (iota subscript).
    return unicodedata.normalize("NFC", acute).lower()


def exact_freq_lookup(form: str, freq_map: dict[str, int]) -> int:
    return freq_map.get(exact_form_key(form), 0)


def freq_bucket(count: int) -> str:
    if count >= 1000:
        return "C"
    if count >= 100:
        return "M"
    if count >= 1:
        return "R"
    return "X"


# Lemma-aggregate thresholds. A lemma whose combined corpus count
# across all its inflected forms exceeds the C threshold is considered
# culturally common enough that every one of its polytonic forms should
# land in bucket C, even if individual inflections are rare. This
# corrects for the corpus-mass dilution effect where highly-inflected
# classical lemmas (e.g. Ζεύς, λόγος, ἀείδω) have their frequency mass
# spread across 100+ surface forms. Numbers were chosen so the C promo
# triggers for lemmas that are unambiguously famous in the classical
# canon; they are higher than per-form thresholds to keep the bucket
# precise (roughly: Zeus-famous, not hapax-famous).
LEMMA_AGG_C_MIN = 20_000
LEMMA_AGG_M_MIN = 5_000


def load_canonical_ag_sets() -> tuple[set[str], set[str]]:
    """Load (canonical_forms, canonical_lemmas) from CANONICAL_AG_FORMS.

    Both sets contain NFC-normalized polytonic strings. The forms set
    is matched by exact surface equality (so monotonic variants do not
    get promoted along with their polytonic siblings). The lemmas set
    is matched against each (form, lemma) pair's lemma text; any form
    of a canonical lemma gets promoted to at least C.
    """
    if not CANONICAL_AG_FORMS.exists():
        return set(), set()
    with open(CANONICAL_AG_FORMS, encoding="utf-8") as f:
        raw = json.load(f)
    forms = {unicodedata.normalize("NFC", s)
             for s in raw.get("forms_c", [])}
    lemmas = {unicodedata.normalize("NFC", s)
              for s in raw.get("lemmas_c", [])}
    return forms, lemmas


def compute_lemma_totals(
    form_lemma: list[tuple[str, str]],
    freq_map: dict[str, int],
) -> dict[str, int]:
    """Return lemma -> sum of corpus counts across all its surface forms.

    Each form's count is looked up via freq_lookup (which tries NFC,
    lowercase, and accent-stripped lowercase keys). A form shared by
    multiple lemmas is counted once per lemma, which is a mild double-
    count, but acceptable because we only use the aggregate to cross a
    fairly conservative promotion threshold, not as a precise measure.
    """
    by_lemma: dict[str, set[str]] = defaultdict(set)
    for form, lemma in form_lemma:
        by_lemma[lemma].add(form)
    totals: dict[str, int] = {}
    for lemma, forms in by_lemma.items():
        t = 0
        for f in forms:
            t += freq_lookup(f, freq_map)
        totals[lemma] = t
    return totals


def bucket_for(
    form: str,
    lemma: str,
    form_count: int,
    lemma_total: int,
    canonical_forms: set[str],
    canonical_lemmas: set[str],
) -> str:
    """Choose a frequency bucket taking canonical seed and lemma total
    into account. Rules, from strongest to weakest:

      1. If the exact polytonic form is in the canonical seed -> C.
      2. If the lemma is in the canonical seed AND this particular form
         carries any polytonic mark (breathing, circumflex, grave, iota
         subscript) -> C. Acute-only or fully-stripped forms of a
         canonical lemma are NOT promoted, so that a polytonic form
         always outranks a monotonic sibling at tiebreak.
      3. Lemma-aggregate: count across all forms of this lemma >= 20K -> C;
         >= 5K -> at least M. Same polytonic-only gating applies.
      4. Otherwise use the per-form corpus count with the default edges.
    """
    if form in canonical_forms:
        return "C"
    form_is_polytonic = has_polytonic(form)
    if lemma in canonical_lemmas and form_is_polytonic:
        return "C"
    if lemma_total >= LEMMA_AGG_C_MIN and form_is_polytonic:
        return "C"
    base = freq_bucket(form_count)
    if (lemma_total >= LEMMA_AGG_M_MIN and form_is_polytonic
            and base == "R"):
        return "M"
    return base


def load_freq_maps() -> tuple[dict[str, int], dict[str, int]]:
    """Return (mg_freq, ag_freq). Both map stripped-lowercase form to count."""
    mg_freq: dict[str, int] = {}
    ag_freq: dict[str, int] = {}

    if MG_FORM_FREQ.exists():
        with open(MG_FORM_FREQ, encoding="utf-8") as f:
            raw = json.load(f)
        for form, count in raw.items():
            mg_freq[form] = count

    if CORPUS_FREQ.exists():
        with open(CORPUS_FREQ, encoding="utf-8") as f:
            raw = json.load(f)
        for form, counts in raw.get("forms", {}).items():
            ag_freq[form] = counts[0] if counts else 0

    return mg_freq, ag_freq


class FormProfileEvidence(NamedTuple):
    """Exact-form corpus evidence from ``form_profile.db``.

    ``exact`` maps :func:`exact_form_key` to the corpus count: the work-
    deduplicated total, or the largest single-source count when that is
    larger;
    ``treebank`` maps the same key to its larger GLAUx or Diorisis count (the
    two treebanks annotate largely the same texts, so they are not summed);
    ``dominant`` maps :func:`mark_skeleton` to the count of the most frequent
    well-formed spelling with those letters.
    """
    exact: dict[str, int]
    dominant: dict[str, int]
    treebank: dict[str, int]
    metadata: dict[str, str]


def load_form_profile_freq() -> FormProfileEvidence:
    """Load full accent-preserving corpus counts and provenance.

    Unlike the language-model vocabulary artifact, this database keeps forms
    with a single occurrence and therefore covers the sparse inflectional
    tail.  We recompute :func:`exact_form_key` from ``forms.form`` instead of
    reading ``form_norm``: the profile's runtime key uses ``casefold()``, which
    folds final sigma and iota subscript, while Hunspell admission deliberately
    preserves both distinctions.
    """
    if not FORM_PROFILE_DB.exists():
        return FormProfileEvidence({}, {}, {}, {})
    conn = sqlite3.connect(str(FORM_PROFILE_DB))
    try:
        metadata = dict(conn.execute("SELECT key, value FROM meta"))
        rows = conn.execute(
            "SELECT f.form, p.total_count, p.source_counts_json "
            "FROM forms f JOIN form_profile p USING(form_id)"
        )
        frequencies: dict[str, int] = defaultdict(int)
        by_treebank: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for form, count, source_counts in rows:
            key = exact_form_key(form)
            sources = json.loads(source_counts or "{}")
            # total_count is deduplicated by work, so a spelling found only
            # in a lower-priority source's copy of a work that a higher-
            # priority source claims has total 0 (199,082 rows). Its own
            # source count still attests it.
            frequencies[key] += max(
                int(count), max((int(n) for n in sources.values()), default=0)
            )
            for source in TREEBANK_PROFILE_SOURCES:
                if sources.get(source):
                    by_treebank[key][source] += int(sources[source])
    finally:
        conn.close()
    treebank = {
        key: max(counts.values()) for key, counts in by_treebank.items()
    }
    dominant: dict[str, int] = {}
    for form, count in frequencies.items():
        if grc_orthography_reason(form) is not None:
            continue
        skeleton = mark_skeleton(form)
        dominant[skeleton] = max(dominant.get(skeleton, 0), count)
    return FormProfileEvidence(frequencies, dominant, treebank, metadata)


def load_top_lsj9_lemmas(limit: int = 2000) -> set[str]:
    """Return the highest-frequency LSJ9 citation headwords."""
    if not LSJ9_FREQUENCY.exists():
        return set()
    raw = json.loads(LSJ9_FREQUENCY.read_text(encoding="utf-8"))
    ranked = sorted(
        ((lemma, int(count)) for lemma, count in raw.items()),
        key=lambda item: (-item[1], item[0]),
    )[:limit]
    normalized: set[str] = set()
    for lemma, _count in ranked:
        # LSJ9 inserts hyphens at inflectional boundaries and records vowel
        # quantity with combining breve/macron. Neither belongs in ordinary
        # citation spelling or in lookup.db's lemma text.
        nfd = unicodedata.normalize("NFD", lemma.replace("-", ""))
        plain = "".join(
            char for char in nfd if ord(char) not in {0x0304, 0x0306}
        )
        normalized.add(unicodedata.normalize("NFC", plain))
    return normalized


def load_lm_head_required_forms() -> set[str]:
    """Return the reviewed real words among the LM's most frequent forms,
    in the spelling the dictionary stores (contextual grave read as acute,
    final elision mark canonicalized)."""
    if not LM_HEAD_FIXTURE.exists() or not LM_HEAD_EXCLUSIONS.exists():
        return set()
    fixture = json.loads(LM_HEAD_FIXTURE.read_text(encoding="utf-8"))
    excluded = {
        row["form"] for row in json.loads(
            LM_HEAD_EXCLUSIONS.read_text(encoding="utf-8")
        )["exclusions"]
    }
    return {
        canonicalize_final_elision(
            contextual_acute(unicodedata.normalize("NFC", row["form"]))
        )
        for row in fixture["forms"]
        if row["form"] not in excluded
    }


def load_grc_compatibility_forms() -> tuple[set[str], dict]:
    """Load the structurally reviewed April acceptance baseline."""
    if not GRC_COMPATIBILITY_FORMS.exists():
        return set(), {}
    with gzip.open(GRC_COMPATIBILITY_FORMS, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"unsupported compatibility fixture: {GRC_COMPATIBILITY_FORMS}"
        )
    return set(payload.get("forms", [])), payload


def load_grc_textbook_forms() -> tuple[set[str], dict]:
    """Load the pinned standard paradigms used by the artifact gate."""
    if not GRC_TEXTBOOK_FORMS.exists():
        return set(), {}
    with gzip.open(GRC_TEXTBOOK_FORMS, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported textbook fixture: {GRC_TEXTBOOK_FORMS}")
    paradigms = payload.get("paradigms", {})
    forms = {
        form for lemma_forms in paradigms.values() for form in lemma_forms
    }
    return forms, payload


def load_independent_ag_headwords() -> set[str]:
    """Load exact-form keys that independently establish AG headwords."""
    headwords: set[str] = set()
    for path in INDEPENDENT_AG_HEADWORD_PATHS:
        if not path.exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        values = raw if isinstance(raw, list) else raw.keys()
        headwords.update(
            exact_form_key(value)
            for value in values
            if isinstance(value, str) and value
        )
    return headwords


def is_bare_elision_fallback(
    form: str,
    lemma: str,
    elided_pair_keys: set[tuple[str, str]],
    independent_headwords: set[str],
) -> bool:
    """Return whether *form* is an apostrophe-less elision fallback.

    The lookup intentionally accepts tolerant stems such as ``ἀφ`` for
    ``ἀφ᾽``. They are unsuitable dictionary words. A bare spelling is rejected
    when lookup also contains its marked counterpart under the same lemma and
    neither DGE nor Cunliffe establishes the bare spelling as a headword. This
    keeps genuine collisions such as ``ἄν`` (whose marked spelling belongs
    to ``ἀνά``), while covering the class beyond a fixed list.
    """
    clean = canonicalize_final_elision(sanitize_form(form))
    if not clean or clean.endswith("\u1fbd"):
        return False
    bases = _greek_bases(clean)
    if (not bases
            or bases[-1][0].lower() in GREEK_VOWELS | frozenset("νρςξψ")):
        return False
    key = exact_form_key(clean)
    if (clean.lower() in AG_FUNCTION_WORDS
            or key in independent_headwords
            or (key, exact_form_key(lemma)) not in elided_pair_keys):
        return False
    return True


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        return out
    except Exception:
        return "unknown"


def read_version_file() -> str:
    vp = ROOT / "VERSION"
    if vp.exists():
        return vp.read_text().strip()
    return "0.0.0"


# Hunspell flag format: we use FLAG num mode so each rule gets an
# integer flag 1..65535. Hunspell limits combined flags on a word via
# comma separation (e.g. word/123,456). This gives us >64K flags,
# plenty for Greek's highly varied inflection signatures.
def gen_flag_names():
    # Reserve 1..9 as system flags (we use none currently but leave
    # headroom for future). Start at 1000 so our rule flags are easy
    # to distinguish from any reserved values.
    i = 1000
    while i < 65500:
        yield str(i)
        i += 1


def select_forms(
    conn: sqlite3.Connection,
    variant: str,
    keep_lemmas: set[str] | None = None,
    attestation_freq: dict[str, int] | None = None,
) -> list[tuple[str, str]]:
    """Return [(form, lemma_text)] for the given variant.

    variant='el'  -> Modern Greek monotonic forms. This is the union of:
                     (a) every src='el' row (native MG overrides), and
                     (b) every monotonic form from src='grc' + lang='all'.
                     Rationale: a lot of shared vocabulary (articles,
                     common verbs, proper names) is stored on the 'grc'
                     side with src='grc' even though it is perfectly
                     valid Modern Greek. Excluding those would miss
                     words like 'δεν', 'καί', 'σκύλος'.
    variant='grc' -> all src='grc' or language-shared rows containing
                     orthographic marks. This is the Ancient + Medieval
                     vocabulary with diacritics. Language-shared rows matter
                     because a headword spelling shared with Modern Greek can
                     be owned by src='el' even when its inflections are grc.
                     We exclude stripped fallback keys.
                     If keep_lemmas is provided, any lemma whose text
                     is in that set is kept even if none of its forms
                     carry a breathing/circumflex/grave/iota-subscript
                     mark. This rescues canonical AG lemmas whose
                     spellings are entirely acute-only (e.g. Πλάτων,
                     Μένανδρος, πόλεμος) which would otherwise be
                     excluded as 'pure-monotonic, not AG'.
                     If attestation_freq is provided, a corpus-attested lemma
                     is also kept when all its forms are acute-only.
    """
    cur = conn.cursor()

    if variant == "el":
        # MG-specific overrides from lookup.db
        rows = cur.execute(
            """
            SELECT k.form, l.text
            FROM lookup k JOIN lemmas l ON k.lemma_id = l.id
            WHERE k.src = 'el'
            """
        )
        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str]] = []
        for form, lemma in rows:
            if has_polytonic(form):
                continue
            key = (form, lemma)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    if variant == "grc":
        # AG vocabulary. The DB stores, per AG lemma, three
        # parallel copies of each form: truly-polytonic, acute-only,
        # and fully-stripped. For the AG keyboard artifact we want:
        #
        #  - every truly-polytonic form (breathing, circumflex, grave,
        #    iota subscript), and
        #  - every acute-only form whose lemma has a polytonic sibling
        #    in its paradigm (so σοφίας rides along with σοφίᾳ).
        #
        # We skip fully-stripped forms (they are just lookup fallback
        # keys and would bloat the artifact without adding correct
        # AG orthography).
        rows = cur.execute(
            """
            SELECT k.form, l.text
            FROM lookup k
            JOIN lemmas l ON k.lemma_id = l.id
            JOIN (
                SELECT DISTINCT lemma_id FROM lookup WHERE src = 'grc'
            ) grc_lemmas ON grc_lemmas.lemma_id = k.lemma_id
            WHERE k.src = 'grc' OR k.lang = 'all'
            """
        )
        by_lemma: dict[str, list[str]] = defaultdict(list)
        elided_pair_keys: set[tuple[str, str]] = set()
        for form, lemma in rows:
            canonical_form = canonicalize_final_elision(sanitize_form(form))
            if canonical_form.endswith("\u1fbd"):
                elided_pair_keys.add((
                    exact_form_key(canonical_form[:-1]),
                    exact_form_key(lemma),
                ))
            if form in BARE_ELISION_STEMS:
                continue
            if not has_any_diacritic(form):
                continue
            by_lemma[lemma].append(form)

        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str]] = []
        keep = keep_lemmas or set()
        independent_headwords = (
            load_independent_ag_headwords() if attestation_freq else set()
        )
        for lemma, forms in by_lemma.items():
            is_canonical_keep = lemma in keep
            is_attested_keep = (
                attestation_freq is not None
                and any(exact_freq_lookup(f, attestation_freq) > 0 for f in forms)
            )
            if not is_canonical_keep and not is_attested_keep:
                if not any(has_polytonic(f) for f in forms):
                    continue
                if not has_any_diacritic(lemma):
                    continue
            for f in forms:
                if (attestation_freq is not None
                        and is_bare_elision_fallback(
                            f,
                            lemma,
                            elided_pair_keys,
                            independent_headwords,
                        )):
                    continue
                key = (f, lemma)
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
        return out

    raise ValueError(f"Unknown variant {variant!r}")


def longest_common_prefix(strs: list[str]) -> str:
    if not strs:
        return ""
    cp = strs[0]
    for s in strs[1:]:
        i = 0
        while i < len(cp) and i < len(s) and cp[i] == s[i]:
            i += 1
        cp = cp[:i]
        if not cp:
            break
    return cp


def filter_by_lemma_freq(
    form_lemma: list[tuple[str, str]],
    freq_map: dict[str, int],
    min_lemma_count: int = 1,
    strict_acute_min: int | None = None,
    strict_form_freq_map: dict[str, int] | None = None,
    keep_forms: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Drop lemmas whose most-frequent form is below min_lemma_count.

    This removes Wiktionary-generated lemmas (and their inflected forms)
    that have never been observed in any corpus, which are typically
    spurious artifacts of Lua paradigm expansion. Lemmas with any
    attested form are kept whole (all their generated forms survive).

    If strict_acute_min is set, then within a surviving lemma, acute-only
    forms (no breathings/circumflex/iota-sub/grave) must individually
    meet that threshold to be kept. Polytonic-bearing forms are always
    kept when their lemma is kept. This is mainly used for the AG
    polytonic variant, where the DB carries many acute-only inflections
    that are not canonical AG spellings (e.g. post-Byzantine usage).
    ``keep_forms`` (pinned textbook cells and citation headwords) are exempt
    from that per-form requirement.
    """
    if min_lemma_count <= 0 and strict_acute_min is None:
        return form_lemma

    pinned_forms = keep_forms or set()
    by_lemma: dict[str, list[str]] = defaultdict(list)
    for form, lemma in form_lemma:
        by_lemma[lemma].append(form)

    keep_lemmas: set[str] = set()
    for lemma, forms in by_lemma.items():
        candidates = set(forms) | {lemma}
        for f in candidates:
            aggregate_count = freq_lookup(f, freq_map)
            exact_count = (
                exact_freq_lookup(f, strict_form_freq_map)
                if strict_form_freq_map is not None else 0
            )
            if max(aggregate_count, exact_count) >= min_lemma_count:
                keep_lemmas.add(lemma)
                break

    out: list[tuple[str, str]] = []
    for form, lemma in form_lemma:
        if lemma not in keep_lemmas:
            continue
        if (strict_acute_min is not None
                and not has_polytonic(form)
                and form not in pinned_forms):
            # acute-only or undecorated form: require per-form count
            if strict_form_freq_map is None:
                count = freq_lookup(form, freq_map)
            else:
                count = exact_freq_lookup(form, strict_form_freq_map)
            if count < strict_acute_min:
                continue
        out.append((form, lemma))
    return out


def sanitize_export_pairs(
    form_lemma: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], int, int]:
    """Sanitize and deduplicate pairs before affix compression.

    Returns ``(pairs, changed_forms, editorial_dropped)``. Editorial notation
    is never meaningful Hunspell input and can make regex-based consumers
    reject the complete affix file.
    """
    sanitized: list[tuple[str, str]] = []
    changed_forms = 0
    editorial_dropped = 0
    dedup: set[tuple[str, str]] = set()
    for form, lemma in form_lemma:
        clean_form = canonicalize_final_elision(sanitize_form(form))
        clean_lemma = canonicalize_final_elision(sanitize_form(lemma))
        if not clean_form:
            continue
        if (has_editorial_sigla(clean_form)
                or has_editorial_sigla(clean_lemma)):
            editorial_dropped += 1
            continue
        pair = (clean_form, clean_lemma)
        if pair in dedup:
            continue
        dedup.add(pair)
        if clean_form != form:
            changed_forms += 1
        sanitized.append(pair)
    return sanitized, changed_forms, editorial_dropped


def build_sfx_rules(
    form_lemma: list[tuple[str, str]],
    freq_map: dict[str, int],
    max_singletons_to_inline: int = 1,
    canonical_forms: set[str] | None = None,
    canonical_lemmas: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Given (form, lemma) pairs, produce (aff_lines, dic_lines).

    Strategy:
      - Group by lemma.
      - Compute the longest common prefix of each paradigm.
      - Compress only when that prefix is itself one of the paradigm's real
        forms; it becomes the flagged .dic base and the remaining tails are
        zero-strip suffix rules.
      - Inline paradigms whose common prefix is only a truncated stem. Hunspell
        accepts every .dic base as a word unless NEEDAFFIX is implemented, and
        Tonos intentionally does not implement NEEDAFFIX or nonzero stripping.
      - Lemmas sharing a suffix signature share a flag.

    The .aff returned contains only the SFX rule blocks, not the header.

    Frequency bucketing uses bucket_for(), which combines per-form
    corpus count, per-lemma aggregate count, and the canonical seed
    lists passed in. When canonical_forms/canonical_lemmas are None,
    behaviour reduces to the original per-form-count bucketing.
    """
    if canonical_forms is None:
        canonical_forms = set()
    if canonical_lemmas is None:
        canonical_lemmas = set()
    lemma_totals = compute_lemma_totals(form_lemma, freq_map)
    # Group by lemma
    by_lemma: dict[str, set[str]] = defaultdict(set)
    for form, lemma in form_lemma:
        if form in BARE_ELISION_STEMS:
            continue
        by_lemma[lemma].add(form)

    # Singletons: emit as plain words.
    singletons: list[tuple[str, str]] = []  # (form, lemma)
    # Multi-form lemmas bucketed by signature
    sig_to_lemmas: dict[tuple, list[tuple[str, str, list[str]]]] = defaultdict(list)
    # values: (lemma, real base form, [suffixes preserving duplicates])

    for lemma, forms in by_lemma.items():
        flist = sorted(forms)
        if len(flist) == 1:
            singletons.append((flist[0], lemma))
            continue
        stem = longest_common_prefix(flist)
        # A flagged .dic entry is accepted literally by both Hunspell and
        # Tonos. Never use a merely mechanical/truncated prefix (λόγ,
        # ἀνθρώπ, αναφερόμεν) as that entry. If the common prefix is not an
        # attested member of this paradigm, inline the paradigm instead.
        if not stem or stem not in forms:
            for f in flist:
                singletons.append((f, lemma))
            continue
        suffixes = [f[len(stem):] for f in flist]
        sig = tuple(sorted(set(suffixes)))
        sig_to_lemmas[sig].append((lemma, stem, suffixes))

    # Only signatures with 2+ lemmas get a flag (real compression win).
    # Unique/singleton signatures are inlined as plain wordlist entries.
    # This keeps the .aff under control and keeps flag IDs below
    # Hunspell's FLAG num cap (65535).
    sorted_sigs = sorted(
        sig_to_lemmas.items(), key=lambda x: (-len(x[1]), x[0])
    )

    flag_iter = gen_flag_names()
    sig_to_flag: dict[tuple, str] = {}
    aff_blocks: list[str] = []
    inlined_lemmas: list[tuple[str, list[str]]] = []  # (lemma, forms)
    MAX_FLAGS = 60_000  # headroom below 65535

    flags_assigned = 0
    for sig, lemma_info in sorted_sigs:
        if len(lemma_info) < 2 or flags_assigned >= MAX_FLAGS:
            # Inline: every form of every lemma becomes a plain entry.
            for lemma, stem, suffixes in lemma_info:
                forms = [stem + s for s in suffixes]
                inlined_lemmas.append((lemma, forms))
            continue
        flag = next(flag_iter)
        flags_assigned += 1
        sig_to_flag[sig] = flag
        unique_suffixes = sig  # already a sorted unique tuple
        n = len(unique_suffixes)
        header = f"SFX {flag} Y {n}"
        body = [header]
        for suf in unique_suffixes:
            add = suf if suf else "0"
            body.append(f"SFX {flag} 0 {add} .")
        aff_blocks.append("\n".join(body))

    # Build .dic lines
    dic_lines: list[str] = []

    def fmt_entry(word: str, flag: str | None, bucket: str) -> str:
        morph = f"fr:{bucket}"
        if flag:
            return f"{word}/{flag}\t{morph}"
        return f"{word}\t{morph}"

    def pick_bucket(form: str, lemma: str, form_count: int) -> str:
        return bucket_for(
            form=form,
            lemma=lemma,
            form_count=form_count,
            lemma_total=lemma_totals.get(lemma, 0),
            canonical_forms=canonical_forms,
            canonical_lemmas=canonical_lemmas,
        )

    # Emit stems with flags (shared-signature lemmas). A stem line
    # represents all forms of possibly-several lemmas sharing the
    # same suffix signature; we pick the strongest (highest) bucket
    # across the covered forms so the stem does not under-rank any
    # iconic form it expands to. Canonical seed membership is checked
    # against each individual expanded form.
    emitted_stems: set[tuple[str, str]] = set()
    bucket_rank = {"C": 3, "M": 2, "R": 1, "X": 0}
    for sig, lemma_info in sorted_sigs:
        if sig not in sig_to_flag:
            continue
        flag = sig_to_flag[sig]
        for lemma, stem, suffixes in lemma_info:
            best_bucket = "X"
            for suf in set(suffixes):
                full = stem + suf
                c = freq_lookup(full, freq_map)
                b = pick_bucket(full, lemma, c)
                if bucket_rank[b] > bucket_rank[best_bucket]:
                    best_bucket = b
            key = (stem, flag)
            if key in emitted_stems:
                continue
            emitted_stems.add(key)
            dic_lines.append(fmt_entry(stem, flag, best_bucket))

    # Emit inlined lemmas (unique-signature, expanded to individual forms)
    emitted_plain: set[str] = set()
    for lemma, forms in inlined_lemmas:
        for form in forms:
            if form in emitted_plain:
                continue
            emitted_plain.add(form)
            c = freq_lookup(form, freq_map)
            dic_lines.append(fmt_entry(form, None, pick_bucket(form, lemma, c)))

    # Emit singletons as plain words
    for form, lemma in singletons:
        if form in emitted_plain:
            continue
        emitted_plain.add(form)
        c = freq_lookup(form, freq_map)
        dic_lines.append(fmt_entry(form, None, pick_bucket(form, lemma, c)))

    return aff_blocks, dic_lines


def write_variant(
    variant: str,
    form_lemma: list[tuple[str, str]],
    freq_map: dict[str, int],
    out_dir: Path,
    dic_name: str,
    lang_tag: str,
    version: str,
    commit: str,
    canonical_forms: set[str] | None = None,
    canonical_lemmas: set[str] | None = None,
) -> dict:
    """Emit <dic_name>.dic, <dic_name>.aff, <dic_name>.version. Return stats."""
    out_dir.mkdir(parents=True, exist_ok=True)

    aff_blocks, dic_lines = build_sfx_rules(
        form_lemma, freq_map,
        canonical_forms=canonical_forms,
        canonical_lemmas=canonical_lemmas,
    )

    # .aff header
    aff_header = [
        f"# Dilemma Hunspell export for {lang_tag}",
        f"# Version {version} (built from commit {commit})",
        f"# Generated by export_hunspell.py",
        "SET UTF-8",
        f"LANG {lang_tag}",
        "FLAG num",
        # TRY order: common Greek letters first, for Hunspell's own
        # suggestion mechanism. Roughly ordered by Modern Greek letter
        # frequency.
        "TRY αεοιτνσρπλκημυδγωβχφξζθψηάέήίόύώϊϋΐΰ"
        "αβγδεζηθικλμνξοπρστυφχψω",
        # Flag used to mark morphological info field
        "WORDCHARS αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
        "άέήίόύώϊϋΐΰᾰᾱῐῑῠῡἀἁἂἃἄἅἆἇἐἑἒἓἔἕἠἡἢἣἤἥἦἧ"
        "ἰἱἲἳἴἵἶἷὀὁὂὃὄὅὐὑὒὓὔὕὖὗὠὡὢὣὤὥὦὧᾳῃῳ'ʼ’",
        "",
    ]
    aff_text = "\n".join(aff_header) + "\n".join(aff_blocks) + "\n"

    aff_path = out_dir / f"{dic_name}.aff"
    aff_path.write_text(aff_text, encoding="utf-8")

    # .dic: first line is count, then entries
    dic_header = f"{len(dic_lines)}\n"
    dic_path = out_dir / f"{dic_name}.dic"
    with open(dic_path, "w", encoding="utf-8") as f:
        f.write(dic_header)
        for line in dic_lines:
            f.write(line)
            f.write("\n")

    # Sidecar version file
    ver_path = out_dir / f"{dic_name}.version"
    ver_path.write_text(
        f"version: {version}\n"
        f"commit: {commit}\n"
        f"variant: {lang_tag}\n"
        f"entries: {len(dic_lines)}\n"
        f"aff_rules: {sum(len(b.splitlines()) - 1 for b in aff_blocks)}\n",
        encoding="utf-8",
    )

    return {
        "aff_bytes": aff_path.stat().st_size,
        "dic_bytes": dic_path.stat().st_size,
        "entries": len(dic_lines),
        "aff_rules": sum(len(b.splitlines()) - 1 for b in aff_blocks),
        "aff_path": str(aff_path),
        "dic_path": str(dic_path),
    }


DEFAULT_MIN_LEMMA_FREQ = {"el": 1, "grc": 3}


def run_export(sanity: int | None, variants: list[str],
               min_lemma_freq: int | None = None) -> None:
    if not LOOKUP_DB.exists():
        print(f"ERROR: {LOOKUP_DB} not found. Download with "
              f"`huggingface-cli download ciscoriordan/dilemma --local-dir . "
              f"--include 'data/*'`",
              file=sys.stderr)
        sys.exit(1)

    commit = get_git_commit()
    version = read_version_file()

    print(f"Dilemma Hunspell export")
    print(f"  version: {version}")
    print(f"  commit:  {commit}")
    print(f"  sanity:  {sanity if sanity else 'off (full export)'}")
    print(f"  output:  {OUT}")
    print()

    print("Loading frequency maps...")
    mg_freq, ag_freq = load_freq_maps()
    profile = (
        load_form_profile_freq() if "grc" in variants
        else FormProfileEvidence({}, {}, {}, {})
    )
    grc_form_freq = profile.exact
    profile_meta = profile.metadata
    print(f"  MG freq: {len(mg_freq):,} entries")
    print(f"  AG freq: {len(ag_freq):,} entries")
    if "grc" in variants:
        if not grc_form_freq:
            print(f"ERROR: {FORM_PROFILE_DB} is required for full, "
                  "accent-preserving grc attestation; download it with "
                  "`python -m dilemma download --with-attestation`",
                  file=sys.stderr)
            sys.exit(1)
        print(f"  AG exact-form profile: {len(grc_form_freq):,} entries")
        print(f"  AG form-profile hash: "
              f"{profile_meta.get('content_hash', 'unknown')}")
    canonical_forms, canonical_lemmas = load_canonical_ag_sets()
    top_lsj9_lemmas = load_top_lsj9_lemmas()
    compatibility_forms, compatibility_meta = (
        load_grc_compatibility_forms() if "grc" in variants else (set(), {})
    )
    textbook_forms, textbook_meta = (
        load_grc_textbook_forms() if "grc" in variants else (set(), {})
    )
    if "grc" in variants and not compatibility_forms:
        print(f"ERROR: {GRC_COMPATIBILITY_FORMS} is required for the "
              "whole-artifact no-regression gate", file=sys.stderr)
        sys.exit(1)
    if "grc" in variants and not textbook_forms:
        print(f"ERROR: {GRC_TEXTBOOK_FORMS} is required for the textbook "
              "paradigm gate", file=sys.stderr)
        sys.exit(1)
    # Closed lists and pinned forms: exempt from the corpus-evidence filters
    # and pinned to the common frequency bucket.
    export_overrides = GRC_CLOSED_LIST_FORMS
    lm_head_forms = load_lm_head_required_forms() if "grc" in variants else set()
    grc_protected_forms = grc_pinned_forms(
        canonical_forms, top_lsj9_lemmas, textbook_forms, lm_head_forms
    )
    print(f"  AG canonical forms (pin to C): {len(canonical_forms):,}")
    print(f"  AG canonical lemmas (pin to C): {len(canonical_lemmas):,}")
    if "grc" in variants:
        baseline = compatibility_meta.get("baseline", {})
        print(f"  AG compatibility forms: {len(compatibility_forms):,} "
              f"(baseline {baseline.get('version', 'unknown')} "
              f"{str(baseline.get('commit', 'unknown'))[:7]})")
        print(f"  AG textbook forms: {len(textbook_forms):,} "
              f"(source {textbook_meta.get('source', {}).get('sha256', 'unknown')[:12]})")
    print()

    conn = sqlite3.connect(str(LOOKUP_DB))
    conn.execute("PRAGMA mmap_size=268435456")

    for variant in variants:
        print(f"=== Variant: {variant} ===")
        keep = (
            canonical_lemmas | top_lsj9_lemmas | GRC_COMPLETE_PARADIGM_LEMMAS
            if variant == "grc" else None
        )
        form_lemma = select_forms(
            conn,
            variant,
            keep_lemmas=keep,
            attestation_freq=grc_form_freq if variant == "grc" else None,
        )
        print(f"  {len(form_lemma):,} raw (form, lemma) pairs from lookup.db")

        if variant == "el":
            # Augment MG from mg_form_freq.json: any form with count >= 1
            # that isn't already present. Resolve its lemma by querying
            # lookup.db for the first match on the form (src='grc' or
            # src='el'). This pulls in widely-attested MG vocabulary
            # (δεν, καί, σκύλος) that is stored only on the 'grc' side
            # of the lookup table.
            existing_forms = {f for f, _ in form_lemma}
            added = 0
            for form, count in mg_freq.items():
                if count < 1:
                    continue
                if form in existing_forms:
                    continue
                if has_polytonic(form):
                    continue
                # Find lemma
                row = conn.execute(
                    "SELECT l.text FROM lookup k "
                    "JOIN lemmas l ON k.lemma_id = l.id "
                    "WHERE k.form = ? LIMIT 1",
                    (form,)
                ).fetchone()
                lemma = row[0] if row else form
                form_lemma.append((form, lemma))
                added += 1
            print(f"  +{added:,} forms augmented from mg_form_freq.json")

        threshold = (min_lemma_freq if min_lemma_freq is not None
                     else DEFAULT_MIN_LEMMA_FREQ.get(variant, 0))
        if threshold > 0:
            fm = mg_freq if variant == "el" else ag_freq
            before = len(form_lemma)
            # Sparse acute-only forms are checked against the complete
            # accent-preserving corpus profile, not the LM vocabulary (whose
            # minimum count is 20).  Textbook paradigms and top citation
            # headwords are explicit, reviewable exceptions.
            strict = 1 if variant == "grc" else None
            form_lemma = filter_by_lemma_freq(
                form_lemma, fm,
                min_lemma_count=threshold,
                strict_acute_min=strict,
                strict_form_freq_map=(grc_form_freq
                                      if variant == "grc" else None),
                keep_forms=((top_lsj9_lemmas | textbook_forms)
                            if variant == "grc" else None),
            )
            print(f"  After freq filter (>= {threshold}"
                  f"{', strict_acute>=1' if strict else ''}): "
                  f"{len(form_lemma):,} pairs (dropped {before-len(form_lemma):,})")
        if sanity:
            # Cap to the first N lemmas for a sanity pass
            seen_lemmas: set[str] = set()
            capped: list[tuple[str, str]] = []
            for f, l in form_lemma:
                if l not in seen_lemmas:
                    if len(seen_lemmas) >= sanity:
                        continue
                    seen_lemmas.add(l)
                capped.append((f, l))
            form_lemma = [(f, l) for f, l in form_lemma if l in seen_lemmas]
            print(f"  Sanity pass: {len(seen_lemmas):,} lemmas, "
                  f"{len(form_lemma):,} forms")

        if variant == "grc":
            # Closed-class words and a handful of independently attested
            # runtime-resolved forms are exceptions to the lookup/frequency
            # gate. Inject them after filtering so sparse source ownership
            # cannot silently remove them.
            # A sanity pass stays small: it does not carry the 1.18M-form
            # April surface, so its output does not satisfy the release audit.
            form_lemma, added = add_grc_reviewed_forms(
                form_lemma,
                export_overrides,
                textbook_meta.get("paradigms", {}),
                set() if sanity else compatibility_forms,
            )
            if added["overrides"]:
                print(f"  +{added['overrides']} AG closed-list/override forms")
            if added["textbook"]:
                print(f"  +{added['textbook']:,} pinned textbook-paradigm forms")
            if added["compatibility"]:
                print(f"  +{added['compatibility']:,} reviewed April "
                      "compatibility forms")

        # Belt-and-braces guard: sanitize every form so a misplaced combining
        # breathing (leading U+0313/U+0314 or trailing U+0313/U+0314 used as
        # an apostrophe) never reaches the .dic. See form_sanitize.sanitize_form
        # for the full rules. Lemmas are sanitized too so that two lemma
        # spellings that differ only by this bug collapse onto one paradigm.
        sanitized, changed_forms, editorial_dropped = sanitize_export_pairs(
            form_lemma
        )
        if changed_forms:
            print(f"  Guard: sanitized {changed_forms:,} forms "
                  f"(misplaced/orphan breathing marks)")
        if editorial_dropped:
            print(f"  Guard: dropped {editorial_dropped:,} "
                  "editorial-notation pairs")
        form_lemma = sanitized

        if variant == "grc":
            form_lemma, report = finalize_grc_pairs(
                form_lemma,
                evidence=profile,
                compatibility_forms=compatibility_forms,
                textbook_forms=textbook_forms,
                export_overrides=export_overrides,
                protected_forms=grc_protected_forms,
            )
            if report["invalid"]:
                detail = ", ".join(
                    f"{reason}={count:,}"
                    for reason, count in sorted(report["invalid"].items())
                )
                print(f"  Guard: dropped {sum(report['invalid'].values()):,} "
                      f"structurally invalid forms ({detail})")
            for key, message in (
                ("new_structural", "new truncated/smooth-upsilon forms"),
                ("unattested", "unreviewed, unattested post-April "
                               "generated forms"),
                ("dominated", "weak new respellings of a more common "
                              "spelling"),
            ):
                if report[key]:
                    print(f"  Guard: dropped {report[key]:,} {message}")
            if report["acute_twins"]:
                print(f"  +{report['acute_twins']:,} contextual-grave "
                      "acute twins")
            if report["twin_dropped"]:
                print(f"  Guard: dropped {report['twin_dropped']:,} derived "
                      "twins (structure or weak respelling)")

        if variant == "el":
            stats = write_variant(
                variant=variant,
                form_lemma=form_lemma,
                freq_map=mg_freq,
                out_dir=OUT,
                dic_name="el_GR_monotonic",
                lang_tag="el_GR",
                version=version,
                commit=commit,
            )
        else:
            stats = write_variant(
                variant=variant,
                form_lemma=form_lemma,
                freq_map=ag_freq,
                out_dir=OUT,
                dic_name="grc_polytonic",
                lang_tag="grc",
                version=version,
                commit=commit,
                # Frequency bucket C: curated forms, top citation
                # headwords, and the reviewed LM head, so an unaccented
                # λυω ranks λύω first. Textbook cells and closed-list forms
                # keep their corpus bucket.
                canonical_forms=(
                    canonical_forms | top_lsj9_lemmas | lm_head_forms
                ),
                canonical_lemmas=canonical_lemmas,
            )

        print(f"  entries: {stats['entries']:,}")
        print(f"  aff rules: {stats['aff_rules']:,}")
        print(f"  .dic: {stats['dic_bytes']/1024/1024:.2f} MB "
              f"({stats['dic_path']})")
        print(f"  .aff: {stats['aff_bytes']/1024/1024:.2f} MB "
              f"({stats['aff_path']})")
        total_mb = (stats['dic_bytes'] + stats['aff_bytes']) / 1024 / 1024
        print(f"  total: {total_mb:.2f} MB")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", type=int, default=None,
                    help="Cap to N lemmas for a quick sanity run")
    ap.add_argument("--variant", choices=["el", "grc", "both"],
                    default="grc",
                    help="Which variant to emit. Defaults to grc "
                         "(Ancient + Medieval polytonic); the el "
                         "(Modern Greek monotonic) variant is retained "
                         "for other downstream consumers but is not "
                         "shipped in the Tonos keyboard.")
    ap.add_argument("--min-lemma-freq", type=int, default=None,
                    help="Override per-variant default: drop lemmas "
                         "whose most-attested form has corpus count < "
                         "this value. Defaults are el=1, grc=3.")
    args = ap.parse_args()

    variants = ["el", "grc"] if args.variant == "both" else [args.variant]
    run_export(args.sanity, variants, min_lemma_freq=args.min_lemma_freq)


if __name__ == "__main__":
    main()

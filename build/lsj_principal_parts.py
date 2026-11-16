#!/usr/bin/env python3
"""Parse principal parts (fut., aor., pf., etc.) out of LSJ head text.

The LSJ entry for a verb starts with a header sentence that lists the
canonical principal parts before the English definition begins, e.g.:

    λύω, fut. λύσω, aor. ἔλυσα, pf. λέλυκα,
    pf. m./p. λέλυμαι, aor. p. ἐλύθην

This module turns that header into a structured dict like::

    {"fut": "λύσω", "aor": "ἔλυσα", "pf": "λέλυκα",
     "pf_mp": "λέλυμαι", "aor_p": "ἐλύθην"}

It is intentionally conservative: a missing or ambiguous principal part
yields no entry rather than a guess. Downstream code can fall back to
present-only expansion when the parser returns nothing.

The parser handles:
  - Leading prosody bracket (``[ᾰ]``, ``[α]``)
  - Bold/wikitext-style headword echo (``**βάλλω** βάλλω,``)
  - Section markers ``:—Med.``, ``:—Pass.``, ``Med. and Pass.``, also
    at the start of a text that opens inside a section
  - Dual aorist labels (``aor. 1``, ``aor. 2``, ``aor. I``, ``aor. II``),
    and two numbered passive aorists of one verb
  - Author-attribution citation suffixes after each form

What it deliberately does NOT do:
  - Resolve cross-reference verbs (``v. καταχέω``)
  - Expand suffix-only abbreviations (``fut. -ψω``, ``fut. ύσω``) - those
    would need to know which letters of the headword to drop, which is
    itself a morphological judgement; we leave that to downstream Lua
    expansion when a full form is given instead.
  - Pick variant forms; the first attestation wins.

:func:`derive_grc_conj_args` turns the parts into ``{{grc-conj}}``
arguments. A past tense needs the stem without its augment for every form
but the indicative; :func:`unaugmented_stem` derives it from the printed
indicative and the headword, and declines to when the augment cannot be
read, in which case only the indicative is built.
"""

from __future__ import annotations

import re
import unicodedata

# Public principal-part keys.
# Tracks both the active triplet (fut, aor, pf, plpf, impf), the perfect
# mediopassive (pf_mp), the aorist passive (aor_p), the future passive
# (fut_p), and middle/passive variants of the future / future perfect.
PART_KEYS = (
    "impf", "fut", "aor", "aor1", "aor2",
    "pf", "plpf",
    "fut_med", "aor_med",
    "fut_p", "aor_p", "aor_p2", "pf_mp", "futp",
)


def strip_diacritics(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if not unicodedata.combining(c))


def _has_tonal_accent(form: str) -> bool:
    return any(mark in unicodedata.normalize("NFD", form)
               for mark in ("\u0300", "\u0301", "\u0342"))


# Greek-letter ranges used for token boundary detection. Includes
# combining marks so accented forms read as single tokens.
def _is_greek_letter(c: str) -> bool:
    if not c:
        return False
    return (
        ("Ͱ" <= c <= "Ͽ") or  # Greek and Coptic
        ("ἀ" <= c <= "῿") or  # Greek Extended
        unicodedata.combining(c)
    )


# Inline label like "fut.", "aor.", "pf.", "Pf.", "perf.", "plpf.",
# "impf.", "imperf.". An optional digit "1" or "2" disambiguates
# aor. 1 / aor. 2; an optional "p." or "pass." right after the label
# tags the form as passive (aor. p. ἐλύθην, pf. p. λέλυμαι).
_LABEL_RE = re.compile(
    r"(?<![A-Za-zΑ-Ωα-ωἀ-ῶ])"
    r"(?P<label>[Ff]ut|[Aa]or|[Pp]f|[Pp]erf|[Pp]lpf|[Pp]lupf|[Ii]mpf|[Ii]mperf)"
    r"\.\s*"
    r"(?:(?P<num>[12]|II?(?![A-Za-z]))\s*)?"
    r"(?P<voice>(?:m\.\s*/\s*p\.|m\.|p\.|mid\.|pass\.|m/p\.))?",
    re.UNICODE,
)
# LSJ numbers the two aorists with Arabic or Roman figures (aor. 1, aor. I).
_LABEL_NUMBERS = {"1": "1", "2": "2", "I": "1", "II": "2"}


# Section marker like ":—Med.", ":—Pass.", ":-Med.,", "—Med.".
# Returns the new section label.
_SECTION_RE = re.compile(
    r"(?::|—|:-)\s*(?:—\s*)?"
    r"(?P<sect>Med\.\s+and\s+Pass\.?|Mediopass\.?|Med\.?|Pass\.?|Mid\.?)",
)
# The same marker when a text opens inside a section: the earlier glosses
# export begins many entries after the dash (λῃστεύω: "Pass. (v. infr.), aor.
# ἐλῃστεύθην").
_LEADING_SECTION_RE = re.compile(
    r"\s*(?P<sect>Med\.\s+and\s+Pass\.|Mediopass\.|Med\.|Pass\.|Mid\.)"
    r"(?=[\s,(])",
)


# Tokens after which a principal part may be quoted. We always grab the
# very next Greek word, but stop at hyphen-trailing fragments
# ("πέμ-\nπέμπεσκε" line-break artifacts) and at fragments shorter than
# three characters (single letters and abbreviations).
_MIN_FORM_LEN = 3


# Greek words that are NOT principal parts: negations, articles,
# conjunctions, particles. We reject these when they show up
# immediately after a label (typical LSJ pattern: ``impf. οὐκ
# ἠρχόμην`` describing periphrastic negation). Stored as plain
# (diacritic-stripped, lowercased) strings so the comparison works
# directly against ``strip_diacritics(token)``.
_STOP_WORDS = {
    # Greek negations / particles / conjunctions
    "ου", "ουκ", "ουχ", "μη", "μηκ", "η", "και",
    "δε", "γαρ", "τε", "νυν", "μεν", "ουδε", "μηδε",
    "οτι", "ως", "πως", "ινα", "οπως", "ην",
    # Latin-script dialect / author tags. LSJ usually puts these
    # immediately after the label (``fut. Ion. βαλέω``).
    "ep", "ion", "att", "dor", "aeol", "hom", "hdt",
    "hp", "pi", "lat",
}


def _is_suffix_fragment(word: str) -> bool:
    """Whether a token is the tail of a word (φιτύω: "fut. ύσω").

    A Greek word that begins with a vowel carries a breathing on its first
    vowel or diphthong; LSJ prints such endings without one, and usually
    with a hyphen.
    """
    nfd = unicodedata.normalize("NFD", word.lower())
    if not nfd or nfd[0] not in "αεηιουω":
        return False
    for char in nfd[1:]:
        if char in "\u0313\u0314":
            return False
        if not unicodedata.combining(char) and char not in "ιυ":
            break
    return True


def _grab_form(text: str, start: int) -> tuple[str, int]:
    """Return the next plausible Greek principal-part form starting at
    or after ``start``.

    Skips leading whitespace, dialect prefix tokens
    (``Ion. βαλέω``, ``Ep. γράψα``), and stop-words (``οὐκ`` before
    ``ἠρχόμην``). Suffix-only abbreviations (``-ψω``) are rejected.
    Returns ("", start) if no acceptable form is found.
    """
    n = len(text)
    i = start

    # We allow up to a couple of "skip" tokens (dialect tags, stop
    # words) before the actual form. This handles ``impf. οὐκ
    # ἠρχόμην`` and ``fut. Ion. βαλέω``.
    for _ in range(4):
        # Skip whitespace.
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            return "", start
        # "-ψω" suffix abbreviation: reject. We cannot reconstruct
        # the dropped headword letters reliably without morphology.
        if text[i] == "-":
            return "", start

        # First check for a Latin-script dialect/author tag like
        # "Ion." or "Ep." that we should skip past. A Latin tag is
        # at most 4 ASCII letters, optionally followed by a dot.
        if "A" <= text[i] <= "Z" or "a" <= text[i] <= "z":
            k = i
            while k < n and (("A" <= text[k] <= "Z") or
                              ("a" <= text[k] <= "z")):
                k += 1
            tag = text[i:k]
            if tag.lower() in _STOP_WORDS:
                i = k
                # Consume trailing dot/comma + whitespace.
                while i < n and text[i] in ".,":
                    i += 1
                continue
            # Unknown Latin token - bail out, this isn't a Greek form.
            return "", start

        j = i
        while j < n and _is_greek_letter(text[j]):
            j += 1
        if j == i:
            return "", start

        word = text[i:j]
        # Reject mid-token line-break fragments. The LSJ OCR sometimes
        # leaves "πέμ-\nπέμπεσκε" in the text; we never accept a form
        # where the next non-Greek char is a hyphen.
        if j < n and text[j] == "-":
            return "", j
        if len(word) < _MIN_FORM_LEN:
            i = j
            continue

        # If the word is a known Greek stop-word, skip it.
        word_plain = strip_diacritics(word).lower()
        if word_plain in _STOP_WORDS:
            i = j
            while i < n and text[i] in ".,":
                i += 1
            continue
        # A breathing or quantity mark does not establish the accent of a
        # finite principal part. OCR sometimes inserts an unaccented token
        # immediately before the intact form, as in "ὑλα ἴηλα". Consider
        # the next token only within this same uninterrupted clause.
        if not _has_tonal_accent(word):
            i = j
            continue
        if _is_suffix_fragment(word):
            return "", start
        return unicodedata.normalize("NFC", word), j

    return "", start


def _strip_header(text: str, headword: str) -> str:
    """Strip the prosody bracket and bold/echoed headword from the start
    of the LSJ entry text.

    LSJ entries commonly start with one or both of:
      ``[ᾰ], `` (prosody hint)
      ``**βάλλω** βάλλω, `` (markdown-bold headword + echo)
    We chop these off so the principal-part labels start the body.
    """
    cleaned = text
    cleaned = re.sub(r"^\s*\[[^\]]*\]\s*,?\s*", "", cleaned)
    cleaned = re.sub(r"^\s*\*\*[^*]+\*\*\s*", "", cleaned)
    # Strip a leading repetition of the headword followed by comma or
    # space. We do NFC-compare so the prosody form of the headword
    # (with combining macron/breve already merged) still matches.
    hw_pattern = re.escape(unicodedata.normalize("NFC", headword))
    cleaned = re.sub(r"^\s*" + hw_pattern + r"\s*,?\s*", "", cleaned)
    return cleaned


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Return list of (section_name, slice_of_text).

    ``section_name`` is one of ``act`` (default), ``med`` (middle),
    ``pass`` (passive). LSJ uses ``:—Med.`` / ``:—Pass.`` markers (or
    ``:-Med.``, ``Mediopass.``) to introduce non-active sections.
    """
    markers: list[tuple[int, str]] = []
    leading = _LEADING_SECTION_RE.match(text)
    for m in ([leading] if leading else []) + list(_SECTION_RE.finditer(text)):
        sect_text = m.group("sect").lower()
        if sect_text.startswith("med") and "and pass" in sect_text:
            # "Med. and Pass." merges middle and passive into one
            # section. We tag it as 'med' which lets pf./aor. fall
            # under pf_mp / aor_p (the same form does double duty).
            sect_name = "med"
        elif sect_text.startswith("med") or sect_text.startswith("mid"):
            sect_name = "med"
        elif sect_text.startswith("pass"):
            sect_name = "pass"
        else:
            continue
        markers.append((m.start(), sect_name))

    if not markers:
        return [("act", text)]

    sections: list[tuple[str, str]] = []
    last_end = 0
    last_name = "act"
    for pos, name in markers:
        sections.append((last_name, text[last_end:pos]))
        last_end = pos
        last_name = name
    sections.append((last_name, text[last_end:]))
    return sections


def _section_part_key(label: str, num: str | None, voice_tag: str | None,
                      section: str) -> str | None:
    """Map (label, num, voice_tag, section) -> canonical part key.

    Returns None when the combination is unsupported (e.g. plpf in the
    passive section, which we silently drop because grc-conj has no
    principal-part slot for it).
    """
    lab = label.lower()
    base = {
        "fut": "fut",
        "aor": "aor",
        "pf": "pf",
        "perf": "pf",
        "plpf": "plpf",
        "plupf": "plpf",
        "impf": "impf",
        "imperf": "impf",
    }.get(lab)
    if base is None:
        return None

    # Numbered aorists: aor. 1 / aor. 2.
    if base == "aor" and num in ("1", "2"):
        if voice_tag and ("p" in voice_tag.lower() or "pass" in voice_tag.lower()):
            # Aorist passive doesn't take 1/2 numbering; drop the
            # number and let the passive section claim it.
            base = "aor"
        else:
            base = f"aor{num}"

    # Voice tags inline (e.g. "aor. p.", "pf. m./p.") override the
    # section context.
    if voice_tag:
        vt = voice_tag.lower().replace(" ", "")
        if "m" in vt and "p" in vt:
            # m./p. - mediopassive
            if base == "pf":
                return "pf_mp"
            if base in ("aor", "aor1", "aor2"):
                return "aor_p"
        if vt.startswith("p") or vt.startswith("pass"):
            if base in ("aor", "aor1", "aor2"):
                return "aor_p"
            if base == "fut":
                return "fut_p"
            if base == "pf":
                return "pf_mp"
            return None
        if vt.startswith("m") or vt.startswith("mid"):
            if base == "fut":
                return "fut_med"
            if base in ("aor", "aor1", "aor2"):
                return "aor_med"
            if base == "pf":
                return "pf_mp"
            return None

    # No inline voice tag - use the section.
    if section == "med":
        if base == "fut":
            return "fut_med"
        if base in ("aor", "aor1", "aor2"):
            return "aor_med"
        if base == "pf":
            return "pf_mp"
        if base == "impf":
            # Don't track imperfect middle separately
            return None
        return None
    if section == "pass":
        if base == "fut":
            return "fut_p"
        if base in ("aor", "aor1", "aor2"):
            return "aor_p"
        if base == "pf":
            return "pf_mp"
        return None

    # Active section - keep base key as-is.
    return base


# Maximum bytes of head-text we look at when scanning for principal
# parts. Some entries (esp. -μι verbs like τίθημι, ἵστημι) bury the
# principal-part header behind a thousand-plus bytes of dialect-form
# enumeration; we use the entire head text since the parser already
# stops at section markers and rejects implausible forms via
# ``_form_looks_like``. The head texts are at most a few KB.
_MAX_SCAN = 8000


def _form_looks_like(form: str, key: str) -> bool:
    """Reject obviously wrong forms based on tense ending.

    Subjunctive / optative forms can sneak past the regex when LSJ has
    an OCR artifact like ``f.it. παιδεύσομαι fut. παιδεύσῃ`` (the
    second token after `fut.` is a subjunctive). A 1sg principal part
    has predictable endings; we sanity-check against them.
    """
    plain = strip_diacritics(form).lower()
    # Active 1sg endings.
    if key in ("fut",):
        return plain.endswith("ω") or plain.endswith("ουμαι") or \
               plain.endswith("ουμι") or plain.endswith("εμαι") or \
               plain.endswith("ομαι") or plain.endswith("εω")
    if key in ("aor", "aor1"):
        return plain.endswith("α") or plain.endswith("ν") or plain.endswith("σα")
    if key == "aor2":
        return plain.endswith("ον") or plain.endswith("ομην")
    if key in ("pf",):
        return plain.endswith("α") or plain.endswith("κα")
    if key == "plpf":
        return plain.endswith("ν") or plain.endswith("ει") or \
               plain.endswith("κειν") or plain.endswith("μην")
    if key == "impf":
        return plain.endswith("ον") or plain.endswith("ν") or \
               plain.endswith("μην") or plain.endswith("κον")
    if key in ("fut_med", "aor_med"):
        return plain.endswith("μαι") or plain.endswith("μην")
    if key in ("fut_p",):
        return plain.endswith("ομαι") or plain.endswith("ησομαι") or \
               plain.endswith("σομαι")
    if key == "aor_p":
        return plain.endswith("ην") or plain.endswith("ν")
    if key == "pf_mp":
        # 1sg: -μαι. Some entries only attest 3sg (-ται), e.g.
        # ``pf. πέπεμπται``; we accept these and let the stem
        # extraction in ``derive_grc_conj_args`` handle either.
        return plain.endswith("μαι") or plain.endswith("ται") or \
               plain.endswith("νται")
    return True


def _find_chained_aor_passive(sect_text: str, label_match: re.Match
                              ) -> str | None:
    """Look for an unlabelled ``-θην`` form chained after an ``aor.``
    label in a passive-bearing section.

    LSJ collapses middle and passive aorists into a single ``aor.``
    clause inside ``:—Med. and Pass.`` (and sometimes ``:—Pass.``)
    sections, e.g. for παύω::

        aor. ἐπαυσάμην Il.14.260; ἐπαύθην, Ep. παύθην, …;
        ἐπαύσθην Hdt.5.94, etc. ; later ἐπάην …

    Here ``ἐπαυσάμην`` (the first form) is captured by the standard
    label-driven scan as ``aor_med``, but the chained ``ἐπαύθην`` is
    the true passive aorist and has no label of its own. This helper
    scans the slice between the label match and the next label /
    section boundary, returning the first ``-θην`` form (length >= 4)
    that isn't a suffix abbreviation, line-break artifact, or
    parenthetical aside.
    """
    n = len(sect_text)
    start = label_match.end()
    # Bound the scan at the earliest of:
    #   1. The next principal-part label (``pf.``, etc.).
    #   2. A blank line (``\n\n``) - LSJ uses these to mark the
    #      end of the principal-parts header and the start of
    #      numbered sub-sections / definition body. Without this
    #      bound, an entry whose principal-parts header has no
    #      later label (e.g. προστρέπω) would scan into the
    #      definition and pick up a noun like ``πάθην`` as a
    #      passive aorist.
    end = n
    next_label = _LABEL_RE.search(sect_text, start)
    if next_label:
        end = min(end, next_label.start())
    blank_pos = sect_text.find("\n\n", start)
    if blank_pos != -1:
        end = min(end, blank_pos)

    # Walk Greek-letter runs in the bounded slice. We skip over
    # parenthesised content (``(v.l. παυθ-)``), forms preceded by a
    # hyphen (``-θην`` suffix abbreviation), and forms followed by a
    # hyphen (line-break artifact ``θην-``).
    i = start
    paren_depth = 0
    while i < end:
        c = sect_text[i]
        if c == "(":
            paren_depth += 1
            i += 1
            continue
        if c == ")":
            if paren_depth > 0:
                paren_depth -= 1
            i += 1
            continue
        if paren_depth > 0:
            i += 1
            continue
        if not _is_greek_letter(c):
            i += 1
            continue
        # Start of a Greek-letter run.
        prev = sect_text[i - 1] if i > 0 else ""
        if prev == "-":
            # ``-θην`` suffix abbreviation; skip the run wholesale.
            j = i
            while j < end and _is_greek_letter(sect_text[j]):
                j += 1
            i = j
            continue
        j = i
        while j < end and _is_greek_letter(sect_text[j]):
            j += 1
        word = sect_text[i:j]
        i = j
        # Reject mid-token line-break fragments ``θην-``.
        if j < n and sect_text[j] == "-":
            continue
        if len(word) < 4:
            continue
        plain = strip_diacritics(word).lower()
        if not plain.endswith("θην"):
            continue
        if not _has_tonal_accent(word):
            continue
        return unicodedata.normalize("NFC", word)
    return None


def ends_at_label(text: str) -> bool:
    """Whether a text stops right after a label ("... fut. -ήσω: aor.").

    The current glosses export cuts some entries there; the earlier export
    of the same entry then opens with the form the label announces.
    """
    cleaned = text.rstrip()
    labels = [m for m in _LABEL_RE.finditer(cleaned)]
    return bool(labels) and labels[-1].end() == len(cleaned)


def parse_principal_parts(text: str, headword: str) -> dict[str, str]:
    """Extract canonical principal parts from an LSJ head text.

    Returns a (possibly empty) dict whose keys are drawn from
    :data:`PART_KEYS`. Missing parts are simply absent.
    """
    if not text:
        return {}
    cleaned = _strip_header(text, headword)
    cleaned = cleaned[:_MAX_SCAN]

    out: dict[str, str] = {}
    for sect_name, sect_text in _split_sections(cleaned):
        for m in _LABEL_RE.finditer(sect_text):
            label = m.group("label")
            num = _LABEL_NUMBERS.get(m.group("num"))
            voice = m.group("voice")
            key = _section_part_key(label, num, voice, sect_name)
            if key is None:
                continue
            form, _ = _grab_form(sect_text, m.end())
            if not form:
                continue
            # Validate that the form actually looks like the claimed
            # tense. Filters out OCR junk like ``fut. παιδεύσῃ`` where
            # the form is actually a subjunctive 2sg.
            if not _form_looks_like(form, key):
                continue
            # First attestation wins; later occurrences are usually
            # variants ("Ion. βαλέω", "later γεγράφηκα") that we want
            # to ignore for the canonical principal-part slot. A verb
            # with two numbered passive aorists keeps both (ἀποπνίγω:
            # aor. I ἀπεπνίχθην, aor. 2 ἀπεπνίγην).
            if key not in out:
                out[key] = form
            elif key == "aor_p" and num and out[key] != form:
                out.setdefault("aor_p2", form)

            # Chained passive aorist: in ``:—Med. and Pass.`` (and
            # ``:—Pass.``) sections, LSJ may bundle the middle and
            # passive aorists under one ``aor.`` label, e.g. παύω's
            # ``aor. ἐπαυσάμην …; ἐπαύθην, …; ἐπαύσθην …``. The
            # standard scan only takes the first form (the middle);
            # we reach into the chain for the first ``-θην`` form
            # and tag it as ``aor_p``. An explicit ``aor. p.``
            # label, when present, has already been honoured above
            # and would have set ``aor_p`` first; the ``key not in
            # out`` guard preserves that precedence.
            if (sect_name in ("pass", "med")
                    and key in ("aor", "aor1", "aor2",
                                "aor_med")
                    and "aor_p" not in out):
                lab = label.lower()
                if lab in ("aor",):
                    chained = _find_chained_aor_passive(sect_text, m)
                    if chained and _form_looks_like(chained, "aor_p"):
                        out["aor_p"] = chained

    # Aorist consolidation. LSJ tends to mention aor.2 only when an
    # aor.1 is also present (e.g. λείπω). Conventionally the older /
    # default aorist is the second; βάλλω's principal part is ἔβᾰλον
    # not ἔβαλα. We therefore prefer aor2 over aor1 when both exist
    # and no unnumbered "aor." was given. Otherwise the first
    # unambiguous aorist wins.
    if "aor" not in out:
        if "aor2" in out:
            out["aor"] = out["aor2"]
        elif "aor1" in out:
            out["aor"] = out["aor1"]

    return out


# ---------------------------------------------------------------------------
# Augment and preverb analysis
# ---------------------------------------------------------------------------
#
# Module:grc-conj takes two stems for a past tense: the augmented one for the
# indicative and the unaugmented one for every other mood, the infinitive and
# the participle. LSJ prints the augmented indicative, so the second stem has
# to be derived. In a compound the augment follows the preverb, and the
# preverb stands before it in its prevocalic form (ἀπο-λύω: ἀπ-έ-λυσα,
# ἐκ-βάλλω: ἐξ-έ-βαλον, συλ-λέγω: συν-έ-λεξα), so removing the augment also
# restores the preverb the present shows.

_PERISPOMENI = chr(0x0342)
_TONE_MARKS = frozenset((chr(0x0300), chr(0x0301), _PERISPOMENI))
_SMOOTH, _ROUGH = chr(0x0313), chr(0x0314)
_BREATHINGS = (_SMOOTH, _ROUGH)
_SUBSCRIPT, _DIAERESIS = chr(0x0345), chr(0x0308)
_MACRON, _BREVE = chr(0x0304), chr(0x0306)
_VOWEL_LETTERS = frozenset("αεηιουω")
_DIPHTHONGS = frozenset(("αι", "ει", "οι", "υι", "αυ", "ευ", "ου", "ηυ"))
# Sounds that answer each other at the start of a stem: an aspirate and its
# plain stop (θάπτω, ἐτάφην), a stop and its fusion with σ (ἅπτω, ἧψα).
_ONSET_CLASSES = ("τθ", "πφ", "κχ", "πβφψ", "κγχξ", "τδθζσ")
_LABIALS, _GUTTURALS = frozenset("πβφψμ"), frozenset("κγχξ")

# Preverb -> (its spellings in composition, its spellings before the augment).
# A composition spelling stands only before the sounds that call for it
# (_composition_fits).
_PREVERBS = {
    "ανα": (("ανα", "αν"), ("αν",)),
    "αντι": (("αντι", "αντ", "ανθ"), ("αντ", "ανθ")),
    "απο": (("απο", "απ", "αφ"), ("απ", "αφ")),
    "δια": (("δια", "δι"), ("δι",)),
    "εισ": (("εισ", "εσ"), ("εισ", "εσ")),
    "εκ": (("εκ", "εξ"), ("εξ",)),
    "εν": (("εν", "εμ", "εγ", "ελ"), ("εν",)),
    "επι": (("επι", "επ", "εφ"), ("επ", "εφ")),
    "κατα": (("κατα", "κατ", "καθ"), ("κατ", "καθ")),
    "μετα": (("μετα", "μετ", "μεθ"), ("μετ", "μεθ")),
    "παρα": (("παρα", "παρ"), ("παρ",)),
    "περι": (("περι",), ("περι",)),
    "προ": (("προ",), ("προ",)),
    "προσ": (("προσ",), ("προσ",)),
    "συν": (("συν", "συμ", "συγ", "συλ", "συρ", "συσ", "συ",
             "ξυν", "ξυμ", "ξυγ", "ξυλ", "ξυρ", "ξυσ", "ξυ"), ("συν", "ξυν")),
    "υπερ": (("υπερ",), ("υπερ",)),
    "υπο": (("υπο", "υπ", "υφ"), ("υπ", "υφ")),
    "αμφι": (("αμφι", "αμφ"), ("αμφ", "αμφι")),
}
_ELIDED = frozenset(("αν", "αντ", "ανθ", "απ", "αφ", "δι", "επ", "εφ", "κατ",
                     "καθ", "μετ", "μεθ", "παρ", "υπ", "υφ", "εξ", "αμφ"))
_FULL = frozenset(("ανα", "αντι", "απο", "δια", "επι", "κατα", "μετα",
                   "παρα", "υπο", "εκ"))
_ALWAYS = frozenset(("περι", "προ", "προσ", "υπερ", "εισ", "εσ", "αμφι"))

# Aorist stems that share no letters with their present: augmented ->
# unaugmented (ἤνεγκα ἐνεγκεῖν, ἦλθον ἐλθεῖν, εἶδον ἰδεῖν, εἷλον ἑλεῖν).
_SUPPLETIVE_AORISTS = {
    "ηνεγκ": "ενεγκ", "ηνεικ": "ενεικ", "ηνεχθ": "ενεχθ",
    "ηλθ": "ελθ", "ηλυθ": "ελυθ", "ειπ": "ειπ", "ειδ": "ιδ", "ειλ": "ελ",
}
_TEMPORAL_AUGMENTS = {
    "α": ("η",), "ε": ("η", "ει"), "ο": ("ω",), "αι": ("η",),
    "ει": ("η",), "οι": ("ω",), "αυ": ("ηυ",), "ευ": ("ηυ",),
}


def _letters(word: str) -> list[str]:
    """Lowercase letters, each with its marks, tone accents removed.

    A circumflex on α, ι or υ shows a long vowel; it is kept as a macron,
    which the conjugation module needs to place the accent again.
    """
    out: list[str] = []
    for char in unicodedata.normalize("NFD", word.lower()):
        if not unicodedata.combining(char):
            out.append("σ" if char == "ς" else char)
        elif not out:
            continue
        elif char not in _TONE_MARKS:
            out[-1] += char
        elif (char == _PERISPOMENI and out[-1][0] in "αιυ"
              and _MACRON not in out[-1] and _BREVE not in out[-1]
              and not (len(out) > 1 and _DIAERESIS not in out[-1]
                       and out[-2][0] + out[-1][0] in _DIPHTHONGS)):
            out[-1] += _MACRON
    return [letter.replace(_MACRON, "") if _SUBSCRIPT in letter else letter
            for letter in out]


def _spell(letters: list[str]) -> str:
    return unicodedata.normalize("NFC", "".join(letters))


def _bare(letters: list[str]) -> str:
    return "".join(letter[0] for letter in letters)


def _unbreathed(letter: str) -> str:
    return "".join(char for char in letter if char not in _BREATHINGS)


def _vowel_group(letters: list[str]) -> int:
    """Number of letters in the initial vowel or diphthong (0, 1 or 2)."""
    if not letters or letters[0][0] not in _VOWEL_LETTERS:
        return 0
    if (len(letters) > 1 and letters[0][0] + letters[1][0] in _DIPHTHONGS
            and _DIAERESIS not in letters[1]
            and not any(mark in letters[0] for mark in _BREATHINGS)):
        return 2
    return 1


def _same_onset(left: str, right: str) -> bool:
    """Whether a stem ``left`` goes on like the present ``right``.

    The same sound (θάπτω, ἐτάφην), or anything at all where the present
    has only its ending left (εὕω, εὗσα).
    """
    if not left or not right:
        return True
    if right in ("ω", "ομαι") and left[0] not in _VOWEL_LETTERS:
        return True
    return left[0] == right[0] or any(
        left[0] in sounds and right[0] in sounds for sounds in _ONSET_CLASSES)


def _composition_fits(spelling: str, following: str) -> bool:
    """Whether a preverb spelling can stand before ``following``."""
    if not following:
        return False
    nxt = following[0]
    vowel = nxt in _VOWEL_LETTERS
    if spelling in _ALWAYS:
        return True
    if spelling in _FULL:
        return not vowel
    if spelling in _ELIDED:
        return vowel
    if spelling in ("εν", "συν", "ξυν"):
        return vowel or nxt in "τδθνζσ"
    if spelling in ("εμ", "συμ", "ξυμ"):
        return nxt in _LABIALS
    if spelling in ("εγ", "συγ", "ξυγ"):
        return nxt in _GUTTURALS
    if spelling in ("ελ", "συλ", "ξυλ"):
        return nxt == "λ"
    if spelling in ("συρ", "ξυρ"):
        return nxt == "ρ"
    if spelling in ("συσ", "ξυσ"):
        return nxt == "σ"
    if spelling in ("συ", "ξυ"):
        return nxt in "ζσ"
    return False


def _compose(canonical: str, following: str, xi: bool = False) -> str:
    """The spelling of a preverb before a consonant-initial stem."""
    nxt = following[:1]
    if canonical == "συν":
        base = "ξυ" if xi else "συ"
        if nxt in _LABIALS:
            return base + "μ"
        if nxt in _GUTTURALS:
            return base + "γ"
        if nxt in "λρ":
            return base + nxt
        if nxt == "ζ" or (nxt == "σ" and following[1:2]
                          and following[1] not in _VOWEL_LETTERS):
            return base
        if nxt == "σ":
            return base + "σ"
        return base + "ν"
    if canonical == "εν":
        if nxt in _LABIALS:
            return "εμ"
        if nxt in _GUTTURALS:
            return "εγ"
        if nxt == "λ":
            return "ελ"
        return "εν"
    return canonical


def _preverb_splits(bare: str) -> list[tuple[int, str, str]]:
    """Preverb chains that can open ``bare``, longest first.

    Each is (length of the chain, last preverb, its spelling there). At
    least two letters must remain as the verb.
    """
    found: dict[int, tuple[int, str, str]] = {}

    def walk(start: int, depth: int) -> None:
        if depth == 3:
            return
        for canonical, (spellings, _) in _PREVERBS.items():
            for spelling in spellings:
                end = start + len(spelling)
                if (bare.startswith(spelling, start) and len(bare) - end >= 2
                        and _composition_fits(spelling, bare[end:])):
                    found.setdefault(end, (end, canonical, spelling))
                    walk(end, depth + 1)

    walk(0, 0)
    return sorted(found.values(), reverse=True)


def _unaugmented_base(base: list[str], rest: list[str], simplex: bool
                      ) -> tuple[list[str] | None, bool]:
    """Remove the augment that opens ``rest``.

    ``base`` is the present from the same point (the verb without its
    preverbs) and ``rest`` the augmented stem from that point. Returns the
    unaugmented letters and whether an augment was seen; ``None`` when the
    stem cannot be read against the present.
    """
    b, r = _bare(base), _bare(rest)
    if not r or not b:
        return None, False
    if r in _SUPPLETIVE_AORISTS:
        plain = list(_SUPPLETIVE_AORISTS[r])
        if simplex:
            breathing = "".join(mark for letter in rest[:2] for mark in letter
                                if mark in _BREATHINGS)[:1]
            plain[1 if "".join(plain[:2]) in _DIPHTHONGS else 0] += breathing
        return plain, True
    if b in ("αιρω", "αιρομαι") and r[:2] == "ηρ":
        # αἴρω: ἦρα, ἤρθην. The aorist stem is ἀρ-, not the present's αἰρ-.
        breathing = "".join(mark for mark in rest[0] if mark in _BREATHINGS)
        return ["α" + (breathing if simplex else "")] + rest[1:], True
    group = _vowel_group(base)
    if not group:
        if r[0] == "ε" and len(r) > 1 and r[1] not in _VOWEL_LETTERS:
            # The syllabic augment. ρ doubles after it (ἔρρηξα), and in
            # epic so do λ, μ, ν and σ (ἔμμορον); a compound keeps the
            # doubled ρ after a vowel (ἀπορρήξω), as its present shows.
            tail = rest[1:]
            if r[1:2] == r[2:3] and r[1] in "ρλμνσ" and b[:2] != r[1:3]:
                tail = rest[2:]
                if simplex and r[1] == "ρ":
                    tail = ["ρ" + _ROUGH] + rest[3:]
            return tail, True
        if r[0] == "η" and r[1:2] == b[0] and b[0] in "βδμ":
            return rest[1:], True  # ἠβουλήθην, ἠδυνήθην, ἤμελλον
        if r[:2] == b[:2]:
            return rest, False  # no augment (epic θάλησα)
        return None, False
    vowel, after = b[:group], b[group:]
    if r.startswith("ερρ") and not b.startswith("ερρ"):
        # ἐρρήθην beside ἐρῶ: the stem is ῥη-.
        return (["ρ" + _ROUGH] if simplex else ["ρ"]) + rest[3:], True
    if r.startswith(vowel) and _same_onset(r[group:], after):
        return rest, False  # ἵδρυσα, ὠφέλησα, εὗρον, epic ἄμειψα
    for augment in _TEMPORAL_AUGMENTS.get(vowel, ()):
        # ηὐ- for αὐ- or εὐ- is evidence enough (αὐαίνω, ηὔηνα).
        if r.startswith(augment) and (len(augment) == 2 or _same_onset(
                r[len(augment):], after)):
            restored = [letter if simplex else _unbreathed(letter)
                        for letter in base[:group]]
            return restored + rest[len(augment):], True
    if r[0] == "ε" and len(r) > 1 and r[1] not in _VOWEL_LETTERS:
        # A consonant-initial aorist beside a vowel-initial present
        # (ἔχω ἔσχον, ἵστημι ἔστησα).
        return rest[1:], True
    return None, False


def unaugmented_stem(headword: str, augmented: str) -> tuple[str | None, str]:
    """The non-indicative stem of an augmented one, and the stem's preverb.

    ``augmented`` is a past indicative stem as LSJ prints it (ἀπελυσ,
    ἀντεταφ, εἰσηνεγκ, ἠκουσ). The first value is the stem without its
    augment, with the preverb as the present has it (ἀπολυσ, ἀντιταφ,
    εἰσενεγκ, ἀκουσ), or ``None`` when the augment cannot be identified
    against ``headword``: then only the indicative is safe to build. The
    second value is the part of ``augmented`` that precedes the augment (""
    for a simplex); the module needs it to keep the accent off the preverb.
    """
    head = _letters(headword)
    stem = _letters(augmented)
    h, a = _bare(head), _bare(stem)
    if not h or not a:
        return None, ""
    splits = _preverb_splits(h)

    def second_augment(whole: list[str], end: int) -> bool:
        # Where the present allows a preverb to end later than this reading
        # takes it, the letter after that point must not be an augment the
        # present lacks: ἠμφεσβήτουν, ἐπηνώρθωσα and ἐδιῄτησα carry a second
        # augment there, προὔβαλον a contracted one.
        bare = _bare(whole)
        for later, _, _ in splits:
            if end < later < len(bare) and bare[later] != h[later] and (
                    bare[later] in "εηω"
                    or (bare[later] in _VOWEL_LETTERS
                        and h[later] not in _VOWEL_LETTERS)):
                return True
        return False

    other: tuple[str | None, str] | None = None
    for end, canonical, spelling in splits:
        earlier = end - len(spelling)
        xi = spelling.startswith("ξ")
        for before in _PREVERBS[canonical][1]:
            if canonical == "συν" and before.startswith("ξ") != xi:
                continue
            lead = earlier + len(before)
            if a[:lead] != h[:earlier] + before or len(a) <= lead:
                continue
            prefix = _spell(stem[:lead])
            base, seen = _unaugmented_base(head[end:], stem[lead:],
                                           simplex=False)
            if base is None:
                whole = None
            elif not seen:
                whole = stem
            elif _bare(base)[0] in _VOWEL_LETTERS:
                whole = stem[:lead] + base
            elif (not _vowel_group(head[end:])
                  and _same_onset(_bare(base), h[end:])):
                whole = head[:end] + base
            else:
                composed = h[:earlier] + _compose(canonical, _bare(base), xi)
                if base[0][0] == "ρ" and composed[-1] in _VOWEL_LETTERS:
                    base = ["ρ"] + base
                whole = [head[0] if composed[0] == h[0] else composed[0]]
                whole += list(composed[1:]) + base
            if whole is not None and second_augment(whole, end):
                whole = None
            if whole is not None and seen:
                return _spell(whole), prefix
            # Without a visible augment the accent is held back only by an
            # augment that may be there unseen: a lengthened vowel (ἀφῑκόμην,
            # ἐνεῖρα), or the vowel of a stem that could not be read
            # (διαφῆκα). Epic περίχευε has none.
            if whole is None and a[lead] not in "εηω":
                continue
            if whole is not None and not _vowel_group(head[end:]):
                prefix = ""
            if other is None or (other[0] is None and whole is not None
                                 and other[1] == prefix):
                other = (_spell(whole) if whole is not None else None, prefix)
    # An augment seen at the front (ἐκάθισα, εἱστίασα beside ἑστιάω, whose
    # letters also allow ἐσ-) outweighs a preverb without one.
    base, seen = _unaugmented_base(head, stem, simplex=True)
    if base is not None and second_augment(base, 0):
        base = None
    if base is not None and (seen or other is None):
        return _spell(base), ""
    if other is not None:
        return other
    return None, ""


def _stem_preverb(head: list[str], letters: list[str]
                  ) -> tuple[int, int, str] | None:
    """Where the preverb of ``head`` ends in a perfect stem.

    Returns (length of the preverb, length of the preverbs before the last
    one, the last preverb), or ``None`` for a simplex or a stem that does
    not show the present's preverb followed by a perfect of the rest:
    its reduplication (ἀπο-λελυκ, δια-πεφυκ), an ἐ- (ἀπ-εσταλκ, παρ-εικ),
    or the vowel of a vowel-initial verb, lengthened or not (ἀπ-ηλλαχ,
    ἀφ-ῑγ, μετ-αγηοχ). The letters alone often allow two divisions
    (ἀπαν-αλίσκω, ἀπανα-λίσκω), and a wrong one misplaces the accent.
    """
    h, s = _bare(head), _bare(letters)
    for end, canonical, spelling in _preverb_splits(h):
        earlier = end - len(spelling)
        base = h[end:]
        spellings, before = _PREVERBS[canonical]
        for candidate in sorted(set(spellings) | set(before), key=len,
                                reverse=True):
            lead = earlier + len(candidate)
            rest = s[lead:]
            if s[:lead] != h[:earlier] + candidate or len(rest) < 2:
                continue
            if base[0] in _VOWEL_LETTERS:
                fits = (rest[0] == base[0]
                        or (base[0] in "αεο" and rest[0] in "ηω")
                        or (base[0] in "ιε" and rest[0] == "ε"))
            else:
                fits = ((rest[1] == "ε" and _same_onset(rest, base))
                        or (rest[0] == "ε" and (rest[1] not in _VOWEL_LETTERS
                                                or rest[1] == "ι")))
            if fits:
                return lead, earlier, canonical
    return None


def stem_preverb(headword: str, stem: str) -> str:
    """The preverb that opens an unaugmented or reduplicated ``stem``.

    The perfect of a compound keeps the accent off its preverb (ἀφῖγμαι),
    so the module is told where the preverb ends. "" for a simplex or when
    the stem does not open with a preverb of ``headword``.
    """
    letters = _letters(stem)
    found = _stem_preverb(_letters(headword), letters)
    return _spell(letters[:found[0]]) if found else ""


def _augmented_pluperfect(headword: str, stem: str) -> tuple[str, str] | None:
    """The augmented pluperfect stem of a perfect stem, and its preverb.

    λελυκ gives ἐλελυκ and ἀπολελυκ gives ἀπελελυκ. A perfect stem that
    begins with a vowel takes no further augment (ἠγγέλκειν, ἐστάλκειν),
    and only a visibly reduplicated one is augmented here.
    """
    letters = _letters(stem)
    if not letters:
        return None
    found = _stem_preverb(_letters(headword), letters)
    if not found:
        if (letters[0][0] in _VOWEL_LETTERS or len(letters) < 3
                or letters[1][0] != "ε"
                or not _same_onset(letters[0][0], letters[2][0])):
            return None
        return _spell(["ε" + _SMOOTH] + letters), ""
    lead, earlier, canonical = found
    if letters[lead][0] in _VOWEL_LETTERS or letters[lead + 1][0] != "ε":
        return None
    before = _PREVERBS[canonical][1][0]
    if _bare(letters)[earlier:lead].startswith("ξ"):
        before = "ξυν"
    opening = letters[:earlier] + list(before)
    if not earlier:
        opening[0] = before[0] + "".join(
            mark for mark in letters[0] if mark in _BREATHINGS)
    return _spell(opening + ["ε"] + letters[lead:]), _spell(opening)


def _accent_position(form: str) -> int:
    """Index of the letter that carries the tone accent, -1 without one."""
    index = -1
    for char in unicodedata.normalize("NFD", form):
        if not unicodedata.combining(char):
            index += 1
        elif char in _TONE_MARKS:
            return index
    return -1


def _contract_class(headword: str) -> str:
    """"e", "a" or "o" for a contract verb, else ""."""
    plain = strip_diacritics(headword).lower()
    for ending, vowel in (("εω", "e"), ("εομαι", "e"), ("αω", "a"),
                          ("αομαι", "a"), ("οω", "o"), ("οομαι", "o")):
        if plain.endswith(ending):
            return vowel
    return ""


# A generator directive, not a template argument: build the indicative only.
INDICATIVE_ONLY = "@indicative"


def derive_grc_conj_args(parts: dict[str, str], headword: str
                         ) -> dict[str, list[str]]:
    """Build per-tense ``{{grc-conj|...}}`` argument lists from
    principal parts.

    Returns a dict like::

        {"fut": ["fut", "λυσ", "λυθ"],
         "aor-1": ["aor-1", "ἐλυσ", "λυσ"],
         "aor-pass": ["aor-1", "", "", "ἐλυθ", "λυθ", "form=pass"],
         "perf": ["perf", "λελυκ", "λελυ"]}

    Each value is the argument vector to feed to grc-conj: the tense code,
    the positional stems, then named arguments. Tenses for which no usable
    stem can be derived are omitted.

    A past tense takes its augmented stem (the indicative) before its
    unaugmented one (every other form), in the active and in the passive
    alike. The unaugmented stem comes from :func:`unaugmented_stem`, which
    reads the augment after the preverb of a compound; when it cannot, the
    list ends in :data:`INDICATIVE_ONLY` and only the indicative is built.
    A compound passes ``prefix=``, so that the accent stays on or after the
    augment (παρέσχον), unless the printed part itself has it on the
    preverb (ἄφευσα, epic περίχευα). A middle principal part (ἐδεξάμην,
    δέξομαι) passes ``form=mid`` and builds no active cells.

    The grc-conj template computes tone accents but requires lexical
    breathings, diaeresis, iota subscript, and length marks. Remove only
    acute, grave, and perispomeni, retaining all other marks in NFC; a
    circumflex on α, ι or υ becomes a macron.
    """
    args: dict[str, list[str]] = {}
    plain_head = strip_diacritics(headword).lower()

    def stripped(form: str) -> str:
        """Remove tone accents without discarding lexical marks."""
        return _spell(_letters(form))

    def cut(form: str, endings: tuple[str, ...]) -> tuple[str, str] | None:
        """Stem and matched ending of ``form``, compared by letters."""
        letters = _letters(form)
        bare = _bare(letters)
        for ending in endings:
            if bare.endswith(ending) and len(bare) > len(ending):
                return _spell(letters[:len(letters) - len(ending)]), ending
        return None

    def named(prefix: str = "", form: str = "", resolved: bool = True,
              printed: str = "") -> list[str]:
        extra = []
        if form:
            extra.append("form=" + form)
        # The printed part shows whether the accent may stand on the
        # preverb: ἄφευσα and περίχευα have it there, παρέσχον does not.
        if prefix and not (printed and 0 <= _accent_position(printed)
                           < len(_letters(prefix))):
            extra.append("prefix=" + prefix)
        if not resolved:
            extra.append(INDICATIVE_ONLY)
        return extra

    def past(code: str, printed: str, augmented: str, voice: str,
             sure: bool = True, passive: bool = False) -> list[str]:
        """Arguments of a past tense from its augmented stem."""
        plain, prefix = unaugmented_stem(headword, augmented)
        resolved = sure and plain is not None
        stems = [augmented, plain if plain is not None else augmented]
        if passive:
            stems = ["", ""] + stems
        return [code] + stems + named(prefix, voice, resolved, printed)

    # Future. An active future in -ω builds the active and the middle, a
    # contracted one (-ῶ, by its circumflex, or uncontracted -έω) the
    # contract paradigm; a middle future (-ομαι, -οῦμαι) builds the middle
    # alone. The passive future stem (λυθήσομαι -> λυθ) joins the active
    # template or stands by itself.
    future = parts.get("fut") or parts.get("fut_med") or ""
    found = cut(future, ("ουμαι", "ομαι", "ω")) if future else None
    fut_p = cut(parts.get("fut_p", ""), ("ησομαι", "ησεται"))
    fut_p_stem = fut_p[0] if fut_p else ""
    if found:
        stem, ending = found
        contracted = _PERISPOMENI in unicodedata.normalize("NFD", future)[-2:]
        if ending == "ομαι":
            args["fut"] = ["fut", stem, "form=mid"]
        elif ending == "ουμαι":
            args["fut"] = ["fut-con-e", stem, "form=mid"]
        elif contracted and plain_head.endswith(
                ("αζω", "αννυμι", "αυνω", "αω")):
            args["fut"] = ["fut-con-a", stem]
        elif contracted:
            args["fut"] = ["fut-con-e", stem]
        elif stem.endswith("ε"):
            args["fut"] = ["fut-con-e", stem[:-1]]
        else:
            args["fut"] = ["fut", stem]
            if fut_p_stem:
                args["fut"].append(fut_p_stem)
                fut_p_stem = ""
    if fut_p_stem:
        args["fut-pass"] = ["fut", "", fut_p_stem]

    # Aorist. aor-1 and aor-2 take the augmented stem, then the unaugmented
    # one; the passive takes its own pair in a call of its own, so that an
    # unresolved augment on one side does not cost the other its paradigm.
    aor1 = parts.get("aor1") or (parts.get("aor") if parts.get("aor", "")
                                  and not parts.get("aor2")
                                  else None)
    aor2 = parts.get("aor2") or (parts.get("aor") if parts.get("aor", "")
                                  and not parts.get("aor1")
                                  else None)
    plain_aor = strip_diacritics(parts.get("aor", "")).lower()
    if (not aor1 and aor2 and parts.get("aor") != aor2
            and plain_aor.endswith(("α", "αμην"))):
        # A plain "aor." beside a numbered second aorist is the first one
        # (ἀναθάλλω: aor. ἀνέθηλα, aor. 2 ἀνέθαλον).
        aor1 = parts["aor"]
    if (not aor2 and aor1 and parts.get("aor") != aor1
            and plain_aor.endswith(("ον", "ομην"))):
        aor2 = parts["aor"]
    if not aor1 and not aor2 and parts.get("aor_med"):
        aor1 = aor2 = parts["aor_med"]
    aor_p = parts.get("aor_p")
    if aor_p and strip_diacritics(aor_p).lower().endswith("μην"):
        # A middle aorist printed in the passive section (ἐσεχύμην).
        if not aor1 and not aor2:
            aor1 = aor2 = aor_p
        aor_p = None
    for form in (aor1, aor2, parts.get("aor")):
        # A passive deponent prints its aorist in -θην under the plain label
        # (ἐβουλήθην); a root aorist in -ην (ἔβην) is not passive.
        if form and not aor_p and strip_diacritics(form).lower().endswith("θην"):
            aor_p = form
    if aor_p:
        found = cut(aor_p, ("ην",))
        if found:
            # ἔστην, ἔβην and ἔφθην are root aorists: only their
            # indicative goes like a passive one.
            root = plain_head.endswith("φθανω") or (
                plain_head.endswith(("ιστημι", "ισταμαι", "βαινω"))
                and not found[0].endswith("θ"))
            args["aor-pass"] = past("aor-1", aor_p, found[0], "pass",
                                    not root, passive=True)
        found = cut(parts.get("aor_p2", ""), ("ην",))
        if found:
            args["aor-pass-2"] = past("aor-1", parts["aor_p2"], found[0],
                                      "pass", passive=True)
    if aor1:
        found = cut(aor1, ("αμην", "α"))
        if found and not found[0].endswith(("μεθ", "σθ")):
            stem, ending = found
            if ending == "αμην":
                args["aor-1"] = past("aor-1", aor1, stem, "mid")
            elif (plain_head.endswith(("ημι", "ωμι"))
                  and stem.endswith(("ηκ", "ωκ"))):
                # ἔθηκα, ἔδωκα, ἧκα: in the active the κ stem belongs to
                # the indicative alone (θῶ, θεῖναι); the middle on it is
                # late but whole (προσηκάμην, προσήκασθαι).
                args["aor-1"] = past("aor-1", aor1, stem, "act", sure=False)
                middle = past("aor-1", aor1, stem, "mid")
                if plain_head.endswith("ιημι") and INDICATIVE_ONLY in middle:
                    # ἡκ- is augmented and unaugmented alike.
                    middle.remove(INDICATIVE_ONLY)
                args["aor-mid"] = middle
            else:
                args["aor-1"] = past("aor-1", aor1, stem, "")
    if aor2:
        found = cut(aor2, ("ομην", "ον"))
        if found:
            stem, ending = found
            args["aor-2"] = past("aor-2", aor2, stem,
                                 "mid" if ending == "ομην" else "")

    # Perfect: stem1 = active perf stem ("λελυκ" from λέλυκα -> strip
    # "α"), stem2 = mediopassive perf stem.
    #
    # The MP stem requires special handling: grc-conj's perf module
    # applies euphonic assimilation rules (μ + π/β/φ -> μμ, etc.)
    # only when the stem ends in a consonant matching one of those
    # patterns. So for ``γέγραμμαι`` (assimilated φ + μαι -> μμαι)
    # the right MP stem is ``γεγραφ`` (the unassimilated form), not
    # ``γεγραμ``. We back-derive from the active perfect stem when
    # available, or attempt a few reverse-assimilation guesses from
    # the surface MP form.
    if "pf" in parts or "pf_mp" in parts:
        act = parts.get("pf", "")
        mp_form = parts.get("pf_mp", "")
        act_stem = ""
        mp_stem = ""
        if act:
            a = stripped(act)
            for end in ("ηκα", "α", "ᾰ"):
                if a.endswith(end):
                    act_stem = a[: -len(end)] + ("ηκ" if end == "ηκα" else "")
                    break
        # If the active perfect is a -κα perfect (κ-perfect) and we
        # have an MP form, the MP stem is typically the active stem
        # minus the κ (vowel-ending stem). λέλυκα -> λελυκ; λέλυμαι
        # -> stem λελυ. Use the active stem as the basis.
        if mp_form:
            m_strip = stripped(mp_form)
            # Drop trailing 1sg "μαι" / 3sg "ται" / 3pl "νται"
            for end in ("νται", "ται", "μαι"):
                if m_strip.endswith(end):
                    mp_stem = m_strip[: -len(end)]
                    break
            # Reverse-assimilate the trailing geminate / cluster.
            # The assimilation rules in grc-conj produce these from
            # base consonants:
            #   π/β/φ + μ -> μμ (stem ends in π/β/φ)
            #   κ/γ/χ + μ -> γμ (stem ends in κ/γ/χ)
            #   τ/δ/θ + μ -> σμ (stem ends in τ/δ/θ via dental cluster)
            # We reverse the most common case: μμ -> use π/β/φ from
            # the active stem if it ends in any of those. γμ -> use
            # κ/γ/χ from the active stem. Otherwise leave the stem
            # alone.
            if act_stem and len(act_stem) > 1 and act_stem[-1] in "πβφκγχτδθ":
                if mp_stem.endswith("μ") and act_stem[-1] in "πβφ":
                    # reverse μμ -> stem-final π/β/φ
                    mp_stem = mp_stem[:-1] + act_stem[-1]
                elif mp_stem.endswith("γ") and act_stem[-1] in "κγχ":
                    mp_stem = mp_stem[:-1] + act_stem[-1]
                elif mp_stem.endswith("σ") and act_stem[-1] in "τδθ":
                    mp_stem = mp_stem[:-1] + act_stem[-1]
        if act_stem or mp_stem:
            prefixes = {stem_preverb(headword, stem)
                        for stem in (act_stem, mp_stem) if stem}
            prefix = prefixes.pop() if len(prefixes) == 1 else ""
            for printed in (act, mp_form):
                if printed and not named(prefix, printed=printed):
                    prefix = ""
            args["perf"] = ["perf", act_stem, mp_stem] + named(prefix)
            # Pluperfect: the unaugmented one on the perfect stems (Homer,
            # the Koine) and the augmented one beside it (ἐλελύκειν).
            args["plup"] = ["plup", act_stem, mp_stem] + named(prefix)
            augmented = [_augmented_pluperfect(headword, stem) if stem else ("", "")
                         for stem in (act_stem, mp_stem)]
            if all(augmented) and len({pair[1] for pair in augmented
                                       if pair[0]}) == 1:
                args["plup-aug"] = (
                    ["plup"] + [pair[0] for pair in augmented]
                    + named(next(pair[1] for pair in augmented if pair[0])))

    # Imperfect: the augmented stem, for the indicative it alone has. A
    # contracted imperfect (ἠγνόουν, ἀτίμων) takes the contract template
    # of its present.
    if "impf" in parts:
        found = cut(parts["impf"],
                    ("ουμην", "ωμην", "ομην", "ουν", "ων", "ον"))
        if found:
            stem, ending = found
            contract = _contract_class(headword)
            code = "imperf"
            if ending in ("ουν", "ουμην"):
                code = {"e": "imperf-con-e", "o": "imperf-con-o"}.get(
                    contract, "")
            elif ending in ("ων", "ωμην"):
                code = "imperf-con-a" if contract == "a" else ""
            if code:
                prefix = unaugmented_stem(headword, stem)[1]
                args["imperf"] = [code, stem] + named(
                    prefix, "mp" if ending.endswith("μην") else "",
                    printed=parts["impf"])

    return args


def future_stems(calls: dict[str, list[str]]) -> frozenset[str]:
    """The stems of the futures among the calls of
    :func:`derive_grc_conj_args`, by their letters."""
    return frozenset(
        _bare(_letters(arg))
        for tense, tense_args in calls.items() if tense.startswith("fut")
        for arg in tense_args[1:]
        if arg and "=" not in arg and arg != INDICATIVE_ONLY)


def derived_cells(tense: str, tense_args: list[str], cells,
                  printed_stems=frozenset()) -> set[str]:
    """The cells of one call of :func:`derive_grc_conj_args` that stand on
    a stem the entry does not print.

    An entry prints a past tense with its augment (ἀντετάφην, ἤγαγον). The
    stem without it (ἀντιταφ-, ἀγαγ-), which carries every form outside the
    indicative, is derived from the headword, and so is the augmented
    pluperfect (ἐλελύκειν beside the printed λέλυκα). A plain ἐ- before the
    stem is no derivation: the conjugation module takes it off by itself
    (ἐζημίωσα, ζημιῶσαι). Neither is a stem that the entry prints in its
    future (``printed_stems``, from :func:`future_stems`): ἀμφιεσ- of
    ἠμφίεσα stands in ἀμφιέσω. The cells of every other call begin with a
    stem the entry prints.
    """
    if tense == "plup-aug":
        return set(cells)
    if not tense.startswith("aor") or INDICATIVE_ONLY in tense_args:
        return set()
    stems = [arg for arg in tense_args[1:] if arg and "=" not in arg]
    if len(stems) < 2:
        return set()
    printed, plain = (_bare(_letters(stem)) for stem in stems[:2])
    if printed in (plain, "ε" + plain) or plain in printed_stems:
        return set()
    return {cell for cell in cells
            if not _bare(_letters(cell)).startswith(printed)}


_ACUTE = chr(0x0301)
# The endings of a participle in -μενος after μεν-, by the length of the
# last syllable: before a long one the accent stands on μέν.
_PARTICIPLE_SHORT = ("ον", "ε", "οι", "α", "αι")
_PARTICIPLE_LONG = ("ου", "ῳ", "ων", "οις", "ους", "η", "ης", "ῃ", "ην",
                    "αις", "ας")


def participle_cases(nominative: str) -> list[str]:
    """The other case forms of a participle in -μενος.

    λυόμενος gives λυόμενον, λυομένου, λυομένη and the rest, with the
    accent on μέν before a long ending; the perfect λελυμένος keeps it
    there throughout. Any other spelling gives nothing.
    """
    nfd = unicodedata.normalize("NFD", nominative)
    if nfd.endswith("με" + _ACUTE + "νος"):
        stem = nfd[:-len("με" + _ACUTE + "νος")]
        if any(mark in stem for mark in _TONE_MARKS):
            return []
        short = long = stem + "με" + _ACUTE + "ν"
    elif nfd.endswith("μενος"):
        stem = nfd[:-len("μενος")]
        if stem.count(_ACUTE) != 1 or _PERISPOMENI in stem:
            return []
        short = stem + "μεν"
        long = stem.replace(_ACUTE, "") + "με" + _ACUTE + "ν"
    else:
        return []
    return [unicodedata.normalize("NFC", base + ending)
            for base, endings in ((short, _PARTICIPLE_SHORT),
                                  (long, _PARTICIPLE_LONG))
            for ending in endings]


# --- Principal parts that no entry prints -----------------------------------
#
# Most LSJ verbs print no principal parts: a compound leaves them to its
# simplex (συγκατατρίβω to τρίβω), a denominative to its class (εἰκονίζω
# goes like νομίζω). The functions below state those parts. They are
# hypotheses, so the caller keeps only the cells a corpus attests.

# The tenses a compound takes from its simplex. The imperfect belongs to the
# present system, which the compound builds itself.
INFERRED_KEYS = ("fut", "fut_med", "aor", "aor1", "aor2", "aor_med", "pf",
                 "pf_mp", "fut_p", "aor_p", "aor_p2")
# Tenses that stand or fall together when an entry prints some of them.
INFERRED_FAMILIES = (
    ("fut", "fut_med"), ("aor", "aor1", "aor2", "aor_med"), ("pf",),
    ("pf_mp",), ("fut_p", "aor_p", "aor_p2"),
)
_ASPIRATES = {"τ": "θ", "π": "φ", "κ": "χ"}
_ASPIRATED_SPELLINGS = frozenset(("ανθ", "αφ", "εφ", "καθ", "μεθ", "υφ"))
_PLAIN_SPELLINGS = frozenset(("αντ", "απ", "επ", "κατ", "μετ", "υπ"))


def _is_rough(letters: list[str]) -> bool:
    return any(_ROUGH in letter for letter in letters[:2])


def _join_preverb(head: list[str], end: int, canonical: str, spelling: str,
                  part: list[str]) -> list[str] | None:
    """The preverbs of a compound before a form of its simplex.

    ``head[:end]`` is the preverb chain of the present, whose last preverb
    is ``canonical``, spelled ``spelling`` there. That preverb is spelled
    again for the sound ``part`` begins with: elided before a vowel and
    aspirated before a rough one (κατ-έτριψα, καθ-εῖλκον), assimilated
    before a consonant (συλ-λήψομαι, ἐγ-κέκλικα), ἐξ before a vowel.
    """
    if not part:
        return None
    earlier = end - len(spelling)
    xi = spelling.startswith("ξ")
    part = list(part)
    if part[0][0] in _VOWEL_LETTERS:
        group = _vowel_group(part)
        if canonical == "συν":
            preverb = "ξυν" if xi else "συν"
        elif canonical == "εισ":
            preverb = spelling
        else:
            preverb = _PREVERBS[canonical][1][0]
            if _is_rough(part) and preverb[-1] in _ASPIRATES:
                preverb = preverb[:-1] + _ASPIRATES[preverb[-1]]
        part[:2] = [_unbreathed(letter) for letter in part[:2]]
        if (preverb[-1] in _VOWEL_LETTERS and group == 1
                and part[0][0] in "ιυ"):
            part[0] += _DIAERESIS  # προϊδεῖν
    elif part[0][0] == "ρ":
        if canonical == "εν":
            return None
        preverb = _compose(canonical, "ρ", xi)
        part[0] = "ρ"
        if preverb[-1] in _VOWEL_LETTERS:
            part = ["ρ"] + part  # καταρρεύσω
    else:
        preverb = (spelling if canonical == "εισ"
                   else _compose(canonical, _bare(part), xi))
    opening = head[:earlier] + list(preverb)
    if not earlier:
        # The breathing of the present's preverb (ἀπ-, εἰσ-).
        for index in range(min(2, len(preverb))):
            if head[index][0] == preverb[index]:
                opening[index] += "".join(
                    mark for mark in head[index] if mark in _BREATHINGS)
    return opening + part


def _compound_split(headword: str, simplex: str
                    ) -> tuple[list[str], int, str, str] | None:
    """Where ``headword`` divides into its preverbs and ``simplex``."""
    head, base = _letters(headword), _letters(simplex)
    end = len(head) - len(base)
    if end <= 0 or _bare(head[end:]) != _bare(base):
        return None
    if base[0][0] == "ρ" and head[end - 1][0] == "ρ":
        end -= 1  # κατα-ρρέω
    for found, canonical, spelling in _preverb_splits(_bare(head)):
        if found == end:
            return head, end, canonical, spelling
    return None


def compound_parts(headword: str, simplex: str, parts: dict[str, str]
                   ) -> dict[str, str]:
    """The principal parts of a compound, from those of its simplex.

    ``parts`` are forms of ``simplex`` (ἔτριψα, τρίψω); the result has the
    same tenses with the preverbs of ``headword`` (συγκατέτριψα,
    συγκατατρίψω), without tone accents. A contracted future keeps its
    circumflex, which names its conjugation.
    """
    split = _compound_split(headword, simplex)
    if split is None:
        return {}
    head, end, canonical, spelling = split
    out: dict[str, str] = {}
    for key in INFERRED_KEYS:
        form = parts.get(key)
        if not form:
            continue
        joined = _join_preverb(head, end, canonical, spelling, _letters(form))
        if joined is None:
            continue
        if _PERISPOMENI in unicodedata.normalize("NFD", form)[-2:]:
            joined[-1] += _PERISPOMENI
        out[key] = _spell(joined)
    return out


def compound_simplexes(headword: str, verbs: dict[str, list[str]]
                       ) -> list[str]:
    """The verbs ``headword`` is a compound of, by its letters.

    ``verbs`` maps the bare letters of every known verb to its spellings.
    The answer is the verb left after the longest chain of preverbs (τρίβω
    for συγκατατρίβω), provided every shorter chain that also leaves a verb
    leaves a compound of it (κατατρίβω). Two readings that disagree
    (ὑπ-ερείδω, ὑπερ-είδω) give none. An aspirated preverb needs a rough
    verb after it and a plain elided one a smooth verb (καθ-έλκω, not
    κατ-έλκω).
    """
    head = _letters(headword)
    bare = _bare(head)
    found: dict[int, list[str]] = {}
    for end, _, spelling in _preverb_splits(bare):
        rest = bare[end:]
        if rest.startswith("ρρ") and spelling[-1] in _VOWEL_LETTERS:
            rest = rest[1:]
        fitting = []
        for verb in verbs.get(rest, ()):
            if rest[0] in _VOWEL_LETTERS:
                rough = _is_rough(_letters(verb))
                if ((spelling in _ASPIRATED_SPELLINGS and not rough)
                        or (spelling in _PLAIN_SPELLINGS and rough)):
                    continue
            fitting.append(verb)
        if fitting:
            found[end] = fitting
    if not found:
        return []
    longest = max(found)
    for end in found:
        if end != longest and longest - end not in {
                split[0] for split in _preverb_splits(bare[end:])}:
            return []
    return found[longest]


# Verb classes whose other tenses follow from the present: the ending of
# the present, the letters that replace it before the tense suffix, and
# whether the stem ends in a dental (νομίζω, ἐνόμισα, νενόμισμαι).
_REGULAR_CLASSES = (
    ("ιζω", "ι", True), ("αζω", "α", True), ("ευω", "ευ", False),
    ("εω", "η", False), ("οω", "ω", False), ("αω", "η", False),
)
_AUGMENTED_VOWELS = {
    "α": ("η",), "ε": ("η",), "ο": ("ω",), "αι": ("ῃ", "αι"),
    "ει": ("ῃ", "ει"), "οι": ("ῳ", "οι"), "αυ": ("ηυ", "αυ"),
    "ευ": ("ηυ", "ευ"),
}


def _augmented(stem: list[str]) -> list[list[str]]:
    """``stem`` with the augment, in each spelling the augment takes.

    A diphthong is also left as it is: εἰκόνισα and εὐλόγησα stand beside
    ᾐκόνισα and ηὐλόγησα.
    """
    first = stem[0][0]
    if first == "ρ":
        return [["ε" + _SMOOTH, "ρ", "ρ"] + stem[1:]]
    if first not in _VOWEL_LETTERS:
        return [["ε" + _SMOOTH] + stem]
    group = _vowel_group(stem)
    vowel = _bare(stem[:group])
    breathing = "".join(mark for letter in stem[:group] for mark in letter
                        if mark in _BREATHINGS)[:1]
    out = []
    for augment in _AUGMENTED_VOWELS.get(vowel, ()):
        if augment == vowel:
            out.append(list(stem))
        elif augment in ("ῃ", "ῳ"):
            long = "η" if augment == "ῃ" else "ω"
            out.append([long + breathing + _SUBSCRIPT] + stem[group:])
        elif len(augment) == 2:
            out.append([augment[0], augment[1] + breathing] + stem[group:])
        else:
            out.append([augment + breathing] + stem[group:])
    return out or [list(stem)]


def _reduplicated(stem: list[str]) -> list[list[str]]:
    """``stem`` as a perfect: reduplicated, or augmented where a verb
    cannot reduplicate (a vowel, ῥ, a double consonant, most clusters)."""
    first = stem[0][0]
    second = stem[1][0] if len(stem) > 1 else ""
    if first in _VOWEL_LETTERS or first == "ρ":
        return _augmented(stem)
    stop_and_liquid = (first in "πβφτδθκγχ" and second in "λρμν"
                       and first + second != "γν")
    if first in "ζξψ" or not (second in _VOWEL_LETTERS or stop_and_liquid):
        return [["ε" + _SMOOTH] + stem]
    plain = {"φ": "π", "θ": "τ", "χ": "κ"}.get(first, first)
    return [[plain, "ε"] + stem]


def regular_parts(verb: str) -> list[dict[str, str]]:
    """The principal parts a regular denominative verb would have.

    For a verb in -ίζω, -άζω, -εύω, -έω, -όω or -άω (or the middle of one)
    with a syllable before that ending: one dict per spelling of the
    augment, each with the future, the aorist, the perfect and the passive
    (εἰκονίζω: εἰκονίσω, ᾐκόνισα or εἰκόνισα, ...), without tone accents.
    A verb in -ίζω also gets its Attic future (νομιῶ) in a dict of its own.
    Anything else gives an empty list.
    """
    letters = _letters(verb)
    bare = _bare(letters)
    for ending, vowel, dental in _REGULAR_CLASSES:
        middle = bare.endswith(ending[:-1] + "ομαι")
        if not middle and not bare.endswith(ending):
            continue
        size = len(ending) + (3 if middle else 0)
        root = letters[:len(letters) - size]
        # A short verb is not a denominative (ἐρέω, ὁράω, ἀρόω, χέω).
        if len(root) < 3 or not any(letter[0] in _VOWEL_LETTERS
                                    for letter in root):
            return []
        if dental or ending == "ευω":
            theme = letters[len(root):len(root) + len(vowel)]
        elif ending == "αω" and root[-1][0] in "ειρ":
            theme = ["α" + _MACRON]
        else:
            theme = [vowel]
        stem = root + theme
        sigma = ["σ"] if dental else []
        variants = []
        augmented, reduplicated = _augmented(stem), _reduplicated(stem)
        for index, past in enumerate(augmented):
            perfect = reduplicated[min(index, len(reduplicated) - 1)]
            parts = {
                "aor_p": _spell(past + sigma + list("θην")),
                "pf_mp": _spell(perfect + sigma + list("μαι")),
                "fut_p": _spell(stem + sigma + list("θησομαι")),
            }
            if middle:
                parts["fut_med"] = _spell(stem + list("σομαι"))
                parts["aor_med"] = _spell(past + list("σαμην"))
            else:
                parts["fut"] = _spell(stem + list("σω"))
                parts["aor"] = _spell(past + list("σα"))
                # A dental stem with nothing before it to mark the perfect
                # would give the adjective in -ικός, -ακός as its participle
                # (ἱππίζω: ἱππικός).
                if not dental or perfect != stem:
                    parts["pf"] = _spell(perfect + list("κα"))
            variants.append(parts)
        if ending == "ιζω":
            if middle:
                variants.append({"fut_med": _spell(stem + list("ουμαι"))})
            else:
                contracted = stem + ["ω" + _PERISPOMENI]
                variants.append({"fut": _spell(contracted)})
        return variants
    return []


def other_voice(bare: str) -> str:
    """The bare spelling of the same verb in the other voice, or ""."""
    for ending, other in (("ομαι", "ω"), ("μαι", "μι"), ("ω", "ομαι"),
                          ("μι", "μαι")):
        if bare.endswith(ending):
            return bare[:-len(ending)] + other
    return ""


# Classes whose regular tenses coincide: -ᾱσα of -άω and -ασα of -άζω, -ησα
# of -έω and of -άω, and the aorist of -όω with the feminine participle of
# -άω (δαπανῶσαι).
_RIVAL_ENDINGS = {"αω": ("αζω", "εω", "οω"), "αζω": ("αω",), "εω": ("αω",),
                  "οω": ("αω",)}


def regular_rivals(verb: str) -> list[str]:
    """Bare spellings of the verbs that would share ``verb``'s regular
    tenses: the same stem in a class with the same tense stems, and the
    same verb with the other of ἀ- and ἐ-, which take one augment
    (ἠπόρημαι is the perfect of ἀπορέω). Each in both voices."""
    bare = _bare(_letters(verb))
    active = bare[:-4] + "ω" if bare.endswith("ομαι") else bare
    rivals = []
    for ending, others in _RIVAL_ENDINGS.items():
        if active.endswith(ending):
            rivals += [active[:-len(ending)] + other for other in others]
            break
    if active[0] in "αε" and active[1:2] not in ("ι", "υ"):
        rivals.append(("ε" if active[0] == "α" else "α") + active[1:])
    return rivals + [other_voice(rival) for rival in rivals]


def regular_compound_parts(headword: str) -> list[dict[str, str]]:
    """Regular parts with the augment after a preverb of ``headword``.

    A denominative whose first element is a preposition takes its augment
    after it (ἐπιστατέω, ἐπεστάτησα) whether or not the rest is a verb.
    One set of :func:`regular_parts` for each preverb chain the letters
    allow; the rest is read as rough after an aspirated preverb (ἐφ-).
    """
    head = _letters(headword)
    out: list[dict[str, str]] = []
    for end, _, spelling in _preverb_splits(_bare(head)):
        rest = list(head[end:])
        if rest[0][0] == "ρ" and head[end - 1][0] in _VOWEL_LETTERS:
            continue
        if rest[0][0] in _VOWEL_LETTERS:
            breathing = _ROUGH if spelling in _ASPIRATED_SPELLINGS else _SMOOTH
            rest[_vowel_group(rest) - 1] += breathing
        simplex = _spell(rest)
        for parts in regular_parts(simplex):
            parts = compound_parts(headword, simplex, parts)
            if parts and parts not in out:
                out.append(parts)
    return out


def short_verb_compound(headword: str) -> bool:
    """Whether ``headword`` may be a compound of a verb too short to be a
    denominative: ἀπο-θέω, ἐμ-πλέω, κατα-ρρέω. Its tenses are not those of
    its ending's class (ἀποθήσω is the future of ἀποτίθημι)."""
    head = _letters(headword)
    return any(not regular_parts(_spell(head[end:]))
               for end, _, _ in _preverb_splits(_bare(head)))


__all__ = [
    "INDICATIVE_ONLY",
    "INFERRED_FAMILIES",
    "INFERRED_KEYS",
    "PART_KEYS",
    "compound_parts",
    "compound_simplexes",
    "parse_principal_parts",
    "derive_grc_conj_args",
    "ends_at_label",
    "other_voice",
    "regular_compound_parts",
    "regular_parts",
    "regular_rivals",
    "derived_cells",
    "future_stems",
    "participle_cases",
    "short_verb_compound",
    "stem_preverb",
    "strip_diacritics",
    "unaugmented_stem",
]


if __name__ == "__main__":
    # Smoke test: print parsed output for a few hand-picked verbs.
    import json
    import sys
    from pathlib import Path
    GLOSSES = Path.home() / "Documents" / "lsj9" / "lsj9_glosses.jsonl"
    if not GLOSSES.exists():
        print(f"Cannot find {GLOSSES}", file=sys.stderr)
        sys.exit(1)
    heads: dict[str, str] = {}
    with open(GLOSSES) as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            hw = e.get("headword")
            if hw and "level" not in e and "number" not in e and hw not in heads:
                heads[hw] = e["text"]
    for v in ("λύω", "γράφω", "λείπω", "βάλλω", "παύω", "παιδεύω",
              "πέμπω", "τρέχω", "πίπτω", "ἔρχομαι"):
        if v in heads:
            print(f"=== {v} ===")
            parts = parse_principal_parts(heads[v], v)
            print(f"  parts: {parts}")
            args = derive_grc_conj_args(parts, v)
            print(f"  grc-conj args: {args}")

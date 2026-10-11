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
  - Section markers ``:—Med.``, ``:—Pass.``, ``Med. and Pass.``
  - Dual aorist labels (``aor. 1``, ``aor. 2``)
  - Author-attribution citation suffixes after each form

What it deliberately does NOT do:
  - Resolve cross-reference verbs (``v. καταχέω``)
  - Expand suffix-only abbreviations (``fut. -ψω``) - those would need
    to know which letters of the headword to drop, which is itself a
    morphological judgement; we leave that to downstream Lua expansion
    when a full form is given instead.
  - Pick variant forms; the first attestation wins.
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
    "fut_p", "aor_p", "pf_mp", "futp",
)


def strip_diacritics(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if not unicodedata.combining(c))


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
    r"(?:(?P<num>[12])\s*)?"
    r"(?P<voice>(?:m\.\s*/\s*p\.|m\.|p\.|mid\.|pass\.|m/p\.))?",
    re.UNICODE,
)


# Section marker like ":—Med.", ":—Pass.", ":-Med.,", "—Med.".
# Returns the new section label.
_SECTION_RE = re.compile(
    r"(?::|—|:-)\s*(?:—\s*)?"
    r"(?P<sect>Med\.\s+and\s+Pass\.?|Mediopass\.?|Med\.?|Pass\.?|Mid\.?)",
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
    for m in _SECTION_RE.finditer(text):
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
        return unicodedata.normalize("NFC", word)
    return None


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
            num = m.group("num")
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
            # to ignore for the canonical principal-part slot.
            if key not in out:
                out[key] = form

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


def _stem_active_indicative_singular(form: str, ending: str) -> str | None:
    """Strip the given ending from ``form`` if it matches.

    Comparison is on the diacritic-free lowercase form so the strip
    works on both ``ἔγραψα`` (ε + accent) and ``εγραψα``.
    """
    plain = strip_diacritics(form).lower()
    if not plain.endswith(ending):
        return None
    # Slice the original form by the same number of characters since
    # NFC code points correspond 1:1 with the stripped string here
    # (the combining marks are folded into base codepoints).
    return form[: -len(ending)]


def derive_grc_conj_args(parts: dict[str, str], headword: str
                         ) -> dict[str, list[str]]:
    """Build per-tense ``{{grc-conj|...}}`` argument lists from
    principal parts.

    Returns a dict like::

        {"fut": ["fut", "λυσ", "λυθ"],
         "aor-1": ["aor-1", "ελυσ", "λυσ", "λυθ", "ελυθ"],
         "perf": ["perf", "λελυκ", "λελυ"]}

    Each value is the argument vector to feed to grc-conj (tense code
    first, then positional stems). Tenses for which we cannot derive a
    usable stem are simply omitted.

    The grc-conj template computes tone accents but requires lexical
    breathings, diaeresis, iota subscript, and length marks. Remove only
    acute, grave, and perispomeni, retaining all other marks in NFC.
    """
    args: dict[str, list[str]] = {}

    def stripped(form: str) -> str:
        """Remove tone accents without discarding lexical marks."""
        return unicodedata.normalize("NFC", "".join(
            char for char in unicodedata.normalize("NFD", form)
            if char not in {"\u0300", "\u0301", "\u0342"}
        )).lower()

    # Future: stem1 = active future stem (strip ω from λύσω -> λυσ),
    # stem2 = passive future stem (strip ησομαι from λυθήσομαι -> λυθ).
    # Deponent futures (πεσοῦμαι, ἐλεύσομαι) don't have an active
    # principal part, so we synthesize stem1 from the middle form
    # by stripping ομαι/ουμαι and feeding it as ``form=mp`` to grc-conj
    # by leaving stem1 empty. This matches Wiktionary's convention.
    if "fut" in parts or "fut_med" in parts:
        fut_form = parts.get("fut", "")
        fut_stem = ""
        if fut_form:
            fp = stripped(fut_form)
            if fp.endswith("ω"):
                fut_stem = fp[:-1]
            elif fp.endswith("ουμαι"):
                # Deponent active-coded as middle: keep the οῦμαι form
                # but strip "ουμαι" -> contract-future stem; we treat
                # it as fut_med below instead and leave fut_stem empty.
                fut_stem = ""
        if not fut_stem and "fut_med" in parts:
            fp = stripped(parts["fut_med"])
            for end in ("ουμαι", "ομαι"):
                if fp.endswith(end):
                    # We don't currently have a stem slot for the
                    # middle, so we skip - conservative default.
                    break
        fut_p = parts.get("fut_p", "")
        fut_p_stem = ""
        if fut_p:
            fp = stripped(fut_p)
            for end in ("ησομαι", "ησεται"):
                if fp.endswith(end):
                    fut_p_stem = fp[: -len(end)]
                    break
        if fut_stem:
            args["fut"] = ["fut", fut_stem]
            if fut_p_stem:
                args["fut"].append(fut_p_stem)

    # Aorist: aor-1 takes augmented + non-augmented + passive +
    # augmented-passive stems. ``ἔλυσα`` -> "ελυσ"; non-augmented
    # ``λύσω``-derived "λυσ"; passive "λυθ" from ``ἐλύθην`` -> "λυθ"
    # after stripping "ην" + augment.
    aor1 = parts.get("aor1") or (parts.get("aor") if parts.get("aor", "")
                                  and not parts.get("aor2")
                                  else None)
    aor2 = parts.get("aor2") or (parts.get("aor") if parts.get("aor", "")
                                  and not parts.get("aor1")
                                  else None)
    aor_p = parts.get("aor_p")

    aor_p_stem = ""
    aor_p_aug_stem = ""
    if aor_p:
        ap = stripped(aor_p)
        # ἐλύθην -> ελυθην; strip "ην" -> ελυθ (augmented). The
        # non-augmented form is the same minus the leading "ε" if
        # the head starts with augment "ε" + consonant.
        if ap.endswith("ην"):
            aug = ap[:-2]
            aor_p_aug_stem = aug
            aor_p_stem = _strip_augment(aug)

    if aor1:
        a = stripped(aor1)
        if a.endswith("α") or a.endswith("ᾰ"):
            aug_stem = a[:-1]
            non_aug_stem = _strip_augment(aug_stem)
            grc_args = ["aor-1", aug_stem, non_aug_stem]
            if aor_p_stem:
                grc_args.append(aor_p_stem)
            if aor_p_aug_stem:
                grc_args.append(aor_p_aug_stem)
            args["aor-1"] = grc_args

    if aor2:
        a = stripped(aor2)
        if a.endswith("ον") or a.endswith("ομην"):
            # ἔλιπον -> ελιπον; strip "ον" -> ελιπ
            cut = 3 if a.endswith("ομην") else 2
            aug_stem = a[:-cut]
            non_aug_stem = _strip_augment(aug_stem)
            grc_args = ["aor-2", aug_stem, non_aug_stem]
            if aor_p_stem:
                grc_args.append(aor_p_stem)
            if aor_p_aug_stem:
                grc_args.append(aor_p_aug_stem)
            args["aor-2"] = grc_args

    # Aorist passive standalone (when no active aorist is provided
    # but a passive aorist is). grc-conj requires an active stem,
    # so we leave stem1 empty (the template treats this as
    # passive-only via args.form='pass').
    if not aor1 and not aor2 and aor_p_stem:
        args["aor-1"] = ["aor-1", "", "", aor_p_stem, aor_p_aug_stem]

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
                # Same logic when the act_stem ends in a vowel + κ
                # (e.g. λελυκ from λέλυκα): leave the MP stem at the
                # vowel since the κ is the perfect-active marker.
                elif act_stem.endswith("κ"):
                    # MP form's stem doesn't include κ; trust the
                    # surface mp_stem.
                    pass
        if act_stem or mp_stem:
            args["perf"] = ["perf", act_stem, mp_stem]

    # Pluperfect: same stems as perfect plus an augment that grc-conj
    # adds itself, so we just rebuild from pf/pf_mp.
    if "perf" in args:
        args["plup"] = ["plup", args["perf"][1], args["perf"][2]]

    # Imperfect: stem1 is the augmented imperfect stem; the present
    # template handles non-augmented forms separately.
    if "impf" in parts:
        imp = stripped(parts["impf"])
        for end in ("ον", "ομην"):
            if imp.endswith(end):
                args["imperf"] = ["imperf", imp[: -len(end)]]
                break

    return args


# Greek vowel set used for augment recognition.
_GREEK_VOWELS = set("αεηιουωᾰᾱ")
_AUG_REPLACEMENTS = {
    "ε": "",     # ε-augment (ε + consonant -> consonant), ἔλυσα -> λυσ
    "η": "α",    # temporal augment α -> η: ἤγαγον -> αγαγον
    "ω": "ο",    # temporal augment ο -> ω: ὤφειλον -> οφειλ-
    "ηυ": "αυ",
    "ευ": "ευ",  # diphthong unchanged
    "οι": "οι",
    "ου": "ου",
    "ει": "ει",
    "ι": "ι",
    "υ": "υ",
}


def _strip_augment(stem: str) -> str:
    """Best-effort stripping of the syllabic / temporal augment.

    Returns the stem unchanged when no recognizable augment is found.
    Heuristics:
      - Leading ``ε`` followed by a consonant -> drop ``ε``
        (syllabic augment)
      - Leading ``η`` -> ``α`` (temporal augment of α-)
      - Leading ``ω`` -> ``ο`` (temporal augment of ο-)
      - Leading ``η`` followed by ``υ`` -> ``α`` + ``υ``

    This preserves the existing simple augment policy while retaining
    marks on every surviving letter.
    Compounds with preposition prefixes (e.g. προσέλιπον) are NOT
    handled; those need a separate prefix table.
    """
    nfd = unicodedata.normalize("NFD", stem)
    if not nfd:
        return stem
    boundary = 1
    while boundary < len(nfd) and unicodedata.combining(nfd[boundary]):
        boundary += 1
    following = nfd[boundary:]
    if nfd[0] == "ε" and following and following[0] not in _GREEK_VOWELS:
        # The augment and its marks disappear together.
        return unicodedata.normalize("NFC", following)
    if nfd[0] in {"η", "ω"}:
        first = "α" if nfd[0] == "η" else "ο"
        return unicodedata.normalize("NFC", first + nfd[1:])
    return unicodedata.normalize("NFC", stem)


__all__ = [
    "PART_KEYS",
    "parse_principal_parts",
    "derive_grc_conj_args",
    "strip_diacritics",
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

"""Read Diorisis word forms without duplicating closing quotation marks.

The XML sometimes puts a closing quotation mark in a word's Beta Code
``form``, with or without repeating it in the following ``punct`` node.
A terminal apostrophe also marks genuine elision, so punctuation alone
cannot distinguish the two. Only an independently present, identically
annotated cell in the same document licenses removing the quotation suffix.
All original readings and witness locations remain in a reversible audit.
"""

from __future__ import annotations

from collections import defaultdict
import unicodedata
import xml.etree.ElementTree as ET


def _beta_key(beta: str) -> str:
    # A contextual grave does not change the lexical cell. Keep case and
    # every other Beta Code mark, including actual elision, unchanged.
    return beta.replace("\\", "/")


def _readings(word: ET.Element):
    lemmas = word.findall("lemma")
    if not lemmas:
        yield "", "", ""
    for lemma in lemmas:
        entry = unicodedata.normalize("NFC", lemma.get("entry") or "")
        pos = (lemma.get("POS") or "").lower()
        analyses = lemma.findall("analysis")
        if not analyses:
            yield entry, pos, ""
        for analysis in analyses:
            morph = " ".join((analysis.get("morph") or "").split())
            yield entry, pos, morph


def _closing_quotes(mark: str) -> int:
    # This is the source's single-quote notation, not a rule about all
    # punctuation or all apostrophes. A breathing/elision ')' is unrelated.
    return len(mark) - len(mark.rstrip("’'"))


def _plain_sentence_punctuation(mark: str) -> bool:
    return bool(mark) and all(char in ".;:!?··" for char in mark)


def _initial_beta_letter(beta: str) -> str | None:
    letters = frozenset("abgdezhqiklmncoprstufxyw")
    if any(char.lower() not in letters and char not in "*()\\/=+|'"
           for char in beta):
        return None
    return next((char.lower() for char in beta
                 if char.isascii() and char.isalpha()), None)


def restore_source_beta(normalized: str, audit_record: dict) -> str:
    """Reverse one audited normalization, refusing a different input cell."""
    if normalized != audit_record["normalized_beta"]:
        raise ValueError("normalized Beta Code does not match its audit record")
    return audit_record["original_beta"]


class DiorisisSourceContext:
    """Resolve supported duplicate quote suffixes within one XML document.

    ``beta(word)`` preserves unresolved apostrophes, including genuine
    elisions before quotation punctuation. ``audit`` records the complete
    finite quotation-candidate set, original Beta Code, proposed cell,
    following punctuation, every annotation, and supporting source loci.
    Conversion to each reader's Unicode/elision convention happens later.
    """

    def __init__(self, root: ET.Element):
        words = []
        sentence_final_words = set()
        for parent in root.iter():
            children = list(parent)
            last_word = next((child for child in reversed(children)
                              if child.tag == "word"), None)
            for index, word in enumerate(children):
                if word.tag != "word":
                    continue
                following = children[index + 1] if index + 1 < len(children) else None
                punctuation = (following.get("mark", "")
                               if following is not None and following.tag == "punct" else "")
                locus = {"sentence_id": parent.get("id") if parent.tag == "sentence" else None,
                         "location": parent.get("location"), "word_id": word.get("id")}
                words.append((word, punctuation, locus))
                if parent.tag == "sentence" and word is last_word:
                    sentence_final_words.add(word)

        plain_quote_boundaries = {}
        for index, (word, punctuation, _locus) in enumerate(words):
            following_beta = (words[index + 1][0].get("form", "")
                              if index + 1 < len(words) else "")
            initial = _initial_beta_letter(following_beta)
            # Across a sentence/verse boundary a following vowel can still
            # license elision. Missing/unknown continuation is ambiguous too.
            if (word in sentence_final_words
                    and _plain_sentence_punctuation(punctuation)
                    and initial is not None and initial not in "aehiouw"):
                plain_quote_boundaries[word] = following_beta

        witnesses = defaultdict(list)
        for word, punctuation, locus in words:
            original = word.get("form", "")
            # An elided cell outside a closing-quotation boundary is also
            # valid evidence: removing an outer quote may leave its elision.
            if original.endswith("'") and (
                    _closing_quotes(punctuation) or word in plain_quote_boundaries):
                continue
            for reading in _readings(word):
                if not all(reading):
                    continue
                witnesses[(_beta_key(original), *reading)].append(
                    {**locus, "original_beta": original})

        self._normalized = {}
        self.audit = []
        for word, punctuation, locus in words:
            original = word.get("form", "")
            count = _closing_quotes(punctuation)
            plain_boundary = not count and word in plain_quote_boundaries
            if plain_boundary:
                count = 1
            if not count or not original.endswith("'" * count):
                continue
            candidate = original[:-count]
            readings = sorted(set(_readings(word)))
            evidence = []
            for entry, pos, morph in readings:
                matching = witnesses.get((_beta_key(candidate), entry, pos, morph), [])
                if matching:
                    evidence.append({"lemma": entry, "pos": pos, "morph": morph,
                                     "witnesses": matching})
            supported = {(row["lemma"], row["pos"], row["morph"]) for row in evidence}
            unresolved = [reading for reading in readings if reading not in supported]
            # One supported reading must not erase another possible elided
            # reading. A single surface is changed only on complete agreement.
            resolved = bool(readings) and not unresolved
            normalized = candidate if resolved else original
            self._normalized[word] = normalized
            self.audit.append({**locus, "original_beta": original,
                               "normalized_beta": normalized,
                               "candidate_beta": candidate,
                               "following_punctuation": punctuation,
                               "reason": (("terminal_quote_with_same_source_cell"
                                           if plain_boundary else
                                           "duplicated_closing_quote_with_same_source_cell")
                                          if resolved else "unresolved_terminal_quote_or_elision"),
                               "readings": [{"lemma": e, "pos": p, "morph": m}
                                            for e, p, m in readings],
                               "evidence": evidence,
                               "unresolved_readings": [{"lemma": e, "pos": p, "morph": m}
                                                       for e, p, m in unresolved]})
            if plain_boundary:
                self.audit[-1]["quote_boundary"] = "plain_sentence_punctuation"
                self.audit[-1]["following_beta"] = plain_quote_boundaries[word]

    def beta(self, word: ET.Element) -> str:
        return self._normalized.get(word, word.get("form", ""))

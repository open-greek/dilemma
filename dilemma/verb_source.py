"""Shared checks for Ancient Greek verb-table source rows."""

# Forms of εἰμί that Wiktionary's periphrastic cells contribute to the verb
# they conjugate: "τετιμηκότες εἶτε" leaves εἶτε tagged as a perfect optative
# of τιμάω. The mood must agree too: ἔστω is also a genuine aorist middle
# indicative of ἵστημι, distinct from the imperative auxiliary.
_EIMI_FORMS_BY_MOOD = {
    "subjunctive": {"ὦ", "ᾖς", "ᾖ", "ὦμεν", "ἦτε", "ὦσι", "ὦσιν"},
    "optative": {"εἴην", "εἴης", "εἴη", "εἶμεν", "εἴημεν", "εἶτε", "εἴητε",
                 "εἶεν", "εἴησαν"},
    "infinitive": {"εἶναι"},
    "imperative": {"ἔστω", "ἔστων", "ὄντων"},
}
EIMI_AUXILIARY_FORMS = set().union(*_EIMI_FORMS_BY_MOOD.values())

# The source pipeline also emits these monotonic copies of the plural
# perfect optative auxiliaries. Do not normalize every auxiliary spelling:
# rough-breathing εἷμεν / εἷτε are genuine aorists of ἵημι, and ἑστῶ
# belongs to ἵστημι. The explicit perfect-optative context distinguishes
# the reviewed copies from those independent verb forms.
_PERFECT_OPTATIVE_AUXILIARY_COPIES = {"είμεν", "είτε", "είεν"}


def is_periphrastic_auxiliary(form, lemma, tags):
    """Whether a source row contains εἰμί rather than its own verb."""
    if lemma in ("εἰμί", "εἶμι"):
        return False
    tags = set(tags)
    moods = tags & {"indicative", "subjunctive", "optative", "imperative",
                    "infinitive", "participle"}
    tenses = tags & {"present", "imperfect", "future", "aorist", "perfect",
                     "pluperfect", "future-perfect"}
    return (any(form in forms and moods == {mood}
                for mood, forms in _EIMI_FORMS_BY_MOOD.items())
            or (form in _PERFECT_OPTATIVE_AUXILIARY_COPIES
                and tenses == {"perfect"} and moods == {"optative"}))



"""Identify complete grammatical readings of stemless Ancient verb tables."""

import unicodedata


_DIALECT_NAMES = {"": "", "att": "Attic", "epi": "Epic", "ion": "Ionic",
                  "koi": "Koine", "dor": "Doric", "aeo": "Aeolic"}

# These are the empty-stem outputs of grc-conj aor-1. A missing argument
# alone is not evidence: indicative-only and passive-only templates omit
# the active/middle nonpast stem deliberately.
_EMPTY_AORIST_NONPAST = {
    "active": {
        "subjunctive": {"ω", "ης", "η", "ητον", "ωμεν", "ητε", "ωσι", "ωσιν"},
        "optative": {"αιμι", "ειας", "αις", "ειε", "ειεν", "αι", "αιτον",
                     "αιτην", "αιμεν", "αιτε", "ειαν", "αιεν"},
        "imperative": {"ον", "ατω", "ατον", "ατων", "ατε", "αντων"},
        "infinitive": {"αι"},
        "participle": {"masculine": {"ας"}, "feminine": {"ασα"}, "neuter": {"αν"}},
    },
    "middle": {
        "subjunctive": {"ωμαι", "η", "ηται", "ησθον", "ωμεθα", "ησθε", "ωνται",
                        "ομαι", "εαι", "εται", "ωμεσθα", "ομεθα", "ομεσθα"},
        "optative": {"αιμην", "αιο", "αιτο", "αισθον", "αισθην", "αιμεθα",
                     "αισθε", "αιντο", "αιμεσθα", "αιατο"},
        "imperative": {"αι", "ασθω", "ασθον", "ασθων", "ασθε"},
        "infinitive": {"ασθαι"},
        "participle": {"masculine": {"αμενος"}, "feminine": {"αμενη"}, "neuter": {"αμενον"}},
    },
}


def stemless_aorist_nonpast_voices(entry):
    """Map dialects to voices positively missing their aor-1 nonpast stem.

    Every competing aorist template for that dialect and voice must lack
    stem 3, supply stem 2, and have no indicative-only declaration. A
    complete template or unsupported template syntax prevents attribution.
    """
    voice_modes = {"": {"active", "middle"}, "am": {"active", "middle"},
                   "act": {"active"}, "mid": {"middle"}, "ap": {"active"},
                   "mp": {"middle"}, "pass": set()}
    witnesses = {}
    for template in entry.get("inflection_templates") or []:
        args = template.get("args") or {}
        if template.get("name") != "grc-conj" or not str(args.get("1", "")).startswith("aor-"):
            continue
        dialect = _DIALECT_NAMES.get(args.get("dial", ""))
        if dialect is None:
            continue
        mode = args.get("form", "")
        voices = voice_modes.get(mode, {"active", "middle"})
        missing = (args.get("1") == "aor-1" and mode in voice_modes
                   and bool(args.get("2")) and not args.get("3")
                   and "indonly" not in args)
        for voice in voices:
            witnesses.setdefault((dialect, voice), []).append(missing)
    result = {}
    for (dialect, voice), claims in witnesses.items():
        if all(claims):
            result.setdefault(dialect, set()).add(voice)
    return {dialect: frozenset(voices) for dialect, voices in result.items()}


def retain_complete_aorist_nonpast_readings(form, tags, *, stemless_voices=frozenset()):
    """Withhold ending-only readings from a positively incomplete source table.

    The original indicative, other tense/mood, complete forms, and competing
    voices remain. Missing or conflicting finite grammatical axes are not
    resolved by guessing. Participle header forms use the supplied gender;
    extraneous person/number labels do not supply the absent lexical stem.
    """
    claims = set(tags)
    tenses = claims & {"present", "imperfect", "future", "aorist", "perfect",
                       "pluperfect", "future-perfect"}
    moods = claims & {"indicative", "subjunctive", "optative", "imperative",
                      "infinitive", "participle"}
    if not stemless_voices or tenses != {"aorist"} or len(moods) != 1:
        return tags
    mood, = moods
    if mood == "indicative":
        return tags
    if mood not in {"infinitive", "participle"}:
        if (len(claims & {"first-person", "second-person", "third-person"}) != 1
                or len(claims & {"singular", "dual", "plural"}) != 1):
            return tags
    letters = "".join(c for c in unicodedata.normalize("NFD", form)
                      if unicodedata.category(c) != "Mn").lower()
    rejected = set()
    for voice in claims & stemless_voices:
        endings = _EMPTY_AORIST_NONPAST[voice][mood]
        if mood == "participle":
            genders = claims & {"masculine", "feminine", "neuter"}
            if len(genders) != 1:
                continue
            gender, = genders
            endings = endings[gender]
        if letters in endings:
            rejected.add(voice)
    if not rejected:
        return tags
    if claims & {"active", "middle", "mediopassive", "passive"} - rejected:
        return [tag for tag in tags if tag not in rejected]
    return None


# An empty passive stem in a source conjugation table produces these
# endings as whole words. The Ionic aorist table of τύπτω demonstrates
# the problem: its active stem exists, but its passive stem is missing.
# Compare the complete grammatical cell, rather than banning a spelling
# that could independently belong to another verb or part of speech.
_STEMLESS_PASSIVE_AORIST = {
    "indicative_1sg": "ην", "indicative_2sg": "ης",
    "indicative_3sg": "η", "indicative_2du": "ητον",
    "indicative_3du": "ητην", "indicative_1pl": "ημεν",
    "indicative_2pl": "ητε", "indicative_3pl": "ησαν",
    "subjunctive_1sg": "ω", "subjunctive_2sg": "ης",
    "subjunctive_3sg": "η", "subjunctive_2du": "ητον",
    "subjunctive_3du": "ητον", "subjunctive_1pl": "ωμεν",
    "subjunctive_2pl": "ητε", "subjunctive_3pl": "ωσι",
    "optative_1sg": "ειην", "optative_2sg": "ειης",
    "optative_3sg": "ειη", "optative_2du": "ειτον",
    "optative_3du": "ειτην", "optative_1pl": "ειμεν",
    "optative_2pl": "ειτε", "optative_3pl": "ειεν",
    "imperative_2sg": "ηθι", "imperative_3sg": "ητω",
    "imperative_2du": "ητον", "imperative_3du": "ητων",
    "imperative_2pl": "ητε", "imperative_3pl": "εντων",
    "infinitive": "ηναι",
    "participle_nom_m_sg": "εις", "participle_voc_m_sg": "εν",
    "participle_nom_f_sg": "εισα", "participle_voc_f_sg": "εισα",
    "participle_nom_n_sg": "εν", "participle_acc_n_sg": "εν",
    "participle_voc_n_sg": "εν",
}


def is_stemless_passive_aorist(form, key):
    """Whether one passive aorist cell contains only its inflectional ending."""
    if not key.startswith("passive_aorist_"):
        return False
    ending = _STEMLESS_PASSIVE_AORIST.get(key[len("passive_aorist_"):])
    if ending is None:
        return False
    letters = "".join(c for c in unicodedata.normalize("NFD", form)
                      if unicodedata.category(c) != "Mn").lower()
    if key in ("passive_aorist_indicative_3sg",
               "passive_aorist_subjunctive_3pl"):
        return letters in (ending, ending + "ν")
    return letters == ending


def stemless_passive_aorist_dialects(entry):
    """Dialects whose sole aorist template explicitly lacks both passive stems.

    A complete competing template makes the table-header attribution
    ambiguous. Unsupported template types or dialect labels do not prove
    a missing stem.
    """
    witnesses = {}
    for template in entry.get("inflection_templates") or []:
        args = template.get("args") or {}
        if template.get("name") != "grc-conj" or args.get("1") != "aor-1":
            continue
        dialect = _DIALECT_NAMES.get(args.get("dial", ""))
        if dialect is None:
            continue
        missing = (args.get("form") == "ap" and args.get("2") and args.get("3")
                   and not args.get("4") and not args.get("5"))
        witnesses.setdefault(dialect, []).append(bool(missing))
    return frozenset(dialect for dialect, claims in witnesses.items() if all(claims))


def retain_complete_passive_aorist_readings(form, tags, *, stemless_table=False):
    """Remove only a uniquely identified stemless passive reading.

    Return None when no alternative voice remains. Missing or conflicting
    grammatical axes are retained without guessing their combinations.
    A positively identified stemless table also proves that its bare
    gender-only participle header forms have no passive stem.
    """
    claims = set(tags)
    if "passive" not in claims:
        return tags
    tenses = claims & {"present", "imperfect", "future", "aorist", "perfect",
                       "pluperfect", "future-perfect"}
    moods = claims & {"indicative", "subjunctive", "optative", "imperative",
                      "infinitive", "participle"}
    if tenses != {"aorist"} or len(moods) != 1:
        return tags
    mood, = moods
    key = "passive_aorist_" + mood
    if mood == "participle":
        genders = claims & {"masculine", "feminine", "neuter"}
        cases = claims & {"nominative", "genitive", "dative", "accusative", "vocative"}
        numbers = claims & {"singular", "dual", "plural"}
        if len(genders) != 1:
            return tags
        if stemless_table and not cases and not numbers:
            cases, numbers = {"nominative"}, {"singular"}
        if len(cases) != 1 or len(numbers) != 1:
            return tags
        gender, = genders
        case, = cases
        number, = numbers
        key += ("_" + {"nominative": "nom", "genitive": "gen", "dative": "dat",
                       "accusative": "acc", "vocative": "voc"}[case]
                + "_" + {"masculine": "m", "feminine": "f", "neuter": "n"}[gender]
                + "_" + {"singular": "sg", "dual": "du", "plural": "pl"}[number])
    elif mood != "infinitive":
        persons = claims & {"first-person", "second-person", "third-person"}
        numbers = claims & {"singular", "dual", "plural"}
        if len(persons) != 1 or len(numbers) != 1:
            return tags
        person, = persons
        number, = numbers
        key += ("_" + {"first-person": "1", "second-person": "2", "third-person": "3"}[person]
                + {"singular": "sg", "dual": "du", "plural": "pl"}[number])
    if not is_stemless_passive_aorist(form, key):
        return tags
    if claims & {"active", "middle", "mediopassive"}:
        return [tag for tag in tags if tag != "passive"]
    return None

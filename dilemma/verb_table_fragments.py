"""Identify complete grammatical readings of stemless Ancient verb tables."""

import unicodedata


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
    dialect_names = {"": "", "att": "Attic", "epi": "Epic", "ion": "Ionic",
                     "koi": "Koine", "dor": "Doric", "aeo": "Aeolic"}
    witnesses = {}
    for template in entry.get("inflection_templates") or []:
        args = template.get("args") or {}
        if template.get("name") != "grc-conj" or args.get("1") != "aor-1":
            continue
        dialect = dialect_names.get(args.get("dial", ""))
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

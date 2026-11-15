"""Crasis resolution for Ancient Greek.

Crasis is the contraction of two words into one, typically an article
or καί + the following word. For example:
  τοὔνομα = τό + ὄνομα  -> lemma: ὄνομα
  κἀγώ = καί + ἐγώ      -> lemma: κἀγώ (has its own entry)
  τἀνδρός = τοῦ + ἀνδρός -> lemma: ἀνήρ

Rules:
  1. If the crasis form has its own Wiktionary entry with no clear
     main word, return the crasis form itself as the lemma.
  2. For τ- crasis (article + noun), return the lemma of the noun.
  3. For κ- crasis (καί + X), return the lemma of X if identifiable.
  4. For forms not in the table, return None (let caller handle it).
"""

# Hand-curated crasis table: form -> lemma of the main content word.
# Sources: EN Wiktionary AG entries with "crasis" or "contraction of"
# in the gloss, plus standard Greek grammar references.
CRASIS_TABLE = {
    # τ- crasis: article + noun/adjective -> noun lemma
    "τοὔνομα": "ὄνομα",
    "τοὔνεκα": "τοὔνεκα",       # adverb, has own entry ("for that reason")
    "τοὐναντίον": "ἐναντίος",
    "τοὐμόν": "ἐμός",
    "τοὐμπαλιν": "ἔμπαλιν",
    "τἀργύριον": "ἀργύριον",
    "τἀδικήματα": "ἀδίκημα",
    "τἀναγκαῖα": "ἀναγκαῖος",
    "τἀνδρός": "ἀνήρ",
    "τἀνδρί": "ἀνήρ",
    "τἆλλα": "ἄλλος",
    "τἀκεῖ": "ἐκεῖ",
    "τὠνόματα": "ὄνομα",
    "τὠυτό": "αὐτός",
    "ταὐτός": "αὐτός",
    "ταὐτά": "αὐτός",
    "θάτερα": "ἕτερος",

    # κ- crasis: καί + X -> lemma of X or self if established word
    "κἀγώ": "ἐγώ",
    "κἀγαθός": "ἀγαθός",
    "κἀκ": "ἐκ",
    "κἀν": "ἐν",
    "κἄν": "ἄν",
    "κἀκείνων": "ἐκεῖνος",
    "κἀκεῖθεν": "ἐκεῖθεν",
    "κἀνταῦθα": "ἐνταῦθα",
    "κεἰς": "εἰς",
    "κᾷτα": "εἶτα",
    "κᾆτα": "εἶτα",
    "κἆτα": "εἶτα",
    "καὐτός": "αὐτός",

    # ὁ/ἡ + X crasis
    "αὑτή": "αὐτός",
    "αὑτοῦ": "αὐτός",
    "οὑκ": "ἐκ",
    "ἁνήρ": "ἀνήρ",
    "ὡνήρ": "ἀνήρ",
    "ὦνερ": "ἀνήρ",
    "ἅνδρες": "ἀνήρ",
    "ἇνδρες": "ἀνήρ",
    "ὧνδρες": "ἀνήρ",
    "ὦνδρες": "ἀνήρ",
    "ὤριστος": "ἄριστος",

    # μέντοι + ἄν
    "μεντἄν": "ἄν",
}


def resolve_crasis(form: str) -> str | None:
    """Resolve a crasis form to the lemma of its main content word.

    Returns the lemma if known, or None if the form is not recognized
    as crasis.
    """
    return CRASIS_TABLE.get(form)

"""Dilemma: diachronic Greek lemmatizer, POS tagger, and dependency parser.

    from dilemma import Dilemma

    d = Dilemma()
    d.lemmatize("ἔφατ̓")  # -> "φημί"

    from dilemma import Tagger  # POS tagger + dependency parser

    t = Tagger(lang="grc")
    t.tag(["μῆνιν ἄειδε θεά"])

See README.md at the repo root for full docs.
"""


from .core import (
    Dilemma,
    LemmaCandidate,
    LookupDB,
    to_monotonic,
    grave_to_acute,
    strip_accents,
)
from ._download import download as download_data
from .morph_diff import (
    diff_form,
    diff_paradigm,
    MorphDiff,
    Role,
)
from .morph_reconcile import (
    reconcile_token,
    reconcile_work,
    vote_readings,
    Reading,
    Provenance,
    ReconciledToken,
)
from .paradigm import (
    generate,
    generate_paradigm,
    iter_slots,
    ParadigmForm,
    ParadigmSlot,
    ParadigmSource,
)


def __getattr__(name):
    """Lazy re-export of `Tagger` from dilemma.tagger.

    Loaded on demand because importing dilemma.tagger pulls in torch and
    transformers; users who only want the lemmatizer should not pay that cost.
    """
    if name == "Tagger":
        from .tagger import Tagger
        return Tagger
    raise AttributeError(f"module 'dilemma' has no attribute {name!r}")


__all__ = [
    "Dilemma",
    "LemmaCandidate",
    "LookupDB",
    "to_monotonic",
    "grave_to_acute",
    "strip_accents",
    "download_data",
    "diff_form",
    "diff_paradigm",
    "MorphDiff",
    "Role",
    "reconcile_token",
    "reconcile_work",
    "vote_readings",
    "Reading",
    "Provenance",
    "ReconciledToken",
    "Tagger",
    "generate",
    "generate_paradigm",
    "iter_slots",
    "ParadigmForm",
    "ParadigmSlot",
    "ParadigmSource",
]

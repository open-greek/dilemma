"""Principal-part stems retain the lexical marks required by Lua."""
import importlib.util
from pathlib import Path
import sys
import unicodedata

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))
import expand_lsj
from lsj_principal_parts import _strip_augment, derive_grc_conj_args


@pytest.mark.parametrize("stem,expected", [
    ("ἐλυσ", "λυσ"), ("ἠρπασ", "ἀρπασ"), ("ὡρισ", "ὁρισ"),
    ("ἠυξ", "ἀυξ"), ("ἀϊσσ", "ἀϊσσ"), ("προσελυσ", "προσελυσ"),
])
def test_existing_augment_policy_retains_marks_on_surviving_letters(stem, expected):
    assert _strip_augment(stem) == expected
    assert _strip_augment(unicodedata.normalize("NFD", stem)) == expected


def test_all_principal_part_stems_preserve_lexical_marks_and_normalize_unicode():
    parts = {"fut": "ἀκούσω", "aor1": "ἤκουσα", "pf": "ἀκήκοα", "pf_mp": "ἤκουσμαι"}
    args = derive_grc_conj_args(parts, "ἀκούω")
    assert args["fut"] == ["fut", "ἀκουσ"]
    assert args["aor-1"] == ["aor-1", "ἠκουσ", "ἀκουσ"]
    assert args["perf"] == ["perf", "ἀκηκο", "ἠκουσ"]
    decomposed = {key: unicodedata.normalize("NFD", value) for key, value in parts.items()}
    assert derive_grc_conj_args(decomposed, "ἀκούω") == args
    for word, stem in [("ἀΐσω", "ἀϊσ"), ("ζῳώσω", "ζῳωσ"), ("κρῑ́νω", "κρῑν"), ("δῐώξω", "δῐωξ")]:
        assert derive_grc_conj_args({"fut": word}, "unused")["fut"] == ["fut", stem]


@pytest.mark.skipif(
    not expand_lsj.WTP_DB.exists() or importlib.util.find_spec("wikitextprocessor") is None,
    reason="requires the optional pinned Wiktionary Lua cache and runtime",
)
@pytest.mark.parametrize("word,parts", [
    ("λύω", {"fut": "λύσω", "aor1": "ἔλυσα", "pf": "λέλυκα", "pf_mp": "λέλυμαι"}),
    ("ἀκούω", {"fut": "ἀκούσω", "aor1": "ἤκουσα", "pf": "ἀκήκοα", "pf_mp": "ἤκουσμαι", "aor_p": "ἠκούσθην"}),
    ("ἁρπάζω", {"fut": "ἁρπάσω", "aor1": "ἥρπασα", "pf": "ἥρπακα"}),
    ("ὁρίζω", {"fut": "ὁρίσω", "aor1": "ὥρισα", "pf": "ὥρικα"}),
])
def test_real_lua_recovers_supplied_marked_parts(word, parts):
    expand_lsj._PP_CACHE.clear()
    wtp = expand_lsj.get_wtp()
    forms = set()
    for args in derive_grc_conj_args(parts, word).values():
        forms.update(expand_lsj._expand_principal_part(wtp, word, args))
    assert set(parts.values()) <= forms

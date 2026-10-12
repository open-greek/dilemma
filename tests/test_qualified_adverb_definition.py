"""An accent-description reference does not create an adverb head."""
from build_data import is_independent_lexical_entry
from build.build_grc_lexical_heads import has_definition


def test_paroxytone_adverb_reference_preserves_separate_lexical_senses():
    entry = {"word": "μέτα", "pos": "adv", "head_templates": [{"name": "grc-adv"}],
             "senses": [{"glosses": ["paroxytone form of μετᾰ́"]}]}
    assert not is_independent_lexical_entry(entry, "en")
    assert not has_definition(entry, "en")
    entry["senses"].append({"glosses": ["among or afterward"]})
    assert is_independent_lexical_entry(entry, "en")
    assert has_definition(entry, "en")

import json

from build.build_grc_contracted_aliases import (
    SOURCES, build_aliases, contracted_verb_targets, pin,
)


def test_explicit_references_do_not_infer_a_target_from_an_ending():
    for gloss, target in [
        ("συνηρημένη μορφή του καταμετρέω", "καταμετρέω"),
        ("contracted form of κῠνηγέω (kunēgéō)", "κυνηγέω"),
    ]:
        entry = {"pos": "verb", "senses": [{"glosses": [gloss]}]}
        assert contracted_verb_targets(entry) == {target}
        assert contracted_verb_targets({**entry, "pos": "noun"}) == set()
    assert contracted_verb_targets({"word": "ποιῶ", "pos": "verb",
                                    "senses": [{"glosses": ["to make"]}]}) == set()
    assert contracted_verb_targets({"pos": "verb", "senses": [{
        "glosses": ["third-person singular of τιμάω"],
        "form_of": [{"word": "τιμάω"}]}]}) == set()


def test_builder_pins_lexical_evidence_and_preserves_tied_references(tmp_path):
    inputs = {}
    for source, relative in SOURCES.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"word": "τιμῶ", "lang_code": "grc", "pos": "verb",
                 "senses": [{"glosses": ["συνηρημένη μορφή του " + target]}]}
                for target in ("τιμάω", "τιμέω", "ἀνύπαρκτος")]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        inputs[source] = pin(path)
    lexical = tmp_path / "heads.json"
    lexical.write_text(json.dumps({"heads": {"τιμάω": ["one"], "τιμέω": ["two"]},
                                   "inputs": inputs}))
    first = build_aliases(tmp_path, lexical)
    assert first == build_aliases(tmp_path, lexical)
    assert first["aliases"] == {"τιμῶ": [
        {"lemma": "τιμάω", "sources": sorted(SOURCES)},
        {"lemma": "τιμέω", "sources": sorted(SOURCES)}]}
    assert first["lexical_heads"] == pin(lexical)
    assert str(tmp_path) not in json.dumps(first)

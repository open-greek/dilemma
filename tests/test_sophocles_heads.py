"""Head rules of the Sophocles expansion (build/expand_sophocles.py)."""
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
import expand_sophocles as S  # noqa: E402


def write_chunk(directory, *entries, name="0750.xml"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        "<root>" + "".join(f"<entry>{entry}</entry>" for entry in entries) + "</root>",
        encoding="utf-8")


def parse(tmp_path, monkeypatch, *entries, **kwargs):
    write_chunk(tmp_path / "chunks", *entries)
    monkeypatch.setattr(S, "SOPH_DIR", tmp_path / "chunks")
    audit = S.new_head_audit()
    return S.parse_sophocles_entries(None, audit=audit, **kwargs), audit


def fixture_inputs(tmp_path, monkeypatch, lookup, entries, heads):
    """Point the generator at fixture chunks, a fixture table and, with
    ``heads`` (head -> sources), a fixture grc_lexical_heads.json."""
    write_chunk(tmp_path / "chunks", *entries)
    table = tmp_path / "ag_lookup.json"
    table.write_text(json.dumps(lookup, ensure_ascii=False), encoding="utf-8")
    lexical = tmp_path / "grc_lexical_heads.json"
    if heads is not None:
        lexical.write_text(json.dumps({"heads": heads}, ensure_ascii=False),
                           encoding="utf-8")
    monkeypatch.setattr(S, "SOPH_DIR", tmp_path / "chunks")
    monkeypatch.setattr(S, "AG_LOOKUP", table)
    monkeypatch.setattr(S, "LEXICAL_HEADS", lexical)
    monkeypatch.setattr(S, "load_breathing_review", lambda *a, **kw: None)
    monkeypatch.setattr(S, "sophocles_noun_input_hashes", lambda: {})
    monkeypatch.setattr(S, "get_wtp", lambda: None)
    monkeypatch.setattr(S, "load_wiktionary_genitives", lambda: {})
    return table


def run_noun_step(tmp_path, monkeypatch, lookup, *entries, forms=None, heads=None):
    """expand_all() over fixture entries with the Lua expansion stubbed:
    ``forms`` maps a head to its paradigm (default: the head alone)."""
    table = fixture_inputs(tmp_path, monkeypatch, lookup, entries, heads)
    calls = []

    def expand_noun(wtp, head, gender, genitive, wikt_genitives=None):
        calls.append((head, gender, genitive))
        _, error = S.resolve_noun_genitive(head, gender, genitive, wikt_genitives)
        return (set(), error) if error else (set((forms or {}).get(head, [head])), "")

    monkeypatch.setattr(S, "expand_noun", expand_noun)
    S.expand_all()
    audit = json.loads((tmp_path / "ag_sophocles_noun_generation_audit.json")
                       .read_text(encoding="utf-8"))
    return json.loads(table.read_text(encoding="utf-8")), audit, calls


def run_verb_step(tmp_path, monkeypatch, lookup, *entries, forms=None, heads=None):
    """expand_verbs() over fixture entries with the Lua expansion stubbed."""
    table = fixture_inputs(tmp_path, monkeypatch, lookup, entries, heads)
    calls = []

    def expand_verb(wtp, head):
        calls.append(head)
        return set((forms or {}).get(head, [head])), ""

    monkeypatch.setattr(S, "expand_verb", expand_verb)
    S.expand_verbs()
    return json.loads(table.read_text(encoding="utf-8")), calls


@pytest.mark.parametrize("orth,head,reason", [
    # another Greek word follows the first one: a phrase, or a split word
    ("Νέα Ῥώμη", "Νέα", "multi-word"),
    ("Κοίλη Συρία", "Κοίλη", "multi-word"),
    ("ἀνθ᾽ ὧν", "ἀνθ᾽", "multi-word"),
    ("κατα- φθάνω", "κατα", "multi-word"),
    ("προ νύσσω", "προ", "multi-word"),
    ("όσπερ ον", "όσπερ", "multi-word"),
    ("ἐπιχοριαμβικὸν μέτρον", "ἐπιχοριαμβικὸν", "multi-word"),
    ("Ἀππία ὁδός", "Ἀππία", "multi-word"),
    ("μή τις", "μή", "multi-word"),
    ("ολεθρεία ου ολεθρία", "ολεθρεία", "multi-word"),
    # the Greek words after the first one are entry text (one_word_head)
    ("κομίτιον ου", "κομίτιον", ""),
    ("γληχωνίτης οἶνος", "γληχωνίτης", ""),
    ("ἑτερολέκτως ἔχειν", "ἑτερολέκτως", ""),
    # entry text follows the head: it stays the head
    ("βιοθανέω ---- βιαιοθανατέω", "βιοθανέω", ""),
    ("φυσιάω — φυσιλοω", "φυσιάω", ""),
    ("ἑψόω — ἔψω. Abuc. 1553 B Ἑψοῦται ἡ τροφή", "ἑψόω", ""),
    ("ἐπιστάνω = ἐφίστημι", "ἐπιστάνω", ""),
    ("γραμματοφυλάκιον--- γραμματοφυλακεῖον", "γραμματοφυλάκιον", ""),
    ("γιγαντιάω (γίγας)", "γιγαντιάω", ""),
    ("νοοειδής. ές", "νοοειδής", ""),
    ("ἰσάγγελος; ov", "ἰσάγγελος", ""),
    ("ἐπισύνθετος oν", "ἐπισύνθετος", ""),
    ("Ἱριάδιτζα or Τριαδίτζα", "Ἱριάδιτζα", ""),
    ("μεσσοπαγής for µεσοπαγής", "μεσσοπαγής", ""),
    ("ὕδατι. Ps. 41", "ὕδατι", ""),
    ("δεκαεπτά (δέκα, ἑπτά) — ἑπτακαίδεκα", "δεκαεπτά", ""),
    ("ἀθεότης", "ἀθεότης", ""),
    # no Greek word at all
    ("Plut. TL", "", ""),
    ("also Σατορνῖλος", "", ""),
])
def test_read_orth_tells_a_head_from_a_phrase(orth, head, reason):
    got, why, rest = S.read_orth(orth)
    assert (got, why) == (head, reason)
    assert S.greek_mu(orth).strip().endswith(rest)
    assert S.clean_headword(orth) == ("" if reason else head)


def test_first_word_of_a_phrase_is_not_an_entry(tmp_path, monkeypatch):
    entries, audit = parse(
        tmp_path, monkeypatch,
        "<orth>Νέα Ῥώμη</orth>, ἡ, New Rome = Κωνσταντινούπολις.",
        "<orth>κατα- φθάνω</orth>, to overtake.",
        "<orth>ἕως οὗ</orth>, — ἕως.",
        "<orth>ἕως</orth>, until.",
        "<orth>βιοθανέω ---- βιαιοθανατέω</orth>, Jejun. 1924 B.",
        "<orth>Ἱριάδιτζα or Τριαδίτζα</orth>, ης, ἡ, Triaditza, a place.")
    assert set(entries) == {"ἕως", "βιοθανέω", "Ἱριάδιτζα"}
    assert audit["multi_word_heads"] == {
        "Νέα Ῥώμη": "Νέα", "κατα- φθάνω": "κατα", "ἕως οὗ": "ἕως"}
    # the single-word entry is read from its own text, not the phrase's
    assert entries["ἕως"] is not audit["printed"]["ἕως"]
    assert entries["Ἱριάδιτζα"]["gender"] == "f"
    assert set(audit["printed"]) == {"Νέα", "κατα", "ἕως", "βιοθανέω", "Ἱριάδιτζα"}


def test_noun_step_skips_and_records_a_phrase(tmp_path, monkeypatch):
    lookup, audit, calls = run_noun_step(
        tmp_path, monkeypatch, {"νέος": "νέος"},
        "<orth>Νέα Ῥώμη</orth>, ἡ, New Rome.",
        "<orth>Νεκρὰ Θάλασσα</orth>, ἡ, the Dead Sea.",
        "<orth>ἄβαγνον</orth>, ου, τό, a kind of rose.",
        forms={"ἄβαγνον": ["ἄβαγνον", "ἀβάγνου"]})
    assert [head for head, _, _ in calls] == ["ἄβαγνον"]
    assert lookup == {"νέος": "νέος", "ἄβαγνον": "ἄβαγνον", "αβαγνον": "ἄβαγνον",
                      "ἀβάγνου": "ἄβαγνον", "αβαγνου": "ἄβαγνον"}
    assert audit["schema_version"] == 1
    assert audit["head_rules"]["multi_word_heads"] == {
        "count": 2, "heads": {"Νέα Ῥώμη": "Νέα", "Νεκρὰ Θάλασσα": "Νεκρὰ"}}


@pytest.mark.parametrize("first,phrase,joined", [
    ("κατα", "κατα- φθάνω", "καταφθάνω"),
    ("κατα", "κατα λιπαίνω", "καταλιπαίνω"),
    ("κοινό", "κοινό δημος", "κοινόδημος"),
    ("ὀγδοηκοντα", "ὀγδοηκοντα- πέντε", "ὀγδοηκονταπέντε"),
    ("μή", "μή τις", "μήτις"),
    # two accents, a grave, an elision mark, three words: no single word
    ("Νέα", "Νέα Ῥώμη", ""),
    ("μέτα", "μέτα ποίησις", ""),
    ("πὴ", "πὴ μὲν", ""),
    ("ἀνθ᾽", "ἀνθ᾽ ὧν", ""),
    ("οπου", "οπου δή ποτε", ""),
])
def test_split_word_joins_the_two_parts_of_a_phrase(first, phrase, joined):
    assert S.split_word(first, phrase)[0] == joined


def test_a_word_the_ocr_split_is_an_entry_under_the_joined_word(tmp_path, monkeypatch):
    sources = ["lsj9_headwords"]
    lookup, audit, calls = run_noun_step(
        tmp_path, monkeypatch, {"καταφθάνω": "καταφθάνω"},
        "<orth>κηνσο φύλαξ</orth>, ακος, ὁ, keeper of the census.",
        "<orth>μοιχο σύνοδος</orth>, ου, ἡ, adulterous council.",
        "<orth>κατα- φθάνω</orth>, to overtake.",
        "<orth>μή τις</orth>, lest any one.",
        "<orth>τετρα- πλατεῖα</orth>, ας, ἡ, a square.",
        "<orth>Νέα Ῥώμη</orth>, ἡ, New Rome.",
        # κατα is a head of the other lexica as well: a combining form
        heads={"κηνσοφύλαξ": sources, "μοιχοσύνοδος": sources,
               "καταφθάνω": sources, "κατα": sources, "φθάνω": sources,
               "μήτις": sources, "μή": sources, "τις": sources},
        forms={"κηνσοφύλαξ": ["κηνσοφύλαξ", "κηνσοφύλακος"]})
    # the joined nouns are expanded with the entry's genitive and gender
    assert calls == [("κηνσοφύλαξ", "m", "ακος"), ("μοιχοσύνοδος", "f", "ου")]
    assert lookup["κηνσοφύλακος"] == "κηνσοφύλαξ"
    assert audit["head_rules"]["repaired_heads"] == {"count": 3, "heads": {
        "κηνσο φύλαξ": "κηνσοφύλαξ", "μοιχο σύνοδος": "μοιχοσύνοδος",
        "κατα- φθάνω": "καταφθάνω"}}
    # no other source lists the joined word, or both parts are words: phrases
    assert audit["head_rules"]["multi_word_heads"] == {"count": 3, "heads": {
        "μή τις": "μή", "τετρα- πλατεῖα": "τετρα", "Νέα Ῥώμη": "Νέα"}}
    assert "split_words" not in audit["head_rules"]


def test_a_proclitic_before_a_word_stays_a_phrase():
    listed = frozenset({"οὐμήν", "οὐ", "μήν", "ἐκδέχομαι", "ἐκ", "δέχομαι",
                        "προνύσσω", "προ", "νύσσω"})
    audit = S.new_head_audit()
    entries = {}
    for first, phrase in (("οὐ", "οὐ μήν"), ("ἐκ", "ἐκ δέχομαι"), ("προ", "προ νύσσω")):
        entry = {"headword": first, "gender": ""}
        audit["multi_word_heads"][phrase] = first
        joined, second = S.split_word(first, phrase)
        audit["split_words"][phrase] = (joined, second, entry)
    settled = S.settle_heads(entries, listed, audit)
    # οὐ and ἐκ are words without an accent; προ is a combining form
    assert list(settled) == ["προνύσσω"]
    assert settled["προνύσσω"] == {"headword": "προνύσσω", "printed": "προ νύσσω",
                                   "gender": ""}
    assert audit["multi_word_heads"] == {"οὐ μήν": "οὐ", "ἐκ δέχομαι": "ἐκ"}
    assert audit["repaired_heads"] == {"προ νύσσω": "προνύσσω"}


def test_verb_step_expands_a_split_verb(tmp_path, monkeypatch):
    lookup, calls = run_verb_step(
        tmp_path, monkeypatch, {},
        "<orth>συμ- ψάλλω</orth>, to sing with.",
        "<orth>προ νύσσω</orth>, to prick before.",
        heads={"συμψάλλω": ["lsj9_headwords"], "συμ": ["lsj9_headwords"],
               "ψάλλω": ["lsj9_headwords"]})
    assert calls == ["συμψάλλω"]
    assert set(lookup.values()) == {"συμψάλλω"}


def test_article_after_the_head_states_the_gender(tmp_path, monkeypatch):
    entries, _ = parse(
        tmp_path, monkeypatch,
        "<orth>καισάριης</orth>, ἡ, the Latin caesaries --- κόμη. Lyd. 95, 11.",
        "<orth>ἀτρεπτότης</orth>, ητος; ἡ, (ἄτρεπτος) immutability.",
        "<orth>καθολικότης</orth>, yros, ἡ, the office of καθολικός.",
        "<orth>ἀντίστροφος</orth>, oυ, ἡ, antistrophe, in versification.",
        "<orth>τένων</orth>, οντος, ὁ. A sinew.",
        # the article belongs to the definition: a guess, not a statement
        "<orth>αἰσχροπάθεια</orth>, ας, η, (αἰσχροπαθής) — τὸ αἰσχρῶς πάσχειν.",
        "<orth>διαγογγυσμός</orth>, οὔ, ὃ, --- τὸ διαγογγύζειν. Ex, 16, 8.",
        "<orth>θάτερον</orth>, = τὸ ἅτερον (ἕτερον).",
        # the feminine ending of an adjective, read as the article by the OCR
        "<orth>ἀρθρικός</orth>, ἡ. ὄν, (ἄρθρον) of the article, in grammar.",
        "<orth>πρωϊνός</orth> ἡ, όν, L. matutinus.")
    stated = {head: (entry["gender"], entry["gender_stated"])
              for head, entry in entries.items()}
    assert stated == {
        "καισάριης": ("f", True), "ἀτρεπτότης": ("f", True),
        "καθολικότης": ("f", True), "ἀντίστροφος": ("f", True),
        "τένων": ("m", True),
        "αἰσχροπάθεια": ("n", False), "διαγογγυσμός": ("n", False),
        "θάτερον": ("n", False),
        "ἀρθρικός": ("f", False), "πρωϊνός": ("f", False)}


def test_ending_class_keeps_to_the_stated_gender():
    assert S.infer_from_ending("καισάριης") == ("m", "καισάριοῦ")
    assert S.infer_from_ending("καισάριης", "f") == ("", "")
    assert S.infer_from_ending("ἀντίστροφος") == ("m", "ἀντίστροφου")
    assert S.infer_from_ending("ἀντίστροφος", "f") == ("f", "ἀντίστροφου")
    assert S.infer_from_ending("βροῦμα", "f") == ("", "")
    assert S.infer_from_ending("κινητήριον", "m") == ("", "")
    assert S.infer_from_ending("αἰσχροπάθεια") == ("f", "αἰσχροπάθείας")


def test_stated_gender_wins_over_the_ending(tmp_path, monkeypatch):
    lookup, audit, calls = run_noun_step(
        tmp_path, monkeypatch, {},
        "<orth>καισάριης</orth>, ἡ, the Latin caesaries --- κόμη.",
        "<orth>φάκιης</orth>, ἡ, facies = ὄψις. Lyd. 145.",
        "<orth>καλλίπενθος</orth>, eos, τὸ, good πένθος for sin.",
        "<orth>ἀντίστροφος</orth>, oυ, ἡ, antistrophe, in versification.",
        "<orth>βροῦμα</orth>, ἡ, the Latin bruma.",
        "<orth>διαγογγυσμός</orth>, οὔ, ὃ, --- τὸ διαγογγύζειν.",
        "<orth>ἀρθρικός</orth>, ἡ. ὄν, (ἄρθρον) of the article.")
    assert calls == [
        ("καισάριης", "f", ""), ("φάκιης", "f", ""), ("καλλίπενθος", "n", ""),
        ("ἀντίστροφος", "f", "ἀντίστροφου"), ("βροῦμα", "f", ""),
        # no stated gender: the ending decides, as before
        ("διαγογγυσμός", "m", "διαγογγυσμου"), ("ἀρθρικός", "m", "ἀρθρικου")]
    # no class fits the stated gender and the ending: nothing is generated
    assert not {"καισάριης", "φάκιης", "καλλίπενθος"} & set(lookup.values())
    assert {"ἀντίστροφος", "βροῦμα", "διαγογγυσμός"} <= set(lookup.values())
    assert audit["failures"] == {"missing-genitive": 3}
    assert audit["head_rules"]["stated_gender"] == {"count": 5, "heads": {
        "καισάριης": {"stated": "f", "ending_class": "m", "result": "missing-genitive"},
        "φάκιης": {"stated": "f", "ending_class": "m", "result": "missing-genitive"},
        "καλλίπενθος": {"stated": "n", "ending_class": "m",
                        "result": "missing-genitive"},
        "ἀντίστροφος": {"stated": "f", "ending_class": "m", "result": "expanded"},
        "βροῦμα": {"stated": "f", "ending_class": "n", "result": "expanded"}}}


LISTED = frozenset({"καθηδύνω", "καταμαρτυρία", "ἡλιοκάμινος", "ὕποκατασκευή"})


@pytest.mark.parametrize("printed,head,rule", [
    ("Ἔρετριάς", "", "two-accents"),
    ("οἴηματικός", "", "two-accents"),
    # the joined word is no other source's head
    ("Σινα-ῖτις", "", "inner-hyphen"),
    ("αἱρέω-ω", "", "inner-hyphen"),
    ("στρέφω—τρέπω", "", "inner-hyphen"),
    # ... or carries two accents, listed or not
    ("ὕπο-κατασκευή", "", "inner-hyphen"),
    ("γαζοφυλακέω--γαζοφύλαξ", "", "inner-hyphen"),
    # another source lists the joined word
    ("καθ-ηδύνω", "καθηδύνω", "joined"),
    ("κατα-μαρτυρία", "καταμαρτυρία", "joined"),
    # one accent, no hyphen: the head stands
    ("ἀθεότης", "ἀθεότης", ""),
    ("πρωΐα", "πρωΐα", ""),
    ("ᾅδης", "ᾅδης", ""),
    ("οφφικιον", "οφφικιον", ""),
])
def test_settle_head_rejects_or_repairs_a_malformed_head(printed, head, rule):
    assert S.settle_head(printed, LISTED) == (head, rule)


def test_hyphen_repair_needs_another_source(tmp_path):
    assert S.settle_head("καθ-ηδύνω") == ("", "inner-hyphen")
    assert S.settle_head("καθ-ηδύνω", frozenset()) == ("", "inner-hyphen")
    assert S.load_lexical_heads(tmp_path / "missing.json") is None
    path = tmp_path / "heads.json"
    path.write_text(json.dumps({"heads": {
        "καθηδύνω": ["lsj9_headwords", "sophocles/0600.xml"],
        "Σιναῖτις": ["sophocles/0950.xml"],
        "πέτρα": ["pd_headwords"]}}, ensure_ascii=False), encoding="utf-8")
    # the lexicon's own heads are the ones being judged: they are no evidence
    assert S.load_lexical_heads(path) == {"καθηδύνω", "πέτρα"}


def test_settle_heads_keeps_the_order_and_the_printed_twin():
    entries = {"ἄβαγνον": {"headword": "ἄβαγνον", "gender": "n"},
               "κατα-μαρτυρία": {"headword": "κατα-μαρτυρία", "gender": "f"},
               "Ἔρετριάς": {"headword": "Ἔρετριάς", "gender": "f"},
               "καθ-ηδύνω": {"headword": "καθ-ηδύνω", "gender": ""},
               "καθηδύνω": {"headword": "καθηδύνω", "gender": "", "own": True},
               "ὕδωρ": {"headword": "ὕδωρ", "gender": "n"}}
    audit = S.new_head_audit()
    settled = S.settle_heads(entries, LISTED, audit)
    assert list(settled) == ["ἄβαγνον", "καταμαρτυρία", "καθηδύνω", "ὕδωρ"]
    assert settled["καταμαρτυρία"] == {
        "headword": "καταμαρτυρία", "printed": "κατα-μαρτυρία", "gender": "f"}
    assert settled["καθηδύνω"]["own"] and settled["ἄβαγνον"] is entries["ἄβαγνον"]
    assert entries["κατα-μαρτυρία"]["headword"] == "κατα-μαρτυρία"
    assert audit["malformed_heads"] == {"Ἔρετριάς": "two-accents"}
    assert audit["repaired_heads"] == {"κατα-μαρτυρία": "καταμαρτυρία",
                                       "καθ-ηδύνω": "καθηδύνω"}
    assert audit["lowercased_heads"] == {}
    # without the lexical heads nothing is repaired
    assert list(S.settle_heads(entries, None)) == ["ἄβαγνον", "καθηδύνω", "ὕδωρ"]


HEADS = {"καταμαρτυρία": ["lbg_headwords", "sophocles/0600.xml"],
         "ἡλιοκάμινος": ["lsj9_headwords"],
         "καθηδύνω": ["lsj9_headwords", "sophocles/0600.xml"],
         "Σιναῖτις": ["sophocles/0950.xml"]}


def test_noun_step_rejects_and_repairs_malformed_heads(tmp_path, monkeypatch):
    lookup, audit, calls = run_noun_step(
        tmp_path, monkeypatch, {"ἡλιοκάμινος": "ἡλιοκάμινος"},
        "<orth>Ἔρετρίάς</orth>, άδος, ἡ, of Eretria.",
        "<orth>Σινα-ῖτις</orth>, ιδος, ἡ, of Sinai.",
        "<orth>ὕπο-κατασκευή</orth>, ῆς, ἡ, preparation.",
        "<orth>κατα-μαρτυρία</orth>, ας, ἡ, testimony against one.",
        "<orth>ἡλιο-κάμινος</orth>, ου, ἡ, a place exposed to the sun.",
        "<orth>ἀνα-πλήρως</orth>, adv. fully.",
        heads=HEADS,
        forms={"καταμαρτυρία": ["καταμαρτυρία", "καταμαρτυρίας"]})
    # the repaired head is expanded; the one the table has is left alone
    assert calls == [("καταμαρτυρία", "f", "ας")]
    assert lookup == {
        "ἡλιοκάμινος": "ἡλιοκάμινος",
        "καταμαρτυρία": "καταμαρτυρία", "καταμαρτυρια": "καταμαρτυρία",
        "καταμαρτυρίας": "καταμαρτυρία", "καταμαρτυριας": "καταμαρτυρία"}
    assert audit["head_rules"]["malformed_heads"] == {"count": 4, "heads": {
        "Ἔρετρίάς": "two-accents", "Σινα-ῖτις": "inner-hyphen",
        "ἀνα-πλήρως": "inner-hyphen", "ὕπο-κατασκευή": "inner-hyphen"}}
    assert audit["head_rules"]["repaired_heads"] == {"count": 2, "heads": {
        "κατα-μαρτυρία": "καταμαρτυρία", "ἡλιο-κάμινος": "ἡλιοκάμινος"}}
    # the historical layer keyed its rows by the printed head, so a completed
    # genitive still supersedes that name, rejected and repaired heads alike;
    # the head with two accents is superseded as such
    assert set(audit["superseded_lemmas"]) == {
        "Σινα-ῖτις", "Ἔρετρίάς", "κατα-μαρτυρία", "ἡλιο-κάμινος"}
    assert set(audit["resolutions"]) == {
        "Σινα-ῖτις", "κατα-μαρτυρία", "ἡλιο-κάμινος"}


def test_verb_step_rejects_and_repairs_malformed_heads(tmp_path, monkeypatch):
    lookup, calls = run_verb_step(
        tmp_path, monkeypatch, {},
        "<orth>καθ-ηδύνω</orth>, to sweeten.",
        "<orth>δια-μαστιγόω</orth>, to scourge.",
        "<orth>θεμάτίζω</orth>, to propose.",
        "<orth>γράφω</orth>, to write.",
        heads=HEADS, forms={"καθηδύνω": ["καθηδύνω", "καθηδύνεις"]})
    assert calls == ["καθηδύνω", "γράφω"]
    assert set(lookup.values()) == {"καθηδύνω", "γράφω"}
    assert "καθηδύνεις" in lookup and "καθηδυνεις" in lookup


COMMON = frozenset({"ἀθεότης", "ἔκπτυξις", "ἱπποθυτέω", "πέτρα", "Πέτρα",
                    "θεοπρεπῶς", "ἑλένη", "Ἑλένη", "ἀγάπη", "Ἀγάπη"})


@pytest.mark.parametrize("printed,head,rule", [
    # only the lowercase spelling is another source's head: a common word
    ("Ἀθεότης", "ἀθεότης", "lowercased"),
    ("Ἔκπτυξις", "ἔκπτυξις", "lowercased"),
    ("Θεοπρεπῶς", "θεοπρεπῶς", "lowercased"),
    ("Ἱππο-θυτέω", "ἱπποθυτέω", "joined"),
    # the OCR prints the capital's breathing as a separate mark
    ("᾿Αθεότης", "ἀθεότης", "lowercased"),
    # another source lists the capitalized head: a name beside a common word
    ("Πέτρα", "Πέτρα", ""),
    ("Ἑλένη", "Ἑλένη", ""),
    ("᾿Αγάπη", "᾿Αγάπη", ""),
    # no lowercase twin
    ("Θεοδοτάκης", "Θεοδοτάκης", ""),
    ("Σιναῖτις", "Σιναῖτις", ""),
    ("ἀθεότης", "ἀθεότης", ""),
])
def test_settle_head_lowercases_a_capitalized_common_word(printed, head, rule):
    assert S.settle_head(printed, COMMON) == (head, rule)
    assert S.settle_head(printed, None)[0] in ("", printed)


def test_noun_step_generates_a_common_word_under_its_lowercase_head(
        tmp_path, monkeypatch):
    sources = ["lsj9_headwords"]
    lookup, audit, calls = run_noun_step(
        tmp_path, monkeypatch, {"ἔκπτυξις": "ἔκπτυξις", "εκπτυξις": "ἔκπτυξις"},
        "<orth>᾿Αθεότης</orth>, ητος, ἡ, ungodliness.",
        "<orth>Ἔκπτυξις</orth>, εως, ἡ, an unfolding.",
        "<orth>Πέτρα</orth>, ας, ἡ, Petra, a city.",
        "<orth>Θεοδοτάκης</orth>, ου, ὁ, dear Θεόδοτος.",
        "<orth>Θεοπρεπῶς</orth>, adv. in a manner worthy of God.",
        heads={"ἀθεότης": sources, "Ἀθεότης": ["sophocles/0057.xml"],
               "ἔκπτυξις": sources, "πέτρα": sources, "Πέτρα": ["pd_headwords"],
               "θεοπρεπῶς": sources},
        forms={"ἀθεότης": ["ἀθεότης", "ἀθεότητος"]})
    # Ἔκπτυξις is skipped: the table already has the lowercase head
    assert [head for head, _, _ in calls] == ["ἀθεότης", "Πέτρα", "Θεοδοτάκης"]
    assert lookup["ἀθεότητος"] == lookup["αθεοτητος"] == "ἀθεότης"
    assert lookup["θεοπρεπῶς"] == lookup["θεοπρεπως"] == "θεοπρεπῶς"
    assert lookup["Πέτρα"] == "Πέτρα" and lookup["Θεοδοτάκης"] == "Θεοδοτάκης"
    assert set(lookup.values()) == {
        "ἔκπτυξις", "ἀθεότης", "Πέτρα", "Θεοδοτάκης", "θεοπρεπῶς"}
    assert audit["head_rules"]["lowercased_heads"] == {"count": 3, "heads": {
        "Ἔκπτυξις": "ἔκπτυξις", "Θεοπρεπῶς": "θεοπρεπῶς", "᾿Αθεότης": "ἀθεότης"}}


def test_verb_step_generates_a_common_word_under_its_lowercase_head(
        tmp_path, monkeypatch):
    lookup, calls = run_verb_step(
        tmp_path, monkeypatch, {},
        "<orth>Ἱπποθυτέω</orth>, to sacrifice horses.",
        "<orth>Ἰουδαΐζω</orth>, to judaize.",
        heads={"ἱπποθυτέω": ["lsj9_headwords"]})
    assert calls == ["ἱπποθυτέω", "Ἰουδαΐζω"]


def test_capitalized_plain_key_does_not_shadow_a_lowercase_reading():
    lookup = {"εστιας": "ἑστία", "πετρας": "Πέτρα", "νεα": "νέος"}
    withheld = {}
    assert not S.add_plain_key(lookup, "Εστιας", "Ἑστιάς", withheld)
    assert withheld == {"Εστιας": {"lemma": "Ἑστιάς", "lowercase_reading": "ἑστία"}}
    # the lowercase key reads the same lemma, or there is none: written
    assert S.add_plain_key(lookup, "Πετρας", "Πέτρα", withheld)
    assert S.add_plain_key(lookup, "Εστιαδος", "Ἑστιάς", withheld)
    # a lowercase key is not held back by anything, and no key is overwritten
    assert S.add_plain_key(lookup, "εστιαδος", "ἑστιάς", withheld)
    assert not S.add_plain_key(lookup, "νεα", "Νέα", withheld)
    assert lookup == {"εστιας": "ἑστία", "πετρας": "Πέτρα", "νεα": "νέος",
                      "Πετρας": "Πέτρα", "Εστιαδος": "Ἑστιάς", "εστιαδος": "ἑστιάς"}
    assert list(withheld) == ["Εστιας"]


def test_noun_step_withholds_shadowing_capitalized_keys(tmp_path, monkeypatch):
    lookup, audit, _ = run_noun_step(
        tmp_path, monkeypatch,
        {"ἑστίας": "ἑστία", "εστιας": "ἑστία", "αττικως": "ἀττικός"},
        "<orth>Ἑστιάς</orth>, άδος, ἡ, Vestal.",
        "<orth>Ἀττικῶς</orth>, adv. in the Attic dialect.",
        forms={"Ἑστιάς": ["Ἑστιάς", "Ἑστιάδος"]})
    assert lookup == {
        "ἑστίας": "ἑστία", "εστιας": "ἑστία", "αττικως": "ἀττικός",
        "Ἑστιάς": "Ἑστιάς", "Ἑστιάδος": "Ἑστιάς", "Εστιαδος": "Ἑστιάς",
        "Ἀττικῶς": "Ἀττικῶς"}
    assert audit["withheld_capitalized_keys"] == {"count": 2, "keys": {
        "Αττικως": {"lemma": "Ἀττικῶς", "lowercase_reading": "ἀττικός"},
        "Εστιας": {"lemma": "Ἑστιάς", "lowercase_reading": "ἑστία"}}}


def test_verb_step_withholds_shadowing_capitalized_keys(tmp_path, monkeypatch):
    lookup, calls = run_verb_step(
        tmp_path, monkeypatch,
        {"ἰουδαΐζεις": "ἰουδαΐζω", "ιουδαιζεις": "ἰουδαΐζω"},
        "<orth>Ἰουδαΐζω</orth>, to judaize.",
        forms={"Ἰουδαΐζω": ["Ἰουδαΐζω", "Ἰουδαΐζεις"]})
    assert calls == ["Ἰουδαΐζω"]
    assert lookup == {
        "ἰουδαΐζεις": "ἰουδαΐζω", "ιουδαιζεις": "ἰουδαΐζω",
        "Ἰουδαΐζω": "Ἰουδαΐζω", "Ιουδαιζω": "Ἰουδαΐζω", "Ἰουδαΐζεις": "Ἰουδαΐζω"}
    audit = json.loads((tmp_path / "ag_sophocles_verb_generation_audit.json")
                       .read_text(encoding="utf-8"))
    assert audit["schema_version"] == 1 and len(audit["generator_sha256"]) == 64
    assert audit["withheld_capitalized_keys"] == {"count": 1, "keys": {
        "Ιουδαιζεις": {"lemma": "Ἰουδαΐζω", "lowercase_reading": "ἰουδαΐζω"}}}


MICRO = "µ"


@pytest.mark.parametrize("orth,head", [
    # the micro sign is the letter mu
    (f"{MICRO}έρος", "μέρος"),
    (f"σχη{MICRO}ατοποιέω", "σχηματοποιέω"),
    (f"{MICRO}ετα-διαιτάω", "μετα-διαιτάω"),
    # the lexicon's asterisk and the OCR's stray marks are no letters
    ("*πιάζω", "πιάζω"),
    ("«ἀθεμιστία", "ἀθεμιστία"),
    ("καβάδιον»", "καβάδιον"),
    ("“προικιμαῖος", "προικιμαῖος"),
    ("|αἴρινος", "αἴρινος"),
    ("φυτη-κομέω>", "φυτη-κομέω"),
    # a koronis before the capital is its breathing
    ("᾽Αβραμιαῖος", "᾿Αβραμιαῖος"),
    # Latin letters after the word are entry text
    ("ἀββᾶςI", "ἀββᾶς"),
    ("συν-υποχωρέω.Plut. II", "συν-υποχωρέω"),
    # Latin letters inside it are misread Greek ones: no head
    ("Kούβρικος", ""),
    ("κόρoς", ""),
    ("Fαῦ", ""),
    ("xnμίον", ""),
])
def test_read_orth_settles_the_ocr_letters(orth, head):
    assert S.read_orth(orth)[:2] == (head, "")
    assert MICRO not in S.read_orth(f"{orth}, for {MICRO}έρος")[2]


def test_legacy_head_is_the_reading_of_the_earlier_tables():
    assert S.legacy_head("Kούβρικος") == "ούβρικος"
    assert S.legacy_head(f"{MICRO}έρος") == f"{MICRO}έρος"
    assert S.legacy_head("*πιάζω") == "*πιάζω"
    assert S.legacy_head("᾽Αντινώ") == "᾽Αντινώ"
    assert S.legacy_head("κομίτιον ου") == "κομίτιον"
    assert S.legacy_head("Plut. TL") == ""


@pytest.mark.parametrize("phrase,rule,ending", [
    ("κομίτιον ου", "run-on-ending", "ου"),
    ("ἐξοδεία ας", "run-on-ending", "ας"),
    ("ξηροφθαλμία ας", "run-on-ending", "ας"),
    ("φύσημα ατος", "run-on-ending", "ατος"),
    ("πολυ-άνθραξ ακος", "run-on-ending", "ακος"),
    ("ορθο-πρίων ονος", "run-on-ending", "ονος"),
    ("θεοηχής ές", "run-on-ending", "ές"),
    ("κελευστός ἡ", "run-on-ending", "ἡ"),
    ("ξηροφαγέω ἤσω", "run-on-ending", "ἤσω"),
    ("πραοπαθέω ήσω", "run-on-ending", "ήσω"),
    ("σχολιάζω άσω", "run-on-ending", "άσω"),
    ("γληχωνίτης οἶνος", "qualified-noun", ""),
    ("δωρίτης ἀγών", "qualified-noun", ""),
    ("ἀστερίτης λίθος", "qualified-noun", ""),
    ("ληκύθιον μέτρον", "qualified-noun", ""),
    ("μελαμ-ψίθιος οἶνος", "qualified-noun", ""),
    ("ἀπαλλακτικῶς ἔχειν", "adverb-idiom", ""),
    ("δυσπαύστως έχω", "adverb-idiom", ""),
    # phrases, names, split words and variants stay phrases
    ("Νέα Ῥώμη", "", ""),
    ("Κοίλη Συρία", "", ""),
    ("κατα λιπαίνω", "", ""),
    ("παρά ξηρος", "", ""),
    ("κοινό δημος", "", ""),
    ("ου μή", "", ""),
    ("νόος νοῦς", "", ""),
    ("ορίας άνεμος", "", ""),
    ("ἐπιχοριαμβικὸν μέτρον", "", ""),
    ("ἀεὶ βασιλεύς", "", ""),
    ("ἕως οὗ", "", ""),
    ("ρόσα ἡ rosa", "", ""),
    ("ολεθρεία ου ολεθρία", "", ""),
    ("ἀπίσσωτος ΟΥ ἀπίττωτος", "", ""),
    # a genitive that does not fit the head's ending is no ending
    ("φύσημα ου", "", ""),
    ("κομίτιον ατος", "", ""),
])
def test_one_word_head_reads_entry_text_after_the_head(phrase, rule, ending):
    head, *following = phrase.split()
    assert S.one_word_head(head, following) == (rule, ending)


def test_run_on_ending_is_the_entrys_first_field(tmp_path, monkeypatch):
    entries, audit = parse(
        tmp_path, monkeypatch,
        "<orth>κομίτιον ου</orth>, τὸ, the Latin comitium. Jos.",
        "<orth>πολυ-άνθραξ ακος</orth>, ὁ, ἡ, rich in coal. Schol.",
        "<orth>θεοηχής ές</orth>, (ἠχέω) instructed by God.",
        "<orth>κελευστός ἡ</orth>,όν, ordered, commanded. Lucian.",
        "<orth>σχολιάζω άσω</orth>, to write σχόλια, to comment on",
        "<orth>δωρίτης ἀγών</orth>, o, (δῶρον) a game in which the",
        "<orth>δυσπαύστως έχω</orth>, = δύσπαυστός εἶμι.Martyr.",
        # no breathing and no review of the first word: still a phrase
        "<orth>ορθο-πρίων ονος</orth>, ό, a surgical instrument for")
    assert set(entries) == {"κομίτιον", "πολυ-άνθραξ", "θεοηχής", "κελευστός",
                            "σχολιάζω", "δωρίτης", "δυσπαύστως"}
    assert (entries["κομίτιον"]["gender"], entries["κομίτιον"]["genitive"]) == ("n", "ου")
    assert (entries["πολυ-άνθραξ"]["gender"],
            entries["πολυ-άνθραξ"]["genitive"]) == ("m", "ακος")
    assert entries["θεοηχής"]["adjective_endings"] == "two"
    assert entries["κελευστός"]["adjective_endings"] == "three"
    assert not entries["σχολιάζω"]["is_adj"] and not entries["σχολιάζω"]["is_xref"]
    # the equals sign glosses the idiom, it is no reference of the adverb
    assert not entries["δυσπαύστως"]["is_xref"]
    assert audit["one_word_heads"] == {
        "κομίτιον ου": {"head": "κομίτιον", "rule": "run-on-ending"},
        "πολυ-άνθραξ ακος": {"head": "πολυ-άνθραξ", "rule": "run-on-ending"},
        "θεοηχής ές": {"head": "θεοηχής", "rule": "run-on-ending"},
        "κελευστός ἡ": {"head": "κελευστός", "rule": "run-on-ending"},
        "σχολιάζω άσω": {"head": "σχολιάζω", "rule": "run-on-ending"},
        "δωρίτης ἀγών": {"head": "δωρίτης", "rule": "qualified-noun"},
        "δυσπαύστως έχω": {"head": "δυσπαύστως", "rule": "adverb-idiom"}}
    assert audit["multi_word_heads"] == {"ορθο-πρίων ονος": "ορθο-πρίων"}


def test_noun_step_expands_a_run_on_head_and_an_idioms_adverb(tmp_path, monkeypatch):
    lookup, audit, calls = run_noun_step(
        tmp_path, monkeypatch, {},
        "<orth>κομίτιον ου</orth>, τὸ, the Latin comitium. Jos.",
        "<orth>ἑτερολέκτως ἔχειν</orth>, to be said differently. Stud.",
        forms={"κομίτιον": ["κομίτιον", "κομιτίου"]})
    assert calls == [("κομίτιον", "n", "ου")]
    assert lookup == {"κομίτιον": "κομίτιον", "κομιτιον": "κομίτιον",
                      "κομιτίου": "κομίτιον", "κομιτιου": "κομίτιον",
                      "ἑτερολέκτως": "ἑτερολέκτως", "ετερολεκτως": "ἑτερολέκτως"}
    assert audit["head_rules"]["multi_word_heads"] == {"count": 0, "heads": {}}
    assert audit["head_rules"]["one_word_heads"]["count"] == 2


def test_micro_sign_heads_are_read_with_mu(tmp_path, monkeypatch):
    lookup, audit, calls = run_noun_step(
        tmp_path, monkeypatch, {"μέρος": "μέρος"},
        f"<orth>{MICRO}έρος</orth>, ους, τὸ, part.",
        f"<orth>{MICRO}ετρίασις</orth>, εως, ἡ, {MICRO}ετριότης.",
        f"<orth>σχῆ{MICRO}α</orth>, ατος, τὸ, figure.",
        forms={"μετρίασις": ["μετρίασις", "μετριάσεως"],
               "σχῆμα": ["σχῆμα", "σχήματος"]})
    # the head the table has under its Greek spelling is not expanded again
    assert [head for head, _, _ in calls] == ["μετρίασις", "σχῆμα"]
    assert MICRO not in json.dumps(lookup, ensure_ascii=False)
    assert MICRO not in json.dumps(audit, ensure_ascii=False)
    assert lookup["σχήματος"] == "σχῆμα" and lookup["μετριάσεως"] == "μετρίασις"
    # only the letter differs from the earlier reading: nothing was re-read
    assert audit["head_rules"]["reread_heads"] == {"count": 0, "heads": {}}
    # ... and the audit says how each of those heads declines, for the
    # overlay's test of the rows the earlier reading wrote
    assert audit["micro_sign_endings"] == {
        "μέρος": "ους", "μετρίασις": "εως", "σχῆμα": "ατος"}
    assert "micro_sign_heads" not in audit["head_rules"]


def test_micro_sign_endings_follow_the_entry(tmp_path, monkeypatch):
    write_chunk(tmp_path / "chunks",
                f"<orth>{MICRO}ελισταγής</orth>, ές, dropping honey.",
                f"<orth>{MICRO}υριστικός</orth>, ή, όν, fragrant.",
                f"<orth>σχη{MICRO}ατοποιέω</orth>, ήσω, to shape.",
                f"<orth>{MICRO}αίωτρα</orth>, ων, τὰ, midwife's fee.",
                "<orth>ἀγαθός</orth>, ή, όν, good.")
    monkeypatch.setattr(S, "SOPH_DIR", tmp_path / "chunks")
    audit = S.new_head_audit()
    S.parse_sophocles_entries(None, audit=audit, corrections={})
    # only the heads the earlier reading spelled with the micro sign
    assert audit["micro_sign_heads"] == {
        "μελισταγής": "μελισταγής", "μυριστικός": "μυριστικός",
        "σχηματοποιέω": "σχηματοποιέω", "μαίωτρα": "μαίωτρα"}
    assert audit["reread_heads"] == {}
    # the adjective's genitive, none for a verb, no record for a plural head
    assert S.micro_sign_endings(audit) == {
        "μελισταγής": "ους", "μυριστικός": "ου", "σχηματοποιέω": ""}


def test_reviewed_corrections_replace_a_misread_head(tmp_path, monkeypatch):
    corrections = S.load_head_corrections()
    assert corrections["ορνιθείον"]["head"] == "ὀρνιθεῖον"
    write_chunk(tmp_path / "chunks",
                "<orth>ορνιθείον</orth>, ου, το, hen-coop. Phryn. P. 85. 54,",
                "<orth>Kούβρικος</orth>, ου, 6, Cubricus, the original name",
                "<orth>κίκιov</orth>,ου, τὸ, = κρότωνος ρίζα. A.",
                "<orth>Fαῦ</orth>, vau, the sixth letter of the alphabet.",
                "<orth>*σχῆμα</orth>, ατος, τὸ, figure, appearance.")
    monkeypatch.setattr(S, "SOPH_DIR", tmp_path / "chunks")
    audit = S.new_head_audit()
    entries = S.parse_sophocles_entries(None, audit=audit)
    assert set(entries) == {"ὀρνιθεῖον", "Κούβρικος", "κίκιον", "σχῆμα"}
    assert entries["ὀρνιθεῖον"]["headword"] == "ὀρνιθεῖον"
    assert (entries["κίκιον"]["gender"], entries["κίκιον"]["genitive"]) == ("n", "ου")
    assert audit["corrected_heads"] == {
        "ορνιθείον": "ὀρνιθεῖον", "Kούβρικος": "Κούβρικος", "κίκιov": "κίκιον"}
    assert audit["reread_heads"] == {
        "ορνιθείον": "ὀρνιθεῖον", "ούβρικος": "Κούβρικος", "κίκι": "κίκιον",
        "αῦ": "", "*σχῆμα": "σχῆμα"}
    # without the record the misread letters make no head at all
    plain = S.parse_sophocles_entries(None, corrections={})
    assert set(plain) == {"ορνιθείον", "κίκι", "σχῆμα"}


def test_noun_audit_supersedes_the_earlier_reading(tmp_path, monkeypatch):
    _, audit, _ = run_noun_step(
        tmp_path, monkeypatch, {},
        "<orth>ορνιθείον</orth>, ου, το, hen-coop. Phryn. P. 85. 54,",
        "<orth>Kούβρικος</orth>, ου, 6, Cubricus, the original name",
        "<orth>κίκιov</orth>,ου, τὸ, = κρότωνος ρίζα. A.",
        "<orth>*σχῆμα</orth>, ατος, τὸ, figure, appearance.",
        "<orth>Fαῦ</orth>, vau, the sixth letter of the alphabet.",
        heads={"κίκι": ["lsj9_headwords"], "σχῆμα": ["lsj9_headwords"]})
    superseded = set(audit["superseded_lemmas"])
    assert {"*σχῆμα", "αῦ", "ορνιθείον", "ούβρικος"} <= superseded
    # κίκι is a word of its own (LSJ lists it): its rows are not touched
    assert "κίκι" not in superseded
    assert audit["head_rules"]["reread_heads"]["heads"]["κίκι"] == "κίκιον"
    # the present heads never stood in the earlier tables under this spelling
    assert {"κίκιον", "σχῆμα"} <= set(audit["superseded_reread_lemmas"]) <= superseded
    assert "legacy_printed" not in audit["head_rules"]


def test_a_head_printed_plainly_too_is_superseded_outright(tmp_path, monkeypatch):
    _, audit, _ = run_noun_step(
        tmp_path, monkeypatch, {},
        f"<orth>σχῆ{MICRO}α</orth>, ατος, τὸ, figure, appearance.",
        f"<orth>δρᾶ{MICRO}α</orth>, ατος, τὸ, deed.",
        "<orth>δρᾶμα</orth>, ατος, τὸ, drama.")
    assert {"σχῆμα", "δρᾶμα"} <= set(audit["superseded_lemmas"])
    assert audit["superseded_reread_lemmas"] == ["σχῆμα"]


def test_noun_audit_supersedes_printed_heads_that_are_no_lemma(tmp_path, monkeypatch):
    sources = ["lsj9_headwords"]
    _, audit, calls = run_noun_step(
        tmp_path, monkeypatch, {"βαρύπυκνος": "βαρύπυκνος"},
        # two accents: adjective, verb and a noun whose genitive is no use
        "<orth>κατάγωγός</orth>, ὄν, bringing down.",
        "<orth>οἷές = ἐές</orth>, Porph. Cer. 47, 18.",
        "<orth>ἀττικῖσμός</orth>, οὔ, ὁ, (ἀττικίζω) Atticism.",
        # another source lists the spelling with its two accents
        "<orth>οἶκόνδε</orth>, adv. homeward.",
        # capitalized common words: adjective, verb, a withheld noun
        "<orth>Βαρύπυκνος</orth>, ον, (πυκνός) in music.",
        "<orth>Ἱπποθυτέω</orth>, to sacrifice horses.",
        "<orth>Θεοπρεπότης</orth>, τος, ἡ, the being θεοπρεπής.",
        # a name: another source has the capital
        "<orth>Πέτρα</orth>, ας, ἡ, Petra, a city.",
        heads={"βαρύπυκνος": sources, "ἱπποθυτέω": sources,
               "θεοπρεπότης": sources, "οἶκόνδε": sources,
               "πέτρα": sources, "Πέτρα": ["pd_headwords"]})
    superseded = set(audit["superseded_lemmas"])
    assert {"κατάγωγός", "οἷές", "ἀττικῖσμός",
            "Βαρύπυκνος", "Ἱπποθυτέω", "Θεοπρεπότης"} <= superseded
    assert not {"οἶκόνδε", "βαρύπυκνος", "θεοπρεπότης"} & superseded
    # a noun among them keeps no published key: it has no resolution record,
    # unlike a name whose genitive was completed
    assert set(audit["resolutions"]) == {"Πέτρα"}
    assert audit["superseded_reread_lemmas"] == []
    # the rows of the capitalized heads are read under the lowercase head; a
    # head rejected for its two accents has no reading to take them
    assert audit["reread_lemmas"] == {
        "Βαρύπυκνος": "βαρύπυκνος", "Ἱπποθυτέω": "ἱπποθυτέω",
        "Θεοπρεπότης": "θεοπρεπότης"}
    # ... where the old paradigm shows the ending the entry prints: no form
    # of the paradigm built on Θεοπρεπότης, -ου ends in the printed τος
    assert audit["reread_endings"] == {
        "Βαρύπυκνος": "ου", "Ἱπποθυτέω": "", "Θεοπρεπότης": "τος"}
    assert S.unread_printed_heads(
        {"malformed_heads": {"ἄμάντευτος": "two-accents", "Σινα-ῖτις": "inner-hyphen"},
         "lowercased_heads": {"᾿Αθεότης": "ἀθεότης"}}) == {"ἄμάντευτος", "᾿Αθεότης"}
    # the breathing review withholds a head whose breathing was read as an
    # acute; one withheld for its missing breathing alone keeps its rows
    review = {"withheld": {"όψαποίημα": {"reason": "two_accents"},
                           "ολοκαρπεύω": {"reason": "both_breathings_attested"}}}
    assert S.unread_printed_heads({}, frozenset(), review) == {"όψαποίημα"}


def test_two_accent_heads_are_corrected_from_the_record(tmp_path, monkeypatch):
    _, audit, _ = run_noun_step(
        tmp_path, monkeypatch, {},
        # a second accent, an acute for the breathing, a circumflex for the
        # diaeresis, a circumflex for a length mark
        "<orth>ἄμάντευτος</orth>, ον, (μαντεύομαι) not prophesied.",
        "<orth>Άσιάρχης</orth>, ου, ὁ, Asiarch of proconsular Asia.",
        "<orth>Ἰουδαῖσμός</orth>, οῦ, ὁ, (ἰουδαΐζω) Judaism.",
        "<orth>κρεᾶνομία</orth>, ας, ἡ, (κρεανόμος) distribution of meat.",
        "<orth>δρυφακτόώὠ</orth>, όσω, (δρύφακτον) to fence, fortify.")
    assert audit["head_rules"]["corrected_heads"]["heads"] == {
        "ἄμάντευτος": "ἀμάντευτος", "Άσιάρχης": "Ἀσιάρχης",
        "Ἰουδαῖσμός": "Ἰουδαϊσμός", "κρεᾶνομία": "κρεανομία",
        "δρυφακτόώὠ": "δρυφακτόω"}
    assert audit["head_rules"]["malformed_heads"] == {"count": 0, "heads": {}}
    # the printed readings keyed rows of the earlier tables
    assert {"ἄμάντευτος", "Άσιάρχης", "Ἰουδαῖσμός", "κρεᾶνομία",
            "δρυφακτόώὠ"} <= set(audit["superseded_lemmas"])
    # their rows are read under the corrected head where it has the same
    # letters; the forms of δρυφακτόώὠ were built on the misreading
    assert audit["reread_lemmas"] == {
        "ἄμάντευτος": "ἀμάντευτος", "Άσιάρχης": "Ἀσιάρχης",
        "Ἰουδαῖσμός": "Ἰουδαϊσμός", "κρεᾶνομία": "κρεανομία"}


def test_reread_printed_heads_keeps_only_a_reading_with_the_same_letters():
    micro = "\u00b5"
    audit = {
        "reread_heads": {
            "ἄμάντευτος": "ἀμάντευτος",          # a second accent
            "ἀραχνοῦφής": "ἀραχνοϋφής",          # a circumflex for the diaeresis
            "Άσμοδαῖος": "Ἀσμοδαῖος",            # an acute for the breathing
            "ούβρικος": "Κούβρικος",             # a letter was lost
            "ἀνικμάέω": "ἀνικμάζω",              # a letter was misread
            f"σχη{micro}ατοποιέω": "σχηματοποιέω",  # the overlay reads the micro sign
            "*Περσικός": "Περσικός",             # no text matches its keys
            "᾽Αβραμιαῖος": "᾿Αβραμιαῖος",
            "οἶκόνδε": "οἰκόνδε",                # another source's head
            "αῦ": "",                            # no reading at all
        },
        "lowercased_heads": {"Θεοτήρητος": "θεοτήρητος",
                             "᾿Αβραμιαῖος": "ἀβραμιαῖος",
                             "Ἀσμοδαῖος": "ἀσμοδαῖος",
                             "Πέτρα": "πέτρα"},
    }
    assert S.reread_printed_heads(audit, frozenset({"οἶκόνδε", "Πέτρα"})) == {
        "ἄμάντευτος": "ἀμάντευτος", "ἀραχνοῦφής": "ἀραχνοϋφής",
        # corrected, then generated under the lowercase head
        "Άσμοδαῖος": "ἀσμοδαῖος", "Ἀσμοδαῖος": "ἀσμοδαῖος",
        "Θεοτήρητος": "θεοτήρητος"}
    assert S.same_letters("Ἰουδαῖσμός", "Ἰουδαϊσμός")
    assert not S.same_letters("καταγωγύός", "καταγωγός")


@pytest.mark.parametrize("entry,ending", [
    # a printed genitive ending is required as it stands
    ("<orth>Εἰλωτίς</orth>, ίδος, 7, (Είλως) female Helot.", "ιδος"),
    ("<orth>Θεοπρεπότης</orth>, τος, ἡ, the being θεοπρεπής.", "τος"),
    ("<orth>Θρύλλημα</orth>, ατος, τὸ, (θρυλλέω) byword.", "ατος"),
    ("<orth>Θεαρχία</orth>, ας, ἡ, (ἀρχὴ) the Supreme Divinity.", "ας"),
    # the genitive ου, also where the OCR gives it as ov or the article is lost
    ("<orth>Γυμνοσοφιστής</orth>, οὔ, ο, (γυμνός) Gymnosophist.", "ου"),
    ("<orth>Διμοιρίτης</orth>, ov, 6, commonly in the plural.", "ου"),
    ("<orth>Θηκάριον</orth>, ον, τὸ, (θήκη) sheath.", "ου"),
    # an adjective in -ος by its endings, one in -ης, -ες by its genitive
    ("<orth>Θεαρχικός</orth>, ἡ, όν, belonging to the Supreme Divinity.", "ου"),
    ("<orth>Θεοτήρητος</orth>, ον, (τηρέω) divinely preserved.", "ου"),
    ("<orth>Βληχηματώδης</orth>, ες, --- βληχώδης. Nil. 821 A.", "ους"),
    # a verb and an adverb are not declined
    ("<orth>Ἱπποθυτέω</orth> (θύτης), to sacrifice horses.", ""),
    ("<orth>Θεοπρεπῶς</orth>, adv. in a manner becoming a god.", ""),
    # no printed ending: the genitive of the head's ending
    ("<orth>Ἱμάντωσις</orth>, δέσις ξύλων ἐμβαλλομένων.", "ιμαντωσεως"),
    # a plural head has no singular to check, nor a head of no known class
    ("<orth>Κελλία</orth>,ων, τὰ, Cellae, a monastic settlement.", None),
    ("<orth>Κλαυσῦς</orth>, ῦ, ὁ, Clausys. Bekker. 1195.", None),
])
def test_reread_ending_is_the_ending_the_entry_prints(tmp_path, monkeypatch,
                                                      entry, ending):
    write_chunk(tmp_path / "chunks", entry)
    monkeypatch.setattr(S, "SOPH_DIR", tmp_path / "chunks")
    (head, parsed), = S.parse_sophocles_entries(None, corrections={}).items()
    assert S.reread_ending(head, parsed) == ending


def test_reread_endings_leave_out_a_head_without_evidence():
    audit = {"reread_heads": {"ἄμάντευτος": "ἀμάντευτος"},
             "printed": {
                 "ἀμάντευτος": {"is_indecl": False, "gender": "", "first_field": "ον"},
                 "Κελλία": {"is_indecl": False, "gender": "n", "first_field": "ων"}}}
    reread = {"ἄμάντευτος": "ἀμάντευτος", "Κελλία": "κελλία", "Ἄγνωστος": "ἄγνωστος"}
    # a corrected head is looked up under its present reading; a plural head
    # and a head no entry prints have no ending
    assert S.reread_endings(reread, audit) == {"ἄμάντευτος": "ου"}


NAMES = {"Νοητός": {"source": "sophocles/0750.xml", "reason": "Noetus"},
         "Ἀγέννητος": {"source": "sophocles/0057.xml", "reason": "a Gnostic figure"}}


def test_the_rows_of_a_reviewed_proper_name_stay_under_the_name():
    """The lowercase rule cannot tell Νοητός (Noetus) from a capitalized
    common word, and the entry is generated under νοητός like one. The rows
    the earlier tables have under the name are the name's: they are neither
    superseded with the printed head nor read under the common word."""
    heads = frozenset({"νοητός", "ἀγέννητος", "θεοτήρητος", "ἀργυρός"})
    assert S.settle_head("Νοητός", heads) == ("νοητός", "lowercased")
    audit = {
        "lowercased_heads": {"Νοητός": "νοητός", "᾿Αγέννητος": "ἀγέννητος",
                             "Θεοτήρητος": "θεοτήρητος", "Ἀργυρός": "ἀργυρός"},
        # a name printed with a second accent, corrected
        "reread_heads": {"Ἄργυρός": "Ἀργυρός"},
    }
    names = dict(NAMES, **{"Ἀργυρός": {"source": "sophocles/0200.xml",
                                      "reason": "Argyrus"}})
    assert S.unread_printed_heads(audit) == {
        "Νοητός", "᾿Αγέννητος", "Θεοτήρητος", "Ἀργυρός"}
    assert S.unread_printed_heads(audit, names=names) == {"Θεοτήρητος"}
    assert S.reread_printed_heads(audit) == {
        "Νοητός": "νοητός", "Θεοτήρητος": "θεοτήρητος",
        "Ἄργυρός": "ἀργυρός", "Ἀργυρός": "ἀργυρός"}
    # the misread spelling of a name is read under the name
    assert S.reread_printed_heads(audit, names=names) == {
        "Θεοτήρητος": "θεοτήρητος", "Ἄργυρός": "Ἀργυρός"}


def test_noun_audit_keeps_the_rows_of_a_proper_name(tmp_path, monkeypatch):
    sources = ["lsj9_headwords"]
    monkeypatch.setattr(S, "load_proper_names", lambda: NAMES)
    _, audit, _ = run_noun_step(
        tmp_path, monkeypatch, {"νοητός": "νοητός", "θεοτήρητος": "θεοτήρητος"},
        "<orth>Νοητός</orth>, οῦ, ὁ, Noetus, the founder of patropassianism.",
        "<orth>Θεοτήρητος</orth>, ον, (τηρέω) divinely preserved.",
        heads={"νοητός": sources, "θεοτήρητος": sources})
    assert audit["head_rules"]["lowercased_heads"]["heads"] == {
        "Νοητός": "νοητός", "Θεοτήρητος": "θεοτήρητος"}
    assert audit["head_rules"]["proper_names"] == {
        "count": 1, "heads": {"Νοητός": "Noetus"}}
    assert "Θεοτήρητος" in audit["superseded_lemmas"]
    assert "Νοητός" not in audit["superseded_lemmas"]
    assert audit["reread_lemmas"] == {"Θεοτήρητος": "θεοτήρητος"}
    assert audit["reread_endings"] == {"Θεοτήρητος": "ου"}


def test_every_proper_name_is_a_capitalized_head_with_a_common_twin():
    names = S.load_proper_names()
    assert len(names) >= 30
    heads = S.load_lexical_heads()
    if heads is None:
        pytest.skip("grc_lexical_heads.json is not installed")
    corrections = S.load_head_corrections()
    for name, record in names.items():
        chunk = (S.SOPH_DIR.parent / "chunks" / record["source"].split("/")[1])
        text = chunk.read_text(encoding="utf-8")
        printed = {S.sanitize_form(S.read_orth(S.settle_orth(orth, corrections))[0])
                   for orth in re.findall(r"<orth>(.*?)</orth>", text)}
        assert name in printed, name
        # without the record the head rules would read it as the common word
        assert S.lowercase_twin(name, heads) == name.lower(), name


def test_proper_names_refuse_a_malformed_record(tmp_path):
    from dilemma_build import sophocles_ocr as O
    path = tmp_path / "corrections.json"

    def load(name="Νοητός", **fields):
        record = {"source": "sophocles/0750.xml", "reason": "Noetus", **fields}
        path.write_text(json.dumps({"schema_version": 1, "corrections": {},
                                    "names": {name: record}}, ensure_ascii=False),
                        encoding="utf-8")
        return O.load_proper_names(path)

    assert load() == {"Νοητός": {"source": "sophocles/0750.xml", "reason": "Noetus"}}
    assert O.load_proper_names(tmp_path / "absent.json") == {}
    path.write_text(json.dumps({"schema_version": 1, "corrections": {}}),
                    encoding="utf-8")
    assert O.load_proper_names(path) == {}
    for bad in ({"name": "νοητός"}, {"name": "Noetus"}, {"name": "ΝΟΗΤΟΣ"},
                {"name": "᾿Αγέννητος"}, {"reason": " "}, {"source": "0750.xml"}):
        with pytest.raises(SystemExit):
            load(**bad)
    # the names are part of the generator's pinned inputs
    assert O.corrections_sha256({}) != O.corrections_sha256(load())


def test_every_correction_names_an_orth_of_its_chunk():
    corrections = S.load_head_corrections()
    assert len(corrections) >= 30
    for orth, record in corrections.items():
        chunk = (S.SOPH_DIR.parent / "chunks" / record["source"].split("/")[1])
        text = S.greek_mu(chunk.read_text(encoding="utf-8"))
        assert re.search(rf"<orth>\s*{re.escape(orth)}\s*</orth>", text), orth
        # the record changes the reading
        assert S.read_orth(orth)[0] != record["head"], orth
        assert S.read_orth(S.settle_orth(orth, corrections))[:2] == (record["head"], "")


def test_head_corrections_refuse_a_malformed_record(tmp_path):
    from dilemma_build import sophocles_ocr as O
    path = tmp_path / "corrections.json"

    def load(head="ὀρνιθεῖον", orth="ορνιθείον", **fields):
        record = {"head": head, "source": "sophocles/0800.xml", "reason": "accent",
                  **fields}
        path.write_text(json.dumps({"schema_version": 1, "corrections": {orth: record}},
                                   ensure_ascii=False), encoding="utf-8")
        return O.load_head_corrections(path)

    assert load()["ορνιθείον"]["head"] == "ὀρνιθεῖον"
    assert O.load_head_corrections(tmp_path / "absent.json") == {}
    for bad in ({"head": "ὀρνιθείὸν"}, {"head": "ορνιθειον"}, {"head": "ὀρνιθεῖον οἶκος"},
                {"head": f"κό{MICRO}η"}, {"head": "Kόρος"}, {"reason": " "},
                {"source": "0800.xml"}, {"orth": f"{MICRO}έρος"}, {"orth": " ορνιθείον"}):
        with pytest.raises(SystemExit):
            load(**bad)


def test_verb_step_expands_an_entry_glossed_by_an_equals_sign(tmp_path, monkeypatch):
    lookup, calls = run_verb_step(
        tmp_path, monkeypatch, {"ἐπιμορφάζω": "ἐπιμορφάζω"},
        "<orth>ἀκρατεύω</orth>, = ἀκρατεύομαι. Plut. Frag. 838 C.",
        "<orth>ἀνακοσμοποιέω</orth>, = preceding. Method. 272 C.",
        "<orth>ἀχρεύω</orth> = ἀχρειόω. Paul. Rom. 3, 12.",
        "<orth>*πιάζω</orth>, = πιέζω, to lay hold of, to seize, catch.",
        "<orth>ἐκνεωτερίξω</orth>, = νεωτερίζω strengthened. Phot. III, 680 C.",
        # a head that sends the reader to another verb is expanded too (#144)
        "<orth>ἀπάχθομαι</orth>, see ἀπέχθομαι.",
        "<orth>φαγέω</orth>, see κρεωφαγέω.",
        # no verb: a name, a dative phrase, a noun in -ώ
        "<orth>Ἰεύω</orth>, = Ἰαῶ.",
        "<orth>ἐπισηκρήτῳ</orth>, = ἀσηκρήτις. Caesarius 851 (titul.).",
        "<orth>σαββώ</orth>, = σάββατον.",
        # already a key
        "<orth>ἐπιμορφάζω</orth>, = μορφάζω.")
    assert calls == ["ἀκρατεύω", "ἀνακοσμοποιέω", "ἀχρεύω", "πιάζω", "ἐκνεωτερίζω",
                     "ἀπάχθομαι", "φαγέω"]
    assert lookup["ἀνακοσμοποιέω"] == "ἀνακοσμοποιέω"
    assert lookup["ἀπάχθομαι"] == "ἀπάχθομαι" and "Ἰεύω" not in lookup


def test_verb_step_expands_a_see_reference_to_another_verb(tmp_path, monkeypatch):
    # The lexicon prints the spelling as a head and sends the reader to the
    # verb's other spelling (#144). A reference to a word that is no verb
    # citation form is no evidence that the head is a verb.
    lookup, calls = run_verb_step(
        tmp_path, monkeypatch, {},
        "<orth>ἀναστάζω</orth>, see ἀνστάζω.",
        "<orth>ὑφέλλω</orth>, see ὑφείλω.",
        "<orth>εἰδέτω</orth>, see οἶδα.",
        "<orth>ἀπεσθέομαι</orth>, see ἀπησθημένος.",
        "<orth>ἀκρατεύω</orth>, = ἀκρατεύομαι.",
        heads={})
    assert sorted(calls) == sorted(["ἀκρατεύω", "ἀναστάζω", "ὑφέλλω"])


def test_see_reference_names_its_target(tmp_path, monkeypatch):
    entries, _ = parse(tmp_path, monkeypatch,
                       "<orth>φαγέω</orth>, see κρεωφαγέω.",
                       "<orth>ἀνακοσμοποιέω</orth>, = preceding.")
    assert entries["φαγέω"]["xref"] == "see"
    assert entries["φαγέω"]["xref_target"] == "κρεωφαγέω"
    assert entries["ἀνακοσμοποιέω"]["xref"] == "equals"
    assert entries["ἀνακοσμοποιέω"]["xref_target"] == ""

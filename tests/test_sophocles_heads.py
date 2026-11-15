"""Head rules of the Sophocles expansion (build/expand_sophocles.py)."""
import json
from pathlib import Path
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
    ("κομίτιον ου", "κομίτιον", "multi-word"),
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
    assert orth.strip().endswith(rest)
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
        "<orth>Ἔρετριάς</orth>, άδος, ἡ, of Eretria.",
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
        "Ἔρετριάς": "two-accents", "Σινα-ῖτις": "inner-hyphen",
        "ἀνα-πλήρως": "inner-hyphen", "ὕπο-κατασκευή": "inner-hyphen"}}
    assert audit["head_rules"]["repaired_heads"] == {"count": 2, "heads": {
        "κατα-μαρτυρία": "καταμαρτυρία", "ἡλιο-κάμινος": "ἡλιοκάμινος"}}
    # the historical layer keyed its rows by the printed head, so a completed
    # genitive still supersedes that name, rejected and repaired heads alike
    assert set(audit["superseded_lemmas"]) == {
        "Σινα-ῖτις", "Ἔρετριάς", "κατα-μαρτυρία", "ἡλιο-κάμινος"}


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
                    "θεοπρεπῶς", "ἑλένη", "Ἑλένη"})


@pytest.mark.parametrize("printed,head,rule", [
    # only the lowercase spelling is another source's head: a common word
    ("Ἀθεότης", "ἀθεότης", "lowercased"),
    ("Ἔκπτυξις", "ἔκπτυξις", "lowercased"),
    ("Θεοπρεπῶς", "θεοπρεπῶς", "lowercased"),
    ("Ἱππο-θυτέω", "ἱπποθυτέω", "joined"),
    # another source lists the capitalized head: a name beside a common word
    ("Πέτρα", "Πέτρα", ""),
    ("Ἑλένη", "Ἑλένη", ""),
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
        "<orth>Ἀθεότης</orth>, ητος, ἡ, ungodliness.",
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
        "Ἀθεότης": "ἀθεότης", "Ἔκπτυξις": "ἔκπτυξις", "Θεοπρεπῶς": "θεοπρεπῶς"}}


def test_verb_step_generates_a_common_word_under_its_lowercase_head(
        tmp_path, monkeypatch):
    lookup, calls = run_verb_step(
        tmp_path, monkeypatch, {},
        "<orth>Ἱπποθυτέω</orth>, to sacrifice horses.",
        "<orth>Ἰουδαΐζω</orth>, to judaize.",
        heads={"ἱπποθυτέω": ["lsj9_headwords"]})
    assert calls == ["ἱπποθυτέω", "Ἰουδαΐζω"]

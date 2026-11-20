"""The verb pass reads only the pinned LSJ9 glosses export."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
import expand_lsj


def _pin(tmp_path, text):
    glosses = tmp_path / "lsj9_glosses.jsonl"
    glosses.write_text(text, encoding="utf-8")
    pin = tmp_path / "pin.json"
    pin.write_text(json.dumps({
        "path": glosses.name, "size": glosses.stat().st_size,
        "sha256": hashlib.sha256(glosses.read_bytes()).hexdigest()}),
        encoding="utf-8")
    return glosses, pin


def test_repo_pin_names_the_glosses_export():
    pin = json.loads(expand_lsj.LSJ9_GLOSSES_PIN.read_text(encoding="utf-8"))
    assert pin["path"] == "lsj9_glosses.jsonl"
    assert len(pin["sha256"]) == 64 and pin["size"] > 0
    supplement = pin[expand_lsj.SUPPLEMENT_PIN_KEY]
    assert supplement["path"] == expand_lsj.LSJ9_GLOSSES_SUPPLEMENT.name
    assert len(supplement["sha256"]) == 64 and supplement["size"] > 0
    assert supplement["sha256"] != pin["sha256"]


def test_matching_export_passes(tmp_path, monkeypatch):
    glosses, pin = _pin(tmp_path, '{"headword": "λύω", "text": "loose"}\n')
    monkeypatch.setattr(expand_lsj, "LSJ9_GLOSSES", glosses)
    monkeypatch.setattr(expand_lsj, "LSJ9_GLOSSES_PIN", pin)
    assert expand_lsj.verify_lsj_glosses()["size"] == glosses.stat().st_size


def test_missing_or_other_export_stops_the_verb_pass(tmp_path, monkeypatch):
    glosses, pin = _pin(tmp_path, '{"headword": "λύω", "text": "loose"}\n')
    monkeypatch.setattr(expand_lsj, "LSJ9_GLOSSES_PIN", pin)
    monkeypatch.setattr(expand_lsj, "LSJ9_GLOSSES", tmp_path / "absent.jsonl")
    with pytest.raises(SystemExit, match="not found"):
        expand_lsj.verify_lsj_glosses()
    glosses.write_text('{"headword": "λύω"}\n', encoding="utf-8")
    monkeypatch.setattr(expand_lsj, "LSJ9_GLOSSES", glosses)
    with pytest.raises(SystemExit, match="does not match"):
        expand_lsj.verify_lsj_glosses()


def _pin_with_supplement(tmp_path, text, supplement_text):
    glosses, pin = _pin(tmp_path, text)
    supplement = tmp_path / "lsj9_glosses_earlier.jsonl"
    supplement.write_text(supplement_text, encoding="utf-8")
    record = json.loads(pin.read_text(encoding="utf-8"))
    record[expand_lsj.SUPPLEMENT_PIN_KEY] = {
        "path": supplement.name, "size": supplement.stat().st_size,
        "sha256": hashlib.sha256(supplement.read_bytes()).hexdigest()}
    pin.write_text(json.dumps(record), encoding="utf-8")
    return glosses, supplement, pin


def test_supplement_is_pinned_too(tmp_path, monkeypatch):
    glosses, supplement, pin = _pin_with_supplement(
        tmp_path, '{"headword": "λύω", "text": "loose"}\n',
        '{"headword": "λύω", "text": "fut. λύσω"}\n')
    monkeypatch.setattr(expand_lsj, "LSJ9_GLOSSES", glosses)
    monkeypatch.setattr(expand_lsj, "LSJ9_GLOSSES_PIN", pin)
    monkeypatch.setattr(expand_lsj, "LSJ9_GLOSSES_SUPPLEMENT", supplement)
    actual = expand_lsj.verify_lsj_glosses()
    assert actual["supplement"]["size"] == supplement.stat().st_size
    supplement.write_text('{"headword": "λύω"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match"):
        expand_lsj.verify_lsj_glosses()
    monkeypatch.setattr(expand_lsj, "LSJ9_GLOSSES_SUPPLEMENT",
                        tmp_path / "absent.jsonl")
    with pytest.raises(SystemExit, match="not found"):
        expand_lsj.verify_lsj_glosses()


def test_supplement_heads_load_only_when_pinned(tmp_path, monkeypatch):
    glosses, supplement, pin = _pin_with_supplement(
        tmp_path, '{"headword": "λύω", "text": "loose"}\n',
        '{"headword": "λύω", "text": "fut. λύσω"}\n'
        '{"headword": "λύω", "text": "sense", "level": "sub"}\n')
    monkeypatch.setattr(expand_lsj, "LSJ9_GLOSSES_PIN", pin)
    monkeypatch.setattr(expand_lsj, "LSJ9_GLOSSES_SUPPLEMENT", supplement)
    monkeypatch.setattr(expand_lsj, "_LSJ_SUPPLEMENT_HEAD_TEXTS", None)
    assert expand_lsj.load_lsj_supplement_head_texts() == {"λύω": "fut. λύσω"}
    _, plain_pin = _pin(tmp_path, '{"headword": "λύω", "text": "loose"}\n')
    monkeypatch.setattr(expand_lsj, "LSJ9_GLOSSES_PIN", plain_pin)
    monkeypatch.setattr(expand_lsj, "_LSJ_SUPPLEMENT_HEAD_TEXTS", None)
    assert expand_lsj.load_lsj_supplement_head_texts() == {}


# κατατέμνω in the two exports: the current text stops at the first
# citation, the earlier one keeps the aorist.
CURRENT_HEAD = "Ion. κατα-τάμνω, fut. -τεμῶ Ar.Ach.301: aor."
EARLIER_HEAD = ("fut. -τεμῶ : aor. κατέτεμον (v. infr.); Ion. and Dor. "
                "κατέταμον , Tab.Heracl.1.14:—cut in pieces, cut up")


def test_supplement_adds_the_parts_the_current_text_lacks():
    # The hyphenated future is written out from the headword.
    assert expand_lsj.merged_principal_parts("κατατέμνω", CURRENT_HEAD) == {
        "fut": "κατατεμῶ"}
    assert expand_lsj.merged_principal_parts(
        "κατατέμνω", CURRENT_HEAD, EARLIER_HEAD) == {
        "fut": "κατατεμῶ", "aor": "κατέτεμον"}


def test_current_text_decides_the_parts_it_gives():
    current = "fut. παύσω: aor. ἔπαυσα: pf. πέπαυκα"
    earlier = "fut. παυσῶ: aor. 2 ἔπαυον: pf. πέπαυχα: aor. Pass. ἐπαύθην"
    own = expand_lsj.merged_principal_parts("παύω", current)
    merged = expand_lsj.merged_principal_parts("παύω", current, earlier)
    assert {key: merged[key] for key in own} == own
    # The aorist slots are filled as a group, so a numbered aorist of the
    # earlier export does not join an aorist the current text gave.
    assert "aor2" not in merged
    assert set(merged) - set(own) <= {"aor_p"}

"""Generated lookup bytes must not depend on Python's randomized set order."""
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = r'''
import importlib, json, pathlib, sys
root, output, name = sys.argv[1:]
sys.path.insert(0, str(pathlib.Path(root) / "build"))
m = importlib.import_module(name)
m.AG_LOOKUP = pathlib.Path(output)
m.AG_LOOKUP.write_text(json.dumps({"κοινόν": "base"}, ensure_ascii=False))
entries = {
    "λόγος": {"gender": "m", "genitive": "λόγου"},
    "λύω": {"gender": "", "genitive": ""},
}
m.get_wtp = lambda: None
m.expand_noun = lambda *a, **kw: ({"λόγος", "λόγου", "λόγῳ", "λόγων", "κοινόν"}, "")
m.expand_verb = lambda *a, **kw: ({"λύω", "λύεις", "λύει", "λῦε", "κοινόν"}, "")
m.load_wiktionary_genitives = lambda: {}
m.noun_generation_input_hashes = lambda *a, **kw: {}
if name == "expand_sophocles":
    m.sophocles_noun_input_hashes = lambda: {}
if name == "expand_lsj":
    m.parse_lsj_entries = lambda: entries
    m.load_wiktionary_forms = lambda: {}
    m.load_lsj_head_texts = lambda: {}
else:
    m.analyze = lambda: (entries, ["λόγος"], [])
    m.parse_sophocles_entries = lambda: entries
    m.find_verb_candidates = lambda *a: ["λύω"]
m.expand_all()
m.expand_verbs()
'''


def test_lookup_bytes_and_source_priority_are_hash_seed_independent(tmp_path):
    for module in ["expand_lsj", "expand_sophocles"]:
        outputs = []
        for seed in ["1", "97"]:
            path = tmp_path / f"{module}-{seed}.json"
            subprocess.run(
                [sys.executable, "-c", SCRIPT, str(ROOT), str(path), module],
                env={**os.environ, "PYTHONHASHSEED": seed}, check=True,
                capture_output=True, text=True,
            )
            outputs.append(path.read_bytes())
        assert outputs[0] == outputs[1]
        lookup = json.loads(outputs[0])
        assert lookup["κοινόν"] == "base"
        assert lookup["λόγῳ"] == "λόγος"
        assert lookup["λύεις"] == "λύω"

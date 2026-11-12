"""A generated paradigm file that lost its accents is caught."""
import ast
from pathlib import Path

from dilemma.form_sanitize import MIN_POLYTONIC_SHARE, polytonic_share

ROOT = Path(__file__).resolve().parent.parent


def test_polytonic_share_counts_forms_with_a_diacritic():
    assert polytonic_share([]) == 1.0
    assert polytonic_share(["ματρωνίκια", "ματρωνικίου", "ἀβάκια"]) == 1.0
    assert polytonic_share(["ματρωνικια", "ματρωνικιου", "ἀβάκια", "αβακια"]) == 0.25
    # The shape of the file that fed the lookup: bare keys under accented lemmas.
    stale = {"ματρωνικια": "ματρωνίκιον", "ματρωνικιου": "ματρωνίκιον"}
    assert polytonic_share(stale) < MIN_POLYTONIC_SHARE


def test_expansion_and_lookup_build_both_check_the_share():
    for name in ("build/expand_lbg.py", "build_lookup_db.py"):
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
        calls = {node.func.id for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        assert "polytonic_share" in calls, name

"""PyPA packaging and the repository's source-validation tools must coexist."""
from pathlib import Path
import subprocess
import sys

import build


ROOT = Path(__file__).resolve().parents[1]


def test_pypa_build_and_project_validation_imports_coexist():
    assert hasattr(build, "ProjectBuilder"), "Install the development extra: pip install -e '.[dev]'"
    from dilemma_build.build_grc_lexical_heads import has_definition
    from build_data import is_independent_lexical_entry

    entry = {"word": "λέξη", "pos": "noun", "senses": [{"glosses": ["a word"]}]}
    assert has_definition(entry, "en")
    assert is_independent_lexical_entry(entry, "en")
    # Resolving project helpers must not replace PyPA's module in sys.modules.
    assert sys.modules["build"] is build
    assert hasattr(build, "ProjectBuilder")


def test_pypa_build_cli_is_available_from_the_checkout():
    result = subprocess.run([sys.executable, "-m", "build", "--version"],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "build" in result.stdout


def test_existing_builder_script_path_works_outside_the_checkout(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "build/build_grc_lexical_heads.py"), "--help"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--kaikki" in result.stdout

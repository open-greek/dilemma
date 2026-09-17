import datetime as dt
import tomllib

import pytest

from scripts import release as release_module
from scripts.release import render_release_metadata


def _release_tree(tmp_path):
    (tmp_path / "VERSION").write_text("1.2.4\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "dilemma-nlp"\nversion = "1.2.4"\n',
        encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n### Added\n- A feature.\n\n"
        "## [1.2.4] - 2026-09-16\n\n- Previous.\n\n"
        "[1.2.4]: https://example.test/1.2.4\n",
        encoding="utf-8",
    )


def test_render_release_metadata_updates_all_authorities(tmp_path):
    _release_tree(tmp_path)
    updates = render_release_metadata(
        tmp_path, "1.3.0", dt.date(2026, 9, 17)
    )

    assert updates[tmp_path / "VERSION"] == "1.3.0\n"
    parsed = tomllib.loads(updates[tmp_path / "pyproject.toml"])
    assert parsed["project"]["version"] == "1.3.0"
    changelog = updates[tmp_path / "CHANGELOG.md"]
    assert "## [Unreleased]\n\n## [1.3.0] - 2026-09-17" in changelog
    assert "[1.3.0]: https://github.com/open-greek/dilemma/releases/tag/1.3.0" in changelog


def test_render_release_metadata_rejects_empty_unreleased_section(tmp_path):
    _release_tree(tmp_path)
    path = tmp_path / "CHANGELOG.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "\n### Added\n- A feature.\n", ""
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no Unreleased entries"):
        render_release_metadata(tmp_path, "1.3.0", dt.date(2026, 9, 17))


def test_render_release_metadata_requires_version_increase(tmp_path):
    _release_tree(tmp_path)
    with pytest.raises(ValueError, match="must be greater"):
        render_release_metadata(tmp_path, "1.2.4", dt.date(2026, 9, 17))


@pytest.mark.parametrize(
    ("local_result", "remote_result", "message"),
    [
        ("1.3.0", "", "local tag 1.3.0 already exists"),
        ("", "abc123\trefs/tags/1.3.0", "remote tag 1.3.0 already exists"),
    ],
)
def test_ensure_tag_available_rejects_collisions(
    monkeypatch, local_result, remote_result, message
):
    def fake_run(*args, capture=False):
        assert capture
        if args[:3] == ("git", "tag", "--list"):
            return local_result
        if args[:3] == ("git", "ls-remote", "--tags"):
            return remote_result
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(release_module, "_run", fake_run)
    with pytest.raises(RuntimeError, match=message):
        release_module._ensure_tag_available("1.3.0")

#!/usr/bin/env python3
"""Prepare, test, tag, and publish a Dilemma release.

Usage:
    python3 scripts/release.py 1.3.0
    python3 scripts/release.py 1.3.0 --dry-run

The command requires a clean, synchronized ``main`` branch. It updates all
version metadata and the changelog, commits and pushes ``main``, waits for the
Tests workflow on that exact commit, then creates and pushes a numeric
annotated tag. The tag workflow verifies the completed Tests run before it
publishes GitHub and PyPI releases.
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "open-greek/dilemma"
SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def _run(*args: str, capture: bool = False) -> str:
    result = subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else ""


def _version_tuple(version: str) -> tuple[int, int, int]:
    match = SEMVER_RE.fullmatch(version)
    if not match:
        raise ValueError(f"invalid semantic version: {version!r}")
    return tuple(int(part) for part in match.groups())


def render_release_metadata(root: Path, new_version: str,
                            release_date: dt.date) -> dict[Path, str]:
    """Return updated release file contents without writing them."""
    new_tuple = _version_tuple(new_version)
    version_path = root / "VERSION"
    pyproject_path = root / "pyproject.toml"
    changelog_path = root / "CHANGELOG.md"

    old_version = version_path.read_text(encoding="utf-8").strip()
    if new_tuple <= _version_tuple(old_version):
        raise ValueError(
            f"new version {new_version} must be greater than {old_version}"
        )

    pyproject_text = pyproject_path.read_text(encoding="utf-8")
    parsed = tomllib.loads(pyproject_text)
    project_version = parsed["project"]["version"]
    if project_version != old_version:
        raise ValueError(
            f"VERSION ({old_version}) and pyproject.toml "
            f"({project_version}) disagree"
        )
    version_pattern = re.compile(
        rf'(?m)^version = "{re.escape(old_version)}"$'
    )
    pyproject_updated, replacements = version_pattern.subn(
        f'version = "{new_version}"', pyproject_text, count=1
    )
    if replacements != 1:
        raise ValueError("could not update [project].version")

    changelog_text = changelog_path.read_text(encoding="utf-8")
    unreleased = "## [Unreleased]\n"
    if changelog_text.count(unreleased) != 1:
        raise ValueError("CHANGELOG.md must contain one Unreleased heading")
    after_unreleased = changelog_text.split(unreleased, 1)[1]
    next_heading = after_unreleased.find("\n## [")
    if next_heading < 0 or not after_unreleased[:next_heading].strip():
        raise ValueError("CHANGELOG.md has no Unreleased entries to publish")
    release_heading = f"## [{new_version}] - {release_date.isoformat()}"
    changelog_updated = changelog_text.replace(
        unreleased,
        f"{unreleased}\n{release_heading}\n",
        1,
    )
    link_marker = f"[{old_version}]:"
    if changelog_updated.count(link_marker) != 1:
        raise ValueError(f"CHANGELOG.md is missing the {old_version} link")
    changelog_updated = changelog_updated.replace(
        link_marker,
        f"[{new_version}]: https://github.com/{REPOSITORY}/releases/tag/"
        f"{new_version}\n{link_marker}",
        1,
    )

    return {
        version_path: f"{new_version}\n",
        pyproject_path: pyproject_updated,
        changelog_path: changelog_updated,
    }


def _ensure_release_context() -> None:
    if shutil.which("git") is None or shutil.which("gh") is None:
        raise RuntimeError("release requires both git and gh")
    if _run("git", "branch", "--show-current", capture=True) != "main":
        raise RuntimeError("release must run from main")
    if _run("git", "status", "--porcelain", capture=True):
        raise RuntimeError("release requires a clean working tree")
    _run("git", "fetch", "origin", "main")
    head = _run("git", "rev-parse", "HEAD", capture=True)
    remote = _run("git", "rev-parse", "origin/main", capture=True)
    if head != remote:
        raise RuntimeError("main must be synchronized with origin/main")


def _ensure_tag_available(version: str) -> None:
    if _run("git", "tag", "--list", version, capture=True):
        raise RuntimeError(f"local tag {version} already exists")
    remote_tag = _run(
        "git", "ls-remote", "--tags", "origin", f"refs/tags/{version}",
        capture=True,
    )
    if remote_tag:
        raise RuntimeError(f"remote tag {version} already exists")


def _find_workflow_run(workflow: str, sha: str) -> dict | None:
    raw = _run(
        "gh", "run", "list",
        "--repo", REPOSITORY,
        "--workflow", workflow,
        "--commit", sha,
        "--event", "push",
        "--limit", "20",
        "--json", "databaseId,status,conclusion,headSha,url,createdAt",
        capture=True,
    )
    runs = json.loads(raw or "[]")
    matches = [run for run in runs if run.get("headSha") == sha]
    if not matches:
        return None
    return max(matches, key=lambda run: run.get("createdAt", ""))


def _wait_for_workflow(workflow: str, sha: str,
                       timeout_seconds: int = 2700) -> dict:
    deadline = time.monotonic() + timeout_seconds
    run = None
    while time.monotonic() < deadline:
        run = _find_workflow_run(workflow, sha)
        if run:
            break
        time.sleep(5)
    if not run:
        raise RuntimeError(f"timed out waiting for {workflow} to start")

    _run(
        "gh", "run", "watch", str(run["databaseId"]),
        "--repo", REPOSITORY,
        "--exit-status",
    )
    completed = _find_workflow_run(workflow, sha)
    if not completed or completed.get("conclusion") != "success":
        raise RuntimeError(f"{workflow} did not complete successfully")
    return completed


def _print_diff(path: Path, old: str, new: str) -> None:
    relative = path.relative_to(ROOT)
    sys.stdout.writelines(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=str(relative),
        tofile=str(relative),
    ))


def release(new_version: str, *, dry_run: bool = False,
            release_date: dt.date | None = None) -> None:
    release_date = release_date or dt.date.today()
    updates = render_release_metadata(ROOT, new_version, release_date)
    if dry_run:
        for path, updated in updates.items():
            _print_diff(path, path.read_text(encoding="utf-8"), updated)
        return

    _ensure_release_context()
    _ensure_tag_available(new_version)

    for path, updated in updates.items():
        path.write_text(updated, encoding="utf-8")

    _run("git", "diff", "--check")
    _run("git", "add", "VERSION", "pyproject.toml", "CHANGELOG.md")
    old_version = _run("git", "show", "HEAD:VERSION", capture=True)
    _run(
        "git", "commit", "-m",
        f"bump version: {old_version} -> {new_version}",
    )
    sha = _run("git", "rev-parse", "HEAD", capture=True)
    _run("git", "push", "origin", "main")
    tests = _wait_for_workflow("test.yml", sha)
    print(f"Tests passed: {tests['url']}")

    _run("git", "tag", "-a", new_version, "-m", f"v{new_version}")
    _run("git", "push", "origin", new_version)
    published = _wait_for_workflow("release.yml", sha)
    print(f"Release published: {published['url']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="new numeric semantic version")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="show metadata changes without writing, committing, or pushing",
    )
    parser.add_argument(
        "--date", type=dt.date.fromisoformat,
        help="release date (YYYY-MM-DD; defaults to today)",
    )
    args = parser.parse_args()
    try:
        release(args.version, dry_run=args.dry_run, release_date=args.date)
    except (KeyError, OSError, RuntimeError, ValueError,
            subprocess.CalledProcessError) as exc:
        print(f"release failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

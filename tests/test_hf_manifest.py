"""Guards for data/hf_manifest.json and the CI steps that consume it.

The lookup artifacts ship on HuggingFace rather than in git, so the manifest is
what ties a git commit to the exact bytes CI must test against. These tests
keep the three pieces (the two pinned lists in scripts/hf_data.py, the
manifest, and the workflow's download step) from drifting apart; the
byte-level check of the real artifacts is `hf_data.py verify`, which runs in
CI right after the download.

`python -m dilemma download` serves more than CI downloads, so the tests also
hold the lists against the files the runtime package names (read from its
source) and run every subcommand against an in-memory stand-in for the Hub.
"""
import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "hf_manifest.json"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "test.yml"

# scripts/ is not a package, so load the module by path.
_spec = importlib.util.spec_from_file_location(
    "hf_data", ROOT / "scripts" / "hf_data.py")
hf_data = importlib.util.module_from_spec(_spec)
sys.modules["hf_data"] = hf_data
_spec.loader.exec_module(hf_data)


@pytest.fixture(scope="module")
def manifest():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def _committed_data_blobs() -> dict:
    """path -> git blob id of every committed file under data/."""
    try:
        listed = subprocess.run(
            ["git", "ls-tree", "-r", "HEAD", "--", "data"], cwd=ROOT,
            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    blobs = {}
    for line in listed.splitlines():
        meta, path = line.split("\t", 1)
        blobs[path] = meta.split()[2]
    if not blobs:
        pytest.skip("no committed data files")
    return blobs


def test_manifest_covers_exactly_the_tracked_artifacts(manifest):
    assert manifest["repo"] == hf_data.REPO
    assert list(manifest["files"]) == hf_data.TRACKED


def test_manifest_covers_exactly_the_pinned_only_files(manifest):
    """The second list: pinned and compared with the Hub, not downloaded by
    CI. Together the two lists are everything the manifest holds (#140)."""
    assert list(manifest["pinned_only"]) == hf_data.PINNED_ONLY
    assert set(manifest) == {"_comment", "repo", "generated", "files", "pinned_only"}
    both = hf_data.TRACKED + hf_data.PINNED_ONLY
    assert len(both) == len(set(both))
    assert list(hf_data.pinned_entries()) == both


def test_runtime_pos_tables_are_pinned():
    """`python -m dilemma download` serves the Hub copies of the POS tables,
    so CI has to wait for, download and verify them like every other table
    the runtime loads (#140)."""
    assert {
        "data/ag_pos_lookup.json",
        "data/glaux_pos_lookup.json",
        "data/treebank_pos_lookup.json",
        "data/mg_pos_lookup.json",
        "data/lsj9_pos_lookup.json",
    } <= set(hf_data.TRACKED)
    # No code reads the Medieval POS table; it is pinned because it is served.
    assert "data/med_pos_lookup.json" in hf_data.PINNED_ONLY


def test_build_tables_are_pinned_and_not_downloaded_by_ci(capsys):
    assert {
        "data/ag_lookup.json", "data/ag_pairs.json",
        "data/med_lookup.json", "data/med_pairs.json",
        "data/ag_source_alias_audit.json", "data/ag_source_alias_manifest.json",
        "data/glaux_pairs.json", "data/lbg_pairs.json", "data/nt_pairs.json",
    } <= set(hf_data.PINNED_ONLY)
    assert hf_data.main(["files"]) == 0
    downloaded = capsys.readouterr().out.split()
    assert downloaded == hf_data.TRACKED
    assert not set(downloaded) & set(hf_data.PINNED_ONLY)


def test_git_tracked_artifacts_are_pinned_at_their_committed_bytes(manifest):
    """A pinned table that is also in git must be pinned at the committed
    blob, or installs (Hub copy) and source checkouts (git copy) disagree.

    Fails from the commit that changes such a table until the release runs
    `hf_data.py push`, which is the point."""
    blobs = _committed_data_blobs()
    pins = {**manifest["files"], **manifest["pinned_only"]}
    wrong = {path: (entry["git_sha1"], blobs[path])
             for path, entry in pins.items()
             if path in blobs and path != "data/hf_manifest.json"
             and entry["git_sha1"] != blobs[path]}
    assert not wrong


def test_manifest_entries_are_well_formed(manifest):
    for name in ("files", "pinned_only"):
        for path, entry in manifest[name].items():
            assert path.startswith(("data/", "model/")), path
            assert set(entry) == {"size", "sha256", "git_sha1"}, path
            assert entry["size"] > 0
            assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), path
            assert re.fullmatch(r"[0-9a-f]{40}", entry["git_sha1"]), path


def test_ci_fetches_modern_training_evidence_inputs():
    """A clean CI checkout must run the real-data invariant, not skip it."""
    assert {
        "data/mg_pairs.json", "data/mg_lookup.json", "data/el_participles.json",
        "data/el_noun_heads.json", "data/el_verb_heads.json",
    } <= set(hf_data.TRACKED)


def test_ci_fetches_ancient_lexical_evidence():
    """The Ancient export must not fall back to inferred lookup heads."""
    assert "data/grc_lexical_heads.json" in hf_data.TRACKED


def test_ci_fetches_export_frequency_evidence_before_full_tests():
    assert "data/corpus_freq.json" in hf_data.TRACKED
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "test_export_creates_output_directory_before_audit" in workflow
    assert workflow.index("- name: Build and audit Ancient Greek Hunspell coverage") < workflow.index("- name: Run all tests")


def test_ci_provisions_and_checks_both_model_backends():
    """Fail model setup early instead of after the full dictionary tests."""
    assert {
        "model/combined-s3/vocab.json",
        "model/combined-s3/encoder.onnx",
        "model/combined-s3/decoder_step.onnx",
        "model/combined-s3/model.pt",
        "model/combined-s3/head_labels.json",
        "model/combined-s3/heads.npz",
    } <= set(hf_data.TRACKED)
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "onnxruntime" in workflow
    assert "python3 -m pytest tests/test_integrity.py" in workflow
    assert workflow.index("pytest tests/test_integrity.py") < workflow.index("- name: Run all tests")


def test_workflow_downloads_the_tracked_list_and_checks_it(manifest):
    """The workflow must take its file list from `hf_data.py files` and run the
    wait/verify steps - a hardcoded list would silently drift from the manifest,
    and skipping the wait brings back the upload/push race."""
    wf = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "scripts/hf_data.py files" in wf
    assert "scripts/hf_data.py wait" in wf
    assert "scripts/hf_data.py verify" in wf
    for path in manifest["files"]:
        assert f"hf download ciscoriordan/dilemma {path}" not in wf


def test_hashes_match_the_hub_encoding(tmp_path):
    """sha256 is the plain file digest and git_sha1 is git's blob id - the two
    forms the Hub reports for LFS/Xet and for small plain-git files."""
    import hashlib
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"dilemma")
    got = hf_data.file_hashes(sample)
    assert got["size"] == 7
    assert got["sha256"] == hashlib.sha256(b"dilemma").hexdigest()
    assert got["git_sha1"] == hashlib.sha1(b"blob 7\0dilemma").hexdigest()


def test_closed_class_companion_is_published_and_generated():
    assert "data/el_closed_class_heads.json" in hf_data.TRACKED
    assert "data/el_closed_class_heads.json" in (ROOT / ".gitignore").read_text()
    source = (ROOT / "build_data.py").read_text()
    assert 'DATA_DIR / "el_closed_class_heads.json"' in source


def test_ancient_reading_companion_is_published_and_generated():
    assert "data/grc_reading_heads.json" in hf_data.TRACKED
    assert "data/grc_reading_heads.json" in (ROOT / ".gitignore").read_text()
    source = (ROOT / "build_data.py").read_text()
    assert 'DATA_DIR / "grc_reading_heads.json"' in source


def test_adjective_companion_is_published_generated_and_downloaded():
    from fnmatch import fnmatch
    from dilemma import _download

    path = "data/el_adjective_heads.json"
    assert path in hf_data.TRACKED
    assert path in (ROOT / ".gitignore").read_text()
    assert 'DATA_DIR / "el_adjective_heads.json"' in (ROOT / "build_data.py").read_text()
    assert any(fnmatch(path, pattern) for pattern in _download.INCLUDES)
    assert not any(fnmatch(path, pattern) for pattern in _download.BASE_IGNORES)


# --- the files the runtime names --------------------------------------------

_FILE_NAME = re.compile(r"(?:data/)?[\w.*-]+\.(?:json|db|gz|npz|onnx|pt)")

# File names in the runtime package that the pinned lists do not hold, each
# with the reason. Anything else the package names has to be pinned.
_RUNTIME_NAMES_OUTSIDE_THE_MANIFEST = {
    # dilemma.paradigm reads these from $DILEMMA_PARADIGM_DATA, a directory
    # the caller supplies; they are not Dilemma data files.
    "jtauber_ag_paradigms.json": "external paradigm directory",
    "ag_noun_paradigms.json": "external paradigm directory",
    "dilemma_ag_verb_paradigms.json": "external paradigm directory",
    "dilemma_ag_noun_paradigms.json": "external paradigm directory",
    # Tagger weights: served under tagger/, which hf_data.UNPINNED lists.
    "tagger.onnx": "tagger weights",
    "tagger_labels.json": "tagger weights",
    "tokenizer.json": "tagger weights",
    "mwt.json": "tagger weights",
    "dilemma.onnx": "a logger name, not a file",
}

# Runtime names that only PINNED_ONLY holds, so CI does not download them.
# A table the default runtime path reads must be in TRACKED instead.
_RUNTIME_NAMES_CI_DOES_NOT_DOWNLOAD = {
    "ag_lookup.json": "fallback for a data directory without lookup.db",
    "med_lookup.json": "fallback for a data directory without lookup.db",
    "ag_verb_paradigms.json": "read from the external paradigm directory; "
                              "the Hub file of that name is a build output",
}


def _runtime_file_names() -> dict:
    """File names the runtime package mentions, with the modules naming them.

    Every string literal of `dilemma/` that is a file name as a whole; an
    f-string contributes its text with `*` for each substitution, so
    `f"{source}_headwords.json"` gives `*_headwords.json`. A name assembled
    in pieces is not seen.
    """
    names = {}
    for path in sorted((ROOT / "dilemma").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        inner = {id(part) for node in ast.walk(tree)
                 if isinstance(node, ast.JoinedStr) for part in node.values}
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                text = "".join(
                    part.value if isinstance(part, ast.Constant) else "*"
                    for part in node.values)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in inner:
                text = node.value
            else:
                continue
            if _FILE_NAME.fullmatch(text):
                names.setdefault(text.removeprefix("data/"), set()).add(
                    path.relative_to(ROOT).as_posix())
    return names


def _pinned_names(paths) -> set:
    """Base names under data/ and under the model directory CI tests."""
    return {Path(p).name for p in paths
            if p.startswith("data/") and p.count("/") == 1
            or p.startswith("model/combined-s3/")}


def test_the_scan_finds_the_runtime_tables():
    names = _runtime_file_names()
    assert {"lookup.db", "treebank_pos_lookup.json", "form_profile.db",
            "verb_subjects.json", "encoder.onnx", "*_headwords.json",
            "convention_*.json"} <= set(names)
    assert names["lookup.db"] == {"dilemma/core.py"}
    assert len(names) >= 50


def test_every_file_the_runtime_names_is_pinned():
    """A data file the runtime opens that is in neither list is what #140 was:
    installs read a Hub copy that no release vouches for."""
    tracked = _pinned_names(hf_data.TRACKED)
    pinned_only = _pinned_names(hf_data.PINNED_ONLY)
    unpinned, not_downloaded = {}, {}
    for name, modules in _runtime_file_names().items():
        if "*" in name or name in _RUNTIME_NAMES_OUTSIDE_THE_MANIFEST:
            continue
        if name in tracked:
            continue
        if name in pinned_only:
            if name not in _RUNTIME_NAMES_CI_DOES_NOT_DOWNLOAD:
                not_downloaded[name] = sorted(modules)
        else:
            unpinned[name] = sorted(modules)
    assert not unpinned, "in neither list of scripts/hf_data.py"
    assert not not_downloaded, "the runtime reads these: move them to TRACKED"


def test_every_file_a_runtime_pattern_can_reach_is_pinned():
    """`f"{source}_headwords.json"` and the like: every committed data file
    the pattern matches must be pinned, since the runtime may open it."""
    patterns = sorted(n for n in _runtime_file_names() if "*" in n)
    assert patterns == ["*_headwords.json", "convention_*.json"]
    tracked = _pinned_names(hf_data.TRACKED)
    committed = {Path(p).name for p in _committed_data_blobs()
                 if p.count("/") == 1}
    for pattern in patterns:
        assert any(fnmatch(name, pattern) for name in tracked), pattern
        loose = sorted(name for name in committed
                       if fnmatch(name, pattern) and name not in tracked)
        assert not loose, pattern


def test_the_exception_lists_are_current():
    names = set(_runtime_file_names())
    pinned = _pinned_names(hf_data.TRACKED + hf_data.PINNED_ONLY)
    for name in _RUNTIME_NAMES_OUTSIDE_THE_MANIFEST:
        assert name in names, f"{name} is no longer named by the runtime"
        assert name not in pinned, f"{name} is pinned now"
    for name in _RUNTIME_NAMES_CI_DOES_NOT_DOWNLOAD:
        assert name in names, f"{name} is no longer named by the runtime"
        assert name in _pinned_names(hf_data.PINNED_ONLY), name


# --- what the Hub serves ------------------------------------------------------

def test_served_prefixes_match_the_download_patterns():
    from dilemma import _download

    assert _download.REPO == _download.TAGGER_REPO == hf_data.REPO
    patterns = _download.INCLUDES + _download.TAGGER_INCLUDES
    assert all(p.endswith("/*") for p in patterns)
    assert tuple(p[:-1] for p in patterns) == hf_data.SERVED_PREFIXES


def test_a_served_file_without_a_pin_is_reported():
    served = [
        "data/lookup.db",                       # TRACKED
        "data/ag_lookup.json",                  # PINNED_ONLY
        "data/changes/glaux_name_lemmas.json",  # TRACKED, in a subdirectory
        "data/new_table.json",
        "data/changes/new_audit.json",
        "model/combined-s3/model.pt",           # TRACKED
        "model/new-model/model.pt",
        "data/kaikki_inputs/20261114/kaikki.org-en-dictionary-Greek.jsonl.gz",
        "tagger/grc/tagger.onnx",
        "training_ckpts/ag_dep/tagger_model.pt",  # not served by the download
        ".gitattributes",
    ]
    assert hf_data.unpinned_served(served) == [
        "data/changes/new_audit.json",
        "data/new_table.json",
        "model/new-model/model.pt",
    ]


def test_the_unpinned_prefixes_have_reasons_and_shadow_no_pin():
    for prefix, reason in hf_data.UNPINNED.items():
        assert prefix.startswith(hf_data.SERVED_PREFIXES) and prefix.endswith("/")
        assert len(reason) > 20
        assert not [p for p in hf_data.TRACKED + hf_data.PINNED_ONLY
                    if p.startswith(prefix)], prefix


# --- the subcommands, against an in-memory Hub --------------------------------

class _FakeHub:
    """The parts of huggingface_hub.HfApi that hf_data.py calls."""

    def __init__(self):
        self.blobs = {}
        self.uploads = []
        self.dates = {}

    def _info(self, path):
        data = self.blobs[path]
        # Large files come back with a sha256, small ones with a blob id only.
        lfs = SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) \
            if len(data) > 16 else None
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        date = self.dates.get(path)
        return SimpleNamespace(
            path=path, size=len(data), lfs=lfs, blob_id=blob,
            last_commit=SimpleNamespace(date=date) if date else None)

    def module(self):
        hub = self

        class HfApi:
            def get_paths_info(self, repo, paths, expand=False):
                assert repo == hf_data.REPO
                return [hub._info(p) for p in paths if p in hub.blobs]

            def list_repo_tree(self, repo, recursive=False):
                assert repo == hf_data.REPO and recursive
                folders = [SimpleNamespace(path="data"), SimpleNamespace(path="model")]
                return folders + [hub._info(p) for p in sorted(hub.blobs)]

            def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id,
                            commit_message):
                assert repo_id == hf_data.REPO
                hub.blobs[path_in_repo] = Path(path_or_fileobj).read_bytes()
                hub.uploads.append(path_in_repo)

        return SimpleNamespace(HfApi=HfApi)


_SMALL = b"small"
_LARGE = b"a table large enough for a sha256"


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A checkout with one CI artifact and three PINNED_ONLY files (one of
    them absent locally), all served by the fake Hub and pinned."""
    root = tmp_path / "checkout"
    (root / "data").mkdir(parents=True)
    hub = _FakeHub()
    monkeypatch.setattr(hf_data, "ROOT", root)
    monkeypatch.setattr(hf_data, "MANIFEST_PATH", root / "data" / "hf_manifest.json")
    monkeypatch.setattr(hf_data, "TRACKED", ["data/lookup.db"])
    monkeypatch.setattr(hf_data, "PINNED_ONLY", [
        "data/ag_lookup.json", "data/med_pos_lookup.json", "data/glaux_freq.json"])
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
    monkeypatch.setattr(hf_data.time, "sleep", lambda seconds: None)
    content = {
        "data/lookup.db": _LARGE + b" 1",
        "data/ag_lookup.json": _LARGE + b" 2",
        "data/med_pos_lookup.json": _SMALL,
        "data/glaux_freq.json": _LARGE + b" 3",
    }
    for rel, data in content.items():
        (root / rel).write_bytes(data)
        hub.blobs[rel] = data
    assert hf_data.main(["manifest"]) == 0
    (root / "data" / "glaux_freq.json").unlink()   # a checkout without it
    return SimpleNamespace(root=root, hub=hub)


def _manifest(tree) -> dict:
    return json.loads((tree.root / "data" / "hf_manifest.json").read_text())


def test_manifest_command_keeps_the_pin_of_an_absent_pinned_only_file(tree):
    before = _manifest(tree)
    assert list(before["files"]) == ["data/lookup.db"]
    assert list(before["pinned_only"]) == hf_data.PINNED_ONLY
    (tree.root / "data" / "ag_lookup.json").write_bytes(_LARGE + b" rebuilt")
    assert hf_data.main(["manifest"]) == 0
    after = _manifest(tree)
    assert after["pinned_only"]["data/ag_lookup.json"]["sha256"] == \
        hashlib.sha256(_LARGE + b" rebuilt").hexdigest()
    assert after["pinned_only"]["data/glaux_freq.json"] == \
        before["pinned_only"]["data/glaux_freq.json"]
    assert after["files"] == before["files"]


def test_check_passes_when_checkout_manifest_and_hub_agree(tree, capsys):
    assert hf_data.main(["check"]) == 0
    assert "4 artifacts, no served file without a pin" in capsys.readouterr().out


@pytest.mark.parametrize("rel", ["data/lookup.db", "data/ag_lookup.json",
                                 "data/med_pos_lookup.json"])
def test_check_fails_on_a_rebuilt_file_of_either_list(tree, capsys, rel):
    (tree.root / rel).write_bytes(_LARGE + b" rebuilt")
    assert hf_data.main(["check"]) == 1
    assert f"{rel} differs from the manifest" in capsys.readouterr().err


@pytest.mark.parametrize("rel", ["data/lookup.db", "data/ag_lookup.json",
                                 "data/med_pos_lookup.json", "data/glaux_freq.json"])
def test_check_and_wait_fail_when_the_hub_serves_other_bytes(tree, capsys, rel):
    tree.hub.blobs[rel] = b"an older upload of this file"
    assert hf_data.main(["check"]) == 1
    assert f"{rel}: the Hub is not serving the manifest's bytes yet" \
        in capsys.readouterr().err
    assert hf_data.main(["wait", "--timeout", "0", "--interval", "0"]) == 1
    assert f"waiting for {rel}" in capsys.readouterr().out


def test_check_fails_on_a_served_file_without_a_pin(tree, capsys):
    tree.hub.blobs["data/uploaded_by_hand.json"] = _LARGE
    tree.hub.blobs["data/kaikki_inputs/20261114/dump.jsonl.gz"] = _LARGE
    tree.hub.blobs["tagger/grc/tagger.onnx"] = _LARGE
    tree.hub.blobs["training_ckpts/run/model.pt"] = _LARGE
    assert hf_data.main(["check"]) == 1
    err = capsys.readouterr().err
    assert "data/uploaded_by_hand.json: served by the Hub without a pin" in err
    assert "kaikki_inputs" not in err and "tagger" not in err
    assert "training_ckpts" not in err


def test_check_fails_when_a_listed_file_has_no_pin(tree, monkeypatch, capsys):
    monkeypatch.setattr(hf_data, "PINNED_ONLY",
                        hf_data.PINNED_ONLY + ["data/lbg_pairs.json"])
    assert hf_data.main(["check"]) == 1
    assert "manifest is missing data/lbg_pairs.json" in capsys.readouterr().err


def test_wait_passes_when_the_hub_serves_both_lists(tree, capsys):
    assert hf_data.main(["wait", "--timeout", "0"]) == 0
    assert "Hub matches the manifest (4 artifacts)" in capsys.readouterr().out


def test_verify_reads_only_what_ci_downloads(tree, capsys):
    (tree.root / "data" / "ag_lookup.json").unlink()
    (tree.root / "data" / "med_pos_lookup.json").write_bytes(b"other bytes")
    assert hf_data.main(["verify"]) == 0
    assert "verified 1 artifacts" in capsys.readouterr().out
    (tree.root / "data" / "lookup.db").write_bytes(b"other bytes")
    assert hf_data.main(["verify"]) == 1


def test_push_uploads_a_changed_file_of_either_list_and_rewrites_its_pin(tree):
    before = _manifest(tree)
    rebuilt = {"data/lookup.db": _LARGE + b" new lookup",
               "data/med_pos_lookup.json": b"tiny2"}
    for rel, data in rebuilt.items():
        (tree.root / rel).write_bytes(data)
    assert hf_data.main(["push"]) == 0
    assert tree.hub.uploads == ["data/lookup.db", "data/med_pos_lookup.json"]
    after = _manifest(tree)
    assert after["files"]["data/lookup.db"]["sha256"] == \
        hashlib.sha256(rebuilt["data/lookup.db"]).hexdigest()
    assert after["pinned_only"]["data/med_pos_lookup.json"]["sha256"] == \
        hashlib.sha256(rebuilt["data/med_pos_lookup.json"]).hexdigest()
    # Untouched and absent files keep their pins, in list order.
    assert list(after["pinned_only"]) == hf_data.PINNED_ONLY
    for rel in ("data/ag_lookup.json", "data/glaux_freq.json"):
        assert after["pinned_only"][rel] == before["pinned_only"][rel]
    assert hf_data.main(["check"]) == 0
    assert hf_data.main(["wait", "--timeout", "0"]) == 0


def test_push_of_named_paths_takes_either_list_and_nothing_else(tree):
    (tree.root / "data" / "ag_lookup.json").write_bytes(_LARGE + b" rebuilt")
    (tree.root / "data" / "lookup.db").write_bytes(_LARGE + b" rebuilt too")
    assert hf_data.main(["push", "data/ag_lookup.json"]) == 0
    assert tree.hub.uploads == ["data/ag_lookup.json"]
    with pytest.raises(SystemExit, match="not tracked by the manifest"):
        hf_data.main(["push", "data/uploaded_by_hand.json"])
    with pytest.raises(SystemExit, match="not found locally"):
        hf_data.main(["push", "data/glaux_freq.json"])
    assert tree.hub.uploads == ["data/ag_lookup.json"]


def test_push_dry_run_uploads_nothing_and_leaves_the_manifest(tree, capsys):
    before = (tree.root / "data" / "hf_manifest.json").read_bytes()
    (tree.root / "data" / "ag_lookup.json").write_bytes(_LARGE + b" rebuilt")
    assert hf_data.main(["push", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "^ data/ag_lookup.json: would upload (0 MB), would rewrite the pin" in out
    assert "= data/lookup.db" in out and "= data/med_pos_lookup.json" in out
    assert "glaux_freq" not in out
    assert tree.hub.uploads == []
    assert (tree.root / "data" / "hf_manifest.json").read_bytes() == before


def test_push_moves_a_pin_between_the_lists(tree, monkeypatch):
    """A file that changes list keeps its pin: push reads both lists."""
    before = _manifest(tree)
    monkeypatch.setattr(hf_data, "TRACKED",
                        ["data/lookup.db", "data/med_pos_lookup.json"])
    monkeypatch.setattr(hf_data, "PINNED_ONLY",
                        ["data/ag_lookup.json", "data/glaux_freq.json"])
    assert hf_data.main(["push"]) == 0
    assert tree.hub.uploads == []
    after = _manifest(tree)
    assert list(after["files"]) == hf_data.TRACKED
    assert list(after["pinned_only"]) == hf_data.PINNED_ONLY
    assert after["files"]["data/med_pos_lookup.json"] == \
        before["pinned_only"]["data/med_pos_lookup.json"]

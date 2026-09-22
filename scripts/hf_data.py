#!/usr/bin/env python3
"""Keep the HuggingFace data artifacts and this git repo in lockstep.

The big lookup artifacts live on the Hub instead of in git, so the two
histories can drift apart. CI downloads them at test time, which means a
`git push` landing while `hf upload` is still running makes CI run new tests
against stale data. That is what broke the 2026-07-13 run: the tests wanted
`pedalion` in lemma_attestation's `_meta.sources`, the Hub was still serving
the copy without it, and the upload finished 80 seconds after CI had already
downloaded.

`data/hf_manifest.json` pins the exact sha256 of every artifact CI downloads.
CI waits for the Hub to serve those bytes before downloading and re-checks the
bytes afterwards, so the race resolves itself instead of failing; the pre-push
hook (scripts/install_git_hooks.sh) refuses a push whose manifest the Hub
cannot satisfy yet.

    python scripts/hf_data.py files      # paths CI downloads (feeds the workflow)
    python scripts/hf_data.py manifest   # rewrite the manifest from local files
    python scripts/hf_data.py push       # upload changed artifacts, rewrite the manifest
    python scripts/hf_data.py check      # local vs manifest vs Hub (pre-push hook)
    python scripts/hf_data.py wait       # CI: block until the Hub serves the manifest
    python scripts/hf_data.py verify     # CI: downloaded bytes vs the manifest
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "hf_manifest.json"

# Mirrors dilemma/_download.py:REPO (the public model repo).
REPO = "ciscoriordan/dilemma"

# Exactly the artifacts .github/workflows/test.yml downloads - the workflow
# gets them from `hf_data.py files`, so this list is the single source of
# truth. Add a file here when a test starts asserting on it.
TRACKED = [
    "data/lookup.db",
    "data/spell_index.db",
    "data/lemma_attestation.json",
    # Exact-form attestation for the grc Hunspell export and its audit.
    "data/form_profile.db",
]

_CHUNK = 1 << 20


# --- hashing ---------------------------------------------------------------

def file_hashes(path: Path) -> dict:
    """sha256 (what the Hub reports for LFS/Xet blobs) + git's blob sha1.

    Small files are stored as plain git objects on the Hub and expose only a
    blob id, so both are recorded and whichever the Hub gives is compared.
    """
    size = path.stat().st_size
    sha = hashlib.sha256()
    blob = hashlib.sha1(b"blob %d\0" % size)
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            sha.update(chunk)
            blob.update(chunk)
    return {"size": size, "sha256": sha.hexdigest(), "git_sha1": blob.hexdigest()}


# --- manifest --------------------------------------------------------------

def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        sys.exit(f"{MANIFEST_PATH} is missing; run: python scripts/hf_data.py manifest")
    with open(MANIFEST_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def manifest_entries() -> dict:
    return load_manifest()["files"]


def write_manifest(files: dict, *, previous: dict | None = None) -> bool:
    """Write the manifest; keep `generated` when nothing actually changed."""
    old = previous if previous is not None else (
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if MANIFEST_PATH.exists() else {}
    )
    if old.get("files") == files and old.get("repo") == REPO:
        return False
    doc = {
        "_comment": (
            "sha256 of every artifact CI downloads from the Hub. Rewritten by "
            "scripts/hf_data.py push; CI waits for the Hub to serve these exact "
            "bytes (hf_data.py wait) before running tests."
        ),
        "repo": REPO,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
    }
    MANIFEST_PATH.write_text(
        json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return True


def local_entries(paths: list[str]) -> dict:
    out = {}
    for rel in paths:
        path = ROOT / rel
        if not path.exists():
            sys.exit(f"{rel} not found locally; build or download it first")
        out[rel] = file_hashes(path)
    return out


# --- Hub -------------------------------------------------------------------

def hub_entries(paths: list[str]) -> dict:
    """path -> {size, sha256|None, git_sha1, last_commit}; missing paths omitted."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")
    infos = HfApi().get_paths_info(REPO, paths=paths, expand=True)
    out = {}
    for info in infos:
        if not hasattr(info, "size"):  # a folder
            continue
        out[info.path] = {
            "size": info.size,
            "sha256": info.lfs.sha256 if info.lfs else None,
            "git_sha1": info.blob_id,
            "last_commit": info.last_commit.date if info.last_commit else None,
        }
    return out


def hub_matches(want: dict, have: dict | None) -> bool:
    """Does the Hub's copy carry the manifest's bytes?"""
    if have is None:
        return False
    if have["sha256"]:
        return have["sha256"] == want["sha256"]
    return have["git_sha1"] == want["git_sha1"]


# --- commands --------------------------------------------------------------

def cmd_files(args) -> int:
    print(" ".join(TRACKED))
    return 0


def cmd_manifest(args) -> int:
    changed = write_manifest(local_entries(TRACKED))
    print(f"{MANIFEST_PATH.relative_to(ROOT)}: "
          f"{'updated' if changed else 'already up to date'}")
    return 0


def cmd_push(args) -> int:
    paths = args.paths or TRACKED
    unknown = [p for p in paths if p not in TRACKED]
    if unknown:
        sys.exit(f"not tracked by the manifest: {', '.join(unknown)}\n"
                 f"add it to TRACKED in {Path(__file__).name} first")
    local = local_entries(paths)
    hub = hub_entries(paths)
    from huggingface_hub import HfApi
    api = HfApi()
    for rel in paths:
        if hub_matches(local[rel], hub.get(rel)):
            print(f"  = {rel} (Hub already has these bytes)")
            continue
        print(f"  ^ {rel} ({local[rel]['size'] / 1e6:.0f} MB) uploading...")
        api.upload_file(
            path_or_fileobj=str(ROOT / rel),
            path_in_repo=rel,
            repo_id=REPO,
            commit_message=f"Upload {rel}",
        )
    files = manifest_entries() if MANIFEST_PATH.exists() else {}
    files.update(local)
    files = {rel: files[rel] for rel in TRACKED if rel in files}
    changed = write_manifest(files)
    print(f"{MANIFEST_PATH.relative_to(ROOT)}: "
          f"{'updated - commit it with the code change' if changed else 'unchanged'}")
    return 0


def cmd_check(args) -> int:
    want = manifest_entries()
    missing = [rel for rel in TRACKED if rel not in want]
    if missing:
        print(f"manifest is missing {', '.join(missing)}; run: "
              f"python scripts/hf_data.py push", file=sys.stderr)
        return 1

    problems = []
    for rel, entry in want.items():
        path = ROOT / rel
        if not path.exists():
            continue  # not every checkout carries the data files
        if path.stat().st_size != entry["size"] or \
                file_hashes(path)["sha256"] != entry["sha256"]:
            problems.append(f"{rel} differs from the manifest (rebuilt but not shipped)")
    if problems:
        for p in problems:
            print(f"  ! {p}", file=sys.stderr)
        print("run: python scripts/hf_data.py push", file=sys.stderr)
        return 1

    try:
        hub = hub_entries(list(want))
    except Exception as e:  # offline, rate limited, no token: don't block work
        print(f"  ? could not reach the Hub ({e}); skipping the remote check")
        return 0
    stale = [rel for rel, entry in want.items() if not hub_matches(entry, hub.get(rel))]
    if stale:
        for rel in stale:
            print(f"  ! {rel}: the Hub is not serving the manifest's bytes yet",
                  file=sys.stderr)
        print("CI downloads these at test time, so pushing now tests stale data.\n"
              "run: python scripts/hf_data.py push", file=sys.stderr)
        return 1
    print(f"  = Hub matches the manifest ({len(want)} artifacts)")
    return 0


def cmd_wait(args) -> int:
    """Block until the Hub serves the manifest's bytes (CI's first step)."""
    want = manifest_entries()
    deadline = time.monotonic() + args.timeout
    generated = _parse_ts(load_manifest().get("generated"))
    while True:
        hub = hub_entries(list(want))
        pending = {rel: e for rel, e in want.items() if not hub_matches(e, hub.get(rel))}
        if not pending:
            print(f"Hub matches the manifest ({len(want)} artifacts)")
            return 0
        for rel, entry in pending.items():
            have = hub.get(rel)
            if have is None:
                got = "nothing"
            else:
                got = (have["sha256"] or have["git_sha1"])[:12]
                if have["last_commit"]:
                    got += have["last_commit"].strftime(" from %Y-%m-%d %H:%MZ")
            print(f"  waiting for {rel}: want sha256 {entry['sha256'][:12]}, "
                  f"Hub has {got}")
            # The Hub moving PAST the manifest is not a race, it is a stale
            # manifest - waiting cannot fix it, so say so and stop.
            if have and generated and have["last_commit"] and \
                    have["last_commit"] > generated:
                print(f"\n{rel} on the Hub is newer than the manifest "
                      f"(generated {generated:%Y-%m-%d %H:%M}Z). The manifest was not "
                      f"refreshed after the upload:\n"
                      f"  python scripts/hf_data.py manifest   # then commit it",
                      file=sys.stderr)
                return 1
        if time.monotonic() >= deadline:
            print(f"\nGave up after {args.timeout}s. The upload for the artifacts "
                  f"above never landed; finish it with:\n"
                  f"  python scripts/hf_data.py push", file=sys.stderr)
            return 1
        time.sleep(args.interval)


def cmd_verify(args) -> int:
    """Check downloaded bytes against the manifest (CI's post-download step)."""
    want = manifest_entries()
    bad = []
    for rel, entry in want.items():
        path = ROOT / rel
        if not path.exists():
            bad.append(f"{rel}: not downloaded")
            continue
        got = file_hashes(path)
        if got["sha256"] != entry["sha256"]:
            bad.append(f"{rel}: sha256 {got['sha256'][:12]}, "
                       f"manifest wants {entry['sha256'][:12]}")
    for line in bad:
        print(f"  ! {line}", file=sys.stderr)
    if bad:
        return 1
    print(f"verified {len(want)} artifacts against the manifest")
    return 0


def _parse_ts(value):
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="hf_data.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("files", help="print the paths CI downloads").set_defaults(fn=cmd_files)
    sub.add_parser("manifest", help="rewrite the manifest from local files").set_defaults(fn=cmd_manifest)

    p_push = sub.add_parser("push", help="upload changed artifacts, then rewrite the manifest")
    p_push.add_argument("paths", nargs="*", help=f"default: all of {', '.join(TRACKED)}")
    p_push.set_defaults(fn=cmd_push)

    sub.add_parser("check", help="local vs manifest vs Hub").set_defaults(fn=cmd_check)

    p_wait = sub.add_parser("wait", help="block until the Hub serves the manifest")
    p_wait.add_argument("--timeout", type=int, default=900, help="seconds (default 900)")
    p_wait.add_argument("--interval", type=int, default=20, help="poll seconds (default 20)")
    p_wait.set_defaults(fn=cmd_wait)

    sub.add_parser("verify", help="check downloaded bytes against the manifest").set_defaults(fn=cmd_verify)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

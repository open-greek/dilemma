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
    python scripts/hf_data.py push -n    # say what push would upload and re-pin
    python scripts/hf_data.py check      # local vs manifest vs Hub (pre-push hook)
    python scripts/hf_data.py wait       # CI: block until the Hub serves the manifest
    python scripts/hf_data.py verify     # CI: downloaded bytes vs the manifest

`python -m dilemma download` serves every file under data/ and model/, not
only what CI downloads, so the manifest has a second list (`pinned_only`, the
PINNED_ONLY list below) for the files that builds and tools read: `push`,
`check` and `wait` cover it, `files` and `verify` do not, because CI does not
download those files. `check` also lists what the Hub serves and fails on a
file that is in neither list (#140).
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
# truth, including provenance that accompanies the generated tables.
# Add a file here when CI needs to receive the same version as the code.
TRACKED = [
    "data/lookup.db",
    "data/spell_index.db",
    "data/lemma_attestation.json",
    # Exact-form attestation for the grc Hunspell export and its audit.
    "data/form_profile.db",
    # Exact source loci share the profile's form IDs and content hash.
    "data/form_citations.db",
    # Merged corpus counts used by the Ancient export's admission filter.
    "data/corpus_freq.json",
    # Source cells preserve elision used by the Ancient alias recovery.
    "data/diorisis_pairs.json",
    # Reversible quotation evidence consumed by the Ancient export gate.
    "data/changes/diorisis_token_normalizations.json",
    # Reversible corrections shared by GLAUx lemma consumers.
    "data/changes/glaux_name_lemmas.json",
    # Reversible, source-backed completion of sparse Modern paradigms.
    "data/changes/el_paradigm_completion.json",
    "data/changes/el_present_completion.json",
    # Source-scoped tense/lemma repairs and independently anchored past cells.
    "data/el_verb_repair_sources.json.gz",
    "data/changes/el_verb_data_repairs.json",
    "data/changes/el_imperfect_completion.json",
    "data/changes/el_verb_pos_repairs.json",
    "data/changes/el_source_tense_corrections.json",
    # Input-pinned provenance for the generated noun expansion layer.
    "data/ag_noun_generation_audit.json",
    "data/ag_sophocles_noun_generation_audit.json",
    # Base tables checked by the real Modern Greek training-evidence test.
    "data/mg_pairs.json",
    "data/mg_lookup.json",
    "data/mg_pos_lookup.json",
    # The generator delta carried onto the Modern Greek snapshot, and what it
    # changed there (scripts/refresh_el_generator_delta.py).
    "data/changes/el_generator_refresh.json",
    "data/changes/el_generator_refresh_audit.json",
    # Source-backed participial cells used by the Modern Greek exporter.
    "data/el_participles.json",
    # Independent noun heads retained beside flat-lookup homographs.
    "data/el_noun_heads.json",
    # Defined adjectives retained beside noun/verb flat homographs.
    "data/el_adjective_heads.json",
    # Independent finite verb heads selected by the source reference graph.
    "data/el_verb_heads.json",
    # Expanded lexical POS coverage exceeds GitHub's regular-file size limit.
    "data/lsj9_pos_lookup.json",
    # POS tables the runtime loads. Git-tracked too, but the wheel carries
    # no data: installs read the Hub copies, so a release has to serve the
    # bytes of its own commit (#140).
    "data/ag_pos_lookup.json",
    "data/glaux_pos_lookup.json",
    "data/treebank_pos_lookup.json",
    # Wiktionary-derived Ancient headwords, for consumers of the downloaded
    # data that validate lemmas without a source checkout.
    "data/ag_headwords.json",
    # Headword lists and small tables the runtime reads. The wheel carries
    # no data, so an install that has only the downloaded files needs them
    # here: without the lists a lemma convention loads an empty headword set.
    "data/cunliffe_headwords.json",
    "data/dge_headwords.json",
    "data/lbg_headwords.json",
    "data/lemma_equivalences.json",
    "data/lgpn_names.json",
    "data/lsj10_headwords.json",
    "data/lsj9_frequency.json",
    "data/lsj9_indeclinables.json",
    "data/lsj_headwords.json",
    "data/mg_headwords.json",
    "data/pd_headwords.json",
    "data/triantafyllidis_headwords.json",
    "data/vlg_headwords.json",
    "data/wip_headwords.json",
    # Lemma-convention remaps (convention="lsj", "cunliffe", "triantafyllidis").
    "data/convention_cunliffe.json",
    "data/convention_lsj.json",
    "data/convention_triantafyllidis.json",
    # Independent lexical heads for sparse Ancient corpus admission.
    "data/grc_lexical_heads.json",
    # Explicit citation references consumed by sparse-form export admission.
    "data/grc_contracted_verb_aliases.json",
    # Exact POS-specific source evidence for non-flat Modern morphology.
    "data/el_closed_class_heads.json",
    # Exact POS-specific definitions for Ancient non-flat training readings.
    "data/grc_reading_heads.json",
    # Regular Ancient adverbs in -ως and their adjectives (issue #117).
    "data/grc_adverb_heads.json",
    # Reviewed POS-table cell corrections the runtime applies at load (#106).
    # Git-tracked too, but the wheel carries no data: installs get it here.
    "data/pos_cell_corrections.json",
    # Explicit, sense-scoped dictionary evidence for optional verb exports.
    "data/verb_subjects.json",
    # Immutable source copies for reproducible monotonic corpus extraction.
    "data/el_lm_inputs/tatoeba_ell_sentences.tsv.bz2",
    "data/el_lm_inputs/wiktionary_examples.jsonl.gz",
    "data/el_lm_inputs/provenance.json",
    # Source-merged capital name owners and exact corpus target audit.
    "data/treebank_pos_lookup.audit.json",
    # Integrity checks exercise both exported and PyTorch model backends.
    "model/combined-s3/vocab.json",
    "model/combined-s3/encoder.onnx",
    "model/combined-s3/decoder_step.onnx",
    "model/combined-s3/model.pt",
    "model/combined-s3/head_labels.json",
    "model/combined-s3/heads.npz",
]

# Pinned like TRACKED, but not downloaded by CI. `python -m dilemma download`
# serves these with the rest of data/ and model/, and a build, a tool or a
# full install reads them, so a release has to say which bytes are its own
# (#140). They are pinned under `pinned_only`, uploaded by `push` when the
# checkout has them and compared with the Hub by `check` and `wait`; `files`
# and `verify` leave them out. A table the default runtime path or a test
# reads belongs in TRACKED instead.
PINNED_ONLY = [
    # Tables of the Ancient and Medieval build chain. The two lookups are
    # also what the runtime falls back to in a data directory without
    # lookup.db.
    "data/ag_lookup.json",
    "data/ag_pairs.json",
    "data/med_lookup.json",
    "data/med_pairs.json",
    "data/ag_source_alias_audit.json",
    "data/ag_source_alias_manifest.json",
    # Written by build_data.py beside the two tables above and tracked in
    # git. No code reads it; it is pinned because the Hub serves it.
    "data/med_pos_lookup.json",
    # Corpus pairs read by the lookup build, the LSJ overlay and the treebank
    # POS table.
    "data/glaux_pairs.json",
    "data/lbg_pairs.json",
    "data/nt_pairs.json",
    "data/ptnk_pairs.json",
    "data/tagnt_pairs.json",
    "data/pedalion_pairs.json",
    # Per-source form frequencies that build/merge_corpus_freq.py merges into
    # corpus_freq.json.
    "data/glaux_freq.json",
    "data/diorisis_freq.json",
    "data/pta_freq.json",
    "data/cog_public_freq.json",
    # Older per-source frequencies that cog_public_freq.json replaced. No
    # code reads them; pinned while the Hub still serves them.
    "data/first1kgreek_freq.json",
    "data/canonical_greeklit_freq.json",
    # Per-lemma Ancient verb paradigms (build/build_grc_verb_paradigms.py).
    "data/ag_verb_paradigms.json",
    # Index of the Herculaneum word search (vesuvius.py).
    "data/vesuvius_index.json.gz",
    # The Ancient-only model. A full download has it, and lang="grc" then
    # loads it in place of combined-s3; CI downloads and tests combined-s3.
    "model/grc-s3/vocab.json",
    "model/grc-s3/encoder.onnx",
    "model/grc-s3/decoder_step.onnx",
    "model/grc-s3/model.pt",
    "model/grc-s3/head_labels.json",
    "model/grc-s3/heads.npz",
]

# What `python -m dilemma download` fetches from the repository: the include
# patterns of dilemma/_download.py (tests/test_hf_manifest.py keeps the two
# in step).
SERVED_PREFIXES = ("data/", "model/", "tagger/")

# Served paths this manifest does not pin, each with its reason. `check`
# fails on any other served file that is in neither list above, so a file
# uploaded outside `push` cannot stay unpinned unnoticed (#140).
UNPINNED = {
    "data/kaikki_inputs/": "dated source archives, pinned by Hub revision and "
                           "sha256 in data/kaikki_inputs.json",
    "tagger/": "tagger weights, fetched by a separate download step into "
               "another directory layout; not pinned here",
}

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


def pinned_only_entries() -> dict:
    """The pins CI does not download (`pinned_only`); empty in an older
    manifest."""
    if not MANIFEST_PATH.exists():
        return {}
    return load_manifest().get("pinned_only", {})


def pinned_entries() -> dict:
    """Everything the Hub must serve: both lists of the manifest."""
    return {**manifest_entries(), **pinned_only_entries()}


def local_pinned_only_entries() -> dict:
    """Pins of the PINNED_ONLY files: the local file where present, else the
    pin the manifest already has (few checkouts carry all of them)."""
    old = pinned_only_entries()
    out = {}
    for rel in PINNED_ONLY:
        path = ROOT / rel
        if path.exists():
            out[rel] = file_hashes(path)
        elif rel in old:
            out[rel] = old[rel]
    return out


def write_manifest(files: dict, pinned_only: dict | None = None, *,
                   previous: dict | None = None) -> bool:
    """Write the manifest; keep `generated` when nothing actually changed.

    `pinned_only=None` keeps the pins of that list as the manifest has them.
    """
    old = previous if previous is not None else (
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if MANIFEST_PATH.exists() else {}
    )
    if pinned_only is None:
        pinned_only = old.get("pinned_only", {})
    if old.get("files") == files and old.get("repo") == REPO \
            and old.get("pinned_only", {}) == pinned_only:
        return False
    doc = {
        "_comment": (
            "sha256 of every artifact CI downloads from the Hub (files) and of "
            "the other files the Hub serves that builds and tools read "
            "(pinned_only). Rewritten by scripts/hf_data.py push; CI waits for "
            "the Hub to serve these exact bytes (hf_data.py wait) before "
            "running tests."
        ),
        "repo": REPO,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
        "pinned_only": pinned_only,
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


def hub_served_paths() -> list[str]:
    """Every file of the repository that `python -m dilemma download` serves."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")
    return sorted(
        entry.path for entry in HfApi().list_repo_tree(REPO, recursive=True)
        if hasattr(entry, "size") and entry.path.startswith(SERVED_PREFIXES))


def unpinned_served(served, pinned=None) -> list[str]:
    """Served paths with no pin: in neither list and under no UNPINNED prefix."""
    if pinned is None:
        pinned = set(TRACKED) | set(PINNED_ONLY)
    return sorted(
        path for path in served
        if path.startswith(SERVED_PREFIXES) and path not in pinned
        and not path.startswith(tuple(UNPINNED)))


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
    changed = write_manifest(local_entries(TRACKED), local_pinned_only_entries())
    print(f"{MANIFEST_PATH.relative_to(ROOT)}: "
          f"{'updated' if changed else 'already up to date'}")
    return 0


def cmd_push(args) -> int:
    # A PINNED_ONLY file is uploaded when the checkout has it; a checkout
    # that rebuilt only the runtime artifacts keeps the pins of the rest.
    paths = args.paths or TRACKED + [
        rel for rel in PINNED_ONLY if (ROOT / rel).exists()]
    unknown = [p for p in paths if p not in TRACKED and p not in PINNED_ONLY]
    if unknown:
        sys.exit(f"not tracked by the manifest: {', '.join(unknown)}\n"
                 f"add it to TRACKED or PINNED_ONLY in {Path(__file__).name} first")
    local = local_entries(paths)
    hub = hub_entries(paths)
    old = pinned_entries() if MANIFEST_PATH.exists() else {}
    if args.dry_run:
        for rel in paths:
            notes = []
            if not hub_matches(local[rel], hub.get(rel)):
                notes.append(f"would upload ({local[rel]['size'] / 1e6:.0f} MB)")
            if old.get(rel) != local[rel]:
                notes.append("would rewrite the pin")
            print(f"  {'^' if notes else '='} {rel}"
                  + (": " + ", ".join(notes) if notes else ""))
        print("dry run: nothing uploaded, manifest not written")
        return 0
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
    old.update(local)
    files = {rel: old[rel] for rel in TRACKED if rel in old}
    pinned_only = {rel: old[rel] for rel in PINNED_ONLY if rel in old}
    changed = write_manifest(files, pinned_only)
    print(f"{MANIFEST_PATH.relative_to(ROOT)}: "
          f"{'updated - commit it with the code change' if changed else 'unchanged'}")
    return 0


def cmd_check(args) -> int:
    want = pinned_entries()
    missing = [rel for rel in TRACKED + PINNED_ONLY if rel not in want]
    if missing:
        print(f"manifest is missing {', '.join(missing)}; run: "
              f"python scripts/hf_data.py push\n"
              f"(from a checkout that has the files: push pins what it finds)",
              file=sys.stderr)
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
        served = hub_served_paths()
    except Exception as e:  # offline, rate limited, no token: don't block work
        print(f"  ? could not reach the Hub ({e}); skipping the remote check")
        return 0
    stale = [rel for rel, entry in want.items() if not hub_matches(entry, hub.get(rel))]
    if stale:
        for rel in stale:
            print(f"  ! {rel}: the Hub is not serving the manifest's bytes yet",
                  file=sys.stderr)
        print("CI and installs download from the Hub, so pushing now ships stale data.\n"
              "run: python scripts/hf_data.py push", file=sys.stderr)
        return 1
    unpinned = unpinned_served(served)
    if unpinned:
        for rel in unpinned:
            print(f"  ! {rel}: served by the Hub without a pin", file=sys.stderr)
        print(f"`python -m dilemma download` hands these to every install. Add each "
              f"to TRACKED\n(CI downloads it) or PINNED_ONLY in "
              f"{Path(__file__).name} and run push, or delete it\nfrom the Hub.",
              file=sys.stderr)
        return 1
    print(f"  = Hub matches the manifest ({len(want)} artifacts, "
          f"no served file without a pin)")
    return 0


def cmd_wait(args) -> int:
    """Block until the Hub serves the manifest's bytes (CI's first step)."""
    want = pinned_entries()
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
    p_push.add_argument("paths", nargs="*",
                        help="default: every TRACKED artifact and every "
                             "PINNED_ONLY file the checkout has")
    p_push.add_argument("-n", "--dry-run", action="store_true",
                        help="say what would be uploaded and re-pinned; "
                             "upload nothing and leave the manifest alone")
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

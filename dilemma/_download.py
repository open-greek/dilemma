"""Download Dilemma data files from HuggingFace Hub.

    python -m dilemma download                  # everything (lemma + tagger weights)
    python -m dilemma download --dir /some/path
    python -m dilemma download --no-tagger      # lemma data only
    python -m dilemma download --only-tagger    # tagger weights only

Both lemma data and tagger weights live at
https://huggingface.co/ciscoriordan/dilemma. The base download is ~5.5 GB
(lemma `data/` ~4.5 GB + `model/` ~0.07 GB + tagger weights ~1.0 GB); the
two opt-in form-attestation DBs add ~1.2 GB more (~6.7 GB for everything).
The tagger weights are under the `tagger/` prefix and the lemma artifacts
are under `data/` and `model/`.
"""

import argparse
import sys
from pathlib import Path

DEFAULT_CACHE = Path.home() / ".cache" / "dilemma"
REPO = "ciscoriordan/dilemma"
INCLUDES = ["data/*", "model/*"]

# The form-attestation DBs are large and only needed for the `attested only`
# gate / form_attestation(). Both are kept OUT of the base download; fetch them
# explicitly. form_profile.db (the gate + usage distribution) comes with
# --with-attestation; the larger form_citations.db (example loci) needs
# --with-citations (which implies the profile).
PROFILE_FILE = "data/form_profile.db"
CITATIONS_FILE = "data/form_citations.db"
ATTEST_FILES = [PROFILE_FILE, CITATIONS_FILE]

TAGGER_REPO = "ciscoriordan/dilemma"
TAGGER_INCLUDES = ["tagger/*"]


def _enable_fast_transfer() -> None:
    """Opt in to the fastest transfer backend available.

    Modern huggingface_hub downloads Xet-backed repos (this repo is fully
    Xet-backed) through hf_xet, but with polite default concurrency;
    HF_XET_HIGH_PERFORMANCE=1 lifts it to saturate the connection. Older
    hub versions instead use the hf_transfer Rust downloader, gated behind
    HF_HUB_ENABLE_HF_TRANSFER, which errors when the package is missing -
    so it is only set when importable. setdefault keeps any explicit user
    setting (including an opt-out of 0) in charge.
    """
    import importlib.util
    import os

    if importlib.util.find_spec("hf_xet"):
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    elif importlib.util.find_spec("hf_transfer"):
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _snapshot_download(*, repo_id, local_dir, allow_patterns, ignore_patterns=None):
    _enable_fast_transfer()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download Dilemma data. "
            "Install with: pip install huggingface_hub"
        ) from e
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )


def download(target_dir: Path | None = None, *, tagger: bool = True,
             dilemma: bool = True, attestation: bool = False,
             citations: bool = False) -> Path:
    """Download Dilemma + tagger artifacts from HuggingFace to `target_dir`.

    Lemma `data/` and `model/` go directly under `target_dir`. Tagger
    weights land at `target_dir/tagger_model/` (mirrors `~/.cache/dilemma/`
    layout: `model/` for the lemmatizer, `tagger_model/` for the tagger).

    The form-attestation DBs are excluded from the base `data/*` download:
    `attestation=True` pulls `form_profile.db` (the gate + usage distribution),
    and `citations=True` additionally pulls the larger `form_citations.db`
    (example loci) and implies the profile.

    Returns the path that was downloaded into. Requires `huggingface_hub`.
    """
    dest = Path(target_dir) if target_dir else DEFAULT_CACHE
    dest.mkdir(parents=True, exist_ok=True)

    if dilemma:
        _snapshot_download(
            repo_id=REPO, local_dir=dest, allow_patterns=INCLUDES,
            ignore_patterns=ATTEST_FILES,
        )
    if attestation or citations:
        _snapshot_download(
            repo_id=REPO, local_dir=dest, allow_patterns=[PROFILE_FILE],
        )
    if citations:
        _snapshot_download(
            repo_id=REPO, local_dir=dest, allow_patterns=[CITATIONS_FILE],
        )
    if tagger:
        tagger_dest = dest / "tagger_model"
        tagger_dest.mkdir(parents=True, exist_ok=True)
        # The HF repo lays out tagger weights under `tagger/<lang>/...`. We
        # strip the leading `tagger/` so files land at
        # `tagger_model/<lang>/...`, which is what
        # dilemma.tagger._WEIGHTS_DIR expects.
        _snapshot_download(
            repo_id=TAGGER_REPO,
            local_dir=tagger_dest.parent / "_tagger_tmp",
            allow_patterns=TAGGER_INCLUDES,
        )
        _flatten_tagger_weights(tagger_dest.parent / "_tagger_tmp", tagger_dest)

    return dest


def _flatten_tagger_weights(src: Path, dst: Path) -> None:
    """Move src/tagger/<lang>/... -> dst/<lang>/...; remove src tree."""
    import shutil
    weights_root = src / "tagger"
    if not weights_root.exists():
        return
    for child in weights_root.iterdir():
        target = dst / child.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(child), str(target))
    shutil.rmtree(src, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dilemma download",
        description="Download Dilemma lookup tables, lemma model, and tagger weights.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"Target directory (default: {DEFAULT_CACHE})",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--no-tagger", action="store_true",
        help="Skip tagger POS / dep weights (lemma data only)",
    )
    group.add_argument(
        "--only-tagger", action="store_true",
        help="Download only tagger weights (skip lemma data + model)",
    )
    parser.add_argument(
        "--with-attestation", action="store_true",
        help="Also fetch form_profile.db (the 'attested only' gate + the "
             "form_attestation usage distribution); excluded from base otherwise",
    )
    parser.add_argument(
        "--with-citations", action="store_true",
        help="Also fetch the large form_citations.db (example loci for "
             "form_attestation); implies --with-attestation",
    )
    args = parser.parse_args(argv)
    dest = download(
        args.dir,
        tagger=not args.no_tagger,
        dilemma=not args.only_tagger,
        attestation=(args.with_attestation or args.with_citations)
        and not args.only_tagger,
        citations=args.with_citations and not args.only_tagger,
    )
    print(f"Downloaded to {dest}")
    print(
        "Dilemma will find this automatically, or set "
        f"DILEMMA_DATA_DIR={dest / 'data'} to override."
    )
    if not args.no_tagger:
        print(
            "Tagger weights at "
            f"{dest / 'tagger_model'}; override with DILEMMA_TAGGER_DIR."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Guards for the PTNK / TAGNT / Pedalion ingestion from cog's standardized
annotation exports (build/extract_{ptnk,tagnt,pedalion}.py + cog_annotations).

The extractor tests are gated on the cog annotations root being present (they
read the gzipped exports), so they skip cleanly on CI where the corpora are
absent. The shipped-artifact invariants (lemma_attestation carries the pedalion
source + pin) are asserted in tests/test_attestation.py and always run.
"""
import json
import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build"))

import cog_annotations as C  # noqa: E402

_HAVE_EXPORTS = C.export_dir("pedalion") is not None
pytestmark = pytest.mark.skipif(
    not _HAVE_EXPORTS,
    reason="cog annotation exports not present (set DILEMMA_COG_ANNOTATIONS)")

_JUNK_FINALS = tuple("᾽'’ʼ`ʹ")


def _pairs_shape_ok(pairs):
    assert isinstance(pairs, list) and pairs
    for p in pairs[:2000]:
        assert set(p) >= {"form", "lemma"}
        assert p["form"] and p["lemma"]
        assert not p["lemma"].endswith(_JUNK_FINALS)


def test_pedalion_skips_gorman_and_strips_homographs():
    import extract_pedalion as E
    export = C.export_dir("pedalion")
    pairs, _manifest, kept, gorman, _skipped = E.extract(export)
    _pairs_shape_ok(pairs)
    # the Gorman held-out rows must be actively filtered, not merely absent
    assert gorman > 0
    # homograph-disambiguation digits (ξένος2) are stripped from every lemma
    assert not any(p["lemma"] and p["lemma"][-1].isdigit() for p in pairs)
    # constructed example-sentence collections still contribute lookup pairs,
    # so kept tokens vastly outnumber the six literary works alone
    assert kept > 50_000


def test_ptnk_is_train_split_only():
    """No dev/test PTNK surface form may leak into the ingested pairs (dev+test
    are held out for eval/eval_ptnk.py)."""
    import extract_ptnk as E
    export = C.export_dir("ptnk")
    pairs, *_ = E.extract(export)
    _pairs_shape_ok(pairs)
    train_forms = set()
    held_out_forms = set()
    for w in C.load_manifest(export)["works"]:
        for rec in C.iter_work_tokens(export, w):
            form = unicodedata.normalize("NFC", (rec.get("form") or "").strip())
            if not form or not rec.get("lemma"):
                continue
            (train_forms if rec.get("split") == "train"
             else held_out_forms).add(form)
    emitted = {p["form"] for p in pairs}
    # every emitted form is attested in train; none is train-absent
    assert emitted <= train_forms
    assert not (emitted - train_forms)


def test_tagnt_splits_multiform_headwords():
    import extract_tagnt as E
    export = C.export_dir("tagnt")
    pairs, *_ = E.extract(export)
    _pairs_shape_ok(pairs)
    # "Δαυείδ, Δαυίδ, Δαβίδ" must be split, so no lemma retains a comma
    assert not any("," in p["lemma"] for p in pairs)


def test_pins_are_cog_exports():
    for name in ("ptnk", "tagnt", "pedalion"):
        manifest = C.load_manifest(C.export_dir(name))
        assert C.pin_line(manifest).startswith(f"cog export {name}-")

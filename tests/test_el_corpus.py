"""Manifest policy, inputs, readers, split, caps, shards and an end-to-end
run of the monotonic Modern Greek corpus extractor
(build/el_corpus.py, extract_el_monotonic_lm.py).

Every input is a small file written by the test; nothing is downloaded.

Run with:
    python -m pytest tests/test_el_corpus.py -x -v
"""

import bz2
import gzip
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import extract_el_monotonic_lm as extractor  # noqa: E402
from dilemma_build import el_corpus as C  # noqa: E402
from dilemma_build import el_text as T  # noqa: E402

SHA0 = "0" * 64
COMMIT0 = "a" * 40


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path, **extra):
    return {"size": Path(path).stat().st_size, "sha256": sha(path), **extra}


def url_source(key, path, reader, **extra):
    return {
        "key": key, "title": key, "role": "el_lm_core", "license": "CC0-1.0",
        "license_evidence": "test fixture",
        "fetch": {"kind": "url", "url": f"https://example.invalid/{Path(path).name}",
                  "filename": Path(path).name, **pin(path)},
        "reader": reader, **extra,
    }


def manifest_of(*sources, max_repeats=5):
    return {"schema_version": 1,
            "split": {"unit": "group", "seed": C.SPLIT_SEED, "dev_fraction": 0.02},
            "max_sentence_repeats": max_repeats,
            "sources": list(sources), "excluded": []}


# -- licenses and manifest -------------------------------------------------

@pytest.mark.parametrize("lic", ["CC-BY-NC-4.0", "CC-BY-NC-SA-4.0", "CC-BY-ND-4.0",
                                 "CC-BY-NC-ND-4.0", "cc-by-nc-4.0"])
def test_nc_and_nd_licenses_are_refused(lic):
    assert "NonCommercial or NoDerivatives" in C.license_refusal(lic)


@pytest.mark.parametrize("lic", ["", "proprietary", "GPL-3.0", "CC-BY-SA-2.5"])
def test_unknown_licenses_are_refused(lic):
    assert C.license_refusal(lic)


@pytest.mark.parametrize("lic", sorted(C.ALLOWED_LICENSES))
def test_allowed_licenses_pass(lic):
    assert C.license_refusal(lic) is None


def test_the_repository_manifest_validates():
    manifest = C.load_manifest()
    keys = {s["key"] for s in manifest["sources"]}
    assert {"parliament", "opengov_comments", "wikipedia", "tatoeba"} <= keys
    for src in manifest["sources"]:
        if src.get("enabled", True):
            assert C.license_refusal(src["license"]) is None
    excluded = {e["key"] for e in manifest["excluded"]}
    assert {"opensubtitles", "triantafyllidis"} <= excluded


def _good_manifest():
    return manifest_of({
        "key": "demo", "role": "el_lm_core", "license": "CC-BY-4.0",
        "license_evidence": "x",
        "fetch": {"kind": "hf_dataset", "repo": "org/demo", "revision": COMMIT0,
                  "files": [{"path": "a.parquet", "size": 10, "sha256": SHA0}]},
        "reader": {"kind": "parquet_text", "id_field": "id", "text_field": "t"},
    })


def test_a_well_formed_manifest_validates():
    C.validate_manifest(_good_manifest())


@pytest.mark.parametrize("change, message", [
    (lambda s: s.update(license="CC-BY-NC-SA-4.0"), "refused"),
    (lambda s: s["fetch"].update(revision="main"), "full commit sha"),
    (lambda s: s["fetch"]["files"][0].update(sha256="abc"), "invalid sha256"),
    (lambda s: s["fetch"]["files"][0].update(size=0), "invalid size"),
    (lambda s: s.update(role="other"), "invalid role"),
    (lambda s: s.pop("license_evidence"), "license_evidence"),
    (lambda s: s.update(token_cap=0), "token_cap"),
    (lambda s: s.update(weight=0.5), "weight"),
    (lambda s: s.update(dev_fraction=0.6), "dev_fraction"),
    (lambda s: s.update(key="Bad-Key"), "key"),
    (lambda s: s["reader"].pop("kind"), "reader kind"),
    (lambda s: s.update(enabled=False), "disabled_reason"),
])
def test_malformed_manifests_are_refused(change, message):
    m = _good_manifest()
    change(m["sources"][0])
    with pytest.raises(C.CorpusError, match=message):
        C.validate_manifest(m)


def test_a_disabled_source_may_carry_an_unlisted_license():
    m = _good_manifest()
    m["sources"][0].update(enabled=False, license="LicenseRef-Terms",
                           disabled_reason="no named license")
    C.validate_manifest(m)


def test_an_excluded_key_cannot_also_be_a_source():
    m = _good_manifest()
    m["excluded"] = [{"key": "demo", "license": "CC-BY-NC-4.0", "reason": "x"}]
    with pytest.raises(C.CorpusError, match="both a source and excluded"):
        C.validate_manifest(m)


# -- inputs -----------------------------------------------------------------

def test_inputs_are_verified_against_their_pins(tmp_path, monkeypatch):
    monkeypatch.setenv("DILEMMA_EL_RAW_DIR", str(tmp_path / "raw"))
    raw = tmp_path / "raw" / "lines"
    raw.mkdir(parents=True)
    f = raw / "lines.txt"
    f.write_text("Καλημέρα σε όλους.\n", encoding="utf-8")
    src = url_source("lines", f, {"kind": "text_lines"})
    assert C.resolve_inputs(src) == [f]
    f.write_text("Καλημέρα σε όλοις.\n", encoding="utf-8")   # same size
    with pytest.raises(C.CorpusError, match="sha256"):
        C.resolve_inputs(src)
    f.write_text("άλλο\n", encoding="utf-8")
    with pytest.raises(C.CorpusError, match="size"):
        C.resolve_inputs(src)
    f.unlink()
    with pytest.raises(C.CorpusError, match="--download"):
        C.resolve_inputs(src)


# -- readers ----------------------------------------------------------------

def test_parliament_reader_groups_by_sitting(tmp_path):
    csv_text = ("member_name,sitting_date,parliamentary_period,parliamentary_session,"
                "parliamentary_sitting,speech\n"
                "Α,2020-01-01,ΙΗ,Α,1,Πρώτη ομιλία.\n"
                "Β,2020-01-01,ΙΗ,Α,1,\"Δεύτερη, με κόμμα.\"\n"
                "Γ,2020-01-02,ΙΗ,Α,2,Τρίτη.\n")
    z = tmp_path / "p.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("dataset_versions/tell_all.csv", csv_text)
    src = {"key": "parliament", "reader": {
        "kind": "parliament_csv", "member": "dataset_versions/tell_all.csv",
        "text_column": "speech",
        "group_columns": ["sitting_date", "parliamentary_period",
                          "parliamentary_session", "parliamentary_sitting"]}}
    docs = list(C.iter_documents(src, [z]))
    assert [d.id for d in docs] == ["parliament:row:0", "parliament:row:1",
                                    "parliament:row:2"]
    assert docs[0].group == docs[1].group == "parliament:2020-01-01:ΙΗ:Α:1"
    assert docs[2].group != docs[0].group
    assert docs[1].text == "Δεύτερη, με κόμμα."
    assert len(list(C.iter_documents(src, [z], max_docs=2))) == 2


def test_tatoeba_and_text_line_readers(tmp_path):
    tb = tmp_path / "ell.tsv.bz2"
    tb.write_bytes(bz2.compress("12\tell\tΠού είσαι;\nbad line\n13\tell\tΕδώ.\n"
                                .encode("utf-8")))
    docs = list(C.iter_documents({"key": "tatoeba",
                                  "reader": {"kind": "tatoeba_tsv_bz2"}}, [tb]))
    assert [(d.id, d.group, d.text) for d in docs] == [
        ("tatoeba:12", "tatoeba:12", "Πού είσαι;"), ("tatoeba:13", "tatoeba:13", "Εδώ.")]
    tl = tmp_path / "lines.txt"
    tl.write_text("Πρώτη.\n\n  Τρίτη.  \n", encoding="utf-8")
    docs = list(C.iter_documents({"key": "cv", "reader": {"kind": "text_lines"}}, [tl]))
    assert [(d.id, d.text) for d in docs] == [("cv:line:0", "Πρώτη."),
                                              ("cv:line:2", "Τρίτη.")]


def test_parquet_readers(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    p = tmp_path / "w.parquet"
    pq.write_table(pa.table({"id": ["1", "2", "3"], "title": ["Α", "Β", "Γ"],
                             "year": ["1900", "400 π.Χ.", None],
                             "text": ["Ένα.", "Δύο.", None]}), p)
    src = {"key": "w", "reader": {"kind": "parquet_text", "id_field": "id",
                                  "text_field": "text", "group_field": "title",
                                  "skip_if_field_matches": {"year": r"π\.Χ"}}}
    docs = list(C.iter_documents(src, [p]))
    assert [(d.id, d.group, d.text) for d in docs] == [
        ("w:1", "w:Α", "Ένα."), ("w:3", "w:Γ", "")]

    og = tmp_path / "OpenGov.parquet"
    pq.write_table(pa.table({
        "consultation_id": [7],
        "articles": [[{"comments": [{"comment_row_id": 1, "content": "Σχόλιο ένα."},
                                    {"comment_row_id": 2, "content": None}]},
                      {"comments": None}]]}), og)
    docs = list(C.iter_documents({"key": "og", "reader": {"kind": "opengov_comments"}},
                                 [og]))
    assert [(d.id, d.group, d.text) for d in docs] == [
        ("og:OpenGov:1", "og:7", "Σχόλιο ένα."), ("og:OpenGov:2", "og:7", "")]


def test_kaikki_reader_keeps_modern_greek_examples(tmp_path):
    entries = [
        {"word": "Σπίτι", "lang_code": "el",
         "senses": [{"examples": [{"text": "Το σπίτι μας."}, {"text": ""}]}]},
        {"word": "οἶκος", "lang_code": "grc",
         "senses": [{"examples": [{"text": "ὁ οἶκος"}]}]},
        {"word": "νερό", "lang_code": "el", "senses": [{"glosses": ["water"]}]},
    ]
    f = tmp_path / "kaikki.org-dictionary-Greek.jsonl"
    f.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
                 encoding="utf-8")
    docs = list(C.iter_documents({"key": "wkt", "reader": {"kind": "kaikki_examples"}},
                                 [f]))
    assert [(d.id, d.group, d.text) for d in docs] == [
        ("wkt:en:0:0:0", "wkt:σπίτι", "Το σπίτι μας.")]


def test_text_options_defaults():
    assert C.text_options({"reader": {"kind": "x"}}) == {
        "line_breaks": "sentence", "require_terminal_punctuation": False,
        "stop_at_lines": frozenset()}


# -- split, caps, shards ----------------------------------------------------

def test_split_is_deterministic_and_close_to_its_fraction():
    groups = [f"g:{i}" for i in range(20000)]
    a = [C.split_of(g, 0.02) for g in groups]
    assert a == [C.split_of(g, 0.02) for g in groups]
    share = a.count("dev") / len(a)
    assert 0.015 < share < 0.025
    assert C.split_of("g:1", 0.02, seed=1) in ("dev", "train")


def test_cap_threshold():
    hist = [10] * 10 + [0] * (C.BUCKETS - 10)
    assert C.cap_threshold(hist, None) == (C.BUCKETS, 100)
    assert C.cap_threshold(hist, 1000) == (C.BUCKETS, 100)
    assert C.cap_threshold(hist, 35) == (3, 30)
    assert C.cap_threshold(hist, 5) == (0, 0)


def test_shard_writer_is_byte_stable_and_round_trips(tmp_path):
    def write(out):
        w = C.ShardWriter(out, "train", shard_tokens=5)
        w.write("d:1", "g:1", ["Καλή μέρα", "σ’ αγαπώ"], 4)
        w.write("d:2", "g:1", ["τρία τέσσερα πέντε"], 3)
        w.write("d:3", "g:2", ["έξι"], 1)
        return w.close()

    a = write(tmp_path / "a")
    b = write(tmp_path / "b")
    assert [s["file"] for s in a] == ["train-00000.jsonl.gz", "train-00001.jsonl.gz"]
    assert [s["tokens"] for s in a] == [7, 1]
    assert a == b
    for s in a:
        assert (tmp_path / "a" / s["file"]).read_bytes() == \
            (tmp_path / "b" / s["file"]).read_bytes()
        assert s["sha256"] == sha(tmp_path / "a" / s["file"])
    docs = list(C.iter_shard_documents(tmp_path / "a" / a[0]["file"]))
    assert docs[0] == {"id": "d:1", "group": "g:1", "s": ["Καλή μέρα", "σ’ αγαπώ"]}
    with gzip.open(tmp_path / "a" / a[0]["file"]) as fh:
        assert fh.read().decode("utf-8").count("\n") == 2


# -- end to end -------------------------------------------------------------

_SYL1 = ["κα", "λο", "μι", "νε", "πα", "ρο", "τι", "σο"]
_SYL2 = ["λό", "μά", "νέ", "πή", "ρί", "τύ", "σώ", "δά"]
LINES_KEY = "lines"
LINES_FRACTION = 0.3


def _corpus_fixture(tmp_path):
    """A text_lines source whose dev and train lines share one sentence and
    whose train lines repeat another, plus a Tatoeba source."""
    raw = tmp_path / "raw"
    sides = {i: C.split_of(f"{LINES_KEY}:line:{i}", LINES_FRACTION)
             for i in range(60)}
    dev_lines = [i for i, s in sides.items() if s == "dev"]
    train_lines = [i for i, s in sides.items() if s == "train"]
    assert len(dev_lines) >= 3 and len(train_lines) >= 20
    lines = []
    for i in range(60):
        word = _SYL1[i % 8] + _SYL2[i // 8]
        text = f"Η λέξη {word} είναι εδώ."
        if i == dev_lines[0]:
            text += " Καλημέρα σε όλους."
        if i in train_lines[:3]:
            text += " καλημέρα σε ΌΛΟΥΣ."          # same key after case folding
        if i in train_lines[3:11]:
            text += " Ευχαριστώ πολύ για όλα."    # 8 copies, cap 5
        lines.append(text)
    f = raw / LINES_KEY / "lines.txt"
    f.parent.mkdir(parents=True)
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tb = raw / "tatoeba" / "ell.tsv.bz2"
    tb.parent.mkdir(parents=True)
    tb.write_bytes(bz2.compress(
        "1\tell\tΠού είσαι;\n2\tell\tἘν ἀρχῇ ἦν ὁ λόγος.\n3\tell\tΕντάξει, ξέρω.\n"
        "4\tell\tΕίμαι εδώ.\n".encode("utf-8")))

    manifest = manifest_of(
        url_source(LINES_KEY, f, {"kind": "text_lines"},
                   dev_fraction=LINES_FRACTION, token_cap=60),
        url_source("tatoeba", tb, {"kind": "tatoeba_tsv_bz2"}, weight=10,
                   dev_fraction=0.05),
    )
    mpath = tmp_path / "corpora.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return mpath, raw, sides


def _run(mpath, out):
    assert extractor.main(["--manifest", str(mpath), "--out-dir", str(out),
                           "--workers", "0"]) == 0
    return out


def test_extractor_end_to_end(tmp_path, monkeypatch):
    mpath, raw, _ = _corpus_fixture(tmp_path)
    monkeypatch.setenv("DILEMMA_EL_RAW_DIR", str(raw))
    out = _run(mpath, tmp_path / "corpus")
    meta = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    lines = meta["sources"][LINES_KEY]
    dev, train = lines["splits"]["dev"], lines["splits"]["train"]
    assert dev["documents"] + train["documents"] == 60
    assert lines["documents_read"] == {"dev": 60, "train": 60}

    groups = {}
    keys = {}
    for split in ("dev", "train"):
        docs = [d for s in lines["splits"][split]["shards"]
                for d in C.iter_shard_documents(out / LINES_KEY / s["file"])]
        groups[split] = {d["group"] for d in docs}
        keys[split] = [T.sentence_key(x.split(" ")) for d in docs for x in d["s"]]
    assert not groups["dev"] & groups["train"]
    assert not set(keys["dev"]) & set(keys["train"])

    assert train["dropped_dev_duplicates"] == {"sentences": 3, "tokens": 9}
    assert train["dropped_repeats"] == {"sentences": 3, "tokens": 12}
    thanks = T.sentence_key(["Ευχαριστώ", "πολύ", "για", "όλα"])
    assert keys["train"].count(thanks) == 5
    assert train["repeat_cap_tokens_removed"]["1"] == 7 * 4
    assert train["most_repeated"][0] == [8, "Ευχαριστώ πολύ για όλα"]

    tatoeba = meta["sources"]["tatoeba"]
    assert tatoeba["weight"] == 10
    kept = (tatoeba["splits"]["dev"]["sentences"]
            + tatoeba["splits"]["train"]["sentences"])
    assert kept == 3
    assert tatoeba["filters"]["docs_dropped_polytonic"] == 1

    # the cap is recorded, not applied to the shards
    cap = lines["cap"]
    assert cap["token_cap"] == 60 and cap["bucket_threshold"] < C.BUCKETS
    capped = sum(len(t) for _, t in C.iter_corpus_sentences(out, LINES_KEY, "train"))
    full = sum(len(t) for _, t in
               C.iter_corpus_sentences(out, LINES_KEY, "train", cap=False))
    assert capped == cap["train_tokens_under_cap"] <= 60 < full == train["tokens"]
    ids = [i for i, _ in C.iter_corpus_sentences(out, LINES_KEY, "dev")]
    assert len(ids) == len(set(ids)) == dev["sentences"]

    reg = meta["sources"]["tatoeba"]["register_per_million_words"]
    assert set(reg) == set(T.REGISTER_WORDS)
    assert meta["totals"]["train"]["tokens"] == (
        train["tokens"] + tatoeba["splits"]["train"]["tokens"])

    # a second run writes the same bytes
    again = _run(mpath, tmp_path / "corpus2")
    files = sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(again) for p in again.rglob("*") if p.is_file())
    for rel in files:
        assert (out / rel).read_bytes() == (again / rel).read_bytes(), rel


def test_a_partial_run_cannot_replace_the_full_corpus():
    with pytest.raises(SystemExit):
        extractor.main(["--sanity", "--out-dir", str(C.DEFAULT_OUT_DIR)])
    with pytest.raises(SystemExit):
        extractor.main(["--sources", "tatoeba", "--out-dir", str(C.DEFAULT_OUT_DIR)])


def test_a_refused_license_stops_the_extractor(tmp_path, monkeypatch):
    mpath, raw, _ = _corpus_fixture(tmp_path)
    m = json.loads(mpath.read_text(encoding="utf-8"))
    m["sources"][1]["license"] = "CC-BY-NC-4.0"
    mpath.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("DILEMMA_EL_RAW_DIR", str(raw))
    with pytest.raises(C.CorpusError, match="refused"):
        extractor.main(["--manifest", str(mpath), "--out-dir",
                        str(tmp_path / "out"), "--workers", "0"])
    assert not (tmp_path / "out").exists()

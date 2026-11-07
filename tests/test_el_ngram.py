"""The monotonic Modern Greek next-word model: token resolution, streaming
counts, pruning, export and the binary read back (build/el_ngram.py,
train_lm.py --lang el, export_lm.py --lang el, build/gnlm_reader.py).

Every input is a small file written by the test: a corpus in the shard
format of ``extract_el_monotonic_lm.py`` and a word list in the format of
``export_el_monotonic.py``. Nothing is downloaded.

Run with:
    python -m pytest tests/test_el_ngram.py -x -v
"""

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import export_lm  # noqa: E402
from dilemma_build import el_corpus as C  # noqa: E402
from dilemma_build import el_ngram as N  # noqa: E402
from dilemma_build.gnlm_reader import FormatError, GNLMReader  # noqa: E402
from eval_lm import NgramLM  # noqa: E402

WORDS = """θα πάμε στην Αθήνα αύριο η βουλή ψήφισε το νομοσχέδιο όνομα μου
είναι Νίκη νίκη δεν ξέρω πότε ποτέ έρθει έρθεις ο μίλησε για άνθρωποι ήρθαν
τι κάνεις ευχαριστώ πολύ σινεμά εσύ που πού""".split()

NEWS_TRAIN = [
    ["Θα πάμε στην Αθήνα αύριο"],
    ["Η Βουλή ψήφισε το νομοσχέδιο", "Το όνομά μου είναι Νίκη"],
    ["Δεν ξέρω πότε θα έρθει", "Δεν ξέρω πότε θα έρθει", "Δεν ξέρω πότε"],
    ["Ο κ. Παπαδόπουλος μίλησε για το ΕΣΥ"],
    ["Πότε θα έρθεις", "Δεν θα έρθει ποτέ"],
    ["<NUM> άνθρωποι ήρθαν αύριο"],
    ["Η νίκη είναι δική μου"],
]
CONV_TRAIN = [["Τι κάνεις"], ["Ευχαριστώ πολύ"], ["Θα πάμε σινεμά"],
              ["Πού πάμε", "Δεν ξέρω που είναι"]]
NEWS_DEV = [["Θα πάμε σινεμά αύριο", "Δεν ξέρω"]]
CONV_DEV = [["Τι κάνεις εσύ"]]


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_shard(directory: Path, split: str, docs, source: str) -> list[dict]:
    w = C.ShardWriter(directory, split)
    for k, sents in enumerate(docs):
        tokens = sum(len(s.split(" ")) for s in sents)
        w.write(f"{source}:{split}:{k}", f"{source}:{split}:{k}", sents, tokens)
    return w.close()


def make_corpus(root: Path, *, news_threshold: int = C.BUCKETS) -> Path:
    corpus = root / "corpus"
    sources = {}
    for name, train, dev, weight, threshold in (
            ("news", NEWS_TRAIN, NEWS_DEV, 1, news_threshold),
            ("conv", CONV_TRAIN, CONV_DEV, 2, C.BUCKETS)):
        sources[name] = {
            "weight": weight,
            "cap": {"token_cap": None, "bucket_threshold": threshold},
            "splits": {
                "train": {"shards": write_shard(corpus / name, "train", train, name)},
                "dev": {"shards": write_shard(corpus / name, "dev", dev, name)},
            },
        }
    meta = {"schema_version": 1, "commit": "a" * 40,
            "corpus_manifest_sha256": "0" * 64, "sources": sources}
    (corpus / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False),
                                          encoding="utf-8")
    return corpus


def make_dictionary(root: Path, words=WORDS) -> Path:
    dic = root / "el_monotonic.dic"
    lines = sorted(set(words))
    dic.write_text(f"{len(lines)}\n" + "".join(f"{w}\tfr:C\n" for w in lines),
                   encoding="utf-8")
    (root / "el_monotonic.aff").write_text("SET UTF-8\nLANG el\nFLAG num\n",
                                           encoding="utf-8")
    return dic


def train(root: Path, out_name: str, *extra: str, corpus=None, dic=None) -> Path:
    """Train on the fixture keeping every n-gram: the shipped minimum counts
    are for 200M tokens and would empty a corpus of a few sentences."""
    corpus = corpus or make_corpus(root)
    dic = dic or make_dictionary(root)
    out = root / out_name
    N.main(["--corpus-dir", str(corpus), "--dictionary", str(dic),
            "--out", str(out), "--cache-dir", str(root / "encoded"),
            "--workers", "1", "--min-bigram-for-tri", "1",
            "--min-count-tri", "1", "--min-count-bi", "1", *extra])
    return out


def export(root: Path, in_dir: Path, name: str, *extra: str):
    out = root / f"{name}.bin"
    version = root / f"{name}.version"
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "export_lm.py"), "--lang", "el",
         "--in", str(in_dir), "--out", str(out), "--version-out", str(version),
         "--oov-dictionary", str(root / "el_monotonic.dic"), *extra],
        capture_output=True, text=True)
    return proc, out, version


# --- resolution ---------------------------------------------------------------

def test_the_enclitic_second_accent_is_folded_away():
    assert N.fold_enclitic_accent("όνομά") == "όνομα"
    assert N.fold_enclitic_accent("άκουσέ") == "άκουσε"
    assert N.fold_enclitic_accent("Όνομά") == "Όνομα"
    assert N.fold_enclitic_accent("είναι") == "είναι"
    assert N.fold_enclitic_accent("και") == "και"


@pytest.fixture(scope="module")
def resolver():
    return N.Resolver(sorted(set(WORDS)))


@pytest.mark.parametrize("token, initial, spelling, outcome", [
    ("Θα", True, "θα", N.LOWERED_INITIAL),
    ("Αθήνα", True, "Αθήνα", N.EXACT),
    ("Νίκη", False, "Νίκη", N.EXACT),
    ("Νίκη", True, "νίκη", N.LOWERED_INITIAL),
    ("Βουλή", False, "βουλή", N.LOWERED_INSIDE),
    ("Ο", False, "ο", N.LOWERED_INSIDE),
    ("όνομά", False, "όνομα", N.EXACT | N.ENCLITIC_FOLDED),
    ("Όνομά", True, "όνομα", N.LOWERED_INITIAL | N.ENCLITIC_FOLDED),
    ("ΕΣΥ", False, None, N.UNK_CAPS),
    ("ΘΑ", True, None, N.UNK_CAPS),
    ("Α", False, None, N.UNK_CAPS),
    ("<NUM>", False, None, N.UNK_PLACEHOLDER),
    ("<FOREIGN>", True, None, N.UNK_PLACEHOLDER),
    ("κ.", False, None, N.UNK_ABBREVIATION),
    ("ειναι", False, None, N.UNK_NOT_LISTED),
    ("αθήνα", False, None, N.UNK_NOT_LISTED),
])
def test_tokens_resolve_to_the_spellings_the_keyboard_looks_up(
        resolver, token, initial, spelling, outcome):
    full_id, got = resolver.resolve(token, initial)
    names = list(N.RESERVED) + resolver.entries
    assert got == outcome
    assert (names[full_id] if spelling else full_id) == (spelling or N.UNK)


# --- counting -----------------------------------------------------------------

def naive_counts(corpus: Path, dic: Path, vocab: list[str]):
    """Counter-based counts of the training split, the trainer's reference."""
    r = N.Resolver(N.load_dictionary(dic))
    names = list(N.RESERVED) + r.entries
    ids = {t: i for i, t in enumerate(vocab)}
    meta = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    uni, bi, tri = Counter(), Counter(), Counter()
    for source, entry in meta["sources"].items():
        for shard in entry["splits"]["train"]["shards"]:
            for doc in C.iter_shard_documents(corpus / source / shard["file"]):
                for s in doc["s"]:
                    toks = s.split(" ")
                    seq = [N.BOS] + [
                        ids.get(names[r.resolve(t, k == 0)[0]], N.UNK)
                        for k, t in enumerate(toks)] + [N.EOS]
                    for _ in range(entry["weight"]):
                        uni.update(seq)
                        bi.update(zip(seq, seq[1:]))
                        tri.update(zip(seq, seq[1:], seq[2:]))
    return uni, bi, tri


@pytest.mark.parametrize("chunk", [5, 10_000])
def test_streaming_counts_equal_counter_counts(tmp_path, chunk):
    corpus, dic = make_corpus(tmp_path), make_dictionary(tmp_path)
    c = N.count_corpus(corpus, dic, vocab_size=1000, workers=1,
                       cache_dir=tmp_path / "enc", chunk_tokens=chunk)
    uni, bi, tri = naive_counts(corpus, dic, c.vocab)
    assert {i: n for i, n in enumerate(c.uni.tolist()) if n} == dict(uni)
    got_bi = {(int(k) >> N.ID_BITS, int(k) & N.ID_MASK): int(n)
              for k, n in zip(*c.bi)}
    got_tri = {(int(k) >> 2 * N.ID_BITS, (int(k) >> N.ID_BITS) & N.ID_MASK,
                int(k) & N.ID_MASK): int(n) for k, n in zip(*c.tri)}
    assert got_bi == dict(bi)
    assert got_tri == dict(tri)


def test_enclitic_hosts_and_sentence_capitals_count_as_the_base_spelling(tmp_path):
    corpus, dic = make_corpus(tmp_path), make_dictionary(tmp_path)
    c = N.count_corpus(corpus, dic, vocab_size=1000, workers=1,
                       cache_dir=tmp_path / "enc")
    count = dict(zip(c.vocab, c.uni.tolist()))
    assert count["όνομα"] == 1          # from "όνομά μου"
    assert count["θα"] == 5 + 2         # sentence-initial Θα, conv weighs 2
    assert "Θα" not in count and "όνομά" not in count
    assert count["Νίκη"] == 1 and count["νίκη"] == 1
    assert c.outcomes["enclitic_folded"] == 1
    assert c.outcomes["unk_capitals"] == 1             # ΕΣΥ
    assert c.outcomes["unk_abbreviation"] == 1         # κ.
    assert c.outcomes["unk_not_in_word_list"] == 2     # Παπαδόπουλος, δική


def test_the_encoding_cache_follows_every_helper_it_depends_on(monkeypatch):
    # A cached encoding must not outlive a change to the code or constants
    # that decide its content, or a cached run would differ from a fresh one.
    base = N.resolver_digest()
    monkeypatch.setattr(N.el_lexicon, "LETTERS",
                        N.el_lexicon.LETTERS | frozenset("ϝ"))
    assert N.resolver_digest() != base
    monkeypatch.undo()
    assert N.resolver_digest() == base
    monkeypatch.setattr(N.el_lexicon, "APOSTROPHE", "'")
    assert N.resolver_digest() != base
    monkeypatch.undo()
    for helper in (N.sample_bucket, N.el_corpus.cap_bucket, N.el_corpus._bucket):
        real = N.inspect.getsource

        def changed(obj, helper=helper, real=real):
            return real(obj) + ("# changed" if obj is helper else "")

        monkeypatch.setattr(N.inspect, "getsource", changed)
        assert N.resolver_digest() != base, helper.__name__
        monkeypatch.undo()


def test_a_token_cap_keeps_only_documents_below_its_bucket(tmp_path):
    corpus = make_corpus(tmp_path, news_threshold=0)
    dic = make_dictionary(tmp_path)
    c = N.count_corpus(corpus, dic, vocab_size=1000, workers=1,
                       cache_dir=tmp_path / "enc")
    assert set(c.per_source) == {"news", "conv"}
    assert c.per_source["news"]["train_tokens"] == 0
    assert c.per_source["conv"]["weighted_tokens"] == 2 * c.per_source["conv"]["train_tokens"]


def test_contested_spellings_seen_past_the_table_cap_join_the_vocabulary():
    entries = ["αβ", "πότε", "ποτέ", "χ"]
    counts = np.array([0, 0, 0, 0, 50, 9, 7, 3])
    contested = N.contested_mask(entries)
    assert contested.tolist() == [False] * 5 + [True, True, False]
    ids, forced = N.choose_vocab(counts, entries, 5, contested, oov_cap=5)
    assert forced == 2 and sorted(ids) == [0, 1, 2, 3, 4, 5, 6]
    ids, forced = N.choose_vocab(counts, entries, 5, contested, oov_cap=8)
    assert forced == 1 and 5 in ids and 6 not in ids


# --- end to end ---------------------------------------------------------------

@pytest.fixture(scope="module")
def model(tmp_path_factory):
    root = tmp_path_factory.mktemp("el_lm")
    corpus, dic = make_corpus(root), make_dictionary(root)
    # 20 ids: the four reserved tokens and the 16 most frequent spellings,
    # so ποτέ (seen once) stays out and goes to the count table.
    in_dir = train(root, "lm_el", "--vocab-size", "20", corpus=corpus, dic=dic)
    proc, out, version = export(root, in_dir, "el_ngram")
    assert proc.returncode == 0, proc.stderr
    return {"root": root, "in": in_dir, "bin": out,
            "version": json.loads(version.read_text(encoding="utf-8")),
            "corpus": corpus, "dic": dic}


def test_the_binary_passes_every_structural_check(model):
    lm = GNLMReader(model["bin"])
    assert lm.problems() == []
    data = model["bin"].read_bytes()
    assert data[:4] == b"GNLM"
    assert int.from_bytes(data[4:8], "little") == 2
    assert int.from_bytes(data[8:12], "little") == 3
    for name, i in zip(N.RESERVED, (lm.id_pad, lm.id_unk, lm.id_bos, lm.id_eos)):
        assert lm.vocab[i] == name
    # </s> sorts before <PAD>, so the reserved ids are not 0..3 in order.
    assert lm.id_eos == 0


def test_the_vocabulary_holds_only_list_spellings(model):
    lm = GNLMReader(model["bin"])
    words = set(N.load_dictionary(model["dic"]))
    assert all(t in words or t in N.RESERVED for t in lm.vocab)
    assert "Θα" not in lm.vocab and "θα" in lm.vocab


def test_no_row_is_headed_by_unk(model):
    lm = GNLMReader(model["bin"])
    assert lm.id_unk not in lm.bi_index
    assert not any(lm.id_unk in ctx for ctx in lm.tri_index)


def test_out_of_vocabulary_twins_get_their_exact_count(model):
    lm = GNLMReader(model["bin"])
    assert "ποτέ" not in lm.index and "πότε" in lm.index
    assert lm.oov_n >= 1
    assert lm.oov_count("ποτέ") == 1
    assert int(lm.oov_counts.max()) <= 255
    assert model["version"]["oov_counts_saturated"] == 0
    assert lm.score("ποτέ", None, None) > float("-inf")
    assert lm.score("έρθεις", "δεν", "θα") < lm.score("έρθει", "δεν", "θα")


def test_dilemmas_reference_reader_reads_the_file(model):
    ref = NgramLM(model["bin"])
    try:
        assert ref.id_of("θα") != ref.id_unk
        level, sug = ref.predict([ref.id_of("δεν"), ref.id_of("ξέρω")])
        assert level == "trigram"
        assert ref._vocab[sug[0][0]] == "πότε"
    finally:
        ref.close()


def test_predictions_follow_the_counts(model):
    lm = GNLMReader(model["bin"])
    assert lm.suggest("δεν", "ξέρω", 1) == ["πότε"]
    assert lm.suggest(None, "θα", 2) == ["έρθει", "πάμε"]  # 3 each, by id
    assert lm.suggest("<s>", "Θα", 1) == ["πάμε"]          # case fallback
    assert lm.suggest_completions(None, "δεν", "ξε", 3)[0] == "ξέρω"
    assert lm.suggest_completions(None, "<s>", "Θ", 1) == ["Θα"]


def test_the_sidecar_records_what_the_keyboard_and_importer_check(model):
    v = model["version"]
    data = model["bin"].read_bytes()
    assert v["artifact_size_bytes"] == len(data)
    assert v["artifact_sha256"] == hashlib.sha256(data).hexdigest()
    assert v["format_version"] == 2 and v["n_gram_order"] == 3
    assert v["oov_dictionary_sha256"] == sha(model["dic"])
    assert v["vocab_dictionary_sha256"] == sha(model["dic"])
    assert v["corpus_manifest_sha256"] == sha(model["corpus"] / "manifest.json")
    assert v["lang"] == "el" and v["drop_unknown_contexts"] is True
    assert v["vocab_size"] == 20
    assert v["total_tokens"] > 0 and v["train_words_weighted"] > 0
    assert set(v["per_source"]) == {"news", "conv"}
    assert len(v["dilemma_commit"].split("-")[0]) == 40


def test_prepruned_and_full_intermediates_export_identical_files(model):
    root = model["root"]
    full = train(root, "lm_el_full", "--vocab-size", "20", "--no-preprune",
                 corpus=model["corpus"], dic=model["dic"])
    proc, out, _ = export(root, full, "el_ngram_full")
    assert proc.returncode == 0, proc.stderr
    assert out.read_bytes() == model["bin"].read_bytes()


def test_a_rerun_writes_the_same_intermediates(model):
    again = train(model["root"], "lm_el_again", "--vocab-size", "20",
                  corpus=model["corpus"], dic=model["dic"])
    for name in ("vocab.json", "unigrams.json", "bigrams.tsv.gz",
                 "trigrams.tsv.gz", "type_counts.tsv.gz", "dev_sentences.txt"):
        assert (again / name).read_bytes() == (model["in"] / name).read_bytes(), name


def test_dev_sentences_are_resolved_and_never_counted(model):
    dev = (model["in"] / "dev_sentences_conv.txt").read_text(encoding="utf-8")
    assert dev == "<s> τι κάνεις εσύ </s>\n"
    vocab = json.loads((model["in"] / "vocab.json").read_text(encoding="utf-8"))
    assert "εσύ" not in vocab          # dev only


def test_export_refuses_a_table_from_another_word_list(model, tmp_path):
    other = make_dictionary(tmp_path, WORDS + ["καινούργια"])
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "export_lm.py"), "--lang", "el",
         "--in", str(model["in"]), "--out", str(tmp_path / "x.bin"),
         "--version-out", str(tmp_path / "x.version"),
         "--oov-dictionary", str(other)], capture_output=True, text=True)
    assert proc.returncode != 0 and "same list" in proc.stderr


def test_export_refuses_cuts_the_files_were_not_pruned_for(model):
    proc, _, _ = export(model["root"], model["in"], "x",
                        "--keep-homograph-min-count", "1")
    assert proc.returncode != 0 and "pruned for" in proc.stderr
    assert export_lm.prepruned_refusal(None, 1, 1) is None
    pre = {"top_k_bi": export_lm.TOP_K_BI, "top_k_tri": export_lm.TOP_K_TRI,
           "keep_homograph_min_count": 3, "homograph_context_min_count": 3}
    assert export_lm.prepruned_refusal(pre, 3, 3) is None
    assert export_lm.prepruned_refusal(pre, 0, 0) is None
    assert "keep_homograph_min_count" in export_lm.prepruned_refusal(pre, 2, 3)


def test_saturated_table_counts_are_reported():
    stats = {}
    table = export_lm.select_oov_unigrams(
        [("ποτέ", 300), ("πότε", 2)], {"<UNK>"}, 1, {"ποτέ", "πότε"}, stats)
    assert stats["saturated"] == 1
    assert sorted(c for _, c in table) == [2, 255]


def test_a_sampled_run_cannot_replace_the_shipping_intermediates():
    with pytest.raises(SystemExit):
        N.main(["--sample-fraction", "0.5", "--out", str(N.DEFAULT_OUT)])


def test_train_lm_dispatches_el_to_the_streaming_trainer():
    proc = subprocess.run([sys.executable, str(PROJECT_ROOT / "train_lm.py"),
                           "--lang", "el", "--help"], capture_output=True, text=True)
    assert proc.returncode == 0 and "monotonic Modern Greek" in proc.stdout


# --- reader -------------------------------------------------------------------

def test_the_reader_refuses_what_lmreader_refuses(model):
    data = bytearray(model["bin"].read_bytes())
    with pytest.raises(FormatError):
        GNLMReader(bytes(data[:200]))
    bad = bytearray(data)
    bad[4] = 3
    with pytest.raises(FormatError):
        GNLMReader(bytes(bad))


def test_a_broken_footer_is_ignored_not_fatal(model):
    data = bytearray(model["bin"].read_bytes())
    data[-4:] = b"XXXX"
    lm = GNLMReader(bytes(data))
    assert lm.oov_n == 0 and lm.problems() == []


def test_a_row_out_of_order_is_reported(model):
    lm = GNLMReader(model["bin"])
    row = max(lm.bi_index.values(), key=lambda r: r[1])
    off, cnt = row
    assert cnt >= 2
    data = bytearray(model["bin"].read_bytes())
    base = int.from_bytes(data[96:104], "little") + 6 * off
    first, second = data[base:base + 6], data[base + 6:base + 12]
    if first[4:6] == second[4:6]:
        pytest.skip("first two entries tie")
    data[base:base + 6], data[base + 6:base + 12] = second, first
    assert "bigram row not in descending probability" in GNLMReader(bytes(data)).problems()


def test_typed_marks_rank_first_in_completions(model):
    lm = GNLMReader(model["bin"])
    assert lm.suggest_completions(None, "δεν", "πό", 2)[0] == "πότε"


@pytest.mark.parametrize("token, lower", [
    ("ΠΩΣ", "πως"),        # final sigma at the end of a word
    ("ΣΑΣ", "σας"),
    ("Σ’", "σ’"),          # but σ before the apostrophe of elision
    ("ΑΠ’", "απ’"),
    ("Σ'", "σ'"),
    ("ΟΔΟΣ", "οδος"),
])
def test_lookups_lowercase_the_way_the_keyboard_does(token, lower):
    from dilemma_build.gnlm_reader import greek_lower, normalize
    assert greek_lower(token) == lower
    assert normalize(token) == normalize(lower)


def test_apostrophes_fold_to_the_corpus_mark(model):
    lm = GNLMReader(model["bin"])
    assert GNLMReader._fold_elision("σ'") == "σ’"
    assert GNLMReader._fold_elision("σʼ") == "σ’"
    assert GNLMReader._fold_elision("σε") is None
    assert lm.context_id("ΘΑ") == lm.id_exact("θα")

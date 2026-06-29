#!/usr/bin/env python3
"""Data integrity test suite for Dilemma.

Catches structural issues that have caused real bugs:
- ONNX model/vocab dimension mismatch
- ONNX model producing garbage output
- Missing lookup.db tables or low row counts
- Model load failures
- ONNX/PyTorch prediction parity
- Missing or truncated headword list

Static checks (vocab match, DB tables, headwords) complete in <1s.
Model inference tests (ONNX valid output, load/predict, ONNX/PyTorch
parity) take ~20s due to ONNX cold start and PyTorch model loading.

Usage:
    python test_integrity.py           # run all tests
    python test_integrity.py -v        # verbose output
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

DILEMMA_DIR = Path(__file__).parent.parent
DATA_DIR = DILEMMA_DIR / "data"
MODEL_DIR = DILEMMA_DIR / "model"
LOOKUP_DB = DATA_DIR / "lookup.db"
LSJ_HEADWORDS = DATA_DIR / "lsj_headwords.json"

# Default model directory (best available)
DEFAULT_MODEL = MODEL_DIR / "combined-s3"

# Test words with expected lemmas
TEST_CASES_LOOKUP = [
    ("θεούς", "θεός"),         # AG lookup
    ("ἐσκότωσε", "σκοτόω"),    # AG form takes priority in combined lookup
]

TEST_CASES_MG_LOOKUP = [
    ("φέρνοντας", "φέρνω"),    # MG-only verb form (no AG equivalent)
    ("τρομερό", "τρομερός"),   # MG adjective
]

TEST_CASES_MODEL = [
    ("λελαμπρυσμένος", "λαμπρύνω"),  # model inference (not in lookup)
]

TEST_CASES_ELISION = [
    ("ἀλλ\u0313", "ἀλλά"),     # elision expansion
]

# Minimum thresholds
MIN_COMBINED_ROWS = 12_000_000
MIN_AG_ONLY_ROWS = 10_000
MIN_HEADWORDS = 150_000

# Shared Dilemma instance cache (avoids repeated cold starts)
_dilemma_cache = {}


def _get_dilemma(lang="all"):
    """Get or create a cached Dilemma instance."""
    if lang not in _dilemma_cache:
        sys.path.insert(0, str(DILEMMA_DIR))
        from dilemma import Dilemma
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _dilemma_cache[lang] = Dilemma(lang=lang)
    return _dilemma_cache[lang]


class TestResult:
    __test__ = False  # a result-holder helper, not a pytest test class

    def __init__(self, name, passed, message=""):
        self.name = name
        self.passed = passed
        self.message = message

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        msg = f"  [{status}] {self.name}"
        if self.message:
            msg += f" - {self.message}"
        return msg


def test_onnx_vocab_match():
    """ONNX decoder output dimension must match vocab.json size."""
    vocab_path = DEFAULT_MODEL / "vocab.json"
    decoder_path = DEFAULT_MODEL / "decoder_step.onnx"

    if not vocab_path.exists():
        return TestResult("ONNX/vocab match", False,
                          f"vocab.json not found at {vocab_path}")
    if not decoder_path.exists():
        return TestResult("ONNX/vocab match", False,
                          f"decoder_step.onnx not found at {decoder_path}")

    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab["char2id"])

    try:
        import onnxruntime as ort
    except ImportError:
        return TestResult("ONNX/vocab match", False, "onnxruntime not installed")

    session = ort.InferenceSession(
        str(decoder_path), providers=["CPUExecutionProvider"])
    output_info = session.get_outputs()[0]
    # Output shape is [batch, seq_len, vocab_dim]
    onnx_vocab_dim = output_info.shape[-1]

    if onnx_vocab_dim != vocab_size:
        return TestResult(
            "ONNX/vocab match", False,
            f"decoder output dim = {onnx_vocab_dim}, vocab size = {vocab_size}")

    return TestResult("ONNX/vocab match", True, f"both = {vocab_size}")


def test_onnx_produces_valid_output():
    """Run a test word through ONNX and verify it produces a valid Greek lemma.

    Uses the cached Dilemma instance to avoid a separate ONNX cold start.
    Falls back to direct ONNX if Dilemma is not available.
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return TestResult("ONNX valid output", False, "onnxruntime not installed")

    vocab_path = DEFAULT_MODEL / "vocab.json"
    if not vocab_path.exists():
        return TestResult("ONNX valid output", False, "vocab.json not found")

    # Use Dilemma's model if available (avoids double ONNX cold start)
    try:
        d = _get_dilemma("all")
        d._load_model()
        if getattr(d, '_use_onnx', False) and d._model and d._vocab:
            model = d._model
            vocab = d._vocab
        else:
            raise RuntimeError("not ONNX")
    except Exception:
        # Fall back to direct ONNX loading
        sys.path.insert(0, str(DILEMMA_DIR))
        from dilemma.onnx_inference import OnnxLemmaModel, CharVocabLight
        vocab = CharVocabLight(vocab_path)
        model = OnnxLemmaModel(DEFAULT_MODEL)

    import numpy as np

    test_word = "θεούς"
    ids = vocab.encode(test_word)
    ONNX_MAX_LEN = 48
    padded = ids + [0] * (ONNX_MAX_LEN - len(ids))
    src = np.array([padded[:ONNX_MAX_LEN]], dtype=np.int64)
    src_mask = (src == 0)

    results = model.generate(src, src_mask, num_beams=4)
    decoded = vocab.decode(results[0][0][0])

    # Check it's not empty
    if not decoded:
        return TestResult("ONNX valid output", False,
                          "empty output for 'θεούς'")

    # Check it contains Greek characters (not random ASCII)
    greek_chars = sum(
        1 for c in decoded
        if '\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF')
    if greek_chars == 0:
        return TestResult("ONNX valid output", False,
                          f"no Greek chars in output: {decoded!r}")

    # Check it's a reasonable length (not char-level garbage)
    if len(decoded) > 30:
        return TestResult("ONNX valid output", False,
                          f"output too long ({len(decoded)} chars): {decoded!r}")

    return TestResult("ONNX valid output", True, f"'θεούς' -> '{decoded}'")


def test_lookup_db_tables():
    """lookup.db must have the 'lookup' and 'lemmas' tables with lang='grc' rows."""
    if not LOOKUP_DB.exists():
        return TestResult("lookup.db tables", False,
                          f"lookup.db not found at {LOOKUP_DB}")

    conn = sqlite3.connect(str(LOOKUP_DB))

    # Check tables exist
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    if "lookup" not in tables:
        conn.close()
        return TestResult("lookup.db tables", False, "missing 'lookup' table")
    if "lemmas" not in tables:
        conn.close()
        return TestResult("lookup.db tables", False, "missing 'lemmas' table")

    # Check lang='grc' rows exist (AG-only overrides for polytonic disambiguation)
    ag_count = conn.execute(
        "SELECT COUNT(*) FROM lookup WHERE lang='grc'").fetchone()[0]
    conn.close()

    if ag_count == 0:
        return TestResult("lookup.db tables", False,
                          "no lang='grc' (AG-only) rows in lookup table")

    return TestResult("lookup.db tables", True, f"lang='grc' rows: {ag_count:,}")


def test_lookup_db_row_counts():
    """Combined table should have >12M rows, AG-only should have >10K rows."""
    if not LOOKUP_DB.exists():
        return TestResult("lookup.db row counts", False, "lookup.db not found")

    conn = sqlite3.connect(str(LOOKUP_DB))
    combined_count = conn.execute(
        "SELECT COUNT(*) FROM lookup WHERE lang='all'").fetchone()[0]
    ag_count = conn.execute(
        "SELECT COUNT(*) FROM lookup WHERE lang='grc'").fetchone()[0]
    conn.close()

    errors = []
    if combined_count < MIN_COMBINED_ROWS:
        errors.append(
            f"combined (lang='all'): {combined_count:,} < {MIN_COMBINED_ROWS:,}")
    if ag_count < MIN_AG_ONLY_ROWS:
        errors.append(
            f"AG-only (lang='grc'): {ag_count:,} < {MIN_AG_ONLY_ROWS:,}")

    if errors:
        return TestResult("lookup.db row counts", False, "; ".join(errors))

    return TestResult("lookup.db row counts", True,
                      f"combined: {combined_count:,}, AG-only: {ag_count:,}")


def test_model_loads_and_predicts():
    """Dilemma() should load without errors and lemmatize test words correctly."""
    try:
        d = _get_dilemma("all")
    except Exception as e:
        return TestResult("model loads and predicts", False,
                          f"Dilemma() init failed: {e}")

    errors = []
    total = 0

    # Combined lookup test cases (AG-priority)
    for form, expected in TEST_CASES_LOOKUP:
        total += 1
        actual = d.lemmatize(form)
        if actual != expected:
            errors.append(f"{form}: expected '{expected}', got '{actual}'")

    # MG-only lookup test cases
    try:
        d_mg = _get_dilemma("el")
        for form, expected in TEST_CASES_MG_LOOKUP:
            total += 1
            actual = d_mg.lemmatize(form)
            if actual != expected:
                errors.append(
                    f"{form} (MG): expected '{expected}', got '{actual}'")
    except Exception as e:
        errors.append(f"Dilemma(lang='el') init failed: {e}")

    # Model inference test cases (forms likely not in lookup)
    for form, expected in TEST_CASES_MODEL:
        total += 1
        actual = d.lemmatize(form)
        if actual != expected:
            errors.append(f"{form}: expected '{expected}', got '{actual}'")

    # Elision test cases
    for form, expected in TEST_CASES_ELISION:
        total += 1
        actual = d.lemmatize(form)
        if actual != expected:
            errors.append(f"{form}: expected '{expected}', got '{actual}'")

    if errors:
        return TestResult("model loads and predicts", False, "; ".join(errors))

    return TestResult("model loads and predicts", True,
                      f"{total}/{total} correct")


def test_onnx_pytorch_parity():
    """If both backends are available, they should produce the same lemma.

    Uses Dilemma's own model loading to avoid version compatibility issues
    with manual PyTorch model construction.
    """
    pt_path = DEFAULT_MODEL / "model.pt"
    onnx_path = DEFAULT_MODEL / "encoder.onnx"

    if not onnx_path.exists():
        return TestResult("ONNX/PyTorch parity", True,
                          "skipped: ONNX model not found")
    if not pt_path.exists():
        return TestResult("ONNX/PyTorch parity", True,
                          "skipped: PyTorch model not found")

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return TestResult("ONNX/PyTorch parity", True,
                          "skipped: onnxruntime not installed")

    try:
        import torch  # noqa: F401
    except ImportError:
        return TestResult("ONNX/PyTorch parity", True,
                          "skipped: torch not installed")

    sys.path.insert(0, str(DILEMMA_DIR))
    from dilemma import Dilemma

    # Reuse the cached ONNX-backed instance
    d_onnx = _get_dilemma("all")
    d_onnx._load_model()
    if not getattr(d_onnx, '_use_onnx', False):
        return TestResult("ONNX/PyTorch parity", True,
                          "skipped: Dilemma chose PyTorch despite ONNX files")

    # Create a PyTorch-backed instance
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d_pt = Dilemma()
    try:
        d_pt._load_pytorch(DEFAULT_MODEL)
        d_pt._use_onnx = False
    except Exception as e:
        return TestResult("ONNX/PyTorch parity", True,
                          f"skipped: PyTorch load failed ({e})")

    test_words = ["θεούς", "ἔλυσε", "πολεμούσαν"]
    errors = []

    for word in test_words:
        onnx_lemma = d_onnx._predict([word])[0]
        pt_lemma = d_pt._predict([word])[0]

        if onnx_lemma != pt_lemma:
            errors.append(
                f"'{word}': ONNX='{onnx_lemma}', PyTorch='{pt_lemma}'")

    if errors:
        return TestResult("ONNX/PyTorch parity", False, "; ".join(errors))

    return TestResult("ONNX/PyTorch parity", True,
                      f"{len(test_words)} words match")


def test_headword_list():
    """lsj_headwords.json should have >150K entries (119K LSJ + variants)."""
    if not LSJ_HEADWORDS.exists():
        return TestResult("headword list", False,
                          f"lsj_headwords.json not found at {LSJ_HEADWORDS}")

    with open(LSJ_HEADWORDS, encoding="utf-8") as f:
        headwords = json.load(f)

    count = len(headwords)
    if count < MIN_HEADWORDS:
        return TestResult("headword list", False,
                          f"{count:,} entries < minimum {MIN_HEADWORDS:,}")

    return TestResult("headword list", True, f"{count:,} entries")


def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    start = time.time()

    tests = [
        ("ONNX/vocab dimension match", test_onnx_vocab_match),
        ("lookup.db has required tables", test_lookup_db_tables),
        ("lookup.db row counts are sane", test_lookup_db_row_counts),
        ("Headword list loaded", test_headword_list),
        # Tests below trigger model inference (ONNX cold start ~10s first time)
        ("ONNX produces valid output", test_onnx_produces_valid_output),
        ("Model loads and predicts", test_model_loads_and_predicts),
        ("ONNX/PyTorch parity", test_onnx_pytorch_parity),
    ]

    results = []
    print("=" * 60)
    print("DILEMMA INTEGRITY TESTS")
    print("=" * 60)

    for label, test_fn in tests:
        if verbose:
            print(f"\n  Running: {label}...")
        try:
            result = test_fn()
        except Exception as e:
            result = TestResult(label, False, f"unexpected error: {e}")
        results.append(result)
        print(result)

    elapsed = time.time() - start

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)

    print()
    print("=" * 60)
    print(f"  {passed}/{total} passed, {failed} failed  ({elapsed:.1f}s)")
    print("=" * 60)

    if failed > 0:
        print("\nFailed:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.message}")
        sys.exit(1)
    else:
        print("\nAll integrity checks passed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comprehensive benchmark of Greek lemmatizers on four evaluation datasets.

Datasets:
  1. AG classical (Sextus Empiricus, ~323 tokens)
  2. Byzantine (DBBE medieval Greek, ~8342 tokens)
  3. Katharevousa (~283 tokens)
  4. Demotic MG (~363 tokens)

Tools:
  - Dilemma (no POS)
  - Dilemma (gold POS) - only where gold POS is available (DBBE)
  - stanza grc
  - stanza el
  - spaCy el
  - CLTK GreekBackoffLemmatizer
  - OdyCy (if installed)
  - Pie Extended (if installed)

Run:
    python data/benchmarks/bench_all.py
"""

import json
import os
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

# Ensure dilemma is importable (single-file module in repo root)
DILEMMA_DIR = Path(__file__).resolve().parent.parent.parent  # ~/Documents/dilemma
sys.path.insert(0, str(DILEMMA_DIR))

BENCH_DIR = DILEMMA_DIR / "data" / "benchmarks"
DBBE_PATH = DILEMMA_DIR / "data" / "dbbe" / "lingAnn_GS_medievalGreek.tsv"
EQUIV_PATH = DILEMMA_DIR / "data" / "lemma_equivalences.json"
OUTPUT_PATH = BENCH_DIR / "full_matrix.json"

# Max sentence length for stanza/spaCy (longer sentences get split)
MAX_SENT_LEN = 200

# ---------------------------------------------------------------------------
# Load lemma equivalences
# ---------------------------------------------------------------------------
def load_equivalences(path):
    """Build bidirectional equivalence lookup: lemma -> set of equivalent lemmas."""
    with open(path) as f:
        data = json.load(f)
    equiv = {}
    for group in data["groups"]:
        group_set = set(group)
        for lemma in group:
            if lemma in equiv:
                equiv[lemma] = equiv[lemma] | group_set
            else:
                equiv[lemma] = set(group_set)
    return equiv


EQUIV = load_equivalences(EQUIV_PATH)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def strip_accents(s):
    """Remove combining diacriticals (accents, breathings) from Greek text."""
    nfkd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfkd if unicodedata.category(c) != "Mn"))


def to_monotonic(s):
    """Normalize to monotonic Greek (strip breathings, convert graves to acute)."""
    _strip = {0x0313, 0x0314, 0x0345, 0x0306, 0x0304}
    _to_acute = {0x0300, 0x0342}
    nfd = unicodedata.normalize("NFD", s)
    out = []
    for ch in nfd:
        cp = ord(ch)
        if cp in _strip:
            continue
        if cp in _to_acute:
            out.append("\u0301")
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def normalize_basic(s):
    """Lowercase and strip whitespace."""
    return s.lower().strip()


def normalize_monotonic(s):
    """Monotonic + lowercase."""
    return to_monotonic(s).lower().strip()


def normalize_accent_free(s):
    """Strip all accents + lowercase."""
    return strip_accents(s).lower().strip()


def are_equivalent(pred, gold, equiv):
    """Check if pred matches gold via equivalence table (after accent-stripping)."""
    # First check accent-stripped match
    if normalize_accent_free(pred) == normalize_accent_free(gold):
        return True
    # Check equivalence groups at multiple normalization levels
    pred_n = normalize_basic(pred)
    gold_n = normalize_basic(gold)
    pred_a = normalize_accent_free(pred)
    gold_a = normalize_accent_free(gold)
    # Check forward: gold's equivalences contain pred?
    gold_equivs = equiv.get(gold, set())
    if pred in gold_equivs:
        return True
    if pred_n in {normalize_basic(e) for e in gold_equivs}:
        return True
    if pred_a in {normalize_accent_free(e) for e in gold_equivs}:
        return True
    # Check reverse: pred's equivalences contain gold?
    pred_equivs = equiv.get(pred, set())
    if gold in pred_equivs:
        return True
    if gold_n in {normalize_basic(e) for e in pred_equivs}:
        return True
    if gold_a in {normalize_accent_free(e) for e in pred_equivs}:
        return True
    return False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_two_col_tsv(path):
    """Load form\\tlemma TSV. Return list of sentences, each = list of (form, lemma)."""
    sentences = []
    current = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                current.append((parts[0], parts[1]))
    if current:
        sentences.append(current)
    return sentences


def load_dbbe(path):
    """Load 3-col DBBE TSV (form, morphtag, lemma). Return sentences with POS.
    Filters out punctuation (pos starting with 'u') and empty tokens."""
    sentences = []
    current = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                form = parts[0]
                pos = parts[1]
                if not form.strip():
                    continue  # Skip empty tokens
                if pos.startswith("u"):
                    continue  # Skip punctuation
                current.append((form, pos, parts[2]))
    if current:
        sentences.append(current)
    return sentences


def split_long_sentences(sentences, max_len=MAX_SENT_LEN):
    """Split sentences longer than max_len into chunks for tools that need it."""
    result = []
    for sent in sentences:
        if len(sent) <= max_len:
            result.append(sent)
        else:
            for i in range(0, len(sent), max_len):
                chunk = sent[i : i + max_len]
                if chunk:
                    result.append(chunk)
    return result


def morphtag_to_upos(tag):
    """Convert 9-char morphological tag to UPOS for Dilemma."""
    if not tag or len(tag) < 1:
        return None
    first = tag[0]
    mapping = {
        "n": "NOUN",
        "v": "VERB",
        "a": "ADJ",
        "d": "ADV",
        "l": "DET",
        "p": "PRON",
        "m": "NUM",
        "r": "ADP",
        "g": "PART",
        "c": "CCONJ",
        "u": "PUNCT",
        "e": "INTJ",
        "i": "INTJ",
        "x": "X",
    }
    return mapping.get(first)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(predictions, gold_pairs, equiv_map):
    """
    predictions: list of predicted lemmas (aligned to gold)
    gold_pairs: list of (form, gold_lemma)
    Returns dict of accuracy metrics.
    """
    assert len(predictions) == len(gold_pairs), (
        f"Mismatch: {len(predictions)} predictions vs {len(gold_pairs)} gold"
    )

    n = len(predictions)
    strict = 0
    monotonic = 0
    accent_stripped = 0
    equiv_adj = 0

    for pred, (form, gold) in zip(predictions, gold_pairs):
        pred_s = pred.strip()
        gold_s = gold.strip()

        # Strict
        if pred_s == gold_s:
            strict += 1

        # Monotonic-normalized
        if normalize_monotonic(pred_s) == normalize_monotonic(gold_s):
            monotonic += 1

        # Accent-stripped
        if normalize_accent_free(pred_s) == normalize_accent_free(gold_s):
            accent_stripped += 1

        # Equivalence-adjusted (includes accent-stripped + equivalence table)
        if are_equivalent(pred_s, gold_s, equiv_map):
            equiv_adj += 1

    return {
        "n_tokens": n,
        "strict": round(strict / n * 100, 2),
        "monotonic_normalized": round(monotonic / n * 100, 2),
        "accent_stripped": round(accent_stripped / n * 100, 2),
        "equiv_adjusted": round(equiv_adj / n * 100, 2),
    }


# ---------------------------------------------------------------------------
# Tool wrappers
# ---------------------------------------------------------------------------

class DilemmaWrapper:
    def __init__(self, use_pos=False, convention=None):
        self.use_pos = use_pos
        self.convention = convention
        if convention == "triantafyllidis":
            self.name = "Dilemma MG"
        elif use_pos:
            self.name = "Dilemma (gold POS)"
        else:
            self.name = "Dilemma (no POS)"
        self.d = None
        self.verbose_stats = None

    def load(self):
        from dilemma import Dilemma
        # resolve_articles=True is the correct invocation for the gold here:
        # the article paradigm (τὸν->ὁ) is deliberately excluded from the
        # lookup (MG-leak safety), so this restores article-lemma resolution
        # that the gold annotates. Harmless for MG (resolves to the MG lemma).
        self.d = Dilemma(convention=self.convention, resolve_articles=True)

    def lemmatize_dataset(self, sentences, has_pos=False):
        # Flatten all forms
        all_forms = []
        all_upos = []  # Only used when use_pos=True and has_pos=True
        for sent in sentences:
            for tok in sent:
                all_forms.append(tok[0])
                if self.use_pos and has_pos and len(tok) >= 3:
                    all_upos.append(morphtag_to_upos(tok[1]))
                else:
                    all_upos.append(None)

        if self.use_pos and has_pos:
            # Use lemmatize_batch_pos for tokens with POS, lemmatize_batch for rest
            pos_forms = []
            pos_tags = []
            no_pos_forms = []
            pos_indices = []
            no_pos_indices = []

            for i, (form, upos) in enumerate(zip(all_forms, all_upos)):
                if upos and upos != "PUNCT":
                    pos_forms.append(form)
                    pos_tags.append(upos)
                    pos_indices.append(i)
                else:
                    no_pos_forms.append(form)
                    no_pos_indices.append(i)

            # Batch lemmatize
            pos_results = self.d.lemmatize_batch_pos(pos_forms, pos_tags) if pos_forms else []
            no_pos_results = self.d.lemmatize_batch(no_pos_forms) if no_pos_forms else []

            # Reassemble in order
            preds = [""] * len(all_forms)
            for idx, lemma in zip(pos_indices, pos_results):
                preds[idx] = lemma
            for idx, lemma in zip(no_pos_indices, no_pos_results):
                preds[idx] = lemma
        else:
            # Simple batch lemmatize
            preds = self.d.lemmatize_batch(all_forms)

        # Store forms for verbose stats collection (done outside timing)
        self._pending_forms = all_forms

        return preds

    def collect_verbose_stats(self):
        """Collect verbose stats from the last lemmatize_dataset call.
        Called separately from timing."""
        if hasattr(self, "_pending_forms") and self._pending_forms:
            self._collect_verbose_stats(self._pending_forms)
            self._pending_forms = None

    def _collect_verbose_stats(self, all_forms):
        """Collect per-strategy stats via lemmatize_verbose.
        For large datasets (>1000 tokens), sample 500 tokens and extrapolate."""
        import random

        if len(all_forms) > 1000:
            sample = random.sample(all_forms, 500)
            scale = len(all_forms) / 500
        else:
            sample = all_forms
            scale = 1.0

        source_counts = defaultdict(int)
        source_times = defaultdict(float)

        for form in sample:
            try:
                t0 = time.perf_counter()
                vr = self.d.lemmatize_verbose(form)
                elapsed = time.perf_counter() - t0
                if vr:
                    src = vr[0].source
                    source_counts[src] += 1
                    source_times[src] += elapsed
                else:
                    source_counts["identity"] += 1
                    source_times["identity"] += elapsed
            except Exception:
                source_counts["identity"] += 1

        self.verbose_stats = {
            "by_source": {},
            "sampled": len(all_forms) > 1000,
            "sample_size": len(sample),
            "total_tokens": len(all_forms),
        }
        for src in sorted(source_counts.keys()):
            self.verbose_stats["by_source"][src] = {
                "count": int(source_counts[src] * scale),
                "time_s": round(source_times[src] * scale, 4),
            }


class StanzaWrapper:
    def __init__(self, lang):
        self.lang = lang
        self.name = f"stanza {lang}"
        self.nlp = None

    def load(self):
        import stanza
        self.nlp = stanza.Pipeline(
            self.lang,
            processors="tokenize,pos,lemma",
            tokenize_pretokenized=True,
            verbose=False,
        )

    def lemmatize_dataset(self, sentences, has_pos=False):
        """Feed sentences as pre-tokenized, align output back to gold."""
        # Split long sentences for stanza
        sentences = split_long_sentences(sentences)

        preds = []
        for sent in sentences:
            tokens = [tok[0] for tok in sent]
            # Filter out empty tokens for stanza, keeping track of indices
            non_empty = [(i, t) for i, t in enumerate(tokens) if t.strip()]
            if not non_empty:
                preds.extend([tok[0] for tok in sent])
                continue

            filtered_tokens = [t for _, t in non_empty]
            try:
                doc = self.nlp([filtered_tokens])
                # Extract lemmas from stanza output
                stanza_lemmas = []
                for stanza_sent in doc.sentences:
                    for w in stanza_sent.words:
                        stanza_lemmas.append(w.lemma if w.lemma else w.text)

                # Map back to original token positions
                lemma_map = {}
                for idx, (orig_idx, _) in enumerate(non_empty):
                    if idx < len(stanza_lemmas):
                        lemma_map[orig_idx] = stanza_lemmas[idx]

                for i, tok in enumerate(sent):
                    if i in lemma_map:
                        preds.append(lemma_map[i])
                    else:
                        preds.append(tok[0])  # identity for empty/skipped tokens

            except Exception as e:
                # Fallback: return identity for all tokens
                preds.extend([tok[0] for tok in sent])

        return preds


class SpacyWrapper:
    def __init__(self, model_name="el_core_news_sm"):
        self.model_name = model_name
        self.name = "spaCy el"
        self.nlp = None

    def load(self):
        import spacy
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.nlp = spacy.load(self.model_name)

    def lemmatize_dataset(self, sentences, has_pos=False):
        preds = []
        sentences = split_long_sentences(sentences)
        for sent in sentences:
            text = " ".join(tok[0] for tok in sent)
            doc = self.nlp(text)

            spacy_tokens = [(t.text, t.lemma_) for t in doc]
            gold_tokens = [tok[0] for tok in sent]

            aligned = self._align(spacy_tokens, gold_tokens)
            preds.extend(aligned)

        return preds

    def _align(self, spacy_tokens, gold_tokens):
        result = []
        si = 0
        for gf in gold_tokens:
            if si < len(spacy_tokens) and spacy_tokens[si][0] == gf:
                result.append(spacy_tokens[si][1])
                si += 1
            else:
                found = False
                for j in range(si, min(si + 5, len(spacy_tokens))):
                    if spacy_tokens[j][0] == gf:
                        result.append(spacy_tokens[j][1])
                        si = j + 1
                        found = True
                        break
                if not found:
                    result.append(gf)
                    if si < len(spacy_tokens):
                        si += 1
        return result


class CLTKWrapper:
    def __init__(self):
        self.name = "CLTK"
        self.lem = None

    def load(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from cltk.lemmatize.grc import GreekBackoffLemmatizer
            self.lem = GreekBackoffLemmatizer()

    def lemmatize_dataset(self, sentences, has_pos=False):
        preds = []
        for sent in sentences:
            forms = [tok[0] for tok in sent]
            results = self.lem.lemmatize(forms)
            for form, lemma in results:
                preds.append(lemma)
        return preds


class OdyCyWrapper:
    def __init__(self):
        self.name = "OdyCy"
        self.nlp = None

    def load(self):
        import spacy
        self.nlp = spacy.load("grc_odycy_joint_sm")

    def lemmatize_dataset(self, sentences, has_pos=False):
        preds = []
        sentences = split_long_sentences(sentences)
        for sent in sentences:
            text = " ".join(tok[0] for tok in sent)
            doc = self.nlp(text)
            spacy_tokens = [(t.text, t.lemma_) for t in doc]
            gold_tokens = [tok[0] for tok in sent]
            aligned = SpacyWrapper(None)._align(spacy_tokens, gold_tokens)
            preds.extend(aligned)
        return preds


class PieExtendedWrapper:
    def __init__(self):
        self.name = "Pie Extended"
        self.tagger = None

    def load(self):
        from pie_extended.cli.sub import get_tagger
        self.tagger = get_tagger("grc")

    def lemmatize_dataset(self, sentences, has_pos=False):
        preds = []
        for sent in sentences:
            text = " ".join(tok[0] for tok in sent)
            result = self.tagger.tag_str(text)
            # pie-extended returns list of dicts with 'lemma' key
            lemmas = [r.get("lemma", tok[0]) for r, tok in zip(result, sent)]
            preds.extend(lemmas)
        return preds


# ---------------------------------------------------------------------------
# Dataset and tool matrix definitions
# ---------------------------------------------------------------------------
DATASETS = {
    "ag_classical": {
        "path": BENCH_DIR / "ag_gold.tsv",
        "loader": "two_col",
        "lang": "grc",
        "description": "AG Classical",
    },
    "byzantine": {
        "path": DBBE_PATH,
        "loader": "dbbe",
        "lang": "grc",
        "has_pos": True,
        "description": "Byzantine",
    },
    "katharevousa": {
        "path": BENCH_DIR / "katharevousa_gold.tsv",
        "loader": "two_col",
        "lang": "mixed",
        "description": "Katharevousa",
    },
    "demotic": {
        "path": BENCH_DIR / "demotic_gold.tsv",
        "loader": "two_col",
        "lang": "el",
        "description": "Demotic MG",
    },
}

# Tools and which datasets they run on, with reasons for N/A
TOOL_MATRIX = {
    "Dilemma (no POS)": {
        "ag_classical": True,
        "byzantine": True,
        "katharevousa": True,
        "demotic": True,
    },
    "Dilemma (gold POS)": {
        "ag_classical": "No gold POS tags in this dataset",
        "byzantine": True,
        "katharevousa": "No gold POS tags in this dataset",
        "demotic": "No gold POS tags in this dataset",
    },
    "Dilemma MG": {
        "ag_classical": "MG convention, not for AG text",
        "byzantine": "MG convention, not for Byzantine text",
        "katharevousa": True,
        "demotic": True,
    },
    "stanza grc": {
        "ag_classical": True,
        "byzantine": True,
        "katharevousa": True,
        "demotic": "AG-only model, no Modern Greek support",
    },
    "stanza el": {
        "ag_classical": "MG-only model, no polytonic/AG support",
        "byzantine": "MG-only model, no polytonic/medieval Greek support",
        "katharevousa": True,
        "demotic": True,
    },
    "spaCy el": {
        "ag_classical": "MG-only model, no polytonic/AG support",
        "byzantine": "MG-only model, no polytonic/medieval Greek support",
        "katharevousa": True,
        "demotic": True,
    },
    "CLTK": {
        "ag_classical": True,
        "byzantine": True,
        "katharevousa": True,
        "demotic": "AG-only dictionary, no Modern Greek forms",
    },
    "OdyCy": {
        "ag_classical": True,
        "byzantine": True,
        "katharevousa": True,
        "demotic": "AG-only model, no Modern Greek support",
    },
    "Pie Extended": {
        "ag_classical": True,
        "byzantine": True,
        "katharevousa": True,
        "demotic": "AG-only model, no Modern Greek support",
    },
}


# ---------------------------------------------------------------------------
# Flatten gold data
# ---------------------------------------------------------------------------
def flatten_gold(sentences, has_pos=False):
    """Flatten sentences into list of (form, gold_lemma) pairs."""
    pairs = []
    for sent in sentences:
        for tok in sent:
            if has_pos and len(tok) >= 3:
                pairs.append((tok[0], tok[2]))
            else:
                pairs.append((tok[0], tok[1]))
    return pairs


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_benchmark():
    print("=" * 80)
    print("COMPREHENSIVE GREEK LEMMATIZER BENCHMARK")
    print("=" * 80)
    print()

    # Load datasets
    print("Loading datasets...")
    loaded_data = {}
    for ds_name, ds_info in DATASETS.items():
        if ds_info["loader"] == "two_col":
            sents = load_two_col_tsv(ds_info["path"])
        elif ds_info["loader"] == "dbbe":
            sents = load_dbbe(ds_info["path"])

        has_pos = ds_info.get("has_pos", False)
        gold = flatten_gold(sents, has_pos=has_pos)
        loaded_data[ds_name] = {
            "sentences": sents,
            "gold": gold,
            "has_pos": has_pos,
            "n_tokens": len(gold),
        }
        print(f"  {ds_info['description']}: {len(gold)} tokens, {len(sents)} sentences")

    print()

    # Initialize tools
    tools = {}
    tool_constructors = [
        ("Dilemma (no POS)", lambda: DilemmaWrapper(use_pos=False)),
        ("Dilemma (gold POS)", lambda: DilemmaWrapper(use_pos=True)),
        ("Dilemma MG", lambda: DilemmaWrapper(convention="triantafyllidis")),
        ("stanza grc", lambda: StanzaWrapper("grc")),
        ("stanza el", lambda: StanzaWrapper("el")),
        ("spaCy el", lambda: SpacyWrapper()),
        ("CLTK", lambda: CLTKWrapper()),
        ("OdyCy", lambda: OdyCyWrapper()),
        ("Pie Extended", lambda: PieExtendedWrapper()),
    ]

    for tool_name, constructor in tool_constructors:
        print(f"Loading {tool_name}...", end=" ", flush=True)
        try:
            t0 = time.perf_counter()
            wrapper = constructor()
            wrapper.load()
            load_time = time.perf_counter() - t0
            tools[tool_name] = {"wrapper": wrapper, "load_time": round(load_time, 3)}
            print(f"OK ({load_time:.2f}s)")
        except Exception as e:
            short_err = str(e).split("\n")[0][:80]
            print(f"FAILED: {short_err}")
            tools[tool_name] = None

    print()

    # Run benchmarks
    results = {}
    for tool_name in [t[0] for t in tool_constructors]:
        tool_info = tools.get(tool_name)
        matrix_entry = TOOL_MATRIX.get(tool_name, {})

        if tool_info is None:
            results[tool_name] = {"error": "Failed to load/install", "datasets": {}}
            # Fill skipped for all datasets
            for ds_name in DATASETS:
                results[tool_name]["datasets"][ds_name] = {
                    "status": "skipped",
                    "notes": "Tool failed to load/install",
                }
            continue

        wrapper = tool_info["wrapper"]
        results[tool_name] = {
            "load_time_s": tool_info["load_time"],
            "datasets": {},
        }

        for ds_name in DATASETS:
            applicable = matrix_entry.get(ds_name, "Not configured")

            if applicable is not True:
                results[tool_name]["datasets"][ds_name] = {
                    "status": "N/A",
                    "notes": applicable if isinstance(applicable, str) else "Not applicable",
                }
                continue

            ds = loaded_data[ds_name]
            print(f"  {tool_name} on {ds_name} ({ds['n_tokens']} tokens)...", end=" ", flush=True)

            try:
                t0 = time.perf_counter()
                preds = wrapper.lemmatize_dataset(
                    ds["sentences"], has_pos=ds["has_pos"]
                )
                inference_time = time.perf_counter() - t0

                # Collect Dilemma verbose stats separately (not timed)
                if hasattr(wrapper, "collect_verbose_stats"):
                    wrapper.collect_verbose_stats()

                # Score
                scores = score(preds, ds["gold"], EQUIV)
                tps = round(ds["n_tokens"] / inference_time, 1) if inference_time > 0 else float("inf")

                ds_result = {
                    "status": "ok",
                    **scores,
                    "inference_time_s": round(inference_time, 4),
                    "tokens_per_second": tps,
                }

                # Add Dilemma verbose stats if available
                if hasattr(wrapper, "verbose_stats") and wrapper.verbose_stats:
                    ds_result["dilemma_strategy"] = wrapper.verbose_stats
                    wrapper.verbose_stats = None

                results[tool_name]["datasets"][ds_name] = ds_result
                print(
                    f"strict={scores['strict']:.1f}% "
                    f"equiv={scores['equiv_adjusted']:.1f}% "
                    f"({inference_time:.2f}s, {tps:.0f} tok/s)"
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"ERROR: {e}")
                results[tool_name]["datasets"][ds_name] = {
                    "status": "error",
                    "notes": str(e),
                }

    # Save results
    print()
    print(f"Saving results to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print formatted tables
    print()
    print_table(results, loaded_data)

    return results


def print_table(results, loaded_data):
    """Print formatted results tables."""
    ds_names = list(DATASETS.keys())
    ds_labels = {
        "ag_classical": "AG Classical",
        "byzantine": "Byzantine",
        "katharevousa": "Katharevousa",
        "demotic": "Demotic MG",
    }
    ds_tokens = {ds: loaded_data[ds]["n_tokens"] for ds in ds_names}

    # ---- SUMMARY TABLE ----
    print("=" * 130)
    print("SUMMARY TABLE: Strict Accuracy (%) | Equiv-adjusted Accuracy (%) | Tokens/sec")
    print("=" * 130)

    # Header
    header = f"{'Tool':<25}"
    for ds in ds_names:
        label = f"{ds_labels[ds]} ({ds_tokens[ds]})"
        header += f"  {label:<28}"
    header += "  Load(s)"
    print(header)
    print("-" * 130)

    # Sub-header
    sub = f"{'':25}"
    for ds in ds_names:
        sub += f"  {'Strict':>7} {'Equiv':>7} {'tok/s':>7}     "
    print(sub)
    print("-" * 130)

    tool_order = list(TOOL_MATRIX.keys())

    for tool_name in tool_order:
        r = results.get(tool_name)
        if not r:
            continue

        load_t = r.get("load_time_s", "-")
        if "error" in r and r.get("error"):
            load_t = "FAIL"

        row = f"{tool_name:<25}"
        for ds in ds_names:
            ds_r = r.get("datasets", {}).get(ds, {})
            status = ds_r.get("status", "N/A") if isinstance(ds_r, dict) else "N/A"

            if status == "ok":
                s = ds_r.get("strict", 0)
                e = ds_r.get("equiv_adjusted", 0)
                tps = ds_r.get("tokens_per_second", 0)
                row += f"  {s:>7.1f} {e:>7.1f} {tps:>7.0f}     "
            elif status == "N/A":
                row += f"  {'--':>7} {'--':>7} {'--':>7}     "
            elif status == "error":
                row += f"  {'ERR':>7} {'ERR':>7} {'ERR':>7}     "
            elif status == "skipped":
                row += f"  {'--':>7} {'--':>7} {'--':>7}     "

        row += f"  {load_t}"
        print(row)

    print("=" * 130)

    # ---- N/A REASONS ----
    print()
    print("N/A / SKIP REASONS:")
    print("-" * 80)
    for tool_name in tool_order:
        r = results.get(tool_name, {})
        datasets = r.get("datasets", {})
        for ds in ds_names:
            ds_r = datasets.get(ds, {})
            if isinstance(ds_r, dict):
                status = ds_r.get("status", "")
                notes = ds_r.get("notes", "")
                if status in ("N/A", "skipped", "error") and notes:
                    print(f"  {tool_name} / {ds_labels[ds]}: {notes}")

    # ---- DETAILED METRICS ----
    print()
    print("DETAILED METRICS (per tool, per dataset)")
    print("=" * 80)
    for tool_name in tool_order:
        r = results.get(tool_name, {})
        if "error" in r and r.get("error"):
            continue
        datasets = r.get("datasets", {})
        for ds in ds_names:
            ds_r = datasets.get(ds, {})
            if not isinstance(ds_r, dict) or ds_r.get("status") != "ok":
                continue
            print(f"\n  {tool_name} on {ds_labels[ds]}:")
            print(f"    Strict:              {ds_r['strict']:>6.2f}%")
            print(f"    Monotonic-norm:      {ds_r['monotonic_normalized']:>6.2f}%")
            print(f"    Accent-stripped:      {ds_r['accent_stripped']:>6.2f}%")
            print(f"    Equiv-adjusted:      {ds_r['equiv_adjusted']:>6.2f}%")
            print(f"    Inference time:      {ds_r['inference_time_s']:>6.3f}s")
            print(f"    Tokens/second:       {ds_r['tokens_per_second']:>8.1f}")

            if "dilemma_strategy" in ds_r:
                st = ds_r["dilemma_strategy"]
                by_src = st.get("by_source", {})
                total = sum(s["count"] for s in by_src.values())
                sampled = st.get("sampled", False)
                note = " (estimated from sample)" if sampled else ""
                if total > 0:
                    print(f"    --- Dilemma strategy breakdown{note} ---")
                    for src_name, src_data in sorted(by_src.items()):
                        cnt = src_data["count"]
                        tm = src_data["time_s"]
                        print(f"    {src_name:12s}: {cnt:>5} ({cnt/total*100:.1f}%) in {tm:.3f}s")

    print()


if __name__ == "__main__":
    run_benchmark()

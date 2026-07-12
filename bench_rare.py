#!/usr/bin/env python3
"""Frequency-stratified benchmark for Dilemma.

Evaluates lemmatization accuracy across corpus-frequency bins using
GLAUx+Diorisis frequencies (27M combined tokens). For each bin, checks
whether Dilemma's output is a valid LSJ/Wiktionary headword.

Frequency bins:
  High:   1000+ corpus occurrences
  Medium: 100-999
  Low:    10-99
  Rare:   1-9
  Unseen: not in GLAUx+Diorisis at all (true OOV)

Test texts:
  - Xenophon, Cyropaedia (Gorman treebank gold)
  - Herodotus, Histories (Perseus canonical-greekLit TEI, tlg0016.tlg001,
    CC BY-SA; point HERODOTUS_TEI at your copy of the grc XML)
  - Sextus Empiricus, Pyrrhoniae Hypotyposes (AG Classical benchmark)

Reference: Novak & Cavar (2025), "Corpus Frequencies in Morphological
Inflection: Do They Matter?", ITAT 2025 (arXiv:2510.23131).
"""

import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

DILEMMA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DILEMMA_DIR))

DATA_DIR = DILEMMA_DIR / "data"
CORPUS_FREQ_PATH = DATA_DIR / "corpus_freq.json"
LSJ_HEADWORDS_PATH = DATA_DIR / "lsj_headwords.json"
AG_HEADWORDS_PATH = DATA_DIR / "ag_headwords.json"
EQUIV_PATH = DATA_DIR / "lemma_equivalences.json"

# Treebank paths
GORMAN_DIR = DATA_DIR / "treebanks" / "Gorman"

# Sextus Empiricus benchmark
SEXTUS_GOLD = DATA_DIR / "benchmarks" / "ag_gold.tsv"

SKIP_UPOS = {"PUNCT", "NUM", "X", "SYM"}

FREQ_BINS = [
    ("High (1000+)", 1000, float("inf")),
    ("Medium (100-999)", 100, 999),
    ("Low (10-99)", 10, 99),
    ("Rare (1-9)", 1, 9),
    ("Unseen (0)", 0, 0),
]


def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn").lower()


def grave_to_acute(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    out = nfd.replace("\u0300", "\u0301")
    return unicodedata.normalize("NFC", out)


def is_capitalized(word: str) -> bool:
    for ch in word:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):
            return cat == "Lu"
    return False


def load_corpus_freqs() -> dict[str, int]:
    """Load form frequencies from corpus_freq.json.

    Keys are accent-stripped forms, values are total occurrence counts.
    """
    with open(CORPUS_FREQ_PATH) as f:
        data = json.load(f)
    forms = data["forms"]
    return {form: counts[0] for form, counts in forms.items()}


def load_headwords() -> set[str]:
    with open(LSJ_HEADWORDS_PATH) as f:
        lsj = json.load(f)
    with open(AG_HEADWORDS_PATH) as f:
        ag = json.load(f)
    headwords = set()
    for hw in lsj + ag:
        hw = hw.strip()
        if hw:
            headwords.add(hw)
            headwords.add(strip_accents(hw))
    return headwords


def load_equivalences() -> dict[str, set[str]]:
    with open(EQUIV_PATH) as f:
        data = json.load(f)
    equiv = {}
    for group in data["groups"]:
        group_set = set(group)
        for lemma in group:
            equiv[lemma] = equiv.get(lemma, set()) | group_set
    return equiv


def is_valid_lemma(lemma: str, headwords: set[str]) -> bool:
    if lemma in headwords:
        return True
    if strip_accents(lemma) in headwords:
        return True
    acute = grave_to_acute(lemma)
    if acute in headwords:
        return True
    return False


def get_freq_bin(freq: int) -> str:
    for name, lo, hi in FREQ_BINS:
        if lo <= freq <= hi:
            return name
    return FREQ_BINS[-1][0]


def tokenize_greek(text: str) -> list[str]:
    return re.findall(r"[\u0370-\u03FF\u1F00-\u1FFF\u0300-\u036F\u02BC\u2019']+", text)


# --- Data loaders ---

def load_cyropaedia_words() -> list[str]:
    """Load Cyropaedia text words, excluding capitalized and punctuation."""
    txt_path = Path("/tmp/test_lemmatizers/data/cyropaedia.txt")
    if not txt_path.exists():
        print("  [SKIP] Cyropaedia text not found at /tmp/test_lemmatizers/data/")
        return []
    text = txt_path.read_text(encoding="utf-8")
    words = tokenize_greek(text)
    return [w for w in words if not is_capitalized(w)]


def load_herodotus_words() -> list[str]:
    """Load Herodotus words from an openly licensed text of the Histories,
    excluding capitalized words (proper-noun proxy, matching the other
    text loaders). Point HERODOTUS_TEI at a Perseus canonical-greekLit
    TEI XML (tlg0016.tlg001.perseus-grc2.xml, CC BY-SA); this is a plain
    word source for the headword-validity metric, so no annotations are
    needed."""
    tei_path = Path(os.environ.get("HERODOTUS_TEI", ""))
    if not tei_path.is_file():
        print("  [SKIP] Herodotus TEI not found; set HERODOTUS_TEI to a "
              "canonical-greekLit tlg0016.tlg001 grc XML")
        return []
    xml = tei_path.read_text(encoding="utf-8")
    # Drop the TEI header (edition metadata), then strip tags; the Greek
    # tokenizer ignores any Latin/markup residue.
    xml = re.sub(r"(?s)<teiHeader.*?</teiHeader>", " ", xml)
    text = re.sub(r"<[^>]+>", " ", xml)
    words = tokenize_greek(text)
    return [w for w in words if not is_capitalized(w)]


def load_sextus_words() -> list[str]:
    """Load Sextus Empiricus words from AG Classical benchmark."""
    if not SEXTUS_GOLD.exists():
        print("  [SKIP] AG benchmark not found")
        return []
    words = []
    with open(SEXTUS_GOLD) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[0].strip():
                form = parts[0]
                if not is_capitalized(form):
                    words.append(form)
    return words


def run_stratified_benchmark():
    print("Loading data...")
    freqs = load_corpus_freqs()
    headwords = load_headwords()
    print(f"  Corpus forms: {len(freqs):,}")
    print(f"  Headwords: {len(headwords):,}")

    from dilemma import Dilemma
    m = Dilemma(lang="all", convention="lsj")
    m.preload()

    datasets = {
        "Cyropaedia": load_cyropaedia_words(),
        "Herodotus": load_herodotus_words(),
        "Sextus Empiricus": load_sextus_words(),
    }

    # Combine all words, deduplicate
    all_words = []
    for name, words in datasets.items():
        print(f"  {name}: {len(words):,} tokens")
        all_words.extend(words)

    # Deduplicate across all texts
    seen = set()
    unique_words = []
    for w in all_words:
        if w not in seen:
            seen.add(w)
            unique_words.append(w)
    print(f"  Combined unique words: {len(unique_words):,}")

    # Bin by frequency
    binned: dict[str, list[str]] = defaultdict(list)
    for w in unique_words:
        key = strip_accents(w)
        freq = freqs.get(key, 0)
        bin_name = get_freq_bin(freq)
        binned[bin_name].append(w)

    # Evaluate per bin
    print(f"\n{'Frequency bin':<22} {'Forms':>7} {'Valid':>7} {'Accuracy':>10}")
    print("-" * 50)

    results = {}
    total_valid = 0
    total_forms = 0

    for bin_name, _, _ in FREQ_BINS:
        words_in_bin = binned.get(bin_name, [])
        if not words_in_bin:
            results[bin_name] = (0, 0, 0.0)
            continue
        valid = 0
        for w in words_in_bin:
            pred = m.lemmatize(w)
            if is_valid_lemma(pred, headwords):
                valid += 1
        n = len(words_in_bin)
        pct = valid / n * 100 if n else 0
        results[bin_name] = (n, valid, pct)
        total_valid += valid
        total_forms += n
        print(f"  {bin_name:<20} {n:>7,} {valid:>7,} {pct:>9.1f}%")

    total_pct = total_valid / total_forms * 100 if total_forms else 0
    print(f"  {'Overall':<20} {total_forms:>7,} {total_valid:>7,} {total_pct:>9.1f}%")

    # Per-text breakdown
    print(f"\n\nPer-text results:")
    for text_name in ["Cyropaedia", "Herodotus", "Sextus Empiricus"]:
        words = datasets[text_name]
        if not words:
            continue
        # Deduplicate per text
        seen_t = set()
        unique_t = []
        for w in words:
            if w not in seen_t:
                seen_t.add(w)
                unique_t.append(w)

        text_binned: dict[str, list[str]] = defaultdict(list)
        for w in unique_t:
            key = strip_accents(w)
            freq = freqs.get(key, 0)
            bin_name = get_freq_bin(freq)
            text_binned[bin_name].append(w)

        print(f"\n  {text_name} ({len(unique_t):,} unique words)")
        print(f"  {'Frequency bin':<22} {'Forms':>7} {'Valid':>7} {'Accuracy':>10}")
        print(f"  {'-' * 50}")

        t_total_valid = 0
        t_total_forms = 0
        for bin_name, _, _ in FREQ_BINS:
            words_in_bin = text_binned.get(bin_name, [])
            if not words_in_bin:
                continue
            valid = 0
            for w in words_in_bin:
                pred = m.lemmatize(w)
                if is_valid_lemma(pred, headwords):
                    valid += 1
            n = len(words_in_bin)
            pct = valid / n * 100 if n else 0
            t_total_valid += valid
            t_total_forms += n
            print(f"  {bin_name:<22} {n:>7,} {valid:>7,} {pct:>9.1f}%")

        t_pct = t_total_valid / t_total_forms * 100 if t_total_forms else 0
        print(f"  {'Overall':<22} {t_total_forms:>7,} {t_total_valid:>7,} {t_pct:>9.1f}%")

    return results


if __name__ == "__main__":
    run_stratified_benchmark()

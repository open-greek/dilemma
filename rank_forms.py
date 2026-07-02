#!/usr/bin/env python3
"""
Rank inflected forms per lemma by corpus frequency for downstream consumers.

Produces:
  data/{prefix}_ranked_forms.json          - lemma -> [forms sorted by frequency]
  data/{prefix}_form_freq.json             - form -> frequency count
  data/{prefix}_ranked_forms_verbose.json  - (with --verbose) per-corpus frequency breakdown

Usage:
  python3 rank_forms.py --lang el    # Modern Greek (default)
  python3 rank_forms.py --lang grc   # Ancient Greek
  python3 rank_forms.py --lang mgr   # Medieval/Byzantine Greek
  python3 rank_forms.py --lang all   # All three
  python3 rank_forms.py --lang el --verbose  # Include per-corpus breakdown
"""

import argparse
import json
import os
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

DATA_DIR = Path(__file__).parent / "data"
HF_REPO_ID = "open-greek/dilemma-data"

# Fallback path for MG frequency data (in the lemma project)
LEMMA_FREQ_PATH = Path.home() / "Documents" / "lemma" / "data" / "el_full.txt"


def load_mg_polytonic_freq():
    """Load MG polytonic frequency data from mg_polytonic_freq.json.

    Returns: {monotonic_form: [(polytonic_form, count), ...]}
    """
    freq_path = DATA_DIR / "mg_polytonic_freq.json"
    if not freq_path.exists():
        print(f"  ERROR: {freq_path} not found")
        sys.exit(1)

    with open(freq_path, encoding="utf-8") as f:
        data = json.load(f)

    forms = data.get("forms", {})
    mono_to_poly = defaultdict(list)
    for poly_form, info in forms.items():
        mono = info.get("monotonic", "")
        count = info.get("count", 0)
        if mono:
            mono_to_poly[mono].append((poly_form, count))

    print(f"  Loaded {len(forms):,} polytonic forms mapping to {len(mono_to_poly):,} monotonic forms")
    return mono_to_poly


def process_mg_polytonic(min_count=3):
    """Process MG polytonic frequency data into ranked variants per monotonic form.

    Loads the polytonic freq data, filters by min_count, sorts variants by
    frequency descending, and writes mg_polytonic_ranked.json.
    """
    print("\n=== MG Polytonic Ranking ===")
    print("  Loading polytonic frequency data...")
    mono_to_poly = load_mg_polytonic_freq()

    ranked = {}
    total_variants = 0
    filtered_out = 0

    for mono, variants in mono_to_poly.items():
        # Filter out low-frequency forms
        kept = [(pf, c) for pf, c in variants if c >= min_count]
        filtered_out += len(variants) - len(kept)

        if not kept:
            continue

        # Sort by frequency descending
        kept.sort(key=lambda x: -x[1])
        ranked[mono] = [pf for pf, _ in kept]
        total_variants += len(kept)

    # Write output
    out_path = DATA_DIR / "mg_polytonic_ranked.json"
    print(f"  Writing {out_path.name}...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ranked, f, ensure_ascii=False, indent=None, separators=(",", ":"))

    print(f"\n  === Stats for MG polytonic ===")
    print(f"  Monotonic forms with variants: {len(ranked):,}")
    print(f"  Total polytonic variants:      {total_variants:,}")
    print(f"  Filtered out (count < {min_count}):   {filtered_out:,}")
    print(f"  Output file: {out_path}")
    print(f"  Output size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Sample output
    for mono in ["από", "αυτός", "είναι", "και"]:
        if mono in ranked:
            print(f"  Sample: {mono} -> {ranked[mono][:5]}")

    return ranked


def load_mg_freq():
    """Load Modern Greek frequency data from FrequencyWords format (word count per line)."""
    freq_path = DATA_DIR / "mg_freq.txt"
    if not freq_path.exists():
        if LEMMA_FREQ_PATH.exists():
            print(f"  mg_freq.txt not found in data/, falling back to {LEMMA_FREQ_PATH}")
            freq_path = LEMMA_FREQ_PATH
        else:
            print("  WARNING: No MG frequency data found, all frequencies will be 0")
            return {}

    freq = {}
    with open(freq_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0]
                try:
                    count = int(parts[1])
                except ValueError:
                    continue
                freq[word] = count
    print(f"  Loaded {len(freq):,} frequency entries from {freq_path.name}")
    return freq


def load_ag_freq():
    """Load Ancient Greek frequency data from corpus_freq.json.

    Format: {forms: {form: [total, genre1, ...]}, _total_tokens: ..., ...}
    """
    freq_path = DATA_DIR / "corpus_freq.json"
    if not freq_path.exists():
        print("  WARNING: corpus_freq.json not found, all frequencies will be 0")
        return {}

    with open(freq_path, encoding="utf-8") as f:
        data = json.load(f)

    freq = {}
    forms = data.get("forms", {})
    if forms:
        for form, counts in forms.items():
            if isinstance(counts, list) and len(counts) > 0:
                freq[form] = counts[0]
    else:
        # Non-metadata keys are the forms directly
        for key, val in data.items():
            if key.startswith("_"):
                continue
            if isinstance(val, list) and len(val) > 0:
                freq[key] = val[0]
            elif isinstance(val, int):
                freq[key] = val

    print(f"  Loaded {len(freq):,} frequency entries from corpus_freq.json")
    return freq


def load_corpus_freq_file(path):
    """Load a corpus frequency JSON file in the standard format.

    Format: {forms: {form: [total, genre1, ...]}, _total_tokens: ..., ...}
    Returns: {form: total_count}
    """
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    freq = {}
    forms = data.get("forms", {})
    if forms:
        for form, counts in forms.items():
            if isinstance(counts, list) and len(counts) > 0:
                freq[form] = counts[0]
    else:
        for key, val in data.items():
            if key.startswith("_"):
                continue
            if isinstance(val, list) and len(val) > 0:
                freq[key] = val[0]
            elif isinstance(val, int):
                freq[key] = val

    return freq


def load_all_verbose_sources():
    """Load all available frequency sources for verbose output.

    Returns: dict of {source_name: {form: count}} for each source that loaded
    successfully. Sources that don't exist are silently skipped.
    """
    sources = {}

    # opensubs (MG frequency)
    freq_path = DATA_DIR / "mg_freq.txt"
    if not freq_path.exists() and LEMMA_FREQ_PATH.exists():
        freq_path = LEMMA_FREQ_PATH
    if freq_path.exists():
        freq = {}
        with open(freq_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        freq[parts[0]] = int(parts[1])
                    except ValueError:
                        continue
        sources["opensubs"] = freq
        print(f"  [verbose] Loaded {len(freq):,} entries from {freq_path.name} (opensubs)")

    # glaux (AG)
    glaux = load_corpus_freq_file(DATA_DIR / "glaux_freq.json")
    if glaux is not None:
        sources["glaux"] = glaux
        print(f"  [verbose] Loaded {len(glaux):,} entries from glaux_freq.json")

    # diorisis (AG)
    diorisis = load_corpus_freq_file(DATA_DIR / "diorisis_freq.json")
    if diorisis is not None:
        sources["diorisis"] = diorisis
        print(f"  [verbose] Loaded {len(diorisis):,} entries from diorisis_freq.json")

    # pg (Byzantine literary, future)
    pg = load_corpus_freq_file(DATA_DIR / "pg_freq.json")
    if pg is not None:
        sources["pg"] = pg
        print(f"  [verbose] Loaded {len(pg):,} entries from pg_freq.json")

    # byz_vern (Byzantine vernacular, future)
    byz_vern = load_corpus_freq_file(DATA_DIR / "byz_vern_freq.json")
    if byz_vern is not None:
        sources["byz_vern"] = byz_vern
        print(f"  [verbose] Loaded {len(byz_vern):,} entries from byz_vern_freq.json")

    return sources


def load_mg_lookup():
    """Load MG lookup: {form: {lemma: str, confidence: int}}."""
    path = DATA_DIR / "mg_lookup_scored.json"
    if not path.exists():
        print(f"  ERROR: {path} not found")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data):,} entries from mg_lookup_scored.json")
    return data


def load_ag_lookup():
    """Load AG lookup. Tries ag_lookup_scored.json first, falls back to ag_lookup.json."""
    scored_path = DATA_DIR / "ag_lookup_scored.json"
    plain_path = DATA_DIR / "ag_lookup.json"

    if scored_path.exists():
        with open(scored_path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Loaded {len(data):,} entries from ag_lookup_scored.json")
        return data, True
    elif plain_path.exists():
        with open(plain_path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Loaded {len(data):,} entries from ag_lookup.json")
        return data, False
    else:
        print("  ERROR: No AG lookup file found")
        sys.exit(1)


def invert_lookup_scored(lookup):
    """Invert {form: {lemma, confidence}} to {lemma: [(form, confidence)]}."""
    lemma_forms = defaultdict(list)
    for form, entry in lookup.items():
        lemma = entry["lemma"]
        confidence = entry.get("confidence", 0)
        lemma_forms[lemma].append((form, confidence))
    return lemma_forms


def invert_lookup_plain(lookup):
    """Invert {form: lemma} to {lemma: [(form, confidence=0)]}."""
    lemma_forms = defaultdict(list)
    for form, lemma in lookup.items():
        lemma_forms[lemma].append((form, 0))
    return lemma_forms


def canonical_lower(form):
    """Return the lowercase version of a form."""
    return form.lower()


def deduplicate_forms(forms_with_scores):
    """Deduplicate case variants, keeping only the canonical (lowercase) form.

    For each group of case variants, keep the lowercase form unless only
    the capitalized form exists (e.g., proper nouns).

    forms_with_scores: list of (form, confidence, frequency)
    Returns: deduplicated list of (form, confidence, frequency)
    """
    # Group by lowercase form
    groups = defaultdict(list)
    for form, conf, freq in forms_with_scores:
        groups[canonical_lower(form)].append((form, conf, freq))

    result = []
    for lower_form, variants in groups.items():
        # Check if lowercase version exists among variants
        has_lowercase = any(f == lower_form for f, _, _ in variants)

        if has_lowercase:
            # Use the lowercase variant; merge frequency from all case variants
            total_freq = sum(fr for _, _, fr in variants)
            max_conf = max(c for _, c, _ in variants)
            result.append((lower_form, max_conf, total_freq))
        else:
            # No lowercase variant exists - keep the most frequent capitalized form
            # (this handles proper nouns where only Capitalized form is attested)
            best = max(variants, key=lambda x: (x[2], x[1]))
            total_freq = sum(fr for _, _, fr in variants)
            result.append((best[0], best[1], total_freq))

    return result


def rank_and_output(lemma_forms, freq, prefix, verbose=False, verbose_sources=None):
    """Rank forms per lemma by frequency, deduplicate, and write output files."""
    print(f"\n  Ranking forms for {len(lemma_forms):,} lemmas...")

    # Build form frequency dict for all forms in the lookup
    all_forms_in_lookup = set()
    for forms_list in lemma_forms.values():
        for form, _ in forms_list:
            all_forms_in_lookup.add(form)

    form_freq = {}
    for form in all_forms_in_lookup:
        f = freq.get(form, 0)
        # Also check lowercase
        if f == 0 and form != form.lower():
            f = freq.get(form.lower(), 0)
        form_freq[form] = f

    # Rank and deduplicate per lemma
    ranked = {}
    ranked_verbose = {}
    total_forms = 0
    forms_per_lemma = []

    for lemma, forms_list in lemma_forms.items():
        # Attach frequency to each form
        scored = []
        for form, confidence in forms_list:
            f = freq.get(form, 0)
            if f == 0 and form != form.lower():
                f = freq.get(form.lower(), 0)
            scored.append((form, confidence, f))

        # Deduplicate case variants
        deduped = deduplicate_forms(scored)

        # Sort: frequency desc, confidence desc, alphabetical asc
        deduped.sort(key=lambda x: (-x[2], -x[1], x[0]))

        ranked[lemma] = [form for form, _, _ in deduped]
        if verbose and verbose_sources:
            verbose_entries = []
            for form, conf, _ in deduped:
                entry = {"form": form, "conf": conf}
                for src_name, src_freq in verbose_sources.items():
                    f_val = src_freq.get(form, 0)
                    if f_val == 0 and form != form.lower():
                        f_val = src_freq.get(form.lower(), 0)
                    entry[f"freq_{src_name}"] = f_val
                verbose_entries.append(entry)
            ranked_verbose[lemma] = verbose_entries
        n = len(ranked[lemma])
        total_forms += n
        forms_per_lemma.append(n)

    # Also build a clean form_freq dict (lowercase-canonical)
    clean_form_freq = {}
    for form, f in form_freq.items():
        lf = form.lower()
        if lf in clean_form_freq:
            clean_form_freq[lf] = max(clean_form_freq[lf], f)
        else:
            clean_form_freq[lf] = f
    # Also include any non-lowerable forms
    for form, f in form_freq.items():
        if form not in clean_form_freq and form.lower() not in clean_form_freq:
            clean_form_freq[form] = f

    # Write outputs
    ranked_path = DATA_DIR / f"{prefix}_ranked_forms.json"
    freq_path = DATA_DIR / f"{prefix}_form_freq.json"

    print(f"  Writing {ranked_path.name}...")
    with open(ranked_path, "w", encoding="utf-8") as f:
        json.dump(ranked, f, ensure_ascii=False, indent=None, separators=(",", ":"))

    print(f"  Writing {freq_path.name}...")
    with open(freq_path, "w", encoding="utf-8") as f:
        json.dump(clean_form_freq, f, ensure_ascii=False, indent=None, separators=(",", ":"))

    if verbose and verbose_sources and ranked_verbose:
        verbose_path = DATA_DIR / f"{prefix}_ranked_forms_verbose.json"
        print(f"  Writing {verbose_path.name}...")
        with open(verbose_path, "w", encoding="utf-8") as f:
            json.dump(ranked_verbose, f, ensure_ascii=False, indent=None, separators=(",", ":"))

    # Print stats
    n_lemmas = len(ranked)
    forms_with_freq = sum(1 for fpl in ranked.values() for form in fpl
                         if freq.get(form, 0) > 0 or freq.get(form.lower(), 0) > 0)
    forms_without_freq = total_forms - forms_with_freq

    print(f"\n  === Stats for {prefix} ===")
    print(f"  Total lemmas:          {n_lemmas:,}")
    print(f"  Total forms:           {total_forms:,}")
    if forms_per_lemma:
        print(f"  Median forms/lemma:    {statistics.median(forms_per_lemma):.0f}")
        print(f"  Mean forms/lemma:      {statistics.mean(forms_per_lemma):.1f}")
        print(f"  Max forms/lemma:       {max(forms_per_lemma):,}")
    print(f"  Forms with freq > 0:   {forms_with_freq:,}")
    print(f"  Forms with freq = 0:   {forms_without_freq:,}")

    return ranked


def process_mg(verbose=False, verbose_sources=None):
    """Process Modern Greek."""
    print("\n=== Modern Greek (el) ===")
    print("  Loading lookup...")
    lookup = load_mg_lookup()
    print("  Loading frequencies...")
    freq = load_mg_freq()
    lemma_forms = invert_lookup_scored(lookup)
    ranked = rank_and_output(lemma_forms, freq, "mg", verbose=verbose,
                             verbose_sources=verbose_sources)
    return ranked


def process_ag(verbose=False, verbose_sources=None):
    """Process Ancient Greek."""
    print("\n=== Ancient Greek (grc) ===")
    print("  Loading lookup...")
    lookup_data, is_scored = load_ag_lookup()
    print("  Loading frequencies...")
    freq = load_ag_freq()
    if is_scored:
        lemma_forms = invert_lookup_scored(lookup_data)
    else:
        lemma_forms = invert_lookup_plain(lookup_data)
    ranked = rank_and_output(lemma_forms, freq, "ag", verbose=verbose,
                             verbose_sources=verbose_sources)
    return ranked


def load_med_lookup():
    """Load Medieval Greek lookup: {form: lemma} (plain format)."""
    path = DATA_DIR / "med_lookup.json"
    if not path.exists():
        print(f"  ERROR: {path} not found")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data):,} entries from med_lookup.json")
    return data


def process_med(verbose=False, verbose_sources=None):
    """Process Medieval/Byzantine Greek.

    Uses AG corpus frequencies as the best available proxy, since no
    dedicated Byzantine frequency corpus exists. GLAUx and Diorisis
    include significant post-Classical and late antique texts that
    overlap with Byzantine vocabulary.
    """
    print("\n=== Medieval/Byzantine Greek (mgr) ===")
    print("  Loading lookup...")
    lookup = load_med_lookup()
    print("  Loading frequencies (AG corpus as proxy)...")
    freq = load_ag_freq()
    lemma_forms = invert_lookup_plain(lookup)
    ranked = rank_and_output(lemma_forms, freq, "med", verbose=verbose,
                             verbose_sources=verbose_sources)
    return ranked


def download_prebuilt(prefix, verbose=False):
    """Try to download pre-built ranked forms from HuggingFace Hub.

    Returns True if all required files were downloaded successfully,
    False otherwise.
    """
    if not HAS_HF_HUB:
        print(f"  huggingface_hub not installed, cannot download pre-built files")
        return False

    files = [
        f"{prefix}_ranked_forms.json",
        f"{prefix}_form_freq.json",
    ]
    if verbose:
        files.append(f"{prefix}_ranked_forms_verbose.json")
    if prefix == "mg":
        files.append("mg_lookup_scored.json")

    try:
        for filename in files:
            print(f"  Downloading {filename} from HF Hub...")
            cached_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type="dataset",
            )
            dest = DATA_DIR / filename
            shutil.copy2(cached_path, dest)
            print(f"  Saved {filename} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"  HF download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Rank inflected forms per lemma by corpus frequency")
    parser.add_argument("--lang", default="el", choices=["el", "grc", "mgr", "all"],
                        help="Language: el (Modern Greek), grc (Ancient Greek), mgr (Medieval/Byzantine), all")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Generate verbose output with per-corpus frequency breakdowns")
    parser.add_argument("--rebuild", action="store_true", default=False,
                        help="Skip HF Hub download and regenerate locally")
    parser.add_argument("--polytonic", action="store_true", default=False,
                        help="Generate polytonic ranked variants for MG")
    args = parser.parse_args()

    # Map lang choices to (prefix, process_fn) pairs
    lang_map = {
        "el": "mg",
        "grc": "ag",
        "mgr": "med",
    }
    langs_to_process = list(lang_map.keys()) if args.lang == "all" else [args.lang]

    verbose_sources = None
    needs_rebuild = set()

    # Try downloading pre-built files unless --rebuild is set
    if not args.rebuild:
        for lang in langs_to_process:
            prefix = lang_map[lang]
            print(f"\n--- Checking HF Hub for pre-built {prefix} files ---")
            if download_prebuilt(prefix, verbose=args.verbose):
                print(f"  Using pre-built files from HF Hub")
            else:
                print(f"  HF download failed, rebuilding locally")
                needs_rebuild.add(lang)
    else:
        needs_rebuild = set(langs_to_process)

    # Load verbose sources only if we need to rebuild something with verbose
    if needs_rebuild and args.verbose:
        print("\nLoading all frequency sources for verbose output...")
        verbose_sources = load_all_verbose_sources()
        if not verbose_sources:
            print("  WARNING: No frequency sources loaded for verbose output")

    results = {}
    if "el" in needs_rebuild:
        results["el"] = process_mg(verbose=args.verbose, verbose_sources=verbose_sources)
    if "grc" in needs_rebuild:
        results["grc"] = process_ag(verbose=args.verbose, verbose_sources=verbose_sources)
    if "mgr" in needs_rebuild:
        results["mgr"] = process_med(verbose=args.verbose, verbose_sources=verbose_sources)

    # Polytonic ranking for MG
    if args.polytonic and ("el" in langs_to_process or args.lang == "all"):
        process_mg_polytonic()

    # Quick sanity check for MG
    if "el" in results:
        ranked = results["el"]
        for lemma in ["τρώω", "είμαι", "έχω", "κάνω"]:
            if lemma in ranked:
                forms = ranked[lemma][:15]
                print(f"\n  Sample: {lemma} -> {forms}")

    # Quick sanity check for AG
    if "grc" in results:
        ranked = results["grc"]
        for lemma in ["λέγω", "εἰμί", "ποιέω", "γίγνομαι"]:
            if lemma in ranked:
                forms = ranked[lemma][:15]
                print(f"\n  Sample: {lemma} -> {forms}")

    # Quick sanity check for Medieval
    if "mgr" in results:
        ranked = results["mgr"]
        for lemma in list(ranked.keys())[:3]:
            forms = ranked[lemma][:10]
            print(f"\n  Sample: {lemma} -> {forms}")

    print("\nDone.")


if __name__ == "__main__":
    main()

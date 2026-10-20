#!/usr/bin/env python3
"""Evaluate separate GlossAPI frequency priors; never replace production data.

Corpus windows are identical before and after cleaning. Development data
selects a mixture weight; a separately pinned test split measures that choice.
The experimental reranker changes only tied lookup candidates, preserves
article contexts, and does not alter Dilemma's runtime ranking implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dilemma import Dilemma, strip_accents, to_monotonic
from dilemma.core import LOOKUP_DB_PATH
from scripts.audit_glossapi_coverage import (collect_corpus_sample, file_sha256,
    iter_greek_tokens, load_manifest, mg_frequency_counts, RawSample, _clean_mg_form)

UD_REVISION = "005bb3e0853a40f95162aab35177250c6f1c7328"
UD_BASE = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Greek-GDT/{UD_REVISION}/"
FIXTURE_SHA256 = {"dev": "53de54735051eecef04aa8c0f5f0fbf2466e0e0d4fd9dfd3d313560c394ddf41",
                  "test": "30cb5881ce6f8ba628de88853a4c105e716a365e8fcd9024a85befe41165c260"}
LABEL = re.compile(r"^\s*(?:[Α-Ωα-ω]{1,3}|\d{1,5})[.)΄’]\s+")
FIELD = re.compile(r"^\s*(?:Αρμόδιος υπάλληλος|Χειροκίνητη ενέργεια)\s*(?::|\||$)", re.I)
ARTICLES = set("ο η το οι τα του της των τον την τους τις ένας μία ένα ενός μιας".split())
WEIGHTS = (0.0, 0.05, 0.10, 0.25)


def accent_restorations(lookup, wanted):
    alternatives = defaultdict(set)
    for form in lookup:
        lower = to_monotonic(form.lower())
        bare = strip_accents(lower)
        if bare in wanted and bare != lower:
            alternatives[bare].add(lower)
    return {bare: next(iter(forms)) for bare, forms in alternatives.items() if len(forms) == 1}


class ContextCounter:
    def __init__(self, restorations):
        self.restorations = restorations
        self.seen = set()
        self.raw = Counter()
        self.clean = Counter()
        self.removed = Counter()

    def observe(self, text, identity, allowance):
        tagged = []
        for line in text.splitlines(keepends=True):
            tokens = list(iter_greek_tokens(line))
            fingerprint = hashlib.sha256(unicodedata.normalize("NFC", " ".join(line.split())).encode()).digest()
            duplicate = len(tokens) >= 8 and fingerprint in self.seen
            if len(tokens) >= 8:
                self.seen.add(fingerprint)
            marker = LABEL.match(line)
            marker_tokens = len(list(iter_greek_tokens(marker.group()))) if marker else 0
            field = bool(FIELD.match(line))
            for index, token in enumerate(tokens):
                reason = ("duplicate_line" if duplicate else "field_label" if field
                          else "enumeration_label" if index < marker_tokens else None)
                tagged.append((token, reason))
        if len(tagged) > allowance:
            seed = hashlib.sha256((identity + "\0" + text[:1024]).encode("utf-8", errors="replace")).digest()
            start = int.from_bytes(seed[:8], "big") % (len(tagged) - allowance + 1)
            tagged = tagged[start:start + allowance]
        for raw, reason in tagged:
            self.raw[raw] += 1
            if reason:
                self.removed[reason] += 1
                continue
            clean = _clean_mg_form(raw)
            if clean is None:
                self.removed["unicode_or_nonlexical"] += 1
                continue
            # Restore stress only on original all-capitals words with a
            # unique accented spelling in the existing Modern lookup.
            if raw.isupper() and len(clean) > 2 and strip_accents(clean) == clean:
                restored = self.restorations.get(clean)
                if restored:
                    self.removed["stress_restored"] += 1
                    clean = restored
            self.clean[clean] += 1


def counts_from_file(path):
    return Counter({parts[0]: int(parts[1]) for line in path.read_text().splitlines()
                    if len(parts := line.split()) == 2})


def lemma_prior(counts, lookup):
    result = Counter()
    for form, count in counts.items():
        lemma = lookup.get(form)
        if lemma:
            result[to_monotonic(lemma.lower())] += count
    total = sum(result.values())
    return {lemma: count / total for lemma, count in result.items()} if total else {}


def load_ud(path):
    sentence = []
    records = []
    for line in path.read_text().splitlines() + [""]:
        if not line:
            records.extend(sentence)
            sentence = []
        elif not line.startswith("#"):
            fields = line.split("\t")
            if fields[0].isdigit():
                previous = sentence[-1]["form"] if sentence else ""
                sentence.append({"form": fields[1], "gold": fields[2], "upos": fields[3], "previous": previous})
    return [r for r in records if r["upos"] not in {"PUNCT", "SYM", "X"} and r["gold"] != "_"]


def candidate_records(records, engine):
    cache = {}
    output = []
    for record in records:
        key = (record["form"], record["previous"])
        if key not in cache:
            candidates = engine.lemmatize_verbose(key[0], guess=False, prev_word=key[1])
            base = candidates[0].lemma if candidates else None
            pool = []
            if candidates and candidates[0].source == "lookup" and key[1].lower() not in ARTICLES:
                first = candidates[0]
                pool = list(dict.fromkeys(c.lemma for c in candidates if c.source == "lookup"
                                         and c.score == first.score and c.proper == first.proper))
            cache[key] = (base, pool)
        base, pool = cache[key]
        output.append({**record, "runtime": base, "pool": pool})
    return output


def predict(record, prior):
    pool = record["pool"]
    if len(pool) < 2:
        return record["runtime"]
    return max(pool, key=lambda lemma: prior.get(to_monotonic(lemma.lower()), 0))


def score(records, prior, baseline=None):
    correct = gains = losses = changed = 0
    examples = []
    for index, record in enumerate(records):
        prediction = predict(record, prior)
        ok = prediction == record["gold"]
        correct += ok
        if baseline is not None:
            old = baseline[index]
            gains += ok and old != record["gold"]
            losses += not ok and old == record["gold"]
            changed += old != prediction
            if old != prediction and len(examples) < 30:
                examples.append({**record, "baseline": old, "experimental": prediction})
    discordant = gains + losses
    p_value = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1)) / 2**discordant) if discordant else 1.0
    return {"tokens": len(records), "correct": correct, "accuracy": correct / len(records),
            "ambiguous_ties": sum(len(r["pool"]) > 1 for r in records),
            "changed": changed, "gains": gains, "losses": losses,
            "paired_exact_p": p_value, "examples": examples}


def blend(base, domain, weight):
    return {key: (1-weight)*base.get(key, 0) + weight*domain.get(key, 0)
            for key in base.keys() | domain.keys()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/glossapi_frequency_evaluation.json")
    parser.add_argument("--counts", type=Path, default=ROOT / "data/glossapi_frequency_clean_counts.json")
    parser.add_argument("--reuse-counts", action="store_true")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--source", choices=("cache", "stream"), default="stream")
    parser.add_argument("--fixture-dir", type=Path, default=ROOT / "data/glossapi_frequency_fixtures")
    parser.add_argument("--limit-eval", type=int)
    parser.add_argument("--corpus", action="append")
    parser.add_argument("--max-documents", type=int, default=4096)
    args = parser.parse_args()
    if (args.max_tokens is not None and args.max_tokens <= 0) or args.max_documents <= 0:
        parser.error("sample limits must be positive")
    recipe = {"max_tokens": args.max_tokens, "corpus": args.corpus, "max_documents": args.max_documents}
    lookup = json.loads(args.lookup.read_text())
    production = ROOT / "data/mg_freq.txt"
    before = file_sha256(production)
    if args.reuse_counts:
        counts = json.loads(args.counts.read_text())
        if counts["lookup_sha256"] != file_sha256(args.lookup):
            raise ValueError("clean counts belong to a different lookup snapshot")
        if recipe != counts["recipe"]:
            raise ValueError("sample limits differ from saved counters")
        if counts["collector_sha256"] != file_sha256(Path(__file__)):
            raise ValueError("clean counts belong to a different collector revision")
        if counts["manifest_sha256"] != file_sha256(ROOT / "data/glossapi_corpora.json"):
            raise ValueError("source manifest differs from saved counters")
    else:
        manifest = load_manifest()
        old_counts = json.loads((ROOT / "data/glossapi_mg_frequency_experiment.json").read_text())
        wanted = set(old_counts["combined_counts"])
        restorations = accent_restorations(lookup, wanted)
        sources = {}
        for corpus in manifest["corpora"]:
            cap = corpus["frequency_token_cap"]
            if cap <= 0:
                continue
            if args.corpus and corpus["key"] not in args.corpus:
                continue
            cap = min(cap, args.max_tokens) if args.max_tokens else cap
            cleaner = ContextCounter(restorations)
            sample = collect_corpus_sample(corpus, source=args.source, token_cap=cap,
                                           document_observer=cleaner.observe, max_documents=args.max_documents)
            if cleaner.raw != sample.tokens:
                raise ValueError(f"{corpus['key']}: cleaning changed the sample windows")
            if not sample.token_count:
                raise ValueError(f"{corpus['key']}: empty sample")
            sources[corpus["key"]] = {"documents": sample.documents, "sampled_tokens": sample.token_count,
                "raw": dict(mg_frequency_counts(sample, cap)), "clean": dict(cleaner.clean),
                "diagnostics": dict(cleaner.removed)}
            print(corpus["key"], sample.documents, sample.token_count, dict(cleaner.removed), flush=True)
        counts = {"sources": sources, "recipe": recipe, "lookup_sha256": file_sha256(args.lookup),
                  "manifest_sha256": file_sha256(ROOT / "data/glossapi_corpora.json"),
                  "collector_sha256": file_sha256(Path(__file__)),
                  "complete": args.max_tokens is None and not args.corpus and args.max_documents == 4096}
        args.counts.write_text(json.dumps(counts, ensure_ascii=False, sort_keys=True) + "\n")
    combined = Counter()
    for source in counts["sources"].values():
        combined.update(source["clean"])
    base = lemma_prior(counts_from_file(production), lookup)
    domain = lemma_prior(combined, lookup)
    args.fixture_dir.mkdir(parents=True, exist_ok=True)
    engine = Dilemma(lang="el", convention="triantafyllidis", skip_pos=True)
    fixtures = {}
    evaluated = {}
    for split in ("dev", "test"):
        name = f"el_gdt-ud-{split}.conllu"
        path = args.fixture_dir / name
        if not path.exists():
            with urlopen(UD_BASE + name, timeout=60) as response:
                path.write_bytes(response.read())
        if file_sha256(path) != FIXTURE_SHA256[split]:
            raise ValueError(f"{split} fixture differs from the pinned GDT bytes")
        fixtures[split] = {"url": UD_BASE + name, "sha256": file_sha256(path)}
        records = load_ud(path)
        evaluated[split] = candidate_records(records[:args.limit_eval] if args.limit_eval else records, engine)
        print("evaluated", split, len(evaluated[split]), flush=True)
    dev = evaluated["dev"]
    results = {str(weight): score(dev, blend(base, domain, weight)) for weight in WEIGHTS}
    weight = max(WEIGHTS, key=lambda w: (results[str(w)]["correct"], -w))
    test = evaluated["test"]
    baseline = [predict(record, base) for record in test]
    selected = score(test, blend(base, domain, weight), baseline)
    fixed = score(test, blend(base, domain, 0.10), baseline)
    runtime_correct = sum(r["runtime"] == r["gold"] for r in test)
    measurable = (weight > 0 and selected["gains"] > selected["losses"] and selected["paired_exact_p"] <= .05
                  and selected["correct"] > runtime_correct and counts["complete"] and args.limit_eval is None)
    report = {"status": "experiment_only", "replaces_mg_freq": False, "production_sha256": before,
              "counts_sha256": file_sha256(args.counts), "fixtures": fixtures,
              "lookup_sha256": file_sha256(args.lookup), "code_sha256": file_sha256(Path(__file__)),
              "runtime_lookup_sha256": file_sha256(LOOKUP_DB_PATH),
              "raw_tokens": sum(sum(s["raw"].values()) for s in counts["sources"].values()),
              "clean_tokens": sum(combined.values()), "clean_types": len(combined),
              "weights_predeclared": WEIGHTS, "development": results, "selected_weight": weight,
              "test": selected, "fixed_ten_percent_test": fixed,
              "shipped_runtime_test_correct": runtime_correct, "measurable_improvement": measurable,
              "decision": "keep_separate_pending_replication" if measurable else "do_not_adopt",
              "source_diagnostics": {k: {f: v for f,v in s.items() if f not in {"raw","clean"}} for k,s in counts["sources"].items()},
              "limitations": ["Dictionary-mapped surface counts are uncertain lemma priors, not annotated lemma counts.",
                 "Only tied lookup candidates without preceding article agreement are reranked.",
                 "The pinned GDT splits come from a separate treebank; exact corpus overlap was not exhaustively checked, and dictionary/tagger source overlap is possible.",
                 "No runtime ranking code or production frequency file is changed."]}
    assert file_sha256(production) == before
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("selected_weight", "test", "measurable_improvement", "decision")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

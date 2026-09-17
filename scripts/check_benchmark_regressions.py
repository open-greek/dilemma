#!/usr/bin/env python3
"""Run deterministic Dilemma-only accuracy regression checks.

The comparative benchmark in ``data/benchmarks/bench_all.py`` intentionally
loads optional third-party tools and model weights.  This gate exercises only
the shipped lookup and deterministic fallback layers (``guess=False``), pins
the gold fixture bytes, and compares every prediction with a reviewed
baseline.  Prediction changes are printed token by token; a reduction in
strict or equivalence-adjusted correctness fails the command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dilemma import Dilemma, strip_accents  # noqa: E402


BENCHMARK_DIR = ROOT / "data" / "benchmarks"
BASELINE_PATH = BENCHMARK_DIR / "dilemma_regression.json"
EQUIVALENCE_PATH = ROOT / "data" / "lemma_equivalences.json"


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    fixture: str
    profile: str
    dilemma_kwargs: dict


SPECS = (
    BenchmarkSpec(
        name="ag_classical",
        fixture="ag_gold.tsv",
        profile="all/wiktionary",
        dilemma_kwargs={"lang": "all", "resolve_articles": True,
                        "skip_pos": True},
    ),
    BenchmarkSpec(
        name="katharevousa",
        fixture="katharevousa_gold.tsv",
        profile="all/wiktionary",
        dilemma_kwargs={"lang": "all", "resolve_articles": True,
                        "skip_pos": True},
    ),
    BenchmarkSpec(
        name="demotic",
        fixture="demotic_gold.tsv",
        profile="el/triantafyllidis",
        dilemma_kwargs={"lang": "el", "convention": "triantafyllidis",
                        "skip_pos": True},
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gold_fixture(path: Path) -> list[tuple[str, str]]:
    """Load and structurally validate a two-column UTF-8/NFC gold fixture."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: not valid UTF-8") from exc

    pairs = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(
                f"{path}:{line_number}: expected exactly two TSV columns"
            )
        form, lemma = parts
        if not form or not lemma:
            raise ValueError(f"{path}:{line_number}: empty form or lemma")
        if form != form.strip() or lemma != lemma.strip():
            raise ValueError(f"{path}:{line_number}: surrounding whitespace")
        if unicodedata.normalize("NFC", form) != form:
            raise ValueError(f"{path}:{line_number}: form is not NFC")
        if unicodedata.normalize("NFC", lemma) != lemma:
            raise ValueError(f"{path}:{line_number}: lemma is not NFC")
        pairs.append((form, lemma))

    if not pairs:
        raise ValueError(f"{path}: fixture has no tokens")
    return pairs


def _load_equivalences(path: Path = EQUIVALENCE_PATH) -> dict[str, set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    equivalences: dict[str, set[str]] = {}
    for group in data["groups"]:
        members = set(group)
        for lemma in members:
            equivalences.setdefault(lemma, set()).update(members)
    return equivalences


def _accent_free(text: str) -> str:
    return strip_accents(text).casefold().strip()


def equivalent(prediction: str | None, gold: str,
               equivalences: dict[str, set[str]]) -> bool:
    if prediction is None:
        return False
    if _accent_free(prediction) == _accent_free(gold.rstrip("?")):
        return True
    pred = prediction.strip()
    gold = gold.rstrip("?").strip()
    pred_key = _accent_free(pred)
    gold_key = _accent_free(gold)
    gold_equivalents = equivalences.get(gold, set())
    if pred_key in {_accent_free(item) for item in gold_equivalents}:
        return True
    pred_equivalents = equivalences.get(pred, set())
    return gold_key in {_accent_free(item) for item in pred_equivalents}


def score_predictions(pairs: list[tuple[str, str]],
                      predictions: list[str | None],
                      equivalences: dict[str, set[str]]) -> dict[str, int]:
    if len(pairs) != len(predictions):
        raise ValueError("prediction count does not match gold token count")
    strict = 0
    equiv = 0
    for (_, raw_gold), prediction in zip(pairs, predictions):
        gold = raw_gold.rstrip("?")
        strict += prediction == gold
        equiv += equivalent(prediction, gold, equivalences)
    return {
        "tokens": len(pairs),
        "strict_correct": strict,
        "equiv_correct": equiv,
    }


def collect_results() -> dict:
    equivalences = _load_equivalences()
    instances: dict[str, Dilemma] = {}
    datasets = {}

    for spec in SPECS:
        path = BENCHMARK_DIR / spec.fixture
        pairs = load_gold_fixture(path)
        if spec.profile not in instances:
            instances[spec.profile] = Dilemma(**spec.dilemma_kwargs)
        predictions = instances[spec.profile].lemmatize_batch(
            [form for form, _ in pairs], guess=False
        )
        datasets[spec.name] = {
            "fixture": spec.fixture,
            "fixture_sha256": _sha256(path),
            "profile": spec.profile,
            "guess": False,
            "metrics": score_predictions(pairs, predictions, equivalences),
            "predictions": predictions,
        }

    return {"schema_version": 1, "datasets": datasets}


def _print_metrics(name: str, metrics: dict[str, int]) -> None:
    tokens = metrics["tokens"]
    strict = 100 * metrics["strict_correct"] / tokens
    equiv = 100 * metrics["equiv_correct"] / tokens
    print(
        f"{name}: {tokens} tokens, strict={strict:.2f}% "
        f"({metrics['strict_correct']}), equiv={equiv:.2f}% "
        f"({metrics['equiv_correct']})"
    )


def check_results(current: dict, baseline: dict) -> list[str]:
    """Print prediction changes and return accuracy regression messages."""
    failures = []
    baseline_datasets = baseline.get("datasets", {})
    for spec in SPECS:
        name = spec.name
        actual = current["datasets"][name]
        expected = baseline_datasets.get(name)
        if not expected:
            failures.append(f"{name}: missing from baseline")
            continue

        _print_metrics(name, actual["metrics"])
        if actual["fixture_sha256"] != expected.get("fixture_sha256"):
            failures.append(
                f"{name}: gold fixture changed; review it and refresh the baseline"
            )
            continue

        expected_predictions = expected.get("predictions", [])
        actual_predictions = actual["predictions"]
        if len(expected_predictions) != len(actual_predictions):
            failures.append(f"{name}: baseline prediction count changed")
            continue

        pairs = load_gold_fixture(BENCHMARK_DIR / spec.fixture)
        changes = []
        for index, ((form, gold), old, new) in enumerate(
                zip(pairs, expected_predictions, actual_predictions), 1):
            if old != new:
                changes.append((index, form, gold, old, new))
        if changes:
            print(f"  {len(changes)} token prediction change(s):")
            for index, form, gold, old, new in changes[:25]:
                print(
                    f"    {index}: {form!r}, gold={gold!r}, "
                    f"baseline={old!r}, current={new!r}"
                )
            if len(changes) > 25:
                print(f"    ... {len(changes) - 25} more")

        for metric in ("strict_correct", "equiv_correct"):
            old_value = expected["metrics"][metric]
            new_value = actual["metrics"][metric]
            if new_value < old_value:
                failures.append(
                    f"{name}: {metric} regressed from {old_value} to {new_value}"
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update", action="store_true",
        help="write the current reviewed predictions as the new baseline",
    )
    args = parser.parse_args()

    current = collect_results()
    if args.update:
        BASELINE_PATH.write_text(
            json.dumps(current, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"updated {BASELINE_PATH}")
        for name, result in current["datasets"].items():
            _print_metrics(name, result["metrics"])
        return 0

    if not BASELINE_PATH.exists():
        print(f"missing baseline: {BASELINE_PATH}", file=sys.stderr)
        return 2
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    failures = check_results(current, baseline)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    print("benchmark regression gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate GlossAPI's Greek variety classifier outside Dilemma's runtime.

The classifier uses a custom two-layer head whose implementation is recorded
and revision-pinned in ``data/glossapi_corpora.json``.  It is evaluated on
Dilemma's independently maintained register fixtures.  This script is an
optional experiment: neither GreekBERT nor Transformers is imported by the
``dilemma`` package or installed as a core dependency.

The Modern Greek fixtures are current Wikipedia prose, so their expected
classifier label is Standard Modern Greek rather than the model's historical
``demotic`` class.  Because the model card says Wiki data was used in training,
source-family overlap cannot be ruled out and is recorded in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "glossapi_corpora.json"
DEFAULT_OUTPUT = ROOT / "data" / "glossapi_variety_evaluation.json"
BENCHMARK_DIR = ROOT / "data" / "benchmarks"


@dataclass(frozen=True)
class Fixture:
    key: str
    path: Path
    expected_label: int | None
    source_note: str


FIXTURES = (
    Fixture(
        "classical_ancient",
        BENCHMARK_DIR / "ag.txt",
        0,
        "Sextus Empiricus, First1KGreek; no reported model training source match.",
    ),
    Fixture(
        "katharevousa",
        BENCHMARK_DIR / "katharevousa.txt",
        3,
        "1868 Sathas excerpt from Wikisource; exact training overlap is unknown.",
    ),
    Fixture(
        "standard_modern_test",
        BENCHMARK_DIR / "demotic.txt",
        1,
        "Current Wikipedia prose; model card reports Wiki training data.",
    ),
    Fixture(
        "standard_modern_dev",
        BENCHMARK_DIR / "demotic_dev.txt",
        1,
        "Current Wikipedia prose; model card reports Wiki training data.",
    ),
    Fixture(
        "byzantine_out_of_scope",
        ROOT / "tests" / "fixtures" / "form_attest"
        / "byzantine_vernacular" / "testbyz.txt",
        None,
        "Tiny synthetic Medieval Greek probe; the model exposes no Medieval label.",
    ),
)


def load_model_spec(path: Path = MANIFEST_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)["variety_model"]


def chunk_words(text: str, words_per_chunk: int = 80, minimum_words: int = 40) -> list[str]:
    """Split a fixture into deterministic fragments and merge a short tail."""
    words = text.split()
    chunks = [words[i:i + words_per_chunk]
              for i in range(0, len(words), words_per_chunk)]
    if len(chunks) > 1 and len(chunks[-1]) < minimum_words:
        chunks[-2].extend(chunks.pop())
    return [" ".join(chunk) for chunk in chunks if chunk]


def load_examples(
    fixtures: tuple[Fixture, ...] = FIXTURES,
    *,
    words_per_chunk: int = 80,
) -> list[dict]:
    examples = []
    for fixture in fixtures:
        text = fixture.path.read_text(encoding="utf-8")
        for index, chunk in enumerate(chunk_words(text, words_per_chunk)):
            examples.append({
                "fixture": fixture.key,
                "chunk": index,
                "text": chunk,
                "expected_label": fixture.expected_label,
                "source_note": fixture.source_note,
            })
    return examples


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_weights(spec: dict, *, local_only: bool) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - core dependency
        raise RuntimeError("huggingface_hub is required") from exc
    weights = spec["weights"]
    resolved = Path(hf_hub_download(
        repo_id=spec["repo"],
        filename=weights["path"],
        revision=spec["revision"],
        local_files_only=local_only,
    ))
    if resolved.stat().st_size != weights["size"]:
        raise RuntimeError("variety model weight size does not match the manifest")
    if _sha256(resolved) != weights["sha256"]:
        raise RuntimeError("variety model SHA-256 does not match the manifest")
    return resolved


def _optional_ml_imports():
    try:
        import torch
        from torch import nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "variety evaluation requires: pip install -e '.[variety-eval]'"
        ) from exc
    return torch, nn, AutoConfig, AutoModel, AutoTokenizer


def load_classifier(spec: dict, *, local_only: bool, device: str):
    torch, nn, AutoConfig, AutoModel, AutoTokenizer = _optional_ml_imports()

    class GlossAPIVarietyClassifier(nn.Module):
        def __init__(self, config, num_labels: int = 4):
            super().__init__()
            self.bert = AutoModel.from_config(config)
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, 256),
                nn.Dropout(0.1),
                nn.Linear(256, num_labels),
            )

        def forward(self, input_ids, attention_mask):
            output = self.bert(input_ids, attention_mask=attention_mask)
            return self.classifier(output.pooler_output)

    config = AutoConfig.from_pretrained(
        spec["repo"], revision=spec["revision"], local_files_only=local_only
    )
    tokenizer = AutoTokenizer.from_pretrained(
        spec["repo"], revision=spec["revision"], local_files_only=local_only
    )
    model = GlossAPIVarietyClassifier(config, num_labels=len(spec["labels"]))
    weights = resolve_weights(spec, local_only=local_only)
    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older supported Torch
        state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return torch, tokenizer, model


def choose_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def predict(
    examples: list[dict],
    spec: dict,
    *,
    local_only: bool,
    requested_device: str,
    batch_size: int,
) -> tuple[list[dict], str]:
    torch, _nn, _AutoConfig, _AutoModel, _AutoTokenizer = _optional_ml_imports()
    device = choose_device(torch, requested_device)
    torch, tokenizer, model = load_classifier(
        spec, local_only=local_only, device=device
    )
    labels = {int(key): value for key, value in spec["labels"].items()}
    predictions = []
    with torch.inference_mode():
        for offset in range(0, len(examples), batch_size):
            batch = examples[offset:offset + batch_size]
            encoded = tokenizer(
                [item["text"] for item in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()
                       if key in {"input_ids", "attention_mask"}}
            logits = model(**encoded)
            probabilities = torch.softmax(logits, dim=-1).cpu()
            for item, probs in zip(batch, probabilities):
                predicted = int(torch.argmax(probs).item())
                predictions.append({
                    "fixture": item["fixture"],
                    "chunk": item["chunk"],
                    "word_count": len(item["text"].split()),
                    "expected_label": item["expected_label"],
                    "predicted_label": predicted,
                    "predicted_name": labels[predicted],
                    "confidence": round(float(probs[predicted]), 6),
                    "correct": (
                        predicted == item["expected_label"]
                        if item["expected_label"] is not None else None
                    ),
                    "source_note": item["source_note"],
                })
    return predictions, device


def summarize(predictions: list[dict], labels: dict[int, str]) -> dict:
    scored = [item for item in predictions if item["expected_label"] is not None]
    correct = sum(bool(item["correct"]) for item in scored)
    by_fixture = defaultdict(list)
    confusion: Counter[tuple[int, int]] = Counter()
    for item in predictions:
        by_fixture[item["fixture"]].append(item)
        if item["expected_label"] is not None:
            confusion[(item["expected_label"], item["predicted_label"])] += 1
    fixture_reports = {}
    for fixture, items in sorted(by_fixture.items()):
        fixture_scored = [item for item in items if item["correct"] is not None]
        fixture_reports[fixture] = {
            "chunks": len(items),
            "accuracy": (
                round(sum(bool(item["correct"]) for item in fixture_scored)
                      / len(fixture_scored), 6)
                if fixture_scored else None
            ),
            "prediction_counts": dict(sorted(Counter(
                item["predicted_name"] for item in items
            ).items())),
        }
    return {
        "scored_chunks": len(scored),
        "correct_chunks": correct,
        "accuracy": round(correct / len(scored), 6) if scored else None,
        "by_fixture": fixture_reports,
        "confusion": [
            {
                "expected": labels[expected],
                "predicted": labels[predicted],
                "count": count,
            }
            for (expected, predicted), count in sorted(confusion.items())
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--words-per-chunk", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--local-only", action="store_true",
        help="require model/tokenizer files to already be in the HF cache",
    )
    args = parser.parse_args(argv)

    spec = load_model_spec(args.manifest)
    examples = load_examples(words_per_chunk=args.words_per_chunk)
    predictions, device = predict(
        examples,
        spec,
        local_only=args.local_only,
        requested_device=args.device,
        batch_size=args.batch_size,
    )
    labels = {int(key): value for key, value in spec["labels"].items()}
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "external_optional_evaluation",
        "core_dependency": False,
        "model": {
            "repo": spec["repo"],
            "revision": spec["revision"],
            "weights_sha256": spec["weights"]["sha256"],
            "implementation_reference": spec["implementation_reference"],
        },
        "device": device,
        "chunking": {
            "words_per_chunk": args.words_per_chunk,
            "minimum_tail_words": 40,
        },
        "summary": summarize(predictions, labels),
        "predictions": predictions,
        "limitations": [
            "The model card reports only a random held-out split of its 5,020 fragments.",
            "Modern test fixtures share the broad Wiki source family named in training data, so leakage cannot be excluded.",
            "The model has no Medieval Greek class; the Byzantine probe is unscored.",
            "Gutenberg variety labels generated by this same model are not independent gold data and are not used here.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = report["summary"]
    print(
        f"GlossAPI variety accuracy: {summary['correct_chunks']}/"
        f"{summary['scored_chunks']} ({summary['accuracy']:.1%})"
    )
    for fixture, item in summary["by_fixture"].items():
        print(f"  {fixture}: {item['prediction_counts']} accuracy={item['accuracy']}")
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

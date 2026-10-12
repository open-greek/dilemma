"""Compare batched decoding to the former serial beam-search algorithm."""
import numpy as np
import pytest
from pathlib import Path
import time

from dilemma.onnx_inference import OnnxLemmaModel


def serial_generate(model, src, mask, max_len=32, bos_id=1, eos_id=2, num_beams=1):
    memory = model._encode(src, mask)
    output = []
    for i in range(len(src)):
        beams, complete = [([bos_id], 0.0)], []
        for _ in range(max_len):
            candidates = []
            for ids, score in beams:
                if ids[-1] == eos_id:
                    complete.append((ids, score))
                    continue
                logits = model._decode_step(ids, memory[i:i + 1], mask[i:i + 1])
                maximum = logits.max()
                probs = logits - maximum - np.log(np.exp(logits - maximum).sum())
                top = np.argpartition(probs, -num_beams)[-num_beams:]
                top = top[np.argsort(probs[top])[::-1]]
                candidates.extend((ids + [int(k)], score + float(probs[k])) for k in top)
            if not candidates:
                break
            candidates.sort(key=lambda row: row[1], reverse=True)
            beams = candidates[:num_beams]
            if all(ids[-1] == eos_id for ids, _ in beams):
                complete.extend(beams)
                break
        complete.extend(row for row in beams if row[0][-1] != eos_id)
        complete.sort(key=lambda row: row[1], reverse=True)
        output.append(complete[:num_beams])
    return output


class FakeDecoder:
    def __init__(self):
        self.batch_sizes = []

    def run(self, _, values):
        tgt, memory = values["tgt"], values["memory"]
        self.batch_sizes.append(len(tgt))
        logits = np.empty((*tgt.shape, 8), dtype=np.float32)
        for row, ids in enumerate(tgt):
            word = int(memory[row, 0, 0])
            for position in range(tgt.shape[1]):
                scores = np.array([-20, -21, -4, -2, -3, -5, -6, -7], dtype=np.float32)
                scores[3 + (word + int(ids[position])) % 3] = 1.5
                scores[2] = 2.5 if position >= word % 4 else -8
                logits[row, position] = scores
        return [logits]


def fake_model():
    model = object.__new__(OnnxLemmaModel)
    model.decoder = FakeDecoder()
    model._encode = lambda src, mask: src[:, :1, None].astype(np.float32)
    return model


@pytest.mark.parametrize("beams", [1, 2, 4])
@pytest.mark.parametrize("max_len", [0, 1, 4, 7])
def test_same_ranked_beams_with_different_endings_and_truncation(beams, max_len):
    src = np.arange(70, dtype=np.int64).reshape(-1, 1)
    mask = np.zeros_like(src, dtype=bool)
    reference = serial_generate(fake_model(), src, mask, max_len=max_len, num_beams=beams)
    model = fake_model()
    actual = model.generate(src, mask, max_len=max_len, num_beams=beams)
    assert actual == reference
    assert all(size <= 128 for size in model.decoder.batch_sizes)
    if max_len > 1 and beams == 4:
        assert 128 in model.decoder.batch_sizes


def test_empty_batch():
    empty = np.empty((0, 1), dtype=np.int64)
    assert fake_model().generate(empty, empty.astype(bool)) == []


def test_encoder_batches_are_bounded_without_reordering_words():
    src = np.arange(300, dtype=np.int64).reshape(-1, 1)
    mask = np.zeros_like(src, dtype=bool)
    expected = serial_generate(fake_model(), src, mask, max_len=4, num_beams=4)
    model = fake_model()
    encode = model._encode
    sizes = []
    def bounded(values, padding):
        sizes.append(len(values))
        return encode(values, padding)
    model._encode = bounded
    assert model.generate(src, mask, max_len=4, num_beams=4) == expected
    assert sizes == [128, 128, 44]


def test_prediction_cache_keeps_beam_width_separate_and_is_bounded():
    from dilemma import Dilemma
    instance = object.__new__(Dilemma)
    calls = []
    def predict(words, num_beams=4):
        calls.append((words, num_beams))
        return [word + str(num_beams) for word in words]
    instance._predict_uncached = predict
    assert instance._predict(["a", "a", "b"]) == ["a4", "a4", "b4"]
    assert instance._predict(["b", "a"]) == ["b4", "a4"]
    assert instance._predict(["a"], num_beams=1) == ["a1"]
    assert calls == [(["a", "b"], 4), (["a"], 1)]
    instance._predict([str(i) for i in range(65540)])
    assert len(instance._prediction_cache) == 65536


@pytest.mark.slow
def test_real_model_preserves_ranked_readings():
    from dilemma.core import MODEL_DIR
    from dilemma.onnx_inference import CharVocabLight
    directory = Path(MODEL_DIR) / "combined-s3"
    if not (directory / "encoder.onnx").exists():
        pytest.skip("Combined ONNX model is not installed")
    vocab = CharVocabLight(directory / "vocab.json")
    model = OnnxLemmaModel(directory)
    forms = ["ἀπομεταθεμελιώσεως", "ὑπερεξαδυναστεύεσθαι", "φιλοσοφοῦντες",
             "προκατανοησάμενος", "Ἐρατοσθένους", "συνεπαναφέρουσιν", "λελυμένοι", "ἀνεγνωκότες"]
    src = np.array([vocab.encode(form) + [0] * (48 - len(vocab.encode(form)))
                    for form in forms], dtype=np.int64)
    mask = src == 0
    start = time.monotonic()
    expected = serial_generate(model, src, mask, num_beams=4)
    serial_seconds = time.monotonic() - start
    start = time.monotonic()
    actual = model.generate(src, mask, num_beams=4)
    batch_seconds = time.monotonic() - start
    assert [[ids for ids, score in rows] for rows in actual] == [
        [ids for ids, score in rows] for rows in expected]
    for before, after in zip(expected, actual):
        assert np.allclose([score for ids, score in before],
                           [score for ids, score in after], atol=1e-4)
    many = model.generate(np.tile(src, (17, 1)), np.tile(mask, (17, 1)), num_beams=4)
    assert [[ids for ids, score in rows] for rows in many] == [
        [ids for ids, score in rows] for rows in expected] * 17
    print(f"serial_seconds={serial_seconds:.3f} batch_seconds={batch_seconds:.3f}")

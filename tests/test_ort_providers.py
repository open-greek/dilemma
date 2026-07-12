"""Device-agnostic onnxruntime provider selection + silent-fallback warning."""
import logging

import pytest

from dilemma._ort_providers import GPU_PROVIDERS, ort_providers


def test_auto_providers_end_with_cpu_fallback():
    provs = ort_providers()
    assert provs, "must return at least one provider"
    assert provs[-1] == "CPUExecutionProvider", "CPU must always be the fallback"


def test_env_override(monkeypatch):
    monkeypatch.setenv("DILEMMA_ORT_PROVIDERS",
                       "CUDAExecutionProvider, CPUExecutionProvider")
    assert ort_providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    monkeypatch.setenv("DILEMMA_ORT_PROVIDERS", "CPUExecutionProvider")
    assert ort_providers() == ["CPUExecutionProvider"]


def test_auto_never_requests_coreml_by_default(monkeypatch):
    # CoreML is opt-in: the auto path must not pick it, so Mac behavior is
    # unchanged unless a caller explicitly asks via the env override.
    monkeypatch.delenv("DILEMMA_ORT_PROVIDERS", raising=False)
    assert "CoreMLExecutionProvider" not in ort_providers()


def test_make_session_warns_on_silent_cpu_fallback(monkeypatch, caplog, tmp_path):
    # Force a GPU request; onnxruntime on a CPU-only box will fall back to CPU.
    monkeypatch.setenv("DILEMMA_ORT_PROVIDERS",
                       "CUDAExecutionProvider,CPUExecutionProvider")
    ort = pytest.importorskip("onnxruntime")
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return  # a real GPU host: no fallback to warn about
    from dilemma._ort_providers import make_session
    # a trivial identity ONNX model so we exercise real session creation
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["x"], ["y"])
    # Pin a released opset and IR version: a fresh `onnx` stamps its own
    # latest (e.g. opset 27), which an older onnxruntime rejects at model
    # load, before the fallback warning under test can be exercised.
    model = helper.make_model(
        helper.make_graph([node], "g", [x], [y]),
        opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    p = tmp_path / "id.onnx"
    p.write_bytes(model.SerializeToString())
    with caplog.at_level(logging.WARNING, logger="dilemma.onnx"):
        sess = make_session(p)
    assert getattr(sess, "dilemma_on_gpu", None) is False
    assert any("GPU acceleration is OFF" in r.message for r in caplog.records)
    _ = GPU_PROVIDERS  # referenced for the import

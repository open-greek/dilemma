"""Device-agnostic onnxruntime provider selection + silent-fallback warning."""
import logging

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
    ort = __import__("onnxruntime")
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return  # a real GPU host: no fallback to warn about
    from dilemma._ort_providers import make_session
    # a trivial identity ONNX model so we exercise real session creation
    try:
        import numpy as np
        from onnx import TensorProto, helper
    except Exception:  # noqa: BLE001 - onnx not installed in this env
        return
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["x"], ["y"])
    model = helper.make_model(helper.make_graph([node], "g", [x], [y]))
    p = tmp_path / "id.onnx"
    p.write_bytes(model.SerializeToString())
    with caplog.at_level(logging.WARNING, logger="dilemma.onnx"):
        sess = make_session(p)
    assert getattr(sess, "dilemma_on_gpu", None) is False
    assert any("GPU acceleration is OFF" in r.message for r in caplog.records)
    _ = GPU_PROVIDERS  # referenced for the import

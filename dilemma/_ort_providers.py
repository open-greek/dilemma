"""Device-agnostic onnxruntime execution-provider selection.

The same code should run CPU on a Mac and CUDA on an NVIDIA box with no edits,
so ONNX sessions ask for providers via `ort_providers()` instead of hardcoding
one. Default order: CUDA if the installed onnxruntime exposes it (i.e. a GPU +
onnxruntime-gpu), else CPU. CPU is always the final fallback.

Override with the env var (comma-separated, highest priority first), e.g.:
    DILEMMA_ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider   # force GPU
    DILEMMA_ORT_PROVIDERS=CoreMLExecutionProvider,CPUExecutionProvider # try Apple NE
    DILEMMA_ORT_PROVIDERS=CPUExecutionProvider                         # force CPU

CoreML is opt-in only (op-coverage for these models varies); the auto path never
picks it, so Mac behavior is unchanged unless explicitly requested.
"""
from __future__ import annotations

import logging
import os

_log = logging.getLogger("dilemma.onnx")

# Providers that put work on an accelerator, not the CPU.
GPU_PROVIDERS = frozenset({"CUDAExecutionProvider", "TensorrtExecutionProvider",
                           "ROCMExecutionProvider", "MIGraphXExecutionProvider"})


def ort_providers() -> list[str]:
    """The execution providers to hand `onnxruntime.InferenceSession(providers=...)`,
    best-first. Honors DILEMMA_ORT_PROVIDERS; else auto-detects CUDA vs CPU."""
    override = os.environ.get("DILEMMA_ORT_PROVIDERS", "").strip()
    if override:
        return [p.strip() for p in override.split(",") if p.strip()]
    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
    except Exception:  # noqa: BLE001 - onnxruntime absent/broken: caller handles
        return ["CPUExecutionProvider"]
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:      # NVIDIA + onnxruntime-gpu
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" not in providers:   # always a fallback
        providers.append("CPUExecutionProvider")
    return providers


def make_session(model_path, sess_options=None):
    """Create an onnxruntime InferenceSession with device-agnostic providers,
    and WARN if a GPU provider was requested but the session silently fell back
    to CPU. That silent fallback (onnxruntime-gpu missing, or a CUDA/cuDNN
    mismatch) is the classic "why is my GPU box no faster than a laptop and the
    GPU at 0% util" trap - onnxruntime runs the model on CPU without erroring.
    The session also carries `.dilemma_on_gpu` so a caller can assert the device
    up front instead of discovering it from a slow run."""
    import onnxruntime as ort

    requested = ort_providers()
    sess = ort.InferenceSession(str(model_path), sess_options=sess_options,
                                providers=requested)
    active = sess.get_providers()
    want_gpu = [p for p in requested if p in GPU_PROVIDERS]
    on_gpu = any(p in active for p in GPU_PROVIDERS)
    if want_gpu and not on_gpu:
        _log.warning(
            "onnxruntime requested %s but is running on %s: GPU acceleration is "
            "OFF (install onnxruntime-gpu and verify CUDA/cuDNN, or set "
            "DILEMMA_ORT_PROVIDERS). The model still runs, on CPU.",
            want_gpu, active)
    try:
        sess.dilemma_on_gpu = on_gpu          # queryable device flag
    except Exception:  # noqa: BLE001 - session may forbid attrs; best-effort
        pass
    return sess
